//! Nonlinear solver using Newton's method, with a user-supplied Jacobian. Ported from SUNDIALS.

use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimName, Dyn, Matrix, OVector, RealField, Scalar,
    Storage, StorageMut, U1,
};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{Error, NLProblem, NLSolver};

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "OVector<T, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "OVector<T, D>: Deserialize<'de>, DefaultAllocator: Allocator<T, D>"
    ))
)]
#[derive(Debug, Clone)]
pub struct Newton<T, D>
where
    D: Dim,
    DefaultAllocator: Allocator<T, D>,
{
    /// Newton update vector
    delta: OVector<T, D>,
    /// Jacobian status, current = `true` / stale = `false`
    jcur: bool,
    /// current number of iterations in a solve attempt
    curiter: usize,
    /// maximum number of iterations in a solve attempt
    maxiters: usize,
    /// total number of nonlinear iterations across all solves
    niters: usize,
    /// total number of convergence failures across all solves
    nconvfails: usize,
}

impl<T, D> Newton<T, D>
where
    D: Dim,
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new Newton solver, statically sized
    ///
    /// # Arguments
    /// * `maxiters` - The maximum number of iterations per solve attempt
    pub fn new(maxiters: usize) -> Self
    where
        T: Scalar + RealField,
        D: DimName,
    {
        Newton {
            delta: OVector::zeros(),
            jcur: false,
            curiter: 0,
            maxiters,
            niters: 0,
            nconvfails: 0,
        }
    }
}

impl<T> Newton<T, Dyn>
where
    DefaultAllocator: Allocator<T, Dyn>,
{
    /// Create a new Newton solver, dynamically sized
    ///
    /// # Arguments
    /// * `size` - The problem size
    /// * `maxiters` - The maximum number of iterations per solve attempt
    pub fn new_dynamic(size: usize, maxiters: usize) -> Self
    where
        T: Scalar + RealField,
    {
        Newton {
            delta: OVector::<T, Dyn>::zeros(size),
            jcur: false,
            curiter: 0,
            maxiters,
            niters: 0,
            nconvfails: 0,
        }
    }
}

impl<T, D> NLSolver<T, D> for Newton<T, D>
where
    T: Scalar + RealField + Copy,
    D: Dim,
    DefaultAllocator: Allocator<T, D>,
{
    fn solve<NLP, SA, SB, SC>(
        &mut self,
        problem: &mut NLP,
        y0: &Matrix<T, D, U1, SA>,
        y: &mut Matrix<T, D, U1, SB>,
        w: &Matrix<T, D, U1, SC>,
        tol: T,
        call_lsetup: bool,
    ) -> Result<(), Error>
    where
        NLP: NLProblem<T, D>,
        SA: Storage<T, D, U1>,
        SB: StorageMut<T, D, U1>,
        SC: Storage<T, D, U1>,
    {
        tracing::trace!("Newton::solve");

        // assume the Jacobian is good
        let mut jbad = false;
        let mut call_lsetup = call_lsetup;

        // looping point for attempts at solution of the nonlinear system: Evaluate the nonlinear
        // residual function (store in delta) Setup the linear solver if necessary Preform Newton
        // iteraion
        let retval: Result<(), Error> = 'outer: loop {
            // compute the nonlinear residual, store in delta
            let retval = problem
                .sys(y0, &mut self.delta)
                .and_then(|_| {
                    // if indicated, setup the linear system
                    if call_lsetup {
                        // NLS->LSetup() aka idaNlsLSetup()
                        problem
                            .setup(y0, &self.delta, jbad)
                            .map(|jcur| self.jcur = jcur)
                    } else {
                        Ok(())
                    }
                })
                .and_then(|_| {
                    // initialize counter curiter
                    self.curiter = 0;

                    // load prediction into y
                    y.copy_from(&y0);

                    // looping point for Newton iteration. Break out on any error.
                    'inner: loop {
                        // increment nonlinear solver iteration counter
                        self.niters += 1;

                        // compute the negative of the residual for the linear system rhs
                        self.delta.neg_mut();

                        // solve the linear system to get Newton update delta
                        let retval = problem.solve(y, &mut self.delta).and_then(|_| {
                            // update the Newton iterate
                            *y += &self.delta;

                            // test for convergence
                            problem
                                .ctest(self, y, &self.delta, tol, w)
                                .and_then(|converged| {
                                    if converged {
                                        // if successful update Jacobian status and return
                                        self.jcur = false;
                                        Ok(true)
                                    } else {
                                        self.curiter += 1;
                                        if self.curiter >= self.maxiters {
                                            Err(Error::ConvergenceRecover {})
                                        } else {
                                            // compute the nonlinear residual, store in delta
                                            // Ok(false) will continue to iterate 'inner
                                            problem.sys(y, &mut self.delta).and(Ok(false))
                                        }
                                    }
                                })
                        });

                        // check if the iteration should continue; otherwise exit Newton loop
                        if let Ok(false) = retval {
                            continue 'inner;
                        } else {
                            break retval.and(Ok(()));
                        }
                    } // end of Newton iteration loop
                });

            // all inner-loop results go here
            match retval {
                Ok(_) => {
                    return retval;
                }

                // If there is a recoverable convergence failure and the Jacobian-related data appears not to be
                // current, increment the convergence failure count and loop again with a call to lsetup in which jbad = true.
                Err(Error::ConvergenceRecover {}) => {
                    if !self.jcur {
                        self.nconvfails += 1;
                        call_lsetup = true;
                        jbad = true;
                        continue 'outer;
                    }
                }

                // Otherwise break out and return.
                _ => {
                    break 'outer retval;
                }
            }
        }; // end of setup loop

        // increment number of convergence failures
        self.nconvfails += 1;

        // all error returns exit here
        retval
    }

    fn get_num_iters(&self) -> usize {
        self.niters
    }

    fn get_cur_iter(&self) -> usize {
        self.curiter
    }

    fn get_num_conv_fails(&self) -> usize {
        self.nconvfails
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use linear::{Dense, LSolver};
    use nalgebra::{Matrix, Storage, U1, U3};

    use crate::norm_wrms::NormWRMS;

    use super::*;

    #[derive(Clone, Debug)]
    struct TestProblem {
        a: nalgebra::Matrix3<f64>,
        x: nalgebra::Vector3<f64>,

        lsolver: linear::Dense<U3>,
    }

    impl TestProblem {
        /// Jacobian of the nonlinear residual function
        ///
        /// ```math
        ///            (2x  2y  2z)
        /// J(x,y,z) = (4x  2y  -4)
        ///            (6x  -4  2z)  
        /// ```
        fn jac<SA, SB, SC>(
            _t: f64,
            y: &Matrix<f64, U3, U1, SA>,
            _fy: &Matrix<f64, U3, U1, SB>,
            j: &mut Matrix<f64, U3, U3, SC>,
        ) where
            SA: Storage<f64, U3, U1>,
            SB: Storage<f64, U3, U1>,
            SC: StorageMut<f64, U3, U3>,
        {
            let (x, y, z) = (y[0], y[1], y[2]);
            j.set_column(0, &nalgebra::Vector3::new(2.0 * x, 4.0 * x, 6.0 * x));
            j.set_column(1, &nalgebra::Vector3::new(2.0 * y, 2.0 * y, -4.0));
            j.set_column(2, &nalgebra::Vector3::new(2.0 * z, -4.0, 2.0 * z));
        }
    }

    impl NLProblem<f64, U3> for TestProblem {
        /// Nonlinear residual function
        ///
        /// ```math
        /// f1(x,y,z) = x^2 + y^2 + z^2 - 1 = 0
        /// f2(x,y,z) = 2x^2 + y^2 - 4z     = 0
        /// f3(x,y,z) = 3x^2 - 4y + z^2     = 0
        /// ```
        fn sys<SA, SB>(
            &mut self,
            ycor: &Matrix<f64, U3, U1, SA>,
            res: &mut Matrix<f64, U3, U1, SB>,
        ) -> Result<(), Error>
        where
            SA: Storage<f64, nalgebra::Const<3>, U1>,
            SB: StorageMut<f64, nalgebra::Const<3>, U1>,
        {
            let (x, y, z) = (ycor[0], ycor[1], ycor[2]);
            res[0] = x.powi(2) + y.powi(2) + z.powi(2) - 1.0;
            res[1] = 2.0 * x.powi(2) + y.powi(2) - 4.0 * z;
            res[2] = 3.0 * x.powi(2) - 4.0 * y + z.powi(2);
            Ok(())
        }

        fn setup<SA, SB>(
            &mut self,
            y: &Matrix<f64, U3, U1, SA>,
            _f: &Matrix<f64, U3, U1, SB>,
            _jbad: bool,
        ) -> Result<bool, Error>
        where
            SA: Storage<f64, U3, U1>,
            SB: Storage<f64, U3, U1>,
        {
            // compute the Jacobian
            Self::jac(0.0, y, &nalgebra::zero(), &mut self.a);
            // setup the linear solver
            self.lsolver.setup(&mut self.a).map_err(Error::from)?;
            Ok(true)
        }

        fn solve<SA, SB>(
            &mut self,
            _y: &Matrix<f64, U3, U1, SA>,
            b: &mut Matrix<f64, U3, U1, SB>,
        ) -> Result<(), Error>
        where
            SA: Storage<f64, U3, U1>,
            SB: StorageMut<f64, U3, U1>,
        {
            // Solve self.A * b = b
            //retval = SUNLinSolSolve(Imem->LS, Imem->A, Imem->x, b, ZERO);
            self.lsolver.solve(&self.a, &mut self.x, b, 0.0)?;

            b.copy_from(&self.x);
            Ok(())
        }

        fn ctest<NLS, SA, SB, SC>(
            &mut self,
            _solver: &NLS,
            _y: &Matrix<f64, U3, U1, SA>,
            del: &Matrix<f64, U3, U1, SB>,
            tol: f64,
            ewt: &Matrix<f64, U3, U1, SC>,
        ) -> Result<bool, Error>
        where
            NLS: NLSolver<f64, U3>,
            SA: Storage<f64, U3, U1>,
            SB: Storage<f64, U3, U1>,
            SC: Storage<f64, U3, U1>,
        {
            // compute the norm of the correction
            let delnrm = del.norm_wrms(ewt);

            //if (delnrm <= tol) return(SUN_NLS_SUCCESS);  /* success       */
            //else               return(SUN_NLS_CONTINUE); /* not converged */
            Ok(delnrm <= tol)
        }
    }

    #[test]
    fn test_newton() {
        // approximate solution
        let y_exp = nalgebra::vector![
            0.785196933062355226,
            0.496611392944656396,
            0.369922830745872357
        ];

        let mut p = TestProblem {
            a: nalgebra::Matrix3::zeros(),
            x: nalgebra::Vector3::zeros(),

            lsolver: Dense::new(),
        };

        // set initial guess
        let y0 = nalgebra::vector![0.5, 0.5, 0.5];
        let mut y = nalgebra::Vector3::zeros();

        // set weights
        let w = nalgebra::vector![1.0, 1.0, 1.0];

        let mut newton = Newton::new(10);
        newton
            .solve(&mut p, &y0, &mut y, &w, 1e-2, true)
            .expect("Should have converged.");

        let expected_err = nalgebra::vector![-0.00578453, 1.0143e-08, 1.47767e-08];

        // print the solution
        println!("Solution: y = {:?}", y);
        println!("Solution Error = {:?}", y - y_exp);
        println!("Number of nonlinear iterations: {}", newton.niters);
        println!("Number of convergence failures: {}", newton.nconvfails);

        assert_relative_eq!(y - y_exp, expected_err, max_relative = 1e-5);
    }
}
