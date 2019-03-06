use ndarray::*;

use crate::nonlinear::traits::*;
use crate::traits::ModelSpec;

struct Newton<P: NLProblem> {
    /// Newton update vector
    delta: Array1<P::Scalar>,
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

impl<
        P: NLProblem<Scalar = impl num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef>,
    > Newton<P>
{
    pub fn new(f: &P) -> Self {
        Newton {
            delta: Array::zeros(f.model_size()),
            jcur: false,
            curiter: 0,
            maxiters: 0,
            niters: 0,
            nconvfails: 0,
        }
    }

    /// The optional function SUNNonlinSolLSetupFn is called by sundials integrators to provide the nonlinear solver with access to its linear solver setup function.
    /// # Arguments
    ///
    /// * `problem` -
    /// * `y0` -
    /// * `y` -
    /// * `w` -
    /// * `tol` -
    /// * `lsetup` - (SUNNonlinSolLSetupFn) a wrapper function to the sundials integrator’s linear
    ///     solver setup function. See section 9.1.4 for the definition of SUNNonlinLSetupFn.
    ///
    /// Note: The SUNNonlinLSetupFn function sets up the linear system `Ax = b` where `A = ∂F/∂y` is
    /// the linearization of the nonlinear residual function `F(y) = 0` (when using sunlinsol direct
    /// linear solvers) or calls the user-defined preconditioner setup function (when using
    /// sunlinsol iterative linear solvers). sunnonlinsol implementations that do not require
    /// solving this system, do not utilize sunlinsol linear solvers, or use sunlinsol linear
    /// solvers that do not require setup may set this operation to NULL.
    ///
    /// Performs the nonlinear solve `F(y) = 0`
    ///
    /// # Returns
    ///
    /// Ok(())
    ///
    /// # Errors
    ///
    /// Recoverable failure return codes (positive):
    /// * SUN_NLS_CONV_RECVR
    /// *_RHSFUNC_RECVR (ODEs) or *_RES_RECVR (DAEs)
    /// *_LSETUP_RECVR
    /// *_LSOLVE_RECVR
    ///
    /// Unrecoverable failure return codes (negative):
    /// *_MEM_NULL
    /// *_RHSFUNC_FAIL (ODEs) or *_RES_FAIL (DAEs)
    /// *_LSETUP_FAIL
    /// *_LSOLVE_FAIL
    pub fn solve<S1, S2>(
        &mut self,
        problem: &mut P,
        y0: &ArrayBase<S1, Ix1>,
        y: &mut ArrayBase<S2, Ix1>,
        w: &ArrayBase<S1, Ix1>,
        tol: P::Scalar,
        mut call_lsetup: bool,
    ) -> Result<(), Error>
    where
        S1: Data<Elem = P::Scalar>,
        S2: DataMut<Elem = P::Scalar>,
    {
        // assume the Jacobian is good
        let mut jbad = false;

        // looping point for attempts at solution of the nonlinear system: Evaluate the nonlinear
        // residual function (store in delta) Setup the linear solver if necessary Preform Newton
        // iteraion
        let retval = 'outer: loop {
            // compute the nonlinear residual, store in delta
            NLProblem::res(problem, y0, &mut self.delta);
            //retval = NEWTON_CONTENT(NLS)->Sys(y0, delta, mem);
            //if (retval != SUN_NLS_SUCCESS) break;

            // if indicated, setup the linear system
            if call_lsetup {
                self.jcur = NLProblem::lsetup(problem, y0, &self.delta.view(), jbad);
            }

            // initialize counter curiter
            self.curiter = 0;

            // load prediction into y
            y.assign(&y0);

            // looping point for Newton iteration. Break out on any error.
            let retval: Result<(), Error> = 'inner: loop {
                // increment nonlinear solver iteration counter
                self.niters += 1;

                // compute the negative of the residual for the linear system rhs
                self.delta.mapv_inplace(P::Scalar::neg);

                // solve the linear system to get Newton update delta
                NLProblem::lsolve(problem, y, &mut self.delta);

                // update the Newton iterate
                //N_VLinearSum(ONE, y, ONE, delta, y);
                *y += &self.delta;

                // test for convergence
                let retval = NLProblem::ctest(problem, y, &self.delta.view(), tol, w);

                match retval {
                    // if successful update Jacobian status and return
                    Ok(true) => {
                        self.jcur = false;
                        //  return(SUN_NLS_SUCCESS);
                        break Ok(());
                    }
                    // check if the iteration should continue; otherwise exit Newton loop
                    Ok(false) => {
                        // not yet converged. Increment curiter and test for max allowed.
                        self.curiter += 1;
                        if self.curiter >= self.maxiters {
                            //  retval = SUN_NLS_CONV_RECVR;
                            break Err(Error::ConvergenceRecover {});
                        }

                        // compute the nonlinear residual, store in delta
                        let retval = NLProblem::res(problem, y, &mut self.delta);

                        //if (retval != SUN_NLS_SUCCESS) break;
                        if retval.is_err() {
                            break retval;
                        }
                    }
                    Err(_) => {
                        //if (retval != SUN_NLS_CONTINUE) break;
                        break retval.map(|_| ());
                    }
                }
            }; // end of Newton iteration loop

            // all inner-loop results go here

            // If there is a recoverable convergence failure and the Jacobian-related data appears
            // not to be current, increment the convergence failure count and loop again with a
            // call to lsetup in which jbad is TRUE. Otherwise break out and return.
            match retval {
                Ok(_) => {
                    return retval;
                }

                Err(_) => {
                    if !self.jcur {
                        self.nconvfails += 1;
                        call_lsetup = true;
                        jbad = true;
                        continue 'outer;
                    } else {
                        break 'outer retval;
                    }
                }
            }
        }; // end of setup loop

        // increment number of convergence failures
        self.nconvfails += 1;

        // all error returns exit here
        retval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::traits::*;
    use crate::traits::ModelSpec;
    use ndarray::*;
    use nearly_eq::assert_nearly_eq;

    #[derive(Clone, Debug)]
    struct TestProblem {
        A: Array<f64, Ix2>,
        x: Array<f64, Ix1>,
    }

    impl ModelSpec for TestProblem {
        type Scalar = f64;
        type Dim = Ix1;

        fn model_size(&self) -> usize {
            3
        }
    }

    impl TestProblem {
        /// Jacobian of the nonlinear residual function
        ///
        ///            ( 2x  2y  2z )
        /// J(x,y,z) = ( 4x  2y  -4 )
        ///            ( 6x  -4  2z )  
        fn jac<S1, S2, S3>(
            &self,
            t: f64,
            y: &ArrayBase<S1, Ix1>,
            fy: &ArrayBase<S2, Ix1>,
            J: &mut ArrayBase<S3, Ix2>,
        ) -> Result<(), failure::Error>
        where
            S1: Data<Elem = <Self as ModelSpec>::Scalar>,
            S2: Data<Elem = <Self as ModelSpec>::Scalar>,
            S3: DataMut<Elem = <Self as ModelSpec>::Scalar>,
        {
            J[[0, 0]] = 2.0 * y[1];
            J[[0, 1]] = 2.0 * y[2];
            J[[0, 2]] = 2.0 * y[3];

            J[[1, 0]] = 4.0 * y[1];
            J[[1, 1]] = 2.0 * y[2];
            J[[1, 2]] = -4.0;

            J[[2, 0]] = 6.0 * y[1];
            J[[2, 1]] = -4.0;
            J[[2, 2]] = 2.0 * y[3];
            Ok(())
        }
    }

    impl NLProblem for TestProblem {
        /// Nonlinear residual function
        ///
        /// f1(x,y,z) = x^2 + y^2 + z^2 - 1 = 0
        /// f2(x,y,z) = 2x^2 + y^2 - 4z     = 0
        /// f3(x,y,z) = 3x^2 - 4y + z^2     = 0
        fn res<S1, S2>(
            &self,
            y: &ArrayBase<S1, Ix1>,
            f: &mut ArrayBase<S2, Ix1>,
        ) -> Result<(), Error>
        where
            S1: Data<Elem = <Self as ModelSpec>::Scalar>,
            S2: DataMut<Elem = <Self as ModelSpec>::Scalar>,
        {
            f[0] = y[1] * y[1] + y[2] * y[2] + y[3] * y[3] - 1.0;
            f[1] = 2.0 * y[1] * y[1] + y[2] * y[2] - 4.0 * y[3];
            f[2] = 3.0 * (y[1] * y[1]) - 4.0 * y[2] + y[3] * y[3];
            Ok(())
        }

        fn lsetup<S1>(
            &mut self,
            y: &ArrayBase<S1, Ix1>,
            F: &ArrayView<<Self as ModelSpec>::Scalar, Ix1>,
            jbad: bool,
        ) -> bool
        where
            S1: Data<Elem = <Self as ModelSpec>::Scalar>,
        {
            // compute the Jacobian
            self.jac(0.0, y, &Array::zeros(self.model_size()), &mut self.A);
            //retval = Jac(ZERO, y, NULL, Imem->A, NULL, NULL, NULL, NULL);
            //if (retval != 0) return(retval);

            // update Jacobian status
            //*jcur = SUNTRUE;

            /* setup the linear solver */
            //retval = SUNLinSolSetup(Imem->LS, Imem->A);

            //return(retval);
            true
        }

        fn lsolve<S1, S2>(&self, y: &ArrayBase<S1, Ix1>, b: &mut ArrayBase<S2, Ix1>)
        where
            S1: Data<Elem = <Self as ModelSpec>::Scalar>,
            S2: DataMut<Elem = <Self as ModelSpec>::Scalar>,
        {
            //retval = SUNLinSolSolve(Imem->LS, Imem->A, Imem->x, b, ZERO);
            //N_VScale(ONE, Imem->x, b);
            use ndarray_linalg::*;
            self.A.solveh_inplace(b).unwrap();
        }

        fn ctest<S1, S2, S3>(
            &self,
            y: &ArrayBase<S1, Ix1>,
            del: &ArrayBase<S2, Ix1>,
            tol: <Self as ModelSpec>::Scalar,
            ewt: &ArrayBase<S3, Ix1>,
        ) -> Result<bool, Error>
        where
            S1: Data<Elem = <Self as ModelSpec>::Scalar>,
            S2: Data<Elem = <Self as ModelSpec>::Scalar>,
            S3: Data<Elem = <Self as ModelSpec>::Scalar>,
        {
            use crate::traits::NormRms;
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
        let Y = array![0.785196933062355226, 0.496611392944656396, 0.369922830745872357];

        let mut p = TestProblem {
            A: Array::zeros((3, 3)),
            x: Array::zeros(3),
        };

        // set initial guess
        let y0 = array![0.5, 0.5, 0.5];

        let mut y = Array::zeros(3);

        // set weights
        let w = array![1.0, 1.0, 1.0];

        let mut newton = Newton::new(&p);
        newton.solve(&mut p, &y0, &mut y, &w, 1e-2, true);

        // print the solution
        println!("Solution: y = {:?}", y);

        // print the solution error
        println!("Solution Error = {:?}", &y - &Y);

        assert_nearly_eq!(y, Y);
    }
}
