use crate::{
    constants::*,
    ida_nls::{self, IdaNLProblem},
    tol_control::{self, TolControl},
    traits::{Jacobian, Residual, Root},
    Error, Ida, IdaCounters, IdaTask,
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, DimName, DimNameAdd, Dyn, Matrix, OMatrix,
    OVector, Storage, StorageMut, Vector, U1, U6,
};
use nonlinear::NLSolver;

use crate::traits::{IdaProblem, IdaReal};

impl<T, D, P, LS, NLS> Ida<T, D, P, LS, NLS>
where
    T: IdaReal,
    D: Dim + DimName,
    P: IdaProblem<T, D>,
    LS: linear::LSolver<T, D>,
    NLS: nonlinear::NLSolver<T, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, D, Const<MXORDP1>> + Allocator<T, D> + Allocator<bool, D>,
{
    /// Creates a new Ida solver given initial Arrays of `yy0` and `yyp`.
    ///
    /// *Panics" if ModelSpec::Scalar is unable to convert any constant initialization value.
    pub fn new<SA, SB>(
        problem: P,
        ls: LS,
        yy0: &Matrix<T, D, U1, SA>,
        yp0: &Matrix<T, D, U1, SB>,
        tol_control: TolControl<T, D>,
    ) -> Self
    where
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>,
    {
        // Initialize the phi array
        let mut ida_phi = OMatrix::<T, D, Const<MXORDP1>>::zeros();
        ida_phi.set_column(0, yy0);
        ida_phi.set_column(1, yp0);

        #[cfg(feature = "data_trace")]
        {
            let mut data_trace = std::fs::File::create("roberts_rs.json").unwrap();
            data_trace.write_all(b"{\"data\":[\n").unwrap();
        }

        //IDAResFn res, realtype t0, N_Vector yy0, N_Vector yp0
        Self {
            ida_setup_done: false,

            tol_control,

            // Set default values for integrator optional inputs
            ida_maxord: MAXORD_DEFAULT as usize,
            ida_mxstep: MXSTEP_DEFAULT as u64,
            ida_hmax_inv: T::from(HMAX_INV_DEFAULT).unwrap(),
            ida_hin: T::zero(),
            ida_eps_newt: T::zero(),
            ida_epcon: T::from(EPCON).unwrap(),
            ida_maxnef: MXNEF as u64,
            ida_maxncf: MXNCF as u64,
            ida_suppressalg: false,
            //ida_id          = NULL;
            ida_constraints: None,

            ida_cjlast: T::zero(),

            // set the saved value maxord_alloc
            //ida_maxord_alloc = MAXORD_DEFAULT;

            // Set default values for IC optional inputs
            //ida_epiccon : T::from(0.01 * EPCON).unwrap(),
            //ida_maxnh   : MAXNH,
            //ida_maxnj   = MAXNJ;
            //ida_maxnit  = MAXNI;
            //ida_maxbacks  = MAXBACKS;
            //ida_lsoff   = SUNFALSE;
            //ida_steptol = SUNRpowerR(self.ida_uround, TWOTHIRDS);
            ida_phi,

            ida_psi: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_alpha: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_beta: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_sigma: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_gamma: OVector::<T, Const<MXORDP1>>::zeros(),

            ida_delta: OVector::<T, D>::zeros(),
            ida_id: Some(OVector::<bool, D>::from_element(false)),

            // Initialize all the counters and other optional output values
            counters: IdaCounters {
                ida_nst: 0,
                ida_ncfn: 0,
                ida_netf: 0,
                ida_nni: 0,

                ida_nre: 0,
            },

            ida_kused: 0,
            ida_hused: T::zero(),
            ida_tolsf: T::one(),
            ida_nge: 0,

            // Initialize root-finding variables
            ida_irfnd: false,
            //ida_glo: Array::zeros(problem.num_roots()),
            //ida_ghi: Array::zeros(problem.num_roots()),
            //ida_grout: Array::zeros(problem.num_roots()),
            //ida_iroots: Array::zeros(problem.num_roots()),
            //ida_rootdir: Array::zeros(problem.num_roots()),
            //ida_nrtfn: problem.num_roots(),
            ida_mxgnull: 1,
            ida_tlo: T::zero(),
            ida_ttol: T::zero(),
            //ida_gactive: Array::from_elem(problem.num_roots(), false),
            ida_taskc: IdaTask::Normal,
            ida_toutc: T::zero(),
            ida_thi: T::zero(),
            ida_trout: T::zero(),

            // Not from ida.c...
            ida_ee: OVector::<T, D>::zeros(),

            ida_tstop: None,

            ida_kk: 0,
            ida_knew: 0,
            ida_phase: 0,
            ida_ns: 0,

            ida_rr: T::zero(),
            ida_tretlast: T::zero(),
            ida_h0u: T::zero(),
            ida_hh: T::zero(),

            ida_cvals: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_dvals: OVector::<T, Const<MAXORD_DEFAULT>>::zeros(),

            //ida_zvecs: Array::zeros((MXORDP1, yy0.shape()[0])),

            // Initialize nonlinear solver
            nls: NLS::new(MAXNLSIT),
            nlp: ida_nls::IdaNLProblem::new(problem, ls, yy0, yp0),

            #[cfg(feature = "data_trace")]
            data_trace,
        }
    }

    //-----------------------------------------------------------------
    // Interpolated output
    //-----------------------------------------------------------------

    /// IDAGetDky
    ///
    /// This routine evaluates the k-th derivative of y(t) as the value of the k-th derivative of the interpolating
    /// polynomial at the independent variable t, and stores the results in the vector dky. It uses the current
    /// independent variable value, tn, and the method order last used, kused.
    ///
    /// The return values are:
    ///   IDA_SUCCESS       if t is legal
    ///   IDA_BAD_T         if t is not within the interval of the last step taken
    ///   IDA_BAD_DKY       if the dky vector is NULL
    ///   IDA_BAD_K         if the requested k is not in the range [0,order used]
    ///   IDA_VTolCTOROP_ERR  if the fused vector operation fails
    pub fn get_dky<S>(&self, t: T, k: usize, dky: &mut Matrix<T, D, U1, S>) -> Result<(), Error>
    where
        S: StorageMut<T, D, U1>,
    {
        if (k < 0) || (k > self.ida_kused) {
            Err(Error::BadK {})?
            //IDAProcessError(IDA_mem, IDA_BAD_K, "IDA", "IDAGetDky", MSG_BAD_K);
            //return(IDA_BAD_K);
        }

        // Check t for legality.  Here tn - hused is t_{n-1}.
        let tfuzz = T::hundred()
            * T::from(f64::EPSILON).unwrap()
            * (self.nlp.ida_tn.abs() + self.ida_hh.abs())
            * self.ida_hh.signum();
        let tp = self.nlp.ida_tn - self.ida_hused - tfuzz;

        if (t - tp) * self.ida_hh < T::zero() {
            Err(Error::BadTimeValue {
                t: t.to_f64().unwrap(),
                tdiff: (self.nlp.ida_tn - self.ida_hused).to_f64().unwrap(),
                tcurr: self.nlp.ida_tn.to_f64().unwrap(),
            })?
            //IDAProcessError(IDA_mem, IDA_BAD_T, "IDA", "IDAGetDky", MSG_BAD_T, t, self.nlp.ida_tn-self.ida_hused, self.nlp.ida_tn);
            //return(IDA_BAD_T);
        }

        // Initialize the c_j^(k) and c_k^(k-1)
        let mut cjk = OVector::<T, Const<MXORDP1>>::zeros();
        let mut cjk_1 = OVector::<T, Const<MXORDP1>>::zeros();

        let delt = t - self.nlp.ida_tn;
        let mut psij_1 = T::zero();

        for i in 0..k + 1 {
            let scalar_i = T::from(i as f64).unwrap();
            // The below reccurence is used to compute the k-th derivative of the solution:
            //    c_j^(k) = ( k * c_{j-1}^(k-1) + c_{j-1}^{k} (Delta+psi_{j-1}) ) / psi_j
            //
            //    Translated in indexes notation:
            //    cjk[j] = ( k*cjk_1[j-1] + cjk[j-1]*(delt+psi[j-2]) ) / psi[j-1]
            //
            //    For k=0, j=1: c_1 = c_0^(-1) + (delt+psi[-1]) / psi[0]
            //
            //    In order to be able to deal with k=0 in the same way as for k>0, the
            //    following conventions were adopted:
            //      - c_0(t) = 1 , c_0^(-1)(t)=0
            //      - psij_1 stands for psi[-1]=0 when j=1
            //                      for psi[j-2]  when j>1
            if i == 0 {
                cjk[i] = T::one();
            } else {
                //                                                i       i-1          1
                // c_i^(i) can be always updated since c_i^(i) = -----  --------  ... -----
                //                                               psi_j  psi_{j-1}     psi_1
                cjk[i] = cjk[i - 1] * scalar_i / self.ida_psi[i - 1];
                psij_1 = self.ida_psi[i - 1];
            }

            // update c_j^(i)
            //j does not need to go till kused
            //for(j=i+1; j<=self.ida_kused-k+i; j++) {
            for j in i + 1..self.ida_kused - k + 1 + 1 {
                cjk[j] =
                    (scalar_i * cjk_1[j - 1] + cjk[j - 1] * (delt + psij_1)) / self.ida_psi[j - 1];
                psij_1 = self.ida_psi[j - 1];
            }

            // save existing c_j^(i)'s
            //for(j=i+1; j<=self.ida_kused-k+i; j++)
            for j in i + 1..self.ida_kused - k + 1 + 1 {
                cjk_1[j] = cjk[j];
            }
        }

        // Compute sum (c_j(t) * phi(t))
        // Sum j=k to j<=self.ida_kused
        //retval = N_VLinearCombination( self.ida_kused - k + 1, cjk + k, self.ida_phi + k, dky);

        let phi = self.ida_phi.columns(k, self.ida_kused + 1);
        //let phi = self .ida_phi .slice_axis(Axis(0), Slice::from(k..self.ida_kused + 1));

        let cvals = cjk.rows(k, self.ida_kused + 1);

        let x = std::ops::Mul::mul(phi, &cvals);
        dbg!(x);

        let x = phi.tr_mul(&cvals);
        //dky.tr_copy_from(&x);
        dbg!(x);

        // We manually broadcast here so we can turn it into a column vec
        /*
        let cvals = cjk.slice(s![k..self.ida_kused + 1]);
        let cvals = cvals
            .broadcast((phi.len_of(Axis(1)), phi.len_of(Axis(0))))
            .unwrap()
            .reversed_axes();

        dky.assign(&(&phi * &cvals).sum_axis(Axis(0)));
        */

        Ok(())
    }

    /// IDAInitialSetup
    ///
    /// This routine is called by `solve` once at the first step. It performs all checks on optional inputs and inputs
    /// to `init`/`reinit` that could not be done before.
    ///
    /// If no error is encountered, IDAInitialSetup returns IDA_SUCCESS. Otherwise, it returns an error flag and
    /// reported to the error handler function.
    fn initial_setup(&mut self) {
        //booleantype conOK;
        //int ier;

        // Initial error weight vector
        self.tol_control
            .ewt_set(&self.ida_phi.column(0), &mut self.nlp.ida_ewt);

        /*
        // Check to see if y0 satisfies constraints.
        if (IDA_mem->ida_constraintsSet) {
          conOK = N_VConstrMask(IDA_mem->ida_constraints, IDA_mem->ida_phi[0], IDA_mem->ida_tempv2);
          if (!conOK) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_Y0_FAIL_CONSTR);
            return(IDA_ILL_INPUT);
          }
        }

        // Call linit function if it exists.
        if (IDA_mem->ida_linit != NULL) {
          ier = IDA_mem->ida_linit(IDA_mem);
          if (ier != 0) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_LINIT_FAIL);
            return(IDA_LINIT_FAIL);
          }
        }
        */

        //return(IDA_SUCCESS);
    }
}
