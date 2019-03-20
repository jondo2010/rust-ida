
/// State variables involved in the linear problem
#[derive(Clone, Debug)]
struct IdaLProblem {
    // Linear solver, matrix and vector objects/pointers
/*
    SUNLinearSolver LS;   // generic linear solver object
    SUNMatrix J;          // J = dF/dy + cj*dF/dy'
    N_Vector ytemp;       // temp vector used by IDAAtimesDQ
    N_Vector yptemp;      // temp vector used by IDAAtimesDQ
    N_Vector x;           // temp vector used by the solve function
    N_Vector ycur;        // current y vector in Newton iteration
    N_Vector ypcur;       // current yp vector in Newton iteration
    N_Vector rcur;        // rcur = F(tn, ycur, ypcur)

    // Iterative solver tolerance
    realtype sqrtN;     // sqrt(N)
    realtype eplifac;   // eplifac = linear convergence factor

    // Statistics and associated parameters
    realtype dqincfac;  // dqincfac = optional increment factor in Jv
    long int nje;       // nje = no. of calls to jac
    long int npe;       // npe = total number of precond calls
    long int nli;       // nli = total number of linear iterations
    long int nps;       // nps = total number of psolve calls
    long int ncfl;      // ncfl = total number of convergence failures
    long int nreDQ;     // nreDQ = total number of calls to res
    long int njtsetup;  // njtsetup = total number of calls to jtsetup
    long int njtimes;   // njtimes = total number of calls to jtimes
    long int nst0;      // nst0 = saved nst (for performance monitor)
    long int nni0;      // nni0 = saved nni (for performance monitor)
    long int ncfn0;     // ncfn0 = saved ncfn (for performance monitor)
    long int ncfl0;     // ncfl0 = saved ncfl (for performance monitor)
    long int nwarn;     // nwarn = no. of warnings (for perf. monitor)

    long int last_flag; // last error return flag

    // Preconditioner computation
    // (a) user-provided:
    //     - pdata == user_data
    //     - pfree == NULL (the user dealocates memory)
    // (b) internal preconditioner module
    //     - pdata == ida_mem
    //     - pfree == set by the prec. module and called in idaLsFree
    IDALsPrecSetupFn pset;
    IDALsPrecSolveFn psolve;
    int (*pfree)(IDAMem IDA_mem);
    void *pdata;

    // Jacobian times vector compuation
    // (a) jtimes function provided by the user:
    //     - jt_data == user_data
    //     - jtimesDQ == SUNFALSE
    // (b) internal jtimes
    //     - jt_data == ida_mem
    //     - jtimesDQ == SUNTRUE
    booleantype jtimesDQ;
    IDALsJacTimesSetupFn jtsetup;
    IDALsJacTimesVecFn jtimes;
    void *jt_data;
*/
}