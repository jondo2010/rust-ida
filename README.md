Implicit Differential-Algebraic Solver
====================
[![Crate](http://meritbadge.herokuapp.com/ida)](https://crates.io/crates/ida)
[![docs.rs](https://docs.rs/ida/badge.svg)](https://docs.rs/ida)
[![pipeline status](https://gitlab.com/jondo2010/rust-ida/badges/master/pipeline.svg)](https://gitlab.com/jondo2010/rust-ida/commits/master)
[![codecov](https://codecov.io/gl/jondo2010/rust-ida/branch/master/graph/badge.svg)](https://codecov.io/gl/jondo2010/rust-ida)

`ida` is a Rust port of the Implicit Differential-Algebraic solver from the Sundials suite. it is a general purpose solver for the initial value problem (IVP) for systems of differentialalgebraic equations (DAEs).

See:
- https://computation.llnl.gov/projects/sundials/ida
- https://computation.llnl.gov/sites/default/files/public/ida_guide.pdf

The original ida was based on earlier numeric codes (daspk), and written in ANSI-standard C. It gave the user the choice between Newton/direct and Inexact Newton/Krylov (iterative) methods for solving the underlying non-linear system. The Rust version also allows for this choice, but is implemented using Rust Traits at compile-time.

`ida` is implemented using the `ndarray` crate to provide variable length vectors and matrices. The scalar datatype is available for the user as a generic parameter.

Goals
-----
I originally started this project in support of my other work in simulating Functional Mockup Units (FMUs), and hence priority will be to support this use case.

The numeric behavior of the algorithms has been written to match *exactly* the original code using `double` datatypes, while allowing the freedom to refactor data structures, error handling code and generics to take advantage of Rusts strengths.

Status
------
As of version 0.1.0:
* The 'Roberts' example should run numerically identically to the original.
* Only the direct linear solver and Newton non-linear are implemented.
* Only the dense vector/matrix math is implemented.
* Lots of code and garbage comments still need to be cleaned up.
* Plenty of additional features and examples still need to be ported (Krylov solver, Jacobian approximation, Constraints, etc.).

License
-------
BSD3-License, see [LICENSE](LICENSE) file.
