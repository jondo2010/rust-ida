#![allow(unused)]

use std::{fs::File, io::Read, path::Path};

use linear::Dense;
use nalgebra::*;

use nonlinear::Newton;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{sundials::SundialsProblem, Ida, IdaProblem};

mod complete_step;
mod get_dky;
mod get_solution;
mod nonlinear_solve;
mod predict;
mod restore;
mod root_finding;
mod set_coeffs;
mod test_error;

fn get_serialized_ida(test_name: &str) -> Ida<f64, SundialsProblem, Dense<Dyn>, Newton<f64, Dyn>> {
    let mut file = File::open(
        std::env::current_dir()
            .unwrap()
            .join(format!("src/tests/data/{test_name}.json")),
    )
    .unwrap();
    let mut s = String::new();
    file.read_to_string(&mut s).unwrap();
    serde_json::from_str(&s).unwrap()
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct Dummy {}

impl IdaProblem<f64> for Dummy {
    type D = U3;
    type R = U3;

    fn res<SA, SB, SC>(
        &self,
        tt: f64,
        yy: &Matrix<f64, Const<3>, U1, SA>,
        yp: &Matrix<f64, Const<3>, U1, SB>,
        rr: &mut Matrix<f64, Const<3>, U1, SC>,
    ) where
        SA: Storage<f64, Const<3>>,
        SB: Storage<f64, Const<3>>,
        SC: StorageMut<f64, Const<3>>,
    {
    }

    fn jac<SA, SB, SC, SD>(
        &self,
        tt: f64,
        cj: f64,
        yy: &Matrix<f64, Const<3>, U1, SA>,
        yp: &Matrix<f64, Const<3>, U1, SB>,
        rr: &Matrix<f64, Const<3>, U1, SC>,
        jac: &mut Matrix<f64, Const<3>, Const<3>, SD>,
    ) where
        SA: Storage<f64, Const<3>>,
        SB: Storage<f64, Const<3>>,
        SC: Storage<f64, Const<3>>,
        SD: StorageMut<f64, Const<3>, Const<3>>,
    {
    }

    fn root<SA, SB, SC>(
        &self,
        t: f64,
        y: &Matrix<f64, Const<3>, U1, SA>,
        yp: &Matrix<f64, Const<3>, U1, SB>,
        gout: &mut Matrix<f64, Const<3>, U1, SC>,
    ) where
        SA: Storage<f64, Const<3>>,
        SB: Storage<f64, Const<3>>,
        SC: StorageMut<f64, Const<3>>,
    {
    }
}
