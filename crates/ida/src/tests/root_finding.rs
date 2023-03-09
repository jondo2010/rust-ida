use approx::assert_relative_eq;
use nalgebra::Dyn;

use super::get_serialized_ida;
use crate::IdaRootData;

fn test_assert_eq(left: &IdaRootData<f64, Dyn>, right: &IdaRootData<f64, Dyn>) {
    assert_eq!(left.ida_iroots, right.ida_iroots);
    assert_eq!(left.ida_rootdir, right.ida_rootdir);
    assert_relative_eq!(left.ida_tlo, right.ida_tlo, epsilon = 1e-12);
    assert_relative_eq!(left.ida_thi, right.ida_thi, epsilon = 1e-12);
    assert_relative_eq!(left.ida_trout, right.ida_trout, epsilon = 1e-12);
    assert_relative_eq!(left.ida_glo, right.ida_glo);
    assert_relative_eq!(left.ida_ghi, right.ida_ghi);
    assert_relative_eq!(left.ida_grout, right.ida_grout);
    assert_relative_eq!(left.ida_toutc, right.ida_toutc);
    assert_relative_eq!(left.ida_ttol, right.ida_ttol);
    assert_eq!(left.ida_taskc, right.ida_taskc);
    assert_eq!(left.ida_irfnd, right.ida_irfnd);
    assert_eq!(left.ida_nge, right.ida_nge);
    assert_eq!(left.ida_gactive, right.ida_gactive);
    assert_eq!(left.ida_mxgnull, right.ida_mxgnull);
}

#[test]
fn test_rcheck1() {
    let mut ida_pre = get_serialized_ida("rcheck1_pre");
    let ida_post = get_serialized_ida("rcheck1_post");
    ida_pre.r_check1().unwrap();
    assert_relative_eq!(ida_pre.roots.ida_ttol, ida_post.roots.ida_ttol);
}

#[test]
fn test_rcheck3() {
    let mut ida_pre = get_serialized_ida("rcheck3_pre");
    dbg!(ida_pre.roots.ida_nge);
    dbg!(ida_pre.nlp.ida_tn);
    dbg!(ida_pre.roots.ida_toutc);

    //let ida_post = get_serialized_ida("rcheck3_post");
    //let res = ida_pre.r_check3().unwrap();
    //assert_eq!(res, crate::private::RootStatus::RootFound);
    //dbg!(ida_pre.get_root_info());
    //test_assert_eq(&ida_pre.roots, &ida_post.roots);
}
