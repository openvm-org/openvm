use super::core::run_divrem;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

#[test]
fn run_divrem_unsigned_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV32_NUM_LIMBS] = [245, 168, 6, 0];
    let r: [u32; RV32_NUM_LIMBS] = [171, 4, 0, 0];

    let (res_q, res_r, x_sign, y_sign) =
        run_divrem::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(false, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert_eq!(x_sign, 0);
    assert_eq!(y_sign, 0);
}

#[test]
fn run_divrem_unsigned_zero_divisor_test() {
    let x: [u32; RV32_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV32_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign) =
        run_divrem::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(false, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert_eq!(x_sign, 0);
    assert_eq!(y_sign, 0);
}

#[test]
fn run_divrem_signed_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV32_NUM_LIMBS] = [74, 60, 255, 255];
    let r: [u32; RV32_NUM_LIMBS] = [212, 240, 255, 255];

    let (res_q, res_r, x_sign, y_sign) = run_divrem::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(true, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert_eq!(x_sign, 1);
    assert_eq!(y_sign, 0);
}

#[test]
fn run_divrem_signed_zero_divisor_test() {
    let x: [u32; RV32_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV32_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign) = run_divrem::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(true, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert_eq!(x_sign, 1);
    assert_eq!(y_sign, 0);
}

#[test]
fn run_divrem_signed_overflow_test() {
    let x: [u32; RV32_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV32_NUM_LIMBS] = [255, 255, 255, 255];
    let r: [u32; RV32_NUM_LIMBS] = [0, 0, 0, 0];

    let (res_q, res_r, x_sign, y_sign) = run_divrem::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(true, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(x[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert_eq!(x_sign, 1);
    assert_eq!(y_sign, 1);
}

#[test]
fn run_divrem_signed_min_dividend_test() {
    let x: [u32; RV32_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV32_NUM_LIMBS] = [123, 34, 255, 255];
    let q: [u32; RV32_NUM_LIMBS] = [236, 147, 0, 0];
    let r: [u32; RV32_NUM_LIMBS] = [156, 149, 255, 255];

    let (res_q, res_r, x_sign, y_sign) = run_divrem::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(true, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert_eq!(x_sign, 1);
    assert_eq!(y_sign, 1);
}
