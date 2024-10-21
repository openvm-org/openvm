use super::core::run_mul;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

#[test]
fn run_mul_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_NUM_LIMBS] = [63, 247, 125, 232];
    let result = run_mul::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(&x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
