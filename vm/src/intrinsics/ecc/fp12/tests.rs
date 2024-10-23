use ax_ecc_primitives::test_utils::bn254_prime;
use p3_baby_bear::BabyBear;

use crate::{arch::{testing::VmChipTestBuilder, ExecutionBridge}, intrinsics::ecc_v2::FIELD_ELEMENT_BITS};

use super::Fp12MultiplyCoreChip;

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
type F = BabyBear;

#[test]
fn test_fp12_multiply() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = bn254_prime();
    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let core = Fp12MultiplyCoreChip::new(
        modulus.clone(),
        NUM_LIMBS,
        LIMB_BITS,
        FIELD_ELEMENT_BITS - 1,
        tester.memory_controller().borrow().range_checker.bus(),
        FP12Opcode::default_offset(),

    );
    let adapter = Rv32VecHeapAdapterChip
}
