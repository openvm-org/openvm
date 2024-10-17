use afs_primitives::{bigint::utils::secp256k1_coord_prime, ecc::SampleEcPoints};
use ax_ecc_primitives::field_expression::FieldVariableConfig;
use axvm_instructions::UsizeOpcode;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{super::FIELD_ELEMENT_BITS, SwEcAddNeChip};
use crate::{
    arch::{
        instructions::EccOpcode,
        testing::{TestAdapterChip, VmChipTestBuilder},
        ExecutionBridge, VmChipWrapper,
    },
    intrinsics::ecc_v2::sw::SwEcDoubleChip,
    system::program::Instruction,
    utils::biguint_to_limbs_vec,
};

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
type F = BabyBear;

#[derive(Clone)]
struct TestConfig {}
impl FieldVariableConfig for TestConfig {
    fn canonical_limb_bits() -> usize {
        LIMB_SIZE
    }

    fn max_limb_bits() -> usize {
        FIELD_ELEMENT_BITS
    }

    fn num_limbs_per_field_element() -> usize {
        NUM_LIMBS
    }
}

#[test]
fn test_add_ne() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = secp256k1_coord_prime();
    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let core = SwEcAddNeChip::<TestConfig>::new(
        modulus.clone(),
        tester.memory_controller().borrow().range_checker.bus(),
        EccOpcode::default_offset(),
    );
    let mut adapter = TestAdapterChip::new(vec![], vec![None], execution_bridge);

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();

    let p1_x_limbs = biguint_to_limbs_vec(p1_x.clone(), LIMB_SIZE, NUM_LIMBS);
    let p1_y_limbs = biguint_to_limbs_vec(p1_y.clone(), LIMB_SIZE, NUM_LIMBS);
    let p2_x_limbs = biguint_to_limbs_vec(p2_x.clone(), LIMB_SIZE, NUM_LIMBS);
    let p2_y_limbs = biguint_to_limbs_vec(p2_y.clone(), LIMB_SIZE, NUM_LIMBS);
    let interface_reads = [p1_x_limbs, p1_y_limbs, p2_x_limbs, p2_y_limbs].concat();
    adapter.prank_reads.push_back(
        interface_reads
            .into_iter()
            .map(BabyBear::from_canonical_u32)
            .collect(),
    );

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip
        .core
        .air
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[2].0);
    assert_eq!(r[2], SampleEcPoints[2].1);

    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 12;
    let addr_ptr3 = 24;

    let data_as = 2;
    let _address1 = 0;
    let _address2 = 128;
    let _address3 = 256;
    let instruction = Instruction::from_isize(
        chip.core.air.offset + EccOpcode::EC_ADD_NE as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(&mut chip, instruction);
}

#[test]
fn test_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = secp256k1_coord_prime();
    let execution_bridge = ExecutionBridge::new(tester.execution_bus(), tester.program_bus());
    let core = SwEcDoubleChip::<TestConfig>::new(
        modulus.clone(),
        tester.memory_controller().borrow().range_checker.bus(),
        EccOpcode::default_offset(),
    );
    let mut adapter = TestAdapterChip::new(vec![], vec![None], execution_bridge);

    let (p1_x, p1_y) = SampleEcPoints[1].clone();

    let p1_x_limbs = biguint_to_limbs_vec(p1_x.clone(), LIMB_SIZE, NUM_LIMBS);
    let p1_y_limbs = biguint_to_limbs_vec(p1_y.clone(), LIMB_SIZE, NUM_LIMBS);
    let interface_reads = [p1_x_limbs, p1_y_limbs].concat();
    adapter.prank_reads.push_back(
        interface_reads
            .into_iter()
            .map(BabyBear::from_canonical_u32)
            .collect(),
    );

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip.core.air.expr.execute(vec![p1_x, p1_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[3].0);
    assert_eq!(r[2], SampleEcPoints[3].1);

    let ptr_as = 1;
    let addr_ptr1 = 0;
    let addr_ptr2 = 12;
    let addr_ptr3 = 24;

    let data_as = 2;
    let _address1 = 0;
    let _address2 = 128;
    let _address3 = 256;
    let instruction = Instruction::from_isize(
        chip.core.air.offset + EccOpcode::EC_DOUBLE as usize,
        addr_ptr3 as isize,
        addr_ptr1 as isize,
        addr_ptr2 as isize,
        ptr_as as isize,
        data_as as isize,
    );
    tester.execute(&mut chip, instruction);
}
