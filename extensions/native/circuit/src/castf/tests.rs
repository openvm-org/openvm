use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestChipHarness, VmChipTestBuilder},
    MemoryConfig,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_native_compiler::{conversion::AS, CastfOpcode};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{CastFChip, CastFCoreAir, CastFCoreCols, CastFStep, LIMB_BITS};
use crate::{
    adapters::{ConvertAdapterAir, ConvertAdapterCols, ConvertAdapterFiller, ConvertAdapterStep},
    castf::run_castf,
    test_utils::write_native_array,
    CastFAir, CastFCoreFiller, CASTF_MAX_BITS,
};

const MAX_INS_CAPACITY: usize = 128;
const READ_SIZE: usize = 1;
const WRITE_SIZE: usize = 4;
type F = BabyBear;
type Harness = TestChipHarness<F, CastFStep, CastFAir, CastFChip<F>>;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker().clone();
    let air = CastFAir::new(
        ConvertAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        CastFCoreAir::new(range_checker.bus()),
    );
    let executor = CastFStep::new(ConvertAdapterStep::<READ_SIZE, WRITE_SIZE>::new());
    let chip = CastFChip::<F>::new(
        CastFCoreFiller::new(ConvertAdapterFiller, range_checker),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness,
    rng: &mut StdRng,
    b: Option<F>,
) {
    let b_val = b.unwrap_or(F::from_canonical_u32(rng.gen_range(0..1 << CASTF_MAX_BITS)));
    let b_ptr = write_native_array(tester, rng, Some([b_val])).1;

    let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    tester.execute(
        harness,
        &Instruction::from_usize(
            CastfOpcode::CASTF.global_opcode(),
            [a, b_ptr, 0, RV32_MEMORY_AS as usize, AS::Native as usize],
        ),
    );
    let expected = run_castf(b_val.as_canonical_u32());
    let result = tester.read::<RV32_REGISTER_NUM_LIMBS>(RV32_MEMORY_AS as usize, a);
    assert_eq!(result.map(|x| x.as_canonical_u32() as u8), expected);
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn castf_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::volatile(MemoryConfig::default());
    let mut harness = create_test_chip(&tester);
    let num_ops = 100;

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut harness, &mut rng, None);
    }

    set_and_execute(&mut tester, &mut harness, &mut rng, Some(F::ZERO));

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
struct CastFPrankValues {
    pub in_val: Option<u32>,
    pub out_val: Option<[u32; 4]>,
    pub a_pointer: Option<u32>,
    pub b_pointer: Option<u32>,
}

fn run_negative_castf_test(prank_vals: CastFPrankValues, b: Option<F>, error: VerificationError) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::volatile(MemoryConfig::default());

    let mut harness = create_test_chip(&tester);
    set_and_execute(&mut tester, &mut harness, &mut rng, b);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);

    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = values.split_at_mut(adapter_width);
        let core_cols: &mut CastFCoreCols<F> = core_row.borrow_mut();
        let adapter_cols: &mut ConvertAdapterCols<F, READ_SIZE, WRITE_SIZE> =
            adapter_row.borrow_mut();

        if let Some(in_val) = prank_vals.in_val {
            // TODO: in_val is actually never used in the AIR, should remove it
            core_cols.in_val = F::from_canonical_u32(in_val);
        }
        if let Some(out_val) = prank_vals.out_val {
            core_cols.out_val = out_val.map(F::from_canonical_u32);
        }
        if let Some(a_pointer) = prank_vals.a_pointer {
            adapter_cols.a_pointer = F::from_canonical_u32(a_pointer);
        }
        if let Some(b_pointer) = prank_vals.b_pointer {
            adapter_cols.b_pointer = F::from_canonical_u32(b_pointer);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(error);
}

#[test]
fn casf_invalid_out_val_test() {
    run_negative_castf_test(
        CastFPrankValues {
            out_val: Some([2 << LIMB_BITS, 0, 0, 0]),
            ..Default::default()
        },
        Some(F::from_canonical_u32(2 << LIMB_BITS)),
        VerificationError::ChallengePhaseError,
    );

    let prime = F::NEG_ONE.as_canonical_u32() + 1;
    run_negative_castf_test(
        CastFPrankValues {
            out_val: Some(prime.to_le_bytes().map(|x| x as u32)),
            ..Default::default()
        },
        Some(F::ZERO),
        VerificationError::ChallengePhaseError,
    );
}

#[test]
fn negative_convert_adapter_test() {
    // overflowing the memory pointer
    run_negative_castf_test(
        CastFPrankValues {
            b_pointer: Some(1 << 30),
            ..Default::default()
        },
        None,
        VerificationError::ChallengePhaseError,
    );

    // Memory address space pointer has to be 4-byte aligned
    run_negative_castf_test(
        CastFPrankValues {
            a_pointer: Some(1),
            ..Default::default()
        },
        None,
        VerificationError::ChallengePhaseError,
    );
}

#[should_panic]
#[test]
fn castf_overflow_in_val_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::volatile(MemoryConfig::default());
    let mut harness = create_test_chip(&tester);
    set_and_execute(&mut tester, &mut harness, &mut rng, Some(F::NEG_ONE));
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn castf_sanity_test() {
    let b = 160558167;
    let expected = [87, 236, 145, 9];
    assert_eq!(run_castf(b), expected);
}
