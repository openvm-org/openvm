use std::{array, borrow::BorrowMut, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{core::NewFieldArithmeticCoreChip, NewFieldArithmeticChip};
use crate::{
    arch::{
        instructions::{AluOpcode, FieldArithmeticOpcode},
        testing::{memory::gen_pointer, VmChipTestBuilder},
        InstructionExecutor, VmChip, VmChipWrapper,
    },
    kernels::{adapters::native_adapter::NativeAdapterChip, core::BYTE_XOR_BUS},
    system::program::Instruction,
};

type F = BabyBear;

fn run_test_case<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: FieldArithmeticOpcode,
    a: F,
    b: F,
    c: F,
    d: F,
    e: F,
    f: F,
) {
    tester.write::<1>(
        e.as_canonical_u32() as usize,
        b.as_canonical_u32() as usize,
        [1].map(F::from_canonical_u32),
    );
    tester.write::<1>(
        f.as_canonical_u32() as usize,
        c.as_canonical_u32() as usize,
        [1].map(F::from_canonical_u32),
    );

    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [a, b, c, d, e, f].map(|x| x.as_canonical_u32() as usize),
        ),
    );

    assert_eq!(
        F::from_canonical_u32(2u32),
        tester.read::<1>(d.as_canonical_u32() as usize, a.as_canonical_u32() as usize)[0]
    );
}

#[test]
fn new_field_arithmetic_air_test() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = NewFieldArithmeticChip::new(
        NativeAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        NewFieldArithmeticCoreChip::new(0),
        tester.memory_controller(),
    );

    run_test_case(
        &mut tester,
        &mut chip,
        FieldArithmeticOpcode::ADD,
        F::from_canonical_u32(1u32),
        F::from_canonical_u32(2u32),
        F::from_canonical_u32(3u32),
        F::from_canonical_u32(4u32),
        F::from_canonical_u32(5u32),
        F::from_canonical_u32(6u32),
    );

    run_test_case(
        &mut tester,
        &mut chip,
        FieldArithmeticOpcode::ADD,
        F::from_canonical_u32(1u32),
        F::from_canonical_u32(2u32),
        F::from_canonical_u32(3u32),
        F::from_canonical_u32(4u32),
        F::from_canonical_u32(5u32),
        F::from_canonical_u32(6u32),
    );

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
