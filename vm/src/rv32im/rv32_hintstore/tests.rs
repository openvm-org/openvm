use std::{array, borrow::BorrowMut, sync::Arc};

use afs_primitives::xor::XorLookupChip;
use afs_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, Chip, ChipUsageGetter,
};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_traits::WrappingSub;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng};

use super::{run_hint_store_data, Rv32HintStoreChip, Rv32HintStoreCoreChip};
use crate::{
    arch::{
        instructions::{
            Rv32HintStoreOpcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder},
        VmAdapterChip,
    },
    rv32im::{
        adapters::{compose, Rv32HintStoreAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        rv32_hintstore::Rv32HintStoreCoreCols,
    },
    system::{
        program::Instruction,
        vm::{chip_set::BYTE_XOR_BUS, Streams},
    },
};

const IMM_BITS: usize = 16;

type F = BabyBear;

fn into_limbs<const NUM_LIMBS: usize, const LIMB_BITS: usize>(num: u32) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| (num >> (LIMB_BITS * i)) & ((1 << LIMB_BITS) - 1))
}
fn sign_extend<const IMM_BITS: usize>(num: u32) -> u32 {
    if num & (1 << (IMM_BITS - 1)) != 0 {
        num | (u32::MAX - (1 << IMM_BITS) + 1)
    } else {
        num
    }
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32HintStoreChip<F>,
    rng: &mut StdRng,
    opcode: Rv32HintStoreOpcode,
    rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_ext = sign_extend::<IMM_BITS>(imm);
    let ptr_val = rng.gen_range(
        0..(1
            << (tester
                .memory_controller()
                .borrow()
                .mem_config
                .pointer_max_bits
                - 2)),
    ) << 2;
    let rs1 = rs1
        .unwrap_or(into_limbs::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            ptr_val.wrapping_sub(&imm_ext),
        ))
        .map(F::from_canonical_u32);
    let b = gen_pointer(rng, 4);

    let ptr_val = imm_ext.wrapping_add(compose(rs1));
    tester.write(1, b, rs1);

    let read_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));
    for data in read_data {
        chip.core.streams.lock().hint_stream.push_back(data);
    }

    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize + Rv32HintStoreOpcode::default_offset(),
            [0, b, imm as usize, 1, 2],
        ),
    );

    let write_data = run_hint_store_data(opcode, read_data);
    assert_eq!(write_data, tester.read::<4>(2, ptr_val as usize));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_hintstore_test() {
    setup_tracing();
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let adapter = Rv32HintStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
    );

    let core = Rv32HintStoreCoreChip::new(
        Arc::new(Mutex::new(Streams::default())),
        xor_lookup_chip.clone(),
        Rv32HintStoreOpcode::default_offset(),
    );
    let mut chip = Rv32HintStoreChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW, None, None);
    }

    drop(range_checker_chip);
    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adaptor is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_hintstore_test(
    opcode: Rv32HintStoreOpcode,
    data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    expected_error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let adapter = Rv32HintStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
    );
    let core = Rv32HintStoreCoreChip::new(
        Arc::new(Mutex::new(Streams::default())),
        xor_lookup_chip.clone(),
        Rv32HintStoreOpcode::default_offset(),
    );
    let adapter_width = BaseAir::<F>::width(adapter.air());
    let mut chip = Rv32HintStoreChip::<F>::new(adapter, core, tester.memory_controller());

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None);

    let hintstore_trace_width = chip.trace_width();
    let mut chip_input = chip.generate_air_proof_input();
    let hintstore_trace = chip_input.raw.common_main.as_mut().unwrap();
    {
        let mut trace_row = hintstore_trace.row_slice(0).to_vec();

        let (_, core_row) = trace_row.split_at_mut(adapter_width);

        let core_cols: &mut Rv32HintStoreCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = data {
            core_cols.data = data.map(F::from_canonical_u32);
        }
        *hintstore_trace = RowMajorMatrix::new(trace_row, hintstore_trace_width);
    }

    drop(range_checker_chip);
    disable_debug_builder();
    let tester = tester
        .build()
        .load_air_proof_input(chip_input)
        .load(xor_lookup_chip)
        .finalize();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    let result = tester.simple_test();
    assert_eq!(result.err(), Some(expected_error), "{}", msg);
}

#[test]
fn negative_hintstore_tests() {
    run_negative_hintstore_test(
        HINT_STOREW,
        Some([92, 187, 45, 280]),
        VerificationError::NonZeroCumulativeSum,
    );
}
///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let adapter = Rv32HintStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
    );
    let core = Rv32HintStoreCoreChip::new(
        Arc::new(Mutex::new(Streams::default())),
        xor_lookup_chip.clone(),
        Rv32HintStoreOpcode::default_offset(),
    );
    let mut chip = Rv32HintStoreChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW, None, None);
    }
}

#[test]
fn run_hintstorew_sanity_test() {
    let read_data = [138, 45, 202, 76].map(F::from_canonical_u32);
    let store_write_data = run_hint_store_data(HINT_STOREW, read_data);
    assert_eq!(store_write_data, read_data);
}
