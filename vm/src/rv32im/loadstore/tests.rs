use std::{array, borrow::BorrowMut};

use afs_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, Chip, ChipUsageGetter,
};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use num_traits::WrappingSub;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{solve_loadstore, LoadStoreCoreChip, Rv32LoadStoreChip};
use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder},
        VmAdapterChip,
    },
    rv32im::{
        adapters::{compose, Rv32LoadStoreAdapterChip, RV32_REGISTER_NUM_LANES},
        loadstore::LoadStoreCoreCols,
    },
    system::program::Instruction,
};

const IMM_BITS: usize = 16;
const ADDR_BITS: usize = 29;

type F = BabyBear;

fn into_limbs(num: u32) -> [u32; 4] {
    array::from_fn(|i| (num >> (8 * i)) & 255)
}
fn sign_extend(num: u32) -> u32 {
    if num & 0x8000 != 0 {
        num | 0xffff0000
    } else {
        num
    }
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32LoadStoreChip<F>,
    rng: &mut StdRng,
    opcode: Rv32LoadStoreOpcode,
    rs1: Option<[u32; RV32_REGISTER_NUM_LANES]>,
    imm: Option<u32>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_ext = sign_extend(imm);

    let ptr_val = rng.gen_range(0..(1 << (ADDR_BITS - 2))) << 2;
    let rs1 = rs1.unwrap_or(into_limbs(ptr_val.wrapping_sub(&imm_ext)));
    let rs1 = rs1.map(F::from_canonical_u32);
    let a = gen_pointer(rng, 4);
    let b = gen_pointer(rng, 4);

    let ptr_val = imm_ext.wrapping_add(compose(rs1));
    tester.write(1, b, rs1);

    let is_load = [LOADW, LOADH, LOADB, LOADHU, LOADBU, HINTLOAD_RV32].contains(&opcode);
    let some_prev_data: [F; RV32_REGISTER_NUM_LANES] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 8))));
    if is_load {
        tester.write(1, a, some_prev_data);
    } else {
        tester.write(2, ptr_val as usize, some_prev_data);
    }

    let read_data: [F; RV32_REGISTER_NUM_LANES] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 8))));
    if is_load {
        tester.write(2, ptr_val as usize, read_data);
    } else {
        tester.write(1, a, read_data);
    }

    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize + Rv32LoadStoreOpcode::default_offset(),
            [a, b, imm as usize, 1, 2],
        ),
    );

    let write_data = solve_loadstore(opcode, read_data, some_prev_data);
    if is_load && opcode != HINTLOAD_RV32 {
        assert_eq!(write_data, tester.read::<4>(1, a));
    } else if !is_load {
        assert_eq!(write_data, tester.read::<4>(2, ptr_val as usize));
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_loadstore_test() {
    setup_tracing();
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let adapter = Rv32LoadStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
        Rv32LoadStoreOpcode::default_offset(),
    );
    let inner = LoadStoreCoreChip::new(Rv32LoadStoreOpcode::default_offset());
    let mut chip = Rv32LoadStoreChip::<F>::new(adapter, inner, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADW, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADBU, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADHU, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREW, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREB, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREH, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, HINTLOAD_RV32, None, None);
    }

    drop(range_checker_chip);
    let tester = tester.build().load(chip).finalize();
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
fn run_negative_loadstore_test(
    opcode: Rv32LoadStoreOpcode,
    read_data: Option<[u32; RV32_REGISTER_NUM_LANES]>,
    prev_data: Option<[u32; RV32_REGISTER_NUM_LANES]>,
    opcodes: Option<[bool; 7]>,
    expected_error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let adapter = Rv32LoadStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
        Rv32LoadStoreOpcode::default_offset(),
    );
    let inner = LoadStoreCoreChip::new(Rv32LoadStoreOpcode::default_offset());
    let adapter_width = BaseAir::<F>::width(adapter.air());
    let mut chip = Rv32LoadStoreChip::<F>::new(adapter, inner, tester.memory_controller());

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None);

    let loadstore_trace_width = chip.trace_width();
    let mut chip_input = chip.generate_air_proof_input();
    let loadstore_trace = chip_input.raw.common_main.as_mut().unwrap();
    {
        let mut trace_row = loadstore_trace.row_slice(0).to_vec();

        let (_, core_row) = trace_row.split_at_mut(adapter_width);

        let core_cols: &mut LoadStoreCoreCols<F, RV32_REGISTER_NUM_LANES> = core_row.borrow_mut();

        if let Some(read_data) = read_data {
            core_cols.read_data = read_data.map(F::from_canonical_u32);
        }

        if let Some(prev_data) = prev_data {
            core_cols.prev_data = prev_data.map(F::from_canonical_u32);
        }

        if let Some(opcodes) = opcodes {
            core_cols.opcode_loadw_flag = F::from_bool(opcodes[0]);
            core_cols.opcode_loadhu_flag = F::from_bool(opcodes[1]);
            core_cols.opcode_loadbu_flag = F::from_bool(opcodes[2]);
            core_cols.opcode_storew_flag = F::from_bool(opcodes[3]);
            core_cols.opcode_storeh_flag = F::from_bool(opcodes[4]);
            core_cols.opcode_storeb_flag = F::from_bool(opcodes[5]);
            core_cols.opcode_hintload_flag = F::from_bool(opcodes[6]);
        }
        *loadstore_trace = RowMajorMatrix::new(trace_row, loadstore_trace_width);
    }

    drop(range_checker_chip);
    disable_debug_builder();
    let tester = tester.build().load_air_proof_input(chip_input).finalize();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    let result = tester.simple_test();
    assert_eq!(result.err(), Some(expected_error), "{}", msg);
}

#[test]
fn negative_loadstore_tests() {
    run_negative_loadstore_test(
        LOADW,
        Some([92, 187, 45, 118]),
        None,
        None,
        VerificationError::NonZeroCumulativeSum,
    );

    run_negative_loadstore_test(
        STOREB,
        None,
        Some([5, 132, 77, 250]),
        None,
        VerificationError::NonZeroCumulativeSum,
    );

    run_negative_loadstore_test(
        LOADHU,
        None,
        None,
        Some([true, false, false, false, false, false, false]),
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
    let adapter = Rv32LoadStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
        Rv32LoadStoreOpcode::default_offset(),
    );
    let inner = LoadStoreCoreChip::new(Rv32LoadStoreOpcode::default_offset());
    let mut chip = Rv32LoadStoreChip::<F>::new(adapter, inner, tester.memory_controller());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADW, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADBU, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADHU, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREW, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREB, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, STOREH, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, HINTLOAD_RV32, None, None);
    }
}

#[test]
fn solve_loadw_storew_sanity_test() {
    let read_data = [138, 45, 202, 76].map(F::from_canonical_u32);
    let prev_data = [159, 213, 89, 34].map(F::from_canonical_u32);
    let store_write_data = solve_loadstore(STOREW, read_data, prev_data);
    let load_write_data = solve_loadstore(LOADW, read_data, prev_data);
    assert_eq!(store_write_data, read_data);
    assert_eq!(load_write_data, read_data);
}

#[test]
fn solve_storeh_sanity_test() {
    let read_data = [250, 123, 67, 198].map(F::from_canonical_u32);
    let prev_data = [144, 56, 175, 92].map(F::from_canonical_u32);
    let write_data = solve_loadstore(STOREH, read_data, prev_data);
    assert_eq!(write_data, [250, 123, 175, 92].map(F::from_canonical_u32));
}

#[test]
fn solve_storeb_sanity_test() {
    let read_data = [221, 104, 58, 147].map(F::from_canonical_u32);
    let prev_data = [199, 83, 243, 12].map(F::from_canonical_u32);
    let write_data = solve_loadstore(STOREB, read_data, prev_data);
    assert_eq!(write_data, [221, 83, 243, 12].map(F::from_canonical_u32));
}

#[test]
fn solve_loadhu_sanity_test() {
    let read_data = [175, 33, 198, 250].map(F::from_canonical_u32);
    let prev_data = [90, 121, 64, 205].map(F::from_canonical_u32);
    let write_data = solve_loadstore(LOADHU, read_data, prev_data);
    assert_eq!(write_data, [175, 33, 0, 0].map(F::from_canonical_u32));
}

#[test]
fn solve_loadbu_sanity_test() {
    let read_data = [131, 74, 186, 29].map(F::from_canonical_u32);
    let prev_data = [142, 67, 210, 88].map(F::from_canonical_u32);
    let write_data = solve_loadstore(LOADBU, read_data, prev_data);
    assert_eq!(write_data, [131, 0, 0, 0].map(F::from_canonical_u32));
}
