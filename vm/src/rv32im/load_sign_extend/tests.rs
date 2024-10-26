use std::{array, borrow::BorrowMut};

use afs_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, Chip, ChipUsageGetter,
};
use ax_sdk::{config::setup_tracing, utils::create_seeded_rng};
use axvm_instructions::instruction::Instruction;
use num_traits::WrappingSub;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{run_write_data_sign_extend, LoadSignExtendCoreChip, Rv32LoadSignExtendChip};
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
        adapters::{compose, Rv32LoadStoreAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        load_sign_extend::LoadSignExtendCoreCols,
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
    chip: &mut Rv32LoadSignExtendChip<F>,
    rng: &mut StdRng,
    opcode: Rv32LoadStoreOpcode,
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
    let a = gen_pointer(rng, 4);
    let b = gen_pointer(rng, 4);

    let ptr_val = imm_ext.wrapping_add(compose(rs1));
    tester.write(1, b, rs1);

    let some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 8))));
    let read_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << 8))));
    tester.write(1, a, some_prev_data);
    tester.write(2, ptr_val as usize, read_data);

    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize + Rv32LoadStoreOpcode::default_offset(),
            [a, b, imm as usize, 1, 2],
        ),
    );

    let write_data = run_write_data_sign_extend::<_, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        opcode,
        read_data,
        some_prev_data,
    );
    assert_eq!(write_data, tester.read::<4>(1, a));
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
    let core =
        LoadSignExtendCoreChip::new(range_checker_chip, Rv32LoadStoreOpcode::default_offset());
    let mut chip = Rv32LoadSignExtendChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 1;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADB, None, None);
        // set_and_execute(&mut tester, &mut chip, &mut rng, LOADH, None, None);
    }

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
    read_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    most_sig_bit: Option<u32>,
    opcodes: Option<[bool; 2]>,
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
    let core = LoadSignExtendCoreChip::new(
        range_checker_chip.clone(),
        Rv32LoadStoreOpcode::default_offset(),
    );
    let adapter_width = BaseAir::<F>::width(adapter.air());
    let mut chip = Rv32LoadSignExtendChip::<F>::new(adapter, core, tester.memory_controller());

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None);

    let loadstore_trace_width = chip.trace_width();
    let mut chip_input = chip.generate_air_proof_input();
    let loadstore_trace = chip_input.raw.common_main.as_mut().unwrap();
    {
        let mut trace_row = loadstore_trace.row_slice(0).to_vec();

        let (_, core_row) = trace_row.split_at_mut(adapter_width);

        let core_cols: &mut LoadSignExtendCoreCols<F, RV32_REGISTER_NUM_LIMBS> =
            core_row.borrow_mut();

        if let Some(read_data) = read_data {
            core_cols.read_data = read_data.map(F::from_canonical_u32);
        }

        if let Some(most_sig_bit) = most_sig_bit {
            core_cols.most_sig_bit = F::from_canonical_u32(most_sig_bit);
        }

        if let Some(opcodes) = opcodes {
            core_cols.opcode_loadb_flag = F::from_bool(opcodes[0]);
            core_cols.opcode_loadh_flag = F::from_bool(opcodes[1]);
        }
        *loadstore_trace = RowMajorMatrix::new(trace_row, loadstore_trace_width);
    }

    drop(range_checker_chip);
    disable_debug_builder();
    let tester = tester.build().load_air_proof_input(chip_input).finalize();
    tester.simple_test_with_expected_error(expected_error);
}

#[test]
fn negative_loadstore_tests() {
    run_negative_loadstore_test(
        LOADB,
        Some([92, 187, 45, 118]),
        None,
        None,
        VerificationError::NonZeroCumulativeSum,
    );

    run_negative_loadstore_test(
        LOADB,
        Some([5, 132, 77, 250]),
        Some(1),
        None,
        VerificationError::NonZeroCumulativeSum,
    );

    run_negative_loadstore_test(
        LOADH,
        None,
        None,
        Some([true, false]),
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
    let core =
        LoadSignExtendCoreChip::new(range_checker_chip, Rv32LoadStoreOpcode::default_offset());
    let mut chip = Rv32LoadSignExtendChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADB, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LOADH, None, None);
    }
}

#[test]
fn solve_loadh_sanity_test() {
    let read_data = [34, 159, 237, 112].map(F::from_canonical_u32);
    let prev_data = [94, 183, 56, 241].map(F::from_canonical_u32);
    let write_data = run_write_data_sign_extend::<_, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        LOADH, read_data, prev_data,
    );
    assert_eq!(write_data, [34, 159, 255, 255].map(F::from_canonical_u32));
}

#[test]
fn solve_loadb_sanity_test() {
    let read_data = [103, 151, 78, 219].map(F::from_canonical_u32);
    let prev_data = [53, 180, 29, 244].map(F::from_canonical_u32);
    let write_data = run_write_data_sign_extend::<_, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        LOADB, read_data, prev_data,
    );
    assert_eq!(write_data, [103, 0, 0, 0].map(F::from_canonical_u32));
}
