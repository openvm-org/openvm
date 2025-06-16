use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    VmAirWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_transpiler::Rv32JalrOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::Rv32JalrCoreAir;
use crate::{
    adapters::{
        compose, Rv32JalrAdapterAir, Rv32JalrAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    jalr::{run_jalr, Rv32JalrChip, Rv32JalrCoreCols, Rv32JalrStep},
    test_utils::get_verification_error,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;

type F = BabyBear;

fn into_limbs(num: u32) -> [u32; 4] {
    array::from_fn(|i| (num >> (8 * i)) & 255)
}

fn create_test_chip(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Rv32JalrChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let range_checker_chip = tester.memory_controller().range_checker.clone();

    let mut chip = Rv32JalrChip::<F>::new(
        VmAirWrapper::new(
            Rv32JalrAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            Rv32JalrCoreAir::new(bitwise_bus, range_checker_chip.bus()),
        ),
        Rv32JalrStep::new(
            Rv32JalrAdapterStep::new(),
            bitwise_chip.clone(),
            range_checker_chip.clone(),
        ),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    (chip, bitwise_chip)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32JalrChip<F>,
    rng: &mut StdRng,
    opcode: Rv32JalrOpcode,
    initial_imm: Option<u32>,
    initial_imm_sign: Option<u32>,
    initial_pc: Option<u32>,
    rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
) {
    let imm = initial_imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_sign = initial_imm_sign.unwrap_or(rng.gen_range(0..2));
    let imm_ext = imm + (imm_sign * 0xffff0000);
    let a = rng.gen_range(0..32) << 2;
    let b = rng.gen_range(1..32) << 2;
    let to_pc = rng.gen_range(0..(1 << PC_BITS));

    let rs1 = rs1.unwrap_or(into_limbs((to_pc as u32).wrapping_sub(imm_ext)));
    let rs1 = rs1.map(F::from_canonical_u32);

    tester.write(1, b, rs1);

    let initial_pc = initial_pc.unwrap_or(rng.gen_range(0..(1 << PC_BITS)));
    tester.execute_with_pc(
        chip,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b,
                imm as usize,
                1,
                0,
                (a != 0) as usize,
                imm_sign as usize,
            ],
        ),
        initial_pc,
    );
    let final_pc = tester.execution.last_to_pc().as_canonical_u32();

    let rs1 = compose(rs1);

    let (next_pc, rd_data) = run_jalr(initial_pc, rs1, imm as u16, imm_sign == 1);
    let rd_data = if a == 0 { [0; 4] } else { rd_data };

    assert_eq!(next_pc & !1, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_jalr_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&mut tester);

    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut chip,
            &mut rng,
            JALR,
            None,
            None,
            None,
            None,
        );
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct JalrPrankValues {
    pub rd_data: Option<[u32; RV32_REGISTER_NUM_LIMBS - 1]>,
    pub rs1_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    pub to_pc_least_sig_bit: Option<u32>,
    pub to_pc_limbs: Option<[u32; 2]>,
    pub imm_sign: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_jalr_test(
    opcode: Rv32JalrOpcode,
    initial_pc: Option<u32>,
    initial_rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    initial_imm: Option<u32>,
    initial_imm_sign: Option<u32>,
    prank_vals: JalrPrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut chip, bitwise_chip) = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        opcode,
        initial_imm,
        initial_imm_sign,
        initial_pc,
        initial_rs1,
    );

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core_cols: &mut Rv32JalrCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_canonical_u32);
        }
        if let Some(data) = prank_vals.rs1_data {
            core_cols.rs1_data = data.map(F::from_canonical_u32);
        }
        if let Some(data) = prank_vals.to_pc_least_sig_bit {
            core_cols.to_pc_least_sig_bit = F::from_canonical_u32(data);
        }
        if let Some(data) = prank_vals.to_pc_limbs {
            core_cols.to_pc_limbs = data.map(F::from_canonical_u32);
        }
        if let Some(data) = prank_vals.imm_sign {
            core_cols.imm_sign = F::from_canonical_u32(data);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn invalid_cols_negative_tests() {
    run_negative_jalr_test(
        JALR,
        None,
        None,
        Some(15362),
        Some(0),
        JalrPrankValues {
            imm_sign: Some(1),
            ..Default::default()
        },
        false,
    );

    run_negative_jalr_test(
        JALR,
        None,
        None,
        Some(15362),
        Some(1),
        JalrPrankValues {
            imm_sign: Some(0),
            ..Default::default()
        },
        false,
    );

    run_negative_jalr_test(
        JALR,
        None,
        Some([23, 154, 67, 28]),
        Some(42512),
        Some(1),
        JalrPrankValues {
            to_pc_least_sig_bit: Some(0),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_jalr_test(
        JALR,
        Some(251),
        None,
        None,
        None,
        JalrPrankValues {
            rd_data: Some([1, 0, 0]),
            ..Default::default()
        },
        true,
    );

    run_negative_jalr_test(
        JALR,
        None,
        Some([0, 0, 0, 0]),
        Some((1 << 15) - 2),
        Some(0),
        JalrPrankValues {
            to_pc_limbs: Some([
                (F::NEG_ONE * F::from_canonical_u32((1 << 14) + 1)).as_canonical_u32(),
                1,
            ]),
            ..Default::default()
        },
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_jalr_sanity_test() {
    let initial_pc = 789456120;
    let imm = -1235_i32 as u32;
    let rs1 = 736482910;
    let (next_pc, rd_data) = run_jalr(initial_pc, rs1, imm as u16, true);
    assert_eq!(next_pc & !1, 736481674);
    assert_eq!(rd_data, [252, 36, 14, 47]);
}
