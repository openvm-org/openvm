use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerChip,
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
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32JalrAdapterRecord, Rv32JalrChipGpu, Rv32JalrCoreRecord},
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::Rv32JalrCoreAir;
use crate::{
    adapters::{
        compose, Rv32JalrAdapterAir, Rv32JalrAdapterExecutor, Rv32JalrAdapterFiller,
        RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    jalr::{run_jalr, Rv32JalrChip, Rv32JalrCoreCols, Rv32JalrExecutor},
    test_utils::get_verification_error,
    Rv32JalrAir, Rv32JalrFiller,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv32JalrExecutor, Rv32JalrAir, Rv32JalrChip<F>>;

fn into_limbs(num: u32) -> [u32; 4] {
    array::from_fn(|i| (num >> (8 * i)) & 255)
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv32JalrAir, Rv32JalrExecutor, Rv32JalrChip<F>) {
    let air = Rv32JalrAir::new(
        Rv32JalrAdapterAir::new(memory_bridge, execution_bridge),
        Rv32JalrCoreAir::new(bitwise_chip.bus(), range_checker_chip.bus()),
    );
    let executor = Rv32JalrExecutor::new(Rv32JalrAdapterExecutor);
    let chip = Rv32JalrChip::<F>::new(
        Rv32JalrFiller::new(
            Rv32JalrAdapterFiller::new(),
            bitwise_chip,
            range_checker_chip,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let range_checker_chip = tester.range_checker().clone();

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_checker_chip,
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
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
        executor,
        arena,
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
    let final_pc = tester.last_to_pc().as_canonical_u32();

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
    let (mut harness, bitwise) = create_harness(&mut tester);

    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            JALR,
            None,
            None,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
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

    let (mut harness, bitwise) = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        initial_imm,
        initial_imm_sign,
        initial_pc,
        initial_rs1,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
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
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
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

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv32JalrExecutor, Rv32JalrAir, Rv32JalrChipGpu, Rv32JalrChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_bus = default_var_range_checker_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32JalrChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_jalr_tracegen() {
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut rng = create_seeded_rng();

    let mut harness = create_cuda_harness(&tester);
    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            JALR,
            None,
            None,
            None,
            None,
        );
    }

    type Record<'a> = (&'a mut Rv32JalrAdapterRecord, &'a mut Rv32JalrCoreRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32JalrAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
