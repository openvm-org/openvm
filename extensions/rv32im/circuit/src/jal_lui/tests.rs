use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode::{self, *};
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
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32RdWriteAdapterRecord, Rv32JalLuiChipGpu, Rv32JalLuiCoreRecord},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{run_jal_lui, Rv32JalLuiChip, Rv32JalLuiCoreAir, Rv32JalLuiExecutor};
use crate::{
    adapters::{
        Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterCols, Rv32CondRdWriteAdapterExecutor,
        Rv32CondRdWriteAdapterFiller, Rv32RdWriteAdapterAir, Rv32RdWriteAdapterExecutor,
        Rv32RdWriteAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, RV_IS_TYPE_IMM_BITS,
    },
    jal_lui::{Rv32JalLuiCoreCols, ADDITIONAL_BITS},
    test_utils::get_verification_error,
    Rv32JalLuiAir, Rv32JalLuiFiller,
};

const IMM_BITS: usize = 20;
const LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<F, Rv32JalLuiExecutor, Rv32JalLuiAir, Rv32JalLuiChip<F>>;

type F = BabyBear;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv32JalLuiAir, Rv32JalLuiExecutor, Rv32JalLuiChip<F>) {
    let air = Rv32JalLuiAir::new(
        Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(memory_bridge, execution_bridge)),
        Rv32JalLuiCoreAir::new(bitwise_chip.bus()),
    );
    let executor = Rv32JalLuiExecutor::new(Rv32CondRdWriteAdapterExecutor::new(
        Rv32RdWriteAdapterExecutor,
    ));
    let chip = Rv32JalLuiChip::<F>::new(
        Rv32JalLuiFiller::new(
            Rv32CondRdWriteAdapterFiller::new(Rv32RdWriteAdapterFiller),
            bitwise_chip,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
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

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv32JalLuiOpcode,
    imm: Option<i32>,
    initial_pc: Option<u32>,
) {
    let imm: i32 = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm = match opcode {
        JAL => ((imm >> 1) << 2) - (1 << IMM_BITS),
        LUI => imm,
    };

    let a = rng.gen_range((opcode == LUI) as usize..32) << 2;
    let needs_write = a != 0 || opcode == LUI;

    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::large_from_isize(
            opcode.global_opcode(),
            a as isize,
            0,
            imm as isize,
            1,
            0,
            needs_write as isize,
            0,
        ),
        initial_pc.unwrap_or(rng.gen_range(imm.unsigned_abs()..(1 << PC_BITS))),
    );
    let initial_pc = tester.last_from_pc().as_canonical_u32();
    let final_pc = tester.last_to_pc().as_canonical_u32();

    let (next_pc, rd_data) = run_jal_lui(opcode == JAL, initial_pc, imm);
    let rd_data = if needs_write { rd_data } else { [0; 4] };

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test_case(JAL, 100)]
#[test_case(LUI, 100)]
fn rand_jal_lui_test(opcode: Rv32JalLuiOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
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
struct JalLuiPrankValues {
    pub rd_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    pub imm: Option<i32>,
    pub is_jal: Option<bool>,
    pub is_lui: Option<bool>,
    pub needs_write: Option<bool>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_jal_lui_test(
    opcode: Rv32JalLuiOpcode,
    initial_imm: Option<i32>,
    initial_pc: Option<u32>,
    prank_vals: JalLuiPrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        initial_imm,
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv32CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut Rv32JalLuiCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_canonical_u32);
        }
        if let Some(imm) = prank_vals.imm {
            core_cols.imm = if imm < 0 {
                F::NEG_ONE * F::from_canonical_u32((-imm) as u32)
            } else {
                F::from_canonical_u32(imm as u32)
            };
        }
        if let Some(is_jal) = prank_vals.is_jal {
            core_cols.is_jal = F::from_bool(is_jal);
        }
        if let Some(is_lui) = prank_vals.is_lui {
            core_cols.is_lui = F::from_bool(is_lui);
        }
        if let Some(needs_write) = prank_vals.needs_write {
            adapter_cols.needs_write = F::from_bool(needs_write);
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
fn opcode_flag_negative_test() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            is_jal: Some(false),
            is_lui: Some(true),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            is_jal: Some(false),
            is_lui: Some(false),
            needs_write: Some(false),
            ..Default::default()
        },
        true,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            is_jal: Some(true),
            is_lui: Some(false),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([LIMB_MAX, LIMB_MAX, LIMB_MAX, LIMB_MAX]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([LIMB_MAX, LIMB_MAX, LIMB_MAX, LIMB_MAX]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([0, LIMB_MAX, LIMB_MAX, LIMB_MAX + 1]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            imm: Some(-1),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            imm: Some(-28),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        Some(251),
        JalLuiPrankValues {
            rd_data: Some([F::NEG_ONE.as_canonical_u32(), 1, 0, 0]),
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
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, _) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LUI,
        Some((1 << IMM_BITS) - 1),
        None,
    );
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        JAL,
        Some((1 << RV_IS_TYPE_IMM_BITS) - 1),
        None,
    );
}

#[test]
fn run_jal_sanity_test() {
    let initial_pc = 28120;
    let imm = -2048;
    let (next_pc, rd_data) = run_jal_lui(true, initial_pc, imm);
    assert_eq!(next_pc, 26072);
    assert_eq!(rd_data, [220, 109, 0, 0]);
}

#[test]
fn run_lui_sanity_test() {
    let initial_pc = 456789120;
    let imm = 853679;
    let (next_pc, rd_data) = run_jal_lui(false, initial_pc, imm);
    assert_eq!(next_pc, 456789124);
    assert_eq!(rd_data, [0, 240, 106, 208]);
}

#[test]
fn test_additional_bits() {
    let last_limb_bits = PC_BITS - RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
    let additional_bits = (last_limb_bits..RV32_CELL_BITS).fold(0, |acc, x| acc + (1u32 << x));
    assert_eq!(additional_bits, ADDITIONAL_BITS);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv32JalLuiExecutor, Rv32JalLuiAir, Rv32JalLuiChipGpu, Rv32JalLuiChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32JalLuiChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(Rv32JalLuiOpcode::JAL, 100)]
#[test_case(Rv32JalLuiOpcode::LUI, 100)]
fn test_cuda_rand_jal_lui_tracegen(opcode: Rv32JalLuiOpcode, num_ops: usize) {
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut rng = create_seeded_rng();

    let mut harness = create_cuda_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv32RdWriteAdapterRecord,
        &'a mut Rv32JalLuiCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32CondRdWriteAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
