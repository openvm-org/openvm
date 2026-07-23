use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor, VmAirWrapper, VmChipWrapper,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_riscv_transpiler::Rv64JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightLimits,
                RvrPreflightTranscript,
            },
            MatrixRecordArena, VmExecutor,
        },
        system::{
            cuda::memory::MemoryInventoryGPU,
            memory::online::{AddressMap, GuestMemory, TracingMemory},
        },
        utils::test_system_config,
    },
    openvm_circuit_primitives::Chip,
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::{
            assert_eq_host_and_device_matrix_col_maj, transport_matrix_d2h_row_major,
        },
        prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::VmExe,
        program::Program,
        riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        SystemOpcode,
    },
    openvm_riscv_transpiler::BaseAluImmOpcode,
    openvm_stark_backend::prover::ColMajorMatrix,
};
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64RdWriteAdapterRecord, Rv64JalLuiChipGpu, Rv64JalLuiCoreRecord},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use crate::{
    adapters::{
        rv64_u16_block_to_bytes, Rv64CondRdWriteAdapterAir, Rv64CondRdWriteAdapterCols,
        Rv64CondRdWriteAdapterExecutor, Rv64CondRdWriteAdapterFiller, Rv64RdWriteAdapterFiller,
        RV64_BYTE_BITS, RV64_PTR_U16_LIMBS, RV_J_TYPE_IMM_BITS,
    },
    jal_lui::{get_signed_imm, run_jal_lui, Rv64JalLuiCoreCols},
    Rv64JalLuiAir, Rv64JalLuiChip, Rv64JalLuiCoreAir, Rv64JalLuiExecutor, Rv64JalLuiFiller,
};

const MAX_INS_CAPACITY: usize = 128;
const LIMB_MAX_U16: u32 = u16::MAX as u32;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64JalLuiExecutor, Rv64JalLuiAir, Rv64JalLuiChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64JalLuiAir, Rv64JalLuiExecutor, Rv64JalLuiChip<F>) {
    let air = VmAirWrapper::new(
        Rv64CondRdWriteAdapterAir::new(crate::adapters::Rv64RdWriteAdapterAir::new(
            memory_bridge,
            execution_bridge,
        )),
        Rv64JalLuiCoreAir::new(range_checker_chip.bus()),
    );
    let executor = Rv64JalLuiExecutor::new(Rv64CondRdWriteAdapterExecutor::new(
        crate::adapters::Rv64RdWriteAdapterExecutor::new(),
    ));
    let chip = VmChipWrapper::<F, _>::new(
        Rv64JalLuiFiller::new(
            Rv64CondRdWriteAdapterFiller::new(Rv64RdWriteAdapterFiller::new()),
            range_checker_chip,
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
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
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
    opcode: Rv64JalLuiOpcode,
    imm: Option<i32>,
    initial_pc: Option<u32>,
    rd_ptr: Option<usize>,
) where
    Rv64JalLuiExecutor: PreflightExecutor<F, RA>,
{
    let is_jal = opcode == JAL;
    let imm = imm.unwrap_or_else(|| {
        if is_jal {
            let raw: i32 = rng.random_range(0..(1 << (RV_J_TYPE_IMM_BITS - 1)));
            if rng.random_bool(0.5) {
                -raw
            } else {
                raw
            }
        } else {
            rng.random_range(0..(1 << 20))
        }
    });
    let a = rd_ptr.unwrap_or_else(|| (rng.random_range(0..32) << 3) as usize);

    let initial_pc = initial_pc.unwrap_or_else(|| {
        if is_jal && imm < 0 {
            rng.random_range((-imm as u32)..(1u32 << 30))
        } else {
            rng.random_range(0..(1u32 << 30).min(1u32 << PC_BITS))
        }
    });
    let imm_field: F = if imm.is_negative() {
        -F::from_u32(imm.unsigned_abs())
    } else {
        F::from_u32(imm.unsigned_abs())
    };
    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                0,
                imm_field.as_canonical_u32() as usize,
                1,
                0,
                (a != 0) as usize,
            ],
        ),
        initial_pc,
    );

    let (_next_pc, rd_data) = run_jal_lui(is_jal, initial_pc, imm);
    if a != 0 {
        let rd_bytes = rv64_u16_block_to_bytes(rd_data);
        assert_eq!(rd_bytes.map(F::from_u8), tester.read_bytes::<8>(1, a));
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test_case(JAL, 100)]
#[test_case(LUI, 100)]
fn rand_jal_lui_test(opcode: Rv64JalLuiOpcode, num_ops: usize) {
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
    pub rd_data: Option<[u32; RV64_PTR_U16_LIMBS]>,
    pub imm: Option<i32>,
    pub imm_low_4: Option<u32>,
    pub is_jal: Option<bool>,
    pub is_lui: Option<bool>,
    pub is_sign_extend: Option<bool>,
    pub rd_ptr: Option<u32>,
    pub needs_write: Option<bool>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_jal_lui_test_with_rd_ptr(
    opcode: Rv64JalLuiOpcode,
    initial_imm: Option<i32>,
    initial_pc: Option<u32>,
    rd_ptr: Option<usize>,
    prank_vals: JalLuiPrankValues,
    _interaction_error: bool,
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
        rd_ptr,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, core_row) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut Rv64JalLuiCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_u32);
        }
        if let Some(imm) = prank_vals.imm {
            core_cols.imm = if imm.is_negative() {
                -F::from_u32(imm.unsigned_abs())
            } else {
                F::from_u32(imm.unsigned_abs())
            };
        }
        if let Some(imm_low_4) = prank_vals.imm_low_4 {
            core_cols.imm_low_4 = F::from_u32(imm_low_4);
        }
        if let Some(is_jal) = prank_vals.is_jal {
            core_cols.is_jal = F::from_bool(is_jal);
        }
        if let Some(is_lui) = prank_vals.is_lui {
            core_cols.is_lui = F::from_bool(is_lui);
        }
        if let Some(is_sign_extend) = prank_vals.is_sign_extend {
            core_cols.is_sign_extend = F::from_bool(is_sign_extend);
        }
        if let Some(rd_ptr) = prank_vals.rd_ptr {
            adapter_cols.inner.rd_ptr = F::from_u32(rd_ptr);
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
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

fn run_negative_jal_lui_test(
    opcode: Rv64JalLuiOpcode,
    initial_imm: Option<i32>,
    initial_pc: Option<u32>,
    prank_vals: JalLuiPrankValues,
    interaction_error: bool,
) {
    run_negative_jal_lui_test_with_rd_ptr(
        opcode,
        initial_imm,
        initial_pc,
        None,
        prank_vals,
        interaction_error,
    );
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
fn write_suppression_boundary_negative_test() {
    run_negative_jal_lui_test_with_rd_ptr(
        JAL,
        Some((1 << 19) + 2),
        Some(28120),
        Some(0),
        JalLuiPrankValues {
            rd_ptr: Some(8),
            ..Default::default()
        },
        true,
    );

    run_negative_jal_lui_test_with_rd_ptr(
        JAL,
        Some((1 << 19) + 2),
        Some(28120),
        Some(8),
        JalLuiPrankValues {
            needs_write: Some(false),
            ..Default::default()
        },
        true,
    );
}

#[test]
fn rd_upper_bytes_trace_tamper_negative_test() {
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    let initial_pc = 0x1234;
    let imm = 16i32;
    let rd_ptr = 16usize;
    let clean_rd_prev = [9u32, 8, 7, 6, 0, 0, 0, 0];

    tester.write_bytes(1, rd_ptr, clean_rd_prev.map(F::from_u32));

    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::large_from_isize(
            LUI.global_opcode(),
            rd_ptr as isize,
            0,
            imm as isize,
            1,
            0,
            1,
            0,
        ),
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();
        adapter_cols.inner.rd_aux_cols.prev_data[1] = F::from_u32(1);
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn sign_extend_flag_negative_tests() {
    // LUI with imm small enough that imm << 12 has bit 31 unset (MSB of rd[1] is 0).
    // is_sign_extend pranked to true should fail.
    run_negative_jal_lui_test(
        LUI,
        Some(1),
        None,
        JalLuiPrankValues {
            is_sign_extend: Some(true),
            ..Default::default()
        },
        true,
    );
    // JAL writes pc+4 with pc < 2^30, so MSB of rd[1] is always 0.
    // is_sign_extend pranked to true should fail.
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            is_sign_extend: Some(true),
            ..Default::default()
        },
        true,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([LIMB_MAX_U16, LIMB_MAX_U16]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        Some((1u32 << 28) - 6),
        JalLuiPrankValues {
            rd_data: Some([0, 0]),
            ..Default::default()
        },
        false,
    );
    // Pin LUI sign bit so this case exercises bad rd arithmetic, not sign-select mismatch.
    run_negative_jal_lui_test(
        LUI,
        Some(1 << 19),
        None,
        JalLuiPrankValues {
            rd_data: Some([0, LIMB_MAX_U16]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([0, LIMB_MAX_U16 + 1]),
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
        LUI,
        Some(1 << 19),
        None,
        JalLuiPrankValues {
            is_sign_extend: Some(false),
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
        Some((1 << 20) - 1),
        None,
        None,
    );
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        JAL,
        Some((1i32 << (RV_J_TYPE_IMM_BITS - 1)) - 1),
        None,
        None,
    );
}

#[test]
fn run_jal_sanity_test() {
    let initial_pc = 28120;
    let imm = -2048;
    let (next_pc, rd_data) = run_jal_lui(true, initial_pc, imm);
    assert_eq!(next_pc, 26072);
    assert_eq!(rd_data, [0x6ddc, 0, 0, 0]);
}

#[test]
fn jal_x0_write_suppression_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, _) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        JAL,
        Some((1 << 19) + 2),
        Some(28120),
        Some(0),
    );
}

#[test]
fn run_lui_sanity_test() {
    let initial_pc = 456789120;
    let imm = 853679;
    let (next_pc, rd_data) = run_jal_lui(false, initial_pc, imm);
    assert_eq!(next_pc, 456789124);
    assert_eq!(rd_data, [0xf000, 0xd06a, 0xffff, 0xffff]);
}

#[test]
fn run_lui_sign_extend_sanity_test() {
    let (_, rd_data) = run_jal_lui(false, 0, 1 << 19);
    assert_eq!(rd_data[2], 0xffff);
    assert_eq!(rd_data[3], 0xffff);
}

#[test]
fn get_signed_imm_test() {
    let imm: i32 = -10;
    let imm_f: F = -F::from_u32(10);
    let signed_imm = get_signed_imm(true, imm_f);
    assert_eq!(signed_imm, imm);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64JalLuiExecutor, Rv64JalLuiAir, Rv64JalLuiChipGpu, Rv64JalLuiChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64JalLuiChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(JAL, 100)]
#[test_case(LUI, 100)]
fn test_cuda_rand_jal_lui_tracegen(opcode: Rv64JalLuiOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(openvm_circuit::arch::testing::default_bitwise_lookup_bus());
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
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64RdWriteAdapterRecord,
        &'a mut Rv64JalLuiCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64CondRdWriteAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_jal_lui_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let jal_lui = |opcode: Rv64JalLuiOpcode, rd: usize, immediate: usize, needs_write: usize| {
        Instruction::<F>::from_usize(
            opcode.global_opcode(),
            [
                reg(rd),
                0,
                immediate,
                RV64_REGISTER_AS as usize,
                0,
                needs_write,
            ],
        )
    };
    let instructions = [
        jal_lui(LUI, 1, 1, 1),
        jal_lui(JAL, 2, 4, 1),
        jal_lui(LUI, 3, 0x80000, 1),
        jal_lui(JAL, 0, 4, 0),
        jal_lui(LUI, 1, 0xfffff, 1),
        jal_lui(JAL, 1, 4, 1),
        jal_lui(LUI, 4, 0x12345, 1),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config.clone())
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(16, 16))
        .unwrap();

    let mut tester = GpuChipTestBuilder::default();
    let initial_image = GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
    tester.memory.memory = TracingMemory::from_image(initial_image);
    let device_ctx = tester.range_checker().device_ctx.clone();
    let hasher_chip = tester.memory.hasher_chip.clone().unwrap();
    tester.memory.inventory = MemoryInventoryGPU::new(
        tester.memory.config.clone(),
        hasher_chip,
        device_ctx.clone(),
    );
    tester
        .memory
        .inventory
        .set_initial_memory(&tester.memory.memory.data().memory);
    let mut harness = create_cuda_harness(&tester);
    for (instruction_index, instruction) in instructions[..7].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            instruction_index as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64RdWriteAdapterRecord,
        &'a mut Rv64JalLuiCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64CondRdWriteAdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(JAL.global_opcode()).len(), 3);
    assert_eq!(d_replay_plan.opcode_range(LUI.global_opcode()).len(), 4);
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();
    let raw_count = |count: &F| {
        const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<u32>()) };
        // CUDA kernels atomically update this shared field-typed buffer as raw u32 counters.
        unsafe { *(std::ptr::from_ref(count).cast::<u32>()) }
    };
    // Six enabled writes contribute six lookups each. The x0 JAL advances the clock without a
    // memory event and contributes only the four core lookups.
    assert_eq!(replay_counts.iter().map(raw_count).sum::<u32>(), 6 * 6 + 4);

    let negative_program = Program::from_instructions(&[
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
        Instruction::<F>::large_from_isize(
            JAL.global_opcode(),
            0,
            0,
            -4,
            RV64_REGISTER_AS as isize,
            0,
            0,
            0,
        ),
    ]);
    let mut negative_from = execution.transcript.program_log[0];
    negative_from.pc = 4;
    negative_from.timestamp = 1;
    let mut negative_terminate = execution.transcript.program_log[0];
    negative_terminate.pc = 0;
    negative_terminate.timestamp = 2;
    let negative_transcript = RvrPreflightTranscript {
        program_log: vec![negative_from, negative_terminate, negative_terminate],
        memory_log: Vec::new(),
        initial_write_log: Vec::new(),
    };
    let d_negative_program =
        GpuRvrProgram::upload(&negative_program, &memory_config, device_ctx).unwrap();
    let (d_negative, d_negative_plan) = d_negative_program
        .upload_transcript(&negative_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let negative_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    Rv64JalLuiChipGpu::new(negative_range_checker.clone(), tester.timestamp_max_bits())
        .generate_proving_ctx_from_rvr(&d_negative_program, &d_negative, &d_negative_plan)
        .unwrap();
    assert_eq!(d_negative.error_code().unwrap(), 0);
    assert_eq!(
        negative_range_checker
            .count
            .to_host_on(device_ctx)
            .unwrap()
            .iter()
            .map(raw_count)
            .sum::<u32>(),
        4
    );

    let run_corrupt = |corrupt_program: &Program<F>,
                       transcript: RvrPreflightTranscript,
                       expected_error: u32,
                       expected_lookup_count: u32| {
        let corrupt_range_checker = Arc::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
                openvm_circuit::arch::testing::default_var_range_checker_bus(),
                device_ctx.clone(),
            ),
        );
        let d_corrupt_program =
            GpuRvrProgram::upload(corrupt_program, &memory_config, device_ctx).unwrap();
        let (d_corrupt, d_corrupt_plan) = d_corrupt_program
            .upload_transcript(&transcript, RvrPreflightEndpoint::Terminated)
            .unwrap();
        Rv64JalLuiChipGpu::new(corrupt_range_checker.clone(), tester.timestamp_max_bits())
            .generate_proving_ctx_from_rvr(&d_corrupt_program, &d_corrupt, &d_corrupt_plan)
            .unwrap();
        assert_eq!(d_corrupt.error_code().unwrap(), expected_error);
        assert_eq!(
            corrupt_range_checker
                .count
                .to_host_on(device_ctx)
                .unwrap()
                .iter()
                .map(raw_count)
                .sum::<u32>(),
            expected_lookup_count,
            "a rejected row must not update the shared lookup histogram"
        );
    };
    let transcript = || RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };

    let negative_lui_timestamp = execution
        .transcript
        .program_log
        .iter()
        .find(|event| event.pc == 8)
        .unwrap()
        .timestamp;
    let negative_lui_write = execution
        .transcript
        .memory_log
        .iter()
        .position(|event| event.timestamp == negative_lui_timestamp)
        .unwrap();
    let mut result_corrupt = transcript();
    result_corrupt.memory_log[negative_lui_write].value[2] = 0;
    run_corrupt(&program, result_corrupt, 186, 34);

    let mut target_instructions = instructions.clone();
    target_instructions[1] = jal_lui(JAL, 2, 8, 1);
    run_corrupt(
        &Program::from_instructions(&target_instructions),
        transcript(),
        187,
        34,
    );

    let mut flag_instructions = instructions.clone();
    flag_instructions[3] = jal_lui(JAL, 0, 4, 1);
    run_corrupt(
        &Program::from_instructions(&flag_instructions),
        transcript(),
        184,
        36,
    );

    let mut bound_instructions = instructions.clone();
    bound_instructions[6] = jal_lui(LUI, 4, 1 << 20, 1);
    run_corrupt(
        &Program::from_instructions(&bound_instructions),
        transcript(),
        189,
        34,
    );

    let x0_timestamp = execution
        .transcript
        .program_log
        .iter()
        .find(|event| event.pc == 12)
        .unwrap()
        .timestamp;
    let mut gap_corrupt = transcript();
    let mut displaced_final_write = gap_corrupt.memory_log.pop().unwrap();
    displaced_final_write.timestamp = x0_timestamp;
    let insertion_index = gap_corrupt
        .memory_log
        .partition_point(|event| event.timestamp < x0_timestamp);
    gap_corrupt
        .memory_log
        .insert(insertion_index, displaced_final_write);
    run_corrupt(&program, gap_corrupt, 185, 30);

    // A preceding ADDI is outside this AIR, but its logged write is the exact predecessor of the
    // JAL write. Corrupting only that predecessor's value reaches error 188 without making another
    // JAL/LUI row fail first.
    let predecessor_instructions = [
        Instruction::<F>::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [
                reg(1),
                0,
                7,
                RV64_REGISTER_AS as usize,
                openvm_instructions::riscv::RV64_IMM_AS as usize,
            ],
        ),
        jal_lui(JAL, 1, 4, 1),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let predecessor_program = Program::from_instructions(&predecessor_instructions);
    let predecessor_execution = VmExecutor::new(config)
        .unwrap()
        .rvr_preflight_instance(&VmExe::new(predecessor_program.clone()), None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(8, 8))
        .unwrap();
    let mut predecessor_corrupt = RvrPreflightTranscript {
        program_log: predecessor_execution.transcript.program_log,
        memory_log: predecessor_execution.transcript.memory_log,
        initial_write_log: predecessor_execution.transcript.initial_write_log,
    };
    let addi_timestamp = predecessor_corrupt.program_log[0].timestamp;
    let predecessor_write = predecessor_corrupt
        .memory_log
        .iter()
        .position(|event| {
            event.timestamp == addi_timestamp + 1 && event.pointer == (reg(1) / 2) as u32
        })
        .unwrap();
    predecessor_corrupt.memory_log[predecessor_write].value[0] = 1 << 16;
    run_corrupt(&predecessor_program, predecessor_corrupt, 188, 0);

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_ctx =
        Rv64JalLuiChipGpu::new(legacy_range_checker.clone(), tester.timestamp_max_bits())
            .generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace =
        <Rv64JalLuiChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
            &harness.cpu_chip,
            harness.matrix_arena,
        )
        .common_main;
    let replay_trace = transport_matrix_d2h_row_major(&replay_ctx.common_main, device_ctx).unwrap();
    let canonical_rows = |matrix: &RowMajorMatrix<F>| {
        let mut rows = (0..matrix.height())
            .map(|row| {
                matrix
                    .row_slice(row)
                    .unwrap()
                    .iter()
                    .map(|value| value.as_canonical_u32())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    };
    assert_eq!(
        canonical_rows(&expected_trace),
        canonical_rows(&replay_trace)
    );
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR JAL/LUI transcript replay proof failed");
}
