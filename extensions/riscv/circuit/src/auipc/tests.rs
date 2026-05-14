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
use openvm_riscv_transpiler::Rv64AuipcOpcode::{self, *};
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
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64RdWriteAdapterRecord, Rv64AuipcChipGpu, Rv64AuipcCoreRecord},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use crate::{
    adapters::{
        Rv64RdWriteAdapterAir, Rv64RdWriteAdapterCols, Rv64RdWriteAdapterExecutor,
        Rv64RdWriteAdapterFiller, RV64_CELL_BITS,
    },
    auipc::{run_auipc, Rv64AuipcCoreCols},
    Rv64AuipcAir, Rv64AuipcChip, Rv64AuipcCoreAir, Rv64AuipcExecutor, Rv64AuipcFiller,
};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AuipcExecutor, Rv64AuipcAir, Rv64AuipcChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AuipcAir, Rv64AuipcExecutor, Rv64AuipcChip<F>) {
    let air = VmAirWrapper::new(
        Rv64RdWriteAdapterAir::new(memory_bridge, execution_bridge),
        Rv64AuipcCoreAir::new(bitwise_chip.bus(), range_checker_chip.bus()),
    );
    let executor = Rv64AuipcExecutor::new(Rv64RdWriteAdapterExecutor::new());
    let chip = VmChipWrapper::<F, _>::new(
        Rv64AuipcFiller::new(
            Rv64RdWriteAdapterFiller::new(),
            bitwise_chip,
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
        BitwiseOperationLookupAir<RV64_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.range_checker(),
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
    opcode: Rv64AuipcOpcode,
    imm: Option<u32>,
    initial_pc: Option<u32>,
) where
    Rv64AuipcExecutor: PreflightExecutor<F, RA>,
{
    let imm = imm.unwrap_or(rng.random_range(0..(1 << IMM_BITS))) as usize;
    let a = rng.random_range(0..32) << 3;

    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [a, 0, imm, 1, 0]),
        initial_pc.unwrap_or(rng.random_range(0..(1 << PC_BITS))),
    );
    let initial_pc = tester.last_from_pc().as_canonical_u32();
    let rd_data = run_auipc(initial_pc, imm as u32);
    // run_auipc returns 4 u16 cells; the byte-view memory read returns 8 bytes.
    let mut rd_bytes = [0u8; 8];
    for (i, &v) in rd_data.iter().enumerate() {
        let [lo, hi] = v.to_le_bytes();
        rd_bytes[2 * i] = lo;
        rd_bytes[2 * i + 1] = hi;
    }
    assert_eq!(rd_bytes.map(F::from_u8), tester.read::<8>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_auipc_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            AUIPC,
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
// part of the trace and check that the chip throws the expected error. Post Pattern B
// u16 migration: rd_data is `[T; 2]` u16 (was `[T; 4]` u8), and `pc_limbs` is no longer
// a column — the chip recovers `pc` via the composite-carry decomposition
// `pc + (imm << 8) = rd[0] + rd[1] * 2^16 + carry * 2^32`, so prank-pc tests now target
// rd_data + imm_limbs directly.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct AuipcPrankValues {
    pub is_sign_extend: Option<u32>,
    /// 2 u16 limbs (low 32 bits of rd).
    pub rd_data: Option<[u32; 2]>,
    /// Low byte of imm.
    pub imm_low_8: Option<u32>,
    /// High 16 bits of imm.
    pub imm_high_16: Option<u32>,
}

fn run_negative_auipc_test(
    opcode: Rv64AuipcOpcode,
    initial_imm: Option<u32>,
    initial_pc: Option<u32>,
    prank_vals: AuipcPrankValues,
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
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core_cols: &mut Rv64AuipcCoreCols<F> = core_row.borrow_mut();

        if let Some(val) = prank_vals.is_sign_extend {
            core_cols.is_sign_extend = F::from_u32(val);
        }
        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_u32);
        }
        if let Some(val) = prank_vals.imm_low_8 {
            core_cols.imm_low_8 = F::from_u32(val);
        }
        if let Some(val) = prank_vals.imm_high_16 {
            core_cols.imm_high_16 = F::from_u32(val);
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

#[test]
fn invalid_imm_limb_negative_test() {
    // Out-of-byte-range imm_low_8 fails the bitwise_lookup range check.
    run_negative_auipc_test(
        AUIPC,
        Some(0x123456),
        None,
        AuipcPrankValues {
            // imm = 0x123456 ⇒ imm_low_8 = 0x56, imm_high_16 = 0x1234. Prank low byte to 0x100.
            imm_low_8: Some(0x100),
            imm_high_16: Some(0x1234),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn invalid_rd_data_negative_test() {
    // Pranking rd_data to a wrong u16 limb breaks the carry-chain constraint linking
    // rd_data to from_pc + (imm << 8).
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(0x12345678),
        AuipcPrankValues {
            rd_data: Some([0x1234, 0x1234]),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn rd_upper_bytes_trace_tamper_negative_test() {
    // Tamper with the rd-aux prev_data column: changing one of the 4 u16 prev_data cells
    // away from the canonical pre-write value should fail the memory-bus permutation.
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    let initial_pc = 0x1234;
    let imm = 16usize;
    let rd_ptr = 16usize;

    let clean_rd_prev = [9u32, 8, 7, 6, 0, 0, 0, 0];
    tester.write(1, rd_ptr, clean_rd_prev.map(F::from_u32));

    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(AUIPC.global_opcode(), [rd_ptr, 0, imm, 1, 0]),
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64RdWriteAdapterCols<F> = adapter_row.borrow_mut();
        // Pattern B: prev_data is `[F; BLOCK_FE_WIDTH=4]` u16 cells; bump one to a different
        // canonical u16 value.
        adapter_cols.rd_aux_cols.prev_data[1] = F::from_u32(1);
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
    // is_sign_extend = 1 when the canonical result fits in 32 bits (MSB of rd[1] is 0).
    // pc=4, imm=0 ⟹ rd_low_32 = 4 ⟹ rd[1] = 0 ⟹ canonical is_sign_extend = 0; prank to 1.
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(4),
        AuipcPrankValues {
            is_sign_extend: Some(1),
            ..Default::default()
        },
        true,
    );
    // is_sign_extend = 0 when canonical bit 31 of rd_low_32 is 1.
    // pc=0, imm=2^23 ⟹ rd_low_32 = 2^31 ⟹ rd[1] = 0x8000 ⟹ canonical is_sign_extend = 1;
    // prank to 0.
    run_negative_auipc_test(
        AUIPC,
        Some(1 << 23),
        Some(0),
        AuipcPrankValues {
            is_sign_extend: Some(0),
            ..Default::default()
        },
        true,
    );
}

#[test]
fn overflow_negative_tests() {
    // Force imm decomposition to disagree with the instruction-encoded imm. The byte-range
    // check rejects out-of-range `imm_low_8` and the u16 range check rejects out-of-range
    // `imm_high_16`.
    run_negative_auipc_test(
        AUIPC,
        Some(256264),
        None,
        AuipcPrankValues {
            // 0x3e988 ⇒ imm_low_8 = 0x88, imm_high_16 = 0x03e9. Prank low byte out of u8 range.
            imm_low_8: Some(3592),
            imm_high_16: Some(0x03e9),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(255),
        None,
        AuipcPrankValues {
            // F::NEG_ONE is way past byte range — fails the bitwise lookup.
            imm_low_8: Some(F::NEG_ONE.as_canonical_u32()),
            imm_high_16: Some(0),
            ..Default::default()
        },
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_auipc_sanity_test() {
    let pc = 0x12345678u32;
    let imm = 0xabcdu32;
    let rd = run_auipc(pc, imm);
    // rd_low_32 = pc + (imm << 8) = 0x12345678 + 0xabcd00 = 0x12e02378.
    assert_eq!(rd[0] as u32 | ((rd[1] as u32) << 16), 0x12e02378);
    // Top bit of rd[1] (=0x12e0) is 0, so no sign extension.
    assert_eq!(rd[2], 0);
    assert_eq!(rd[3], 0);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64AuipcExecutor, Rv64AuipcAir, Rv64AuipcChipGpu, Rv64AuipcChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = openvm_circuit::arch::testing::default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AuipcChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_auipc_tracegen() {
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(openvm_circuit::arch::testing::default_bitwise_lookup_bus());
    let mut rng = create_seeded_rng();
    let mut harness = create_cuda_harness(&tester);

    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            AUIPC,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64RdWriteAdapterRecord,
        &'a mut Rv64AuipcCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64RdWriteAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
