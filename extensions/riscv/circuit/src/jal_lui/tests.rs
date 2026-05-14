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
        Rv64CondRdWriteAdapterAir, Rv64CondRdWriteAdapterCols, Rv64CondRdWriteAdapterExecutor,
        Rv64CondRdWriteAdapterFiller, Rv64RdWriteAdapterFiller, RV64_CELL_BITS, RV_J_TYPE_IMM_BITS,
    },
    jal_lui::{get_signed_imm, run_jal_lui, Rv64JalLuiCoreCols},
    Rv64JalLuiAir, Rv64JalLuiChip, Rv64JalLuiCoreAir, Rv64JalLuiExecutor, Rv64JalLuiFiller,
};

const MAX_INS_CAPACITY: usize = 128;
const LIMB_MAX_U16: u32 = (1 << 16) - 1;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64JalLuiExecutor, Rv64JalLuiAir, Rv64JalLuiChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64JalLuiAir, Rv64JalLuiExecutor, Rv64JalLuiChip<F>) {
    let air = VmAirWrapper::new(
        Rv64CondRdWriteAdapterAir::new(crate::adapters::Rv64RdWriteAdapterAir::new(
            memory_bridge,
            execution_bridge,
        )),
        Rv64JalLuiCoreAir::new(bitwise_chip.bus(), range_checker_chip.bus()),
    );
    let executor = Rv64JalLuiExecutor::new(Rv64CondRdWriteAdapterExecutor::new(
        crate::adapters::Rv64RdWriteAdapterExecutor::new(),
    ));
    let chip = VmChipWrapper::<F, _>::new(
        Rv64JalLuiFiller::new(
            Rv64CondRdWriteAdapterFiller::new(Rv64RdWriteAdapterFiller::new()),
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
            let raw = rng.random_range(0..(1 << (RV_J_TYPE_IMM_BITS - 1))) as i32;
            if rng.random_bool(0.5) {
                -raw
            } else {
                raw
            }
        } else {
            rng.random_range(0..(1 << 20)) as i32
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
    let imm_field: F = if imm < 0 {
        -F::from_u32((-imm) as u32)
    } else {
        F::from_u32(imm as u32)
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
        let mut rd_bytes = [0u8; 8];
        for (i, &v) in rd_data.iter().enumerate() {
            let [lo, hi] = v.to_le_bytes();
            rd_bytes[2 * i] = lo;
            rd_bytes[2 * i + 1] = hi;
        }
        assert_eq!(rd_bytes.map(F::from_u8), tester.read::<8>(1, a));
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
// Pattern B u16: `rd_data` is `[T; 2]` u16 limbs (was `[T; 4]` u8); the chip also stores
// `imm_low_4` for LUI's low-4-bit imm witness. Prank patterns are adapted accordingly.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct JalLuiPrankValues {
    /// 2 u16 limbs of rd_low_32.
    pub rd_data: Option<[u32; 2]>,
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
            core_cols.imm = if imm < 0 {
                F::NEG_ONE * F::from_u32((-imm) as u32)
            } else {
                F::from_u32(imm as u32)
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
    // Swap is_jal ↔ is_lui: instruction bus mismatch.
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
    // Clear both flags: instruction bus has nothing matching.
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
    // rd_ptr = 0 means JAL should suppress the write. Pranking rd_ptr to a non-zero value
    // tries to claim a write happened to that address; the memory bus permutation rejects.
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
    // Conversely, with rd_ptr = 8 (real write) pranking needs_write to false skips the
    // permutation send.
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
    // Tamper with one of the 4 u16 prev_data cells; the memory bus permutation rejects.
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    let initial_pc = 0x1234;
    let imm = 16i32;
    let rd_ptr = 16usize;
    let clean_rd_prev = [9u32, 8, 7, 6, 0, 0, 0, 0];
    tester.write(1, rd_ptr, clean_rd_prev.map(F::from_u32));

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
        // u16 cells (post Pattern B): bump the high cell.
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
    // is_sign_extend pranked to true should fail the sign-bit range check on rd[1].
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
    // JAL writes pc+4 with pc < 2^30 so MSB of rd[1] is always 0; same prank fails.
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
    // Pattern B u16: out-of-canonical rd_data fails the per-limb range check via range_bus.
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
    // LUI: pinning imm sign to high-bit case so this exercises bad rd arithmetic.
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
    // rd_data with high limb > 2^16 fails the range check.
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
    // imm wrap to negative on LUI: violates the constraint imm = imm_low_4 + rd[1] * 16.
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
    // LUI imm = 2^19 → high bit set → canonical is_sign_extend = 1; prank to false fails.
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
        Some(((1 << (RV_J_TYPE_IMM_BITS - 1)) - 1) as i32),
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
    // rd_low_32 = pc + 4 = 28124 = 0x6ddc → u16[0] = 0x6ddc = 28124, u16[1] = 0; no sign ext.
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
    // rd_low_32 = imm << 12 = 853679 << 12 = 0xd06af000. u16: lo = 0xf000, hi = 0xd06a.
    // Top bit of hi is 1 ⇒ sign extension fills upper cells with 0xffff.
    assert_eq!(rd_data, [0xf000, 0xd06a, 0xffff, 0xffff]);
}

#[test]
fn run_lui_sign_extend_sanity_test() {
    // imm = 2^19 (high bit set) → imm << 12 = 2^31 → bit 31 of rd_low_32 is 1.
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
    let gpu_chip = Rv64JalLuiChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );
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
