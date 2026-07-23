use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::ShiftOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
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
    openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip},
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::assert_eq_host_and_device_matrix_col_maj, prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        instruction::Instruction,
        program::Program,
        riscv::RV64_REGISTER_AS,
        SystemOpcode,
    },
    openvm_stark_backend::prover::ColMajorMatrix,
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64BaseAluRegU16AdapterRecord, Rv64ShiftRightArithmeticChipGpu,
        ShiftRightArithmeticCoreRecord,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    core::run_shift_right_arithmetic, Rv64ShiftRightArithmeticChip, ShiftRightArithmeticCoreAir,
    ShiftRightArithmeticCoreCols,
};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, Rv64BaseAluRegU16AdapterAir,
        Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller, RV64_REGISTER_NUM_LIMBS,
        U16_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
    Rv64ShiftRightArithmeticAir, Rv64ShiftRightArithmeticExecutor, ShiftRightArithmeticFiller,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const REGISTER_SHIFT_AMOUNTS: [u8; 8] = [0, 1, 15, 16, 31, 32, 63, 64];
type Harness = TestChipHarness<
    F,
    Rv64ShiftRightArithmeticExecutor,
    Rv64ShiftRightArithmeticAir,
    Rv64ShiftRightArithmeticChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64ShiftRightArithmeticAir,
    Rv64ShiftRightArithmeticExecutor,
    Rv64ShiftRightArithmeticChip<F>,
) {
    let air = Rv64ShiftRightArithmeticAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        ShiftRightArithmeticCoreAir::new(range_checker_chip.bus(), ShiftOpcode::CLASS_OFFSET),
    );
    let executor = Rv64ShiftRightArithmeticExecutor::new(
        Rv64BaseAluRegU16AdapterExecutor,
        ShiftOpcode::CLASS_OFFSET,
    );
    let chip = Rv64ShiftRightArithmeticChip::<F>::new(
        ShiftRightArithmeticFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: ShiftOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let b_u16 = rv64_bytes_to_u16_block(b);
    let c_u16 = rv64_bytes_to_u16_block(c);
    let (a_u16, _, _) = run_shift_right_arithmetic::<BLOCK_FE_WIDTH, U16_BITS>(&b_u16, &c_u16);
    let a_bytes = rv64_u16_block_to_bytes(a_u16);
    assert_eq!(
        a_bytes.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

fn execute_boundary_shifts<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: ShiftOpcode,
) {
    for top in [0x12, 0xBC] {
        let b = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, top];
        for shift in REGISTER_SHIFT_AMOUNTS {
            let mut c = [0u8; RV64_REGISTER_NUM_LIMBS];
            c[0] = shift;
            set_and_execute(tester, executor, arena, rng, opcode, Some(b), Some(c));
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////
#[test_case(SRA, 100)]
fn run_rv64_shift_right_arithmetic_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&tester);

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

    execute_boundary_shifts(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct ShiftPrankValues<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: Option<[u32; NUM_LIMBS]>,
    pub b_sign: Option<u32>,
    pub bit_shift_marker: Option<[u32; LIMB_BITS]>,
    pub limb_shift_marker: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_aux: Option<[u32; NUM_LIMBS]>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_shift_test(
    opcode: ShiftOpcode,
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_vals: ShiftPrankValues<BLOCK_FE_WIDTH, U16_BITS>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftRightArithmeticCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(a) = prank_vals.a {
            cols.a = a.map(F::from_u32);
        }
        if let Some(b_sign) = prank_vals.b_sign {
            cols.b_sign = F::from_u32(b_sign);
        }
        if let Some(bit_shift_marker) = prank_vals.bit_shift_marker {
            cols.bit_shift_marker = bit_shift_marker.map(F::from_u32);
        }
        if let Some(limb_shift_marker) = prank_vals.limb_shift_marker {
            cols.limb_shift_marker = limb_shift_marker.map(F::from_u32);
        }
        if let Some(bit_shift_carry) = prank_vals.bit_shift_carry {
            cols.bit_shift_carry = bit_shift_carry.map(F::from_u32);
        }
        if let Some(bit_shift_aux) = prank_vals.bit_shift_aux {
            cols.bit_shift_aux = bit_shift_aux.map(F::from_u32);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_sra_wrong_a_negative_test() {
    // SRA(1, 1) = 0; pranking a to 1 is wrong.
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        a: Some([1, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRA, b, c, prank_vals, false);
}

#[test]
fn rv64_sra_wrong_bit_shift_negative_test() {
    // SRA([0,...,0,128], 9): pranking bit_shift_marker to index 2 binds the core to a shift of 2,
    // which disagrees with the register operand bound by the execution interaction.
    let b = [0, 0, 0, 0, 0, 0, 0, 128];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let mut bit_shift_marker = [0u32; U16_BITS];
    bit_shift_marker[2] = 1;
    let prank_vals = ShiftPrankValues {
        bit_shift_marker: Some(bit_shift_marker),
        ..Default::default()
    };
    run_negative_shift_test(SRA, b, c, prank_vals, false);
}

#[test]
fn rv64_sra_wrong_limb_shift_negative_test() {
    // Shift by exactly one u16 limb; pranking limb_shift_marker to the wrong index breaks the
    // output recombination.
    let b = [0, 0, 0, 0, 0, 0, 0, 128];
    let c = [16, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 0, 1, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRA, b, c, prank_vals, false);
}

#[test]
fn rv64_sra_wrong_carry_negative_test() {
    // b = all 0xFFFF, shift by 9 bits. The low bits that cross the limb boundary are nonzero;
    // zeroing the carry breaks the decomposition (and the carry range check).
    let b = [255; RV64_REGISTER_NUM_LIMBS];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_carry: Some([0; BLOCK_FE_WIDTH]),
        ..Default::default()
    };
    run_negative_shift_test(SRA, b, c, prank_vals, true);
}

#[test]
fn rv64_sra_wrong_aux_negative_test() {
    // Zeroing the aux part breaks the b = carry + aux * 2^bit_shift decomposition.
    let b = [255; RV64_REGISTER_NUM_LIMBS];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_aux: Some([0; BLOCK_FE_WIDTH]),
        ..Default::default()
    };
    run_negative_shift_test(SRA, b, c, prank_vals, false);
}

#[test]
fn rv64_sra_wrong_sign_negative_test() {
    // b is negative (top u16 limb has its sign bit set), so b_sign should be 1. Pranking b_sign
    // to 0 fails the b_sign range check (b[NUM_LIMBS-1] no longer fits in LIMB_BITS-1 bits).
    let b = [0, 0, 0, 0, 0, 0, 0, 128];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        b_sign: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SRA, b, c, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sra_sanity_test() {
    let x = rv64_bytes_to_u16_block([31, 190, 221, 200, 45, 7, 61, 186]);
    let y = rv64_bytes_to_u16_block([81, 20, 50, 80, 49, 190, 190, 113]);
    let (result, limb_shift, bit_shift) =
        run_shift_right_arithmetic::<BLOCK_FE_WIDTH, U16_BITS>(&x, &y);
    // Reference: arithmetic shift the full signed 64-bit value right by (y[0] % 64) bits.
    let expected = ((i64::from_le_bytes([31, 190, 221, 200, 45, 7, 61, 186]) >> (81u32 % 64))
        as u64)
        .to_le_bytes();
    assert_eq!(rv64_u16_block_to_bytes(result), expected);
    let shift = (y[0] as usize) % (BLOCK_FE_WIDTH * U16_BITS);
    assert_eq!(shift / U16_BITS, limb_shift);
    assert_eq!(shift % U16_BITS, bit_shift);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64ShiftRightArithmeticExecutor,
    Rv64ShiftRightArithmeticAir,
    Rv64ShiftRightArithmeticChipGpu,
    Rv64ShiftRightArithmeticChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip =
        Rv64ShiftRightArithmeticChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(ShiftOpcode::SRA, 100)]
fn test_cuda_rand_shift_right_arithmetic_tracegen(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

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

    execute_boundary_shifts(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        opcode,
    );

    type Record<'a> = (
        &'a mut Rv64BaseAluRegU16AdapterRecord,
        &'a mut ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegU16AdapterExecutor>::new(),
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
fn test_cuda_shift_right_arithmetic_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let instruction = |rd: usize, rs1: usize, rs2: usize| {
        Instruction::<F>::from_usize(
            SRA.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                reg(rs2),
                RV64_REGISTER_AS as usize,
                RV64_REGISTER_AS as usize,
            ],
        )
    };
    let instructions = [
        // The upper rs2 limbs are nonzero and its low limb is 65, so only the low six bits apply.
        instruction(23, 1, 2),
        instruction(24, 3, 4),
        instruction(25, 1, 5),
        instruction(26, 1, 6),
        instruction(27, 1, 7),
        instruction(28, 1, 8),
        instruction(29, 1, 9),
        instruction(30, 1, 10),
        instruction(31, 3, 11),
        instruction(12, 0, 2),
        instruction(13, 1, 0),
        instruction(14, 14, 5),
        instruction(15, 1, 15),
        instruction(17, 18, 18),
        instruction(19, 19, 19),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let init_registers = [
        (1usize, 0x8123_4567_89ab_cdefu64),
        (2, 0xfeed_face_cafe_0041),
        (3, 0x7123_4567_89ab_cdef),
        (4, 0),
        (5, 15),
        (6, 16),
        (7, 31),
        (8, 32),
        (9, 63),
        (10, 64),
        (11, 1),
        (14, 0x9123_4567_89ab_cdef),
        (15, 48),
        (18, 0x7123_4567_89ab_cdef),
        (19, 0x8123_4567_89ab_cdef),
    ];
    let init_memory: SparseMemoryImage = init_registers
        .into_iter()
        .flat_map(|(register, value)| {
            value
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(move |(offset, byte)| {
                    ((RV64_REGISTER_AS, (reg(register) + offset) as u32), byte)
                })
        })
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config)
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(20, 45))
        .unwrap();

    let mut tester = GpuChipTestBuilder::default();
    let mut initial_image = GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
    initial_image.memory.set_from_sparse(&init_memory);
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
    for (pc, instruction) in instructions[..15].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64BaseAluRegU16AdapterRecord,
        &'a mut ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegU16AdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(SRA.global_opcode()).len(), 15);
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();

    let mut corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let first_write_timestamp = corrupt_transcript.program_log[0].timestamp + 2;
    corrupt_transcript
        .memory_log
        .iter_mut()
        .find(|event| event.timestamp == first_write_timestamp)
        .unwrap()
        .value[3] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_chip = Rv64ShiftRightArithmeticChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        tester.timestamp_max_bits(),
    );
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 158);

    // On the rd == rs1 == rs2 row, alter only an upper limb of the second read. That limb is
    // ignored by shift arithmetic, so output validation passes, but predecessor resolution must
    // reject it because the first read is its immediate predecessor.
    let mut predecessor_corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let alias_timestamp = predecessor_corrupt_transcript.program_log[14].timestamp;
    let rs1_index = predecessor_corrupt_transcript
        .memory_log
        .iter()
        .position(|event| event.timestamp == alias_timestamp)
        .unwrap();
    predecessor_corrupt_transcript.memory_log[rs1_index + 1].value[3] ^= 1;
    let (d_predecessor_corrupt, d_predecessor_corrupt_plan) = d_program
        .upload_transcript(
            &predecessor_corrupt_transcript,
            RvrPreflightEndpoint::Terminated,
        )
        .unwrap();
    let predecessor_corrupt_chip = Rv64ShiftRightArithmeticChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        tester.timestamp_max_bits(),
    );
    predecessor_corrupt_chip
        .generate_proving_ctx_from_rvr(
            &d_program,
            &d_predecessor_corrupt,
            &d_predecessor_corrupt_plan,
        )
        .unwrap();
    assert_eq!(d_predecessor_corrupt.error_code().unwrap(), 159);

    // This AIR always emits a destination write, so replay must reject x0 rather than synthesize
    // a disabled write row.
    let mut x0_instructions = instructions;
    x0_instructions[0] = instruction(0, 1, 2);
    let x0_program = Program::from_instructions(&x0_instructions);
    let d_x0_program = GpuRvrProgram::upload(&x0_program, &memory_config, device_ctx).unwrap();
    let (d_x0, d_x0_plan) = d_x0_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    let x0_chip = Rv64ShiftRightArithmeticChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        tester.timestamp_max_bits(),
    );
    x0_chip
        .generate_proving_ctx_from_rvr(&d_x0_program, &d_x0, &d_x0_plan)
        .unwrap();
    assert_eq!(d_x0.error_code().unwrap(), 154);

    let legacy_range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let legacy_chip = Rv64ShiftRightArithmeticChipGpu::new(
        legacy_range_checker.clone(),
        tester.timestamp_max_bits(),
    );
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace = <Rv64ShiftRightArithmeticChip<F> as Chip<
        MatrixRecordArena<F>,
        CpuBackend<SC>,
    >>::generate_proving_ctx(&harness.cpu_chip, harness.matrix_arena)
    .common_main;
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &replay_ctx.common_main, device_ctx);
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR SRA transcript replay proof failed");
}
