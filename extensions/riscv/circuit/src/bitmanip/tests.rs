use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::rngs::StdRng;
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::{Rv64BaseAluImmU16AdapterRecord, Rv64BaseAluRegU16AdapterRecord},
        Rv64BitManipImmChipGpu, Rv64BitManipRegChipGpu,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{
    core::*, Rv64BitManipImmAir, Rv64BitManipImmChip, Rv64BitManipImmExecutor, Rv64BitManipRegAir,
    Rv64BitManipRegChip, Rv64BitManipRegExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, Rv64BaseAluRegU16AdapterAir,
        Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller, RV64_REGISTER_NUM_LIMBS,
    },
    test_utils::rv64_rand_write_register_or_imm,
    BitManipImmCoreAir, BitManipImmFiller, BitManipRegCoreAir, BitManipRegFiller,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const REG_CASES: [(usize, u64, u64); REG_OP_COUNT] = [
    (SH1ADD, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
    (SH2ADD, 0x8000_0000_0000_0001, 0x7fff_ffff_ffff_ffff),
    (SH3ADD, 0xffff_ffff_ffff_ffff, 2),
    (ADD_UW, 0xffff_ffff_8000_0001, 9),
    (SH1ADD_UW, 0xffff_ffff_8000_0001, 9),
    (SH2ADD_UW, 0xffff_ffff_8000_0001, 9),
    (SH3ADD_UW, 0xffff_ffff_8000_0001, 9),
    (ANDN, 0x55aa_0ff0_f00f_aa55, 0x0f0f_f0f0_3333_cccc),
    (ORN, 0x55aa_0ff0_f00f_aa55, 0x0f0f_f0f0_3333_cccc),
    (XNOR, 0x55aa_0ff0_f00f_aa55, 0x0f0f_f0f0_3333_cccc),
    (ROL, 0x0123_4567_89ab_cdef, 13),
    (ROR, 0x0123_4567_89ab_cdef, 45),
    (ROLW, 0x89ab_cdef, 7),
    (RORW, 0x89ab_cdef, 11),
    (MIN, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (MINU, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (MAX, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (MAXU, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (BCLR, 0xffff_ffff_ffff_ffff, 63),
    (BSET, 0, 37),
    (BINV, 0x1000, 12),
    (BEXT, 0x8000_0000_0000_0000, 63),
];
const IMM_CASES: [(usize, u64, u32); IMM_OP_COUNT] = [
    (SLLI_UW, 0xffff_ffff_8000_0001, 37),
    (RORI, 0x0123_4567_89ab_cdef, 45),
    (RORIW, 0x0000_0000_89ab_cdef, 21),
    (CLZ, 0x0000_8000_0000_0000, 0),
    (CTZ, 0x0000_0000_0000_8000, 0),
    (CLZW, 0x0000_0000_0000_8000, 0),
    (CTZW, 0x0000_0000_8000_0000, 0),
    (CPOP, 0xf0f0_f0f0_1234_5678, 0),
    (CPOPW, 0xffff_0000_1234_5678, 0),
    (SEXT_B, 0x80, 0),
    (SEXT_H, 0x8001, 0),
    (ZEXT_H, 0xffff_ffff_ffff_8001, 0),
    (ORC_B, 0x0001_0200_0004_0000, 0),
    (REV8, 0x0123_4567_89ab_cdef, 0),
    (BCLRI, 0xffff_ffff_ffff_ffff, 47),
    (BSETI, 0, 48),
    (BINVI, 0x1000, 12),
    (BEXTI, 0x8000_0000_0000_0000, 63),
];

type RegHarness =
    TestChipHarness<F, Rv64BitManipRegExecutor, Rv64BitManipRegAir, Rv64BitManipRegChip<F>>;
type ImmHarness =
    TestChipHarness<F, Rv64BitManipImmExecutor, Rv64BitManipImmAir, Rv64BitManipImmChip<F>>;

fn create_reg_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BitManipRegAir,
    Rv64BitManipRegExecutor,
    Rv64BitManipRegChip<F>,
) {
    let air = Rv64BitManipRegAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        BitManipRegCoreAir::new(),
    );
    let executor = Rv64BitManipRegExecutor::new(Rv64BaseAluRegU16AdapterExecutor);
    let chip = Rv64BitManipRegChip::new(
        BitManipRegFiller::new(Rv64BaseAluRegU16AdapterFiller::new()),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_reg_harness(tester: &VmChipTestBuilder<F>) -> RegHarness {
    let (air, executor, chip) = create_reg_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
    );
    RegHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_imm_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BitManipImmAir,
    Rv64BitManipImmExecutor,
    Rv64BitManipImmChip<F>,
) {
    let air = Rv64BitManipImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge),
        BitManipImmCoreAir::new(),
    );
    let executor = Rv64BitManipImmExecutor::new(Rv64BaseAluImmU16AdapterExecutor);
    let chip = Rv64BitManipImmChip::new(
        BitManipImmFiller::new(Rv64BaseAluImmU16AdapterFiller::new()),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_imm_harness(tester: &VmChipTestBuilder<F>) -> ImmHarness {
    let (air, executor, chip) = create_imm_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
    );
    ImmHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn execute_reg<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    local_opcode: usize,
    rs1: u64,
    rs2: u64,
) {
    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        rs1.to_le_bytes(),
        rs2.to_le_bytes(),
        None,
        BITMANIP_OFFSET + local_opcode,
        rng,
    );
    tester.execute(executor, arena, &instruction);
    let expected = run_bitmanip_reg(local_opcode, rs1, rs2).to_le_bytes();
    assert_eq!(
        expected.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
}

fn execute_imm<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    local_opcode: usize,
    rs1: u64,
    imm: u32,
) {
    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        rs1.to_le_bytes(),
        [0; RV64_REGISTER_NUM_LIMBS],
        Some(imm as usize),
        BITMANIP_OFFSET + local_opcode,
        rng,
    );
    tester.execute(executor, arena, &instruction);
    let expected = run_bitmanip_imm(local_opcode, rs1, imm).to_le_bytes();
    assert_eq!(
        expected.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
}

#[test]
fn rv64_bitmanip_reg_chip_smoke() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_reg_harness(&tester);

    for (local_opcode, rs1, rs2) in REG_CASES {
        execute_reg(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            local_opcode,
            rs1,
            rs2,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("verification failed");
}

#[test]
fn rv64_bitmanip_imm_chip_smoke() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_imm_harness(&tester);

    for (local_opcode, rs1, imm) in IMM_CASES {
        execute_imm(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            local_opcode,
            rs1,
            imm,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("verification failed");
}

#[cfg(feature = "cuda")]
type GpuRegHarness = GpuTestChipHarness<
    F,
    Rv64BitManipRegExecutor,
    Rv64BitManipRegAir,
    Rv64BitManipRegChipGpu,
    Rv64BitManipRegChip<F>,
>;

#[cfg(feature = "cuda")]
type GpuImmHarness = GpuTestChipHarness<
    F,
    Rv64BitManipImmExecutor,
    Rv64BitManipImmAir,
    Rv64BitManipImmChipGpu,
    Rv64BitManipImmChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_reg_harness(tester: &GpuChipTestBuilder) -> GpuRegHarness {
    let (air, executor, cpu_chip) = create_reg_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BitManipRegChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_cuda_imm_harness(tester: &GpuChipTestBuilder) -> GpuImmHarness {
    let (air, executor, cpu_chip) = create_imm_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BitManipImmChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitmanip_reg_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_reg_harness(&tester);

    for (local_opcode, rs1, rs2) in REG_CASES {
        execute_reg(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            local_opcode,
            rs1,
            rs2,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluRegU16AdapterRecord,
        &'a mut BitManipRegCoreRecord,
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

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitmanip_imm_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_imm_harness(&tester);

    for (local_opcode, rs1, imm) in IMM_CASES {
        execute_imm(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            local_opcode,
            rs1,
            imm,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut BitManipImmCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
