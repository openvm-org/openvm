use std::sync::Arc;

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
    var_range::SharedVariableRangeCheckerChip,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::rngs::StdRng;
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::{
            Rv64BaseAluImmU16AdapterRecord, Rv64BaseAluRegAdapterRecord,
            Rv64BaseAluRegU16AdapterRecord,
        },
        Rv64BitManipBitwiseInvChipGpu, Rv64BitManipImmChipGpu, Rv64BitManipMinMaxChipGpu,
        Rv64BitManipRegChipGpu, Rv64BitManipShAddChipGpu, Rv64BitManipSlliUwChipGpu,
    },
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use super::{
    core::*, Rv64BitManipBitwiseInvAir, Rv64BitManipBitwiseInvChip, Rv64BitManipBitwiseInvExecutor,
    Rv64BitManipImmAir, Rv64BitManipImmChip, Rv64BitManipImmExecutor, Rv64BitManipMinMaxAir,
    Rv64BitManipMinMaxChip, Rv64BitManipMinMaxExecutor, Rv64BitManipRegAir, Rv64BitManipRegChip,
    Rv64BitManipRegExecutor, Rv64BitManipShAddAir, Rv64BitManipShAddChip,
    Rv64BitManipShAddExecutor, Rv64BitManipSlliUwAir, Rv64BitManipSlliUwChip,
    Rv64BitManipSlliUwExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, Rv64BaseAluRegAdapterAir, Rv64BaseAluRegAdapterExecutor,
        Rv64BaseAluRegAdapterFiller, Rv64BaseAluRegU16AdapterAir, Rv64BaseAluRegU16AdapterExecutor,
        Rv64BaseAluRegU16AdapterFiller, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    test_utils::rv64_rand_write_register_or_imm,
    BitManipBitwiseInvCoreAir, BitManipBitwiseInvFiller, BitManipImmCoreAir, BitManipImmFiller,
    BitManipMinMaxCoreAir, BitManipMinMaxFiller, BitManipRegCoreAir, BitManipRegFiller,
    BitManipSlliUwCoreAir, BitManipSlliUwFiller,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const SHADD_CASES: [(usize, u64, u64); SHADD_OP_COUNT] = [
    (SH1ADD, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
    (SH2ADD, 0x8000_0000_0000_0001, 0x7fff_ffff_ffff_ffff),
    (SH3ADD, 0xffff_ffff_ffff_ffff, 2),
    (ADD_UW, 0xffff_ffff_8000_0001, 9),
    (SH1ADD_UW, 0xffff_ffff_8000_0001, 9),
    (SH2ADD_UW, 0xffff_ffff_8000_0001, 9),
    (SH3ADD_UW, 0xffff_ffff_8000_0001, 9),
];
const SLLI_UW_CASES: [(usize, u64, u32); 4] = [
    (SLLI_UW, 0xffff_ffff_8000_0001, 0),
    (SLLI_UW, 0xffff_ffff_8000_0001, 15),
    (SLLI_UW, 0xffff_ffff_8000_0001, 37),
    (SLLI_UW, 0x0000_0000_dead_beef, 63),
];
const REG_CASES: [(usize, u64, u64); 8] = [
    (ROL, 0x0123_4567_89ab_cdef, 13),
    (ROR, 0x0123_4567_89ab_cdef, 45),
    (ROLW, 0x89ab_cdef, 7),
    (RORW, 0x89ab_cdef, 11),
    (BCLR, 0xffff_ffff_ffff_ffff, 63),
    (BSET, 0, 37),
    (BINV, 0x1000, 12),
    (BEXT, 0x8000_0000_0000_0000, 63),
];
const BITWISE_INV_CASES: [(usize, u64, u64); 5] = [
    (ANDN, 0x55aa_0ff0_f00f_aa55, 0x0f0f_f0f0_3333_cccc),
    (ORN, 0x55aa_0ff0_f00f_aa55, 0x0f0f_f0f0_3333_cccc),
    (XNOR, 0x55aa_0ff0_f00f_aa55, 0x0f0f_f0f0_3333_cccc),
    (ANDN, 0, 0),
    (XNOR, u64::MAX, u64::MAX),
];
const MIN_MAX_CASES: [(usize, u64, u64); 8] = [
    (MIN, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (MINU, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (MAX, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    (MAXU, 0xffff_ffff_ffff_ff00, 0x7fff_ffff_ffff_ffff),
    // Equal operands (no differing limb).
    (MIN, 0x0123_4567_89ab_cdef, 0x0123_4567_89ab_cdef),
    // Signed boundary: i64::MIN vs zero.
    (MAX, 0x8000_0000_0000_0000, 0),
    (MINU, 0x8000_0000_0000_0000, 0),
    // Difference only in the lowest limb.
    (MAXU, 0xffff_ffff_ffff_ffff, 0xffff_ffff_ffff_fffe),
];
const IMM_CASES: [(usize, u64, u32); IMM_OP_COUNT] = [
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
type ShAddHarness =
    TestChipHarness<F, Rv64BitManipShAddExecutor, Rv64BitManipShAddAir, Rv64BitManipShAddChip<F>>;
type SlliUwHarness = TestChipHarness<
    F,
    Rv64BitManipSlliUwExecutor,
    Rv64BitManipSlliUwAir,
    Rv64BitManipSlliUwChip<F>,
>;
type ImmHarness =
    TestChipHarness<F, Rv64BitManipImmExecutor, Rv64BitManipImmAir, Rv64BitManipImmChip<F>>;
type BitwiseInvHarness = TestChipHarness<
    F,
    Rv64BitManipBitwiseInvExecutor,
    Rv64BitManipBitwiseInvAir,
    Rv64BitManipBitwiseInvChip<F>,
>;
type MinMaxHarness = TestChipHarness<
    F,
    Rv64BitManipMinMaxExecutor,
    Rv64BitManipMinMaxAir,
    Rv64BitManipMinMaxChip<F>,
>;

fn create_shadd_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
) -> (
    Rv64BitManipShAddAir,
    Rv64BitManipShAddExecutor,
    Rv64BitManipShAddChip<F>,
) {
    let air = Rv64BitManipShAddAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        BitManipShAddCoreAir::new(range_checker.bus()),
    );
    let executor = Rv64BitManipShAddExecutor::new(Rv64BaseAluRegU16AdapterExecutor);
    let chip = Rv64BitManipShAddChip::new(
        BitManipShAddFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_shadd_harness(tester: &VmChipTestBuilder<F>) -> ShAddHarness {
    let (air, executor, chip) = create_shadd_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
        tester.range_checker(),
    );
    ShAddHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_slli_uw_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
) -> (
    Rv64BitManipSlliUwAir,
    Rv64BitManipSlliUwExecutor,
    Rv64BitManipSlliUwChip<F>,
) {
    let air = Rv64BitManipSlliUwAir::new(
        Rv64BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge),
        BitManipSlliUwCoreAir::new(range_checker.bus()),
    );
    let executor = Rv64BitManipSlliUwExecutor::new(Rv64BaseAluImmU16AdapterExecutor);
    let chip = Rv64BitManipSlliUwChip::new(
        BitManipSlliUwFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_slli_uw_harness(tester: &VmChipTestBuilder<F>) -> SlliUwHarness {
    let (air, executor, chip) = create_slli_uw_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
        tester.range_checker(),
    );
    SlliUwHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

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

fn create_bitwise_inv_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
    bitwise_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
) -> (
    Rv64BitManipBitwiseInvAir,
    Rv64BitManipBitwiseInvExecutor,
    Rv64BitManipBitwiseInvChip<F>,
) {
    let air = Rv64BitManipBitwiseInvAir::new(
        Rv64BaseAluRegAdapterAir::new(execution_bridge, memory_bridge),
        BitManipBitwiseInvCoreAir::new(bitwise_chip.bus()),
    );
    let executor = Rv64BitManipBitwiseInvExecutor::new(Rv64BaseAluRegAdapterExecutor::new());
    let chip = Rv64BitManipBitwiseInvChip::new(
        BitManipBitwiseInvFiller::new(Rv64BaseAluRegAdapterFiller, bitwise_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_bitwise_inv_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    BitwiseInvHarness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let (air, executor, chip) = create_bitwise_inv_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
        bitwise_chip.clone(),
    );
    let harness = BitwiseInvHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    (harness, (bitwise_chip.air, bitwise_chip))
}

fn create_min_max_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    memory_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
) -> (
    Rv64BitManipMinMaxAir,
    Rv64BitManipMinMaxExecutor,
    Rv64BitManipMinMaxChip<F>,
) {
    let air = Rv64BitManipMinMaxAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        BitManipMinMaxCoreAir::new(range_checker.bus()),
    );
    let executor = Rv64BitManipMinMaxExecutor::new(Rv64BaseAluRegU16AdapterExecutor);
    let chip = Rv64BitManipMinMaxChip::new(
        BitManipMinMaxFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_min_max_harness(tester: &VmChipTestBuilder<F>) -> MinMaxHarness {
    let (air, executor, chip) = create_min_max_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.memory_helper(),
        tester.range_checker(),
    );
    MinMaxHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
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
fn rv64_bitmanip_shadd_chip_smoke() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_shadd_harness(&tester);

    for (local_opcode, rs1, rs2) in SHADD_CASES {
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
fn rv64_bitmanip_slli_uw_chip_smoke() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_slli_uw_harness(&tester);

    for (local_opcode, rs1, imm) in SLLI_UW_CASES {
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

#[test]
fn rv64_bitmanip_bitwise_inv_chip_smoke() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_bitwise_inv_harness(&tester);

    for (local_opcode, rs1, rs2) in BITWISE_INV_CASES {
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

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("verification failed");
}

#[test]
fn rv64_bitmanip_min_max_chip_smoke() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_min_max_harness(&tester);

    for (local_opcode, rs1, rs2) in MIN_MAX_CASES {
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

#[cfg(feature = "cuda")]
type GpuRegHarness = GpuTestChipHarness<
    F,
    Rv64BitManipRegExecutor,
    Rv64BitManipRegAir,
    Rv64BitManipRegChipGpu,
    Rv64BitManipRegChip<F>,
>;

#[cfg(feature = "cuda")]
type GpuBitwiseInvHarness = GpuTestChipHarness<
    F,
    Rv64BitManipBitwiseInvExecutor,
    Rv64BitManipBitwiseInvAir,
    Rv64BitManipBitwiseInvChipGpu,
    Rv64BitManipBitwiseInvChip<F>,
>;

#[cfg(feature = "cuda")]
type GpuMinMaxHarness = GpuTestChipHarness<
    F,
    Rv64BitManipMinMaxExecutor,
    Rv64BitManipMinMaxAir,
    Rv64BitManipMinMaxChipGpu,
    Rv64BitManipMinMaxChip<F>,
>;

#[cfg(feature = "cuda")]
type GpuShAddHarness = GpuTestChipHarness<
    F,
    Rv64BitManipShAddExecutor,
    Rv64BitManipShAddAir,
    Rv64BitManipShAddChipGpu,
    Rv64BitManipShAddChip<F>,
>;

#[cfg(feature = "cuda")]
type GpuSlliUwHarness = GpuTestChipHarness<
    F,
    Rv64BitManipSlliUwExecutor,
    Rv64BitManipSlliUwAir,
    Rv64BitManipSlliUwChipGpu,
    Rv64BitManipSlliUwChip<F>,
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
fn create_cuda_shadd_harness(tester: &GpuChipTestBuilder) -> GpuShAddHarness {
    // The CPU chip only regenerates the trace for comparison against the GPU
    // trace; its range-checker counts must be discarded (the GPU kernel already
    // adds them), so give it a dummy chip rather than the hybrid mirror.
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_shadd_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
        dummy_range_checker_chip,
    );
    let gpu_chip =
        Rv64BitManipShAddChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_cuda_slli_uw_harness(tester: &GpuChipTestBuilder) -> GpuSlliUwHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_slli_uw_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
        dummy_range_checker_chip,
    );
    let gpu_chip =
        Rv64BitManipSlliUwChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

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
fn create_cuda_bitwise_inv_harness(tester: &GpuChipTestBuilder) -> GpuBitwiseInvHarness {
    // The CPU chip only regenerates the trace for comparison; its lookup
    // requests must be discarded (the GPU kernel already adds them).
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));
    let (air, executor, cpu_chip) = create_bitwise_inv_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
        dummy_bitwise_chip,
    );
    let gpu_chip = Rv64BitManipBitwiseInvChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_cuda_min_max_harness(tester: &GpuChipTestBuilder) -> GpuMinMaxHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_min_max_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.dummy_memory_helper(),
        dummy_range_checker_chip,
    );
    let gpu_chip =
        Rv64BitManipMinMaxChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitmanip_shadd_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_shadd_harness(&tester);

    for (local_opcode, rs1, rs2) in SHADD_CASES {
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
        &'a mut BitManipShAddCoreRecord,
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
fn test_cuda_bitmanip_slli_uw_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_slli_uw_harness(&tester);

    for (local_opcode, rs1, imm) in SLLI_UW_CASES {
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
        &'a mut BitManipSlliUwCoreRecord,
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

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitmanip_bitwise_inv_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut harness = create_cuda_bitwise_inv_harness(&tester);

    for (local_opcode, rs1, rs2) in BITWISE_INV_CASES {
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
        &'a mut Rv64BaseAluRegAdapterRecord,
        &'a mut BitManipBitwiseInvCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegAdapterExecutor>::new(),
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
fn test_cuda_bitmanip_min_max_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_min_max_harness(&tester);

    for (local_opcode, rs1, rs2) in MIN_MAX_CASES {
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
        &'a mut BitManipMinMaxCoreRecord,
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
