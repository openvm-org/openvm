use std::sync::Arc;

use openvm_bigint_circuit::*;
use openvm_circuit::{
    arch::{
        testing::RANGE_TUPLE_CHECKER_BUS, DenseRecordArena, EmptyAdapterCoreLayout,
        InstructionExecutor, VmAirWrapper, VmChipWrapper,
    },
    utils::generate_long_number,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChip,
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip},
    var_range::VariableRangeCheckerChip,
};
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode, VmOpcode,
};
use openvm_rv32_adapters::{
    Rv32HeapAdapterAir, Rv32HeapAdapterFiller, Rv32HeapAdapterStep, Rv32HeapBranchAdapterAir,
    Rv32HeapBranchAdapterFiller, Rv32HeapBranchAdapterStep,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS, RV_B_TYPE_IMM_BITS},
    BaseAluCoreAir, BaseAluFiller, BranchEqualCoreAir, BranchEqualFiller, BranchLessThanCoreAir,
    BranchLessThanFiller, LessThanCoreAir, LessThanFiller, MultiplicationCoreAir,
    MultiplicationFiller, ShiftCoreAir, ShiftFiller,
};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use super::*;
use crate::testing::{
    default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
    GpuTestChipHarness,
};

type F = BabyBear;

const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_BRANCH: isize = 1 << (RV_B_TYPE_IMM_BITS - 1);

fn rv32_write_heap_default<const NUM_LIMBS: usize>(
    tester: &mut GpuChipTestBuilder,
    addr1_writes: Vec<[F; NUM_LIMBS]>,
    addr2_writes: Vec<[F; NUM_LIMBS]>,
    opcode_with_offset: usize,
) -> Instruction<F> {
    let (reg1, _) = tester.write_heap_default::<NUM_LIMBS>(4, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) = tester.write_heap_default::<NUM_LIMBS>(4, 128, addr2_writes);
        reg2
    };
    let (reg3, _) = tester.write_heap_pointer_default(4, 128);

    Instruction::from_isize(
        VmOpcode::from_usize(opcode_with_offset),
        reg3 as isize,
        reg1 as isize,
        reg2 as isize,
        1_isize,
        2_isize,
    )
}

fn rv32_heap_branch_default<const NUM_LIMBS: usize>(
    tester: &mut GpuChipTestBuilder,
    addr1_writes: Vec<[F; NUM_LIMBS]>,
    addr2_writes: Vec<[F; NUM_LIMBS]>,
    imm: isize,
    opcode_with_offset: usize,
) -> Instruction<F> {
    let (reg1, _) = tester.write_heap_default::<NUM_LIMBS>(4, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) = tester.write_heap_default::<NUM_LIMBS>(4, 128, addr2_writes);
        reg2
    };

    Instruction::from_isize(
        VmOpcode::from_usize(opcode_with_offset),
        reg1 as isize,
        reg2 as isize,
        imm,
        1_isize,
        2_isize,
    )
}

#[allow(clippy::type_complexity)]
fn set_and_execute_rand<STEP, AIR, GpuChip, CpuChip>(
    tester: &mut GpuChipTestBuilder,
    harness: &mut GpuTestChipHarness<F, STEP, AIR, GpuChip, CpuChip>,
    rng: &mut StdRng,
    opcode: usize,
    branch_fn: Option<fn(usize, &[u32; INT256_NUM_LIMBS], &[u32; INT256_NUM_LIMBS]) -> bool>,
) where
    STEP: InstructionExecutor<F, DenseRecordArena>,
{
    let branch = branch_fn.is_some();

    let b = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(rng);
    let c = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(rng);
    if branch {
        let imm = rng.gen_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
        let instruction = rv32_heap_branch_default(
            tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            imm,
            opcode,
        );

        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
            rng.gen_range((ABS_MAX_BRANCH as u32)..(1 << (PC_BITS - 1))),
        );
    } else {
        let instruction = rv32_write_heap_default(
            tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            opcode,
        );
        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
        );
    }
}

type BaseAluHarness = GpuTestChipHarness<
    F,
    Rv32BaseAlu256Step,
    Rv32BaseAlu256Air,
    BaseAlu256ChipGpu,
    VmChipWrapper<
        F,
        BaseAluFiller<
            Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
            INT256_NUM_LIMBS,
            RV32_CELL_BITS,
        >,
    >,
>;

type BranchEqualHarness = GpuTestChipHarness<
    F,
    Rv32BranchEqual256Step,
    Rv32BranchEqual256Air,
    BranchEqual256ChipGpu,
    VmChipWrapper<
        F,
        BranchEqualFiller<Rv32HeapBranchAdapterFiller<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>,
    >,
>;

type LessThanHarness = GpuTestChipHarness<
    F,
    Rv32LessThan256Step,
    Rv32LessThan256Air,
    LessThan256ChipGpu,
    VmChipWrapper<
        F,
        LessThanFiller<
            Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
            INT256_NUM_LIMBS,
            RV32_CELL_BITS,
        >,
    >,
>;

type BranchLessThanHarness = GpuTestChipHarness<
    F,
    Rv32BranchLessThan256Step,
    Rv32BranchLessThan256Air,
    BranchLessThan256ChipGpu,
    VmChipWrapper<
        F,
        BranchLessThanFiller<
            Rv32HeapBranchAdapterFiller<2, INT256_NUM_LIMBS>,
            INT256_NUM_LIMBS,
            RV32_CELL_BITS,
        >,
    >,
>;

type Shift256Harness = GpuTestChipHarness<
    F,
    Rv32Shift256Step,
    Rv32Shift256Air,
    Shift256ChipGpu,
    VmChipWrapper<
        F,
        ShiftFiller<
            Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
            INT256_NUM_LIMBS,
            RV32_CELL_BITS,
        >,
    >,
>;

type Multiplication256Harness = GpuTestChipHarness<
    F,
    Rv32Multiplication256Step,
    Rv32Multiplication256Air,
    Multiplication256ChipGpu,
    VmChipWrapper<
        F,
        MultiplicationFiller<
            Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
            INT256_NUM_LIMBS,
            RV32_CELL_BITS,
        >,
    >,
>;

fn create_alu_test_harness(tester: &GpuChipTestBuilder) -> BaseAluHarness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = VmAirWrapper::new(
        Rv32HeapAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        BaseAluCoreAir::new(bitwise_bus, BaseAluOpcode::CLASS_OFFSET),
    );
    let executor = Rv32BaseAlu256Step::new(
        Rv32HeapAdapterStep::new(tester.address_bits()),
        BaseAluOpcode::CLASS_OFFSET,
    );

    let cpu_chip = VmChipWrapper::new(
        BaseAluFiller::new(
            Rv32HeapAdapterFiller::<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
                tester.address_bits(),
                dummy_bitwise_chip.clone(),
            ),
            dummy_bitwise_chip,
            BaseAluOpcode::CLASS_OFFSET,
        ),
        tester.dummy_memory_helper(),
    );

    let gpu_chip = BaseAlu256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn create_branch_equal_test_harness(tester: &GpuChipTestBuilder) -> BranchEqualHarness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = VmAirWrapper::new(
        Rv32HeapBranchAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = Rv32BranchEqual256Step::new(
        Rv32HeapBranchAdapterStep::new(tester.address_bits()),
        BranchEqualOpcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );

    let cpu_chip = VmChipWrapper::new(
        BranchEqualFiller::new(
            Rv32HeapBranchAdapterFiller::<2, INT256_NUM_LIMBS>::new(
                tester.address_bits(),
                dummy_bitwise_chip.clone(),
            ),
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        tester.dummy_memory_helper(),
    );

    let gpu_chip = BranchEqual256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn create_less_than_test_harness(tester: &GpuChipTestBuilder) -> LessThanHarness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = VmAirWrapper::new(
        Rv32HeapAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        LessThanCoreAir::new(bitwise_bus, LessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv32LessThan256Step::new(
        Rv32HeapAdapterStep::new(tester.address_bits()),
        LessThanOpcode::CLASS_OFFSET,
    );

    let cpu_chip = VmChipWrapper::new(
        LessThanFiller::new(
            Rv32HeapAdapterFiller::<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
                tester.address_bits(),
                dummy_bitwise_chip.clone(),
            ),
            dummy_bitwise_chip,
            LessThanOpcode::CLASS_OFFSET,
        ),
        tester.dummy_memory_helper(),
    );

    let gpu_chip = LessThan256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn create_branch_less_than_test_harness(tester: &GpuChipTestBuilder) -> BranchLessThanHarness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = VmAirWrapper::new(
        Rv32HeapBranchAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        BranchLessThanCoreAir::new(bitwise_bus, BranchLessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv32BranchLessThan256Step::new(
        Rv32HeapBranchAdapterStep::new(tester.address_bits()),
        BranchLessThanOpcode::CLASS_OFFSET,
    );

    let cpu_chip = VmChipWrapper::new(
        BranchLessThanFiller::new(
            Rv32HeapBranchAdapterFiller::<2, INT256_NUM_LIMBS>::new(
                tester.address_bits(),
                dummy_bitwise_chip.clone(),
            ),
            dummy_bitwise_chip,
            BranchLessThanOpcode::CLASS_OFFSET,
        ),
        tester.dummy_memory_helper(),
    );

    let gpu_chip = BranchLessThan256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn create_shift_test_harness(tester: &GpuChipTestBuilder) -> Shift256Harness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_bus = default_var_range_checker_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let air = VmAirWrapper::new(
        Rv32HeapAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        ShiftCoreAir::new(bitwise_bus, range_bus, ShiftOpcode::CLASS_OFFSET),
    );
    let executor = Rv32Shift256Step::new(
        Rv32HeapAdapterStep::new(tester.address_bits()),
        ShiftOpcode::CLASS_OFFSET,
    );

    let cpu_chip = VmChipWrapper::new(
        ShiftFiller::new(
            Rv32HeapAdapterFiller::<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
                tester.address_bits(),
                dummy_bitwise_chip.clone(),
            ),
            dummy_bitwise_chip,
            dummy_range_checker,
            ShiftOpcode::CLASS_OFFSET,
        ),
        tester.dummy_memory_helper(),
    );

    let gpu_chip = Shift256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn create_multiplication_test_harness(tester: &GpuChipTestBuilder) -> Multiplication256Harness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [
            1 << RV32_CELL_BITS,
            (INT256_NUM_LIMBS * (1 << RV32_CELL_BITS)) as u32,
        ],
    );

    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let air = VmAirWrapper::new(
        Rv32HeapAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        ),
        MultiplicationCoreAir::new(range_tuple_bus, MulOpcode::CLASS_OFFSET),
    );
    let executor = Rv32Multiplication256Step::new(
        Rv32HeapAdapterStep::new(tester.address_bits()),
        MulOpcode::CLASS_OFFSET,
    );

    let cpu_chip = VmChipWrapper::new(
        MultiplicationFiller::new(
            Rv32HeapAdapterFiller::<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::new(
                tester.address_bits(),
                dummy_bitwise_chip.clone(),
            ),
            dummy_range_tuple_chip.clone(),
            MulOpcode::CLASS_OFFSET,
        ),
        tester.dummy_memory_helper(),
    );

    let gpu_chip = Multiplication256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[test_case(BaseAluOpcode::ADD.global_opcode_usize(), 24)]
#[test_case(BaseAluOpcode::SUB.global_opcode_usize(), 24)]
#[test_case(BaseAluOpcode::XOR.global_opcode_usize(), 24)]
#[test_case(BaseAluOpcode::OR.global_opcode_usize(), 24)]
#[test_case(BaseAluOpcode::AND.global_opcode_usize(), 24)]
fn rand_alu_256_tracegen_test(opcode: usize, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_alu_test_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute_rand(&mut tester, &mut harness, &mut rng, opcode, None);
    }

    type Record<'a> = (
        &'a mut BaseAlu256AdapterRecord,
        &'a mut BaseAlu256CoreRecord,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(BranchEqualOpcode::BEQ.global_opcode_usize(), 24)]
#[test_case(BranchEqualOpcode::BNE.global_opcode_usize(), 24)]
fn rand_branch_equal_256_tracegen_test(opcode: usize, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_branch_equal_test_harness(&tester);

    let branch_fn = |opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| {
        x.iter()
            .zip(y.iter())
            .fold(true, |acc, (x, y)| acc && (x == y))
            ^ (opcode == BranchEqualOpcode::BNE.local_usize() + BranchEqualOpcode::CLASS_OFFSET)
    };

    for _ in 0..num_ops {
        set_and_execute_rand(&mut tester, &mut harness, &mut rng, opcode, Some(branch_fn));
    }

    type Record<'a> = (
        &'a mut BranchEqual256AdapterRecord,
        &'a mut BranchEqual256CoreRecord,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(LessThanOpcode::SLT, 24)]
#[test_case(LessThanOpcode::SLTU, 24)]
fn run_lt_256_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_less_than_test_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode.local_usize() + LessThanOpcode::CLASS_OFFSET,
            None,
        );
    }

    type Record<'a> = (
        &'a mut LessThan256AdapterRecord,
        &'a mut LessThan256CoreRecord,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(BranchLessThanOpcode::BLT, 24)]
#[test_case(BranchLessThanOpcode::BLTU, 24)]
#[test_case(BranchLessThanOpcode::BGE, 24)]
#[test_case(BranchLessThanOpcode::BGEU, 24)]
fn run_blt_256_rand_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let branch_fn =
        |opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| -> bool {
            let opcode =
                BranchLessThanOpcode::from_usize(opcode - BranchLessThanOpcode::CLASS_OFFSET);
            let (is_ge, is_signed) = match opcode {
                BranchLessThanOpcode::BLT => (false, true),
                BranchLessThanOpcode::BLTU => (false, false),
                BranchLessThanOpcode::BGE => (true, true),
                BranchLessThanOpcode::BGEU => (true, false),
            };
            let x_sign = x[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) != 0 && is_signed;
            let y_sign = y[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) != 0 && is_signed;
            for (x, y) in x.iter().rev().zip(y.iter().rev()) {
                if x != y {
                    return (x < y) ^ x_sign ^ y_sign ^ is_ge;
                }
            }
            is_ge
        };

    let mut harness = create_branch_less_than_test_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode.local_usize() + BranchLessThanOpcode::CLASS_OFFSET,
            Some(branch_fn),
        );
    }

    type Record<'a> = (
        &'a mut BranchLessThan256AdapterRecord,
        &'a mut BranchLessThan256CoreRecord,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(ShiftOpcode::SLL, 24)]
#[test_case(ShiftOpcode::SRL, 24)]
#[test_case(ShiftOpcode::SRA, 24)]
fn run_shift_256_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_shift_test_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode.local_usize() + ShiftOpcode::CLASS_OFFSET,
            None,
        );
    }

    type Record<'a> = (&'a mut Shift256AdapterRecord, &'a mut Shift256CoreRecord);

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test_case(MulOpcode::MUL, 24)]
fn run_mul_256_rand_test(opcode: MulOpcode, num_ops: usize) {
    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [
            1 << RV32_CELL_BITS,
            (INT256_NUM_LIMBS * (1 << RV32_CELL_BITS)) as u32,
        ],
    );
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(default_bitwise_lookup_bus())
        .with_range_tuple_checker(range_tuple_bus);

    let mut harness = create_multiplication_test_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode.local_usize() + MulOpcode::CLASS_OFFSET,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Multiplication256AdapterRecord,
        &'a mut Multiplication256CoreRecord,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
