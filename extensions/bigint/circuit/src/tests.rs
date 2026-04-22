use std::sync::Arc;

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        testing::{
            TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
            RANGE_TUPLE_CHECKER_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::generate_long_number,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip},
    var_range::VariableRangeCheckerChip,
};
use openvm_instructions::{
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV64_CELL_BITS,
    LocalOpcode,
};
use openvm_riscv_adapters::{
    rv64_heap_branch_default, rv64_write_heap_default, Rv64VecHeapAdapterAir,
    Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller, Rv64VecHeapBranchAdapterAir,
    Rv64VecHeapBranchAdapterExecutor, Rv64VecHeapBranchAdapterFiller,
};
use openvm_riscv_circuit::{
    adapters::{INT256_NUM_LIMBS, RV_B_TYPE_IMM_BITS},
    BaseAluCoreAir, BaseAluFiller, BranchEqualCoreAir, BranchEqualFiller, BranchLessThanCoreAir,
    BranchLessThanFiller, LessThanCoreAir, LessThanFiller, MultiplicationCoreAir,
    MultiplicationFiller, ShiftCoreAir, ShiftFiller,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{
        BaseAlu256AdapterRecord, BaseAlu256ChipGpu, BaseAlu256CoreRecord,
        BranchEqual256AdapterRecord, BranchEqual256ChipGpu, BranchEqual256CoreRecord,
        BranchLessThan256AdapterRecord, BranchLessThan256ChipGpu, BranchLessThan256CoreRecord,
        LessThan256AdapterRecord, LessThan256ChipGpu, LessThan256CoreRecord,
        Multiplication256AdapterRecord, Multiplication256ChipGpu, Multiplication256CoreRecord,
        Shift256AdapterRecord, Shift256ChipGpu, Shift256CoreRecord, INT256_NUM_BLOCKS, NUM_READS,
    },
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout, DEFAULT_BLOCK_SIZE,
    },
};

use crate::{
    AluAdapterAir, AluAdapterExecutor, BranchAdapterAir, BranchAdapterExecutor, Rv64BaseAlu256Air,
    Rv64BaseAlu256Chip, Rv64BaseAlu256Executor, Rv64BranchEqual256Air, Rv64BranchEqual256Chip,
    Rv64BranchEqual256Executor, Rv64BranchLessThan256Air, Rv64BranchLessThan256Chip,
    Rv64BranchLessThan256Executor, Rv64LessThan256Air, Rv64LessThan256Chip,
    Rv64LessThan256Executor, Rv64Multiplication256Air, Rv64Multiplication256Chip,
    Rv64Multiplication256Executor, Rv64Shift256Air, Rv64Shift256Chip, Rv64Shift256Executor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_BRANCH: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
const RANGE_TUPLE_SIZES: [u32; 2] = [
    1 << RV64_CELL_BITS,
    (INT256_NUM_LIMBS * (1 << RV64_CELL_BITS)) as u32,
];

fn create_alu_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64BaseAlu256Air,
    Rv64BaseAlu256Executor,
    Rv64BaseAlu256Chip<F>,
) {
    let air = Rv64BaseAlu256Air::new(
        AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
            execution_bridge,
            memory_bridge,
            bitwise_chip.bus(),
            address_bits,
        )),
        BaseAluCoreAir::new(bitwise_chip.bus(), Rv64BaseAlu256Opcode::CLASS_OFFSET),
    );
    let executor = Rv64BaseAlu256Executor::new(
        AluAdapterExecutor::new(Rv64VecHeapAdapterExecutor::new(address_bits)),
        Rv64BaseAlu256Opcode::CLASS_OFFSET,
    );
    let chip = Rv64BaseAlu256Chip::new(
        BaseAluFiller::new(
            Rv64VecHeapAdapterFiller::new(address_bits, bitwise_chip.clone()),
            bitwise_chip,
            Rv64BaseAlu256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_lt_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64LessThan256Air,
    Rv64LessThan256Executor,
    Rv64LessThan256Chip<F>,
) {
    let air = Rv64LessThan256Air::new(
        AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
            execution_bridge,
            memory_bridge,
            bitwise_chip.bus(),
            address_bits,
        )),
        LessThanCoreAir::new(bitwise_chip.bus(), Rv64LessThan256Opcode::CLASS_OFFSET),
    );
    let executor = Rv64LessThan256Executor::new(
        AluAdapterExecutor::new(Rv64VecHeapAdapterExecutor::new(address_bits)),
        Rv64LessThan256Opcode::CLASS_OFFSET,
    );
    let chip = Rv64LessThan256Chip::new(
        LessThanFiller::new(
            Rv64VecHeapAdapterFiller::new(address_bits, bitwise_chip.clone()),
            bitwise_chip.clone(),
            Rv64LessThan256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_mul_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64Multiplication256Air,
    Rv64Multiplication256Executor,
    Rv64Multiplication256Chip<F>,
) {
    let air = Rv64Multiplication256Air::new(
        AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
            execution_bridge,
            memory_bridge,
            bitwise_chip.bus(),
            address_bits,
        )),
        MultiplicationCoreAir::new(*range_tuple_chip.bus(), Rv64Mul256Opcode::CLASS_OFFSET),
    );
    let executor = Rv64Multiplication256Executor::new(
        AluAdapterExecutor::new(Rv64VecHeapAdapterExecutor::new(address_bits)),
        Rv64Mul256Opcode::CLASS_OFFSET,
    );
    let chip = Rv64Multiplication256Chip::<F>::new(
        MultiplicationFiller::new(
            Rv64VecHeapAdapterFiller::new(address_bits, bitwise_chip),
            range_tuple_chip,
            Rv64Mul256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_shift_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (Rv64Shift256Air, Rv64Shift256Executor, Rv64Shift256Chip<F>) {
    let air = Rv64Shift256Air::new(
        AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
            execution_bridge,
            memory_bridge,
            bitwise_chip.bus(),
            address_bits,
        )),
        ShiftCoreAir::new(
            bitwise_chip.bus(),
            range_checker_chip.bus(),
            Rv64Shift256Opcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64Shift256Executor::new(
        AluAdapterExecutor::new(Rv64VecHeapAdapterExecutor::new(address_bits)),
        Rv64Shift256Opcode::CLASS_OFFSET,
    );
    let chip = Rv64Shift256Chip::new(
        ShiftFiller::new(
            Rv64VecHeapAdapterFiller::new(address_bits, bitwise_chip.clone()),
            bitwise_chip.clone(),
            range_checker_chip.clone(),
            Rv64Shift256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_beq_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64BranchEqual256Air,
    Rv64BranchEqual256Executor,
    Rv64BranchEqual256Chip<F>,
) {
    let air = Rv64BranchEqual256Air::new(
        BranchAdapterAir::new(Rv64VecHeapBranchAdapterAir::new(
            execution_bridge,
            memory_bridge,
            bitwise_chip.bus(),
            address_bits,
        )),
        BranchEqualCoreAir::new(Rv64BranchEqual256Opcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = Rv64BranchEqual256Executor::new(
        BranchAdapterExecutor::new(Rv64VecHeapBranchAdapterExecutor::new(address_bits)),
        Rv64BranchEqual256Opcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );
    let chip = Rv64BranchEqual256Chip::new(
        BranchEqualFiller::new(
            Rv64VecHeapBranchAdapterFiller::new(address_bits, bitwise_chip),
            Rv64BranchEqual256Opcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_blt_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64BranchLessThan256Air,
    Rv64BranchLessThan256Executor,
    Rv64BranchLessThan256Chip<F>,
) {
    let air = Rv64BranchLessThan256Air::new(
        BranchAdapterAir::new(Rv64VecHeapBranchAdapterAir::new(
            execution_bridge,
            memory_bridge,
            bitwise_chip.bus(),
            address_bits,
        )),
        BranchLessThanCoreAir::new(
            bitwise_chip.bus(),
            Rv64BranchLessThan256Opcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64BranchLessThan256Executor::new(
        BranchAdapterExecutor::new(Rv64VecHeapBranchAdapterExecutor::new(address_bits)),
        Rv64BranchLessThan256Opcode::CLASS_OFFSET,
    );
    let chip = Rv64BranchLessThan256Chip::new(
        BranchLessThanFiller::new(
            Rv64VecHeapBranchAdapterFiller::new(address_bits, bitwise_chip.clone()),
            bitwise_chip,
            Rv64BranchLessThan256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn beq_branch_fn(opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]) -> bool {
    x.iter()
        .zip(y.iter())
        .fold(true, |acc, (x, y)| acc && (x == y))
        ^ (opcode == BranchEqualOpcode::BNE.local_usize() + Rv64BranchEqual256Opcode::CLASS_OFFSET)
}

fn blt_branch_fn(opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]) -> bool {
    let opcode =
        BranchLessThanOpcode::from_usize(opcode - Rv64BranchLessThan256Opcode::CLASS_OFFSET);
    let (is_ge, is_signed) = match opcode {
        BranchLessThanOpcode::BLT => (false, true),
        BranchLessThanOpcode::BLTU => (false, false),
        BranchLessThanOpcode::BGE => (true, true),
        BranchLessThanOpcode::BGEU => (true, false),
    };
    let x_sign = x[INT256_NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1) != 0 && is_signed;
    let y_sign = y[INT256_NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1) != 0 && is_signed;
    for (x, y) in x.iter().rev().zip(y.iter().rev()) {
        if x != y {
            return (x < y) ^ x_sign ^ y_sign ^ is_ge;
        }
    }
    is_ge
}

#[allow(clippy::type_complexity)]
fn set_and_execute_rand<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: usize,
    branch_fn: Option<fn(usize, &[u32; INT256_NUM_LIMBS], &[u32; INT256_NUM_LIMBS]) -> bool>,
) {
    let branch = branch_fn.is_some();

    let b = generate_long_number::<INT256_NUM_LIMBS, RV64_CELL_BITS>(rng);
    let c = generate_long_number::<INT256_NUM_LIMBS, RV64_CELL_BITS>(rng);
    if branch {
        let imm = rng.random_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
        let instruction = rv64_heap_branch_default(
            tester,
            vec![b.map(F::from_u32)],
            vec![c.map(F::from_u32)],
            imm as isize,
            opcode,
        );

        tester.execute_with_pc(
            executor,
            arena,
            &instruction,
            rng.random_range((ABS_MAX_BRANCH as u32)..(1 << (PC_BITS - 1))),
        );

        let cmp_result = branch_fn.unwrap()(opcode, &b, &c);
        let from_pc = tester.last_from_pc().as_canonical_u32() as i32;
        let to_pc = tester.last_to_pc().as_canonical_u32() as i32;
        assert_eq!(to_pc, from_pc + if cmp_result { imm } else { 4 });
    } else {
        let instruction = rv64_write_heap_default(
            tester,
            vec![b.map(F::from_u32)],
            vec![c.map(F::from_u32)],
            opcode,
        );
        tester.execute(executor, arena, &instruction);
    }
}

#[test_case(BaseAluOpcode::ADD, 24)]
#[test_case(BaseAluOpcode::SUB, 24)]
#[test_case(BaseAluOpcode::XOR, 24)]
#[test_case(BaseAluOpcode::OR, 24)]
#[test_case(BaseAluOpcode::AND, 24)]
fn run_alu_256_rand_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64BaseAlu256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_alu_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }
    let tester = tester
        .build()
        .load(harness)
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(LessThanOpcode::SLT, 24)]
#[test_case(LessThanOpcode::SLTU, 24)]
fn run_lt_256_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64LessThan256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_lt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }
    let tester = tester
        .build()
        .load(harness)
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(MulOpcode::MUL, 24)]
fn run_mul_256_rand_test(opcode: MulOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64Mul256Opcode::CLASS_OFFSET;

    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, RANGE_TUPLE_SIZES);
    let range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_mul_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_tuple_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }
    let tester = tester
        .build()
        .load(harness)
        .load_periphery((range_tuple_chip.air, range_tuple_chip))
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(ShiftOpcode::SLL, 24)]
#[test_case(ShiftOpcode::SRL, 24)]
#[test_case(ShiftOpcode::SRA, 24)]
fn run_shift_256_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64Shift256Opcode::CLASS_OFFSET;

    let range_checker_chip = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_shift_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_checker_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(BranchEqualOpcode::BEQ, 24)]
#[test_case(BranchEqualOpcode::BNE, 24)]
fn run_beq_256_rand_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64BranchEqual256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_beq_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode.local_usize() + offset,
            Some(beq_branch_fn),
        );
    }
    let tester = tester
        .build()
        .load(harness)
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(BranchLessThanOpcode::BLT, 24)]
#[test_case(BranchLessThanOpcode::BLTU, 24)]
#[test_case(BranchLessThanOpcode::BGE, 24)]
#[test_case(BranchLessThanOpcode::BGEU, 24)]
fn run_blt_256_rand_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64BranchLessThan256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_blt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode.local_usize() + offset,
            Some(blt_branch_fn),
        );
    }
    let tester = tester
        .build()
        .load(harness)
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(BaseAluOpcode::ADD, 24)]
#[test_case(BaseAluOpcode::SUB, 24)]
#[test_case(BaseAluOpcode::XOR, 24)]
#[test_case(BaseAluOpcode::OR, 24)]
#[test_case(BaseAluOpcode::AND, 24)]
fn run_alu_256_rand_test_cuda(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_alu_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = BaseAlu256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.local_usize() + Rv32BaseAlu256Opcode::CLASS_OFFSET,
            None,
        );
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
            EmptyAdapterCoreLayout::<F, AluAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test_case(LessThanOpcode::SLT, 24)]
#[test_case(LessThanOpcode::SLTU, 24)]
fn run_lt_256_rand_test_cuda(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_lt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = LessThan256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.local_usize() + Rv32LessThan256Opcode::CLASS_OFFSET,
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
            EmptyAdapterCoreLayout::<F, AluAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test_case(MulOpcode::MUL, 24)]
fn run_mul_256_rand_test_cuda(opcode: MulOpcode, num_ops: usize) {
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, RANGE_TUPLE_SIZES);
    let bitwise_bus = default_bitwise_lookup_bus();

    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(default_bitwise_lookup_bus())
        .with_range_tuple_checker(range_tuple_bus);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_tuple_chip = Arc::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, cpu_chip) = create_mul_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_tuple_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Multiplication256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.local_usize() + Rv32Mul256Opcode::CLASS_OFFSET,
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
            EmptyAdapterCoreLayout::<F, AluAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test_case(ShiftOpcode::SLL, 24)]
#[test_case(ShiftOpcode::SRL, 24)]
#[test_case(ShiftOpcode::SRA, 24)]
fn run_shift_256_rand_test_cuda(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let bitwise_bus = default_bitwise_lookup_bus();
    let range_bus = default_var_range_checker_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_shift_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Shift256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.local_usize() + Rv32Shift256Opcode::CLASS_OFFSET,
            None,
        );
    }

    type Record<'a> = (&'a mut Shift256AdapterRecord, &'a mut Shift256CoreRecord);

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, AluAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test_case(BranchEqualOpcode::BEQ, 24)]
#[test_case(BranchEqualOpcode::BNE, 24)]
fn run_beq_256_rand_test_cuda(opcode: BranchEqualOpcode, num_ops: usize) {
    let bitwise_bus = default_bitwise_lookup_bus();

    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default().with_bitwise_op_lookup(bitwise_bus);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_beq_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = BranchEqual256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.local_usize() + Rv32BranchEqual256Opcode::CLASS_OFFSET,
            Some(beq_branch_fn),
        );
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
            EmptyAdapterCoreLayout::<
                F,
                Rv32VecHeapBranchAdapterExecutor<NUM_READS, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
            >::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test_case(BranchLessThanOpcode::BLT, 24)]
#[test_case(BranchLessThanOpcode::BLTU, 24)]
#[test_case(BranchLessThanOpcode::BGE, 24)]
#[test_case(BranchLessThanOpcode::BGEU, 24)]
fn run_blt_256_rand_test_cuda(opcode: BranchLessThanOpcode, num_ops: usize) {
    let bitwise_bus = default_bitwise_lookup_bus();

    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default().with_bitwise_op_lookup(bitwise_bus);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_blt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = BranchLessThan256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
        tester.address_bits(),
    );
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY);

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.local_usize() + Rv32BranchLessThan256Opcode::CLASS_OFFSET,
            Some(blt_branch_fn),
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
            EmptyAdapterCoreLayout::<
                F,
                Rv32VecHeapBranchAdapterExecutor<NUM_READS, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
            >::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
