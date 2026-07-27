use std::sync::Arc;

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        testing::{
            TestBuilder, TestChipHarness, TestPreflight, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
            RANGE_TUPLE_CHECKER_BUS,
        },
        ExecutionBridge, Executor, MemoryConfig, Postflight, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::generate_long_number,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip},
    var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::{
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV64_BYTE_BITS,
    LocalOpcode,
};
use openvm_riscv_adapters::{
    rv64_heap_branch_default, rv64_write_heap_default, Rv64VecHeapAdapterAir,
    Rv64VecHeapBranchU16AdapterAir, Rv64VecHeapU16AdapterAir,
};
use openvm_riscv_circuit::{
    adapters::RV_B_TYPE_IMM_BITS, AddSubCoreAir, AddSubFiller, BitwiseLogicCoreAir,
    BitwiseLogicFiller, BranchEqualCoreAir, BranchEqualFiller, BranchLessThanCoreAir,
    BranchLessThanFiller, LessThanCoreAir, LessThanFiller, MultiplicationCoreAir,
    MultiplicationFiller, ShiftLogicalCoreAir, ShiftLogicalFiller, ShiftRightArithmeticCoreAir,
    ShiftRightArithmeticFiller,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::{
        AddSub256ChipGpu, BitwiseLogic256ChipGpu, BranchEqual256ChipGpu, BranchLessThan256ChipGpu,
        LessThan256ChipGpu, Multiplication256ChipGpu, ShiftLogical256ChipGpu,
        ShiftRightArithmetic256ChipGpu,
    },
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use crate::{
    trace::{
        generate_add_sub_trace, generate_bitwise_trace, generate_branch_equal_trace,
        generate_branch_less_than_trace, generate_less_than_trace, generate_multiplication_trace,
        generate_shift_arithmetic_trace, generate_shift_logical_trace,
    },
    AluAdapterAir, AluU16AdapterAir, BranchAdapterAir, Rv64AddSub256Air, Rv64AddSub256Chip,
    Rv64AddSub256Executor, Rv64BitwiseLogic256Air, Rv64BitwiseLogic256Chip,
    Rv64BitwiseLogic256Executor, Rv64BranchEqual256Air, Rv64BranchEqual256Chip,
    Rv64BranchEqual256Executor, Rv64BranchLessThan256Air, Rv64BranchLessThan256Chip,
    Rv64BranchLessThan256Executor, Rv64LessThan256Air, Rv64LessThan256Chip,
    Rv64LessThan256Executor, Rv64Multiplication256Air, Rv64Multiplication256Chip,
    Rv64Multiplication256Executor, Rv64ShiftLogical256Air, Rv64ShiftLogical256Chip,
    Rv64ShiftLogical256Executor, Rv64ShiftRightArithmetic256Air, Rv64ShiftRightArithmetic256Chip,
    Rv64ShiftRightArithmetic256Executor, INT256_NUM_U8_LIMBS,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_BRANCH: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
const RANGE_TUPLE_SIZES: [u32; 2] = [
    1 << RV64_BYTE_BITS,
    (INT256_NUM_U8_LIMBS * (1 << RV64_BYTE_BITS)) as u32,
];

fn create_add_sub_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64AddSub256Air,
    Rv64AddSub256Executor,
    Rv64AddSub256Chip<F>,
) {
    let air = Rv64AddSub256Air::new(
        AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        AddSubCoreAir::new(range_checker_chip.bus(), Rv64BaseAlu256Opcode::CLASS_OFFSET),
    );
    let executor = Rv64AddSub256Executor;
    let chip = Rv64AddSub256Chip::new(AddSubFiller::new(range_checker_chip), memory_helper);
    (air, executor, chip)
}

fn create_bitwise_logic_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64BitwiseLogic256Air,
    Rv64BitwiseLogic256Executor,
    Rv64BitwiseLogic256Chip<F>,
) {
    let air = Rv64BitwiseLogic256Air::new(
        AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        BitwiseLogicCoreAir::new(bitwise_chip.bus(), Rv64BaseAlu256Opcode::CLASS_OFFSET),
    );
    let executor = Rv64BitwiseLogic256Executor;
    let chip = Rv64BitwiseLogic256Chip::new(BitwiseLogicFiller::new(bitwise_chip), memory_helper);
    (air, executor, chip)
}

#[allow(clippy::too_many_arguments)]
fn create_lt_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64LessThan256Air,
    Rv64LessThan256Executor,
    Rv64LessThan256Chip<F>,
) {
    let air = Rv64LessThan256Air::new(
        AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        LessThanCoreAir::new(
            range_checker_chip.bus(),
            Rv64LessThan256Opcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64LessThan256Executor;
    let chip = Rv64LessThan256Chip::new(LessThanFiller::new(range_checker_chip), memory_helper);
    (air, executor, chip)
}

fn create_mul_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    range_checker_chip: SharedVariableRangeCheckerChip,
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
            range_checker_chip.bus(),
            address_bits,
        )),
        MultiplicationCoreAir::new(
            *range_tuple_chip.bus(),
            bitwise_chip.bus(),
            Rv64Mul256Opcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64Multiplication256Executor;
    let chip = Rv64Multiplication256Chip::<F>::new(
        MultiplicationFiller::new(
            range_tuple_chip,
            bitwise_chip,
            Rv64Mul256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_shift_logical_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64ShiftLogical256Air,
    Rv64ShiftLogical256Executor,
    Rv64ShiftLogical256Chip<F>,
) {
    let air = Rv64ShiftLogical256Air::new(
        AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        ShiftLogicalCoreAir::new(range_checker_chip.bus(), Rv64Shift256Opcode::CLASS_OFFSET),
    );
    let executor = Rv64ShiftLogical256Executor;
    let chip =
        Rv64ShiftLogical256Chip::new(ShiftLogicalFiller::new(range_checker_chip), memory_helper);
    (air, executor, chip)
}

fn create_shift_right_arithmetic_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64ShiftRightArithmetic256Air,
    Rv64ShiftRightArithmetic256Executor,
    Rv64ShiftRightArithmetic256Chip<F>,
) {
    let air = Rv64ShiftRightArithmetic256Air::new(
        AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        ShiftRightArithmeticCoreAir::new(
            range_checker_chip.bus(),
            Rv64Shift256Opcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64ShiftRightArithmetic256Executor;
    let chip = Rv64ShiftRightArithmetic256Chip::new(
        ShiftRightArithmeticFiller::new(range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

#[allow(clippy::too_many_arguments)]
fn create_beq_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64BranchEqual256Air,
    Rv64BranchEqual256Executor,
    Rv64BranchEqual256Chip<F>,
) {
    let air = Rv64BranchEqual256Air::new(
        BranchAdapterAir::new(Rv64VecHeapBranchU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        BranchEqualCoreAir::new(Rv64BranchEqual256Opcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = Rv64BranchEqual256Executor;
    let chip = Rv64BranchEqual256Chip::new(
        BranchEqualFiller::new(Rv64BranchEqual256Opcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        memory_helper,
    );
    (air, executor, chip)
}

#[allow(clippy::too_many_arguments)]
fn create_blt_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64BranchLessThan256Air,
    Rv64BranchLessThan256Executor,
    Rv64BranchLessThan256Chip<F>,
) {
    let air = Rv64BranchLessThan256Air::new(
        BranchAdapterAir::new(Rv64VecHeapBranchU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
            address_bits,
        )),
        BranchLessThanCoreAir::new(
            range_checker_chip.bus(),
            Rv64BranchLessThan256Opcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64BranchLessThan256Executor;
    let chip = Rv64BranchLessThan256Chip::new(
        BranchLessThanFiller::new(
            range_checker_chip,
            Rv64BranchLessThan256Opcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn beq_branch_fn(
    opcode: usize,
    x: &[u32; INT256_NUM_U8_LIMBS],
    y: &[u32; INT256_NUM_U8_LIMBS],
) -> bool {
    x.iter()
        .zip(y.iter())
        .fold(true, |acc, (x, y)| acc && (x == y))
        ^ (opcode == BranchEqualOpcode::BNE.local_usize() + Rv64BranchEqual256Opcode::CLASS_OFFSET)
}

fn blt_branch_fn(
    opcode: usize,
    x: &[u32; INT256_NUM_U8_LIMBS],
    y: &[u32; INT256_NUM_U8_LIMBS],
) -> bool {
    let opcode =
        BranchLessThanOpcode::from_usize(opcode - Rv64BranchLessThan256Opcode::CLASS_OFFSET);
    let (is_ge, is_signed) = match opcode {
        BranchLessThanOpcode::BLT => (false, true),
        BranchLessThanOpcode::BLTU => (false, false),
        BranchLessThanOpcode::BGE => (true, true),
        BranchLessThanOpcode::BGEU => (true, false),
    };
    let x_sign = x[INT256_NUM_U8_LIMBS - 1] >> (RV64_BYTE_BITS - 1) != 0 && is_signed;
    let y_sign = y[INT256_NUM_U8_LIMBS - 1] >> (RV64_BYTE_BITS - 1) != 0 && is_signed;
    for (x, y) in x.iter().rev().zip(y.iter().rev()) {
        if x != y {
            return (x < y) ^ x_sign ^ y_sign ^ is_ge;
        }
    }
    is_ge
}

#[allow(clippy::type_complexity)]
fn set_and_execute_rand<E: Executor<F> + Clone>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut TestPreflight<F>,
    rng: &mut StdRng,
    opcode: usize,
    branch_fn: Option<fn(usize, &[u32; INT256_NUM_U8_LIMBS], &[u32; INT256_NUM_U8_LIMBS]) -> bool>,
) {
    let branch = branch_fn.is_some();

    let b = generate_long_number::<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>(rng);
    let c = generate_long_number::<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>(rng);
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
            preflight,
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
        tester.execute(executor, preflight, &instruction);
    }
}

#[test_case(BaseAluOpcode::ADD, 24)]
#[test_case(BaseAluOpcode::SUB, 24)]
fn run_add_sub_256_rand_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64BaseAlu256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_add_sub_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let address_bits = tester.address_bits();
    let mut harness = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| generate_add_sub_trace(chip, postflight, address_bits),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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

#[test]
fn add_sub_postflight_rejects_truncated_history() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let address_bits = tester.address_bits();
    let (air, executor, chip) = create_add_sub_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        address_bits,
    );
    let mut harness: TestChipHarness<F, _, _, _> = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| generate_add_sub_trace(chip, postflight, address_bits),
    );
    let b = generate_long_number::<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>(&mut rng);
    let c = generate_long_number::<INT256_NUM_U8_LIMBS, RV64_BYTE_BITS>(&mut rng);
    let instruction = rv64_write_heap_default(
        &mut tester,
        vec![b.map(F::from_u32)],
        vec![c.map(F::from_u32)],
        Rv64BaseAlu256Opcode(BaseAluOpcode::ADD)
            .global_opcode()
            .as_usize(),
    );
    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.preflight,
        &instruction,
        0,
    );
    let history = &mut harness.preflight.executions[0].history;
    let program = openvm_instructions::program::Program::new_without_debug_infos(
        &[instruction.clone(), instruction],
        0,
    );
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::new(&program, history, &memory_config, None).unwrap();
    let actual = generate_add_sub_trace(&harness.chip, &postflight, address_bits).unwrap();
    assert!(!actual.values.is_empty());

    history.memory.accesses[0].pointer += BLOCK_FE_WIDTH as u32;
    let malformed = Postflight::new(&program, history, &memory_config, None).unwrap();
    assert!(generate_add_sub_trace(&harness.chip, &malformed, address_bits).is_err());
}

#[test_case(BaseAluOpcode::XOR, 24)]
#[test_case(BaseAluOpcode::OR, 24)]
#[test_case(BaseAluOpcode::AND, 24)]
fn run_bitwise_logic_256_rand_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64BaseAlu256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_bitwise_logic_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let address_bits = tester.address_bits();
    let range_checker = tester.range_checker();
    let mut harness = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| {
            generate_bitwise_trace(chip, postflight, address_bits, &range_checker)
        },
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_lt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let address_bits = tester.address_bits();
    let mut harness = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| generate_less_than_trace(chip, postflight, address_bits),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_mul_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_tuple_chip.clone(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let address_bits = tester.address_bits();
    let range_checker = tester.range_checker();
    let mut harness = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| {
            generate_multiplication_trace(chip, postflight, address_bits, &range_checker)
        },
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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

    // SLL/SRL and SRA use separate u16 core types, so each needs its own harness.
    if opcode == ShiftOpcode::SRA {
        let (air, executor, chip) = create_shift_right_arithmetic_harness_fields(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker_chip.clone(),
            tester.memory_helper(),
            tester.address_bits(),
        );
        let address_bits = tester.address_bits();
        let mut harness = TestChipHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| generate_shift_arithmetic_trace(chip, postflight, address_bits),
        );

        for _ in 0..num_ops {
            set_and_execute_rand(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                opcode.local_usize() + offset,
                None,
            );
        }

        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    } else {
        let (air, executor, chip) = create_shift_logical_harness_fields(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker_chip.clone(),
            tester.memory_helper(),
            tester.address_bits(),
        );
        let address_bits = tester.address_bits();
        let mut harness = TestChipHarness::with_capacity(
            executor,
            air,
            chip,
            MAX_INS_CAPACITY,
            move |chip, postflight| generate_shift_logical_trace(chip, postflight, address_bits),
        );

        for _ in 0..num_ops {
            set_and_execute_rand(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                opcode.local_usize() + offset,
                None,
            );
        }

        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    }
}

#[test_case(BranchEqualOpcode::BEQ, 24)]
#[test_case(BranchEqualOpcode::BNE, 24)]
fn run_beq_256_rand_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv64BranchEqual256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_beq_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let address_bits = tester.address_bits();
    let range_checker = tester.range_checker();
    let mut harness = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| {
            generate_branch_equal_trace(chip, postflight, address_bits, &range_checker)
        },
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_blt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let address_bits = tester.address_bits();
    let mut harness = TestChipHarness::with_capacity(
        executor,
        air,
        chip,
        MAX_INS_CAPACITY,
        move |chip, postflight| generate_branch_less_than_trace(chip, postflight, address_bits),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
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

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(BaseAluOpcode::ADD, 24)]
#[test_case(BaseAluOpcode::SUB, 24)]
fn run_add_sub_256_rand_test_cuda(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_add_sub_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = AddSub256ChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let address_bits = tester.address_bits();
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| generate_add_sub_trace(chip, postflight, address_bits),
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode.local_usize() + Rv64BaseAlu256Opcode::CLASS_OFFSET,
            None,
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(BaseAluOpcode::XOR, 24)]
#[test_case(BaseAluOpcode::OR, 24)]
#[test_case(BaseAluOpcode::AND, 24)]
fn run_bitwise_logic_256_rand_test_cuda(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_bitwise_logic_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker_chip.clone(),
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = BitwiseLogic256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let address_bits = tester.address_bits();
    let range_checker = dummy_range_checker_chip;
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_bitwise_trace(chip, postflight, address_bits, &range_checker)
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode.local_usize() + Rv64BaseAlu256Opcode::CLASS_OFFSET,
            None,
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(LessThanOpcode::SLT, 24)]
#[test_case(LessThanOpcode::SLTU, 24)]
fn run_lt_256_rand_test_cuda(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_lt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = LessThan256ChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let address_bits = tester.address_bits();
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| generate_less_than_trace(chip, postflight, address_bits),
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode.local_usize() + Rv64LessThan256Opcode::CLASS_OFFSET,
            None,
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(MulOpcode::MUL, 24)]
fn run_mul_256_rand_test_cuda(opcode: MulOpcode, num_ops: usize) {
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, RANGE_TUPLE_SIZES);
    let bitwise_bus = default_bitwise_lookup_bus();

    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(default_bitwise_lookup_bus())
        .with_range_tuple_checker(range_tuple_bus);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_tuple_chip = Arc::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_mul_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_tuple_chip,
        dummy_range_checker_chip.clone(),
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Multiplication256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let address_bits = tester.address_bits();
    let range_checker = dummy_range_checker_chip;
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_multiplication_trace(chip, postflight, address_bits, &range_checker)
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode.local_usize() + Rv64Mul256Opcode::CLASS_OFFSET,
            None,
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(ShiftOpcode::SLL, 24)]
#[test_case(ShiftOpcode::SRL, 24)]
#[test_case(ShiftOpcode::SRA, 24)]
fn run_shift_256_rand_test_cuda(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

    let range_bus = default_var_range_checker_bus();
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

    if opcode == ShiftOpcode::SRA {
        let (air, executor, cpu_chip) = create_shift_right_arithmetic_harness_fields(
            tester.memory_bridge(),
            tester.execution_bridge(),
            dummy_range_checker_chip,
            tester.dummy_memory_helper(),
            tester.address_bits(),
        );
        let gpu_chip = ShiftRightArithmetic256ChipGpu::new(
            tester.range_checker(),
            tester.address_bits(),
            tester.timestamp_max_bits(),
        );
        let address_bits = tester.address_bits();
        let mut harness =
            GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
                .with_trace_generators(
                    move |chip, postflight| {
                        generate_shift_arithmetic_trace(chip, postflight, address_bits)
                    },
                    |chip, program, transcript, plan| {
                        chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                    },
                );

        for _ in 0..num_ops {
            set_and_execute_rand(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                opcode.local_usize() + Rv64Shift256Opcode::CLASS_OFFSET,
                None,
            );
        }

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    } else {
        let (air, executor, cpu_chip) = create_shift_logical_harness_fields(
            tester.memory_bridge(),
            tester.execution_bridge(),
            dummy_range_checker_chip,
            tester.dummy_memory_helper(),
            tester.address_bits(),
        );
        let gpu_chip = ShiftLogical256ChipGpu::new(
            tester.range_checker(),
            tester.address_bits(),
            tester.timestamp_max_bits(),
        );
        let address_bits = tester.address_bits();
        let mut harness =
            GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
                .with_trace_generators(
                    move |chip, postflight| {
                        generate_shift_logical_trace(chip, postflight, address_bits)
                    },
                    |chip, program, transcript, plan| {
                        chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                    },
                );

        for _ in 0..num_ops {
            set_and_execute_rand(
                &mut tester,
                &mut harness.executor,
                &mut harness.preflight,
                &mut rng,
                opcode.local_usize() + Rv64Shift256Opcode::CLASS_OFFSET,
                None,
            );
        }

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(BranchEqualOpcode::BEQ, 24)]
#[test_case(BranchEqualOpcode::BNE, 24)]
fn run_beq_256_rand_test_cuda(opcode: BranchEqualOpcode, num_ops: usize) {
    let bitwise_bus = default_bitwise_lookup_bus();

    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default().with_bitwise_op_lookup(bitwise_bus);

    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_beq_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip.clone(),
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = BranchEqual256ChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let address_bits = tester.address_bits();
    let range_checker = dummy_range_checker_chip;
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_branch_equal_trace(chip, postflight, address_bits, &range_checker)
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode.local_usize() + Rv64BranchEqual256Opcode::CLASS_OFFSET,
            Some(beq_branch_fn),
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test_case(BranchLessThanOpcode::BLT, 24)]
#[test_case(BranchLessThanOpcode::BLTU, 24)]
#[test_case(BranchLessThanOpcode::BGE, 24)]
#[test_case(BranchLessThanOpcode::BGEU, 24)]
fn run_blt_256_rand_test_cuda(opcode: BranchLessThanOpcode, num_ops: usize) {
    let bitwise_bus = default_bitwise_lookup_bus();

    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default().with_bitwise_op_lookup(bitwise_bus);

    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_blt_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = BranchLessThan256ChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    let address_bits = tester.address_bits();
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
            .with_trace_generators(
                move |chip, postflight| {
                    generate_branch_less_than_trace(chip, postflight, address_bits)
                },
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            opcode.local_usize() + Rv64BranchLessThan256Opcode::CLASS_OFFSET,
            Some(blt_branch_fn),
        );
    }

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
