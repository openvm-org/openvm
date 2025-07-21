use openvm_circuit::{
    arch::{
        testing::{BITWISE_OP_LOOKUP_BUS, RANGE_TUPLE_CHECKER_BUS},
        DenseRecordArena, EmptyAdapterCoreLayout, InstructionExecutor, MatrixRecordArena,
        NewVmChipWrapper, VmAirWrapper,
    },
    utils::generate_long_number,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
};
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode, VmOpcode,
};
use openvm_rv32_adapters::{
    Rv32HeapAdapterAir, Rv32HeapAdapterStep, Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterStep,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS, RV_B_TYPE_IMM_BITS},
    BaseAluCoreAir, BranchEqualCoreAir, BranchLessThanCoreAir, LessThanCoreAir,
    MultiplicationCoreAir, ShiftCoreAir,
};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use super::*;
use crate::testing::GpuChipTestBuilder;

type F = BabyBear;

const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_BRANCH: isize = 1 << (RV_B_TYPE_IMM_BITS - 1);

#[allow(dead_code)]
fn beq_fn(_opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]) -> bool {
    x.iter().zip(y.iter()).all(|(a, b)| a == b)
}

#[allow(dead_code)]
fn bne_fn(_opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]) -> bool {
    !x.iter().zip(y.iter()).all(|(a, b)| a == b)
}

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
fn set_and_execute_rand<E: InstructionExecutor<F>>(
    tester: &mut GpuChipTestBuilder,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: usize,
    branch_fn: Option<fn(usize, &[u32; INT256_NUM_LIMBS], &[u32; INT256_NUM_LIMBS]) -> bool>,
) {
    let branch = branch_fn.is_some();

    let b = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(rng);
    let c = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(rng);

    let b_data = vec![b.map(F::from_canonical_u32)];
    let c_data = vec![c.map(F::from_canonical_u32)];

    if branch {
        let imm = rng.gen_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
        let instruction =
            rv32_heap_branch_default::<INT256_NUM_LIMBS>(tester, b_data, c_data, imm, opcode);

        tester.execute_with_pc(
            chip,
            &instruction,
            rng.gen_range((ABS_MAX_BRANCH as u32)..(1 << (PC_BITS - 1))),
        );
    } else {
        let instruction =
            rv32_write_heap_default::<INT256_NUM_LIMBS>(tester, b_data, c_data, opcode);

        tester.execute(chip, &instruction);
    }
}

use openvm_bigint_circuit::{
    Rv32BaseAlu256Step, Rv32BranchEqual256Step, Rv32BranchLessThan256Step, Rv32LessThan256Step,
    Rv32Multiplication256Step, Rv32Shift256Step,
};

type DenseChip<F> = NewVmChipWrapper<F, Rv32BaseAlu256Air, Rv32BaseAlu256Step, DenseRecordArena>;
type SparseChip<F> =
    NewVmChipWrapper<F, Rv32BaseAlu256Air, Rv32BaseAlu256Step, MatrixRecordArena<F>>;

type BranchEqual256DenseChip<F> =
    NewVmChipWrapper<F, Rv32BranchEqual256Air, Rv32BranchEqual256Step, DenseRecordArena>;
type BranchEqual256SparseChip<F> =
    NewVmChipWrapper<F, Rv32BranchEqual256Air, Rv32BranchEqual256Step, MatrixRecordArena<F>>;

type LessThan256DenseChip<F> =
    NewVmChipWrapper<F, Rv32LessThan256Air, Rv32LessThan256Step, DenseRecordArena>;
type LessThan256SparseChip<F> =
    NewVmChipWrapper<F, Rv32LessThan256Air, Rv32LessThan256Step, MatrixRecordArena<F>>;

type BranchLessThan256DenseChip<F> =
    NewVmChipWrapper<F, Rv32BranchLessThan256Air, Rv32BranchLessThan256Step, DenseRecordArena>;
type BranchLessThan256SparseChip<F> =
    NewVmChipWrapper<F, Rv32BranchLessThan256Air, Rv32BranchLessThan256Step, MatrixRecordArena<F>>;

type Shift256DenseChip<F> =
    NewVmChipWrapper<F, Rv32Shift256Air, Rv32Shift256Step, DenseRecordArena>;
type Shift256SparseChip<F> =
    NewVmChipWrapper<F, Rv32Shift256Air, Rv32Shift256Step, MatrixRecordArena<F>>;

type Multiplication256DenseChip<F> =
    NewVmChipWrapper<F, Rv32Multiplication256Air, Rv32Multiplication256Step, DenseRecordArena>;
type Multiplication256SparseChip<F> =
    NewVmChipWrapper<F, Rv32Multiplication256Air, Rv32Multiplication256Step, MatrixRecordArena<F>>;

fn create_alu_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> DenseChip<F> {
    let mut chip = DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            BaseAluCoreAir::new(bitwise.bus(), BaseAluOpcode::CLASS_OFFSET),
        ),
        Rv32BaseAlu256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            BaseAluOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_alu_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> SparseChip<F> {
    let mut chip = SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            BaseAluCoreAir::new(bitwise.bus(), BaseAluOpcode::CLASS_OFFSET),
        ),
        Rv32BaseAlu256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            BaseAluOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_branch_equal_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> BranchEqual256DenseChip<F> {
    let mut chip = BranchEqual256DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapBranchAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        ),
        Rv32BranchEqual256Step::new(
            Rv32HeapBranchAdapterStep::new(tester.address_bits(), bitwise.clone()),
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_branch_equal_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> BranchEqual256SparseChip<F> {
    let mut chip = BranchEqual256SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapBranchAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        ),
        Rv32BranchEqual256Step::new(
            Rv32HeapBranchAdapterStep::new(tester.address_bits(), bitwise.clone()),
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_less_than_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> LessThan256DenseChip<F> {
    let mut chip = LessThan256DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            LessThanCoreAir::new(bitwise.bus(), LessThanOpcode::CLASS_OFFSET),
        ),
        Rv32LessThan256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            LessThanOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_less_than_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> LessThan256SparseChip<F> {
    let mut chip = LessThan256SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            LessThanCoreAir::new(bitwise.bus(), LessThanOpcode::CLASS_OFFSET),
        ),
        Rv32LessThan256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            LessThanOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_branch_less_than_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> BranchLessThan256DenseChip<F> {
    let mut chip = BranchLessThan256DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapBranchAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            BranchLessThanCoreAir::new(bitwise.bus(), BranchLessThanOpcode::CLASS_OFFSET),
        ),
        Rv32BranchLessThan256Step::new(
            Rv32HeapBranchAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            BranchLessThanOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_branch_less_than_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> BranchLessThan256SparseChip<F> {
    let mut chip = BranchLessThan256SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapBranchAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            BranchLessThanCoreAir::new(bitwise.bus(), BranchLessThanOpcode::CLASS_OFFSET),
        ),
        Rv32BranchLessThan256Step::new(
            Rv32HeapBranchAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            BranchLessThanOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_shift256_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> Shift256DenseChip<F> {
    let mut chip = Shift256DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            ShiftCoreAir::new(
                bitwise.bus(),
                tester.cpu_range_checker().bus(),
                ShiftOpcode::CLASS_OFFSET,
            ),
        ),
        Rv32Shift256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            tester.cpu_range_checker().clone(),
            ShiftOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_shift256_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> Shift256SparseChip<F> {
    let mut chip = Shift256SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            ShiftCoreAir::new(
                bitwise.bus(),
                tester.cpu_range_checker().bus(),
                ShiftOpcode::CLASS_OFFSET,
            ),
        ),
        Rv32Shift256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            bitwise.clone(),
            tester.cpu_range_checker().clone(),
            ShiftOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_multiplication_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    range_tuple: SharedRangeTupleCheckerChip<2>,
) -> Multiplication256DenseChip<F> {
    let mut chip = Multiplication256DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            MultiplicationCoreAir::new(*range_tuple.bus(), MulOpcode::CLASS_OFFSET),
        ),
        Rv32Multiplication256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            range_tuple.clone(),
            MulOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_multiplication_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    range_tuple: SharedRangeTupleCheckerChip<2>,
) -> Multiplication256SparseChip<F> {
    let mut chip = Multiplication256SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise.bus(),
                tester.address_bits(),
            ),
            MultiplicationCoreAir::new(*range_tuple.bus(), MulOpcode::CLASS_OFFSET),
        ),
        Rv32Multiplication256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise.clone()),
            range_tuple.clone(),
            MulOpcode::CLASS_OFFSET,
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

#[test_case(BaseAluOpcode::ADD, 24; "BaseAluOpcode::ADD")]
#[test_case(BaseAluOpcode::SUB, 24; "BaseAluOpcode::SUB")]
#[test_case(BaseAluOpcode::XOR, 24; "BaseAluOpcode::XOR")]
#[test_case(BaseAluOpcode::OR, 24; "BaseAluOpcode::OR")]
#[test_case(BaseAluOpcode::AND, 24; "BaseAluOpcode::AND")]
fn run_alu_256_rand_test(opcode: BaseAluOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(bitwise_bus);
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut dense_chip = create_alu_dense_chip(&tester, cpu_bitwise_chip.clone());

    let mut gpu_chip = BaseAlu256ChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        None,
    );
    let mut cpu_chip = create_alu_sparse_chip(&tester, cpu_bitwise_chip.clone());

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode.local_usize() + BaseAluOpcode::CLASS_OFFSET,
            None,
        );
    }

    type Record<'a> = (
        &'a mut BaseAlu256AdapterRecord,
        &'a mut BaseAlu256CoreRecord,
    );
    dense_chip
        .arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}

#[test_case(BranchEqualOpcode::BEQ, 24; "BranchEqualOpcode::BEQ")]
#[test_case(BranchEqualOpcode::BNE, 24; "BranchEqualOpcode::BNE")]
fn run_beq_256_rand_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(bitwise_bus);
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut dense_chip = create_branch_equal_dense_chip(&tester, cpu_bitwise_chip.clone());

    let mut gpu_chip = BranchEqual256ChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        None,
    );
    let mut cpu_chip = create_branch_equal_sparse_chip(&tester, cpu_bitwise_chip.clone());

    let branch_fn = |op_code: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| {
        x.iter()
            .zip(y.iter())
            .fold(true, |acc, (x, y)| acc && (x == y))
            ^ (op_code == BranchEqualOpcode::BNE.local_usize() + BranchEqualOpcode::CLASS_OFFSET)
    };

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode.local_usize() + BranchEqualOpcode::CLASS_OFFSET,
            Some(branch_fn),
        );
    }

    type BranchRecord<'a> = (
        &'a mut BranchEqual256AdapterRecord,
        &'a mut BranchEqual256CoreRecord,
    );
    dense_chip
        .arena
        .get_record_seeker::<BranchRecord, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>>::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}

#[test_case(LessThanOpcode::SLT, 24)]
#[test_case(LessThanOpcode::SLTU, 24)]
fn run_lt_256_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(bitwise_bus);
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut dense_chip = create_less_than_dense_chip(&tester, cpu_bitwise_chip.clone());

    let mut gpu_chip = LessThan256ChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        None,
    );
    let mut cpu_chip = create_less_than_sparse_chip(&tester, cpu_bitwise_chip.clone());

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode.local_usize() + LessThanOpcode::CLASS_OFFSET,
            None,
        );
    }

    type LessThanRecord<'a> = (
        &'a mut LessThan256AdapterRecord,
        &'a mut LessThan256CoreRecord,
    );
    dense_chip
        .arena
        .get_record_seeker::<LessThanRecord, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}

#[test_case(BranchLessThanOpcode::BLT, 24)]
#[test_case(BranchLessThanOpcode::BLTU, 24)]
#[test_case(BranchLessThanOpcode::BGE, 24)]
#[test_case(BranchLessThanOpcode::BGEU, 24)]
fn run_blt_256_rand_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(bitwise_bus);
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut dense_chip = create_branch_less_than_dense_chip(&tester, cpu_bitwise_chip.clone());

    let mut gpu_chip = BranchLessThan256ChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        None,
    );
    let mut cpu_chip = create_branch_less_than_sparse_chip(&tester, cpu_bitwise_chip.clone());

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

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode.local_usize() + BranchLessThanOpcode::CLASS_OFFSET,
            Some(branch_fn),
        );
    }

    type BranchLessThanRecord<'a> = (
        &'a mut BranchLessThan256AdapterRecord,
        &'a mut BranchLessThan256CoreRecord,
    );
    dense_chip
        .arena
        .get_record_seeker::<BranchLessThanRecord, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>>::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}

#[test_case(ShiftOpcode::SLL, 24)]
#[test_case(ShiftOpcode::SRL, 24)]
#[test_case(ShiftOpcode::SRA, 24)]
fn run_shift_256_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(bitwise_bus);
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut dense_chip = create_shift256_dense_chip(&tester, cpu_bitwise_chip.clone());

    let mut gpu_chip = Shift256ChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        None,
    );
    let mut cpu_chip = create_shift256_sparse_chip(&tester, cpu_bitwise_chip.clone());

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode.local_usize() + ShiftOpcode::CLASS_OFFSET,
            None,
        );
    }

    type Shift256Record<'a> = (&'a mut Shift256AdapterRecord, &'a mut Shift256CoreRecord);
    dense_chip
        .arena
        .get_record_seeker::<Shift256Record, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<F, Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>>::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
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
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(bitwise_bus)
        .with_range_tuple_checker(range_tuple_bus);
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let cpu_range_tuple_chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);

    let mut dense_chip = create_multiplication_dense_chip(
        &tester,
        cpu_bitwise_chip.clone(),
        cpu_range_tuple_chip.clone(),
    );

    let mut gpu_chip = Multiplication256ChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        None,
    );
    let mut cpu_chip = create_multiplication_sparse_chip(
        &tester,
        cpu_bitwise_chip.clone(),
        cpu_range_tuple_chip.clone(),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            opcode.local_usize() + MulOpcode::CLASS_OFFSET,
            None,
        );
    }

    type Multiplication256Record<'a> = (
        &'a mut Multiplication256AdapterRecord,
        &'a mut Multiplication256CoreRecord,
    );
    dense_chip
        .arena
        .get_record_seeker::<Multiplication256Record, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<
                F,
                Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
            >::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}
