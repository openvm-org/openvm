use openvm_bigint_transpiler::{
    Rv32BaseAlu256Opcode, Rv32BranchEqual256Opcode, Rv32BranchLessThan256Opcode,
    Rv32LessThan256Opcode, Rv32Mul256Opcode, Rv32Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS, RANGE_TUPLE_CHECKER_BUS},
        InstructionExecutor, VmAirWrapper,
    },
    utils::generate_long_number,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
};
use openvm_instructions::{
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV32_CELL_BITS,
    LocalOpcode,
};
use openvm_rv32_adapters::{
    rv32_heap_branch_default, rv32_write_heap_default, Rv32HeapAdapterAir, Rv32HeapAdapterStep,
    Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterStep,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV_B_TYPE_IMM_BITS},
    BaseAluCoreAir, BranchEqualCoreAir, BranchLessThanCoreAir, LessThanCoreAir,
    MultiplicationCoreAir, ShiftCoreAir,
};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use crate::{
    Rv32BaseAlu256Step, Rv32BranchEqual256Step, Rv32BranchLessThan256Step, Rv32LessThan256Step,
    Rv32Multiplication256Step, Rv32Shift256Step,
};

use super::{
    Rv32BaseAlu256Chip, Rv32BranchEqual256Chip, Rv32BranchLessThan256Chip, Rv32LessThan256Chip,
    Rv32Multiplication256Chip, Rv32Shift256Chip,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_BRANCH: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);

#[allow(clippy::type_complexity)]
fn set_and_execute_rand<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: usize,
    branch_fn: Option<fn(usize, &[u32; INT256_NUM_LIMBS], &[u32; INT256_NUM_LIMBS]) -> bool>,
) {
    let branch = branch_fn.is_some();

    let b = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(rng);
    let c = generate_long_number::<INT256_NUM_LIMBS, RV32_CELL_BITS>(rng);
    if branch {
        let imm = rng.gen_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
        let instruction = rv32_heap_branch_default(
            tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            imm as isize,
            opcode,
        );

        tester.execute_with_pc(
            chip,
            &instruction,
            rng.gen_range((ABS_MAX_BRANCH as u32)..(1 << (PC_BITS - 1))),
        );

        let cmp_result = branch_fn.unwrap()(opcode, &b, &c);
        let from_pc = tester.execution.last_from_pc().as_canonical_u32() as i32;
        let to_pc = tester.execution.last_to_pc().as_canonical_u32() as i32;
        assert_eq!(to_pc, from_pc + if cmp_result { imm } else { 4 });
    } else {
        let instruction = rv32_write_heap_default(
            tester,
            vec![b.map(F::from_canonical_u32)],
            vec![c.map(F::from_canonical_u32)],
            opcode,
        );
        tester.execute(chip, &instruction);
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
    let offset = Rv32BaseAlu256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32BaseAlu256Chip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            BaseAluCoreAir::new(bitwise_bus, offset),
        ),
        Rv32BaseAlu256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
            bitwise_chip.clone(),
            offset,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut chip,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(LessThanOpcode::SLT, 24)]
#[test_case(LessThanOpcode::SLTU, 24)]
fn run_lt_256_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv32LessThan256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Rv32LessThan256Chip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            LessThanCoreAir::new(bitwise_bus, offset),
        ),
        Rv32LessThan256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
            bitwise_chip.clone(),
            offset,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut chip,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(MulOpcode::MUL, 24)]
fn run_mul_256_rand_test(opcode: MulOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv32Mul256Opcode::CLASS_OFFSET;

    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [
            1 << RV32_CELL_BITS,
            (INT256_NUM_LIMBS * (1 << RV32_CELL_BITS)) as u32,
        ],
    );
    let range_tuple_chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32Multiplication256Chip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            MultiplicationCoreAir::new(range_tuple_bus, offset),
        ),
        Rv32Multiplication256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
            range_tuple_chip.clone(),
            offset,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut chip,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }
    let tester = tester
        .build()
        .load(chip)
        .load(range_tuple_chip)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(ShiftOpcode::SLL, 24)]
#[test_case(ShiftOpcode::SRL, 24)]
#[test_case(ShiftOpcode::SRA, 24)]
fn run_shift_256_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv32Shift256Opcode::CLASS_OFFSET;

    let range_checker_chip = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32Shift256Chip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            ShiftCoreAir::new(bitwise_bus, range_checker_chip.bus(), offset),
        ),
        Rv32Shift256Step::new(
            Rv32HeapAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
            bitwise_chip.clone(),
            range_checker_chip.clone(),
            offset,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut chip,
            &mut rng,
            opcode.local_usize() + offset,
            None,
        );
    }

    drop(range_checker_chip);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(BranchEqualOpcode::BEQ, 24)]
#[test_case(BranchEqualOpcode::BNE, 24)]
fn run_beq_256_rand_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv32BranchEqual256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Rv32BranchEqual256Chip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapBranchAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            BranchEqualCoreAir::new(offset, DEFAULT_PC_STEP),
        ),
        Rv32BranchEqual256Step::new(
            Rv32HeapBranchAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
            offset,
            DEFAULT_PC_STEP,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    let branch_fn = |opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| {
        x.iter()
            .zip(y.iter())
            .fold(true, |acc, (x, y)| acc && (x == y))
            ^ (opcode
                == BranchEqualOpcode::BNE.local_usize() + Rv32BranchEqual256Opcode::CLASS_OFFSET)
    };

    for _ in 0..num_ops {
        set_and_execute_rand(
            &mut tester,
            &mut chip,
            &mut rng,
            opcode.local_usize() + offset,
            Some(branch_fn),
        );
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(BranchLessThanOpcode::BLT, 24)]
#[test_case(BranchLessThanOpcode::BLTU, 24)]
#[test_case(BranchLessThanOpcode::BGE, 24)]
#[test_case(BranchLessThanOpcode::BGEU, 24)]
fn run_blt_256_rand_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let offset = Rv32BranchLessThan256Opcode::CLASS_OFFSET;

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32BranchLessThan256Chip::<F>::new(
        VmAirWrapper::new(
            Rv32HeapBranchAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
                tester.address_bits(),
            ),
            BranchLessThanCoreAir::new(bitwise_bus, offset),
        ),
        Rv32BranchLessThan256Step::new(
            Rv32HeapBranchAdapterStep::new(tester.address_bits(), bitwise_chip.clone()),
            bitwise_chip.clone(),
            offset,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );
    let branch_fn =
        |opcode: usize, x: &[u32; INT256_NUM_LIMBS], y: &[u32; INT256_NUM_LIMBS]| -> bool {
            let opcode = BranchLessThanOpcode::from_usize(
                opcode - Rv32BranchLessThan256Opcode::CLASS_OFFSET,
            );
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
            &mut chip,
            &mut rng,
            opcode.local_usize() + offset,
            Some(branch_fn),
        );
    }
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}
