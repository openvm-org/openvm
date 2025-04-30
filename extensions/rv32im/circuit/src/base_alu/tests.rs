use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{
        testing::{
            test_adapter::TestAdapterAir, TestAdapterChip, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
        },
        AdapterExecutorE1, AdapterTraceStep, NewVmChipWrapper, VmAirWrapper, VmChipWrapper,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
    utils::generate_long_number,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
    ChipUsageGetter,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use super::{core::run_alu, BaseAluCoreAir, BaseAluStep, Rv32BaseAluChip, Rv32BaseAluStep};
use crate::{
    adapters::{
        tracing_read, tracing_read_imm, Rv32BaseAluAdapterAir, Rv32BaseAluAdapterCols,
        Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    base_alu::BaseAluCoreCols,
    test_utils::{generate_rv32_is_type_immediate, rv32_rand_write_register_or_imm},
};

const MAX_INS_CAPACITY: usize = 128;

type F = BabyBear;

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    Rv32BaseAluChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let chip = Rv32BaseAluChip::new(
        VmAirWrapper::new(
            Rv32BaseAluAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
            ),
            BaseAluCoreAir::new(bitwise_bus, BaseAluOpcode::CLASS_OFFSET),
        ),
        Rv32BaseAluStep::new(
            Rv32BaseAluAdapterStep::new(),
            bitwise_chip.clone(),
            BaseAluOpcode::CLASS_OFFSET,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    (chip, bitwise_chip)
}
//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

fn run_rv32_alu_rand_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&tester);

    for _ in 0..num_ops {
        let b = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng)
            .map(|x| x as u8);
        let (c_imm, c) = if rng.gen_bool(0.5) {
            (
                None,
                generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng)
                    .map(|x| x as u8),
            )
        } else {
            let (imm, c) = generate_rv32_is_type_immediate(&mut rng);
            (Some(imm), c)
        };

        let (instruction, rd) = rv32_rand_write_register_or_imm(
            &mut tester,
            b,
            c,
            c_imm,
            opcode.global_opcode().as_usize(),
            &mut rng,
        );
        tester.execute(&mut chip, &instruction);

        let a = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c)
            .map(F::from_canonical_u8);
        assert_eq!(a, tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd))
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_alu_add_rand_test() {
    run_rv32_alu_rand_test(BaseAluOpcode::ADD, 100);
}

#[test]
fn rv32_alu_sub_rand_test() {
    run_rv32_alu_rand_test(BaseAluOpcode::SUB, 100);
}

#[test]
fn rv32_alu_xor_rand_test() {
    run_rv32_alu_rand_test(BaseAluOpcode::XOR, 100);
}

#[test]
fn rv32_alu_or_rand_test() {
    run_rv32_alu_rand_test(BaseAluOpcode::OR, 100);
}

#[test]
fn rv32_alu_and_rand_test() {
    run_rv32_alu_rand_test(BaseAluOpcode::AND, 100);
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// the write part of the trace and check that the core chip throws the expected error.
// A dummy adapter is used so memory interactions don't indirectly cause false passes.
//////////////////////////////////////////////////////////////////////////////////////

// type Rv32BaseAluTestChip<F> = NewVmChipWrapper<
//     F,
//     VmAirWrapper<TestAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>,
//     BaseAluStep<TestAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
// >;

// // TODO: FIX NEGATIVE TESTS

// #[allow(clippy::too_many_arguments)]
// fn run_rv32_alu_negative_test(
//     opcode: BaseAluOpcode,
//     a: [u32; RV32_REGISTER_NUM_LIMBS],
//     b: [u32; RV32_REGISTER_NUM_LIMBS],
//     c: [u32; RV32_REGISTER_NUM_LIMBS],
//     interaction_error: bool,
// ) {
//     let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
//     let (mut chip, bitwise_chip) = create_test_chip(&tester);
//     let mut chip = Rv32BaseAluTestChip::<F>::new(
//         TestAdapterChip::new(
//             vec![[b.map(F::from_canonical_u32), c.map(F::from_canonical_u32)].concat()],
//             vec![None],
//             ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
//         ),
//         BaseAluStep::new(bitwise_chip.clone(), BaseAluOpcode::CLASS_OFFSET),
//         tester.offline_memory_mutex_arc(),
//     );

//     tester.execute(
//         &mut chip,
//         &Instruction::from_usize(opcode.global_opcode(), [0, 0, 0, 1, 1]),
//     );

//     let trace_width = chip.trace_width();
//     let adapter_width = Rv32BaseAluAdapterCols::<F>::width();

//     if (opcode == BaseAluOpcode::ADD || opcode == BaseAluOpcode::SUB)
//         && a.iter().all(|&a_val| a_val < (1 << RV32_CELL_BITS))
//     {
//         bitwise_chip.clear();
//         for a_val in a {
//             bitwise_chip.request_xor(a_val, a_val);
//         }
//     }

//     let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
//         let mut values = trace.row_slice(0).to_vec();
//         let cols: &mut BaseAluCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
//             values.split_at_mut(adapter_width).1.borrow_mut();
//         cols.a = a.map(F::from_canonical_u32);
//         *trace = RowMajorMatrix::new(values, trace_width);
//     };

//     disable_debug_builder();
//     let tester = tester
//         .build()
//         .load_and_prank_trace(chip, modify_trace)
//         .load(bitwise_chip)
//         .finalize();
//     tester.simple_test_with_expected_error(if interaction_error {
//         VerificationError::ChallengePhaseError
//     } else {
//         VerificationError::OodEvaluationMismatch
//     });
// }

// #[test]
// fn rv32_alu_add_wrong_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::ADD,
//         [246, 0, 0, 0],
//         [250, 0, 0, 0],
//         [250, 0, 0, 0],
//         false,
//     );
// }

// #[test]
// fn rv32_alu_add_out_of_range_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::ADD,
//         [500, 0, 0, 0],
//         [250, 0, 0, 0],
//         [250, 0, 0, 0],
//         true,
//     );
// }

// #[test]
// fn rv32_alu_sub_wrong_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::SUB,
//         [255, 0, 0, 0],
//         [1, 0, 0, 0],
//         [2, 0, 0, 0],
//         false,
//     );
// }

// #[test]
// fn rv32_alu_sub_out_of_range_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::SUB,
//         [F::NEG_ONE.as_canonical_u32(), 0, 0, 0],
//         [1, 0, 0, 0],
//         [2, 0, 0, 0],
//         true,
//     );
// }

// #[test]
// fn rv32_alu_xor_wrong_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::XOR,
//         [255, 255, 255, 255],
//         [0, 0, 1, 0],
//         [255, 255, 255, 255],
//         true,
//     );
// }

// #[test]
// fn rv32_alu_or_wrong_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::OR,
//         [255, 255, 255, 255],
//         [255, 255, 255, 254],
//         [0, 0, 0, 0],
//         true,
//     );
// }

// #[test]
// fn rv32_alu_and_wrong_negative_test() {
//     run_rv32_alu_negative_test(
//         BaseAluOpcode::AND,
//         [255, 255, 255, 255],
//         [0, 0, 1, 0],
//         [0, 0, 0, 0],
//         true,
//     );
// }

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_add_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [23, 205, 73, 49];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BaseAluOpcode::ADD, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_sub_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [179, 118, 240, 172];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BaseAluOpcode::SUB, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_xor_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [215, 138, 49, 173];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BaseAluOpcode::XOR, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_or_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [247, 171, 61, 239];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BaseAluOpcode::OR, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_and_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [32, 33, 12, 66];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BaseAluOpcode::AND, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// ADAPTER TESTS
//
// Ensure that the adapter is correct.
//////////////////////////////////////////////////////////////////////////////////////

// A pranking chip where `preprocess` can have `rs2` limbs that overflow.
#[derive(derive_new::new)]
struct Rv32BaseAluAdapterTestStep(Rv32BaseAluAdapterStep<RV32_CELL_BITS>);

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32BaseAluAdapterTestStep
where
    F: PrimeField32,
{
    const WIDTH: usize =
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::WIDTH;
    type ReadData = <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::ReadData;
    type WriteData =
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::WriteData;
    type TraceContext<'a> =
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::TraceContext<'a>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::start(
            pc,
            memory,
            adapter_row,
        );
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_REGISTER_AS || e.as_canonical_u32() == RV32_IMM_AS
        );

        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.rs1_ptr = b;
        let rs1 = tracing_read(
            memory,
            d.as_canonical_u32(),
            b.as_canonical_u32(),
            &mut adapter_row.reads_aux[0],
        );

        let rs2 = if e.as_canonical_u32() == RV32_REGISTER_AS {
            adapter_row.rs2_as = e;
            adapter_row.rs2 = c;

            tracing_read(
                memory,
                e.as_canonical_u32(),
                c.as_canonical_u32(),
                &mut adapter_row.reads_aux[1],
            )
        } else {
            adapter_row.rs2_as = e;

            // Here we use values that can overflow
            let c_u32 = c.as_canonical_u32();
            let mask1 = (1 << 9) - 1; // Allowing overflow
            let mask2 = (1 << 3) - 2; // Allowing overflow

            let rs2 = [
                (c_u32 & mask1) as u8,
                ((c_u32 >> 8) & mask2) as u8,
                (c_u32 >> 16) as u8,
                (c_u32 >> 16) as u8,
            ];

            tracing_read_imm(memory, c.as_canonical_u32(), &mut adapter_row.rs2);
            rs2
        };

        (rs1, rs2)
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::write(
            &self.0,
            memory,
            instruction,
            adapter_row,
            data,
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        bitwise_lookup_chip: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterTraceStep<F, CTX>>::fill_trace_row(
            &self.0,
            mem_helper,
            bitwise_lookup_chip,
            adapter_row,
        );
    }
}

impl<F> AdapterExecutorE1<F> for Rv32BaseAluAdapterTestStep
where
    F: PrimeField32,
{
    type ReadData = <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterExecutorE1<F>>::ReadData;
    type WriteData = <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterExecutorE1<F>>::WriteData;

    #[inline(always)]
    fn read<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData
    where
        Mem: GuestMemory,
    {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_REGISTER_AS || e.as_canonical_u32() == RV32_IMM_AS
        );

        let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
            unsafe { memory.read(d.as_canonical_u32(), b.as_canonical_u32()) };

        let rs2 = if e.as_canonical_u32() == RV32_REGISTER_AS {
            let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
                unsafe { memory.read(e.as_canonical_u32(), c.as_canonical_u32()) };
            rs2
        } else {
            // Here we use values that can overflow
            let imm = c.as_canonical_u32();

            debug_assert_eq!(imm >> 24, 0);

            let mask1 = (1 << 9) - 1; // Allowing overflow
            let mask2 = (1 << 3) - 2; // Allowing overflow

            let mut imm_le = [
                (imm & mask1) as u8,
                ((imm >> 8) & mask2) as u8,
                (imm >> 16) as u8,
                (imm >> 16) as u8,
            ];
            imm_le[3] = imm_le[2];
            imm_le
        };

        (rs1, rs2)
    }

    #[inline(always)]
    fn write<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>, data: &Self::WriteData)
    where
        Mem: GuestMemory,
    {
        <Rv32BaseAluAdapterStep<RV32_CELL_BITS> as AdapterExecutorE1<F>>::write(
            &self.0,
            memory,
            instruction,
            data,
        );
    }
}

// #[test]
// fn rv32_alu_adapter_unconstrained_imm_limb_test() {
//     let mut rng = create_seeded_rng();
//     let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
//     let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

//     let mut tester = VmChipTestBuilder::default();

//     let mut chip = NewVmChipWrapper::<F, _, _>::new(
//         VmAirWrapper::new(
//             Rv32BaseAluAdapterAir::new(
//                 tester.execution_bridge(),
//                 tester.memory_bridge(),
//                 bitwise_bus,
//             ),
//             BaseAluCoreAir::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::new(
//                 bitwise_bus,
//                 BaseAluOpcode::CLASS_OFFSET,
//             ),
//         ),
//         BaseAluStep::new(
//             Rv32BaseAluAdapterTestStep(Rv32BaseAluAdapterStep::new()),
//             bitwise_chip.clone(),
//             BaseAluOpcode::CLASS_OFFSET,
//         ),
//         MAX_INS_CAPACITY,
//         tester.memory_helper(),
//     );

//     let b = [0, 0, 0, 0];
//     let (c_imm, c) = {
//         let imm = (1 << 11) - 1;
//         let fake_c: [u32; 4] = [(1 << 9) - 1, (1 << 3) - 2, 0, 0];
//         let fake_c = fake_c.map(|x| x as u8);
//         (Some(imm), fake_c)
//     };

//     let (instruction, _rd) = rv32_rand_write_register_or_imm(
//         &mut tester,
//         b,
//         c,
//         c_imm,
//         BaseAluOpcode::ADD.global_opcode().as_usize(),
//         &mut rng,
//     );
//     tester.execute(&mut chip, &instruction);

//     disable_debug_builder();
//     let tester = tester.build().load(chip).load(bitwise_chip).finalize();
//     tester.simple_test_with_expected_error(VerificationError::ChallengePhaseError);
// }

#[test]
fn rv32_alu_adapter_unconstrained_rs2_read_test() {
    let mut rng = create_seeded_rng();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32BaseAluChip::new(
        VmAirWrapper::new(
            Rv32BaseAluAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
            ),
            BaseAluCoreAir::new(bitwise_bus, BaseAluOpcode::CLASS_OFFSET),
        ),
        Rv32BaseAluStep::new(
            Rv32BaseAluAdapterStep::new(),
            bitwise_chip.clone(),
            BaseAluOpcode::CLASS_OFFSET,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    let b = [1, 1, 1, 1];
    let c = [1, 1, 1, 1];
    let (instruction, _rd) = rv32_rand_write_register_or_imm(
        &mut tester,
        b,
        c,
        None,
        BaseAluOpcode::ADD.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut chip, &instruction);

    let trace_width = chip.trace_width();
    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let mut dummy_values = values.clone();
        let cols: &mut BaseAluCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            dummy_values.split_at_mut(adapter_width).1.borrow_mut();
        cols.opcode_add_flag = F::ZERO;
        values.extend(dummy_values);
        *trace = RowMajorMatrix::new(values, trace_width);
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(VerificationError::OodEvaluationMismatch);
}
