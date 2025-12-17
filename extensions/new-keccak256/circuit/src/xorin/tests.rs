use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::get_random_message,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_new_keccak256_transpiler::XorinOpcode;
use openvm_stark_backend::{p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use crate::xorin::{
    air::XorinVmAir,
    columns::{XorinInstructionCols, XorinMemoryCols, XorinSpongeCols},
    XorinVmChip, XorinVmExecutor, XorinVmFiller,
};

type F = BabyBear;
type Harness = TestChipHarness<F, XorinVmExecutor, XorinVmAir, XorinVmChip<F>>;
use openvm_stark_backend::verifier::VerificationError;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (XorinVmAir, XorinVmExecutor, XorinVmChip<F>) {
    let air = XorinVmAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        address_bits,
        XorinOpcode::CLASS_OFFSET,
    );

    let executor = XorinVmExecutor::new(XorinOpcode::CLASS_OFFSET, address_bits);
    let chip = XorinVmChip::new(
        XorinVmFiller::new(bitwise_chip, address_bits),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );

    const MAX_TRACE_ROWS: usize = 4096;

    let harness = Harness::with_capacity(executor, air, chip, MAX_TRACE_ROWS);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: XorinOpcode,
    buffer_length: Option<usize>,
) {
    const MAX_LEN: usize = 136;

    let buffer_length = match buffer_length {
        Some(length) => length,
        None => MAX_LEN,
    };

    assert!(buffer_length.is_multiple_of(4));

    let rand_buffer = get_random_message(rng, MAX_LEN);
    let mut rand_buffer_arr = [0u8; MAX_LEN];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

    let rand_input = get_random_message(rng, MAX_LEN);
    let mut rand_input_arr = [0u8; MAX_LEN];
    rand_input_arr.copy_from_slice(&rand_input);

    use openvm_circuit::arch::testing::memory::gen_pointer;
    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let buffer_ptr = gen_pointer(rng, buffer_length);
    let input_ptr = gen_pointer(rng, buffer_length);

    let rand_buffer_arr_f = rand_buffer_arr.map(F::from_canonical_u8);
    let rand_input_arr_f = rand_input_arr.map(F::from_canonical_u8);

    for i in 0..(buffer_length / 4) {
        let buffer_chunk: [F; 4] = rand_buffer_arr_f[4 * i..4 * i + 4]
            .try_into()
            .expect("slice has length 4");
        tester.write(2, buffer_ptr + 4 * i, buffer_chunk);

        let input_chunk: [F; 4] = rand_input_arr_f[4 * i..4 * i + 4]
            .try_into()
            .expect("slice has length 4");
        tester.write(2, input_ptr + 4 * i, input_chunk);
    }

    tester.write(1, rd, buffer_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs1, input_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(
        1,
        rs2,
        buffer_length.to_le_bytes().map(F::from_canonical_u8),
    );
    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let mut expected_output = [0u8; MAX_LEN];
    for i in 0..buffer_length {
        expected_output[i] = rand_buffer_arr[i] ^ rand_input_arr[i];
    }

    let mut output_buffer = [F::from_canonical_u8(0); MAX_LEN];

    for i in 0..(buffer_length / 4) {
        let output_chunk: [F; 4] = tester.read(2, buffer_ptr + 4 * i);
        output_buffer[4 * i..4 * i + 4].copy_from_slice(&output_chunk);
    }

    for i in 0..buffer_length {
        assert_eq!(F::from_canonical_u8(expected_output[i]), output_buffer[i]);
    }
}

#[test]
fn xorin_chip_positive_tests() {
    let num_ops: usize = 100;

    for _ in 0..num_ops {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let (mut harness, bitwise) = create_test_harness(&mut tester);

        let buffer_length = Some(rng.gen_range(1..=34) * 4 as usize);

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            XorinOpcode::XORIN,
            buffer_length,
        );

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(bitwise)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }
}

fn run_xorin_chip_negative_tests(
    prank_sponge: Option<XorinSpongeCols<F>>,
    prank_instruction: Option<XorinInstructionCols<F>>,
    prank_mem_oc: Option<XorinMemoryCols<F>>,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    let buffer_length = Some(rng.gen_range(1..=34) * 4_usize);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        XorinOpcode::XORIN,
        buffer_length,
    );

    use openvm_stark_backend::p3_matrix::dense::DenseMatrix;
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        use std::borrow::BorrowMut;

        use openvm_stark_backend::p3_matrix::Matrix;

        use crate::xorin::columns::XorinVmCols;

        let mut values = trace.row_slice(0).to_vec();
        let width = XorinVmCols::<F>::width();
        // split_at_mut() to avoid the compiler saying that it is
        // unable to determine the size during compile time
        let cols: &mut XorinVmCols<F> = values.split_at_mut(width).0.borrow_mut();

        if let Some(prank_sponge) = prank_sponge {
            cols.sponge = prank_sponge;
        }
        if let Some(prank_instruction) = prank_instruction {
            cols.instruction = prank_instruction;
        }
        if let Some(prank_mem_oc) = prank_mem_oc.clone() {
            cols.mem_oc = prank_mem_oc;
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    use openvm_stark_backend::utils::disable_debug_builder;

    disable_debug_builder();

    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();

    if interaction_error {
        tester.simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    } else {
        tester.simple_test_with_expected_error(VerificationError::OodEvaluationMismatch);
    }
}
