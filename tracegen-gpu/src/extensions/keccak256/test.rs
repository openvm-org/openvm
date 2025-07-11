use std::array;

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, BITWISE_OP_LOOKUP_BUS, RANGE_CHECKER_BUS},
        InstructionExecutor, MemoryConfig, NewVmChipWrapper,
    },
    utils::get_random_message,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::VariableRangeCheckerBus,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_keccak256_circuit::{utils::keccak256, KeccakVmChip, KeccakVmStep};
use openvm_keccak256_transpiler::Rv32KeccakOpcode::{self, *};
use openvm_stark_backend::{p3_field::FieldAlgebra, verifier::VerificationError};
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::{rngs::StdRng, Rng};

#[cfg(test)]
use super::*;
use crate::testing::GpuChipTestBuilder;

const MAX_INS_CAPACITY: usize = 1024;

type KeccakVmDenseChip<F> = NewVmChipWrapper<F, KeccakVmAir, KeccakVmStep, DenseRecordArena>;

fn create_test_dense_chip(tester: &GpuChipTestBuilder) -> KeccakVmDenseChip<F> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = KeccakVmDenseChip::new(
        KeccakVmAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
            Rv32KeccakOpcode::CLASS_OFFSET,
        ),
        KeccakVmStep::new(
            bitwise_chip.clone(),
            Rv32KeccakOpcode::CLASS_OFFSET,
            tester.address_bits(),
        ),
        tester.cpu_memory_helper(),
    );

    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_test_sparse_chip(
    tester: &mut GpuChipTestBuilder,
) -> (KeccakVmChip<F>, SharedBitwiseOperationLookupChip<8>) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<8>::new(bitwise_bus);
    let mut chip = KeccakVmChip::new(
        KeccakVmAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
            Rv32KeccakOpcode::CLASS_OFFSET,
        ),
        KeccakVmStep::new(
            bitwise_chip.clone(),
            Rv32KeccakOpcode::CLASS_OFFSET,
            tester.address_bits(),
        ),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    (chip, bitwise_chip)
}

fn set_and_execute<E: InstructionExecutor<F>>(
    tester: &mut GpuChipTestBuilder,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: Rv32KeccakOpcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) {
    let len = len.unwrap_or(rng.gen_range(1..500));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let max_mem_ptr: u32 = 1 << tester.address_bits();
    let dst_ptr = rng.gen_range(0..max_mem_ptr);
    let dst_ptr = dst_ptr ^ (dst_ptr & 3);
    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_canonical_u8));
    let src_ptr = rng.gen_range(0..(max_mem_ptr - len as u32));
    let src_ptr = src_ptr ^ (src_ptr & 3);
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_canonical_u8));

    message.chunks(4).enumerate().for_each(|(i, chunk)| {
        let chunk: [&u8; 4] = array::from_fn(|i| chunk.get(i).unwrap_or(&0));
        tester.write(
            2,
            src_ptr as usize + i * 4,
            chunk.map(|&x| F::from_canonical_u8(x)),
        );
    });

    tester.execute(
        chip,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let output = keccak256(message);
    assert_eq!(
        output.map(F::from_canonical_u8),
        tester.read::<32>(2, dst_ptr as usize)
    );
}

#[test]
fn test_keccak256_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));

    // CPU execution
    let mut dense_chip = create_test_dense_chip(&tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut dense_chip,
            &mut rng,
            KECCAK256,
            None,
            None,
        );
    }

    set_and_execute(
        &mut tester,
        &mut dense_chip,
        &mut rng,
        KECCAK256,
        None,
        Some(2000),
    );

    let (mut sparse_chip, sparse_bitwise_chip) = create_test_sparse_chip(&mut tester);
    dense_chip
        .arena
        .get_record_seeker::<KeccakVmRecordMut, _>()
        .transfer_to_matrix_arena(&mut sparse_chip.arena);

    // GPU tracegen
    let bitwise_gpu_chip = Arc::new(BitwiseOperationLookupChipGPU::<RV32_CELL_BITS>::new(
        sparse_bitwise_chip.bus(),
    ));
    let mem_config = MemoryConfig::default();
    let var_range_gpu_chip = Arc::new(VariableRangeCheckerChipGPU::new(
        VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, mem_config.decomp),
    ));
    let gpu_chip = Keccak256ChipGpu::new(
        sparse_chip.air,
        var_range_gpu_chip,
        bitwise_gpu_chip,
        mem_config.pointer_max_bits as u32,
        dense_chip.arena,
    );

    tester
        .build()
        .load_and_compare(gpu_chip, sparse_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}
