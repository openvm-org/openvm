use std::array;

use openvm_circuit::{arch::testing::memory::gen_pointer, utils::get_random_message};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_keccak256_circuit::{
    utils::keccak256, KeccakVmAir, KeccakVmChip, KeccakVmExecutor, KeccakVmFiller,
};
use openvm_keccak256_transpiler::Rv32KeccakOpcode::{self, *};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::{rngs::StdRng, Rng};

#[cfg(test)]
use super::*;
use crate::testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness};

const MAX_INS_CAPACITY: usize = 1024;
type Harness =
    GpuTestChipHarness<F, KeccakVmExecutor, KeccakVmAir, Keccak256ChipGpu, KeccakVmChip<F>>;

fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = KeccakVmAir::new(
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_bus,
        tester.address_bits(),
        Rv32KeccakOpcode::CLASS_OFFSET,
    );

    let executor = KeccakVmExecutor::new(Rv32KeccakOpcode::CLASS_OFFSET, tester.address_bits());
    let cpu_chip = KeccakVmChip::new(
        KeccakVmFiller::new(dummy_bitwise_chip, tester.address_bits()),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Keccak256ChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits() as u32,
        tester.timestamp_max_bits() as u32,
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn set_and_execute(
    tester: &mut GpuChipTestBuilder,
    harness: &mut Harness,
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
        &mut harness.executor,
        &mut harness.dense_arena,
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
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_test_harness(&tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut harness, &mut rng, KECCAK256, None, None);
    }

    // Test special length edge cases:
    for len in [0, 135, 136, 137, 2000] {
        println!("Testing length: {}", len);
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            KECCAK256,
            None,
            Some(len),
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<KeccakVmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
