use std::{array, sync::Arc};

use hex::FromHex;
use openvm_circuit::{arch::testing::memory::gen_pointer, utils::get_random_message};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChip;
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_sha256_air::{get_sha256_num_blocks, SHA256_BLOCK_U8S};
use openvm_sha256_circuit::{Sha256VmAir, Sha256VmChip, Sha256VmFiller, Sha256VmStep};
use openvm_sha256_transpiler::Rv32Sha256Opcode::{self, *};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::*;
use crate::{
    extensions::sha256::Sha256VmChipGpu,
    testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
};

type F = BabyBear;

const MAX_INS_CAPACITY: usize = 8192;
type Harness = GpuTestChipHarness<F, Sha256VmStep, Sha256VmAir, Sha256VmChipGpu, Sha256VmChip<F>>;

fn create_test_harness(tester: &GpuChipTestBuilder) -> Harness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = Sha256VmAir::new(
        tester.system_port(),
        bitwise_bus,
        tester.address_bits(),
        Rv32Sha256Opcode::CLASS_OFFSET as u16,
    );

    let executor = Sha256VmStep::new(Rv32Sha256Opcode::CLASS_OFFSET, tester.address_bits());
    let cpu_chip = Sha256VmChip::new(
        Sha256VmFiller::new(dummy_bitwise_chip, tester.address_bits()),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Sha256VmChipGpu::new(
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
    opcode: Rv32Sha256Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) {
    let len = len.unwrap_or(rng.gen_range(1..500));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);
    let len = message.len();

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let dst_ptr = gen_pointer(rng, 4);
    let src_ptr = gen_pointer(rng, 4);

    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_canonical_u8));

    let num_blocks = get_sha256_num_blocks(len as u32) as usize;

    for offset in (0..num_blocks * SHA256_BLOCK_U8S).step_by(4) {
        let chunk: [F; 4] = array::from_fn(|i| {
            if offset + i < message.len() {
                F::from_canonical_u8(message[offset + i])
            } else {
                F::from_canonical_u8(rng.gen())
            }
        });

        tester.write(2, src_ptr + offset, chunk);
    }

    tester.execute(
        &mut harness.executor,
        &mut harness.dense_arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );
}

#[test]
fn test_sha256_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_test_harness(&tester);

    let num_ops = 70;
    for i in 1..=num_ops {
        set_and_execute(&mut tester, &mut harness, &mut rng, SHA256, None, Some(i));
    }

    harness
        .dense_arena
        .get_record_seeker::<Sha256VmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[test]
fn test_sha256_known_vectors() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_test_harness(&tester);

    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (
            "98c1c0bdb7d5fea9a88859f06c6c439f",
            "b6b2c9c9b6f30e5c66c977f1bd7ad97071bee739524aecf793384890619f2b05",
        ),
        ("5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c", "ac0e25049870b91d78ef6807bb87fce4603c81abd3c097fba2403fd18b6ce0b7"),
        ("9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e", "080ad71239852124fc26758982090611b9b19abf22d22db3a57f67a06e984a23")
    ];

    for (input, _) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();

        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            SHA256,
            Some(&input),
            None,
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<Sha256VmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
