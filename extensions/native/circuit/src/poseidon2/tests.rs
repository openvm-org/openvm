use std::cmp::min;

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, VmChipTestBuilder, VmChipTester},
    verify_single, VirtualMachine,
};
use openvm_instructions::{instruction::Instruction, program::Program, LocalOpcode, SystemOpcode};
use openvm_native_compiler::{
    conversion::AS, FieldArithmeticOpcode, Poseidon2Opcode, Poseidon2Opcode::*,
    VerifyBatchOpcode::VERIFY_BATCH,
};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32, PrimeField64},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_blake3::{BabyBearBlake3Config, BabyBearBlake3Engine},
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::{rngs::StdRng, Rng};

use super::air::VerifyBatchBus;
use crate::{
    poseidon2::{new_native_poseidon2_chip, CHUNK},
    NativeConfig,
};

const VERIFY_BATCH_BUS: VerifyBatchBus = VerifyBatchBus::new(7);
const MAX_INS_CAPACITY: usize = 1 << 15;

fn compute_commit<F: Field>(
    dim: &[usize],
    opened: &[Vec<F>],
    proof: &[[F; CHUNK]],
    sibling_is_on_right: &[bool],
    hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
) -> [F; CHUNK] {
    let mut log_height = dim[0] as isize;
    let mut proof_index = 0;
    let mut opened_index = 0;
    let mut root = [F::ZERO; CHUNK];
    while log_height >= 0 {
        let mut concat = vec![];
        while opened_index < opened.len() && dim[opened_index] == log_height as usize {
            concat.extend(opened[opened_index].clone());
            opened_index += 1;
        }
        if !concat.is_empty() {
            let mut left = [F::ZERO; CHUNK];
            let mut right = [F::ZERO; CHUNK];
            for i in (0..concat.len()).step_by(CHUNK) {
                left[..(min(i + CHUNK, concat.len()) - i)]
                    .copy_from_slice(&concat[i..min(i + CHUNK, concat.len())]);
                (left, right) = hash_function(left, right);
            }
            root = if log_height as usize == dim[0] {
                left
            } else {
                hash_function(root, left).0
            }
        }
        if log_height > 0 {
            let sibling = proof[proof_index];
            let (left, right) = if sibling_is_on_right[proof_index] {
                (sibling, root)
            } else {
                (root, sibling)
            };
            root = hash_function(left, right).0;
        }
        log_height -= 1;
        proof_index += 1;
    }
    root
}

type F = BabyBear;

#[derive(Debug, Clone)]
struct VerifyBatchInstance {
    dim: Vec<usize>,
    opened: Vec<Vec<F>>,
    proof: Vec<[F; CHUNK]>,
    sibling_is_on_right: Vec<bool>,
    commit: [F; CHUNK],
}

fn random_instance(
    rng: &mut StdRng,
    row_lengths: Vec<Vec<usize>>,
    opened_element_size: usize,
    hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
) -> VerifyBatchInstance {
    let mut dims = vec![];
    let mut opened = vec![];
    let mut proof = vec![];
    let mut sibling_is_on_right = vec![];
    for (log_height, row_lengths) in row_lengths.iter().enumerate() {
        for &row_length in row_lengths {
            dims.push(log_height);
            let mut opened_row = vec![];
            for _ in 0..opened_element_size * row_length {
                opened_row.push(rng.gen());
            }
            opened.push(opened_row);
        }
        if log_height > 0 {
            proof.push(std::array::from_fn(|_| rng.gen()));
            sibling_is_on_right.push(rng.gen());
        }
    }

    dims.reverse();
    opened.reverse();
    proof.reverse();
    sibling_is_on_right.reverse();

    let commit = compute_commit(&dims, &opened, &proof, &sibling_is_on_right, hash_function);

    VerifyBatchInstance {
        dim: dims,
        opened,
        proof,
        sibling_is_on_right,
        commit,
    }
}

const SBOX_REGISTERS: usize = 1;

struct Case {
    row_lengths: Vec<Vec<usize>>,
    opened_element_size: usize,
}

fn test<const N: usize>(cases: [Case; N]) {
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    // single op
    let address_space = AS::Native as usize;

    let mut tester = VmChipTestBuilder::default_native();
    let mut chip = new_native_poseidon2_chip::<F, SBOX_REGISTERS>(
        tester.system_port(),
        Poseidon2Config::default(),
        VERIFY_BATCH_BUS,
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    let mut rng = create_seeded_rng();
    for Case {
        row_lengths,
        opened_element_size,
    } in cases
    {
        let instance =
            random_instance(&mut rng, row_lengths, opened_element_size, |left, right| {
                let concatenated =
                    std::array::from_fn(|i| if i < CHUNK { left[i] } else { right[i - CHUNK] });
                let permuted = chip.step.subchip.permute(concatenated);
                (
                    std::array::from_fn(|i| permuted[i]),
                    std::array::from_fn(|i| permuted[i + CHUNK]),
                )
            });
        let VerifyBatchInstance {
            dim,
            opened,
            proof,
            sibling_is_on_right,
            commit,
        } = instance;

        let dim_register = gen_pointer(&mut rng, 1);
        let opened_register = gen_pointer(&mut rng, 1);
        let opened_length_register = gen_pointer(&mut rng, 1);
        let proof_id = gen_pointer(&mut rng, 1);
        let index_register = gen_pointer(&mut rng, 1);
        let commit_register = gen_pointer(&mut rng, 1);

        let dim_base_pointer = gen_pointer(&mut rng, 1);
        let opened_base_pointer = gen_pointer(&mut rng, 2);
        let index_base_pointer = gen_pointer(&mut rng, 1);
        let commit_pointer = gen_pointer(&mut rng, 1);

        tester.write_usize(address_space, dim_register, [dim_base_pointer]);
        tester.write_usize(address_space, opened_register, [opened_base_pointer]);
        tester.write_usize(address_space, opened_length_register, [opened.len()]);
        tester.write_usize(address_space, proof_id, [tester.streams.hint_space.len()]);
        tester.write_usize(address_space, index_register, [index_base_pointer]);
        tester.write_usize(address_space, commit_register, [commit_pointer]);

        for (i, &dim_value) in dim.iter().enumerate() {
            tester.write_usize(address_space, dim_base_pointer + i, [dim_value]);
        }
        for (i, opened_row) in opened.iter().enumerate() {
            let row_pointer = gen_pointer(&mut rng, 1);
            tester.write_usize(
                address_space,
                opened_base_pointer + (2 * i),
                [row_pointer, opened_row.len() / opened_element_size],
            );
            for (j, &opened_value) in opened_row.iter().enumerate() {
                tester.write(address_space, row_pointer + j, [opened_value]);
            }
        }
        tester
            .streams
            .hint_space
            .push(proof.iter().flatten().copied().collect());
        for (i, &bit) in sibling_is_on_right.iter().enumerate() {
            tester.write(address_space, index_base_pointer + i, [F::from_bool(bit)]);
        }
        tester.write(address_space, commit_pointer, commit);

        let opened_element_size_inv = F::from_canonical_usize(opened_element_size)
            .inverse()
            .as_canonical_u32() as usize;
        tester.execute(
            &mut chip,
            &Instruction::from_usize(
                VERIFY_BATCH.global_opcode(),
                [
                    dim_register,
                    opened_register,
                    opened_length_register,
                    proof_id,
                    index_register,
                    commit_register,
                    opened_element_size_inv,
                ],
            ),
        );
    }

    let mut tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    let trace = tester.air_proof_inputs[2]
        .1
        .raw
        .common_main
        .as_mut()
        .unwrap();
    let row_index = 0;
    trace.row_mut(row_index);

    let p2_chip = Poseidon2SubChip::<F, SBOX_REGISTERS>::new(Poseidon2Config::default().constants);
    let inner_trace = p2_chip.generate_trace(vec![[F::ZERO; 2 * CHUNK]]);
    let inner_width = p2_chip.air.width();

    trace.row_mut(row_index)[..inner_width].copy_from_slice(&inner_trace.values);
    // Run a test after pranking the poseidon2 stuff
    assert_eq!(
        tester.simple_test().err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn verify_batch_test_felt() {
    test([Case {
        row_lengths: vec![vec![3], vec![], vec![9, 2, 1, 13, 4], vec![16]],
        opened_element_size: 1,
    }]);
}

#[test]
fn verify_batch_test_felt_multiple() {
    test([
        Case {
            row_lengths: vec![vec![1, 1, 1, 2, 3], vec![9], vec![8]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![], vec![], vec![], vec![1]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![8], vec![7], vec![6]],
            opened_element_size: 1,
        },
    ])
}

#[test]
fn verify_batch_test_ext() {
    test([Case {
        row_lengths: vec![vec![3], vec![], vec![1, 2, 1], vec![4]],
        opened_element_size: 4,
    }]);
}

#[test]
fn verify_batch_test_ext_multiple() {
    test([
        Case {
            row_lengths: vec![vec![1, 1, 1], vec![3], vec![2]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![], vec![], vec![], vec![1]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![4], vec![3], vec![2]],
            opened_element_size: 4,
        },
    ])
}

#[test]
fn verify_batch_test_felt_and_ext() {
    test([
        Case {
            row_lengths: vec![vec![3], vec![], vec![9, 2, 1, 13, 4], vec![16]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![1, 1, 1], vec![3], vec![2]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![8], vec![7], vec![6]],
            opened_element_size: 1,
        },
        Case {
            row_lengths: vec![vec![], vec![], vec![], vec![1]],
            opened_element_size: 4,
        },
        Case {
            row_lengths: vec![vec![4], vec![3], vec![2]],
            opened_element_size: 4,
        },
    ])
}

/// Create random instructions for the poseidon2 chip.
fn random_instructions(num_ops: usize) -> Vec<Instruction<BabyBear>> {
    let mut rng = create_seeded_rng();
    (0..num_ops)
        .map(|_| {
            let [a, b, c] =
                std::array::from_fn(|_| BabyBear::from_canonical_usize(gen_pointer(&mut rng, 1)));
            Instruction {
                opcode: if rng.gen_bool(0.5) {
                    PERM_POS2
                } else {
                    COMP_POS2
                }
                .global_opcode(),
                a,
                b,
                c,
                d: BabyBear::from_canonical_usize(4),
                e: BabyBear::from_canonical_usize(4),
                f: BabyBear::ZERO,
                g: BabyBear::ZERO,
            }
        })
        .collect()
}

fn tester_with_random_poseidon2_ops(num_ops: usize) -> VmChipTester<BabyBearBlake3Config> {
    let elem_range = || 1..=100;

    let mut tester = VmChipTestBuilder::default_native();
    let mut chip = new_native_poseidon2_chip::<F, SBOX_REGISTERS>(
        tester.system_port(),
        Poseidon2Config::default(),
        VERIFY_BATCH_BUS,
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    let mut rng = create_seeded_rng();

    for instruction in random_instructions(num_ops) {
        let opcode = Poseidon2Opcode::from_usize(
            instruction
                .opcode
                .local_opcode_idx(Poseidon2Opcode::CLASS_OFFSET),
        );
        let [a, b, c, d, e] = [
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
        ]
        .map(|elem| elem.as_canonical_u64() as usize);

        let dst = gen_pointer(&mut rng, CHUNK) / 2;
        let lhs = gen_pointer(&mut rng, CHUNK) / 2;
        let rhs = gen_pointer(&mut rng, CHUNK) / 2;

        let data: [_; 2 * CHUNK] =
            std::array::from_fn(|_| BabyBear::from_canonical_usize(rng.gen_range(elem_range())));

        let hash = chip.step.subchip.permute(data);

        tester.write(d, a, [BabyBear::from_canonical_usize(dst)]);
        tester.write(d, b, [BabyBear::from_canonical_usize(lhs)]);
        if opcode == COMP_POS2 {
            tester.write(d, c, [BabyBear::from_canonical_usize(rhs)]);
        }

        let data_left: [_; CHUNK] = std::array::from_fn(|i| data[i]);
        let data_right: [_; CHUNK] = std::array::from_fn(|i| data[CHUNK + i]);
        match opcode {
            COMP_POS2 => {
                tester.write(e, lhs, data_left);
                tester.write(e, rhs, data_right);
            }
            PERM_POS2 => {
                tester.write(e, lhs, data_left);
                tester.write(e, lhs + CHUNK, data_right);
            }
        }

        tester.execute(&mut chip, &instruction);

        match opcode {
            COMP_POS2 => {
                let expected: [_; CHUNK] = std::array::from_fn(|i| hash[i]);
                let actual = tester.read::<{ CHUNK }>(e, dst);
                assert_eq!(expected, actual);
            }
            PERM_POS2 => {
                let actual_0 = tester.read::<{ CHUNK }>(e, dst);
                let actual_1 = tester.read::<{ CHUNK }>(e, dst + CHUNK);
                let actual = [actual_0, actual_1].concat();
                assert_eq!(&hash, &actual[..]);
            }
        }
    }
    tester.build().load(chip).finalize()
}

fn get_engine() -> BabyBearBlake3Engine {
    BabyBearBlake3Engine::new(FriParameters::new_for_testing(3))
}

#[test]
fn verify_batch_chip_simple_1() {
    let tester = tester_with_random_poseidon2_ops(1);
    tester.test(get_engine).expect("Verification failed");
}

#[test]
fn verify_batch_chip_simple_3() {
    let tester = tester_with_random_poseidon2_ops(3);
    tester.test(get_engine).expect("Verification failed");
}

#[test]
fn verify_batch_chip_simple_50() {
    let tester = tester_with_random_poseidon2_ops(50);
    tester.test(get_engine).expect("Verification failed");
}

// log_blowup = 3 for poseidon2 chip
fn air_test_with_compress_poseidon2(
    poseidon2_max_constraint_degree: usize,
    program: Program<BabyBear>,
) {
    let fri_params = if matches!(std::env::var("OPENVM_FAST_TEST"), Ok(x) if &x == "1") {
        FriParameters {
            log_blowup: 3,
            log_final_poly_len: 0,
            num_queries: 2,
            proof_of_work_bits: 0,
        }
    } else {
        standard_fri_params_with_100_bits_conjectured_security(3)
    };
    let engine = BabyBearPoseidon2Engine::new(fri_params);

    let config = NativeConfig::aggregation(0, poseidon2_max_constraint_degree);
    let vm = VirtualMachine::new(engine, config);

    let pk = vm.keygen();
    let vk = pk.get_vk();
    let segments = vm
        .execute_metered(program.clone(), vec![], &vk.num_interactions())
        .unwrap();
    let result = vm.execute_and_generate(program, vec![], &segments).unwrap();
    let proofs = vm.prove(&pk, result);
    for proof in proofs {
        verify_single(&vm.engine, &pk.get_vk(), &proof).expect("Verification failed");
    }
}

#[test]
fn test_vm_compress_poseidon2_as4() {
    let mut rng = create_seeded_rng();

    let mut instructions = vec![];

    let lhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [lhs_ptr + i]_4 <- rnd()
        instructions.push(Instruction::large_from_isize(
            FieldArithmeticOpcode::ADD.global_opcode(),
            lhs_ptr + i,
            rng.gen_range(1..1 << 20),
            0,
            4,
            0,
            0,
            0,
        ));
    }
    let rhs_ptr = gen_pointer(&mut rng, CHUNK) as isize;
    for i in 0..CHUNK as isize {
        // [rhs_ptr + i]_4 <- rnd()
        instructions.push(Instruction::large_from_isize(
            FieldArithmeticOpcode::ADD.global_opcode(),
            rhs_ptr + i,
            rng.gen_range(1..1 << 20),
            0,
            4,
            0,
            0,
            0,
        ));
    }
    let dst_ptr = gen_pointer(&mut rng, CHUNK) as isize;

    // [11]_4 <- lhs_ptr
    instructions.push(Instruction::large_from_isize(
        FieldArithmeticOpcode::ADD.global_opcode(),
        11,
        lhs_ptr,
        0,
        4,
        0,
        0,
        0,
    ));

    // [22]_4 <- rhs_ptr
    instructions.push(Instruction::large_from_isize(
        FieldArithmeticOpcode::ADD.global_opcode(),
        22,
        rhs_ptr,
        0,
        4,
        0,
        0,
        0,
    ));
    // [33]_4 <- dst_ptr
    instructions.push(Instruction::large_from_isize(
        FieldArithmeticOpcode::ADD.global_opcode(),
        33,
        0,
        dst_ptr,
        4,
        0,
        0,
        0,
    ));

    instructions.push(Instruction::from_isize(
        COMP_POS2.global_opcode(),
        33,
        11,
        22,
        4,
        4,
    ));
    instructions.push(Instruction::from_isize(
        SystemOpcode::TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    ));

    let program = Program::from_instructions(&instructions);

    air_test_with_compress_poseidon2(3, program.clone());
    air_test_with_compress_poseidon2(7, program.clone());
}
