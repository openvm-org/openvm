use std::cmp::min;

use openvm_circuit::arch::testing::{memory::gen_pointer, VmChipTestBuilder, VmChipTester};
use openvm_instructions::{
    instruction::Instruction, Poseidon2Opcode, Poseidon2Opcode::*, UsizeOpcode, VmOpcode,
};
use openvm_native_compiler::{VerifyBatchOpcode, VerifyBatchOpcode::VERIFY_BATCH};
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
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::{rngs::StdRng, Rng};

use crate::verify_batch::{chip::VerifyBatchChip, CHUNK};

fn compute_commit<F: Field>(
    dim: &[usize],
    opened: &[Vec<F>],
    proof: &[[F; CHUNK]],
    root_is_on_right: &[bool],
    hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
) -> [F; CHUNK] {
    let mut height = dim[0];
    let mut proof_index = 0;
    let mut opened_index = 0;
    let mut root = [F::ZERO; CHUNK];
    while height >= 1 {
        let mut concat = vec![];
        while opened_index < opened.len() && dim[opened_index] == height {
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
            root = if height == dim[0] {
                left
            } else {
                hash_function(root, left).0
            }
        }
        if height > 1 {
            let sibling = proof[proof_index];
            let (left, right) = if root_is_on_right[proof_index] {
                (sibling, root)
            } else {
                (root, sibling)
            };
            root = hash_function(left, right).0;
        }
        height /= 2;
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
    root_is_on_right: Vec<bool>,
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
    let mut root_is_on_right = vec![];
    for (lg_height, row_lengths) in row_lengths.iter().enumerate() {
        let height = 1 << lg_height;
        for &row_length in row_lengths {
            dims.push(height);
            let mut opened_row = vec![];
            for _ in 0..opened_element_size * row_length {
                opened_row.push(rng.gen());
            }
            opened.push(opened_row);
        }
        if height > 1 {
            proof.push(std::array::from_fn(|_| rng.gen()));
            root_is_on_right.push(rng.gen());
        }
    }

    dims.reverse();
    opened.reverse();
    proof.reverse();
    root_is_on_right.reverse();

    let commit = compute_commit(&dims, &opened, &proof, &root_is_on_right, hash_function);

    VerifyBatchInstance {
        dim: dims,
        opened,
        proof,
        root_is_on_right,
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
    let address_space = 5;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = VerifyBatchChip::<F, SBOX_REGISTERS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        VerifyBatchOpcode::default_offset(),
        Poseidon2Opcode::default_offset(),
        tester.offline_memory_mutex_arc(),
        Poseidon2Config::default(),
    );

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
                let permuted = chip.subchip.permute(concatenated);
                (
                    std::array::from_fn(|i| permuted[i]),
                    std::array::from_fn(|i| permuted[i + CHUNK]),
                )
            });
        let VerifyBatchInstance {
            dim,
            opened,
            proof,
            root_is_on_right,
            commit,
        } = instance;

        let dim_register = gen_pointer(&mut rng, 1);
        let opened_register = gen_pointer(&mut rng, 1);
        let opened_length_register = gen_pointer(&mut rng, 1);
        let sibling_register = gen_pointer(&mut rng, 1);
        let index_register = gen_pointer(&mut rng, 1);
        let commit_register = gen_pointer(&mut rng, 1);

        let dim_base_pointer = gen_pointer(&mut rng, 1);
        let opened_base_pointer = gen_pointer(&mut rng, 2);
        let sibling_base_pointer = gen_pointer(&mut rng, 1);
        let index_base_pointer = gen_pointer(&mut rng, 1);
        let commit_pointer = gen_pointer(&mut rng, 1);

        tester.write_usize(address_space, dim_register, [dim_base_pointer]);
        tester.write_usize(address_space, opened_register, [opened_base_pointer]);
        tester.write_usize(address_space, opened_length_register, [opened.len()]);
        tester.write_usize(address_space, sibling_register, [sibling_base_pointer]);
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
                tester.write_cell(address_space, row_pointer + j, opened_value);
            }
        }
        for (i, &sibling) in proof.iter().enumerate() {
            let row_pointer = gen_pointer(&mut rng, 1);
            tester.write_usize(
                address_space,
                sibling_base_pointer + (2 * i),
                [row_pointer, CHUNK],
            );
            tester.write(address_space, row_pointer, sibling);
        }
        for (i, &bit) in root_is_on_right.iter().enumerate() {
            tester.write_cell(address_space, index_base_pointer + i, F::from_bool(bit));
        }
        tester.write(address_space, commit_pointer, commit);

        let opened_element_size_inv = F::from_canonical_usize(opened_element_size)
            .inverse()
            .as_canonical_u32() as usize;
        tester.execute(
            &mut chip,
            &Instruction::from_usize(
                VmOpcode::from_usize(VERIFY_BATCH as usize + VerifyBatchOpcode::default_offset()),
                [
                    dim_register,
                    opened_register,
                    opened_length_register,
                    sibling_register,
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
    let trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
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
                opcode: VmOpcode::from_usize(
                    if rng.gen_bool(0.5) {
                        PERM_POS2
                    } else {
                        COMP_POS2
                    } as usize
                        + Poseidon2Opcode::default_offset(),
                ),
                a,
                b,
                c,
                d: BabyBear::ONE,
                e: BabyBear::TWO,
                f: BabyBear::ZERO,
                g: BabyBear::ZERO,
            }
        })
        .collect()
}

fn tester_with_random_poseidon2_ops(num_ops: usize) -> VmChipTester<BabyBearBlake3Config> {
    let elem_range = || 1..=100;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = VerifyBatchChip::<F, SBOX_REGISTERS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        VerifyBatchOpcode::default_offset(),
        Poseidon2Opcode::default_offset(),
        tester.offline_memory_mutex_arc(),
        Poseidon2Config::default(),
    );

    let mut rng = create_seeded_rng();

    for instruction in random_instructions(num_ops) {
        let opcode = Poseidon2Opcode::from_usize(
            instruction.opcode.as_usize() - Poseidon2Opcode::default_offset(),
        );
        let [a, b, c, d, e] = [
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
        ]
        .map(|elem| elem.as_canonical_u64() as usize);

        let dst = gen_pointer(&mut rng, CHUNK);
        let lhs = gen_pointer(&mut rng, CHUNK);
        let rhs = gen_pointer(&mut rng, CHUNK);

        let data: [_; 2 * CHUNK] =
            std::array::from_fn(|_| BabyBear::from_canonical_usize(rng.gen_range(elem_range())));

        let hash = chip.subchip.permute(data);

        tester.write_cell(d, a, BabyBear::from_canonical_usize(dst));
        tester.write_cell(d, b, BabyBear::from_canonical_usize(lhs));
        if opcode == COMP_POS2 {
            tester.write_cell(d, c, BabyBear::from_canonical_usize(rhs));
        }

        match opcode {
            COMP_POS2 => {
                let data_left: [_; CHUNK] = std::array::from_fn(|i| data[i]);
                let data_right: [_; CHUNK] = std::array::from_fn(|i| data[CHUNK + i]);
                tester.write(e, lhs, data_left);
                tester.write(e, rhs, data_right);
            }
            PERM_POS2 => {
                tester.write(e, lhs, data);
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
                let actual = tester.read::<{ 2 * CHUNK }>(e, dst);
                assert_eq!(hash, actual);
            }
        }
    }
    tester.build().load(chip).finalize()
}

fn get_engine() -> BabyBearBlake3Engine {
    BabyBearBlake3Engine::new(standard_fri_params_with_100_bits_conjectured_security(3))
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