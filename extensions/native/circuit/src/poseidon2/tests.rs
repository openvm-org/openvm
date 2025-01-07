use openvm_circuit::arch::testing::{memory::gen_pointer, VmChipTestBuilder, VmChipTester};
use openvm_instructions::{instruction::Instruction, Poseidon2Opcode, UsizeOpcode, VmOpcode};
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField64};
use openvm_stark_sdk::{
    config::{
        baby_bear_blake3::{BabyBearBlake3Config, BabyBearBlake3Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::Rng;

use super::{NativePoseidon2Chip, NATIVE_POSEIDON2_CHUNK_SIZE, NATIVE_POSEIDON2_WIDTH};

/// Create random instructions for the poseidon2 chip.
fn random_instructions(num_ops: usize) -> Vec<Instruction<BabyBear>> {
    let mut rng = create_seeded_rng();
    (0..num_ops)
        .map(|_| {
            let [a, b, c] =
                std::array::from_fn(|_| BabyBear::from_canonical_usize(gen_pointer(&mut rng, 1)));
            Instruction {
                opcode: if rng.gen_bool(0.5) {
                    VmOpcode::from_usize(Poseidon2Opcode::PERM_POS2 as usize)
                } else {
                    VmOpcode::from_usize(Poseidon2Opcode::COMP_POS2 as usize)
                },
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

fn tester_with_random_poseidon2_ops(
    num_ops: usize,
    max_constraint_degree: usize,
) -> VmChipTester<BabyBearBlake3Config> {
    let elem_range = || 1..=100;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = NativePoseidon2Chip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_bridge(),
        Poseidon2Config::default(),
        0,
        max_constraint_degree,
        tester.offline_memory_mutex_arc(),
    );

    let mut rng = create_seeded_rng();

    for instruction in random_instructions(num_ops) {
        let opcode = Poseidon2Opcode::from_usize(instruction.opcode.as_usize());
        let [a, b, c, d, e] = [
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
        ]
        .map(|elem| elem.as_canonical_u64() as usize);

        let dst = gen_pointer(&mut rng, NATIVE_POSEIDON2_CHUNK_SIZE);
        let lhs = gen_pointer(&mut rng, NATIVE_POSEIDON2_CHUNK_SIZE);
        let rhs = gen_pointer(&mut rng, NATIVE_POSEIDON2_CHUNK_SIZE);

        let data: [_; NATIVE_POSEIDON2_WIDTH] =
            std::array::from_fn(|_| BabyBear::from_canonical_usize(rng.gen_range(elem_range())));

        let hash = match &chip {
            NativePoseidon2Chip::Register0(chip) => chip.subchip.permute(data),
            NativePoseidon2Chip::Register1(chip) => chip.subchip.permute(data),
        };

        tester.write_cell(d, a, BabyBear::from_canonical_usize(dst));
        tester.write_cell(d, b, BabyBear::from_canonical_usize(lhs));
        if opcode == Poseidon2Opcode::COMP_POS2 {
            tester.write_cell(d, c, BabyBear::from_canonical_usize(rhs));
        }

        match opcode {
            Poseidon2Opcode::COMP_POS2 => {
                let data_left: [_; NATIVE_POSEIDON2_CHUNK_SIZE] = std::array::from_fn(|i| data[i]);
                let data_right: [_; NATIVE_POSEIDON2_CHUNK_SIZE] =
                    std::array::from_fn(|i| data[NATIVE_POSEIDON2_CHUNK_SIZE + i]);
                tester.write(e, lhs, data_left);
                tester.write(e, rhs, data_right);
            }
            Poseidon2Opcode::PERM_POS2 => {
                tester.write(e, lhs, data);
            }
        }

        tester.execute(&mut chip, instruction);

        match opcode {
            Poseidon2Opcode::COMP_POS2 => {
                let expected: [_; NATIVE_POSEIDON2_CHUNK_SIZE] = std::array::from_fn(|i| hash[i]);
                let actual = tester.read::<NATIVE_POSEIDON2_CHUNK_SIZE>(e, dst);
                assert_eq!(expected, actual);
            }
            Poseidon2Opcode::PERM_POS2 => {
                let actual = tester.read::<NATIVE_POSEIDON2_WIDTH>(e, dst);
                assert_eq!(hash, actual);
            }
        }
    }
    tester.build().load(chip).finalize()
}

fn get_engine() -> BabyBearBlake3Engine {
    BabyBearBlake3Engine::new(standard_fri_params_with_100_bits_conjectured_security(3))
}

/// Checking that 50 random instructions pass.
#[test]
fn poseidon2_chip_random_max_constraint_degree_7() {
    let tester = tester_with_random_poseidon2_ops(50, 7);
    tester.test(get_engine).expect("Verification failed");
}

#[test]
fn poseidon2_chip_random_max_constraint_degree_3() {
    let tester = tester_with_random_poseidon2_ops(50, 3);
    tester.test(get_engine).expect("Verification failed");
}
