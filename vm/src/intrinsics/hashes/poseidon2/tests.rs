use afs_primitives::sub_chip::LocalTraceInstructions;
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm, BabyBearPoseidon2Engine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Config};
use rand::Rng;

use super::{Poseidon2Chip, CHUNK, WIDTH};
use crate::{
    arch::{
        instructions::{
            Poseidon2Opcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder, VmChipTester},
    },
    intrinsics::hashes::poseidon2::Poseidon2VmIoCols,
    system::program::Instruction,
};

fn get_engine(max_trace_height: usize) -> BabyBearPoseidon2Engine {
    let max_log_degree = log2_strict_usize(max_trace_height);
    let perm = random_perm();
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    engine_from_perm(perm, max_log_degree, fri_params)
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
                    PERM_POS2 as usize
                } else {
                    COMP_POS2 as usize
                },
                op_a: a,
                op_b: b,
                op_c: c,
                d: BabyBear::one(),
                e: BabyBear::two(),
                op_f: BabyBear::zero(),
                op_g: BabyBear::zero(),
                debug: String::new(),
            }
        })
        .collect()
}

fn tester_with_random_poseidon2_ops(num_ops: usize) -> VmChipTester {
    let elem_range = || 1..=100;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Poseidon2Chip::from_poseidon2_config(
        Poseidon2Config::<16, _>::new_p3_baby_bear_16(),
        7,
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        0,
    );

    let mut rng = create_seeded_rng();

    for instruction in random_instructions(num_ops) {
        let opcode = Poseidon2Opcode::from_usize(instruction.opcode);
        let [a, b, c, d, e] = [
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
        ]
        .map(|elem| elem.as_canonical_u64() as usize);

        let dst = gen_pointer(&mut rng, CHUNK);
        let lhs = gen_pointer(&mut rng, CHUNK);
        let rhs = gen_pointer(&mut rng, CHUNK);

        let data: [_; WIDTH] =
            std::array::from_fn(|_| BabyBear::from_canonical_usize(rng.gen_range(elem_range())));

        let hash = LocalTraceInstructions::generate_trace_row(&chip.air.inner, data)
            .io
            .output;

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

        tester.execute(&mut chip, instruction);

        match opcode {
            COMP_POS2 => {
                let expected: [_; CHUNK] = std::array::from_fn(|i| hash[i]);
                let actual = tester.read::<CHUNK>(e, dst);
                assert_eq!(expected, actual);
            }
            PERM_POS2 => {
                let actual = tester.read::<WIDTH>(e, dst);
                assert_eq!(hash, actual);
            }
        }
    }
    tester.build().load(chip).finalize()
}

/// Checking that 50 random instructions pass.
#[test]
fn poseidon2_chip_random_50_test_new() {
    let tester = tester_with_random_poseidon2_ops(50);
    tester.test(get_engine).expect("Verification failed");
}

/// Negative test, pranking internal poseidon2 trace values.
#[test]
fn poseidon2_negative_test() {
    let mut rng = create_seeded_rng();
    let num_ops = 1;
    let mut tester = tester_with_random_poseidon2_ops(num_ops);

    tester.test(get_engine).expect("Verification failed");

    disable_debug_builder();
    // test is slow, avoid too many repetitions
    for _ in 0..5 {
        // TODO: better way to modify existing traces in tester
        let trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
        let original_trace = trace.clone();

        // avoid pranking IO cols or dst,lhs,rhs
        let start_prank_col = Poseidon2VmIoCols::<u8>::get_width() + 3;
        let end_prank_col = start_prank_col + Poseidon2Air::<16, BabyBear>::default().get_width();
        let width = rng.gen_range(start_prank_col..end_prank_col);
        let height = rng.gen_range(0..num_ops);
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        println!("Pranking row {height} column {width}");
        trace.row_mut(height)[width] += rand;

        let test_result = tester.test(get_engine);

        assert_eq!(
            test_result.err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        tester.air_proof_inputs[2].raw.common_main = Some(original_trace);
    }
}

// /// Test that the direct bus interactions work.
// #[test]
// fn poseidon2_direct_test() {
//     let mut rng = create_seeded_rng();
//     const NUM_OPS: usize = 50;
//     const CHUNKS: usize = 8;
//     let correct_height = NUM_OPS.next_power_of_two();
//     let hashes: [([BabyBear; CHUNKS], [BabyBear; CHUNKS]); NUM_OPS] = std::array::from_fn(|_| {
//         (
//             std::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
//             std::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
//         )
//     });
//
//     let mut chip = Poseidon2Chip::<16, BabyBear>::from_poseidon2_config(
//         Poseidon2Config::default(),
//         ExecutionBus(0),
//         MemoryTester::new(MemoryBus(1)).chip(),
//     );
//
//     let outs: [[BabyBear; CHUNKS]; NUM_OPS] =
//         std::array::from_fn(|i| chip.hash(hashes[i].0, hashes[i].1));
//
//     let width = Poseidon2VmAir::<16, BabyBear>::direct_interaction_width();
//
//     let dummy_direct_cpu = DummyInteractionAir::new(width, true, POSEIDON2_DIRECT_BUS);
//
//     let mut dummy_direct_cpu_trace = RowMajorMatrix::new(
//         outs.iter()
//             .enumerate()
//             .flat_map(|(i, out)| {
//                 vec![BabyBear::one()]
//                     .into_iter()
//                     .chain(hashes[i].0)
//                     .chain(hashes[i].1)
//                     .chain(out.iter().cloned())
//             })
//             .collect::<Vec<_>>(),
//         width + 1,
//     );
//     dummy_direct_cpu_trace.values.extend(vec![
//         BabyBear::zero();
//         (width + 1) * (correct_height - NUM_OPS)
//     ]);
//
//     let chip_trace = chip.generate_trace();
//
//     // engine generation
//     let max_trace_height = chip_trace.height();
//     let engine = get_engine(max_trace_height);
//
//     // positive test
//     engine
//         .run_simple_test(
//             vec![&dummy_direct_cpu, &chip.air],
//             vec![dummy_direct_cpu_trace, chip_trace],
//             vec![vec![]; 2],
//         )
//         .expect("Verification failed");
// }
