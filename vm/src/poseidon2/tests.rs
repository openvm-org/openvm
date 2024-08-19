use std::{cell::RefCell, collections::HashMap, rc::Rc};

use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use rand::{Rng, RngCore};

use afs_primitives::sub_chip::LocalTraceInstructions;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Engine, engine_from_perm, random_perm},
        fri_params::fri_params_with_80_bits_of_security,
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use poseidon2_air::poseidon2::Poseidon2Config;

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::MachineChip,
        testing::{ExecutionTester, MachineChipTester, MemoryTester},
    },
    cpu::{
        OpCode::{COMP_POS2, PERM_POS2},
        POSEIDON2_BUS,
        POSEIDON2_DIRECT_BUS, trace::Instruction,
    },
    memory::{offline_checker::MemoryChip, tree::Hasher},
};

use super::{CHUNK, Poseidon2Chip, Poseidon2VmAir, WIDTH};

const ADDRESS_BITS: usize = 29;

fn get_engine(max_trace_height: usize) -> BabyBearPoseidon2Engine {
    let max_log_degree = log2_strict_usize(max_trace_height);
    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    engine_from_perm(perm, max_log_degree, fri_params)
}

/// Create random instructions for the poseidon2 chip.
fn random_instructions(num_ops: usize) -> Vec<Instruction<BabyBear>> {
    let mut rng = create_seeded_rng();
    (0..num_ops)
        .map(|_| {
            let [a, b, c] = std::array::from_fn(|_| {
                BabyBear::from_wrapped_u32(rng.next_u32() % (1 << ADDRESS_BITS))
            });
            Instruction {
                opcode: if rng.gen_bool(0.5) {
                    PERM_POS2
                } else {
                    COMP_POS2
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

fn setup_test(
    num_ops: usize,
) -> (
    ExecutionTester,
    MemoryTester<BabyBear>,
    Poseidon2Chip<WIDTH, BabyBear>,
) {
    let elem_range = || 1..=100;
    let address_range = || 0usize..1 << ADDRESS_BITS;

    let execution_bus = ExecutionBus(0);
    let memory_bus = 1;
    let mut execution_tester = ExecutionTester::new(execution_bus, create_seeded_rng());
    let mut memory_tester = MemoryTester::new(memory_bus);
    let mut poseidon2_chip = Poseidon2Chip::from_poseidon2_config(
        Poseidon2Config::<16, _>::new_p3_baby_bear_16(),
        POSEIDON2_BUS,
        execution_bus,
        memory_tester.get_memory_chip(),
    );

    let mut rng = create_seeded_rng();

    for instruction in random_instructions(num_ops) {
        let opcode = instruction.opcode;
        let [a, b, c, d, e] = [
            instruction.op_a,
            instruction.op_b,
            instruction.op_c,
            instruction.d,
            instruction.e,
        ]
        .map(|elem| elem.as_canonical_u64() as usize);

        let dst = rng.gen_range(address_range());
        let lhs = rng.gen_range(address_range());
        let rhs = rng.gen_range(address_range());

        let data: [_; WIDTH] =
            std::array::from_fn(|_| BabyBear::from_canonical_usize(rng.gen_range(elem_range())));
        let hash = LocalTraceInstructions::generate_trace_row(&poseidon2_chip.air.inner, data)
            .io
            .output;

        memory_tester.install(d, a, [BabyBear::from_canonical_usize(dst)]);
        memory_tester.install(d, b, [BabyBear::from_canonical_usize(lhs)]);
        if opcode == COMP_POS2 {
            memory_tester.install(d, c, [BabyBear::from_canonical_usize(rhs)]);
        }

        match opcode {
            COMP_POS2 => {
                let data_left: [_; CHUNK] = std::array::from_fn(|i| data[i]);
                let data_right: [_; CHUNK] = std::array::from_fn(|i| data[CHUNK + i]);
                memory_tester.install(e, lhs, data_left);
                memory_tester.install(e, rhs, data_right);
            }
            PERM_POS2 => {
                memory_tester.install(e, lhs, data);
            }
            _ => panic!(),
        }

        execution_tester.execute(&mut poseidon2_chip, instruction);

        match opcode {
            COMP_POS2 => {
                let data_partial: [_; CHUNK] = std::array::from_fn(|i| hash[i]);
                memory_tester.expect(e, dst, data_partial);
            }
            PERM_POS2 => {
                memory_tester.expect(e, dst, hash);
            }
            _ => panic!(),
        }
        memory_tester.check();
    }
    (execution_tester, memory_tester, poseidon2_chip)
}

/// Checking that 50 random instructions pass.
#[test]
fn poseidon2_chip_random_50_test_new() {
    let (mut execution_tester, mut memory_tester, mut poseidon2_chip) = setup_test(50);
    MachineChipTester::default()
        .add(&mut execution_tester)
        .add(&mut memory_tester)
        .add(&mut poseidon2_chip)
        .engine_test(get_engine)
        .expect("Verification failed");
}

/// Negative test, pranking internal poseidon2 trace values.
#[test]
fn poseidon2_negative_test() {
    let mut rng = create_seeded_rng();
    let (mut execution_tester, mut memory_tester, mut poseidon2_chip) = setup_test(1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    for _ in 0..10 {
        let mut trace = poseidon2_chip.generate_trace();
        let width = rng.gen_range(24..trace.width() - 16);
        let height = rng.gen_range(0..trace.height());
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        trace.row_mut(height)[width] += rand;

        let test_result = MachineChipTester::default()
            .add(&mut execution_tester)
            .add(&mut memory_tester)
            .add_with_custom_trace(&mut poseidon2_chip, trace)
            .engine_test(get_engine);
        assert_eq!(
            test_result,
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
    }
}

/// Test that the direct bus interactions work.
#[test]
fn poseidon2_direct_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    const CHUNKS: usize = 8;
    let correct_height = NUM_OPS.next_power_of_two();
    let hashes: [([BabyBear; CHUNKS], [BabyBear; CHUNKS]); NUM_OPS] = std::array::from_fn(|_| {
        (
            std::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
            std::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
        )
    });
    let mut chip = Poseidon2Chip::<16, BabyBear>::from_poseidon2_config(
        Poseidon2Config::default(),
        POSEIDON2_BUS,
        ExecutionBus(0),
        Rc::new(RefCell::new(MemoryChip::new(0, 0, 0, 1, HashMap::new()))),
    );

    let outs: [[BabyBear; CHUNKS]; NUM_OPS] =
        std::array::from_fn(|i| chip.hash(hashes[i].0, hashes[i].1));

    let width = Poseidon2VmAir::<16, BabyBear>::direct_interaction_width();

    let dummy_direct_cpu = DummyInteractionAir::new(width, true, POSEIDON2_DIRECT_BUS);

    let mut dummy_direct_cpu_trace = RowMajorMatrix::new(
        outs.iter()
            .enumerate()
            .flat_map(|(i, out)| {
                vec![BabyBear::one()]
                    .into_iter()
                    .chain(hashes[i].0)
                    .chain(hashes[i].1)
                    .chain(out.iter().cloned())
            })
            .collect::<Vec<_>>(),
        width + 1,
    );
    dummy_direct_cpu_trace.values.extend(vec![
        BabyBear::zero();
        (width + 1) * (correct_height - NUM_OPS)
    ]);

    let chip_trace = chip.generate_trace();

    // engine generation
    let max_trace_height = chip_trace.height();
    let engine = get_engine(max_trace_height);

    // positive test
    engine
        .run_simple_test(
            vec![&dummy_direct_cpu, &chip.air],
            vec![dummy_direct_cpu_trace, chip_trace],
            vec![vec![]; 2],
        )
        .expect("Verification failed");
}
