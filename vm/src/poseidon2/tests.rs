use core::array::from_fn;
use std::collections::HashMap;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm, BabyBearPoseidon2Engine},
        fri_params::fri_params_with_80_bits_of_security,
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use rand::{Rng, RngCore};

use super::{Poseidon2Chip, Poseidon2VmAir, CHUNK, WIDTH};
use crate::{
    cpu::{
        trace::Instruction,
        OpCode::{COMP_POS2, PERM_POS2},
        POSEIDON2_BUS,
    },
    poseidon2::Poseidon2VmCols,
    program::Program,
    vm::{
        config::{MemoryConfig, VmConfig, DEFAULT_MAX_SEGMENT_LEN},
        VirtualMachine,
    },
};

const NUM_WORDS: usize = 8;
const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 24;
const DECOMP: usize = 8;

#[derive(Debug)]
struct WriteOps {
    ad_s: BabyBear,
    address: BabyBear,
    data: [BabyBear; WORD_SIZE],
}

#[test]
fn p2_flatten_fromslice_roundtrip() {
    const NUM_WORDS: usize = 8;
    const WORD_SIZE: usize = 1;

    let program = Program {
        instructions: vec![],
        debug_infos: vec![],
    };

    let mut vm = VirtualMachine::<NUM_WORDS, WORD_SIZE, BabyBear>::new(
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: false,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
            memory_config: MemoryConfig::new(LIMB_BITS, LIMB_BITS, LIMB_BITS, DECOMP),
            num_public_values: 4,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            ..Default::default()
        },
        program,
        vec![],
    );
    let segment = &mut vm.segments[0];

    // let p2_air = Poseidon2VmAir::<WIDTH, WORD_SIZE, BabyBear>::new(Poseidon2Air::new());
    let num_cols =
        Poseidon2VmCols::<WIDTH, WORD_SIZE, BabyBear>::width(&segment.poseidon2_chip.air);
    let all_cols = (0..num_cols)
        .map(BabyBear::from_canonical_usize)
        .collect::<Vec<BabyBear>>();

    let cols_numbered = Poseidon2VmCols::<WIDTH, WORD_SIZE, BabyBear>::from_slice(
        &all_cols,
        &segment.poseidon2_chip.air,
    );
    let flattened = cols_numbered.flatten();

    assert_eq!(flattened, all_cols);
}

fn run_perm_ops(
    instructions: Vec<Instruction<BabyBear>>,
    data: Vec<[BabyBear; WIDTH]>,
) -> (
    VirtualMachine<NUM_WORDS, WORD_SIZE, BabyBear>,
    BabyBearPoseidon2Engine,
    DummyInteractionAir,
    Vec<RowMajorMatrix<BabyBear>>,
) {
    const NUM_WORDS: usize = 8;
    const WORD_SIZE: usize = 1;

    let num_ops = instructions.len();
    assert_eq!(data.len(), num_ops);
    let mut rng = create_seeded_rng();

    let program = Program {
        instructions: vec![],
        debug_infos: vec![],
    };

    // default VM with poseidon2 enabled
    let mut vm = VirtualMachine::<NUM_WORDS, WORD_SIZE, BabyBear>::new(
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: false,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
            memory_config: MemoryConfig::new(LIMB_BITS, LIMB_BITS, LIMB_BITS, DECOMP),
            num_public_values: 4,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            ..Default::default()
        },
        program,
        vec![],
    );
    let segment = &mut vm.segments[0];

    let emb = |x| {
        let mut word = [BabyBear::zero(); WORD_SIZE];
        word[0] = x;
        word
    };

    let mut write_ops: Vec<WriteOps> = Vec::new();

    for i in 0..num_ops {
        // CAUTION: we assume there will be no collisions between lhs..lhs+CHUNK and rhs..rhs+CHUNK
        const ADDR_MAX: u32 = (1 << LIMB_BITS) - WIDTH as u32;
        let dst = BabyBear::from_wrapped_u32(rng.next_u32() % ADDR_MAX);
        let lhs = BabyBear::from_wrapped_u32(rng.next_u32() % (ADDR_MAX / 2));
        let rhs = lhs + BabyBear::from_wrapped_u32(rng.next_u32() % (ADDR_MAX / 2));
        assert!((lhs.as_canonical_u32() + CHUNK as u32) < rhs.as_canonical_u32());

        let instr = &instructions[i];
        write_ops.push(WriteOps {
            ad_s: instr.d,
            address: instr.op_a,
            data: emb(dst),
        });
        write_ops.push(WriteOps {
            ad_s: instr.d,
            address: instr.op_b,
            data: emb(lhs),
        });
        if instr.opcode == COMP_POS2 {
            write_ops.push(WriteOps {
                ad_s: instr.d,
                address: instr.op_c,
                data: emb(rhs),
            });
        }

        for j in 0..WIDTH {
            write_ops.push(if j < CHUNK {
                WriteOps {
                    ad_s: instr.e,
                    address: lhs + BabyBear::from_canonical_usize(j),
                    data: emb(data[i][j]),
                }
            } else {
                let address = if instr.opcode == COMP_POS2 {
                    rhs + BabyBear::from_canonical_usize(j - CHUNK)
                } else {
                    lhs + BabyBear::from_canonical_usize(j)
                };
                WriteOps {
                    ad_s: instr.e,
                    address,
                    data: emb(data[i][j]),
                }
            });
        }
    }

    let mut initial_memory = HashMap::new();
    write_ops.iter().for_each(|op| {
        initial_memory.insert((op.ad_s, op.address), op.data);
    });

    for ((addr_space, pointer), data) in initial_memory {
        segment
            .memory_manager
            .borrow_mut()
            .unsafe_write_word(addr_space, pointer, data);

        segment
            .memory_manager
            .borrow_mut()
            .interface_chip
            .touch_address(addr_space, pointer, data);
    }

    let time_per =
        Poseidon2Chip::<16, 16, WORD_SIZE, BabyBear>::max_accesses_per_instruction(PERM_POS2);

    (0..num_ops).for_each(|i| {
        segment
            .poseidon2_chip
            .calculate(instructions[i].clone(), false);
    });

    let mut timestamp = 1;
    // dummy air to send poseidon2 opcodes (pretending to be like cpu)
    let dummy_cpu_poseidon2 = DummyInteractionAir::new(
        Poseidon2VmAir::<16, WORD_SIZE, BabyBear>::opcode_interaction_width(),
        true,
        POSEIDON2_BUS,
    );
    let width = Poseidon2VmAir::<16, WORD_SIZE, BabyBear>::opcode_interaction_width() + 1;
    let dummy_cpu_poseidon2_trace = RowMajorMatrix::new(
        {
            let height = num_ops.next_power_of_two();
            let mut vec: Vec<BabyBear> = (0..num_ops)
                .flat_map(|i| {
                    let mut row = Poseidon2VmAir::<16, WORD_SIZE, BabyBear>::make_io_cols(
                        BabyBear::from_canonical_usize(timestamp),
                        instructions[i].clone(),
                    )
                    .flatten();
                    row.remove(1); // remove is_direct
                    timestamp += time_per;

                    row
                })
                .collect();
            vec.resize(width * height, BabyBear::zero());
            vec
        },
        width,
    );

    let memory_interface_trace = segment
        .memory_manager
        .borrow()
        .generate_memory_interface_trace();
    let poseidon2_trace = segment.poseidon2_chip.generate_trace();
    let range_checker_trace = segment.range_checker.generate_trace();

    let traces = vec![
        range_checker_trace,
        memory_interface_trace,
        poseidon2_trace,
        dummy_cpu_poseidon2_trace,
    ];

    // engine generation
    let max_trace_height = traces.iter().map(|trace| trace.height()).max().unwrap();
    let max_log_degree = log2_strict_usize(max_trace_height);
    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    (vm, engine, dummy_cpu_poseidon2, traces)
}

/// Create random instructions for the poseidon2 chip.
fn random_instructions(num_ops: usize) -> Vec<Instruction<BabyBear>> {
    let mut rng = create_seeded_rng();
    (0..num_ops)
        .map(|_| {
            let [a, b, c] =
                from_fn(|_| BabyBear::from_wrapped_u32(rng.next_u32() % (1 << LIMB_BITS)));
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

/// Checking that 50 random instructions pass.
#[test]
fn poseidon2_chip_random_50_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    let instructions = random_instructions(NUM_OPS);
    let data = (0..NUM_OPS)
        .map(|_| from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))))
        .collect_vec();

    let (vm, engine, dummy_cpu_poseidon2, traces) = run_perm_ops(instructions.to_vec(), data);

    // positive test
    engine
        .run_simple_test(
            vec![
                &vm.segments[0].range_checker.air,
                &vm.segments[0].memory_manager.borrow().get_audit_air(),
                &vm.segments[0].poseidon2_chip.air,
                &dummy_cpu_poseidon2,
            ],
            traces,
            vec![vec![]; 4],
        )
        .expect("Verification failed");
}

/// Negative test, pranking internal poseidon2 trace values.
#[test]
fn poseidon2_negative_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    let instructions = random_instructions(NUM_OPS);
    let data = (0..NUM_OPS)
        .map(|_| from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))))
        .collect_vec();

    let (vm, engine, dummy_cpu_poseidon2, mut traces) = run_perm_ops(instructions, data);
    let poseidon2_trace_index = 2;

    // negative test
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    for _ in 0..10 {
        let width = rng.gen_range(24..traces[poseidon2_trace_index].width() - 16);
        let height = rng.gen_range(0..traces[poseidon2_trace_index].height());
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        traces[poseidon2_trace_index].row_mut(height)[width] += rand;
        assert_eq!(
            engine.run_simple_test(
                vec![
                    &vm.segments[0].range_checker.air,
                    &vm.segments[0].memory_manager.borrow().get_audit_air(),
                    &vm.segments[0].poseidon2_chip.air,
                    &dummy_cpu_poseidon2,
                ],
                traces.clone(),
                vec![vec![]; 4],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        traces[poseidon2_trace_index].row_mut(height)[width] -= rand;
    }
}

// /// Test that the direct bus interactions work.
// #[test]
// fn poseidon2_direct_test() {
//     let mut rng = create_seeded_rng();
//     const NUM_OPS: usize = 50;
//     const CHUNKS: usize = 8;
//     let correct_height = NUM_OPS.next_power_of_two();
//     let hashes: [([BabyBear; CHUNKS], [BabyBear; CHUNKS]); NUM_OPS] = from_fn(|_| {
//         (
//             from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
//             from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
//         )
//     });
//     let mut chip = Poseidon2Chip::<16, 16, WORD_SIZE, BabyBear>::from_poseidon2_config(
//         Poseidon2Config::default(),
//         MemoryConfig::new(LIMB_BITS, LIMB_BITS, LIMB_BITS, DECOMP),
//         POSEIDON2_BUS,
//     );

//     let outs: [[BabyBear; CHUNKS]; NUM_OPS] = from_fn(|i| chip.hash(hashes[i].0, hashes[i].1));

//     let width = Poseidon2VmAir::<16, WORD_SIZE, BabyBear>::direct_interaction_width();

//     let dummy_direct_cpu = DummyInteractionAir::new(width, true, POSEIDON2_DIRECT_BUS);

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

//     let chip_trace = chip.generate_trace();

//     // engine generation
//     let max_trace_height = chip_trace.height();
//     let max_log_degree = log2_strict_usize(max_trace_height);
//     let perm = random_perm();
//     let fri_params = fri_params_with_80_bits_of_security()[1];
//     let engine = engine_from_perm(perm, max_log_degree, fri_params);

//     // positive test
//     engine
//         .run_simple_test(
//             vec![&dummy_direct_cpu, &chip.air],
//             vec![dummy_direct_cpu_trace, chip_trace],
//             vec![vec![]; 2],
//         )
//         .expect("Verification failed");
// }
