use core::array::from_fn;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::fri_params_with_80_bits_of_security,
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::Poseidon2Config;
use rand::{Rng, RngCore};

use super::{Poseidon2Chip, Poseidon2VmAir};
use crate::{
    cpu::{
        trace::Instruction,
        OpCode::{COMP_POS2, PERM_POS2},
        MEMORY_BUS, POSEIDON2_BUS, POSEIDON2_DIRECT_BUS,
    },
    memory::tree::Hasher,
    vm::{
        config::{VmConfig, DEFAULT_MAX_SEGMENT_LEN},
        VirtualMachine,
    },
};

const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 16;
const DECOMP: usize = 8;

struct WriteOps {
    clk: usize,
    ad_s: BabyBear,
    address: BabyBear,
    data: [BabyBear; WORD_SIZE],
}

impl WriteOps {
    fn flatten(&self) -> Vec<BabyBear> {
        vec![
            BabyBear::from_canonical_usize(self.clk),
            BabyBear::from_bool(true),
            self.ad_s,
            self.address,
            self.data[0],
        ]
    }
}

macro_rules! run_perm_ops {
    ($instructions:expr, $num_ops:expr, $data:expr) => {{
        let tot_ops: usize = ($num_ops as usize).next_power_of_two();

        // default VM with poseidon2 enabled
        let mut vm = VirtualMachine::<1, BabyBear>::new(
            VmConfig {
                field_arithmetic_enabled: true,
                field_extension_enabled: false,
                compress_poseidon2_enabled: true,
                perm_poseidon2_enabled: true,
                limb_bits: LIMB_BITS,
                decomp: DECOMP,
                num_public_values: 4,
                max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
                ..Default::default()
            },
            vec![],
            vec![],
        );
        let mut segment = &mut vm.segments[0];

        let write_ops: [[WriteOps; 16]; $num_ops] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                if j < 8 {
                    WriteOps {
                        clk: 16 * i + j,
                        ad_s: $instructions[i].e,
                        address: $instructions[i].op_b + BabyBear::from_canonical_usize(j),
                        data: [$data[i][j]],
                    }
                } else {
                    WriteOps {
                        clk: 16 * i + j,
                        ad_s: $instructions[i].e,
                        address: $instructions[i].op_c + BabyBear::from_canonical_usize(j - 8),
                        data: [$data[i][j]],
                    }
                }
            })
        });

        write_ops.iter().flatten().for_each(|op| {
            segment
                .memory_chip
                .write_word(op.clk, op.ad_s, op.address, op.data);
        });

        let time_per = Poseidon2Chip::<16, BabyBear>::max_accesses_per_instruction(COMP_POS2);

        (0..$num_ops).for_each(|i| {
            let start_timestamp = 16 * $num_ops + (time_per * i);
            Poseidon2Chip::<16, BabyBear>::poseidon2_perm(
                &mut segment,
                start_timestamp,
                $instructions[i].clone(),
            );
        });

        let dummy_cpu_poseidon2 = DummyInteractionAir::new(
            Poseidon2VmAir::<16, BabyBear>::opcode_interaction_width(),
            true,
            POSEIDON2_BUS,
        );
        let dummy_cpu_poseidon2_trace = RowMajorMatrix::new(
            {
                let mut vec: Vec<BabyBear> = (0..$num_ops)
                    .flat_map(|i| {
                        Poseidon2VmAir::<16, BabyBear>::make_io_cols(
                            16 * $num_ops + (time_per * i),
                            $instructions[i].clone(),
                        )
                        .flatten()
                        .iter()
                        .enumerate()
                        .filter(|&(index, _)| index != 1)
                        .map(|(_, value)| *value)
                        .collect::<Vec<BabyBear>>()
                    })
                    .collect();
                for _ in 0..(tot_ops - $num_ops)
                    * (Poseidon2VmAir::<16, BabyBear>::opcode_interaction_width() + 1)
                {
                    vec.push(BabyBear::zero());
                }
                vec
            },
            Poseidon2VmAir::<16, BabyBear>::opcode_interaction_width() + 1,
        );

        let dummy_cpu_memory = DummyInteractionAir::new(5, true, MEMORY_BUS);
        let dummy_cpu_memory_trace = RowMajorMatrix::new(
            {
                let mut vec: Vec<_> = write_ops
                    .iter()
                    .flat_map(|ops| {
                        ops.iter().flat_map(|op| {
                            let mut vec = op.flatten();
                            vec.insert(0, BabyBear::one());
                            vec
                        })
                    })
                    .collect();
                for _ in 0..(16 * (tot_ops - $num_ops)) * (5 + 1) {
                    vec.push(BabyBear::zero());
                }
                vec
            },
            5 + 1,
        );

        let memory_chip_trace = segment
            .memory_chip
            .generate_trace(segment.range_checker.clone());
        let range_checker_trace = segment.range_checker.generate_trace();
        let poseidon2_trace = segment.poseidon2_chip.generate_trace();

        let traces = vec![
            range_checker_trace,
            memory_chip_trace,
            poseidon2_trace,
            dummy_cpu_memory_trace,
            dummy_cpu_poseidon2_trace,
        ];

        // engine generation
        let max_trace_height = traces.iter().map(|trace| trace.height()).max().unwrap();
        let max_log_degree = log2_strict_usize(max_trace_height);
        let perm = random_perm();
        let fri_params = fri_params_with_80_bits_of_security()[1];
        let engine = engine_from_perm(perm, max_log_degree, fri_params);

        (vm, engine, dummy_cpu_memory, dummy_cpu_poseidon2, traces)
    }};
}

/// Create random instructions for the poseidon2 chip.
fn random_instructions<const NUM_OPS: usize>() -> [Instruction<BabyBear>; NUM_OPS] {
    let mut rng = create_seeded_rng();
    from_fn(|_| {
        let [a, b, c, e] = from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 6) + 1));
        Instruction {
            opcode: if rng.next_u32() % 2 == 0 {
                PERM_POS2
            } else {
                COMP_POS2
            },
            op_a: a,
            op_b: b,
            op_c: c,
            d: BabyBear::zero(),
            e,
            debug: String::new(),
        }
    })
}

/// Checking that 50 random instructions pass.
#[test]
fn poseidon2_chip_random_50_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    let instructions: [Instruction<BabyBear>; NUM_OPS] = random_instructions::<NUM_OPS>();
    let data: [[BabyBear; 16]; NUM_OPS] =
        from_fn(|_| from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))));

    let (vm, engine, dummy_cpu_memory, dummy_cpu_poseidon2, traces) =
        run_perm_ops!(instructions, NUM_OPS, data);

    // positive test
    engine
        .run_simple_test(
            vec![
                &vm.segments[0].range_checker.air,
                &vm.segments[0].memory_chip.air,
                &vm.segments[0].poseidon2_chip.air,
                &dummy_cpu_memory,
                &dummy_cpu_poseidon2,
            ],
            traces,
            vec![vec![]; 5],
        )
        .expect("Verification failed");
}

/// Negative test, pranking internal poseidon2 trace values.
#[test]
fn poseidon2_negative_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 1;
    let instructions: [Instruction<BabyBear>; NUM_OPS] = random_instructions::<NUM_OPS>();
    let data: [[BabyBear; 16]; NUM_OPS] =
        from_fn(|_| from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))));

    let (vm, engine, dummy_cpu_memory, dummy_cpu_poseidon2, mut traces) =
        run_perm_ops!(instructions, NUM_OPS, data);
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
                    &vm.segments[0].memory_chip.air,
                    &vm.segments[0].poseidon2_chip.air,
                    &dummy_cpu_memory,
                    &dummy_cpu_poseidon2,
                ],
                traces.clone(),
                vec![vec![]; 5],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        traces[poseidon2_trace_index].row_mut(height)[width] -= rand;
    }
}

/// Test that the direct bus interactions work.
#[test]
fn poseidon2_direct_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    const CHUNKS: usize = 8;
    let correct_height = NUM_OPS.next_power_of_two();
    let hashes: [([BabyBear; CHUNKS], [BabyBear; CHUNKS]); NUM_OPS] = from_fn(|_| {
        (
            from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
            from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))),
        )
    });
    let mut chip = Poseidon2Chip::<16, BabyBear>::from_poseidon2_config(
        Poseidon2Config::default(),
        POSEIDON2_BUS,
    );

    let outs: [[BabyBear; CHUNKS]; NUM_OPS] = from_fn(|i| chip.hash(hashes[i].0, hashes[i].1));

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
    let max_log_degree = log2_strict_usize(max_trace_height);
    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    // positive test
    engine
        .run_simple_test(
            vec![&dummy_direct_cpu, &chip.air],
            vec![dummy_direct_cpu_trace, chip_trace],
            vec![vec![]; 2],
        )
        .expect("Verification failed");
}
