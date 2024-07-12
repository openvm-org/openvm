use super::{make_io_cols, Poseidon2Chip};
use crate::cpu::trace::Instruction;
use crate::cpu::OpCode::{COMP_POS2, PERM_POS2};
use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS};
use crate::vm::config::{VmConfig, VmParamsConfig};
use crate::vm::VirtualMachine;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::config::{
    baby_bear_poseidon2::{engine_from_perm, random_perm},
    fri_params::fri_params_with_80_bits_of_security,
};
use afs_test_utils::engine::StarkEngine;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::create_seeded_rng;
use core::array::from_fn;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::RngCore;

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
                vm: VmParamsConfig {
                    field_arithmetic_enabled: true,
                    field_extension_enabled: false,
                    compress_poseidon2_enabled: true,
                    perm_poseidon2_enabled: true,
                    limb_bits: LIMB_BITS,
                    decomp: DECOMP,
                },
            },
            vec![],
            vec![],
        );

        let write_ops: [[WriteOps; 16]; $num_ops] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                if j < 8 {
                    WriteOps {
                        clk: 16 * i + j,
                        ad_s: $instructions[i].d,
                        address: $instructions[i].op_a + BabyBear::from_canonical_usize(j),
                        data: [$data[i][j]],
                    }
                } else {
                    WriteOps {
                        clk: 16 * i + j,
                        ad_s: $instructions[i].d,
                        address: $instructions[i].op_b + BabyBear::from_canonical_usize(j - 8),
                        data: [$data[i][j]],
                    }
                }
            })
        });

        write_ops.iter().flatten().for_each(|op| {
            vm.memory_chip
                .write_word(op.clk, op.ad_s, op.address, op.data);
        });

        let time_per = Poseidon2Chip::<16, BabyBear>::max_accesses_per_instruction(COMP_POS2);

        (0..$num_ops).for_each(|i| {
            let start_timestamp = 16 * $num_ops + (time_per * i);
            Poseidon2Chip::<16, BabyBear>::poseidon2_perm(
                &mut vm,
                start_timestamp,
                $instructions[i],
            );
        });

        let dummy_cpu_poseidon2 = DummyInteractionAir::new(
            Poseidon2Chip::<16, BabyBear>::interaction_width(),
            true,
            POSEIDON2_BUS,
        );
        let dummy_cpu_poseidon2_trace = RowMajorMatrix::new(
            {
                let mut vec: Vec<_> = (0..$num_ops)
                    .flat_map(|i| {
                        make_io_cols(16 * $num_ops + (time_per * i), $instructions[i]).flatten()
                    })
                    .collect();
                for _ in 0..(tot_ops - $num_ops)
                    * (Poseidon2Chip::<16, BabyBear>::interaction_width() + 1)
                {
                    vec.push(BabyBear::zero());
                }
                vec
            },
            Poseidon2Chip::<16, BabyBear>::interaction_width() + 1,
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

        let memory_chip_trace = vm.memory_chip.generate_trace(vm.range_checker.clone());
        let range_checker_trace = vm.range_checker.generate_trace();
        let poseidon2_trace = vm.poseidon2_chip.generate_trace();

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
    core::array::from_fn(|_| {
        let [a, b, c, d, e] =
            core::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 6) + 1));
        Instruction {
            opcode: if rng.next_u32() % 2 == 0 {
                PERM_POS2
            } else {
                COMP_POS2
            },
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
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
                &vm.range_checker.air,
                &vm.memory_chip.air,
                &vm.poseidon2_chip,
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
                    &vm.range_checker.air,
                    &vm.memory_chip.air,
                    &vm.poseidon2_chip,
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
