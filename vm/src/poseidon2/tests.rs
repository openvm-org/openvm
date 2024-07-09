use super::columns::Poseidon2ChipIoCols;
use super::{make_io_cols, Poseidon2Chip};
use crate::cpu::trace::Instruction;
use crate::cpu::OpCode::{COMPRESS_POSEIDON2, PERM_POSEIDON2};
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
use poseidon2_air::poseidon2::Poseidon2Config;
use rand::RngCore;

const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 8;
const DECOMP: usize = 4;

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

impl Poseidon2ChipIoCols<BabyBear> {
    fn random() -> Self {
        let mut rng = create_seeded_rng();
        let [clk, a, b, c, d, e] =
            core::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)));
        Poseidon2ChipIoCols {
            clk,
            is_alloc: BabyBear::from_bool(true),
            a,
            b,
            c,
            d,
            e,
            cmp: BabyBear::from_canonical_u32(rng.next_u32() % 2),
        }
    }
}

#[test]
fn poseidon2_chip_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 4;

    let instructions: [Instruction<BabyBear>; NUM_OPS] = core::array::from_fn(|_| {
        let [a, b, c, d, e] =
            core::array::from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 2) + 2));
        Instruction {
            opcode: if rng.next_u32() % 2 == 0 {
                COMPRESS_POSEIDON2
            } else {
                PERM_POSEIDON2
            },
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
        }
    });

    let mut vm = VirtualMachine::<1, BabyBear>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                field_extension_enabled: false,
                limb_bits: LIMB_BITS,
                decomp: DECOMP,
            },
        },
        vec![],
        Poseidon2Config::<16, BabyBear>::horizen_config(),
    );

    let chunk1: [[BabyBear; 8]; NUM_OPS] =
        from_fn(|_| from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))));

    let chunk2: [[BabyBear; 8]; NUM_OPS] =
        from_fn(|_| from_fn(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30))));

    let write_ops: [[WriteOps; 16]; NUM_OPS] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            if j < 8 {
                WriteOps {
                    clk: 16 * i + j,
                    ad_s: instructions[i].d,
                    address: instructions[i].op_a + BabyBear::from_canonical_usize(j),
                    data: [chunk1[i][j]],
                }
            } else {
                WriteOps {
                    clk: 16 * i + j,
                    ad_s: instructions[i].d,
                    address: instructions[i].op_b + BabyBear::from_canonical_usize(j - 8),
                    data: [chunk2[i][j - 8]],
                }
            }
        })
    });

    for i in 0..NUM_OPS {
        for j in 0..16 {
            vm.memory_chip.write_word(
                write_ops[i][j].clk,
                write_ops[i][j].ad_s,
                write_ops[i][j].address,
                write_ops[i][j].data,
            );
        }
    }

    let time_per = Poseidon2Chip::<16, BabyBear>::max_accesses_per_instruction(COMPRESS_POSEIDON2);

    for i in 0..NUM_OPS {
        let start_timestamp = 16 * NUM_OPS + (time_per * i);
        Poseidon2Chip::<16, BabyBear>::poseidon2_perm(&mut vm, start_timestamp, instructions[i]);
    }

    let dummy_cpu_poseidon2 = DummyInteractionAir::new(
        Poseidon2Chip::<16, BabyBear>::interaction_width(),
        true,
        POSEIDON2_BUS,
    );
    let dummy_cpu_poseidon2_trace = RowMajorMatrix::new(
        (0..NUM_OPS)
            .flat_map(|i| make_io_cols(16 * NUM_OPS + (time_per * i), instructions[i]).flatten())
            .collect(),
        Poseidon2Chip::<16, BabyBear>::interaction_width() + 1,
    );

    let dummy_cpu_memory = DummyInteractionAir::new(5, true, MEMORY_BUS);
    let dummy_cpu_memory_trace = RowMajorMatrix::new(
        write_ops
            .iter()
            .flat_map(|ops| {
                ops.iter().flat_map(|op| {
                    let mut vec = op.flatten();
                    vec.insert(0, BabyBear::one());
                    vec
                })
            })
            .collect(),
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
    // let max_log_degree = 6;
    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

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
