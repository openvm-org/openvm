use super::{make_io_cols, Poseidon2Chip};
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
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::Poseidon2Config;
use rand::RngCore;
use crate::cpu::OpCode::COMPRESS_POSEIDON2;
use crate::cpu::trace::Instruction;

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

#[test]
fn poseidon2_chip_test() {
    let mut rng = create_seeded_rng();
    let a = BabyBear::from_canonical_u32(10);
    let b = BabyBear::from_canonical_u32(30);
    let c = BabyBear::from_canonical_u32(20);
    let d = BabyBear::from_canonical_u32(10);
    let e = BabyBear::from_canonical_u32(12);

    let same_instruction = Instruction {
        opcode: COMPRESS_POSEIDON2,
        op_a: a,
        op_b: b,
        op_c: c,
        d,
        e,
    };
    let num_instructions = 4;

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

    let chunk1 = (0..8)
        .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
        .collect::<Vec<_>>();
    let chunk2 = (8..16)
        .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
        .collect::<Vec<_>>();

    let write_ops: [WriteOps; 16] = core::array::from_fn(|i| {
        if i < 8 {
            WriteOps {
                clk: i,
                ad_s: d,
                address: a + BabyBear::from_canonical_usize(i),
                data: [chunk1[i]],
            }
        } else {
            WriteOps {
                clk: i,
                ad_s: d,
                address: b + BabyBear::from_canonical_usize(i - 8),
                data: [chunk2[i - 8]],
            }
        }
    });

    for op in &write_ops {
        vm.memory_chip
            .write_word(op.clk, op.ad_s, op.address, op.data);
    }

    let time_per = Poseidon2Chip::<16, BabyBear>::max_accesses_per_instruction(COMPRESS_POSEIDON2);

    for i in 0..num_instructions {
        let start_timestamp = write_ops.len() + (time_per * i);
        Poseidon2Chip::<16, BabyBear>::poseidon2_perm(&mut vm, start_timestamp, same_instruction);
    }
    let dummy_cpu_poseidon2 =
        DummyInteractionAir::new(Poseidon2Chip::<16, BabyBear>::interaction_width(), true, POSEIDON2_BUS);
    let dummy_cpu_poseidon2_trace = RowMajorMatrix::new(
        (0..num_instructions)
            .flat_map(|i| make_io_cols(write_ops.len() + (time_per * i), same_instruction).flatten())
            .collect(),
        Poseidon2Chip::<16, BabyBear>::interaction_width() + 1,
    );

    let dummy_cpu_memory = DummyInteractionAir::new(5, true, MEMORY_BUS);
    let dummy_cpu_memory_trace = RowMajorMatrix::new(
        write_ops
            .into_iter()
            .flat_map(|op| {
                let mut vec = op.flatten();
                vec.insert(0, BabyBear::one());
                vec
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

