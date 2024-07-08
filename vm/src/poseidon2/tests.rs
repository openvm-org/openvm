use super::{Poseidon2Chip, Poseidon2Query};
use crate::cpu::{MEMORY_BUS, POSEIDON2_BUS};
use crate::vm::config::{VmConfig, VmParamsConfig};
use crate::vm::VirtualMachine;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use poseidon2::poseidon2::Poseidon2Config;

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
    let a = BabyBear::from_canonical_u32(10);
    let b = BabyBear::from_canonical_u32(10);
    let c = BabyBear::from_canonical_u32(10);
    let d = BabyBear::from_canonical_u32(10);
    let e = BabyBear::from_canonical_u32(10);
    let ops = vec![Poseidon2Query {
        clk: 1,
        a,
        b,
        c,
        d,
        e,
        cmp: BabyBear::from_canonical_u32(0),
    }];

    let mut vm = VirtualMachine::<1, BabyBear>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                limb_bits: LIMB_BITS,
                decomp: DECOMP,
            },
        },
        vec![],
        Poseidon2Config::<16, BabyBear>::horizen_config(),
    );

    let chunk1 = (0..8)
        .map(BabyBear::from_canonical_usize)
        .collect::<Vec<_>>();
    let chunk2 = (8..16)
        .map(BabyBear::from_canonical_usize)
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

    for op in &ops {
        Poseidon2Chip::<16, BabyBear>::poseidon2_perm(&mut vm, op.clone());
    }
    let dummy_cpu_poseidon2 =
        DummyInteractionAir::new(Poseidon2Query::<BabyBear>::width(), true, POSEIDON2_BUS);
    let dummy_cpu_poseidon2_trace = RowMajorMatrix::new(
        ops.into_iter()
            .flat_map(|op| op.to_io_cols().flatten())
            .collect(),
        Poseidon2Query::<BabyBear>::width() + 1,
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

    run_simple_test_no_pis(
        vec![
            &vm.range_checker.air,
            &vm.memory_chip.air,
            &vm.poseidon2_chip,
            &dummy_cpu_memory,
            &dummy_cpu_poseidon2,
        ],
        vec![
            range_checker_trace,
            memory_chip_trace,
            poseidon2_trace,
            dummy_cpu_memory_trace,
            dummy_cpu_poseidon2_trace,
        ],
    )
    .expect("Verification failed");
}
