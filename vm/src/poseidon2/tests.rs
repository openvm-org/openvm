use super::{Poseidon2Chip, Poseidon2Query};
use crate::cpu::MEMORY_BUS;
use crate::vm::config::{VmConfig, VmParamsConfig};
use crate::vm::VirtualMachine;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use poseidon2::poseidon2::Poseidon2Config;

const WORD_SIZE: usize = 1;
const LIMB_BITS: usize = 8;
const DECOMP: usize = 4;

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
        cmp: BabyBear::from_canonical_u32(1),
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

    for i in 0..10 {
        vm.memory_chip.write_word(
            i,
            d,
            a + BabyBear::from_canonical_usize(i),
            [BabyBear::from_canonical_usize(i); WORD_SIZE],
        );
    }

    for op in &ops {
        Poseidon2Chip::<16, BabyBear>::poseidon2_perm(&mut vm, op.clone());
    }
    let dummy_cpu = DummyInteractionAir::new(Poseidon2Query::<BabyBear>::width(), true, MEMORY_BUS);
    let dummy_cpu_trace = RowMajorMatrix::new(
        ops.into_iter()
            .flat_map(|op| op.to_io_cols().flatten())
            .collect(),
        Poseidon2Query::<BabyBear>::width() + 1,
    );

    let memory_chip_trace = vm.memory_chip.generate_trace(vm.range_checker.clone());
    let range_checker_trace = vm.range_checker.generate_trace();
    let poseidon2_trace = vm.poseidon2_chip.generate_trace();

    run_simple_test_no_pis(
        vec![
            &vm.range_checker.air,
            &vm.memory_chip.air,
            &vm.poseidon2_chip,
            &dummy_cpu,
        ],
        vec![
            range_checker_trace,
            memory_chip_trace,
            poseidon2_trace,
            dummy_cpu_trace,
        ],
    )
    .expect("Verification failed");
}
