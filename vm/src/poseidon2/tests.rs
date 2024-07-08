use std::{iter, sync::Arc};

use super::{Poseidon2Chip, Poseidon2Query};
use crate::cpu::{MEMORY_BUS, RANGE_CHECKER_BUS};
use crate::memory::offline_checker::MemoryChip;
use crate::vm::config::{VmConfig, VmParamsConfig};
use crate::vm::VirtualMachine;
use afs_chips::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

const WORD_SIZE: usize = 1;
const ADDR_SPACE_LIMB_BITS: usize = 8;
const POINTER_LIMB_BITS: usize = 8;
const CLK_LIMB_BITS: usize = 8;
const DECOMP: usize = 4;
const RANGE_MAX: u32 = 1 << DECOMP;

const TRACE_DEGREE: usize = 16;

#[test]
fn poseidon2_chip_test() {
    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, RANGE_MAX));
    let mut chip: MemoryChip<WORD_SIZE, BabyBear> = MemoryChip::new(
        ADDR_SPACE_LIMB_BITS,
        POINTER_LIMB_BITS,
        CLK_LIMB_BITS,
        DECOMP,
    );
    let requester = DummyInteractionAir::new(2 + chip.air.mem_width(), true, MEMORY_BUS);

    let ops = vec![Poseidon2Query {
        clk: 1,
        a: BabyBear::from_canonical_u32(10),
        b: BabyBear::from_canonical_u32(10),
        c: BabyBear::from_canonical_u32(10),
        d: BabyBear::from_canonical_u32(10),
        e: BabyBear::from_canonical_u32(10),
        cmp: BabyBear::from_canonical_u32(1),
    }];

    let mut vm = VirtualMachine::<1, BabyBear>::new(
        VmConfig {
            vm: VmParamsConfig {
                field_arithmetic_enabled: true,
                limb_bits: CLK_LIMB_BITS,
                decomp: DECOMP,
            },
        },
        vec![],
    );

    for op in ops {
        Poseidon2Chip::<16, BabyBear>::poseidon2_perm(&mut vm, op);
    }
}
