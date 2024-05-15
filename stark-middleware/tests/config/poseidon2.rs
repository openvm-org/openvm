use std::sync::Arc;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

type Val = BabyBear;
type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
pub type Challenger = DuplexChallenger<Val, Perm, 16>;
type Dft = Radix2DitParallel;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
pub type StarkConfigPoseidon2 = StarkConfig<Pcs, Challenge, Challenger>;

type InstrHash = Instrumented<MyHash>;
type InstrCompress = Instrumented<MyCompress>;
type InstrValMmcs = FieldMerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    InstrHash,
    InstrCompress,
    8,
>;
type InstrChallengeMmcs = ExtensionMmcs<Val, Challenge, InstrValMmcs>;
type InstrPcs = TwoAdicFriPcs<Val, Dft, InstrValMmcs, InstrChallengeMmcs>;
pub type InstrumentedStarkConfigPoseidon2 = StarkConfig<InstrPcs, Challenge, Challenger>;

use rand::{rngs::StdRng, SeedableRng};

use super::instrument::{InstrumentCounter, Instrumented};

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm, pcs_log_degree: usize) -> StarkConfigPoseidon2 {
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(pcs_log_degree, dft, val_mmcs, fri_config);
    StarkConfigPoseidon2::new(pcs)
}

pub fn instrumented_config(
    perm: &Perm,
    pcs_log_degree: usize,
    log_blowup: usize,
    num_queries: usize,
    proof_of_work_bits: usize,
) -> (
    InstrumentedStarkConfigPoseidon2,
    InstrumentCounter,
    InstrumentCounter,
) {
    let hash = Instrumented::new(MyHash::new(perm.clone()));
    let hash_counter = Arc::clone(&hash.input_lens_by_type);
    let compress = Instrumented::new(MyCompress::new(perm.clone()));
    let compress_counter = Arc::clone(&compress.input_lens_by_type);
    let val_mmcs = InstrValMmcs::new(hash, compress);
    let challenge_mmcs = InstrChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let fri_config = FriConfig {
        log_blowup,
        num_queries,
        proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = InstrPcs::new(pcs_log_degree, dft, val_mmcs, fri_config);
    (
        InstrumentedStarkConfigPoseidon2::new(pcs),
        hash_counter,
        compress_counter,
    )
}

pub fn random_perm() -> Perm {
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);
    Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        &mut rng,
    )
}
