use std::any::type_name;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

const RATE: usize = 8;
// permutation width
const WIDTH: usize = 16; // rate + capacity
const DIGEST_WIDTH: usize = 8;

type Val = BabyBear;
type PackedVal = <Val as Field>::Packing;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, WIDTH, 7>;
type InstrPerm = Instrumented<Perm>;

// Generic over P: CryptographicPermutation<[F; WIDTH]>
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, DIGEST_WIDTH, WIDTH>;
type ValMmcs<P> =
    FieldMerkleTreeMmcs<PackedVal, <Val as Field>::Packing, Hash<P>, Compress<P>, DIGEST_WIDTH>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
pub type Challenger<P> = DuplexChallenger<Val, P, WIDTH>;
type Dft = Radix2DitParallel;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;

pub type BabyBearPoseidon2Config<P> = StarkConfig<Pcs<P>, Challenge, Challenger<P>>;

pub type StarkConfigPoseidon2 = BabyBearPoseidon2Config<Perm>;

use rand::{rngs::StdRng, SeedableRng};

use super::{
    instrument::{InstrumentCounter, Instrumented},
    FriParameters,
};

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm, pcs_log_degree: usize) -> StarkConfigPoseidon2 {
    // target 100 bits of security, with conjectures:
    let fri_params = FriParameters {
        log_blowup: 4,
        num_queries: 55,
        proof_of_work_bits: 0,
    };
    config_from_perm(perm, pcs_log_degree, fri_params)
}

pub fn config_from_perm<P>(
    perm: &P,
    pcs_log_degree: usize,
    fri_params: FriParameters,
) -> BabyBearPoseidon2Config<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let fri_config = FriConfig {
        log_blowup: fri_params.log_blowup,
        num_queries: fri_params.num_queries,
        proof_of_work_bits: fri_params.proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(pcs_log_degree, dft, val_mmcs, fri_config);
    BabyBearPoseidon2Config::new(pcs)
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

pub fn random_instrumented_perm() -> InstrPerm {
    let perm = random_perm();
    Instrumented::new(perm)
}

/// Logs hash count statistics to stdout and returns as struct.
/// Count of 1 corresponds to a Poseidon2 permutation with rate RATE that outputs OUT field elements
#[allow(dead_code)]
pub fn print_hash_counts(hash_counter: &InstrumentCounter, compress_counter: &InstrumentCounter) {
    let hash_counter = hash_counter.lock().unwrap();
    let mut hash_count = 0;
    hash_counter.iter().for_each(|(name, lens)| {
        if name == type_name::<(Val, [Val; DIGEST_WIDTH])>() {
            let count = lens
                .iter()
                .fold(0, |count, len| count + (len + RATE - 1) / RATE);
            println!("Hash: {name}, Count: {count}");
            hash_count += count;
        } else {
            panic!("Hash type not yet supported: {}", name);
        }
    });
    drop(hash_counter);
    let compress_counter = compress_counter.lock().unwrap();
    let mut compress_count = 0;
    compress_counter.iter().for_each(|(name, lens)| {
        if name == type_name::<[Val; DIGEST_WIDTH]>() {
            let count = lens.iter().fold(0, |count, len| {
                // len should always be N=2 for TruncatedPermutation
                count + (DIGEST_WIDTH * len + WIDTH - 1) / WIDTH
            });
            println!("Compress: {name}, Count: {count}");
            compress_count += count;
        } else {
            panic!("Compress type not yet supported: {}", name);
        }
    });
    let total_count = hash_count + compress_count;
    println!("Total Count: {total_count}");
}

pub fn get_perm_count(perm: &InstrPerm) -> usize {
    let counter = perm.input_lens_by_type.lock().unwrap();
    counter.iter().fold(0, |total, (name, lens)| {
        if name == type_name::<[Val; WIDTH]>() {
            let count: usize = lens.iter().sum();
            println!("Permutation: {name}, Count: {count}");
            total + count
        } else {
            panic!("Permutation type not yet supported: {}", name);
        }
    })
}
