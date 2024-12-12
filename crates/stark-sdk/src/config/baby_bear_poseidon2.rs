use std::any::type_name;

use openvm_stark_backend::{
    config::StarkConfig,
    interaction::stark_log_up::StarkLogUpPhase,
    p3_challenger::DuplexChallenger,
    p3_commit::ExtensionMmcs,
    p3_field::{extension::BinomialExtensionField, AbstractField, Field},
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_dft::Radix2DitParallel;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, TruncatedPermutation};
use rand::{rngs::StdRng, SeedableRng};
use zkhash::{
    ark_ff::PrimeField as _, fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::poseidon2_instance_babybear::RC16,
};

use super::{
    instrument::{HashStatistics, InstrumentCounter, Instrumented, StarkHashStatistics},
    FriParameters,
};
use crate::{
    assert_sc_compatible_with_serde,
    engine::{StarkEngine, StarkEngineWithHashInstrumentation, StarkFriEngine},
};

const RATE: usize = 8;
// permutation width
const WIDTH: usize = 16; // rate + capacity
const DIGEST_WIDTH: usize = 8;

type Val = BabyBear;
type PackedVal = <Val as Field>::Packing;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<WIDTH>;
type InstrPerm = Instrumented<Perm>;

// Generic over P: CryptographicPermutation<[F; WIDTH]>
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, DIGEST_WIDTH, WIDTH>;
type ValMmcs<P> =
    MerkleTreeMmcs<PackedVal, <Val as Field>::Packing, Hash<P>, Compress<P>, DIGEST_WIDTH>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
pub type Challenger<P> = DuplexChallenger<Val, P, WIDTH, RATE>;
type Dft = Radix2DitParallel<Val>;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;
type RapPhase<P> = StarkLogUpPhase<Val, Challenge, Challenger<P>>;

pub type BabyBearPermutationConfig<P> = StarkConfig<Pcs<P>, RapPhase<P>, Challenge, Challenger<P>>;
pub type BabyBearPoseidon2Config = BabyBearPermutationConfig<Perm>;
pub type BabyBearPoseidon2Engine = BabyBearPermutationEngine<Perm>;

assert_sc_compatible_with_serde!(BabyBearPoseidon2Config);

pub struct BabyBearPermutationEngine<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    pub fri_params: FriParameters,
    pub config: BabyBearPermutationConfig<P>,
    pub perm: P,
}

impl<P> StarkEngine<BabyBearPermutationConfig<P>> for BabyBearPermutationEngine<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    fn config(&self) -> &BabyBearPermutationConfig<P> {
        &self.config
    }

    fn new_challenger(&self) -> Challenger<P> {
        Challenger::new(self.perm.clone())
    }
}

impl<P> StarkEngineWithHashInstrumentation<BabyBearPermutationConfig<Instrumented<P>>>
    for BabyBearPermutationEngine<Instrumented<P>>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    fn clear_instruments(&mut self) {
        self.perm.input_lens_by_type.lock().unwrap().clear();
    }
    fn stark_hash_statistics<T>(&self, custom: T) -> StarkHashStatistics<T> {
        let counter = self.perm.input_lens_by_type.lock().unwrap();
        let permutations = counter.iter().fold(0, |total, (name, lens)| {
            if name == type_name::<[Val; WIDTH]>() {
                let count: usize = lens.iter().sum();
                println!("Permutation: {name}, Count: {count}");
                total + count
            } else {
                panic!("Permutation type not yet supported: {}", name);
            }
        });

        StarkHashStatistics {
            name: type_name::<P>().to_string(),
            stats: HashStatistics { permutations },
            fri_params: self.fri_params,
            custom,
        }
    }
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_engine() -> BabyBearPoseidon2Engine {
    default_engine_impl(FriParameters::standard_fast())
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
fn default_engine_impl(fri_params: FriParameters) -> BabyBearPoseidon2Engine {
    let perm = default_perm();
    engine_from_perm(perm, fri_params)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm) -> BabyBearPoseidon2Config {
    let fri_params = FriParameters::standard_fast();
    config_from_perm(perm, fri_params)
}

pub fn engine_from_perm<P>(perm: P, fri_params: FriParameters) -> BabyBearPermutationEngine<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    let config = config_from_perm(&perm, fri_params);
    BabyBearPermutationEngine {
        config,
        perm,
        fri_params,
    }
}

pub fn config_from_perm<P>(perm: &P, fri_params: FriParameters) -> BabyBearPermutationConfig<P>
where
    P: CryptographicPermutation<[Val; WIDTH]>
        + CryptographicPermutation<[PackedVal; WIDTH]>
        + Clone,
{
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_config = FriConfig {
        log_blowup: fri_params.log_blowup,
        num_queries: fri_params.num_queries,
        proof_of_work_bits: fri_params.proof_of_work_bits,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let rap_phase = StarkLogUpPhase::new();
    BabyBearPermutationConfig::new(pcs, rap_phase)
}

/// Uses HorizenLabs Poseidon2 round constants, but plonky3 Mat4 and also
/// with a p3 Monty reduction factor.
pub fn default_perm() -> Perm {
    let (external_constants, internal_constants) = horizen_round_consts_16();
    Perm::new(external_constants, internal_constants)
}

pub fn random_perm() -> Perm {
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);
    Perm::new_from_rng_128(&mut rng)
}

pub fn random_instrumented_perm() -> InstrPerm {
    let perm = random_perm();
    Instrumented::new(perm)
}

fn horizen_to_p3(horizen_babybear: HorizenBabyBear) -> BabyBear {
    BabyBear::from_canonical_u64(horizen_babybear.into_bigint().0[0])
}

pub fn horizen_round_consts_16() -> (ExternalLayerConstants<BabyBear, 16>, Vec<BabyBear>) {
    let p3_rc16: Vec<Vec<BabyBear>> = RC16
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|babybear| horizen_to_p3(*babybear))
                .collect()
        })
        .collect();

    let rounds_f = 8;
    let rounds_p = 13;
    let rounds_f_beginning = rounds_f / 2;
    let p_end = rounds_f_beginning + rounds_p;
    let initial: Vec<[BabyBear; 16]> = p3_rc16[..rounds_f_beginning]
        .iter()
        .cloned()
        .map(|round| round.try_into().unwrap())
        .collect();
    let terminal: Vec<[BabyBear; 16]> = p3_rc16[p_end..]
        .iter()
        .cloned()
        .map(|round| round.try_into().unwrap())
        .collect();
    let internal_round_constants: Vec<BabyBear> = p3_rc16[rounds_f_beginning..p_end]
        .iter()
        .map(|round| round[0])
        .collect();
    (
        ExternalLayerConstants::new(initial, terminal),
        internal_round_constants,
    )
}

/// Logs hash count statistics to stdout and returns as struct.
/// Count of 1 corresponds to a Poseidon2 permutation with rate RATE that outputs OUT field elements
#[allow(dead_code)]
pub fn print_hash_counts(hash_counter: &InstrumentCounter, compress_counter: &InstrumentCounter) {
    let hash_counter = hash_counter.lock().unwrap();
    let mut hash_count = 0;
    hash_counter.iter().for_each(|(name, lens)| {
        if name == type_name::<(Val, [Val; DIGEST_WIDTH])>() {
            let count = lens.iter().fold(0, |count, len| count + len.div_ceil(RATE));
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
                count + (DIGEST_WIDTH * len).div_ceil(WIDTH)
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

impl StarkFriEngine<BabyBearPoseidon2Config> for BabyBearPoseidon2Engine {
    fn new(fri_params: FriParameters) -> Self {
        default_engine_impl(fri_params)
    }
    fn fri_params(&self) -> FriParameters {
        self.fri_params
    }
}
