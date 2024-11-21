use std::any::type_name;

use ff::PrimeField;
use p3_baby_bear::BabyBear;
use p3_bn254_fr::{Bn254Fr, FFBn254Fr, Poseidon2Bn254};
use p3_challenger::MultiField32Challenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{CryptographicPermutation, MultiField32PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
use zkhash::{
    ark_ff::{BigInteger, PrimeField as _},
    fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

use super::{
    instrument::{HashStatistics, InstrumentCounter, Instrumented, StarkHashStatistics},
    FriParameters,
};
use crate::{
    assert_sc_compatible_with_serde,
    engine::{StarkEngine, StarkEngineWithHashInstrumentation, StarkFriEngine},
};

const WIDTH: usize = 3;
/// Poseidon rate in F. <Poseidon RATE>(2) * <# of F in a N>(8) = 16
const RATE: usize = 16;
const DIGEST_WIDTH: usize = 1;

/// A configuration for  recursion.
type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2Bn254<WIDTH>;
type Hash<P> = MultiField32PaddingFreeSponge<Val, Bn254Fr, P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, 1, WIDTH>;
type ValMmcs<P> = MerkleTreeMmcs<BabyBear, Bn254Fr, Hash<P>, Compress<P>, 1>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
type Dft = Radix2DitParallel<Val>;
type Challenger<P> = MultiField32Challenger<Val, Bn254Fr, P, WIDTH, 2>;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;

pub type BabyBearPermutationOuterConfig<P> = StarkConfig<Pcs<P>, Challenge, Challenger<P>>;
pub type BabyBearPoseidon2OuterConfig = BabyBearPermutationOuterConfig<Perm>;
pub type BabyBearPoseidon2OuterEngine = BabyBearPermutationOuterEngine<Perm>;

assert_sc_compatible_with_serde!(BabyBearPoseidon2OuterConfig);

pub struct BabyBearPermutationOuterEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    pub fri_params: FriParameters,
    pub config: BabyBearPermutationOuterConfig<P>,
    pub perm: P,
}

impl<P> StarkEngine<BabyBearPermutationOuterConfig<P>> for BabyBearPermutationOuterEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    fn config(&self) -> &BabyBearPermutationOuterConfig<P> {
        &self.config
    }

    fn new_challenger(&self) -> Challenger<P> {
        Challenger::new(self.perm.clone()).unwrap()
    }
}

impl<P> StarkEngineWithHashInstrumentation<BabyBearPermutationOuterConfig<Instrumented<P>>>
    for BabyBearPermutationOuterEngine<Instrumented<P>>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
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
pub fn default_engine() -> BabyBearPoseidon2OuterEngine {
    default_engine_impl(FriParameters::standard_fast())
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
fn default_engine_impl(fri_params: FriParameters) -> BabyBearPoseidon2OuterEngine {
    let perm = outer_perm();
    engine_from_perm(perm, fri_params)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm) -> BabyBearPoseidon2OuterConfig {
    let fri_params = FriParameters::standard_fast();
    config_from_perm(perm, fri_params)
}

pub fn engine_from_perm<P>(perm: P, fri_params: FriParameters) -> BabyBearPermutationOuterEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    let config = config_from_perm(&perm, fri_params);
    BabyBearPermutationOuterEngine {
        config,
        perm,
        fri_params,
    }
}

pub fn config_from_perm<P>(perm: &P, fri_params: FriParameters) -> BabyBearPermutationOuterConfig<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    let hash = Hash::new(perm.clone()).unwrap();
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
    BabyBearPermutationOuterConfig::new(pcs)
}

/// The permutation for outer recursion.
pub fn outer_perm() -> Perm {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let terminal = round_constants.split_off(internal_end);
    let internal_round_constants = round_constants.split_off(ROUNDS_F / 2);
    let internal_round_constants = internal_round_constants
        .into_iter()
        .map(|vec| vec[0])
        .collect::<Vec<_>>();
    let initial = round_constants;

    let external_round_constants = ExternalLayerConstants::new(initial, terminal);
    Perm::new(external_round_constants, internal_round_constants)
}

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Fr {
    let bytes = input.into_bigint().to_bytes_le();

    let mut res = <FFBn254Fr as ff::PrimeField>::Repr::default();

    for (i, digit) in res.as_mut().iter_mut().enumerate() {
        *digit = bytes[i];
    }

    let value = FFBn254Fr::from_repr(res);

    if value.is_some().into() {
        Bn254Fr {
            value: value.unwrap(),
        }
    } else {
        panic!("Invalid field element")
    }
}

fn bn254_poseidon2_rc3() -> Vec<[Bn254Fr; 3]> {
    RC3.iter()
        .map(|vec| {
            vec.iter()
                .cloned()
                .map(bn254_from_ark_ff)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect()
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

impl StarkFriEngine<BabyBearPoseidon2OuterConfig> for BabyBearPoseidon2OuterEngine {
    fn new(fri_params: FriParameters) -> Self {
        default_engine_impl(fri_params)
    }
    fn fri_params(&self) -> FriParameters {
        self.fri_params
    }
}
