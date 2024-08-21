use std::any::type_name;

use afs_stark_backend::{rap::AnyRap, verifier::VerificationError};
use ff::PrimeField;
use p3_baby_bear::BabyBear;
use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254, FFBn254Fr};
use p3_challenger::MultiField32Challenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CryptographicPermutation, MultiField32PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, StarkGenericConfig};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use zkhash::{
    ark_ff::{BigInteger, PrimeField as _},
    fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

use super::{
    fri_params::default_fri_params,
    instrument::{HashStatistics, InstrumentCounter, Instrumented, StarkHashStatistics},
    FriParameters,
};
use crate::engine::{StarkEngine, StarkEngineWithHashInstrumentation};

const WIDTH: usize = 3;
const RATE: usize = 16;
const DIGEST_WIDTH: usize = 1;

/// A configuration for  recursion.
type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, WIDTH, 5>;
type Hash<P> = MultiField32PaddingFreeSponge<Val, Bn254Fr, P, WIDTH, RATE, DIGEST_WIDTH>;
type Compress<P> = TruncatedPermutation<P, 2, 1, WIDTH>;
type ValMmcs<P> = FieldMerkleTreeMmcs<BabyBear, Bn254Fr, Hash<P>, Compress<P>, 1>;
type ChallengeMmcs<P> = ExtensionMmcs<Val, Challenge, ValMmcs<P>>;
type Dft = Radix2DitParallel;
type Challenger<P> = MultiField32Challenger<Val, Bn254Fr, P, WIDTH>;
type Pcs<P> = TwoAdicFriPcs<Val, Dft, ValMmcs<P>, ChallengeMmcs<P>>;

pub type BabyBearPermutationOuterConfig<P> = StarkConfig<Pcs<P>, Challenge, Challenger<P>>;
pub type BabyBearPoseidon2OuterConfig = BabyBearPermutationOuterConfig<Perm>;
pub type BabyBearPoseidon2OuterEngine = BabyBearPermutationOuterEngine<Perm>;

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
pub fn default_engine(pcs_log_degree: usize) -> BabyBearPoseidon2OuterEngine {
    let perm = outer_perm();
    let fri_params = default_fri_params();
    engine_from_perm(perm, pcs_log_degree, fri_params)
}

/// `pcs_log_degree` is the upper bound on the log_2(PCS polynomial degree).
pub fn default_config(perm: &Perm, pcs_log_degree: usize) -> BabyBearPoseidon2OuterConfig {
    // target 80 bits of security, with conjectures:
    let fri_params = default_fri_params();
    config_from_perm(perm, pcs_log_degree, fri_params)
}

pub fn engine_from_perm<P>(
    perm: P,
    pcs_log_degree: usize,
    fri_params: FriParameters,
) -> BabyBearPermutationOuterEngine<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    let config = config_from_perm(&perm, pcs_log_degree, fri_params);
    BabyBearPermutationOuterEngine {
        config,
        perm,
        fri_params,
    }
}

pub fn config_from_perm<P>(
    perm: &P,
    pcs_log_degree: usize,
    fri_params: FriParameters,
) -> BabyBearPermutationOuterConfig<P>
where
    P: CryptographicPermutation<[Bn254Fr; WIDTH]> + Clone,
{
    let hash = Hash::new(perm.clone()).unwrap();
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
    BabyBearPermutationOuterConfig::new(pcs)
}

/// The permutation for outer recursion.
pub fn outer_perm() -> Perm {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_start = ROUNDS_F / 2;
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let internal_round_constants = round_constants
        .drain(internal_start..internal_end)
        .map(|vec| vec[0])
        .collect::<Vec<_>>();
    let external_round_constants = round_constants;
    Perm::new(
        ROUNDS_F,
        external_round_constants,
        Poseidon2ExternalMatrixGeneral,
        ROUNDS_P,
        internal_round_constants,
        DiffusionMatrixBN254,
    )
}

/// The FRI config for outer recursion.
pub fn outer_fri_config() -> FriConfig<ChallengeMmcs<Perm>> {
    let perm = outer_perm();
    let hash = Hash::new(perm.clone()).unwrap();
    let compress = Compress::new(perm.clone());
    let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 25,
    };
    FriConfig {
        log_blowup: 4,
        num_queries,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    }
}

#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<BabyBearPoseidon2Outer>")]
pub struct BabyBearPoseidon2Outer {
    pub perm: Perm,
    pub pcs: Pcs<Perm>,
}

impl Clone for BabyBearPoseidon2Outer {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Serialize for BabyBearPoseidon2Outer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<BabyBearPoseidon2Outer>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<BabyBearPoseidon2Outer>> for BabyBearPoseidon2Outer {
    fn from(_: std::marker::PhantomData<BabyBearPoseidon2Outer>) -> Self {
        Self::new()
    }
}

impl BabyBearPoseidon2Outer {
    pub fn new() -> Self {
        let perm = outer_perm();
        let hash = Hash::new(perm.clone()).unwrap();
        let compress = Compress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let dft = Dft {};
        let fri_config = outer_fri_config();
        let pcs = Pcs::new(27, dft, val_mmcs, fri_config);
        Self { pcs, perm }
    }
}

impl Default for BabyBearPoseidon2Outer {
    fn default() -> Self {
        Self::new()
    }
}

impl StarkGenericConfig for BabyBearPoseidon2Outer {
    type Pcs = Pcs<Perm>;
    type Challenge = Challenge;
    type Challenger = Challenger<Perm>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }
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

/// Runs a single end-to-end test for a given set of chips and traces.
/// This includes proving/verifying key generation, creating a proof, and verifying the proof.
/// This function should only be used on chips where the main trace is **not** partitioned.
///
/// Do not use this if you want to generate proofs for different traces with the same proving key.
///
/// - `chips`, `traces`, `public_values` should be zipped.
pub fn run_simple_test(
    chips: Vec<&dyn AnyRap<BabyBearPoseidon2OuterConfig>>,
    traces: Vec<DenseMatrix<BabyBear>>,
    public_values: Vec<Vec<BabyBear>>,
) -> Result<(), VerificationError> {
    let max_trace_height = traces.iter().map(|trace| trace.height()).max().unwrap();
    let max_log_degree = log2_strict_usize(max_trace_height);
    let engine = default_engine(max_log_degree);
    engine.run_simple_test(chips, traces, public_values)
}

/// [run_simple_test] without public values
pub fn run_simple_test_no_pis(
    chips: Vec<&dyn AnyRap<BabyBearPoseidon2OuterConfig>>,
    traces: Vec<DenseMatrix<BabyBear>>,
) -> Result<(), VerificationError> {
    let num_chips = chips.len();
    run_simple_test(chips, traces, vec![vec![]; num_chips])
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
