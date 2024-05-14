use afs_middleware::{
    prover::{
        trace::TraceCommitter,
        types::{ProvenMultiMatrixAirTrace, ProverRap},
        PartitionProver,
    },
    setup::PartitionSetup,
    verifier::{types::VerifierRap, PartitionVerifier},
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::DenseMatrix;
use p3_maybe_rayon::prelude::IntoParallelRefIterator;
use p3_uni_stark::StarkGenericConfig;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use std::sync::{Arc, Mutex};

use afs_chips::{range, xor};
mod list;
mod xor_requester;

mod config;

use crate::config::poseidon2::StarkConfigPoseidon2;

use rand::{rngs::StdRng, SeedableRng};

#[test]
fn test_list_range_checker() {
    use rand::Rng;

    use list::ListChip;
    use range::RangeCheckerChip;

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();

    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);

    let bus_index = 0;

    const LOG_TRACE_DEGREE_RANGE: usize = 3;
    const MAX: u32 = 1 << LOG_TRACE_DEGREE_RANGE;

    const LOG_TRACE_DEGREE_LIST: usize = 6;
    const LIST_LEN: usize = 1 << LOG_TRACE_DEGREE_LIST;

    let trace_degree_max: usize = std::cmp::max(LOG_TRACE_DEGREE_LIST, LOG_TRACE_DEGREE_RANGE);

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, trace_degree_max);

    // Creating a RangeCheckerChip
    let range_checker = Arc::new(RangeCheckerChip::<MAX>::new(bus_index));

    // Generating random lists
    let num_lists = 10;
    let lists_vals = (0..num_lists)
        .map(|_| {
            (0..LIST_LEN)
                .map(|_| rng.gen::<u32>() % MAX)
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>();

    // define a bnach of ListChips
    let lists = lists_vals
        .iter()
        .map(|vals| ListChip::new(bus_index, vals.to_vec(), Arc::clone(&range_checker)))
        .collect::<Vec<ListChip<MAX>>>();

    let pis = [];

    let lists_prep_traces = lists
        .par_iter()
        .map(|list| list.preprocessed_trace())
        .collect::<Vec<Option<DenseMatrix<BabyBear>>>>();

    let lists_traces = lists
        .par_iter()
        .map(|list| list.generate_trace())
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let prep_trace_range = range_checker.preprocessed_trace();
    let trace_range = range_checker.generate_trace();

    let mut prep_traces = lists_prep_traces;
    prep_traces.push(prep_trace_range);

    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(prep_traces);

    let mut traces = lists_traces;
    traces.push(trace_range);

    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(traces);

    let mut airs_prover: Vec<&dyn ProverRap<StarkConfigPoseidon2>> = lists
        .iter()
        .map(|list| list as &dyn ProverRap<StarkConfigPoseidon2>)
        .collect();
    airs_prover.push(&*range_checker);

    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: airs_prover,
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    let mut airs_verifier: Vec<&dyn VerifierRap<StarkConfigPoseidon2>> = lists
        .iter()
        .map(|list| list as &dyn VerifierRap<StarkConfigPoseidon2>)
        .collect();
    airs_verifier.push(&*range_checker);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, airs_verifier, proof, &pis)
        .expect("Verification failed");
}

#[test]
fn test_xor_chip() {
    use rand::Rng;
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);

    use xor::XorChip;
    use xor_requester::XorRequesterChip;

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();

    let bus_index = 0;

    const BITS: usize = 3;
    const MAX: u32 = 1 << BITS;

    const LOG_XOR_REQUESTS: usize = 4;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    const LOG_NUM_REQUESTERS: usize = 3;
    const NUM_REQUESTERS: usize = 1 << LOG_NUM_REQUESTERS;

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, LOG_XOR_REQUESTS + LOG_NUM_REQUESTERS);

    let xor_chip = Arc::new(Mutex::new(XorChip::<BITS>::new(bus_index, vec![])));

    let pis = [];

    let mut requesters = (0..NUM_REQUESTERS)
        .map(|_| XorRequesterChip::new(bus_index, vec![], Arc::clone(&xor_chip)))
        .collect::<Vec<XorRequesterChip<BITS>>>();

    for requester in &mut requesters {
        for _ in 0..XOR_REQUESTS {
            requester.add_request(rng.gen::<u32>() % MAX, rng.gen::<u32>() % MAX);
        }
    }

    let requesters_prep_traces = requesters
        .par_iter()
        .map(|requester| requester.preprocessed_trace())
        .collect::<Vec<Option<DenseMatrix<BabyBear>>>>();

    let requesters_traces = requesters
        .par_iter()
        .map(|requester| requester.generate_trace())
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let xor_chip_locked = xor_chip.lock().unwrap();
    let prep_trace_xor_chip = xor_chip_locked.preprocessed_trace();
    let trace_xor_chip = xor_chip_locked.generate_trace();

    let mut prep_traces = requesters_prep_traces;
    prep_traces.push(prep_trace_xor_chip);

    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(prep_traces);

    let mut traces = requesters_traces;
    traces.push(trace_xor_chip);

    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(traces);

    let mut airs_prover: Vec<&dyn ProverRap<StarkConfigPoseidon2>> = requesters
        .iter()
        .map(|requester| requester as &dyn ProverRap<StarkConfigPoseidon2>)
        .collect();
    airs_prover.push(&*xor_chip_locked);

    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: airs_prover,
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    let mut airs_verifier: Vec<&dyn VerifierRap<StarkConfigPoseidon2>> = requesters
        .iter()
        .map(|requester| requester as &dyn VerifierRap<StarkConfigPoseidon2>)
        .collect();
    airs_verifier.push(&*xor_chip_locked);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(&mut challenger, vk, airs_verifier, proof, &pis)
        .expect("Verification failed");
}
