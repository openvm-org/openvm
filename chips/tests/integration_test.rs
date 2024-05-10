use afs_middleware::{
    prover::{trace::TraceCommitter, types::ProvenMultiMatrixAirTrace, PartitionProver},
    setup::PartitionSetup,
    verifier::PartitionVerifier,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_uni_stark::StarkGenericConfig;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use rand::thread_rng;
use std::sync::{Arc, Mutex};

use afs_chips::list;
use afs_chips::range;

mod config;

use crate::config::poseidon2::StarkConfigPoseidon2;

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

    let mut rng = thread_rng();

    let bus_index = 0;

    const LOG_TRACE_DEGREE_RANGE: usize = 3;
    const MAX: u32 = 1 << LOG_TRACE_DEGREE_RANGE;

    const LOG_TRACE_DEGREE_LIST: usize = 6;
    const LIST_LEN: usize = 1 << LOG_TRACE_DEGREE_LIST;

    let trace_degree_max: usize = std::cmp::max(LOG_TRACE_DEGREE_LIST, LOG_TRACE_DEGREE_RANGE);

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, trace_degree_max);

    // Creating a RangeCheckerChip
    let range_checker = Arc::new(Mutex::new(RangeCheckerChip::<MAX>::new(bus_index)));

    // Creating a ListChip with random values
    let rand_vals = (0..LIST_LEN)
        .map(|_| rng.gen::<u32>() % MAX)
        .collect::<Vec<u32>>();

    let mut list = ListChip::new(bus_index, rand_vals, Some(Arc::clone(&range_checker)));

    let pis = [].map(BabyBear::from_canonical_u32);

    let prep_trace_list = list.preprocessed_trace();
    let trace_list = list.generate_trace();

    let range_checker_locked = range_checker.lock().unwrap();
    let prep_trace_range = range_checker_locked.preprocessed_trace();
    let trace_range = range_checker_locked.generate_trace();

    let setup = PartitionSetup::new(&config);
    let (pk, vk) = setup.setup(vec![prep_trace_list, prep_trace_range]);

    let trace_committer = TraceCommitter::<StarkConfigPoseidon2>::new(config.pcs());
    let proven_trace = trace_committer.commit(vec![trace_list, trace_range]);

    let proven = ProvenMultiMatrixAirTrace {
        trace_data: &proven_trace,
        airs: vec![&list, &*range_checker_locked],
    };

    let prover = PartitionProver::new(config);
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, vec![proven], &pis);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = PartitionVerifier::new(prover.config);
    verifier
        .verify(
            &mut challenger,
            vk,
            vec![&list, &*range_checker_locked],
            proof,
            &pis,
        )
        .expect("Verification failed");
}
