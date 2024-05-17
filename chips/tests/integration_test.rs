use std::{iter, sync::Arc};

use rand::{rngs::StdRng, SeedableRng};

use afs_chips::range;
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, types::ProverRap, MultiTraceStarkProver},
    verifier::{types::VerifierRap, MultiTraceStarkVerifier},
};
use p3_baby_bear::BabyBear;
use p3_matrix::dense::DenseMatrix;
use p3_maybe_rayon::prelude::IntoParallelRefIterator;
use p3_uni_stark::StarkGenericConfig;

mod config;
mod list;

#[test]
fn test_list_range_checker() {
    use rand::Rng;

    use list::ListChip;
    use range::RangeCheckerChip;

    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);

    let bus_index = 0;

    const LOG_TRACE_DEGREE_RANGE: usize = 3;
    const MAX: u32 = 1 << LOG_TRACE_DEGREE_RANGE;

    const LOG_TRACE_DEGREE_LIST: usize = 6;
    const LIST_LEN: usize = 1 << LOG_TRACE_DEGREE_LIST;

    let log_trace_degree_max: usize = std::cmp::max(LOG_TRACE_DEGREE_LIST, LOG_TRACE_DEGREE_RANGE);

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, log_trace_degree_max);

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

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    for list in &lists {
        let n = list.vals().len();
        keygen_builder.add_air(list, n, 0);
    }
    keygen_builder.add_air(&*range_checker, MAX as usize, 0);
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let lists_traces = lists
        .par_iter()
        .map(|list| list.generate_trace())
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let range_trace = range_checker.generate_trace();

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    for trace in lists_traces {
        trace_builder.load_trace(trace)
    }
    trace_builder.load_trace(range_trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &vk,
        lists
            .iter()
            .map(|list| list as &dyn ProverRap<_>)
            .chain(iter::once(&*range_checker as &dyn ProverRap<_>))
            .collect(),
    );

    let pis = vec![vec![]; vk.per_air.len()];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(
            &mut challenger,
            vk,
            lists
                .iter()
                .map(|list| list as &dyn VerifierRap<_>)
                .chain(iter::once(&*range_checker as &dyn VerifierRap<_>))
                .collect(),
            proof,
            &pis,
        )
        .expect("Verification failed");
}
