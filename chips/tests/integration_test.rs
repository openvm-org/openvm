use std::{iter, sync::Arc};

use afs_chips::{range, xor_bits};
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, types::ProverRap, MultiTraceStarkProver},
    verifier::{types::VerifierRap, MultiTraceStarkVerifier, VerificationError},
};
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::DenseMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::IntoParallelRefIterator;
use p3_uni_stark::StarkGenericConfig;
use rand::{rngs::StdRng, SeedableRng};

mod config;
mod list;
mod xor_requester;

type Val = BabyBear;

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

#[test]
fn test_xor_chip() {
    use rand::Rng;
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);

    use xor_bits::XorBitsChip;
    use xor_requester::XorRequesterChip;

    let bus_index = 0;

    const BITS: usize = 3;
    const MAX: u32 = 1 << BITS;

    const LOG_XOR_REQUESTS: usize = 4;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    const LOG_NUM_REQUESTERS: usize = 3;
    const NUM_REQUESTERS: usize = 1 << LOG_NUM_REQUESTERS;

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, LOG_XOR_REQUESTS + LOG_NUM_REQUESTERS);

    let xor_chip = Arc::new(XorBitsChip::<BITS>::new(bus_index, vec![]));

    let mut requesters = (0..NUM_REQUESTERS)
        .map(|_| XorRequesterChip::new(bus_index, vec![], Arc::clone(&xor_chip)))
        .collect::<Vec<XorRequesterChip<BITS>>>();

    for requester in &mut requesters {
        for _ in 0..XOR_REQUESTS {
            requester.add_request(rng.gen::<u32>() % MAX, rng.gen::<u32>() % MAX);
        }
    }

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    for requester in &requesters {
        let n = requester.requests.len();
        keygen_builder.add_air(requester, n, 0);
    }

    keygen_builder.add_air(&*xor_chip, NUM_REQUESTERS * XOR_REQUESTS, 0);
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let requesters_traces = requesters
        .par_iter()
        .map(|requester| requester.generate_trace())
        .collect::<Vec<DenseMatrix<BabyBear>>>();

    let xor_chip_trace = xor_chip.generate_trace();

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    for trace in requesters_traces {
        trace_builder.load_trace(trace)
    }
    trace_builder.load_trace(xor_chip_trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(
        &vk,
        requesters
            .iter()
            .map(|requester| requester as &dyn ProverRap<_>)
            .chain(iter::once(&*xor_chip as &dyn ProverRap<_>))
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
            requesters
                .iter()
                .map(|requester| requester as &dyn VerifierRap<_>)
                .chain(iter::once(&*xor_chip as &dyn VerifierRap<_>))
                .collect(),
            proof,
            &pis,
        )
        .expect("Verification failed");
}

#[test]
fn negative_test_xor_chip() {
    use rand::Rng;
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);

    use xor_bits::XorBitsChip;

    let bus_index = 0;

    const BITS: usize = 3;
    const MAX: u32 = 1 << BITS;

    const LOG_XOR_REQUESTS: usize = 4;
    const XOR_REQUESTS: usize = 1 << LOG_XOR_REQUESTS;

    let perm = config::poseidon2::random_perm();
    let config = config::poseidon2::default_config(&perm, LOG_XOR_REQUESTS);

    let xor_chip = Arc::new(XorBitsChip::<BITS>::new(bus_index, vec![]));

    let dummy_requester = DummyInteractionAir::new(3, true, 0);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);

    keygen_builder.add_air(&dummy_requester, XOR_REQUESTS, 0);

    keygen_builder.add_air(&*xor_chip, XOR_REQUESTS, 0);

    let mut reqs = vec![];
    for _ in 0..XOR_REQUESTS {
        let x = rng.gen::<u32>() % MAX;
        let y = rng.gen::<u32>() % MAX;
        reqs.push((1, vec![x, y, x ^ y]));
        xor_chip.request(x, y);
    }

    // Modifying one of the values to send incompatible values
    reqs[0].1[2] = reqs[0].1[2] + 1;

    let xor_chip_trace = xor_chip.generate_trace();

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let dummy_trace = RowMajorMatrix::new(
        reqs.into_iter()
            .flat_map(|(count, fields)| iter::once(count).chain(fields))
            .map(Val::from_wrapped_u32)
            .collect(),
        4,
    );

    let prover = MultiTraceStarkProver::new(config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.config.pcs());
    trace_builder.load_trace(dummy_trace);
    trace_builder.load_trace(xor_chip_trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&dummy_requester, &*xor_chip]);

    let pis = vec![vec![]; vk.per_air.len()];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    let result = verifier.verify(
        &mut challenger,
        vk,
        vec![&dummy_requester, &*xor_chip],
        proof,
        &pis,
    );

    assert_eq!(
        result,
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected verification to fail, but it passed"
    );
}
