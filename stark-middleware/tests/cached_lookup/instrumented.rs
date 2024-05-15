use std::iter;

use afs_middleware::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    verifier::{MultiTraceStarkVerifier, VerificationError},
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    config::{self, poseidon2::print_hash_counts, tracing_setup},
    interaction::dummy_interaction_air::DummyInteractionAir,
};

type Val = BabyBear;

// Lookup table is cached, everything else (including counts) is committed together
fn prove_and_verify(trace: Vec<(u32, Vec<u32>)>, partition: bool) {
    // tracing_setup();
    let degree = trace.len();
    let log_degree = log2_ceil_usize(degree);

    let perm = config::poseidon2::random_perm();
    let (config, hash_counts, compress_counts) =
        config::poseidon2::instrumented_config(&perm, log_degree, 3, 33, 16);

    let air = DummyInteractionAir::new(trace[0].1.len(), false, 0);

    // Single row major matrix for |count|fields[..]|
    let nopart_trace = RowMajorMatrix::new(
        trace
            .iter()
            .cloned()
            .flat_map(|(count, fields)| {
                assert_eq!(fields.len(), air.field_width());
                iter::once(count).chain(fields)
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        air.field_width() + 1,
    );
    let (count, fields): (Vec<_>, Vec<_>) = trace.into_iter().unzip();
    let part_count_trace =
        RowMajorMatrix::new(count.into_iter().map(Val::from_wrapped_u32).collect(), 1);
    let part_fields_trace = RowMajorMatrix::new(
        fields
            .into_iter()
            .flat_map(|fields| {
                assert_eq!(fields.len(), air.field_width());
                fields
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        air.field_width(),
    );

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    if partition {
        let fields_ptr = keygen_builder.add_cached_main_matrix(air.field_width());
        let count_ptr = keygen_builder.add_main_matrix(1);
        keygen_builder.add_partitioned_air(&air, degree, 0, vec![count_ptr, fields_ptr]);
    } else {
        keygen_builder.add_air(&air, degree, 0);
    }
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = MultiTraceStarkProver::new(config);
    // Must add trace matrices in the same order as above
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    if partition {
        // Receiver fields table is cached
        let cached_trace_data = trace_builder
            .committer
            .commit(vec![part_fields_trace.clone()]);
        trace_builder.load_cached_trace(part_fields_trace, cached_trace_data);
        trace_builder.load_trace(part_count_trace);
    } else {
        trace_builder.load_trace(nopart_trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&air]);
    let pis = vec![vec![]];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    hash_counts.lock().unwrap().clear();
    compress_counts.lock().unwrap().clear();
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    // Do not check cumulative sum
    verifier
        .verify_raps(&mut challenger, vk, vec![&air], proof, &pis)
        .unwrap();
    print_hash_counts(&hash_counts, &compress_counts);
}

fn generate_random_trace(
    mut rng: impl Rng,
    field_width: usize,
    height: usize,
) -> Vec<(u32, Vec<u32>)> {
    (0..height)
        .map(|_| {
            (
                rng.gen_range(0..1000),
                (0..field_width).map(|_| rng.gen()).collect(),
            )
        })
        .collect()
}

fn bench_comparison(field_width: usize, height: usize) {
    let rng = StdRng::seed_from_u64(0);
    let trace = generate_random_trace(rng, field_width, height);
    println!("Without cached trace:");
    prove_and_verify(trace.clone(), false);

    println!("With cached trace:");
    prove_and_verify(trace, true);
}

// Run with `cargo t --release -- --ignored <test name>`
#[test]
#[ignore = "bench"]
fn bench_cached_trace() {
    bench_comparison(1, 1 << 20);
}
