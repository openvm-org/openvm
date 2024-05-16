use std::iter;

use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    verifier::{MultiTraceStarkVerifier, VerificationError},
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;

use crate::config;
use crate::interaction::dummy_interaction_air::DummyInteractionAir;

type Val = BabyBear;

// Lookup table is cached, everything else (including counts) is committed together
pub fn prove_and_verify_indexless_lookups(
    sender: Vec<(u32, Vec<u32>)>,
    receiver: Vec<(u32, Vec<u32>)>,
) -> Result<(), VerificationError> {
    let sender_degree = sender.len();
    let receiver_degree = receiver.len();
    let [sender_log_degree, receiver_log_degree] =
        [sender_degree, receiver_degree].map(log2_ceil_usize);

    let perm = config::poseidon2::random_perm();
    let config =
        config::poseidon2::default_config(&perm, sender_log_degree.max(receiver_log_degree));

    let sender_air = DummyInteractionAir::new(sender[0].1.len(), true, 0);
    let receiver_air = DummyInteractionAir::new(receiver[0].1.len(), false, 0);

    // Single row major matrix for |count|fields[..]|
    let sender_trace = RowMajorMatrix::new(
        sender
            .into_iter()
            .flat_map(|(count, fields)| {
                assert_eq!(fields.len(), sender_air.field_width());
                iter::once(count).chain(fields)
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        sender_air.field_width() + 1,
    );
    let (recv_count, recv_fields): (Vec<_>, Vec<_>) = receiver.into_iter().unzip();
    let recv_count_trace = RowMajorMatrix::new(
        recv_count.into_iter().map(Val::from_wrapped_u32).collect(),
        1,
    );
    let recv_fields_trace = RowMajorMatrix::new(
        recv_fields
            .into_iter()
            .flat_map(|fields| {
                assert_eq!(fields.len(), receiver_air.field_width());
                fields
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        receiver_air.field_width(),
    );

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    // Cached table pointer:
    let recv_fields_ptr = keygen_builder.add_cached_main_matrix(receiver_air.field_width());
    // Everything else together
    let recv_count_ptr = keygen_builder.add_main_matrix(1);
    keygen_builder.add_partitioned_air(
        &receiver_air,
        receiver_degree,
        0,
        vec![recv_count_ptr, recv_fields_ptr],
    );
    // Auto-adds sender matrix
    keygen_builder.add_air(&sender_air, sender_degree, 0);
    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = MultiTraceStarkProver::new(config);
    // Must add trace matrices in the same order as above
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    // Receiver fields table is cached
    let cached_trace_data = trace_builder
        .committer
        .commit(vec![recv_fields_trace.clone()]);
    trace_builder.load_cached_trace(recv_fields_trace, cached_trace_data);
    // Load x normally
    trace_builder.load_trace(recv_count_trace);
    trace_builder.load_trace(sender_trace);
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, vec![&receiver_air, &sender_air]);
    let pis = vec![vec![]; 2];

    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pis);

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier.verify(
        &mut challenger,
        vk,
        vec![&receiver_air, &sender_air],
        proof,
        &pis,
    )
}
