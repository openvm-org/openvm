use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_stark_backend::interaction::trace::generate_permutation_trace;
use afs_stark_backend::keygen::types::MultiStarkProvingKey;
use afs_stark_backend::prover::types::MultiAirCommittedTraceData;
use afs_stark_backend::rap::AnyRap;
use afs_test_utils::engine::StarkEngine;
use p3_maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

use benchmark::utils::bench::{gen_ops_sender_trace, generate_page_and_ops, get_dummy_ptd};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::PrimeField;
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use pprof::criterion::{Output, PProfProfiler}; // Add this line

use afs_page::page_rw_checker::page_controller::PageController;
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    interaction::dummy_interaction_air::DummyInteractionAir,
};

pub fn perm_trace_gen_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace gen");
    group.sample_size(10);

    let idx_len = 16;
    let data_len = 16;
    let log_page_height = 15;
    let log_num_ops = 15;
    let idx_limb_bits = 16;
    let idx_decomp = 16;

    let page_bus_index = 0;
    let range_bus_index = 1;
    let ops_bus_index = 2;

    const MAX_VAL: u32 = 1 << 28;

    let page_height = 1 << log_page_height;
    let num_ops = 1 << log_num_ops;
    let oc_trace_degree = num_ops * 4;
    let max_idx = 1 << idx_limb_bits;

    let (page, ops) =
        generate_page_and_ops(idx_len, data_len, page_height, num_ops, max_idx, MAX_VAL);

    let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
        page_bus_index,
        range_bus_index,
        ops_bus_index,
        idx_len,
        data_len,
        idx_limb_bits,
        idx_decomp,
    );
    let ops_sender = DummyInteractionAir::new(idx_len + data_len + 2, true, ops_bus_index);

    let engine = config::baby_bear_poseidon2::default_engine(
        idx_decomp.max(log_page_height.max(3 + log_num_ops)),
    );
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    page_controller.set_up_keygen_builder(&mut keygen_builder, &ops_sender);

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let dummy_ptd = get_dummy_ptd(&mut trace_builder.committer);

    let (init_page_pdata, final_page_pdata) = page_controller.load_page_and_ops(
        black_box(&page),
        black_box(Some(Arc::new(dummy_ptd.clone()))),
        black_box(Some(Arc::new(dummy_ptd.clone()))),
        black_box(&ops),
        black_box(oc_trace_degree),
        black_box(&mut trace_builder.committer),
    );

    let ops_sender_trace = gen_ops_sender_trace(black_box(&ops_sender), black_box(&ops));
    let pk = keygen_builder.generate_pk();
    pc_prove_with_group(
        group,
        &page_controller,
        &engine,
        &pk,
        &mut trace_builder,
        init_page_pdata,
        final_page_pdata,
        &ops_sender,
        ops_sender_trace,
    );
}
/// This function clears the trace_builder, loads in the traces for all involved chips
/// (including the range_checker and the ops_sender, which is passed in along with its trace),
/// commits them, and then generates the proof.
/// cached_traces_prover_data is a vector of ProverTraceData object for the cached pages
/// (init_page, final_page), which is returned by load_page_and_ops
#[allow(clippy::too_many_arguments)]
pub fn pc_prove_with_group<SC: StarkGenericConfig>(
    group: BenchmarkGroup<WallTime>,
    page_controller: &PageController<SC>,
    engine: &impl StarkEngine<SC>,
    pk: &MultiStarkProvingKey<SC>,
    trace_builder: &mut TraceCommitmentBuilder<SC>,
    init_page_pdata: Arc<ProverTraceData<SC>>,
    final_page_pdata: Arc<ProverTraceData<SC>>,
    ops_sender: &dyn AnyRap<SC>,
    ops_sender_trace: DenseMatrix<Val<SC>>,
) where
    Val<SC>: PrimeField,
    Domain<SC>: Send + Sync,
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let traces = page_controller.traces().as_ref().unwrap();

    trace_builder.clear();

    trace_builder.load_cached_trace(
        traces.init_page_trace().clone(),
        match Arc::try_unwrap(init_page_pdata) {
            Ok(data) => data,
            Err(_) => panic!("Prover data should have only one owner"),
        },
    );
    trace_builder.load_cached_trace(
        traces.final_page_trace().clone(),
        match Arc::try_unwrap(final_page_pdata) {
            Ok(data) => data,
            Err(_) => panic!("Prover data should have only one owner"),
        },
    );
    trace_builder.load_trace(traces.final_page_aux_trace().clone());
    trace_builder.load_trace(traces.offline_checker_trace().clone());
    trace_builder.load_trace(page_controller.range_checker.generate_trace());
    trace_builder.load_trace(ops_sender_trace);

    tracing::info_span!("Prove trace commitment").in_scope(|| trace_builder.commit_current());

    let vk = pk.vk();

    let main_trace_data = trace_builder.view(
        &vk,
        vec![
            page_controller.init_chip(),
            page_controller.final_chip(),
            page_controller.offline_checker(),
            &page_controller.range_checker.air,
            ops_sender,
        ],
    );

    let pis = vec![vec![]; vk.per_air.len()];
    let mut challenger = engine.new_challenger();
    partial_prove_with_group(group, &mut challenger, pk, main_trace_data, &pis)
}

pub fn partial_prove_with_group<'a, SC: StarkGenericConfig>(
    mut group: BenchmarkGroup<WallTime>,
    challenger: &mut SC::Challenger,
    pk: &'a MultiStarkProvingKey<SC>,
    main_trace_data: MultiAirCommittedTraceData<'a, SC>,
    public_values: &'a [Vec<Val<SC>>],
) where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    // Challenger must observe public values
    for pis in public_values.iter() {
        challenger.observe_slice(pis);
    }

    let preprocessed_commits: Vec<_> = pk.preprocessed_commits().cloned().collect();
    challenger.observe_slice(&preprocessed_commits);

    // Challenger must observe all trace commitments
    let main_trace_commitments = main_trace_data.commits().cloned().collect_vec();
    assert_eq!(main_trace_commitments.len(), pk.num_main_trace_commitments);
    challenger.observe_slice(&main_trace_commitments);

    // TODO: this is not needed if there are no interactions. Number of challenge rounds should be specified in proving key
    // Generate 2 permutation challenges
    assert!(pk.num_challenges_to_sample.len() <= 1);
    let challenges: Vec<_> = pk
        .num_challenges_to_sample
        .iter()
        .map(|&num_challenges| {
            (0..num_challenges)
                .map(|_| challenger.sample_ext_element::<SC::Challenge>())
                .collect_vec()
        })
        .collect();

    // TODO: ===== Permutation Trace Generation should be moved to separate module ====
    // Generate permutation traces
    let perm_challenges = challenges.first().map(|c| [c[0], c[1]]); // must have 2 challenges

    // group.bench_function("offline checker perm trace gen", |b| {
    //     b.iter(|| {
    //         let idx = 2;
    //         let pk = &pk.per_air[idx];
    //         let main = &main_trace_data.air_traces[idx];
    //         let public_values = &public_values[idx];
    //         let interactions = &pk.vk.symbolic_constraints.interactions;
    //         let preprocessed_trace = pk.preprocessed_data.as_ref().map(|d| d.trace.as_view());
    //         generate_permutation_trace(
    //             interactions,
    //             &preprocessed_trace,
    //             &main.partitioned_main_trace,
    //             public_values,
    //             perm_challenges,
    //         );
    //     })
    // });

    // group.bench_function("init page trace gen", |b| {
    //     b.iter(|| {
    //         let idx = 0;
    //         let pk = &pk.per_air[idx];
    //         let main = &main_trace_data.air_traces[idx];
    //         let public_values = &public_values[idx];
    //         let interactions = &pk.vk.symbolic_constraints.interactions;
    //         let preprocessed_trace = pk.preprocessed_data.as_ref().map(|d| d.trace.as_view());
    //         generate_permutation_trace(
    //             interactions,
    //             &preprocessed_trace,
    //             &main.partitioned_main_trace,
    //             public_values,
    //             perm_challenges,
    //         );
    //     })
    // });

    group.bench_function("final page trace gen", |b| {
        b.iter(|| {
            let idx = 1;
            let pk = &pk.per_air[idx];
            let main = &main_trace_data.air_traces[idx];
            let public_values = &public_values[idx];
            let interactions = &pk.vk.symbolic_constraints.interactions;
            let preprocessed_trace = pk.preprocessed_data.as_ref().map(|d| d.trace.as_view());
            generate_permutation_trace(
                interactions,
                &preprocessed_trace,
                &main.partitioned_main_trace,
                public_values,
                perm_challenges,
            );
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = perm_trace_gen_benchmark
}
criterion_main!(benches);
