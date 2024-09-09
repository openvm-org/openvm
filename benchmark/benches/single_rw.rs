use std::sync::Arc;

use afs_page::page_rw_checker::page_controller::PageController;
use afs_stark_backend::{
    commit::CommittedSingleMatrixView,
    config::{Com, PcsProof, PcsProverData},
    interaction::trace::generate_permutation_trace,
    keygen::{types::MultiStarkProvingKey, MultiStarkKeygenBuilder},
    prover::{
        quotient::QuotientCommitter,
        trace::{ProverTraceData, SingleRapCommittedTraceView, TraceCommitmentBuilder},
        types::MultiAirCommittedTraceData,
        MultiTraceStarkProver,
    },
    rap::AnyRap,
};
use ax_sdk::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
};
use benchmark::utils::bench::{gen_ops_sender_trace, generate_page_and_ops, get_dummy_ptd};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Pcs;
use p3_field::{AbstractExtensionField, PrimeField};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_maybe_rayon::prelude::ParallelIterator;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use pprof::criterion::{Output, PProfProfiler}; // Add this line

pub fn perm_trace_gen_benchmark(c: &mut Criterion) {
    let idx_len = 16;
    let data_len = 64;
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
    let pk = keygen_builder.generate_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    let dummy_ptd = get_dummy_ptd(&mut trace_builder.committer);

    let (init_page_pdata, final_page_pdata) = page_controller.load_page_and_ops(
        &page,
        None,
        None,
        &ops,
        oc_trace_degree,
        &mut trace_builder.committer,
    );

    let mut group = c.benchmark_group("main_trace_gen");
    group.sample_size(10);
    group.bench_function("main trace gen", |b| {
        b.iter(|| {
            page_controller.load_page_and_ops(
                black_box(&page),
                black_box(Some(Arc::new(dummy_ptd.clone()))),
                black_box(Some(Arc::new(dummy_ptd.clone()))),
                black_box(&ops),
                black_box(oc_trace_degree),
                black_box(&mut trace_builder.committer),
            );

            gen_ops_sender_trace(black_box(&ops_sender), black_box(&ops));
        })
    });
    drop(group);

    let ops_sender_trace = gen_ops_sender_trace(black_box(&ops_sender), black_box(&ops));
    pc_prove_with_group(
        c,
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
    c: &mut Criterion,
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
        traces.init_page_trace.clone(),
        match Arc::try_unwrap(init_page_pdata) {
            Ok(data) => data,
            Err(_) => panic!("Prover data should have only one owner"),
        },
    );
    trace_builder.load_cached_trace(
        traces.final_page_trace.clone(),
        match Arc::try_unwrap(final_page_pdata) {
            Ok(data) => data,
            Err(_) => panic!("Prover data should have only one owner"),
        },
    );
    trace_builder.load_trace(traces.final_page_aux_trace.clone());
    trace_builder.load_trace(traces.offline_checker_trace.clone());
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
    let mut prover = engine.prover();
    partial_prove_with_group(c, &mut prover, &mut challenger, pk, main_trace_data, &pis)
}

pub fn partial_prove_with_group<'a, SC: StarkGenericConfig>(
    c: &mut Criterion,
    prover: &mut MultiTraceStarkProver<SC>,
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
    let pcs = prover.config.pcs();

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

    let interaction_chunk_size = pk.interaction_chunk_size;

    // TODO: ===== Permutation Trace Generation should be moved to separate module ====
    // Generate permutation traces
    {
        let mut group = c.benchmark_group("perm_trace_gen");
        group.sample_size(10);
        let perm_challenges = challenges.first().map(|c| [c[0], c[1]]); // must have 2 challenges

        group.bench_function("offline checker perm trace gen", |b| {
            b.iter(|| {
                let idx = 2;
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
                    interaction_chunk_size,
                );
            })
        });

        group.bench_function("init page perm trace gen", |b| {
            b.iter(|| {
                let idx = 0;
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
                    interaction_chunk_size,
                );
            })
        });

        group.bench_function("final page perm trace gen", |b| {
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
                    interaction_chunk_size,
                );
            })
        });
    }
    let (perm_traces, cumulative_sums_and_indices): (Vec<Option<_>>, Vec<Option<_>>) =
        tracing::info_span!("generate permutation traces").in_scope(|| {
            let perm_challenges = challenges.first().map(|c| [c[0], c[1]]); // must have 2 challenges
            let perm_traces = pk
                .per_air
                .iter()
                .zip_eq(main_trace_data.air_traces.iter())
                .zip_eq(public_values.iter())
                .map(|((pk, main), public_values)| {
                    let interactions = &pk.vk.symbolic_constraints.interactions;
                    let preprocessed_trace =
                        pk.preprocessed_data.as_ref().map(|d| d.trace.as_view());
                    generate_permutation_trace(
                        interactions,
                        &preprocessed_trace,
                        &main.partitioned_main_trace,
                        public_values,
                        perm_challenges,
                        interaction_chunk_size,
                    )
                })
                .collect::<Vec<_>>();
            let mut count = 0usize;
            let cumulative_sums_and_indices = perm_traces
                .iter()
                .map(|opt_trace| {
                    opt_trace.as_ref().map(|trace| {
                        // The cumulative sum is the element in last row of phi, which is the last column in perm_trace
                        let cumulative_sum = *trace.row_slice(trace.height() - 1).last().unwrap();
                        let matrix_index = count;
                        count += 1;
                        (cumulative_sum, matrix_index)
                    })
                })
                .collect();
            (perm_traces, cumulative_sums_and_indices)
        });

    // Challenger needs to observe permutation_exposed_values (aka cumulative_sums)
    for (cumulative_sum, _) in cumulative_sums_and_indices.iter().flatten() {
        challenger.observe_slice(cumulative_sum.as_base_slice());
    }

    // Commit to permutation traces: this means only 1 challenge round right now
    // One shared commit for all permutation traces
    let perm_pcs_data = tracing::info_span!("commit to permutation traces").in_scope(|| {
        let flattened_traces_with_domains: Vec<_> = perm_traces
            .into_iter()
            .zip_eq(&main_trace_data.air_traces)
            .flat_map(|(perm_trace, data)| {
                perm_trace.map(|trace| (data.domain, trace.flatten_to_base()))
            })
            .collect();
        // Only commit if there are permutation traces
        if !flattened_traces_with_domains.is_empty() {
            let (commit, data) = pcs.commit(flattened_traces_with_domains);
            // Challenger observes commitment
            challenger.observe(commit.clone());
            Some((commit, data))
        } else {
            None
        }
    });
    // Either 0 or 1 after_challenge commits, depending on if there are any permutation traces
    let after_challenge_pcs_data: Vec<_> = perm_pcs_data.into_iter().collect();
    let main_pcs_data = &main_trace_data.pcs_data;

    // Prepare the proven RAP trace views
    // Abstraction boundary: after this, we consider InteractiveAIR as a RAP with virtual columns included in the trace.
    let (raps, trace_views): (Vec<_>, Vec<_>) = izip!(
        main_trace_data.air_traces,
        &pk.per_air,
        cumulative_sums_and_indices
    )
    .map(|(main, pk, cumulative_sum_and_index)| {
        // The AIR will be treated as the full RAP with virtual columns after this
        let rap = main.air;
        let domain = main.domain;
        let preprocessed = pk.preprocessed_data.as_ref().map(|p| {
            // TODO: currently assuming each chip has it's own preprocessed commitment
            CommittedSingleMatrixView::new(p.data.as_ref(), 0)
        });
        let matrix_ptrs = &pk.vk.main_graph.matrix_ptrs;
        assert_eq!(main.partitioned_main_trace.len(), matrix_ptrs.len());
        let partitioned_main = matrix_ptrs
            .iter()
            .map(|ptr| {
                CommittedSingleMatrixView::new(main_pcs_data[ptr.commit_index].1, ptr.matrix_index)
            })
            .collect_vec();

        // There will be either 0 or 1 after_challenge traces
        let after_challenge = if let Some((cumulative_sum, index)) = cumulative_sum_and_index {
            let matrix = CommittedSingleMatrixView::new(&after_challenge_pcs_data[0].1, index);
            let exposed_values = vec![cumulative_sum];
            vec![(matrix, exposed_values)]
        } else {
            Vec::new()
        };
        let trace_view = SingleRapCommittedTraceView {
            domain,
            preprocessed,
            partitioned_main,
            after_challenge,
        };
        (rap, trace_view)
    })
    .unzip();
    // === END of logic specific to Interactions/permutations, we can now deal with general RAP ===

    prove_raps_with_committed_traces_with_groups(
        c,
        prover,
        challenger,
        pk,
        raps,
        trace_views,
        &challenges,
        public_values,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn prove_raps_with_committed_traces_with_groups<'a, SC: StarkGenericConfig>(
    c: &mut Criterion,
    prover: &MultiTraceStarkProver<SC>,
    challenger: &mut SC::Challenger,
    pk: &'a MultiStarkProvingKey<SC>,
    raps: Vec<&'a dyn AnyRap<SC>>,
    trace_views: Vec<SingleRapCommittedTraceView<'a, SC>>,
    challenges: &[Vec<SC::Challenge>],
    public_values: &'a [Vec<Val<SC>>],
) where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let pcs = prover.config.pcs();

    // Generate `alpha` challenge
    let alpha: SC::Challenge = challenger.sample_ext_element();
    tracing::debug!("alpha: {alpha:?}");
    let quotient_committer = QuotientCommitter::new(pcs, challenges, alpha);

    let mut group = c.benchmark_group("calc_quot_values");
    group.sample_size(10);
    group.bench_function("quotient poly", |b| {
        b.iter(|| {
            let _ = quotient_committer.quotient_values(
                raps.clone(),
                pk,
                trace_views.clone(),
                public_values,
            );
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = perm_trace_gen_benchmark
}
criterion_main!(benches);
