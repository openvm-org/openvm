use std::sync::Arc;

use benchmark::utils::bench::{gen_ops_sender_trace, generate_page_and_ops, get_dummy_ptd};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use pprof::criterion::{Output, PProfProfiler};

use afs_page::page_rw_checker::page_controller::PageController;
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    interaction::dummy_interaction_air::DummyInteractionAir,
};

pub fn trace_gen_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace gen");
    group.sample_size(100);

    let idx_len = 16;
    let data_len = 512;
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

    group.bench_function("trace gen", |b| {
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
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = trace_gen_benchmark
}
criterion_main!(benches);
