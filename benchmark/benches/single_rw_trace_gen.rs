use std::iter;
use std::{collections::HashSet, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;

use afs_page::common::page::Page;
use afs_page::page_rw_checker::page_controller::{OpType, Operation, PageController};
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    prover::{
        trace::{TraceCommitmentBuilder, TraceCommitter},
        MultiTraceStarkProver,
    },
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};

type Val = BabyBear;

pub fn trace_gen_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace gen");
    group.sample_size(10);

    let idx_len = 16;
    let data_len = 512;
    let log_page_height = 15;
    let log_num_ops = 16;
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

fn generate_page_and_ops(
    idx_len: usize,
    data_len: usize,
    page_height: usize,
    num_ops: usize,
    max_idx: u32,
    max_data: u32,
) -> (Page, Vec<Operation>) {
    let mut rng = create_seeded_rng();

    // Generating a random page with distinct indices
    let initial_page = Page::random(
        &mut rng,
        idx_len,
        data_len,
        max_idx,
        max_data,
        page_height,
        page_height,
    );

    // We will generate the final page from the initial page below
    // while generating the operations
    let mut final_page = initial_page.clone();

    // Generating random sorted distinct timestamps for operations
    let mut clks = HashSet::new();
    while clks.len() < num_ops {
        clks.insert(rng.gen::<usize>() % (max_data as usize - 2) + 1);
    }
    let mut clks: Vec<usize> = clks.into_iter().collect();
    clks.sort();

    let mut ops: Vec<Operation> = vec![];
    for &clk in clks.iter() {
        let op_type = {
            if rng.gen::<u32>() % 3 == 0 {
                OpType::Read
            } else if rng.gen::<u32>() % 3 == 1 {
                OpType::Write
            } else {
                OpType::Delete
            }
        };

        let mut idx = final_page.get_random_idx(&mut rng);

        // if this is a write operation, make it an insert sometimes
        if op_type == OpType::Write && rng.gen::<u32>() % 2 == 0 && !final_page.is_full() {
            idx = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
        }

        let data = {
            if op_type == OpType::Read {
                final_page[&idx].clone()
            } else if op_type == OpType::Write {
                (0..data_len).map(|_| rng.gen::<u32>() % max_data).collect()
            } else {
                vec![0; data_len]
            }
        };

        if op_type == OpType::Write {
            if final_page.contains(&idx) {
                final_page[black_box(&idx)].clone_from(&data);
            } else {
                final_page.insert(&idx, &data);
            }
        } else if op_type == OpType::Delete {
            final_page.delete(black_box(&idx));
        }

        ops.push(Operation::new(clk, idx, data, op_type));
    }

    (initial_page, ops)
}

fn gen_ops_sender_trace(
    ops_sender: &DummyInteractionAir,
    ops: &[Operation],
) -> RowMajorMatrix<Val> {
    RowMajorMatrix::new(
        ops.iter()
            .flat_map(|op| {
                iter::once(Val::one())
                    .chain(iter::once(Val::from_canonical_usize(op.clk)))
                    .chain(iter::once(Val::from_canonical_u8(op.op_type as u8)))
                    .chain(op.idx.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(op.data.iter().map(|x| Val::from_canonical_u32(*x)))
            })
            .collect(),
        1 + ops_sender.field_width(),
    )
}

fn get_dummy_ptd(
    trace_committer: &mut TraceCommitter<BabyBearPoseidon2Config>,
) -> ProverTraceData<BabyBearPoseidon2Config> {
    let simple_trace = RowMajorMatrix::new_col(vec![Val::from_canonical_u32(1)]);
    trace_committer.commit(black_box(vec![simple_trace]))
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = trace_gen_benchmark
}
criterion_main!(benches);
