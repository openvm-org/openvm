use std::collections::{HashMap, HashSet};
use std::{iter, panic};

use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver, USE_DEBUG_BUILDER},
    verifier::VerificationError,
};
use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use crate::common::page::Page;
use crate::common::page_cols::PageCols;
use crate::page_rw_checker::{
    self,
    page_controller::{self, OpType, Operation},
};

type Val = BabyBear;

#[allow(clippy::too_many_arguments)]
fn load_page_test(
    engine: &BabyBearPoseidon2Engine,
    page_init: &Page,
    idx_decomp: usize,
    ops: &Vec<Operation>,
    page_controller: &mut page_controller::PageController<BabyBearPoseidon2Config>,
    ops_sender: &DummyInteractionAir,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
    trace_degree: usize,
    num_ops: usize,
) -> Result<(), VerificationError> {
    let page_height = page_init.height();
    assert!(page_height > 0);

    let (page_traces, mut prover_data) = page_controller.load_page_and_ops(
        &page_init,
        ops.clone(),
        trace_degree,
        &mut trace_builder.committer,
    );

    let offline_checker_trace = page_controller.offline_checker_trace();
    let final_page_aux_trace = page_controller.final_page_aux_trace();
    let range_checker_trace = page_controller.range_checker_trace();

    // Generating trace for ops_sender and making sure it has height num_ops
    let ops_sender_trace = RowMajorMatrix::new(
        ops.iter()
            .flat_map(|op| {
                iter::once(Val::one())
                    .chain(iter::once(Val::from_canonical_usize(op.clk)))
                    .chain(op.idx.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(op.data.iter().map(|x| Val::from_canonical_u32(*x)))
                    .chain(iter::once(Val::from_canonical_u8(op.op_type.clone() as u8)))
            })
            .chain(
                iter::repeat_with(|| iter::repeat(Val::zero()).take(1 + ops_sender.field_width()))
                    .take(num_ops - ops.len())
                    .flatten(),
            )
            .collect(),
        1 + ops_sender.field_width(),
    );

    // Clearing the range_checker counts
    page_controller.update_range_checker(idx_decomp);

    trace_builder.clear();

    trace_builder.load_cached_trace(page_traces[0].clone(), prover_data.remove(0));
    trace_builder.load_cached_trace(page_traces[1].clone(), prover_data.remove(0));
    trace_builder.load_trace(final_page_aux_trace);
    trace_builder.load_trace(offline_checker_trace.clone());
    trace_builder.load_trace(range_checker_trace);
    trace_builder.load_trace(ops_sender_trace);

    trace_builder.commit_current();

    let partial_vk = partial_pk.partial_vk();

    let main_trace_data = trace_builder.view(
        &partial_vk,
        vec![
            &page_controller.init_chip,
            &page_controller.final_chip,
            &page_controller.offline_checker,
            &page_controller.range_checker.air,
            ops_sender,
        ],
    );

    let pis = vec![vec![]; partial_vk.per_air.len()];

    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    let result = verifier.verify(
        &mut challenger,
        partial_vk,
        vec![
            &page_controller.init_chip,
            &page_controller.final_chip,
            &page_controller.offline_checker,
            &page_controller.range_checker.air,
            ops_sender,
        ],
        proof,
        &pis,
    );

    result
}

#[test]
fn page_read_write_test() {
    let mut rng = create_seeded_rng();

    let page_bus_index = 0;
    let range_bus_index = 1;
    let ops_bus_index = 2;

    use page_rw_checker::page_controller::PageController;

    const MAX_VAL: u32 = 0x78000001 / 2; // The prime used by BabyBear / 2

    let log_page_height = 4;
    let log_num_ops = 3;

    let page_width = 6;
    let page_height = 1 << log_page_height;
    let num_ops: usize = 1 << log_num_ops;

    let trace_degree = num_ops * 8;

    let idx_len = rng.gen::<usize>() % ((page_width - 1) - 1) + 1;
    let data_len = (page_width - 1) - idx_len;
    let idx_limb_bits = 10;
    let idx_decomp = 4;
    let max_idx = 1 << idx_limb_bits;

    // Generating a random page with distinct indices
    let mut page: Vec<Vec<u32>> = vec![];
    let mut idx_data_map = HashMap::new();
    for _ in 0..page_height {
        let mut idx;
        loop {
            idx = (0..idx_len)
                .map(|_| rng.gen::<u32>() % max_idx)
                .collect::<Vec<u32>>();
            if !idx_data_map.contains_key(&idx) {
                break;
            }
        }

        let data: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
        idx_data_map.insert(idx.clone(), data.clone());
        page.push(iter::once(1).chain(idx).chain(data).collect());
    }

    let mut page = Page::from_2d_vec(&page, idx_len, data_len);

    // Generating random sorted distinct timestamps for operations
    let mut clks = HashSet::new();
    while clks.len() < num_ops {
        clks.insert(rng.gen::<usize>() % (MAX_VAL as usize - 2) + 1);
    }
    let mut clks: Vec<usize> = clks.into_iter().collect();
    clks.sort();

    let mut ops: Vec<Operation> = vec![];
    for i in 0..num_ops {
        let clk = clks[i];
        let idx = idx_data_map
            .iter()
            .nth(rng.gen::<usize>() % idx_data_map.len())
            .unwrap()
            .0
            .to_vec();

        let op_type = {
            if rng.gen::<bool>() {
                OpType::Read
            } else {
                OpType::Write
            }
        };

        let data = {
            if op_type == OpType::Read {
                idx_data_map[&idx].to_vec()
            } else {
                (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
            }
        };

        if op_type == OpType::Write {
            idx_data_map.insert(idx.clone(), data.clone());
        }

        ops.push(Operation::new(clk, idx, data, op_type));
    }

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

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(3 + log_num_ops));
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let init_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let final_page_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let final_page_aux_ptr = keygen_builder.add_main_matrix(page_controller.final_chip.aux_width());
    let offline_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.offline_checker.air_width());
    let range_checker_ptr =
        keygen_builder.add_main_matrix(page_controller.range_checker.air_width());
    let ops_sender_ptr = keygen_builder.add_main_matrix(1 + ops_sender.field_width());

    keygen_builder.add_partitioned_air(
        &page_controller.init_chip,
        page_height,
        0,
        vec![init_page_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.final_chip,
        page_height,
        0,
        vec![final_page_ptr, final_page_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.offline_checker,
        trace_degree,
        0,
        vec![offline_checker_ptr],
    );

    keygen_builder.add_partitioned_air(
        &page_controller.range_checker.air,
        1 << idx_decomp,
        0,
        vec![range_checker_ptr],
    );

    keygen_builder.add_partitioned_air(&ops_sender, num_ops, 0, vec![ops_sender_ptr]);

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // Testing a fully allocated page
    load_page_test(
        &engine,
        &page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        num_ops,
    )
    .expect("Verification failed");

    // Testing a partially-allocated page
    let rows_allocated = rng.gen::<usize>() % (page_height + 1);
    for i in rows_allocated..page_height {
        // Making sure the first operation using this index is a write
        let idx = page.rows[i].idx.clone();
        for op in ops.iter_mut() {
            if op.idx == idx {
                op.op_type = OpType::Write;
                break;
            }
        }

        // Zeroing out the row
        page.rows[i] = PageCols::from_slice(
            vec![0; idx_len + data_len + 1].as_slice(),
            idx_len,
            data_len,
        );
    }

    load_page_test(
        &engine,
        &page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        num_ops,
    )
    .expect("Verification failed");

    // Testing a fully unallocated page
    for i in 0..page_height {
        // Making sure the first operation that uses every index is a write
        let idx = page[i].idx.clone();
        for op in ops.iter_mut() {
            if op.idx == idx {
                op.op_type = OpType::Write;
                break;
            }
        }

        page.rows[i] = PageCols::from_slice(
            vec![0; 1 + idx_len + data_len].as_slice(),
            idx_len,
            data_len,
        );
    }

    load_page_test(
        &engine,
        &page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        num_ops,
    )
    .expect("Verification failed");

    // Testing writing only 1 index into an unallocated page
    ops = vec![Operation::new(
        10,
        (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect(),
        (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
        OpType::Write,
    )];

    load_page_test(
        &engine,
        &page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        num_ops,
    )
    .expect("Verification failed");

    // Negative tests

    // Testing reading from a non-existing index (in a fully-unallocated page)
    ops = vec![Operation::new(
        1,
        (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect(),
        (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
        OpType::Read,
    )];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        load_page_test(
            &engine,
            &page,
            idx_decomp,
            &ops,
            &mut page_controller,
            &ops_sender,
            &mut trace_builder,
            &partial_pk,
            trace_degree,
            num_ops,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraints to fail"
    );

    // Testing reading wrong data from an existing index
    let idx: Vec<u32> = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
    let data_1: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
    let mut data_2 = data_1.clone();
    data_2[0] += 1; // making sure data_2 is different

    ops = vec![
        Operation::new(1, idx.clone(), data_1, OpType::Write),
        Operation::new(2, idx, data_2, OpType::Read),
    ];

    assert_eq!(
        load_page_test(
            &engine,
            &page,
            idx_decomp,
            &ops,
            &mut page_controller,
            &ops_sender,
            &mut trace_builder,
            &partial_pk,
            trace_degree,
            num_ops,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraints to fail"
    );

    // Testing writing too many indices to a fully unallocated page
    let mut idx_map = HashSet::new();
    for _ in 0..page_height + 1 {
        let mut idx: Vec<u32>;
        loop {
            idx = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
            if !idx_map.contains(&idx) {
                break;
            }
        }

        idx_map.insert(idx);
    }

    ops.clear();
    for (i, idx) in idx_map.iter().enumerate() {
        ops.push(Operation::new(
            i + 1,
            idx.clone(),
            (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
            OpType::Write,
        ));
    }

    let engine_ref = &engine;
    let result = panic::catch_unwind(move || {
        let _ = load_page_test(
            engine_ref,
            &page,
            idx_decomp,
            &ops,
            &mut page_controller,
            &ops_sender,
            &mut trace_builder,
            &partial_pk,
            trace_degree,
            num_ops,
        );
    });

    assert!(
        result.is_err(),
        "Expected to fail when allocating too many indices"
    );
}
