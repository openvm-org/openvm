use std::collections::HashSet;
use std::panic;

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
    utils::create_seeded_rng,
};
use rand::Rng;

use crate::common::{page::Page, page_cols::PageCols};
use crate::execution_air::ExecutionAir;
use crate::page_rw_checker::{
    self,
    page_controller::{self, OpType, Operation},
};

#[allow(clippy::too_many_arguments)]
fn load_page_test(
    engine: &BabyBearPoseidon2Engine,
    page_init: &Page,
    idx_decomp: usize,
    ops: &[Operation],
    page_controller: &mut page_controller::PageController<BabyBearPoseidon2Config>,
    ops_sender: &ExecutionAir,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
    trace_degree: usize,
    exec_trace_degree: usize,
    spacing: usize,
) -> Result<(), VerificationError> {
    let page_height = page_init.height();
    assert!(page_height > 0);

    // Clearing the range_checker counts
    page_controller.reset_range_checker(idx_decomp);

    let (init_page_pdata, final_page_pdata) = page_controller.load_page_and_ops(
        page_init,
        None,
        None,
        ops.to_vec(),
        trace_degree,
        &mut trace_builder.committer,
    );

    // Generating trace for ops_sender and making sure it has height exec_trace_degree
    let ops_sender_trace = ops_sender.generate_trace_testing(ops, exec_trace_degree, spacing);

    let proof = page_controller.prove(
        engine,
        partial_pk,
        trace_builder,
        init_page_pdata,
        final_page_pdata,
        ops_sender,
        ops_sender_trace,
    );

    page_controller.verify(engine, partial_pk.partial_vk(), proof, ops_sender)
}

#[test]
fn execution_air_test() {
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

    let trace_degree = num_ops * 4;

    let idx_len = rng.gen::<usize>() % ((page_width - 1) - 1) + 1;
    let data_len = (page_width - 1) - idx_len;
    let idx_limb_bits = 10;
    let idx_decomp = 4;
    let max_idx = 1 << idx_limb_bits;

    // Generating a random page with distinct indices
    let mut initial_page = Page::random(
        &mut rng,
        idx_len,
        data_len,
        max_idx,
        MAX_VAL,
        page_height,
        page_height,
    );

    // We will generate the final page from the initial page below
    // while generating the operations
    let mut final_page = initial_page.clone();

    let clks: Vec<usize> = (1..num_ops + 1).collect();

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
        if op_type == OpType::Write && rng.gen::<u32>() % 2 == 0 {
            idx = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
        }

        let data = {
            if op_type == OpType::Read {
                final_page[&idx].clone()
            } else if op_type == OpType::Write {
                (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect()
            } else {
                vec![0; data_len]
            }
        };

        if op_type == OpType::Write {
            if final_page.contains(&idx) {
                final_page[&idx].clone_from(&data);
            } else {
                final_page.insert(&idx, &data);
            }
        } else if op_type == OpType::Delete {
            final_page.delete(&idx);
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
    let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
    // let ops_sender = DummyInteractionAir::new(idx_len + data_len + 2, true, ops_bus_index);

    let engine = config::baby_bear_poseidon2::default_engine(
        idx_decomp.max(log_page_height.max(2 + log_num_ops)),
    );
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    page_controller.set_up_keygen_builder(
        &mut keygen_builder,
        page_height,
        trace_degree,
        &ops_sender,
        4 * num_ops,
    );

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // Testing a fully allocated page
    load_page_test(
        &engine,
        &initial_page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        4 * num_ops,
        1,
    )
    .expect("Verification failed");

    // Testing a partially-allocated page
    let rows_allocated = rng.gen::<usize>() % (page_height + 1);
    for i in rows_allocated..page_height {
        // Making sure the first operation using this index is a write
        let idx = initial_page.rows[i].idx.clone();
        for op in ops.iter_mut() {
            if op.idx == idx {
                op.op_type = OpType::Write;
                break;
            }
        }

        // Zeroing out the row
        initial_page.rows[i] = PageCols::from_slice(
            vec![0; idx_len + data_len + 1].as_slice(),
            idx_len,
            data_len,
        );
    }

    load_page_test(
        &engine,
        &initial_page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        4 * num_ops,
        4,
    )
    .expect("Verification failed");

    // Testing a fully unallocated page
    for i in 0..page_height {
        // Making sure the first operation that uses every index is a write
        let idx = initial_page.rows[i].idx.clone();
        for op in ops.iter_mut() {
            if op.idx == idx {
                op.op_type = OpType::Write;
                break;
            }
        }

        initial_page.rows[i] = PageCols::from_slice(
            vec![0; 1 + idx_len + data_len].as_slice(),
            idx_len,
            data_len,
        );
    }

    load_page_test(
        &engine,
        &initial_page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        4 * num_ops,
        1,
    )
    .expect("Verification failed");

    // Testing writing only 1 index into an unallocated page
    ops = vec![Operation::new(
        1,
        (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect(),
        (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect(),
        OpType::Write,
    )];

    load_page_test(
        &engine,
        &initial_page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        4 * num_ops,
        1,
    )
    .expect("Verification failed");

    // Making a test where we write, delete, write, then read an idx
    // in a fully-unallocated page
    let idx: Vec<u32> = (0..idx_len).map(|_| rng.gen::<u32>() % max_idx).collect();
    let data_1: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
    let data_2: Vec<u32> = (0..data_len).map(|_| rng.gen::<u32>() % MAX_VAL).collect();
    ops = vec![
        Operation::new(1, idx.clone(), data_1.clone(), OpType::Write),
        Operation::new(2, idx.clone(), vec![0; data_len], OpType::Delete),
        Operation::new(3, idx.clone(), data_2.clone(), OpType::Write),
        Operation::new(4, idx, data_2, OpType::Read),
    ];

    load_page_test(
        &engine,
        &initial_page,
        idx_decomp,
        &ops,
        &mut page_controller,
        &ops_sender,
        &mut trace_builder,
        &partial_pk,
        trace_degree,
        4 * num_ops,
        1,
    )
    .expect("Verification failed");

    // Negative tests

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

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
            &initial_page,
            idx_decomp,
            &ops,
            &mut page_controller,
            &ops_sender,
            &mut trace_builder,
            &partial_pk,
            trace_degree,
            4 * num_ops,
            1,
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
            &initial_page,
            idx_decomp,
            &ops,
            &mut page_controller,
            &ops_sender,
            &mut trace_builder,
            &partial_pk,
            trace_degree,
            4 * num_ops,
            1,
        );
    });

    assert!(
        result.is_err(),
        "Expected to fail when allocating too many indices"
    );
}
