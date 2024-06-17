use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    verifier::VerificationError,
};
use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::StarkEngine,
    utils::create_seeded_rng,
};
use rand::Rng;

use crate::common::page::Page;
use crate::inner_join::{self, controller};

#[allow(clippy::too_many_arguments)]
fn load_tables_test(
    engine: &BabyBearPoseidon2Engine,
    t1: &Page,
    t2: &Page,
    decomp: usize,
    ij_controller: &mut controller::InnerJoinController<BabyBearPoseidon2Config>,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
    intersector_trace_degree: usize,
) -> Result<(), VerificationError> {
    // Clearing the range_checker counts
    ij_controller.update_range_checker(decomp);

    let (table_traces, mut prover_data) = ij_controller.load_tables(
        t1,
        t2,
        intersector_trace_degree,
        &mut trace_builder.committer,
    );

    let range_checker_trace = ij_controller.range_checker.generate_trace();

    trace_builder.clear();

    trace_builder.load_cached_trace(table_traces.t1_main_trace, prover_data.remove(0));
    trace_builder.load_cached_trace(table_traces.t2_main_trace, prover_data.remove(0));
    trace_builder.load_cached_trace(table_traces.output_main_trace, prover_data.remove(0));
    trace_builder.load_trace(table_traces.t1_aux_trace);
    trace_builder.load_trace(table_traces.t2_aux_trace);
    trace_builder.load_trace(table_traces.output_aux_trace);
    trace_builder.load_trace(table_traces.intersector_trace);
    trace_builder.load_trace(range_checker_trace);

    trace_builder.commit_current();

    let partial_vk = partial_pk.partial_vk();

    let main_trace_data = trace_builder.view(
        &partial_vk,
        vec![
            &ij_controller.t1_chip,
            &ij_controller.t2_chip,
            &ij_controller.output_chip,
            &ij_controller.intersector_chip,
            &ij_controller.range_checker.air,
        ],
    );

    let pis = vec![vec![]; partial_vk.per_air.len()];

    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    verifier.verify(
        &mut challenger,
        partial_vk,
        vec![
            &ij_controller.t1_chip,
            &ij_controller.t2_chip,
            &ij_controller.output_chip,
            &ij_controller.intersector_chip,
            &ij_controller.range_checker.air,
        ],
        proof,
        &pis,
    )
}

#[test]
fn inner_join_test() {
    let mut rng = create_seeded_rng();

    let range_bus_index = 0;
    let t1_intersector_bus_index = 1;
    let t2_intersector_bus_index = 2;
    let intersector_t2_bus_index = 3;
    let t1_output_bus_index = 4;
    let t2_output_bus_index = 5;

    use inner_join::controller::InnerJoinController;

    const MAX_VAL: u32 = 0x78000001 / 2; // The prime used by BabyBear / 2

    let log_t1_height = 4;
    let log_t2_height = 3;

    let t1_idx_len = rng.gen::<usize>() % 2 + 2;
    let t1_data_len = rng.gen::<usize>() % 2 + 2;

    let t2_idx_len = rng.gen::<usize>() % 2 + 2;
    let t2_data_len = rng.gen::<usize>() % 2 + t1_idx_len;

    let t1_height = 1 << log_t1_height;
    let t2_height = 1 << log_t2_height;

    let intersector_trace_degree = 2 * t1_height.max(t2_height);

    let fkey_start = rng.gen::<usize>() % (t2_data_len - t1_idx_len);
    let fkey_end = fkey_start + t1_idx_len;

    let idx_limb_bits = 10;
    let decomp = 4;
    let max_idx = 1 << idx_limb_bits;

    // Generating a fully-allocated random table t1 with primary key
    let t1 = Page::random(
        &mut rng,
        t1_idx_len,
        t1_data_len,
        max_idx,
        MAX_VAL,
        t1_height,
        t1_height,
    );

    // Generating a fully-allocated random table t2
    let mut t2 = Page::random(
        &mut rng,
        t2_idx_len,
        t2_data_len,
        max_idx,
        MAX_VAL,
        t2_height,
        t2_height,
    );

    // Assigning foreign key in t2 rows
    for row in t2.rows.iter_mut() {
        row.data[fkey_start..fkey_end].clone_from_slice(&t1.get_random_idx(&mut rng));
    }

    let mut ij_controller: InnerJoinController<BabyBearPoseidon2Config> = InnerJoinController::new(
        range_bus_index,
        t1_intersector_bus_index,
        t2_intersector_bus_index,
        intersector_t2_bus_index,
        t1_output_bus_index,
        t2_output_bus_index,
        fkey_start,
        fkey_end,
        t1_idx_len,
        t1_data_len,
        t2_idx_len,
        t2_data_len,
        idx_limb_bits,
        decomp,
    );

    let engine = config::baby_bear_poseidon2::default_engine(
        decomp.max(log_t1_height.max(log_t2_height) + 1),
    );
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let t1_main_ptr = keygen_builder.add_cached_main_matrix(ij_controller.t1_chip.table_width());
    let t2_main_ptr = keygen_builder.add_cached_main_matrix(ij_controller.t2_chip.table_width());
    let output_main_ptr =
        keygen_builder.add_cached_main_matrix(ij_controller.output_chip.table_width());
    let t1_aux_ptr = keygen_builder.add_main_matrix(ij_controller.t1_chip.aux_width());
    let t2_aux_ptr = keygen_builder.add_main_matrix(ij_controller.t2_chip.aux_width());
    let output_aux_ptr = keygen_builder.add_main_matrix(ij_controller.output_chip.aux_width());

    keygen_builder.add_partitioned_air(
        &ij_controller.t1_chip,
        t1_height,
        0,
        vec![t1_main_ptr, t1_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &ij_controller.t2_chip,
        t2_height,
        0,
        vec![t2_main_ptr, t2_aux_ptr],
    );

    keygen_builder.add_partitioned_air(
        &ij_controller.output_chip,
        t2_height,
        0,
        vec![output_main_ptr, output_aux_ptr],
    );

    keygen_builder.add_air(&ij_controller.intersector_chip, intersector_trace_degree, 0);
    keygen_builder.add_air(&ij_controller.range_checker.air, 1 << decomp, 0);

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    // Testing a fully allocated tables
    load_tables_test(
        &engine,
        &t1,
        &t2,
        decomp,
        &mut ij_controller,
        &mut trace_builder,
        &partial_pk,
        intersector_trace_degree,
    )
    .expect("Verification failed");

    // Making a test where foreign key sometimes doesn't exist in t1
    for row in t2.rows.iter_mut() {
        if rng.gen::<bool>() {
            row.data[fkey_start..fkey_end].clone_from_slice(
                (fkey_start..fkey_end)
                    .map(|_| rng.gen::<u32>() % MAX_VAL)
                    .collect::<Vec<u32>>()
                    .as_slice(),
            );
        }
    }

    load_tables_test(
        &engine,
        &t1,
        &t2,
        decomp,
        &mut ij_controller,
        &mut trace_builder,
        &partial_pk,
        intersector_trace_degree,
    )
    .expect("Verification failed");

    // Making a test where foreign key always doens't exist in t1
    // This should produce a fully-unallocated output page
    for row in t2.rows.iter_mut() {
        row.data[fkey_start..fkey_end].clone_from_slice(
            (fkey_start..fkey_end)
                .map(|_| rng.gen::<u32>() % MAX_VAL)
                .collect::<Vec<u32>>()
                .as_slice(),
        );
    }

    load_tables_test(
        &engine,
        &t1,
        &t2,
        decomp,
        &mut ij_controller,
        &mut trace_builder,
        &partial_pk,
        intersector_trace_degree,
    )
    .expect("Verification failed");

    // Testing partially-allocated t1 and t2, where foreign key sometimes
    // doesn't exist in t1
    let t1 = Page::random(
        &mut rng,
        t1_idx_len,
        t1_data_len,
        max_idx,
        MAX_VAL,
        t1_height,
        t1_height / 2,
    );

    let mut t2 = Page::random(
        &mut rng,
        t2_idx_len,
        t2_data_len,
        max_idx,
        MAX_VAL,
        t2_height,
        t2_height / 2,
    );

    for row in t2.rows.iter_mut() {
        if rng.gen::<bool>() {
            row.data[fkey_start..fkey_end].clone_from_slice(&t1.get_random_idx(&mut rng));
        }
    }

    load_tables_test(
        &engine,
        &t1,
        &t2,
        decomp,
        &mut ij_controller,
        &mut trace_builder,
        &partial_pk,
        intersector_trace_degree,
    )
    .expect("Verification failed");
}
