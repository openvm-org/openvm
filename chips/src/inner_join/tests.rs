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
    utils::create_seeded_rng,
};
use rand::Rng;

use crate::inner_join::{
    self,
    controller::{self, T2Format, TableFormat},
};
use crate::{common::page::Page, inner_join::controller::IJBuses};

#[allow(clippy::too_many_arguments)]
fn load_tables_test(
    engine: &BabyBearPoseidon2Engine,
    t1: &Page,
    t2: &Page,
    decomp: usize,
    ij_controller: &mut controller::FKInnerJoinController<BabyBearPoseidon2Config>,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
    intersector_trace_degree: usize,
) -> Result<(), VerificationError> {
    // Clearing the range_checker counts
    ij_controller.reset_range_checker(decomp);

    let prover_data = ij_controller.load_tables(
        t1,
        t2,
        None,
        None,
        None,
        intersector_trace_degree,
        &mut trace_builder.committer,
    );
    let proof = ij_controller.prove(engine, partial_pk, trace_builder, prover_data);
    ij_controller.verify(engine, partial_pk.partial_vk(), proof)
}

#[test]
fn inner_join_test() {
    let mut rng = create_seeded_rng();

    let ij_buses = IJBuses {
        range_bus_index: 0,
        t1_intersector_bus_index: 1,
        t2_intersector_bus_index: 2,
        intersector_t2_bus_index: 3,
        t1_output_bus_index: 4,
        t2_output_bus_index: 5,
    };

    use inner_join::controller::FKInnerJoinController;

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
    for row in t2.iter_mut() {
        row.data[fkey_start..fkey_end].clone_from_slice(&t1.get_random_idx(&mut rng));
    }

    let mut ij_controller: FKInnerJoinController<BabyBearPoseidon2Config> =
        FKInnerJoinController::new(
            ij_buses,
            TableFormat::new(t1_idx_len, t1_data_len, idx_limb_bits),
            T2Format::new(
                TableFormat::new(t2_idx_len, t2_data_len, idx_limb_bits),
                fkey_start,
                fkey_end,
            ),
            decomp,
        );

    let engine = config::baby_bear_poseidon2::default_engine(
        decomp.max(log_t1_height.max(log_t2_height) + 1),
    );
    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    ij_controller.set_up_keygen_builder(&mut keygen_builder);

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
    for row in t2.iter_mut() {
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
    for row in t2.iter_mut() {
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

    for row in t2.iter_mut() {
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
