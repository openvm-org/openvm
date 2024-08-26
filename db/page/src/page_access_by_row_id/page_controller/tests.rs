use std::iter;

use afs_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    prover::{trace::TraceCommitmentBuilder, USE_DEBUG_BUILDER},
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

use super::PageController;

type Val = BabyBear;

#[allow(clippy::too_many_arguments)]
fn load_page_test(
    engine: &BabyBearPoseidon2Engine,
    page_to_receive: &[Vec<u32>],
    page_to_send: &[Vec<u32>],
    page_controller: &mut PageController<BabyBearPoseidon2Config>,
    page_requester: &DummyInteractionAir,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    pk: &MultiStarkProvingKey<BabyBearPoseidon2Config>,
    num_requests: usize,
) -> Result<(), VerificationError> {
    let mut rng = create_seeded_rng();

    let page_height = page_to_receive.len();
    assert!(page_height > 0);
    let page_width = page_to_receive[0].len();

    let requests = (0..num_requests)
        .map(|_| rng.gen::<usize>() % page_height)
        .collect::<Vec<usize>>();

    let (page_trace, prover_data) =
        page_controller.load_page(&mut trace_builder.committer, page_to_receive.to_vec());

    let requester_trace = RowMajorMatrix::new(
        requests
            .iter()
            .flat_map(|i| {
                page_controller.request(*i);
                iter::once(1)
                    .chain(iter::once(*i as u32))
                    .chain(page_to_send[*i].clone())
            })
            .map(Val::from_wrapped_u32)
            .collect(),
        2 + page_width,
    );

    let page_metadata_trace = page_controller.generate_trace();

    trace_builder.clear();

    trace_builder.load_cached_trace(page_trace, prover_data);
    trace_builder.load_trace(page_metadata_trace);
    trace_builder.load_trace(requester_trace);

    trace_builder.commit_current();

    let vk = pk.vk();

    let main_trace_data =
        trace_builder.view(&vk, vec![&page_controller.page_access_air, page_requester]);

    let pis = vec![vec![]; vk.per_air.len()];

    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    verifier.verify(&mut challenger, &vk, &proof, &pis)
}

#[test]
fn page_access_chip_test() {
    let mut rng = create_seeded_rng();
    let bus_index = 0;

    use super::PageController;

    let log_page_height = 3;
    let log_num_requests = 5;

    let page_width = 4;
    let page_height = 1 << log_page_height;
    let num_requests: usize = 1 << log_num_requests;

    let pages = (0..2)
        .map(|_| {
            (0..page_height)
                .map(|_| {
                    (0..page_width)
                        .map(|_| rng.gen::<u32>())
                        .collect::<Vec<u32>>()
                })
                .collect::<Vec<Vec<u32>>>()
        })
        .collect::<Vec<Vec<Vec<u32>>>>();

    let mut page_controller = PageController::new(bus_index);
    let page_requester = DummyInteractionAir::new(1 + page_width, true, bus_index);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(log_num_requests));

    let mut keygen_builder = engine.keygen_builder();

    let page_data_ptr = keygen_builder.add_cached_main_matrix(page_width);
    let page_metadata_ptr = keygen_builder.add_main_matrix(2);
    keygen_builder.add_partitioned_air(
        &page_controller.page_access_air,
        0,
        vec![page_data_ptr, page_metadata_ptr],
    );

    keygen_builder.add_air(&page_requester, 0);

    let pk = keygen_builder.generate_pk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    load_page_test(
        &engine,
        &pages[0],
        &pages[0],
        &mut page_controller,
        &page_requester,
        &mut trace_builder,
        &pk,
        num_requests,
    )
    .expect("Verification failed");

    load_page_test(
        &engine,
        &pages[1],
        &pages[1],
        &mut page_controller,
        &page_requester,
        &mut trace_builder,
        &pk,
        num_requests,
    )
    .expect("Verification failed");

    // We expect failure, so we turn off debug assertions
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    let result = load_page_test(
        &engine,
        &pages[0],
        &pages[1],
        &mut page_controller,
        &page_requester,
        &mut trace_builder,
        &pk,
        num_requests,
    );

    assert_eq!(
        result,
        Err(VerificationError::NonZeroCumulativeSum),
        "Verification failed"
    );
}
