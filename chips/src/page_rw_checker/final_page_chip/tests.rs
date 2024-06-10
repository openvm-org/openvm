use std::collections::HashSet;
use std::{iter, sync::Arc};

use crate::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    keygen::{types::MultiStarkPartialProvingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver, USE_DEBUG_BUILDER},
    verifier::VerificationError,
};
use afs_test_utils::{
    config::{
        self, baby_bear_poseidon2::BabyBearPoseidon2Config,
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

type Val = BabyBear;

fn test_single_page(
    engine: &BabyBearPoseidon2Engine,
    page: Vec<Vec<u32>>,
    final_page_chip: &super::FinalPageChip,
    range_checker: Arc<RangeCheckerGateChip>,
    page_sender: &DummyInteractionAir,
    trace_builder: &mut TraceCommitmentBuilder<BabyBearPoseidon2Config>,
    partial_pk: &MultiStarkPartialProvingKey<BabyBearPoseidon2Config>,
) -> Result<(), VerificationError> {
    let page_width = page[0].len();

    let page_trace = final_page_chip.gen_page_trace::<BabyBearPoseidon2Config>(page.clone());
    let page_prover_data = trace_builder.committer.commit(vec![page_trace.clone()]);

    let aux_trace = final_page_chip
        .gen_aux_trace::<BabyBearPoseidon2Config>(page.clone(), range_checker.clone());
    let range_checker_trace = range_checker.generate_trace();

    let page_receiver_trace = RowMajorMatrix::new(
        page.iter()
            .flat_map(|row| {
                row.iter()
                    .cloned()
                    .map(Val::from_canonical_u32)
                    .collect::<Vec<Val>>()
            })
            .collect(),
        page_width,
    );

    trace_builder.clear();

    trace_builder.load_cached_trace(page_trace, page_prover_data);
    trace_builder.load_trace(aux_trace);
    trace_builder.load_trace(range_checker_trace);
    trace_builder.load_trace(page_receiver_trace);

    trace_builder.commit_current();

    let partial_vk = partial_pk.partial_vk();

    let main_trace_data = trace_builder.view(
        &partial_vk,
        vec![final_page_chip, &*range_checker, page_sender],
    );

    let pis = vec![vec![]; partial_vk.per_air.len()];

    let prover = engine.prover();
    let verifier = engine.verifier();

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &partial_pk, main_trace_data, &pis);

    let mut challenger = engine.new_challenger();
    verifier.verify(
        &mut challenger,
        partial_vk,
        vec![final_page_chip, &*range_checker, page_sender],
        proof,
        &pis,
    )
}

#[test]
fn final_page_chip_test() {
    let mut rng = create_seeded_rng();
    let page_bus_index = 0;
    let sorted_bus_index = 1;

    use super::FinalPageChip;

    let log_page_height = 3;

    let page_width = 6;
    let page_height = 1 << log_page_height;

    let idx_len = rng.gen::<usize>() % ((page_width - 1) - 1) + 1;
    let data_len = (page_width - 1) - idx_len;

    let idx_limb_bits = 5;
    let idx_decomp = 2;

    let max_val: u32 = 1 << idx_limb_bits;

    let allocated_rows = ((page_height as f64) * (3.0 / 4.0)) as usize;

    // Creating a list of sorted distinct indices
    let mut all_idx = HashSet::new();
    while all_idx.len() < allocated_rows {
        all_idx.insert(
            (0..idx_len)
                .map(|_| (rng.gen::<u32>() % max_val))
                .collect::<Vec<u32>>(),
        );
    }
    let mut all_idx: Vec<Vec<u32>> = all_idx.into_iter().collect();
    all_idx.sort();

    let mut page: Vec<Vec<u32>> = (0..page_height)
        .map(|x| {
            if x < allocated_rows {
                iter::once(1)
                    .chain(all_idx[x].iter().cloned())
                    .chain((0..data_len).map(|_| (rng.gen::<u32>() % max_val)))
                    .collect()
            } else {
                iter::once(0)
                    .chain((0..idx_len + data_len).map(|_| (rng.gen::<u32>() % max_val)))
                    .collect()
            }
        })
        .collect();

    let final_page_chip = FinalPageChip::new(
        page_bus_index,
        sorted_bus_index,
        idx_len,
        data_len,
        idx_limb_bits,
        idx_decomp,
    );
    let range_checker = Arc::new(RangeCheckerGateChip::new(
        sorted_bus_index,
        1 << idx_limb_bits,
    ));
    let page_sender = DummyInteractionAir::new(page_width - 1, true, page_bus_index);

    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(idx_limb_bits));

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&engine.config);

    let page_data_ptr = keygen_builder.add_cached_main_matrix(final_page_chip.page_width());
    let page_aux_ptr = keygen_builder.add_main_matrix(final_page_chip.aux_width());
    keygen_builder.add_partitioned_air(
        &final_page_chip,
        page_height,
        0,
        vec![page_data_ptr, page_aux_ptr],
    );

    keygen_builder.add_air(&*range_checker, 1 << idx_limb_bits, 0);
    keygen_builder.add_air(&page_sender, page_height, 0);

    let partial_pk = keygen_builder.generate_partial_pk();

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

    test_single_page(
        &engine,
        page.clone(),
        &final_page_chip,
        range_checker.clone(),
        &page_sender,
        &mut trace_builder,
        &partial_pk,
    )
    .expect("Verification Failed");

    // Swap the first two rows of the page so it's no longer sorted
    page.swap(0, 1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        test_single_page(
            &engine,
            page,
            &final_page_chip,
            range_checker.clone(),
            &page_sender,
            &mut trace_builder,
            &partial_pk,
        ),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
