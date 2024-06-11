use super::DummyHashAir;

use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;

#[test]
fn test_single_dummy_hash() {
    let chip: DummyHashAir = DummyHashAir {
        bus_index: 0,
        hash_width: 5,
        rate: 3,
    };
    let requester = DummyInteractionAir::new(chip.get_width(), true, 0);
    let x = [1, 2, 3, 4, 5]
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();
    let y = [1, 2, 3]
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();

    let hash_trace = chip.generate_trace(vec![x], vec![y]);
    let mut dummy_trace = RowMajorMatrix::default(chip.get_width() + 1, 1);
    dummy_trace.values[0] = AbstractField::from_canonical_u32(1);
    for i in 1..=chip.get_width() {
        dummy_trace.values[i] = hash_trace.values[i - 1];
    }

    run_simple_test_no_pis(vec![&chip, &requester], vec![hash_trace, dummy_trace])
        .expect("Verification failed");
}
