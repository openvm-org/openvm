use super::DummyHashChip;

use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_field::AbstractField;

#[test]
fn test_single_dummy_hash() {
    let chip: DummyHashChip<5, 3> = DummyHashChip {
        bus_index: 0,
        width: 5,
        rate: 3,
    };
    let x = [1, 2, 3, 4, 5]
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();
    let y = [1, 2, 3]
        .iter()
        .map(|x| AbstractField::from_canonical_u32(*x))
        .collect();

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}
