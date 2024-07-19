use std::collections::{HashMap, HashSet};

use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use rand::RngCore;

use afs_test_utils::config::baby_bear_blake3::run_simple_test_no_pis;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::create_seeded_rng;

use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::tests::util::HashTestChip;
use crate::memory::expand::{ExpandChip, EXPAND_BUS};
use crate::memory::tree::trees_from_full_memory;

mod util;

const DEFAULT_CHUNK: usize = 8;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = ExpandCols::<DEFAULT_CHUNK, usize>::get_width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ExpandCols::<DEFAULT_CHUNK, _>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn test<const CHUNK: usize>(
    height: usize,
    initial_memory: &HashMap<(BabyBear, BabyBear), BabyBear>,
    touched_addresses: HashSet<(BabyBear, BabyBear)>,
    final_memory: &HashMap<(BabyBear, BabyBear), BabyBear>,
) {
    // checking validity of test data
    for (address, value) in final_memory {
        assert!((address.0.as_canonical_u64() as usize) < (1 << height));
        if initial_memory.get(address) != Some(value) {
            assert!(touched_addresses.contains(address));
        }
    }
    for address in initial_memory.keys() {
        assert!(final_memory.contains_key(address));
    }
    for address in touched_addresses.iter() {
        assert!(final_memory.contains_key(address));
    }

    let address_spaces = [BabyBear::one(), BabyBear::two()];
    let initial_trees = trees_from_full_memory(
        height,
        &address_spaces,
        initial_memory,
        &mut HashTestChip::new(),
    );
    let final_trees_check = trees_from_full_memory(
        height,
        &address_spaces,
        final_memory,
        &mut HashTestChip::new(),
    );

    let mut chip = ExpandChip::new(height, initial_trees.clone());
    for &(address_space, address) in touched_addresses.iter() {
        chip.touch_address(address_space, address);
    }

    let trace_degree = chip.get_trace_height().next_power_of_two();
    let mut hash_test_chip = HashTestChip::new();
    let (trace, final_trees) =
        chip.generate_trace_and_final_tree(final_memory, trace_degree, &mut hash_test_chip);

    assert_eq!(final_trees, final_trees_check);

    let dummy_interaction_air = DummyInteractionAir::new(4 + CHUNK, false, EXPAND_BUS);
    let mut dummy_interaction_trace_rows = vec![];
    let mut interaction = |receive: bool,
                           is_final: bool,
                           address_space: BabyBear,
                           height: usize,
                           node_label: usize,
                           hash: [BabyBear; CHUNK]| {
        dummy_interaction_trace_rows.push(if receive {
            BabyBear::one()
        } else {
            BabyBear::neg_one()
        });
        dummy_interaction_trace_rows.push(BabyBear::from_bool(is_final));
        dummy_interaction_trace_rows.push(address_space);
        dummy_interaction_trace_rows.push(BabyBear::from_canonical_usize(height));
        dummy_interaction_trace_rows.push(BabyBear::from_canonical_usize(node_label));
        dummy_interaction_trace_rows.extend(hash);
    };
    for (address_space, root) in initial_trees {
        interaction(false, false, address_space, height, 0, root.hash());
    }
    for (address_space, root) in final_trees_check {
        interaction(true, true, address_space, height, 0, root.hash());
    }
    let touched_leaves: HashSet<_> = touched_addresses
        .iter()
        .map(|(address_space, address)| {
            (address_space, (address.as_canonical_u64() as usize) / CHUNK)
        })
        .collect();
    for (&address_space, label) in touched_leaves {
        let initial_values = std::array::from_fn(|i| {
            *initial_memory
                .get(&(
                    address_space,
                    BabyBear::from_canonical_usize((CHUNK * label) + i),
                ))
                .unwrap_or(&BabyBear::zero())
        });
        interaction(true, false, address_space, 0, label, initial_values);
        let final_values = std::array::from_fn(|i| {
            *final_memory
                .get(&(
                    address_space,
                    BabyBear::from_canonical_usize((CHUNK * label) + i),
                ))
                .unwrap_or(&BabyBear::zero())
        });
        interaction(false, true, address_space, 0, label, final_values);
    }

    while !(dummy_interaction_trace_rows.len() / (dummy_interaction_air.field_width() + 1))
        .is_power_of_two()
    {
        dummy_interaction_trace_rows.push(BabyBear::zero());
    }
    let dummy_interaction_trace = RowMajorMatrix::new(
        dummy_interaction_trace_rows,
        dummy_interaction_air.field_width() + 1,
    );

    run_simple_test_no_pis(
        vec![&chip.air(), &dummy_interaction_air, &hash_test_chip.air()],
        vec![trace, dummy_interaction_trace, hash_test_chip.trace()],
    )
    .expect("Verification failed");
}

fn random_test<const CHUNK: usize>(
    height: usize,
    max_value: usize,
    mut num_initial_addresses: usize,
    mut num_touched_addresses: usize,
) {
    let mut rng = create_seeded_rng();
    let mut next_usize = || rng.next_u64() as usize;

    let mut initial_memory = HashMap::new();
    let mut final_memory = HashMap::new();
    let mut seen_addresses = HashSet::new();
    let mut touched_addresses = HashSet::new();

    while num_initial_addresses != 0 || num_touched_addresses != 0 {
        let address = (
            BabyBear::from_canonical_usize((next_usize() & 1) + 1),
            BabyBear::from_canonical_usize(next_usize() % (CHUNK << height)),
        );
        if seen_addresses.insert(address) {
            let is_initial = next_usize() & 1 == 0;
            let initial_value = BabyBear::from_canonical_usize(next_usize() % max_value);
            let is_touched = next_usize() & 1 == 0;
            let value_changes = next_usize() & 1 == 0;
            let changed_value = BabyBear::from_canonical_usize(next_usize() % max_value);

            if is_initial && num_initial_addresses != 0 {
                num_initial_addresses -= 1;
                initial_memory.insert(address, initial_value);
                final_memory.insert(address, initial_value);
            }
            if is_touched && num_touched_addresses != 0 {
                num_touched_addresses -= 1;
                touched_addresses.insert(address);
                if value_changes || !is_initial {
                    final_memory.insert(address, changed_value);
                }
            }
        }
    }

    test::<CHUNK>(height, &initial_memory, touched_addresses, &final_memory);
}

#[test]
fn expand_test_1() {
    random_test::<DEFAULT_CHUNK>(10, 3000, 400, 30);
}

#[test]
fn expand_test_2() {
    random_test::<DEFAULT_CHUNK>(3, 3000, 3, 2);
}

#[test]
#[should_panic]
fn expand_negative_test() {
    let height = 1;

    let address_spaces = [BabyBear::one()];
    let memory = HashMap::new();
    let trees = trees_from_full_memory::<DEFAULT_CHUNK, _>(
        height,
        &address_spaces,
        &memory,
        &mut HashTestChip::new(),
    );

    let chip = ExpandChip::new(height, trees.clone());

    let trace_degree = 2;
    let mut hash_test_chip = HashTestChip::new();
    let (trace, _) = chip.generate_trace_and_final_tree(&memory, trace_degree, &mut hash_test_chip);

    run_simple_test_no_pis(
        vec![&chip.air(), &hash_test_chip.air()],
        vec![trace, hash_test_chip.trace()],
    )
    .expect("This should occur");
}
