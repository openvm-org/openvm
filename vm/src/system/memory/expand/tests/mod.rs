use std::collections::{HashMap, HashSet};

use afs_stark_backend::interaction::InteractionType;
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::RngCore;

use crate::system::memory::{
    expand::{
        columns::MemoryMerkleCols, tests::util::HashTestChip, MemoryDimensions, MemoryMerkleBus,
        MemoryMerkleChip,
    },
    tree::MemoryNode,
};

mod util;

const DEFAULT_CHUNK: usize = 8;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = MemoryMerkleCols::<DEFAULT_CHUNK, usize>::get_width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = MemoryMerkleCols::<DEFAULT_CHUNK, _>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn test<const CHUNK: usize>(
    memory_dimensions: MemoryDimensions,
    initial_memory: &HashMap<(BabyBear, BabyBear), BabyBear>,
    touched_addresses: HashSet<(BabyBear, BabyBear)>,
    final_memory: &HashMap<(BabyBear, BabyBear), BabyBear>,
) {
    let MemoryDimensions {
        as_height,
        address_height,
        as_offset,
    } = memory_dimensions;
    let merkle_bus = MemoryMerkleBus(20);

    // checking validity of test data
    for (address, value) in final_memory {
        assert!((address.0.as_canonical_u32() as usize) - as_offset < (1 << as_height));
        assert!((address.1.as_canonical_u32() as usize) < (CHUNK << address_height));
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

    let mut hash_test_chip = HashTestChip::new();

    let initial_tree =
        MemoryNode::tree_from_memory(memory_dimensions, initial_memory, &hash_test_chip);
    let final_tree_check =
        MemoryNode::tree_from_memory(memory_dimensions, final_memory, &hash_test_chip);

    let mut chip = MemoryMerkleChip::<CHUNK, _>::new(memory_dimensions, merkle_bus);
    for &(address_space, address) in touched_addresses.iter() {
        chip.touch_address(address_space, address);
    }

    println!("trace height = {}", chip.current_height());
    let (trace, final_tree) =
        chip.generate_trace_and_final_tree(&initial_tree, final_memory, &mut hash_test_chip);

    assert_eq!(final_tree, final_tree_check);

    let dummy_interaction_air = DummyInteractionAir::new(4 + CHUNK, true, merkle_bus.0);
    let mut dummy_interaction_trace_rows = vec![];
    let mut interaction = |interaction_type: InteractionType,
                           is_compress: bool,
                           height: usize,
                           as_label: usize,
                           address_label: usize,
                           hash: [BabyBear; CHUNK]| {
        let expand_direction = if is_compress {
            BabyBear::neg_one()
        } else {
            BabyBear::one()
        };
        dummy_interaction_trace_rows.push(match interaction_type {
            InteractionType::Send => expand_direction,
            InteractionType::Receive => -expand_direction,
        });
        dummy_interaction_trace_rows.extend([
            expand_direction,
            BabyBear::from_canonical_usize(height),
            BabyBear::from_canonical_usize(as_label),
            BabyBear::from_canonical_usize(address_label),
        ]);
        dummy_interaction_trace_rows.extend(hash);
    };

    let touched_leaves: HashSet<_> = touched_addresses
        .iter()
        .map(|(address_space, address)| {
            (
                ((address_space.as_canonical_u32() as usize) - as_offset) << address_height,
                (address.as_canonical_u32() as usize) / CHUNK,
            )
        })
        .collect();
    for (as_label, address_label) in touched_leaves {
        let initial_values = std::array::from_fn(|i| {
            *initial_memory
                .get(&(
                    BabyBear::from_canonical_usize((as_label >> address_height) + as_offset),
                    BabyBear::from_canonical_usize((CHUNK * address_label) + i),
                ))
                .unwrap_or(&BabyBear::zero())
        });
        interaction(
            InteractionType::Send,
            false,
            0,
            as_label,
            address_label,
            initial_values,
        );
        let final_values = std::array::from_fn(|i| {
            *final_memory
                .get(&(
                    BabyBear::from_canonical_usize((as_label >> address_height) + as_offset),
                    BabyBear::from_canonical_usize((CHUNK * address_label) + i),
                ))
                .unwrap_or(&BabyBear::zero())
        });
        interaction(
            InteractionType::Send,
            true,
            0,
            as_label,
            address_label,
            final_values,
        );
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

    let mut public_values = vec![vec![]; 3];
    public_values[0].extend(initial_tree.hash());
    public_values[0].extend(final_tree_check.hash());

    let hash_test_chip_air = hash_test_chip.air();
    BabyBearPoseidon2Engine::run_simple_test_fast(
        any_rap_arc_vec![chip.air, dummy_interaction_air, hash_test_chip_air],
        vec![trace, dummy_interaction_trace, hash_test_chip.trace()],
        public_values,
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

    test::<CHUNK>(
        MemoryDimensions {
            as_height: 1,
            address_height: height,
            as_offset: 1,
        },
        &initial_memory,
        touched_addresses,
        &final_memory,
    );
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
fn expand_test_no_accesses() {
    let memory_dimensions = MemoryDimensions {
        as_height: 2,
        address_height: 1,
        as_offset: 7,
    };
    let mut hash_test_chip = HashTestChip::new();

    let memory = HashMap::new();
    let tree = MemoryNode::<DEFAULT_CHUNK, _>::tree_from_memory(
        memory_dimensions,
        &memory,
        &hash_test_chip,
    );

    let mut chip: MemoryMerkleChip<DEFAULT_CHUNK, _> =
        MemoryMerkleChip::new(memory_dimensions, MemoryMerkleBus(20));

    let (trace, _) = chip.generate_trace_and_final_tree(&tree, &memory, &mut hash_test_chip);

    let mut public_values = vec![vec![]; 2];
    public_values[0].extend(tree.hash());
    public_values[0].extend(tree.hash());

    let hash_test_chip_air = hash_test_chip.air();
    BabyBearPoseidon2Engine::run_simple_test_fast(
        any_rap_arc_vec![chip.air, hash_test_chip_air],
        vec![trace, hash_test_chip.trace()],
        public_values,
    )
    .expect("This should occur");
}

#[test]
#[should_panic]
fn expand_test_negative() {
    let memory_dimensions = MemoryDimensions {
        as_height: 2,
        address_height: 1,
        as_offset: 7,
    };

    let mut hash_test_chip = HashTestChip::new();

    let memory = HashMap::new();
    let tree = MemoryNode::<DEFAULT_CHUNK, _>::tree_from_memory(
        memory_dimensions,
        &memory,
        &hash_test_chip,
    );

    let mut chip =
        MemoryMerkleChip::<DEFAULT_CHUNK, _>::new(memory_dimensions, MemoryMerkleBus(20));

    let (trace, _) = chip.generate_trace_and_final_tree(&tree, &memory, &mut hash_test_chip);
    let mut new_rows = vec![];
    for i in 0..trace.height() {
        let row: Vec<_> = trace.row(i).collect();
        let mut cols = MemoryMerkleCols::<DEFAULT_CHUNK, _>::from_slice(&row);
        if cols.expand_direction == BabyBear::neg_one() {
            cols.left_direction_different = BabyBear::zero();
            cols.right_direction_different = BabyBear::zero();
        }
        new_rows.extend(cols.flatten());
    }
    let new_trace = RowMajorMatrix::new(
        new_rows,
        MemoryMerkleCols::<DEFAULT_CHUNK, BabyBear>::get_width(),
    );

    let mut public_values = vec![vec![]; 2];
    public_values[0].extend(tree.hash());
    public_values[0].extend(tree.hash());

    let hash_test_chip_air = hash_test_chip.air();
    BabyBearPoseidon2Engine::run_simple_test_fast(
        any_rap_arc_vec![chip.air, hash_test_chip_air],
        vec![new_trace, hash_test_chip.trace()],
        public_values,
    )
    .expect("This should occur");
}
