use std::{
    array,
    collections::{BTreeMap, BTreeSet, HashSet},
};

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
    merkle::{
        columns::MemoryMerkleCols, tests::util::HashTestChip, MemoryDimensions, MemoryMerkleBus,
        MemoryMerkleChip,
    },
    tree::MemoryNode,
    Equipartition,
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
    initial_memory: &Equipartition<BabyBear, CHUNK>,
    touched_chunks: BTreeSet<(BabyBear, usize)>,
    final_memory: &Equipartition<BabyBear, CHUNK>,
) {
    let MemoryDimensions {
        as_height,
        address_height,
        as_offset,
    } = memory_dimensions;
    let merkle_bus = MemoryMerkleBus(20);

    // checking validity of test data
    for (&(address_space, chunk_label), value) in final_memory {
        assert!((address_space.as_canonical_u32() as usize) - as_offset < (1 << as_height));
        assert!(chunk_label < (1 << address_height));
        if initial_memory.get(&(address_space, chunk_label)) != Some(value) {
            assert!(touched_chunks.contains(&(address_space, chunk_label)));
        }
    }
    for key in initial_memory.keys() {
        assert!(final_memory.contains_key(key));
    }
    for &(address_space, chunk_label) in touched_chunks.iter() {
        assert!(final_memory.contains_key(&(address_space, chunk_label)));
    }

    let mut hash_test_chip = HashTestChip::new();

    let initial_tree =
        MemoryNode::tree_from_memory(memory_dimensions, initial_memory, &hash_test_chip);
    let final_tree_check =
        MemoryNode::tree_from_memory(memory_dimensions, final_memory, &hash_test_chip);

    let mut chip = MemoryMerkleChip::<CHUNK, _>::new(memory_dimensions, merkle_bus);
    for &(address_space, chunk_label) in touched_chunks.iter() {
        for i in 0..CHUNK {
            chip.touch_address(
                address_space,
                BabyBear::from_canonical_usize(chunk_label * CHUNK + i),
            );
        }
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
                           as_label: u64,
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
            BabyBear::from_canonical_u64(as_label),
            BabyBear::from_canonical_usize(address_label),
        ]);
        dummy_interaction_trace_rows.extend(hash);
    };

    for (address_space, chunk_label) in touched_chunks {
        let initial_values = *initial_memory
            .get(&(address_space, chunk_label))
            .unwrap_or(&[BabyBear::zero(); CHUNK]);

        let as_label = memory_dimensions.as_to_as_label(address_space);
        interaction(
            InteractionType::Send,
            false,
            0,
            as_label,
            chunk_label * 2,
            initial_values,
        );
        interaction(
            InteractionType::Send,
            false,
            0,
            as_label,
            chunk_label * 2 + 1,
            [BabyBear::zero(); CHUNK],
        );
        let final_values = *final_memory.get(&(address_space, chunk_label)).unwrap();
        interaction(
            InteractionType::Send,
            true,
            0,
            as_label,
            chunk_label * 2,
            final_values,
        );
        interaction(
            InteractionType::Send,
            true,
            0,
            as_label,
            chunk_label * 2 + 1,
            [BabyBear::zero(); CHUNK],
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

    let mut initial_memory = Equipartition::new();
    let mut final_memory = Equipartition::new();
    let mut seen_labels = HashSet::new();
    let mut touched_chunks = BTreeSet::new();

    while num_initial_addresses != 0 || num_touched_addresses != 0 {
        let address_space = BabyBear::from_canonical_usize((next_usize() & 1) + 1);
        let chunk_label = next_usize() % (1 << height);

        if seen_labels.insert(chunk_label) {
            let is_initial = next_usize() & 1 == 0;
            let initial_values =
                array::from_fn(|_| BabyBear::from_canonical_usize(next_usize() % max_value));
            let is_touched = next_usize() & 1 == 0;
            let value_changes = next_usize() & 1 == 0;

            if is_initial && num_initial_addresses != 0 {
                num_initial_addresses -= 1;
                initial_memory.insert((address_space, chunk_label), initial_values);
                final_memory.insert((address_space, chunk_label), initial_values);
            }
            if is_touched && num_touched_addresses != 0 {
                num_touched_addresses -= 1;
                touched_chunks.insert((address_space, chunk_label));
                if value_changes || !is_initial {
                    let changed_values = array::from_fn(|_| {
                        BabyBear::from_canonical_usize(next_usize() % max_value)
                    });
                    final_memory.insert((address_space, chunk_label), changed_values);
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
        touched_chunks,
        &final_memory,
    );
}

#[test]
fn expand_test_0() {
    random_test::<DEFAULT_CHUNK>(2, 3000, 2, 3);
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

    let memory = BTreeMap::new();
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

    let memory = Equipartition::new();
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
