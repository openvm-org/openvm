use std::collections::{HashMap, HashSet};

use afs_test_utils::{
    config::baby_bear_blake3::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use rand::RngCore;

use crate::memory::{
    expand::MemoryDimensions,
    interface::{
        columns::MemoryInterfaceCols, MemoryInterfaceChip, EXPAND_BUS, MEMORY_INTERFACE_BUS,
    },
    OpType::{Read, Write},
};

const DEFAULT_CHUNK: usize = 8;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = MemoryInterfaceCols::<DEFAULT_CHUNK, usize>::get_width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = MemoryInterfaceCols::<DEFAULT_CHUNK, _>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    assert_eq!(flattened, all_cols);

    assert_eq!(num_cols, flattened.len());
}

fn random_test<const CHUNK: usize>(
    height: usize,
    max_value: usize,
    mut num_initial_addresses: usize,
    mut num_touched_addresses: usize,
) {
    let mut rng = create_seeded_rng();
    let mut next_usize = || rng.next_u64() as usize;

    macro_rules! next_bool {
        () => {
            next_usize() & 1 == 0
        };
    }
    macro_rules! next_value {
        () => {
            BabyBear::from_canonical_usize(next_usize() % max_value)
        };
    }

    let mut seen_addresses = HashSet::new();

    let mut initial_memory = HashMap::new();
    let mut final_memory = HashMap::new();
    let memory_dimensions = MemoryDimensions {
        as_height: 1,
        address_height: height,
        as_offset: 1,
    };
    let mut chip = MemoryInterfaceChip::<CHUNK, _>::new(memory_dimensions);

    let mut touched_leaves = HashSet::new();

    let mut dummy_offline_checker_trace_rows = vec![];
    let mut offline_checker_interaction =
        |is_final: bool, address_space: BabyBear, address: BabyBear, value: BabyBear| {
            let expand_direction = if is_final {
                BabyBear::neg_one()
            } else {
                BabyBear::one()
            };
            dummy_offline_checker_trace_rows.push(BabyBear::two() * expand_direction);
            dummy_offline_checker_trace_rows.extend(&[
                expand_direction,
                address_space,
                address,
                value,
            ]);
        };

    while num_initial_addresses != 0 || num_touched_addresses != 0 {
        let address = (
            BabyBear::from_canonical_usize((next_usize() & 1) + 1),
            BabyBear::from_canonical_usize(next_usize() % (CHUNK << height)),
        );
        if seen_addresses.insert(address) {
            let is_initial = next_bool!();
            let initial_value = if is_initial {
                next_value!()
            } else {
                BabyBear::zero()
            };
            let is_touched = next_bool!();

            if num_initial_addresses != 0 && is_initial {
                num_initial_addresses -= 1;
                initial_memory.insert(address, initial_value);
                final_memory.insert(address, initial_value);
            }

            if is_touched && num_touched_addresses != 0 {
                num_touched_addresses -= 1;
                let leaf_label = (address.1.as_canonical_u64() as usize) / CHUNK;
                touched_leaves.insert((address.0.as_canonical_u64() as usize, leaf_label));

                let initially_read = next_bool!();
                let new_value = if initially_read {
                    initial_value
                } else {
                    next_value!()
                };
                final_memory.insert(address, new_value);
                chip.touch_address(
                    address.0,
                    address.1,
                    if initially_read { Read } else { Write },
                    initial_value,
                );
                if initially_read {
                    offline_checker_interaction(false, address.0, address.1, initial_value);
                }

                if next_bool!() {
                    let read_now = next_bool!();
                    let final_value = if initially_read {
                        new_value
                    } else {
                        next_value!()
                    };
                    final_memory.insert(address, final_value);
                    chip.touch_address(
                        address.0,
                        address.1,
                        if read_now { Read } else { Write },
                        new_value,
                    );

                    offline_checker_interaction(true, address.0, address.1, final_value);
                } else {
                    offline_checker_interaction(true, address.0, address.1, new_value);
                }
            }
        }
    }

    let mut dummy_expand_trace_rows = vec![];
    let mut expand_interaction =
        |is_final: bool, address_space: usize, node_label: usize, hash: [BabyBear; CHUNK]| {
            dummy_expand_trace_rows.push(if is_final {
                BabyBear::neg_one()
            } else {
                BabyBear::one()
            });
            dummy_expand_trace_rows.push(BabyBear::from_bool(is_final));
            dummy_expand_trace_rows.push(BabyBear::zero());
            dummy_expand_trace_rows.push(BabyBear::from_canonical_usize(
                (address_space - memory_dimensions.as_offset) << memory_dimensions.address_height,
            ));
            dummy_expand_trace_rows.push(BabyBear::from_canonical_usize(node_label));
            dummy_expand_trace_rows.extend(hash);
        };

    for (address_space, label) in touched_leaves {
        for (is_final, memory) in [(false, &initial_memory), (true, &final_memory)] {
            let values = std::array::from_fn(|i| {
                *memory
                    .get(&(
                        BabyBear::from_canonical_usize(address_space),
                        BabyBear::from_canonical_usize((CHUNK * label) + i),
                    ))
                    .unwrap_or(&BabyBear::zero())
            });
            expand_interaction(is_final, address_space, label, values);
        }
    }

    let dummy_offline_checker_air = DummyInteractionAir::new(4, true, MEMORY_INTERFACE_BUS);
    while !(dummy_offline_checker_trace_rows.len() / (dummy_offline_checker_air.field_width() + 1))
        .is_power_of_two()
    {
        dummy_offline_checker_trace_rows.push(BabyBear::zero());
    }
    let dummy_offline_checker_trace = RowMajorMatrix::new(
        dummy_offline_checker_trace_rows,
        dummy_offline_checker_air.field_width() + 1,
    );
    let dummy_expand_air = DummyInteractionAir::new(4 + CHUNK, false, EXPAND_BUS);

    while !(dummy_expand_trace_rows.len() / (dummy_expand_air.field_width() + 1)).is_power_of_two()
    {
        dummy_expand_trace_rows.push(BabyBear::zero());
    }
    let dummy_expand_trace =
        RowMajorMatrix::new(dummy_expand_trace_rows, dummy_expand_air.field_width() + 1);

    let trace = chip.generate_trace(&final_memory, chip.get_trace_height().next_power_of_two());

    run_simple_test_no_pis(
        vec![&chip.air, &dummy_offline_checker_air, &dummy_expand_air],
        vec![trace, dummy_offline_checker_trace, dummy_expand_trace],
    )
    .expect("Verification failed");
}

#[test]
fn memory_interface_test_1() {
    random_test::<DEFAULT_CHUNK>(10, 3000, 400, 30);
}

#[test]
fn memory_interface_test_2() {
    random_test::<DEFAULT_CHUNK>(3, 3000, 3, 2);
}

#[test]
#[should_panic]
fn memory_interface_negative_test() {
    let mut chip = MemoryInterfaceChip::<DEFAULT_CHUNK, _>::new(MemoryDimensions {
        as_height: 1,
        address_height: 3,
        as_offset: 1,
    });
    chip.touched_leaves.insert((BabyBear::one(), 0));
    let trace = chip.generate_trace(&HashMap::new(), chip.get_trace_height().next_power_of_two());
    run_simple_test_no_pis(vec![&chip.air], vec![trace]).expect("This should occur");
}
