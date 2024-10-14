use std::collections::{HashMap, HashSet};

use ax_sdk::{
    config::baby_bear_blake3::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use rand::RngCore;

use crate::{
    kernels::core::{EXPAND_BUS, MEMORY_BUS, WORD_SIZE},
    system::memory::{
        expand_interface::{columns::MemoryExpandInterfaceCols, MemoryExpandInterfaceChip},
        manager::{access_cell::AccessCell, dimensions::MemoryDimensions},
    },
};

const TEST_CHUNK: usize = 8;
const TEST_NUM_WORDS: usize = TEST_CHUNK / WORD_SIZE;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = MemoryExpandInterfaceCols::<TEST_NUM_WORDS, WORD_SIZE, usize>::width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered =
        MemoryExpandInterfaceCols::<TEST_NUM_WORDS, WORD_SIZE, _>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    assert_eq!(flattened, all_cols);
    assert_eq!(num_cols, flattened.len());
}

fn random_test<const CHUNK: usize, const NUM_WORDS: usize>(
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
    let mut chip = MemoryExpandInterfaceChip::<NUM_WORDS, WORD_SIZE, _>::new(memory_dimensions);

    let mut touched_leaves = HashSet::new();

    let mut dummy_offline_checker_trace_rows = vec![];
    let mut offline_checker_interaction =
        |is_final: bool,
         address_space: BabyBear,
         address: BabyBear,
         value: [BabyBear; WORD_SIZE]| {
            let expand_direction: BabyBear = if is_final {
                BabyBear::neg_one()
            } else {
                BabyBear::one()
            };
            dummy_offline_checker_trace_rows.push(BabyBear::two() * expand_direction);
            dummy_offline_checker_trace_rows.extend(&[expand_direction, address_space, address]);
            dummy_offline_checker_trace_rows.extend(value);
        };

    while num_initial_addresses != 0 || num_touched_addresses != 0 {
        let address = (
            BabyBear::from_canonical_usize((next_usize() & 1) + 1),
            BabyBear::from_canonical_usize(next_usize() % (CHUNK << height)),
        );
        if seen_addresses.insert(address) {
            let initial_value = [next_value!(); WORD_SIZE];
            let is_touched = next_bool!();

            if num_initial_addresses != 0 {
                num_initial_addresses -= 1;
                initial_memory.insert(
                    address,
                    AccessCell {
                        data: initial_value,
                        clk: BabyBear::zero(),
                    },
                );
                final_memory.insert(
                    address,
                    AccessCell {
                        data: initial_value,
                        clk: BabyBear::zero(),
                    },
                );
            }

            if is_touched && num_touched_addresses != 0 {
                num_touched_addresses -= 1;
                let leaf_label = (address.1.as_canonical_u64() as usize) / CHUNK;
                touched_leaves.insert((address.0.as_canonical_u64() as usize, leaf_label));

                let new_value = [next_value!(); WORD_SIZE];
                final_memory.insert(
                    address,
                    AccessCell {
                        data: new_value,
                        clk: BabyBear::zero(),
                    },
                );
                chip.touch_address(address.0, address.1, initial_value, BabyBear::zero());

                // if next_bool!() {
                //     // let read_now = next_bool!();
                //     // let final_value = if initially_read {
                //     //     new_value
                //     // } else {
                //     //     [next_value!(); WORD_SIZE]
                //     // };
                //     // final_memory.insert(
                //     //     address,
                //     //     AccessCell {
                //     //         data: final_value,
                //     //         clk: BabyBear::zero(),
                //     //     },
                //     // );
                //     // chip.touch_address(address.0, address.1, new_value, BabyBear::zero());

                //     offline_checker_interaction(true, address.0, address.1, final_value);
                // } else {
                //     offline_checker_interaction(true, address.0, address.1, new_value);
                // }
            }
        }
    }

    let mut dummy_expand_trace_rows = vec![];
    let mut expand_interaction =
        |is_final: bool,
         address_space: usize,
         node_label: usize,
         hash: [[BabyBear; WORD_SIZE]; NUM_WORDS]| {
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
            dummy_expand_trace_rows.extend(hash.into_iter().flatten());
        };

    for (address_space, label) in touched_leaves {
        for (is_final, memory) in [(false, &initial_memory), (true, &final_memory)] {
            let values = std::array::from_fn(|i| {
                memory
                    .get(&(
                        BabyBear::from_canonical_usize(address_space),
                        BabyBear::from_canonical_usize((CHUNK * label) + i),
                    ))
                    .map_or([BabyBear::zero(); WORD_SIZE], |cell| cell.data)
            });
            expand_interaction(is_final, address_space, label, values);
        }
    }

    let dummy_offline_checker_air = DummyInteractionAir::new(4, true, MEMORY_BUS.0);
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
    random_test::<TEST_CHUNK, TEST_NUM_WORDS>(10, 3000, 400, 30);
}

#[test]
fn memory_interface_test_2() {
    random_test::<TEST_CHUNK, TEST_NUM_WORDS>(3, 3000, 3, 2);
}

#[test]
#[should_panic]
fn memory_interface_negative_test() {
    let mut chip =
        MemoryExpandInterfaceChip::<TEST_CHUNK, TEST_NUM_WORDS, _>::new(MemoryDimensions {
            as_height: 1,
            address_height: 3,
            as_offset: 1,
        });
    chip.touched_leaves.insert((BabyBear::one(), 0));
    let trace = chip.generate_trace(&HashMap::new(), chip.get_trace_height().next_power_of_two());
    run_simple_test_no_pis(vec![&chip.air], vec![trace]).expect("This should occur");
}
