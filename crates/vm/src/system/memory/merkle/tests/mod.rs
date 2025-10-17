use std::{
    array,
    borrow::BorrowMut,
    collections::{BTreeMap, BTreeSet, HashSet},
    sync::Arc,
};

use openvm_stark_backend::{
    interaction::{PermutationCheckBus, PermutationInteractionType},
    p3_field::FieldAlgebra,
    p3_matrix::dense::RowMajorMatrix,
    prover::types::AirProvingContext,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::RngCore;

use crate::{
    arch::{
        testing::{MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS},
        AddressSpaceHostConfig, MemoryCellType, MemoryConfig, ADDR_SPACE_OFFSET,
    },
    system::memory::{
        merkle::{
            memory_to_vec_partition, tests::util::HashTestChip, MemoryDimensions, MemoryMerkleChip,
            MemoryMerkleCols, MerkleTree,
        },
        online::{GuestMemory, LinearMemory},
        AddressMap, MemoryImage,
    },
};

mod util;

const CHUNK: usize = 8;
const COMPRESSION_BUS: PermutationCheckBus = PermutationCheckBus::new(POSEIDON2_DIRECT_BUS);
type F = BabyBear;

fn test(
    memory_dimensions: MemoryDimensions,
    initial_memory: &MemoryImage,
    touched_labels: BTreeSet<(u32, u32)>,
    final_memory: &MemoryImage,
) {
    let MemoryDimensions {
        addr_space_height,
        address_height,
    } = memory_dimensions;

    let merkle_bus = PermutationCheckBus::new(MEMORY_MERKLE_BUS);

    for address_space in 0..final_memory.config.len() {
        for pointer in 0..final_memory.mem[address_space].size() / 4 {
            if unsafe {
                initial_memory.get_f::<F>(address_space as u32, pointer as u32)
                    != final_memory.get_f(address_space as u32, pointer as u32)
            } {
                let label = (pointer / CHUNK) as u32;
                assert!(address_space - (ADDR_SPACE_OFFSET as usize) < (1 << addr_space_height));
                assert!(pointer < (CHUNK << address_height));
                assert!(touched_labels.contains(&(address_space as u32, label)));
            }
        }
    }

    let mut hash_test_chip = HashTestChip::new();

    let final_tree_check =
        MerkleTree::from_memory(final_memory, &memory_dimensions, &hash_test_chip);

    let mut chip =
        MemoryMerkleChip::<CHUNK, _>::new(memory_dimensions, merkle_bus, COMPRESSION_BUS);
    let final_partition: BTreeMap<_, [F; CHUNK]> =
        memory_to_vec_partition::<F, CHUNK>(final_memory, &memory_dimensions)
            .into_iter()
            .map(|(idx, values)| {
                let address_space =
                    (idx >> memory_dimensions.address_height) as u32 + ADDR_SPACE_OFFSET;
                let label = (idx & ((1 << memory_dimensions.address_height) - 1)) as u32;
                ((address_space, label * (CHUNK as u32)), values)
            })
            .collect();
    let final_partition = final_partition
        .into_iter()
        .filter(|((address_space, pointer), _)| {
            touched_labels.contains(&(*address_space, pointer / CHUNK as u32))
        })
        .collect();
    chip.finalize(initial_memory, &final_partition, &hash_test_chip);

    assert_eq!(
        chip.final_state.as_ref().unwrap().final_root,
        final_tree_check.root()
    );
    let chip_api = chip.generate_proving_ctx();

    let dummy_interaction_air = DummyInteractionAir::new(4 + CHUNK, true, merkle_bus.index);
    let mut dummy_interaction_trace_rows = vec![];
    let mut interaction = |interaction_type: PermutationInteractionType,
                           is_compress: bool,
                           height: usize,
                           as_label: u32,
                           address_label: u32,
                           hash: [BabyBear; CHUNK]| {
        let expand_direction = if is_compress {
            BabyBear::NEG_ONE
        } else {
            BabyBear::ONE
        };
        dummy_interaction_trace_rows.push(match interaction_type {
            PermutationInteractionType::Send => expand_direction,
            PermutationInteractionType::Receive => -expand_direction,
        });
        dummy_interaction_trace_rows.extend([
            expand_direction,
            BabyBear::from_canonical_usize(height),
            BabyBear::from_canonical_u32(as_label),
            BabyBear::from_canonical_u32(address_label),
        ]);
        dummy_interaction_trace_rows.extend(hash);
    };

    for (address_space, address_label) in touched_labels {
        let initial_values = unsafe {
            array::from_fn(|i| {
                initial_memory.get((address_space, address_label * CHUNK as u32 + i as u32))
            })
        };
        let as_label = address_space - ADDR_SPACE_OFFSET;
        interaction(
            PermutationInteractionType::Send,
            false,
            0,
            as_label,
            address_label,
            initial_values,
        );
        let final_values = *final_partition
            .get(&(address_space, address_label * (CHUNK as u32)))
            .unwrap();
        interaction(
            PermutationInteractionType::Send,
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
        dummy_interaction_trace_rows.push(BabyBear::ZERO);
    }
    let dummy_interaction_trace = RowMajorMatrix::new(
        dummy_interaction_trace_rows,
        dummy_interaction_air.field_width() + 1,
    );
    let dummy_interaction_api = AirProvingContext::simple_no_pis(Arc::new(dummy_interaction_trace));

    BabyBearPoseidon2Engine::run_test_fast(
        vec![
            Arc::new(chip.air),
            Arc::new(dummy_interaction_air),
            Arc::new(hash_test_chip.air()),
        ],
        vec![
            chip_api,
            dummy_interaction_api,
            hash_test_chip.generate_proving_ctx(),
        ],
    )
    .expect("Verification failed");
}

fn random_test(
    height: usize,
    max_value: u32,
    mut num_initial_addresses: usize,
    mut num_touched_addresses: usize,
) {
    let mut rng = create_seeded_rng();
    let mut next_u32 = || rng.next_u64() as u32;

    let mem_config = MemoryConfig::new(
        1,
        vec![
            AddressSpaceHostConfig {
                num_cells: 0,
                min_block_size: 0,
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: CHUNK << height,
                min_block_size: 1,
                layout: MemoryCellType::Native { size: 4 },
            },
            AddressSpaceHostConfig {
                num_cells: CHUNK << height,
                min_block_size: 1,
                layout: MemoryCellType::Native { size: 4 },
            },
        ],
        height + 3,
        20,
        17,
        32,
    );

    let mut initial_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
    let mut final_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));

    let mut seen = HashSet::new();
    let mut touched_labels = BTreeSet::new();

    while num_initial_addresses != 0 || num_touched_addresses != 0 {
        let address_space = (next_u32() & 1) + 1;
        let label = next_u32() % (1 << height);
        let pointer = label * CHUNK as u32 + (next_u32() % CHUNK as u32);

        if seen.insert(pointer) {
            let is_initial = next_u32() & 1 == 0;
            let is_touched = next_u32() & 1 == 0;
            let value_changes = next_u32() & 1 == 0;

            if is_initial && num_initial_addresses != 0 {
                num_initial_addresses -= 1;
                let value = BabyBear::from_canonical_u32(next_u32() % max_value);
                unsafe {
                    initial_memory.write(address_space, pointer, [value]);
                    final_memory.write(address_space, pointer, [value]);
                }
            }
            if is_touched && num_touched_addresses != 0 {
                num_touched_addresses -= 1;
                touched_labels.insert((address_space, label));
                if value_changes || !is_initial {
                    let value = BabyBear::from_canonical_u32(next_u32() % max_value);
                    unsafe {
                        final_memory.write(address_space, pointer, [value]);
                    }
                }
            }
        }
    }

    test(
        MemoryDimensions {
            addr_space_height: 1,
            address_height: height,
        },
        &initial_memory.memory,
        touched_labels,
        &final_memory.memory,
    );
}

#[test]
fn expand_test_0() {
    random_test(2, 3000, 2, 3);
}

#[test]
fn expand_test_1() {
    random_test(10, 3000, 400, 30);
}

#[test]
fn expand_test_2() {
    random_test(3, 3000, 3, 2);
}

#[test]
fn expand_test_no_accesses() {
    let mut hash_test_chip = HashTestChip::new();
    let height = 1;

    let mem_config = MemoryConfig::new(
        1,
        vec![
            AddressSpaceHostConfig {
                num_cells: 0,
                min_block_size: 0,
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: CHUNK << height,
                min_block_size: 1,
                layout: MemoryCellType::Native { size: 4 },
            },
            AddressSpaceHostConfig {
                num_cells: CHUNK << height,
                min_block_size: 1,
                layout: MemoryCellType::Native { size: 4 },
            },
        ],
        height + 3,
        20,
        17,
        32,
    );
    let md = mem_config.memory_dimensions();

    let memory = AddressMap::from_mem_config(&mem_config);

    let mut chip: MemoryMerkleChip<CHUNK, _> = MemoryMerkleChip::new(
        md,
        PermutationCheckBus::new(MEMORY_MERKLE_BUS),
        COMPRESSION_BUS,
    );

    chip.finalize(&memory, &BTreeMap::new(), &hash_test_chip);
    let trace = chip.generate_proving_ctx();
    BabyBearPoseidon2Engine::run_test_fast(
        vec![Arc::new(chip.air), Arc::new(hash_test_chip.air())],
        vec![trace, hash_test_chip.generate_proving_ctx()],
    )
    .expect("Empty touched memory doesn't work");
}

#[test]
#[should_panic]
fn expand_test_negative() {
    let mut hash_test_chip = HashTestChip::new();
    let height = 1;

    let mem_config = MemoryConfig::new(
        1,
        vec![
            AddressSpaceHostConfig {
                num_cells: 0,
                min_block_size: 0,
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: CHUNK << height,
                min_block_size: 1,
                layout: MemoryCellType::Native { size: 4 },
            },
            AddressSpaceHostConfig {
                num_cells: CHUNK << height,
                min_block_size: 1,
                layout: MemoryCellType::Native { size: 4 },
            },
        ],
        height + 3,
        20,
        17,
        32,
    );
    let md = mem_config.memory_dimensions();

    let memory = AddressMap::from_mem_config(&mem_config);

    let mut chip: MemoryMerkleChip<CHUNK, _> = MemoryMerkleChip::new(
        md,
        PermutationCheckBus::new(MEMORY_MERKLE_BUS),
        COMPRESSION_BUS,
    );

    chip.finalize(&memory, &BTreeMap::new(), &hash_test_chip);
    let mut chip_ctx = chip.generate_proving_ctx();
    {
        let mut trace = (*chip_ctx.clone().common_main.unwrap()).clone();
        for row in trace.rows_mut() {
            let row: &mut MemoryMerkleCols<_, CHUNK> = row.borrow_mut();
            if row.expand_direction == BabyBear::NEG_ONE {
                row.left_direction_different = BabyBear::ZERO;
                row.right_direction_different = BabyBear::ZERO;
            }
        }
        chip_ctx.common_main.replace(Arc::new(trace));
    }

    BabyBearPoseidon2Engine::run_test_fast(
        vec![Arc::new(chip.air), Arc::new(hash_test_chip.air())],
        vec![chip_ctx, hash_test_chip.generate_proving_ctx()],
    )
    .expect("We tinkered with the trace and now it doesn't pass");
}
