use std::{
    array,
    borrow::BorrowMut,
    collections::{BTreeMap, BTreeSet, HashSet},
    sync::Arc,
};

use openvm_circuit_primitives::Chip;
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::{
    interaction::{PermutationCheckBus, PermutationInteractionType},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    prover::AirProvingContext,
    test_utils::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    StarkEngine,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::RngCore;

use crate::{
    arch::{
        hasher::{Hasher, HasherChip},
        testing::{MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS},
        vm_poseidon2_config, AddressSpaceHostConfig, MemoryCellType, MemoryConfig,
        ADDR_SPACE_OFFSET,
    },
    system::{
        memory::{
            merkle::{
                memory_to_vec_partition, tests::util::HashTestChip, MemoryDimensions,
                MemoryMerkleChip, MemoryMerkleCols, MerkleTree,
            },
            online::{GuestMemory, LinearMemory},
            persistent::DirtyLeaves,
            ptr_bits_from_address_height, AddressMap, MemoryImage,
        },
        poseidon2::Poseidon2PeripheryChip,
    },
    utils::test_cpu_engine,
};

mod util;

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
                let label = (pointer / VM_DIGEST_WIDTH) as u32;
                assert!(address_space - (ADDR_SPACE_OFFSET as usize) < (1 << addr_space_height));
                assert!(pointer < (VM_DIGEST_WIDTH << address_height));
                assert!(touched_labels.contains(&(address_space as u32, label)));
            }
        }
    }

    let mut hash_test_chip = HashTestChip::new();

    let final_tree_check =
        MerkleTree::from_memory(final_memory, &memory_dimensions, &hash_test_chip);

    let mut chip =
        MemoryMerkleChip::<VM_DIGEST_WIDTH, _>::new(memory_dimensions, merkle_bus, COMPRESSION_BUS);
    let final_partition: BTreeMap<_, [F; VM_DIGEST_WIDTH]> =
        memory_to_vec_partition::<F, VM_DIGEST_WIDTH>(final_memory, &memory_dimensions)
            .into_iter()
            .map(|(idx, values)| {
                let address_space =
                    (idx >> memory_dimensions.address_height) as u32 + ADDR_SPACE_OFFSET;
                let label = (idx & ((1 << memory_dimensions.address_height) - 1)) as u32;
                ((address_space, label * (VM_DIGEST_WIDTH as u32)), values)
            })
            .collect();
    let final_partition: BTreeMap<_, _> = final_partition
        .into_iter()
        .filter(|((address_space, pointer), _)| {
            touched_labels.contains(&(*address_space, pointer / VM_DIGEST_WIDTH as u32))
        })
        .collect();
    // Dirtiness is per *write* and the test scenario defines its write pattern here:
    // exactly the touched leaves whose values changed are treated as written (the
    // minimal valid dirty set; any superset would also be sound).
    let dirty_leaves: DirtyLeaves = final_partition
        .iter()
        .filter(|((address_space, pointer), values)| {
            let init_values: [F; VM_DIGEST_WIDTH] = array::from_fn(|i| unsafe {
                initial_memory.get_f::<F>(*address_space, *pointer + i as u32)
            });
            init_values != **values
        })
        .map(|(&key, _)| key)
        .collect();
    chip.finalize(
        initial_memory,
        &final_partition,
        &dirty_leaves,
        &hash_test_chip,
    );

    assert_eq!(
        chip.final_state.as_ref().unwrap().final_root,
        final_tree_check.root()
    );
    let chip_api = chip.generate_proving_ctx();

    let dummy_interaction_air =
        DummyInteractionAir::new(4 + VM_DIGEST_WIDTH, true, merkle_bus.index);
    let mut dummy_interaction_trace_rows = vec![];
    let mut interaction = |interaction_type: PermutationInteractionType,
                           is_compress: bool,
                           height: usize,
                           as_label: u32,
                           address_label: u32,
                           hash: [BabyBear; VM_DIGEST_WIDTH]| {
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
            BabyBear::from_usize(height),
            BabyBear::from_u32(as_label),
            BabyBear::from_u32(address_label),
        ]);
        dummy_interaction_trace_rows.extend(hash);
    };

    for (address_space, address_label) in touched_labels {
        let initial_values = unsafe {
            array::from_fn(|i| {
                initial_memory.get((
                    address_space,
                    address_label * VM_DIGEST_WIDTH as u32 + i as u32,
                ))
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
        let leaf_ptr = address_label * (VM_DIGEST_WIDTH as u32);
        let final_values = *final_partition.get(&(address_space, leaf_ptr)).unwrap();
        // Like the real boundary chip, the dummy references a leaf's final state only
        // when the leaf is dirty (the final-claim multiplicity is `is_dirty`).
        if dirty_leaves.contains(&(address_space, leaf_ptr)) {
            interaction(
                PermutationInteractionType::Send,
                true,
                0,
                as_label,
                address_label,
                final_values,
            );
        }
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
    let dummy_interaction_api = AirProvingContext::simple_no_pis(dummy_interaction_trace);

    test_cpu_engine()
        .run_test(
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
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::F { size: 4 },
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::F { size: 4 },
            },
        ],
        ptr_bits_from_address_height(height),
        20,
        17,
    );

    let mut initial_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
    let mut final_memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));

    let mut seen = HashSet::new();
    let mut touched_labels = BTreeSet::new();

    while num_initial_addresses != 0 || num_touched_addresses != 0 {
        let address_space = (next_u32() & 1) + 1;
        let label = next_u32() % (1 << height);
        let pointer = label * VM_DIGEST_WIDTH as u32 + (next_u32() % VM_DIGEST_WIDTH as u32);

        if seen.insert(pointer) {
            let is_initial = next_u32() & 1 == 0;
            let is_touched = next_u32() & 1 == 0;
            let value_changes = next_u32() & 1 == 0;

            if is_initial && num_initial_addresses != 0 {
                num_initial_addresses -= 1;
                let value = BabyBear::from_u32(next_u32() % max_value);
                unsafe {
                    initial_memory.write(address_space, pointer, [value]);
                    final_memory.write(address_space, pointer, [value]);
                }
            }
            if is_touched && num_touched_addresses != 0 {
                num_touched_addresses -= 1;
                touched_labels.insert((address_space, label));
                if value_changes || !is_initial {
                    let value = BabyBear::from_u32(next_u32() % max_value);
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
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::F { size: 4 },
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::F { size: 4 },
            },
        ],
        ptr_bits_from_address_height(height),
        20,
        17,
    );
    let md = mem_config.memory_dimensions();

    let memory = AddressMap::from_mem_config(&mem_config);

    let mut chip: MemoryMerkleChip<VM_DIGEST_WIDTH, _> = MemoryMerkleChip::new(
        md,
        PermutationCheckBus::new(MEMORY_MERKLE_BUS),
        COMPRESSION_BUS,
    );

    chip.finalize(
        &memory,
        &BTreeMap::new(),
        &DirtyLeaves::default(),
        &hash_test_chip,
    );
    let trace = chip.generate_proving_ctx();
    test_cpu_engine()
        .run_test(
            vec![Arc::new(chip.air), Arc::new(hash_test_chip.air())],
            vec![trace, hash_test_chip.generate_proving_ctx()],
        )
        .expect("Empty touched memory doesn't work");
}

#[test]
fn expand_test_negative() {
    let mut hash_test_chip = HashTestChip::new();
    let height = 1;

    let mem_config = MemoryConfig::new(
        1,
        vec![
            AddressSpaceHostConfig {
                num_cells: 0,
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::F { size: 4 },
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::F { size: 4 },
            },
        ],
        ptr_bits_from_address_height(height),
        20,
        17,
    );
    let md = mem_config.memory_dimensions();

    let memory = AddressMap::from_mem_config(&mem_config);

    let mut chip: MemoryMerkleChip<VM_DIGEST_WIDTH, _> = MemoryMerkleChip::new(
        md,
        PermutationCheckBus::new(MEMORY_MERKLE_BUS),
        COMPRESSION_BUS,
    );

    chip.finalize(
        &memory,
        &BTreeMap::new(),
        &DirtyLeaves::default(),
        &hash_test_chip,
    );
    let mut chip_ctx = chip.generate_proving_ctx();
    {
        for row in chip_ctx.common_main.rows_mut() {
            let row: &mut MemoryMerkleCols<_, VM_DIGEST_WIDTH> = row.borrow_mut();
            if row.expand_direction == BabyBear::NEG_ONE {
                row.left_child_mode = BabyBear::ZERO;
                row.right_child_mode = BabyBear::ZERO;
            }
        }
    }

    assert!(test_cpu_engine()
        .run_test(
            vec![Arc::new(chip.air), Arc::new(hash_test_chip.air())],
            vec![chip_ctx, hash_test_chip.generate_proving_ctx()],
        )
        .is_err());
}

const BELOW_LEAF_PATH_LEN: usize = 31;
const COUNTEREXAMPLE_TRACE_HEIGHT: usize = 256;

fn counterexample_digest(seed: u32) -> [BabyBear; VM_DIGEST_WIDTH] {
    array::from_fn(|i| BabyBear::from_u32(seed.wrapping_add(17 * i as u32)))
}

fn below_path_has_prefix(path: u32, depth: usize, prefix: u32) -> bool {
    if depth == 0 {
        prefix == 0
    } else {
        path >> (BELOW_LEAF_PATH_LEN - depth) == prefix
    }
}

fn final_direction_different(direction: BabyBear, child_has_expansion: bool) -> BabyBear {
    if direction == BabyBear::NEG_ONE && !child_has_expansion {
        BabyBear::ONE
    } else {
        BabyBear::ZERO
    }
}

/// The merged `*_child_mode` value for these hand-built fixtures. They reference each
/// touched child exactly once, so an initial row's mode is always 1; a final row's mode
/// is the dd bit (1 iff the child is borrowed from the initial tree); padding is 0.
fn child_mode(direction: BabyBear, child_has_expansion: bool) -> BabyBear {
    if direction == BabyBear::ONE {
        BabyBear::ONE
    } else {
        final_direction_different(direction, child_has_expansion)
    }
}

fn build_below_leaf_swap_subtree(
    hasher: &Poseidon2PeripheryChip<BabyBear>,
    direction: BabyBear,
    alpha_digest: [BabyBear; VM_DIGEST_WIDTH],
    beta_digest: [BabyBear; VM_DIGEST_WIDTH],
    swap_digests: bool,
) -> (
    [BabyBear; VM_DIGEST_WIDTH],
    Vec<MemoryMerkleCols<BabyBear, VM_DIGEST_WIDTH>>,
) {
    #[allow(clippy::too_many_arguments)]
    fn rec(
        hasher: &Poseidon2PeripheryChip<BabyBear>,
        rows: &mut Vec<MemoryMerkleCols<BabyBear, VM_DIGEST_WIDTH>>,
        direction: BabyBear,
        alpha_digest: [BabyBear; VM_DIGEST_WIDTH],
        beta_digest: [BabyBear; VM_DIGEST_WIDTH],
        swap_digests: bool,
        depth: usize,
        prefix: u32,
    ) -> [BabyBear; VM_DIGEST_WIDTH] {
        let alpha_path = 0;
        let beta_path = BabyBear::ORDER_U32;

        if depth == BELOW_LEAF_PATH_LEN {
            if prefix == alpha_path {
                return if swap_digests {
                    beta_digest
                } else {
                    alpha_digest
                };
            }
            if prefix == beta_path {
                return if swap_digests {
                    alpha_digest
                } else {
                    beta_digest
                };
            }
            unreachable!("recursive subtree only follows the two selected paths")
        }

        let child_depth = depth + 1;
        let left_prefix = prefix << 1;
        let right_prefix = left_prefix | 1;
        let left_has_path = below_path_has_prefix(alpha_path, child_depth, left_prefix)
            || below_path_has_prefix(beta_path, child_depth, left_prefix);
        let right_has_path = below_path_has_prefix(alpha_path, child_depth, right_prefix)
            || below_path_has_prefix(beta_path, child_depth, right_prefix);

        let left_child_hash = if left_has_path {
            rec(
                hasher,
                rows,
                direction,
                alpha_digest,
                beta_digest,
                swap_digests,
                child_depth,
                left_prefix,
            )
        } else {
            counterexample_digest(
                10_000u32
                    .wrapping_add(child_depth as u32)
                    .wrapping_add(left_prefix),
            )
        };
        let right_child_hash = if right_has_path {
            rec(
                hasher,
                rows,
                direction,
                alpha_digest,
                beta_digest,
                swap_digests,
                child_depth,
                right_prefix,
            )
        } else {
            counterexample_digest(
                20_000u32
                    .wrapping_add(child_depth as u32)
                    .wrapping_add(right_prefix),
            )
        };
        let parent_hash = hasher.compress_and_record(&left_child_hash, &right_child_hash);

        rows.push(MemoryMerkleCols {
            expand_direction: direction,
            height_section: BabyBear::ZERO,
            parent_height: -BabyBear::from_usize(depth + 1),
            parent_height_inv: (-BabyBear::from_usize(depth + 1)).inverse(),
            is_root: BabyBear::ZERO,
            parent_as_label: BabyBear::ZERO,
            parent_address_label: BabyBear::from_u32(prefix),
            parent_hash,
            left_child_hash,
            right_child_hash,
            left_child_mode: child_mode(
                direction,
                left_has_path && child_depth < BELOW_LEAF_PATH_LEN,
            ),
            right_child_mode: child_mode(
                direction,
                right_has_path && child_depth < BELOW_LEAF_PATH_LEN,
            ),
        });

        parent_hash
    }

    let mut rows = vec![];
    let root = rec(
        hasher,
        &mut rows,
        direction,
        alpha_digest,
        beta_digest,
        swap_digests,
        0,
        0,
    );
    (root, rows)
}

fn counterexample_zero_node_hash(
    hasher: &impl Hasher<VM_DIGEST_WIDTH, BabyBear>,
    height: usize,
) -> [BabyBear; VM_DIGEST_WIDTH] {
    let mut hash = hasher.hash(&[BabyBear::ZERO; VM_DIGEST_WIDTH]);
    for _ in 0..height {
        hash = hasher.compress(&hash, &hash);
    }
    hash
}

fn counterexample_parent_labels(
    memory_dimensions: MemoryDimensions,
    height: usize,
    prefix: u64,
) -> (u32, u32) {
    if height > memory_dimensions.address_height {
        (prefix as u32, 0)
    } else {
        let address_prefix_bits = memory_dimensions.address_height - height;
        let address_mask = (1u64 << address_prefix_bits) - 1;
        (
            (prefix >> address_prefix_bits) as u32,
            (prefix & address_mask) as u32,
        )
    }
}

#[derive(Clone, Copy)]
struct CounterexampleLeafUpdate {
    index: u64,
    initial_hash: [BabyBear; VM_DIGEST_WIDTH],
    final_hash: [BabyBear; VM_DIGEST_WIDTH],
}

fn build_counterexample_canonical_rows(
    hasher: &Poseidon2PeripheryChip<BabyBear>,
    memory_dimensions: MemoryDimensions,
    leaf_updates: &[CounterexampleLeafUpdate],
) -> (
    [BabyBear; VM_DIGEST_WIDTH],
    [BabyBear; VM_DIGEST_WIDTH],
    Vec<MemoryMerkleCols<BabyBear, VM_DIGEST_WIDTH>>,
) {
    let mut current = BTreeMap::new();
    for update in leaf_updates {
        assert!(
            current
                .insert(update.index, (update.initial_hash, update.final_hash))
                .is_none(),
            "duplicate leaf update"
        );
    }

    let overall_height = memory_dimensions.overall_height();
    let mut rows_by_height = (0..=overall_height).map(|_| Vec::new()).collect::<Vec<_>>();

    #[allow(clippy::needless_range_loop)]
    for height in 1..=overall_height {
        let parent_prefixes = current
            .keys()
            .map(|index| index >> 1)
            .collect::<BTreeSet<_>>();
        let mut next = BTreeMap::new();

        for parent_prefix in parent_prefixes {
            let left_prefix = parent_prefix << 1;
            let right_prefix = left_prefix | 1;
            let (left_initial_hash, left_final_hash, left_changed) =
                if let Some(&(initial_hash, final_hash)) = current.get(&left_prefix) {
                    (initial_hash, final_hash, true)
                } else {
                    let hash = counterexample_zero_node_hash(hasher, height - 1);
                    (hash, hash, false)
                };
            let (right_initial_hash, right_final_hash, right_changed) =
                if let Some(&(initial_hash, final_hash)) = current.get(&right_prefix) {
                    (initial_hash, final_hash, true)
                } else {
                    let hash = counterexample_zero_node_hash(hasher, height - 1);
                    (hash, hash, false)
                };

            let initial_hash = hasher.compress_and_record(&left_initial_hash, &right_initial_hash);
            let final_hash = hasher.compress_and_record(&left_final_hash, &right_final_hash);
            let (parent_as_label, parent_address_label) =
                counterexample_parent_labels(memory_dimensions, height, parent_prefix);
            let height_section = BabyBear::from_bool(height > memory_dimensions.address_height);
            let is_root = BabyBear::from_bool(height == overall_height);

            rows_by_height[height].push(MemoryMerkleCols {
                expand_direction: BabyBear::ONE,
                height_section,
                parent_height: BabyBear::from_usize(height),
                parent_height_inv: BabyBear::from_usize(height).inverse(),
                is_root,
                parent_as_label: BabyBear::from_u32(parent_as_label),
                parent_address_label: BabyBear::from_u32(parent_address_label),
                parent_hash: initial_hash,
                left_child_hash: left_initial_hash,
                right_child_hash: right_initial_hash,
                left_child_mode: BabyBear::ONE,
                right_child_mode: BabyBear::ONE,
            });
            rows_by_height[height].push(MemoryMerkleCols {
                expand_direction: BabyBear::NEG_ONE,
                height_section,
                parent_height: BabyBear::from_usize(height),
                parent_height_inv: BabyBear::from_usize(height).inverse(),
                is_root,
                parent_as_label: BabyBear::from_u32(parent_as_label),
                parent_address_label: BabyBear::from_u32(parent_address_label),
                parent_hash: final_hash,
                left_child_hash: left_final_hash,
                right_child_hash: right_final_hash,
                left_child_mode: BabyBear::from_bool(!left_changed),
                right_child_mode: BabyBear::from_bool(!right_changed),
            });

            next.insert(parent_prefix, (initial_hash, final_hash));
        }

        current = next;
    }

    let (root_index, (initial_root, final_root)) =
        current.into_iter().next().expect("missing root");
    assert_eq!(root_index, 0);

    let mut rows = vec![];
    for height in (1..=overall_height).rev() {
        rows.append(&mut rows_by_height[height]);
    }

    (initial_root, final_root, rows)
}

fn build_hidden_leaf_expansion_row(
    hasher: &Poseidon2PeripheryChip<BabyBear>,
    direction: BabyBear,
    address_space_label: u32,
    leaf_label: u32,
    below_leaf_root: [BabyBear; VM_DIGEST_WIDTH],
    unchanged_sibling_hash: [BabyBear; VM_DIGEST_WIDTH],
) -> (
    [BabyBear; VM_DIGEST_WIDTH],
    MemoryMerkleCols<BabyBear, VM_DIGEST_WIDTH>,
) {
    let parent_hash = hasher.compress_and_record(&below_leaf_root, &unchanged_sibling_hash);
    (
        parent_hash,
        MemoryMerkleCols {
            expand_direction: direction,
            height_section: BabyBear::ZERO,
            parent_height: BabyBear::ZERO,
            parent_height_inv: BabyBear::ZERO,
            is_root: BabyBear::ZERO,
            parent_as_label: BabyBear::from_u32(address_space_label),
            parent_address_label: BabyBear::from_u32(leaf_label),
            parent_hash,
            left_child_hash: below_leaf_root,
            right_child_hash: unchanged_sibling_hash,
            left_child_mode: child_mode(direction, true),
            right_child_mode: child_mode(direction, false),
        },
    )
}

/// Builds the fraudulent below-leaf-swap merkle trace and recording
/// [`Poseidon2PeripheryChip`], with no boundary leaf and no memory-bus
/// interaction.
///
/// A "hidden" leaf's hash is silently changed by swapping two digests in a
/// fabricated subtree *below* it. Since the leaf comes from a merkle expansion
/// row (at `parent_height = 0`) rather than the boundary chip, the fraud records
/// no memory write yet changes the Merkle root. Everything balances on the
/// merkle and compression buses alone, so only {merkle, poseidon2} need
/// replacing in the real VM key.
///
/// Returns the trace, its public values (`initial_root || final_root`), and the
/// Poseidon2 chip.
fn build_below_leaf_swap_fraud_merkle(
    memory_dimensions: MemoryDimensions,
    poseidon2_max_constraint_degree: usize,
) -> (
    RowMajorMatrix<BabyBear>,
    Vec<BabyBear>,
    Poseidon2PeripheryChip<BabyBear>,
) {
    assert_eq!(BabyBear::ORDER_U32, 0b1111000000000000000000000000001);

    let poseidon2_chip = Poseidon2PeripheryChip::<BabyBear>::new(
        vm_poseidon2_config(),
        poseidon2_max_constraint_degree,
    );

    let alpha_digest = counterexample_digest(1);
    let beta_digest = counterexample_digest(2);
    let (initial_below_root, initial_below_rows) = build_below_leaf_swap_subtree(
        &poseidon2_chip,
        BabyBear::ONE,
        alpha_digest,
        beta_digest,
        false,
    );
    let (final_below_root, final_below_rows) = build_below_leaf_swap_subtree(
        &poseidon2_chip,
        BabyBear::NEG_ONE,
        alpha_digest,
        beta_digest,
        true,
    );
    assert_ne!(initial_below_root, final_below_root);

    let hidden_address_space_label = 0;
    let hidden_leaf_label = 0;
    let hidden_leaf_index = memory_dimensions.label_to_index((
        ADDR_SPACE_OFFSET + hidden_address_space_label,
        hidden_leaf_label,
    ));
    let hidden_unchanged_sibling_hash = [BabyBear::ZERO; VM_DIGEST_WIDTH];
    let (initial_hidden_leaf_hash, initial_hidden_leaf_row) = build_hidden_leaf_expansion_row(
        &poseidon2_chip,
        BabyBear::ONE,
        hidden_address_space_label,
        hidden_leaf_label,
        initial_below_root,
        hidden_unchanged_sibling_hash,
    );
    let (final_hidden_leaf_hash, final_hidden_leaf_row) = build_hidden_leaf_expansion_row(
        &poseidon2_chip,
        BabyBear::NEG_ONE,
        hidden_address_space_label,
        hidden_leaf_label,
        final_below_root,
        hidden_unchanged_sibling_hash,
    );
    assert_ne!(initial_hidden_leaf_hash, final_hidden_leaf_hash);

    let (initial_root, final_root, mut rows) = build_counterexample_canonical_rows(
        &poseidon2_chip,
        memory_dimensions,
        &[CounterexampleLeafUpdate {
            index: hidden_leaf_index,
            initial_hash: initial_hidden_leaf_hash,
            final_hash: final_hidden_leaf_hash,
        }],
    );
    assert_ne!(initial_root, final_root);

    rows.push(initial_hidden_leaf_row);
    rows.push(final_hidden_leaf_row);

    let mut initial_below_by_depth = (0..BELOW_LEAF_PATH_LEN)
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut final_below_by_depth = (0..BELOW_LEAF_PATH_LEN)
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for row in initial_below_rows {
        let depth = (BabyBear::ORDER_U32 - row.parent_height.as_canonical_u32()) as usize;
        initial_below_by_depth[depth - 1].push(row);
    }
    for row in final_below_rows {
        let depth = (BabyBear::ORDER_U32 - row.parent_height.as_canonical_u32()) as usize;
        final_below_by_depth[depth - 1].push(row);
    }
    for depth in 0..BELOW_LEAF_PATH_LEN {
        rows.append(&mut initial_below_by_depth[depth]);
        rows.append(&mut final_below_by_depth[depth]);
    }

    let trace_height = rows
        .len()
        .next_power_of_two()
        .max(COUNTEREXAMPLE_TRACE_HEIGHT);
    // Pad with inert rows: `parent_height` must match the last real row so the
    // descending-`parent_height` constraint holds; all else is zero.
    let padding_height = rows.last().unwrap().parent_height;
    while rows.len() < trace_height {
        rows.push(MemoryMerkleCols {
            expand_direction: BabyBear::ZERO,
            height_section: BabyBear::ZERO,
            parent_height: padding_height,
            parent_height_inv: padding_height.inverse(),
            is_root: BabyBear::ZERO,
            parent_as_label: BabyBear::ZERO,
            parent_address_label: BabyBear::ZERO,
            parent_hash: [BabyBear::ZERO; VM_DIGEST_WIDTH],
            left_child_hash: [BabyBear::ZERO; VM_DIGEST_WIDTH],
            right_child_hash: [BabyBear::ZERO; VM_DIGEST_WIDTH],
            left_child_mode: BabyBear::ZERO,
            right_child_mode: BabyBear::ZERO,
        });
    }

    let merkle_width = MemoryMerkleCols::<BabyBear, VM_DIGEST_WIDTH>::width();
    let mut merkle_trace = BabyBear::zero_vec(merkle_width * trace_height);
    for (trace_row, row) in merkle_trace.chunks_exact_mut(merkle_width).zip(rows) {
        *trace_row.borrow_mut() = row;
    }

    let public_values = initial_root.into_iter().chain(final_root).collect();
    (
        RowMajorMatrix::new(merkle_trace, merkle_width),
        public_values,
        poseidon2_chip,
    )
}

/// Regression test for the below-leaf-swap counterexample against the **real**
/// OpenVM verifying key (the production persistent-memory `SystemConfig`), not a
/// hand-picked subset of AIRs.
///
/// We key-gen the real config, run a trivial program to obtain a fully
/// bus-balanced proving context, then overwrite only the merkle and poseidon2
/// contexts with the fraudulent ones and run the real prover + verifier. The
/// merkle, compression and memory buses balance among only {boundary (empty),
/// merkle, poseidon2} -- exactly the chips a real persistent-memory VM uses
/// there -- so no permissive dummy chip is involved.
///
/// The fraud hides a Merkle root change inside a leaf-expansion row at
/// `parent_height = 0` (plus a fabricated subtree below it), backed by zero
/// memory operations. The fixed `MemoryMerkleAir` forbids `parent_height = 0`
/// whenever `expand_direction != 0`, so the production verifier must now
/// **reject** this proof.
#[test]
fn real_vm_keygen_verifier_rejects_below_leaf_swap_counterexample() {
    use openvm_instructions::{
        exe::VmExe, instruction::Instruction, program::Program, LocalOpcode,
        SystemOpcode::TERMINATE,
    };

    use crate::{
        arch::{PreflightExecutionOutput, Streams, SystemConfig, VirtualMachine, VmState},
        system::{
            memory::{online::GuestMemory, AddressMap},
            SystemCpuBuilder,
        },
    };

    let vm_config = SystemConfig::default();
    // Build the counterexample for the active default config; CUDA and CPU
    // proving configs can use different memory dimensions.
    let memory_dimensions = vm_config.memory_config.memory_dimensions();

    let engine = test_cpu_engine();
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(engine, SystemCpuBuilder, vm_config.clone()).unwrap();
    let vk = pk.get_vk();

    let merkle_air_id = vm_config.memory_merkle_air_id();

    // The poseidon2 periphery AIR index isn't a fixed constant, so find it by name.
    let poseidon2_air_id = vm
        .air_names()
        .position(|name| name.contains("Poseidon2Periphery"))
        .unwrap();

    // Run a trivial program (single TERMINATE) for a valid, bus-balanced proving
    // context. It touches no memory, so the boundary AIR is empty and the
    // merkle/compression/memory buses are left to {merkle, poseidon2}, which we
    // overwrite below.
    let program = Program::from_instructions(&[Instruction::<BabyBear>::from_isize(
        TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    )]);
    let vm_exe: VmExe<BabyBear> = program.into();
    let max_trace_heights = vec![0; vk.inner.per_air.len()];
    let memory = GuestMemory::new(AddressMap::from_mem_config(&vm_config.memory_config));
    vm.transport_init_memory_to_device(&memory);
    vm.load_program(vm.commit_program_on_device(&vm_exe.program));
    let from_state = VmState::new_with_defaults(0, memory, Streams::default(), 0);
    let mut interpreter = vm.preflight_interpreter(&vm_exe).unwrap();
    let PreflightExecutionOutput {
        system_records,
        record_arenas,
        ..
    } = vm
        .execute_preflight(&mut interpreter, from_state, &max_trace_heights)
        .unwrap();
    let mut ctx = vm
        .generate_proving_ctx(system_records, record_arenas)
        .unwrap();

    // Overwrite the merkle + poseidon2 contexts with the fraudulent ones.
    // `prove` requires `per_air` to be sorted by AIR id, so we re-sort after
    // swapping the entries in.
    let (merkle_trace, merkle_pvs, poseidon2_chip) =
        build_below_leaf_swap_fraud_merkle(memory_dimensions, vm_config.max_constraint_degree);
    ctx.per_trace
        .retain(|(id, _)| *id != merkle_air_id && *id != poseidon2_air_id);
    ctx.per_trace.push((
        merkle_air_id,
        AirProvingContext::simple(merkle_trace, merkle_pvs),
    ));
    ctx.per_trace
        .push((poseidon2_air_id, poseidon2_chip.generate_proving_ctx(())));
    ctx.per_trace.sort_by_key(|(id, _)| *id);

    let proof = vm.engine.prove(vm.pk(), ctx).unwrap();
    assert!(
        vm.engine.verify(&vk, &proof).is_err(),
        "fixed OpenVM verifier must reject the fraudulent below-leaf-swap proof"
    );
}

#[test]
#[should_panic]
fn expand_test_label_rebinding_attack() {
    let mut hash_test_chip = HashTestChip::new();
    let height = 4;
    let fake_label = 8u32;
    let claimed_label = 0u32;

    let mem_config = MemoryConfig::new(
        1,
        vec![
            AddressSpaceHostConfig {
                num_cells: 0,
                layout: MemoryCellType::Null,
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::field32(),
            },
            AddressSpaceHostConfig {
                num_cells: VM_DIGEST_WIDTH << height,
                layout: MemoryCellType::field32(),
            },
        ],
        height + 3,
        20,
        17,
    );
    let md = mem_config.memory_dimensions();

    let mut memory = GuestMemory::new(AddressMap::from_mem_config(&mem_config));
    unsafe {
        memory.write(
            1,
            fake_label * VM_DIGEST_WIDTH as u32,
            [BabyBear::from_u8(69)],
        );
    }

    let touched_labels_for_chip = BTreeSet::from([(1u32, fake_label)]);
    let touched_labels_for_dummy = BTreeSet::from([(1u32, claimed_label)]);

    let final_partition_for_chip: BTreeMap<_, [BabyBear; VM_DIGEST_WIDTH]> =
        memory_to_vec_partition::<BabyBear, VM_DIGEST_WIDTH>(&memory.memory, &md)
            .into_iter()
            .map(|(idx, values)| {
                let address_space = (idx >> md.address_height) as u32 + ADDR_SPACE_OFFSET;
                let label = (idx & ((1 << md.address_height) - 1)) as u32;
                ((address_space, label * (VM_DIGEST_WIDTH as u32)), values)
            })
            .filter(|((address_space, pointer), _)| {
                touched_labels_for_chip
                    .contains(&(*address_space, pointer / VM_DIGEST_WIDTH as u32))
            })
            .collect();

    let merkle_bus = PermutationCheckBus::new(MEMORY_MERKLE_BUS);
    let mut chip = MemoryMerkleChip::<VM_DIGEST_WIDTH, _>::new(md, merkle_bus, COMPRESSION_BUS);
    // The touched leaf's final values equal its initial ones (the write happened before
    // the initial snapshot), so no leaf is dirty.
    chip.finalize(
        &memory.memory,
        &final_partition_for_chip,
        &DirtyLeaves::default(),
        &hash_test_chip,
    );
    let mut chip_ctx = chip.generate_proving_ctx();

    {
        let half = BabyBear::TWO.inverse();
        // Rebind the path of fake_label to claimed_label by solving labels bottom-up:
        // x_{h} = (x_{h-1} - bit_{h-1}(fake_label)) / 2, x_0 = claimed_label.
        let mut per_height = vec![BabyBear::ZERO; md.overall_height() + 1];
        let mut curr = BabyBear::from_u32(claimed_label);
        #[allow(clippy::needless_range_loop)]
        for h in 1..=md.address_height {
            let bit = (fake_label >> (h - 1)) & 1;
            curr = (curr - BabyBear::from_u32(bit)) * half;
            per_height[h] = curr;
        }
        let root_address_label = per_height[md.address_height];
        for dst in per_height
            .iter_mut()
            .take(md.overall_height() + 1)
            .skip(md.address_height + 1)
        {
            *dst = root_address_label;
        }

        for row in chip_ctx.common_main.rows_mut() {
            let row: &mut MemoryMerkleCols<BabyBear, VM_DIGEST_WIDTH> = row.borrow_mut();
            if row.expand_direction == BabyBear::ZERO {
                continue;
            }
            let h = row.parent_height.as_canonical_u32() as usize;
            row.parent_address_label = per_height[h];
        }
    }

    let dummy_interaction_air =
        DummyInteractionAir::new(4 + VM_DIGEST_WIDTH, true, merkle_bus.index);
    let mut dummy_interaction_trace_rows = vec![];
    let mut interaction = |interaction_type: PermutationInteractionType,
                           is_compress: bool,
                           height: usize,
                           as_label: u32,
                           address_label: u32,
                           hash: [BabyBear; VM_DIGEST_WIDTH]| {
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
            BabyBear::from_usize(height),
            BabyBear::from_u32(as_label),
            BabyBear::from_u32(address_label),
        ]);
        dummy_interaction_trace_rows.extend(hash);
    };

    for (address_space, address_label) in touched_labels_for_dummy {
        let values = unsafe {
            array::from_fn(|i| {
                memory.memory.get((
                    address_space,
                    fake_label * VM_DIGEST_WIDTH as u32 + i as u32,
                ))
            })
        };
        let as_label = address_space - ADDR_SPACE_OFFSET;
        interaction(
            PermutationInteractionType::Send,
            false,
            0,
            as_label,
            address_label,
            values,
        );
        // No final-state claim: the leaf is touched but clean, and a real boundary row
        // would have `is_dirty = 0`.
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
    let dummy_interaction_api = AirProvingContext::simple_no_pis(dummy_interaction_trace);

    test_cpu_engine()
        .run_test(
            vec![
                Arc::new(chip.air),
                Arc::new(dummy_interaction_air),
                Arc::new(hash_test_chip.air()),
            ],
            vec![
                chip_ctx,
                dummy_interaction_api,
                hash_test_chip.generate_proving_ctx(),
            ],
        )
        .expect("Label-rebinding attack unexpectedly failed");
}
