use std::{
    cmp::{max, Ordering},
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    mem,
};

use p3_field::PrimeField32;
use p3_util::log2_strict_usize;

use crate::system::memory::{Equipartition, TimestampedEquipartition, TimestampedValues};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Block<T> {
    ContainedInActive,
    Active { timestamp: u32, data: Vec<T> },
    ContainsActive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessAdapterRecordKind {
    Split,
    Merge {
        left_timestamp: u32,
        right_timestamp: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessAdapterRecord<T> {
    pub timestamp: u32,
    pub address_space: T,
    pub start_index: T,
    pub data: Vec<T>,
    pub kind: AccessAdapterRecordKind,
}

/// Represents a single or batch memory write operation.
/// Can be used to generate [MemoryWriteAuxCols].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryWriteRecord<T, const N: usize> {
    pub address_space: T,
    pub address: T,
    pub timestamp: u32,
    pub prev_timestamp: u32,
    pub data: [T; N],
    pub prev_data: [T; N],
}

impl<T: Copy> MemoryWriteRecord<T, 1> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

/// Represents a single or batch memory read operation.
///
/// Also used for "reads" from address space 0 (immediates).
/// Can be used to generate [MemoryReadAuxCols] or [MemoryReadOrImmediateAuxCols].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryReadRecord<T, const N: usize> {
    pub address_space: T,
    pub address: T,
    pub timestamp: u32,
    pub prev_timestamp: u32,
    pub data: [T; N],
}

impl<T: Copy> MemoryReadRecord<T, 1> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

/// The ID of the node corresponding to the block in the binary trie. (See [Memory] for more details.)
///
/// The ID of the root node is 1. The ID of the left child of a node with id `node_id` is `2*node_id`,
/// and the ID of the right child is `2*node_id + 1`.
///
/// Thus, the ith block of size 2^j has ID `2^{address_max_bits - j} + i`.
type BlockId = usize;

/// Tracks the state of memory cells specified by `(address_space, index)` tuples.
/// Internally, maintains a partition of memory cells into blocks, where each block has length a
/// power of two and starts at a position divisible by its length (i.e., is block-aligned).
/// Partitioning is managed using a binary trie (segment tree), with each block corresponding to a
/// node in the trie. `Memory` maintains the partition by tracking a set of active nodes,
/// ensuring exactly one active node on each root-to-leaf path.
#[derive(Debug, Clone)]
pub struct Memory<T> {
    timestamp: u32,
    address_max_bits: usize,
    blocks: HashMap<(T, BlockId), Block<T>>,
    initial_block_len: usize,
}

/// The timestamp corresponding to initial memory.
pub const INITIAL_TIMESTAMP: u32 = 0;

impl<F: PrimeField32> Memory<F> {
    /// Creates a new `Memory` instance with the given `address_max_bits` from a partition in which
    /// every part has length `N`.
    pub fn new<const N: usize>(
        initial_memory: &Equipartition<F, N>,
        address_max_bits: usize,
    ) -> Self {
        assert!(N.is_power_of_two());

        let mut blocks = HashMap::new();
        for ((address_space, label), values) in initial_memory {
            let block_id = ((1 << address_max_bits) + label * N) >> log2_strict_usize(N);
            blocks.insert(
                (*address_space, block_id),
                Block::Active {
                    timestamp: INITIAL_TIMESTAMP,
                    data: values.into(),
                },
            );
        }
        Self {
            timestamp: INITIAL_TIMESTAMP + 1,
            blocks,
            address_max_bits,
            initial_block_len: N,
        }
    }

    /// Returns the current timestamp.
    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Increments the current timestamp by one and returns the new value.
    pub fn increment_timestamp(&mut self) -> u32 {
        self.timestamp += 1;
        self.timestamp
    }

    /// Increments the current timestamp by a specified delta and returns the new value.
    pub fn increment_timestamp_by(&mut self, delta: u32) -> u32 {
        self.timestamp += delta;
        self.timestamp
    }

    /// Writes an array of values to the memory at the specified address space and start index.
    pub fn write<const N: usize>(
        &mut self,
        address_space: F,
        start_index: usize,
        values: [F; N],
    ) -> (MemoryWriteRecord<F, N>, Vec<AccessAdapterRecord<F>>) {
        let active_block_records = self.access(address_space, start_index, N);
        let block_id = self.block_id(start_index, N);

        if let Some(Block::Active { data, timestamp }) =
            self.blocks.get_mut(&(address_space, block_id))
        {
            let prev_data = mem::replace(data, values.to_vec());
            let prev_timestamp = *timestamp;

            debug_assert!(
                prev_timestamp < self.timestamp,
                "previous timestamp ({prev_timestamp}) not less than timestamp ({})",
                self.timestamp
            );

            *timestamp = self.timestamp;
            let record = MemoryWriteRecord {
                address_space,
                address: F::from_canonical_usize(start_index),
                timestamp: self.timestamp,
                prev_timestamp,
                data: values,
                prev_data: prev_data.try_into().unwrap(),
            };

            self.timestamp += 1;

            (record, active_block_records)
        } else {
            unreachable!()
        }
    }

    /// Reads an array of values from the memory at the specified address space and start index.
    pub fn read<const N: usize>(
        &mut self,
        address_space: F,
        start_index: usize,
    ) -> (MemoryReadRecord<F, N>, Vec<AccessAdapterRecord<F>>) {
        let new_active_block_records = self.access(address_space, start_index, N);
        let block_id = self.block_id(start_index, N);

        if let Some(Block::Active { data, timestamp }) =
            self.blocks.get_mut(&(address_space, block_id))
        {
            let prev_timestamp = *timestamp;

            debug_assert!(
                prev_timestamp < self.timestamp,
                "previous timestamp ({}) not less than timestamp ({})",
                prev_timestamp,
                self.timestamp
            );

            *timestamp = self.timestamp;
            let record = MemoryReadRecord {
                address_space,
                address: F::from_canonical_usize(start_index),
                timestamp: self.timestamp,
                prev_timestamp,
                data: data.clone().try_into().unwrap(),
            };

            self.timestamp += 1;

            (record, new_active_block_records)
        } else {
            unreachable!()
        }
    }

    pub(super) fn access(
        &mut self,
        address_space: F,
        start_index: usize,
        len: usize,
    ) -> Vec<AccessAdapterRecord<F>> {
        assert_eq!(
            start_index % len,
            0,
            "start index ({start_index}) must be divisible by len ({len})"
        );

        let block_id = self.block_id(start_index, len);
        let mut records = vec![];

        self.node_access(address_space, block_id, start_index, len, &mut records);
        records
    }

    // Returns the block ID of the block of length `len` containing the index `index`.
    fn block_id(&self, index: usize, len: usize) -> BlockId {
        // Leaves have labels from (1 << address_max_bits)..(1 << (address_max_bits + 1)) - 1.
        ((1 << self.address_max_bits) + index) >> log2_strict_usize(len)
    }

    /// Recursively makes a memory block active and produces adapter records.
    ///
    /// # Note
    ///
    /// This function could be optimized further. In particular, currently when splitting an active
    /// node, we allocate memory for the left and right halves and copy the data into the children.
    /// But the children can just store slices to the reference. One possible implementation---leveraging
    /// that `len` is at most 32 or 64---is that we maintain a disjoint array-backed segment tree for
    /// each block of max size.
    fn node_access(
        &mut self,
        address_space: F,
        block_id: BlockId,
        start: usize,
        len: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) -> (u32, Vec<F>) {
        // Lazily create the initial block if it doesn't exist. In initial memory,
        // all active blocks are the leaves of the tree and have timestamp `INITIAL_TIMESTAMP`.
        let block_state =
            self.blocks
                .entry((address_space, block_id))
                .or_insert_with(|| match self.initial_block_len.cmp(&len) {
                    Ordering::Less => Block::ContainsActive,
                    Ordering::Equal => Block::Active {
                        timestamp: INITIAL_TIMESTAMP,
                        data: vec![F::default(); len],
                    },
                    Ordering::Greater => Block::ContainedInActive,
                });

        match block_state {
            Block::Active {
                timestamp: prev_timestamp,
                data,
            } => (*prev_timestamp, data.clone()),
            Block::ContainsActive => {
                // Recursively access left and right.
                let left_id = 2 * block_id;
                let right_id = 2 * block_id + 1;

                let (left_timestamp, left_data) =
                    self.node_access(address_space, left_id, start, len / 2, records);
                let (right_timestamp, right_data) =
                    self.node_access(address_space, right_id, start + len / 2, len / 2, records);

                let data = [left_data, right_data].concat();
                let timestamp = max(left_timestamp, right_timestamp);

                // Change state of left and right to reflect merge.
                self.blocks
                    .insert((address_space, left_id), Block::ContainedInActive);
                self.blocks
                    .insert((address_space, right_id), Block::ContainedInActive);
                self.blocks.insert(
                    (address_space, block_id),
                    Block::Active {
                        timestamp,
                        data: data.clone(),
                    },
                );

                records.push(AccessAdapterRecord {
                    timestamp,
                    address_space,
                    start_index: F::from_canonical_usize(start),
                    data: data.clone(),
                    kind: AccessAdapterRecordKind::Merge {
                        left_timestamp,
                        right_timestamp,
                    },
                });
                (timestamp, data)
            }
            Block::ContainedInActive => {
                // Recursively access parent.
                let parent_id = block_id >> 1;
                let parent_start = start - (block_id & 1) * len;

                let (parent_timestamp, parent_data) =
                    self.node_access(address_space, parent_id, parent_start, len * 2, records);

                let sibling_id = block_id ^ 1;

                let mut left_data = vec![F::default(); len];
                let mut right_data = vec![F::default(); len];
                left_data.clone_from_slice(&parent_data[..len]);
                right_data.clone_from_slice(&parent_data[len..]);

                let (data, sibling_data) = if block_id & 1 == 0 {
                    (left_data, right_data)
                } else {
                    (right_data, left_data)
                };

                // Change state of parent and sibling to reflect split.
                self.blocks
                    .insert((address_space, parent_id), Block::ContainsActive);
                self.blocks.insert(
                    (address_space, sibling_id),
                    Block::Active {
                        timestamp: parent_timestamp,
                        data: sibling_data,
                    },
                );
                self.blocks.insert(
                    (address_space, block_id),
                    Block::Active {
                        timestamp: parent_timestamp,
                        data: data.clone(),
                    },
                );

                records.push(AccessAdapterRecord {
                    timestamp: parent_timestamp,
                    address_space,
                    start_index: F::from_canonical_usize(parent_start),
                    data: parent_data,
                    kind: AccessAdapterRecordKind::Split,
                });
                (parent_timestamp, data)
            }
        }
    }

    /// Finalizes the memory state to a partition in which all parts have size `N`.
    pub fn finalize<const N: usize>(
        &mut self,
    ) -> (TimestampedEquipartition<F, N>, Vec<AccessAdapterRecord<F>>) {
        let mut adapter_records = vec![];
        let mut blocks = HashSet::new();

        let target_block_height = log2_strict_usize(N);

        for ((address_space, block_id), block) in self.blocks.iter() {
            if !matches!(block, Block::Active { .. }) {
                continue;
            }
            // Leaves are at `block_height = 0`.
            let block_height =
                (self.address_max_bits + 1) - (usize::BITS - block_id.leading_zeros()) as usize;

            match block_height.cmp(&target_block_height) {
                Ordering::Less => {
                    // we need to access the block of height `target_block_height` containing this block
                    blocks.insert((
                        *address_space,
                        block_id >> (target_block_height - block_height),
                    ));
                }
                Ordering::Greater => {
                    let start_block_id = block_id << (block_height - target_block_height);
                    for i in 0..(1 << (block_height - target_block_height)) {
                        blocks.insert((*address_space, start_block_id + i));
                    }
                }
                Ordering::Equal => {
                    blocks.insert((*address_space, *block_id));
                }
            }
        }

        let mut final_memory = BTreeMap::new();

        for (address_space, block_id) in blocks {
            let start_index = (block_id << target_block_height) - (1 << self.address_max_bits);
            let label = start_index / N;
            let (timestamp, data) = self.node_access(
                address_space,
                block_id,
                start_index,
                N,
                &mut adapter_records,
            );
            final_memory.insert(
                (address_space, label),
                TimestampedValues {
                    values: data.try_into().unwrap(),
                    timestamp,
                },
            );
        }
        (final_memory, adapter_records)
    }

    /// Retrieves the value and timestamp at a specific memory address within an address space.
    pub fn get(&self, address_space: F, address: usize) -> Option<(u32, &F)> {
        let mut block_len = self.initial_block_len;
        let mut block_id = self.block_id(address, block_len);

        while let Some(block) = self.blocks.get(&(address_space, block_id)) {
            match block {
                Block::ContainedInActive => {
                    block_id /= 2;
                    block_len *= 2;
                }
                Block::Active { data, timestamp } => {
                    let index_in_block = address & (block_len - 1);
                    return Some((*timestamp, data.get(index_in_block)?));
                }
                Block::ContainsActive => {
                    let index_in_block = address & (block_len - 1);
                    if index_in_block < block_len / 2 {
                        block_id *= 2;
                    } else {
                        block_id = block_id * 2 + 1;
                    }
                    block_len /= 2;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    use super::{AccessAdapterRecord, AccessAdapterRecordKind, Memory};
    use crate::system::memory::{
        Equipartition, MemoryReadRecord, MemoryWriteRecord, TimestampedValues,
    };

    macro_rules! bb {
        ($x:expr) => {
            BabyBear::from_canonical_u32($x)
        };
    }

    macro_rules! bba {
        [$($x:expr),*] => {
            [$(BabyBear::from_canonical_u32($x)),*]
        }
    }

    macro_rules! bbvec {
        [$($x:expr),*] => {
            vec![$(BabyBear::from_canonical_u32($x)),*]
        }
    }

    #[test]
    fn test_write_read_initial_block_len_1() {
        let initial_memory = Equipartition::<BabyBear, 1>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);
        let address_space = BabyBear::one();

        memory.write(address_space, 0, bba![1, 2, 3, 4]);

        let (read_record, _) = memory.read::<2>(address_space, 0);
        assert_eq!(read_record.data, bba![1, 2]);

        memory.write(address_space, 2, bba![100]);

        let (read_record, _) = memory.read::<4>(address_space, 0);
        assert_eq!(read_record.data, bba![1, 2, 100, 4]);
    }

    #[test]
    fn test_write_read_initial_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);
        let address_space = BabyBear::one();

        memory.write(address_space, 0, bba![1, 2, 3, 4]);

        let (read_record, _) = memory.read::<2>(address_space, 0);
        assert_eq!(read_record.data, bba![1, 2]);

        memory.write(address_space, 2, bba![100]);

        let (read_record, _) = memory.read::<4>(address_space, 0);
        assert_eq!(read_record.data, bba![1, 2, 100, 4]);
    }

    #[test]
    fn test_records_initial_block_len_1() {
        let initial_memory = Equipartition::<BabyBear, 1>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);

        let (write_record, adapter_records) = memory.write(bb!(1), 0, bba![1, 2, 3, 4]);

        // Above write first causes merge of [0:1] and [1:2] into [0:2].
        assert_eq!(
            adapter_records[0],
            AccessAdapterRecord {
                timestamp: 0,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![0, 0],
                kind: AccessAdapterRecordKind::Merge {
                    left_timestamp: 0,
                    right_timestamp: 0,
                },
            }
        );
        // then merge [2:3] and [3:4] into [2:4].
        assert_eq!(
            adapter_records[1],
            AccessAdapterRecord {
                timestamp: 0,
                address_space: bb!(1),
                start_index: bb!(2),
                data: bbvec![0, 0],
                kind: AccessAdapterRecordKind::Merge {
                    left_timestamp: 0,
                    right_timestamp: 0,
                },
            }
        );
        // then merge [0:2] and [2:4] into [0:4].
        assert_eq!(
            adapter_records[2],
            AccessAdapterRecord {
                timestamp: 0,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![0, 0, 0, 0],
                kind: AccessAdapterRecordKind::Merge {
                    left_timestamp: 0,
                    right_timestamp: 0,
                },
            }
        );
        // At time 1 we write [0:4].
        assert_eq!(
            write_record,
            MemoryWriteRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 1,
                prev_timestamp: 0,
                data: bba![1, 2, 3, 4],
                prev_data: bba![0, 0, 0, 0],
            }
        );
        assert_eq!(memory.timestamp(), 2);

        let (read_record, adapter_records) = memory.read::<4>(bb!(1), 0);
        // At time 2 we read [0:4].
        assert_eq!(adapter_records.len(), 0);
        assert_eq!(
            read_record,
            MemoryReadRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 2,
                prev_timestamp: 1,
                data: bba![1, 2, 3, 4],
            }
        );
        assert_eq!(memory.timestamp(), 3);

        let (read_record, adapter_records) = memory.write::<2>(bb!(1), 0, bba![10, 11]);
        // write causes split [0:4] into [0:2] and [2:4] (to prepare for write to [0:2]).
        assert_eq!(adapter_records.len(), 1);
        assert_eq!(
            adapter_records[0],
            AccessAdapterRecord {
                timestamp: 2,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![1, 2, 3, 4],
                kind: AccessAdapterRecordKind::Split,
            }
        );

        // At time 3 we write [10, 11] into [0, 2].
        assert_eq!(
            read_record,
            MemoryWriteRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 3,
                prev_timestamp: 2,
                data: bba![10, 11],
                prev_data: bba![1, 2],
            }
        );

        let (read_record, adapter_records) = memory.read::<4>(bb!(1), 0);
        assert_eq!(adapter_records.len(), 1);
        assert_eq!(
            adapter_records[0],
            AccessAdapterRecord {
                timestamp: 3,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![10, 11, 3, 4],
                kind: AccessAdapterRecordKind::Merge {
                    left_timestamp: 3,
                    right_timestamp: 2
                },
            }
        );
        // At time 9 we read [0:4].
        assert_eq!(
            read_record,
            MemoryReadRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 4,
                prev_timestamp: 3,
                data: bba![10, 11, 3, 4],
            }
        );
    }

    #[test]
    fn test_records_initial_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);

        let (write_record, adapter_records) = memory.write(bb!(1), 0, bba![1, 2, 3, 4]);

        // Above write first causes split of [0:8] into [0:4] and [4:8].
        assert_eq!(adapter_records.len(), 1);
        assert_eq!(
            adapter_records[0],
            AccessAdapterRecord {
                timestamp: 0,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![0, 0, 0, 0, 0, 0, 0, 0],
                kind: AccessAdapterRecordKind::Split,
            }
        );
        // At time 1 we write [0:4].
        assert_eq!(
            write_record,
            MemoryWriteRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 1,
                prev_timestamp: 0,
                data: bba![1, 2, 3, 4],
                prev_data: bba![0, 0, 0, 0],
            }
        );
        assert_eq!(memory.timestamp(), 2);

        let (read_record, adapter_records) = memory.read::<4>(bb!(1), 0);
        // At time 2 we read [0:4].
        assert_eq!(adapter_records.len(), 0);
        assert_eq!(
            read_record,
            MemoryReadRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 2,
                prev_timestamp: 1,
                data: bba![1, 2, 3, 4],
            }
        );
        assert_eq!(memory.timestamp(), 3);

        let (read_record, adapter_records) = memory.write::<2>(bb!(1), 0, bba![10, 11]);
        // write causes split [0:4] into [0:2] and [2:4] (to prepare for write to [0:2]).
        assert_eq!(adapter_records.len(), 1);
        assert_eq!(
            adapter_records[0],
            AccessAdapterRecord {
                timestamp: 2,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![1, 2, 3, 4],
                kind: AccessAdapterRecordKind::Split,
            }
        );

        // At time 3 we write [10, 11] into [0, 2].
        assert_eq!(
            read_record,
            MemoryWriteRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 3,
                prev_timestamp: 2,
                data: bba![10, 11],
                prev_data: bba![1, 2],
            }
        );

        let (read_record, adapter_records) = memory.read::<4>(bb!(1), 0);
        assert_eq!(adapter_records.len(), 1);
        assert_eq!(
            adapter_records[0],
            AccessAdapterRecord {
                timestamp: 3,
                address_space: bb!(1),
                start_index: bb!(0),
                data: bbvec![10, 11, 3, 4],
                kind: AccessAdapterRecordKind::Merge {
                    left_timestamp: 3,
                    right_timestamp: 2
                },
            }
        );
        // At time 9 we read [0:4].
        assert_eq!(
            read_record,
            MemoryReadRecord {
                address_space: bb!(1),
                address: bb!(0),
                timestamp: 4,
                prev_timestamp: 3,
                data: bba![10, 11, 3, 4],
            }
        );
    }

    #[test]
    fn test_get_initial_block_len_1() {
        let initial_memory = Equipartition::<BabyBear, 1>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);

        memory.write(bb!(1), 0, bba![4, 3, 2, 1]);

        assert_eq!(
            memory.get(bb!(1), 0),
            Some((1, &BabyBear::from_canonical_u32(4)))
        );
        assert_eq!(
            memory.get(bb!(1), 1),
            Some((1, &BabyBear::from_canonical_u32(3)))
        );
        assert_eq!(
            memory.get(bb!(1), 2),
            Some((1, &BabyBear::from_canonical_u32(2)))
        );
        assert_eq!(
            memory.get(bb!(1), 3),
            Some((1, &BabyBear::from_canonical_u32(1)))
        );
        assert_eq!(memory.get(bb!(1), 5), None);

        assert_eq!(memory.get(bb!(0), 0), None);
    }

    #[test]
    fn test_get_initial_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);

        memory.write(bb!(1), 0, bba![4, 3, 2, 1]);

        assert_eq!(
            memory.get(bb!(1), 0),
            Some((1, &BabyBear::from_canonical_u32(4)))
        );
        assert_eq!(
            memory.get(bb!(1), 1),
            Some((1, &BabyBear::from_canonical_u32(3)))
        );
        assert_eq!(
            memory.get(bb!(1), 2),
            Some((1, &BabyBear::from_canonical_u32(2)))
        );
        assert_eq!(
            memory.get(bb!(1), 3),
            Some((1, &BabyBear::from_canonical_u32(1)))
        );
        assert_eq!(memory.get(bb!(1), 5), Some((0, &BabyBear::zero())));
        assert_eq!(memory.get(bb!(1), 9), None);

        assert_eq!(memory.get(bb!(0), 0), None);
    }

    #[test]
    fn test_finalize_empty() {
        let initial_memory = Equipartition::<BabyBear, 4>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);

        let (memory, records) = memory.finalize::<4>();
        assert_eq!(memory.len(), 0);
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_finalize_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory, 29);
        // Make block 0:4 in address space 1 active.
        memory.write(bb!(1), 0, bba![1, 2, 3, 4]);

        // Make block 16:32 in address space 1 active.
        memory.write(
            bb!(1),
            16,
            bba![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        );

        // Make block 64:72 in address space 2 active.
        memory.write(bb!(2), 64, bba![8, 7, 6, 5, 4, 3, 2, 1]);

        // Finalize to a partition of size 8.
        let (final_memory, records) = memory.finalize::<8>();
        assert_eq!(final_memory.len(), 4);
        assert_eq!(
            final_memory.get(&(bb!(1), 0)),
            Some(&TimestampedValues {
                values: bba![1, 2, 3, 4, 0, 0, 0, 0],
                timestamp: 1,
            })
        );
        // start_index = 16 corresponds to label = 2
        assert_eq!(
            final_memory.get(&(bb!(1), 2)),
            Some(&TimestampedValues {
                values: bba![1, 1, 1, 1, 1, 1, 1, 1],
                timestamp: 2,
            })
        );
        // start_index = 24 corresponds to label = 3
        assert_eq!(
            final_memory.get(&(bb!(1), 3)),
            Some(&TimestampedValues {
                values: bba![1, 1, 1, 1, 1, 1, 1, 1],
                timestamp: 2,
            })
        );
        // start_index = 64 corresponds to label = 8
        assert_eq!(
            final_memory.get(&(bb!(2), 8)),
            Some(&TimestampedValues {
                values: bba![8, 7, 6, 5, 4, 3, 2, 1],
                timestamp: 3,
            })
        );

        // We need to do 1 + 1 + 0 = 2 adapters.
        assert_eq!(records.len(), 2);
    }
}
