use std::{
    array,
    cmp::max,
    collections::{
        BTreeMap,
        Bound::{Included, Unbounded},
        HashMap,
    },
    fmt::Debug,
};

use p3_field::PrimeField;

use crate::system::memory::{Equipartition, TimestampedEquipartition, TimestampedValues};

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
    pub pointer: T,
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
    pub pointer: T,
    pub timestamp: u32,
    pub prev_timestamp: u32,
    pub data: [T; N],
}

impl<T: Copy> MemoryReadRecord<T, 1> {
    pub fn value(&self) -> T {
        self.data[0]
    }
}

pub const INITIAL_TIMESTAMP: u32 = 0;

type BlockKey<F> = (F, usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Block<F> {
    address_space: F,
    pointer: usize,
    size: usize,
    timestamp: u32,
}

impl<F: PartialEq> Block<F> {
    pub fn contains(&self, address_space: F, pointer: usize) -> bool {
        self.address_space == address_space
            && self.pointer <= pointer
            && pointer < self.pointer + self.size
    }
}

/// A partition of data into blocks where each block has size a power of two.
#[derive(Debug)]
pub struct Memory<F> {
    blocks: BTreeMap<BlockKey<F>, Block<F>>,
    data: HashMap<BlockKey<F>, F>,
    initial_block_size: usize,
    timestamp: u32,
}

impl<F: PrimeField> Memory<F> {
    /// Creates a new partition with the given initial block size.
    ///
    /// Panics if the initial block size is not a power of two.
    pub fn new<const N: usize>(initial_memory: &Equipartition<F, N>) -> Self {
        assert!(N.is_power_of_two());

        let mut blocks = BTreeMap::new();
        let mut data = HashMap::new();
        for (&(address_space, block_idx), values) in initial_memory {
            let pointer = block_idx * N;
            blocks.insert(
                (address_space, pointer),
                Block {
                    address_space,
                    pointer,
                    size: N,
                    timestamp: INITIAL_TIMESTAMP,
                },
            );
            for (i, value) in values.iter().enumerate() {
                data.insert((address_space, pointer + i), *value);
            }
        }
        Self {
            blocks,
            data,
            initial_block_size: N,
            timestamp: INITIAL_TIMESTAMP + 1,
        }
    }

    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Increments the current timestamp by one and returns the new value.
    pub fn increment_timestamp(&mut self) {
        self.timestamp += 1;
    }

    /// Increments the current timestamp by a specified delta and returns the new value.
    pub fn increment_timestamp_by(&mut self, delta: u32) {
        self.timestamp += delta;
    }

    /// Writes an array of values to the memory at the specified address space and start index.
    pub fn write<const N: usize>(
        &mut self,
        address_space: F,
        pointer: usize,
        values: [F; N],
    ) -> (MemoryWriteRecord<F, N>, Vec<AccessAdapterRecord<F>>) {
        assert!(N.is_power_of_two());

        let mut adapter_records = vec![];
        self.access(address_space, pointer, N, &mut adapter_records);

        let block = self.blocks.get_mut(&(address_space, pointer)).unwrap();
        let prev_timestamp = block.timestamp;
        block.timestamp = self.timestamp;

        debug_assert!(prev_timestamp < self.timestamp);

        let prev_data = array::from_fn(|i| {
            self.data
                .insert((address_space, pointer + i), values[i])
                .unwrap_or(F::zero())
        });

        let record = MemoryWriteRecord {
            address_space,
            pointer: F::from_canonical_usize(pointer),
            timestamp: self.timestamp,
            prev_timestamp,
            data: values,
            prev_data,
        };
        self.increment_timestamp();
        (record, adapter_records)
    }

    /// Reads an array of values from the memory at the specified address space and start index.
    pub fn read<const N: usize>(
        &mut self,
        address_space: F,
        pointer: usize,
    ) -> (MemoryReadRecord<F, N>, Vec<AccessAdapterRecord<F>>) {
        assert!(N.is_power_of_two());

        let mut adapter_records = vec![];
        self.access(address_space, pointer, N, &mut adapter_records);

        let block = self.blocks.get_mut(&(address_space, pointer)).unwrap();
        let prev_timestamp = block.timestamp;
        block.timestamp = self.timestamp;

        debug_assert!(prev_timestamp < self.timestamp);

        let record = MemoryReadRecord {
            address_space,
            pointer: F::from_canonical_usize(pointer),
            timestamp: self.timestamp,
            prev_timestamp,
            data: self.get_range_array::<N>(address_space, pointer),
        };

        self.increment_timestamp();
        (record, adapter_records)
    }

    pub fn finalize<const N: usize>(
        &mut self,
    ) -> (TimestampedEquipartition<F, N>, Vec<AccessAdapterRecord<F>>) {
        let mut adapter_records = vec![];

        // First make sure the partition we maintain in self.blocks is an equipartition.
        let mut left = Unbounded;
        while let Some((_, &block)) = self.blocks.range((left, Unbounded)).next() {
            let aligned_pointer = (block.pointer / N) * N;
            if aligned_pointer != block.pointer || block.size != N {
                self.access(
                    block.address_space,
                    aligned_pointer,
                    N,
                    &mut adapter_records,
                );
            }
            left = Included((block.address_space, aligned_pointer + N));
        }

        let mut equipartition = TimestampedEquipartition::<F, N>::new();
        for (&(address_space, pointer), block) in self.blocks.iter() {
            debug_assert_eq!(block.pointer % N, 0);
            debug_assert_eq!(block.size, N);

            equipartition.insert(
                (address_space, pointer / N),
                TimestampedValues {
                    timestamp: block.timestamp,
                    values: self.get_range_array::<N>(address_space, pointer),
                },
            );
        }

        (equipartition, adapter_records)
    }

    // Modifies the partition to ensure that there is a block starting at (address_space, pointer).
    fn make_boundary(
        &mut self,
        address_space: F,
        pointer: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        loop {
            let b = self.block_containing(address_space, pointer);
            if b.pointer == pointer {
                break;
            }
            self.split_block(address_space, b.pointer, records);
        }
    }

    fn access(
        &mut self,
        address_space: F,
        pointer: usize,
        size: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        self.make_boundary(address_space, pointer, records);
        self.make_boundary(address_space, pointer + size, records);

        let b = self
            .blocks
            .entry((address_space, pointer))
            .or_insert_with(|| Block {
                address_space,
                pointer,
                size: self.initial_block_size,
                timestamp: INITIAL_TIMESTAMP,
            });
        if b.pointer == pointer && b.size == size {
            return;
        }
        assert!(size > 1);

        // Now recursively access left and right blocks to ensure they are in the partition.
        let half_size = size / 2;
        self.access(address_space, pointer, half_size, records);
        self.access(address_space, pointer + half_size, half_size, records);

        self.merge_block_with_next(address_space, pointer, records);
    }

    // Splits the block starting at (address_space, pointer) into two blocks.
    //
    // Panics if there is no block starting at (address_space, pointer).
    fn split_block(
        &mut self,
        address_space: F,
        pointer: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        let block = self
            .blocks
            .remove(&(address_space, pointer))
            .unwrap_or(self.initial_block(address_space, pointer));

        assert!(block.size >= 2);

        let half_size = block.size / 2;
        self.blocks.insert(
            (address_space, pointer),
            Block {
                address_space,
                pointer,
                size: half_size,
                timestamp: block.timestamp,
            },
        );
        self.blocks.insert(
            (address_space, pointer + half_size),
            Block {
                address_space,
                pointer: pointer + half_size,
                size: half_size,
                timestamp: block.timestamp,
            },
        );
        records.push(AccessAdapterRecord {
            timestamp: block.timestamp,
            address_space,
            start_index: F::from_canonical_usize(pointer),
            data: self.get_range(address_space, pointer, block.size),
            kind: AccessAdapterRecordKind::Split,
        });
    }

    /// Merges the two adjacent blocks starting at (address_space, pointer).
    ///
    /// Panics if there is no block starting at (address_space, pointer) or if the two blocks
    /// do not have the same size.
    fn merge_block_with_next(
        &mut self,
        address_space: F,
        pointer: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        let left = self
            .blocks
            .remove(&(address_space, pointer))
            .unwrap_or(self.initial_block(address_space, pointer));
        let size = left.size;
        let right = self
            .blocks
            .remove(&(address_space, pointer + size))
            .unwrap_or(self.initial_block(address_space, pointer + size));
        assert_eq!(size, right.size);

        let timestamp = max(left.timestamp, right.timestamp);
        self.blocks.insert(
            (address_space, pointer),
            Block {
                address_space,
                pointer,
                size: 2 * size,
                timestamp,
            },
        );
        records.push(AccessAdapterRecord {
            timestamp,
            address_space,
            start_index: F::from_canonical_usize(pointer),
            data: self.get_range(address_space, pointer, 2 * size),
            kind: AccessAdapterRecordKind::Merge {
                left_timestamp: left.timestamp,
                right_timestamp: right.timestamp,
            },
        });
    }

    fn block_containing(&mut self, address_space: F, pointer: usize) -> Block<F> {
        // Look for the block with the largest key <= (address_space, pointer)
        let key = (address_space, pointer);

        if let Some((_, block)) = self.blocks.range((Unbounded, Included(key))).next_back() {
            if block.contains(address_space, pointer) {
                return *block;
            }
        }
        let aligned_pointer = (pointer / self.initial_block_size) * self.initial_block_size;
        self.initial_block(address_space, aligned_pointer)
    }

    fn initial_block(&self, address_space: F, pointer: usize) -> Block<F> {
        Block {
            address_space,
            pointer,
            size: self.initial_block_size,
            timestamp: INITIAL_TIMESTAMP,
        }
    }

    pub fn get(&self, address_space: F, pointer: usize) -> F {
        *self
            .data
            .get(&(address_space, pointer))
            .unwrap_or(&F::zero())
    }

    fn get_range_array<const N: usize>(&self, address_space: F, pointer: usize) -> [F; N] {
        array::from_fn(|i| {
            *self
                .data
                .get(&(address_space, pointer + i))
                .unwrap_or(&F::zero())
        })
    }

    fn get_range(&self, address_space: F, pointer: usize, len: usize) -> Vec<F> {
        (0..len)
            .map(|i| {
                *self
                    .data
                    .get(&(address_space, pointer + i))
                    .unwrap_or(&F::zero())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    use super::{Block, Memory};
    use crate::system::memory::{
        manager::memory::{AccessAdapterRecord, AccessAdapterRecordKind},
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
    fn test_partition() {
        type F = BabyBear;

        let mut partition = Memory::<F>::new(&Equipartition::<F, 8>::new());
        assert_eq!(
            partition.block_containing(F::zero(), 13),
            Block {
                address_space: F::zero(),
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(F::zero(), 13),
            Block {
                address_space: F::zero(),
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(F::zero(), 8),
            Block {
                address_space: F::zero(),
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(F::zero(), 15),
            Block {
                address_space: F::zero(),
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(F::zero(), 16),
            Block {
                address_space: F::zero(),
                pointer: 16,
                size: 8,
                timestamp: 0,
            }
        );
    }

    #[test]
    fn test_write_read_initial_block_len_1() {
        let initial_memory = Equipartition::<BabyBear, 1>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);
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
        let mut memory = Memory::<BabyBear>::new(&initial_memory);
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
        let mut memory = Memory::<BabyBear>::new(&initial_memory);

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
                pointer: bb!(0),
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
                pointer: bb!(0),
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
                pointer: bb!(0),
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
                pointer: bb!(0),
                timestamp: 4,
                prev_timestamp: 3,
                data: bba![10, 11, 3, 4],
            }
        );
    }

    #[test]
    fn test_records_initial_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);

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
                pointer: bb!(0),
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
                pointer: bb!(0),
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
                pointer: bb!(0),
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
                pointer: bb!(0),
                timestamp: 4,
                prev_timestamp: 3,
                data: bba![10, 11, 3, 4],
            }
        );
    }

    #[test]
    fn test_get_initial_block_len_1() {
        let initial_memory = Equipartition::<BabyBear, 1>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);

        memory.write(bb!(1), 0, bba![4, 3, 2, 1]);

        assert_eq!(memory.get(bb!(1), 0), BabyBear::from_canonical_u32(4));
        assert_eq!(memory.get(bb!(1), 1), BabyBear::from_canonical_u32(3));
        assert_eq!(memory.get(bb!(1), 2), BabyBear::from_canonical_u32(2));
        assert_eq!(memory.get(bb!(1), 3), BabyBear::from_canonical_u32(1));
        assert_eq!(memory.get(bb!(1), 5), BabyBear::zero());

        assert_eq!(memory.get(bb!(0), 0), BabyBear::zero());
    }

    #[test]
    fn test_get_initial_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);

        memory.write(bb!(1), 0, bba![4, 3, 2, 1]);

        assert_eq!(memory.get(bb!(1), 0), BabyBear::from_canonical_u32(4));
        assert_eq!(memory.get(bb!(1), 1), BabyBear::from_canonical_u32(3));
        assert_eq!(memory.get(bb!(1), 2), BabyBear::from_canonical_u32(2));
        assert_eq!(memory.get(bb!(1), 3), BabyBear::from_canonical_u32(1));
        assert_eq!(memory.get(bb!(1), 5), BabyBear::zero());
        assert_eq!(memory.get(bb!(1), 9), BabyBear::zero());
        assert_eq!(memory.get(bb!(0), 0), BabyBear::zero());
    }

    #[test]
    fn test_finalize_empty() {
        let initial_memory = Equipartition::<BabyBear, 4>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);

        let (memory, records) = memory.finalize::<4>();
        assert_eq!(memory.len(), 0);
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_finalize_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);
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
