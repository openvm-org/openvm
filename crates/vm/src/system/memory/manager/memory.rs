use std::{array, cmp::max, fmt::Debug};

use openvm_stark_backend::p3_field::PrimeField32;
use rustc_hash::{FxHashMap, FxHashSet};

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

/// (address_space, pointer)
type Address = (usize, usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct BlockData {
    pointer: usize,
    size: usize,
    timestamp: u32,
}

/// A partition of data into blocks where each block has size a power of two.
#[derive(Debug)]
pub struct Memory<F> {
    block_data: FxHashMap<Address, BlockData>,
    data: FxHashMap<Address, F>,
    initial_block_size: usize,
    timestamp: u32,
}

impl<F: PrimeField32> Memory<F> {
    /// Creates a new partition with the given initial block size.
    ///
    /// Panics if the initial block size is not a power of two.
    pub fn new<const N: usize>(initial_memory: &Equipartition<F, N>) -> Self {
        assert!(N.is_power_of_two());

        let mut block_data = FxHashMap::default();
        let mut data = FxHashMap::default();
        for (&(address_space, block_idx), values) in initial_memory {
            let address_space_usize = address_space.as_canonical_u32() as usize;
            let pointer = block_idx * N;
            let block = BlockData {
                pointer,
                size: N,
                timestamp: INITIAL_TIMESTAMP,
            };
            for (i, value) in values.iter().enumerate() {
                data.insert((address_space_usize, pointer + i), *value);
                block_data.insert((address_space_usize, pointer + i), block);
            }
        }
        Self {
            block_data,
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
        address_space: usize,
        pointer: usize,
        values: [F; N],
    ) -> (MemoryWriteRecord<F, N>, Vec<AccessAdapterRecord<F>>) {
        assert!(N.is_power_of_two());

        let mut adapter_records = vec![];
        let prev_timestamp =
            self.access_updating_timestamp(address_space, pointer, N, &mut adapter_records);

        debug_assert!(prev_timestamp < self.timestamp);

        let prev_data = array::from_fn(|i| {
            self.data
                .insert((address_space, pointer + i), values[i])
                .unwrap_or(F::ZERO)
        });

        let record = MemoryWriteRecord {
            address_space: F::from_canonical_usize(address_space),
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
        address_space: usize,
        pointer: usize,
    ) -> (MemoryReadRecord<F, N>, Vec<AccessAdapterRecord<F>>) {
        assert!(N.is_power_of_two());

        let mut adapter_records = vec![];
        let prev_timestamp =
            self.access_updating_timestamp(address_space, pointer, N, &mut adapter_records);

        debug_assert!(prev_timestamp < self.timestamp);

        let record = MemoryReadRecord {
            address_space: F::from_canonical_usize(address_space),
            pointer: F::from_canonical_usize(pointer),
            timestamp: self.timestamp,
            prev_timestamp,
            data: self.range_array::<N>(address_space, pointer),
        };

        self.increment_timestamp();
        (record, adapter_records)
    }

    pub fn finalize<const N: usize>(
        &mut self,
    ) -> (TimestampedEquipartition<F, N>, Vec<AccessAdapterRecord<F>>) {
        let mut adapter_records = vec![];

        // First make sure the partition we maintain in self.block_data is an equipartition.
        // Grab all aligned pointers that need to be re-accessed.
        let to_access: FxHashSet<_> = self
            .block_data
            .keys()
            .map(|&(address_space, pointer)| (address_space, (pointer / N) * N))
            .collect();

        for &(address_space, pointer) in to_access.iter() {
            let block = self.block_data.get(&(address_space, pointer)).unwrap();
            if block.pointer != pointer || block.size != N {
                self.access(address_space, pointer, N, &mut adapter_records);
            }
        }

        let mut equipartition = TimestampedEquipartition::<F, N>::new();
        for (address_space, pointer) in to_access {
            let block = self.block_data.get(&(address_space, pointer)).unwrap();

            debug_assert_eq!(block.pointer % N, 0);
            debug_assert_eq!(block.size, N);

            equipartition.insert(
                (F::from_canonical_usize(address_space), pointer / N),
                TimestampedValues {
                    timestamp: block.timestamp,
                    values: self.range_array::<N>(address_space, pointer),
                },
            );
        }

        (equipartition, adapter_records)
    }

    // Modifies the partition to ensure that there is a block starting at (address_space, query).
    fn split_to_make_boundary(
        &mut self,
        address_space: usize,
        query: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        let original_block = self.block_containing(address_space, query);
        if original_block.pointer == query {
            return;
        }

        let data = self.range_vec(address_space, original_block.pointer, original_block.size);

        let timestamp = original_block.timestamp;

        let mut cur_ptr = original_block.pointer;
        let mut cur_size = original_block.size;
        while cur_size > 0 {
            // Split.
            records.push(AccessAdapterRecord {
                timestamp,
                address_space: F::from_canonical_usize(address_space),
                start_index: F::from_canonical_usize(cur_ptr),
                data: data
                    [cur_ptr - original_block.pointer..cur_ptr - original_block.pointer + cur_size]
                    .to_vec(),
                kind: AccessAdapterRecordKind::Split,
            });

            let half_size = cur_size / 2;

            if query <= cur_ptr + half_size {
                // The right is finalized; add it to the partition.
                let block = BlockData {
                    pointer: cur_ptr + half_size,
                    size: half_size,
                    timestamp,
                };
                for i in 0..half_size {
                    self.block_data
                        .insert((address_space, cur_ptr + half_size + i), block);
                }
            }
            if query >= cur_ptr + half_size {
                // The left is finalized; add it to the partition.
                let block = BlockData {
                    pointer: cur_ptr,
                    size: half_size,
                    timestamp,
                };
                for i in 0..half_size {
                    self.block_data.insert((address_space, cur_ptr + i), block);
                }
            }

            if cur_ptr + half_size <= query {
                cur_ptr += half_size;
            }

            if cur_ptr == query {
                break;
            }
            cur_size = half_size;
        }
    }

    fn access_updating_timestamp(
        &mut self,
        address_space: usize,
        pointer: usize,
        size: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) -> u32 {
        self.access(address_space, pointer, size, records);

        let mut prev_timestamp = None;

        for i in 0..size {
            let block = self
                .block_data
                .get_mut(&(address_space, pointer + i))
                .unwrap();
            debug_assert!(i == 0 || prev_timestamp == Some(block.timestamp));
            prev_timestamp = Some(block.timestamp);
            block.timestamp = self.timestamp;
        }
        prev_timestamp.unwrap()
    }

    fn access(
        &mut self,
        address_space: usize,
        pointer: usize,
        size: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        self.split_to_make_boundary(address_space, pointer, records);
        self.split_to_make_boundary(address_space, pointer + size, records);

        let block_data = self
            .block_data
            .get(&(address_space, pointer))
            .copied()
            .unwrap_or_else(|| {
                for i in 0..size {
                    self.block_data.insert(
                        (address_space, pointer + i),
                        self.initial_block_data(pointer + i),
                    );
                }
                self.initial_block_data(pointer)
            });

        if block_data.pointer == pointer && block_data.size == size {
            return;
        }
        assert!(size > 1);

        // Now recursively access left and right blocks to ensure they are in the partition.
        let half_size = size / 2;
        self.access(address_space, pointer, half_size, records);
        self.access(address_space, pointer + half_size, half_size, records);

        self.merge_block_with_next(address_space, pointer, records);
    }

    /// Merges the two adjacent blocks starting at (address_space, pointer).
    ///
    /// Panics if there is no block starting at (address_space, pointer) or if the two blocks
    /// do not have the same size.
    fn merge_block_with_next(
        &mut self,
        address_space: usize,
        pointer: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        let left_block = self.block_data.get(&(address_space, pointer)).unwrap();

        let left_timestamp = left_block.timestamp;
        let size = left_block.size;

        let right_timestamp = self
            .block_data
            .get(&(address_space, pointer + size))
            .map(|b| b.timestamp)
            .unwrap_or(INITIAL_TIMESTAMP);

        let timestamp = max(left_timestamp, right_timestamp);
        for i in 0..2 * size {
            self.block_data.insert(
                (address_space, pointer + i),
                BlockData {
                    pointer,
                    size: 2 * size,
                    timestamp,
                },
            );
        }
        records.push(AccessAdapterRecord {
            timestamp,
            address_space: F::from_canonical_usize(address_space),
            start_index: F::from_canonical_usize(pointer),
            data: self.range_vec(address_space, pointer, 2 * size),
            kind: AccessAdapterRecordKind::Merge {
                left_timestamp,
                right_timestamp,
            },
        });
    }

    fn block_containing(&mut self, address_space: usize, pointer: usize) -> BlockData {
        if let Some(block_data) = self.block_data.get(&(address_space, pointer)) {
            *block_data
        } else {
            self.initial_block_data(pointer)
        }
    }

    fn initial_block_data(&self, pointer: usize) -> BlockData {
        let aligned_pointer = (pointer / self.initial_block_size) * self.initial_block_size;
        BlockData {
            pointer: aligned_pointer,
            size: self.initial_block_size,
            timestamp: INITIAL_TIMESTAMP,
        }
    }

    pub fn get(&self, address_space: usize, pointer: usize) -> F {
        *self.data.get(&(address_space, pointer)).unwrap_or(&F::ZERO)
    }

    fn range_array<const N: usize>(&self, address_space: usize, pointer: usize) -> [F; N] {
        array::from_fn(|i| self.get(address_space, pointer + i))
    }

    fn range_vec(&self, address_space: usize, pointer: usize, len: usize) -> Vec<F> {
        (0..len)
            .map(|i| self.get(address_space, pointer + i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::AbstractField;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::{BlockData, Memory};
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
            partition.block_containing(0, 13),
            BlockData {
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(0, 8),
            BlockData {
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(0, 15),
            BlockData {
                pointer: 8,
                size: 8,
                timestamp: 0,
            }
        );

        assert_eq!(
            partition.block_containing(0, 16),
            BlockData {
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
        let address_space = 1;

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
        let address_space = 1;

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

        let (write_record, adapter_records) = memory.write(1, 0, bba![1, 2, 3, 4]);

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

        let (read_record, adapter_records) = memory.read::<4>(1, 0);
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

        let (read_record, adapter_records) = memory.write::<2>(1, 0, bba![10, 11]);
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

        let (read_record, adapter_records) = memory.read::<4>(1, 0);
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

        let (write_record, adapter_records) = memory.write(1, 0, bba![1, 2, 3, 4]);

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

        let (read_record, adapter_records) = memory.read::<4>(1, 0);
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

        let (read_record, adapter_records) = memory.write::<2>(1, 0, bba![10, 11]);
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

        let (read_record, adapter_records) = memory.read::<4>(1, 0);
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

        memory.write(1, 0, bba![4, 3, 2, 1]);

        assert_eq!(memory.get(1, 0), BabyBear::from_canonical_u32(4));
        assert_eq!(memory.get(1, 1), BabyBear::from_canonical_u32(3));
        assert_eq!(memory.get(1, 2), BabyBear::from_canonical_u32(2));
        assert_eq!(memory.get(1, 3), BabyBear::from_canonical_u32(1));
        assert_eq!(memory.get(1, 5), BabyBear::ZERO);

        assert_eq!(memory.get(0, 0), BabyBear::ZERO);
    }

    #[test]
    fn test_get_initial_block_len_8() {
        let initial_memory = Equipartition::<BabyBear, 8>::new();
        let mut memory = Memory::<BabyBear>::new(&initial_memory);

        memory.write(1, 0, bba![4, 3, 2, 1]);

        assert_eq!(memory.get(1, 0), BabyBear::from_canonical_u32(4));
        assert_eq!(memory.get(1, 1), BabyBear::from_canonical_u32(3));
        assert_eq!(memory.get(1, 2), BabyBear::from_canonical_u32(2));
        assert_eq!(memory.get(1, 3), BabyBear::from_canonical_u32(1));
        assert_eq!(memory.get(1, 5), BabyBear::ZERO);
        assert_eq!(memory.get(1, 9), BabyBear::ZERO);
        assert_eq!(memory.get(0, 0), BabyBear::ZERO);
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
        memory.write(1, 0, bba![1, 2, 3, 4]);

        // Make block 16:32 in address space 1 active.
        memory.write(1, 16, bba![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

        // Make block 64:72 in address space 2 active.
        memory.write(2, 64, bba![8, 7, 6, 5, 4, 3, 2, 1]);

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

    #[test]
    fn test_write_read_initial_block_len_8_initial_memory() {
        type F = BabyBear;

        // Initialize initial memory with blocks at indices 0 and 2
        let mut initial_memory = Equipartition::<F, 8>::new();
        initial_memory.insert((F::ONE, 0), bba![1, 2, 3, 4, 5, 6, 7, 8]); // Block 0, pointers 0–8
        initial_memory.insert((F::ONE, 2), bba![1, 2, 3, 4, 5, 6, 7, 8]); // Block 2, pointers 16–24

        let mut memory = Memory::new(&initial_memory);

        // Verify initial state of block 0 (pointers 0–8)
        let (initial_read_record_0, _) = memory.read::<8>(1, 0);
        assert_eq!(initial_read_record_0.data, bba![1, 2, 3, 4, 5, 6, 7, 8]);

        // Verify initial state of block 2 (pointers 16–24)
        let (initial_read_record_2, _) = memory.read::<8>(1, 16);
        assert_eq!(initial_read_record_2.data, bba![1, 2, 3, 4, 5, 6, 7, 8]);

        // Test: Write a partial block to block 0 (pointer 0) and read back partially and fully
        memory.write(1, 0, bba![9, 9, 9, 9]);
        let (partial_read_record, _) = memory.read::<2>(1, 0);
        assert_eq!(partial_read_record.data, bba![9, 9]);

        let (full_read_record_0, _) = memory.read::<8>(1, 0);
        assert_eq!(full_read_record_0.data, bba![9, 9, 9, 9, 5, 6, 7, 8]);

        // Test: Write a single element to pointer 2 and verify read in different lengths
        memory.write(1, 2, bba![100]);
        let (read_record_4, _) = memory.read::<4>(1, 1);
        assert_eq!(read_record_4.data, bba![9, 100, 9, 5]);

        let (full_read_record_2, _) = memory.read::<8>(1, 2);
        assert_eq!(full_read_record_2.data, bba![100, 9, 5, 6, 7, 8, 0, 0]);

        // Test: Write and read at the last pointer in block 2 (pointer 23, part of key (1, 2))
        memory.write(1, 23, bba![77]);
        let (boundary_read_record, _) = memory.read::<2>(1, 23);
        assert_eq!(boundary_read_record.data, bba![77, 0]); // Last byte modified, ensuring boundary check

        // Test: Reading from an uninitialized block (should default to 0)
        let (default_read_record, _) = memory.read::<4>(1, 10);
        assert_eq!(default_read_record.data, bba![0, 0, 0, 0]);

        let (default_read_record, _) = memory.read::<4>(1, 100);
        assert_eq!(default_read_record.data, bba![0, 0, 0, 0]);

        // Test: Overwrite entire memory pointer 16–24 and verify
        memory.write(1, 16, bba![50, 50, 50, 50, 50, 50, 50, 50]);
        let (overwrite_read_record, _) = memory.read::<8>(1, 16);
        assert_eq!(
            overwrite_read_record.data,
            bba![50, 50, 50, 50, 50, 50, 50, 50]
        ); // Verify entire block overwrite
    }
}
