use std::{array, sync::Arc};

use openvm_circuit_primitives::{
    assert_less_than::AssertLtSubAir, var_range::VariableRangeCheckerChip,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{paged_vec::PagedVec, van_emde_boas::VebTree};
use crate::{
    arch::MemoryConfig,
    system::memory::{
        adapter::{AccessAdapterRecord, AccessAdapterRecordKind},
        offline_checker::{MemoryBridge, MemoryBus},
        paged_vec::PAGE_SIZE,
        van_emde_boas::VanEmdeBoas,
        MemoryAuxColsFactory, MemoryImage, RecordId, TimestampedEquipartition, TimestampedValues,
    },
};

pub const INITIAL_TIMESTAMP: u32 = 0;

#[derive(Clone, PartialEq, Eq, Debug)]
struct BlockData {
    size: usize,
    timestamp: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecord<T> {
    pub address_space: T,
    pub pointer: T,
    pub timestamp: u32,
    pub prev_timestamp: u32,
    pub data: Vec<T>,
    /// None if a read.
    pub prev_data: Option<Vec<T>>,
}

pub struct OfflineMemory<F> {
    block_data: Vec<VebTree<BlockData>>,
    data: Vec<PagedVec<F, PAGE_SIZE>>,
    as_offset: u32,
    initial_block_size: usize,
    timestamp: u32,
    timestamp_max_bits: usize,

    memory_bus: MemoryBus,
    range_checker: Arc<VariableRangeCheckerChip>,

    log: Vec<Option<MemoryRecord<F>>>,
}

impl<F: PrimeField32> OfflineMemory<F> {
    /// Creates a new partition with the given initial block size.
    ///
    /// Panics if the initial block size is not a power of two.
    pub fn new(
        initial_memory: MemoryImage<F>,
        initial_block_size: usize,
        memory_bus: MemoryBus,
        range_checker: Arc<VariableRangeCheckerChip>,
        timestamp_max_bits: usize,
        config: MemoryConfig,
    ) -> Self {
        assert!(initial_block_size.is_power_of_two());
        eprintln!("initial_block_size: {}", initial_block_size);

        Self {
            block_data: vec![VanEmdeBoas::new(); 1 << config.as_height],
            data: Self::memory_image_to_paged_vec(initial_memory, config),
            as_offset: config.as_offset,
            initial_block_size,
            timestamp: INITIAL_TIMESTAMP + 1,
            timestamp_max_bits,
            memory_bus,
            range_checker,
            log: vec![],
        }
    }

    pub fn set_initial_memory(&mut self, initial_memory: MemoryImage<F>, config: MemoryConfig) {
        assert_eq!(self.timestamp, INITIAL_TIMESTAMP + 1);
        self.data = Self::memory_image_to_paged_vec(initial_memory, config);
    }

    fn memory_image_to_paged_vec(
        memory_image: MemoryImage<F>,
        config: MemoryConfig,
    ) -> Vec<PagedVec<F, PAGE_SIZE>> {
        let start = std::time::Instant::now();
        let mut paged_vec =
            vec![PagedVec::new((1 << config.pointer_max_bits) / PAGE_SIZE); 1 << config.as_height];
        for ((addr_space, pointer), value) in memory_image.items() {
            paged_vec[(addr_space - config.as_offset) as usize].insert(pointer as usize, value);
        }
        eprintln!(
            "- - - - - - - - - - - - - - - memory_image_to_paged_vec time: {:?}",
            start.elapsed()
        );
        paged_vec
    }

    pub(super) fn set_log_capacity(&mut self, access_capacity: usize) {
        assert!(self.log.is_empty());
        self.log = Vec::with_capacity(access_capacity);
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        MemoryBridge::new(
            self.memory_bus,
            self.timestamp_max_bits,
            self.range_checker.bus(),
        )
    }

    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Increments the current timestamp by one and returns the new value.
    pub fn increment_timestamp(&mut self) {
        self.increment_timestamp_by(1)
    }

    /// Increments the current timestamp by a specified delta and returns the new value.
    pub fn increment_timestamp_by(&mut self, delta: u32) {
        self.log.push(None);
        self.timestamp += delta;
    }

    /// Writes an array of values to the memory at the specified address space and start index.
    pub fn write(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: Vec<F>,
    ) -> Vec<AccessAdapterRecord<F>> {
        let len = values.len();
        assert!(len.is_power_of_two());
        assert_ne!(address_space, 0);

        let mut adapter_records = vec![];
        let prev_timestamp =
            self.access_updating_timestamp(address_space, pointer, len, &mut adapter_records);

        debug_assert!(prev_timestamp < self.timestamp);

        let pointer = pointer as usize;
        let prev_data = self.data[(address_space - self.as_offset) as usize]
            .set_range(pointer..pointer + len, &values);

        let record = MemoryRecord {
            address_space: F::from_canonical_u32(address_space),
            pointer: F::from_canonical_usize(pointer),
            timestamp: self.timestamp,
            prev_timestamp,
            data: values,
            prev_data: Some(prev_data),
        };
        self.log.push(Some(record));
        self.timestamp += 1;
        adapter_records
    }

    /// Reads an array of values from the memory at the specified address space and start index.
    pub fn read(
        &mut self,
        address_space: u32,
        pointer: u32,
        len: usize,
    ) -> Vec<AccessAdapterRecord<F>> {
        assert!(len.is_power_of_two());
        if address_space == 0 {
            let pointer = F::from_canonical_u32(pointer);
            self.log.push(Some(MemoryRecord {
                address_space: F::ZERO,
                pointer,
                timestamp: self.timestamp,
                prev_timestamp: 0,
                data: vec![pointer],
                prev_data: None,
            }));
            self.timestamp += 1;
            return vec![];
        }

        let mut adapter_records = vec![];
        let prev_timestamp =
            self.access_updating_timestamp(address_space, pointer, len, &mut adapter_records);

        debug_assert!(prev_timestamp < self.timestamp);

        let values = self.range_vec(address_space, pointer, len);

        self.log.push(Some(MemoryRecord {
            address_space: F::from_canonical_u32(address_space),
            pointer: F::from_canonical_u32(pointer),
            timestamp: self.timestamp,
            prev_timestamp,
            data: values,
            prev_data: None,
        }));
        self.timestamp += 1;
        adapter_records
    }

    pub fn record_by_id(&self, id: RecordId) -> &MemoryRecord<F> {
        self.log[id.0].as_ref().unwrap()
    }

    pub fn finalize<const N: usize>(
        &mut self,
    ) -> (TimestampedEquipartition<F, N>, Vec<AccessAdapterRecord<F>>) {
        let start = std::time::Instant::now();
        let mut adapter_records = vec![];
        eprintln!("- - - - - - - - - - - - - - - finalize start");

        // First make sure the partition we maintain in self.block_data is an equipartition.
        // Grab all aligned pointers that need to be re-accessed.
        let as_offset = self.as_offset;
        let to_access: Vec<_> = self
            .block_data
            .iter()
            .enumerate()
            .flat_map(|(address_space, tree)| {
                tree.iter().flat_map(move |(pointer, block_data)| {
                    (pointer..pointer + block_data.size as u32)
                        .filter(move |&p| p % N as u32 == 0)
                        .map(move |ptr| (address_space as u32 + as_offset, ptr))
                })
            })
            .collect::<Vec<_>>();
        eprintln!(
            "- - - - - - - - - - - - - - - to_access len: {}",
            to_access.len()
        );
        eprintln!(
            "- - - - - - - - - - - - - - - total data len: {:?}",
            self.log
                .iter()
                .map(|x| x.as_ref().map_or(0, |y| y.data.len()))
                .sum::<usize>()
        );

        for &(address_space, pointer) in to_access.iter() {
            self.access(address_space, pointer, N, &mut adapter_records);
        }

        let mut equipartition = TimestampedEquipartition::<F, N>::new();
        for (address_space, pointer) in to_access {
            let block = self.block_data[(address_space - self.as_offset) as usize]
                .get(pointer)
                .unwrap();

            debug_assert_eq!(pointer % N as u32, 0);
            debug_assert_eq!(block.size, N);

            equipartition.insert(
                (address_space, pointer / N as u32),
                TimestampedValues {
                    timestamp: block.timestamp,
                    values: self.range_array::<N>(address_space, pointer),
                },
            );
        }

        eprintln!(
            "- - - - - - - - - - - - - - - finalize time: {:?}",
            start.elapsed()
        );

        (equipartition, adapter_records)
    }

    fn access_updating_timestamp(
        &mut self,
        address_space: u32,
        pointer: u32,
        size: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) -> u32 {
        self.access(address_space, pointer, size, records);
        let tree = &mut self.block_data[(address_space - self.as_offset) as usize];
        let key = tree.max_not_exceeding(pointer).unwrap();
        tree.insert(
            key,
            BlockData {
                size,
                timestamp: self.timestamp,
            },
        )
        .unwrap()
        .timestamp
    }

    #[allow(clippy::too_many_arguments)]
    fn split_into_aligned(
        address_space: u32,
        left: u32,
        len: u32,
        target_left: u32,
        target_right: u32,
        timestamp: u32,
        data: &[F],
        records: &mut Vec<AccessAdapterRecord<F>>,
        blocks_to_merge: &mut Vec<(u32, u32, u32)>,
    ) {
        if left >= target_right || left + len <= target_left {
            blocks_to_merge.push((left, len, timestamp));
            return;
        }
        if left.wrapping_sub(target_left) & (len - 1) == 0
            && left >= target_left
            && left + len <= target_right
        {
            blocks_to_merge.push((left, len, timestamp));
            return;
        }
        records.push(AccessAdapterRecord {
            timestamp,
            address_space: F::from_canonical_u32(address_space),
            start_index: F::from_canonical_u32(left),
            data: data.to_vec(),
            kind: AccessAdapterRecordKind::Split,
        });
        let len = len / 2;
        Self::split_into_aligned(
            address_space,
            left,
            len,
            target_left,
            target_right,
            timestamp,
            &data[..len as usize],
            records,
            blocks_to_merge,
        );
        Self::split_into_aligned(
            address_space,
            left + len,
            len,
            target_left,
            target_right,
            timestamp,
            &data[len as usize..],
            records,
            blocks_to_merge,
        );
    }

    fn access(
        &mut self,
        address_space: u32,
        pointer: u32,
        size: usize,
        records: &mut Vec<AccessAdapterRecord<F>>,
    ) {
        let left = pointer;
        let right = pointer + size as u32;
        let data = self.range_vec(address_space, left, size);
        let tree = &mut self.block_data[(address_space - self.as_offset) as usize];
        let mut done_left = right;
        // Maintain the following invariant: the half-interval [done_left, right) is splitted into aligned initialized blocks.
        let mut blocks_to_split = vec![];
        while done_left > left {
            if let Some(key) = tree.max_not_exceeding(done_left - 1) {
                let block_data = tree.get(key).unwrap().clone();
                let new_right = key + block_data.size as u32;
                while done_left > new_right && done_left > left {
                    done_left -= 1 + (done_left - 1) % self.initial_block_size as u32;
                    blocks_to_split.push((
                        done_left,
                        self.initial_block_size as u32,
                        INITIAL_TIMESTAMP,
                    ));
                }
                if key + block_data.size as u32 <= left {
                    break;
                }
                tree.erase(key);
                debug_assert!(block_data.size.is_power_of_two());
                debug_assert!(done_left <= key + block_data.size as u32);
                blocks_to_split.push((key, block_data.size as u32, block_data.timestamp));
                done_left = key;
            } else {
                while done_left > left {
                    done_left -= 1 + (done_left - 1) % self.initial_block_size as u32;
                    blocks_to_split.push((
                        done_left,
                        self.initial_block_size as u32,
                        INITIAL_TIMESTAMP,
                    ));
                }
                break;
            }
        }

        let mut blocks_to_merge = vec![];
        for (left, len, timestamp) in blocks_to_split {
            let data = self.data[(address_space - self.as_offset) as usize]
                .get_range(left as usize..(left + len) as usize);
            Self::split_into_aligned(
                address_space,
                left,
                len,
                pointer,
                pointer + size as u32,
                timestamp,
                &data,
                records,
                &mut blocks_to_merge,
            );
        }

        blocks_to_merge.sort_by(|a, b| a.0.cmp(&b.0));

        let mut stack = vec![];
        for (left, len, timestamp) in blocks_to_merge {
            if left + len <= pointer || left >= pointer + size as u32 {
                tree.insert(
                    left,
                    BlockData {
                        size: len as usize,
                        timestamp,
                    },
                );
                continue;
            }
            stack.push((left, len, timestamp));
            while stack.len() > 1 && stack[stack.len() - 1].1 == stack[stack.len() - 2].1 {
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let timestamp = lhs.2.max(rhs.2);
                records.push(AccessAdapterRecord {
                    timestamp,
                    address_space: F::from_canonical_u32(address_space),
                    start_index: F::from_canonical_u32(lhs.0),
                    data: data
                        [(lhs.0 - pointer) as usize..(lhs.0 + lhs.1 + rhs.1 - pointer) as usize]
                        .to_vec(),
                    kind: AccessAdapterRecordKind::Merge {
                        left_timestamp: lhs.2,
                        right_timestamp: rhs.2,
                    },
                });
                stack.push((lhs.0, lhs.1 + rhs.1, timestamp));
            }
        }
        tree.insert(
            pointer,
            BlockData {
                size,
                timestamp: stack.last().unwrap().2,
            },
        );
    }

    pub fn get(&self, address_space: u32, pointer: u32) -> F {
        if let Some(x) = self.data[(address_space - self.as_offset) as usize].get(pointer as usize)
        {
            *x
        } else {
            F::ZERO
        }
    }

    fn range_array<const N: usize>(&self, address_space: u32, pointer: u32) -> [F; N] {
        array::from_fn(|i| self.get(address_space, pointer + i as u32))
    }

    fn range_vec(&self, address_space: u32, pointer: u32, len: usize) -> Vec<F> {
        let pointer = pointer as usize;
        self.data[(address_space - self.as_offset) as usize].get_range(pointer..pointer + len)
    }

    pub fn aux_cols_factory(&self) -> MemoryAuxColsFactory<F> {
        let range_bus = self.range_checker.bus();
        MemoryAuxColsFactory {
            range_checker: self.range_checker.clone(),
            timestamp_lt_air: AssertLtSubAir::new(range_bus, self.timestamp_max_bits),
            _marker: Default::default(),
        }
    }

    // just for unit testing
    #[cfg(test)]
    fn last_record(&self) -> &MemoryRecord<F> {
        self.log.last().unwrap().as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::{MemoryRecord, OfflineMemory};
    use crate::{
        arch::MemoryConfig,
        system::memory::{
            adapter::{AccessAdapterRecord, AccessAdapterRecordKind},
            offline_checker::MemoryBus,
            MemoryImage, TimestampedValues,
        },
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
        let initial_memory = MemoryImage::new(1, 1, 1 << 29);
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            1,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            Default::default(),
        );
        let address_space = 1;

        memory.write(address_space, 0, bbvec![1, 2, 3, 4]);

        memory.read(address_space, 0, 2);
        let read_record = memory.last_record();
        assert_eq!(read_record.data, bba![1, 2]);

        memory.write(address_space, 2, bbvec![100]);

        memory.read(address_space, 0, 4);
        let read_record = memory.last_record();
        assert_eq!(read_record.data, bba![1, 2, 100, 4]);
    }

    #[test]
    fn test_records_initial_block_len_1() {
        let initial_memory = MemoryImage::new(1, 1, 1 << 29);
        // TODO: Ideally we don't need to instantiate all this stuff since we are just testing the data structure.
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            1,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            Default::default(),
        );

        let adapter_records = memory.write(1, 0, bbvec![1, 2, 3, 4]);

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
        let write_record = memory.last_record();
        assert_eq!(
            write_record,
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 1,
                prev_timestamp: 0,
                data: bbvec![1, 2, 3, 4],
                prev_data: Some(bbvec![0, 0, 0, 0]),
            }
        );
        assert_eq!(memory.timestamp(), 2);

        let adapter_records = memory.read(1, 0, 4);
        let read_record = memory.last_record();
        // At time 2 we read [0:4].
        assert_eq!(adapter_records.len(), 0);
        assert_eq!(
            read_record,
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 2,
                prev_timestamp: 1,
                data: bbvec![1, 2, 3, 4],
                prev_data: None,
            }
        );
        assert_eq!(memory.timestamp(), 3);

        let adapter_records = memory.write(1, 0, bbvec![10, 11]);
        let write_record = memory.last_record();
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
            write_record,
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 3,
                prev_timestamp: 2,
                data: bbvec![10, 11],
                prev_data: Some(bbvec![1, 2]),
            }
        );

        let adapter_records = memory.read(1, 0, 4);
        let read_record = memory.last_record();
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
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 4,
                prev_timestamp: 3,
                data: bbvec![10, 11, 3, 4],
                prev_data: None,
            }
        );
    }

    #[test]
    fn test_records_initial_block_len_8() {
        let initial_memory = MemoryImage::new(1, 1, 1 << 29);
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            8,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            Default::default(),
        );

        let adapter_records = memory.write(1, 0, bbvec![1, 2, 3, 4]);
        let write_record = memory.last_record();

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
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 1,
                prev_timestamp: 0,
                data: bbvec![1, 2, 3, 4],
                prev_data: Some(bbvec![0, 0, 0, 0]),
            }
        );
        assert_eq!(memory.timestamp(), 2);

        let adapter_records = memory.read(1, 0, 4);
        let read_record = memory.last_record();
        // At time 2 we read [0:4].
        assert_eq!(adapter_records.len(), 0);
        assert_eq!(
            read_record,
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 2,
                prev_timestamp: 1,
                data: bbvec![1, 2, 3, 4],
                prev_data: None,
            }
        );
        assert_eq!(memory.timestamp(), 3);

        let adapter_records = memory.write(1, 0, bbvec![10, 11]);
        let write_record = memory.last_record();
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
            write_record,
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 3,
                prev_timestamp: 2,
                data: bbvec![10, 11],
                prev_data: Some(bbvec![1, 2]),
            }
        );

        let adapter_records = memory.read(1, 0, 4);
        let read_record = memory.last_record();
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
            &MemoryRecord {
                address_space: bb!(1),
                pointer: bb!(0),
                timestamp: 4,
                prev_timestamp: 3,
                data: bbvec![10, 11, 3, 4],
                prev_data: None,
            }
        );
    }

    #[test]
    fn test_get_initial_block_len_1() {
        let initial_memory = MemoryImage::new(0, 2, 16);
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            1,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            MemoryConfig {
                as_offset: 0,
                ..Default::default()
            },
        );

        memory.write(1, 0, bbvec![4, 3, 2, 1]);

        assert_eq!(memory.get(1, 0), BabyBear::from_canonical_u32(4));
        assert_eq!(memory.get(1, 1), BabyBear::from_canonical_u32(3));
        assert_eq!(memory.get(1, 2), BabyBear::from_canonical_u32(2));
        assert_eq!(memory.get(1, 3), BabyBear::from_canonical_u32(1));
        assert_eq!(memory.get(1, 5), BabyBear::ZERO);

        assert_eq!(memory.get(0, 0), BabyBear::ZERO);
    }

    #[test]
    fn test_get_initial_block_len_8() {
        let initial_memory = MemoryImage::new(0, 2, 16);
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            8,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            MemoryConfig {
                as_offset: 0,
                ..Default::default()
            },
        );

        memory.write(1, 0, bbvec![4, 3, 2, 1]);

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
        let initial_memory = MemoryImage::new(1, 1, 1 << 29);
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            4,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            Default::default(),
        );

        let (memory, records) = memory.finalize::<4>();
        assert_eq!(memory.len(), 0);
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_finalize_block_len_8() {
        let initial_memory = MemoryImage::new(1, 1, 1 << 29);
        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            8,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            Default::default(),
        );
        // Make block 0:4 in address space 1 active.
        memory.write(1, 0, bbvec![1, 2, 3, 4]);

        // Make block 16:32 in address space 1 active.
        memory.write(
            1,
            16,
            bbvec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        );

        // Make block 64:72 in address space 2 active.
        memory.write(2, 64, bbvec![8, 7, 6, 5, 4, 3, 2, 1]);

        // Finalize to a partition of size 8.
        let (final_memory, records) = memory.finalize::<8>();
        assert_eq!(final_memory.len(), 4);
        assert_eq!(
            final_memory.get(&(1, 0)),
            Some(&TimestampedValues {
                values: bba![1, 2, 3, 4, 0, 0, 0, 0],
                timestamp: 1,
            })
        );
        // start_index = 16 corresponds to label = 2
        assert_eq!(
            final_memory.get(&(1, 2)),
            Some(&TimestampedValues {
                values: bba![1, 1, 1, 1, 1, 1, 1, 1],
                timestamp: 2,
            })
        );
        // start_index = 24 corresponds to label = 3
        assert_eq!(
            final_memory.get(&(1, 3)),
            Some(&TimestampedValues {
                values: bba![1, 1, 1, 1, 1, 1, 1, 1],
                timestamp: 2,
            })
        );
        // start_index = 64 corresponds to label = 8
        assert_eq!(
            final_memory.get(&(2, 8)),
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
        let mut initial_memory = MemoryImage::new(1, 1, 1 << 29);
        for i in 0..8 {
            initial_memory.insert((1, i), F::from_canonical_u32(i + 1));
            initial_memory.insert((1, 16 + i), F::from_canonical_u32(i + 1));
        }

        let mut memory = OfflineMemory::<BabyBear>::new(
            initial_memory,
            8,
            MemoryBus(0),
            Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
                1, 29,
            ))),
            29,
            Default::default(),
        );

        // Verify initial state of block 0 (pointers 0–8)
        memory.read(1, 0, 8);
        let initial_read_record_0 = memory.last_record();
        assert_eq!(initial_read_record_0.data, bbvec![1, 2, 3, 4, 5, 6, 7, 8]);

        // Verify initial state of block 2 (pointers 16–24)
        memory.read(1, 16, 8);
        let initial_read_record_2 = memory.last_record();
        assert_eq!(initial_read_record_2.data, bbvec![1, 2, 3, 4, 5, 6, 7, 8]);

        // Test: Write a partial block to block 0 (pointer 0) and read back partially and fully
        memory.write(1, 0, bbvec![9, 9, 9, 9]);
        memory.read(1, 0, 2);
        let partial_read_record = memory.last_record();
        assert_eq!(partial_read_record.data, bbvec![9, 9]);

        memory.read(1, 0, 8);
        let full_read_record_0 = memory.last_record();
        assert_eq!(full_read_record_0.data, bbvec![9, 9, 9, 9, 5, 6, 7, 8]);

        // Test: Write a single element to pointer 2 and verify read in different lengths
        memory.write(1, 2, bbvec![100]);
        memory.read(1, 1, 4);
        let read_record_4 = memory.last_record();
        assert_eq!(read_record_4.data, bbvec![9, 100, 9, 5]);

        memory.read(1, 2, 8);
        let full_read_record_2 = memory.last_record();
        assert_eq!(full_read_record_2.data, bba![100, 9, 5, 6, 7, 8, 0, 0]);

        // Test: Write and read at the last pointer in block 2 (pointer 23, part of key (1, 2))
        memory.write(1, 23, bbvec![77]);
        memory.read(1, 23, 2);
        let boundary_read_record = memory.last_record();
        assert_eq!(boundary_read_record.data, bba![77, 0]); // Last byte modified, ensuring boundary check

        // Test: Reading from an uninitialized block (should default to 0)
        memory.read(1, 10, 4);
        let default_read_record = memory.last_record();
        assert_eq!(default_read_record.data, bba![0, 0, 0, 0]);

        memory.read(1, 100, 4);
        let default_read_record = memory.last_record();
        assert_eq!(default_read_record.data, bba![0, 0, 0, 0]);

        // Test: Overwrite entire memory pointer 16–24 and verify
        memory.write(1, 16, bbvec![50, 50, 50, 50, 50, 50, 50, 50]);
        memory.read(1, 16, 8);
        let overwrite_read_record = memory.last_record();
        assert_eq!(
            overwrite_read_record.data,
            bba![50, 50, 50, 50, 50, 50, 50, 50]
        ); // Verify entire block overwrite
    }
}
