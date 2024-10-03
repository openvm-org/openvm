use std::{borrow::BorrowMut, cmp::max, collections::HashMap};

use afs_primitives::is_less_than::columns::IsLessThanAuxColsMut;
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::{
    adapter::AccessAdapterCols, manager::memory::AccessAdapterRecordKind, AddressSpace,
    MemoryAddress, MemoryChip, TimestampedValue,
};

impl<F: PrimeField32> MemoryChip<F> {
    pub fn generate_memory_interface_trace(&self) -> RowMajorMatrix<F> {
        let all_addresses = self.interface_chip.all_addresses();
        let mut final_memory = HashMap::new();
        for (addr_space, pointer) in all_addresses {
            let (timestamp, &value) = self
                .memory
                .get(
                    AddressSpace(addr_space.as_canonical_u32()),
                    pointer.as_canonical_u32() as usize,
                )
                .unwrap();
            final_memory.insert(
                (addr_space, pointer),
                TimestampedValue {
                    timestamp: F::from_canonical_u32(timestamp),
                    value,
                },
            );
        }

        self.interface_chip.generate_trace(final_memory)
    }

    pub fn generate_access_adapter_trace<const N: usize>(&self) -> RowMajorMatrix<F> {
        let air = self.access_adapter_air::<N>();
        let width = BaseAir::<F>::width(&air);

        match self.adapter_records.get(&N) {
            None => RowMajorMatrix::new(vec![], width),
            Some(records) => {
                let height = records.len().next_power_of_two();
                let mut values = vec![F::zero(); height * width];

                for (row, record) in values.chunks_mut(width).zip(records) {
                    let row: &mut AccessAdapterCols<F, N> = row.borrow_mut();

                    row.is_valid = F::one();
                    row.values = record.data.clone().try_into().unwrap();
                    row.address = MemoryAddress::new(record.address_space, record.start_index);

                    match record.kind {
                        AccessAdapterRecordKind::Split => {
                            row.left_timestamp = F::from_canonical_u32(record.timestamp);
                            row.right_timestamp = F::from_canonical_u32(record.timestamp);
                            row.is_split = F::one();
                        }
                        AccessAdapterRecordKind::Merge {
                            left_timestamp,
                            right_timestamp,
                        } => {
                            assert_eq!(max(left_timestamp, right_timestamp), record.timestamp);
                            row.left_timestamp = F::from_canonical_u32(left_timestamp);
                            row.right_timestamp = F::from_canonical_u32(right_timestamp);
                            row.is_split = F::zero();
                            row.is_right_larger = F::from_bool(left_timestamp < right_timestamp);
                        }
                    }
                    air.lt_air.generate_trace_row_aux(
                        row.left_timestamp.as_canonical_u32(),
                        row.right_timestamp.as_canonical_u32(),
                        &self.range_checker,
                        &mut IsLessThanAuxColsMut {
                            lower_decomp: &mut row.lt_aux,
                        },
                    );
                }
                RowMajorMatrix::new(values, width)
            }
        }
    }
}
