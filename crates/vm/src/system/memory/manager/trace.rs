use std::{borrow::BorrowMut, cmp::max};

use ax_circuit_primitives::{utils::next_power_of_two_or_zero, TraceSubRowGenerator};
use ax_stark_backend::{p3_air::BaseAir, p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use crate::system::memory::{
    adapter::AccessAdapterCols, manager::memory::AccessAdapterRecordKind, MemoryAddress,
    MemoryController,
};

impl<F: PrimeField32> MemoryController<F> {
    pub fn generate_access_adapter_trace<const N: usize>(&self) -> RowMajorMatrix<F> {
        let air = self.access_adapter_air::<N>();
        let width = BaseAir::<F>::width(&air);

        match self.adapter_records.get(&N) {
            None => {
                if let Some(oh) = self.get_access_adapter_overridden_height(N) {
                    RowMajorMatrix::new(F::zero_vec(oh * width), width)
                } else {
                    RowMajorMatrix::new(vec![], width)
                }
            }
            Some(records) => {
                let height = if let Some(oh) = self.get_access_adapter_overridden_height(N) {
                    assert!(
                        oh >= records.len(),
                        "Overridden height is less than the required height"
                    );
                    oh
                } else {
                    records.len()
                };
                let height = next_power_of_two_or_zero(height);
                let mut values = F::zero_vec(height * width);

                for (row, record) in values.chunks_mut(width).zip(records) {
                    let row: &mut AccessAdapterCols<F, N> = row.borrow_mut();

                    row.is_valid = F::ONE;
                    row.values = record.data.clone().try_into().unwrap();
                    row.address = MemoryAddress::new(record.address_space, record.start_index);

                    let (left_timestamp, right_timestamp) = match record.kind {
                        AccessAdapterRecordKind::Split => (record.timestamp, record.timestamp),
                        AccessAdapterRecordKind::Merge {
                            left_timestamp,
                            right_timestamp,
                        } => (left_timestamp, right_timestamp),
                    };
                    debug_assert_eq!(max(left_timestamp, right_timestamp), record.timestamp);

                    row.left_timestamp = F::from_canonical_u32(left_timestamp);
                    row.right_timestamp = F::from_canonical_u32(right_timestamp);
                    row.is_split = F::from_bool(record.kind == AccessAdapterRecordKind::Split);

                    air.lt_air.generate_subrow(
                        (&self.range_checker, left_timestamp, right_timestamp),
                        (&mut row.lt_aux, &mut row.is_right_larger),
                    );
                }
                RowMajorMatrix::new(values, width)
            }
        }
    }
    fn get_access_adapter_overridden_height(&self, n: usize) -> Option<usize> {
        self.overridden_heights
            .as_ref()
            .and_then(|oh| oh.access_adapters_ref().get(&n).copied())
    }
}
