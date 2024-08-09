use std::collections::HashMap;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::AccessCell;
use crate::memory::interface::{columns::MemoryInterfaceCols, MemoryInterfaceChip};

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryInterfaceChip<NUM_WORDS, WORD_SIZE, F>
{
    pub fn generate_trace(
        &self,
        final_memory: &HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for &(address_space, label) in self.touched_leaves.iter() {
            let mut initial_values = [[F::zero(); WORD_SIZE]; NUM_WORDS];
            let mut initial_clks = [F::zero(); NUM_WORDS];
            let mut final_values = [[F::zero(); WORD_SIZE]; NUM_WORDS];
            let mut final_clks = [F::zero(); NUM_WORDS];

            for word_idx in 0..NUM_WORDS {
                let full_address = &(
                    address_space,
                    F::from_canonical_usize((NUM_WORDS * WORD_SIZE * label) + word_idx * WORD_SIZE),
                );

                let initial_cell = self.initial_memory.get(full_address).unwrap();
                initial_values[word_idx] = initial_cell.value;
                initial_clks[word_idx] = initial_cell.timestamp;

                // TODO[osama]: consider making the HAshMap final_memory have [F; WORD_SIZE] key
                let final_cell = final_memory.get(full_address).unwrap();

                final_values[word_idx] = final_cell.value;
                final_clks[word_idx] = final_cell.timestamp;
            }
            let initial_cols = MemoryInterfaceCols {
                expand_direction: F::one(),
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: initial_values,
                clks: initial_clks,
            };
            let final_cols = MemoryInterfaceCols {
                expand_direction: F::neg_one(),
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: final_values,
                clks: final_clks,
            };
            rows.extend(initial_cols.flatten());
            rows.extend(final_cols.flatten());
        }
        while rows.len()
            != trace_degree * MemoryInterfaceCols::<NUM_WORDS, WORD_SIZE, F>::get_width()
        {
            rows.extend(Self::unused_row().flatten());
        }
        RowMajorMatrix::new(
            rows,
            MemoryInterfaceCols::<NUM_WORDS, WORD_SIZE, F>::get_width(),
        )
    }

    fn unused_row() -> MemoryInterfaceCols<NUM_WORDS, WORD_SIZE, F> {
        MemoryInterfaceCols::from_slice(&vec![
            F::zero();
            MemoryInterfaceCols::<NUM_WORDS, WORD_SIZE, F>::get_width()
        ])
    }
}
