use std::collections::HashMap;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::MemoryExpandInterfaceCols, AccessCell, MemoryExpandInterfaceChip};

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryExpandInterfaceChip<NUM_WORDS, WORD_SIZE, F>
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
                let address = &(
                    address_space,
                    F::from_canonical_usize((NUM_WORDS * WORD_SIZE * label) + word_idx * WORD_SIZE),
                );

                let initial_cell = *self.initial_memory.get(address).unwrap_or(&AccessCell {
                    data: [F::zero(); WORD_SIZE],
                    clk: F::zero(),
                });
                initial_values[word_idx] = initial_cell.data;
                initial_clks[word_idx] = initial_cell.clk;

                let final_cell = *final_memory.get(address).unwrap();
                final_values[word_idx] = final_cell.data;
                final_clks[word_idx] = final_cell.clk;
            }
            let initial_cols = MemoryExpandInterfaceCols {
                expand_direction: F::one(),
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: initial_values,
                clks: initial_clks,
            };
            let final_cols = MemoryExpandInterfaceCols {
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
            < trace_degree * MemoryExpandInterfaceCols::<NUM_WORDS, WORD_SIZE, F>::width()
        {
            rows.extend(Self::unused_row().flatten());
        }
        RowMajorMatrix::new(
            rows,
            MemoryExpandInterfaceCols::<NUM_WORDS, WORD_SIZE, F>::width(),
        )
    }

    fn unused_row() -> MemoryExpandInterfaceCols<NUM_WORDS, WORD_SIZE, F> {
        MemoryExpandInterfaceCols::from_slice(&vec![
            F::zero();
            MemoryExpandInterfaceCols::<
                NUM_WORDS,
                WORD_SIZE,
                F,
            >::width()
        ])
    }
}
