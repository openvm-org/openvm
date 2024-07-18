use std::collections::HashMap;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::memory::interface::columns::MemoryInterfaceCols;
use crate::memory::interface::MemoryInterfaceChip;

impl<const CHUNK: usize, F: PrimeField32> MemoryInterfaceChip<CHUNK, F> {
    pub fn generate_trace(
        &self,
        final_memory: &HashMap<(F, F), F>,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for &(address_space, label) in self.touched_leaves.iter() {
            let mut initial_values = [F::zero(); CHUNK];
            let mut initial_values_matter = [F::zero(); CHUNK];
            let mut final_values = [F::zero(); CHUNK];
            let mut final_values_are_from_offline_checker = [F::zero(); CHUNK];
            for i in 0..CHUNK {
                let full_address = &(address_space, F::from_canonical_usize((CHUNK * label) + i));
                final_values[i] = *final_memory.get(full_address).unwrap_or(&F::zero());
                match self.touched_addresses.get(full_address) {
                    Some(cell) => {
                        initial_values[i] = cell.initial_value;
                        initial_values_matter[i] = F::from_bool(cell.read_initially);
                        final_values_are_from_offline_checker[i] = F::from_bool(true);
                    }
                    None => {
                        initial_values[i] = final_values[i];
                        initial_values_matter[i] = F::from_bool(true);
                        final_values_are_from_offline_checker[i] = F::from_bool(false);
                    }
                }
            }
            let initial_cols = MemoryInterfaceCols {
                direction: F::one(),
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: initial_values,
                auxes: initial_values_matter,
                temp_multiplicity: initial_values_matter,
                temp_is_final: [F::zero(); CHUNK],
            };
            let final_cols = MemoryInterfaceCols {
                direction: F::neg_one(),
                address_space,
                leaf_label: F::from_canonical_usize(label),
                values: final_values,
                auxes: final_values_are_from_offline_checker,
                temp_multiplicity: [F::neg_one(); CHUNK],
                temp_is_final: final_values_are_from_offline_checker,
            };
            rows.extend(initial_cols.flatten());
            rows.extend(final_cols.flatten());
        }
        while rows.len() != trace_degree * MemoryInterfaceCols::<CHUNK, F>::get_width() {
            rows.extend(Self::unused_row().flatten());
        }
        let trace = RowMajorMatrix::new(rows, MemoryInterfaceCols::<CHUNK, F>::get_width());
        trace
    }

    fn unused_row() -> MemoryInterfaceCols<CHUNK, F> {
        let mut cols = MemoryInterfaceCols::from_slice(&vec![
            F::zero();
            MemoryInterfaceCols::<CHUNK, F>::get_width()
        ]);

        for i in 0..CHUNK {
            cols.temp_multiplicity[i] = cols.direction
                * (F::one()
                    - ((cols.direction + F::one())
                        * F::two().inverse()
                        * (F::one() - cols.auxes[i])));
            cols.temp_is_final[i] =
                (F::one() - cols.direction) * F::two().inverse() * cols.auxes[i];
        }

        cols
    }
}
