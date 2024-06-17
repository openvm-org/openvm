use crate::sub_chip::SubAirBridge;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::GroupByCols;
use super::GroupByAir;

impl<F: PrimeField64> SubAirBridge<F> for GroupByAir {
    fn sends(&self, col_indices: GroupByCols<usize>) -> Vec<Interaction<F>> {
        let internal_sent_fields = self
            .group_by_cols
            .iter()
            .chain(std::iter::once(&self.aggregated_col))
            .map(|i| VirtualPairCol::single_main(col_indices.io.page[*i]))
            .collect();
        let internal_count = VirtualPairCol::single_main(col_indices.io.is_allocated);

        let output_sent_fields: Vec<VirtualPairCol<F>> = col_indices
            .aux
            .sorted_group_by
            .iter()
            .chain(std::iter::once(&col_indices.aux.aggregated))
            .map(|&i| VirtualPairCol::single_main(i))
            .collect();
        let output_count = VirtualPairCol::single_main(col_indices.aux.is_final);

        vec![
            Interaction {
                fields: internal_sent_fields,
                count: internal_count,
                argument_index: self.internal_bus,
            },
            Interaction {
                fields: output_sent_fields,
                count: output_count,
                argument_index: self.output_bus,
            },
        ]
    }

    fn receives(&self, col_indices: GroupByCols<usize>) -> Vec<Interaction<F>> {
        let internal_received_fields: Vec<VirtualPairCol<F>> = col_indices
            .aux
            .sorted_group_by
            .iter()
            .chain(std::iter::once(&col_indices.aux.aggregated))
            .map(|&i| VirtualPairCol::single_main(i))
            .collect();
        let internal_count = VirtualPairCol::single_main(col_indices.io.is_allocated);

        vec![Interaction {
            fields: internal_received_fields,
            count: internal_count,
            argument_index: self.internal_bus,
        }]
    }
}

impl<F: PrimeField64> AirBridge<F> for GroupByAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_indices_vec: Vec<usize> = (0..self.get_width()).collect();
        let col_indices = GroupByCols::from_slice(&col_indices_vec, self);
        SubAirBridge::sends(self, col_indices)
    }
}
