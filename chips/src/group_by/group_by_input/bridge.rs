use std::iter;

use crate::sub_chip::SubAirBridge;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::columns::GroupByCols;
use super::GroupByAir;

impl<F: PrimeField64> AirBridge<F> for GroupByAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let col_indices_vec: Vec<usize> = (0..self.get_width()).collect();
        let col_indices = GroupByCols::from_slice(&col_indices_vec, self);
        SubAirBridge::sends(self, col_indices)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let col_indices_vec: Vec<usize> = (0..self.get_width()).collect();
        let col_indices = GroupByCols::from_slice(&col_indices_vec, self);
        SubAirBridge::receives(self, col_indices)
    }
}

impl<F: PrimeField64> SubAirBridge<F> for GroupByAir {
    /// Sends desired columns (group_by and to_aggregate) from input page internally with count
    /// `is_alloc`, and sends answer columns with count `is_final`.
    fn sends(&self, col_indices: GroupByCols<usize>) -> Vec<Interaction<F>> {
        assert_eq!(col_indices.aux.grouped.is_some(), !self.sorted);
        let group_by_col_indices = if let Some(grouped) = col_indices.aux.grouped {
            grouped.group_by
        } else {
            self.group_by_cols
                .iter()
                .map(|&i| col_indices.page.data[i])
                .collect()
        };
        // fields = group_by cols, partial_aggregated
        // count = is_final
        let output_sent_fields = group_by_col_indices
            .into_iter()
            .chain(iter::once(col_indices.aux.partial_aggregated))
            .map(VirtualPairCol::single_main)
            .collect();
        let output_count = VirtualPairCol::single_main(col_indices.aux.is_final);
        let mut interactions = vec![Interaction {
            fields: output_sent_fields,
            count: output_count,
            argument_index: self.output_bus,
        }];
        if !self.sorted {
            // Must do internal grouping of page based on group_by columns
            // Sends from columns in input page to internal bus
            let internal_sent_fields = self
                .group_by_cols
                .iter()
                .chain(iter::once(&self.aggregated_col))
                .map(|&i| VirtualPairCol::single_main(col_indices.page.data[i]))
                .collect();
            let internal_count = VirtualPairCol::single_main(col_indices.page.is_alloc);

            interactions.push(Interaction {
                fields: internal_sent_fields,
                count: internal_count,
                argument_index: self.internal_bus,
            });
        }
        interactions
    }

    /// Receives desired columns (`sorted_group_by` and `aggregated`) internally with count
    /// `is_alloc`.
    fn receives(&self, col_indices: GroupByCols<usize>) -> Vec<Interaction<F>> {
        if self.sorted {
            return vec![];
        }

        let grouped = col_indices.aux.grouped.unwrap();
        let internal_received_fields: Vec<VirtualPairCol<F>> = grouped
            .group_by
            .into_iter()
            .chain(iter::once(grouped.to_aggregate))
            .map(VirtualPairCol::single_main)
            .collect();
        let internal_count = VirtualPairCol::single_main(grouped.is_alloc);

        vec![Interaction {
            fields: internal_received_fields,
            count: internal_count,
            argument_index: self.internal_bus,
        }]
    }
}
