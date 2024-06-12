use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::GroupByCols, GroupByAir};

impl<F: PrimeField64> AirBridge<F> for GroupByAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let index_map = GroupByCols::<F>::index_map(self);

        let internal_sent_fields = (index_map.page_start..index_map.page_end)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();
        let internal_count = VirtualPairCol::single_main(index_map.allocated_idx);

        let output_sent_fields = (index_map.sorted_group_by_start..index_map.sorted_group_by_end)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();
        let output_count = VirtualPairCol::single_main(index_map.is_final);

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

    fn receives(&self) -> Vec<Interaction<F>> {
        let index_map = GroupByCols::<F>::index_map(self);

        let internal_received_fields = (index_map.sorted_group_by_start
            ..index_map.sorted_group_by_end)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();
        let internal_count = VirtualPairCol::single_main(index_map.allocated_idx);

        vec![Interaction {
            fields: internal_received_fields,
            count: internal_count,
            argument_index: self.internal_bus,
        }]
    }
}
