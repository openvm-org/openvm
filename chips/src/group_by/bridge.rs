use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::GroupByAir;

impl<F: PrimeField64> AirBridge<F> for GroupByAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let internal_sent_fields = (0..self.page_width)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();
        let output_sent_fields = (self.page_width..self.page_width + self.group_by_cols.len() + 1)
            .map(|i| VirtualPairCol::single_main(i))
            .collect();

        vec![
            Interaction {
                fields: internal_sent_fields,
                count: VirtualPairCol::one(),
                argument_index: self.internal_bus,
            },
            Interaction {
                fields: output_sent_fields,
                count: VirtualPairCol::single_main(self.page_width + self.group_by_cols.len() + 1),
                argument_index: self.output_bus,
            },
        ]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let internal_received_fields = (self.page_width
            ..self.page_width + self.group_by_cols.len())
            .map(|i| VirtualPairCol::single_main(i))
            .collect();
        vec![Interaction {
            fields: internal_received_fields,
            count: VirtualPairCol::one(),
            argument_index: self.internal_bus,
        }]
    }
}
