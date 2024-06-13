use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::{columns::IntersectorCols, IntersectorAir};
use crate::sub_chip::SubAirBridge;

impl<F: PrimeField> SubAirBridge<F> for IntersectorAir {
    fn sends(&self, col_indices: IntersectorCols<usize>) -> Vec<Interaction<F>> {
        vec![Interaction {
            fields: col_indices
                .idx
                .iter()
                .copied()
                .map(VirtualPairCol::single_main)
                .collect(),
            count: VirtualPairCol::single_main(col_indices.out_mult),
            argument_index: self.intersector_t2_bus_index,
        }]
    }

    fn receives(&self, col_indices: IntersectorCols<usize>) -> Vec<Interaction<F>> {
        vec![
            Interaction {
                fields: col_indices
                    .idx
                    .iter()
                    .copied()
                    .map(VirtualPairCol::single_main)
                    .collect(),
                count: VirtualPairCol::single_main(col_indices.t1_mult),
                argument_index: self.t1_intersector_bus_index,
            },
            Interaction {
                fields: col_indices
                    .idx
                    .iter()
                    .copied()
                    .map(VirtualPairCol::single_main)
                    .collect(),
                count: VirtualPairCol::single_main(col_indices.t2_mult),
                argument_index: self.t2_intersector_bus_index,
            },
        ]
    }
}

impl<F: PrimeField> AirBridge<F> for IntersectorAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let intersector_cols = IntersectorCols::<usize>::from_slice(&all_cols, self.idx_len);

        SubAirBridge::sends(self, intersector_cols)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let intersector_cols = IntersectorCols::<usize>::from_slice(&all_cols, self.idx_len);

        SubAirBridge::receives(self, intersector_cols)
    }
}
