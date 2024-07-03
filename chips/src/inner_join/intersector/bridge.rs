use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::{columns::IntersectorCols, IntersectorAir};
use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAirBridge,
    utils::to_vcols,
};

impl<F: PrimeField> SubAirBridge<F> for IntersectorAir {
    /// Sends interactions required by the IsLessThanTuple SubAir
    /// Sends idx with multiplicity out_mult on the intersector_t2_bus (received by t2_chip)
    fn sends(&self, col_indices: IntersectorCols<usize>) -> Vec<Interaction<F>> {
        let mut interactions = SubAirBridge::<F>::sends(
            &self.lt_chip,
            IsLessThanTupleCols {
                io: IsLessThanTupleIOCols {
                    x: vec![usize::MAX; 1 + self.idx_len],
                    y: vec![usize::MAX; 1 + self.idx_len],
                    tuple_less_than: usize::MAX,
                },
                aux: col_indices.aux.lt_aux,
            },
        );

        interactions.push(Interaction {
            fields: to_vcols(&col_indices.io.idx),
            count: VirtualPairCol::single_main(col_indices.io.out_mult),
            argument_index: self.buses.intersector_t2_bus_index,
        });

        interactions
    }

    /// Receives idx with multiplicity t1_mult on the t1_intersector_bus (sent by t1_chip)
    /// Receives idx with multiplicity t2_mult on the t2_intersector_bus (sent by t2_chip)
    fn receives(&self, col_indices: IntersectorCols<usize>) -> Vec<Interaction<F>> {
        vec![
            Interaction {
                fields: to_vcols(&col_indices.io.idx),
                count: VirtualPairCol::single_main(col_indices.io.t1_mult),
                argument_index: self.buses.t1_intersector_bus_index,
            },
            Interaction {
                fields: to_vcols(&col_indices.io.idx),
                count: VirtualPairCol::single_main(col_indices.io.t2_mult),
                argument_index: self.buses.t2_intersector_bus_index,
            },
        ]
    }
}

impl<F: PrimeField> AirBridge<F> for IntersectorAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let intersector_cols = IntersectorCols::<usize>::from_slice(&all_cols, self);

        SubAirBridge::sends(self, intersector_cols)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let intersector_cols = IntersectorCols::<usize>::from_slice(&all_cols, self);

        SubAirBridge::receives(self, intersector_cols)
    }
}
