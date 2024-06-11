use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::{AirConfig, SubAir},
};

use super::{columns::PageIndexScanInputCols, PageIndexScanInputAir};

impl AirConfig for PageIndexScanInputAir {
    type Cols<T> = PageIndexScanInputCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanInputAir {
    fn width(&self) -> usize {
        PageIndexScanInputCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air.limb_bits().clone(),
            self.is_less_than_tuple_air.decomp(),
        )
    }
}

impl<AB: PartitionedAirBuilder + AirBuilderWithPublicValues> Air<AB> for PageIndexScanInputAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let page_main = &builder.partitioned_main()[0].clone();
        let aux_main = &builder.partitioned_main()[1].clone();

        // get the public value x
        let pis = builder.public_values();
        let x = pis[..self.idx_len].to_vec();

        let local_page = page_main.row_slice(0);
        let local_aux = aux_main.row_slice(0);
        let local_vec = local_page
            .iter()
            .chain(local_aux.iter())
            .cloned()
            .collect::<Vec<AB::Var>>();
        let local = local_vec.as_slice();

        let local_cols = PageIndexScanInputCols::<AB::Var>::from_slice(
            local,
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air.limb_bits().clone(),
            self.is_less_than_tuple_air.decomp(),
        );

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: local_cols.idx,
                y: local_cols.x.clone(),
                tuple_less_than: local_cols.satisfies_pred,
            },
            aux: local_cols.is_less_than_tuple_aux,
        };

        // constrain that the public value x is the same as the column x
        for (&local_x, &pub_x) in local_cols.x.iter().zip(x.iter()) {
            builder.assert_eq(local_x, pub_x);
        }

        // constrain that we send the row iff the row is allocated and satisfies the predicate
        builder.assert_eq(
            local_cols.is_alloc * local_cols.satisfies_pred,
            local_cols.send_row,
        );
        builder.assert_bool(local_cols.send_row);

        // constrain the indicator that we used to check wheter key < x is correct
        SubAir::eval(
            &self.is_less_than_tuple_air,
            &mut builder.when_transition(),
            is_less_than_tuple_cols.io,
            is_less_than_tuple_cols.aux,
        );
    }
}
