use std::borrow::Borrow;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::{AirConfig, SubAir},
};

use super::{columns::PageIndexScanOutputCols, PageIndexScanOutputAir};

impl AirConfig for PageIndexScanOutputAir {
    type Cols<T> = PageIndexScanOutputCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanOutputAir {
    fn width(&self) -> usize {
        PageIndexScanOutputCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        )
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for PageIndexScanOutputAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let page_main = &builder.partitioned_main()[0].clone();
        let aux_main = &builder.partitioned_main()[1].clone();

        // get the current row and the next row
        let (local_page, next_page) = (page_main.row_slice(0), page_main.row_slice(1));
        let local_page: &[AB::Var] = (*local_page).borrow();
        let next_page: &[AB::Var] = (*next_page).borrow();

        let (local_aux, next_aux) = (aux_main.row_slice(0), aux_main.row_slice(1));
        let local_aux: &[AB::Var] = (*local_aux).borrow();
        let next_aux: &[AB::Var] = (*next_aux).borrow();

        let local_vec = local_page
            .iter()
            .chain(local_aux.iter())
            .cloned()
            .collect::<Vec<AB::Var>>();
        let local = local_vec.as_slice();
        let next_vec = next_page
            .iter()
            .chain(next_aux.iter())
            .cloned()
            .collect::<Vec<AB::Var>>();
        let next = next_vec.as_slice();

        let local_cols = PageIndexScanOutputCols::<AB::Var>::from_slice(
            local,
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );
        let next_cols = PageIndexScanOutputCols::<AB::Var>::from_slice(
            next,
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );

        // check that is_alloc is a bool
        builder.when_transition().assert_bool(local_cols.is_alloc);
        // if local_cols.is_alloc is 1, then next_cols.is_alloc can be 0 or 1
        builder
            .when_transition()
            .assert_bool(local_cols.is_alloc * next_cols.is_alloc);
        // if local_cols.is_alloc is 0, then next_cols.is_alloc must be 0
        builder
            .when_transition()
            .assert_zero((AB::Expr::one() - local_cols.is_alloc) * next_cols.is_alloc);

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: local_cols.idx,
                y: next_cols.idx,
                tuple_less_than: local_cols.less_than_next_idx,
            },
            aux: local_cols.is_less_than_tuple_aux,
        };

        // constrain the indicator that we used to check whether the current key < next key is correct
        SubAir::eval(
            self.is_less_than_tuple_air(),
            &mut builder.when_transition(),
            is_less_than_tuple_cols.io,
            is_less_than_tuple_cols.aux,
        );

        // if the next row exists, then the current row index must be less than the next row index
        builder
            .when_transition()
            .assert_zero(next_cols.is_alloc * (AB::Expr::one() - local_cols.less_than_next_idx));
    }
}
