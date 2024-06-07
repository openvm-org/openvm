use std::iter;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{FinalPageAuxCols, FinalPageCols},
    FinalPageChip,
};
use crate::{
    is_less_than_tuple::{
        columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
        IsLessThanTupleAir,
    },
    page_rw_checker::page_chip::columns::PageCols,
    sub_chip::{AirConfig, SubAir},
};

impl<F: Field> BaseAir<F> for FinalPageChip {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for FinalPageChip {
    type Cols<T> = FinalPageCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for FinalPageChip
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // TODO: consider deleting this
        let conv_to_expr =
            |slc: &[AB::Var]| slc.iter().map(|x| (*x).into()).collect::<Vec<AB::Expr>>();

        let page_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        let (page_local, page_next) = (page_trace.row_slice(0), page_trace.row_slice(1));

        let page_local_cols =
            PageCols::<AB::Var>::from_slice(&page_local, self.idx_len, self.data_len);
        let page_next_cols =
            PageCols::<AB::Var>::from_slice(&page_next, self.idx_len, self.data_len);

        // TODO: fix limb_bits
        // TODO: consider renaming aux to sort or something

        // The auxiliary columns to compare local index and next index are stored in the next row
        let aux_next = aux_trace.row_slice(1);
        let aux_next = conv_to_expr(&aux_next);

        let aux_next_cols = FinalPageAuxCols::<AB::Expr>::from_slice(
            &aux_next,
            self.idx_limb_bits,
            self.idx_decomp,
            1 + self.idx_len,
        );

        // Ensuring that is_alloc is always bool
        builder.assert_bool(page_local_cols.is_alloc);

        // TODO: maybe add a general helper to convert to AB::Expr?

        // Ensuring that rows are sorted by (1-is_alloc, idx)
        let lt_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: iter::once(AB::Expr::one() - page_local_cols.is_alloc)
                    .chain(page_local_cols.idx.iter().map(|x| (*x).into()))
                    .collect(),
                y: iter::once(AB::Expr::one() - page_next_cols.is_alloc)
                    .chain(page_next_cols.idx.iter().map(|x| (*x).into()))
                    .collect(),
                tuple_less_than: aux_next_cols.lt_out.clone(),
            },
            aux: aux_next_cols.lt_cols,
        };

        // TODO: why is there a bus_index here?
        let lt_air = IsLessThanTupleAir::new(
            self.sorted_bus_index,
            1 << self.idx_limb_bits,
            vec![self.idx_limb_bits; self.idx_len + 1],
            self.idx_decomp,
        );

        SubAir::eval(
            &lt_air,
            &mut builder.when_transition(),
            lt_cols.io,
            lt_cols.aux,
        );

        // Helper
        let or = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b;

        // Ensuring the keys are strictly sorted (for allocated rows)
        builder.when_transition().assert_one(or(
            AB::Expr::one() - page_local_cols.is_alloc,
            or(
                AB::Expr::one() - page_next_cols.is_alloc,
                aux_next_cols.lt_out,
            ),
        ));
    }
}
