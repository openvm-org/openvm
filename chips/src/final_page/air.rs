use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{FinalPageAuxCols, FinalPageCols},
    FinalPageAir,
};
use crate::{
    common::page_cols::PageCols,
    is_less_than_tuple::{
        columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
        IsLessThanTupleAir,
    },
    sub_chip::{AirConfig, SubAir},
};

impl<F: Field> BaseAir<F> for FinalPageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for FinalPageAir {
    type Cols<T> = FinalPageCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for FinalPageAir
where
    AB::M: Clone,
{
    // This function assumes that there are (at least) two partitions for the trace.
    // The first partition is the page itself, and the first self.aux_width() columns of the
    // second partition correspond to the auxiliary columns necessary for sorting.
    // Under this assumption, this function can be called directly from a superair (without needing
    // to construct the columns manually before calling SubAir)
    fn eval(&self, builder: &mut AB) {
        assert!(builder.partitioned_main().len() >= 2);

        let page_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let aux_trace: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        let (page_local, page_next) = (page_trace.row_slice(0), page_trace.row_slice(1));

        let page_local_cols =
            PageCols::<AB::Var>::from_slice(&page_local, self.idx_len, self.data_len);
        let page_next_cols =
            PageCols::<AB::Var>::from_slice(&page_next, self.idx_len, self.data_len);

        // The auxiliary columns to compare local index and next index are stored in the next row
        let aux_next = aux_trace.row_slice(1);

        let aux_next_cols = FinalPageAuxCols::from_slice(
            &aux_next[0..self.aux_width()],
            self.idx_limb_bits,
            self.idx_decomp,
            self.idx_len,
        );

        SubAir::eval(
            self,
            builder,
            [page_local_cols, page_next_cols],
            aux_next_cols,
        );
    }
}

impl<AB: AirBuilder> SubAir<AB> for FinalPageAir {
    type IoView = [PageCols<AB::Var>; 2];
    type AuxView = FinalPageAuxCols<AB::Var>;

    /// Ensuring the page is in the proper format: allocated rows come first
    /// and are sorted by idx, which are distinct. Moreover `idx` and `data`
    /// for unallocated rows should be all zeros.
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux_next: Self::AuxView) {
        let (page_local, page_next) = (&io[0], &io[1]);

        // Helpers
        let or = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b;
        let implies = |a: AB::Expr, b: AB::Expr| or(AB::Expr::one() - a, b);

        // Ensuring that is_alloc is always bool
        builder.assert_bool(page_local.is_alloc);

        // Ensuring that all unallocated rows are at the bottom
        builder.when_transition().assert_one(implies(
            page_next.is_alloc.into(),
            page_local.is_alloc.into(),
        ));

        // Ensuring that rows are sorted by idx
        let lt_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: page_local.idx.clone(),
                y: page_next.idx.clone(),
                tuple_less_than: aux_next.lt_out,
            },
            aux: aux_next.lt_cols.clone(),
        };

        let lt_air = IsLessThanTupleAir::new(
            self.range_bus_index,
            vec![self.idx_limb_bits; self.idx_len],
            self.idx_decomp,
        );

        SubAir::eval(
            &lt_air,
            &mut builder.when_transition(),
            lt_cols.io,
            lt_cols.aux,
        );

        // Ensuring the keys are strictly sorted (for allocated rows)
        builder.when_transition().assert_one(or(
            AB::Expr::one() - page_next.is_alloc,
            aux_next.lt_out.into(),
        ));

        // Making sure `idx` and `data` for unallocated rows are all zeros
        for i in 0..page_local.idx.len() {
            builder.assert_zero((AB::Expr::one() - page_local.is_alloc) * page_local.idx[i]);
        }

        for i in 0..page_local.data.len() {
            builder.assert_zero((AB::Expr::one() - page_local.is_alloc) * page_local.data[i]);
        }
    }
}
