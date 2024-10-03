use afs_primitives::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIoCols},
    sub_chip::{AirConfig, SubAir},
    utils::{implies, or},
};
use afs_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{IndexedOutputPageAuxCols, IndexedOutputPageCols},
    IndexedOutputPageAir,
};
use crate::common::page_cols::PageCols;

impl<F: Field> BaseAirWithPublicValues<F> for IndexedOutputPageAir {}
impl<F: Field> PartitionedBaseAir<F> for IndexedOutputPageAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![self.page_width()]
    }
    fn common_main_width(&self) -> usize {
        self.aux_width()
    }
}
impl<F: Field> BaseAir<F> for IndexedOutputPageAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for IndexedOutputPageAir {
    type Cols<T> = IndexedOutputPageCols<T>;
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for IndexedOutputPageAir {
    // This function assumes that there are (at least) two partitions for the trace.
    // The first partition is the page itself, and the first self.aux_width() columns of the
    // second partition correspond to the auxiliary columns necessary for sorting.
    // Under this assumption, this function can be called directly from a superair (without needing
    // to construct the columns manually before calling SubAir)
    #[inline]
    fn eval(&self, builder: &mut AB) {
        assert_eq!(builder.cached_mains().len(), 1);

        let page_trace: &<AB as AirBuilder>::M = &builder.cached_mains()[0];
        let aux_trace: &<AB as AirBuilder>::M = builder.common_main();

        let [page_local, page_next] = [0, 1].map(|i| {
            PageCols::<AB::Var>::from_slice(&page_trace.row_slice(i), self.idx_len, self.data_len)
        });

        // The auxiliary columns to compare local index and next index are stored in the next row
        let aux_next = IndexedOutputPageAuxCols::from_slice(&aux_trace.row_slice(1), self);

        SubAir::eval(self, builder, [page_local, page_next], aux_next);
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for IndexedOutputPageAir {
    type IoView = [PageCols<AB::Var>; 2];
    type AuxView = IndexedOutputPageAuxCols<AB::Var>;

    /// Ensuring the page is in the proper format: allocated rows come first
    /// and are sorted by idx, which are distinct. Moreover `idx` and `data`
    /// for unallocated rows should be all zeros.
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux_next: Self::AuxView) {
        let (page_local, page_next) = (&io[0], &io[1]);

        // Ensuring that is_alloc is always bool
        builder.assert_bool(page_local.is_alloc);

        // Ensuring that all unallocated rows are at the bottom
        builder
            .when_transition()
            .assert_one(implies(page_next.is_alloc, page_local.is_alloc));

        // Ensuring that rows are sorted by idx
        let lt_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIoCols {
                x: page_local.idx.clone(),
                y: page_next.idx.clone(),
                tuple_less_than: aux_next.lt_out,
            },
            aux: aux_next.lt_cols.clone(),
        };

        // Note: we do not use AssertSortedAir because it constrains keys are strictly sorted on every row,
        // whereas we only want it on allocated rows.
        self.lt_air
            .eval_when_transition(builder, lt_cols.io, lt_cols.aux);

        // Ensuring the keys are strictly sorted for allocated rows
        builder.when_transition().assert_one(or::<AB::Expr>(
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
