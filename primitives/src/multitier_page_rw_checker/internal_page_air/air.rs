use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{InternalPageCols, InternalPageMetadataCols, PtrPageCols},
    InternalPageAir,
};
use crate::{
    is_less_than_tuple::columns::IsLessThanTupleIoCols,
    is_zero::columns::IsZeroIoCols,
    sub_chip::{AirConfig, SubAir},
    utils::implies,
};

impl<F: Field, const COMMITMENT_LEN: usize> BaseAir<F> for InternalPageAir<COMMITMENT_LEN> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const COMMITMENT_LEN: usize> AirConfig for InternalPageAir<COMMITMENT_LEN> {
    type Cols<T> = InternalPageCols<T>;
}

impl<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder + InteractionBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for InternalPageAir<COMMITMENT_LEN>
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        self.eval_without_interactions(builder);
        let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[1];
        let local = main.row_slice(0);
        let pi = builder.public_values().to_vec();
        let data: &<AB as AirBuilder>::M = &builder.partitioned_main()[0];
        let metadata = InternalPageMetadataCols::from_slice(
            &local,
            self.idx_len,
            self.is_init,
            self.is_less_than_tuple_param.clone(),
        );
        let cached_data = PtrPageCols::from_slice(&data.row_slice(0), self.idx_len, COMMITMENT_LEN);
        let cols = InternalPageCols {
            metadata,
            cache_cols: cached_data,
        };
        drop(local);
        self.eval_interactions(builder, &cols, &pi);
    }
}

impl<const COMMITMENT_LEN: usize> InternalPageAir<COMMITMENT_LEN> {
    pub(crate) fn eval_without_interactions<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder + InteractionBuilder,
    >(
        &self,
        builder: &mut AB,
    ) where
        AB::M: Clone,
    {
        // only constrain that own_commitment is accurate
        // partition is physical page data vs metadata
        let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[1];
        let local = main.row_slice(0);
        let data: &<AB as AirBuilder>::M = &builder.partitioned_main()[0];
        let metadata = InternalPageMetadataCols::from_slice(
            &local,
            self.idx_len,
            self.is_init,
            self.is_less_than_tuple_param.clone(),
        );
        let [cached_data, next_data] = [0, 1]
            .map(|i| PtrPageCols::from_slice(&data.row_slice(i), self.idx_len, COMMITMENT_LEN));
        drop(local);
        builder.assert_eq(cached_data.internal_marker, AB::Expr::from_canonical_u64(2));
        builder.assert_eq(metadata.mult_alloc, cached_data.is_alloc * metadata.mult);
        builder.assert_eq(
            metadata.mult_alloc_minus_one,
            metadata.mult_alloc - AB::Expr::one(),
        );
        builder.assert_eq(
            metadata.mult_minus_one_alloc,
            cached_data.is_alloc * metadata.mult_alloc_minus_one,
        );

        if !self.is_init {
            // assert that next_idx is the same as the thing in the next row
            // will do the allocated rows are at the top check later probably
            // Ensuring that all unallocated rows are at the bottom
            builder.when_transition().assert_one({
                let x: AB::Expr =
                    implies::<AB>(next_data.is_alloc.into(), cached_data.is_alloc.into());
                x
            });
            let prove_sort_cols = metadata.prove_sort_cols.unwrap();
            builder.when_transition().assert_zero(
                next_data.is_alloc
                    * (prove_sort_cols.end_less_than_next
                        - prove_sort_cols.end_less_than_next * prove_sort_cols.end_less_than_start
                        - AB::Expr::one()),
            );
            let range_inclusion_cols = metadata.range_inclusion_cols.unwrap();
            let less_than_start = range_inclusion_cols.less_than_start;
            let greater_than_end = range_inclusion_cols.greater_than_end;
            builder.assert_zero(cached_data.is_alloc * (less_than_start + greater_than_end));
            builder.assert_bool(cached_data.is_alloc);
            let subair_aux_cols = metadata.subair_aux_cols.unwrap();
            let subairs = self.is_less_than_tuple_air.clone().unwrap();
            {
                let io = IsLessThanTupleIoCols {
                    x: cached_data.child_start.clone(),
                    y: range_inclusion_cols.start.clone(),
                    tuple_less_than: range_inclusion_cols.less_than_start,
                };
                let aux = subair_aux_cols.idx1_start.clone();
                subairs
                    .idx1_start
                    .eval_without_interactions(builder, io, aux);
            }
            {
                let io = IsLessThanTupleIoCols {
                    x: range_inclusion_cols.end.clone(),
                    y: cached_data.child_end.clone(),
                    tuple_less_than: range_inclusion_cols.greater_than_end,
                };
                let aux = subair_aux_cols.end_idx2.clone();
                subairs.end_idx2.eval_without_interactions(builder, io, aux);
            }
            {
                let io = IsLessThanTupleIoCols {
                    x: cached_data.child_end.clone(),
                    y: next_data.child_start.clone(),
                    tuple_less_than: prove_sort_cols.end_less_than_next,
                };
                let aux = subair_aux_cols.idx2_next.clone();
                subairs
                    .idx2_next
                    .eval_without_interactions(builder, io, aux);
            }
            {
                let io = IsLessThanTupleIoCols {
                    x: cached_data.child_end.clone(),
                    y: cached_data.child_start.clone(),
                    tuple_less_than: prove_sort_cols.end_less_than_start,
                };
                let aux = subair_aux_cols.idx2_idx1.clone();
                subairs
                    .idx2_idx1
                    .eval_without_interactions(builder, io, aux);
            }
            {
                let io = IsZeroIoCols {
                    x: metadata.mult_alloc_minus_one,
                    is_zero: metadata.mult_alloc_is_1,
                };
                let aux = subair_aux_cols.mult_inv;
                SubAir::eval(&subairs.mult_is_1, builder, io, aux);
            }
        }
    }
}
