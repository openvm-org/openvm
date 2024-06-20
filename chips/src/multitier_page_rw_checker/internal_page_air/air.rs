use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{InternalPageCols, InternalPageMetadataCols, PtrPageCols},
    InternalPageAir,
};
use crate::{
    is_less_than_tuple::columns::IsLessThanTupleIOCols,
    is_zero::columns::IsZeroIOCols,
    sub_chip::{AirConfig, SubAir},
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
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for InternalPageAir<COMMITMENT_LEN>
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // only constrain that own_commitment is accurate
        // partition is physical page data vs metadata
        let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();
        let local = main.row_slice(0);
        let pi = builder.public_values().to_vec();
        let data: &<AB as AirBuilder>::M = &builder.partitioned_main()[0].clone();
        let metadata = InternalPageMetadataCols::from_slice(
            &local,
            self.idx_len,
            COMMITMENT_LEN,
            self.is_init,
            self.is_less_than_tuple_param.clone(),
        );
        let next_data = PtrPageCols::from_slice(&data.row_slice(1), self.idx_len, COMMITMENT_LEN);
        let cached_data = PtrPageCols::from_slice(&data.row_slice(0), self.idx_len, COMMITMENT_LEN);
        for (i, p) in pi.iter().enumerate().take(COMMITMENT_LEN) {
            builder.assert_eq(*p, metadata.own_commitment[i]);
        }
        // assert that own id is correct
        builder.assert_eq(metadata.id, AB::Expr::from_canonical_u64(self.id as u64));
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

            let prove_sort_cols = metadata.prove_sort_cols.unwrap();
            for i in 0..self.idx_len {
                builder.when_transition().assert_zero(
                    next_data.is_alloc * (next_data.start[i] - prove_sort_cols.next_idx[i]),
                );
            }
            builder.when_transition().assert_zero(
                next_data.is_alloc
                    * (prove_sort_cols.end_less_than_next
                        - prove_sort_cols.end_less_than_next * prove_sort_cols.end_less_than_start
                        - AB::Expr::one()),
            );
            let range_inclusion_cols = metadata.range_inclusion_cols.unwrap();
            let less_than_start = range_inclusion_cols.less_than_start;
            let greater_than_end = range_inclusion_cols.greater_than_end;
            builder.assert_zero(cached_data.is_alloc * (less_than_start.0 + greater_than_end.0));
            builder.assert_zero(cached_data.is_alloc * (less_than_start.1 + greater_than_end.1));
            builder.assert_bool(cached_data.is_alloc);
            let subair_aux_cols = metadata.subchip_aux_cols.unwrap();
            let subairs = self.is_less_than_tuple_air.clone().unwrap();
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.start.clone(),
                    y: range_inclusion_cols.start.clone(),
                    tuple_less_than: range_inclusion_cols.less_than_start.0,
                };
                let aux = subair_aux_cols.idx1_start.clone();
                SubAir::eval(&subairs.idx1_start, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: range_inclusion_cols.end.clone(),
                    y: cached_data.start.clone(),
                    tuple_less_than: range_inclusion_cols.greater_than_end.0,
                };
                let aux = subair_aux_cols.end_idx1.clone();
                SubAir::eval(&subairs.end_idx1, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.end.clone(),
                    y: range_inclusion_cols.start.clone(),
                    tuple_less_than: range_inclusion_cols.less_than_start.1,
                };
                let aux = subair_aux_cols.idx2_start.clone();
                SubAir::eval(&subairs.idx2_start, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: range_inclusion_cols.end.clone(),
                    y: cached_data.end.clone(),
                    tuple_less_than: range_inclusion_cols.greater_than_end.1,
                };
                let aux = subair_aux_cols.end_idx2.clone();
                SubAir::eval(&subairs.end_idx2, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.end.clone(),
                    y: prove_sort_cols.next_idx.clone(),
                    tuple_less_than: prove_sort_cols.end_less_than_next,
                };
                let aux = subair_aux_cols.end_next.clone();
                SubAir::eval(&subairs.end_next, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.end.clone(),
                    y: cached_data.start.clone(),
                    tuple_less_than: prove_sort_cols.end_less_than_start,
                };
                let aux = subair_aux_cols.end_start.clone();
                SubAir::eval(&subairs.end_start, builder, io, aux);
            }
            {
                let io = IsZeroIOCols {
                    x: metadata.mult_alloc_minus_one,
                    is_zero: metadata.mult_alloc_is_1,
                };
                let aux = subair_aux_cols.mult_inv;
                SubAir::eval(&subairs.mult_is_1, builder, io, aux);
            }
        }
    }
}
