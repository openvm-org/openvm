use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{InternalPageCols, InternalPageMetadataCols, PtrPageCols},
    InternalPageChip,
};
use crate::{
    is_equal::columns::IsEqualIOCols,
    is_less_than_tuple::{columns::IsLessThanTupleIOCols, IsLessThanTupleAir},
    is_zero::columns::IsZeroIOCols,
    sub_chip::{AirConfig, SubAir},
};

impl<F: Field, const COMMITMENT_LEN: usize> BaseAir<F> for InternalPageChip<COMMITMENT_LEN> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const COMMITMENT_LEN: usize> AirConfig for InternalPageChip<COMMITMENT_LEN> {
    type Cols<T> = InternalPageCols<T>;
}

impl<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for InternalPageChip<COMMITMENT_LEN>
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
        for i in 0..COMMITMENT_LEN {
            builder.assert_eq(pi[i], metadata.own_commitment[i]);
        }
        builder.assert_eq(metadata.mult_alloc, cached_data.is_alloc * metadata.mult);

        if self.is_init {
            return;
        }
        // assert stuff is in range
        else {
            // assert that next_key is the same as the thing in the next row
            // will do the allocated rows are at the top check later probably

            let prove_sort_cols = metadata.prove_sort_cols.unwrap();
            for i in 0..self.idx_len {
                builder.when_transition().assert_zero(
                    next_data.is_alloc * (next_data.start[i] - prove_sort_cols.next_key[i]),
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
                    tuple_less_than: range_inclusion_cols.less_than_start.0.clone(),
                };
                let aux = subair_aux_cols.key1_start.clone();
                SubAir::eval(&subairs.key1_start, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: range_inclusion_cols.end.clone(),
                    y: cached_data.start.clone(),
                    tuple_less_than: range_inclusion_cols.greater_than_end.0.clone(),
                };
                let aux = subair_aux_cols.end_key1.clone();
                SubAir::eval(&subairs.end_key1, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.end.clone(),
                    y: range_inclusion_cols.start.clone(),
                    tuple_less_than: range_inclusion_cols.less_than_start.1.clone(),
                };
                let aux = subair_aux_cols.key2_start.clone();
                SubAir::eval(&subairs.key2_start, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: range_inclusion_cols.end.clone(),
                    y: cached_data.end.clone(),
                    tuple_less_than: range_inclusion_cols.greater_than_end.1.clone(),
                };
                let aux = subair_aux_cols.end_key2.clone();
                SubAir::eval(&subairs.end_key2, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.end.clone(),
                    y: cached_data.start.clone(),
                    tuple_less_than: prove_sort_cols.end_less_than_start.clone(),
                };
                let aux = subair_aux_cols.end_start.clone();
                SubAir::eval(&subairs.end_start, builder, io, aux);
            }
            {
                let io = IsLessThanTupleIOCols {
                    x: cached_data.end.clone(),
                    y: prove_sort_cols.next_key.clone(),
                    tuple_less_than: prove_sort_cols.end_less_than_next.clone(),
                };
                let aux = subair_aux_cols.end_next.clone();
                SubAir::eval(&subairs.end_next, builder, io, aux);
            }
            {
                let io = IsZeroIOCols {
                    x: metadata.mult_alloc_minus_one.clone(),
                    is_zero: metadata.mult_alloc_is_1,
                };
                let aux = subair_aux_cols.mult_inv.clone();
                SubAir::eval(&subairs.mult_is_1, builder, io, aux);
            }
        }
    }
}
