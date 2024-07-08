use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{
    columns::{LeafPageCols, LeafPageMetadataCols},
    LeafPageAir, PageRWAir,
};
use crate::{
    common::page_cols::PageCols,
    is_less_than_tuple::columns::IsLessThanTupleIOCols,
    sub_chip::{AirConfig, SubAir},
};

impl<F: Field, const COMMITMENT_LEN: usize> BaseAir<F> for LeafPageAir<COMMITMENT_LEN> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const COMMITMENT_LEN: usize> AirConfig for LeafPageAir<COMMITMENT_LEN> {
    type Cols<T> = LeafPageCols<T>;
}

impl<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for LeafPageAir<COMMITMENT_LEN>
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
        let cached_data = PageCols::from_slice(&data.row_slice(0), self.idx_len, self.data_len);
        let next_data = PageCols::from_slice(&data.row_slice(1), self.idx_len, self.data_len);
        for i in 0..COMMITMENT_LEN {
            builder.assert_eq(pi[i], local[i]);
        }
        // assert that own id is correct
        builder.assert_eq(
            local[COMMITMENT_LEN],
            AB::Expr::from_canonical_u64(self.air_id as u64),
        );
        match &self.page_chip {
            PageRWAir::Initial(i) => {
                SubAir::eval(i, builder, cached_data, ());
            }
            PageRWAir::Final(f) => {
                let metadata = LeafPageMetadataCols::from_slice(
                    &local,
                    self.idx_len,
                    COMMITMENT_LEN,
                    false,
                    self.is_less_than_tuple_param.clone(),
                );
                let next_aux = LeafPageMetadataCols::from_slice(
                    &main.row_slice(1),
                    self.idx_len,
                    COMMITMENT_LEN,
                    false,
                    self.is_less_than_tuple_param.clone(),
                )
                .subchip_aux_cols
                .unwrap()
                .final_page_aux;
                let range_inclusion_cols = metadata.range_inclusion_cols.unwrap();
                let less_than_start = range_inclusion_cols.less_than_start;
                let greater_than_end = range_inclusion_cols.greater_than_end;
                builder.assert_zero(cached_data.is_alloc * (less_than_start + greater_than_end));
                let subair_aux_cols = metadata.subchip_aux_cols.unwrap();
                let subairs = self.is_less_than_tuple_air.clone().unwrap();
                {
                    let io = IsLessThanTupleIOCols {
                        x: cached_data.idx.clone(),
                        y: range_inclusion_cols.start.clone(),
                        tuple_less_than: range_inclusion_cols.less_than_start,
                    };
                    let aux = subair_aux_cols.idx_start.clone();
                    SubAir::eval(&subairs.idx_start, builder, io, aux);
                }
                {
                    let io = IsLessThanTupleIOCols {
                        x: range_inclusion_cols.end.clone(),
                        y: cached_data.idx.clone(),
                        tuple_less_than: range_inclusion_cols.greater_than_end,
                    };
                    let aux = subair_aux_cols.end_idx.clone();
                    SubAir::eval(&subairs.end_idx, builder, io, aux);
                }
                SubAir::eval(f, builder, [cached_data, next_data], next_aux);
            }
        };
    }
}
