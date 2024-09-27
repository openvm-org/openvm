use afs_primitives::{
    is_less_than_tuple::columns::IsLessThanTupleIoCols,
    sub_chip::{AirConfig, SubAir},
};
use afs_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder,
    rap::BaseAirWithPublicValues,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{
    columns::{LeafPageCols, LeafPageMetadataCols},
    LeafPageAir, PageRwAir,
};
use crate::common::page_cols::PageCols;

impl<F: Field, const COMMITMENT_LEN: usize> BaseAir<F> for LeafPageAir<COMMITMENT_LEN> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<F: Field, const COMMITMENT_LEN: usize> BaseAirWithPublicValues<F>
    for LeafPageAir<COMMITMENT_LEN>
{
    fn num_public_values(&self) -> usize {
        COMMITMENT_LEN
    }
}

impl<const COMMITMENT_LEN: usize> AirConfig for LeafPageAir<COMMITMENT_LEN> {
    type Cols<T> = LeafPageCols<T>;
}

impl<
        AB: AirBuilder + AirBuilderWithPublicValues + PartitionedAirBuilder + InteractionBuilder,
        const COMMITMENT_LEN: usize,
    > Air<AB> for LeafPageAir<COMMITMENT_LEN>
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // only constrain that own_commitment is accurate
        // partition is physical page data vs metadata
        let pi = builder.public_values().to_vec();
        match &self.page_chip {
            PageRwAir::Initial(i) => {
                let data: &<AB as AirBuilder>::M = &builder.partitioned_main()[0];
                let cached_data =
                    PageCols::from_slice(&data.row_slice(0), self.idx_len, self.data_len);
                let page_cols = LeafPageCols {
                    metadata: LeafPageMetadataCols {
                        range_inclusion_cols: None,
                        subair_aux_cols: None,
                    },
                    cache_cols: cached_data,
                };
                self.eval_interactions(builder, &page_cols, &pi);
                SubAir::eval(i, builder, page_cols.cache_cols, ());
            }
            PageRwAir::Final(fin) => {
                let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[1];
                let [local, next] = [0, 1].map(|i| main.row_slice(i));
                let data: &<AB as AirBuilder>::M = &builder.partitioned_main()[0];
                let cached_data =
                    PageCols::from_slice(&data.row_slice(0), self.idx_len, self.data_len);
                let next_data =
                    PageCols::from_slice(&data.row_slice(1), self.idx_len, self.data_len);
                let page_cols = LeafPageCols {
                    metadata: LeafPageMetadataCols::from_slice(
                        &local,
                        self.idx_len,
                        self.is_init,
                        self.is_less_than_tuple_param.clone(),
                    ),
                    cache_cols: cached_data,
                };
                let next_aux = LeafPageMetadataCols::from_slice(
                    &next,
                    self.idx_len,
                    false,
                    self.is_less_than_tuple_param.clone(),
                )
                .subair_aux_cols
                .unwrap()
                .final_page_aux;
                drop(local);
                drop(next);
                self.eval_interactions(builder, &page_cols, &pi);

                let range_inclusion_cols = page_cols.metadata.range_inclusion_cols.unwrap();
                let less_than_start = range_inclusion_cols.less_than_start;
                let greater_than_end = range_inclusion_cols.greater_than_end;
                builder.assert_zero(
                    page_cols.cache_cols.is_alloc * (less_than_start + greater_than_end),
                );
                let subair_aux_cols = page_cols.metadata.subair_aux_cols.unwrap();
                let subairs = self.is_less_than_tuple_air.clone().unwrap();
                {
                    let io = IsLessThanTupleIoCols {
                        x: page_cols.cache_cols.idx.clone(),
                        y: range_inclusion_cols.start.clone(),
                        tuple_less_than: range_inclusion_cols.less_than_start,
                    };
                    let aux = subair_aux_cols.idx_start.clone();
                    SubAir::eval(&subairs.idx_start, builder, io, aux);
                }
                {
                    let io = IsLessThanTupleIoCols {
                        x: range_inclusion_cols.end.clone(),
                        y: page_cols.cache_cols.idx.clone(),
                        tuple_less_than: range_inclusion_cols.greater_than_end,
                    };
                    let aux = subair_aux_cols.end_idx.clone();
                    SubAir::eval(&subairs.end_idx, builder, io, aux);
                }
                SubAir::eval(
                    fin,
                    builder,
                    [page_cols.cache_cols, next_data],
                    [subair_aux_cols.final_page_aux, next_aux],
                );
            }
        };
    }
}
