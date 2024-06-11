use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::{AirConfig, SubAir},
};

use super::{columns::PageIndexScanInputCols, Comp, PageIndexScanInputAir};

impl AirConfig for PageIndexScanInputAir {
    type Cols<T> = PageIndexScanInputCols<T>;
}

impl<F: Field> BaseAir<F> for PageIndexScanInputAir {
    fn width(&self) -> usize {
        match &self {
            PageIndexScanInputAir::Lt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => PageIndexScanInputCols::<F>::get_width(
                *idx_len,
                *data_len,
                is_less_than_tuple_air.limb_bits().clone(),
                is_less_than_tuple_air.decomp(),
                Comp::Lt,
            ),
            PageIndexScanInputAir::Gt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => PageIndexScanInputCols::<F>::get_width(
                *idx_len,
                *data_len,
                is_less_than_tuple_air.limb_bits().clone(),
                is_less_than_tuple_air.decomp(),
                Comp::Gt,
            ),
        }
    }
}

impl<AB: PartitionedAirBuilder + AirBuilderWithPublicValues> Air<AB> for PageIndexScanInputAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        match &self {
            PageIndexScanInputAir::Lt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

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
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits().clone(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Lt,
                );

                match local_cols {
                    PageIndexScanInputCols::Lt {
                        is_alloc,
                        idx,
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        // here, we are checking if idx < x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: idx,
                                y: x.clone(),
                                tuple_less_than: satisfies_pred,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);

                        // constrain the indicator that we used to check wheter key < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );
                    }
                    PageIndexScanInputCols::Gt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Gt"
                        );
                    }
                }
            }
            PageIndexScanInputAir::Gt {
                idx_len,
                data_len,
                is_less_than_tuple_air,
                ..
            } => {
                let page_main = &builder.partitioned_main()[0].clone();
                let aux_main = &builder.partitioned_main()[1].clone();

                // get the public value x
                let pis = builder.public_values();
                let public_x = pis[..*idx_len].to_vec();

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
                    *idx_len,
                    *data_len,
                    is_less_than_tuple_air.limb_bits().clone(),
                    is_less_than_tuple_air.decomp(),
                    Comp::Gt,
                );

                match local_cols {
                    PageIndexScanInputCols::Lt { .. } => {
                        panic!(
                            "expected PageIndexScanInputCols::Lt, got PageIndexScanInputCols::Gt"
                        );
                    }
                    PageIndexScanInputCols::Gt {
                        is_alloc,
                        idx,
                        x,
                        satisfies_pred,
                        send_row,
                        is_less_than_tuple_aux,
                        ..
                    } => {
                        // here, we are checking if idx > x
                        let is_less_than_tuple_cols = IsLessThanTupleCols {
                            io: IsLessThanTupleIOCols {
                                x: x.clone(),
                                y: idx,
                                tuple_less_than: satisfies_pred,
                            },
                            aux: is_less_than_tuple_aux,
                        };

                        // constrain that the public value x is the same as the column x
                        for (&local_x, &pub_x) in x.iter().zip(public_x.iter()) {
                            builder.assert_eq(local_x, pub_x);
                        }

                        // constrain that we send the row iff the row is allocated and satisfies the predicate
                        builder.assert_eq(is_alloc * satisfies_pred, send_row);
                        builder.assert_bool(send_row);

                        // constrain the indicator that we used to check wheter key < x is correct
                        SubAir::eval(
                            is_less_than_tuple_air,
                            &mut builder.when_transition(),
                            is_less_than_tuple_cols.io,
                            is_less_than_tuple_cols.aux,
                        );
                    }
                }
            }
        }
    }
}
