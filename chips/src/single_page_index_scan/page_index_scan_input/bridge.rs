use crate::{
    is_less_than_tuple::columns::{
        IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIOCols,
    },
    sub_chip::SubAirBridge,
};

use super::{
    columns::{
        EqCompAuxCols, NonStrictCompAuxCols, PageIndexScanInputAuxCols, PageIndexScanInputCols,
        StrictCompAuxCols,
    },
    Comp, EqCompAir, NonStrictCompAir, PageIndexScanInputAirVariants, StrictCompAir,
};
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::PageIndexScanInputAir;

impl<F: PrimeField64> AirBridge<F> for PageIndexScanInputAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let mut interactions: Vec<Interaction<F>> = vec![];

        // when comparator is = we can use dummy values for idx_limb_bits and decomp
        let (idx_limb_bits, decomp) = match &self.variant_air {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => (
                is_less_than_tuple_air.limb_bits(),
                is_less_than_tuple_air.decomp(),
            ),
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => (vec![], 0),
        };

        let cmp = match &self.variant_air {
            PageIndexScanInputAirVariants::Lt(..) => Comp::Lt,
            PageIndexScanInputAirVariants::Gt(..) => Comp::Gt,
            PageIndexScanInputAirVariants::Lte(..) => Comp::Lte,
            PageIndexScanInputAirVariants::Gte(..) => Comp::Gte,
            PageIndexScanInputAirVariants::Eq(..) => Comp::Eq,
        };

        let num_cols = PageIndexScanInputCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            idx_limb_bits.clone(),
            decomp,
            cmp.clone(),
        );
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            idx_limb_bits.clone(),
            decomp,
            cmp.clone(),
        );

        // construct the row to send
        let mut cols = vec![];
        cols.push(cols_numbered.page_cols.is_alloc);
        cols.extend(cols_numbered.page_cols.idx.clone());
        cols.extend(cols_numbered.page_cols.data);

        let virtual_cols = cols
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        interactions.push(Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols_numbered.local_cols.send_row),
            argument_index: self.page_bus_index,
        });

        // here, we generate the flattened aux columns for IsLessThanTuple, and get the indicator associated with the strict comparison
        // when the comparator is =, we can just generate dummy values
        let (is_less_than_tuple_aux_flattened, strict_comp_ind): (
            Option<Vec<usize>>,
            Option<usize>,
        ) = match cols_numbered.local_cols.aux_cols {
            PageIndexScanInputAuxCols::Lt(StrictCompAuxCols {
                is_less_than_tuple_aux,
                ..
            })
            | PageIndexScanInputAuxCols::Gt(StrictCompAuxCols {
                is_less_than_tuple_aux,
                ..
            }) => (
                Some(is_less_than_tuple_aux.flatten()),
                Some(cols_numbered.local_cols.satisfies_pred),
            ),
            PageIndexScanInputAuxCols::Lte(NonStrictCompAuxCols {
                satisfies_strict_comp,
                is_less_than_tuple_aux,
                ..
            })
            | PageIndexScanInputAuxCols::Gte(NonStrictCompAuxCols {
                satisfies_strict_comp,
                is_less_than_tuple_aux,
                ..
            }) => (
                Some(is_less_than_tuple_aux.flatten()),
                Some(satisfies_strict_comp),
            ),
            PageIndexScanInputAuxCols::Eq(EqCompAuxCols { .. }) => (None, None),
        };

        // get interactions from IsLessThanTuple subchip
        let mut subchip_interactions = match &self.variant_air {
            PageIndexScanInputAirVariants::Lt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                let is_less_than_tuple_cols = IsLessThanTupleCols {
                    io: IsLessThanTupleIOCols {
                        x: cols_numbered.page_cols.idx.clone(),
                        y: cols_numbered.local_cols.x.clone(),
                        tuple_less_than: strict_comp_ind.unwrap(),
                    },
                    aux: IsLessThanTupleAuxCols::from_slice(
                        &is_less_than_tuple_aux_flattened.unwrap(),
                        idx_limb_bits,
                        decomp,
                        self.idx_len,
                    ),
                };

                SubAirBridge::<F>::sends(is_less_than_tuple_air, is_less_than_tuple_cols)
            }
            PageIndexScanInputAirVariants::Gt(StrictCompAir {
                is_less_than_tuple_air,
                ..
            })
            | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                is_less_than_tuple_air,
                ..
            }) => {
                let is_less_than_tuple_cols = IsLessThanTupleCols {
                    io: IsLessThanTupleIOCols {
                        x: cols_numbered.local_cols.x.clone(),
                        y: cols_numbered.page_cols.idx.clone(),
                        tuple_less_than: strict_comp_ind.unwrap(),
                    },
                    aux: IsLessThanTupleAuxCols::from_slice(
                        &is_less_than_tuple_aux_flattened.unwrap(),
                        idx_limb_bits,
                        decomp,
                        self.idx_len,
                    ),
                };

                SubAirBridge::<F>::sends(is_less_than_tuple_air, is_less_than_tuple_cols)
            }
            PageIndexScanInputAirVariants::Eq(EqCompAir { .. }) => vec![],
        };

        interactions.append(&mut subchip_interactions);

        interactions
    }
}
