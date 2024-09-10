use afs_primitives::{
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    var_range::bus::VariableRangeCheckerBus,
};

use crate::multitier_page_rw_checker::page_controller::MyLessThanTupleParams;

#[derive(Clone)]
pub struct InternalPageCols<T> {
    pub cache_cols: PtrPageCols<T>,
    pub metadata: InternalPageMetadataCols<T>,
}

#[derive(Clone)]
pub struct PtrPageCols<T> {
    pub internal_marker: T,
    pub is_alloc: T,
    pub child_start: Vec<T>,
    pub child_end: Vec<T>,
    pub commitment: Vec<T>,
}

#[derive(Clone)]
pub struct InternalPageSubAirCols<T> {
    // check if the 1st idx of this row is less than the lower bound assigned to this page -> want this to be false
    pub idx1_start: IsLessThanTupleAuxCols<T>,
    // check if the upper bound assigned to this page is less than the 2nd idx of this row -> want this to be false
    pub end_idx2: IsLessThanTupleAuxCols<T>,
    // check if the 2nd idx of this row is less than the 1st idx of this row -> want this to be false
    pub idx2_idx1: IsLessThanTupleAuxCols<T>,
    // check if the 2nd idx of this row is less than the 1st idx of the next -> want this to be true
    pub idx2_next: IsLessThanTupleAuxCols<T>,
    // aux for is_zero of mult_minus_one_alloc
    pub mult_inv: T,
}

/// A parent of this page will assign some range of keys - we must prove that range is accurate
#[derive(Clone)]
pub struct TwoRangeInclusionCols<T> {
    pub start: Vec<T>,
    pub end: Vec<T>,
    pub less_than_start: T,
    pub greater_than_end: T,
}

#[derive(Clone)]
pub struct ProveSortCols<T> {
    // we want this to be true
    pub end_less_than_next: T,
    // we want this to be false
    pub end_less_than_start: T,
}

#[derive(Clone)]
pub struct InternalPageMetadataCols<T> {
    pub child_air_id: T,
    pub mult: T,
    pub mult_alloc: T,
    pub mult_alloc_minus_one: T,
    pub mult_alloc_is_1: T,
    pub mult_minus_one_alloc: T,
    pub prove_sort_cols: Option<ProveSortCols<T>>,
    pub range_inclusion_cols: Option<TwoRangeInclusionCols<T>>,
    pub subair_aux_cols: Option<InternalPageSubAirCols<T>>,
}

impl<T> InternalPageCols<T> {
    pub fn from_slice(
        cols: &[T],
        idx_len: usize,
        commitment_len: usize,
        is_init: bool,
        is_less_than_tuple_params: MyLessThanTupleParams,
    ) -> Self
    where
        T: Clone,
    {
        InternalPageCols {
            cache_cols: PtrPageCols::from_slice(
                &cols[0..2 + 2 * idx_len + commitment_len],
                idx_len,
                commitment_len,
            ),
            metadata: InternalPageMetadataCols::from_slice(
                &cols[2 + 2 * idx_len + commitment_len..],
                idx_len,
                is_init,
                is_less_than_tuple_params,
            ),
        }
    }
}

impl<T> PtrPageCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, commitment_len: usize) -> Self
    where
        T: Clone,
    {
        PtrPageCols {
            internal_marker: cols[0].clone(),
            is_alloc: cols[1].clone(),
            child_start: cols[2..2 + idx_len].to_vec(),
            child_end: cols[2 + idx_len..2 + 2 * idx_len].to_vec(),
            commitment: cols[2 + 2 * idx_len..2 + 2 * idx_len + commitment_len].to_vec(),
        }
    }
}

impl<T> InternalPageMetadataCols<T> {
    pub fn from_slice(
        cols: &[T],
        idx_len: usize,
        is_init: bool,
        is_less_than_tuple_params: MyLessThanTupleParams,
    ) -> Self
    where
        T: Clone,
    {
        if is_init {
            InternalPageMetadataCols {
                child_air_id: cols[0].clone(),
                mult: cols[1].clone(),
                mult_alloc: cols[2].clone(),
                mult_alloc_is_1: cols[3].clone(),
                mult_alloc_minus_one: cols[4].clone(),
                mult_minus_one_alloc: cols[5].clone(),
                prove_sort_cols: None,
                range_inclusion_cols: None,
                subair_aux_cols: None,
            }
        } else {
            let mut new_start = 6;
            let prove_sort_cols = ProveSortCols {
                end_less_than_next: cols[new_start].clone(),
                end_less_than_start: cols[new_start + 1].clone(),
            };
            new_start += 2;
            let range_inclusion_cols = TwoRangeInclusionCols {
                start: cols[new_start..new_start + idx_len].to_vec(),
                end: cols[new_start + idx_len..new_start + 2 * idx_len].to_vec(),
                less_than_start: cols[new_start + 2 * idx_len].clone(),
                greater_than_end: cols[new_start + 2 * idx_len + 1].clone(),
            };
            new_start += 2 * idx_len + 2;
            let mut aux_allocs = vec![];
            let range_bus = VariableRangeCheckerBus::new(0, is_less_than_tuple_params.decomp);
            let aux_size = IsLessThanTupleAuxCols::<T>::width(&IsLessThanTupleAir::new(
                range_bus,
                vec![is_less_than_tuple_params.limb_bits; idx_len],
            ));
            for i in 0..4 {
                aux_allocs.push(IsLessThanTupleAuxCols::from_slice(
                    &cols[new_start + i * aux_size..new_start + (i + 1) * aux_size],
                    &IsLessThanTupleAir::new(
                        range_bus,
                        vec![is_less_than_tuple_params.limb_bits; idx_len],
                    ),
                ))
            }
            let subair_cols = InternalPageSubAirCols {
                idx1_start: aux_allocs[0].clone(),
                end_idx2: aux_allocs[1].clone(),
                idx2_next: aux_allocs[2].clone(),
                idx2_idx1: aux_allocs[3].clone(),
                mult_inv: cols[new_start + 4 * aux_size].clone(),
            };
            InternalPageMetadataCols {
                child_air_id: cols[0].clone(),
                mult: cols[1].clone(),
                mult_alloc: cols[2].clone(),
                mult_alloc_is_1: cols[3].clone(),
                mult_alloc_minus_one: cols[4].clone(),
                mult_minus_one_alloc: cols[5].clone(),
                prove_sort_cols: Some(prove_sort_cols),
                range_inclusion_cols: Some(range_inclusion_cols),
                subair_aux_cols: Some(subair_cols),
            }
        }
    }
}
