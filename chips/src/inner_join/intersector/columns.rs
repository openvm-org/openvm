use std::iter;

use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

use super::IntersectorAir;

#[derive(Debug)]
pub struct IntersectorIOCols<T> {
    /// index for the row
    pub idx: Vec<T>,
    /// Multiplicity of idx in t2
    pub t1_mult: T,
    /// Multiplicity of idx in t2
    pub t2_mult: T,
    /// Multiplicity of idx in output_table
    pub out_mult: T,
    /// Indiates if this row is extra and should be ignored
    pub is_extra: T,
}

impl<T: Clone> IntersectorIOCols<T> {
    pub fn from_slice(slc: &[T], intersector_air: &IntersectorAir) -> Self {
        Self {
            idx: slc[..intersector_air.idx_len].to_vec(),
            t1_mult: slc[intersector_air.idx_len].clone(),
            t2_mult: slc[intersector_air.idx_len + 1].clone(),
            out_mult: slc[intersector_air.idx_len + 2].clone(),
            is_extra: slc[intersector_air.idx_len + 3].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.idx
            .clone()
            .into_iter()
            .chain(iter::once(self.t1_mult.clone()))
            .chain(iter::once(self.t2_mult.clone()))
            .chain(iter::once(self.out_mult.clone()))
            .chain(iter::once(self.is_extra.clone()))
            .collect()
    }
}

#[derive(Debug)]
pub struct IntersectorAuxCols<T> {
    /// Columns used by the IsLessThanTupleAir to ensure sorting
    pub lt_aux: IsLessThanTupleAuxCols<T>,
    /// Indicates if idx is greater than idx in the previous row
    pub lt_out: T,
}

impl<T: Clone> IntersectorAuxCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        self.lt_aux
            .flatten()
            .into_iter()
            .chain(iter::once(self.lt_out.clone()))
            .collect()
    }
}

#[derive(Debug)]
pub struct IntersectorCols<T> {
    pub io: IntersectorIOCols<T>,
    pub aux: IntersectorAuxCols<T>,
}

impl<T: Clone> IntersectorCols<T> {
    pub fn from_slice(slc: &[T], intersector_air: &IntersectorAir) -> Self {
        assert!(slc.len() == intersector_air.air_width());
        let idx_len = intersector_air.idx_len;

        Self {
            io: IntersectorIOCols::from_slice(&slc[..idx_len + 4], intersector_air),
            aux: IntersectorAuxCols {
                lt_aux: IsLessThanTupleAuxCols::from_slice(
                    // Note: if we start using passing an AIR reference to the from_slice everywhere, I can pass in intersector_air.lt_chip here
                    &slc[idx_len + 4..],
                    intersector_air.lt_chip.limb_bits(),
                    intersector_air.lt_chip.decomp(),
                    intersector_air.lt_chip.tuple_len(),
                ),
                lt_out: slc[slc.len() - 1].clone(),
            },
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }
}
