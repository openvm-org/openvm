use afs_primitives::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

use super::IntersectorAir;

#[derive(Debug)]
pub struct IntersectorIoCols<T> {
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

impl<T: Clone> IntersectorIoCols<T> {
    pub fn from_slice(slc: &[T], intersector_air: &IntersectorAir) -> Self {
        Self {
            idx: slc[..intersector_air.idx_len].to_vec(),
            t1_mult: slc[intersector_air.idx_len].clone(),
            t2_mult: slc[intersector_air.idx_len + 1].clone(),
            out_mult: slc[intersector_air.idx_len + 2].clone(),
            is_extra: slc[intersector_air.idx_len + 3].clone(),
        }
    }

    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        buf[start..start + self.idx.len()].clone_from_slice(&self.idx);
        buf[start + self.idx.len()] = self.t1_mult.clone();
        buf[start + self.idx.len() + 1] = self.t2_mult.clone();
        buf[start + self.idx.len() + 2] = self.out_mult.clone();
        buf[start + self.idx.len() + 3] = self.is_extra.clone();
        self.idx.len() + 4
    }

    pub fn width(intersector_air: &IntersectorAir) -> usize {
        intersector_air.idx_len + 4
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
    pub fn from_slice(slc: &[T], intersector_air: &IntersectorAir) -> Self {
        Self {
            lt_aux: IsLessThanTupleAuxCols::from_slice(
                &slc[..slc.len() - 1],
                &intersector_air.lt_chip,
            ),
            lt_out: slc[slc.len() - 1].clone(),
        }
    }

    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        let lt_len = self.lt_aux.flatten(buf, start);
        buf[start + lt_len] = self.lt_out.clone();
        lt_len + 1
    }

    pub fn width(intersector_air: &IntersectorAir) -> usize {
        IsLessThanTupleAuxCols::<usize>::width(&intersector_air.lt_chip) + 1
    }
}

#[derive(Debug)]
pub struct IntersectorCols<T> {
    pub io: IntersectorIoCols<T>,
    pub aux: IntersectorAuxCols<T>,
}

impl<T: Clone> IntersectorCols<T> {
    pub fn from_slice(slc: &[T], intersector_air: &IntersectorAir) -> Self {
        assert!(slc.len() == intersector_air.air_width());

        Self {
            io: IntersectorIoCols::from_slice(&slc[..intersector_air.io_width()], intersector_air),
            aux: IntersectorAuxCols::from_slice(
                &slc[intersector_air.io_width()..],
                intersector_air,
            ),
        }
    }

    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        let io_len = self.io.flatten(buf, start);
        let aux_len = self.aux.flatten(buf, start + io_len);
        io_len + aux_len
    }

    pub fn width(intersector_air: &IntersectorAir) -> usize {
        IntersectorIoCols::<usize>::width(intersector_air)
            + IntersectorAuxCols::<usize>::width(intersector_air)
    }
}
