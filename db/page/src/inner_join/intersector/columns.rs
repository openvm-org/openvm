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

    pub fn to_buf(self, buf: &mut Vec<T>) {
        buf.extend(self.idx);
        buf.push(self.t1_mult);
        buf.push(self.t2_mult);
        buf.push(self.out_mult);
        buf.push(self.is_extra);
    }

    pub fn flatten(self, intersector_air: &IntersectorAir) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::width(intersector_air));
        self.to_buf(&mut buf);
        buf
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

    pub fn to_buf(self, buf: &mut Vec<T>) {
        self.lt_aux.to_buf(buf);
        buf.push(self.lt_out);
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

    pub fn to_buf(self, buf: &mut Vec<T>) {
        self.io.to_buf(buf);
        self.aux.to_buf(buf);
    }

    pub fn flatten(self, intersector_air: &IntersectorAir) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::width(intersector_air));
        self.to_buf(&mut buf);
        buf
    }

    pub fn width(intersector_air: &IntersectorAir) -> usize {
        IntersectorIoCols::<usize>::width(intersector_air)
            + IntersectorAuxCols::<usize>::width(intersector_air)
    }
}
