use afs_derive::AlignedBorrow;

use super::IsLessThanAir;

#[derive(Default, AlignedBorrow)]
pub struct IsLessThanIoCols<T> {
    pub x: T,
    pub y: T,
    pub less_than: T,
}

impl<T: Clone> IsLessThanIoCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            x: slc[0].clone(),
            y: slc[1].clone(),
            less_than: slc[2].clone(),
        }
    }

    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        buf[start] = self.x.clone();
        buf[start + 1] = self.y.clone();
        buf[start + 2] = self.less_than.clone();
        3
    }

    pub fn width() -> usize {
        3
    }
}

#[derive(Debug, Clone)]
pub struct IsLessThanAuxCols<T> {
    pub lower: T,
    // lower_decomp consists of lower decomposed into limbs of size decomp where we also shift
    // the final limb and store it as the last element of lower decomp so we can range check
    pub lower_decomp: Vec<T>,
}

impl<T: Clone> IsLessThanAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            lower: slc[0].clone(),
            lower_decomp: slc[1..].to_vec(),
        }
    }

    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        buf[start] = self.lower.clone();
        buf[start + 1..start + 1 + self.lower_decomp.len()].clone_from_slice(&self.lower_decomp);
        1 + self.lower_decomp.len()
    }

    pub fn width(lt_air: &IsLessThanAir) -> usize {
        1 + lt_air.num_limbs + (lt_air.max_bits % lt_air.decomp != 0) as usize
    }
}

pub struct IsLessThanCols<T> {
    pub io: IsLessThanIoCols<T>,
    pub aux: IsLessThanAuxCols<T>,
}

impl<T: Clone> IsLessThanCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let io = IsLessThanIoCols::from_slice(&slc[..3]);
        let aux = IsLessThanAuxCols::from_slice(&slc[3..]);

        Self { io, aux }
    }

    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        let io_len = self.io.flatten(buf, start);
        let aux_len = self.aux.flatten(buf, start + io_len);
        io_len + aux_len
    }

    pub fn width(lt_air: &IsLessThanAir) -> usize {
        IsLessThanIoCols::<T>::width() + IsLessThanAuxCols::<T>::width(lt_air)
    }
}
