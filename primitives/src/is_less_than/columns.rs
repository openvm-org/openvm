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

    pub fn to_buf(self, buf: &mut Vec<T>) {
        buf.push(self.x);
        buf.push(self.y);
        buf.push(self.less_than);
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

    pub fn to_buf(self, buf: &mut Vec<T>) {
        buf.push(self.lower);
        buf.extend(self.lower_decomp);
    }

    pub fn flatten(self, lt_air: &IsLessThanAir) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::width(lt_air));
        self.to_buf(&mut buf);
        buf
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

    pub fn to_buf(self, buf: &mut Vec<T>) {
        self.io.to_buf(buf);
        self.aux.to_buf(buf);
    }

    pub fn flatten(self, lt_air: &IsLessThanAir) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::width(lt_air));
        self.to_buf(&mut buf);
        buf
    }

    pub fn width(lt_air: &IsLessThanAir) -> usize {
        IsLessThanIoCols::<T>::width() + IsLessThanAuxCols::<T>::width(lt_air)
    }
}
