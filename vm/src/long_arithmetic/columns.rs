use super::num_limbs;

pub struct LongArithmeticCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub io: LongArithmeticIoCols<ARG_SIZE, LIMB_SIZE, T>,
    pub aux: LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>,
}

pub struct LongArithmeticIoCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub rcv_count: T,
    pub opcode: T,
    pub x_limbs: Vec<T>,
    pub y_limbs: Vec<T>,
    pub z_limbs: Vec<T>,
}

pub struct LongArithmeticAuxCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub opcode_lo: T,
    pub opcode_hi: T,
    pub carry: Vec<T>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let io =
            LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::from_slice(&slc[..3 * num_limbs + 2]);
        let aux =
            LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::from_slice(&slc[3 * num_limbs + 2..]);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.extend(self.io.flatten());
        flattened.extend(self.aux.flatten());

        flattened
    }

    pub const fn get_width() -> usize {
        LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::get_width()
            + LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::get_width()
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticIoCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        3 * num_limbs::<ARG_SIZE, LIMB_SIZE>() + 2
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let rcv_count = slc[0].clone();
        let opcode = slc[1].clone();
        let x_limbs = slc[2..2 + num_limbs].to_vec();
        let y_limbs = slc[2 + num_limbs..2 + 2 * num_limbs].to_vec();
        let z_limbs = slc[2 + 2 * num_limbs..2 + 3 * num_limbs].to_vec();

        Self {
            rcv_count,
            opcode,
            x_limbs,
            y_limbs,
            z_limbs,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.push(self.rcv_count.clone());
        flattened.push(self.opcode.clone());
        flattened.extend(self.x_limbs.clone());
        flattened.extend(self.y_limbs.clone());
        flattened.extend(self.z_limbs.clone());

        flattened
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        2 + num_limbs::<ARG_SIZE, LIMB_SIZE>()
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let opcode_lo = slc[0].clone();
        let opcode_hi = slc[1].clone();
        let carry = slc[2..2 + num_limbs].to_vec();

        Self {
            opcode_lo,
            opcode_hi,
            carry,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.push(self.opcode_lo.clone());
        flattened.push(self.opcode_hi.clone());
        flattened.extend(self.carry.clone());

        flattened
    }
}
