use std::iter;

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
    pub cmp_result: T,
}

pub struct LongArithmeticAuxCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub opcode_add_flag: T, // 1 if z_limbs should contain the result of addition
    pub opcode_sub_flag: T, // 1 if z_limbs should contain the result of subtraction (means that opcode is SUB or LT)
    pub opcode_lt_flag: T,  // 1 if opcode is LT
    pub opcode_eq_flag: T,  // 1 if opcode is EQ
    // buffer is the carry of the addition/subtraction,
    // or may serve as a single-nonzero-inverse helper vector for EQ256.
    // Refer to air.rs for more details.
    pub buffer: Vec<T>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let io = LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::from_iterator(iter.by_ref());
        let aux = LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::from_iterator(iter.by_ref());

        Self { io, aux }
    }
    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        self.io.flatten().chain(self.aux.flatten())
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
        3 * num_limbs::<ARG_SIZE, LIMB_SIZE>() + 3
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let rcv_count = iter.next().unwrap();
        let opcode = iter.next().unwrap();
        let x_limbs = iter.by_ref().take(num_limbs).collect();
        let y_limbs = iter.by_ref().take(num_limbs).collect();
        let z_limbs = iter.by_ref().take(num_limbs).collect();
        let cmp_result = iter.next().unwrap();

        Self {
            rcv_count,
            opcode,
            x_limbs,
            y_limbs,
            z_limbs,
            cmp_result,
        }
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        iter::once(&self.rcv_count)
            .chain(iter::once(&self.opcode))
            .chain(self.x_limbs.iter())
            .chain(self.y_limbs.iter())
            .chain(self.z_limbs.iter())
            .chain(iter::once(&self.cmp_result))
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        4 + num_limbs::<ARG_SIZE, LIMB_SIZE>()
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let opcode_add_flag = iter.next().unwrap();
        let opcode_sub_flag = iter.next().unwrap();
        let opcode_lt_flag = iter.next().unwrap();
        let opcode_eq_flag = iter.next().unwrap();
        let buffer = iter.by_ref().take(num_limbs).collect();

        Self {
            opcode_add_flag,
            opcode_sub_flag,
            opcode_lt_flag,
            opcode_eq_flag,
            buffer,
        }
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        iter::once(&self.opcode_add_flag)
            .chain(iter::once(&self.opcode_sub_flag))
            .chain(iter::once(&self.opcode_lt_flag))
            .chain(iter::once(&self.opcode_eq_flag))
            .chain(self.buffer.iter())
    }
}
