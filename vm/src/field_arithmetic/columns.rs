use afs_derive::AlignedBorrow;
use p3_field::Field;

/// Columns for field arithmetic chip.
///
/// Four IO columns for opcode, x, y, result.
/// Eight aux columns for interpreting opcode, evaluating indicators, inverse, and explicit computations.
#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticCols<T> {
    pub io: FieldArithmeticIOCols<T>,
    pub aux: FieldArithmeticAuxCols<T>,
}

#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticIOCols<T> {
    pub opcode: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticAuxCols<T> {
    pub opcode_lo: T,
    pub opcode_hi: T,
    pub is_mul: T,
    pub is_div: T,
    pub sum_or_diff: T,
    pub product: T,
    pub quotient: T,
    pub divisor_inv: T,
}

impl<T> FieldArithmeticCols<T>
where
    T: Field,
{
    pub const NUM_COLS: usize = 12;
    pub const NUM_IO_COLS: usize = 4;
    pub const NUM_AUX_COLS: usize = 8;

    pub fn get_width() -> usize {
        FieldArithmeticIOCols::<T>::get_width() + FieldArithmeticAuxCols::<T>::get_width()
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }
}

impl<T: Field> FieldArithmeticIOCols<T> {
    pub fn get_width() -> usize {
        4
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.opcode, self.x, self.y, self.z]
    }
}

impl<T: Field> FieldArithmeticAuxCols<T> {
    pub fn get_width() -> usize {
        8
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.opcode_lo,
            self.opcode_hi,
            self.is_mul,
            self.is_div,
            self.sum_or_diff,
            self.product,
            self.quotient,
            self.divisor_inv,
        ]
    }
}
