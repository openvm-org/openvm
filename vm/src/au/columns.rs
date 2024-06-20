use crate::cpu::OpCode;
use afs_derive::AlignedBorrow;
use p3_field::Field;

use crate::au::FieldArithmeticAir;

/// Columns for field arithmetic chip.
///
/// Four IO columns for opcode, x, y, result.
/// Seven aux columns for interpreting opcode, evaluating indicators, and explicit computations.
#[derive(AlignedBorrow)]
pub struct FieldArithmeticCols<T> {
    pub io: FieldArithmeticIOCols<T>,
    pub aux: FieldArithmeticAuxCols<T>,
}

pub struct FieldArithmeticIOCols<T> {
    pub opcode: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

pub struct FieldArithmeticAuxCols<T> {
    pub opcode_lo: T,
    pub opcode_hi: T,
    pub is_mul: T,
    pub is_div: T,
    pub sum_or_diff: T,
    pub product: T,
    pub quotient: T,
}

impl<T> FieldArithmeticCols<T>
where
    T: Field,
{
    pub const NUM_COLS: usize = 11;
    pub const NUM_IO_COLS: usize = 4;
    pub const NUM_AUX_COLS: usize = 6;

    /// Constructs a new set of columns (including auxiliary columns) given inputs.
    pub fn new(op: OpCode, x: T, y: T) -> Self {
        let opcode = op as u32;
        let opcode_value = opcode - FieldArithmeticAir::BASE_OP as u32;
        let opcode_lo_u32 = opcode_value % 2;
        let opcode_hi_u32 = opcode_value / 2;
        let opcode_lo = T::from_canonical_u32(opcode_lo_u32);
        let opcode_hi = T::from_canonical_u32(opcode_hi_u32);
        let is_div = T::from_bool(op == OpCode::FDIV);
        let is_mul = T::from_bool(op == OpCode::FMUL);
        let sum_or_diff = x + y - T::two() * opcode_lo * y;
        let product = x * y;
        let quotient = if y == T::zero() {
            T::zero()
        } else {
            x * y.inverse()
        };
        let z = is_mul * product + is_div * quotient + (T::one() - opcode_hi) * sum_or_diff;

        Self {
            io: FieldArithmeticIOCols {
                opcode: T::from_canonical_u32(opcode),
                x,
                y,
                z,
            },
            aux: FieldArithmeticAuxCols {
                opcode_lo,
                opcode_hi,
                is_mul,
                is_div,
                sum_or_diff,
                product,
                quotient,
            },
        }
    }

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
        7
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
        ]
    }
}
