use crate::cpu::OpCode;
use afs_derive::AlignedBorrow;
use p3_field::Field;

use crate::au::FieldArithmeticAir;

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
    pub opcode_0bit: T,
    pub opcode_1bit: T,
    pub is_mul: T,
    pub is_div: T,
    pub lin_term: T,
    pub mul_result: T,
    pub div_result: T,
}

impl<T> FieldArithmeticCols<T>
where
    T: Field,
{
    pub const NUM_COLS: usize = 11;
    pub const NUM_IO_COLS: usize = 4;
    pub const NUM_AUX_COLS: usize = 6;

    pub fn new(op: OpCode, x: T, y: T) -> Self {
        let opcode = op as u32;
        let opcode_value = opcode - FieldArithmeticAir::BASE_OP as u32;
        let opcode_0bit_u32 = opcode_value % 2;
        let opcode_1bit_u32 = opcode_value / 2;
        let opcode_0bit = T::from_canonical_u32(opcode_0bit_u32);
        let opcode_1bit = T::from_canonical_u32(opcode_1bit_u32);
        let lin_term = x + y - T::two() * opcode_0bit * y;
        let is_div = T::from_bool(op == OpCode::FDIV);
        let is_mul = T::from_bool(op == OpCode::FMUL);
        let mul_result = x * y;
        let div_result = x * y.inverse();
        let z = is_mul * mul_result + is_div * div_result + (T::one() - opcode_1bit) * lin_term;

        Self {
            io: FieldArithmeticIOCols {
                opcode: T::from_canonical_u32(opcode),
                x,
                y,
                z,
            },
            aux: FieldArithmeticAuxCols {
                opcode_0bit,
                opcode_1bit,
                is_mul,
                is_div,
                lin_term,
                mul_result,
                div_result,
            },
        }
    }

    pub fn get_width() -> usize {
        11
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
            self.opcode_0bit,
            self.opcode_1bit,
            self.is_mul,
            self.is_div,
            self.lin_term,
            self.mul_result,
            self.div_result,
        ]
    }
}
