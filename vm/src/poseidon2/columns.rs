use super::Poseidon2Chip;
use p3_field::Field;
use poseidon2_air::poseidon2::columns::{Poseidon2Cols, Poseidon2ColsIndexMap};
use poseidon2_air::poseidon2::Poseidon2Air;
/// Columns for field arithmetic chip.
///
/// Five IO columns for rcv_count, opcode, x, y, result.
/// Eight aux columns for interpreting opcode, evaluating indicators, inverse, and explicit computations.
pub struct Poseidon2ChipCols<const WIDTH: usize, T> {
    pub io: Poseidon2ChipIoCols<T>,
    pub internal: Poseidon2Cols<WIDTH, T>,
}

/// IO columns for Poseidon2Chip.
/// * `is_alloc`: whether the row is allocated
/// * `clk`: the clock cycle (NOT timestamp)
/// * `a`, `b`, `c`: addresses
/// * `d`, `e`: address spaces
/// * `cmp`: boolean for compression vs. permutation
pub struct Poseidon2ChipIoCols<T> {
    pub is_alloc: T,
    pub clk: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub cmp: T,
}

impl<const WIDTH: usize, T: Clone> Poseidon2ChipCols<WIDTH, T> {
    pub fn get_width(poseidon2_chip: &Poseidon2Chip<WIDTH, T>) -> usize {
        Poseidon2ChipIoCols::<T>::get_width()
            + Poseidon2Cols::<WIDTH, T>::get_width(&poseidon2_chip.air)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.internal.flatten());
        result
    }

    pub fn from_slice(
        slice: &[T],
        index_map: &Poseidon2ColsIndexMap<WIDTH>,
    ) -> Poseidon2ChipCols<WIDTH, T> {
        let io_width = Poseidon2ChipIoCols::<T>::get_width();
        Self {
            io: Poseidon2ChipIoCols::<T>::from_slice(&slice[..io_width]),
            internal: Poseidon2Cols::<WIDTH, T>::from_slice(&slice[io_width..], index_map),
        }
    }
}

impl<const WIDTH: usize, T: Field> Poseidon2ChipCols<WIDTH, T> {
    /// Blank row with all zero input (poseidon2 internal hash values are nonzero)
    /// and `is_alloc` set to 0.
    pub fn blank_row(poseidon2_air: &Poseidon2Air<WIDTH, T>) -> Self {
        Self {
            io: Poseidon2ChipIoCols::<T>::blank_row(),
            internal: Poseidon2Cols::<WIDTH, T>::blank_row(poseidon2_air),
        }
    }
}

impl<T: Clone> Poseidon2ChipIoCols<T> {
    pub fn get_width() -> usize {
        8
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.is_alloc.clone(),
            self.clk.clone(),
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
            self.e.clone(),
            self.cmp.clone(),
        ]
    }

    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            is_alloc: slice[0].clone(),
            clk: slice[1].clone(),
            a: slice[2].clone(),
            b: slice[3].clone(),
            c: slice[4].clone(),
            d: slice[5].clone(),
            e: slice[6].clone(),
            cmp: slice[7].clone(),
        }
    }
}
impl<T: Field> Poseidon2ChipIoCols<T> {
    pub fn blank_row() -> Self {
        Self {
            is_alloc: T::zero(),
            clk: T::zero(),
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::zero(),
            e: T::zero(),
            cmp: T::zero(),
        }
    }
}
