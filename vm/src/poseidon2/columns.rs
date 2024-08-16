use p3_field::Field;

use poseidon2_air::poseidon2::{
    columns::{Poseidon2Cols, Poseidon2ColsIndexMap},
    Poseidon2Air,
};

use super::Poseidon2VmAir;

/// Columns for Poseidon2Vm AIR.
pub struct Poseidon2VmCols<const WIDTH: usize, T> {
    pub io: Poseidon2VmIoCols<T>,
    pub aux: Poseidon2VmAuxCols<WIDTH, T>,
}

/// IO columns for Poseidon2Chip.
/// * `is_opcode`: whether the row is for an opcode (either COMPRESS or PERMUTE)
/// * `is_direct`: whether the row is for a direct hash
/// * `clk`: the clock cycle (NOT timestamp)
/// * `a`, `b`, `c`: addresses
/// * `d`, `e`: address spaces
/// * `cmp`: boolean for compression vs. permutation
#[derive(Clone, Copy, Debug)]
pub struct Poseidon2VmIoCols<T> {
    pub is_opcode: T,
    pub is_direct: T,
    pub pc: T,
    pub start_timestamp: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub cmp: T,
}

/// Auxiliary columns for Poseidon2Chip.
/// * `addresses`: addresses where inputs/outputs for Poseidon2 are located
/// * `internal`: auxiliary columns used by Poseidon2Air for interpreting opcode, evaluating indicators, inverse, and explicit computations.
#[derive(Clone, Debug)]
pub struct Poseidon2VmAuxCols<const WIDTH: usize, T> {
    pub dst: T,
    pub lhs: T,
    pub rhs: T,
    pub internal: Poseidon2Cols<WIDTH, T>,
}

impl<const WIDTH: usize, T: Clone> Poseidon2VmCols<WIDTH, T> {
    pub fn get_width(poseidon2_chip: &Poseidon2VmAir<WIDTH, T>) -> usize {
        Poseidon2VmIoCols::<T>::get_width()
            + Poseidon2VmAuxCols::<WIDTH, T>::get_width(&poseidon2_chip.inner)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }

    pub fn from_slice(
        slice: &[T],
        index_map: &Poseidon2ColsIndexMap<WIDTH>,
    ) -> Poseidon2VmCols<WIDTH, T> {
        let io_width = Poseidon2VmIoCols::<T>::get_width();
        Self {
            io: Poseidon2VmIoCols::<T>::from_slice(&slice[..io_width]),
            aux: Poseidon2VmAuxCols::<WIDTH, T>::from_slice(&slice[io_width..], index_map),
        }
    }
}

impl<const WIDTH: usize, T: Field> Poseidon2VmCols<WIDTH, T> {
    /// Blank row with all zero input (poseidon2 internal hash values are nonzero)
    /// and `is_alloc` set to 0.
    pub fn blank_row(poseidon2_air: &Poseidon2Air<WIDTH, T>) -> Self {
        Self {
            io: Poseidon2VmIoCols::<T>::blank_row(),
            aux: Poseidon2VmAuxCols::<WIDTH, T>::blank_row(poseidon2_air),
        }
    }
}

impl<T: Clone> Poseidon2VmIoCols<T> {
    pub fn get_width() -> usize {
        10
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.is_opcode.clone(),
            self.is_direct.clone(),
            self.pc.clone(),
            self.start_timestamp.clone(),
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
            is_opcode: slice[0].clone(),
            is_direct: slice[1].clone(),
            pc: slice[2].clone(),
            start_timestamp: slice[3].clone(),
            a: slice[4].clone(),
            b: slice[5].clone(),
            c: slice[6].clone(),
            d: slice[7].clone(),
            e: slice[8].clone(),
            cmp: slice[9].clone(),
        }
    }
}
impl<T: Field> Poseidon2VmIoCols<T> {
    pub fn blank_row() -> Self {
        Self {
            is_opcode: T::zero(),
            is_direct: T::zero(),
            pc: T::zero(),
            start_timestamp: T::zero(),
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::one(),
            e: T::zero(),
            cmp: T::zero(),
        }
    }

    pub fn direct_io_cols() -> Self {
        Self {
            is_opcode: T::zero(),
            is_direct: T::one(),
            pc: T::zero(),
            start_timestamp: T::zero(),
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::one(),
            e: T::zero(),
            cmp: T::zero(),
        }
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2VmAuxCols<WIDTH, T> {
    pub fn get_width(air: &Poseidon2Air<WIDTH, T>) -> usize {
        3 + Poseidon2Cols::<WIDTH, T>::get_width(air)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.dst.clone(), self.lhs.clone(), self.rhs.clone()];
        result.extend(self.internal.flatten());
        result
    }

    pub fn from_slice(slice: &[T], index_map: &Poseidon2ColsIndexMap<WIDTH>) -> Self {
        Self {
            dst: slice[0].clone(),
            lhs: slice[1].clone(),
            rhs: slice[2].clone(),
            internal: Poseidon2Cols::from_slice(&slice[3..], index_map),
        }
    }
}
impl<const WIDTH: usize, T: Field> Poseidon2VmAuxCols<WIDTH, T> {
    pub fn blank_row(air: &Poseidon2Air<WIDTH, T>) -> Self {
        Self {
            dst: T::default(),
            lhs: T::default(),
            rhs: T::default(),
            internal: Poseidon2Cols::blank_row(air),
        }
    }
}
