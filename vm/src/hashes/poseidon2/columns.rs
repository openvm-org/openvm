use std::array;

use p3_field::Field;
use poseidon2_air::poseidon2::columns::Poseidon2Cols;

use super::air::Poseidon2VmAir;
use crate::{
    hashes::poseidon2::{CHUNK, WIDTH},
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

/// Columns for Poseidon2Vm AIR.
#[derive(Clone, Debug)]
pub struct Poseidon2VmCols<T> {
    pub io: Poseidon2VmIoCols<T>,
    pub aux: Poseidon2VmAuxCols<T>,
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
    pub timestamp: T,
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
pub struct Poseidon2VmAuxCols<T> {
    pub dst_ptr: T,
    pub lhs_ptr: T,
    pub rhs_ptr: T,
    pub internal: Poseidon2Cols<WIDTH, T>,
    pub ptr_aux_cols: [MemoryReadAuxCols<1, T>; 3],
    pub input_aux_cols: [MemoryReadAuxCols<CHUNK, T>; 2],
    pub output_aux_cols: [MemoryWriteAuxCols<CHUNK, T>; 2],
}

impl<T: Clone> Poseidon2VmCols<T> {
    pub fn width(p2_air: &Poseidon2VmAir<T>) -> usize {
        Poseidon2VmIoCols::<T>::get_width() + Poseidon2VmAuxCols::<T>::width(p2_air)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }

    pub fn from_slice<F: Clone>(slice: &[T], air: &Poseidon2VmAir<F>) -> Poseidon2VmCols<T> {
        let io_width = Poseidon2VmIoCols::<T>::get_width();
        Self {
            io: Poseidon2VmIoCols::<T>::from_slice(&slice[..io_width]),
            aux: Poseidon2VmAuxCols::<T>::from_slice(&slice[io_width..], air),
        }
    }
}

impl<F: Field> Poseidon2VmCols<F> {
    /// Blank row with all zero input (poseidon2 internal hash values are nonzero)
    /// and `is_alloc` set to 0.
    ///
    /// Due to how memory timestamps are currently managed, even blank rows must have consistent timestamps.
    ///
    /// Warning: the aux memory columns have capacity reserved but are not initialized.
    pub fn blank_row(air: &Poseidon2VmAir<F>) -> Self {
        Self {
            io: Poseidon2VmIoCols::<F>::blank_row(),
            aux: Poseidon2VmAuxCols::<F>::blank_row(air),
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
            self.timestamp.clone(),
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
            timestamp: slice[3].clone(),
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
            timestamp: T::zero(),
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::one(),
            e: T::one(),
            cmp: T::zero(),
        }
    }

    pub fn direct_io_cols(timestamp: T) -> Self {
        Self {
            is_opcode: T::zero(),
            is_direct: T::one(),
            pc: T::zero(),
            timestamp,
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::one(),
            e: T::one(),
            cmp: T::zero(),
        }
    }
}

impl<T: Clone> Poseidon2VmAuxCols<T> {
    pub fn width(air: &Poseidon2VmAir<T>) -> usize {
        3 + Poseidon2Cols::<WIDTH, T>::get_width(&air.inner)
            + 3 * MemoryReadAuxCols::<1, T>::width()
            + 2 * MemoryReadAuxCols::<CHUNK, T>::width()
            + 2 * MemoryWriteAuxCols::<CHUNK, T>::width()
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.dst_ptr.clone(),
            self.lhs_ptr.clone(),
            self.rhs_ptr.clone(),
        ];
        result.extend(self.internal.flatten());
        result.extend(
            self.ptr_aux_cols
                .iter()
                .flat_map(|col| col.clone().flatten()),
        );
        result.extend(
            self.input_aux_cols
                .iter()
                .flat_map(|col| col.clone().flatten()),
        );
        result.extend(
            self.output_aux_cols
                .iter()
                .flat_map(|col| col.clone().flatten()),
        );
        result
    }

    pub fn from_slice<F: Clone>(slc: &[T], air: &Poseidon2VmAir<F>) -> Self {
        let p2_index_map = Poseidon2Cols::index_map(&air.inner);

        let dst = slc[0].clone();
        let lhs = slc[1].clone();
        let rhs = slc[2].clone();

        let mut start = 3;
        let mut end = start + Poseidon2Cols::<WIDTH, T>::get_width(&air.inner);
        let internal = Poseidon2Cols::from_slice(&slc[start..end], &p2_index_map);

        let ptr_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryReadAuxCols::<1, T>::width();
            MemoryReadAuxCols::from_slice(&slc[start..end])
        });
        let input_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryReadAuxCols::<CHUNK, T>::width();
            MemoryReadAuxCols::from_slice(&slc[start..end])
        });
        let output_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryWriteAuxCols::<CHUNK, T>::width();
            MemoryWriteAuxCols::from_slice(&slc[start..end])
        });

        Self {
            dst_ptr: dst,
            lhs_ptr: lhs,
            rhs_ptr: rhs,
            internal,
            ptr_aux_cols,
            input_aux_cols,
            output_aux_cols,
        }
    }
}

impl<T: Field> Poseidon2VmAuxCols<T> {
    pub fn blank_row(air: &Poseidon2VmAir<T>) -> Self {
        Self {
            dst_ptr: T::default(),
            lhs_ptr: T::default(),
            rhs_ptr: T::default(),
            internal: Poseidon2Cols::blank_row(&air.inner),
            ptr_aux_cols: array::from_fn(|_| MemoryReadAuxCols::disabled()),
            input_aux_cols: array::from_fn(|_| MemoryReadAuxCols::disabled()),
            output_aux_cols: array::from_fn(|_| MemoryWriteAuxCols::disabled()),
        }
    }
}
