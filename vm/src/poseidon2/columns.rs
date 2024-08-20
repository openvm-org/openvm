use p3_field::Field;
use poseidon2_air::poseidon2::{columns::Poseidon2Cols, Poseidon2Air};

use super::air::Poseidon2VmAir;
use crate::memory::offline_checker::columns::MemoryOfflineCheckerAuxCols;

/// Columns for Poseidon2Vm AIR.
#[derive(Clone, Debug)]
pub struct Poseidon2VmCols<const WIDTH: usize, const WORD_SIZE: usize, T> {
    pub io: Poseidon2VmIoCols<T>,
    pub aux: Poseidon2VmAuxCols<WIDTH, WORD_SIZE, T>,
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
    pub clk: T,
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
pub struct Poseidon2VmAuxCols<const WIDTH: usize, const WORD_SIZE: usize, T> {
    pub dst: T,
    pub lhs: T,
    pub rhs: T,
    pub internal: Poseidon2Cols<WIDTH, T>,
    // There are 3+2*WIDTH memory accesses
    pub mem_oc_aux_cols: Vec<MemoryOfflineCheckerAuxCols<WORD_SIZE, T>>,
}

impl<const WIDTH: usize, const WORD_SIZE: usize, T: Clone> Poseidon2VmCols<WIDTH, WORD_SIZE, T> {
    pub fn width(p2_air: &Poseidon2VmAir<WIDTH, WORD_SIZE, T>) -> usize {
        Poseidon2VmIoCols::<T>::get_width()
            + Poseidon2VmAuxCols::<WIDTH, WORD_SIZE, T>::width(p2_air)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }

    pub fn from_slice<F: Clone>(
        slice: &[T],
        air: &Poseidon2VmAir<WIDTH, WORD_SIZE, F>,
    ) -> Poseidon2VmCols<WIDTH, WORD_SIZE, T> {
        let io_width = Poseidon2VmIoCols::<T>::get_width();
        Self {
            io: Poseidon2VmIoCols::<T>::from_slice(&slice[..io_width]),
            aux: Poseidon2VmAuxCols::<WIDTH, WORD_SIZE, T>::from_slice(&slice[io_width..], air),
        }
    }
}

impl<const WIDTH: usize, const WORD_SIZE: usize, F: Field> Poseidon2VmCols<WIDTH, WORD_SIZE, F> {
    /// Blank row with all zero input (poseidon2 internal hash values are nonzero)
    /// and `is_alloc` set to 0.
    ///
    /// Due to how memory timestamps are currently managed, even blank rows must have consistent timestamps.
    ///
    /// Warning: the aux memory columns have capacity reserved but are not initialized.
    pub fn blank_row(poseidon2_air: &Poseidon2Air<WIDTH, F>, timestamp: F) -> Self {
        Self {
            io: Poseidon2VmIoCols::<F>::blank_row(timestamp),
            aux: Poseidon2VmAuxCols::<WIDTH, WORD_SIZE, F>::blank_row(poseidon2_air),
        }
    }
}

impl<T: Clone> Poseidon2VmIoCols<T> {
    pub fn get_width() -> usize {
        9
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.is_opcode.clone(),
            self.is_direct.clone(),
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
            is_opcode: slice[0].clone(),
            is_direct: slice[1].clone(),
            clk: slice[2].clone(),
            a: slice[3].clone(),
            b: slice[4].clone(),
            c: slice[5].clone(),
            d: slice[6].clone(),
            e: slice[7].clone(),
            cmp: slice[8].clone(),
        }
    }
}
impl<T: Field> Poseidon2VmIoCols<T> {
    pub fn blank_row(timestamp: T) -> Self {
        Self {
            is_opcode: T::zero(),
            is_direct: T::zero(),
            clk: timestamp,
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::one(),
            e: T::zero(),
            cmp: T::zero(),
        }
    }

    pub fn direct_io_cols(timestamp: T) -> Self {
        Self {
            is_opcode: T::zero(),
            is_direct: T::one(),
            clk: timestamp,
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::one(),
            e: T::zero(),
            cmp: T::zero(),
        }
    }
}

impl<const WIDTH: usize, const WORD_SIZE: usize, T: Clone> Poseidon2VmAuxCols<WIDTH, WORD_SIZE, T> {
    pub fn width(air: &Poseidon2VmAir<WIDTH, WORD_SIZE, T>) -> usize {
        3 + Poseidon2Cols::<WIDTH, T>::get_width(&air.inner)
            + (3 + 2 * WIDTH) * MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::width(&air.mem_oc)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.dst.clone(), self.lhs.clone(), self.rhs.clone()];
        result.extend(self.internal.flatten());
        result.extend(
            self.mem_oc_aux_cols
                .iter()
                .flat_map(|col| col.clone().flatten()),
        );
        result
    }

    pub fn from_slice<F: Clone>(slc: &[T], air: &Poseidon2VmAir<WIDTH, WORD_SIZE, F>) -> Self {
        let p2_index_map = Poseidon2Cols::index_map(&air.inner);

        let dst = slc[0].clone();
        let lhs = slc[1].clone();
        let rhs = slc[2].clone();

        let mut start = 3;
        let mut end = start + Poseidon2Cols::<WIDTH, T>::get_width(&air.inner);
        let internal = Poseidon2Cols::from_slice(&slc[start..end], &p2_index_map);

        let mut mem_oc_aux_cols = Vec::with_capacity(3 + 2 * WIDTH);
        for _ in 0..3 + 2 * WIDTH {
            start = end;
            end += MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::width(&air.mem_oc);
            mem_oc_aux_cols.push(MemoryOfflineCheckerAuxCols::from_slice(&slc[start..end]));
        }

        Self {
            dst,
            lhs,
            rhs,
            internal,
            mem_oc_aux_cols,
        }
    }
}
impl<const WIDTH: usize, const WORD_SIZE: usize, T: Field> Poseidon2VmAuxCols<WIDTH, WORD_SIZE, T> {
    pub fn blank_row(air: &Poseidon2Air<WIDTH, T>) -> Self {
        Self {
            dst: T::default(),
            lhs: T::default(),
            rhs: T::default(),
            internal: Poseidon2Cols::blank_row(air),
            mem_oc_aux_cols: Vec::with_capacity(3 + 2 * WIDTH),
        }
    }
}
