//! Defines auxiliary columns for memory operations: `MemoryReadAuxCols`,
//! `MemoryReadWithImmediateAuxCols`, and `MemoryWriteAuxCols`.

use openvm_circuit_primitives::is_less_than::LessThanAuxCols;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};

use crate::system::memory::offline_checker::bridge::AUX_LEN;

// repr(C) is needed to make sure that the compiler does not reorder the fields
// we assume the order of the fields when using borrow or borrow_mut
#[repr(C)]
/// Base structure for auxiliary memory columns.
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct MemoryBaseAuxCols<T> {
    /// The previous timestamps in which the cells were accessed.
    pub(super) prev_timestamp: T,
    /// The auxiliary columns to perform the less than check.
    pub(super) clk_lt_aux: LessThanAuxCols<T, AUX_LEN>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct MemoryWriteAuxCols<T, const N: usize> {
    pub base: MemoryBaseAuxCols<T>,
    pub prev_data: [T; N],
}

impl<const N: usize, T> MemoryWriteAuxCols<T, N> {
    pub fn new(prev_data: [T; N], prev_timestamp: T, lt_aux: LessThanAuxCols<T, AUX_LEN>) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp,
                clk_lt_aux: lt_aux,
            },
            prev_data,
        }
    }
}

impl<const N: usize, T> MemoryWriteAuxCols<T, N> {
    pub fn from_base(base: MemoryBaseAuxCols<T>, prev_data: [T; N]) -> Self {
        Self { base, prev_data }
    }

    pub fn get_base(self) -> MemoryBaseAuxCols<T> {
        self.base
    }
}

impl<const N: usize, F: FieldAlgebra> MemoryWriteAuxCols<F, N> {
    pub const fn disabled() -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp: F::ZERO,
                clk_lt_aux: LessThanAuxCols {
                    lower_decomp: [F::ZERO; AUX_LEN],
                },
            },
            prev_data: [F::ZERO; N],
        }
    }
}

/// The auxiliary columns for a memory read operation with block size `N`.
/// These columns should be automatically managed by the memory controller.
/// To fully constrain a memory read, in addition to these columns,
/// the address space, pointer, and data must be provided.
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct MemoryReadAuxCols<T, const N: usize> {
    pub(super) base: MemoryBaseAuxCols<T>,
}

impl<const N: usize, F: PrimeField32> MemoryReadAuxCols<F, N> {
    pub fn new(prev_timestamp: u32, clk_lt_aux: LessThanAuxCols<F, AUX_LEN>) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp: F::from_canonical_u32(prev_timestamp),
                clk_lt_aux,
            },
        }
    }
}

impl<const N: usize, F: FieldAlgebra + Copy> MemoryReadAuxCols<F, N> {
    pub const fn disabled() -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp: F::ZERO,
                clk_lt_aux: LessThanAuxCols {
                    lower_decomp: [F::ZERO; AUX_LEN],
                },
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct MemoryReadOrImmediateAuxCols<T> {
    pub(super) base: MemoryBaseAuxCols<T>,
    pub(super) is_immediate: T,
    pub(super) is_zero_aux: T,
}

impl<T> MemoryReadOrImmediateAuxCols<T> {
    pub fn new(
        prev_timestamp: T,
        is_immediate: T,
        is_zero_aux: T,
        clk_lt_aux: LessThanAuxCols<T, AUX_LEN>,
    ) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp,
                clk_lt_aux,
            },
            is_immediate,
            is_zero_aux,
        }
    }
}

impl<F: FieldAlgebra + Copy> MemoryReadOrImmediateAuxCols<F> {
    pub const fn disabled() -> Self {
        MemoryReadOrImmediateAuxCols {
            base: MemoryBaseAuxCols {
                prev_timestamp: F::ZERO,
                clk_lt_aux: LessThanAuxCols {
                    lower_decomp: [F::ZERO; AUX_LEN],
                },
            },
            is_immediate: F::ZERO,
            is_zero_aux: F::ZERO,
        }
    }
}
