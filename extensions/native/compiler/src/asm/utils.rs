use ax_stark_backend::p3_field::PrimeField32;

use crate::prelude::{MemIndex, RVar};

/// Represents a memory index triple.
pub enum IndexTriple<F> {
    Var(i32, F, F),
    Const(F, F, F),
}

impl<F: PrimeField32> MemIndex<F> {
    pub fn fp(&self) -> IndexTriple<F> {
        match &self.index {
            RVar::Const(index) => IndexTriple::Const(
                *index,
                F::from_canonical_usize(self.offset),
                F::from_canonical_usize(self.size),
            ),
            RVar::Val(index) => IndexTriple::Var(
                index.fp(),
                F::from_canonical_usize(self.offset),
                F::from_canonical_usize(self.size),
            ),
        }
    }
}

/// A value or a constant.
pub enum ValueOrConst<F, EF> {
    Val(i32),
    ExtVal(i32),
    Const(F),
    ExtConst(EF),
}
