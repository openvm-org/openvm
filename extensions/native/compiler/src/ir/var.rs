use std::array;

use itertools::izip;
use serde::{Deserialize, Serialize};

use super::{Builder, Config, Ptr, RVar};

pub trait Variable<C: Config>: Clone {
    type Expression: From<Self>;

    fn uninit(builder: &mut Builder<C>) -> Self;

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>);

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    );

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    );

    fn eval(builder: &mut Builder<C>, expr: impl Into<Self::Expression>) -> Self {
        let dst: Self = builder.uninit();
        dst.assign(expr.into(), builder);
        dst
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemIndex<N> {
    pub index: RVar<N>,
    pub offset: usize,
    pub size: usize,
}

pub trait MemVariable<C: Config>: Variable<C> {
    fn size_of() -> usize;
    /// Loads the variable from the heap.
    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>);
    /// Stores the variable to the heap.
    fn store(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>);
}

pub trait FromConstant<C: Config> {
    type Constant;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self;
}

impl<C: Config, T: Variable<C>, const N: usize> Variable<C> for [T; N] {
    type Expression = [T; N];

    fn uninit(builder: &mut Builder<C>) -> Self {
        array::from_fn(|_| T::uninit(builder))
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        self.iter()
            .zip(src)
            .for_each(|(d, s)| d.assign(s.into(), builder));
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        izip!(lhs.into(), rhs.into()).for_each(|(l, r)| T::assert_eq(l, r, builder));
    }

    fn assert_ne(
        _lhs: impl Into<Self::Expression>,
        _rhs: impl Into<Self::Expression>,
        _builder: &mut Builder<C>,
    ) {
        unimplemented!("assert_ne cannot be implemented for slices")
    }
}

impl<C: Config, T: MemVariable<C>, const N: usize> MemVariable<C> for [T; N] {
    fn size_of() -> usize {
        N * T::size_of()
    }

    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        for (i, v) in self.iter().enumerate() {
            let mut v_idx = index;
            v_idx.offset += i * T::size_of();
            v.load(ptr, v_idx, builder);
        }
    }

    fn store(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        for (i, v) in self.iter().enumerate() {
            let mut v_idx = index;
            v_idx.offset += i * T::size_of();
            v.store(ptr, v_idx, builder);
        }
    }
}
