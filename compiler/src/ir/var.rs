use p3_field::PrimeField;

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

#[derive(Debug, Clone, Copy)]
pub struct MemIndex<N: PrimeField> {
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
