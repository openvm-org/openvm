use crate::ir::{Builder, Config, Ext, Felt, Var};

pub trait CanSelect<C: Config> {
    fn select(builder: &mut Builder<C>, cond: Var<C::N>, a: Self, b: Self) -> Self;
}

impl<C: Config> CanSelect<C> for Var<C::N> {
    fn select(builder: &mut Builder<C>, cond: Var<C::N>, a: Self, b: Self) -> Self {
        builder.select_v(cond, a, b)
    }
}

impl<C: Config> CanSelect<C> for Felt<C::F> {
    fn select(builder: &mut Builder<C>, cond: Var<C::N>, a: Self, b: Self) -> Self {
        builder.select_f(cond, a, b)
    }
}

impl<C: Config> CanSelect<C> for Ext<C::F, C::EF> {
    fn select(builder: &mut Builder<C>, cond: Var<C::N>, a: Self, b: Self) -> Self {
        builder.select_ef(cond, a, b)
    }
}
