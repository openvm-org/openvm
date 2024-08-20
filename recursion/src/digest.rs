use afs_compiler::{
    ir::{Array, Builder, Config, Felt, FromConstant, MemIndex, Ptr, Usize, Var, Variable},
    prelude::MemVariable,
};

#[derive(Clone)]
pub enum DigestVal<C: Config> {
    F(Vec<C::F>),
    N(Vec<C::N>),
}

impl<C: Config> DigestVal<C> {
    pub fn len(&self) -> usize {
        match self {
            DigestVal::F(v) => v.len(),
            DigestVal::N(v) => v.len(),
        }
    }
}

#[derive(Clone)]
pub enum DigestVariable<C: Config> {
    Felt(Array<C, Felt<C::F>>),
    Var(Array<C, Var<C::N>>),
}

impl<C: Config> Variable<C> for DigestVariable<C> {
    type Expression = Self;

    fn uninit(builder: &mut Builder<C>) -> Self {
        Self::Felt(builder.uninit())
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        match (self, src) {
            (Self::Felt(lhs), Self::Felt(rhs)) => builder.assign(lhs, rhs),
            (Self::Var(lhs), Self::Var(rhs)) => builder.assign(lhs, rhs),
            _ => panic!("Assignment types mismatch"),
        }
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        match (lhs.into(), rhs.into()) {
            (Self::Felt(lhs), Self::Felt(rhs)) => builder.assert_eq::<Array<C, _>>(lhs, rhs),
            (Self::Var(lhs), Self::Var(rhs)) => builder.assert_eq::<Array<C, _>>(lhs, rhs),
            _ => panic!("Assertion types mismatch"),
        }
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        match (lhs.into(), rhs.into()) {
            (Self::Felt(lhs), Self::Felt(rhs)) => builder.assert_ne::<Array<C, _>>(lhs, rhs),
            (Self::Var(lhs), Self::Var(rhs)) => builder.assert_ne::<Array<C, _>>(lhs, rhs),
            _ => panic!("Assertion types mismatch"),
        }
    }
}

impl<C: Config> MemVariable<C> for DigestVariable<C> {
    fn size_of() -> usize {
        Array::<C, Felt<C::F>>::size_of()
    }

    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        match self {
            DigestVariable::Felt(array) => array.load(ptr, index, builder),
            DigestVariable::Var(array) => array.load(ptr, index, builder),
        }
    }

    fn store(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        match self {
            DigestVariable::Felt(array) => array.store(ptr, index, builder),
            DigestVariable::Var(array) => array.store(ptr, index, builder),
        }
    }
}

impl<C: Config> FromConstant<C> for DigestVariable<C> {
    type Constant = DigestVal<C>;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self {
        match value {
            DigestVal::F(value) => {
                let mut array = builder.array(value.len());
                for (i, val) in value.into_iter().enumerate() {
                    let val = Felt::constant(val, builder);
                    builder.set(&mut array, i, val);
                }
                Self::Felt(array)
            }
            DigestVal::N(value) => {
                let mut array = builder.array(value.len());
                for (i, val) in value.into_iter().enumerate() {
                    let val = Var::constant(val, builder);
                    builder.set(&mut array, i, val);
                }
                Self::Var(array)
            }
        }
    }
}

impl<C: Config> DigestVariable<C> {
    pub fn len(&self) -> Usize<C::N> {
        match self {
            DigestVariable::Felt(array) => array.len(),
            DigestVariable::Var(array) => array.len(),
        }
    }
}
