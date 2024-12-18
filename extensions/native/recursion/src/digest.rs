use openvm_native_compiler::{
    ir::{Array, Builder, Config, Ext, Felt, FromConstant, MemIndex, Ptr, Usize, Var, Variable},
    prelude::MemVariable,
};

use crate::{outer_poseidon2::Poseidon2CircuitBuilder, vars::OuterDigestVariable};

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
    pub fn is_empty(&self) -> bool {
        match self {
            DigestVal::F(v) => v.is_empty(),
            DigestVal::N(v) => v.is_empty(),
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
                let array = builder.array(value.len());
                for (i, val) in value.into_iter().enumerate() {
                    let val = Felt::constant(val, builder);
                    builder.set(&array, i, val);
                }
                Self::Felt(array)
            }
            DigestVal::N(value) => {
                let array = builder.array(value.len());
                for (i, val) in value.into_iter().enumerate() {
                    let val = Var::constant(val, builder);
                    builder.set(&array, i, val);
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
    /// Cast to OuterDigestVariable. This should only be used in static mode.
    pub fn into_outer_digest(self) -> OuterDigestVariable<C> {
        match self {
            DigestVariable::Var(array) => array.vec().try_into().unwrap(),
            DigestVariable::Felt(_) => panic!("Trying to get Var array from Felt array"),
        }
    }
    /// Cast to an inner digest. This should only be used in dynamic mode.
    pub fn into_inner_digest(self) -> Array<C, Felt<C::F>> {
        match self {
            DigestVariable::Felt(array) => array,
            DigestVariable::Var(_) => panic!("Trying to get Felt array from Var array"),
        }
    }
}

impl<C: Config> From<Array<C, Felt<C::F>>> for DigestVariable<C> {
    fn from(value: Array<C, Felt<C::F>>) -> Self {
        Self::Felt(value)
    }
}

impl<C: Config> From<Array<C, Var<C::N>>> for DigestVariable<C> {
    fn from(value: Array<C, Var<C::N>>) -> Self {
        Self::Var(value)
    }
}

pub trait CanPoseidon2Digest<C: Config> {
    fn p2_digest(&self, builder: &mut Builder<C>) -> DigestVariable<C>;
}

impl<C: Config> CanPoseidon2Digest<C> for Array<C, Array<C, Felt<C::F>>> {
    fn p2_digest(&self, builder: &mut Builder<C>) -> DigestVariable<C> {
        if builder.flags.static_only {
            let digest_vec = builder.p2_hash(&flatten_fixed(self));
            DigestVariable::Var(builder.vec(digest_vec.to_vec()))
        } else {
            DigestVariable::Felt(builder.poseidon2_hash_x(self))
        }
    }
}

impl<C: Config> CanPoseidon2Digest<C> for Array<C, Array<C, Ext<C::F, C::EF>>> {
    fn p2_digest(&self, builder: &mut Builder<C>) -> DigestVariable<C> {
        if builder.flags.static_only {
            let flat_felts: Vec<_> = flatten_fixed(self)
                .into_iter()
                .flat_map(|ext| builder.ext2felt_circuit(ext).to_vec())
                .collect();
            let digest_vec = builder.p2_hash(&flat_felts);
            DigestVariable::Var(builder.vec(digest_vec.to_vec()))
        } else {
            DigestVariable::Felt(builder.poseidon2_hash_ext(self))
        }
    }
}

fn flatten_fixed<C: Config, V: MemVariable<C>>(arr: &Array<C, Array<C, V>>) -> Vec<V> {
    arr.vec()
        .into_iter()
        .flat_map(|felt_arr| felt_arr.vec())
        .collect()
}
