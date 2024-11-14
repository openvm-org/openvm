use alloc::{format, rc::Rc};
use core::marker::PhantomData;
use std::{cell::RefCell, collections::HashMap, hash::Hash};

use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField};
use serde::{Deserialize, Serialize};

use super::{
    utils::prime_field_to_usize, Builder, Config, DslIr, ExtConst, FromConstant, MemIndex,
    MemVariable, Ptr, RVar, SymbolicExt, SymbolicFelt, SymbolicVar, Variable,
};

/// A variable that represents a native field element.
///
/// Used for counters, simple loops, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var<N>(pub u32, pub PhantomData<N>);

/// A variable that represents an emulated field element.
///
/// Used to do field arithmetic for recursive verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Felt<F>(pub u32, pub PhantomData<F>);

/// A variable that represents an emulated extension field element.
///
/// Used to do extension field arithmetic for recursive verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ext<F, EF>(pub u32, pub PhantomData<(F, EF)>);

/// A variable that represents either a constant or variable counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Usize<N: PrimeField> {
    /// A compile time variable. It should only be used in static mode.
    Const(Rc<RefCell<N>>),
    Var(Var<N>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Witness<C: Config> {
    pub vars: Vec<C::N>,
    pub felts: Vec<C::F>,
    pub exts: Vec<C::EF>,
    pub vkey_hash: C::N,
    pub commited_values_digest: C::N,
}

impl<C: Config> Witness<C> {
    pub fn size(&self) -> usize {
        self.vars.len() + self.felts.len() + self.exts.len() + 2
    }

    pub fn write_vkey_hash(&mut self, vkey_hash: C::N) {
        self.vars.push(vkey_hash);
        self.vkey_hash = vkey_hash;
    }

    pub fn write_commited_values_digest(&mut self, commited_values_digest: C::N) {
        self.vars.push(commited_values_digest);
        self.commited_values_digest = commited_values_digest
    }
}

impl<N: PrimeField> Usize<N> {
    pub fn is_const(&self) -> bool {
        match self {
            Usize::Const(_) => true,
            Usize::Var(_) => false,
        }
    }

    pub fn value(&self) -> usize {
        match self {
            Usize::Const(c) => prime_field_to_usize(*c.borrow()),
            Usize::Var(_) => panic!("Cannot get the value of a variable"),
        }
    }

    pub fn get_var(&self) -> Var<N> {
        match self {
            Usize::Const(_) => panic!("Cannot get the variable of a constant"),
            Usize::Var(v) => *v,
        }
    }

    pub fn from_field(value: N) -> Self {
        Usize::Const(Rc::new(RefCell::new(value)))
    }

    pub fn materialize<C: Config<N = N>>(&self, builder: &mut Builder<C>) -> Var<C::N> {
        match self {
            Usize::Const(c) => builder.eval(*c.borrow()),
            Usize::Var(v) => *v,
        }
    }
}

impl<N: PrimeField> From<Var<N>> for Usize<N> {
    fn from(v: Var<N>) -> Self {
        Usize::Var(v)
    }
}

impl<N: PrimeField> From<usize> for Usize<N> {
    fn from(c: usize) -> Self {
        Usize::Const(Rc::new(RefCell::new(N::from_canonical_usize(c))))
    }
}

impl<N: PrimeField> From<Usize<N>> for RVar<N> {
    fn from(u: Usize<N>) -> Self {
        match u {
            Usize::Const(c) => RVar::Const(*c.borrow()),
            Usize::Var(v) => RVar::Val(v),
        }
    }
}

impl<N> Var<N> {
    pub const fn new(id: u32) -> Self {
        Self(id, PhantomData)
    }

    pub fn id(&self) -> String {
        format!("var{}", self.0)
    }

    pub fn loc(&self) -> String {
        self.0.to_string()
    }
}

impl<F> Felt<F> {
    pub const fn new(id: u32) -> Self {
        Self(id, PhantomData)
    }

    pub fn id(&self) -> String {
        format!("felt{}", self.0)
    }

    pub fn loc(&self) -> String {
        self.0.to_string()
    }

    pub fn inverse(&self) -> SymbolicFelt<F>
    where
        F: Field,
    {
        SymbolicFelt::<F>::ONE / *self
    }
}

impl<F, EF> Ext<F, EF> {
    pub const fn new(id: u32) -> Self {
        Self(id, PhantomData)
    }

    pub fn id(&self) -> String {
        format!("ext{}", self.0)
    }

    pub fn loc(&self) -> String {
        self.0.to_string()
    }

    pub fn inverse(&self) -> SymbolicExt<F, EF>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        SymbolicExt::<F, EF>::ONE / *self
    }
}

impl<C: Config> Variable<C> for Usize<C::N> {
    type Expression = SymbolicVar<C::N>;

    fn uninit(builder: &mut Builder<C>) -> Self {
        builder.uninit::<Var<C::N>>().into()
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        match self {
            Usize::Const(c) => {
                *c.borrow_mut() = if let SymbolicVar::Const(c, _) = src {
                    if !builder.is_sub_builder {
                        c
                    } else {
                        panic!("cannot assign Usize::Const inside a closure")
                    }
                } else {
                    panic!("cannot assign Usize::Const with a variable")
                }
            }
            Usize::Var(v) => {
                builder.assign(v, src);
            }
        }
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        Var::<C::N>::assert_eq(lhs, rhs, builder);
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        Var::<C::N>::assert_ne(lhs, rhs, builder);
    }

    fn eval(builder: &mut Builder<C>, expr: impl Into<Self::Expression>) -> Self {
        let expr = expr.into();
        match expr {
            SymbolicVar::Const(c, _) => {
                // Usize::Const should only be used in static mode.
                if builder.flags.static_only {
                    Usize::from_field(c)
                } else {
                    Usize::Var(builder.eval(c))
                }
            }
            _ => Usize::Var(builder.eval(expr)),
        }
    }
}

impl<N: Field> Var<N> {
    fn assign_with_cache<C: Config<N = N>>(
        &self,
        src: SymbolicVar<N>,
        builder: &mut Builder<C>,
        cache: &mut HashMap<SymbolicVar<N>, Self>,
    ) {
        if let Some(v) = cache.get(&src) {
            builder.operations.push(DslIr::AddVI(*self, *v, C::N::ZERO));
            return;
        }
        match src {
            SymbolicVar::Const(c, _) => {
                builder.operations.push(DslIr::ImmV(*self, c));
            }
            SymbolicVar::Val(v, _) => {
                builder.operations.push(DslIr::AddVI(*self, v, C::N::ZERO));
            }
            SymbolicVar::Add(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _)) => {
                    let sum = *lhs + *rhs;
                    builder.operations.push(DslIr::ImmV(*self, sum));
                }
                (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _)) => {
                    builder.operations.push(DslIr::AddVI(*self, *rhs, *lhs));
                }
                (SymbolicVar::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign(rhs.clone(), builder);
                    builder.push(DslIr::AddVI(*self, rhs_value, *lhs));
                }
                (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _)) => {
                    builder.push(DslIr::AddVI(*self, *lhs, *rhs));
                }
                (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _)) => {
                    builder.push(DslIr::AddV(*self, *lhs, *rhs));
                }
                (SymbolicVar::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign(rhs.clone(), builder);
                    builder.push(DslIr::AddV(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicVar::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign(lhs.clone(), builder);
                    builder.push(DslIr::AddVI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicVar::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign(lhs.clone(), builder);
                    builder.push(DslIr::AddV(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddV(*self, lhs_value, rhs_value));
                }
            },
            SymbolicVar::Mul(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _)) => {
                    let product = *lhs * *rhs;
                    builder.push(DslIr::ImmV(*self, product));
                }
                (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _)) => {
                    builder.push(DslIr::MulVI(*self, *rhs, *lhs));
                }
                (SymbolicVar::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulVI(*self, rhs_value, *lhs));
                }
                (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _)) => {
                    builder.push(DslIr::MulVI(*self, *lhs, *rhs));
                }
                (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _)) => {
                    builder.push(DslIr::MulV(*self, *lhs, *rhs));
                }
                (SymbolicVar::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulV(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicVar::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::MulVI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicVar::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::MulV(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulV(*self, lhs_value, rhs_value));
                }
            },
            SymbolicVar::Sub(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _)) => {
                    let difference = *lhs - *rhs;
                    builder.push(DslIr::ImmV(*self, difference));
                }
                (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _)) => {
                    builder.push(DslIr::SubVIN(*self, *lhs, *rhs));
                }
                (SymbolicVar::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubVIN(*self, *lhs, rhs_value));
                }
                (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _)) => {
                    builder.push(DslIr::SubVI(*self, *lhs, *rhs));
                }
                (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _)) => {
                    builder.push(DslIr::SubV(*self, *lhs, *rhs));
                }
                (SymbolicVar::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubV(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicVar::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::SubVI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicVar::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::SubV(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubV(*self, lhs_value, rhs_value));
                }
            },
            SymbolicVar::Neg(operand, _) => match &*operand {
                SymbolicVar::Const(operand, _) => {
                    let negated = -*operand;
                    builder.push(DslIr::ImmV(*self, negated));
                }
                SymbolicVar::Val(operand, _) => {
                    builder.push(DslIr::SubVIN(*self, C::N::ZERO, *operand));
                }
                operand => {
                    let operand_value = Self::uninit(builder);
                    operand_value.assign_with_cache(operand.clone(), builder, cache);
                    cache.insert(operand.clone(), operand_value);
                    builder.push(DslIr::SubVIN(*self, C::N::ZERO, operand_value));
                }
            },
        }
    }
}

impl<C: Config> Variable<C> for Var<C::N> {
    type Expression = SymbolicVar<C::N>;

    fn uninit(builder: &mut Builder<C>) -> Self {
        builder.var_count += 1;
        Var(builder.var_count, PhantomData)
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        self.assign_with_cache(src, builder, &mut HashMap::new());
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _)) => {
                assert_eq!(lhs, rhs, "Assertion failed at compile time");
            }
            (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqVI(rhs, lhs));
            }
            (SymbolicVar::Const(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqVI(rhs_value, lhs));
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqVI(lhs, rhs));
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqV(lhs, rhs));
            }
            (SymbolicVar::Val(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqV(lhs, rhs_value));
            }
            (lhs, rhs) => {
                let lhs_value = Self::uninit(builder);
                lhs_value.assign(lhs, builder);
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqV(lhs_value, rhs_value));
            }
        }
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _)) => {
                assert_ne!(lhs, rhs, "Assertion failed at compile time");
            }
            (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeVI(rhs, lhs));
            }
            (SymbolicVar::Const(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeVI(rhs_value, lhs));
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeVI(lhs, rhs));
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeV(lhs, rhs));
            }
            (SymbolicVar::Val(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeV(lhs, rhs_value));
            }
            (lhs, rhs) => {
                let lhs_value = Self::uninit(builder);
                lhs_value.assign(lhs, builder);
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeV(lhs_value, rhs_value));
            }
        }
    }
}

impl<C: Config> MemVariable<C> for Var<C::N> {
    fn size_of() -> usize {
        1
    }

    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        builder.push(DslIr::LoadV(*self, ptr, index));
    }

    fn store(&self, ptr: Ptr<<C as Config>::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        builder.push(DslIr::StoreV(*self, ptr, index));
    }
}

impl<C: Config> MemVariable<C> for Usize<C::N> {
    fn size_of() -> usize {
        1
    }

    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        match self {
            Usize::Const(_) => {
                panic!("Usize::Const should not be loaded");
            }
            Usize::Var(v) => {
                builder.push(DslIr::LoadV(*v, ptr, index));
            }
        }
    }

    fn store(&self, ptr: Ptr<<C as Config>::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        match self {
            Usize::Const(_) => {
                panic!("Usize::Const should not be stored");
            }
            Usize::Var(v) => {
                builder.push(DslIr::StoreV(*v, ptr, index));
            }
        }
    }
}

impl<F: Field> Felt<F> {
    fn assign_with_cache<C: Config<F = F>>(
        &self,
        src: SymbolicFelt<F>,
        builder: &mut Builder<C>,
        cache: &mut HashMap<SymbolicFelt<F>, Self>,
    ) {
        if let Some(v) = cache.get(&src) {
            builder.operations.push(DslIr::AddFI(*self, *v, C::F::ZERO));
            return;
        }
        match src {
            SymbolicFelt::Const(c, _) => {
                builder.operations.push(DslIr::ImmF(*self, c));
            }
            SymbolicFelt::Val(v, _) => {
                builder.operations.push(DslIr::AddFI(*self, v, C::F::ZERO));
            }
            SymbolicFelt::Add(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    let sum = *lhs + *rhs;
                    builder.operations.push(DslIr::ImmF(*self, sum));
                }
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.operations.push(DslIr::AddFI(*self, *rhs, *lhs));
                }
                (SymbolicFelt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddFI(*self, rhs_value, *lhs));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    builder.push(DslIr::AddFI(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::AddF(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddF(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicFelt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::AddFI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicFelt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::AddF(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddF(*self, lhs_value, rhs_value));
                }
            },
            SymbolicFelt::Mul(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    let product = *lhs * *rhs;
                    builder.push(DslIr::ImmF(*self, product));
                }
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::MulFI(*self, *rhs, *lhs));
                }
                (SymbolicFelt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulFI(*self, rhs_value, *lhs));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    builder.push(DslIr::MulFI(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::MulF(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulF(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicFelt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::MulFI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicFelt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::MulF(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulF(*self, lhs_value, rhs_value));
                }
            },
            SymbolicFelt::Sub(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    let difference = *lhs - *rhs;
                    builder.push(DslIr::ImmF(*self, difference));
                }
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::SubFIN(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubFIN(*self, *lhs, rhs_value));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    builder.push(DslIr::SubFI(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::SubF(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubF(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicFelt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::SubFI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicFelt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::SubF(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubF(*self, lhs_value, rhs_value));
                }
            },
            SymbolicFelt::Div(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    let quotient = *lhs / *rhs;
                    builder.push(DslIr::ImmF(*self, quotient));
                }
                (SymbolicFelt::Const(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::DivFIN(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::DivFIN(*self, *lhs, rhs_value));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                    builder.push(DslIr::DivFI(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                    builder.push(DslIr::DivF(*self, *lhs, *rhs));
                }
                (SymbolicFelt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::DivF(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicFelt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::DivFI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicFelt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::DivF(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_cache(lhs.clone(), builder, cache);
                    cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_cache(rhs.clone(), builder, cache);
                    cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::DivF(*self, lhs_value, rhs_value));
                }
            },
            SymbolicFelt::Neg(operand, _) => match &*operand {
                SymbolicFelt::Const(operand, _) => {
                    let negated = -*operand;
                    builder.push(DslIr::ImmF(*self, negated));
                }
                SymbolicFelt::Val(operand, _) => {
                    builder.push(DslIr::SubFIN(*self, C::F::ZERO, *operand));
                }
                operand => {
                    let operand_value = Self::uninit(builder);
                    operand_value.assign_with_cache(operand.clone(), builder, cache);
                    cache.insert(operand.clone(), operand_value);
                    builder.push(DslIr::SubFIN(*self, C::F::ZERO, operand_value));
                }
            },
        }
    }
}

impl<C: Config> Variable<C> for Felt<C::F> {
    type Expression = SymbolicFelt<C::F>;

    fn uninit(builder: &mut Builder<C>) -> Self {
        builder.felt_count += 1;
        Felt(builder.felt_count, PhantomData)
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        self.assign_with_cache(src, builder, &mut HashMap::new());
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (SymbolicFelt::Const(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                assert_eq!(lhs, rhs, "Assertion failed at compile time");
            }
            (SymbolicFelt::Const(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqFI(rhs, lhs));
            }
            (SymbolicFelt::Const(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqFI(rhs_value, lhs));
            }
            (SymbolicFelt::Val(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqFI(lhs, rhs));
            }
            (SymbolicFelt::Val(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqF(lhs, rhs));
            }
            (SymbolicFelt::Val(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqF(lhs, rhs_value));
            }
            (lhs, rhs) => {
                let lhs_value = Self::uninit(builder);
                lhs_value.assign(lhs, builder);
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqF(lhs_value, rhs_value));
            }
        }
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (SymbolicFelt::Const(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                assert_ne!(lhs, rhs, "Assertion failed at compile time");
            }
            (SymbolicFelt::Const(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeFI(rhs, lhs));
            }
            (SymbolicFelt::Const(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeFI(rhs_value, lhs));
            }
            (SymbolicFelt::Val(lhs, _), SymbolicFelt::Const(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeFI(lhs, rhs));
            }
            (SymbolicFelt::Val(lhs, _), SymbolicFelt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeF(lhs, rhs));
            }
            (SymbolicFelt::Val(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeF(lhs, rhs_value));
            }
            (lhs, rhs) => {
                let lhs_value = Self::uninit(builder);
                lhs_value.assign(lhs, builder);
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeF(lhs_value, rhs_value));
            }
        }
    }
}

impl<C: Config> MemVariable<C> for Felt<C::F> {
    fn size_of() -> usize {
        1
    }

    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        builder.push(DslIr::LoadF(*self, ptr, index));
    }

    fn store(&self, ptr: Ptr<<C as Config>::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        builder.push(DslIr::StoreF(*self, ptr, index));
    }
}

impl<F: Field, EF: ExtensionField<F>> Ext<F, EF> {
    fn assign_with_caches<C: Config<F = F, EF = EF>>(
        &self,
        src: SymbolicExt<F, EF>,
        builder: &mut Builder<C>,
        ext_cache: &mut HashMap<SymbolicExt<F, EF>, Ext<F, EF>>,
        base_cache: &mut HashMap<SymbolicFelt<F>, Felt<F>>,
    ) {
        if let Some(v) = ext_cache.get(&src) {
            builder
                .operations
                .push(DslIr::AddEI(*self, *v, C::EF::ZERO));
            return;
        }
        match src {
            SymbolicExt::Base(v, _) => match &*v {
                SymbolicFelt::Const(c, _) => {
                    builder
                        .operations
                        .push(DslIr::ImmE(*self, C::EF::from_base(*c)));
                }
                SymbolicFelt::Val(v, _) => {
                    builder
                        .operations
                        .push(DslIr::AddEFFI(*self, *v, C::EF::ZERO));
                }
                v => {
                    let v_value = Felt::uninit(builder);
                    v_value.assign(v.clone(), builder);
                    builder.push(DslIr::AddEFFI(*self, v_value, C::EF::ZERO));
                }
            },
            SymbolicExt::Const(c, _) => {
                builder.operations.push(DslIr::ImmE(*self, c));
            }
            SymbolicExt::Val(v, _) => {
                builder.operations.push(DslIr::AddEI(*self, v, C::EF::ZERO));
            }
            SymbolicExt::Add(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicExt::Const(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    let sum = *lhs + *rhs;
                    builder.operations.push(DslIr::ImmE(*self, sum));
                }
                (SymbolicExt::Const(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.operations.push(DslIr::AddEI(*self, *rhs, *lhs));
                }
                (SymbolicExt::Const(lhs, _), SymbolicExt::Base(rhs, _)) => match rhs.as_ref() {
                    SymbolicFelt::Const(rhs, _) => {
                        let sum = *lhs + C::EF::from_base(*rhs);
                        builder.operations.push(DslIr::ImmE(*self, sum));
                    }
                    SymbolicFelt::Val(rhs, _) => {
                        builder.operations.push(DslIr::AddEFFI(*self, *rhs, *lhs));
                    }
                    rhs => {
                        let rhs_value: Felt<_> = Felt::uninit(builder);
                        rhs_value.assign_with_cache(rhs.clone(), builder, base_cache);
                        base_cache.insert(rhs.clone(), rhs_value);
                        builder
                            .operations
                            .push(DslIr::AddEFFI(*self, rhs_value, *lhs));
                    }
                },
                (SymbolicExt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddEI(*self, rhs_value, *lhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    builder.push(DslIr::AddEI(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Base(rhs, _)) => match rhs.as_ref() {
                    SymbolicFelt::Const(rhs, _) => {
                        builder.push(DslIr::AddEFI(*self, *lhs, *rhs));
                    }
                    SymbolicFelt::Val(rhs, _) => {
                        builder.push(DslIr::AddEF(*self, *lhs, *rhs));
                    }
                    rhs => {
                        let rhs = builder.eval(rhs.clone());
                        builder.push(DslIr::AddEF(*self, *lhs, rhs));
                    }
                },
                (SymbolicExt::Val(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::AddE(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddE(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicExt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::AddEI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicExt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::AddE(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::AddE(*self, lhs_value, rhs_value));
                }
            },
            SymbolicExt::Mul(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicExt::Const(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    let product = *lhs * *rhs;
                    builder.push(DslIr::ImmE(*self, product));
                }
                (SymbolicExt::Const(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::MulEI(*self, *rhs, *lhs));
                }
                (SymbolicExt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulEI(*self, rhs_value, *lhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    builder.push(DslIr::MulEI(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::MulE(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Base(rhs, _)) => match rhs.as_ref() {
                    SymbolicFelt::Const(rhs, _) => {
                        builder.push(DslIr::MulEFI(*self, *lhs, *rhs));
                    }
                    SymbolicFelt::Val(rhs, _) => {
                        builder.push(DslIr::MulEF(*self, *lhs, *rhs));
                    }
                    rhs => {
                        let rhs = builder.eval(rhs.clone());
                        builder.push(DslIr::MulEF(*self, *lhs, rhs));
                    }
                },
                (SymbolicExt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulE(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicExt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::MulEI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicExt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::MulE(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::MulE(*self, lhs_value, rhs_value));
                }
            },
            SymbolicExt::Sub(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicExt::Const(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    let difference = *lhs - *rhs;
                    builder.push(DslIr::ImmE(*self, difference));
                }
                (SymbolicExt::Const(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::SubEIN(*self, *lhs, *rhs));
                }
                (SymbolicExt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubEIN(*self, *lhs, rhs_value));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    builder.push(DslIr::SubEI(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::SubE(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Base(rhs, _)) => match rhs.as_ref() {
                    SymbolicFelt::Const(rhs, _) => {
                        builder.push(DslIr::SubEFI(*self, *lhs, *rhs));
                    }
                    SymbolicFelt::Val(rhs, _) => {
                        builder.push(DslIr::SubEF(*self, *lhs, *rhs));
                    }
                    rhs => {
                        let rhs = builder.eval(rhs.clone());
                        builder.push(DslIr::SubEF(*self, *lhs, rhs));
                    }
                },
                (SymbolicExt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::SubE(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicExt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::SubEI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicExt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::SubE(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign(rhs.clone(), builder);
                    builder.push(DslIr::SubE(*self, lhs_value, rhs_value));
                }
            },
            SymbolicExt::Div(lhs, rhs, _) => match (&*lhs, &*rhs) {
                (SymbolicExt::Const(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    let quotient = *lhs / *rhs;
                    builder.push(DslIr::ImmE(*self, quotient));
                }
                (SymbolicExt::Const(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::DivEIN(*self, *lhs, *rhs));
                }
                (SymbolicExt::Const(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::DivEIN(*self, *lhs, rhs_value));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Const(rhs, _)) => {
                    builder.push(DslIr::DivEI(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Val(rhs, _)) => {
                    builder.push(DslIr::DivE(*self, *lhs, *rhs));
                }
                (SymbolicExt::Val(lhs, _), SymbolicExt::Base(rhs, _)) => match rhs.as_ref() {
                    SymbolicFelt::Const(rhs, _) => {
                        builder.push(DslIr::DivEFI(*self, *lhs, *rhs));
                    }
                    SymbolicFelt::Val(rhs, _) => {
                        builder.push(DslIr::DivEF(*self, *lhs, *rhs));
                    }
                    rhs => {
                        let rhs = builder.eval(rhs.clone());
                        builder.push(DslIr::DivEF(*self, *lhs, rhs));
                    }
                },
                (SymbolicExt::Val(lhs, _), rhs) => {
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::DivE(*self, *lhs, rhs_value));
                }
                (lhs, SymbolicExt::Const(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::DivEI(*self, lhs_value, *rhs));
                }
                (lhs, SymbolicExt::Val(rhs, _)) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    builder.push(DslIr::DivE(*self, lhs_value, *rhs));
                }
                (lhs, rhs) => {
                    let lhs_value = Self::uninit(builder);
                    lhs_value.assign_with_caches(lhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(lhs.clone(), lhs_value);
                    let rhs_value = Self::uninit(builder);
                    rhs_value.assign_with_caches(rhs.clone(), builder, ext_cache, base_cache);
                    ext_cache.insert(rhs.clone(), rhs_value);
                    builder.push(DslIr::DivE(*self, lhs_value, rhs_value));
                }
            },
            SymbolicExt::Neg(operand, _) => match &*operand {
                SymbolicExt::Const(operand, _) => {
                    let negated = -*operand;
                    builder.push(DslIr::ImmE(*self, negated));
                }
                SymbolicExt::Val(operand, _) => {
                    builder.push(DslIr::NegE(*self, *operand));
                }
                operand => {
                    let operand_value = Self::uninit(builder);
                    operand_value.assign_with_caches(
                        operand.clone(),
                        builder,
                        ext_cache,
                        base_cache,
                    );
                    ext_cache.insert(operand.clone(), operand_value);
                    builder.push(DslIr::NegE(*self, operand_value));
                }
            },
        }
    }
}

impl<C: Config> Variable<C> for Ext<C::F, C::EF> {
    type Expression = SymbolicExt<C::F, C::EF>;

    fn uninit(builder: &mut Builder<C>) -> Self {
        builder.ext_count += 1;
        Ext(builder.ext_count, PhantomData)
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        self.assign_with_caches(src, builder, &mut HashMap::new(), &mut HashMap::new());
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (SymbolicExt::Const(lhs, _), SymbolicExt::Const(rhs, _)) => {
                assert_eq!(lhs, rhs, "Assertion failed at compile time");
            }
            (SymbolicExt::Const(lhs, _), SymbolicExt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqEI(rhs, lhs));
            }
            (SymbolicExt::Const(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqEI(rhs_value, lhs));
            }
            (SymbolicExt::Val(lhs, _), SymbolicExt::Const(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqEI(lhs, rhs));
            }
            (SymbolicExt::Val(lhs, _), SymbolicExt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertEqE(lhs, rhs));
            }
            (SymbolicExt::Val(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqE(lhs, rhs_value));
            }
            (lhs, rhs) => {
                let lhs_value = Self::uninit(builder);
                lhs_value.assign(lhs, builder);
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertEqE(lhs_value, rhs_value));
            }
        }
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (SymbolicExt::Const(lhs, _), SymbolicExt::Const(rhs, _)) => {
                assert_ne!(lhs, rhs, "Assertion failed at compile time");
            }
            (SymbolicExt::Const(lhs, _), SymbolicExt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeEI(rhs, lhs));
            }
            (SymbolicExt::Const(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeEI(rhs_value, lhs));
            }
            (SymbolicExt::Val(lhs, _), SymbolicExt::Const(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeEI(lhs, rhs));
            }
            (SymbolicExt::Val(lhs, _), SymbolicExt::Val(rhs, _)) => {
                builder.trace_push(DslIr::AssertNeE(lhs, rhs));
            }
            (SymbolicExt::Val(lhs, _), rhs) => {
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeE(lhs, rhs_value));
            }
            (lhs, rhs) => {
                let lhs_value = Self::uninit(builder);
                lhs_value.assign(lhs, builder);
                let rhs_value = Self::uninit(builder);
                rhs_value.assign(rhs, builder);
                builder.trace_push(DslIr::AssertNeE(lhs_value, rhs_value));
            }
        }
    }
}

impl<C: Config> MemVariable<C> for Ext<C::F, C::EF> {
    fn size_of() -> usize {
        C::EF::D
    }

    fn load(&self, ptr: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        builder.push(DslIr::LoadE(*self, ptr, index));
    }

    fn store(&self, ptr: Ptr<<C as Config>::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        builder.push(DslIr::StoreE(*self, ptr, index));
    }
}

impl<C: Config> FromConstant<C> for Var<C::N> {
    type Constant = C::N;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self {
        builder.eval(value)
    }
}

impl<C: Config> FromConstant<C> for Felt<C::F> {
    type Constant = C::F;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self {
        builder.eval(value)
    }
}

impl<C: Config> FromConstant<C> for Ext<C::F, C::EF> {
    type Constant = C::EF;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self {
        builder.eval(value.cons())
    }
}
