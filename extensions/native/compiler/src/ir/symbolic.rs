use alloc::sync::Arc;
use core::{
    any::Any,
    ops::{Add, Div, Mul, Neg, Sub},
};
use std::{
    any::TypeId,
    hash::Hash,
    iter::{Product, Sum},
    mem,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use openvm_stark_backend::p3_field::{ExtensionField, Field, FieldAlgebra, FieldArray, PrimeField};
use serde::{Deserialize, Serialize};

use super::{utils::prime_field_to_usize, Ext, Felt, Usize, Var};

const NUM_RANDOM_ELEMENTS: usize = 4;

pub type Digest<T> = FieldArray<T, NUM_RANDOM_ELEMENTS>;

pub fn elements<F: Field>() -> Digest<F> {
    let powers = [1671541671, 1254988180, 442438744, 1716490559];
    let generator = F::GENERATOR;

    Digest::from(powers.map(|p| generator.exp_u64(p)))
}

pub fn ext_elements<F: Field, EF: ExtensionField<F>>() -> Digest<EF> {
    let powers = [1021539871, 1430550064, 447478069, 1248903325];
    let generator = EF::GENERATOR;

    Digest::from(powers.map(|p| generator.exp_u64(p)))
}

fn digest_id<F: Field>(id: u32) -> Digest<F> {
    let elements = elements();
    Digest::from(elements.0.map(|e: F| {
        (e + F::from_canonical_u32(id))
            .try_inverse()
            .unwrap_or(F::ONE)
    }))
}

fn digest_id_ext<F: Field, EF: ExtensionField<F>>(id: u32) -> Digest<EF> {
    let elements = ext_elements();
    Digest::from(elements.0.map(|e: EF| {
        (e + EF::from_canonical_u32(id))
            .try_inverse()
            .unwrap_or(EF::ONE)
    }))
}

fn div_digests<F: Field>(a: Digest<F>, b: Digest<F>) -> Digest<F> {
    Digest::from(core::array::from_fn(|i| a.0[i] / b.0[i]))
}

/// A symbolic variable. For any binary operator, at least one of the operands must be variable.
#[derive(Debug, Clone)]
pub enum SymbolicVar<N: Field> {
    Const(N, Digest<N>),
    Val(Var<N>, Digest<N>),
    Add(Arc<SymbolicVar<N>>, Arc<SymbolicVar<N>>, Digest<N>),
    Mul(Arc<SymbolicVar<N>>, Arc<SymbolicVar<N>>, Digest<N>),
    Sub(Arc<SymbolicVar<N>>, Arc<SymbolicVar<N>>, Digest<N>),
    Neg(Arc<SymbolicVar<N>>, Digest<N>),
}

#[derive(Debug, Clone)]
pub enum SymbolicFelt<F: Field> {
    Const(F, Digest<F>),
    Val(Felt<F>, Digest<F>),
    Add(Arc<SymbolicFelt<F>>, Arc<SymbolicFelt<F>>, Digest<F>),
    Mul(Arc<SymbolicFelt<F>>, Arc<SymbolicFelt<F>>, Digest<F>),
    Sub(Arc<SymbolicFelt<F>>, Arc<SymbolicFelt<F>>, Digest<F>),
    Div(Arc<SymbolicFelt<F>>, Arc<SymbolicFelt<F>>, Digest<F>),
    Neg(Arc<SymbolicFelt<F>>, Digest<F>),
}

#[derive(Debug, Clone)]
pub enum SymbolicExt<F: Field, EF: Field> {
    Const(EF, Digest<EF>),
    Base(Arc<SymbolicFelt<F>>, Digest<EF>),
    Val(Ext<F, EF>, Digest<EF>),
    Add(Arc<SymbolicExt<F, EF>>, Arc<SymbolicExt<F, EF>>, Digest<EF>),
    Mul(Arc<SymbolicExt<F, EF>>, Arc<SymbolicExt<F, EF>>, Digest<EF>),
    Sub(Arc<SymbolicExt<F, EF>>, Arc<SymbolicExt<F, EF>>, Digest<EF>),
    Div(Arc<SymbolicExt<F, EF>>, Arc<SymbolicExt<F, EF>>, Digest<EF>),
    Neg(Arc<SymbolicExt<F, EF>>, Digest<EF>),
}

/// A right value of Var. It should never be assigned with a value.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum RVar<N> {
    Const(N),
    Val(Var<N>),
}

impl<N: PrimeField> RVar<N> {
    pub fn zero() -> Self {
        RVar::Const(N::ZERO)
    }
    pub fn one() -> Self {
        RVar::Const(N::ONE)
    }
    pub fn from_field(n: N) -> Self {
        RVar::Const(n)
    }
    pub fn is_const(&self) -> bool {
        match self {
            RVar::Const(_) => true,
            RVar::Val(_) => false,
        }
    }
    pub fn value(&self) -> usize {
        match self {
            RVar::Const(c) => prime_field_to_usize(*c),
            _ => panic!("RVar::value() called on non-const value"),
        }
    }
    pub fn field_value(&self) -> N {
        match self {
            RVar::Const(c) => *c,
            _ => panic!("RVar::field_value() called on non-const value"),
        }
    }

    pub fn variable(&self) -> Var<N> {
        match self {
            RVar::Const(_) => panic!("RVar::variable() called on const value"),
            RVar::Val(var) => *var,
        }
    }
}

impl<N: Field> Hash for SymbolicVar<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for elem in self.digest().0.iter() {
            elem.hash(state);
        }
    }
}

impl<N: Field> PartialEq for SymbolicVar<N> {
    fn eq(&self, other: &Self) -> bool {
        if self.digest() != other.digest() {
            return false;
        }
        match (self, other) {
            (SymbolicVar::Const(a, _), SymbolicVar::Const(b, _)) => a == b,
            (SymbolicVar::Val(a, _), SymbolicVar::Val(b, _)) => a == b,
            (SymbolicVar::Add(a, b, _), SymbolicVar::Add(c, d, _)) => a == c && b == d,
            (SymbolicVar::Mul(a, b, _), SymbolicVar::Mul(c, d, _)) => a == c && b == d,
            (SymbolicVar::Sub(a, b, _), SymbolicVar::Sub(c, d, _)) => a == c && b == d,
            (SymbolicVar::Neg(a, _), SymbolicVar::Neg(b, _)) => a == b,
            _ => false,
        }
    }
}

impl<N: Field> Eq for SymbolicVar<N> {}

impl<F: Field> Hash for SymbolicFelt<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for elem in self.digest().0.iter() {
            elem.hash(state);
        }
    }
}

impl<F: Field> PartialEq for SymbolicFelt<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.digest() != other.digest() {
            return false;
        }
        match (self, other) {
            (SymbolicFelt::Const(a, _), SymbolicFelt::Const(b, _)) => a == b,
            (SymbolicFelt::Val(a, _), SymbolicFelt::Val(b, _)) => a == b,
            (SymbolicFelt::Add(a, b, _), SymbolicFelt::Add(c, d, _)) => a == c && b == d,
            (SymbolicFelt::Mul(a, b, _), SymbolicFelt::Mul(c, d, _)) => a == c && b == d,
            (SymbolicFelt::Sub(a, b, _), SymbolicFelt::Sub(c, d, _)) => a == c && b == d,
            (SymbolicFelt::Div(a, b, _), SymbolicFelt::Div(c, d, _)) => a == c && b == d,
            (SymbolicFelt::Neg(a, _), SymbolicFelt::Neg(b, _)) => a == b,
            _ => false,
        }
    }
}

impl<F: Field> Eq for SymbolicFelt<F> {}

impl<F: Field, EF: Field> Hash for SymbolicExt<F, EF> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for elem in self.digest().0.iter() {
            elem.hash(state);
        }
    }
}

impl<F: Field, EF: Field> PartialEq for SymbolicExt<F, EF> {
    fn eq(&self, other: &Self) -> bool {
        if self.digest() != other.digest() {
            return false;
        }
        match (self, other) {
            (SymbolicExt::Const(a, _), SymbolicExt::Const(b, _)) => a == b,
            (SymbolicExt::Base(a, _), SymbolicExt::Base(b, _)) => a == b,
            (SymbolicExt::Val(a, _), SymbolicExt::Val(b, _)) => a == b,
            (SymbolicExt::Add(a, b, _), SymbolicExt::Add(c, d, _)) => a == c && b == d,
            (SymbolicExt::Mul(a, b, _), SymbolicExt::Mul(c, d, _)) => a == c && b == d,
            (SymbolicExt::Sub(a, b, _), SymbolicExt::Sub(c, d, _)) => a == c && b == d,
            (SymbolicExt::Div(a, b, _), SymbolicExt::Div(c, d, _)) => a == c && b == d,
            (SymbolicExt::Neg(a, _), SymbolicExt::Neg(b, _)) => a == b,
            _ => false,
        }
    }
}

impl<F: Field, EF: Field> Eq for SymbolicExt<F, EF> {}

impl<N: Field> SymbolicVar<N> {
    pub(crate) const fn digest(&self) -> Digest<N> {
        match self {
            SymbolicVar::Const(_, d) => *d,
            SymbolicVar::Val(_, d) => *d,
            SymbolicVar::Add(_, _, d) => *d,
            SymbolicVar::Mul(_, _, d) => *d,
            SymbolicVar::Sub(_, _, d) => *d,
            SymbolicVar::Neg(_, d) => *d,
        }
    }
}

impl<F: Field> SymbolicFelt<F> {
    pub(crate) const fn digest(&self) -> Digest<F> {
        match self {
            SymbolicFelt::Const(_, d) => *d,
            SymbolicFelt::Val(_, d) => *d,
            SymbolicFelt::Add(_, _, d) => *d,
            SymbolicFelt::Mul(_, _, d) => *d,
            SymbolicFelt::Sub(_, _, d) => *d,
            SymbolicFelt::Div(_, _, d) => *d,
            SymbolicFelt::Neg(_, d) => *d,
        }
    }
}

impl<F: Field, EF: Field> SymbolicExt<F, EF> {
    pub(crate) const fn digest(&self) -> Digest<EF> {
        match self {
            SymbolicExt::Const(_, d) => *d,
            SymbolicExt::Base(_, d) => *d,
            SymbolicExt::Val(_, d) => *d,
            SymbolicExt::Add(_, _, d) => *d,
            SymbolicExt::Mul(_, _, d) => *d,
            SymbolicExt::Sub(_, _, d) => *d,
            SymbolicExt::Div(_, _, d) => *d,
            SymbolicExt::Neg(_, d) => *d,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExtOperand<F: Field, EF: ExtensionField<F>> {
    Base(F),
    Const(EF),
    Felt(Felt<F>),
    Ext(Ext<F, EF>),
    SymFelt(SymbolicFelt<F>),
    Sym(SymbolicExt<F, EF>),
}

impl<F: Field, EF: ExtensionField<F>> ExtOperand<F, EF> {
    pub fn digest(&self) -> Digest<EF> {
        match self {
            ExtOperand::Base(f) => SymbolicFelt::from(*f).digest().0.map(EF::from_base).into(),
            ExtOperand::Const(ef) => (*ef).into(),
            ExtOperand::Felt(f) => SymbolicFelt::from(*f).digest().0.map(EF::from_base).into(),
            ExtOperand::Ext(e) => digest_id_ext::<F, EF>(e.0),
            ExtOperand::SymFelt(f) => f.digest().0.map(EF::from_base).into(),
            ExtOperand::Sym(e) => e.digest(),
        }
    }

    pub fn symbolic(self) -> SymbolicExt<F, EF> {
        let digest = self.digest();
        match self {
            ExtOperand::Base(f) => SymbolicExt::Base(Arc::new(SymbolicFelt::from(f)), digest),
            ExtOperand::Const(ef) => SymbolicExt::Const(ef, digest),
            ExtOperand::Felt(f) => SymbolicExt::Base(Arc::new(SymbolicFelt::from(f)), digest),
            ExtOperand::Ext(e) => SymbolicExt::Val(e, digest),
            ExtOperand::SymFelt(f) => SymbolicExt::Base(Arc::new(f), digest),
            ExtOperand::Sym(e) => e,
        }
    }
}

pub trait ExtConst<F: Field, EF: ExtensionField<F>> {
    fn cons(self) -> SymbolicExt<F, EF>;
}

impl<F: Field, EF: ExtensionField<F>> ExtConst<F, EF> for EF {
    fn cons(self) -> SymbolicExt<F, EF> {
        SymbolicExt::Const(self, self.into())
    }
}

pub trait ExtensionOperand<F: Field, EF: ExtensionField<F>> {
    fn to_operand(self) -> ExtOperand<F, EF>;
}

impl<N: Field> FieldAlgebra for SymbolicVar<N> {
    type F = N;

    const ZERO: Self = SymbolicVar::Const(N::ZERO, FieldArray([N::ZERO; 4]));
    const ONE: Self = SymbolicVar::Const(N::ONE, FieldArray([N::ONE; 4]));
    const TWO: Self = SymbolicVar::Const(N::TWO, FieldArray([N::TWO; 4]));
    const NEG_ONE: Self = SymbolicVar::Const(N::NEG_ONE, FieldArray([N::NEG_ONE; 4]));

    fn from_f(f: Self::F) -> Self {
        SymbolicVar::from(f)
    }
    fn from_bool(b: bool) -> Self {
        SymbolicVar::from(N::from_bool(b))
    }
    fn from_canonical_u8(n: u8) -> Self {
        SymbolicVar::from(N::from_canonical_u8(n))
    }
    fn from_canonical_u16(n: u16) -> Self {
        SymbolicVar::from(N::from_canonical_u16(n))
    }
    fn from_canonical_u32(n: u32) -> Self {
        SymbolicVar::from(N::from_canonical_u32(n))
    }
    fn from_canonical_u64(n: u64) -> Self {
        SymbolicVar::from(N::from_canonical_u64(n))
    }
    fn from_canonical_usize(n: usize) -> Self {
        SymbolicVar::from(N::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        SymbolicVar::from(N::from_wrapped_u32(n))
    }
    fn from_wrapped_u64(n: u64) -> Self {
        SymbolicVar::from(N::from_wrapped_u64(n))
    }
}

/// Trait to exclude SymbolicVar in generic parameters.
trait NotSymbolicVar {}
impl<N: Field> NotSymbolicVar for N {}
impl<N: Field> NotSymbolicVar for Var<N> {}
impl<N: PrimeField> NotSymbolicVar for Usize<N> {}
impl<N: Field> NotSymbolicVar for RVar<N> {}

impl<F: Field> FieldAlgebra for SymbolicFelt<F> {
    type F = F;

    const ZERO: Self = SymbolicFelt::Const(F::ZERO, FieldArray([F::ZERO; 4]));
    const ONE: Self = SymbolicFelt::Const(F::ONE, FieldArray([F::ONE; 4]));
    const TWO: Self = SymbolicFelt::Const(F::TWO, FieldArray([F::TWO; 4]));
    const NEG_ONE: Self = SymbolicFelt::Const(F::NEG_ONE, FieldArray([F::NEG_ONE; 4]));

    fn from_f(f: Self::F) -> Self {
        SymbolicFelt::from(f)
    }
    fn from_bool(b: bool) -> Self {
        SymbolicFelt::from(F::from_bool(b))
    }
    fn from_canonical_u8(n: u8) -> Self {
        SymbolicFelt::from(F::from_canonical_u8(n))
    }
    fn from_canonical_u16(n: u16) -> Self {
        SymbolicFelt::from(F::from_canonical_u16(n))
    }
    fn from_canonical_u32(n: u32) -> Self {
        SymbolicFelt::from(F::from_canonical_u32(n))
    }
    fn from_canonical_u64(n: u64) -> Self {
        SymbolicFelt::from(F::from_canonical_u64(n))
    }
    fn from_canonical_usize(n: usize) -> Self {
        SymbolicFelt::from(F::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        SymbolicFelt::from(F::from_wrapped_u32(n))
    }
    fn from_wrapped_u64(n: u64) -> Self {
        SymbolicFelt::from(F::from_wrapped_u64(n))
    }
}

impl<F: Field, EF: ExtensionField<F>> FieldAlgebra for SymbolicExt<F, EF> {
    type F = EF;

    const ZERO: Self = SymbolicExt::Const(EF::ZERO, FieldArray([EF::ZERO; 4]));
    const ONE: Self =
        SymbolicExt::Const(EF::ONE, FieldArray([EF::ZERO, EF::ZERO, EF::ZERO, EF::ONE]));
    const TWO: Self =
        SymbolicExt::Const(EF::TWO, FieldArray([EF::ZERO, EF::ZERO, EF::ZERO, EF::TWO]));
    const NEG_ONE: Self = SymbolicExt::Const(
        EF::NEG_ONE,
        FieldArray([EF::ZERO, EF::ZERO, EF::ZERO, EF::NEG_ONE]),
    );

    fn from_f(f: Self::F) -> Self {
        SymbolicExt::Const(f, f.into())
    }
    fn from_bool(b: bool) -> Self {
        SymbolicExt::from_f(EF::from_bool(b))
    }
    fn from_canonical_u8(n: u8) -> Self {
        SymbolicExt::from_f(EF::from_canonical_u8(n))
    }
    fn from_canonical_u16(n: u16) -> Self {
        SymbolicExt::from_f(EF::from_canonical_u16(n))
    }
    fn from_canonical_u32(n: u32) -> Self {
        SymbolicExt::from_f(EF::from_canonical_u32(n))
    }
    fn from_canonical_u64(n: u64) -> Self {
        SymbolicExt::from_f(EF::from_canonical_u64(n))
    }
    fn from_canonical_usize(n: usize) -> Self {
        SymbolicExt::from_f(EF::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        SymbolicExt::from_f(EF::from_wrapped_u32(n))
    }
    fn from_wrapped_u64(n: u64) -> Self {
        SymbolicExt::from_f(EF::from_wrapped_u64(n))
    }
}

// Implement all conversions from constants N, F, EF, to the corresponding symbolic types

impl<N: Field> From<N> for SymbolicVar<N> {
    fn from(n: N) -> Self {
        SymbolicVar::Const(n, n.into())
    }
}

impl<F: Field> From<F> for SymbolicFelt<F> {
    fn from(f: F) -> Self {
        SymbolicFelt::Const(f, f.into())
    }
}

impl<F: Field, EF: ExtensionField<F>> From<F> for SymbolicExt<F, EF> {
    fn from(f: F) -> Self {
        f.to_operand().symbolic()
    }
}

// Implement all conversions from Var<N>, Felt<F>, Ext<F, EF> to the corresponding symbolic types

impl<N: Field> From<Var<N>> for SymbolicVar<N> {
    fn from(v: Var<N>) -> Self {
        SymbolicVar::Val(v, digest_id(v.0))
    }
}

impl<F: Field> From<Felt<F>> for SymbolicFelt<F> {
    fn from(f: Felt<F>) -> Self {
        SymbolicFelt::Val(f, digest_id(f.0))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<Ext<F, EF>> for SymbolicExt<F, EF> {
    fn from(e: Ext<F, EF>) -> Self {
        e.to_operand().symbolic()
    }
}

// Implement all operations for SymbolicVar<N>, SymbolicFelt<F>, SymbolicExt<F, EF>

impl<N: Field> Add for SymbolicVar<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let digest = self.digest() + rhs.digest();
        match (&self, &rhs) {
            (SymbolicVar::Const(a, _), SymbolicVar::Const(b, _)) => {
                return SymbolicVar::Const(*a + *b, digest);
            }
            (SymbolicVar::Const(a, _), _) => {
                if a.is_zero() {
                    return rhs;
                }
            }
            (_, SymbolicVar::Const(b, _)) => {
                if b.is_zero() {
                    return self;
                }
            }
            _ => (),
        }
        SymbolicVar::Add(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<N: Field, RHS: Into<SymbolicVar<N>> + NotSymbolicVar> Add<RHS> for SymbolicVar<N> {
    type Output = Self;

    fn add(self, rhs: RHS) -> Self::Output {
        self + rhs.into()
    }
}

impl<F: Field> Add for SymbolicFelt<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let digest = self.digest() + rhs.digest();
        SymbolicFelt::Add(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<F: Field, EF: ExtensionField<F>, E: ExtensionOperand<F, EF>> Add<E> for SymbolicExt<F, EF> {
    type Output = Self;

    fn add(self, rhs: E) -> Self::Output {
        let rhs = rhs.to_operand().symbolic();
        let digest = self.digest() + rhs.digest();
        SymbolicExt::Add(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<N: Field> Mul for SymbolicVar<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let digest = self.digest() * rhs.digest();
        match (&self, &rhs) {
            (SymbolicVar::Const(a, _), SymbolicVar::Const(b, _)) => {
                return SymbolicVar::Const(*a * *b, digest);
            }
            (SymbolicVar::Const(a, _), _) => {
                if a.is_zero() {
                    return self;
                }
                if a.is_one() {
                    return rhs;
                }
            }
            (_, SymbolicVar::Const(b, _)) => {
                if b.is_zero() {
                    return rhs;
                }
                if b.is_one() {
                    return self;
                }
            }
            _ => (),
        }
        SymbolicVar::Mul(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<N: Field, RHS: Into<SymbolicVar<N>> + NotSymbolicVar> Mul<RHS> for SymbolicVar<N> {
    type Output = Self;

    fn mul(self, rhs: RHS) -> Self::Output {
        self * rhs.into()
    }
}

impl<F: Field> Mul for SymbolicFelt<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let digest = self.digest() * rhs.digest();
        SymbolicFelt::Mul(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Mul<E> for SymbolicExt<F, EF> {
    type Output = Self;

    fn mul(self, rhs: E) -> Self::Output {
        let rhs = rhs.to_operand();
        let rhs_digest = rhs.digest();
        let prod_digest = self.digest() * rhs_digest;
        match rhs {
            ExtOperand::Base(f) => SymbolicExt::Mul(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(
                    Arc::new(SymbolicFelt::from(f)),
                    rhs_digest,
                )),
                prod_digest,
            ),
            ExtOperand::Const(ef) => SymbolicExt::Mul(
                Arc::new(self),
                Arc::new(SymbolicExt::Const(ef, rhs_digest)),
                prod_digest,
            ),
            ExtOperand::Felt(f) => SymbolicExt::Mul(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(
                    Arc::new(SymbolicFelt::from(f)),
                    rhs_digest,
                )),
                prod_digest,
            ),
            ExtOperand::Ext(e) => SymbolicExt::Mul(
                Arc::new(self),
                Arc::new(SymbolicExt::Val(e, rhs_digest)),
                prod_digest,
            ),
            ExtOperand::SymFelt(f) => SymbolicExt::Mul(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(Arc::new(f), rhs_digest)),
                prod_digest,
            ),
            ExtOperand::Sym(e) => SymbolicExt::Mul(Arc::new(self), Arc::new(e), prod_digest),
        }
    }
}

impl<N: Field> Sub for SymbolicVar<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let digest = self.digest() - rhs.digest();
        match (&self, &rhs) {
            (SymbolicVar::Const(a, _), SymbolicVar::Const(b, _)) => {
                return SymbolicVar::Const(*a - *b, digest);
            }
            (SymbolicVar::Const(a, _), _) => {
                if a.is_zero() {
                    return rhs;
                }
            }
            (_, SymbolicVar::Const(b, _)) => {
                if b.is_zero() {
                    return self;
                }
            }
            _ => (),
        }
        SymbolicVar::Sub(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<N: Field, RHS: Into<SymbolicVar<N>> + NotSymbolicVar> Sub<RHS> for SymbolicVar<N> {
    type Output = Self;

    fn sub(self, rhs: RHS) -> Self::Output {
        self - rhs.into()
    }
}

impl<F: Field> Sub for SymbolicFelt<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let digest = self.digest() - rhs.digest();
        SymbolicFelt::Sub(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Sub<E> for SymbolicExt<F, EF> {
    type Output = Self;

    fn sub(self, rhs: E) -> Self::Output {
        let rhs = rhs.to_operand();
        let rhs_digest = rhs.digest();
        let digest = self.digest() - rhs_digest;
        match rhs {
            ExtOperand::Base(f) => SymbolicExt::Sub(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(
                    Arc::new(SymbolicFelt::from(f)),
                    rhs_digest,
                )),
                digest,
            ),
            ExtOperand::Const(ef) => SymbolicExt::Sub(
                Arc::new(self),
                Arc::new(SymbolicExt::Const(ef, rhs_digest)),
                digest,
            ),
            ExtOperand::Felt(f) => SymbolicExt::Sub(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(
                    Arc::new(SymbolicFelt::from(f)),
                    rhs_digest,
                )),
                digest,
            ),
            ExtOperand::Ext(e) => SymbolicExt::Sub(
                Arc::new(self),
                Arc::new(SymbolicExt::Val(e, rhs_digest)),
                digest,
            ),
            ExtOperand::SymFelt(f) => SymbolicExt::Sub(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(Arc::new(f), rhs_digest)),
                digest,
            ),
            ExtOperand::Sym(e) => SymbolicExt::Sub(Arc::new(self), Arc::new(e), digest),
        }
    }
}

impl<F: Field> Div for SymbolicFelt<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let self_digest = self.digest();
        let rhs_digest = rhs.digest();
        let digest = div_digests(self_digest, rhs_digest);
        SymbolicFelt::Div(Arc::new(self), Arc::new(rhs), digest)
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Div<E> for SymbolicExt<F, EF> {
    type Output = Self;

    fn div(self, rhs: E) -> Self::Output {
        let rhs = rhs.to_operand();
        let rhs_digest = rhs.digest();
        let digest = div_digests(self.digest(), rhs_digest);
        match rhs {
            ExtOperand::Base(f) => SymbolicExt::Div(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(
                    Arc::new(SymbolicFelt::from(f)),
                    rhs_digest,
                )),
                digest,
            ),
            ExtOperand::Const(ef) => SymbolicExt::Div(
                Arc::new(self),
                Arc::new(SymbolicExt::Const(ef, rhs_digest)),
                digest,
            ),
            ExtOperand::Felt(f) => SymbolicExt::Div(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(
                    Arc::new(SymbolicFelt::from(f)),
                    rhs_digest,
                )),
                digest,
            ),
            ExtOperand::Ext(e) => SymbolicExt::Div(
                Arc::new(self),
                Arc::new(SymbolicExt::Val(e, rhs_digest)),
                digest,
            ),
            ExtOperand::SymFelt(f) => SymbolicExt::Div(
                Arc::new(self),
                Arc::new(SymbolicExt::Base(Arc::new(f), rhs_digest)),
                digest,
            ),
            ExtOperand::Sym(e) => SymbolicExt::Div(Arc::new(self), Arc::new(e), digest),
        }
    }
}

impl<N: Field> Neg for SymbolicVar<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let digest = -self.digest();
        match &self {
            SymbolicVar::Const(c, _) => SymbolicVar::Const(-*c, digest),
            _ => SymbolicVar::Neg(Arc::new(self), digest),
        }
    }
}

impl<F: Field> Neg for SymbolicFelt<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let digest = -self.digest();
        SymbolicFelt::Neg(Arc::new(self), digest)
    }
}

impl<F: Field, EF: ExtensionField<F>> Neg for SymbolicExt<F, EF> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let digest = -self.digest();
        SymbolicExt::Neg(Arc::new(self), digest)
    }
}

// Implement all operations between N, F, EF, and SymbolicVar<N>, SymbolicFelt<F>, SymbolicExt<F,
// EF>

impl<F: Field> Add<F> for SymbolicFelt<F> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        self + SymbolicFelt::from(rhs)
    }
}

impl<F: Field> Mul<F> for SymbolicFelt<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        self * SymbolicFelt::from(rhs)
    }
}

impl<F: Field> Sub<F> for SymbolicFelt<F> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self - SymbolicFelt::from(rhs)
    }
}

// Implement all operations between SymbolicVar<N>, SymbolicFelt<F>, SymbolicExt<F, EF>, and Var<N>,
//  Felt<F>, Ext<F, EF>.

impl<F: Field> Add<Felt<F>> for SymbolicFelt<F> {
    type Output = SymbolicFelt<F>;

    fn add(self, rhs: Felt<F>) -> Self::Output {
        self + SymbolicFelt::from(rhs)
    }
}

impl<F: Field> Mul<Felt<F>> for SymbolicFelt<F> {
    type Output = SymbolicFelt<F>;

    fn mul(self, rhs: Felt<F>) -> Self::Output {
        self * SymbolicFelt::from(rhs)
    }
}

impl<F: Field> Sub<Felt<F>> for SymbolicFelt<F> {
    type Output = SymbolicFelt<F>;

    fn sub(self, rhs: Felt<F>) -> Self::Output {
        self - SymbolicFelt::from(rhs)
    }
}

impl<F: Field> Div<SymbolicFelt<F>> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn div(self, rhs: SymbolicFelt<F>) -> Self::Output {
        SymbolicFelt::<F>::from(self) / rhs
    }
}

// Implement operations between constants N, F, EF, and Var<N>, Felt<F>, Ext<F, EF>.

impl<F: Field> Add for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn add(self, rhs: Self) -> Self::Output {
        SymbolicFelt::<F>::from(self) + rhs
    }
}

impl<F: Field> Add<F> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn add(self, rhs: F) -> Self::Output {
        SymbolicFelt::from(self) + rhs
    }
}

impl<N: Field> Mul for Var<N> {
    type Output = SymbolicVar<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        SymbolicVar::<N>::from(self) * rhs
    }
}

impl<N: Field> Mul<N> for Var<N> {
    type Output = SymbolicVar<N>;

    fn mul(self, rhs: N) -> Self::Output {
        SymbolicVar::from(self) * rhs
    }
}

impl<F: Field> Mul for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn mul(self, rhs: Self) -> Self::Output {
        SymbolicFelt::<F>::from(self) * rhs
    }
}

impl<F: Field> Mul<F> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn mul(self, rhs: F) -> Self::Output {
        SymbolicFelt::from(self) * rhs
    }
}

impl<F: Field> Sub for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        SymbolicFelt::<F>::from(self) - rhs
    }
}

impl<F: Field> Sub<F> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn sub(self, rhs: F) -> Self::Output {
        SymbolicFelt::from(self) - rhs
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Add<E> for Ext<F, EF> {
    type Output = SymbolicExt<F, EF>;

    fn add(self, rhs: E) -> Self::Output {
        let rhs: ExtOperand<F, EF> = rhs.to_operand();
        let self_sym = self.to_operand().symbolic();
        self_sym + rhs
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Mul<E> for Ext<F, EF> {
    type Output = SymbolicExt<F, EF>;

    fn mul(self, rhs: E) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym * rhs
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Sub<E> for Ext<F, EF> {
    type Output = SymbolicExt<F, EF>;

    fn sub(self, rhs: E) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym - rhs
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> Div<E> for Ext<F, EF> {
    type Output = SymbolicExt<F, EF>;

    fn div(self, rhs: E) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym / rhs
    }
}

impl<F: Field, EF: ExtensionField<F>> Add<SymbolicExt<F, EF>> for Felt<F> {
    type Output = SymbolicExt<F, EF>;

    fn add(self, rhs: SymbolicExt<F, EF>) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym + rhs
    }
}

impl<F: Field, EF: ExtensionField<F>> Mul<SymbolicExt<F, EF>> for Felt<F> {
    type Output = SymbolicExt<F, EF>;

    fn mul(self, rhs: SymbolicExt<F, EF>) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym * rhs
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub<SymbolicExt<F, EF>> for Felt<F> {
    type Output = SymbolicExt<F, EF>;

    fn sub(self, rhs: SymbolicExt<F, EF>) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym - rhs
    }
}

impl<F: Field, EF: ExtensionField<F>> Div<SymbolicExt<F, EF>> for Felt<F> {
    type Output = SymbolicExt<F, EF>;

    fn div(self, rhs: SymbolicExt<F, EF>) -> Self::Output {
        let self_sym = self.to_operand().symbolic();
        self_sym / rhs
    }
}

impl<F: Field> Div for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn div(self, rhs: Self) -> Self::Output {
        SymbolicFelt::<F>::from(self) / rhs
    }
}

impl<F: Field> Div<F> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn div(self, rhs: F) -> Self::Output {
        SymbolicFelt::from(self) / rhs
    }
}

impl<F: Field> Div<Felt<F>> for SymbolicFelt<F> {
    type Output = SymbolicFelt<F>;

    fn div(self, rhs: Felt<F>) -> Self::Output {
        self / SymbolicFelt::from(rhs)
    }
}

impl<F: Field> Div<F> for SymbolicFelt<F> {
    type Output = SymbolicFelt<F>;

    fn div(self, rhs: F) -> Self::Output {
        self / SymbolicFelt::from(rhs)
    }
}

impl<N: Field> Product for SymbolicVar<N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SymbolicVar::ONE, |acc, x| acc * x)
    }
}

impl<N: Field> Sum for SymbolicVar<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SymbolicVar::ZERO, |acc, x| acc + x)
    }
}

impl<N: Field> AddAssign for SymbolicVar<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<N: Field> SubAssign for SymbolicVar<N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<N: Field> MulAssign for SymbolicVar<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<N: Field> Default for SymbolicVar<N> {
    fn default() -> Self {
        SymbolicVar::ZERO
    }
}

impl<F: Field> Sum for SymbolicFelt<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SymbolicFelt::ZERO, |acc, x| acc + x)
    }
}

impl<F: Field> Product for SymbolicFelt<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SymbolicFelt::ONE, |acc, x| acc * x)
    }
}

impl<F: Field> AddAssign for SymbolicFelt<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field> SubAssign for SymbolicFelt<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field> MulAssign for SymbolicFelt<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F: Field> Default for SymbolicFelt<F> {
    fn default() -> Self {
        SymbolicFelt::ZERO
    }
}

impl<F: Field, EF: ExtensionField<F>> Sum for SymbolicExt<F, EF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SymbolicExt::ZERO, |acc, x| acc + x)
    }
}

impl<F: Field, EF: ExtensionField<F>> Product for SymbolicExt<F, EF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(SymbolicExt::ONE, |acc, x| acc * x)
    }
}

impl<F: Field, EF: ExtensionField<F>> Default for SymbolicExt<F, EF> {
    fn default() -> Self {
        SymbolicExt::ZERO
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> AddAssign<E> for SymbolicExt<F, EF> {
    fn add_assign(&mut self, rhs: E) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> SubAssign<E> for SymbolicExt<F, EF> {
    fn sub_assign(&mut self, rhs: E) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> MulAssign<E> for SymbolicExt<F, EF> {
    fn mul_assign(&mut self, rhs: E) {
        *self = self.clone() * rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> DivAssign<E> for SymbolicExt<F, EF> {
    fn div_assign(&mut self, rhs: E) {
        *self = self.clone() / rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>, E: Any> ExtensionOperand<F, EF> for E {
    fn to_operand(self) -> ExtOperand<F, EF> {
        match self.type_id() {
            ty if ty == TypeId::of::<F>() => {
                // *Safety*: We know that E is a F and we can transmute it to F which implements
                // the Copy trait.
                let value = unsafe { mem::transmute_copy::<E, F>(&self) };
                ExtOperand::<F, EF>::Base(value)
            }
            ty if ty == TypeId::of::<EF>() => {
                // *Safety*: We know that E is a EF and we can transmute it to EF which implements
                // the Copy trait.
                let value = unsafe { mem::transmute_copy::<E, EF>(&self) };
                ExtOperand::<F, EF>::Const(value)
            }
            ty if ty == TypeId::of::<Felt<F>>() => {
                // *Safety*: We know that E is a Felt<F> and we can transmute it to Felt<F> which
                // implements the Copy trait.
                let value = unsafe { mem::transmute_copy::<E, Felt<F>>(&self) };
                ExtOperand::<F, EF>::Felt(value)
            }
            ty if ty == TypeId::of::<Ext<F, EF>>() => {
                // *Safety*: We know that E is a Ext<F, EF> and we can transmute it to Ext<F, EF>
                // which implements the Copy trait.
                let value = unsafe { mem::transmute_copy::<E, Ext<F, EF>>(&self) };
                ExtOperand::<F, EF>::Ext(value)
            }
            ty if ty == TypeId::of::<SymbolicFelt<F>>() => {
                // *Safety*: We know that E is a Symbolic Felt<F> and we can transmute it to
                // SymbolicFelt<F> but we need to clone the pointer.
                let value_ref = unsafe { mem::transmute::<&E, &SymbolicFelt<F>>(&self) };
                let value = value_ref.clone();
                ExtOperand::<F, EF>::SymFelt(value)
            }
            ty if ty == TypeId::of::<SymbolicExt<F, EF>>() => {
                // *Safety*: We know that E is a SymbolicExt<F, EF> and we can transmute it to
                // SymbolicExt<F, EF> but we need to clone the pointer.
                let value_ref = unsafe { mem::transmute::<&E, &SymbolicExt<F, EF>>(&self) };
                let value = value_ref.clone();
                ExtOperand::<F, EF>::Sym(value)
            }
            ty if ty == TypeId::of::<ExtOperand<F, EF>>() => {
                let value_ref = unsafe { mem::transmute::<&E, &ExtOperand<F, EF>>(&self) };
                value_ref.clone()
            }
            _ => unimplemented!("unsupported type"),
        }
    }
}

impl<F: Field> Add<SymbolicFelt<F>> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn add(self, rhs: SymbolicFelt<F>) -> Self::Output {
        SymbolicFelt::<F>::from(self) + rhs
    }
}

impl<F: Field, EF: ExtensionField<F>> From<Felt<F>> for SymbolicExt<F, EF> {
    fn from(value: Felt<F>) -> Self {
        value.to_operand().symbolic()
    }
}

impl<F: Field, EF: ExtensionField<F>> Neg for Ext<F, EF> {
    type Output = SymbolicExt<F, EF>;
    fn neg(self) -> Self::Output {
        -SymbolicExt::from(self)
    }
}

impl<F: Field> Neg for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn neg(self) -> Self::Output {
        -SymbolicFelt::from(self)
    }
}

impl<N: Field> Neg for Var<N> {
    type Output = SymbolicVar<N>;

    fn neg(self) -> Self::Output {
        -SymbolicVar::from(self)
    }
}

impl<F: Field> MulAssign<Felt<F>> for SymbolicFelt<F> {
    fn mul_assign(&mut self, rhs: Felt<F>) {
        *self = self.clone() * Self::from(rhs);
    }
}

impl<F: Field> Mul<SymbolicFelt<F>> for Felt<F> {
    type Output = SymbolicFelt<F>;

    fn mul(self, rhs: SymbolicFelt<F>) -> Self::Output {
        SymbolicFelt::<F>::from(self) * rhs
    }
}

impl<N: Field> Mul<SymbolicVar<N>> for Var<N> {
    type Output = SymbolicVar<N>;

    fn mul(self, rhs: SymbolicVar<N>) -> Self::Output {
        SymbolicVar::<N>::from(self) * rhs
    }
}

impl<N: Field, RHS: Into<SymbolicVar<N>>> Add<RHS> for Var<N> {
    type Output = SymbolicVar<N>;

    fn add(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) + rhs.into()
    }
}

impl<N: Field, RHS: Into<SymbolicVar<N>>> Sub<RHS> for Var<N> {
    type Output = SymbolicVar<N>;

    fn sub(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) - rhs.into()
    }
}

impl<N: PrimeField, RHS: Into<SymbolicVar<N>>> Add<RHS> for Usize<N> {
    type Output = SymbolicVar<N>;

    fn add(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) + rhs.into()
    }
}

impl<N: PrimeField, RHS: Into<SymbolicVar<N>>> Sub<RHS> for Usize<N> {
    type Output = SymbolicVar<N>;

    fn sub(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) - rhs.into()
    }
}

impl<N: PrimeField, RHS: Into<SymbolicVar<N>>> Mul<RHS> for Usize<N> {
    type Output = SymbolicVar<N>;

    fn mul(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) * rhs.into()
    }
}

impl<N: PrimeField> From<Usize<N>> for SymbolicVar<N> {
    fn from(value: Usize<N>) -> Self {
        match value {
            Usize::Const(n) => SymbolicVar::from(*n.borrow()),
            Usize::Var(n) => SymbolicVar::from(n),
        }
    }
}

impl<N: PrimeField, RHS: Into<SymbolicVar<N>>> Add<RHS> for RVar<N> {
    type Output = SymbolicVar<N>;

    fn add(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) + rhs.into()
    }
}

impl<N: PrimeField, RHS: Into<SymbolicVar<N>>> Sub<RHS> for RVar<N> {
    type Output = SymbolicVar<N>;

    fn sub(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) - rhs.into()
    }
}

impl<N: PrimeField, RHS: Into<SymbolicVar<N>>> Mul<RHS> for RVar<N> {
    type Output = SymbolicVar<N>;

    fn mul(self, rhs: RHS) -> Self::Output {
        SymbolicVar::from(self) * rhs.into()
    }
}

impl<N: Field> From<RVar<N>> for SymbolicVar<N> {
    fn from(value: RVar<N>) -> Self {
        match value {
            RVar::Const(n) => SymbolicVar::from(n),
            RVar::Val(n) => SymbolicVar::from(n),
        }
    }
}

impl<N: PrimeField> From<usize> for RVar<N> {
    fn from(value: usize) -> Self {
        Self::from_field(N::from_canonical_usize(value))
    }
}

impl<N: Field> From<Var<N>> for RVar<N> {
    fn from(value: Var<N>) -> Self {
        RVar::Val(value)
    }
}
