use p3_air::VirtualPairCol;
#[cfg(any(feature = "test-traits", test))]
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};

// TODO: Ideally upstream PrimeField implements From<T>
pub trait FieldFrom<T> {
    fn from_val(value: T) -> Self;
}

#[cfg(any(feature = "test-traits", test))]
impl FieldFrom<u8> for BabyBear {
    fn from_val(value: u8) -> Self {
        BabyBear::from_canonical_u8(value)
    }
}

#[cfg(any(feature = "test-traits", test))]
impl FieldFrom<BabyBear> for BabyBear {
    fn from_val(value: BabyBear) -> Self {
        value
    }
}

pub fn to_vcols<F: Field>(cols: &[usize]) -> Vec<VirtualPairCol<F>> {
    cols.iter()
        .copied()
        .map(VirtualPairCol::single_main)
        .collect()
}

pub fn fill_slc_to_f<F: Field>(dest: &mut [F], src: &[u32]) {
    dest.iter_mut()
        .zip(src.iter())
        .for_each(|(d, s)| *d = F::from_canonical_u32(*s));
}

pub fn to_field_vec<F: Field>(src: &[u32]) -> Vec<F> {
    src.iter().map(|s| F::from_canonical_u32(*s)).collect()
}

pub fn not<F: AbstractField>(a: F) -> F {
    F::one() - a
}

pub fn and<F: AbstractField>(a: F, b: F) -> F {
    a * b
}

/// Assumes that a and b are boolean
pub fn or<F: AbstractField>(a: F, b: F) -> F {
    a.clone() + b.clone() - and(a, b)
}

/// Assumes that a and b are boolean
pub fn implies<F: AbstractField>(a: F, b: F) -> F {
    or(F::one() - a, b)
}
