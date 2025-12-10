extern crate alloc;

use alloc::vec::Vec;
use core::{mem, ops};

use rand::Rng;

pub trait Generate {
    fn generate<R: Rng>(rng: &mut R) -> Self;
}

impl Generate for () {
    fn generate<R: Rng>(_: &mut R) -> Self {}
}

impl Generate for bool {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        rng.gen_bool(0.5)
    }
}

macro_rules! impl_generate {
    ($ty:ty) => {
        impl Generate for $ty {
            fn generate<R: Rng>(rng: &mut R) -> Self {
                rng.gen()
            }
        }
    };
}

impl_generate!(u8);
impl_generate!(u16);
impl_generate!(u32);
impl_generate!(u64);
impl_generate!(u128);
impl_generate!(usize);
impl_generate!(i8);
impl_generate!(i16);
impl_generate!(i32);
impl_generate!(i64);
impl_generate!(i128);
impl_generate!(isize);
impl_generate!(f32);
impl_generate!(f64);

macro_rules! impl_tuple {
    () => {};
    ($first:ident, $($rest:ident,)*) => {
        impl<$first: Generate, $($rest: Generate,)*> Generate for ($first, $($rest,)*) {
            fn generate<R: Rng>(rng: &mut R) -> Self {
                ($first::generate(rng), $($rest::generate(rng),)*)
            }
        }

        impl_tuple!($($rest,)*);
    };
}

impl_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,);

impl<T: Generate, const N: usize> Generate for [T; N] {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        let mut result = mem::MaybeUninit::<Self>::uninit();
        let result_ptr = result.as_mut_ptr().cast::<T>();
        for i in 0..N {
            unsafe {
                result_ptr.add(i).write(T::generate(rng));
            }
        }
        unsafe { result.assume_init() }
    }
}

impl<T: Generate> Generate for Option<T> {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        if rng.gen_bool(0.5) {
            Some(T::generate(rng))
        } else {
            None
        }
    }
}

pub fn generate_vec<R: Rng, T: Generate>(rng: &mut R, range: ops::Range<usize>) -> Vec<T> {
    let len = rng.gen_range(range);
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        result.push(T::generate(rng));
    }
    result
}
