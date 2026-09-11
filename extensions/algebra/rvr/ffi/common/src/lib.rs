//! Shared helpers for the algebra modular and fp2 FFI staticlibs.

use std::ffi::c_void;

use halo2curves_axiom::ff::{Field, PrimeField};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_ffi_common::{read_mem_words, write_mem_words};

/// Size of a 256-bit field element in bytes.
pub const FIELD_256_BYTES: usize = 32;
/// Number of 8-byte words in a 256-bit field element.
pub const FIELD_256_WORDS: usize = FIELD_256_BYTES / WORD_SIZE;
/// Size of a BLS12-381 base field element in bytes.
pub const BLS12_381_ELEM_BYTES: usize = 48;

// ── FieldArith trait ─────────────────────────────────────────────────────────

/// Arithmetic + I/O for a single element type, parameterized on a wrapper.
pub trait FieldArith {
    type Elem;

    /// # Safety
    /// `state` must be a valid pointer to the C `RvState` struct.
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem;
    /// # Safety
    /// `state` must be a valid pointer to the C `RvState` struct.
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem);

    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn sub(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn mul(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn div(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn is_eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool;
}

// ── KnownFieldArith trait ─────────────────────────────────────────────────────

/// Narrower trait for known (native) field types whose element type supports
/// standard arithmetic via operator overloads. Implementing this trait
/// automatically provides a full [`FieldArith`] impl via a blanket impl —
/// only `read_elem` and `write_elem` need to be supplied.
pub trait KnownFieldArith {
    type Elem: Field;

    /// # Safety
    /// `state` must be a valid pointer to the C `RvState` struct.
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem;
    /// # Safety
    /// `state` must be a valid pointer to the C `RvState` struct.
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem);
}

impl<T: KnownFieldArith> FieldArith for T {
    type Elem = T::Elem;

    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem {
        KnownFieldArith::read_elem(self, state, ptr)
    }

    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem) {
        KnownFieldArith::write_elem(self, state, ptr, val)
    }

    #[inline(always)]
    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a + b
    }
    #[inline(always)]
    fn sub(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a - b
    }
    #[inline(always)]
    fn mul(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a * b
    }
    #[inline(always)]
    fn div(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        a * b.invert().unwrap()
    }
    #[inline(always)]
    fn is_eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool {
        a == b
    }
}

// ── 256-bit field I/O ───────────────────────────────────────────────────────

/// Read a 256-bit field element through the VM memory interface.
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
pub unsafe fn read_field_256<F: PrimeField<Repr = [u8; 32]>>(state: *mut c_void, ptr: u64) -> F {
    let mut words = [0u64; FIELD_256_WORDS];
    read_mem_words(state, ptr, &mut words);
    let mut bytes = [0u8; FIELD_256_BYTES];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    F::from_repr_vartime(bytes).unwrap_or_else(|| {
        let modulus =
            BigUint::parse_bytes(F::MODULUS.trim_start_matches("0x").as_bytes(), 16).unwrap();
        let value = BigUint::from_bytes_le(&bytes);
        let reduced = value % modulus;
        let le = reduced.to_bytes_le();
        let mut reduced_bytes = [0u8; FIELD_256_BYTES];
        reduced_bytes[..le.len()].copy_from_slice(&le);
        F::from_repr_vartime(reduced_bytes).unwrap()
    })
}

/// Write a 256-bit field element through the VM memory interface.
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
pub unsafe fn write_field_256<F: PrimeField<Repr = [u8; 32]>>(
    state: *mut c_void,
    ptr: u64,
    val: &F,
) {
    let bytes = val.to_repr();
    let mut words = [0u64; FIELD_256_WORDS];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u64::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    write_mem_words(state, ptr, &words);
}

// ── BigUint helpers (for unknown-modulus fallbacks) ──────────────────────────

/// Convert a limb count in bytes to RV64 words.
#[inline]
pub fn limb_bytes_to_words(num_limbs: u32) -> u32 {
    assert_eq!(
        num_limbs % WORD_SIZE as u32,
        0,
        "limb byte count must be a multiple of WORD_SIZE"
    );
    num_limbs / WORD_SIZE as u32
}

/// Read a `num_limbs`-byte little-endian BigUint through the VM memory interface.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; `num_limbs` must be a multiple
/// of `WORD_SIZE`.
#[inline]
pub unsafe fn read_bigint(state: *mut c_void, ptr: u64, num_limbs: u32) -> BigUint {
    let num_words = limb_bytes_to_words(num_limbs) as usize;
    let mut words = vec![0u64; num_words];
    read_mem_words(state, ptr, &mut words);
    let mut bytes = vec![0u8; num_limbs as usize];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

/// Write a zero-padded BigUint through the VM memory interface.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; `num_limbs` must be a multiple
/// of `WORD_SIZE` and large enough to hold `value`.
#[inline]
pub unsafe fn write_bigint(state: *mut c_void, ptr: u64, value: &BigUint, num_limbs: u32) {
    let num_words = limb_bytes_to_words(num_limbs) as usize;
    let mut bytes = value.to_bytes_le();
    bytes.resize(num_limbs as usize, 0);
    let mut words = vec![0u64; num_words];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u64::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    write_mem_words(state, ptr, &words);
}

/// Modular inverse of `a` modulo `p` via extended GCD. Caller must ensure
/// `a != 0`.
#[inline]
pub fn mod_inverse(a: &BigUint, p: &BigUint) -> BigUint {
    debug_assert!(
        a != &BigUint::ZERO,
        "mod_inverse expects a non-zero element"
    );
    let result = BigInt::from(a.clone()).extended_gcd(&BigInt::from(p.clone()));
    result
        .x
        .mod_floor(&BigInt::from(p.clone()))
        .to_biguint()
        .unwrap()
}

// ── FFI generation macros (shared by modular and fp2) ───────────────────────

/// Generate a known-field FFI function. `$wrapper` must be a
/// `PhantomData`-newtype that implements [`KnownFieldArith`] for `$field`.
#[macro_export]
macro_rules! known_field_op_fn {
    ($name:ident, $wrapper:ident, $field:ty, $op:ident) => {
        /// # Safety
        /// `state` must be a valid `RvState` pointer.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut ::std::ffi::c_void,
            rd: u64,
            rs1: u64,
            rs2: u64,
        ) {
            let f = $wrapper::<$field>(::std::marker::PhantomData);
            $crate::exec_op(&f, state, rd, rs1, rs2, |f, a, b| {
                $crate::FieldArith::$op(f, a, b)
            });
        }
    };
}

/// Generate a generic (unknown-modulus) field-op FFI function.
///
/// `$field_ty` must be a struct with `modulus: BigUint` and `num_limbs: u32`
/// fields that implements [`FieldArith`].
#[macro_export]
macro_rules! unknown_field_op_fn {
    ($name:ident, $field_ty:ident, $op:ident) => {
        /// # Safety
        /// `state` must be a valid `RvState` pointer. `modulus_ptr` must point
        /// to `num_limbs` bytes.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut ::std::ffi::c_void,
            rd_ptr: u64,
            rs1_ptr: u64,
            rs2_ptr: u64,
            num_limbs: u32,
            modulus_ptr: *const u8,
        ) {
            let modulus = ::num_bigint::BigUint::from_bytes_le(::std::slice::from_raw_parts(
                modulus_ptr,
                num_limbs as ::std::primitive::usize,
            ));
            let f = $field_ty { modulus, num_limbs };
            $crate::exec_op(&f, state, rd_ptr, rs1_ptr, rs2_ptr, |f, a, b| {
                $crate::FieldArith::$op(f, a, b)
            });
        }
    };
}

// ── Instruction execution helper ─────────────────────────────────────────────

/// Read both operands, apply `op`, write the result.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; the three pointers must be valid
/// for the element size of `F::Elem`.
#[inline(always)]
pub unsafe fn exec_op<F, Op>(
    f: &F,
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
    op: Op,
) where
    F: FieldArith,
    Op: FnOnce(&F, F::Elem, F::Elem) -> F::Elem,
{
    let a = f.read_elem(state, rs1_ptr);
    let b = f.read_elem(state, rs2_ptr);
    let result = op(f, a, b);
    f.write_elem(state, rd_ptr, &result);
}
