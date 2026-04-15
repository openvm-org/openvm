//! FFI functions for the algebra extension (modular arithmetic + Fp2).
//!
//! Four field struct types implement the unified `FieldArith` trait:
//!   - `KnownPrimeField<F>`   — known prime field, generic on `F`
//!   - `KnownComplexField<F>` — known complex (Fp2) field, generic on `F`
//!   - `UnknownPrimeField`    — unknown prime field, carries modulus
//!   - `UnknownComplexField`  — unknown complex field, carries modulus

use std::ffi::c_void;
use std::marker::PhantomData;

use halo2curves_axiom::ff::{Field, PrimeField};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::One;
use rvr_openvm_ext_ffi_common::{
    ext_hint_stream_set, rd_mem_u32_range_wrapper, rd_mem_words_traced, trace_mem_access_range,
    wr_mem_words_traced, AS_MEMORY, WORD_SIZE,
};

/// Size of a 256-bit field element in bytes.
pub const FIELD_256_BYTES: usize = 32;
/// Number of 4-byte words in a 256-bit field element.
pub const FIELD_256_WORDS: usize = FIELD_256_BYTES / WORD_SIZE;

/// Size of a BLS12-381 Fq element in bytes.
pub const BLS12_381_ELEM_BYTES: usize = 48;
/// Number of 4-byte words in a BLS12-381 Fq element.
pub const BLS12_381_ELEM_WORDS: usize = BLS12_381_ELEM_BYTES / WORD_SIZE;

// ── FieldArith trait ─────────────────────────────────────────────────────────

pub trait FieldArith {
    type Elem;

    /// # Safety
    /// `state` must be a valid pointer to the C `RvState` struct.
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem;
    /// # Safety
    /// `state` must be a valid pointer to the C `RvState` struct.
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem);

    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn sub(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn mul(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn div(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem;
    fn is_eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool;
}

// ── Field structs ────────────────────────────────────────────────────────────

pub struct KnownPrimeField<F>(pub PhantomData<F>);
struct KnownComplexField<F>(PhantomData<F>);

struct UnknownPrimeField {
    modulus: BigUint,
    num_limbs: u32,
}

struct UnknownComplexField {
    modulus: BigUint,
    num_limbs: u32,
}

// ── I/O helpers ──────────────────────────────────────────────────────────────

/// Read a 256-bit field element from guest memory (traced).
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
pub unsafe fn read_field_256<F: PrimeField<Repr = [u8; 32]>>(state: *mut c_void, ptr: u32) -> F {
    let mut words = [0u32; FIELD_256_WORDS];
    rd_mem_words_traced(state, ptr, &mut words);
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

/// Write a 256-bit field element to guest memory (traced).
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
pub unsafe fn write_field_256<F: PrimeField<Repr = [u8; 32]>>(
    state: *mut c_void,
    ptr: u32,
    val: &F,
) {
    let bytes = val.to_repr();
    let mut words = [0u32; FIELD_256_WORDS];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u32::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    wr_mem_words_traced(state, ptr, &words);
}

/// Read a BLS12-381 Fq element (48 bytes) from guest memory (traced).
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
pub unsafe fn read_bls12_381_fq(state: *mut c_void, ptr: u32) -> blstrs::Fp {
    let mut words = [0u32; BLS12_381_ELEM_WORDS];
    rd_mem_words_traced(state, ptr, &mut words);
    let mut bytes = [0u8; BLS12_381_ELEM_BYTES];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    blstrs::Fp::from_bytes_le(&bytes).unwrap_or_else(|| {
        let modulus = BigUint::from_bytes_le(&blstrs::Fp::char());
        let value = BigUint::from_bytes_le(&bytes);
        let reduced = value % modulus;
        let le = reduced.to_bytes_le();
        let mut reduced_bytes = [0u8; BLS12_381_ELEM_BYTES];
        reduced_bytes[..le.len()].copy_from_slice(&le);
        blstrs::Fp::from_bytes_le(&reduced_bytes).unwrap()
    })
}

/// Write a BLS12-381 Fq element (48 bytes) to guest memory (traced).
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
pub unsafe fn write_bls12_381_fq(state: *mut c_void, ptr: u32, val: &blstrs::Fp) {
    let bytes = val.to_bytes_le();
    let mut words = [0u32; BLS12_381_ELEM_WORDS];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u32::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    wr_mem_words_traced(state, ptr, &words);
}

// ── BigUint helpers ──────────────────────────────────────────────────────────

unsafe fn read_bigint(state: *mut c_void, ptr: u32, num_limbs: u32) -> BigUint {
    let num_words = (num_limbs / WORD_SIZE as u32) as usize;
    let mut words = vec![0u32; num_words];
    rd_mem_words_traced(state, ptr, &mut words);
    let mut bytes = vec![0u8; num_limbs as usize];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

unsafe fn write_bigint(state: *mut c_void, ptr: u32, value: &BigUint, num_limbs: u32) {
    let num_words = (num_limbs / WORD_SIZE as u32) as usize;
    let mut bytes = value.to_bytes_le();
    bytes.resize(num_limbs as usize, 0);
    let mut words = vec![0u32; num_words];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u32::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    wr_mem_words_traced(state, ptr, &words);
}

fn mod_inverse(a: &BigUint, p: &BigUint) -> BigUint {
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

// ── KnownPrimeField impl (256-bit via generic bound) ────────────────────────

impl<F: PrimeField<Repr = [u8; 32]>> FieldArith for KnownPrimeField<F> {
    type Elem = F;
    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem {
        read_field_256(state, ptr)
    }
    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem) {
        write_field_256(state, ptr, val)
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

// ── KnownPrimeField impl (BLS12-381 Fq, 48 bytes) ──────────────────────────
//
// blstrs::Fp uses a different `ff` crate version than halo2curves_axiom, so it
// doesn't actually impl halo2curves_axiom::ff::PrimeField. However the compiler
// can't prove that, so a concrete impl on KnownPrimeField<blstrs::Fp> would
// conflict with the generic one above. We use a newtype to sidestep coherence.

pub struct Bls12381Fq;

impl FieldArith for KnownPrimeField<Bls12381Fq> {
    type Elem = blstrs::Fp;
    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem {
        read_bls12_381_fq(state, ptr)
    }
    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem) {
        write_bls12_381_fq(state, ptr, val)
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

// ── KnownComplexField impls ─────────────────────────────────────────────────

impl FieldArith for KnownComplexField<halo2curves_axiom::bn256::Fq2> {
    type Elem = halo2curves_axiom::bn256::Fq2;

    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem {
        halo2curves_axiom::bn256::Fq2::new(
            read_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr),
            read_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr + FIELD_256_BYTES as u32),
        )
    }

    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem) {
        write_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr, &val.c0);
        write_field_256::<halo2curves_axiom::bn256::Fq>(
            state,
            ptr + FIELD_256_BYTES as u32,
            &val.c1,
        );
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

impl FieldArith for KnownComplexField<blstrs::Fp2> {
    type Elem = blstrs::Fp2;

    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem {
        blstrs::Fp2::new(
            read_bls12_381_fq(state, ptr),
            read_bls12_381_fq(state, ptr + BLS12_381_ELEM_BYTES as u32),
        )
    }

    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem) {
        write_bls12_381_fq(state, ptr, &val.c0());
        write_bls12_381_fq(state, ptr + BLS12_381_ELEM_BYTES as u32, &val.c1());
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

// ── UnknownPrimeField impl ──────────────────────────────────────────────────

impl FieldArith for UnknownPrimeField {
    type Elem = BigUint;

    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem {
        read_bigint(state, ptr, self.num_limbs)
    }

    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem) {
        write_bigint(state, ptr, val, self.num_limbs)
    }

    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        (a + b) % &self.modulus
    }

    fn sub(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        (a + &self.modulus - b) % &self.modulus
    }

    fn mul(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        (a * b) % &self.modulus
    }

    fn div(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let b_inv = mod_inverse(&b, &self.modulus);
        (a * b_inv) % &self.modulus
    }

    fn is_eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool {
        a % &self.modulus == b % &self.modulus
    }
}

// ── UnknownComplexField impl ────────────────────────────────────────────────

impl FieldArith for UnknownComplexField {
    type Elem = (BigUint, BigUint);

    unsafe fn read_elem(&self, state: *mut c_void, ptr: u32) -> Self::Elem {
        let c0 = read_bigint(state, ptr, self.num_limbs);
        let c1 = read_bigint(state, ptr + self.num_limbs, self.num_limbs);
        (c0, c1)
    }

    unsafe fn write_elem(&self, state: *mut c_void, ptr: u32, val: &Self::Elem) {
        write_bigint(state, ptr, &val.0, self.num_limbs);
        write_bigint(state, ptr + self.num_limbs, &val.1, self.num_limbs);
    }

    fn add(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let p = &self.modulus;
        ((&a.0 + &b.0) % p, (&a.1 + &b.1) % p)
    }

    fn sub(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let p = &self.modulus;
        ((&a.0 + p - &b.0) % p, (&a.1 + p - &b.1) % p)
    }

    fn mul(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let p = &self.modulus;
        // Add p * p before subtraction so the intermediate stays non-negative.
        let c0 = (&a.0 * &b.0 + p * p - &a.1 * &b.1) % p;
        let c1 = (&a.0 * &b.1 + &a.1 * &b.0) % p;
        (c0, c1)
    }

    fn div(&self, a: Self::Elem, b: Self::Elem) -> Self::Elem {
        let p = &self.modulus;
        let norm = (&b.0 * &b.0 + &b.1 * &b.1) % p;
        let norm_inv = mod_inverse(&norm, p);
        let bi0 = (&b.0 * &norm_inv) % p;
        let bi1 = ((p - &b.1) * &norm_inv) % p;
        // Add p * p before subtraction so the intermediate stays non-negative.
        let c0 = (&a.0 * &bi0 + p * p - &a.1 * &bi1) % p;
        let c1 = (&a.0 * &bi1 + &a.1 * &bi0) % p;
        (c0, c1)
    }

    fn is_eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool {
        let p = &self.modulus;
        &a.0 % p == &b.0 % p && &a.1 % p == &b.1 % p
    }
}

// ── Instruction execution helpers ────────────────────────────────────────────

/// Read both operands, apply `op`, write the result. The `op` is monomorphised
/// per call site so there is no runtime dispatch.
#[inline(always)]
unsafe fn exec_op<F, Op>(f: &F, state: *mut c_void, rd_ptr: u32, rs1_ptr: u32, rs2_ptr: u32, op: Op)
where
    F: FieldArith,
    Op: FnOnce(&F, F::Elem, F::Elem) -> F::Elem,
{
    let a = f.read_elem(state, rs1_ptr);
    let b = f.read_elem(state, rs2_ptr);
    let result = op(f, a, b);
    f.write_elem(state, rd_ptr, &result);
}

#[inline(always)]
unsafe fn exec_iseq<F: FieldArith>(f: &F, state: *mut c_void, rs1_ptr: u32, rs2_ptr: u32) -> u32 {
    let a = f.read_elem(state, rs1_ptr);
    let b = f.read_elem(state, rs2_ptr);
    if f.is_eq(&a, &b) {
        1
    } else {
        0
    }
}

// ── FFI generation macros ────────────────────────────────────────────────────

/// Generate one specialized `extern "C"` arithmetic entry point for a known
/// field. `$wrap` selects `KnownPrimeField` or `KnownComplexField`, and `$op`
/// is the [`FieldArith`] method invoked on the field.
macro_rules! field_op_fn {
    ($name:ident, $wrap:ident, $field:ty, $op:ident) => {
        /// # Safety
        /// `state` must be a valid `RvState` pointer.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd: u32, rs1: u32, rs2: u32) {
            let f = $wrap::<$field>(PhantomData);
            exec_op(&f, state, rd, rs1, rs2, |f, a, b| f.$op(a, b));
        }
    };
}

macro_rules! define_mod_ffi {
    ($field:ty, $suffix:ident) => {
        paste::paste! {
            field_op_fn!([<rvr_ext_mod_add_ $suffix>], KnownPrimeField, $field, add);
            field_op_fn!([<rvr_ext_mod_sub_ $suffix>], KnownPrimeField, $field, sub);
            field_op_fn!([<rvr_ext_mod_mul_ $suffix>], KnownPrimeField, $field, mul);
            field_op_fn!([<rvr_ext_mod_div_ $suffix>], KnownPrimeField, $field, div);
            /// # Safety
            /// `state` must be a valid `RvState` pointer.
            #[no_mangle]
            pub unsafe extern "C" fn [<rvr_ext_mod_iseq_ $suffix>](
                state: *mut c_void, rs1: u32, rs2: u32,
            ) -> u32 { exec_iseq(&KnownPrimeField::<$field>(PhantomData), state, rs1, rs2) }
        }
    };
}

macro_rules! define_fp2_ffi {
    ($field:ty, $suffix:ident) => {
        paste::paste! {
            field_op_fn!([<rvr_ext_fp2_add_ $suffix>], KnownComplexField, $field, add);
            field_op_fn!([<rvr_ext_fp2_sub_ $suffix>], KnownComplexField, $field, sub);
            field_op_fn!([<rvr_ext_fp2_mul_ $suffix>], KnownComplexField, $field, mul);
            field_op_fn!([<rvr_ext_fp2_div_ $suffix>], KnownComplexField, $field, div);
        }
    };
}

// ── Specialized per-curve FFI instantiations ─────────────────────────────────

// k256_coord and k256_scalar are provided by rvr_ext_k256.c so these Rust
// entry points are intentionally not generated here.
define_mod_ffi!(halo2curves_axiom::secp256r1::Fp, p256_coord);
define_mod_ffi!(halo2curves_axiom::secp256r1::Fq, p256_scalar);
define_mod_ffi!(halo2curves_axiom::bn256::Fq, bn254_fq);
define_mod_ffi!(halo2curves_axiom::bn256::Fr, bn254_fr);
define_mod_ffi!(Bls12381Fq, bls12_381_fq);
define_mod_ffi!(halo2curves_axiom::bls12_381::Fr, bls12_381_fr);

define_fp2_ffi!(halo2curves_axiom::bn256::Fq2, bn254);
define_fp2_ffi!(blstrs::Fp2, bls12_381);

// ── Generic FFI (fallback for unknown moduli) ────────────────────────────────

/// Generate one specialized `extern "C"` arithmetic entry point over an
/// unknown field carried at runtime. `$wrap` is `UnknownPrimeField` or
/// `UnknownComplexField`; `$op` is the [`FieldArith`] method.
macro_rules! unknown_field_op_fn {
    ($name:ident, $wrap:ident, $op:ident) => {
        /// # Safety
        /// `state` must be a valid `RvState` pointer. `modulus_ptr` must point
        /// to `num_limbs` bytes.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u32,
            rs1_ptr: u32,
            rs2_ptr: u32,
            num_limbs: u32,
            modulus_ptr: *const u8,
        ) {
            let modulus =
                BigUint::from_bytes_le(std::slice::from_raw_parts(modulus_ptr, num_limbs as usize));
            let f = $wrap { modulus, num_limbs };
            exec_op(&f, state, rd_ptr, rs1_ptr, rs2_ptr, |f, a, b| f.$op(a, b));
        }
    };
}

unknown_field_op_fn!(rvr_ext_mod_add, UnknownPrimeField, add);
unknown_field_op_fn!(rvr_ext_mod_sub, UnknownPrimeField, sub);
unknown_field_op_fn!(rvr_ext_mod_mul, UnknownPrimeField, mul);
unknown_field_op_fn!(rvr_ext_mod_div, UnknownPrimeField, div);

/// # Safety
/// `state` must be a valid `RvState` pointer. `modulus_ptr` must point to `num_limbs` bytes.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_mod_iseq(
    state: *mut c_void,
    rs1_ptr: u32,
    rs2_ptr: u32,
    num_limbs: u32,
    modulus_ptr: *const u8,
) -> u32 {
    let modulus =
        BigUint::from_bytes_le(std::slice::from_raw_parts(modulus_ptr, num_limbs as usize));
    let f = UnknownPrimeField { modulus, num_limbs };
    exec_iseq(&f, state, rs1_ptr, rs2_ptr)
}

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_mod_setup(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
    num_limbs: u32,
) {
    let num_words = num_limbs / WORD_SIZE as u32;
    debug_assert!(num_words >= 1);

    // Source comes from rs1 (we pass it through to rd). rs2 isn't used as
    // data, only its memory access needs to be traced.
    let mut src_words = vec![0u32; num_words as usize];
    rd_mem_words_traced(state, rs1_ptr, &mut src_words);
    trace_mem_access_range(state, rs2_ptr, num_words, AS_MEMORY);
    wr_mem_words_traced(state, rd_ptr, &src_words);
}

unknown_field_op_fn!(rvr_ext_fp2_add, UnknownComplexField, add);
unknown_field_op_fn!(rvr_ext_fp2_sub, UnknownComplexField, sub);
unknown_field_op_fn!(rvr_ext_fp2_mul, UnknownComplexField, mul);
unknown_field_op_fn!(rvr_ext_fp2_div, UnknownComplexField, div);

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_fp2_setup(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
    num_limbs: u32,
) {
    let total_limbs = num_limbs * 2;
    let num_words = total_limbs / WORD_SIZE as u32;
    debug_assert!(num_words >= 1);

    let mut src_words = vec![0u32; num_words as usize];
    rd_mem_words_traced(state, rs1_ptr, &mut src_words);
    trace_mem_access_range(state, rs2_ptr, num_words, AS_MEMORY);
    wr_mem_words_traced(state, rd_ptr, &src_words);
}

// ── Phantom instructions ─────────────────────────────────────────────────────

fn mod_sqrt_impl(x: &BigUint, modulus: &BigUint, non_qr: &BigUint) -> Option<BigUint> {
    if modulus % 4u32 == BigUint::from(3u8) {
        let exponent = (modulus + BigUint::one()) >> 2;
        let ret = x.modpow(&exponent, modulus);
        if &ret * &ret % modulus == x % modulus {
            Some(ret)
        } else {
            None
        }
    } else {
        let mut q = modulus - BigUint::one();
        let mut s = 0u32;
        while &q % 2u32 == BigUint::ZERO {
            s += 1;
            q /= 2u32;
        }
        let mut m = s;
        let mut c = non_qr.modpow(&q, modulus);
        let mut t = x.modpow(&q, modulus);
        let mut r = x.modpow(&((&q + BigUint::one()) >> 1), modulus);
        loop {
            if t == BigUint::ZERO {
                return Some(BigUint::ZERO);
            }
            if t == BigUint::one() {
                return Some(r);
            }
            let mut i = 0u32;
            let mut tmp = t.clone();
            while tmp != BigUint::one() && i < m {
                tmp = &tmp * &tmp % modulus;
                i += 1;
            }
            if i == m {
                return None;
            }
            for _ in 0..m - i - 1 {
                c = &c * &c % modulus;
            }
            let b = c;
            m = i;
            c = &b * &b % modulus;
            t = ((t * &b % modulus) * &b) % modulus;
            r = (r * b) % modulus;
        }
    }
}

/// # Safety
/// `state` must be a valid `RvState` pointer. `modulus_ptr` and `non_qr_ptr`
/// must point to `num_limbs` valid bytes each.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_algebra_hint_sqrt(
    state: *mut c_void,
    rs1_ptr: u32,
    num_limbs: u32,
    modulus_ptr: *const u8,
    non_qr_ptr: *const u8,
) {
    let modulus =
        BigUint::from_bytes_le(std::slice::from_raw_parts(modulus_ptr, num_limbs as usize));
    let non_qr = BigUint::from_bytes_le(std::slice::from_raw_parts(non_qr_ptr, num_limbs as usize));

    let num_words = (num_limbs / WORD_SIZE as u32) as usize;
    let mut words = vec![0u32; num_words];
    rd_mem_u32_range_wrapper(state, rs1_ptr, words.as_mut_ptr(), num_words as u32);
    // Phantom: untraced read.
    let mut x_bytes = vec![0u8; num_limbs as usize];
    for (i, &w) in words.iter().enumerate() {
        x_bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    let x = BigUint::from_bytes_le(&x_bytes);

    let (success, sqrt) = match mod_sqrt_impl(&x, &modulus, &non_qr) {
        Some(sqrt) => (true, sqrt),
        None => {
            let sqrt = mod_sqrt_impl(&(&x * &non_qr % &modulus), &modulus, &non_qr)
                .expect("Either x or x * non_qr should be a square");
            (false, sqrt)
        }
    };

    let mut hint = vec![0u8; WORD_SIZE + num_limbs as usize];
    hint[0] = if success { 1 } else { 0 };
    let sqrt_bytes = sqrt.to_bytes_le();
    hint[WORD_SIZE..WORD_SIZE + sqrt_bytes.len()].copy_from_slice(&sqrt_bytes);

    ext_hint_stream_set(hint.as_ptr(), hint.len() as u32);
}
