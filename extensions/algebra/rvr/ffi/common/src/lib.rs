//! Shared helpers for the algebra modular and fp2 FFI staticlibs.

use std::ffi::c_void;

use halo2curves_axiom::ff::PrimeField;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use rvr_openvm_ext_ffi_common::{rd_mem_words_traced, wr_mem_words_traced, WORD_SIZE};

/// Size of a 256-bit field element in bytes.
pub const FIELD_256_BYTES: usize = 32;
/// Number of 4-byte words in a 256-bit field element.
pub const FIELD_256_WORDS: usize = FIELD_256_BYTES / WORD_SIZE;

/// Size of a BLS12-381 Fq element in bytes.
pub const BLS12_381_ELEM_BYTES: usize = 48;
/// Number of 4-byte words in a BLS12-381 Fq element.
pub const BLS12_381_ELEM_WORDS: usize = BLS12_381_ELEM_BYTES / WORD_SIZE;

// ── FieldArith trait ─────────────────────────────────────────────────────────

/// Arithmetic + I/O for a single element type, parameterized on a wrapper.
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

// ── 256-bit / 384-bit field I/O ─────────────────────────────────────────────

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

// ── BigUint helpers (for unknown-modulus fallbacks) ──────────────────────────

/// Read a `num_limbs`-byte little-endian BigUint from guest memory (traced).
///
/// # Safety
/// `state` must be a valid `RvState` pointer; `num_limbs` must be a multiple
/// of `WORD_SIZE`.
#[inline]
pub unsafe fn read_bigint(state: *mut c_void, ptr: u32, num_limbs: u32) -> BigUint {
    let num_words = (num_limbs / WORD_SIZE as u32) as usize;
    let mut words = vec![0u32; num_words];
    rd_mem_words_traced(state, ptr, &mut words);
    let mut bytes = vec![0u8; num_limbs as usize];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

/// Write a BigUint to guest memory (traced), zero-padded to `num_limbs` bytes.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; `num_limbs` must be a multiple
/// of `WORD_SIZE` and large enough to hold `value`.
#[inline]
pub unsafe fn write_bigint(state: *mut c_void, ptr: u32, value: &BigUint, num_limbs: u32) {
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
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
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
