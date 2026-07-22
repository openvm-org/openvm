//! Projective short-Weierstrass point add/double, per curve.
//!
//! Each operation delegates to the circuit's native projective formulas
//! (`openvm_ecc_circuit::curves`) so rvr execution matches the circuit exactly:
//!   - a = 0 curves (bn254, bls12_381): Algorithms 7/9,
//!   - general a (p256): Algorithms 1/3.
//!
//! Callers use the per-curve convenience functions (`ec_add_proj_bn254`,
//! `ec_double_proj_p256`, …); the `a`/`3b` curve constants are encapsulated
//! here. k256 add/double are not defined here — they come from libsecp256k1 in
//! the modular staticlib.

use std::ffi::c_void;

use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use openvm_ecc_circuit::curves::{
    ec_add_proj_impl_a0, ec_add_proj_impl_general, ec_double_proj_impl_a0,
    ec_double_proj_impl_general,
};
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_algebra_ffi_common::{
    read_field_256, write_field_256, BLS12_381_ELEM_BYTES, FIELD_256_BYTES,
};
use rvr_openvm_ext_ffi_common::{read_mem_words, write_mem_words};

/// Number of RV64 words in a BLS12-381 base field element.
const BLS12_381_ELEM_WORDS: usize = BLS12_381_ELEM_BYTES / WORD_SIZE;

/// Read a BLS12-381 base field element through the VM memory interface.
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
unsafe fn read_bls12_381_fq(state: *mut c_void, ptr: u64) -> blstrs::Fp {
    let mut words = [0u64; BLS12_381_ELEM_WORDS];
    read_mem_words(state, ptr, &mut words);
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

/// Write a BLS12-381 base field element through the VM memory interface.
///
/// # Safety
/// `state` must be a valid pointer to the C `RvState` struct.
#[inline(always)]
unsafe fn write_bls12_381_fq(state: *mut c_void, ptr: u64, val: &blstrs::Fp) {
    let bytes = val.to_bytes_le();
    let mut words = [0u64; BLS12_381_ELEM_WORDS];
    for (i, w) in words.iter_mut().enumerate() {
        *w = u64::from_le_bytes(
            bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE]
                .try_into()
                .unwrap(),
        );
    }
    write_mem_words(state, ptr, &words);
}

/// `|a|` for P256 (a = -3).
const P256_A_ABS: u64 = 3;
/// `3b` coefficient for BN254 (b = 3).
const BN254_B3: u64 = 9;
/// `3b` coefficient for BLS12-381 (b = 4).
const BLS12_381_B3: u64 = 12;

/// Byte stride between the coordinates of a 256-bit projective point.
const COORD_256: u64 = FIELD_256_BYTES as u64;
/// Byte stride between the coordinates of a BLS12-381 projective point.
const COORD_BLS12_381: u64 = BLS12_381_ELEM_BYTES as u64;

// TODO: precompute the constant
/// `3b` for P256, computed as `3 * b` from the standard `b` constant.
#[inline(always)]
fn p256_b3<F: PrimeField<Repr = [u8; FIELD_256_BYTES]>>() -> F {
    let b_bytes: [u8; FIELD_256_BYTES] =
        hex_literal::hex!("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");
    let mut le_bytes = b_bytes;
    le_bytes.reverse();
    let b = F::from_repr(le_bytes).unwrap();
    b + b + b
}

// A projective point is three contiguous coordinates (X, Y, Z) in memory.

#[inline(always)]
unsafe fn read_proj_256<F: PrimeField<Repr = [u8; FIELD_256_BYTES]>>(
    state: *mut c_void,
    ptr: u64,
) -> (F, F, F) {
    (
        read_field_256(state, ptr),
        read_field_256(state, ptr + COORD_256),
        read_field_256(state, ptr + 2 * COORD_256),
    )
}

#[inline(always)]
unsafe fn write_proj_256<F: PrimeField<Repr = [u8; FIELD_256_BYTES]>>(
    state: *mut c_void,
    ptr: u64,
    (x, y, z): (F, F, F),
) {
    write_field_256(state, ptr, &x);
    write_field_256(state, ptr + COORD_256, &y);
    write_field_256(state, ptr + 2 * COORD_256, &z);
}

#[inline(always)]
unsafe fn read_proj_bls12_381(
    state: *mut c_void,
    ptr: u64,
) -> (blstrs::Fp, blstrs::Fp, blstrs::Fp) {
    (
        read_bls12_381_fq(state, ptr),
        read_bls12_381_fq(state, ptr + COORD_BLS12_381),
        read_bls12_381_fq(state, ptr + 2 * COORD_BLS12_381),
    )
}

#[inline(always)]
unsafe fn write_proj_bls12_381(
    state: *mut c_void,
    ptr: u64,
    (x, y, z): (blstrs::Fp, blstrs::Fp, blstrs::Fp),
) {
    write_bls12_381_fq(state, ptr, &x);
    write_bls12_381_fq(state, ptr + COORD_BLS12_381, &y);
    write_bls12_381_fq(state, ptr + 2 * COORD_BLS12_381, &z);
}

// ── P256 (general a = -3) ───────────────────────────────────────────────────

pub(crate) unsafe fn ec_add_proj_p256(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64, rs2_ptr: u64) {
    type Fp = halo2curves_axiom::secp256r1::Fp;
    let (x1, y1, z1) = read_proj_256::<Fp>(state, rs1_ptr);
    let (x2, y2, z2) = read_proj_256::<Fp>(state, rs2_ptr);
    let a = -Fp::from(P256_A_ABS);
    let out = ec_add_proj_impl_general(x1, y1, z1, x2, y2, z2, a, p256_b3::<Fp>());
    write_proj_256(state, rd_ptr, out);
}

pub(crate) unsafe fn ec_double_proj_p256(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) {
    type Fp = halo2curves_axiom::secp256r1::Fp;
    let (x1, y1, z1) = read_proj_256::<Fp>(state, rs1_ptr);
    let a = -Fp::from(P256_A_ABS);
    let out = ec_double_proj_impl_general(x1, y1, z1, a, p256_b3::<Fp>());
    write_proj_256(state, rd_ptr, out);
}

// ── BN254 (a = 0) ───────────────────────────────────────────────────────────

pub(crate) unsafe fn ec_add_proj_bn254(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
) {
    type Fq = halo2curves_axiom::bn256::Fq;
    let (x1, y1, z1) = read_proj_256::<Fq>(state, rs1_ptr);
    let (x2, y2, z2) = read_proj_256::<Fq>(state, rs2_ptr);
    let out = ec_add_proj_impl_a0(x1, y1, z1, x2, y2, z2, Fq::from(BN254_B3));
    write_proj_256(state, rd_ptr, out);
}

pub(crate) unsafe fn ec_double_proj_bn254(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) {
    type Fq = halo2curves_axiom::bn256::Fq;
    let (x1, y1, z1) = read_proj_256::<Fq>(state, rs1_ptr);
    let out = ec_double_proj_impl_a0(x1, y1, z1, Fq::from(BN254_B3));
    write_proj_256(state, rd_ptr, out);
}

// ── BLS12-381 (a = 0) ───────────────────────────────────────────────────────

pub(crate) unsafe fn ec_add_proj_bls12_381(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
) {
    let (x1, y1, z1) = read_proj_bls12_381(state, rs1_ptr);
    let (x2, y2, z2) = read_proj_bls12_381(state, rs2_ptr);
    let out = ec_add_proj_impl_a0(x1, y1, z1, x2, y2, z2, blstrs::Fp::from(BLS12_381_B3));
    write_proj_bls12_381(state, rd_ptr, out);
}

pub(crate) unsafe fn ec_double_proj_bls12_381(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) {
    let (x1, y1, z1) = read_proj_bls12_381(state, rs1_ptr);
    let out = ec_double_proj_impl_a0(x1, y1, z1, blstrs::Fp::from(BLS12_381_B3));
    write_proj_bls12_381(state, rd_ptr, out);
}
