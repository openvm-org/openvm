//! FFI functions for the ECC extension (Weierstrass EC point operations).
//!
//! Uses native halo2curves / blstrs field arithmetic for the 4 known curves,
//! reusing the `FieldArith` trait and `KnownPrimeField` types from the algebra
//! FFI crate.

use std::ffi::c_void;

use halo2curves_axiom::ff::Field;
use num_bigint::BigUint;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_ecc_circuit::{ec_add_ne_expr, ec_double_ne_expr};
use openvm_mod_circuit_builder::{run_field_expression_precomputed, ExprBuilderConfig};
use rvr_openvm_ext_algebra_ffi::{
    read_bls12_381_fq, read_field_256, write_bls12_381_fq, write_field_256, BLS12_381_ELEM_BYTES,
    FIELD_256_BYTES,
};
use rvr_openvm_ext_ffi_common::{rd_mem_words_traced, wr_mem_words_traced, WORD_SIZE};

/// Affine point: two field coordinates (x, y).
const AFFINE_COORDS: u32 = 2;
/// Size of a 256-bit affine point in bytes.
const POINT_256_BYTES: u32 = AFFINE_COORDS * FIELD_256_BYTES as u32;
/// Size of a BLS12-381 affine point in bytes.
const POINT_BLS12_381_BYTES: u32 = AFFINE_COORDS * BLS12_381_ELEM_BYTES as u32;

// ── Native EC formulas ────────────────────────────────────────────────────────

/// EC point addition (non-equal x-coords) using native field ops.
///
///   λ = (y2 − y1) / (x2 − x1)
///   x3 = λ² − x1 − x2
///   y3 = λ·(x1 − x3) − y1
///
/// Note: this opcode assumes `x1 != x2`.
#[inline(always)]
fn ec_add_ne_impl<F: Field>(x1: F, y1: F, x2: F, y2: F) -> (F, F) {
    let dx = x2 - x1;
    debug_assert!(!bool::from(dx.is_zero()));
    let lambda = (y2 - y1) * dx.invert().unwrap();
    let x3 = lambda.square() - x1 - x2;
    let y3 = lambda * (x1 - x3) - y1;
    (x3, y3)
}

/// EC point doubling using native field ops.
///
///   λ = (3·x1² + a) / (2·y1)
///   x3 = λ² − 2·x1
///   y3 = λ·(x1 − x3) − y1
///
/// Note: this opcode assumes `y1 != 0` and the input is not the point at infinity.
#[inline(always)]
fn ec_double_impl<F: Field>(x1: F, y1: F, a: F) -> (F, F) {
    let x1_sq = x1.square();
    let three_x1_sq = x1_sq.double() + x1_sq;
    let two_y1 = y1.double();
    debug_assert!(!bool::from(two_y1.is_zero()));
    let lambda = (three_x1_sq + a) * two_y1.invert().unwrap();
    let x3 = lambda.square() - x1.double();
    let y3 = lambda * (x1 - x3) - y1;
    (x3, y3)
}

// ── 256-bit curve helpers ─────────────────────────────────────────────────────

/// Execute ec_add_ne for a 256-bit PrimeField curve.
#[inline(always)]
unsafe fn ec_add_ne_256<F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>>(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
) {
    let x1: F = read_field_256(state, rs1_ptr);
    let y1: F = read_field_256(state, rs1_ptr + FIELD_256_BYTES as u32);
    let x2: F = read_field_256(state, rs2_ptr);
    let y2: F = read_field_256(state, rs2_ptr + FIELD_256_BYTES as u32);
    let (x3, y3) = ec_add_ne_impl(x1, y1, x2, y2);
    write_field_256(state, rd_ptr, &x3);
    write_field_256(state, rd_ptr + FIELD_256_BYTES as u32, &y3);
}

/// Execute ec_double for a 256-bit PrimeField curve.
#[inline(always)]
unsafe fn ec_double_256<F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>>(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    a: F,
) {
    let x1: F = read_field_256(state, rs1_ptr);
    let y1: F = read_field_256(state, rs1_ptr + FIELD_256_BYTES as u32);
    let (x3, y3) = ec_double_impl(x1, y1, a);
    write_field_256(state, rd_ptr, &x3);
    write_field_256(state, rd_ptr + FIELD_256_BYTES as u32, &y3);
}

// ── BLS12-381 helpers ─────────────────────────────────────────────────────────

unsafe fn ec_add_ne_bls12_381(state: *mut c_void, rd_ptr: u32, rs1_ptr: u32, rs2_ptr: u32) {
    let x1 = read_bls12_381_fq(state, rs1_ptr);
    let y1 = read_bls12_381_fq(state, rs1_ptr + BLS12_381_ELEM_BYTES as u32);
    let x2 = read_bls12_381_fq(state, rs2_ptr);
    let y2 = read_bls12_381_fq(state, rs2_ptr + BLS12_381_ELEM_BYTES as u32);
    let (x3, y3) = ec_add_ne_impl(x1, y1, x2, y2);
    write_bls12_381_fq(state, rd_ptr, &x3);
    write_bls12_381_fq(state, rd_ptr + BLS12_381_ELEM_BYTES as u32, &y3);
}

unsafe fn ec_double_bls12_381(state: *mut c_void, rd_ptr: u32, rs1_ptr: u32) {
    let x1 = read_bls12_381_fq(state, rs1_ptr);
    let y1 = read_bls12_381_fq(state, rs1_ptr + BLS12_381_ELEM_BYTES as u32);
    // BLS12-381 has a = 0
    let (x3, y3) = ec_double_impl(x1, y1, blstrs::Fp::ZERO);
    write_bls12_381_fq(state, rd_ptr, &x3);
    write_bls12_381_fq(state, rd_ptr + BLS12_381_ELEM_BYTES as u32, &y3);
}

// ── Curve constants ──────────────────────────────────────────────────────────

const P256_A_ABS: u64 = 3;

unsafe fn trace_read_bytes(state: *mut c_void, ptr: u32, len: u32) -> Vec<u8> {
    debug_assert_eq!(len % WORD_SIZE as u32, 0);
    let num_words = (len as usize) / WORD_SIZE;
    let mut words = vec![0u32; num_words];
    rd_mem_words_traced(state, ptr, &mut words);
    let mut bytes = Vec::with_capacity(len as usize);
    for &w in &words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

unsafe fn trace_write_bytes(state: *mut c_void, ptr: u32, bytes: &[u8]) {
    debug_assert_eq!(bytes.len() % WORD_SIZE, 0);
    let words: Vec<u32> = bytes
        .chunks_exact(WORD_SIZE)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    wr_mem_words_traced(state, ptr, &words);
}

fn ecc_setup_expr(point_bytes: u32, setup_bytes: &[u8], is_double: bool) -> Vec<u8> {
    let coord_bytes = (point_bytes / 2) as usize;
    let modulus = BigUint::from_bytes_le(&setup_bytes[..coord_bytes]);
    let config = ExprBuilderConfig {
        modulus,
        num_limbs: coord_bytes,
        limb_bits: 8,
    };
    let expr = if is_double {
        let a_biguint = BigUint::from_bytes_le(&setup_bytes[coord_bytes..point_bytes as usize]);
        ec_double_ne_expr(
            config,
            // TODO: use a real range bus here, or remove the requirement entirely;
            // OpenVM currently uses the same dummy bus.
            VariableRangeCheckerBus::new(u16::MAX, 16),
            a_biguint,
        )
    } else {
        ec_add_ne_expr(
            config,
            // TODO: use a real range bus here, or remove the requirement entirely;
            // OpenVM currently uses the same dummy bus.
            VariableRangeCheckerBus::new(u16::MAX, 16),
        )
    };
    let flag_idx = expr.num_flags();
    let writes = run_field_expression_precomputed::<true>(&expr, flag_idx, setup_bytes);
    writes.into()
}

unsafe fn ec_add_ne_setup(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
    point_bytes: u32,
) {
    let mut setup_bytes = trace_read_bytes(state, rs1_ptr, point_bytes);
    setup_bytes.extend_from_slice(&trace_read_bytes(state, rs2_ptr, point_bytes));
    let output = ecc_setup_expr(point_bytes, &setup_bytes, false);
    trace_write_bytes(state, rd_ptr, &output);
}

unsafe fn ec_double_setup(state: *mut c_void, rd_ptr: u32, rs1_ptr: u32, point_bytes: u32) {
    let setup_bytes = trace_read_bytes(state, rs1_ptr, point_bytes);
    let output = ecc_setup_expr(point_bytes, &setup_bytes, true);
    trace_write_bytes(state, rd_ptr, &output);
}

unsafe fn ec_add_ne_256_entry<
    F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>,
>(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
) {
    ec_add_ne_256::<F>(state, rd_ptr, rs1_ptr, rs2_ptr);
}

unsafe fn ec_double_256_entry<
    F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>,
>(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    a: F,
) {
    ec_double_256(state, rd_ptr, rs1_ptr, a);
}

macro_rules! ecc_add_ne_entry {
    ($name:ident, $curve:ty) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid affine point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u32,
            rs1_ptr: u32,
            rs2_ptr: u32,
        ) {
            ec_add_ne_256_entry::<$curve>(state, rd_ptr, rs1_ptr, rs2_ptr);
        }
    };
}

macro_rules! ecc_double_entry {
    ($name:ident, $curve:ty, $a:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid affine point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd_ptr: u32, rs1_ptr: u32) {
            ec_double_256_entry::<$curve>(state, rd_ptr, rs1_ptr, $a);
        }
    };
}

macro_rules! ecc_add_ne_setup_entry {
    ($name:ident, $point_bytes:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid affine point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u32,
            rs1_ptr: u32,
            rs2_ptr: u32,
        ) {
            ec_add_ne_setup(state, rd_ptr, rs1_ptr, rs2_ptr, $point_bytes);
        }
    };
}

macro_rules! ecc_double_setup_entry {
    ($name:ident, $point_bytes:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid affine point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd_ptr: u32, rs1_ptr: u32) {
            ec_double_setup(state, rd_ptr, rs1_ptr, $point_bytes);
        }
    };
}

// k256 add_ne and double are provided by rvr_ext_k256.c so these Rust
// entry points are intentionally not generated here.
ecc_add_ne_setup_entry!(rvr_ext_setup_ec_add_ne_k256, POINT_256_BYTES);
ecc_double_setup_entry!(rvr_ext_setup_ec_double_k256, POINT_256_BYTES);

ecc_add_ne_entry!(rvr_ext_ec_add_ne_p256, halo2curves_axiom::secp256r1::Fp);
ecc_add_ne_setup_entry!(rvr_ext_setup_ec_add_ne_p256, POINT_256_BYTES);
ecc_double_entry!(
    rvr_ext_ec_double_p256,
    halo2curves_axiom::secp256r1::Fp,
    -halo2curves_axiom::secp256r1::Fp::from(P256_A_ABS)
);
ecc_double_setup_entry!(rvr_ext_setup_ec_double_p256, POINT_256_BYTES);

ecc_add_ne_entry!(rvr_ext_ec_add_ne_bn254, halo2curves_axiom::bn256::Fq);
ecc_add_ne_setup_entry!(rvr_ext_setup_ec_add_ne_bn254, POINT_256_BYTES);
ecc_double_entry!(
    rvr_ext_ec_double_bn254,
    halo2curves_axiom::bn256::Fq,
    halo2curves_axiom::bn256::Fq::ZERO
);
ecc_double_setup_entry!(rvr_ext_setup_ec_double_bn254, POINT_256_BYTES);

/// # Safety
///
/// `state` must point to a valid native tracer state for this execution.
/// Pointer parameters must point to valid 48-byte BLS12-381 affine point
/// coordinates.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_ec_add_ne_bls12_381(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
) {
    ec_add_ne_bls12_381(state, rd_ptr, rs1_ptr, rs2_ptr);
}

ecc_add_ne_setup_entry!(rvr_ext_setup_ec_add_ne_bls12_381, POINT_BLS12_381_BYTES);

/// # Safety
///
/// `state` must point to a valid native tracer state for this execution.
/// Pointer parameters must point to valid 48-byte BLS12-381 affine point
/// coordinates.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_ec_double_bls12_381(
    state: *mut c_void,
    rd_ptr: u32,
    rs1_ptr: u32,
) {
    ec_double_bls12_381(state, rd_ptr, rs1_ptr);
}

ecc_double_setup_entry!(rvr_ext_setup_ec_double_bls12_381, POINT_BLS12_381_BYTES);
