//! Rust FFI for P-256 and BN254 point operations and setup operations for all
//! supported curves.
//!
//! Point operations use `halo2curves_axiom`. Setup operations evaluate
//! OpenVM's precomputed field expressions.

use std::{ffi::c_void, iter, sync::LazyLock};

use halo2curves_axiom::ff::Field;
use openvm_circuit_primitives::U16_BITS;
use openvm_ecc_circuit::{ec_add_ne_program, ec_double_ne_program, CurveType};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, ExprBuilderConfig, FieldExpressionProgram,
};
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_algebra_ffi_common::{
    read_field_256, write_field_256, BLS12_381_ELEM_BYTES, FIELD_256_BYTES,
};
use rvr_openvm_ext_ffi_common::{read_mem_words, write_mem_words};

/// BN254 base field element size in bytes, as `u64` for address arithmetic.
const BN254_FQ_BYTES: u64 = FIELD_256_BYTES as u64;
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
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
) {
    let x1: F = read_field_256(state, rs1_ptr);
    let y1: F = read_field_256(state, rs1_ptr + BN254_FQ_BYTES);
    let x2: F = read_field_256(state, rs2_ptr);
    let y2: F = read_field_256(state, rs2_ptr + BN254_FQ_BYTES);
    let (x3, y3) = ec_add_ne_impl(x1, y1, x2, y2);
    write_field_256(state, rd_ptr, &x3);
    write_field_256(state, rd_ptr + BN254_FQ_BYTES, &y3);
}

/// Execute ec_double for a 256-bit PrimeField curve.
#[inline(always)]
unsafe fn ec_double_256<F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>>(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    a: F,
) {
    let x1: F = read_field_256(state, rs1_ptr);
    let y1: F = read_field_256(state, rs1_ptr + BN254_FQ_BYTES);
    let (x3, y3) = ec_double_impl(x1, y1, a);
    write_field_256(state, rd_ptr, &x3);
    write_field_256(state, rd_ptr + BN254_FQ_BYTES, &y3);
}

// ── Curve constants ──────────────────────────────────────────────────────────

const P256_A_ABS: u64 = 3;

unsafe fn trace_read_bytes(state: *mut c_void, ptr: u64, len: u32) -> Vec<u8> {
    debug_assert_eq!(len % WORD_SIZE as u32, 0);
    let num_words = (len as usize) / WORD_SIZE;
    let mut words = vec![0u64; num_words];
    read_mem_words(state, ptr, &mut words);
    let mut bytes = Vec::with_capacity(len as usize);
    for &w in &words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

unsafe fn trace_write_bytes(state: *mut c_void, ptr: u64, bytes: &[u8]) {
    debug_assert_eq!(bytes.len() % WORD_SIZE, 0);
    let words: Vec<u64> = bytes
        .chunks_exact(WORD_SIZE)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    write_mem_words(state, ptr, &words);
}

fn field_config(curve: CurveType, num_limbs: usize) -> ExprBuilderConfig {
    ExprBuilderConfig {
        modulus: curve.coordinate_modulus().clone(),
        num_limbs,
        limb_bits: 8,
    }
}

fn ec_add_ne_setup_program(curve: CurveType, coord_bytes: usize) -> FieldExpressionProgram {
    ec_add_ne_program(field_config(curve, coord_bytes), U16_BITS)
}

fn ec_double_setup_program(curve: CurveType, coord_bytes: usize) -> FieldExpressionProgram {
    ec_double_ne_program(
        field_config(curve, coord_bytes),
        U16_BITS,
        curve.a_coefficient().clone(),
    )
}

fn setup_values_match(
    program: &FieldExpressionProgram,
    coord_bytes: usize,
    setup_bytes: &[u8],
) -> bool {
    let expected = iter::once(program.prime()).chain(program.setup_values());
    expected.enumerate().all(|(i, expected)| {
        let start = i * coord_bytes;
        let Some(bytes) = setup_bytes.get(start..start + coord_bytes) else {
            return false;
        };
        let expected = expected.to_bytes_le();
        bytes
            .get(..expected.len())
            .is_some_and(|value| value == expected.as_slice())
            && bytes[expected.len()..].iter().all(|&byte| byte == 0)
    })
}

fn ecc_setup_expr(
    program: &FieldExpressionProgram,
    point_bytes: u32,
    setup_bytes: &[u8],
) -> Option<Vec<u8>> {
    let coord_bytes = (point_bytes / 2) as usize;
    setup_values_match(program, coord_bytes, setup_bytes).then(|| {
        let flag_idx = program.num_flags();
        let writes = run_field_expression_precomputed::<true>(program, flag_idx, setup_bytes);
        writes.into()
    })
}

unsafe fn ec_add_ne_setup(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
    point_bytes: u32,
    program: &FieldExpressionProgram,
) -> bool {
    let mut setup_bytes = trace_read_bytes(state, rs1_ptr, point_bytes);
    setup_bytes.extend_from_slice(&trace_read_bytes(state, rs2_ptr, point_bytes));
    let Some(output) = ecc_setup_expr(program, point_bytes, &setup_bytes) else {
        return false;
    };
    trace_write_bytes(state, rd_ptr, &output);
    true
}

unsafe fn ec_double_setup(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    point_bytes: u32,
    program: &FieldExpressionProgram,
) -> bool {
    let setup_bytes = trace_read_bytes(state, rs1_ptr, point_bytes);
    let Some(output) = ecc_setup_expr(program, point_bytes, &setup_bytes) else {
        return false;
    };
    trace_write_bytes(state, rd_ptr, &output);
    true
}

unsafe fn ec_add_ne_256_entry<
    F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>,
>(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
) {
    ec_add_ne_256::<F>(state, rd_ptr, rs1_ptr, rs2_ptr);
}

unsafe fn ec_double_256_entry<
    F: halo2curves_axiom::ff::PrimeField<Repr = [u8; FIELD_256_BYTES]>,
>(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
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
            rd_ptr: u64,
            rs1_ptr: u64,
            rs2_ptr: u64,
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
        pub unsafe extern "C" fn $name(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) {
            ec_double_256_entry::<$curve>(state, rd_ptr, rs1_ptr, $a);
        }
    };
}

macro_rules! ecc_add_ne_setup_entry {
    ($name:ident, $point_bytes:expr, $curve:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid affine point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u64,
            rs1_ptr: u64,
            rs2_ptr: u64,
        ) -> bool {
            static PROGRAM: LazyLock<FieldExpressionProgram> =
                LazyLock::new(|| ec_add_ne_setup_program($curve, ($point_bytes / 2) as usize));
            ec_add_ne_setup(state, rd_ptr, rs1_ptr, rs2_ptr, $point_bytes, &PROGRAM)
        }
    };
}

macro_rules! ecc_double_setup_entry {
    ($name:ident, $point_bytes:expr, $curve:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid affine point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) -> bool {
            static PROGRAM: LazyLock<FieldExpressionProgram> =
                LazyLock::new(|| ec_double_setup_program($curve, ($point_bytes / 2) as usize));
            ec_double_setup(state, rd_ptr, rs1_ptr, $point_bytes, &PROGRAM)
        }
    };
}

ecc_add_ne_setup_entry!(
    rvr_ext_setup_ec_add_ne_k256,
    POINT_256_BYTES,
    CurveType::K256
);
ecc_double_setup_entry!(
    rvr_ext_setup_ec_double_k256,
    POINT_256_BYTES,
    CurveType::K256
);

ecc_add_ne_entry!(rvr_ext_ec_add_ne_p256, halo2curves_axiom::secp256r1::Fp);
ecc_add_ne_setup_entry!(
    rvr_ext_setup_ec_add_ne_p256,
    POINT_256_BYTES,
    CurveType::P256
);
ecc_double_entry!(
    rvr_ext_ec_double_p256,
    halo2curves_axiom::secp256r1::Fp,
    -halo2curves_axiom::secp256r1::Fp::from(P256_A_ABS)
);
ecc_double_setup_entry!(
    rvr_ext_setup_ec_double_p256,
    POINT_256_BYTES,
    CurveType::P256
);

ecc_add_ne_entry!(rvr_ext_ec_add_ne_bn254, halo2curves_axiom::bn256::Fq);
ecc_add_ne_setup_entry!(
    rvr_ext_setup_ec_add_ne_bn254,
    POINT_256_BYTES,
    CurveType::BN254
);
ecc_double_entry!(
    rvr_ext_ec_double_bn254,
    halo2curves_axiom::bn256::Fq,
    halo2curves_axiom::bn256::Fq::ZERO
);
ecc_double_setup_entry!(
    rvr_ext_setup_ec_double_bn254,
    POINT_256_BYTES,
    CurveType::BN254
);

ecc_add_ne_setup_entry!(
    rvr_ext_setup_ec_add_ne_bls12_381,
    POINT_BLS12_381_BYTES,
    CurveType::BLS12_381
);

ecc_double_setup_entry!(
    rvr_ext_setup_ec_double_bls12_381,
    POINT_BLS12_381_BYTES,
    CurveType::BLS12_381
);

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use super::*;

    fn parse_hex(value: &str) -> BigUint {
        BigUint::parse_bytes(value.as_bytes(), 16).unwrap()
    }

    fn write_le(value: &BigUint, out: &mut [u8]) {
        let bytes = value.to_bytes_le();
        out[..bytes.len()].copy_from_slice(&bytes);
    }

    #[test]
    fn setup_programs_validate_each_curve() {
        let cases = [
            (
                CurveType::K256,
                POINT_256_BYTES,
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
                "0",
            ),
            (
                CurveType::P256,
                POINT_256_BYTES,
                "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
                "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
            ),
            (
                CurveType::BN254,
                POINT_256_BYTES,
                "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
                "0",
            ),
            (
                CurveType::BLS12_381,
                POINT_BLS12_381_BYTES,
                "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
                "0",
            ),
        ];

        for (curve, point_bytes, modulus, a) in cases {
            let coord_bytes = point_bytes as usize / 2;
            let modulus = parse_hex(modulus);
            let a = parse_hex(a);

            let add_program = ec_add_ne_setup_program(curve, coord_bytes);
            assert_eq!(add_program.prime(), &modulus);
            let mut add_setup = vec![0u8; 2 * point_bytes as usize];
            write_le(&modulus, &mut add_setup[..coord_bytes]);
            write_le(&a, &mut add_setup[coord_bytes..2 * coord_bytes]);
            add_setup[2 * coord_bytes] = 1;
            add_setup[3 * coord_bytes] = 1;
            assert!(ecc_setup_expr(&add_program, point_bytes, &add_setup).is_some());
            add_setup[0] ^= 1;
            assert!(ecc_setup_expr(&add_program, point_bytes, &add_setup).is_none());

            let double_program = ec_double_setup_program(curve, coord_bytes);
            assert_eq!(double_program.prime(), &modulus);
            assert_eq!(double_program.setup_values(), std::slice::from_ref(&a));
            let mut double_setup = vec![0u8; point_bytes as usize];
            write_le(&modulus, &mut double_setup[..coord_bytes]);
            write_le(&a, &mut double_setup[coord_bytes..]);
            assert!(ecc_setup_expr(&double_program, point_bytes, &double_setup).is_some());
            double_setup[coord_bytes] ^= 1;
            assert!(ecc_setup_expr(&double_program, point_bytes, &double_setup).is_none());
        }
    }
}
