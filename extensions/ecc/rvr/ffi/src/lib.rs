//! Rust FFI for projective P-256, BN254, and BLS12-381 point operations and
//! setup operations for all supported curves.
//!
//! Point operations use the circuit's native projective formulas (see [`ec`]) so
//! rvr execution matches the circuit exactly. Setup operations evaluate OpenVM's
//! precomputed field expressions and validate the on-chip curve parameters.

use std::{ffi::c_void, iter, sync::LazyLock};

use num_bigint::BigUint;
use openvm_circuit_primitives::U16_BITS;
use openvm_ecc_circuit::{ec_add_proj_program, ec_double_proj_program, CurveType};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, ExprBuilderConfig, FieldExpressionProgram,
};
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_algebra_ffi_common::{BLS12_381_ELEM_BYTES, FIELD_256_BYTES};
use rvr_openvm_ext_ffi_common::{read_mem_words, write_mem_words};

mod ec;
use ec::{
    ec_add_proj_bls12_381, ec_add_proj_bn254, ec_add_proj_p256, ec_double_proj_bls12_381,
    ec_double_proj_bn254, ec_double_proj_p256,
};

/// Projective point: three field coordinates (X, Y, Z).
const PROJECTIVE_COORDS: u32 = 3;
/// Size of a 256-bit projective point in bytes.
const POINT_256_BYTES: u32 = PROJECTIVE_COORDS * FIELD_256_BYTES as u32;
/// Size of a BLS12-381 projective point in bytes.
const POINT_BLS12_381_BYTES: u32 = PROJECTIVE_COORDS * BLS12_381_ELEM_BYTES as u32;

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

/// The `b` coefficient of `y² = x³ + ax + b` for each supported curve.
///
/// [`CurveType`] exposes the coordinate modulus and `a`, but the projective
/// formulas (and their setup programs) also need `b`, so it is encoded here.
fn curve_b(curve: CurveType) -> BigUint {
    match curve {
        CurveType::K256 => BigUint::from(7u32),
        CurveType::P256 => BigUint::parse_bytes(
            b"5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b",
            16,
        )
        .unwrap(),
        CurveType::BN254 => BigUint::from(3u32),
        CurveType::BLS12_381 => BigUint::from(4u32),
    }
}

fn ec_add_proj_setup_program(curve: CurveType, coord_bytes: usize) -> FieldExpressionProgram {
    ec_add_proj_program(
        field_config(curve, coord_bytes),
        U16_BITS,
        curve.a_coefficient().clone(),
        curve_b(curve),
    )
}

fn ec_double_proj_setup_program(curve: CurveType, coord_bytes: usize) -> FieldExpressionProgram {
    ec_double_proj_program(
        field_config(curve, coord_bytes),
        U16_BITS,
        curve.a_coefficient().clone(),
        curve_b(curve),
    )
}

/// Checks that the on-chip setup values (encoded in the first point's X, Y, Z
/// coordinates as modulus, a, b) match the program's prime and setup values.
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
    let coord_bytes = (point_bytes / PROJECTIVE_COORDS) as usize;
    setup_values_match(program, coord_bytes, setup_bytes).then(|| {
        let flag_idx = program.num_flags();
        let writes = run_field_expression_precomputed::<true>(program, flag_idx, setup_bytes);
        writes.into()
    })
}

unsafe fn ec_add_proj_setup(
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

unsafe fn ec_double_proj_setup(
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

/// Projective point-add entry point, forwarding to the per-curve `$f` in [`ec`].
macro_rules! ecc_add_entry {
    ($name:ident, $f:path) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid projective point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u64,
            rs1_ptr: u64,
            rs2_ptr: u64,
        ) {
            $f(state, rd_ptr, rs1_ptr, rs2_ptr);
        }
    };
}

/// Projective point-double entry point, forwarding to the per-curve `$f` in [`ec`].
macro_rules! ecc_double_entry {
    ($name:ident, $f:path) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid projective point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) {
            $f(state, rd_ptr, rs1_ptr);
        }
    };
}

macro_rules! ecc_add_proj_setup_entry {
    ($name:ident, $point_bytes:expr, $curve:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid projective point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            state: *mut c_void,
            rd_ptr: u64,
            rs1_ptr: u64,
            rs2_ptr: u64,
        ) -> bool {
            static PROGRAM: LazyLock<FieldExpressionProgram> = LazyLock::new(|| {
                ec_add_proj_setup_program($curve, ($point_bytes / PROJECTIVE_COORDS) as usize)
            });
            ec_add_proj_setup(state, rd_ptr, rs1_ptr, rs2_ptr, $point_bytes, &PROGRAM)
        }
    };
}

macro_rules! ecc_double_proj_setup_entry {
    ($name:ident, $point_bytes:expr, $curve:expr) => {
        /// # Safety
        ///
        /// `state` must point to a valid native tracer state for this execution.
        /// Pointer parameters must point to valid projective point coordinates.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd_ptr: u64, rs1_ptr: u64) -> bool {
            static PROGRAM: LazyLock<FieldExpressionProgram> = LazyLock::new(|| {
                ec_double_proj_setup_program($curve, ($point_bytes / PROJECTIVE_COORDS) as usize)
            });
            ec_double_proj_setup(state, rd_ptr, rs1_ptr, $point_bytes, &PROGRAM)
        }
    };
}

// k256 add/double come from libsecp256k1 in the modular staticlib, so only the
// setup entry points are generated here.
ecc_add_proj_setup_entry!(
    rvr_ext_setup_ec_add_proj_k256,
    POINT_256_BYTES,
    CurveType::K256
);
ecc_double_proj_setup_entry!(
    rvr_ext_setup_ec_double_proj_k256,
    POINT_256_BYTES,
    CurveType::K256
);

ecc_add_entry!(rvr_ext_ec_add_proj_p256, ec_add_proj_p256);
ecc_add_proj_setup_entry!(
    rvr_ext_setup_ec_add_proj_p256,
    POINT_256_BYTES,
    CurveType::P256
);
ecc_double_entry!(rvr_ext_ec_double_proj_p256, ec_double_proj_p256);
ecc_double_proj_setup_entry!(
    rvr_ext_setup_ec_double_proj_p256,
    POINT_256_BYTES,
    CurveType::P256
);

ecc_add_entry!(rvr_ext_ec_add_proj_bn254, ec_add_proj_bn254);
ecc_add_proj_setup_entry!(
    rvr_ext_setup_ec_add_proj_bn254,
    POINT_256_BYTES,
    CurveType::BN254
);
ecc_double_entry!(rvr_ext_ec_double_proj_bn254, ec_double_proj_bn254);
ecc_double_proj_setup_entry!(
    rvr_ext_setup_ec_double_proj_bn254,
    POINT_256_BYTES,
    CurveType::BN254
);

ecc_add_entry!(rvr_ext_ec_add_proj_bls12_381, ec_add_proj_bls12_381);
ecc_add_proj_setup_entry!(
    rvr_ext_setup_ec_add_proj_bls12_381,
    POINT_BLS12_381_BYTES,
    CurveType::BLS12_381
);
ecc_double_entry!(rvr_ext_ec_double_proj_bls12_381, ec_double_proj_bls12_381);
ecc_double_proj_setup_entry!(
    rvr_ext_setup_ec_double_proj_bls12_381,
    POINT_BLS12_381_BYTES,
    CurveType::BLS12_381
);
