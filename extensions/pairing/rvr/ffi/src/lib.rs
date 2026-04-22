//! FFI function for the pairing extension (HintFinalExp phantom).
//!
//! Computes the multi-Miller loop and final exponentiation hint, then sets
//! the hint stream to the result bytes.

use std::ffi::c_void;

use halo2curves_axiom::{bls12_381, bn256, ff::PrimeField};
use openvm_ecc_guest::{algebra::field::FieldExtension, AffinePoint};
use openvm_pairing_guest::{
    halo2curves_shims::{bls12_381::Bls12_381, bn254::Bn254},
    pairing::{FinalExp, MultiMillerLoop},
};
use rvr_openvm_ext_algebra_ffi::{BLS12_381_ELEM_BYTES, FIELD_256_BYTES};
use rvr_openvm_ext_ffi_common::{
    ext_hint_stream_set, rd_mem_u32_range_wrapper, rd_mem_u32_wrapper, WORD_SIZE,
};

/// BN254 base field element size in bytes.
const BN254_FQ_BYTES: u32 = FIELD_256_BYTES as u32;
/// BLS12-381 base field element size in bytes.
const BLS12_381_FQ_BYTES: u32 = BLS12_381_ELEM_BYTES as u32;
/// G1 affine point: two field coordinates (x, y).
const G1_AFFINE_COORDS: u32 = 2;
/// G2 affine point: two Fp2 coordinates, each containing two Fp elements.
const G2_AFFINE_COORDS: u32 = 4;
/// Offset of `len` in a guest slice header `(data_ptr, len)`.
const SLICE_LEN_OFFSET: u32 = WORD_SIZE as u32;

unsafe fn set_hint_stream(bytes: &[u8]) {
    let len: u32 = bytes.len().try_into().unwrap();
    ext_hint_stream_set(bytes.as_ptr(), len);
}

unsafe fn clear_hint_stream() {
    set_hint_stream(&[]);
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Read `N` bytes from guest memory (untraced, word-aligned reads).
unsafe fn read_bytes<const N: usize>(state: *mut c_void, ptr: u32) -> [u8; N] {
    const {
        assert!(
            N.is_multiple_of(WORD_SIZE),
            "N must be a multiple of WORD_SIZE"
        );
    }
    let num_words = N / WORD_SIZE;
    let mut words = vec![0u32; num_words];
    rd_mem_u32_range_wrapper(state, ptr, words.as_mut_ptr(), num_words as u32);
    let mut bytes = [0u8; N];
    for (i, &w) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// Read an Fq element from guest memory for BN254.
unsafe fn read_bn254_fq(state: *mut c_void, ptr: u32) -> Option<bn256::Fq> {
    let bytes = read_bytes::<{ BN254_FQ_BYTES as usize }>(state, ptr);
    Option::from(bn256::Fq::from_repr(bytes))
}

/// Read an Fq element from guest memory for BLS12-381.
unsafe fn read_bls12_381_fq(state: *mut c_void, ptr: u32) -> Option<bls12_381::Fq> {
    let bytes = read_bytes::<{ BLS12_381_FQ_BYTES as usize }>(state, ptr);
    Option::from(bls12_381::Fq::from_repr(bytes.into()))
}

fn point_base(ptr: u32, idx: u32, words_per_point: u32) -> Option<u32> {
    idx.checked_mul(words_per_point)?.checked_add(ptr)
}

// ── BN254 pairing hint ──────────────────────────────────────────────────

unsafe fn hint_bn254(state: *mut c_void, rs1_val: u32, rs2_val: u32) -> Option<Vec<u8>> {
    let p_ptr = rd_mem_u32_wrapper(state, rs1_val);
    let p_len = rd_mem_u32_wrapper(state, rs1_val + SLICE_LEN_OFFSET);
    let q_ptr = rd_mem_u32_wrapper(state, rs2_val);
    let q_len = rd_mem_u32_wrapper(state, rs2_val + SLICE_LEN_OFFSET);

    if p_len != q_len {
        return None;
    }

    let p: Vec<_> = (0..p_len)
        .map(|i| {
            let base = point_base(p_ptr, i, G1_AFFINE_COORDS * BN254_FQ_BYTES)?;
            Some(AffinePoint::new(
                read_bn254_fq(state, base)?,
                read_bn254_fq(state, base + BN254_FQ_BYTES)?,
            ))
        })
        .collect::<Option<_>>()?;

    let q: Vec<_> = (0..q_len)
        .map(|i| {
            let base = point_base(q_ptr, i, G2_AFFINE_COORDS * BN254_FQ_BYTES)?;
            // BN254 Fq2 exposes a constructor helper.
            let x = bn256::Fq2::new(
                read_bn254_fq(state, base)?,
                read_bn254_fq(state, base + BN254_FQ_BYTES)?,
            );
            let y = bn256::Fq2::new(
                read_bn254_fq(state, base + 2 * BN254_FQ_BYTES)?,
                read_bn254_fq(state, base + 3 * BN254_FQ_BYTES)?,
            );
            Some(AffinePoint::new(x, y))
        })
        .collect::<Option<_>>()?;

    let f: bn256::Fq12 = Bn254::multi_miller_loop(&p, &q);
    let (c, u) = Bn254::final_exp_hint(&f);

    // Serialize hint: c (Fp12) then u (Fp12), each as 12 Fq elements
    Some(
        c.to_coeffs()
            .into_iter()
            .chain(u.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect(),
    )
}

// ── BLS12-381 pairing hint ──────────────────────────────────────────────

unsafe fn hint_bls12_381(state: *mut c_void, rs1_val: u32, rs2_val: u32) -> Option<Vec<u8>> {
    let p_ptr = rd_mem_u32_wrapper(state, rs1_val);
    let p_len = rd_mem_u32_wrapper(state, rs1_val + SLICE_LEN_OFFSET);
    let q_ptr = rd_mem_u32_wrapper(state, rs2_val);
    let q_len = rd_mem_u32_wrapper(state, rs2_val + SLICE_LEN_OFFSET);

    if p_len != q_len {
        return None;
    }

    let p: Vec<_> = (0..p_len)
        .map(|i| {
            let base = point_base(p_ptr, i, G1_AFFINE_COORDS * BLS12_381_FQ_BYTES)?;
            Some(AffinePoint::new(
                read_bls12_381_fq(state, base)?,
                read_bls12_381_fq(state, base + BLS12_381_FQ_BYTES)?,
            ))
        })
        .collect::<Option<_>>()?;

    let q: Vec<_> = (0..q_len)
        .map(|i| {
            let base = point_base(q_ptr, i, G2_AFFINE_COORDS * BLS12_381_FQ_BYTES)?;
            // BLS12-381 Fq2 uses struct fields instead of an `Fq2::new` helper.
            let x = bls12_381::Fq2 {
                c0: read_bls12_381_fq(state, base)?,
                c1: read_bls12_381_fq(state, base + BLS12_381_FQ_BYTES)?,
            };
            let y = bls12_381::Fq2 {
                c0: read_bls12_381_fq(state, base + 2 * BLS12_381_FQ_BYTES)?,
                c1: read_bls12_381_fq(state, base + 3 * BLS12_381_FQ_BYTES)?,
            };
            Some(AffinePoint::new(x, y))
        })
        .collect::<Option<_>>()?;

    let f: bls12_381::Fq12 = Bls12_381::multi_miller_loop(&p, &q);
    let (c, u) = Bls12_381::final_exp_hint(&f);

    Some(
        c.to_coeffs()
            .into_iter()
            .chain(u.to_coeffs())
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect(),
    )
}

// ── FFI entry point ─────────────────────────────────────────────────────

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_pairing_hint_final_exp_bn254(
    state: *mut c_void,
    rs1_val: u32,
    rs2_val: u32,
) {
    if let Some(hint_bytes) = hint_bn254(state, rs1_val, rs2_val) {
        set_hint_stream(&hint_bytes);
    } else {
        clear_hint_stream();
    }
}

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_pairing_hint_final_exp_bls12_381(
    state: *mut c_void,
    rs1_val: u32,
    rs2_val: u32,
) {
    if let Some(hint_bytes) = hint_bls12_381(state, rs1_val, rs2_val) {
        set_hint_stream(&hint_bytes);
    } else {
        clear_hint_stream();
    }
}
