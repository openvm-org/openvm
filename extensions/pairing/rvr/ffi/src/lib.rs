//! FFI function for the pairing extension (HintFinalExp phantom).
//!
//! Computes the multi-Miller loop and final exponentiation hint, then sets
//! the hint stream to the result bytes.

use std::{
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
};

use halo2curves_axiom::{bls12_381, bn256, ff::PrimeField};
use openvm_ecc_guest::{
    algebra::{field::FieldExtension, Field},
    AffinePoint,
};
use openvm_pairing_guest::{
    halo2curves_shims::{bls12_381::Bls12_381, bn254::Bn254},
    pairing::{FinalExp, MultiMillerLoop},
};
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_algebra_ffi_common::{BLS12_381_ELEM_BYTES, FIELD_256_BYTES};
use rvr_openvm_ext_ffi_common::{
    checked_guest_memory_range, checked_peek_mem_words, ext_hint_stream_set, peek_mem_words,
};

/// BN254 base field element size in bytes.
const BN254_FQ_BYTES: u64 = FIELD_256_BYTES as u64;
const BN254_FQ_WORDS: usize = FIELD_256_BYTES / WORD_SIZE;
/// BLS12-381 base field element size in bytes.
const BLS12_381_FQ_BYTES: u64 = BLS12_381_ELEM_BYTES as u64;
const BLS12_381_FQ_WORDS: usize = BLS12_381_ELEM_BYTES / WORD_SIZE;
/// G1 affine point: two field coordinates (x, y).
const G1_AFFINE_COORDS: u64 = 2;
/// G2 affine point: two Fp2 coordinates, each containing two Fp elements.
const G2_AFFINE_COORDS: u64 = 4;

unsafe fn set_hint_stream(bytes: &[u8]) {
    let len = u64::try_from(bytes.len()).unwrap();
    ext_hint_stream_set(bytes.as_ptr(), len);
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Peek at `N` bytes in a range validated by [`read_pairing_points`].
unsafe fn read_bytes<const N: usize, const WORDS: usize>(state: *mut c_void, ptr: u64) -> [u8; N] {
    const {
        assert!(N == WORDS * WORD_SIZE, "word count must cover N bytes");
    }
    let mut words = [0u64; WORDS];
    peek_mem_words(state, ptr, &mut words);
    let mut bytes = [0u8; N];
    for (i, &word) in words.iter().enumerate() {
        bytes[i * WORD_SIZE..(i + 1) * WORD_SIZE].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Read an Fq element from guest memory for BN254.
unsafe fn read_bn254_fq(state: *mut c_void, ptr: u64) -> Option<bn256::Fq> {
    let bytes = read_bytes::<FIELD_256_BYTES, BN254_FQ_WORDS>(state, ptr);
    Option::from(bn256::Fq::from_repr(bytes))
}

/// Read an Fq element from guest memory for BLS12-381.
unsafe fn read_bls12_381_fq(state: *mut c_void, ptr: u64) -> Option<bls12_381::Fq> {
    let bytes = read_bytes::<BLS12_381_ELEM_BYTES, BLS12_381_FQ_WORDS>(state, ptr);
    Option::from(bls12_381::Fq::from_repr(bytes.into()))
}

fn point_base(ptr: u64, idx: usize, bytes_per_point: u64) -> Option<u64> {
    let offset = u64::try_from(idx).ok()?.checked_mul(bytes_per_point)?;
    ptr.checked_add(offset)
}

// ── Shared scaffolding ──────────────────────────────────────────────────

type PairingPoints<Fq, Fq2> = (Vec<AffinePoint<Fq>>, Vec<AffinePoint<Fq2>>);

/// Read G1 and G2 point vectors from guest memory.
///
/// `read_fq` reads one base-field element. `make_fq2` constructs an Fq2
/// from two consecutive Fq elements; use `Fq2::new` or `|c0, c1| Fq2 { c0, c1 }`
/// depending on the curve's API.
unsafe fn read_pairing_points<Fq: Field, Fq2: Field>(
    state: *mut c_void,
    rs1_val: u64,
    rs2_val: u64,
    fq_bytes: u64,
    read_fq: impl Fn(*mut c_void, u64) -> Option<Fq>,
    make_fq2: impl Fn(Fq, Fq) -> Fq2,
) -> Option<PairingPoints<Fq, Fq2>> {
    const SLICE_HEADER_BYTES: u64 = 2 * WORD_SIZE as u64;

    // Validate both slice headers before reading either one.
    checked_guest_memory_range(rs1_val, SLICE_HEADER_BYTES)?;
    checked_guest_memory_range(rs2_val, SLICE_HEADER_BYTES)?;

    let mut p_header = [0u64; 2];
    let mut q_header = [0u64; 2];
    checked_peek_mem_words(state, rs1_val, &mut p_header)?;
    checked_peek_mem_words(state, rs2_val, &mut q_header)?;
    let [p_ptr, p_len] = p_header;
    let [q_ptr, q_len] = q_header;

    if p_len != q_len {
        return None;
    }

    let p_point_bytes = G1_AFFINE_COORDS.checked_mul(fq_bytes)?;
    let q_point_bytes = G2_AFFINE_COORDS.checked_mul(fq_bytes)?;
    let p_bytes = p_len.checked_mul(p_point_bytes)?;
    let q_bytes = q_len.checked_mul(q_point_bytes)?;
    // Point payloads are not read or allocated until both complete ranges
    // have been checked.
    if !p_ptr.is_multiple_of(WORD_SIZE as u64) || !q_ptr.is_multiple_of(WORD_SIZE as u64) {
        return None;
    }
    checked_guest_memory_range(p_ptr, p_bytes)?;
    checked_guest_memory_range(q_ptr, q_bytes)?;

    let len = usize::try_from(p_len).ok()?;
    let mut p = Vec::new();
    let mut q = Vec::new();
    p.try_reserve_exact(len).ok()?;
    q.try_reserve_exact(len).ok()?;
    for i in 0..len {
        let base = point_base(p_ptr, i, p_point_bytes)?;
        p.push(AffinePoint::new(
            read_fq(state, base)?,
            read_fq(state, base.checked_add(fq_bytes)?)?,
        ));

        let base = point_base(q_ptr, i, q_point_bytes)?;
        let x = make_fq2(
            read_fq(state, base)?,
            read_fq(state, base.checked_add(fq_bytes)?)?,
        );
        let y = make_fq2(
            read_fq(state, base.checked_add(2 * fq_bytes)?)?,
            read_fq(state, base.checked_add(3 * fq_bytes)?)?,
        );
        q.push(AffinePoint::new(x, y));
    }

    Some((p, q))
}

// ── BN254 pairing hint ──────────────────────────────────────────────────

unsafe fn hint_bn254(state: *mut c_void, rs1_val: u64, rs2_val: u64) -> Option<Vec<u8>> {
    let (p, q) = read_pairing_points(
        state,
        rs1_val,
        rs2_val,
        BN254_FQ_BYTES,
        |s, ptr| unsafe { read_bn254_fq(s, ptr) },
        bn256::Fq2::new,
    )?;
    let f: bn256::Fq12 = Bn254::multi_miller_loop(&p, &q);
    let (c, u) = Bn254::final_exp_hint(&f);
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

unsafe fn hint_bls12_381(state: *mut c_void, rs1_val: u64, rs2_val: u64) -> Option<Vec<u8>> {
    let (p, q) = read_pairing_points(
        state,
        rs1_val,
        rs2_val,
        BLS12_381_FQ_BYTES,
        |s, ptr| unsafe { read_bls12_381_fq(s, ptr) },
        |c0, c1| bls12_381::Fq2 { c0, c1 },
    )?;
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

unsafe fn run_hint(
    state: *mut c_void,
    rs1_val: u64,
    rs2_val: u64,
    hint: unsafe fn(*mut c_void, u64, u64) -> Option<Vec<u8>>,
) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(hint_bytes) = hint(state, rs1_val, rs2_val) else {
            return false;
        };
        set_hint_stream(&hint_bytes);
        true
    }))
    .unwrap_or(false)
}

// ── FFI entry point ─────────────────────────────────────────────────────

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_pairing_hint_final_exp_bn254(
    state: *mut c_void,
    rs1_val: u64,
    rs2_val: u64,
) -> bool {
    run_hint(state, rs1_val, rs2_val, hint_bn254)
}

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_pairing_hint_final_exp_bls12_381(
    state: *mut c_void,
    rs1_val: u64,
    rs2_val: u64,
) -> bool {
    run_hint(state, rs1_val, rs2_val, hint_bls12_381)
}
