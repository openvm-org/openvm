//! Rust FFI for BN254 and generic Fp2 arithmetic.
//!
//! BN254 uses `halo2curves_axiom`; other moduli use `BigUint`.
//!
//! Shared helpers (I/O, `FieldArith`, `exec_op`) live in
//! `rvr-openvm-ext-algebra-ffi-common`.

use std::{ffi::c_void, marker::PhantomData};

use num_bigint::BigUint;
use openvm_instructions::riscv::RV64_MEMORY_AS;
use rvr_openvm_ext_algebra_ffi_common::{
    known_field_op_fn, limb_bytes_to_words, mod_inverse, read_bigint, read_field_256, write_bigint,
    write_field_256, FieldArith, KnownFieldArith,
};
use rvr_openvm_ext_ffi_common::{rd_mem_words_traced, trace_mem_access_range, wr_mem_words_traced};

const FIELD_256_BYTES: u64 = rvr_openvm_ext_algebra_ffi_common::FIELD_256_BYTES as u64;

// ── Field structs ───────────────────────────────────────────────────────────

struct KnownComplexField<F>(PhantomData<F>);

struct UnknownComplexField {
    modulus: BigUint,
    num_limbs: u32,
}

// ── KnownComplexField impls ─────────────────────────────────────────────────

impl KnownFieldArith for KnownComplexField<halo2curves_axiom::bn256::Fq2> {
    type Elem = halo2curves_axiom::bn256::Fq2;

    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem {
        halo2curves_axiom::bn256::Fq2::new(
            read_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr),
            read_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr + FIELD_256_BYTES),
        )
    }

    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem) {
        write_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr, &val.c0);
        write_field_256::<halo2curves_axiom::bn256::Fq>(state, ptr + FIELD_256_BYTES, &val.c1);
    }
}

// ── UnknownComplexField impl ────────────────────────────────────────────────

impl FieldArith for UnknownComplexField {
    type Elem = (BigUint, BigUint);

    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem {
        let c0 = read_bigint(state, ptr, self.num_limbs);
        let c1 = read_bigint(state, ptr + u64::from(self.num_limbs), self.num_limbs);
        (c0, c1)
    }

    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem) {
        write_bigint(state, ptr, &val.0, self.num_limbs);
        write_bigint(
            state,
            ptr + u64::from(self.num_limbs),
            &val.1,
            self.num_limbs,
        );
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
        let c0 = (&a.0 * &bi0 + p * p - &a.1 * &bi1) % p;
        let c1 = (&a.0 * &bi1 + &a.1 * &bi0) % p;
        (c0, c1)
    }

    fn is_eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool {
        let p = &self.modulus;
        &a.0 % p == &b.0 % p && &a.1 % p == &b.1 % p
    }
}

// ── Macros ──────────────────────────────────────────────────────────────────

macro_rules! define_fp2_ffi {
    ($field:ty, $suffix:ident) => {
        paste::paste! {
            known_field_op_fn!([<rvr_ext_fp2_add_ $suffix>], KnownComplexField, $field, add);
            known_field_op_fn!([<rvr_ext_fp2_sub_ $suffix>], KnownComplexField, $field, sub);
            known_field_op_fn!([<rvr_ext_fp2_mul_ $suffix>], KnownComplexField, $field, mul);
            known_field_op_fn!([<rvr_ext_fp2_div_ $suffix>], KnownComplexField, $field, div);
        }
    };
}

define_fp2_ffi!(halo2curves_axiom::bn256::Fq2, bn254);

// ── Generic FFI (fallback for unknown moduli) ────────────────────────────────

use rvr_openvm_ext_algebra_ffi_common::unknown_field_op_fn;

unknown_field_op_fn!(rvr_ext_fp2_add, UnknownComplexField, add);
unknown_field_op_fn!(rvr_ext_fp2_sub, UnknownComplexField, sub);
unknown_field_op_fn!(rvr_ext_fp2_mul, UnknownComplexField, mul);
unknown_field_op_fn!(rvr_ext_fp2_div, UnknownComplexField, div);

/// # Safety
/// `state` must be a valid `RvState` pointer.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_fp2_setup(
    state: *mut c_void,
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
    num_limbs: u32,
) {
    let total_limbs = num_limbs.checked_mul(2).expect("Fp2 limb count overflow");
    let num_words = limb_bytes_to_words(total_limbs);
    debug_assert!(num_words >= 1);

    let mut input_words = vec![0u64; num_words as usize];
    rd_mem_words_traced(state, rs1_ptr, &mut input_words);
    trace_mem_access_range(state, rs2_ptr, num_words, RV64_MEMORY_AS);

    // Setup validates that the guest-provided base-field modulus and setup
    // inputs match the constants configured into this chip.
    //
    // In mod-builder, `Input(0)` means the first input slot. For setup, that
    // slot is the modulus p read from rs1. VM evaluates inputs modulo p,
    // so each setup coordinate writes p % p = 0.
    let output_words = vec![0u64; num_words as usize];
    wr_mem_words_traced(state, rd_ptr, &output_words);
}
