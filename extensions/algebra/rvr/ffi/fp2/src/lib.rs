//! Fp2 (complex extension field) arithmetic FFI. Per-curve specializations
//! dispatch to native `halo2curves_axiom::bn256::Fq2` / `blstrs::Fp2`;
//! unknown moduli fall back to a BigUint-pair path.
//!
//! Shared helpers (I/O, `FieldArith`, `exec_op`) live in
//! `rvr-openvm-ext-algebra-ffi-common`.

use std::{ffi::c_void, marker::PhantomData};

use halo2curves_axiom::ff::Field;
use num_bigint::BigUint;
use rvr_openvm_ext_algebra_ffi_common::{
    exec_op, mod_inverse, read_bigint, read_bls12_381_fq, read_field_256, write_bigint,
    write_bls12_381_fq, write_field_256, FieldArith, BLS12_381_ELEM_BYTES, FIELD_256_BYTES,
};
use rvr_openvm_ext_ffi_common::{
    rd_mem_words_traced, trace_mem_access_range, wr_mem_words_traced, AS_MEMORY, WORD_SIZE,
};

// ── Field structs ───────────────────────────────────────────────────────────

struct KnownComplexField<F>(PhantomData<F>);

struct UnknownComplexField {
    modulus: BigUint,
    num_limbs: u32,
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

macro_rules! field_op_fn {
    ($name:ident, $field:ty, $op:ident) => {
        /// # Safety
        /// `state` must be a valid `RvState` pointer.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd: u32, rs1: u32, rs2: u32) {
            let f = KnownComplexField::<$field>(PhantomData);
            exec_op(&f, state, rd, rs1, rs2, |f, a, b| f.$op(a, b));
        }
    };
}

macro_rules! define_fp2_ffi {
    ($field:ty, $suffix:ident) => {
        paste::paste! {
            field_op_fn!([<rvr_ext_fp2_add_ $suffix>], $field, add);
            field_op_fn!([<rvr_ext_fp2_sub_ $suffix>], $field, sub);
            field_op_fn!([<rvr_ext_fp2_mul_ $suffix>], $field, mul);
            field_op_fn!([<rvr_ext_fp2_div_ $suffix>], $field, div);
        }
    };
}

define_fp2_ffi!(halo2curves_axiom::bn256::Fq2, bn254);
define_fp2_ffi!(blstrs::Fp2, bls12_381);

// ── Generic FFI (fallback for unknown moduli) ────────────────────────────────

macro_rules! unknown_field_op_fn {
    ($name:ident, $op:ident) => {
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
            let f = UnknownComplexField { modulus, num_limbs };
            exec_op(&f, state, rd_ptr, rs1_ptr, rs2_ptr, |f, a, b| f.$op(a, b));
        }
    };
}

unknown_field_op_fn!(rvr_ext_fp2_add, add);
unknown_field_op_fn!(rvr_ext_fp2_sub, sub);
unknown_field_op_fn!(rvr_ext_fp2_mul, mul);
unknown_field_op_fn!(rvr_ext_fp2_div, div);

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
