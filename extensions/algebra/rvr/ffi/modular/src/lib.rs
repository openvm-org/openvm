//! Modular arithmetic FFI: 256-bit / 384-bit prime fields + the algebra
//! sqrt phantom hint. Per-curve specializations dispatch to native
//! `halo2curves_axiom` / `blstrs` arithmetic; unknown moduli fall back to a
//! BigUint-based path.
//!
//! This crate also bundles libsecp256k1 (with ECC) — see `c/rvr_ext_modular.c`
//! and `build.rs`.
//!
//! Shared helpers (I/O, `FieldArith`, `exec_op`, the BLS12-381 newtype) live in
//! `rvr-openvm-ext-algebra-ffi-common` so the fp2 staticlib can reuse them
//! without re-defining strong globals.

use std::{ffi::c_void, marker::PhantomData};

use halo2curves_axiom::ff::{Field, PrimeField};
use num_bigint::BigUint;
use num_traits::One;
use rvr_openvm_ext_algebra_ffi_common::{
    exec_op, mod_inverse, read_bigint, read_bls12_381_fq, read_field_256, write_bigint,
    write_bls12_381_fq, write_field_256, FieldArith,
};
use rvr_openvm_ext_ffi_common::{
    ext_hint_stream_set, rd_mem_u32_range_wrapper, rd_mem_words_traced, trace_mem_access_range,
    wr_mem_words_traced, AS_MEMORY, WORD_SIZE,
};

// ── Field structs ────────────────────────────────────────────────────────────

pub struct KnownPrimeField<F>(pub PhantomData<F>);

/// Newtype tag for BLS12-381 Fq used by `KnownPrimeField<Bls12381Fq>`.
/// `blstrs::Fp` uses a different `ff` crate version than halo2curves_axiom, so
/// it does not impl `halo2curves_axiom::ff::PrimeField`, but the compiler can't
/// prove that — a concrete impl on `KnownPrimeField<blstrs::Fp>` would conflict
/// with the generic blanket impl below. The newtype must live in this crate
/// (not the shared `…-ffi-common`) so coherence allows the blanket + concrete
/// impl pair: with `Bls12381Fq` local, the compiler knows no upstream
/// `PrimeField<Repr=[u8;32]>` impl can ever appear for it.
pub struct Bls12381Fq;

struct UnknownPrimeField {
    modulus: BigUint,
    num_limbs: u32,
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

// ── Instruction execution helpers ────────────────────────────────────────────

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

macro_rules! field_op_fn {
    ($name:ident, $field:ty, $op:ident) => {
        /// # Safety
        /// `state` must be a valid `RvState` pointer.
        #[no_mangle]
        pub unsafe extern "C" fn $name(state: *mut c_void, rd: u32, rs1: u32, rs2: u32) {
            let f = KnownPrimeField::<$field>(PhantomData);
            exec_op(&f, state, rd, rs1, rs2, |f, a, b| f.$op(a, b));
        }
    };
}

macro_rules! define_mod_ffi {
    ($field:ty, $suffix:ident) => {
        paste::paste! {
            field_op_fn!([<rvr_ext_mod_add_ $suffix>], $field, add);
            field_op_fn!([<rvr_ext_mod_sub_ $suffix>], $field, sub);
            field_op_fn!([<rvr_ext_mod_mul_ $suffix>], $field, mul);
            field_op_fn!([<rvr_ext_mod_div_ $suffix>], $field, div);
            /// # Safety
            /// `state` must be a valid `RvState` pointer.
            #[no_mangle]
            pub unsafe extern "C" fn [<rvr_ext_mod_iseq_ $suffix>](
                state: *mut c_void, rs1: u32, rs2: u32,
            ) -> u32 { exec_iseq(&KnownPrimeField::<$field>(PhantomData), state, rs1, rs2) }
        }
    };
}

// ── Specialized per-curve FFI instantiations ─────────────────────────────────

// k256_coord and k256_scalar are provided by the C wrapper (see c/rvr_ext_modular.c)
// using libsecp256k1; no Rust entry points generated for them.
define_mod_ffi!(halo2curves_axiom::secp256r1::Fp, p256_coord);
define_mod_ffi!(halo2curves_axiom::secp256r1::Fq, p256_scalar);
define_mod_ffi!(halo2curves_axiom::bn256::Fq, bn254_fq);
define_mod_ffi!(halo2curves_axiom::bn256::Fr, bn254_fr);
define_mod_ffi!(Bls12381Fq, bls12_381_fq);
define_mod_ffi!(halo2curves_axiom::bls12_381::Fr, bls12_381_fr);

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
            let f = UnknownPrimeField { modulus, num_limbs };
            exec_op(&f, state, rd_ptr, rs1_ptr, rs2_ptr, |f, a, b| f.$op(a, b));
        }
    };
}

unknown_field_op_fn!(rvr_ext_mod_add, add);
unknown_field_op_fn!(rvr_ext_mod_sub, sub);
unknown_field_op_fn!(rvr_ext_mod_mul, mul);
unknown_field_op_fn!(rvr_ext_mod_div, div);

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

    let mut src_words = vec![0u32; num_words as usize];
    rd_mem_words_traced(state, rs1_ptr, &mut src_words);
    trace_mem_access_range(state, rs2_ptr, num_words, AS_MEMORY);
    wr_mem_words_traced(state, rd_ptr, &src_words);
}

// ── Phantom: HintSqrt ────────────────────────────────────────────────────────

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
