//! Rust FFI for P-256, BN254, generic modular arithmetic, setup, and
//! square-root hints.
//!
//! P-256 and BN254 use `halo2curves_axiom`; other moduli use `BigUint`.
//!
//! Shared I/O and arithmetic helpers live in
//! `rvr-openvm-ext-algebra-ffi-common`.

use std::{ffi::c_void, marker::PhantomData};

use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::One;
use openvm_instructions::riscv::RV64_MEMORY_AS;
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_algebra_ffi_common::{
    known_field_op_fn, mod_inverse, read_bigint, read_field_256, write_bigint, write_field_256,
    FieldArith, KnownFieldArith,
};
use rvr_openvm_ext_ffi_common::{
    ext_hint_stream_set, rd_mem_u64_range_wrapper, rd_mem_words_traced, trace_mem_access_range,
    wr_mem_words_traced,
};

// ── Field structs ────────────────────────────────────────────────────────────

pub struct KnownPrimeField<F>(pub PhantomData<F>);

struct UnknownPrimeField {
    modulus: BigUint,
    num_limbs: u32,
}

// ── KnownPrimeField impl (256-bit via generic bound) ────────────────────────

impl<F: PrimeField<Repr = [u8; 32]>> KnownFieldArith for KnownPrimeField<F> {
    type Elem = F;
    #[inline(always)]
    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem {
        read_field_256(state, ptr)
    }
    #[inline(always)]
    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem) {
        write_field_256(state, ptr, val)
    }
}

// ── UnknownPrimeField impl ──────────────────────────────────────────────────

impl FieldArith for UnknownPrimeField {
    type Elem = BigUint;

    unsafe fn read_elem(&self, state: *mut c_void, ptr: u64) -> Self::Elem {
        read_bigint(state, ptr, self.num_limbs)
    }

    unsafe fn write_elem(&self, state: *mut c_void, ptr: u64, val: &Self::Elem) {
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
unsafe fn exec_iseq<F: FieldArith>(f: &F, state: *mut c_void, rs1_ptr: u64, rs2_ptr: u64) -> u32 {
    let a = f.read_elem(state, rs1_ptr);
    let b = f.read_elem(state, rs2_ptr);
    if f.is_eq(&a, &b) {
        1
    } else {
        0
    }
}

// ── FFI generation macros ────────────────────────────────────────────────────

macro_rules! define_mod_ffi {
    ($field:ty, $suffix:ident) => {
        paste::paste! {
            known_field_op_fn!([<rvr_ext_mod_add_ $suffix>], KnownPrimeField, $field, add);
            known_field_op_fn!([<rvr_ext_mod_sub_ $suffix>], KnownPrimeField, $field, sub);
            known_field_op_fn!([<rvr_ext_mod_mul_ $suffix>], KnownPrimeField, $field, mul);
            known_field_op_fn!([<rvr_ext_mod_div_ $suffix>], KnownPrimeField, $field, div);
            /// # Safety
            /// `state` must be a valid `RvState` pointer.
            #[no_mangle]
            pub unsafe extern "C" fn [<rvr_ext_mod_iseq_ $suffix>](
                state: *mut c_void, rs1: u64, rs2: u64,
            ) -> u32 { exec_iseq(&KnownPrimeField::<$field>(PhantomData), state, rs1, rs2) }
        }
    };
}

// ── Specialized per-curve FFI instantiations ─────────────────────────────────

define_mod_ffi!(halo2curves_axiom::secp256r1::Fp, p256_coord);
define_mod_ffi!(halo2curves_axiom::secp256r1::Fq, p256_scalar);
define_mod_ffi!(halo2curves_axiom::bn256::Fq, bn254_fq);
define_mod_ffi!(halo2curves_axiom::bn256::Fr, bn254_fr);

// ── Generic FFI (fallback for unknown moduli) ────────────────────────────────

use rvr_openvm_ext_algebra_ffi_common::unknown_field_op_fn;

unknown_field_op_fn!(rvr_ext_mod_add, UnknownPrimeField, add);
unknown_field_op_fn!(rvr_ext_mod_sub, UnknownPrimeField, sub);
unknown_field_op_fn!(rvr_ext_mod_mul, UnknownPrimeField, mul);
unknown_field_op_fn!(rvr_ext_mod_div, UnknownPrimeField, div);

/// # Safety
/// `state` must be a valid `RvState` pointer. `modulus_ptr` must point to `num_limbs` bytes.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_mod_iseq(
    state: *mut c_void,
    rs1_ptr: u64,
    rs2_ptr: u64,
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
    rd_ptr: u64,
    rs1_ptr: u64,
    rs2_ptr: u64,
    num_limbs: u32,
) {
    let num_words = num_limbs / WORD_SIZE as u32;
    debug_assert!(num_words >= 1);

    let mut input_words = vec![0u64; num_words as usize];
    rd_mem_words_traced(state, rs1_ptr, &mut input_words);
    trace_mem_access_range(state, rs2_ptr, num_words, RV64_MEMORY_AS);

    // Setup validates that the guest-provided modulus and setup inputs match
    // the constants configured into this chip.
    //
    // In mod-builder, `Input(0)` means the first input slot. For setup, that
    // slot is the modulus p read from rs1. VM evaluates inputs modulo p,
    // so the setup output is p % p = 0.
    let output_words = vec![0u64; num_words as usize];
    wr_mem_words_traced(state, rd_ptr, &output_words);
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
    rs1_ptr: u64,
    num_limbs: u32,
    modulus_ptr: *const u8,
    non_qr_ptr: *const u8,
) {
    let modulus =
        BigUint::from_bytes_le(std::slice::from_raw_parts(modulus_ptr, num_limbs as usize));
    let non_qr = BigUint::from_bytes_le(std::slice::from_raw_parts(non_qr_ptr, num_limbs as usize));

    let num_words = (num_limbs / WORD_SIZE as u32) as usize;
    let mut words = vec![0u64; num_words];
    rd_mem_u64_range_wrapper(state, rs1_ptr, words.as_mut_ptr(), num_words as u32);
    // Note: no trace here — this is a phantom (hint) instruction, reads are not traced
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
