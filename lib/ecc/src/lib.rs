// #![no_std]

extern crate alloc;

pub use axvm_algebra as algebra;
#[cfg(feature = "halo2curves")]
pub use halo2curves_axiom as halo2curves;

mod affine_point;
pub use affine_point::*;
mod group;
pub use group::*;
mod msm;
pub use msm::*;
mod ecdsa;
pub use ecdsa::*;

/// Implementation of this library's traits on halo2curves types.
/// Used for testing and also VM runtime execution.
/// These should **only** be importable on a host machine.
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
pub mod halo2curves_shims;
/// Traits for optimal Ate pairing check using intrinsic functions.
pub mod pairing;
/// Weierstrass curve traits
pub mod sw;

/// Types for BLS12-381 curve with intrinsic functions.
pub mod bls12_381;
/// Types for BN254 curve with intrinsic functions.
pub mod bn254;

// TEMPORARY[jpw]: this should be moved into pairing
#[cfg(target_os = "zkvm")]
pub mod pairing_hint {
    use core::mem::MaybeUninit;

    use axvm_platform::constants::{Custom1Funct3, PairingBaseFunct7, CUSTOM_1, PAIRING_MAX_KINDS};

    // TODO: using PairingCurve enum
    const BN254_IDX: u8 = 0;
    const BLS12_381_IDX: u8 = 1;

    // TODO: transmute to Fp12 types

    /// Writes hint to stack and returns (residue_witness, scaling_factor)
    pub fn bn254_final_exp_hint(f: &[u8]) -> [u8; 32 * 12 * 2] {
        debug_assert_eq!(f.len(), 32 * 12);
        let hint = MaybeUninit::<[u8; 32 * 12 * 2]>::uninit();
        unsafe {
            bn254_hint_final_exp(f.as_ptr());
            let mut ptr = hint.as_ptr() as *const u8;
            // NOTE[jpw]: this loop could be unrolled using seq_macro and hint_store_u32(ptr, $imm)
            for _ in (0..32 * 12 * 2).step_by(4) {
                axvm::hint_store_u32!(ptr, 0);
                ptr = ptr.add(4);
            }
            hint.assume_init()
        }
    }

    /// Writes hint to stack and returns (residue_witness, scaling_factor)
    pub fn bls12_381_final_exp_hint(f: &[u8]) -> [u8; 48 * 12 * 2] {
        debug_assert_eq!(f.len(), 48 * 12);
        let hint = MaybeUninit::<[u8; 48 * 12 * 2]>::uninit();
        unsafe {
            bls12_381_hint_final_exp(f.as_ptr());
            let mut ptr = hint.as_ptr() as *const u8;
            // NOTE[jpw]: this loop could be unrolled using seq_macro and hint_store_u32(ptr, $imm)
            for _ in (0..48 * 12 * 2).step_by(4) {
                axvm::hint_store_u32!(ptr, 0);
                ptr = ptr.add(4);
            }
            hint.assume_init()
        }
    }

    /// Only resets the hint stream, does not write anything to memory
    #[inline(always)]
    unsafe fn bn254_hint_final_exp(f: *const u8) {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, x0, {rs}, x0",
            opcode = const CUSTOM_1,
            funct3 = const (Custom1Funct3::Pairing as u8),
            funct7 = const (BN254_IDX * PAIRING_MAX_KINDS + PairingBaseFunct7::HintFinalExp as u8),
            rs = in(reg) f
        );
    }

    /// Only resets the hint stream, does not write anything to memory
    #[inline(always)]
    unsafe fn bls12_381_hint_final_exp(f: *const u8) {
        core::arch::asm!(
            ".insn r {opcode}, {funct3}, {funct7}, x0, {rs}, x0",
            opcode = const CUSTOM_1,
            funct3 = const (Custom1Funct3::Pairing as u8),
            funct7 = const (BLS12_381_IDX * PAIRING_MAX_KINDS + PairingBaseFunct7::HintFinalExp as u8),
            rs = in(reg) f
        );
    }
}
