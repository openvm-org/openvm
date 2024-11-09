#![no_std]

extern crate alloc;

pub mod field;
pub mod point;
pub mod sw;

#[cfg(feature = "halo2curves")]
pub mod curve;

#[cfg(feature = "halo2curves")]
pub mod pairing;

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
            // Assuming the compiler will unroll this loop
            // so `i` is a compile time constant:
            for i in (0..32 * 12 * 2).step_by(4) {
                axvm::hint_store_u32!(hint.as_ptr() as *const u8, i);
            }
            hint.assume_init()
        }
    }

    // The length must fit into an I-type immediate
    const _: () = assert!(48 * 12 * 2 < (1 << 11));

    /// Writes hint to stack and returns (residue_witness, scaling_factor)
    pub fn bls12_381_final_exp_hint(f: &[u8]) -> [u8; 48 * 12 * 2] {
        debug_assert_eq!(f.len(), 48 * 12);
        let hint = MaybeUninit::<[u8; 48 * 12 * 2]>::uninit();
        unsafe {
            bls12_381_hint_final_exp(f.as_ptr());
            // Assuming the compiler will unroll this loop
            // so `i` is a compile time constant:
            for i in (0..48 * 12 * 2).step_by(4) {
                axvm::hint_store_u32!(hint.as_ptr() as *const u8, i);
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
