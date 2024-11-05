#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;

#[cfg(target_os = "zkvm")]
use axvm_platform::constants::{Custom0Funct3, CUSTOM_0};

/// The keccak256 cryptographic hash function.
#[inline(always)]
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut output = [0u8; 32];
        set_keccak256(input, &mut output);
        output
    }
    #[cfg(target_os = "zkvm")]
    {
        let output = MaybeUninit::<[u8; 32]>::uninit();
        axvm_platform::custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Keccak256 as u8,
            0x0,
            output.as_ptr(),
            input.as_ptr(),
            input.len()
        );
        unsafe { output.assume_init() }
    }
}

/// Sets `output` to the keccak256 hash of `input`.
#[inline(always)]
pub fn set_keccak256(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use tiny_keccak::Hasher;
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(input);
        hasher.finalize(output);
    }
    #[cfg(target_os = "zkvm")]
    {
        axvm_platform::custom_insn_r!(
            CUSTOM_0,
            Custom0Funct3::Keccak256 as u8,
            0x0,
            output.as_ptr(),
            input.as_ptr(),
            input.len()
        );
    }
}
