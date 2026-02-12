#![no_std]

use openvm_groth16_guest::{
    groth16_verify_intrinsic,
    Groth16Proof,
    Groth16VerifyingKey
};

/// High-level function for verifying a Groth16 proof inside OpenVM.
pub fn verify_groth16(
    proof: &Groth16Proof,
    vk: &Groth16VerifyingKey,
    public_inputs: &[u32],
) {
    unsafe {
        groth16_verify_intrinsic(
            proof as *const Groth16Proof,
            vk as *const Groth16VerifyingKey,
            public_inputs.as_ptr(),
        );
    }
}
