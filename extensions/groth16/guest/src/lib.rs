#![no_std]

pub const GROTH16_VERIFY_OPCODE: u8 = 0x50;

#[repr(C)]
pub struct Groth16Proof {
    pub a: [u32; 8],  // G1
    pub b: [u32; 16], // G2
    pub c: [u32; 8],  // G1
}

#[repr(C)]
pub struct Groth16VerifyingKey {
    pub alpha_g1: [u32; 8],
    pub beta_g2: [u32; 16],
    pub gamma_g2: [u32; 16],
    pub delta_g2: [u32; 16],
}

#[inline(always)]
pub fn groth16_verify_intrinsic(proof: *const Groth16Proof, vk: *const Groth16VerifyingKey, public_ptr: *const u32) {
    #[cfg(target_os = "zkvm")]
    unsafe {
        use core::arch::asm;
        asm!(
            ".insn r 0x7b, 0, {opcode}, x0, {rs1}, {rs2}",
            opcode = const GROTH16_VERIFY_OPCODE,
            rs1 = in(reg) proof,
            rs2 = in(reg) vk,
        );
    }
    #[cfg(not(target_os = "zkvm"))]
    {
        let _ = (proof, vk, public_ptr);
    }
}
