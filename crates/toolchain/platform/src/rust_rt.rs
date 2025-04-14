//! This module contains the components required to link a Rust binary.
//!
//! In particular:
//! * It defines an entrypoint ensuring initialization and finalization are done properly.
//! * It includes a panic handler.
//! * It includes an allocator.

/// WARNING: the [SYSTEM_OPCODE] here should be equal to `SYSTEM_OPCODE` in
/// `extensions_rv32im_guest` Can't import `openvm_rv32im_guest` here because would create a
/// circular dependency
#[cfg(target_os = "zkvm")]
/// This is custom-0 defined in RISC-V spec document
const SYSTEM_OPCODE: u8 = 0x0b;

extern crate alloc;

#[inline(always)]
pub fn terminate<const EXIT_CODE: u8>() {
    #[cfg(target_os = "zkvm")]
    crate::custom_insn_i!(
        opcode = SYSTEM_OPCODE,
        funct3 = 0,
        rd = Const "x0",
        rs1 = Const "x0",
        imm = Const EXIT_CODE
    );
    #[cfg(not(target_os = "zkvm"))]
    {
        unimplemented!()
    }
}
