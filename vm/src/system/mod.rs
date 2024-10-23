pub mod memory;
pub mod program;
pub mod vm;

/// We use default PC step of 4 whenever possible for consistency with RISC-V, where 4 comes
/// from the fact that each standard RISC-V instruction is 32-bits = 4 bytes.
pub const DEFAULT_PC_STEP: u32 = 4;
pub const PC_BITS: usize = 30;
