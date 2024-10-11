//! A transpiler from custom RISC-V ELFs to axVM machine code.

pub mod elf;
pub mod rrs;
pub mod util;

#[cfg(test)]
mod tests;
