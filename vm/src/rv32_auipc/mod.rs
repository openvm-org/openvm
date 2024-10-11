mod integration;

pub use integration::*;

use crate::arch::{MachineChipWrapper, Rv32RdWriteAdapter};

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip<F> = MachineChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32AuipcIntegration>;
