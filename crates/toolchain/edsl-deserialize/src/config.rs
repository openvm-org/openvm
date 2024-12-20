use derive_more::derive::From;
use openvm_circuit::arch::{
    SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmConfig, VmInventoryError,
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_native_circuit::{Native, *};
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::castf_extension::{CastF, CastFExecutor, CastFPeriphery};

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32WithKernelsConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub native: Native,
    #[extension]
    pub castf: CastF,
}

impl Default for Rv32WithKernelsConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            native: Native,
            castf: CastF,
        }
    }
}
