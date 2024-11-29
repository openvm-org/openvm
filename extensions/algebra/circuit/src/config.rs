use ax_circuit_derive::{Chip, ChipUsageGetter};
use axvm_circuit::arch::{
    SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmGenericConfig, VmInventoryError,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor, VmGenericConfig};
use axvm_rv32im_circuit::*;
use derive_more::derive::From;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;

use super::*;

#[derive(Clone, Debug, VmGenericConfig)]
pub struct Rv32ModularConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub modular: ModularExtension,
}

impl Rv32ModularConfig {
    pub fn new(moduli: Vec<BigUint>) -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(moduli),
        }
    }
}

#[derive(Clone, Debug, VmGenericConfig)]
pub struct Rv32ModularWithFp2Config {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub modular: ModularExtension,
    #[extension]
    pub fp2: Fp2Extension,
}

impl Rv32ModularWithFp2Config {
    pub fn new(moduli: Vec<BigUint>) -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(moduli.clone()),
            fp2: Fp2Extension::new(moduli),
        }
    }
}
