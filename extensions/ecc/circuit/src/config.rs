use ax_circuit_derive::{Chip, ChipUsageGetter};
use axvm_algebra_circuit::*;
use axvm_circuit::arch::{
    SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmConfig, VmInventoryError,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use axvm_rv32im_circuit::*;
use derive_more::derive::From;
use ax_stark_backend::p3_field::PrimeField32;

use super::*;

#[derive(Clone, Debug, VmConfig)]
pub struct Rv32WeierstrassConfig {
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
    pub weierstrass: WeierstrassExtension,
}

impl Rv32WeierstrassConfig {
    pub fn new(curves: Vec<CurveConfig>) -> Self {
        let mut primes: Vec<_> = curves.iter().map(|c| c.modulus.clone()).collect();
        primes.extend(curves.iter().map(|c| c.scalar.clone()));
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(primes),
            weierstrass: WeierstrassExtension::new(curves),
        }
    }
}
