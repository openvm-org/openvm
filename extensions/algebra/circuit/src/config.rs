use num_bigint::BigUint;
use openvm_circuit::arch::SystemConfig;
use openvm_circuit_derive::VmConfig;
use openvm_rv32im_circuit::*;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
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

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
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
