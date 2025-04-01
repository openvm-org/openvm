use openvm_algebra_circuit::*;
use openvm_circuit::arch::SystemConfig;
use openvm_circuit_derive::VmConfig;
use openvm_ecc_circuit::*;
use openvm_rv32im_circuit::*;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct Rv32PairingConfig {
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
    #[extension]
    pub weierstrass: WeierstrassExtension,
    #[extension]
    pub pairing: PairingExtension,
}

impl Rv32PairingConfig {
    pub fn new(curves: Vec<PairingCurve>) -> Self {
        let mut primes: Vec<_> = curves
            .iter()
            .map(|c| c.curve_config().modulus.clone())
            .collect();
        primes.extend(curves.iter().map(|c| c.curve_config().scalar.clone()));
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(primes.to_vec()),
            fp2: Fp2Extension::new(primes.to_vec()),
            weierstrass: WeierstrassExtension::new(
                curves.iter().map(|c| c.curve_config()).collect(),
            ),
            pairing: PairingExtension::new(curves),
        }
    }
}
