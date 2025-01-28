use openvm_algebra_circuit::*;
use openvm_circuit::arch::SystemConfig;
use openvm_circuit_derive::VmConfig;
use openvm_rv32im_circuit::*;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct Rv32EccConfig {
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
    pub ecc: EccExtension,
}

impl Rv32EccConfig {
    pub fn new(
        sw_curves: Vec<CurveConfig<SwCurveCoeffs>>,
        te_curves: Vec<CurveConfig<TeCurveCoeffs>>,
    ) -> Self {
        let sw_primes: Vec<_> = sw_curves
            .iter()
            .flat_map(|c| [c.modulus.clone(), c.scalar.clone()])
            .collect();
        let te_primes: Vec<_> = te_curves
            .iter()
            .flat_map(|c| [c.modulus.clone(), c.scalar.clone()])
            .collect();
        let primes = sw_primes.into_iter().chain(te_primes).collect();
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(primes),
            ecc: EccExtension::new(sw_curves, te_curves),
        }
    }
}
