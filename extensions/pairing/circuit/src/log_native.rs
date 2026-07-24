//! Log-native registration for pairing configurations.

use openvm_algebra_circuit::{log_native::ModularRecordArena, Rv64ModularConfig};
use openvm_circuit::arch::{
    rvr::{LogNativeAssemblerRegistry, VmRvrLogNativeExtension},
    Arena,
};
use openvm_ecc_circuit::log_native::WeierstrassRecordArena;
use openvm_riscv_circuit::log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{PairingExtension, Rv64PairingConfig};

/// Record-arena requirements contributed by a composed pairing configuration.
///
/// Miller-loop and sparse-line operations lower to Fp2 instructions in the
/// current pairing implementation. The config also owns Weierstrass G1
/// instructions; their arena bounds include both two-source Fp2-compatible
/// layouts and the one-source EC-double layouts. The pairing extension's
/// final-exp hint is a shared PHANTOM instruction and contributes no additional
/// record layout.
pub trait PairingRecordArena<F>: WeierstrassRecordArena<F> {}

impl<F, RA> PairingRecordArena<F> for RA where RA: WeierstrassRecordArena<F> {}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for PairingExtension
where
    F: PrimeField32,
    RA: PairingRecordArena<F>,
{
    fn extend_rvr_log_native(&self, _registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        // HARD-1: Rv64I owns SystemOpcode::PHANTOM. Pairing's hint uses that
        // assembler and must not duplicate the shared opcode registration.
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Rv64PairingConfig
where
    F: PrimeField32,
    RA: Arena
        + Rv64IRecordArena<F>
        + Rv64MRecordArena<F>
        + Rv64IoRecordArena<F>
        + ModularRecordArena<F>
        + PairingRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        Rv64ModularConfig::extend_rvr_log_native(&self.modular, registry);
        self.fp2.extend_rvr_log_native(registry);
        self.weierstrass.extend_rvr_log_native(registry);
        self.pairing.extend_rvr_log_native(registry);
    }
}

#[cfg(test)]
mod tests {
    use openvm_algebra_transpiler::Fp2Opcode;
    use openvm_circuit::arch::MatrixRecordArena;
    use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode, SystemOpcode,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;
    use crate::{PairingCurve, BN254_COMPLEX_STRUCT_NAME};

    #[test]
    fn mixed_weierstrass_and_pairing_registry_composes() {
        let config = Rv64PairingConfig::new(
            vec![PairingCurve::Bn254],
            vec![BN254_COMPLEX_STRUCT_NAME.to_string()],
        );
        let mut registry =
            LogNativeAssemblerRegistry::<BabyBear, MatrixRecordArena<BabyBear>>::new();
        config.extend_rvr_log_native(&mut registry);

        let instruction = |opcode| {
            Instruction::from_usize(
                opcode,
                [1, 2, 3, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
            )
        };
        assert!(registry.contains_instruction(&instruction(Fp2Opcode::ADD.global_opcode())));
        assert!(registry.contains_instruction(&instruction(
            Rv64WeierstrassOpcode::EC_ADD_NE.global_opcode(),
        )));
        assert!(registry.contains_instruction(&Instruction::from_usize(
            SystemOpcode::PHANTOM.global_opcode(),
            [0; 5],
        )));
    }
}
