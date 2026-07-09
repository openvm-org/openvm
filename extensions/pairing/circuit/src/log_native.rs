//! Log-native registration for pairing configurations.

use openvm_algebra_circuit::{
    log_native::{Fp2RecordArena, ModularRecordArena},
    Rv64ModularConfig,
};
use openvm_circuit::arch::{
    rvr::{LogNativeAssemblerRegistry, VmRvrLogNativeExtension},
    Arena,
};
use openvm_riscv_circuit::log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{PairingExtension, Rv64PairingConfig};

/// Record-arena requirements contributed by pairing arithmetic.
///
/// Miller-loop and sparse-line operations lower to Fp2 instructions in the
/// current pairing implementation, so they use the algebra extension's
/// VecHeap-backed record layouts. The pairing extension's final-exp hint is a
/// shared PHANTOM instruction and contributes no additional record layout.
pub trait PairingRecordArena<F>: Fp2RecordArena<F> {}

impl<F, RA> PairingRecordArena<F> for RA where RA: Fp2RecordArena<F> {}

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
        self.pairing.extend_rvr_log_native(registry);
    }
}
