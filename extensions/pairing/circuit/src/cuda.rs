//! GPU builder where the [Rv64ModularBuilder], [AlgebraProverExt], and [EccProverExt] will use
//! either cuda tracegen or hybrid CPU tracegen depending on what [openvm_algebra_circuit] and
//! [openvm_ecc_circuit] crates export.
use openvm_algebra_circuit::{AlgebraProverExt, Rv64ModularBuilder};
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex,
        VmProverExtension,
    },
    system::cuda::SystemChipInventoryGPU,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_ecc_circuit::EccProverExt;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{PairingProverExt, Rv64PairingConfig};

#[derive(Clone)]
pub struct Rv64PairingGpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv64PairingGpuBuilder {
    type VmConfig = Rv64PairingConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64PairingConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ModularBuilder,
            &config.modular,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&AlgebraProverExt, &config.fp2, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&EccProverExt, &config.weierstrass, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&PairingProverExt, &config.pairing, inventory)?;
        Ok(chip_complex)
    }
}
