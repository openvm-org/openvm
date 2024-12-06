use ax_stark_sdk::config::FriParameters;
use axvm_algebra_circuit::{
    Fp2Extension, Fp2ExtensionExecutor, Fp2ExtensionPeriphery, ModularExtension,
    ModularExtensionExecutor, ModularExtensionPeriphery,
};
use axvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery};
use axvm_circuit::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmConfig, VmInventoryError,
    },
    circuit_derive::{Chip, ChipUsageGetter},
    derive::{AnyEnum, InstructionExecutor},
};
use axvm_ecc_circuit::{
    WeierstrassExtension, WeierstrassExtensionExecutor, WeierstrassExtensionPeriphery,
};
use axvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
use axvm_native_circuit::{Native, NativeExecutor, NativePeriphery};
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_pairing_circuit::{PairingExtension, PairingExtensionExecutor, PairingExtensionPeriphery};
use axvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use bon::Builder;
use derive_more::derive::From;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::F;

#[derive(Clone, Debug)]
pub struct AppConfig<VC: VmConfig<F>> {
    pub app_fri_params: FriParameters,
    pub app_vm_config: VC,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AggConfig {
    pub max_num_user_public_values: usize,
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub root_fri_params: FriParameters,
    pub compiler_options: CompilerOptions,
}

#[derive(Builder, Clone, Debug)]
pub struct SdkVmConfig {
    pub system: SystemConfig,
    pub rv32i: Option<Rv32I>,
    pub rv32m: Option<Rv32M>,
    pub io: Option<Rv32Io>,
    pub bigint: Option<Int256>,
    pub modular: Option<ModularExtension>,
    pub fp2: Option<Fp2Extension>,
    pub pairing: Option<PairingExtension>,
    pub ecc: Option<WeierstrassExtension>,
    pub native: Option<Native>,
    pub keccak: Option<Keccak256>,
}

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum SdkVmConfigExecutor<F: PrimeField32> {
    #[any_enum]
    System(SystemExecutor<F>),
    #[any_enum]
    Rv32i(Rv32IExecutor<F>),
    #[any_enum]
    Rv32m(Rv32MExecutor<F>),
    #[any_enum]
    Io(Rv32IoExecutor<F>),
    #[any_enum]
    BigInt(Int256Executor<F>),
    #[any_enum]
    Modular(ModularExtensionExecutor<F>),
    #[any_enum]
    Fp2(Fp2ExtensionExecutor<F>),
    #[any_enum]
    Pairing(PairingExtensionExecutor<F>),
    #[any_enum]
    Ecc(WeierstrassExtensionExecutor<F>),
    #[any_enum]
    Native(NativeExecutor<F>),
    #[any_enum]
    Keccak(Keccak256Executor<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum SdkVmConfigPeriphery<F: PrimeField32> {
    #[any_enum]
    System(SystemPeriphery<F>),
    #[any_enum]
    Rv32i(Rv32IPeriphery<F>),
    #[any_enum]
    Rv32m(Rv32MPeriphery<F>),
    #[any_enum]
    Io(Rv32IoPeriphery<F>),
    #[any_enum]
    BigInt(Int256Periphery<F>),
    #[any_enum]
    Modular(ModularExtensionPeriphery<F>),
    #[any_enum]
    Fp2(Fp2ExtensionPeriphery<F>),
    #[any_enum]
    Pairing(PairingExtensionPeriphery<F>),
    #[any_enum]
    Ecc(WeierstrassExtensionPeriphery<F>),
    #[any_enum]
    Native(NativePeriphery<F>),
    #[any_enum]
    Keccak(Keccak256Periphery<F>),
}

impl<F: PrimeField32> VmConfig<F> for SdkVmConfig {
    type Executor = SdkVmConfigExecutor<F>;
    type Periphery = SdkVmConfigPeriphery<F>;

    fn system(&self) -> &SystemConfig {
        &self.system
    }

    fn system_mut(&mut self) -> &mut SystemConfig {
        &mut self.system
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut complex = self.system.create_chip_complex()?.transmute();

        if let Some(ref rv32i) = self.rv32i {
            complex = complex.extend(rv32i)?;
        }
        if let Some(ref rv32m) = self.rv32m {
            complex = complex.extend(rv32m)?;
        }
        if let Some(ref io) = self.io {
            complex = complex.extend(io)?;
        }
        if let Some(ref bigint) = self.bigint {
            complex = complex.extend(bigint)?;
        }
        if let Some(ref modular) = self.modular {
            complex = complex.extend(modular)?;
        }
        if let Some(ref fp2) = self.fp2 {
            complex = complex.extend(fp2)?;
        }
        if let Some(ref pairing) = self.pairing {
            complex = complex.extend(pairing)?;
        }
        if let Some(ref ecc) = self.ecc {
            complex = complex.extend(ecc)?;
        }
        if let Some(ref native) = self.native {
            complex = complex.extend(native)?;
        }
        if let Some(ref keccak) = self.keccak {
            complex = complex.extend(keccak)?;
        }

        Ok(complex)
    }
}
