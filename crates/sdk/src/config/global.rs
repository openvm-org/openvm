use bon::Builder;
use derive_more::derive::From;
use openvm_algebra_circuit::{
    Fp2Extension, Fp2ExtensionExecutor, Fp2ExtensionPeriphery, ModularExtension,
    ModularExtensionExecutor, ModularExtensionPeriphery,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        InitFileGenerator, SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmConfig,
        VmInventoryError,
    },
    circuit_derive::{Chip, ChipUsageGetter},
    derive::{AnyEnum, InstructionExecutor},
};
use openvm_ecc_circuit::{
    WeierstrassExtension, WeierstrassExtensionExecutor, WeierstrassExtensionPeriphery,
};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_native_circuit::{
    CastFExtension, CastFExtensionExecutor, CastFExtensionPeriphery, Native, NativeExecutor,
    NativePeriphery,
};
use openvm_native_transpiler::LongFormTranspilerExtension;
use openvm_pairing_circuit::{
    PairingExtension, PairingExtensionExecutor, PairingExtensionPeriphery,
};
use openvm_pairing_transpiler::PairingTranspilerExtension;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha256Periphery};
use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::transpiler::Transpiler;
use serde::{Deserialize, Serialize};

use crate::F;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
pub struct SdkVmConfig {
    #[serde(default)]
    pub system: SdkSystemConfig,

    pub rv32i: Option<UnitStruct>,
    pub io: Option<UnitStruct>,
    pub keccak: Option<UnitStruct>,
    pub sha256: Option<UnitStruct>,
    pub native: Option<UnitStruct>,
    pub castf: Option<UnitStruct>,

    pub rv32m: Option<Rv32M>,
    pub bigint: Option<Int256>,
    pub modular: Option<ModularExtension>,
    pub fp2: Option<Fp2Extension>,
    pub pairing: Option<PairingExtension>,
    pub ecc: Option<WeierstrassExtension>,
}

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum SdkVmConfigExecutor<F: PrimeField32> {
    #[any_enum]
    System(SystemExecutor<F>),
    #[any_enum]
    Rv32i(Rv32IExecutor<F>),
    #[any_enum]
    Io(Rv32IoExecutor<F>),
    #[any_enum]
    Keccak(Keccak256Executor<F>),
    #[any_enum]
    Sha256(Sha256Executor<F>),
    #[any_enum]
    Native(NativeExecutor<F>),
    #[any_enum]
    Rv32m(Rv32MExecutor<F>),
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
    CastF(CastFExtensionExecutor<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum SdkVmConfigPeriphery<F: PrimeField32> {
    #[any_enum]
    System(SystemPeriphery<F>),
    #[any_enum]
    Rv32i(Rv32IPeriphery<F>),
    #[any_enum]
    Io(Rv32IoPeriphery<F>),
    #[any_enum]
    Keccak(Keccak256Periphery<F>),
    #[any_enum]
    Sha256(Sha256Periphery<F>),
    #[any_enum]
    Native(NativePeriphery<F>),
    #[any_enum]
    Rv32m(Rv32MPeriphery<F>),
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
    CastF(CastFExtensionPeriphery<F>),
}

impl SdkVmConfig {
    pub fn transpiler(&self) -> Transpiler<F> {
        let mut transpiler = Transpiler::default();
        if self.rv32i.is_some() {
            transpiler = transpiler.with_extension(Rv32ITranspilerExtension);
        }
        if self.io.is_some() {
            transpiler = transpiler.with_extension(Rv32IoTranspilerExtension);
        }
        if self.keccak.is_some() {
            transpiler = transpiler.with_extension(Keccak256TranspilerExtension);
        }
        if self.sha256.is_some() {
            transpiler = transpiler.with_extension(Sha256TranspilerExtension);
        }
        if self.native.is_some() {
            transpiler = transpiler.with_extension(LongFormTranspilerExtension);
        }
        if self.rv32m.is_some() {
            transpiler = transpiler.with_extension(Rv32MTranspilerExtension);
        }
        if self.bigint.is_some() {
            transpiler = transpiler.with_extension(Int256TranspilerExtension);
        }
        if self.modular.is_some() {
            transpiler = transpiler.with_extension(ModularTranspilerExtension);
        }
        if self.fp2.is_some() {
            transpiler = transpiler.with_extension(Fp2TranspilerExtension);
        }
        if self.pairing.is_some() {
            transpiler = transpiler.with_extension(PairingTranspilerExtension);
        }
        if self.ecc.is_some() {
            transpiler = transpiler.with_extension(EccTranspilerExtension);
        }
        transpiler
    }
}

impl<F: PrimeField32> VmConfig<F> for SdkVmConfig {
    type Executor = SdkVmConfigExecutor<F>;
    type Periphery = SdkVmConfigPeriphery<F>;

    fn system(&self) -> &SystemConfig {
        &self.system.config
    }

    fn system_mut(&mut self) -> &mut SystemConfig {
        &mut self.system.config
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut complex = self.system.config.create_chip_complex()?.transmute();

        if self.rv32i.is_some() {
            complex = complex.extend(&Rv32I)?;
        }
        if self.io.is_some() {
            complex = complex.extend(&Rv32Io)?;
        }
        if self.keccak.is_some() {
            complex = complex.extend(&Keccak256)?;
        }
        if self.sha256.is_some() {
            complex = complex.extend(&Sha256)?;
        }
        if self.native.is_some() {
            complex = complex.extend(&Native)?;
        }
        if self.castf.is_some() {
            complex = complex.extend(&CastFExtension)?;
        }

        if let Some(rv32m) = self.rv32m {
            let mut rv32m = rv32m;
            if let Some(ref bigint) = self.bigint {
                rv32m.range_tuple_checker_sizes[0] =
                    rv32m.range_tuple_checker_sizes[0].max(bigint.range_tuple_checker_sizes[0]);
                rv32m.range_tuple_checker_sizes[1] =
                    rv32m.range_tuple_checker_sizes[1].max(bigint.range_tuple_checker_sizes[1]);
            }
            complex = complex.extend(&rv32m)?;
        }
        if let Some(bigint) = self.bigint {
            let mut bigint = bigint;
            if let Some(ref rv32m) = self.rv32m {
                bigint.range_tuple_checker_sizes[0] =
                    rv32m.range_tuple_checker_sizes[0].max(bigint.range_tuple_checker_sizes[0]);
                bigint.range_tuple_checker_sizes[1] =
                    rv32m.range_tuple_checker_sizes[1].max(bigint.range_tuple_checker_sizes[1]);
            }
            complex = complex.extend(&bigint)?;
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

        Ok(complex)
    }
}

impl InitFileGenerator for SdkVmConfig {
    fn generate_init_file_contents(&self) -> Option<String> {
        if self.modular.is_some() || self.fp2.is_some() || self.ecc.is_some() {
            let mut contents = String::new();
            contents.push_str(
                "// This file is automatically generated by cargo openvm. Do not rename or edit.\n",
            );

            if let Some(modular_config) = &self.modular {
                contents.push_str(&modular_config.generate_moduli_init());
                contents.push('\n');
            }

            if let Some(fp2_config) = &self.fp2 {
                assert!(
                    self.modular.is_some(),
                    "ModularExtension is required for Fp2Extension"
                );
                let modular_config = self.modular.as_ref().unwrap();
                contents.push_str(&fp2_config.generate_complex_init(modular_config));
                contents.push('\n');
            }

            if let Some(ecc_config) = &self.ecc {
                contents.push_str(&ecc_config.generate_sw_init());
                contents.push('\n');
            }

            Some(contents)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SdkSystemConfig {
    pub config: SystemConfig,
}

// Default implementation uses no init file
impl InitFileGenerator for SdkSystemConfig {}

impl Default for SdkSystemConfig {
    fn default() -> Self {
        Self {
            config: SystemConfig::default().with_continuations(),
        }
    }
}

impl From<SystemConfig> for SdkSystemConfig {
    fn from(config: SystemConfig) -> Self {
        Self { config }
    }
}

/// A struct that is used to represent a unit struct in the config, used for
/// serialization and deserialization.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct UnitStruct {}

impl From<Rv32I> for UnitStruct {
    fn from(_: Rv32I) -> Self {
        UnitStruct {}
    }
}

impl From<Rv32Io> for UnitStruct {
    fn from(_: Rv32Io) -> Self {
        UnitStruct {}
    }
}

impl From<Keccak256> for UnitStruct {
    fn from(_: Keccak256) -> Self {
        UnitStruct {}
    }
}

impl From<Sha256> for UnitStruct {
    fn from(_: Sha256) -> Self {
        UnitStruct {}
    }
}

impl From<Native> for UnitStruct {
    fn from(_: Native) -> Self {
        UnitStruct {}
    }
}

impl From<CastFExtension> for UnitStruct {
    fn from(_: CastFExtension) -> Self {
        UnitStruct {}
    }
}
