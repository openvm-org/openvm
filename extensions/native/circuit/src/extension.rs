use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::{
    arch::{
        MemoryConfig, SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmExtension,
        VmGenericConfig, VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    intrinsics::hashes::poseidon2::Poseidon2Chip,
    rv32im::BranchEqualCoreChip,
    system::{native_adapter::NativeAdapterChip, public_values::PublicValuesChip},
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::*;
use branch_native_adapter::BranchNativeAdapterChip;
use derive_more::derive::From;
use jal_native_adapter::JalNativeAdapterChip;
use loadstore_native_adapter::NativeLoadStoreAdapterChip;
use native_vectorized_adapter::NativeVectorizedAdapterChip;
use program::DEFAULT_PC_STEP;
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeConfig {
    pub system: SystemConfig,
    pub native: Native,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            native: Default::default(),
        }
    }
}

impl NativeConfig {
    pub fn aggregation(num_public_values: usize, poseidon2_max_constraint_degree: usize) -> Self {
        Self {
            system: SystemConfig::new(
                poseidon2_max_constraint_degree,
                MemoryConfig {
                    max_access_adapter_n: 8,
                    ..Default::default()
                },
                num_public_values,
            )
            .with_max_segment_len((1 << 24) - 100),
            native: Default::default(),
        }
    }
}

impl<F: PrimeField32> VmGenericConfig<F> for NativeConfig {
    type Executor = NativeExecutor<F>;
    type Periphery = NativePeriphery<F>;

    fn system(&self) -> &SystemConfig {
        &self.system
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let base = SystemConfig::default().with_continuations();
        let complex = base.create_chip_complex()?;
        complex.extend(&self.native)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Native;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum NativeExecutor<F: PrimeField32> {
    #[any_enum]
    System(SystemExecutor<F>),
    LoadStore(KernelLoadStoreChip<F, 1>),
    BranchEqual(KernelBranchEqChip<F>),
    Jal(KernelJalChip<F>),
    FieldArithmetic(FieldArithmeticChip<F>),
    FieldExtension(FieldExtensionChip<F>),
    PublicValues(PublicValuesChip<F>),
    Poseidon2(Poseidon2Chip<F>),
    FriReducedOpening(FriReducedOpeningChip<F>),
    CastF(CastFChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum NativePeriphery<F: PrimeField32> {
    #[any_enum]
    System(SystemPeriphery<F>),
}

impl<F: PrimeField32> VmExtension<F> for Native {
    type Executor = NativeExecutor<F>;
    type Periphery = NativePeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<NativeExecutor<F>, NativePeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();

        let load_store_chip = KernelLoadStoreChip::<F, 1>::new(
            NativeLoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                NativeLoadStoreOpcode::default_offset(),
            ),
            KernelLoadStoreCoreChip::new(NativeLoadStoreOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            load_store_chip,
            NativeLoadStoreOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let branch_equal_chip = KernelBranchEqChip::new(
            BranchNativeAdapterChip::<_>::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
            ),
            BranchEqualCoreChip::new(NativeBranchEqualOpcode::default_offset(), DEFAULT_PC_STEP),
            memory_controller.clone(),
        );
        inventory.add_executor(
            branch_equal_chip,
            NativeBranchEqualOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let jal_chip = KernelJalChip::new(
            JalNativeAdapterChip::<_>::new(execution_bus, program_bus, memory_controller.clone()),
            JalCoreChip::new(NativeJalOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            jal_chip,
            NativeJalOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let field_arithmetic_chip = FieldArithmeticChip::new(
            NativeAdapterChip::<F, 2, 1>::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
            ),
            FieldArithmeticCoreChip::new(FieldArithmeticOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            field_arithmetic_chip,
            FieldArithmeticOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let field_extension_chip = FieldExtensionChip::new(
            NativeVectorizedAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            FieldExtensionCoreChip::new(FieldExtensionOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            field_extension_chip,
            FieldExtensionOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let fri_reduced_opening_chip = FriReducedOpeningChip::new(
            memory_controller.clone(),
            execution_bus,
            program_bus,
            FriOpcode::default_offset(),
        );
        inventory.add_executor(
            fri_reduced_opening_chip,
            FriOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        Ok(inventory)
    }
}
