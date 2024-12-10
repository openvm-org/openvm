use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_poseidon2_air::poseidon2::air::SBOX_DEGREE;
use ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::{
    arch::{
        vm_poseidon2_config, MemoryConfig, SystemConfig, SystemExecutor, SystemPeriphery,
        SystemPort, VmChipComplex, VmConfig, VmExtension, VmInventory, VmInventoryBuilder,
        VmInventoryError,
    },
    system::{native_adapter::NativeAdapterChip, phantom::PhantomChip, poseidon2::Poseidon2Chip},
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use axvm_instructions::{
    program::DEFAULT_PC_STEP, AxVmOpcode, PhantomDiscriminant, Poseidon2Opcode, UsizeOpcode,
};
use axvm_native_compiler::{
    FieldArithmeticOpcode, FieldExtensionOpcode, FriOpcode, NativeBranchEqualOpcode,
    NativeJalOpcode, NativeLoadStoreOpcode, NativePhantom,
};
use axvm_rv32im_circuit::BranchEqualCoreChip;
use branch_native_adapter::BranchNativeAdapterChip;
use derive_more::derive::From;
use jal_native_adapter::JalNativeAdapterChip;
use loadstore_native_adapter::NativeLoadStoreAdapterChip;
use native_vectorized_adapter::NativeVectorizedAdapterChip;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, phantom::*, *};

#[derive(Clone, Debug, Serialize, Deserialize, VmConfig, derive_new::new)]
pub struct NativeConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
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
    pub fn with_continuations(mut self) -> Self {
        self.system = self.system.with_continuations();
        self
    }

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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Native;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum NativeExecutor<F: PrimeField32> {
    LoadStore(NativeLoadStoreChip<F, 1>),
    BranchEqual(NativeBranchEqChip<F>),
    Jal(NativeJalChip<F>),
    FieldArithmetic(FieldArithmeticChip<F>),
    FieldExtension(FieldExtensionChip<F>),
    Poseidon2(Poseidon2Chip<F>),
    FriReducedOpening(FriReducedOpeningChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum NativePeriphery<F: PrimeField32> {
    Phantom(PhantomChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for Native {
    type Executor = NativeExecutor<F>;
    type Periphery = NativePeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<NativeExecutor<F>, NativePeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_controller,
        } = builder.system_port();

        let mut load_store_chip = NativeLoadStoreChip::<F, 1>::new(
            NativeLoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                NativeLoadStoreOpcode::default_offset(),
            ),
            NativeLoadStoreCoreChip::new(NativeLoadStoreOpcode::default_offset()),
            memory_controller.clone(),
        );
        load_store_chip.core.set_streams(builder.streams().clone());

        inventory.add_executor(
            load_store_chip,
            NativeLoadStoreOpcode::iter().map(AxVmOpcode::with_default_offset),
        )?;

        let branch_equal_chip = NativeBranchEqChip::new(
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
            NativeBranchEqualOpcode::iter().map(AxVmOpcode::with_default_offset),
        )?;

        let jal_chip = NativeJalChip::new(
            JalNativeAdapterChip::<_>::new(execution_bus, program_bus, memory_controller.clone()),
            JalCoreChip::new(NativeJalOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            jal_chip,
            NativeJalOpcode::iter().map(AxVmOpcode::with_default_offset),
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
            FieldArithmeticOpcode::iter().map(AxVmOpcode::with_default_offset),
        )?;

        let field_extension_chip = FieldExtensionChip::new(
            NativeVectorizedAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            FieldExtensionCoreChip::new(FieldExtensionOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            field_extension_chip,
            FieldExtensionOpcode::iter().map(AxVmOpcode::with_default_offset),
        )?;

        let fri_reduced_opening_chip = FriReducedOpeningChip::new(
            memory_controller.clone(),
            execution_bus,
            program_bus,
            FriOpcode::default_offset(),
        );
        inventory.add_executor(
            fri_reduced_opening_chip,
            FriOpcode::iter().map(AxVmOpcode::with_default_offset),
        )?;

        let poseidon2_chip = Poseidon2Chip::from_poseidon2_config(
            vm_poseidon2_config(),
            builder
                .system_config()
                .max_constraint_degree
                .min(SBOX_DEGREE),
            execution_bus,
            program_bus,
            memory_controller.clone(),
            builder.new_bus_idx(),
            Poseidon2Opcode::default_offset(),
        );
        inventory.add_executor(
            poseidon2_chip,
            Poseidon2Opcode::iter().map(AxVmOpcode::with_default_offset),
        )?;

        builder.add_phantom_sub_executor(
            NativeHintInputSubEx,
            PhantomDiscriminant(NativePhantom::HintInput as u16),
        )?;

        builder.add_phantom_sub_executor(
            NativeHintBitsSubEx,
            PhantomDiscriminant(NativePhantom::HintBits as u16),
        )?;

        builder.add_phantom_sub_executor(
            NativePrintSubEx,
            PhantomDiscriminant(NativePhantom::Print as u16),
        )?;

        Ok(inventory)
    }
}

pub(crate) mod phantom {
    use ax_stark_backend::p3_field::{Field, PrimeField32};
    use axvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use axvm_instructions::PhantomDiscriminant;
    use eyre::bail;

    pub struct NativeHintInputSubEx;
    pub struct NativePrintSubEx;
    pub struct NativeHintBitsSubEx;

    impl<F: Field> PhantomSubExecutor<F> for NativeHintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            let hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.clear();
            streams
                .hint_stream
                .push_back(F::from_canonical_usize(hint.len()));
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativePrintSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            _: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            _: F,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let addr_space = F::from_canonical_u16(c_upper);
            let value = memory.unsafe_read_cell(addr_space, a);
            println!("{}", value);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for NativeHintBitsSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            c_upper: u16,
        ) -> eyre::Result<()> {
            let addr_space = F::from_canonical_u16(c_upper);
            let val = memory.unsafe_read_cell(addr_space, a);
            let mut val = val.as_canonical_u32();

            let len = b.as_canonical_u32();
            streams.hint_stream.clear();
            for _ in 0..len {
                streams
                    .hint_stream
                    .push_back(F::from_canonical_u32(val & 1));
                val >>= 1;
            }
            Ok(())
        }
    }
}