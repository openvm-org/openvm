use air::VerifyBatchBus;
use branch_native_adapter::BranchNativeAdapterChip;
use derive_more::derive::From;
use jal_native_adapter::JalNativeAdapterChip;
use loadstore_native_adapter::NativeLoadStoreAdapterChip;
use native_vectorized_adapter::NativeVectorizedAdapterChip;
use openvm_circuit::{
    arch::{
        MemoryConfig, SystemConfig, SystemExecutor, SystemPeriphery, SystemPort, VmChipComplex,
        VmConfig, VmExtension, VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::{native_adapter::NativeAdapterChip, phantom::PhantomChip},
};
use openvm_circuit_derive::{AnyEnum, InstructionExecutor, VmConfig};
use openvm_circuit_primitives_derive::{BytesStateful, Chip, ChipUsageGetter};
use openvm_instructions::{program::DEFAULT_PC_STEP, PhantomDiscriminant, UsizeOpcode, VmOpcode};
use openvm_native_compiler::{
    CastfOpcode, FieldArithmeticOpcode, FieldExtensionOpcode, FriOpcode, NativeBranchEqualOpcode,
    NativeJalOpcode, NativeLoadStore4Opcode, NativeLoadStoreOpcode, NativePhantom, Poseidon2Opcode,
    VerifyBatchOpcode, BLOCK_LOAD_STORE_SIZE,
};
use openvm_poseidon2_air::Poseidon2Config;
use openvm_rv32im_circuit::{
    BranchEqualCoreChip, Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor,
    Rv32IoPeriphery, Rv32M, Rv32MExecutor, Rv32MPeriphery,
};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{
    adapters::{convert_adapter::ConvertAdapterChip, *},
    chip::NativePoseidon2Chip,
    phantom::*,
    *,
};

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

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum, BytesStateful)]
pub enum NativeExecutor<F: PrimeField32> {
    LoadStore(NativeLoadStoreChip<F, 1>),
    BlockLoadStore(NativeLoadStoreChip<F, 4>),
    BranchEqual(NativeBranchEqChip<F>),
    Jal(NativeJalChip<F>),
    FieldArithmetic(FieldArithmeticChip<F>),
    FieldExtension(FieldExtensionChip<F>),
    FriReducedOpening(FriReducedOpeningChip<F>),
    VerifyBatch(NativePoseidon2Chip<F, 1>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum, BytesStateful)]
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
            memory_bridge,
        } = builder.system_port();
        let offline_memory = builder.system_base().offline_memory();

        let mut load_store_chip = NativeLoadStoreChip::<F, 1>::new(
            NativeLoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                NativeLoadStoreOpcode::default_offset(),
            ),
            NativeLoadStoreCoreChip::new(NativeLoadStoreOpcode::default_offset()),
            offline_memory.clone(),
        );
        load_store_chip.core.set_streams(builder.streams().clone());

        inventory.add_executor(
            load_store_chip,
            NativeLoadStoreOpcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let mut block_load_store_chip = NativeLoadStoreChip::<F, BLOCK_LOAD_STORE_SIZE>::new(
            NativeLoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                NativeLoadStore4Opcode::default_offset(),
            ),
            NativeLoadStoreCoreChip::new(NativeLoadStore4Opcode::default_offset()),
            offline_memory.clone(),
        );
        block_load_store_chip
            .core
            .set_streams(builder.streams().clone());

        inventory.add_executor(
            block_load_store_chip,
            NativeLoadStore4Opcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let branch_equal_chip = NativeBranchEqChip::new(
            BranchNativeAdapterChip::<_>::new(execution_bus, program_bus, memory_bridge),
            BranchEqualCoreChip::new(NativeBranchEqualOpcode::default_offset(), DEFAULT_PC_STEP),
            offline_memory.clone(),
        );
        inventory.add_executor(
            branch_equal_chip,
            NativeBranchEqualOpcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let jal_chip = NativeJalChip::new(
            JalNativeAdapterChip::<_>::new(execution_bus, program_bus, memory_bridge),
            JalCoreChip::new(NativeJalOpcode::default_offset()),
            offline_memory.clone(),
        );
        inventory.add_executor(
            jal_chip,
            NativeJalOpcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let field_arithmetic_chip = FieldArithmeticChip::new(
            NativeAdapterChip::<F, 2, 1>::new(execution_bus, program_bus, memory_bridge),
            FieldArithmeticCoreChip::new(FieldArithmeticOpcode::default_offset()),
            offline_memory.clone(),
        );
        inventory.add_executor(
            field_arithmetic_chip,
            FieldArithmeticOpcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let field_extension_chip = FieldExtensionChip::new(
            NativeVectorizedAdapterChip::new(execution_bus, program_bus, memory_bridge),
            FieldExtensionCoreChip::new(FieldExtensionOpcode::default_offset()),
            offline_memory.clone(),
        );
        inventory.add_executor(
            field_extension_chip,
            FieldExtensionOpcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let fri_reduced_opening_chip = FriReducedOpeningChip::new(
            execution_bus,
            program_bus,
            memory_bridge,
            FriOpcode::default_offset(),
            offline_memory.clone(),
        );
        inventory.add_executor(
            fri_reduced_opening_chip,
            FriOpcode::iter().map(VmOpcode::with_default_offset),
        )?;

        let poseidon2_chip = NativePoseidon2Chip::new(
            builder.system_port(),
            VerifyBatchOpcode::default_offset(),
            Poseidon2Opcode::default_offset(),
            offline_memory.clone(),
            Poseidon2Config::default(),
            VerifyBatchBus(builder.new_bus_idx()),
        );
        inventory.add_executor(
            poseidon2_chip,
            [
                VmOpcode::with_default_offset(VerifyBatchOpcode::VERIFY_BATCH),
                VmOpcode::with_default_offset(Poseidon2Opcode::PERM_POS2),
                VmOpcode::with_default_offset(Poseidon2Opcode::COMP_POS2),
            ],
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
    use eyre::bail;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use openvm_instructions::PhantomDiscriminant;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};

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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CastFExtension;

#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum, BytesStateful)]
pub enum CastFExtensionExecutor<F: PrimeField32> {
    CastF(CastFChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum, BytesStateful)]
pub enum CastFExtensionPeriphery<F: PrimeField32> {
    Placeholder(CastFChip<F>),
}

impl<F: PrimeField32> VmExtension<F> for CastFExtension {
    type Executor = CastFExtensionExecutor<F>;
    type Periphery = CastFExtensionPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();
        let offline_memory = builder.system_base().offline_memory();
        let range_checker = builder.system_base().range_checker_chip.clone();

        let castf_chip = CastFChip::new(
            ConvertAdapterChip::new(execution_bus, program_bus, memory_bridge),
            CastFCoreChip::new(range_checker.clone(), CastfOpcode::default_offset()),
            offline_memory.clone(),
        );
        inventory.add_executor(
            castf_chip,
            [VmOpcode::with_default_offset(CastfOpcode::CASTF)],
        )?;

        Ok(inventory)
    }
}

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32WithKernelsConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub native: Native,
    #[extension]
    pub castf: CastFExtension,
}

impl Default for Rv32WithKernelsConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            native: Native,
            castf: CastFExtension,
        }
    }
}
