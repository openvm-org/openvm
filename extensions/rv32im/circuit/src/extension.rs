use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmExtension, VmGenericConfig,
        VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor, VmGenericConfig};
use axvm_instructions::*;
use derive_more::derive::From;
use program::DEFAULT_PC_STEP;
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

/// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, VmGenericConfig, derive_new::new)]
pub struct Rv32IConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub io: Rv32Io,
}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, VmGenericConfig, derive_new::new)]
pub struct Rv32ImConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub io: Rv32Io,
}

impl Default for Rv32IConfig {
    fn default() -> Self {
        let system = SystemConfig::default().with_continuations();
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Default for Rv32ImConfig {
    fn default() -> Self {
        let inner = Rv32IConfig::default();
        Self {
            system: inner.system,
            base: inner.base,
            mul: Default::default(),
            io: Default::default(),
        }
    }
}

// ============ Extension Implementations ============

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32I;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32Io;

/// RISC-V 32-bit Multiplication Extension (RV32M) Extension
#[derive(Clone, Copy, Debug)]
pub struct Rv32M {
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Rv32M {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: [1 << 8, 8 * (1 << 8)],
        }
    }
}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 32-bit Base (RV32I) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32IExecutor<F: PrimeField32> {
    // Rv32 (for standard 32-bit integers):
    BaseAlu(Rv32BaseAluChip<F>),
    LessThan(Rv32LessThanChip<F>),
    Shift(Rv32ShiftChip<F>),
    LoadStore(Rv32LoadStoreChip<F>),
    LoadSignExtend(Rv32LoadSignExtendChip<F>),
    BranchEqual(Rv32BranchEqualChip<F>),
    BranchLessThan(Rv32BranchLessThanChip<F>),
    JalLui(Rv32JalLuiChip<F>),
    Jalr(Rv32JalrChip<F>),
    Auipc(Rv32AuipcChip<F>),
}

/// RISC-V 32-bit Multiplication Extension (RV32M) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32MExecutor<F: PrimeField32> {
    Multiplication(Rv32MultiplicationChip<F>),
    MultiplicationHigh(Rv32MulHChip<F>),
    DivRem(Rv32DivRemChip<F>),
}

/// RISC-V 32-bit Io Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32IoExecutor<F: PrimeField32> {
    HintStore(Rv32HintStoreChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32IPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32MPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    /// Only needed for multiplication extension
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32IoPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExtension<F> for Rv32I {
    type Executor = Rv32IExecutor<F>;
    type Periphery = Rv32IPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Rv32IExecutor<F>, Rv32IPeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
        let range_checker = builder.system_base().range_checker_chip.clone();
        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            Arc::clone(chip)
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lu_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let base_alu_chip = Rv32BaseAluChip::new(
            Rv32BaseAluAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            BaseAluCoreChip::new(bitwise_lu_chip.clone(), BaseAluOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            base_alu_chip,
            BaseAluOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let lt_chip = Rv32LessThanChip::new(
            Rv32BaseAluAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            LessThanCoreChip::new(bitwise_lu_chip.clone(), LessThanOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            lt_chip,
            LessThanOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let shift_chip = Rv32ShiftChip::new(
            Rv32BaseAluAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            ShiftCoreChip::new(
                bitwise_lu_chip.clone(),
                range_checker.clone(),
                ShiftOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            shift_chip,
            ShiftOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let load_store_chip = Rv32LoadStoreChip::new(
            Rv32LoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                range_checker.clone(),
                Rv32LoadStoreOpcode::default_offset(),
            ),
            LoadStoreCoreChip::new(Rv32LoadStoreOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            load_store_chip,
            Rv32LoadStoreOpcode::iter()
                .take(Rv32LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.with_default_offset()),
        )?;

        let load_sign_extend_chip = Rv32LoadSignExtendChip::new(
            Rv32LoadStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                range_checker.clone(),
                Rv32LoadStoreOpcode::default_offset(),
            ),
            LoadSignExtendCoreChip::new(
                range_checker.clone(),
                Rv32LoadStoreOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            load_sign_extend_chip,
            [Rv32LoadStoreOpcode::LOADB, Rv32LoadStoreOpcode::LOADH]
                .map(|x| x.with_default_offset()),
        )?;

        let beq_chip = Rv32BranchEqualChip::new(
            Rv32BranchAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            BranchEqualCoreChip::new(BranchEqualOpcode::default_offset(), DEFAULT_PC_STEP),
            memory_controller.clone(),
        );
        inventory.add_executor(
            beq_chip,
            BranchEqualOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let blt_chip = Rv32BranchLessThanChip::new(
            Rv32BranchAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            BranchLessThanCoreChip::new(
                bitwise_lu_chip.clone(),
                BranchLessThanOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            blt_chip,
            BranchLessThanOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let jal_lui_chip = Rv32JalLuiChip::new(
            Rv32CondRdWriteAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            Rv32JalLuiCoreChip::new(bitwise_lu_chip.clone(), Rv32JalLuiOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            jal_lui_chip,
            Rv32JalLuiOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let jalr_chip = Rv32JalrChip::new(
            Rv32JalrAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            Rv32JalrCoreChip::new(
                bitwise_lu_chip.clone(),
                range_checker.clone(),
                Rv32JalrOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            jalr_chip,
            Rv32JalrOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let auipc_chip = Rv32AuipcChip::new(
            Rv32RdWriteAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            Rv32AuipcCoreChip::new(bitwise_lu_chip.clone(), Rv32AuipcOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(
            auipc_chip,
            Rv32AuipcOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        builder.add_phantom_sub_executor(
            phantom::Rv32HintInputSubEx,
            PhantomDiscriminant(Rv32Phantom::HintInput as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::Rv32PrintStrSubEx,
            PhantomDiscriminant(Rv32Phantom::PrintStr as u16),
        )?;

        Ok(inventory)
    }
}

impl<F: PrimeField32> VmExtension<F> for Rv32M {
    type Executor = Rv32MExecutor<F>;
    type Periphery = Rv32MPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Rv32MExecutor<F>, Rv32MPeriphery<F>>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();

        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            Arc::clone(chip)
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lu_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let range_tuple_checker = if let Some(chip) = builder
            .find_chip::<Arc<RangeTupleCheckerChip<2>>>()
            .into_iter()
            .find(|c| {
                c.bus().sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus().sizes[1] >= self.range_tuple_checker_sizes[1]
            }) {
            chip.clone()
        } else {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(builder.new_bus_idx(), self.range_tuple_checker_sizes);
            let chip = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mul_chip = Rv32MultiplicationChip::new(
            Rv32MultAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            MultiplicationCoreChip::new(range_tuple_checker.clone(), MulOpcode::default_offset()),
            memory_controller.clone(),
        );
        inventory.add_executor(mul_chip, MulOpcode::iter().map(|x| x.with_default_offset()))?;

        let mul_h_chip = Rv32MulHChip::new(
            Rv32MultAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            MulHCoreChip::new(
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
                MulHOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            mul_h_chip,
            MulHOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        let div_rem_chip = Rv32DivRemChip::new(
            Rv32MultAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
            DivRemCoreChip::new(
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
                DivRemOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        inventory.add_executor(
            div_rem_chip,
            DivRemOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        Ok(inventory)
    }
}

impl<F: PrimeField32> VmExtension<F> for Rv32Io {
    type Executor = Rv32IoExecutor<F>;
    type Periphery = Rv32IoPeriphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError> {
        let mut inventory = VmInventory::new();
        let execution_bus = builder.system_base().execution_bus();
        let program_bus = builder.system_base().program_bus();
        let memory_controller = builder.memory_controller().clone();
        let range_checker = builder.system_base().range_checker_chip.clone();
        let bitwise_lu_chip = if let Some(chip) = builder
            .find_chip::<Arc<BitwiseOperationLookupChip<8>>>()
            .first()
        {
            Arc::clone(chip)
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lu_bus));
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mut hintstore_chip = Rv32HintStoreChip::new(
            Rv32HintStoreAdapterChip::new(
                execution_bus,
                program_bus,
                memory_controller.clone(),
                range_checker.clone(),
            ),
            Rv32HintStoreCoreChip::new(
                bitwise_lu_chip.clone(),
                Rv32HintStoreOpcode::default_offset(),
            ),
            memory_controller.clone(),
        );
        hintstore_chip.core.set_streams(builder.streams().clone());

        inventory.add_executor(
            hintstore_chip,
            Rv32HintStoreOpcode::iter().map(|x| x.with_default_offset()),
        )?;

        Ok(inventory)
    }
}

/// Phantom sub-executors
mod phantom {
    use ax_stark_backend::p3_field::{Field, PrimeField32};
    use axvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::MemoryController,
    };
    use axvm_instructions::PhantomDiscriminant;
    use eyre::bail;

    use crate::adapters::unsafe_read_rv32_register;

    pub struct Rv32HintInputSubEx;
    pub struct Rv32PrintStrSubEx;

    impl<F: Field> PhantomSubExecutor<F> for Rv32HintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &MemoryController<F>,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: F,
            _: F,
            _: u16,
        ) -> eyre::Result<()> {
            let mut hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                (hint.len() as u32)
                    .to_le_bytes()
                    .iter()
                    .map(|b| F::from_canonical_u8(*b)),
            );
            // Extend by 0 for 4 byte alignment
            let capacity = hint.len().div_ceil(4) * 4;
            hint.resize(capacity, F::ZERO);
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32PrintStrSubEx {
        fn phantom_execute(
            &mut self,
            memory: &MemoryController<F>,
            _: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: F,
            b: F,
            _: u16,
        ) -> eyre::Result<()> {
            let rd = unsafe_read_rv32_register(memory, a);
            let rs1 = unsafe_read_rv32_register(memory, b);
            let bytes = (0..rs1)
                .map(|i| -> eyre::Result<u8> {
                    let val = memory.unsafe_read_cell(F::TWO, F::from_canonical_u32(rd + i));
                    let byte: u8 = val.as_canonical_u32().try_into()?;
                    Ok(byte)
                })
                .collect::<eyre::Result<Vec<u8>>>()?;
            let peeked_str = String::from_utf8(bytes)?;
            println!("{peeked_str}");
            Ok(())
        }
    }
}
