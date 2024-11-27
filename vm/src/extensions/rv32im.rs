use std::sync::Arc;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use axvm_circuit_derive::{AnyEnum, InstructionExecutor};
use axvm_instructions::*;
use derive_more::derive::From;
use p3_field::PrimeField32;
use program::DEFAULT_PC_STEP;
use strum::IntoEnumIterator;

use crate::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmExtension, VmGenericConfig,
        VmInventory, VmInventoryBuilder, VmInventoryError,
    },
    rv32im::{adapters::*, *},
    system::phantom::PhantomChip,
};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32IConfig {
    pub system: SystemConfig,
    pub base: Rv32I,
    // todo: hintstore
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32ImConfig {
    pub system: SystemConfig,
    pub base: Rv32I,
    pub mul: Rv32M,
    // todo: hintstore
}

impl Default for Rv32IConfig {
    fn default() -> Self {
        let system = SystemConfig::default().with_continuations();
        Self {
            system,
            base: Default::default(),
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
        }
    }
}

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32I;

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

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32Periphery<F: PrimeField32> {
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    /// Only needed for multiplication extension
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

// TODO: generate this by proc-macro
/// RISC-V 32-bit IM (Base + Multiplication) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
pub enum Rv32ImExecutor<F: PrimeField32> {
    #[any_enum]
    System(SystemExecutor<F>),
    #[any_enum]
    Base(Rv32IExecutor<F>),
    #[any_enum]
    Mul(Rv32MExecutor<F>),
}

// TODO: generate this by proc-macro
#[derive(ChipUsageGetter, Chip, From, AnyEnum)]
pub enum Rv32ImPeriphery<F: PrimeField32> {
    #[any_enum]
    System(SystemPeriphery<F>),
    #[any_enum]
    Rv32(Rv32Periphery<F>),
}

// TODO: generate this by proc-macro
impl<F: PrimeField32> VmGenericConfig<F> for Rv32IConfig {
    type Executor = Rv32ImExecutor<F>;
    type Periphery = Rv32ImPeriphery<F>;

    fn system(&self) -> &SystemConfig {
        &self.system
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let complex = self.system.create_chip_complex()?;
        let complex = complex.extend(&self.base)?;
        Ok(complex)
    }
}

// TODO: generate this by proc-macro
impl<F: PrimeField32> VmGenericConfig<F> for Rv32ImConfig {
    type Executor = Rv32ImExecutor<F>;
    type Periphery = Rv32ImPeriphery<F>;

    fn system(&self) -> &SystemConfig {
        &self.system
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let base = Rv32IConfig {
            system: self.system,
            base: self.base,
        };
        let complex = base.create_chip_complex()?;
        let complex = complex.extend(&self.mul)?;
        Ok(complex)
    }
}

impl<F: PrimeField32> VmExtension<F> for Rv32I {
    type Executor = Rv32IExecutor<F>;
    type Periphery = Rv32Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Rv32IExecutor<F>, Rv32Periphery<F>>, VmInventoryError> {
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
    type Periphery = Rv32Periphery<F>;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Rv32MExecutor<F>, Rv32Periphery<F>>, VmInventoryError> {
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

pub(crate) mod phantom {
    use axvm_instructions::PhantomDiscriminant;
    use eyre::bail;
    use p3_field::{Field, PrimeField32};

    use crate::{
        arch::{PhantomSubExecutor, Streams},
        rv32im::adapters::unsafe_read_rv32_register,
        system::memory::MemoryController,
    };

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
