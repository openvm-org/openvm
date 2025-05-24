use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        ExecutionBridge, SystemConfig, SystemPort, VmAirWrapper, VmExtension, VmInventory,
        VmInventoryBuilder, VmInventoryError,
    },
    system::phantom::PhantomChip,
};
use openvm_circuit_derive::{AnyEnum, InsExecutorE1, InstructionExecutor, VmConfig};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, Rv32AuipcOpcode, Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode,
    Rv32LoadStoreOpcode, Rv32Phantom, ShiftOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

// TODO(ayush): this should be decided after e2 execution
const MAX_INS_CAPACITY: usize = 1 << 22;

/// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32IConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub io: Rv32Io,
}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32ImConfig {
    #[config]
    pub rv32i: Rv32IConfig,
    #[extension]
    pub mul: Rv32M,
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

impl Rv32IConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv32ImConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        Self {
            rv32i: Rv32IConfig::with_public_values(public_values),
            mul: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        Self {
            rv32i: Rv32IConfig::with_public_values_and_segment_len(public_values, segment_len),
            mul: Default::default(),
        }
    }
}

// ============ Extension Implementations ============

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv32I;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv32Io;

/// RISC-V 32-bit Multiplication Extension (RV32M) Extension
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv32M {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Rv32M {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 8 * (1 << 8)]
}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 32-bit Base (RV32I) Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, InsExecutorE1, From, AnyEnum)]
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
#[derive(ChipUsageGetter, Chip, InstructionExecutor, InsExecutorE1, From, AnyEnum)]
pub enum Rv32MExecutor<F: PrimeField32> {
    Multiplication(Rv32MultiplicationChip<F>),
    MultiplicationHigh(Rv32MulHChip<F>),
    DivRem(Rv32DivRemChip<F>),
}

/// RISC-V 32-bit Io Instruction Executors
#[derive(ChipUsageGetter, Chip, InstructionExecutor, InsExecutorE1, From, AnyEnum)]
pub enum Rv32IoExecutor<F: PrimeField32> {
    HintStore(Rv32HintStoreChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32IPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32MPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
    /// Only needed for multiplication extension
    RangeTupleChecker(SharedRangeTupleCheckerChip<2>),
    // We put this only to get the <F> generic to work
    Phantom(PhantomChip<F>),
}

#[derive(From, ChipUsageGetter, Chip, AnyEnum)]
pub enum Rv32IoPeriphery<F: PrimeField32> {
    BitwiseOperationLookup(SharedBitwiseOperationLookupChip<8>),
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
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();

        let range_checker = builder.system_base().range_checker_chip.clone();
        let pointer_max_bits = builder.system_config().memory_config.pointer_max_bits;

        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let base_alu_chip = Rv32BaseAluChip::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                    bitwise_lu_chip.bus(),
                ),
                BaseAluCoreAir::new(bitwise_lu_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
            ),
            Rv32BaseAluStep::new(
                Rv32BaseAluAdapterStep::new(bitwise_lu_chip.clone()),
                bitwise_lu_chip.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            base_alu_chip,
            BaseAluOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let lt_chip = Rv32LessThanChip::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                    bitwise_lu_chip.bus(),
                ),
                LessThanCoreAir::new(bitwise_lu_chip.bus(), LessThanOpcode::CLASS_OFFSET),
            ),
            LessThanStep::new(
                Rv32BaseAluAdapterStep::new(bitwise_lu_chip.clone()),
                bitwise_lu_chip.clone(),
                LessThanOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(lt_chip, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_chip = Rv32ShiftChip::new(
            VmAirWrapper::new(
                Rv32BaseAluAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                    bitwise_lu_chip.bus(),
                ),
                ShiftCoreAir::new(
                    bitwise_lu_chip.bus(),
                    range_checker.bus(),
                    ShiftOpcode::CLASS_OFFSET,
                ),
            ),
            ShiftStep::new(
                Rv32BaseAluAdapterStep::new(bitwise_lu_chip.clone()),
                bitwise_lu_chip.clone(),
                range_checker.clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(shift_chip, ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let load_store_chip = Rv32LoadStoreChip::new(
            VmAirWrapper::new(
                Rv32LoadStoreAdapterAir::new(
                    memory_bridge,
                    ExecutionBridge::new(execution_bus, program_bus),
                    range_checker.bus(),
                    pointer_max_bits,
                ),
                LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
            ),
            LoadStoreStep::new(
                Rv32LoadStoreAdapterStep::new(pointer_max_bits),
                range_checker.clone(),
                Rv32LoadStoreOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            load_store_chip,
            Rv32LoadStoreOpcode::iter()
                .take(Rv32LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_chip = Rv32LoadSignExtendChip::new(
            VmAirWrapper::new(
                Rv32LoadStoreAdapterAir::new(
                    memory_bridge,
                    ExecutionBridge::new(execution_bus, program_bus),
                    range_checker.bus(),
                    pointer_max_bits,
                ),
                LoadSignExtendCoreAir::new(range_checker.bus()),
            ),
            LoadSignExtendStep::new(
                Rv32LoadStoreAdapterStep::new(pointer_max_bits),
                range_checker.clone(),
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            load_sign_extend_chip,
            [Rv32LoadStoreOpcode::LOADB, Rv32LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let beq_chip = Rv32BranchEqualChip::new(
            VmAirWrapper::new(
                Rv32BranchAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
            ),
            BranchEqualStep::new(
                Rv32BranchAdapterStep::new(),
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            beq_chip,
            BranchEqualOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let blt_chip = Rv32BranchLessThanChip::<F>::new(
            VmAirWrapper::new(
                Rv32BranchAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                BranchLessThanCoreAir::new(
                    bitwise_lu_chip.bus(),
                    BranchLessThanOpcode::CLASS_OFFSET,
                ),
            ),
            BranchLessThanStep::new(
                Rv32BranchAdapterStep::new(),
                bitwise_lu_chip.clone(),
                BranchLessThanOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            blt_chip,
            BranchLessThanOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let jal_lui_chip = Rv32JalLuiChip::<F>::new(
            VmAirWrapper::new(
                Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(
                    memory_bridge,
                    ExecutionBridge::new(execution_bus, program_bus),
                )),
                Rv32JalLuiCoreAir::new(bitwise_lu_chip.bus()),
            ),
            Rv32JalLuiStep::new(
                Rv32CondRdWriteAdapterStep::new(Rv32RdWriteAdapterStep::new()),
                bitwise_lu_chip.clone(),
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            jal_lui_chip,
            Rv32JalLuiOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let jalr_chip = Rv32JalrChip::<F>::new(
            VmAirWrapper::new(
                Rv32JalrAdapterAir::new(
                    memory_bridge,
                    ExecutionBridge::new(execution_bus, program_bus),
                ),
                Rv32JalrCoreAir::new(bitwise_lu_chip.bus(), range_checker.bus()),
            ),
            Rv32JalrStep::new(
                Rv32JalrAdapterStep::new(),
                bitwise_lu_chip.clone(),
                range_checker.clone(),
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(jalr_chip, Rv32JalrOpcode::iter().map(|x| x.global_opcode()))?;

        let auipc_chip = Rv32AuipcChip::<F>::new(
            VmAirWrapper::new(
                Rv32RdWriteAdapterAir::new(
                    memory_bridge,
                    ExecutionBridge::new(execution_bus, program_bus),
                ),
                Rv32AuipcCoreAir::new(bitwise_lu_chip.bus()),
            ),
            Rv32AuipcStep::new(Rv32RdWriteAdapterStep::new(), bitwise_lu_chip.clone()),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            auipc_chip,
            Rv32AuipcOpcode::iter().map(|x| x.global_opcode()),
        )?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        builder.add_phantom_sub_executor(
            phantom::Rv32HintInputSubEx,
            PhantomDiscriminant(Rv32Phantom::HintInput as u16),
        )?;
        builder.add_phantom_sub_executor(
            phantom::Rv32HintRandomSubEx::new(),
            PhantomDiscriminant(Rv32Phantom::HintRandom as u16),
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
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();

        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let range_tuple_checker = if let Some(chip) = builder
            .find_chip::<SharedRangeTupleCheckerChip<2>>()
            .into_iter()
            .find(|c| {
                c.bus().sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus().sizes[1] >= self.range_tuple_checker_sizes[1]
            }) {
            chip.clone()
        } else {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(builder.new_bus_idx(), self.range_tuple_checker_sizes);
            let chip = SharedRangeTupleCheckerChip::new(range_tuple_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mul_chip = Rv32MultiplicationChip::<F>::new(
            VmAirWrapper::new(
                Rv32MultAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                // TODO(ayush): bus should return value not reference
                MultiplicationCoreAir::new(*range_tuple_checker.bus(), MulOpcode::CLASS_OFFSET),
            ),
            MultiplicationStep::new(
                Rv32MultAdapterStep::new(),
                range_tuple_checker.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(mul_chip, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_h_chip = Rv32MulHChip::<F>::new(
            VmAirWrapper::new(
                Rv32MultAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                MulHCoreAir::new(bitwise_lu_chip.bus(), *range_tuple_checker.bus()),
            ),
            MulHStep::new(
                Rv32MultAdapterStep::new(),
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(mul_h_chip, MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let div_rem_chip = Rv32DivRemChip::<F>::new(
            VmAirWrapper::new(
                Rv32MultAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                DivRemCoreAir::new(
                    bitwise_lu_chip.bus(),
                    *range_tuple_checker.bus(),
                    DivRemOpcode::CLASS_OFFSET,
                ),
            ),
            DivRemStep::new(
                Rv32MultAdapterStep::new(),
                bitwise_lu_chip.clone(),
                range_tuple_checker.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        inventory.add_executor(
            div_rem_chip,
            DivRemOpcode::iter().map(|x| x.global_opcode()),
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
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = builder.system_port();

        let bitwise_lu_chip = if let Some(&chip) = builder
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .first()
        {
            chip.clone()
        } else {
            let bitwise_lu_bus = BitwiseOperationLookupBus::new(builder.new_bus_idx());
            let chip = SharedBitwiseOperationLookupChip::new(bitwise_lu_bus);
            inventory.add_periphery_chip(chip.clone());
            chip
        };

        let mut hintstore_chip = Rv32HintStoreChip::<F>::new(
            Rv32HintStoreAir::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bitwise_lu_chip.bus(),
                Rv32HintStoreOpcode::CLASS_OFFSET,
                builder.system_config().memory_config.pointer_max_bits,
            ),
            Rv32HintStoreStep::new(
                bitwise_lu_chip,
                builder.system_config().memory_config.pointer_max_bits,
                Rv32HintStoreOpcode::CLASS_OFFSET,
            ),
            MAX_INS_CAPACITY,
            builder.system_base().memory_controller.helper(),
        );
        hintstore_chip.step.set_streams(builder.streams().clone());

        inventory.add_executor(
            hintstore_chip,
            Rv32HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(inventory)
    }
}

/// Phantom sub-executors
mod phantom {
    use eyre::bail;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_instructions::PhantomDiscriminant;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};
    use rand::{rngs::OsRng, Rng};

    use crate::adapters::{memory_read, new_read_rv32_register};

    pub struct Rv32HintInputSubEx;
    pub struct Rv32HintRandomSubEx {
        rng: OsRng,
    }
    impl Rv32HintRandomSubEx {
        pub fn new() -> Self {
            Self { rng: OsRng }
        }
    }
    pub struct Rv32PrintStrSubEx;

    impl<F: Field> PhantomSubExecutor<F> for Rv32HintInputSubEx {
        fn phantom_execute(
            &mut self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32HintRandomSubEx {
        fn phantom_execute(
            &mut self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let len = new_read_rv32_register(memory, 1, a) as usize;
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                std::iter::repeat_with(|| F::from_canonical_u8(self.rng.gen::<u8>())).take(len * 4),
            );
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32PrintStrSubEx {
        fn phantom_execute(
            &mut self,
            memory: &GuestMemory,
            _: &mut Streams<F>,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let rd = new_read_rv32_register(memory, 1, a);
            let rs1 = new_read_rv32_register(memory, 1, b);
            let bytes = (0..rs1)
                .map(|i| memory_read::<1>(memory, 2, rd + i)[0])
                .collect::<Vec<u8>>();
            let peeked_str = String::from_utf8(bytes)?;
            print!("{peeked_str}");
            Ok(())
        }
    }
}
