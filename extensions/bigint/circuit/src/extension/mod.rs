use std::sync::Arc;

use derive_more::derive::From;
use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::{LogNativeAssemblerRegistry, VmRvrLogNativeExtension};
use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, MatrixRecordArena,
        RowMajorMatrixArena, VmBuilder, VmChipComplex, VmCircuitExtension, VmExecutionExtension,
        VmField, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        SharedRangeTupleCheckerChip,
    },
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_adapters::{
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterExecutor, Rv64VecHeapAdapterFiller,
    Rv64VecHeapBranchU16AdapterAir, Rv64VecHeapBranchU16AdapterExecutor,
    Rv64VecHeapBranchU16AdapterFiller, Rv64VecHeapU16AdapterAir, Rv64VecHeapU16AdapterExecutor,
    Rv64VecHeapU16AdapterFiller,
};
use openvm_riscv_circuit::Rv64ImCpuProverExt;
use openvm_riscv_transpiler::{BaseAluOpcode, ShiftOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, StarkProtocolConfig, Val};
#[cfg(feature = "rvr")]
use rvr_openvm_ext_bigint::Int256Extension;
#[cfg(feature = "rvr")]
use rvr_openvm_lift::{RvrExtensionCtx, RvrExtensions, VmRvrExtension};
use serde::{Deserialize, Serialize};

use crate::{
    AluAdapterAir, AluAdapterExecutor, AluU16AdapterAir, AluU16AdapterExecutor, BranchAdapterAir,
    BranchAdapterExecutor, *,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::*;
        pub use self::cuda::{
            Int256GpuProverExt as Int256ProverExt,
            Int256Rv64GpuBuilder as Int256Rv64Builder,
        };
    } else {
        pub use self::{
            Int256CpuProverExt as Int256ProverExt,
            Int256Rv64CpuBuilder as Int256Rv64Builder,
        };
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Int256 {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Int256 {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 32 * (1 << 8)]
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for Int256 {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, ctx: Option<&RvrExtensionCtx>) {
        let ext = Int256Extension::new(ctx).expect("failed to construct rvr Int256Extension");
        extensions.register_lifter(ext);
    }
}

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Int256Executor {
    AddSub256(Rv64AddSub256Executor),
    BitwiseLogic256(Rv64BitwiseLogic256Executor),
    LessThan256(Rv64LessThan256Executor),
    BranchEqual256(Rv64BranchEqual256Executor),
    BranchLessThan256(Rv64BranchLessThan256Executor),
    Multiplication256(Rv64Multiplication256Executor),
    ShiftLogical256(Rv64ShiftLogical256Executor),
    ShiftRightArithmetic256(Rv64ShiftRightArithmetic256Executor),
}

impl VmExecutionExtension for Int256 {
    type Executor = Int256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Int256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());

        let add_sub = Rv64AddSub256Executor::new(
            AluU16AdapterExecutor::new(Rv64VecHeapU16AdapterExecutor::new(byte_ptr_max_bits)),
            Rv64BaseAlu256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            add_sub,
            [BaseAluOpcode::ADD, BaseAluOpcode::SUB]
                .map(|op| Rv64BaseAlu256Opcode(op).global_opcode()),
        )?;

        let bitwise = Rv64BitwiseLogic256Executor::new(
            AluAdapterExecutor::new(Rv64VecHeapAdapterExecutor::new(byte_ptr_max_bits)),
            Rv64BaseAlu256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            bitwise,
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND]
                .map(|op| Rv64BaseAlu256Opcode(op).global_opcode()),
        )?;

        let lt = Rv64LessThan256Executor::new(
            AluU16AdapterExecutor::new(Rv64VecHeapU16AdapterExecutor::new(byte_ptr_max_bits)),
            Rv64LessThan256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(lt, Rv64LessThan256Opcode::iter().map(|x| x.global_opcode()))?;

        // Note: `iter()` registers all branch opcode variants, but only BEQ256 is currently
        // generated by the transpiler. The guest uses SLTU + standard 32-bit branches for ordering
        // comparisons. Chips BNE256, BLT256, BLTU256, BGE256, BGEU256 are unused.
        let beq = Rv64BranchEqual256Executor::new(
            BranchAdapterExecutor::new(Rv64VecHeapBranchU16AdapterExecutor::new(byte_ptr_max_bits)),
            Rv64BranchEqual256Opcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        inventory.add_executor(
            beq,
            Rv64BranchEqual256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let blt = Rv64BranchLessThan256Executor::new(
            BranchAdapterExecutor::new(Rv64VecHeapBranchU16AdapterExecutor::new(byte_ptr_max_bits)),
            Rv64BranchLessThan256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            blt,
            Rv64BranchLessThan256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let mult = Rv64Multiplication256Executor::new(
            AluAdapterExecutor::new(Rv64VecHeapAdapterExecutor::new(byte_ptr_max_bits)),
            Rv64Mul256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(mult, Rv64Mul256Opcode::iter().map(|x| x.global_opcode()))?;

        let shift_logical = Rv64ShiftLogical256Executor::new(
            AluU16AdapterExecutor::new(Rv64VecHeapU16AdapterExecutor::new(byte_ptr_max_bits)),
            Rv64Shift256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_logical,
            [
                Rv64Shift256Opcode(ShiftOpcode::SLL),
                Rv64Shift256Opcode(ShiftOpcode::SRL),
            ]
            .map(|x| x.global_opcode()),
        )?;

        let shift_right_arithmetic = Rv64ShiftRightArithmetic256Executor::new(
            AluU16AdapterExecutor::new(Rv64VecHeapU16AdapterExecutor::new(byte_ptr_max_bits)),
            Rv64Shift256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_right_arithmetic,
            [Rv64Shift256Opcode(ShiftOpcode::SRA)].map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Int256 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());

        let bitwise_lu = {
            // A trick to get around Rust's borrow rules
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };

        let range_tuple_checker = {
            let existing_air = inventory.find_air::<RangeTupleCheckerAir<2>>().find(|c| {
                c.bus.sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus.sizes[1] >= self.range_tuple_checker_sizes[1]
            });
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = RangeTupleCheckerBus::new(
                    inventory.new_bus_idx(),
                    self.range_tuple_checker_sizes,
                );
                let air = RangeTupleCheckerAir { bus };
                inventory.add_air(air);
                air.bus
            }
        };

        let add_sub = Rv64AddSub256Air::new(
            AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            AddSubCoreAir::new(range_checker, Rv64BaseAlu256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(add_sub);

        let bitwise = Rv64BitwiseLogic256Air::new(
            AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            BitwiseLogicCoreAir::new(bitwise_lu, Rv64BaseAlu256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(bitwise);

        let lt = Rv64LessThan256Air::new(
            AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            LessThanCoreAir::new(range_checker, Rv64LessThan256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(lt);

        let beq = Rv64BranchEqual256Air::new(
            BranchAdapterAir::new(Rv64VecHeapBranchU16AdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            BranchEqualCoreAir::new(Rv64BranchEqual256Opcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(beq);

        let blt = Rv64BranchLessThan256Air::new(
            BranchAdapterAir::new(Rv64VecHeapBranchU16AdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            BranchLessThanCoreAir::new(range_checker, Rv64BranchLessThan256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(blt);

        let mult = Rv64Multiplication256Air::new(
            AluAdapterAir::new(Rv64VecHeapAdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            MultiplicationCoreAir::new(
                range_tuple_checker,
                bitwise_lu,
                Rv64Mul256Opcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(mult);

        let shift_logical = Rv64ShiftLogical256Air::new(
            AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            ShiftLogicalCoreAir::new(range_checker, Rv64Shift256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_logical);

        let shift_right_arithmetic = Rv64ShiftRightArithmetic256Air::new(
            AluU16AdapterAir::new(Rv64VecHeapU16AdapterAir::new(
                exec_bridge,
                memory_bridge,
                range_checker,
                byte_ptr_max_bits,
            )),
            ShiftRightArithmeticCoreAir::new(range_checker, Rv64Shift256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_right_arithmetic);

        Ok(())
    }
}

pub struct Int256CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, E, RA> VmProverExtension<E, RA, Int256> for Int256CpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &Int256,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let byte_ptr_max_bits =
            to_byte_ptr_bits(inventory.airs().config().memory_config.pointer_max_bits);

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<SharedRangeTupleCheckerChip<2>>()
                .find(|c| {
                    c.bus().sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.bus().sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &RangeTupleCheckerAir<2> = inventory.next_air()?;
                let chip = SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv64AddSub256Air>()?;
        let add_sub = Rv64AddSub256Chip::new(
            AddSubFiller::new(
                Rv64VecHeapU16AdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(add_sub);

        inventory.next_air::<Rv64BitwiseLogic256Air>()?;
        let bitwise = Rv64BitwiseLogic256Chip::new(
            BitwiseLogicFiller::new(
                Rv64VecHeapAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(bitwise);

        inventory.next_air::<Rv64LessThan256Air>()?;
        let lt = Rv64LessThan256Chip::new(
            LessThanFiller::new(
                Rv64VecHeapU16AdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64BranchEqual256Air>()?;
        let beq = Rv64BranchEqual256Chip::new(
            BranchEqualFiller::new(
                Rv64VecHeapBranchU16AdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64BranchEqual256Opcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThan256Air>()?;
        let blt = Rv64BranchLessThan256Chip::new(
            BranchLessThanFiller::new(
                Rv64VecHeapBranchU16AdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                range_checker.clone(),
                Rv64BranchLessThan256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64Multiplication256Air>()?;
        let mult = Rv64Multiplication256Chip::new(
            MultiplicationFiller::new(
                Rv64VecHeapAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                range_tuple_checker.clone(),
                bitwise_lu.clone(),
                Rv64Mul256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64ShiftLogical256Air>()?;
        let shift_logical = Rv64ShiftLogical256Chip::new(
            ShiftLogicalFiller::new(
                Rv64VecHeapU16AdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_logical);

        inventory.next_air::<Rv64ShiftRightArithmetic256Air>()?;
        let shift_right_arithmetic = Rv64ShiftRightArithmetic256Chip::new(
            ShiftRightArithmeticFiller::new(
                Rv64VecHeapU16AdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_right_arithmetic);
        Ok(())
    }
}

#[derive(Clone)]
pub struct Int256Rv64CpuBuilder;

impl<SC, E> VmBuilder<E> for Int256Rv64CpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Int256Rv64Config;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Int256Rv64Config,
        circuit: AirInventory<E::SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.rv64i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.rv64m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Int256CpuProverExt,
            &config.bigint,
            inventory,
        )?;
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<E::SC>, Self::RecordArena>
    where
        Val<E::SC>: PrimeField32,
    {
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }
}
