use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena,
        VmCircuitExtension, VmExecutionExtension, VmField, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::AnyEnum;
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
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, LessThanImmOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, MulWOpcode, Rv64AuipcOpcode, Rv64HintStoreOpcode, Rv64JalLuiOpcode,
    Rv64JalrOpcode, Rv64LoadStoreOpcode, Rv64Phantom, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode,
    ShiftWOpcode,
};
#[cfg(feature = "rvr")]
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val};
#[cfg(feature = "rvr")]
use rvr_openvm_ext_riscv::{
    Rv64IExtension, Rv64IRuntimeHooks, Rv64IoExtension, Rv64IoRuntimeHooks,
};
#[cfg(feature = "rvr")]
use rvr_openvm_lift::{RvrExtensionCtx, RvrExtensions, VmRvrExtension};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::{
            Rv64ImGpuProverExt as Rv64ImGpuProverExt,
        };
    } else {
        pub use self::{
            Rv64ImCpuProverExt as Rv64ImProverExt,
        };
    }
}

// ============ Extension Struct Definitions ============

/// RISC-V 64-bit Base (RV64I) Extension
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64I;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64Io;

/// RISC-V 64-bit Multiplication Extension (RV64M) Extension
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv64M {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Rv64M {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [
        // range for a single limb
        1 << RV64_BYTE_BITS,
        // carry bound across a column of an N-limb × N-limb multiplication
        2 * RV64_REGISTER_NUM_LIMBS as u32 * (1 << RV64_BYTE_BITS),
    ]
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for Rv64I {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, _ctx: Option<&RvrExtensionCtx>) {
        extensions.register_lifter(Rv64IExtension::new());
        extensions.register_runtime_hook(Rv64IRuntimeHooks);
    }
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for Rv64Io {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, ctx: Option<&RvrExtensionCtx>) {
        extensions.register_lifter(
            Rv64IoExtension::new(ctx).expect("Rv64IoExtension chip resolution failed"),
        );
        extensions.register_runtime_hook(Rv64IoRuntimeHooks);
    }
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for Rv64M {}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 64-bit Base (RV64I) Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Rv64IExecutor {
    AddSub(Rv64AddSubExecutor),
    AddI(Rv64AddIExecutor),
    BitwiseLogic(Rv64BitwiseLogicExecutor),
    BitwiseLogicImm(Rv64BitwiseLogicImmExecutor),
    LessThanImm(Rv64LessThanImmExecutor),
    ShiftLogicalImm(Rv64ShiftLogicalImmExecutor),
    ShiftRightArithmeticImm(Rv64ShiftRightArithmeticImmExecutor),
    AddSubW(Rv64AddSubWExecutor),
    AddIW(Rv64AddIWExecutor),
    LessThan(Rv64LessThanExecutor),
    ShiftLogical(Rv64ShiftLogicalExecutor),
    ShiftRightArithmetic(Rv64ShiftRightArithmeticExecutor),
    ShiftWLogical(Rv64ShiftWLogicalExecutor),
    ShiftWRightArithmetic(Rv64ShiftWRightArithmeticExecutor),
    ShiftWLogicalImm(Rv64ShiftWLogicalImmExecutor),
    ShiftWRightArithmeticImm(Rv64ShiftWRightArithmeticImmExecutor),
    BranchEqual(Rv64BranchEqualExecutor),
    BranchLessThan(Rv64BranchLessThanExecutor),
    JalLui(Rv64JalLuiExecutor),
    Jalr(Rv64JalrExecutor),
    Auipc(Rv64AuipcExecutor),
    LoadSignExtendByte(Rv64LoadSignExtendByteExecutor),
    LoadByte(Rv64LoadByteExecutor),
    StoreByte(Rv64StoreByteExecutor),
    LoadSignExtendHalfword(Rv64LoadSignExtendHalfwordExecutor),
    LoadHalfword(Rv64LoadHalfwordExecutor),
    StoreHalfword(Rv64StoreHalfwordExecutor),
    LoadSignExtendWord(Rv64LoadSignExtendWordExecutor),
    LoadWord(Rv64LoadWordExecutor),
    StoreWord(Rv64StoreWordExecutor),
    LoadDoubleword(Rv64LoadDoublewordExecutor),
    StoreDoubleword(Rv64StoreDoublewordExecutor),
}

/// RISC-V 64-bit Multiplication Extension (RV64M) Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Rv64MExecutor {
    Multiplication(Rv64MultiplicationExecutor),
    MulW(Rv64MulWExecutor),
    MultiplicationHigh(Rv64MulHExecutor),
    DivRem(Rv64DivRemExecutor),
    DivRemW(Rv64DivRemWExecutor),
}

/// RISC-V 64-bit Io Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Rv64IoExecutor {
    HintStore(Rv64HintStoreExecutor),
}

// ============ VmExtension Implementations ============

impl VmExecutionExtension for Rv64I {
    type Executor = Rv64IExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Rv64IExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());

        let add_sub = Rv64AddSubExecutor::new(
            Rv64BaseAluRegU16AdapterExecutor,
            BaseAluOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            add_sub,
            [BaseAluOpcode::ADD, BaseAluOpcode::SUB].map(|x| x.global_opcode()),
        )?;

        let bitwise_logic = Rv64BitwiseLogicExecutor::new(
            Rv64BaseAluRegAdapterExecutor,
            BaseAluOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            bitwise_logic,
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND].map(|x| x.global_opcode()),
        )?;

        let add_sub_w = Rv64AddSubWExecutor::new(
            Rv64BaseAluWRegU16AdapterExecutor,
            BaseAluWOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(add_sub_w, BaseAluWOpcode::iter().map(|x| x.global_opcode()))?;

        let lt = Rv64LessThanExecutor::new(
            Rv64BaseAluRegU16AdapterExecutor,
            LessThanOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(lt, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_logical = Rv64ShiftLogicalExecutor::new(
            Rv64BaseAluRegU16AdapterExecutor,
            ShiftOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_logical,
            [ShiftOpcode::SLL, ShiftOpcode::SRL].map(|x| x.global_opcode()),
        )?;

        let shift_right_arithmetic = Rv64ShiftRightArithmeticExecutor::new(
            Rv64BaseAluRegU16AdapterExecutor,
            ShiftOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_right_arithmetic,
            [ShiftOpcode::SRA].map(|x| x.global_opcode()),
        )?;

        let shift_w_logical = Rv64ShiftWLogicalExecutor::new(
            Rv64BaseAluWRegU16AdapterExecutor::new(),
            ShiftWOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_w_logical,
            [ShiftWOpcode::SLLW, ShiftWOpcode::SRLW].map(|x| x.global_opcode()),
        )?;

        let shift_w_right_arithmetic = Rv64ShiftWRightArithmeticExecutor::new(
            Rv64BaseAluWRegU16AdapterExecutor::new(),
            ShiftWOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_w_right_arithmetic,
            [ShiftWOpcode::SRAW].map(|x| x.global_opcode()),
        )?;

        let addi_w = Rv64AddIWExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            BaseAluWImmOpcode::CLASS_OFFSET,
            BaseAluWImmOpcode::ADDIW as usize,
        );
        inventory.add_executor(
            addi_w,
            [BaseAluWImmOpcode::ADDIW].map(|x| x.global_opcode()),
        )?;

        let shift_w_logical_imm = Rv64ShiftWLogicalImmExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            ShiftWImmOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_w_logical_imm,
            [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW].map(|x| x.global_opcode()),
        )?;

        let shift_w_right_arithmetic_imm = Rv64ShiftWRightArithmeticImmExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            ShiftWImmOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_w_right_arithmetic_imm,
            [ShiftWImmOpcode::SRAIW].map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_byte = Rv64LoadSignExtendByteExecutor::new(
            Rv64LoadByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_sign_extend_byte,
            [Rv64LoadStoreOpcode::LOADB].map(|x| x.global_opcode()),
        )?;

        let load_byte = Rv64LoadByteExecutor::new(
            Rv64LoadByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_byte,
            [Rv64LoadStoreOpcode::LOADBU].map(|x| x.global_opcode()),
        )?;

        let store_byte = Rv64StoreByteExecutor::new(
            Rv64StoreByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            store_byte,
            [Rv64LoadStoreOpcode::STOREB].map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_halfword = Rv64LoadSignExtendHalfwordExecutor::new(
            Rv64LoadMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_sign_extend_halfword,
            [Rv64LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let load_halfword = Rv64LoadHalfwordExecutor::new(
            Rv64LoadMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_halfword,
            [Rv64LoadStoreOpcode::LOADHU].map(|x| x.global_opcode()),
        )?;

        let store_halfword = Rv64StoreHalfwordExecutor::new(
            Rv64StoreMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            store_halfword,
            [Rv64LoadStoreOpcode::STOREH].map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_word = Rv64LoadSignExtendWordExecutor::new(
            Rv64LoadMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_sign_extend_word,
            [Rv64LoadStoreOpcode::LOADW].map(|x| x.global_opcode()),
        )?;

        let load_word = Rv64LoadWordExecutor::new(
            Rv64LoadMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_word,
            [Rv64LoadStoreOpcode::LOADWU].map(|x| x.global_opcode()),
        )?;

        let store_word = Rv64StoreWordExecutor::new(
            Rv64StoreMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            store_word,
            [Rv64LoadStoreOpcode::STOREW].map(|x| x.global_opcode()),
        )?;

        let load_doubleword = Rv64LoadDoublewordExecutor::new(
            Rv64LoadMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_doubleword,
            [Rv64LoadStoreOpcode::LOADD].map(|x| x.global_opcode()),
        )?;

        let store_doubleword = Rv64StoreDoublewordExecutor::new(
            Rv64StoreMultiByteAdapterExecutor::new(byte_ptr_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            store_doubleword,
            [Rv64LoadStoreOpcode::STORED].map(|x| x.global_opcode()),
        )?;

        let beq = BranchEqualExecutor::new(
            Rv64BranchAdapterExecutor,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        inventory.add_executor(beq, BranchEqualOpcode::iter().map(|x| x.global_opcode()))?;

        let blt = BranchLessThanExecutor::new(
            Rv64BranchAdapterExecutor,
            BranchLessThanOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(blt, BranchLessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let jal_lui = Rv64JalLuiExecutor::new(Rv64CondRdWriteAdapterExecutor::new(
            Rv64RdWriteAdapterExecutor,
        ));
        inventory.add_executor(jal_lui, Rv64JalLuiOpcode::iter().map(|x| x.global_opcode()))?;

        let jalr = Rv64JalrExecutor::new(Rv64JalrAdapterExecutor);
        inventory.add_executor(jalr, Rv64JalrOpcode::iter().map(|x| x.global_opcode()))?;

        let auipc = Rv64AuipcExecutor::new(Rv64RdWriteAdapterExecutor);
        inventory.add_executor(auipc, Rv64AuipcOpcode::iter().map(|x| x.global_opcode()))?;

        let addi = Rv64AddIExecutor::new(
            Rv64BaseAluImmU16AdapterExecutor,
            BaseAluImmOpcode::CLASS_OFFSET,
            BaseAluImmOpcode::ADDI as usize,
        );
        inventory.add_executor(addi, [BaseAluImmOpcode::ADDI].map(|x| x.global_opcode()))?;

        let shift_logical_imm = Rv64ShiftLogicalImmExecutor::new(
            Rv64BaseAluImmU16AdapterExecutor,
            ShiftImmOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_logical_imm,
            [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI].map(|x| x.global_opcode()),
        )?;

        let shift_right_arithmetic_imm = Rv64ShiftRightArithmeticImmExecutor::new(
            Rv64BaseAluImmU16AdapterExecutor,
            ShiftImmOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            shift_right_arithmetic_imm,
            [ShiftImmOpcode::SRAI].map(|x| x.global_opcode()),
        )?;

        let less_than_imm = Rv64LessThanImmExecutor::new(
            Rv64BaseAluImmU16AdapterExecutor,
            LessThanImmOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            less_than_imm,
            LessThanImmOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let bitwise_logic_imm = Rv64BitwiseLogicImmExecutor::new(
            Rv64BaseAluImmAdapterExecutor,
            BaseAluImmOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            bitwise_logic_imm,
            [
                BaseAluImmOpcode::XORI,
                BaseAluImmOpcode::ORI,
                BaseAluImmOpcode::ANDI,
            ]
            .map(|x| x.global_opcode()),
        )?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        inventory.add_phantom_sub_executor(
            phantom::Rv64HintInputSubEx,
            PhantomDiscriminant(Rv64Phantom::HintInput as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::Rv64HintRandomSubEx,
            PhantomDiscriminant(Rv64Phantom::HintRandom as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::Rv64PrintStrSubEx,
            PhantomDiscriminant(Rv64Phantom::PrintStr as u16),
        )?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Rv64I {
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

        let add_sub = Rv64AddSubAir::new(
            Rv64BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            AddSubCoreAir::new(range_checker, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(add_sub);

        let bitwise_logic = Rv64BitwiseLogicAir::new(
            Rv64BaseAluRegAdapterAir::new(exec_bridge, memory_bridge),
            BitwiseLogicCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(bitwise_logic);

        let add_sub_w = Rv64AddSubWAir::new(
            Rv64BaseAluWRegU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            crate::add_sub_w::AddSubWCoreAir::new(range_checker, BaseAluWOpcode::CLASS_OFFSET),
        );
        inventory.add_air(add_sub_w);

        let lt = Rv64LessThanAir::new(
            Rv64BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            LessThanCoreAir::new(range_checker, LessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(lt);

        let shift_logical = Rv64ShiftLogicalAir::new(
            Rv64BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftLogicalCoreAir::new(range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_logical);

        let shift_right_arithmetic = Rv64ShiftRightArithmeticAir::new(
            Rv64BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftRightArithmeticCoreAir::new(range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_right_arithmetic);

        let shift_w_logical = Rv64ShiftWLogicalAir::new(
            Rv64BaseAluWRegU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            crate::shift_w::ShiftWLogicalCoreAir::new(range_checker, ShiftWOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_w_logical);

        let shift_w_right_arithmetic = Rv64ShiftWRightArithmeticAir::new(
            Rv64BaseAluWRegU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            crate::shift_w::ShiftWRightArithmeticCoreAir::new(
                range_checker,
                ShiftWOpcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(shift_w_right_arithmetic);

        let addi_w = Rv64AddIWAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            AddICoreAir::new(
                range_checker,
                BaseAluWImmOpcode::CLASS_OFFSET,
                BaseAluWImmOpcode::ADDIW as usize,
            ),
        );
        inventory.add_air(addi_w);

        let shift_w_logical_imm = Rv64ShiftWLogicalImmAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            ShiftLogicalImmCoreAir::new(range_checker, ShiftWImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_w_logical_imm);

        let shift_w_right_arithmetic_imm = Rv64ShiftWRightArithmeticImmAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            ShiftRightArithmeticImmCoreAir::new(range_checker, ShiftWImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_w_right_arithmetic_imm);

        let load_sign_extend_byte = Rv64LoadSignExtendByteAir::new(
            Rv64LoadByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadSignExtendByteCoreAir::new(
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu,
                range_checker,
            ),
        );
        inventory.add_air(load_sign_extend_byte);

        let load_byte = Rv64LoadByteAir::new(
            Rv64LoadByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_byte);

        let store_byte = Rv64StoreByteAir::new(
            Rv64StoreByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_byte);

        let load_sign_extend_halfword = Rv64LoadSignExtendHalfwordAir::new(
            Rv64LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadSignExtendHalfwordCoreAir::new(
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu,
                range_checker,
            ),
        );
        inventory.add_air(load_sign_extend_halfword);

        let load_halfword = Rv64LoadHalfwordAir::new(
            Rv64LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_halfword);

        let store_halfword = Rv64StoreHalfwordAir::new(
            Rv64StoreMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_halfword);

        let load_sign_extend_word = Rv64LoadSignExtendWordAir::new(
            Rv64LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadSignExtendWordCoreAir::new(
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu,
                range_checker,
            ),
        );
        inventory.add_air(load_sign_extend_word);

        let load_word = Rv64LoadWordAir::new(
            Rv64LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_word);

        let store_word = Rv64StoreWordAir::new(
            Rv64StoreMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_word);

        let load_doubleword = Rv64LoadDoublewordAir::new(
            Rv64LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_doubleword);

        let store_doubleword = Rv64StoreDoublewordAir::new(
            Rv64StoreMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_doubleword);

        let beq = Rv64BranchEqualAir::new(
            Rv64BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(beq);

        let blt = Rv64BranchLessThanAir::new(
            Rv64BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchLessThanCoreAir::new(range_checker, BranchLessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(blt);

        let jal_lui = Rv64JalLuiAir::new(
            Rv64CondRdWriteAdapterAir::new(Rv64RdWriteAdapterAir::new(memory_bridge, exec_bridge)),
            Rv64JalLuiCoreAir::new(range_checker),
        );
        inventory.add_air(jal_lui);

        let jalr = Rv64JalrAir::new(
            Rv64JalrAdapterAir::new(memory_bridge, exec_bridge),
            Rv64JalrCoreAir::new(range_checker),
        );
        inventory.add_air(jalr);

        let auipc = Rv64AuipcAir::new(
            Rv64RdWriteAdapterAir::new(memory_bridge, exec_bridge),
            Rv64AuipcCoreAir::new(range_checker),
        );
        inventory.add_air(auipc);

        let addi = Rv64AddIAir::new(
            Rv64BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            AddICoreAir::new(
                range_checker,
                BaseAluImmOpcode::CLASS_OFFSET,
                BaseAluImmOpcode::ADDI as usize,
            ),
        );
        inventory.add_air(addi);

        let shift_logical_imm = Rv64ShiftLogicalImmAir::new(
            Rv64BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftLogicalImmCoreAir::new(range_checker, ShiftImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_logical_imm);

        let shift_right_arithmetic_imm = Rv64ShiftRightArithmeticImmAir::new(
            Rv64BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftRightArithmeticImmCoreAir::new(range_checker, ShiftImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_right_arithmetic_imm);

        let less_than_imm = Rv64LessThanImmAir::new(
            Rv64BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            LessThanImmCoreAir::new(range_checker, LessThanImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(less_than_imm);

        let bitwise_logic_imm = Rv64BitwiseLogicImmAir::new(
            Rv64BaseAluImmAdapterAir::new(exec_bridge, memory_bridge),
            BitwiseLogicImmCoreAir::new(bitwise_lu, BaseAluImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(bitwise_logic_imm);

        Ok(())
    }
}

pub struct Rv64ImCpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Rv64I> for Rv64ImCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64AddSubAir>()?;
        let add_sub = Rv64AddSubChip::new(
            AddSubFiller::new(Rv64BaseAluRegU16AdapterFiller, range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(add_sub);

        inventory.next_air::<Rv64BitwiseLogicAir>()?;
        let bitwise_logic = Rv64BitwiseLogicChip::new(
            BitwiseLogicFiller::new(Rv64BaseAluRegAdapterFiller, bitwise_lu.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(bitwise_logic);

        inventory.next_air::<Rv64AddSubWAir>()?;
        let add_sub_w = Rv64AddSubWChip::new(
            crate::add_sub_w::AddSubWFiller::new(
                Rv64BaseAluWRegU16AdapterFiller::new(range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(add_sub_w);

        inventory.next_air::<Rv64LessThanAir>()?;
        let lt = Rv64LessThanChip::new(
            LessThanFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64ShiftLogicalAir>()?;
        let shift_logical = Rv64ShiftLogicalChip::new(
            ShiftLogicalFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_logical);

        inventory.next_air::<Rv64ShiftRightArithmeticAir>()?;
        let shift_right_arithmetic = Rv64ShiftRightArithmeticChip::new(
            ShiftRightArithmeticFiller::new(
                Rv64BaseAluRegU16AdapterFiller::new(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_right_arithmetic);

        inventory.next_air::<Rv64ShiftWLogicalAir>()?;
        let shift_w_logical = Rv64ShiftWLogicalChip::new(
            crate::shift_w::ShiftWLogicalFiller::new(
                Rv64BaseAluWRegU16AdapterFiller::new(range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_w_logical);

        inventory.next_air::<Rv64ShiftWRightArithmeticAir>()?;
        let shift_w_right_arithmetic = Rv64ShiftWRightArithmeticChip::new(
            crate::shift_w::ShiftWRightArithmeticFiller::new(
                Rv64BaseAluWRegU16AdapterFiller::new(range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_w_right_arithmetic);

        inventory.next_air::<Rv64AddIWAir>()?;
        let addi_w = Rv64AddIWChip::new(
            AddIFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(addi_w);

        inventory.next_air::<Rv64ShiftWLogicalImmAir>()?;
        let shift_w_logical_imm = Rv64ShiftWLogicalImmChip::new(
            ShiftLogicalImmFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_w_logical_imm);

        inventory.next_air::<Rv64ShiftWRightArithmeticImmAir>()?;
        let shift_w_right_arithmetic_imm = Rv64ShiftWRightArithmeticImmChip::new(
            ShiftRightArithmeticImmFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_w_right_arithmetic_imm);

        inventory.next_air::<Rv64LoadSignExtendByteAir>()?;
        let load_sign_extend_byte_chip = Rv64LoadSignExtendByteChip::new(
            LoadSignExtendByteFiller::new(
                Rv64LoadByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_sign_extend_byte_chip);

        inventory.next_air::<Rv64LoadByteAir>()?;
        let load_byte_chip = Rv64LoadByteChip::new(
            LoadByteFiller::new(
                Rv64LoadByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_byte_chip);

        inventory.next_air::<Rv64StoreByteAir>()?;
        let store_byte_chip = Rv64StoreByteChip::new(
            StoreByteFiller::new(
                Rv64StoreByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(store_byte_chip);

        inventory.next_air::<Rv64LoadSignExtendHalfwordAir>()?;
        let load_sign_extend_halfword_chip = Rv64LoadSignExtendHalfwordChip::new(
            LoadSignExtendHalfwordFiller::new(
                Rv64LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_sign_extend_halfword_chip);

        inventory.next_air::<Rv64LoadHalfwordAir>()?;
        let load_halfword_chip = Rv64LoadHalfwordChip::new(
            LoadHalfwordFiller::new(
                Rv64LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_halfword_chip);

        inventory.next_air::<Rv64StoreHalfwordAir>()?;
        let store_halfword_chip = Rv64StoreHalfwordChip::new(
            StoreHalfwordFiller::new(
                Rv64StoreMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(store_halfword_chip);

        inventory.next_air::<Rv64LoadSignExtendWordAir>()?;
        let load_sign_extend_word_chip = Rv64LoadSignExtendWordChip::new(
            LoadSignExtendWordFiller::new(
                Rv64LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_sign_extend_word_chip);

        inventory.next_air::<Rv64LoadWordAir>()?;
        let load_word_chip = Rv64LoadWordChip::new(
            LoadWordFiller::new(
                Rv64LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_word_chip);

        inventory.next_air::<Rv64StoreWordAir>()?;
        let store_word_chip = Rv64StoreWordChip::new(
            StoreWordFiller::new(
                Rv64StoreMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(store_word_chip);

        inventory.next_air::<Rv64LoadDoublewordAir>()?;
        let load_doubleword_chip = Rv64LoadDoublewordChip::new(
            LoadDoublewordFiller::new(
                Rv64LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_doubleword_chip);

        inventory.next_air::<Rv64StoreDoublewordAir>()?;
        let store_doubleword_chip = Rv64StoreDoublewordChip::new(
            StoreDoublewordFiller::new(
                Rv64StoreMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(store_doubleword_chip);

        inventory.next_air::<Rv64BranchEqualAir>()?;
        let beq = Rv64BranchEqualChip::new(
            BranchEqualFiller::new(
                Rv64BranchAdapterFiller,
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThanAir>()?;
        let blt = Rv64BranchLessThanChip::new(
            BranchLessThanFiller::new(
                Rv64BranchAdapterFiller,
                range_checker.clone(),
                BranchLessThanOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64JalLuiAir>()?;
        let jal_lui = Rv64JalLuiChip::new(
            Rv64JalLuiFiller::new(
                Rv64CondRdWriteAdapterFiller::new(Rv64RdWriteAdapterFiller),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv64JalrAir>()?;
        let jalr = Rv64JalrChip::new(
            Rv64JalrFiller::new(Rv64JalrAdapterFiller, range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv64AuipcAir>()?;
        let auipc = Rv64AuipcChip::new(
            Rv64AuipcFiller::new(Rv64RdWriteAdapterFiller, range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(auipc);

        inventory.next_air::<Rv64AddIAir>()?;
        let addi = Rv64AddIChip::new(
            AddIFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(addi);

        inventory.next_air::<Rv64ShiftLogicalImmAir>()?;
        let shift_logical_imm = Rv64ShiftLogicalImmChip::new(
            ShiftLogicalImmFiller::new(
                Rv64BaseAluImmU16AdapterFiller::new(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_logical_imm);

        inventory.next_air::<Rv64ShiftRightArithmeticImmAir>()?;
        let shift_right_arithmetic_imm = Rv64ShiftRightArithmeticImmChip::new(
            ShiftRightArithmeticImmFiller::new(
                Rv64BaseAluImmU16AdapterFiller::new(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift_right_arithmetic_imm);

        inventory.next_air::<Rv64LessThanImmAir>()?;
        let less_than_imm = Rv64LessThanImmChip::new(
            LessThanImmFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(less_than_imm);

        inventory.next_air::<Rv64BitwiseLogicImmAir>()?;
        let bitwise_logic_imm = Rv64BitwiseLogicImmChip::new(
            BitwiseLogicImmFiller::new(Rv64BaseAluImmAdapterFiller::new(), bitwise_lu.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(bitwise_logic_imm);

        Ok(())
    }
}

impl VmExecutionExtension for Rv64M {
    type Executor = Rv64MExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Rv64MExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let mult =
            Rv64MultiplicationExecutor::new(Rv64MultAdapterExecutor, MulOpcode::CLASS_OFFSET);
        inventory.add_executor(mult, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_w = Rv64MulWExecutor::new(Rv64MultWAdapterExecutor, MulWOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_w, MulWOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_h = Rv64MulHExecutor::new(Rv64MultAdapterExecutor, MulHOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_h, MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let div_rem = Rv64DivRemExecutor::new(Rv64MultAdapterExecutor, DivRemOpcode::CLASS_OFFSET);
        inventory.add_executor(div_rem, DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        let divrem_w =
            Rv64DivRemWExecutor::new(Rv64MultWAdapterExecutor, DivRemWOpcode::CLASS_OFFSET);
        inventory.add_executor(divrem_w, DivRemWOpcode::iter().map(|x| x.global_opcode()))?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Rv64M {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);

        let bitwise_lu = {
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

        let mult = Rv64MultiplicationAir::new(
            Rv64MultAdapterAir::new(exec_bridge, memory_bridge),
            MultiplicationCoreAir::new(range_tuple_checker, bitwise_lu, MulOpcode::CLASS_OFFSET),
        );
        inventory.add_air(mult);

        let mul_w = Rv64MulWAir::new(
            Rv64MultWAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            crate::mul_w::MulWCoreAir::new(
                range_tuple_checker,
                bitwise_lu,
                MulWOpcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(mul_w);

        let mul_h = Rv64MulHAir::new(
            Rv64MultAdapterAir::new(exec_bridge, memory_bridge),
            MulHCoreAir::new(bitwise_lu, range_tuple_checker),
        );
        inventory.add_air(mul_h);

        let div_rem = Rv64DivRemAir::new(
            Rv64MultAdapterAir::new(exec_bridge, memory_bridge),
            DivRemCoreAir::new(bitwise_lu, range_tuple_checker, DivRemOpcode::CLASS_OFFSET),
        );
        inventory.add_air(div_rem);

        let divrem_w = Rv64DivRemWAir::new(
            Rv64MultWAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            crate::divrem_w::DivRemWCoreAir::new(
                bitwise_lu,
                range_tuple_checker,
                DivRemWOpcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(divrem_w);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Rv64M> for Rv64ImCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &Rv64M,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64MultiplicationAir>()?;
        let mult = Rv64MultiplicationChip::new(
            MultiplicationFiller::new(
                Rv64MultAdapterFiller,
                range_tuple_checker.clone(),
                bitwise_lu.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64MulWAir>()?;
        let mul_w = Rv64MulWChip::new(
            crate::mul_w::MulWFiller::new(
                Rv64MultWAdapterFiller::new(bitwise_lu.clone()),
                range_tuple_checker.clone(),
                bitwise_lu.clone(),
                MulWOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mul_w);

        inventory.next_air::<Rv64MulHAir>()?;
        let mul_h = Rv64MulHChip::new(
            MulHFiller::new(
                Rv64MultAdapterFiller,
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv64DivRemAir>()?;
        let div_rem = Rv64DivRemChip::new(
            DivRemFiller::new(
                Rv64MultAdapterFiller,
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(div_rem);

        inventory.next_air::<Rv64DivRemWAir>()?;
        let divrem_w = Rv64DivRemWChip::new(
            crate::divrem_w::DivRemWFiller::new(
                Rv64MultWAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
                DivRemWOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(divrem_w);

        Ok(())
    }
}

impl VmExecutionExtension for Rv64Io {
    type Executor = Rv64IoExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Rv64IoExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());
        let hint_store =
            Rv64HintStoreExecutor::new(byte_ptr_max_bits, Rv64HintStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            hint_store,
            Rv64HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Rv64Io {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.pointer_max_bits());

        let hint_store = Rv64HintStoreAir::new(
            exec_bridge,
            memory_bridge,
            range_checker,
            Rv64HintStoreOpcode::CLASS_OFFSET,
            byte_ptr_max_bits,
        );
        inventory.add_air(hint_store);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Rv64Io> for Rv64ImCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());

        inventory.next_air::<Rv64HintStoreAir>()?;
        let hint_store = Rv64HintStoreChip::new(
            Rv64HintStoreFiller::new(byte_ptr_max_bits, range_checker.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
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
    use rand::{rngs::StdRng, Rng};

    use crate::adapters::{memory_read, read_rv64_register_as_u32, RV64_REGISTER_NUM_LIMBS};

    const HINT_DWORD_BYTES: usize = RV64_REGISTER_NUM_LIMBS;

    pub struct Rv64HintInputSubEx;
    pub struct Rv64HintRandomSubEx;
    pub struct Rv64PrintStrSubEx;

    impl PhantomSubExecutor for Rv64HintInputSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams,
            _: &mut StdRng,
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
            let hint_len = hint.len() as u64;
            streams.hint_stream.extend(hint_len.to_le_bytes());
            // Pad the hint payload to full dwords so RV64 `HINT_BUFFER` reads can consume it.
            let capacity = hint.len().div_ceil(HINT_DWORD_BYTES) * HINT_DWORD_BYTES;
            hint.resize(capacity, 0u8);
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl PhantomSubExecutor for Rv64HintRandomSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams,
            rng: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                eprintln!("WARNING: Using fixed-seed RNG for deterministic randomness. Consider security implications for your use case.");
            });

            let byte_len = read_rv64_register_as_u32(memory, a) as usize * HINT_DWORD_BYTES;
            streams.hint_stream.clear();
            streams
                .hint_stream
                .extend(std::iter::repeat_with(|| rng.random::<u8>()).take(byte_len));
            Ok(())
        }
    }

    impl PhantomSubExecutor for Rv64PrintStrSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            _: &mut Streams,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let rd = read_rv64_register_as_u32(memory, a);
            let rs1 = read_rv64_register_as_u32(memory, b);
            let bytes = (0..rs1)
                .map(|i| memory_read::<1>(memory, 2, rd + i)[0])
                .collect::<Vec<u8>>();
            let peeked_str = String::from_utf8(bytes)?;
            print!("{peeked_str}");
            Ok(())
        }
    }
}
