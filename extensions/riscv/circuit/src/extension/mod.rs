use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension,
        VmExecutionExtension, VmField, VmProverExtension,
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
    Chip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BranchEqualOpcode,
    BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, LessThanImmOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, MulWOpcode, AuipcOpcode, HintStoreOpcode, JalLuiOpcode,
    JalrOpcode, LoadStoreOpcode, Rv64Phantom, RevealOpcode, ShiftImmOpcode,
    ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
};
#[cfg(feature = "rvr")]
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_backend::{prover::AirProvingContext, StarkEngine, StarkProtocolConfig, Val};
#[cfg(feature = "rvr")]
use rvr_openvm_ext_riscv::{
    Rv64IExtension, Rv64IoExtension, Rv64IoRuntimeHooks, Rv64MExtension, Rv64PhantomExtension,
    Rv64PhantomRuntimeHooks,
};
#[cfg(feature = "rvr")]
use rvr_openvm_lift::{RvrExtensionCtx, RvrExtensions, VmRvrExtension};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

macro_rules! add_executor_chip_with_tracegen {
    ($inventory:expr, $chip:expr, $generate:path) => {
        $inventory.add_executor_chip_with_tracegen($chip, |chip, postflight| {
            $generate(chip, postflight).map(AirProvingContext::simple_no_pis)
        });
    };
}

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::{
            Rv64ImGpuProverExt as Rv64ImGpuProverExt,
            Rv64ImPreflightGpuTracegen,
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
        1 << BYTE_BITS,
        // carry bound across a column of an N-limb × N-limb multiplication
        2 * REGISTER_NUM_LIMBS as u32 * (1 << BYTE_BITS),
    ]
}

#[cfg(feature = "rvr")]
impl<F: PrimeField32> VmRvrExtension<F> for Rv64I {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, _ctx: Option<&RvrExtensionCtx>) {
        extensions.register_lifter(Rv64IExtension::new());
        extensions.register_lifter(Rv64PhantomExtension::new());
        extensions.register_runtime_hook(Rv64PhantomRuntimeHooks);
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
impl<F: PrimeField32> VmRvrExtension<F> for Rv64M {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, _ctx: Option<&RvrExtensionCtx>) {
        extensions.register_lifter(Rv64MExtension::new());
    }
}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 64-bit Base (RV64I) Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor)]
pub enum Rv64IExecutor {
    AddSub(AddSubExecutor),
    AddI(AddIExecutor),
    BitwiseLogic(BitwiseLogicExecutor),
    BitwiseLogicImm(BitwiseLogicImmExecutor),
    LessThanImm(LessThanImmExecutor),
    ShiftLogicalImm(ShiftLogicalImmExecutor),
    ShiftRightArithmeticImm(ShiftRightArithmeticImmExecutor),
    AddSubW(AddSubWExecutor),
    AddIW(AddIWExecutor),
    LessThan(LessThanExecutor),
    ShiftLogical(ShiftLogicalExecutor),
    ShiftRightArithmetic(ShiftRightArithmeticExecutor),
    ShiftWLogical(ShiftWLogicalExecutor),
    ShiftWRightArithmetic(ShiftWRightArithmeticExecutor),
    ShiftWLogicalImm(ShiftWLogicalImmExecutor),
    ShiftWRightArithmeticImm(ShiftWRightArithmeticImmExecutor),
    BranchEqual(BranchEqualExecutor),
    BranchLessThan(BranchLessThanExecutor),
    JalLui(JalLuiExecutor),
    Jalr(JalrExecutor),
    Auipc(AuipcExecutor),
    LoadSignExtendByte(LoadSignExtendByteExecutor),
    LoadByte(LoadByteExecutor),
    StoreByte(StoreByteExecutor),
    LoadSignExtendHalfword(LoadSignExtendHalfwordExecutor),
    LoadHalfword(LoadHalfwordExecutor),
    StoreHalfword(StoreHalfwordExecutor),
    LoadSignExtendWord(LoadSignExtendWordExecutor),
    LoadWord(LoadWordExecutor),
    StoreWord(StoreWordExecutor),
    LoadDoubleword(LoadDoublewordExecutor),
    StoreDoubleword(StoreDoublewordExecutor),
}

/// RISC-V 64-bit Multiplication Extension (RV64M) Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor)]
pub enum Rv64MExecutor {
    Multiplication(MultiplicationExecutor),
    MulW(MulWExecutor),
    MultiplicationHigh(MulHExecutor),
    DivRem(DivRemExecutor),
    DivRemW(DivRemWExecutor),
}

/// RISC-V 64-bit Io Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor)]
pub enum Rv64IoExecutor {
    HintStore(HintStoreExecutor),
    Reveal(RevealExecutor),
}

// ============ VmExtension Implementations ============

impl VmExecutionExtension for Rv64I {
    type Executor = Rv64IExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Rv64IExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let add_sub = AddSubExecutor::new(BaseAluOpcode::CLASS_OFFSET);
        inventory.add_executor(
            add_sub,
            [BaseAluOpcode::ADD, BaseAluOpcode::SUB].map(|x| x.global_opcode()),
        )?;

        let bitwise_logic = BitwiseLogicExecutor::new(BaseAluOpcode::CLASS_OFFSET);
        inventory.add_executor(
            bitwise_logic,
            [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND].map(|x| x.global_opcode()),
        )?;

        let add_sub_w = AddSubWExecutor::new(BaseAluWOpcode::CLASS_OFFSET);
        inventory.add_executor(add_sub_w, BaseAluWOpcode::iter().map(|x| x.global_opcode()))?;

        let lt = LessThanExecutor::new(LessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(lt, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_logical = ShiftLogicalExecutor::new(ShiftOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_logical,
            [ShiftOpcode::SLL, ShiftOpcode::SRL].map(|x| x.global_opcode()),
        )?;

        let shift_right_arithmetic =
            ShiftRightArithmeticExecutor::new(ShiftOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_right_arithmetic,
            [ShiftOpcode::SRA].map(|x| x.global_opcode()),
        )?;

        let shift_w_logical = ShiftWLogicalExecutor::new(ShiftWOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_w_logical,
            [ShiftWOpcode::SLLW, ShiftWOpcode::SRLW].map(|x| x.global_opcode()),
        )?;

        let shift_w_right_arithmetic =
            ShiftWRightArithmeticExecutor::new(ShiftWOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_w_right_arithmetic,
            [ShiftWOpcode::SRAW].map(|x| x.global_opcode()),
        )?;

        let addi_w = AddIWExecutor::new(BaseAluWImmOpcode::CLASS_OFFSET);
        inventory.add_executor(
            addi_w,
            [BaseAluWImmOpcode::ADDIW].map(|x| x.global_opcode()),
        )?;

        let shift_w_logical_imm = ShiftWLogicalImmExecutor::new(ShiftWImmOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_w_logical_imm,
            [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW].map(|x| x.global_opcode()),
        )?;

        let shift_w_right_arithmetic_imm =
            ShiftWRightArithmeticImmExecutor::new(ShiftWImmOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_w_right_arithmetic_imm,
            [ShiftWImmOpcode::SRAIW].map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_byte =
            LoadSignExtendByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_sign_extend_byte,
            [LoadStoreOpcode::LOADB].map(|x| x.global_opcode()),
        )?;

        let load_byte = LoadByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_byte,
            [LoadStoreOpcode::LOADBU].map(|x| x.global_opcode()),
        )?;

        let store_byte = StoreByteExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            store_byte,
            [LoadStoreOpcode::STOREB].map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_halfword =
            LoadSignExtendHalfwordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_sign_extend_halfword,
            [LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let load_halfword = LoadHalfwordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_halfword,
            [LoadStoreOpcode::LOADHU].map(|x| x.global_opcode()),
        )?;

        let store_halfword = StoreHalfwordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            store_halfword,
            [LoadStoreOpcode::STOREH].map(|x| x.global_opcode()),
        )?;

        let load_sign_extend_word =
            LoadSignExtendWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_sign_extend_word,
            [LoadStoreOpcode::LOADW].map(|x| x.global_opcode()),
        )?;

        let load_word = LoadWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_word,
            [LoadStoreOpcode::LOADWU].map(|x| x.global_opcode()),
        )?;

        let store_word = StoreWordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            store_word,
            [LoadStoreOpcode::STOREW].map(|x| x.global_opcode()),
        )?;

        let load_doubleword = LoadDoublewordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_doubleword,
            [LoadStoreOpcode::LOADD].map(|x| x.global_opcode()),
        )?;

        let store_doubleword = StoreDoublewordExecutor::new(LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            store_doubleword,
            [LoadStoreOpcode::STORED].map(|x| x.global_opcode()),
        )?;

        let beq = BranchEqualCoreExecutor::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP);
        inventory.add_executor(beq, BranchEqualOpcode::iter().map(|x| x.global_opcode()))?;

        let blt = BranchLessThanCoreExecutor::new(BranchLessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(blt, BranchLessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let jal_lui = JalLuiExecutor::new();
        inventory.add_executor(jal_lui, JalLuiOpcode::iter().map(|x| x.global_opcode()))?;

        let jalr = JalrExecutor::new();
        inventory.add_executor(jalr, JalrOpcode::iter().map(|x| x.global_opcode()))?;

        let auipc = AuipcExecutor::new();
        inventory.add_executor(auipc, AuipcOpcode::iter().map(|x| x.global_opcode()))?;

        let addi = AddIExecutor::new(BaseAluImmOpcode::CLASS_OFFSET);
        inventory.add_executor(addi, [BaseAluImmOpcode::ADDI].map(|x| x.global_opcode()))?;

        let shift_logical_imm = ShiftLogicalImmExecutor::new(ShiftImmOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_logical_imm,
            [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI].map(|x| x.global_opcode()),
        )?;

        let shift_right_arithmetic_imm =
            ShiftRightArithmeticImmExecutor::new(ShiftImmOpcode::CLASS_OFFSET);
        inventory.add_executor(
            shift_right_arithmetic_imm,
            [ShiftImmOpcode::SRAI].map(|x| x.global_opcode()),
        )?;

        let less_than_imm = LessThanImmExecutor::new(LessThanImmOpcode::CLASS_OFFSET);
        inventory.add_executor(
            less_than_imm,
            LessThanImmOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let bitwise_logic_imm = BitwiseLogicImmExecutor::new(BaseAluImmOpcode::CLASS_OFFSET);
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
            phantom::HintInputSubEx,
            PhantomDiscriminant(Rv64Phantom::HintInput as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::HintRandomSubEx,
            PhantomDiscriminant(Rv64Phantom::HintRandom as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::PrintStrSubEx,
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

        let add_sub = AddSubAir::new(
            BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            AddSubCoreAir::new(range_checker, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(add_sub);

        let bitwise_logic = BitwiseLogicAir::new(
            BaseAluRegAdapterAir::new(exec_bridge, memory_bridge),
            BitwiseLogicCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(bitwise_logic);

        let add_sub_w = AddSubWAir::new(
            BaseAluWRegU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            crate::add_sub_w::AddSubWCoreAir::new(range_checker, BaseAluWOpcode::CLASS_OFFSET),
        );
        inventory.add_air(add_sub_w);

        let lt = LessThanAir::new(
            BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            LessThanCoreAir::new(range_checker, LessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(lt);

        let shift_logical = ShiftLogicalAir::new(
            BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftLogicalCoreAir::new(range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_logical);

        let shift_right_arithmetic = ShiftRightArithmeticAir::new(
            BaseAluRegU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftRightArithmeticCoreAir::new(range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_right_arithmetic);

        let shift_w_logical = ShiftWLogicalAir::new(
            BaseAluWRegU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            crate::shift_w::ShiftWLogicalCoreAir::new(range_checker, ShiftWOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_w_logical);

        let shift_w_right_arithmetic = ShiftWRightArithmeticAir::new(
            BaseAluWRegU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            crate::shift_w::ShiftWRightArithmeticCoreAir::new(
                range_checker,
                ShiftWOpcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(shift_w_right_arithmetic);

        let addi_w = AddIWAir::new(
            BaseAluWImmU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            AddICoreAir::new(
                range_checker,
                BaseAluWImmOpcode::CLASS_OFFSET,
                BaseAluWImmOpcode::ADDIW as usize,
            ),
        );
        inventory.add_air(addi_w);

        let shift_w_logical_imm = ShiftWLogicalImmAir::new(
            BaseAluWImmU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            ShiftLogicalImmCoreAir::new(range_checker, ShiftWImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_w_logical_imm);

        let shift_w_right_arithmetic_imm = ShiftWRightArithmeticImmAir::new(
            BaseAluWImmU16AdapterAir::new(exec_bridge, memory_bridge, range_checker),
            ShiftRightArithmeticImmCoreAir::new(range_checker, ShiftWImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_w_right_arithmetic_imm);

        let load_sign_extend_byte = LoadSignExtendByteAir::new(
            LoadByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadSignExtendByteCoreAir::new(
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu,
                range_checker,
            ),
        );
        inventory.add_air(load_sign_extend_byte);

        let load_byte = LoadByteAir::new(
            LoadByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadByteCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_byte);

        let store_byte = StoreByteAir::new(
            StoreByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreByteCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_byte);

        let load_sign_extend_halfword = LoadSignExtendHalfwordAir::new(
            LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadSignExtendHalfwordCoreAir::new(
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu,
                range_checker,
            ),
        );
        inventory.add_air(load_sign_extend_halfword);

        let load_halfword = LoadHalfwordAir::new(
            LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadHalfwordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_halfword);

        let store_halfword = StoreHalfwordAir::new(
            StoreMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreHalfwordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_halfword);

        let load_sign_extend_word = LoadSignExtendWordAir::new(
            LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadSignExtendWordCoreAir::new(
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu,
                range_checker,
            ),
        );
        inventory.add_air(load_sign_extend_word);

        let load_word = LoadWordAir::new(
            LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadWordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_word);

        let store_word = StoreWordAir::new(
            StoreMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreWordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_word);

        let load_doubleword = LoadDoublewordAir::new(
            LoadMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            LoadDoublewordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(load_doubleword);

        let store_doubleword = StoreDoublewordAir::new(
            StoreMultiByteAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                byte_ptr_max_bits,
            ),
            StoreDoublewordCoreAir::new(LoadStoreOpcode::CLASS_OFFSET, bitwise_lu),
        );
        inventory.add_air(store_doubleword);

        let beq = BranchEqualAir::new(
            BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(beq);

        let blt = BranchLessThanAir::new(
            BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchLessThanCoreAir::new(range_checker, BranchLessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(blt);

        let jal_lui = JalLuiAir::new(
            CondRdWriteAdapterAir::new(RdWriteAdapterAir::new(memory_bridge, exec_bridge)),
            JalLuiCoreAir::new(range_checker),
        );
        inventory.add_air(jal_lui);

        let jalr = JalrAir::new(
            JalrAdapterAir::new(memory_bridge, exec_bridge),
            JalrCoreAir::new(range_checker),
        );
        inventory.add_air(jalr);

        let auipc = AuipcAir::new(
            RdWriteAdapterAir::new(memory_bridge, exec_bridge),
            AuipcCoreAir::new(range_checker),
        );
        inventory.add_air(auipc);

        let addi = AddIAir::new(
            BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            AddICoreAir::new(
                range_checker,
                BaseAluImmOpcode::CLASS_OFFSET,
                BaseAluImmOpcode::ADDI as usize,
            ),
        );
        inventory.add_air(addi);

        let shift_logical_imm = ShiftLogicalImmAir::new(
            BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftLogicalImmCoreAir::new(range_checker, ShiftImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_logical_imm);

        let shift_right_arithmetic_imm = ShiftRightArithmeticImmAir::new(
            BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            ShiftRightArithmeticImmCoreAir::new(range_checker, ShiftImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift_right_arithmetic_imm);

        let less_than_imm = LessThanImmAir::new(
            BaseAluImmU16AdapterAir::new(exec_bridge, memory_bridge),
            LessThanImmCoreAir::new(range_checker, LessThanImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(less_than_imm);

        let bitwise_logic_imm = BitwiseLogicImmAir::new(
            BaseAluImmAdapterAir::new(exec_bridge, memory_bridge),
            BitwiseLogicImmCoreAir::new(bitwise_lu, BaseAluImmOpcode::CLASS_OFFSET),
        );
        inventory.add_air(bitwise_logic_imm);

        Ok(())
    }
}

pub struct Rv64ImCpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC> VmProverExtension<E, Rv64I> for Rv64ImCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<SC, CpuBackend<SC>>,
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
                inventory.add_periphery_chip_with_tracegen(chip.clone(), |chip, _| {
                    Ok(chip.generate_proving_ctx())
                });
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<AddSubAir>()?;
        let add_sub =
            AddSubChip::new(AddSubFiller::new(range_checker.clone()), mem_helper.clone());
        add_executor_chip_with_tracegen!(
            inventory,
            add_sub,
            crate::add_sub::trace::generate_trace_from_postflight
        );

        inventory.next_air::<BitwiseLogicAir>()?;
        let bitwise_logic = BitwiseLogicChip::new(
            BitwiseLogicFiller::new(bitwise_lu.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            bitwise_logic,
            crate::bitwise_logic::trace::generate_trace_from_postflight
        );

        inventory.next_air::<AddSubWAir>()?;
        let add_sub_w = AddSubWChip::new(
            crate::add_sub_w::AddSubWFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            add_sub_w,
            crate::add_sub_w::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LessThanAir>()?;
        let lt = LessThanChip::new(
            LessThanFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            lt,
            crate::less_than::trace::generate_trace_from_postflight
        );

        inventory.next_air::<ShiftLogicalAir>()?;
        let shift_logical = ShiftLogicalChip::new(
            ShiftLogicalFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_logical,
            crate::shift_logical::trace::generate_trace_from_postflight
        );

        inventory.next_air::<ShiftRightArithmeticAir>()?;
        let shift_right_arithmetic = ShiftRightArithmeticChip::new(
            ShiftRightArithmeticFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_right_arithmetic,
            crate::shift_right_arithmetic::trace::generate_trace_from_postflight
        );

        inventory.next_air::<ShiftWLogicalAir>()?;
        let shift_w_logical = ShiftWLogicalChip::new(
            crate::shift_w::ShiftWLogicalFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_w_logical,
            crate::shift_w::trace::generate_logical_trace_from_postflight
        );

        inventory.next_air::<ShiftWRightArithmeticAir>()?;
        let shift_w_right_arithmetic = ShiftWRightArithmeticChip::new(
            crate::shift_w::ShiftWRightArithmeticFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_w_right_arithmetic,
            crate::shift_w::trace::generate_right_arithmetic_trace_from_postflight
        );

        inventory.next_air::<AddIWAir>()?;
        let addi_w = AddIWChip::new(AddIFiller::new(range_checker.clone()), mem_helper.clone());
        add_executor_chip_with_tracegen!(
            inventory,
            addi_w,
            crate::addi::trace::generate_w_trace_from_postflight
        );

        inventory.next_air::<ShiftWLogicalImmAir>()?;
        let shift_w_logical_imm = ShiftWLogicalImmChip::new(
            ShiftLogicalImmFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_w_logical_imm,
            crate::shift_logical_imm::trace::generate_word_trace_from_postflight
        );

        inventory.next_air::<ShiftWRightArithmeticImmAir>()?;
        let shift_w_right_arithmetic_imm = ShiftWRightArithmeticImmChip::new(
            ShiftRightArithmeticImmFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_w_right_arithmetic_imm,
            crate::shift_right_arithmetic_imm::trace::generate_word_trace_from_postflight
        );

        inventory.next_air::<LoadSignExtendByteAir>()?;
        let load_sign_extend_byte_chip = LoadSignExtendByteChip::new(
            LoadSignExtendByteFiller::new(
                LoadByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_sign_extend_byte_chip,
            crate::load_sign_extend::byte::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LoadByteAir>()?;
        let load_byte_chip = LoadByteChip::new(
            LoadByteFiller::new(
                LoadByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_byte_chip,
            crate::load::byte::trace::generate_trace_from_postflight
        );

        inventory.next_air::<StoreByteAir>()?;
        let store_byte_chip = StoreByteChip::new(
            StoreByteFiller::new(
                StoreByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            store_byte_chip,
            crate::store::byte::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LoadSignExtendHalfwordAir>()?;
        let load_sign_extend_halfword_chip = LoadSignExtendHalfwordChip::new(
            LoadSignExtendHalfwordFiller::new(
                LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_sign_extend_halfword_chip,
            crate::load_sign_extend::halfword::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LoadHalfwordAir>()?;
        let load_halfword_chip = LoadHalfwordChip::new(
            LoadHalfwordFiller::new(
                LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_halfword_chip,
            crate::load::halfword::trace::generate_trace_from_postflight
        );

        inventory.next_air::<StoreHalfwordAir>()?;
        let store_halfword_chip = StoreHalfwordChip::new(
            StoreHalfwordFiller::new(
                StoreMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            store_halfword_chip,
            crate::store::halfword::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LoadSignExtendWordAir>()?;
        let load_sign_extend_word_chip = LoadSignExtendWordChip::new(
            LoadSignExtendWordFiller::new(
                LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_sign_extend_word_chip,
            crate::load_sign_extend::word::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LoadWordAir>()?;
        let load_word_chip = LoadWordChip::new(
            LoadWordFiller::new(
                LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_word_chip,
            crate::load::word::trace::generate_trace_from_postflight
        );

        inventory.next_air::<StoreWordAir>()?;
        let store_word_chip = StoreWordChip::new(
            StoreWordFiller::new(
                StoreMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            store_word_chip,
            crate::store::word::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LoadDoublewordAir>()?;
        let load_doubleword_chip = LoadDoublewordChip::new(
            LoadDoublewordFiller::new(
                LoadMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            load_doubleword_chip,
            crate::load::doubleword::trace::generate_trace_from_postflight
        );

        inventory.next_air::<StoreDoublewordAir>()?;
        let store_doubleword_chip = StoreDoublewordChip::new(
            StoreDoublewordFiller::new(
                StoreMultiByteAdapterFiller::new(byte_ptr_max_bits, range_checker.clone()),
                LoadStoreOpcode::CLASS_OFFSET,
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            store_doubleword_chip,
            crate::store::doubleword::trace::generate_trace_from_postflight
        );

        inventory.next_air::<BranchEqualAir>()?;
        let beq =
            BranchEqualChip::new(BranchEqualFiller::new(DEFAULT_PC_STEP), mem_helper.clone());
        add_executor_chip_with_tracegen!(
            inventory,
            beq,
            crate::branch_eq::trace::generate_trace_from_postflight
        );

        inventory.next_air::<BranchLessThanAir>()?;
        let blt = BranchLessThanChip::new(
            BranchLessThanFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            blt,
            crate::branch_lt::trace::generate_trace_from_postflight
        );

        inventory.next_air::<JalLuiAir>()?;
        let jal_lui = JalLuiChip::new(
            JalLuiFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            jal_lui,
            crate::jal_lui::trace::generate_trace_from_postflight
        );

        inventory.next_air::<JalrAir>()?;
        let jalr = JalrChip::new(
            JalrFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            jalr,
            crate::jalr::trace::generate_trace_from_postflight
        );

        inventory.next_air::<AuipcAir>()?;
        let auipc = AuipcChip::new(
            AuipcFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            auipc,
            crate::auipc::trace::generate_trace_from_postflight
        );

        inventory.next_air::<AddIAir>()?;
        let addi = AddIChip::new(AddIFiller::new(range_checker.clone()), mem_helper.clone());
        add_executor_chip_with_tracegen!(
            inventory,
            addi,
            crate::addi::trace::generate_trace_from_postflight
        );

        inventory.next_air::<ShiftLogicalImmAir>()?;
        let shift_logical_imm = ShiftLogicalImmChip::new(
            ShiftLogicalImmFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_logical_imm,
            crate::shift_logical_imm::trace::generate_trace_from_postflight
        );

        inventory.next_air::<ShiftRightArithmeticImmAir>()?;
        let shift_right_arithmetic_imm = ShiftRightArithmeticImmChip::new(
            ShiftRightArithmeticImmFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            shift_right_arithmetic_imm,
            crate::shift_right_arithmetic_imm::trace::generate_trace_from_postflight
        );

        inventory.next_air::<LessThanImmAir>()?;
        let less_than_imm = LessThanImmChip::new(
            LessThanImmFiller::new(range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            less_than_imm,
            crate::less_than_imm::trace::generate_trace_from_postflight
        );

        inventory.next_air::<BitwiseLogicImmAir>()?;
        let bitwise_logic_imm = BitwiseLogicImmChip::new(
            BitwiseLogicImmFiller::new(bitwise_lu.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            bitwise_logic_imm,
            crate::bitwise_logic_imm::trace::generate_trace_from_postflight
        );

        Ok(())
    }
}

impl VmExecutionExtension for Rv64M {
    type Executor = Rv64MExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Rv64MExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let mult = MultiplicationExecutor::new(MulOpcode::CLASS_OFFSET);
        inventory.add_executor(mult, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_w = MulWExecutor::new(MulWOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_w, MulWOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_h = MulHExecutor::new(MulHOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_h, MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let div_rem = DivRemExecutor::new(DivRemOpcode::CLASS_OFFSET);
        inventory.add_executor(div_rem, DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        let divrem_w = DivRemWExecutor::new(DivRemWOpcode::CLASS_OFFSET);
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

        let mult = MultiplicationAir::new(
            MultAdapterAir::new(exec_bridge, memory_bridge),
            MultiplicationCoreAir::new(range_tuple_checker, bitwise_lu, MulOpcode::CLASS_OFFSET),
        );
        inventory.add_air(mult);

        let mul_w = MulWAir::new(
            MultWAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            crate::mul_w::MulWCoreAir::new(
                range_tuple_checker,
                bitwise_lu,
                MulWOpcode::CLASS_OFFSET,
            ),
        );
        inventory.add_air(mul_w);

        let mul_h = MulHAir::new(
            MultAdapterAir::new(exec_bridge, memory_bridge),
            MulHCoreAir::new(bitwise_lu, range_tuple_checker),
        );
        inventory.add_air(mul_h);

        let div_rem = DivRemAir::new(
            MultAdapterAir::new(exec_bridge, memory_bridge),
            DivRemCoreAir::new(bitwise_lu, range_tuple_checker, DivRemOpcode::CLASS_OFFSET),
        );
        inventory.add_air(div_rem);

        let divrem_w = DivRemWAir::new(
            MultWAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
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
impl<E, SC> VmProverExtension<E, Rv64M> for Rv64ImCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &Rv64M,
        inventory: &mut ChipInventory<SC, CpuBackend<SC>>,
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
                inventory.add_periphery_chip_with_tracegen(chip.clone(), |chip, _| {
                    Ok(chip.generate_proving_ctx())
                });
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
                inventory.add_periphery_chip_with_tracegen(chip.clone(), |chip, _| {
                    Ok(chip.generate_proving_ctx())
                });
                chip
            }
        };

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<MultiplicationAir>()?;
        let mult = MultiplicationChip::new(
            MultiplicationFiller::new(range_tuple_checker.clone(), bitwise_lu.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            mult,
            crate::mul::trace::generate_trace_from_postflight
        );

        inventory.next_air::<MulWAir>()?;
        let mul_w = MulWChip::new(
            crate::mul_w::MulWFiller::new(range_tuple_checker.clone(), bitwise_lu.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            mul_w,
            crate::mul_w::trace::generate_trace_from_postflight
        );

        inventory.next_air::<MulHAir>()?;
        let mul_h = MulHChip::new(
            MulHFiller::new(bitwise_lu.clone(), range_tuple_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            mul_h,
            crate::mulh::trace::generate_trace_from_postflight
        );

        inventory.next_air::<DivRemAir>()?;
        let div_rem = DivRemChip::new(
            DivRemFiller::new(
                MultAdapterFiller,
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            div_rem,
            crate::divrem::trace::generate_trace_from_postflight
        );

        inventory.next_air::<DivRemWAir>()?;
        let divrem_w = DivRemWChip::new(
            crate::divrem_w::DivRemWFiller::new(
                MultWAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
            ),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            divrem_w,
            crate::divrem_w::trace::generate_trace_from_postflight
        );

        Ok(())
    }
}

impl VmExecutionExtension for Rv64Io {
    type Executor = Rv64IoExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<Rv64IoExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let hint_store = HintStoreExecutor::new(HintStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            hint_store,
            HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;
        let reveal = RevealExecutor::new(RevealOpcode::CLASS_OFFSET);
        inventory.add_executor(reveal, RevealOpcode::iter().map(|x| x.global_opcode()))?;

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

        let hint_store = HintStoreAir::new(
            exec_bridge,
            memory_bridge,
            range_checker,
            HintStoreOpcode::CLASS_OFFSET,
            byte_ptr_max_bits,
        );
        inventory.add_air(hint_store);

        let reveal = RevealAir::new(
            RevealAdapterAir::new(memory_bridge, exec_bridge, range_checker, byte_ptr_max_bits),
            RevealCoreAir::new(bitwise_lu),
        );
        inventory.add_air(reveal);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC> VmProverExtension<E, Rv64Io> for Rv64ImCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<SC, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip_with_tracegen(chip.clone(), |chip, _| {
                    Ok(chip.generate_proving_ctx())
                });
                chip
            }
        };

        inventory.next_air::<HintStoreAir>()?;
        let hint_store = HintStoreChip::new(
            HintStoreFiller::new(byte_ptr_max_bits, range_checker.clone()),
            mem_helper.clone(),
        );
        add_executor_chip_with_tracegen!(
            inventory,
            hint_store,
            crate::hintstore::trace::generate_trace_from_postflight
        );

        inventory.next_air::<RevealAir>()?;
        let reveal = RevealChip::new(
            RevealFiller::new(byte_ptr_max_bits, range_checker, bitwise_lu),
            mem_helper,
        );
        add_executor_chip_with_tracegen!(
            inventory,
            reveal,
            crate::reveal::trace::generate_trace_from_postflight
        );

        Ok(())
    }
}

/// Phantom sub-executors
mod phantom {
    use std::{iter::repeat_with, sync::Once};

    use eyre::{bail, eyre, WrapErr};
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_instructions::{riscv::MEMORY_AS, PhantomDiscriminant};
    use openvm_platform::memory::MEM_SIZE;
    use rand::{rngs::StdRng, Rng};

    use crate::adapters::{read_register, REGISTER_NUM_LIMBS};

    const HINT_DWORD_BYTES: usize = REGISTER_NUM_LIMBS;

    pub struct HintInputSubEx;
    pub struct HintRandomSubEx;
    pub struct PrintStrSubEx;

    impl PhantomSubExecutor for HintInputSubEx {
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
            let input = match streams.input_stream.pop_front() {
                Some(input) => input,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.set_input(input);
            Ok(())
        }
    }

    impl PhantomSubExecutor for HintRandomSubEx {
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
            static WARN_ONCE: Once = Once::new();
            WARN_ONCE.call_once(|| {
                eprintln!("WARNING: Using fixed-seed RNG for deterministic randomness. Consider security implications for your use case.");
            });

            let num_words: u64 = read_register(memory, a);
            let num_bytes: u64 = num_words
                .checked_mul(HINT_DWORD_BYTES as u64)
                .ok_or_else(|| eyre!("HINT_RANDOM byte count overflow"))?;
            if num_bytes > MEM_SIZE as u64 {
                bail!("HINT_RANDOM byte count {num_bytes} exceeds resource limit {MEM_SIZE}");
            }
            let num_bytes = num_bytes as usize;
            streams
                .hint_stream
                .try_set_hint_from_iter(
                    num_bytes,
                    repeat_with(|| rng.random::<u8>()).take(num_bytes),
                )
                .wrap_err("failed to reserve HINT_RANDOM stream")?;
            Ok(())
        }
    }

    impl PhantomSubExecutor for PrintStrSubEx {
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
            let ptr = read_register(memory, a);
            let len = read_register(memory, b);
            let bytes = memory
                .checked_u8_slice(MEMORY_AS, ptr, len)
                .map_err(|error| eyre!("PRINT_STR {error}"))?;
            let peeked_str = std::str::from_utf8(bytes)?;
            print!("{peeked_str}");
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use openvm_circuit::{
            arch::{MemoryConfig, Streams},
            system::memory::online::{AddressMap, GuestMemory},
        };
        use openvm_instructions::riscv::{
            MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS,
        };
        use rand::{rngs::StdRng, SeedableRng};

        use super::*;
        use crate::adapters::memory_write;

        const OPERAND_A_REG: u32 = REGISTER_NUM_LIMBS as u32;
        const OPERAND_B_REG: u32 = 2 * REGISTER_NUM_LIMBS as u32;

        fn memory_with_operands(first: u64, second: u64) -> GuestMemory {
            let mut config = MemoryConfig::default();
            config.addr_spaces[MEMORY_AS as usize].num_cells = 512;
            let mut memory = GuestMemory::new(AddressMap::from_mem_config(&config));
            memory_write(
                &mut memory,
                REGISTER_AS,
                OPERAND_A_REG,
                first.to_le_bytes(),
            );
            memory_write(
                &mut memory,
                REGISTER_AS,
                OPERAND_B_REG,
                second.to_le_bytes(),
            );
            memory
        }

        fn phantom_error(executor: &dyn PhantomSubExecutor, memory: &GuestMemory) -> String {
            let mut streams = Streams::default();
            let mut rng = StdRng::seed_from_u64(0);
            executor
                .phantom_execute(
                    memory,
                    &mut streams,
                    &mut rng,
                    PhantomDiscriminant(0),
                    OPERAND_A_REG,
                    OPERAND_B_REG,
                    0,
                )
                .unwrap_err()
                .to_string()
        }

        #[test]
        fn print_str_checks_full_register_length_against_guest_memory() {
            let memory = memory_with_operands(0x400, 1u64 << 32);
            let message = phantom_error(&PrintStrSubEx, &memory);

            assert!(
                message.contains("PRINT_STR memory range out of bounds"),
                "unexpected error: {message}"
            );
        }

        #[test]
        fn print_str_rejects_full_register_range_overflow() {
            let memory = memory_with_operands(u64::MAX, 1);
            let message = phantom_error(&PrintStrSubEx, &memory);

            assert_eq!(message, "PRINT_STR range overflow");
        }

        #[test]
        fn hint_random_applies_resource_limit_after_full_register_multiplication() {
            let memory = memory_with_operands(u64::from(u32::MAX) + 1, 0);
            let message = phantom_error(&HintRandomSubEx, &memory);

            assert!(
                message.contains("exceeds resource limit"),
                "unexpected error: {message}"
            );
        }

        #[test]
        fn hint_random_rejects_full_register_byte_count_overflow() {
            let memory = memory_with_operands(u64::MAX, 0);
            let message = phantom_error(&HintRandomSubEx, &memory);

            assert_eq!(message, "HINT_RANDOM byte count overflow");
        }
    }
}
