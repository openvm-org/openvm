use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_derive::{Chip, ChipUsageGetter};
use afs_primitives::{
    range_tuple::RangeTupleCheckerChip, var_range::VariableRangeCheckerChip,
    xor::lookup::XorLookupChip,
};
use enum_dispatch::enum_dispatch;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;
use strum_macros::IntoStaticStr;

use crate::{
    arch::ExecutionState,
    intrinsics::{
        ecc::{EcAddUnequalChip, EcDoubleChip},
        hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
        modular::{ModularAddSubChip, ModularMulDivChip},
    },
    kernels::{
        branch_eq::KernelBranchEqChip,
        castf::CastFChip,
        core::CoreChip,
        field_arithmetic::FieldArithmeticChip,
        field_extension::FieldExtensionChip,
        jal::KernelJalChip,
        modular::{KernelModularAddSubChip, KernelModularMulDivChip},
    },
    old::{
        alu::ArithmeticLogicChip, shift::ShiftChip, uint_multiplication::UintMultiplicationChip,
    },
    rv32im::{
        base_alu::Rv32BaseAluChip, branch_eq::Rv32BranchEqualChip,
        branch_lt::Rv32BranchLessThanChip, load_sign_extend::Rv32LoadSignExtendChip,
        loadstore::Rv32LoadStoreChip, new_divrem::Rv32DivRemChip, new_lt::Rv32LessThanChip,
        new_mul::Rv32MultiplicationChip, new_mulh::Rv32MulHChip, new_shift::Rv32ShiftChip,
        rv32_auipc::Rv32AuipcChip, rv32_jal_lui::Rv32JalLuiChip, rv32_jalr::Rv32JalrChip,
    },
    system::program::{ExecutionError, Instruction},
};

#[enum_dispatch]
pub trait InstructionExecutor<F> {
    /// Runtime execution of the instruction, if the instruction is owned by the
    /// current instance. May internally store records of this call for later trace generation.
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError>;

    /// For display purposes. From absolute opcode as `usize`, return the string name of the opcode
    /// if it is a supported opcode by the present executor.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

impl<F, C: InstructionExecutor<F>> InstructionExecutor<F> for Rc<RefCell<C>> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        prev_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        self.borrow_mut().execute(instruction, prev_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.borrow().get_opcode_name(opcode)
    }
}

/// ATTENTION: CAREFULLY MODIFY THE ORDER OF ENTRIES. the order of entries determines the AIR ID of
/// each chip. Change of the order may cause break changes of VKs.
#[derive(Clone, EnumDiscriminants)]
#[strum_discriminants(derive(Serialize, Deserialize, Ord, PartialOrd))]
#[strum_discriminants(name(ExecutorName))]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum AxVmInstructionExecutor<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    BranchEqual(Rc<RefCell<KernelBranchEqChip<F>>>),
    Jal(Rc<RefCell<KernelJalChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ArithmeticLogicUnitRv32(Rc<RefCell<Rv32BaseAluChip<F>>>),
    ArithmeticLogicUnit256(Rc<RefCell<ArithmeticLogicChip<F, 32, 8>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    U256Multiplication(Rc<RefCell<UintMultiplicationChip<F, 32, 8>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    Shift256(Rc<RefCell<ShiftChip<F, 32, 8>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    LoadSignExtendRv32(Rc<RefCell<Rv32LoadSignExtendChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    // Intrinsics:
    ModularAddSubRv32_1x32(Rc<RefCell<ModularAddSubChip<F, 1, 32>>>),
    ModularMulDivRv32_1x32(Rc<RefCell<ModularMulDivChip<F, 1, 32>>>),
    ModularAddSubRv32_3x16(Rc<RefCell<ModularAddSubChip<F, 3, 16>>>),
    ModularMulDivRv32_3x16(Rc<RefCell<ModularMulDivChip<F, 3, 16>>>),
    // TO BE REPLACED:
    CastF(Rc<RefCell<CastFChip<F>>>),
    ModularAddSub(Rc<RefCell<KernelModularAddSubChip<F, 32>>>),
    ModularMultDiv(Rc<RefCell<KernelModularMulDivChip<F, 32>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}

/// ATTENTION: CAREFULLY MODIFY THE ORDER OF ENTRIES. the order of entries determines the AIR ID of
/// each chip. Change of the order may cause break changes of VKs.
#[derive(Clone, IntoStaticStr, ChipUsageGetter, Chip)]
pub enum AxVmChip<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    BranchEqual(Rc<RefCell<KernelBranchEqChip<F>>>),
    Jal(Rc<RefCell<KernelJalChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    RangeChecker(Arc<VariableRangeCheckerChip>),
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ByteXor(Arc<XorLookupChip<8>>),
    ArithmeticLogicUnitRv32(Rc<RefCell<Rv32BaseAluChip<F>>>),
    ArithmeticLogicUnit256(Rc<RefCell<ArithmeticLogicChip<F, 32, 8>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    U256Multiplication(Rc<RefCell<UintMultiplicationChip<F, 32, 8>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    Shift256(Rc<RefCell<ShiftChip<F, 32, 8>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    LoadSignExtendRv32(Rc<RefCell<Rv32LoadSignExtendChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    // Intrinsics:
    ModularAddSubRv32_1x32(Rc<RefCell<ModularAddSubChip<F, 1, 32>>>),
    ModularMulDivRv32_1x32(Rc<RefCell<ModularMulDivChip<F, 1, 32>>>),
    ModularAddSubRv32_3x16(Rc<RefCell<ModularAddSubChip<F, 3, 16>>>),
    ModularMulDivRv32_3x16(Rc<RefCell<ModularMulDivChip<F, 3, 16>>>),
    // TO BE REPLACED:
    CastF(Rc<RefCell<CastFChip<F>>>),
    ModularAddSub(Rc<RefCell<KernelModularAddSubChip<F, 32>>>),
    ModularMultDiv(Rc<RefCell<KernelModularMulDivChip<F, 32>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}
