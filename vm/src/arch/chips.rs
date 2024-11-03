use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChip, range_tuple::RangeTupleCheckerChip,
    var_range::VariableRangeCheckerChip,
};
use axvm_instructions::instruction::Instruction;
use enum_dispatch::enum_dispatch;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;
use strum_macros::IntoStaticStr;

use crate::{
    arch::ExecutionState,
    intrinsics::{
        ecc::{
            pairing::{
                EcLineMul013By013Chip, EcLineMulBy01234Chip, MillerDoubleAndAddStepChip,
                MillerDoubleStepChip,
            },
            sw::{EcAddNeChip, EcDoubleChip},
        },
        hashes::{keccak256::KeccakVmChip, poseidon2::Poseidon2Chip},
        int256::{
            Rv32BaseAlu256Chip, Rv32LessThan256Chip, Rv32Multiplication256Chip, Rv32Shift256Chip,
        },
        modular::{ModularAddSubChip, ModularMulDivChip},
    },
    kernels::{
        branch_eq::KernelBranchEqChip, castf::CastFChip, field_arithmetic::FieldArithmeticChip,
        field_extension::FieldExtensionChip, fri::FriMatOpeningChip, jal::KernelJalChip,
        loadstore::KernelLoadStoreChip, public_values::PublicValuesChip,
    },
    rv32im::*,
    system::{phantom::PhantomChip, program::ExecutionError},
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
#[derive(EnumDiscriminants)]
#[strum_discriminants(derive(Serialize, Deserialize, Ord, PartialOrd))]
#[strum_discriminants(name(ExecutorName))]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum AxVmInstructionExecutor<F: PrimeField32> {
    Phantom(Rc<RefCell<PhantomChip<F>>>),
    // Native kernel:
    LoadStore(Rc<RefCell<KernelLoadStoreChip<F, 1>>>),
    BranchEqual(Rc<RefCell<KernelBranchEqChip<F>>>),
    Jal(Rc<RefCell<KernelJalChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionChip<F>>>),
    PublicValues(Rc<RefCell<PublicValuesChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    FriMatOpening(Rc<RefCell<FriMatOpeningChip<F>>>),
    CastF(Rc<RefCell<CastFChip<F>>>),
    // Rv32 (for standard 32-bit integers):
    BaseAluRv32(Rc<RefCell<Rv32BaseAluChip<F>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    LoadSignExtendRv32(Rc<RefCell<Rv32LoadSignExtendChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    // Intrinsics:
    HintStoreRv32(Rc<RefCell<Rv32HintStoreChip<F>>>),
    Keccak256Rv32(Rc<RefCell<KeccakVmChip<F>>>),
    // 256Rv32 (for 256-bit integers):
    BaseAlu256Rv32(Rc<RefCell<Rv32BaseAlu256Chip<F>>>),
    Shift256Rv32(Rc<RefCell<Rv32Shift256Chip<F>>>),
    LessThan256Rv32(Rc<RefCell<Rv32LessThan256Chip<F>>>),
    Multiplication256Rv32(Rc<RefCell<Rv32Multiplication256Chip<F>>>),
    // Modular arithmetic:
    ModularAddSubRv32_1x32(Rc<RefCell<ModularAddSubChip<F, 1, 32>>>),
    ModularMulDivRv32_1x32(Rc<RefCell<ModularMulDivChip<F, 1, 32>>>),
    ModularAddSubRv32_3x16(Rc<RefCell<ModularAddSubChip<F, 3, 16>>>),
    ModularMulDivRv32_3x16(Rc<RefCell<ModularMulDivChip<F, 3, 16>>>),
    EcAddNeRv32_2x32(Rc<RefCell<EcAddNeChip<F, 2, 32>>>),
    EcDoubleRv32_2x32(Rc<RefCell<EcDoubleChip<F, 2, 32>>>),
    EcAddNeRv32_6x16(Rc<RefCell<EcAddNeChip<F, 6, 16>>>),
    EcDoubleRv32_6x16(Rc<RefCell<EcDoubleChip<F, 6, 16>>>),
    EcLineMul013By013(Rc<RefCell<EcLineMul013By013Chip<F, 4, 10, 32>>>),
    EcLineMulBy01234(Rc<RefCell<EcLineMulBy01234Chip<F, 12, 12, 32>>>),
    // 32-bytes or 48-bytes prime.
    MillerDoubleStepRv32_32(Rc<RefCell<MillerDoubleStepChip<F, 4, 8, 32>>>),
    MillerDoubleStepRv32_48(Rc<RefCell<MillerDoubleStepChip<F, 12, 24, 16>>>),
    MillerDoubleAndAddStepRv32_32(Rc<RefCell<MillerDoubleAndAddStepChip<F, 4, 12, 32>>>),
    MillerDoubleAndAddStepRv32_48(Rc<RefCell<MillerDoubleAndAddStepChip<F, 12, 36, 16>>>),
}

/// ATTENTION: CAREFULLY MODIFY THE ORDER OF ENTRIES. the order of entries determines the AIR ID of
/// each chip. Change of the order may cause break changes of VKs.
#[derive(IntoStaticStr, ChipUsageGetter, Chip)]
pub enum AxVmChip<F: PrimeField32> {
    // Lookup tables that are not executors:
    RangeChecker(Arc<VariableRangeCheckerChip>),
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    BitwiseOperationLookup(Arc<BitwiseOperationLookupChip<8>>),
    // System:
    Phantom(Rc<RefCell<PhantomChip<F>>>),
    // Native kernel:
    LoadStore(Rc<RefCell<KernelLoadStoreChip<F, 1>>>),
    BranchEqual(Rc<RefCell<KernelBranchEqChip<F>>>),
    Jal(Rc<RefCell<KernelJalChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    FriMatOpening(Rc<RefCell<FriMatOpeningChip<F>>>),
    CastF(Rc<RefCell<CastFChip<F>>>),
    // Rv32 (for standard 32-bit integers):
    BaseAluRv32(Rc<RefCell<Rv32BaseAluChip<F>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    LoadSignExtendRv32(Rc<RefCell<Rv32LoadSignExtendChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    // Intrinsics:
    HintStoreRv32(Rc<RefCell<Rv32HintStoreChip<F>>>),
    Keccak256Rv32(Rc<RefCell<KeccakVmChip<F>>>),
    // 256Rv32 (for 256-bit integers):
    BaseAlu256Rv32(Rc<RefCell<Rv32BaseAlu256Chip<F>>>),
    LessThan256Rv32(Rc<RefCell<Rv32LessThan256Chip<F>>>),
    Multiplication256Rv32(Rc<RefCell<Rv32Multiplication256Chip<F>>>),
    Shift256Rv32(Rc<RefCell<Rv32Shift256Chip<F>>>),
    // Modular arithmetic:
    ModularAddSubRv32_1x32(Rc<RefCell<ModularAddSubChip<F, 1, 32>>>),
    ModularMulDivRv32_1x32(Rc<RefCell<ModularMulDivChip<F, 1, 32>>>),
    ModularAddSubRv32_3x16(Rc<RefCell<ModularAddSubChip<F, 3, 16>>>),
    ModularMulDivRv32_3x16(Rc<RefCell<ModularMulDivChip<F, 3, 16>>>),
    EcAddNeRv32_2x32(Rc<RefCell<EcAddNeChip<F, 2, 32>>>),
    EcDoubleRv32_2x32(Rc<RefCell<EcDoubleChip<F, 2, 32>>>),
    EcAddNeRv32_6x16(Rc<RefCell<EcAddNeChip<F, 6, 16>>>),
    EcDoubleRv32_6x16(Rc<RefCell<EcDoubleChip<F, 6, 16>>>),
    EcLineMul013By013(Rc<RefCell<EcLineMul013By013Chip<F, 4, 10, 32>>>),
    EcLineMulBy01234(Rc<RefCell<EcLineMulBy01234Chip<F, 12, 12, 32>>>),
    // 32-bytes or 48-bytes prime.
    MillerDoubleStepRv32_32(Rc<RefCell<MillerDoubleStepChip<F, 4, 8, 32>>>),
    MillerDoubleStepRv32_48(Rc<RefCell<MillerDoubleStepChip<F, 12, 24, 16>>>),
    MillerDoubleAndAddStepRv32_32(Rc<RefCell<MillerDoubleAndAddStepChip<F, 4, 12, 32>>>),
    MillerDoubleAndAddStepRv32_48(Rc<RefCell<MillerDoubleAndAddStepChip<F, 12, 36, 16>>>),
}
