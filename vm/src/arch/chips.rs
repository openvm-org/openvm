use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_derive::Chip;
use afs_primitives::{
    range_tuple::RangeTupleCheckerChip, var_range::VariableRangeCheckerChip,
    xor::lookup::XorLookupChip,
};
use afs_stark_backend::rap::get_air_name;
use enum_dispatch::enum_dispatch;
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;
use strum_macros::IntoStaticStr;

use crate::{
    arch::ExecutionState,
    intrinsics::{
        castf::CastFChip,
        ecc::{EcAddUnequalChip, EcDoubleChip},
        hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
        modular_addsub::ModularAddSubChip,
        modular_multdiv::ModularMultDivChip,
        uint_multiplication::UintMultiplicationChip,
    },
    kernels::{
        core::CoreChip, field_extension::chip::FieldExtensionArithmeticChip,
        new_field_arithmetic::NewFieldArithmeticChip,
    },
    old::{alu::ArithmeticLogicChip, shift::ShiftChip},
    rv32im::{
        branch_eq::Rv32BranchEqualChip, branch_lt::Rv32BranchLessThanChip,
        loadstore::Rv32LoadStoreChip, new_alu::Rv32ArithmeticLogicChip, new_divrem::Rv32DivRemChip,
        new_lt::Rv32LessThanChip, new_mul::Rv32MultiplicationChip, new_mulh::Rv32MulHChip,
        new_shift::Rv32ShiftChip, rv32_auipc::Rv32AuipcChip, rv32_jal_lui::Rv32JalLuiChip,
        rv32_jalr::Rv32JalrChip,
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

// TODO[jpw]: consider renaming this BaseChip and moving to stark-backend
/// This trait contains the functions of a chip that do not need to know about the STARK config.
/// Currently also specialized to AIRs with only a single common main trace matrix and no cached trace.
/// For proving, the trait [Chip](afs_stark_backend::chip::Chip) must also be implemented.
#[enum_dispatch]
pub trait VmChip<F>: Sized {
    fn generate_trace(self) -> RowMajorMatrix<F>;
    fn air_name(&self) -> String;
    fn generate_public_values(&mut self) -> Vec<F> {
        vec![]
    }
    fn current_trace_height(&self) -> usize;
    /// Width of the common main trace
    fn trace_width(&self) -> usize;

    /// For metrics collection
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }
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

impl<F, C: VmChip<F>> VmChip<F> for Rc<RefCell<C>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        match Rc::try_unwrap(self) {
            Ok(ref_cell) => ref_cell.into_inner().generate_trace(),
            Err(_) => panic!("cannot generate trace while other chips still hold a reference"),
        }
    }

    fn generate_public_values(&mut self) -> Vec<F> {
        self.borrow_mut().generate_public_values()
    }

    fn air_name(&self) -> String {
        self.borrow().air_name()
    }

    fn current_trace_height(&self) -> usize {
        self.borrow().current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.borrow().trace_width()
    }
}

#[derive(Clone, EnumDiscriminants)]
#[strum_discriminants(derive(Serialize, Deserialize))]
#[strum_discriminants(name(ExecutorName))]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum AxVmInstructionExecutor<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    FieldArithmetic(Rc<RefCell<NewFieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ArithmeticLogicUnitRv32(Rc<RefCell<Rv32ArithmeticLogicChip<F>>>),
    ArithmeticLogicUnit256(Rc<RefCell<ArithmeticLogicChip<F, 32, 8>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    U256Multiplication(Rc<RefCell<UintMultiplicationChip<F, 32, 8>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    Shift256(Rc<RefCell<ShiftChip<F, 32, 8>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    // TO BE REPLACED:
    CastF(Rc<RefCell<CastFChip<F>>>),
    ModularAddSub(Rc<RefCell<ModularAddSubChip<F, 32, 8>>>),
    ModularMultDiv(Rc<RefCell<ModularMultDivChip<F, 63, 32, 8>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}

#[derive(Clone, IntoStaticStr, Chip)]
#[enum_dispatch(VmChip<F>)]
pub enum AxVmChip<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    FieldArithmetic(Rc<RefCell<NewFieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    RangeChecker(Arc<VariableRangeCheckerChip>),
    RangeTupleChecker(Arc<RangeTupleCheckerChip<2>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ByteXor(Arc<XorLookupChip<8>>),
    ArithmeticLogicUnitRv32(Rc<RefCell<Rv32ArithmeticLogicChip<F>>>),
    ArithmeticLogicUnit256(Rc<RefCell<ArithmeticLogicChip<F, 32, 8>>>),
    LessThanRv32(Rc<RefCell<Rv32LessThanChip<F>>>),
    MultiplicationRv32(Rc<RefCell<Rv32MultiplicationChip<F>>>),
    MultiplicationHighRv32(Rc<RefCell<Rv32MulHChip<F>>>),
    U256Multiplication(Rc<RefCell<UintMultiplicationChip<F, 32, 8>>>),
    DivRemRv32(Rc<RefCell<Rv32DivRemChip<F>>>),
    ShiftRv32(Rc<RefCell<Rv32ShiftChip<F>>>),
    Shift256(Rc<RefCell<ShiftChip<F, 32, 8>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    JalrRv32(Rc<RefCell<Rv32JalrChip<F>>>),
    AuipcRv32(Rc<RefCell<Rv32AuipcChip<F>>>),
    // TO BE REPLACED:
    CastF(Rc<RefCell<CastFChip<F>>>),
    ModularAddSub(Rc<RefCell<ModularAddSubChip<F, 32, 8>>>),
    ModularMultDiv(Rc<RefCell<ModularMultDivChip<F, 63, 32, 8>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}

impl<F: PrimeField32> VmChip<F> for Arc<VariableRangeCheckerChip> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        VariableRangeCheckerChip::generate_trace(&self)
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        1 << (1 + self.air.bus.range_max_bits)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<F: PrimeField32, const N: usize> VmChip<F> for Arc<RangeTupleCheckerChip<N>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        RangeTupleCheckerChip::generate_trace(&self)
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.air.height() as usize
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<F: PrimeField32, const M: usize> VmChip<F> for Arc<XorLookupChip<M>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        XorLookupChip::generate_trace(&self)
    }

    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        1 << (2 * M)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
