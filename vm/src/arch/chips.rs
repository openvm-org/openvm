use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    range_tuple::RangeTupleCheckerChip, var_range::VariableRangeCheckerChip,
    xor::lookup::XorLookupChip,
};
use afs_stark_backend::rap::{get_air_name, AnyRap};
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;
use strum_macros::IntoStaticStr;

use crate::{
    alu::ArithmeticLogicChip,
    arch::ExecutionState,
    branch_eq::Rv32BranchEqualChip,
    branch_lt::Rv32BranchLessThanChip,
    castf::CastFChip,
    core::CoreChip,
    ecc::{EcAddUnequalChip, EcDoubleChip},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    loadstore::Rv32LoadStoreChip,
    memory::MemoryChipRef,
    modular_addsub::ModularAddSubChip,
    modular_multdiv::ModularMultDivChip,
    new_alu::Rv32ArithmeticLogicChip,
    new_divrem::Rv32DivRemChip,
    new_lt::Rv32LessThanChip,
    new_mul::Rv32MultiplicationChip,
    new_mulh::Rv32MulHChip,
    new_shift::Rv32ShiftChip,
    program::{ExecutionError, Instruction},
    rv32_jal_lui::Rv32JalLuiChip,
    shift::ShiftChip,
    ui::UiChip,
    uint_multiplication::UintMultiplicationChip,
};

#[enum_dispatch]
pub trait InstructionExecutor<F> {
    /// Runtime execution of the instruction, if the instruction is owned by the
    /// current instance. May internally store records of this call for later trace generation.
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError>;

    /// For display purposes. From absolute opcode as `usize`, return the string name of the opcode
    /// if it is a supported opcode by the present executor.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

#[enum_dispatch]
pub trait MachineChip<F>: Sized {
    // Functions for when chip owns a single AIR
    fn generate_trace(self) -> RowMajorMatrix<F>;
    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>;
    fn air_name(&self) -> String;
    fn generate_public_values(&mut self) -> Vec<F> {
        vec![]
    }
    fn current_trace_height(&self) -> usize;
    fn trace_width(&self) -> usize;

    // Functions for when chip owns multiple AIRs.
    // Default implementations fallback to single AIR functions, but
    // these can be overridden, in which case the single AIR functions
    // should be `unreachable!()`.
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        vec![self.generate_trace()]
    }
    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        vec![self.air()]
    }
    fn air_names(&self) -> Vec<String> {
        vec![self.air_name()]
    }
    fn generate_public_values_per_air(&mut self) -> Vec<Vec<F>> {
        vec![self.generate_public_values()]
    }
    fn current_trace_heights(&self) -> Vec<usize> {
        vec![self.current_trace_height()]
    }
    fn trace_widths(&self) -> Vec<usize> {
        vec![self.trace_width()]
    }

    /// For metrics collection
    fn current_trace_cells(&self) -> Vec<usize> {
        self.trace_widths()
            .into_iter()
            .zip_eq(self.current_trace_heights())
            .map(|(width, height)| width * height)
            .collect()
    }
}

impl<F, C: InstructionExecutor<F>> InstructionExecutor<F> for Rc<RefCell<C>> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        self.borrow_mut().execute(instruction, prev_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.borrow().get_opcode_name(opcode)
    }
}

impl<F, C: MachineChip<F>> MachineChip<F> for Rc<RefCell<C>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        match Rc::try_unwrap(self) {
            Ok(ref_cell) => ref_cell.into_inner().generate_trace(),
            Err(_) => panic!("cannot generate trace while other chips still hold a reference"),
        }
    }

    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        match Rc::try_unwrap(self) {
            Ok(ref_cell) => ref_cell.into_inner().generate_traces(),
            Err(_) => panic!("cannot generate trace while other chips still hold a reference"),
        }
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.borrow().air()
    }
    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.borrow().airs()
    }

    fn generate_public_values(&mut self) -> Vec<F> {
        self.borrow_mut().generate_public_values()
    }
    fn generate_public_values_per_air(&mut self) -> Vec<Vec<F>> {
        self.borrow_mut().generate_public_values_per_air()
    }

    fn air_name(&self) -> String {
        self.borrow().air_name()
    }
    fn air_names(&self) -> Vec<String> {
        self.borrow().air_names()
    }

    fn current_trace_height(&self) -> usize {
        self.borrow().current_trace_height()
    }
    fn current_trace_heights(&self) -> Vec<usize> {
        self.borrow().current_trace_heights()
    }

    fn trace_width(&self) -> usize {
        self.borrow().trace_width()
    }
    fn trace_widths(&self) -> Vec<usize> {
        self.borrow().trace_widths()
    }
}

#[derive(Debug, Clone, EnumDiscriminants)]
#[strum_discriminants(derive(Serialize, Deserialize))]
#[strum_discriminants(name(ExecutorName))]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum InstructionExecutorVariant<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ModularAddSub(Rc<RefCell<ModularAddSubChip<F, 32, 8>>>),
    ModularMultDiv(Rc<RefCell<ModularMultDivChip<F, 63, 32, 8>>>),
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
    Ui(Rc<RefCell<UiChip<F>>>),
    CastF(Rc<RefCell<CastFChip<F>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}

#[derive(Debug, Clone, IntoStaticStr)]
#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    Memory(MemoryChipRef<F>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
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
    Ui(Rc<RefCell<UiChip<F>>>),
    LoadStoreRv32(Rc<RefCell<Rv32LoadStoreChip<F>>>),
    BranchEqualRv32(Rc<RefCell<Rv32BranchEqualChip<F>>>),
    BranchLessThanRv32(Rc<RefCell<Rv32BranchLessThanChip<F>>>),
    JalLuiRv32(Rc<RefCell<Rv32JalLuiChip<F>>>),
    CastF(Rc<RefCell<CastFChip<F>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}

impl<F: PrimeField32> MachineChip<F> for Arc<VariableRangeCheckerChip> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        VariableRangeCheckerChip::generate_trace(&self)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
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

impl<F: PrimeField32, const N: usize> MachineChip<F> for Arc<RangeTupleCheckerChip<N>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        RangeTupleCheckerChip::generate_trace(&self)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
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

impl<F: PrimeField32, const M: usize> MachineChip<F> for Arc<XorLookupChip<M>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        XorLookupChip::generate_trace(&self)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air)
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
