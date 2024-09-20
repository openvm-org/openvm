use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    range_tuple::RangeTupleCheckerChip, var_range::VariableRangeCheckerChip,
    xor::lookup::XorLookupChip,
};
use afs_stark_backend::rap::AnyRap;
use enum_dispatch::enum_dispatch;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use strum_macros::IntoStaticStr;

use crate::{
    arch::columns::ExecutionState,
    castf::CastFChip,
    core::CoreChip,
    ecc::{EcAddUnequalChip, EcDoubleChip},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    memory::MemoryChip,
    modular_arithmetic::{ModularArithmeticAirVariant, ModularArithmeticChip},
    program::{ExecutionError, Instruction, ProgramChip},
    shift::ShiftChip,
    ui::UiChip,
    uint_arithmetic::UintArithmeticChip,
    uint_multiplication::UintMultiplicationChip,
};

#[enum_dispatch]
pub trait InstructionExecutor<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError>;
}

#[enum_dispatch]
pub trait MachineChip<F> {
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>>;
    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>;
    fn generate_public_values(&mut self) -> Vec<Vec<F>> {
        (0..self.trace_widths().len()).map(|_| vec![]).collect()
    }
    fn current_trace_heights(&self) -> Vec<usize>;
    fn trace_widths(&self) -> Vec<usize>;
    fn current_trace_cells(&self) -> Vec<usize> {
        self.current_trace_heights()
            .iter()
            .zip(self.trace_widths().iter())
            .map(|(height, width)| height * width)
            .collect()
    }
}

pub trait SingleAirMachineChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F>;
    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>;
    fn generate_public_values(&mut self) -> Vec<F> {
        vec![]
    }
    fn current_trace_height(&self) -> usize;
    fn trace_width(&self) -> usize;
    fn current_trace_cells(&self) -> usize {
        self.current_trace_height() * self.trace_width()
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
}

impl<F, C: SingleAirMachineChip<F>> SingleAirMachineChip<F> for Rc<RefCell<C>> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        match Rc::try_unwrap(self) {
            Ok(ref_cell) => ref_cell.into_inner().generate_trace(),
            Err(_) => panic!("cannot generate trace while other chips still hold a reference"),
        }
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        self.borrow().air()
    }

    fn generate_public_values(&mut self) -> Vec<F> {
        self.borrow_mut().generate_public_values()
    }

    fn current_trace_height(&self) -> usize {
        self.borrow().current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.borrow().trace_width()
    }
}

#[derive(Debug)]
#[enum_dispatch(InstructionExecutor<F>)]
pub enum InstructionExecutorVariant<F: PrimeField32> {
    Core(Rc<RefCell<CoreChip<F>>>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ModularArithmetic(Rc<RefCell<ModularArithmeticChip<F, ModularArithmeticAirVariant>>>),
    U256Arithmetic(Rc<RefCell<UintArithmeticChip<256, 8, F>>>),
    U256Multiplication(Rc<RefCell<UintMultiplicationChip<F, 32, 8>>>),
    Shift256(Rc<RefCell<ShiftChip<F, 32, 8>>>),
    Ui(Rc<RefCell<UiChip<F>>>),
    CastF(Rc<RefCell<CastFChip<F>>>),
    Secp256k1AddUnequal(Rc<RefCell<EcAddUnequalChip<F>>>),
    Secp256k1Double(Rc<RefCell<EcDoubleChip<F>>>),
}

type ChipRef<C> = SingleAirChipAdapter<Rc<RefCell<C>>>;

#[derive(IntoStaticStr)]
#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F: PrimeField32> {
    Core(ChipRef<CoreChip<F>>),
    Program(ChipRef<ProgramChip<F>>),
    Memory(ChipRef<MemoryChip<F>>),
    FieldArithmetic(ChipRef<FieldArithmeticChip<F>>),
    FieldExtension(ChipRef<FieldExtensionArithmeticChip<F>>),
    Poseidon2(ChipRef<Poseidon2Chip<F>>),
    Keccak256(ChipRef<KeccakVmChip<F>>),
    U256Arithmetic(ChipRef<UintArithmeticChip<256, 8, F>>),
    U256Multiplication(ChipRef<UintMultiplicationChip<F, 32, 8>>),
    Shift256(ChipRef<ShiftChip<F, 32, 8>>),
    Ui(ChipRef<UiChip<F>>),
    CastF(ChipRef<CastFChip<F>>),
    Secp256k1AddUnequal(ChipRef<EcAddUnequalChip<F>>),
    Secp256k1Double(ChipRef<EcDoubleChip<F>>),

    RangeChecker(Arc<VariableRangeCheckerChip>),
    RangeTupleChecker(Arc<RangeTupleCheckerChip>),
    ByteXor(Arc<XorLookupChip<8>>),
}

pub struct SingleAirChipAdapter<C> {
    inner: C,
}

impl<C> SingleAirChipAdapter<C> {
    pub fn new(inner: C) -> Self {
        Self { inner }
    }
}

impl<F, C: SingleAirMachineChip<F>> MachineChip<F> for SingleAirChipAdapter<C> {
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        vec![self.inner.generate_trace()]
    }

    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        vec![self.inner.air()]
    }

    fn generate_public_values(&mut self) -> Vec<Vec<F>> {
        vec![self.inner.generate_public_values()]
    }

    fn current_trace_heights(&self) -> Vec<usize> {
        vec![self.inner.current_trace_height()]
    }

    fn trace_widths(&self) -> Vec<usize> {
        vec![self.inner.trace_width()]
    }
}

impl<F: PrimeField32> MachineChip<F> for Arc<VariableRangeCheckerChip> {
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        vec![VariableRangeCheckerChip::generate_trace(&self)]
    }

    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        vec![Box::new(self.air)]
    }

    fn current_trace_heights(&self) -> Vec<usize> {
        vec![1 << (1 + self.air.bus.range_max_bits)]
    }

    fn trace_widths(&self) -> Vec<usize> {
        vec![BaseAir::<F>::width(&self.air)]
    }
}

impl<F: PrimeField32> MachineChip<F> for Arc<RangeTupleCheckerChip> {
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        vec![RangeTupleCheckerChip::generate_trace(&self)]
    }

    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        vec![Box::new(self.air.clone())]
    }

    fn current_trace_heights(&self) -> Vec<usize> {
        vec![self.air.height() as usize]
    }

    fn trace_widths(&self) -> Vec<usize> {
        vec![BaseAir::<F>::width(&self.air)]
    }
}

impl<F: PrimeField32, const M: usize> MachineChip<F> for Arc<XorLookupChip<M>> {
    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        vec![XorLookupChip::generate_trace(&self)]
    }

    fn airs<SC: StarkGenericConfig>(&self) -> Vec<Box<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        vec![Box::new(self.air.clone())]
    }

    fn current_trace_heights(&self) -> Vec<usize> {
        vec![1 << (2 * M)]
    }

    fn trace_widths(&self) -> Vec<usize> {
        vec![BaseAir::<F>::width(&self.air)]
    }
}
