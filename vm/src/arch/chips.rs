use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{var_range::VariableRangeCheckerChip, xor::lookup::XorLookupChip};
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
    cpu::{trace::Instruction, CpuChip},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    memory::MemoryChipRef,
    modular_arithmetic::ModularArithmeticChip as NewModularArithmeticChip,
    modular_multiplication::ModularArithmeticChip,
    program::ProgramChip,
    uint_arithmetic::UintArithmeticChip,
};

#[enum_dispatch]
pub trait InstructionExecutor<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize>;
}

#[enum_dispatch]
pub trait MachineChip<F> {
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
    ) -> ExecutionState<usize> {
        self.borrow_mut().execute(instruction, prev_state)
    }
}

impl<F, C: MachineChip<F>> MachineChip<F> for Rc<RefCell<C>> {
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
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ModularArithmetic(Rc<RefCell<ModularArithmeticChip<F>>>),
    // FIXME: remove above and rename (no New)
    NewModularArithmetic(Rc<RefCell<NewModularArithmeticChip<F>>>),
    U256Arithmetic(Rc<RefCell<UintArithmeticChip<256, 8, F>>>),
}

#[derive(Debug, IntoStaticStr)]
#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F: PrimeField32> {
    Cpu(Rc<RefCell<CpuChip<F>>>),
    Program(Rc<RefCell<ProgramChip<F>>>),
    Memory(MemoryChipRef<F>),
    FieldArithmetic(Rc<RefCell<FieldArithmeticChip<F>>>),
    FieldExtension(Rc<RefCell<FieldExtensionArithmeticChip<F>>>),
    Poseidon2(Rc<RefCell<Poseidon2Chip<F>>>),
    RangeChecker(Arc<VariableRangeCheckerChip>),
    Keccak256(Rc<RefCell<KeccakVmChip<F>>>),
    ByteXor(Arc<XorLookupChip<8>>),
    U256Arithmetic(Rc<RefCell<UintArithmeticChip<256, 8, F>>>),
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

    fn current_trace_height(&self) -> usize {
        1 << (1 + self.air.bus.range_max_bits)
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

    fn current_trace_height(&self) -> usize {
        1 << (2 * M)
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
