use std::{cell::RefCell, collections::HashMap, ops::Deref, rc::Rc, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    interaction::InteractionBuilder, rap::AnyRap, verifier::VerificationError,
};
use afs_test_utils::{
    config::baby_bear_poseidon2::{run_simple_test, BabyBearPoseidon2Config},
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
};
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::{AbstractField, Field, PrimeField32, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig};
use rand::{rngs::StdRng, RngCore};

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::{MachineChip, OpCodeExecutor},
        columns::{ExecutionState, InstructionCols},
    },
    cpu::trace::Instruction,
    memory::{offline_checker::MemoryChip, MemoryAccess, OpType::Write},
};

pub struct MemoryTester<F: PrimeField32> {
    memory_bus: usize,
    memory: HashMap<(F, F), F>,
    memory_chip: Rc<RefCell<MemoryChip<1, F>>>,
    accesses: Vec<MemoryAccess<1, F>>,
}

impl<F: PrimeField32> MemoryTester<F> {
    pub fn new(memory_bus: usize) -> Self {
        Self {
            memory_bus,
            memory: HashMap::new(),
            memory_chip: Rc::new(RefCell::new(MemoryChip::new(
                30,
                30,
                30,
                16,
                HashMap::new(),
                Arc::new(RangeCheckerGateChip::new(0, 1)),
            ))),
            accesses: vec![],
        }
    }
    fn set<const N: usize>(
        memory: &mut HashMap<(F, F), F>,
        address_space: usize,
        start_address: usize,
        cells: [F; N],
    ) {
        for (i, &cell) in cells.iter().enumerate() {
            memory.insert(
                (
                    F::from_canonical_usize(address_space),
                    F::from_canonical_usize(start_address + i),
                ),
                cell,
            );
        }
    }

    pub fn install<const N: usize>(
        &mut self,
        address_space: usize,
        start_address: usize,
        cells: [F; N],
    ) {
        self.expect(address_space, start_address, cells);
        Self::set(
            &mut self.memory_chip.borrow_mut().memory,
            address_space,
            start_address,
            cells,
        );
    }

    pub fn expect<const N: usize>(
        &mut self,
        address_space: usize,
        start_address: usize,
        cells: [F; N],
    ) {
        Self::set(&mut self.memory, address_space, start_address, cells);
    }

    pub fn get_memory_chip(&self) -> Rc<RefCell<MemoryChip<1, F>>> {
        self.memory_chip.clone()
    }

    pub fn check(&mut self) {
        assert_eq!(self.memory_chip.borrow().memory, self.memory);
        self.accesses
            .append(&mut self.memory_chip.borrow_mut().accesses);
        self.memory_chip.borrow_mut().last_timestamp = None;
        self.memory_chip.borrow_mut().memory.clear();
        self.memory.clear();
    }

    fn dummy_air(&self) -> DummyInteractionAir {
        DummyInteractionAir::new(5, false, self.memory_bus)
    }
}

impl<F: PrimeField32> MachineChip<F> for MemoryTester<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for &MemoryAccess {
            timestamp,
            op_type,
            address_space,
            address,
            data: [data],
        } in self.accesses.iter()
        {
            rows.push(F::one());
            rows.extend([
                F::from_canonical_usize(timestamp),
                F::from_bool(op_type == Write),
                address_space,
                address,
                data,
            ]);
        }

        while !(rows.len() / self.width()).is_power_of_two() {
            rows.push(F::zero());
        }

        RowMajorMatrix::new(rows, self.width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.dummy_air())
    }

    fn current_trace_height(&self) -> usize {
        self.accesses.len()
    }

    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.dummy_air())
    }
}

#[derive(Clone)]
struct Execution {
    initial_state: ExecutionState<usize>,
    final_state: ExecutionState<usize>,
    instruction: InstructionCols<usize>,
}

#[derive(Clone)]
pub struct ExecutionTester {
    execution_bus: ExecutionBus,
    rng: StdRng,
    executions: Vec<Execution>,
}

impl ExecutionTester {
    pub fn new(execution_bus: ExecutionBus, rng: StdRng) -> Self {
        Self {
            execution_bus,
            rng,
            executions: vec![],
        }
    }
    fn next_elem_size_usize<F: Field>(&mut self) -> usize {
        (self.rng.next_u32() % (1 << (F::bits() - 1))) as usize
    }
    pub fn execute<F: PrimeField64, E: OpCodeExecutor<F>>(
        &mut self,
        executor: &mut E,
        instruction: Instruction<F>,
    ) {
        let initial_state = ExecutionState {
            pc: self.next_elem_size_usize::<F>(),
            timestamp: self.next_elem_size_usize::<F>(),
        };
        let final_state = executor.execute(&instruction, initial_state);
        self.executions.push(Execution {
            initial_state,
            final_state,
            instruction: InstructionCols::from_instruction(&instruction)
                .map(|elem| elem.as_canonical_u64() as usize),
        })
    }
    fn test_execution_with_expected_changes<F: PrimeField64, E: OpCodeExecutor<F>>(
        &mut self,
        executor: &mut E,
        instruction: Instruction<F>,
        expected_pc_change: usize,
        expected_timestamp_change: usize,
    ) {
        let initial_state = ExecutionState {
            pc: self.next_elem_size_usize::<F>(),
            timestamp: self.next_elem_size_usize::<F>(),
        };
        let final_state = ExecutionState {
            pc: initial_state.pc + expected_pc_change,
            timestamp: initial_state.timestamp + expected_timestamp_change,
        };
        assert_eq!(executor.execute(&instruction, initial_state), final_state);
        self.executions.push(Execution {
            initial_state,
            final_state,
            instruction: InstructionCols::from_instruction(&instruction)
                .map(|elem| elem.as_canonical_u64() as usize),
        });
    }
}

impl<F: Field> BaseAir<F> for ExecutionTester {
    fn width(&self) -> usize {
        (2 * ExecutionState::<usize>::get_width()) + InstructionCols::<usize>::get_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ExecutionTester {
    fn eval(&self, builder: &mut AB) {
        for Execution {
            initial_state,
            final_state,
            instruction,
        } in self.executions.iter()
        {
            let initial_state = initial_state
                .map(AB::F::from_canonical_usize)
                .map(Into::into);
            let final_state = final_state.map(AB::F::from_canonical_usize).map(Into::into);
            let instruction_cols = instruction.map(AB::F::from_canonical_usize).map(Into::into);
            self.execution_bus.execute(
                builder,
                AB::Expr::neg_one(),
                initial_state,
                final_state,
                instruction_cols,
            );
        }
    }
}

impl<F: Field> MachineChip<F> for ExecutionTester {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(vec![F::zero()], 1)
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.clone())
    }

    fn current_trace_height(&self) -> usize {
        1
    }

    fn width(&self) -> usize {
        1
    }
}

#[derive(Default)]
pub struct MachineChipTester {
    airs: Vec<Box<dyn AnyRap<BabyBearPoseidon2Config>>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    public_values: Vec<Vec<BabyBear>>,
}

impl MachineChipTester {
    pub fn add<C: MachineChip<BabyBear>>(&mut self, chip: &mut C) -> &mut Self {
        self.traces.push(chip.generate_trace());
        self.public_values.push(chip.get_public_values());
        self.airs.push(chip.air());
        self
    }
    pub fn add_with_custom_trace<C: MachineChip<BabyBear>>(
        &mut self,
        chip: &mut C,
        trace: RowMajorMatrix<BabyBear>,
    ) -> &mut Self {
        self.traces.push(trace);
        self.public_values.push(chip.get_public_values());
        self.airs.push(chip.air());
        self
    }
    pub fn simple_test(&mut self) -> Result<(), VerificationError> {
        run_simple_test(
            self.airs.iter().map(|x| x.deref()).collect(),
            self.traces.clone(),
            self.public_values.clone(),
        )
    }
    fn max_trace_height(&self) -> usize {
        self.traces
            .iter()
            .map(RowMajorMatrix::height)
            .max()
            .unwrap()
    }
    /// Given a function to produce an engine from the max trace height,
    /// runs a simple test on that engine
    pub fn engine_test<E: StarkEngine<BabyBearPoseidon2Config>, P: Fn(usize) -> E>(
        &mut self,
        engine_provider: P,
    ) -> Result<(), VerificationError> {
        engine_provider(self.max_trace_height()).run_simple_test(
            self.airs.iter().map(|x| x.deref()).collect(),
            self.traces.clone(),
            self.public_values.clone(),
        )
    }
}
