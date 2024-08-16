use std::{cell::RefCell, collections::HashMap, rc::Rc};

use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::StarkGenericConfig;
use rand::{RngCore, rngs::StdRng};

use afs_stark_backend::{
    interaction::InteractionBuilder, rap::AnyRap, verifier::VerificationError,
};
use afs_test_utils::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, run_simple_test},
    interaction::dummy_interaction_air::DummyInteractionAir,
};

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::{MachineChip, OpCodeExecutor},
        columns::{ExecutionState, InstructionCols},
    },
    cpu::trace::Instruction,
    memory::{MemoryAccess, offline_checker::MemoryChip, OpType::Write},
};

pub struct MemoryTester<F: PrimeField32> {
    dummy_interaction_air: DummyInteractionAir,
    memory: HashMap<(F, F), F>,
    memory_chip: Rc<RefCell<MemoryChip<1, F>>>,
    accesses: Vec<MemoryAccess<1, F>>,
}

impl<F: PrimeField32> MemoryTester<F> {
    pub fn new(memory_bus: usize) -> Self {
        Self {
            dummy_interaction_air: DummyInteractionAir::new(5, false, memory_bus),
            memory: HashMap::new(),
            memory_chip: Rc::new(RefCell::new(MemoryChip::new(
                30,
                30,
                30,
                16,
                HashMap::new(),
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

        let width = BaseAir::<F>::width(&self.dummy_interaction_air);
        while !(rows.len() / width).is_power_of_two() {
            rows.push(F::zero());
        }

        RowMajorMatrix::new(rows, width)
    }

    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC> {
        &self.dummy_interaction_air
    }
}

struct Execution {
    initial_state: ExecutionState<usize>,
    final_state: ExecutionState<usize>,
    instruction: InstructionCols<usize>,
}

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
            self.execution_bus.interact_execute_with_multiplicity(
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
    fn air<SC: StarkGenericConfig>(&self) -> &dyn AnyRap<SC> {
        self
    }
}

#[derive(Default)]
pub struct MachineChipTester<'a> {
    airs: Vec<&'a dyn AnyRap<BabyBearPoseidon2Config>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    public_values: Vec<Vec<BabyBear>>,
}

impl<'a> MachineChipTester<'a> {
    pub fn add<C: MachineChip<BabyBear>>(&mut self, chip: &'a mut C) -> &mut Self {
        self.traces.push(chip.generate_trace());
        self.public_values.push(chip.get_public_values());
        self.airs.push(chip.air());
        self
    }
    pub fn add_with_custom_trace<C: MachineChip<BabyBear>>(
        &mut self,
        chip: &'a mut C,
        trace: RowMajorMatrix<BabyBear>,
    ) -> &mut Self {
        self.traces.push(trace);
        self.public_values.push(chip.get_public_values());
        self.airs.push(chip.air());
        self
    }
    pub fn simple_test(&mut self) -> Result<(), VerificationError> {
        run_simple_test(
            self.airs.clone(),
            self.traces.clone(),
            self.public_values.clone(),
        )
    }
}
