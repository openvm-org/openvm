use std::{cell::RefCell, collections::HashMap, ops::Deref, rc::Rc, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{
    interaction::InteractionBuilder, rap::AnyRap, verifier::VerificationError,
};
use afs_test_utils::{
    config::baby_bear_poseidon2::{run_simple_test, BabyBearPoseidon2Config},
    engine::StarkEngine,
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
        chips::{InstructionExecutor, MachineChip},
        columns::{ExecutionState, InstructionCols},
    },
    cpu::{trace::Instruction, RANGE_CHECKER_BUS},
    memory::{manager::MemoryManager, offline_checker::bus::MemoryBus},
    vm::config::MemoryConfig,
};

pub struct MemoryTester<F: PrimeField32> {
    memory_bus: MemoryBus,
    memory: HashMap<(F, F), F>,
    range_checker: Arc<RangeCheckerGateChip>,
    manager: Rc<RefCell<MemoryManager<F>>>,
    trace_rows: Vec<F>,
}

impl<F: PrimeField32> MemoryTester<F> {
    fn memory_config() -> MemoryConfig {
        MemoryConfig {
            decomp: 8, // speed
            ..Default::default()
        }
    }
    pub fn new(memory_bus: MemoryBus) -> Self {
        let range_checker = Arc::new(RangeCheckerGateChip::new(
            RANGE_CHECKER_BUS,
            1 << Self::memory_config().decomp,
        ));
        Self {
            memory_bus,
            memory: HashMap::new(),
            range_checker: range_checker.clone(),
            manager: Rc::new(RefCell::new(Self::make_manager(memory_bus, range_checker))),
            trace_rows: vec![],
        }
    }

    fn make_manager(
        memory_bus: MemoryBus,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> MemoryManager<F> {
        MemoryManager::new_for_testing(memory_bus, Self::memory_config(), range_checker)
    }

    pub fn install<const N: usize>(
        &mut self,
        address_space: usize,
        start_address: usize,
        cells: [F; N],
    ) {
        self.expect(address_space, start_address, cells);
        for (i, &cell) in cells.iter().enumerate() {
            self.manager.borrow_mut().unsafe_write_word(
                F::from_canonical_usize(address_space),
                F::from_canonical_usize(start_address + i),
                [cell],
            );
        }
    }

    pub fn expect<const N: usize>(
        &mut self,
        address_space: usize,
        start_address: usize,
        cells: [F; N],
    ) {
        for (i, &cell) in cells.iter().enumerate() {
            self.memory.insert(
                (
                    F::from_canonical_usize(address_space),
                    F::from_canonical_usize(start_address + i),
                ),
                cell,
            );
        }
    }

    pub fn get_memory_manager(&self) -> Rc<RefCell<MemoryManager<F>>> {
        self.manager.clone()
    }

    pub fn check(&mut self) {
        let map_memory: HashMap<(F, F), F> = self
            .manager
            .borrow()
            .memory
            .iter()
            .map(|(key, value)| (*key, value.data[0]))
            .collect();

        assert_eq!(map_memory, self.memory);
        let height = self.manager.borrow().interface_chip.current_height();
        self.trace_rows.extend(
            self.manager
                .borrow_mut()
                .generate_memory_interface_trace_with_height(height)
                .values,
        );
        *self.manager.borrow_mut() =
            Self::make_manager(self.memory_bus, self.range_checker.clone());
        self.memory.clear();
    }
}

impl<F: PrimeField32> MachineChip<F> for MemoryTester<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let curr_height = self.trace_rows.len() / self.trace_width();
        let desired_height = curr_height.next_power_of_two();
        self.trace_rows.extend(
            self.manager
                .borrow_mut()
                .generate_memory_interface_trace_with_height(desired_height - curr_height)
                .values,
        );
        RowMajorMatrix::new(self.trace_rows.clone(), self.trace_width())
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.manager.borrow().get_audit_air())
    }

    fn current_trace_height(&self) -> usize {
        self.current_trace_cells() / self.trace_width()
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.manager.borrow().get_audit_air())
    }

    fn current_trace_cells(&self) -> usize {
        self.trace_rows.len()
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

    fn next_timestamp(&mut self) -> usize {
        (self.rng.next_u32() % (1 << MemoryConfig::default().clk_max_bits)) as usize
    }
    pub fn execute<F: PrimeField32, E: InstructionExecutor<F>>(
        &mut self,
        memory_tester: &mut MemoryTester<F>, // should merge MemoryTester and ExecutionTester into one struct (MachineChipTestBuilder?)
        executor: &mut E,
        instruction: Instruction<F>,
    ) {
        let initial_state = ExecutionState {
            pc: self.next_elem_size_usize::<F>(),
            timestamp: self.next_timestamp(),
        };
        println!("timestamp = {}", initial_state.timestamp);
        memory_tester.manager.borrow_mut().timestamp =
            F::from_canonical_usize(initial_state.timestamp);
        let final_state = executor.execute(&instruction, initial_state);
        self.executions.push(Execution {
            initial_state,
            final_state,
            instruction: InstructionCols::from_instruction(&instruction)
                .map(|elem| elem.as_canonical_u64() as usize),
        })
    }
    // for use by CoreChip, needs to be modified to setup memorytester (or just merge them before writing CoreChip)
    /*fn test_execution_with_expected_changes<F: PrimeField64, E: InstructionExecutor<F>>(
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
    }*/
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

    fn trace_width(&self) -> usize {
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
        self.public_values.push(chip.generate_public_values());
        self.airs.push(chip.air());

        let trace_cells = self.traces.last().unwrap().values.len();
        let width = self.airs.last().unwrap().width();
        println!(
            "trace cells = {}, width = {}, height = {}",
            trace_cells,
            width,
            trace_cells / width
        );
        self
    }
    pub fn add_memory(&mut self, memory_tester: &mut MemoryTester<BabyBear>) -> &mut Self {
        self.add(memory_tester);
        self.add(&mut memory_tester.range_checker);
        self
    }
    pub fn add_with_custom_trace<C: MachineChip<BabyBear>>(
        &mut self,
        chip: &mut C,
        trace: RowMajorMatrix<BabyBear>,
    ) -> &mut Self {
        self.traces.push(trace);
        self.public_values.push(chip.generate_public_values());
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
