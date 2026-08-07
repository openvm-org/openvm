mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod execution;
pub mod memory;
pub mod program;
mod utils;

pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
pub use execution::ExecutionTester;
pub use memory::MemoryTester;
#[cfg(feature = "tco")]
use openvm_instructions::exe::VmExe;
use openvm_instructions::{instruction::Instruction, program::Program};
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_air::BaseAir,
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
};
pub use utils::*;

#[cfg(not(feature = "tco"))]
use crate::arch::{
    execution_mode::PreflightCtx, interpreter::AlignedBuf, ExecutionCtxTrait, VmExecState,
};
#[cfg(feature = "tco")]
use crate::arch::{ExecutorInventory, InterpretedInstance, SystemConfig};
use crate::{
    arch::{
        ExecutionState, Executor, MemoryConfig, Postflight, PostflightError, PreflightHistory,
        PreflightOutput, Streams, VmField, VmState,
    },
    system::memory::online::GuestMemory,
};

pub const EXECUTION_BUS: BusIndex = 0;
pub const MEMORY_BUS: BusIndex = 1;
pub const POSEIDON2_DIRECT_BUS: BusIndex = 6;
pub const READ_INSTRUCTION_BUS: BusIndex = 8;
pub const BITWISE_OP_LOOKUP_BUS: BusIndex = 9;
pub const BYTE_XOR_BUS: BusIndex = 10;
pub const RANGE_TUPLE_CHECKER_BUS: BusIndex = 11;
pub const MEMORY_MERKLE_BUS: BusIndex = 12;

pub const RANGE_CHECKER_BUS: BusIndex = 4;

#[derive(Clone)]
pub struct TestPreflightExecution<F> {
    pub program: Program<F>,
    pub history: PreflightHistory,
}

#[derive(Clone, Default)]
pub struct TestPreflight<F> {
    pub executions: Vec<TestPreflightExecution<F>>,
}

type TestTraceGenerator<F, C> =
    Box<dyn for<'a> Fn(&C, &Postflight<'a, F>) -> Result<RowMajorMatrix<F>, PostflightError>>;
type TestBatchTraceGenerator<F, C> =
    Box<dyn for<'a> Fn(&C, &[Postflight<'a, F>]) -> Result<RowMajorMatrix<F>, PostflightError>>;
type TestTracePadding<F> = Box<dyn Fn(&mut [F])>;
type TestTraceRows<F> = Box<dyn Fn(&RowMajorMatrix<F>) -> usize>;

pub struct TestChipHarness<F, E, A, C> {
    pub executor: E,
    pub air: A,
    pub chip: C,
    pub preflight: TestPreflight<F>,
    pub generate_trace: TestTraceGenerator<F, C>,
    pub generate_batch_trace: Option<TestBatchTraceGenerator<F, C>>,
    pub rows_used: TestTraceRows<F>,
    pub fill_padding: TestTracePadding<F>,
    pub balance_memory: bool,
}

#[cfg(not(feature = "tco"))]
pub(crate) fn execute_test_preflight<F, E>(
    _memory_config: &MemoryConfig,
    executor: &E,
    program: &Program<F>,
    state: VmState<GuestMemory>,
) -> PreflightOutput
where
    F: VmField,
    E: Executor<F>,
{
    let instruction = &program
        .get_instruction_and_debug_info(0)
        .expect("test program must contain a starting instruction")
        .0;

    let pre_compute_size = executor.pre_compute_size().next_power_of_two();
    let pre_compute_buf = AlignedBuf::uninit(pre_compute_size, pre_compute_size);
    let pre_compute = unsafe {
        // SAFETY: `pre_compute_buf` owns exactly `pre_compute_size` writable bytes and outlives the
        // handler invocation below.
        std::slice::from_raw_parts_mut(pre_compute_buf.ptr, pre_compute_size)
    };
    let handler = executor
        .pre_compute::<PreflightCtx>(state.pc(), instruction, pre_compute)
        .expect("test instruction must be statically valid");
    let ctx = PreflightCtx::new::<F>(&state.memory, Some(1));
    let mut exec_state = VmExecState::new(state, ctx);
    assert!(!PreflightCtx::should_suspend(&mut exec_state));
    let pc = exec_state.pc();
    PreflightCtx::on_instruction_start(&mut exec_state, pc);
    unsafe {
        // SAFETY: `handler` and its aligned pre-compute data were produced by this executor for
        // `instruction`, and `pre_compute` outlives the call.
        handler(pre_compute_buf.ptr, &mut exec_state);
    }

    let exit_code = exec_state
        .exit_code
        .expect("test instruction preflight must succeed");
    let pc = exec_state.vm_state.pc();
    let history = exec_state.ctx.finish(pc);
    PreflightOutput {
        history,
        state: exec_state.vm_state,
        exit_code,
    }
}

#[cfg(feature = "tco")]
pub(crate) fn execute_test_preflight<F, E>(
    memory_config: &MemoryConfig,
    executor: &E,
    program: &Program<F>,
    state: VmState<GuestMemory>,
) -> PreflightOutput
where
    F: VmField,
    E: Executor<F> + Clone,
{
    let instruction = &program
        .get_instruction_and_debug_info(0)
        .expect("test program must contain a starting instruction")
        .0;
    let exe = VmExe::new(program.clone()).with_pc_start(program.pc_base);
    let system_config = SystemConfig::default_from_memory(memory_config.clone());
    let mut inventory = ExecutorInventory::<E>::new(system_config);
    inventory
        .add_executor(executor.clone(), [instruction.opcode])
        .expect("test executor opcode must be unique");
    let interpreter = InterpretedInstance::new(&inventory, &exe)
        .expect("test instruction must be statically valid");
    interpreter
        .execute_preflight_from_state::<F>(state, Some(1))
        .expect("test instruction preflight must succeed")
}

impl<F, E, A, C> TestChipHarness<F, E, A, C>
where
    F: PrimeField32,
    A: BaseAir<F>,
{
    pub fn with_capacity<G>(executor: E, air: A, chip: C, height: usize, generate_trace: G) -> Self
    where
        G: for<'a> Fn(&C, &Postflight<'a, F>) -> Result<RowMajorMatrix<F>, PostflightError>
            + 'static,
    {
        Self {
            executor,
            air,
            chip,
            preflight: TestPreflight {
                executions: Vec::with_capacity(height),
            },
            generate_trace: Box::new(generate_trace),
            generate_batch_trace: None,
            rows_used: Box::new(|trace| trace.height()),
            fill_padding: Box::new(|_| {}),
            balance_memory: true,
        }
    }

    pub fn with_batch_trace_generator(
        mut self,
        generate_trace: impl for<'a> Fn(&C, &[Postflight<'a, F>]) -> Result<RowMajorMatrix<F>, PostflightError>
            + 'static,
    ) -> Self {
        self.generate_batch_trace = Some(Box::new(generate_trace));
        self
    }

    pub fn with_rows_used(
        mut self,
        rows_used: impl Fn(&RowMajorMatrix<F>) -> usize + 'static,
    ) -> Self {
        self.rows_used = Box::new(rows_used);
        self
    }

    pub fn with_padding(mut self, fill_padding: impl Fn(&mut [F]) + 'static) -> Self {
        self.fill_padding = Box::new(fill_padding);
        self
    }

    pub fn without_memory_balance(mut self) -> Self {
        self.balance_memory = false;
        self
    }
}

pub trait TestBuilder<F: PrimeField32> {
    fn execute<E: Executor<F> + Clone>(
        &mut self,
        executor: &mut E,
        preflight: &mut TestPreflight<F>,
        instruction: &Instruction<F>,
    );

    fn execute_with_pc<E: Executor<F> + Clone>(
        &mut self,
        executor: &mut E,
        preflight: &mut TestPreflight<F>,
        instruction: &Instruction<F>,
        initial_pc: u32,
    );

    fn write<const N: usize>(&mut self, address_space: usize, pointer: usize, value: [F; N]);
    fn read<const N: usize>(&mut self, address_space: usize, pointer: usize) -> [F; N];
    fn write_bytes<const N: usize>(&mut self, address_space: usize, byte_ptr: usize, value: [F; N]);
    fn read_bytes<const N: usize>(&mut self, address_space: usize, byte_ptr: usize) -> [F; N];

    fn write_usize<const N: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        value: [usize; N],
    );

    /// Bit width for RISC-V byte pointers used by extension tests and adapter configs.
    fn address_bits(&self) -> usize;

    /// Byte pc of the last execution's final state (byte pcs do not fit in a field element).
    fn last_to_pc(&self) -> u32;
    /// Byte pc of the last execution's initial state.
    fn last_from_pc(&self) -> u32;

    /// Byte-pc state of the last execution's final state.
    fn execution_final_state(&self) -> ExecutionState<u32>;
    fn streams_mut(&mut self) -> &mut Streams;

    fn get_default_register(&mut self, increment: usize) -> usize;
    fn get_default_pointer(&mut self, increment: usize) -> usize;

    fn get_default_registers<const N: usize>(&mut self, increment: usize) -> [usize; N] {
        let start = self.get_default_register(N * increment);
        std::array::from_fn(|index| start + index * increment)
    }

    fn write_heap_pointer_default(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
    ) -> (usize, usize);

    fn write_heap_default<const NUM_LIMBS: usize>(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
        writes: Vec<[F; NUM_LIMBS]>,
    ) -> (usize, usize);
}
