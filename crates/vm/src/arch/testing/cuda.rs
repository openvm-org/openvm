use std::sync::Arc;

use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        BitwiseOperationLookupChipGPU, SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        RangeTupleCheckerChipGPU, SharedRangeTupleCheckerChip,
    },
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerAir, VariableRangeCheckerBus,
        VariableRangeCheckerChip, VariableRangeCheckerChipGPU,
    },
    Chip,
};
#[cfg(feature = "rvr")]
use openvm_cuda_backend::{
    base::DeviceMatrix, data_transporter::assert_eq_host_and_device_matrix_col_maj,
};
use openvm_cuda_backend::{
    prelude::{EF, F, SC},
    BabyBearPoseidon2GpuEngine, GpuBackend, ProverError,
};
#[cfg(feature = "rvr")]
use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
use openvm_cuda_common::{
    common::get_device,
    stream::{CudaStream, GpuDeviceCtx, StreamGuard},
};
use openvm_instructions::{
    instruction::Instruction,
    program::{Program, PC_BITS},
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
};
#[cfg(feature = "rvr")]
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir};
use openvm_stark_backend::{
    interaction::{LookupBus, PermutationCheckBus},
    p3_field::PrimeCharacteristicRing,
    prover::AirProvingContext,
    AirRef, AnyAir, StarkEngine, VerificationData,
};
#[cfg(feature = "rvr")]
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{ColMajorMatrix, MatrixDimensions as DeviceMatrixDimensions},
    Val,
};
use openvm_stark_sdk::utils::setup_tracing_with_log_level;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::Level;

#[cfg(feature = "rvr")]
use crate::arch::cuda::postflight::{
    GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
};
#[cfg(feature = "rvr")]
use crate::arch::PreflightHistory;
#[cfg(feature = "touchemall")]
use crate::primitives::utils::check_trace_validity;
#[cfg(feature = "rvr")]
use crate::utils::next_power_of_two_or_zero;
use crate::{
    arch::{
        testing::{
            default_tracing_memory, default_var_range_checker_bus, dummy_memory_helper,
            execute_test_preflight,
            execution::{air::ExecutionDummyAir, DeviceExecutionTester},
            memory::DeviceMemoryTester,
            program::{air::ProgramDummyAir, DeviceProgramTester},
            TestBuilder, TestChipHarness, TestPreflight, TestPreflightExecution, EXECUTION_BUS,
            MEMORY_BUS, MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS, PUBLIC_VALUES_BUS,
            READ_INSTRUCTION_BUS,
        },
        to_byte_ptr_bits, ExecutionBridge, ExecutionBus, ExecutionState, Executor, MemoryConfig,
        Postflight, PublicValuesState, Streams, VmState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
        NUM_REGISTERS,
    },
    system::{
        cuda::poseidon2::Poseidon2PeripheryChipGPU,
        memory::{
            offline_checker::{MemoryBridge, MemoryBus},
            online::{AddressMap, GuestMemory},
            MemoryAirInventory, SharedMemoryHelper,
        },
        poseidon2::air::Poseidon2PeripheryAir,
        program::ProgramBus,
        public_values::PublicValuesBus,
        SystemPort,
    },
    utils::test_gpu_engine,
};

#[cfg(feature = "rvr")]
type CpuPostflightTraceGenerator<F, C> = Box<
    dyn for<'a> Fn(
        &C,
        &Postflight<'a, F>,
    ) -> Result<RowMajorMatrix<F>, crate::arch::PostflightError>,
>;
#[cfg(feature = "rvr")]
type CpuPostflightBatchTraceGenerator<F, C> = Box<
    dyn for<'a> Fn(
        &C,
        &[Postflight<'a, F>],
    ) -> Result<RowMajorMatrix<F>, crate::arch::PostflightError>,
>;
#[cfg(feature = "rvr")]
type GpuPostflightTraceGenerator<G> = Box<
    dyn Fn(
        &G,
        &GpuPostflightProgram,
        &GpuPostflightTranscript,
        &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError>,
>;
#[cfg(feature = "rvr")]
type RowsUsed<F> = Box<dyn Fn(&RowMajorMatrix<F>) -> usize>;
#[cfg(feature = "rvr")]
type FillPadding<F> = Box<dyn Fn(&mut [F])>;

#[cfg(feature = "rvr")]
pub struct GpuTestChipHarness<F, Executor, AIR, GpuChip, CpuChip> {
    pub executor: Executor,
    pub air: AIR,
    pub gpu_chip: GpuChip,
    pub cpu_chip: CpuChip,
    pub preflight: TestPreflight<F>,
    generate_cpu_trace: Option<CpuPostflightTraceGenerator<F, CpuChip>>,
    generate_cpu_batch_trace: Option<CpuPostflightBatchTraceGenerator<F, CpuChip>>,
    generate_gpu_trace: Option<GpuPostflightTraceGenerator<GpuChip>>,
    rows_used: RowsUsed<F>,
    fill_padding: FillPadding<F>,
    balance_memory: bool,
}

#[cfg(feature = "rvr")]
impl<F, Executor, AIR, GpuChip, CpuChip> GpuTestChipHarness<F, Executor, AIR, GpuChip, CpuChip>
where
    F: PrimeField32,
    AIR: BaseAir<F>,
{
    pub fn with_capacity(
        executor: Executor,
        air: AIR,
        gpu_chip: GpuChip,
        cpu_chip: CpuChip,
        height: usize,
    ) -> Self {
        Self {
            executor,
            air,
            gpu_chip,
            cpu_chip,
            preflight: TestPreflight {
                executions: Vec::with_capacity(height),
            },
            generate_cpu_trace: None,
            generate_cpu_batch_trace: None,
            generate_gpu_trace: None,
            rows_used: Box::new(Matrix::height),
            fill_padding: Box::new(|_| {}),
            balance_memory: true,
        }
    }

    pub fn with_trace_generators<CpuGenerate, GpuGenerate>(
        mut self,
        generate_cpu_trace: CpuGenerate,
        generate_gpu_trace: GpuGenerate,
    ) -> Self
    where
        CpuGenerate: for<'a> Fn(
                &CpuChip,
                &Postflight<'a, F>,
            ) -> Result<RowMajorMatrix<F>, crate::arch::PostflightError>
            + 'static,
        GpuGenerate: Fn(
                &GpuChip,
                &GpuPostflightProgram,
                &GpuPostflightTranscript,
                &GpuPostflightPlan,
            ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError>
            + 'static,
    {
        self.generate_cpu_trace = Some(Box::new(generate_cpu_trace));
        self.generate_gpu_trace = Some(Box::new(generate_gpu_trace));
        self
    }

    pub fn with_batch_trace_generator(
        mut self,
        generate_trace: impl for<'a> Fn(
                &CpuChip,
                &[Postflight<'a, F>],
            ) -> Result<RowMajorMatrix<F>, crate::arch::PostflightError>
            + 'static,
    ) -> Self {
        self.generate_cpu_batch_trace = Some(Box::new(generate_trace));
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

impl TestBuilder<F> for GpuChipTestBuilder {
    fn execute<E>(
        &mut self,
        executor: &mut E,
        preflight: &mut TestPreflight<F>,
        instruction: &Instruction<F>,
    ) where
        E: Executor<F> + Clone,
    {
        let initial_pc = self.rng.random_range(0..(1 << PC_BITS));
        self.execute_with_pc(executor, preflight, instruction, initial_pc);
    }

    fn execute_with_pc<E>(
        &mut self,
        executor: &mut E,
        preflight: &mut TestPreflight<F>,
        instruction: &Instruction<F>,
        initial_pc: u32,
    ) where
        E: Executor<F> + Clone,
    {
        let program =
            Program::new_without_debug_infos(std::slice::from_ref(instruction), initial_pc);
        let empty_memory = GuestMemory::new(AddressMap::from_mem_config(&self.memory.config));
        let memory = std::mem::replace(&mut self.memory.memory.data, empty_memory);
        let mut state = VmState::new_with_public_values(
            initial_pc,
            memory,
            self.public_values.clone(),
            self.streams.clone(),
            0,
        );
        state.rng = self.rng.clone();
        let output = execute_test_preflight(&self.memory.config, executor, &program, state);
        let initial_state = ExecutionState::new(initial_pc, 1u32);
        let final_event = *output
            .history
            .program
            .last()
            .expect("preflight always emits a final sentinel");
        let final_state = ExecutionState::new(final_event.pc, final_event.timestamp);

        self.memory.memory.data = output.state.memory;
        self.public_values = output.state.public_values;
        self.streams = output.state.streams;
        self.rng = output.state.rng;
        let postflight = Postflight::new_for_test(&program, &output.history, &self.memory.config);
        postflight
            .expect("test preflight history must be valid")
            .record_test_writes(&mut self.memory);
        self.program.execute(instruction, &initial_state);
        self.execution.execute(initial_state, final_state);
        preflight.executions.push(TestPreflightExecution {
            program,
            history: output.history,
        });
    }

    fn read<const N: usize>(&mut self, address_space: usize, pointer: usize) -> [F; N] {
        const { assert!(N == BLOCK_FE_WIDTH) };
        let data = self.memory.read::<BLOCK_FE_WIDTH>(address_space, pointer);
        std::array::from_fn(|i| data[i])
    }

    fn write<const N: usize>(&mut self, address_space: usize, pointer: usize, value: [F; N]) {
        const { assert!(N == BLOCK_FE_WIDTH) };
        self.memory.write::<BLOCK_FE_WIDTH>(
            address_space,
            pointer,
            std::array::from_fn(|i| value[i]),
        );
    }

    fn read_bytes<const N: usize>(&mut self, address_space: usize, byte_ptr: usize) -> [F; N] {
        self.memory.read_bytes(address_space, byte_ptr)
    }

    fn write_bytes<const N: usize>(
        &mut self,
        address_space: usize,
        byte_ptr: usize,
        value: [F; N],
    ) {
        self.memory.write_bytes(address_space, byte_ptr, value);
    }

    fn write_usize<const N: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        value: [usize; N],
    ) {
        self.write(address_space, pointer, value.map(F::from_usize));
    }

    fn address_bits(&self) -> usize {
        to_byte_ptr_bits(self.memory.config.pointer_max_bits)
    }

    fn last_to_pc(&self) -> F {
        self.execution.0.last_to_pc()
    }

    fn last_from_pc(&self) -> F {
        self.execution.0.last_from_pc()
    }

    fn execution_final_state(&self) -> ExecutionState<F> {
        self.execution.0.records.last().unwrap().final_state
    }

    fn streams_mut(&mut self) -> &mut Streams {
        &mut self.streams
    }

    fn get_default_register(&mut self, increment: usize) -> usize {
        let register_file_bytes = NUM_REGISTERS * REGISTER_NUM_LIMBS;
        assert!(increment <= register_file_bytes);
        if self.default_register + increment > register_file_bytes {
            self.default_register = 0;
        }
        let register = self.default_register;
        self.default_register += increment;
        register
    }

    fn get_default_pointer(&mut self, increment: usize) -> usize {
        self.default_pointer += increment;
        self.default_pointer - increment
    }

    fn write_heap_pointer_default(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
    ) -> (usize, usize) {
        let register = self.get_default_register(reg_increment);
        let pointer = self.get_default_pointer(pointer_increment);
        self.write_bytes::<MEMORY_BLOCK_BYTES>(
            1,
            register,
            (pointer as u64).to_le_bytes().map(F::from_u8),
        );
        (register, pointer)
    }

    fn write_heap_default<const NUM_LIMBS: usize>(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
        writes: Vec<[F; NUM_LIMBS]>,
    ) -> (usize, usize) {
        let register = self.get_default_register(reg_increment);
        let pointer = self.get_default_pointer(pointer_increment);
        self.write_heap(register, pointer, writes);
        (register, pointer)
    }
}

pub struct GpuChipTestBuilder {
    pub memory: DeviceMemoryTester,
    pub execution: DeviceExecutionTester,
    pub program: DeviceProgramTester,
    pub streams: Streams,
    pub public_values: PublicValuesState,

    var_range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_op_lookup: Option<Arc<BitwiseOperationLookupChipGPU<8>>>,
    range_tuple_checker: Option<Arc<RangeTupleCheckerChipGPU<2>>>,

    rng: StdRng,
    default_register: usize,
    default_pointer: usize,
}

impl Default for GpuChipTestBuilder {
    fn default() -> Self {
        let mut mem_config = MemoryConfig::default();
        // Tests generate register pointers across the full AS-native pointer range.
        mem_config.addr_spaces[REGISTER_AS as usize].num_cells = 1 << mem_config.pointer_max_bits;
        Self::new(mem_config, default_var_range_checker_bus())
    }
}

impl GpuChipTestBuilder {
    pub fn new(mem_config: MemoryConfig, bus: VariableRangeCheckerBus) -> Self {
        setup_tracing_with_log_level(Level::INFO);
        let mem_bus = MemoryBus::new(MEMORY_BUS);
        let device_ctx = GpuDeviceCtx {
            device_id: get_device().unwrap() as u32,
            stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
        };
        let range_checker = Arc::new(VariableRangeCheckerChipGPU::hybrid(
            Arc::new(VariableRangeCheckerChip::new(bus)),
            device_ctx.clone(),
        ));
        Self {
            memory: DeviceMemoryTester::new(
                default_tracing_memory(&mem_config),
                mem_bus,
                mem_config,
                range_checker.clone(),
                device_ctx.clone(),
            ),
            execution: DeviceExecutionTester::new(
                ExecutionBus::new(EXECUTION_BUS),
                device_ctx.clone(),
            ),
            program: DeviceProgramTester::new(ProgramBus::new(READ_INSTRUCTION_BUS), device_ctx),
            streams: Default::default(),
            public_values: PublicValuesState::new(32),
            var_range_checker: range_checker,
            bitwise_op_lookup: None,
            range_tuple_checker: None,
            rng: StdRng::seed_from_u64(0),
            default_register: 0,
            default_pointer: 0,
        }
    }

    pub fn with_bitwise_op_lookup(mut self, bus: BitwiseOperationLookupBus) -> Self {
        let device_ctx = self.var_range_checker.device_ctx.clone();
        self.bitwise_op_lookup = Some(Arc::new(BitwiseOperationLookupChipGPU::hybrid(
            Arc::new(BitwiseOperationLookupChip::new(bus)),
            device_ctx,
        )));
        self
    }

    pub fn with_range_tuple_checker(mut self, bus: RangeTupleCheckerBus<2>) -> Self {
        let device_ctx = self.var_range_checker.device_ctx.clone();
        self.range_tuple_checker = Some(Arc::new(RangeTupleCheckerChipGPU::hybrid(
            Arc::new(RangeTupleCheckerChip::new(bus)),
            device_ctx,
        )));
        self
    }

    pub fn execute_harness<E, A, C>(
        &mut self,
        harness: &mut TestChipHarness<F, E, A, C>,
        instruction: &Instruction<F>,
    ) where
        E: Executor<F> + Clone,
    {
        self.execute(&mut harness.executor, &mut harness.preflight, instruction);
    }

    pub fn execute_with_pc_harness<E, A, C>(
        &mut self,
        harness: &mut TestChipHarness<F, E, A, C>,
        instruction: &Instruction<F>,
        initial_pc: u32,
    ) where
        E: Executor<F> + Clone,
    {
        self.execute_with_pc(
            &mut harness.executor,
            &mut harness.preflight,
            instruction,
            initial_pc,
        );
    }

    pub fn write_heap<const NUM_LIMBS: usize>(
        &mut self,
        register: usize,
        pointer: usize,
        writes: Vec<[F; NUM_LIMBS]>,
    ) {
        self.write_bytes::<MEMORY_BLOCK_BYTES>(
            1usize,
            register,
            (pointer as u64).to_le_bytes().map(F::from_u8),
        );
        for (i, &write) in writes.iter().enumerate() {
            let ptr = pointer + i * NUM_LIMBS;
            for j in (0..NUM_LIMBS).step_by(MEMORY_BLOCK_BYTES) {
                self.write_bytes::<MEMORY_BLOCK_BYTES>(
                    2usize,
                    ptr + j,
                    write[j..j + MEMORY_BLOCK_BYTES].try_into().unwrap(),
                );
            }
        }
    }

    pub fn system_port(&self) -> SystemPort {
        SystemPort {
            execution_bus: self.execution_bus(),
            program_bus: self.program_bus(),
            memory_bridge: self.memory_bridge(),
            public_values_bus: PublicValuesBus::new(PUBLIC_VALUES_BUS),
        }
    }
    pub fn execution_bridge(&self) -> ExecutionBridge {
        ExecutionBridge::new(self.execution.bus(), self.program.bus())
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        self.memory.memory_bridge()
    }

    pub fn execution_bus(&self) -> ExecutionBus {
        self.execution.bus()
    }

    pub fn program_bus(&self) -> ProgramBus {
        self.program.bus()
    }

    pub fn memory_bus(&self) -> MemoryBus {
        self.memory.mem_bus
    }

    pub fn rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }

    pub fn range_checker(&self) -> Arc<VariableRangeCheckerChipGPU> {
        self.var_range_checker.clone()
    }

    pub fn bitwise_op_lookup(&self) -> Arc<BitwiseOperationLookupChipGPU<8>> {
        self.bitwise_op_lookup
            .clone()
            .expect("Initialize GpuChipTestBuilder with .with_bitwise_op_lookup()")
    }

    pub fn range_tuple_checker(&self) -> Arc<RangeTupleCheckerChipGPU<2>> {
        self.range_tuple_checker
            .clone()
            .expect("Initialize GpuChipTestBuilder with .with_range_tuple_checker()")
    }

    // WARNING: This CPU chip is meant for hybrid chip use, its usage WILL
    // result in altered tracegen. For a dummy primitive chip for trace
    // comparison, see utils::dummy_range_checker.
    pub fn cpu_range_checker(&self) -> SharedVariableRangeCheckerChip {
        self.var_range_checker.cpu_chip.clone().unwrap()
    }

    // WARNING: This CPU chip is meant for hybrid chip use, its usage WILL
    // result in altered tracegen. For a dummy primitive chip for trace
    // comparison, see utils::dummy_bitwise_op_lookup.
    pub fn cpu_bitwise_op_lookup(&self) -> SharedBitwiseOperationLookupChip<8> {
        self.bitwise_op_lookup
            .as_ref()
            .expect("Initialize GpuChipTestBuilder with .with_bitwise_op_lookup()")
            .cpu_chip
            .clone()
            .unwrap()
    }

    // WARNING: This CPU chip is meant for hybrid chip use, its usage WILL
    // result in altered tracegen. For a dummy primitive chip for trace
    // comparison, see utils::dummy_range_tuple_checker.
    pub fn cpu_range_tuple_checker(&self) -> SharedRangeTupleCheckerChip<2> {
        self.range_tuple_checker
            .as_ref()
            .expect("Initialize GpuChipTestBuilder with .with_range_tuple_checker()")
            .cpu_chip
            .clone()
            .unwrap()
    }

    // WARNING: This utility is meant for hybrid chip use, its usage WILL
    // result in altered tracegen. For use during trace comparison, see
    // utils::dummy_memory_helper.
    pub fn cpu_memory_helper(&self) -> SharedMemoryHelper<F> {
        SharedMemoryHelper::new(
            self.cpu_range_checker(),
            self.memory.config.timestamp_max_bits,
        )
    }

    // See [cpu_memory_helper]. Use this utility for creation of CPU chips that
    // are meant for tracegen comparison purposes which should not update other
    // periphery chips (e.g., range checker).
    pub fn dummy_memory_helper(&self) -> SharedMemoryHelper<F> {
        dummy_memory_helper(self.cpu_range_checker().bus(), self.timestamp_max_bits())
    }

    pub fn timestamp_max_bits(&self) -> usize {
        self.memory.config.timestamp_max_bits
    }

    #[cfg(feature = "rvr")]
    pub fn record_preflight_history(
        &mut self,
        program: &Program<F>,
        history: &PreflightHistory,
        exit_code: Option<u32>,
    ) {
        let postflight = Postflight::new(program, history, &self.memory.config, exit_code)
            .expect("test preflight history must be valid");
        for events in history.program.windows(2) {
            let initial = events[0];
            let final_state = events[1];
            let offset = initial
                .pc
                .checked_sub(program.pc_base)
                .expect("test history PC must not precede the program");
            assert_eq!(
                offset % DEFAULT_PC_STEP,
                0,
                "test history PC must be instruction-aligned"
            );
            let instruction = &program
                .get_instruction_and_debug_info((offset / DEFAULT_PC_STEP) as usize)
                .expect("test history PC must resolve to an instruction")
                .0;
            if instruction.opcode == SystemOpcode::TERMINATE.global_opcode() {
                continue;
            }
            let initial_state = ExecutionState::new(initial.pc, initial.timestamp);
            self.program.execute(instruction, &initial_state);
            self.execution.execute(
                initial_state,
                ExecutionState::new(final_state.pc, final_state.timestamp),
            );
        }
        postflight.record_test_writes(&mut self.memory);
    }

    pub fn build(self) -> GpuChipTester {
        GpuChipTester {
            var_range_checker: Some(self.var_range_checker),
            bitwise_op_lookup: self.bitwise_op_lookup,
            range_tuple_checker: self.range_tuple_checker,
            memory: Some(self.memory),
            ..Default::default()
        }
        .load(ExecutionDummyAir::new(self.execution.bus()), self.execution)
        .load(ProgramDummyAir::new(self.program.bus()), self.program)
    }
}

#[derive(Default)]
pub struct GpuChipTester {
    pub airs: Vec<AirRef<SC>>,
    pub ctxs: Vec<AirProvingContext<GpuBackend>>,
    pub memory: Option<DeviceMemoryTester>,
    pub var_range_checker: Option<Arc<VariableRangeCheckerChipGPU>>,
    pub bitwise_op_lookup: Option<Arc<BitwiseOperationLookupChipGPU<8>>>,
    pub range_tuple_checker: Option<Arc<RangeTupleCheckerChipGPU<2>>>,
}

impl GpuChipTester {
    pub fn load<A, G>(mut self, air: A, gpu_chip: G) -> Self
    where
        A: AnyAir<SC> + 'static,
        G: Chip<GpuBackend>,
    {
        let proving_ctx = gpu_chip.generate_proving_ctx();
        if proving_ctx.height() > 0 {
            self = self.load_air_proving_ctx(Arc::new(air) as AirRef<SC>, proving_ctx);
        }
        self
    }

    pub fn load_periphery<A, G>(self, air: A, gpu_chip: G) -> Self
    where
        A: AnyAir<SC> + 'static,
        G: Chip<GpuBackend>,
    {
        self.load(air, gpu_chip)
    }

    pub fn load_air_proving_ctx(
        mut self,
        air: AirRef<SC>,
        proving_ctx: AirProvingContext<GpuBackend>,
    ) -> Self {
        #[cfg(feature = "touchemall")]
        {
            check_trace_validity(&proving_ctx, &air.name());
        }
        self.airs.push(air);
        self.ctxs.push(proving_ctx);
        self
    }

    #[cfg(feature = "rvr")]
    pub fn balance_preflight_memory(&mut self, preflight: &TestPreflight<Val<SC>>) {
        for execution in &preflight.executions {
            self.balance_preflight_history(&execution.program, &execution.history, None);
        }
    }

    #[cfg(feature = "rvr")]
    pub fn balance_preflight_history(
        &mut self,
        program: &Program<Val<SC>>,
        history: &PreflightHistory,
        exit_code: Option<u32>,
    ) {
        let memory = self
            .memory
            .as_mut()
            .expect("chip traces must be loaded before memory finalization");
        let memory_config = memory.config.clone();
        let postflight = match exit_code {
            Some(exit_code) => Postflight::new(program, history, &memory_config, Some(exit_code)),
            None => Postflight::new_for_test(program, history, &memory_config),
        }
        .expect("test preflight history must be valid");
        postflight.balance_test_memory(&mut memory.chip.0);
    }

    #[cfg(feature = "rvr")]
    pub fn load_gpu_harness<E, A, GpuChip, CpuChip>(
        mut self,
        harness: GpuTestChipHarness<Val<SC>, E, A, GpuChip, CpuChip>,
    ) -> Self
    where
        A: AnyAir<SC> + 'static,
    {
        let GpuTestChipHarness {
            air,
            gpu_chip,
            cpu_chip,
            preflight,
            generate_cpu_trace,
            generate_cpu_batch_trace,
            generate_gpu_trace,
            rows_used,
            fill_padding,
            balance_memory,
            ..
        } = harness;
        let generate_cpu_trace =
            generate_cpu_trace.expect("GPU test harness requires a CPU postflight generator");
        let generate_gpu_trace =
            generate_gpu_trace.expect("GPU test harness requires a GPU postflight generator");
        if balance_memory {
            self.balance_preflight_memory(&preflight);
        }
        let memory = self
            .memory
            .as_mut()
            .expect("chip traces must be loaded before memory finalization");
        let memory_config = memory.config.clone();
        let device_ctx = self
            .var_range_checker
            .as_ref()
            .expect("GPU tests require a range checker")
            .device_ctx
            .clone();
        let air = Arc::new(air) as AirRef<SC>;
        let mut used_rows = Vec::new();
        let mut trace_width = None;
        let mut public_values = None;
        let postflights = preflight
            .executions
            .iter()
            .map(|execution| {
                Postflight::new_for_test(&execution.program, &execution.history, &memory_config)
                    .expect("test preflight history must be valid")
            })
            .collect::<Vec<_>>();

        // Chip microtests intentionally execute isolated histories. Full postflight tests cover
        // production multi-step kernel indexing; this path preserves one chip-shaped proof trace.
        for (execution, postflight) in preflight.executions.iter().zip(&postflights) {
            let program =
                GpuPostflightProgram::upload(&execution.program, &memory_config, &device_ctx)
                    .expect("test program must upload");
            let (transcript, plan) = program
                .upload_isolated_history_for_test(&execution.program, &execution.history)
                .expect("test preflight history must upload");
            let proving_ctx = generate_gpu_trace(&gpu_chip, &program, &transcript, &plan)
                .expect("GPU postflight trace generation must succeed");
            transcript
                .synchronize()
                .expect("GPU postflight trace generation must synchronize");
            assert_eq!(
                transcript
                    .error_code()
                    .expect("GPU error code must download"),
                0,
                "GPU postflight trace generation rejected valid test history",
            );

            let trace = generate_cpu_trace(&cpu_chip, postflight)
                .expect("CPU postflight trace generation must succeed");
            let width = Matrix::width(&trace);
            let used = rows_used(&trace);
            assert!(used <= Matrix::height(&trace));
            let height = next_power_of_two_or_zero(used);
            let mut values = trace.values[..used * width].to_vec();
            values.resize(height * width, F::ZERO);
            for row_index in used..height {
                fill_padding(&mut values[row_index * width..(row_index + 1) * width]);
            }
            let expected_trace = RowMajorMatrix::new(values, width);
            let expected_trace_cm = ColMajorMatrix::from_row_major(&expected_trace);
            assert_eq_host_and_device_matrix_col_maj(
                &expected_trace_cm,
                &proving_ctx.common_main,
                &device_ctx,
            );

            assert!(
                proving_ctx.cached_mains.is_empty(),
                "GPU chip test harness does not support cached traces"
            );
            let gpu_height = DeviceMatrixDimensions::height(&proving_ctx.common_main);
            let gpu_width = DeviceMatrixDimensions::width(&proving_ctx.common_main);
            assert_eq!(gpu_width, width);
            let gpu_values = proving_ctx
                .common_main
                .buffer()
                .to_host_on(&device_ctx)
                .expect("GPU trace must download");
            if generate_cpu_batch_trace.is_none() {
                for row in 0..used {
                    for column in 0..width {
                        used_rows.push(gpu_values[column * gpu_height + row]);
                    }
                }
            }
            if used > 0 {
                assert_eq!(
                    *trace_width.get_or_insert(width),
                    width,
                    "all histories for one chip must have the same trace width"
                );
                match &public_values {
                    Some(expected) => assert_eq!(
                        expected, &proving_ctx.public_values,
                        "all histories for one chip must have the same public values"
                    ),
                    None => public_values = Some(proving_ctx.public_values),
                }
            }
        }

        if let Some(generate_batch_trace) = generate_cpu_batch_trace {
            let trace = generate_batch_trace(&cpu_chip, &postflights)
                .expect("CPU batch postflight trace generation must succeed");
            let width = Matrix::width(&trace);
            let used = rows_used(&trace);
            assert!(used <= Matrix::height(&trace));
            let height = next_power_of_two_or_zero(used);
            let mut values = trace.values[..used * width].to_vec();
            values.resize(height * width, F::ZERO);
            for row_index in used..height {
                fill_padding(&mut values[row_index * width..(row_index + 1) * width]);
            }
            let trace = ColMajorMatrix::from_row_major(&RowMajorMatrix::new(values, width));
            let buffer = trace
                .values
                .to_device_on(&device_ctx)
                .expect("batch proof trace must upload");
            self.airs.push(air);
            self.ctxs.push(AirProvingContext {
                cached_mains: vec![],
                common_main: DeviceMatrix::new(Arc::new(buffer), height, width),
                public_values: public_values.unwrap_or_default(),
            });
            return self;
        }

        if let Some(width) = trace_width {
            let used = used_rows.len() / width;
            let height = next_power_of_two_or_zero(used);
            used_rows.resize(height * width, F::ZERO);
            for row in used..height {
                fill_padding(&mut used_rows[row * width..(row + 1) * width]);
            }
            let trace = ColMajorMatrix::from_row_major(&RowMajorMatrix::new(used_rows, width));
            let buffer = trace
                .values
                .to_device_on(&device_ctx)
                .expect("aggregated GPU trace must upload");
            self.airs.push(air);
            self.ctxs.push(AirProvingContext {
                cached_mains: vec![],
                common_main: DeviceMatrix::new(Arc::new(buffer), height, width),
                public_values: public_values.unwrap_or_default(),
            });
        }
        self
    }

    pub fn finalize(mut self) -> Self {
        if let Some(memory_tester) = self.memory.take() {
            let DeviceMemoryTester {
                chip,
                mut memory,
                mut inventory,
                hasher_chip,
                config,
                mem_bus,
                range_bus,
            } = memory_tester;
            let touched_memory = memory.finalize::<F>();
            let memory_bridge = MemoryBridge::new(mem_bus, config.timestamp_max_bits, range_bus);
            self = self.load_periphery(chip.0.air, chip);

            let airs = MemoryAirInventory::new(
                memory_bridge,
                &config,
                PermutationCheckBus::new(MEMORY_MERKLE_BUS),
                PermutationCheckBus::new(POSEIDON2_DIRECT_BUS),
            )
            .into_airs();
            let ctxs = inventory.generate_proving_ctxs(touched_memory);
            for (air, ctx) in airs
                .into_iter()
                .zip(ctxs)
                .filter(|(_, ctx)| ctx.height() > 0)
            {
                self = self.load_air_proving_ctx(air, ctx);
            }

            if let Some(hasher_chip) = hasher_chip {
                let air: AirRef<SC> = match hasher_chip.as_ref() {
                    Poseidon2PeripheryChipGPU::Register0(_) => {
                        let config = Poseidon2Config::default();
                        Arc::new(Poseidon2PeripheryAir::new(
                            Arc::new(Poseidon2SubAir::<F, 0>::new(config.constants.into())),
                            LookupBus::new(POSEIDON2_DIRECT_BUS),
                        ))
                    }
                    Poseidon2PeripheryChipGPU::Register1(_) => {
                        let config = Poseidon2Config::default();
                        Arc::new(Poseidon2PeripheryAir::new(
                            Arc::new(Poseidon2SubAir::<F, 1>::new(config.constants.into())),
                            LookupBus::new(POSEIDON2_DIRECT_BUS),
                        ))
                    }
                };
                let ctx = hasher_chip.generate_proving_ctx();
                self = self.load_air_proving_ctx(air, ctx);
            }
        }
        if let Some(var_range_checker) = self.var_range_checker.take() {
            self = self.load_periphery(
                VariableRangeCheckerAir::new(var_range_checker.cpu_chip.as_ref().unwrap().bus()),
                var_range_checker,
            );
        }
        if let Some(bitwise_op_lookup) = self.bitwise_op_lookup.take() {
            self = self.load_periphery(
                BitwiseOperationLookupAir::<8>::new(
                    bitwise_op_lookup.cpu_chip.as_ref().unwrap().bus(),
                ),
                bitwise_op_lookup,
            );
        }
        if let Some(range_tuple_checker) = self.range_tuple_checker.take() {
            self = self.load_periphery(
                RangeTupleCheckerAir {
                    bus: *range_tuple_checker.cpu_chip.as_ref().unwrap().bus(),
                },
                range_tuple_checker,
            );
        }
        self
    }

    pub fn test<P: Fn() -> BabyBearPoseidon2GpuEngine>(
        self,
        engine_provider: P,
    ) -> Result<VerificationData<SC>, TestGpuStarkError> {
        engine_provider().run_test(self.airs, self.ctxs)
    }

    pub fn simple_test(self) -> Result<VerificationData<SC>, TestGpuStarkError> {
        self.test(test_gpu_engine)
    }
}

/// Concrete `StarkTestError` type alias for BabyBear Poseidon2 GPU tests.
pub type TestGpuStarkError = openvm_stark_backend::StarkTestError<ProverError, EF>;
