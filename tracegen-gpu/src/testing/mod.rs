use std::sync::Arc;

use openvm_circuit::{
    arch::{
        instructions::instruction::Instruction,
        testing::{EXECUTION_BUS, MEMORY_BUS, RANGE_CHECKER_BUS, READ_INSTRUCTION_BUS},
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor, MemoryConfig, Streams,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryBus},
            MemoryController, SharedMemoryHelper,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    range_tuple::RangeTupleCheckerBus,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_stark_backend::{
    engine::VerificationData, p3_field::Field, p3_util::log2_strict_usize,
    verifier::VerificationError, AirRef, Chip,
};
use openvm_stark_sdk::{
    config::{setup_tracing_with_log_level, FriParameters},
    engine::StarkFriEngine,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use stark_backend_gpu::{
    engine::GpuBabyBearPoseidon2Engine,
    prover_backend::GpuBackend,
    types::{DeviceAirProofRawInput, F, SC},
};
use tracing::Level;

use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
        var_range::VariableRangeCheckerChipGPU,
    },
    testing::{
        execution::DeviceExecutionTester, memory::DeviceMemoryTester, program::DeviceProgramTester,
    },
    DeviceChip,
};

pub mod cuda;
pub mod execution;
pub mod memory;
pub mod program;
mod utils;
pub use utils::*;

pub struct GpuChipTestBuilder {
    pub memory: DeviceMemoryTester,
    pub execution: DeviceExecutionTester,
    pub program: DeviceProgramTester,
    pub streams: Streams<F>,

    var_range_checker: Option<Arc<VariableRangeCheckerChipGPU>>,
    bitwise_op_lookup: Option<Arc<BitwiseOperationLookupChipGPU<8>>>,
    range_tuple_checker: Option<Arc<RangeTupleCheckerChipGPU<2>>>,

    rng: StdRng,
}

impl Default for GpuChipTestBuilder {
    fn default() -> Self {
        Self::volatile(MemoryConfig::default())
    }
}

impl GpuChipTestBuilder {
    pub fn new() -> Self {
        // TODO[stephen]: allow for custom test builder configuration
        Self::default()
    }

    pub fn volatile(mem_config: MemoryConfig) -> Self {
        setup_tracing_with_log_level(Level::INFO);
        let mem_bus = MemoryBus::new(MEMORY_BUS);
        let range_checker = SharedVariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            RANGE_CHECKER_BUS,
            mem_config.decomp,
        ));
        let max_access_adapter_n = log2_strict_usize(mem_config.max_access_adapter_n);
        let mut memory_controller =
            MemoryController::with_volatile_memory(mem_bus, mem_config, range_checker);
        memory_controller
            .memory
            .access_adapter_inventory
            .set_arena_from_trace_heights(&vec![1 << 16; max_access_adapter_n]);
        Self {
            memory: DeviceMemoryTester::new(memory_controller),
            execution: DeviceExecutionTester::new(ExecutionBus::new(EXECUTION_BUS)),
            program: DeviceProgramTester::new(ProgramBus::new(READ_INSTRUCTION_BUS)),
            streams: Default::default(),
            var_range_checker: None,
            bitwise_op_lookup: None,
            range_tuple_checker: None,
            rng: StdRng::seed_from_u64(0),
        }
    }

    pub fn with_variable_range_checker(mut self) -> Self {
        let bus = self.memory.controller.range_checker.bus();
        self.var_range_checker = Some(Arc::new(VariableRangeCheckerChipGPU::new(bus)));
        self
    }

    pub fn with_bitwise_op_lookup(mut self, bus: BitwiseOperationLookupBus) -> Self {
        self.bitwise_op_lookup = Some(Arc::new(BitwiseOperationLookupChipGPU::new(bus)));
        self
    }

    pub fn with_range_tuple_checker(mut self, bus: RangeTupleCheckerBus<2>) -> Self {
        self.range_tuple_checker = Some(Arc::new(RangeTupleCheckerChipGPU::new(bus)));
        self
    }

    fn next_elem_size_u32(&mut self) -> u32 {
        self.rng.next_u32() % (1 << (F::bits() - 2))
    }

    pub fn execute<E: InstructionExecutor<F>>(
        &mut self,
        executor: &mut E,
        instruction: &Instruction<F>,
    ) {
        let initial_pc = self.next_elem_size_u32();
        self.execute_with_pc(executor, instruction, initial_pc);
    }

    pub fn execute_with_pc<E: InstructionExecutor<F>>(
        &mut self,
        executor: &mut E,
        instruction: &Instruction<F>,
        initial_pc: u32,
    ) {
        let initial_state = ExecutionState {
            pc: initial_pc,
            timestamp: self.memory.controller.timestamp(),
        };
        tracing::debug!(?initial_state.timestamp);
        let final_state = executor
            .execute(
                &mut self.memory.controller,
                &mut self.streams,
                &mut self.rng,
                instruction,
                initial_state,
            )
            .expect("Expected the execution not to fail");

        self.program.execute(instruction, &initial_state);
        self.execution.execute(initial_state, final_state);
    }

    pub fn read_cell(&mut self, address_space: usize, pointer: usize) -> F {
        self.read::<1>(address_space, pointer)[0]
    }

    pub fn write_cell(&mut self, address_space: usize, pointer: usize, value: F) {
        self.write(address_space, pointer, [value]);
    }

    pub fn read<const N: usize>(&mut self, address_space: usize, pointer: usize) -> [F; N] {
        self.memory.read(address_space, pointer)
    }

    pub fn write<const N: usize>(&mut self, address_space: usize, pointer: usize, value: [F; N]) {
        self.memory.write(address_space, pointer, value);
    }

    pub fn execution_bridge(&self) -> ExecutionBridge {
        ExecutionBridge::new(self.execution.bus(), self.program.bus())
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        self.memory.controller.memory_bridge()
    }

    pub fn execution_bus(&self) -> ExecutionBus {
        self.execution.bus()
    }

    pub fn program_bus(&self) -> ProgramBus {
        self.program.bus()
    }

    pub fn memory_bus(&self) -> MemoryBus {
        self.memory.controller.memory_bus
    }

    pub fn memory_controller(&self) -> &MemoryController<F> {
        &self.memory.controller
    }

    pub fn address_bits(&self) -> usize {
        self.memory.controller.mem_config().pointer_max_bits
    }

    pub fn rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }

    pub fn range_checker(&self) -> Arc<VariableRangeCheckerChipGPU> {
        self.var_range_checker
            .clone()
            .expect("Initialize GpuChipTestBuilder with .with_variable_range_checker()")
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

    pub fn cpu_range_checker(&self) -> SharedVariableRangeCheckerChip {
        self.memory.controller.range_checker.clone()
    }

    pub fn cpu_memory_helper(&self) -> SharedMemoryHelper<F> {
        self.memory.controller.helper()
    }

    pub fn build(self) -> GpuChipTester {
        GpuChipTester {
            var_range_checker: self.var_range_checker,
            bitwise_op_lookup: self.bitwise_op_lookup,
            range_tuple_checker: self.range_tuple_checker,
            ..Default::default()
        }
        .load(self.execution)
        .load(self.program)
    }
}

#[derive(Default)]
pub struct GpuChipTester {
    pub airs: Vec<AirRef<SC>>,
    pub raw_inputs: Vec<DeviceAirProofRawInput<GpuBackend>>,
    pub var_range_checker: Option<Arc<VariableRangeCheckerChipGPU>>,
    pub bitwise_op_lookup: Option<Arc<BitwiseOperationLookupChipGPU<8>>>,
    pub range_tuple_checker: Option<Arc<RangeTupleCheckerChipGPU<2>>>,
}

impl GpuChipTester {
    pub fn load<G: DeviceChip<SC, GpuBackend>>(mut self, trace_generator: G) -> Self {
        if trace_generator.current_trace_height() > 0 {
            self.airs.push(trace_generator.air());
            self.raw_inputs
                .push(trace_generator.generate_device_air_proof_input());
        }
        self
    }

    pub fn load_and_compare<G: DeviceChip<SC, GpuBackend>, C: Chip<SC>>(
        mut self,
        trace_generator: G,
        expected_chip: C,
    ) -> Self {
        if trace_generator.current_trace_height() > 0 {
            self.airs.push(trace_generator.air());
            let raw_input = trace_generator.generate_device_air_proof_input();
            let expected_trace = Arc::new(
                expected_chip
                    .generate_air_proof_input()
                    .raw
                    .common_main
                    .unwrap(),
            );
            assert_eq_cpu_and_gpu_matrix(expected_trace, raw_input.common_main.as_ref().unwrap());
            self.raw_inputs.push(raw_input);
        }
        self
    }

    pub fn finalize(mut self) -> Self {
        // TODO[stephen]: finalize memory
        if let Some(var_range_checker) = self.var_range_checker.clone() {
            self = self.load(var_range_checker);
        }
        if let Some(bitwise_op_lookup) = self.bitwise_op_lookup.clone() {
            self = self.load(bitwise_op_lookup);
        }
        if let Some(range_tuple_checker) = self.range_tuple_checker.clone() {
            self = self.load(range_tuple_checker);
        }
        self
    }

    pub fn test<P: Fn() -> GpuBabyBearPoseidon2Engine>(
        self,
        engine_provider: P,
    ) -> Result<VerificationData<SC>, VerificationError> {
        engine_provider().gpu_run_test(self.airs, self.raw_inputs)
    }

    pub fn simple_test(self) -> Result<VerificationData<SC>, VerificationError> {
        self.test(|| GpuBabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1)))
    }

    pub fn simple_test_with_expected_error(self, expected_error: VerificationError) {
        let msg = format!(
            "Expected verification to fail with {:?}, but it didn't",
            &expected_error
        );
        let result = self.simple_test();
        assert_eq!(result.err(), Some(expected_error), "{}", msg);
    }
}
