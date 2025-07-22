use std::sync::Arc;

use openvm_circuit::{
    arch::{
        instructions::instruction::Instruction,
        testing::{
            execution::air::ExecutionDummyAir, program::air::ProgramDummyAir, TestChipHarness,
            EXECUTION_BUS, MEMORY_BUS, RANGE_CHECKER_BUS, READ_INSTRUCTION_BUS,
        },
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor, MemoryConfig, Streams,
        VmStateMut,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryBus},
            MemoryController, SharedMemoryHelper,
        },
        program::ProgramBus,
        SystemPort,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        SharedRangeTupleCheckerChip,
    },
    var_range::{
        SharedVariableRangeCheckerChip, VariableRangeCheckerAir, VariableRangeCheckerBus,
        VariableRangeCheckerChip,
    },
};
use openvm_stark_backend::{
    p3_field::{Field, FieldAlgebra},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::AnyRap,
    verifier::VerificationError,
    AirRef, Chip,
};
use openvm_stark_sdk::{
    config::{setup_tracing_with_log_level, FriParameters},
    engine::{StarkFriEngine, VerificationDataWithFriParams},
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use stark_backend_gpu::{
    engine::GpuBabyBearPoseidon2Engine,
    prover_backend::GpuBackend,
    types::{F, SC},
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
    default_register: usize,
    default_pointer: usize,
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
        let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            RANGE_CHECKER_BUS,
            mem_config.decomp,
        )));
        Self {
            memory: DeviceMemoryTester::new(
                default_tracing_memory(&mem_config, 1),
                MemoryController::with_volatile_memory(mem_bus, mem_config, range_checker),
            ),
            execution: DeviceExecutionTester::new(ExecutionBus::new(EXECUTION_BUS)),
            program: DeviceProgramTester::new(ProgramBus::new(READ_INSTRUCTION_BUS)),
            streams: Default::default(),
            var_range_checker: None,
            bitwise_op_lookup: None,
            range_tuple_checker: None,
            rng: StdRng::seed_from_u64(0),
            default_register: 0,
            default_pointer: 0,
        }
    }

    pub fn with_variable_range_checker(mut self, bus: VariableRangeCheckerBus) -> Self {
        self.var_range_checker = Some(Arc::new(VariableRangeCheckerChipGPU::hybrid(Arc::new(
            VariableRangeCheckerChip::new(bus),
        ))));
        self
    }

    pub fn with_bitwise_op_lookup(mut self, bus: BitwiseOperationLookupBus) -> Self {
        self.bitwise_op_lookup = Some(Arc::new(BitwiseOperationLookupChipGPU::hybrid(Arc::new(
            BitwiseOperationLookupChip::new(bus),
        ))));
        self
    }

    pub fn with_range_tuple_checker(mut self, bus: RangeTupleCheckerBus<2>) -> Self {
        self.range_tuple_checker = Some(Arc::new(RangeTupleCheckerChipGPU::hybrid(Arc::new(
            RangeTupleCheckerChip::new(bus),
        ))));
        self
    }

    fn next_elem_size_u32(&mut self) -> u32 {
        self.rng.next_u32() % (1 << (F::bits() - 2))
    }

    pub fn execute<E, RA>(&mut self, executor: &mut E, arena: &mut RA, instruction: &Instruction<F>)
    where
        E: InstructionExecutor<F, RA>,
    {
        let initial_pc = self.next_elem_size_u32();
        self.execute_with_pc(executor, arena, instruction, initial_pc);
    }

    pub fn execute_harness<E, A, C, RA>(
        &mut self,
        harness: &mut TestChipHarness<F, E, A, C, RA>,
        instruction: &Instruction<F>,
    ) where
        E: InstructionExecutor<F, RA>,
    {
        self.execute(&mut harness.executor, &mut harness.arena, instruction);
    }

    pub fn execute_with_pc<E, RA>(
        &mut self,
        executor: &mut E,
        arena: &mut RA,
        instruction: &Instruction<F>,
        initial_pc: u32,
    ) where
        E: InstructionExecutor<F, RA>,
    {
        let initial_state = ExecutionState {
            pc: initial_pc,
            timestamp: self.memory.memory.timestamp(),
        };
        tracing::debug!("initial_timestamp={}", initial_state.timestamp);

        let mut pc = initial_pc;
        let state_mut = VmStateMut {
            pc: &mut pc,
            memory: &mut self.memory.memory,
            streams: &mut self.streams,
            rng: &mut self.rng,
            ctx: arena,
        };

        executor
            .execute(state_mut, instruction)
            .expect("Expected the execution not to fail");
        let final_state = ExecutionState {
            pc,
            timestamp: self.memory.memory.timestamp(),
        };

        self.program.execute(instruction, &initial_state);
        self.execution.execute(initial_state, final_state);
    }

    pub fn execute_with_pc_harness<E, A, C, RA>(
        &mut self,
        harness: &mut TestChipHarness<F, E, A, C, RA>,
        instruction: &Instruction<F>,
        initial_pc: u32,
    ) where
        E: InstructionExecutor<F, RA>,
    {
        self.execute_with_pc(
            &mut harness.executor,
            &mut harness.arena,
            instruction,
            initial_pc,
        );
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

    pub fn write_usize<const N: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        value: [usize; N],
    ) {
        self.write(address_space, pointer, value.map(F::from_canonical_usize));
    }

    pub fn write_heap<const NUM_LIMBS: usize>(
        &mut self,
        register: usize,
        pointer: usize,
        writes: Vec<[F; NUM_LIMBS]>,
    ) {
        self.write(
            1usize,
            register,
            pointer.to_le_bytes().map(F::from_canonical_u8),
        );
        if NUM_LIMBS.is_power_of_two() {
            for (i, &write) in writes.iter().enumerate() {
                self.write(2usize, pointer + i * NUM_LIMBS, write);
            }
        } else {
            for (i, &write) in writes.iter().enumerate() {
                let ptr = pointer + i * NUM_LIMBS;
                for j in (0..NUM_LIMBS).step_by(4) {
                    self.write::<4>(2usize, ptr + j, write[j..j + 4].try_into().unwrap());
                }
            }
        }
    }

    pub fn get_default_register(&mut self, increment: usize) -> usize {
        self.default_register += increment;
        self.default_register - increment
    }

    pub fn get_default_pointer(&mut self, increment: usize) -> usize {
        self.default_pointer += increment;
        self.default_pointer - increment
    }

    pub fn write_heap_pointer_default(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
    ) -> (usize, usize) {
        let register = self.get_default_register(reg_increment);
        let pointer = self.get_default_pointer(pointer_increment);
        self.write(1, register, pointer.to_le_bytes().map(F::from_canonical_u8));
        (register, pointer)
    }

    pub fn write_heap_default<const NUM_LIMBS: usize>(
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

    pub fn system_port(&self) -> SystemPort {
        SystemPort {
            execution_bus: self.execution_bus(),
            program_bus: self.program_bus(),
            memory_bridge: self.memory_bridge(),
        }
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
        self.var_range_checker
            .as_ref()
            .expect("Initialize GpuChipTestBuilder with .with_variable_range_checker()")
            .cpu_chip
            .clone()
            .unwrap()
    }

    pub fn cpu_bitwise_op_lookup(&self) -> SharedBitwiseOperationLookupChip<8> {
        self.bitwise_op_lookup
            .as_ref()
            .expect("Initialize GpuChipTestBuilder with .with_bitwise_op_lookup()")
            .cpu_chip
            .clone()
            .unwrap()
    }

    pub fn cpu_range_tuple_checker(&self) -> SharedRangeTupleCheckerChip<2> {
        self.range_tuple_checker
            .as_ref()
            .expect("Initialize GpuChipTestBuilder with .with_range_tuple_checker()")
            .cpu_chip
            .clone()
            .unwrap()
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
        .load(
            ExecutionDummyAir::new(self.execution.bus()),
            self.execution,
            (),
        )
        .load(ProgramDummyAir::new(self.program.bus()), self.program, ())
    }
}

#[derive(Default)]
pub struct GpuChipTester {
    pub airs: Vec<AirRef<SC>>,
    pub ctxs: Vec<AirProvingContext<GpuBackend>>,
    pub var_range_checker: Option<Arc<VariableRangeCheckerChipGPU>>,
    pub bitwise_op_lookup: Option<Arc<BitwiseOperationLookupChipGPU<8>>>,
    pub range_tuple_checker: Option<Arc<RangeTupleCheckerChipGPU<2>>>,
}

impl GpuChipTester {
    pub fn load<A, G, RA>(mut self, air: A, gpu_chip: G, gpu_arena: RA) -> Self
    where
        A: AnyRap<SC> + 'static,
        G: Chip<RA, GpuBackend>,
    {
        let proving_ctx = gpu_chip.generate_proving_ctx(gpu_arena);
        if proving_ctx.common_main.is_some() {
            self.airs.push(Arc::new(air) as AirRef<SC>);
            self.ctxs.push(proving_ctx);
        }
        self
    }

    pub fn load_harness<E, A, G, RA>(self, harness: TestChipHarness<F, E, A, G, RA>) -> Self
    where
        A: AnyRap<SC> + 'static,
        G: Chip<RA, GpuBackend>,
    {
        self.load(harness.air, harness.chip, harness.arena)
    }

    pub fn load_periphery<A, G>(self, air: A, gpu_chip: G) -> Self
    where
        A: AnyRap<SC> + 'static,
        G: Chip<(), GpuBackend>,
    {
        self.load(air, gpu_chip, ())
    }

    pub fn load_air_proving_ctx<A>(
        mut self,
        air: A,
        proving_ctx: AirProvingContext<GpuBackend>,
    ) -> Self
    where
        A: AnyRap<SC> + 'static,
    {
        self.airs.push(Arc::new(air) as AirRef<SC>);
        self.ctxs.push(proving_ctx);
        self
    }

    pub fn load_and_compare<A, G, RA, C, CRA>(
        mut self,
        air: A,
        gpu_chip: G,
        gpu_arena: RA,
        cpu_chip: C,
        cpu_arena: CRA,
    ) -> Self
    where
        A: AnyRap<SC> + 'static,
        C: Chip<CRA, CpuBackend<SC>>,
        G: Chip<RA, GpuBackend>,
    {
        let proving_ctx = gpu_chip.generate_proving_ctx(gpu_arena);
        let expected_trace = cpu_chip.generate_proving_ctx(cpu_arena).common_main;
        if proving_ctx.common_main.is_none() {
            assert!(expected_trace.is_none());
            return self;
        }
        assert_eq_cpu_and_gpu_matrix(
            expected_trace.unwrap(),
            proving_ctx.common_main.as_ref().unwrap(),
        );
        self.airs.push(Arc::new(air) as AirRef<SC>);
        self.ctxs.push(proving_ctx);
        self
    }

    pub fn finalize(mut self) -> Self {
        // TODO[stephen]: memory chips tracegen
        if let Some(var_range_checker) = self.var_range_checker.clone() {
            self = self.load(
                VariableRangeCheckerAir::new(var_range_checker.cpu_chip.as_ref().unwrap().bus()),
                var_range_checker,
                (),
            );
        }
        if let Some(bitwise_op_lookup) = self.bitwise_op_lookup.clone() {
            self = self.load(
                BitwiseOperationLookupAir::<8>::new(
                    bitwise_op_lookup.cpu_chip.as_ref().unwrap().bus(),
                ),
                bitwise_op_lookup,
                (),
            );
        }
        if let Some(range_tuple_checker) = self.range_tuple_checker.clone() {
            self = self.load(
                RangeTupleCheckerAir {
                    bus: *range_tuple_checker.cpu_chip.as_ref().unwrap().bus(),
                },
                range_tuple_checker,
                (),
            );
        }
        self
    }

    pub fn test<P: Fn() -> GpuBabyBearPoseidon2Engine>(
        self,
        engine_provider: P,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        engine_provider().run_test(self.airs, self.ctxs)
    }

    pub fn simple_test(self) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
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
