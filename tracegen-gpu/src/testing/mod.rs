use std::sync::Arc;

use openvm_circuit::{
    arch::{
        instructions::instruction::Instruction,
        testing::{
            execution::air::ExecutionDummyAir, program::air::ProgramDummyAir, TestChipHarness,
            EXECUTION_BUS, MEMORY_BUS, MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS,
            READ_INSTRUCTION_BUS,
        },
        Arena, DenseRecordArena, ExecutionBridge, ExecutionBus, ExecutionState,
        InstructionExecutor, MatrixRecordArena, MemoryConfig, Streams, VmStateMut,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryBus},
            MemoryAirInventory, SharedMemoryHelper,
        },
        poseidon2::air::Poseidon2PeripheryAir,
        program::ProgramBus,
        SystemPort,
    },
    utils::next_power_of_two_or_zero,
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
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir};
use openvm_stark_backend::{
    config::Val,
    interaction::{LookupBus, PermutationCheckBus},
    p3_field::{Field, FieldAlgebra},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::AnyRap,
    utils::disable_debug_builder,
    verifier::VerificationError,
    AirRef, Chip,
};
use openvm_stark_sdk::{
    config::{setup_tracing_with_log_level, FriParameters},
    engine::{StarkFriEngine, VerificationDataWithFriParams},
};
use p3_air::BaseAir;
use p3_field::PrimeField32;
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
    system::poseidon2::Poseidon2PeripheryChipGPU,
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

pub struct GpuTestChipHarness<F, Executor, AIR, GpuChip, CpuChip> {
    pub executor: Executor,
    pub air: AIR,
    pub gpu_chip: GpuChip,
    pub cpu_chip: CpuChip,
    pub dense_arena: DenseRecordArena,
    pub matrix_arena: MatrixRecordArena<F>,
}

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
        let width = air.width();
        let height = next_power_of_two_or_zero(height);
        let dense_arena = DenseRecordArena::with_capacity(height, width);
        let matrix_arena = MatrixRecordArena::with_capacity(height, width);
        Self {
            executor,
            air,
            gpu_chip,
            cpu_chip,
            dense_arena,
            matrix_arena,
        }
    }
}

pub struct GpuChipTestBuilder {
    pub memory: DeviceMemoryTester,
    pub execution: DeviceExecutionTester,
    pub program: DeviceProgramTester,
    pub streams: Streams<F>,

    var_range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_op_lookup: Option<Arc<BitwiseOperationLookupChipGPU<8>>>,
    range_tuple_checker: Option<Arc<RangeTupleCheckerChipGPU<2>>>,

    rng: StdRng,
    default_register: usize,
    default_pointer: usize,
}

impl Default for GpuChipTestBuilder {
    fn default() -> Self {
        Self::volatile(MemoryConfig::default(), default_var_range_checker_bus())
    }
}

impl GpuChipTestBuilder {
    pub fn new() -> Self {
        // TODO[stephen]: allow for custom test builder configuration
        Self::default()
    }

    pub fn volatile(mem_config: MemoryConfig, bus: VariableRangeCheckerBus) -> Self {
        setup_tracing_with_log_level(Level::INFO);
        let mem_bus = MemoryBus::new(MEMORY_BUS);
        let range_checker = Arc::new(VariableRangeCheckerChipGPU::hybrid(Arc::new(
            VariableRangeCheckerChip::new(bus),
        )));
        Self {
            memory: DeviceMemoryTester::volatile(
                default_tracing_memory(&mem_config, 1),
                mem_bus,
                mem_config,
                range_checker.clone(),
            ),
            execution: DeviceExecutionTester::new(ExecutionBus::new(EXECUTION_BUS)),
            program: DeviceProgramTester::new(ProgramBus::new(READ_INSTRUCTION_BUS)),
            streams: Default::default(),
            var_range_checker: range_checker,
            bitwise_op_lookup: None,
            range_tuple_checker: None,
            rng: StdRng::seed_from_u64(0),
            default_register: 0,
            default_pointer: 0,
        }
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

    pub fn address_bits(&self) -> usize {
        self.memory.config.pointer_max_bits
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
        SharedMemoryHelper::new(self.cpu_range_checker(), self.memory.config.clk_max_bits)
    }

    pub fn build(self) -> GpuChipTester {
        GpuChipTester {
            var_range_checker: Some(self.var_range_checker),
            bitwise_op_lookup: self.bitwise_op_lookup,
            range_tuple_checker: self.range_tuple_checker,
            memory: Some(self.memory),
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
    pub memory: Option<DeviceMemoryTester>,
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

    pub fn load_air_proving_ctx(
        mut self,
        air: AirRef<SC>,
        proving_ctx: AirProvingContext<GpuBackend>,
    ) -> Self {
        self.airs.push(air);
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

    pub fn load_gpu_harness<E, A, GpuChip, CpuChip>(
        self,
        harness: GpuTestChipHarness<Val<SC>, E, A, GpuChip, CpuChip>,
    ) -> Self
    where
        A: AnyRap<SC> + 'static,
        CpuChip: Chip<MatrixRecordArena<Val<SC>>, CpuBackend<SC>>,
        GpuChip: Chip<DenseRecordArena, GpuBackend>,
    {
        self.load_and_compare(
            harness.air,
            harness.gpu_chip,
            harness.dense_arena,
            harness.cpu_chip,
            harness.matrix_arena,
        )
    }

    pub fn finalize(mut self) -> Self {
        if let Some(mut memory_tester) = self.memory.take() {
            let is_persistent = memory_tester.inventory.continuation_enabled();
            let touched_memory = memory_tester.memory.finalize::<F>(is_persistent);
            let memory_bridge = memory_tester.memory_bridge();

            for chip in memory_tester.chip_for_block.into_values() {
                self = self.load_periphery(chip.0.air, chip);
            }

            let airs = MemoryAirInventory::<SC>::new(
                memory_bridge,
                &memory_tester.config,
                memory_tester.range_bus,
                is_persistent.then_some((
                    PermutationCheckBus::new(MEMORY_MERKLE_BUS),
                    PermutationCheckBus::new(POSEIDON2_DIRECT_BUS),
                )),
            )
            .into_airs();
            let ctxs = memory_tester
                .inventory
                .generate_proving_ctxs(memory_tester.memory.access_adapter_records, touched_memory);
            for (air, ctx) in airs
                .into_iter()
                .zip(ctxs)
                .filter(|(_, ctx)| ctx.common_main.is_some())
            {
                self = self.load_air_proving_ctx(air, ctx);
            }

            if let Some(hasher_chip) = memory_tester.hasher_chip {
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
                let ctx = hasher_chip.generate_proving_ctx(());
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
        disable_debug_builder();
        let msg = format!(
            "Expected verification to fail with {:?}, but it didn't",
            &expected_error
        );
        let result = self.simple_test();
        assert_eq!(result.err(), Some(expected_error), "{}", msg);
    }
}
