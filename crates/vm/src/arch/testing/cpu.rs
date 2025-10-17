use std::sync::Arc;

use itertools::zip_eq;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus, VariableRangeCheckerChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, NATIVE_AS};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::VerificationData,
    interaction::PermutationCheckBus,
    p3_field::{Field, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_util::log2_strict_usize,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        types::AirProvingContext,
    },
    rap::AnyRap,
    verifier::VerificationError,
    AirRef, Chip,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_blake3::{BabyBearBlake3Config, BabyBearBlake3Engine},
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing_with_log_level, FriParameters,
    },
    engine::{StarkEngine, StarkFriEngine},
    p3_baby_bear::BabyBear,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use tracing::Level;

use crate::{
    arch::{
        testing::{
            execution::air::ExecutionDummyAir,
            program::{air::ProgramDummyAir, ProgramTester},
            ExecutionTester, MemoryTester, TestBuilder, TestChipHarness, EXECUTION_BUS, MEMORY_BUS,
            MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS, RANGE_CHECKER_BUS, READ_INSTRUCTION_BUS,
        },
        vm_poseidon2_config, Arena, ExecutionBridge, ExecutionBus, ExecutionState,
        MatrixRecordArena, MemoryConfig, PreflightExecutor, Streams, VmStateMut,
    },
    system::{
        memory::{
            adapter::records::arena_size_bound,
            offline_checker::{MemoryBridge, MemoryBus},
            online::TracingMemory,
            MemoryAirInventory, MemoryController, SharedMemoryHelper, CHUNK,
        },
        poseidon2::Poseidon2PeripheryChip,
        program::ProgramBus,
        SystemPort,
    },
};

pub struct VmChipTestBuilder<F: Field> {
    pub memory: MemoryTester<F>,
    pub streams: Streams<F>,
    pub rng: StdRng,
    pub execution: ExecutionTester<F>,
    pub program: ProgramTester<F>,
    internal_rng: StdRng,
    custom_pvs: Vec<Option<F>>,
    default_register: usize,
    default_pointer: usize,
}

impl<F> TestBuilder<F> for VmChipTestBuilder<F>
where
    F: PrimeField32,
{
    fn execute<E, RA>(&mut self, executor: &mut E, arena: &mut RA, instruction: &Instruction<F>)
    where
        E: PreflightExecutor<F, RA>,
        RA: Arena,
    {
        let initial_pc = self.next_elem_size_u32();
        self.execute_with_pc(executor, arena, instruction, initial_pc);
    }

    fn execute_with_pc<E, RA>(
        &mut self,
        executor: &mut E,
        arena: &mut RA,
        instruction: &Instruction<F>,
        initial_pc: u32,
    ) where
        E: PreflightExecutor<F, RA>,
        RA: Arena,
    {
        let initial_state = ExecutionState {
            pc: initial_pc,
            timestamp: self.memory.memory.timestamp(),
        };
        tracing::debug!("initial_timestamp={}", self.memory.memory.timestamp());

        let mut pc = initial_pc;
        let state_mut = VmStateMut {
            pc: &mut pc,
            memory: &mut self.memory.memory,
            streams: &mut self.streams,
            rng: &mut self.rng,
            custom_pvs: &mut self.custom_pvs,
            ctx: arena,
            #[cfg(feature = "metrics")]
            metrics: &mut Default::default(),
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

    fn read<const N: usize>(&mut self, address_space: usize, pointer: usize) -> [F; N] {
        self.memory.read(address_space, pointer)
    }

    fn write<const N: usize>(&mut self, address_space: usize, pointer: usize, value: [F; N]) {
        self.memory.write(address_space, pointer, value);
    }

    fn write_usize<const N: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        value: [usize; N],
    ) {
        self.memory
            .write(address_space, pointer, value.map(F::from_canonical_usize));
    }

    fn write_cell(&mut self, address_space: usize, pointer: usize, value: F) {
        self.write(address_space, pointer, [value]);
    }

    fn read_cell(&mut self, address_space: usize, pointer: usize) -> F {
        self.read::<1>(address_space, pointer)[0]
    }

    fn address_bits(&self) -> usize {
        self.memory.controller.memory_config().pointer_max_bits
    }

    fn last_to_pc(&self) -> F {
        self.execution.last_to_pc()
    }

    fn last_from_pc(&self) -> F {
        self.execution.last_from_pc()
    }

    fn execution_final_state(&self) -> ExecutionState<F> {
        self.execution.records.last().unwrap().final_state
    }

    fn streams_mut(&mut self) -> &mut Streams<F> {
        &mut self.streams
    }

    fn get_default_register(&mut self, increment: usize) -> usize {
        self.default_register += increment;
        self.default_register - increment
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
        self.write(1, register, pointer.to_le_bytes().map(F::from_canonical_u8));
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

impl<F: PrimeField32> VmChipTestBuilder<F> {
    pub fn new(
        controller: MemoryController<F>,
        memory: TracingMemory,
        streams: Streams<F>,
        rng: StdRng,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        internal_rng: StdRng,
    ) -> Self {
        setup_tracing_with_log_level(Level::WARN);
        Self {
            memory: MemoryTester::new(controller, memory),
            streams,
            rng,
            custom_pvs: Vec::new(),
            execution: ExecutionTester::new(execution_bus),
            program: ProgramTester::new(program_bus),
            internal_rng,
            default_register: 0,
            default_pointer: 0,
        }
    }

    fn next_elem_size_u32(&mut self) -> u32 {
        self.internal_rng.next_u32() % (1 << (F::bits() - 2))
    }

    pub fn set_num_public_values(&mut self, num_public_values: usize) {
        self.custom_pvs.resize(num_public_values, None);
    }

    fn write_heap<const NUM_LIMBS: usize>(
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

    pub fn system_port(&self) -> SystemPort {
        SystemPort {
            execution_bus: self.execution.bus,
            program_bus: self.program.bus,
            memory_bridge: self.memory_bridge(),
        }
    }

    pub fn execution_bridge(&self) -> ExecutionBridge {
        ExecutionBridge::new(self.execution.bus, self.program.bus)
    }

    pub fn execution_bus(&self) -> ExecutionBus {
        self.execution.bus
    }

    pub fn program_bus(&self) -> ProgramBus {
        self.program.bus
    }

    pub fn memory_bus(&self) -> MemoryBus {
        self.memory.controller.memory_bus
    }

    pub fn range_checker(&self) -> SharedVariableRangeCheckerChip {
        self.memory.controller.range_checker.clone()
    }

    pub fn memory_bridge(&self) -> MemoryBridge {
        self.memory.controller.memory_bridge()
    }

    pub fn memory_helper(&self) -> SharedMemoryHelper<F> {
        self.memory.controller.helper()
    }
}

// Use Blake3 as hash for faster tests.
pub type TestSC = BabyBearBlake3Config;

impl VmChipTestBuilder<BabyBear> {
    pub fn build(self) -> VmChipTester<TestSC> {
        let tester = VmChipTester {
            memory: Some(self.memory),
            ..Default::default()
        };
        let tester =
            tester.load_periphery((ExecutionDummyAir::new(self.execution.bus), self.execution));
        tester.load_periphery((ProgramDummyAir::new(self.program.bus), self.program))
    }
    pub fn build_babybear_poseidon2(self) -> VmChipTester<BabyBearPoseidon2Config> {
        let tester = VmChipTester {
            memory: Some(self.memory),
            ..Default::default()
        };
        let tester =
            tester.load_periphery((ExecutionDummyAir::new(self.execution.bus), self.execution));
        tester.load_periphery((ProgramDummyAir::new(self.program.bus), self.program))
    }
}

impl<F: PrimeField32> VmChipTestBuilder<F> {
    pub fn default_persistent() -> Self {
        let mut mem_config = MemoryConfig::default();
        mem_config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 1 << 29;
        mem_config.addr_spaces[NATIVE_AS as usize].num_cells = 0;
        Self::persistent(mem_config)
    }

    pub fn default_native() -> Self {
        Self::volatile(MemoryConfig::aggregation())
    }

    fn range_checker_and_memory(
        mem_config: &MemoryConfig,
        init_block_size: usize,
    ) -> (SharedVariableRangeCheckerChip, TracingMemory) {
        let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
            RANGE_CHECKER_BUS,
            mem_config.decomp,
        )));
        let max_access_adapter_n = log2_strict_usize(mem_config.max_access_adapter_n);
        let arena_size_bound = arena_size_bound(&vec![1 << 16; max_access_adapter_n]);
        let memory = TracingMemory::new(mem_config, init_block_size, arena_size_bound);

        (range_checker, memory)
    }

    pub fn persistent(mem_config: MemoryConfig) -> Self {
        setup_tracing_with_log_level(Level::INFO);
        let (range_checker, memory) = Self::range_checker_and_memory(&mem_config, CHUNK);
        let hasher_chip = Arc::new(Poseidon2PeripheryChip::new(
            vm_poseidon2_config(),
            POSEIDON2_DIRECT_BUS,
            3,
        ));
        let memory_controller = MemoryController::with_persistent_memory(
            MemoryBus::new(MEMORY_BUS),
            mem_config,
            range_checker,
            PermutationCheckBus::new(MEMORY_MERKLE_BUS),
            PermutationCheckBus::new(POSEIDON2_DIRECT_BUS),
            hasher_chip,
        );
        Self {
            memory: MemoryTester::new(memory_controller, memory),
            streams: Default::default(),
            rng: StdRng::seed_from_u64(0),
            custom_pvs: Vec::new(),
            execution: ExecutionTester::new(ExecutionBus::new(EXECUTION_BUS)),
            program: ProgramTester::new(ProgramBus::new(READ_INSTRUCTION_BUS)),
            internal_rng: StdRng::seed_from_u64(0),
            default_register: 0,
            default_pointer: 0,
        }
    }

    pub fn volatile(mem_config: MemoryConfig) -> Self {
        setup_tracing_with_log_level(Level::INFO);
        let (range_checker, memory) = Self::range_checker_and_memory(&mem_config, 1);
        let memory_controller = MemoryController::with_volatile_memory(
            MemoryBus::new(MEMORY_BUS),
            mem_config,
            range_checker,
        );
        Self {
            memory: MemoryTester::new(memory_controller, memory),
            streams: Default::default(),
            rng: StdRng::seed_from_u64(0),
            custom_pvs: Vec::new(),
            execution: ExecutionTester::new(ExecutionBus::new(EXECUTION_BUS)),
            program: ProgramTester::new(ProgramBus::new(READ_INSTRUCTION_BUS)),
            internal_rng: StdRng::seed_from_u64(0),
            default_register: 0,
            default_pointer: 0,
        }
    }
}

impl<F: PrimeField32> Default for VmChipTestBuilder<F> {
    fn default() -> Self {
        let mut mem_config = MemoryConfig::default();
        // TODO[jpw]: this is because old tests use `gen_pointer` on address space 1; this can be
        // removed when tests are updated.
        mem_config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 1 << 29;
        mem_config.addr_spaces[NATIVE_AS as usize].num_cells = 0;
        Self::volatile(mem_config)
    }
}

pub struct VmChipTester<SC: StarkGenericConfig> {
    pub memory: Option<MemoryTester<Val<SC>>>,
    pub air_ctxs: Vec<(AirRef<SC>, AirProvingContext<CpuBackend<SC>>)>,
}

impl<SC: StarkGenericConfig> Default for VmChipTester<SC> {
    fn default() -> Self {
        Self {
            memory: None,
            air_ctxs: vec![],
        }
    }
}

impl<SC: StarkGenericConfig> VmChipTester<SC>
where
    Val<SC>: PrimeField32,
{
    pub fn load<E, A, C>(
        mut self,
        harness: TestChipHarness<Val<SC>, E, A, C, MatrixRecordArena<Val<SC>>>,
    ) -> Self
    where
        A: AnyRap<SC> + 'static,
        C: Chip<MatrixRecordArena<Val<SC>>, CpuBackend<SC>>,
    {
        let arena = harness.arena;
        let rows_used = arena.trace_offset.div_ceil(arena.width);
        if rows_used > 0 {
            let air = Arc::new(harness.air) as AirRef<SC>;
            let ctx = harness.chip.generate_proving_ctx(arena);
            tracing::debug!("Generated air proving context for {}", air.name());
            self.air_ctxs.push((air, ctx));
        }

        self
    }

    pub fn load_periphery<A, C>(self, (air, chip): (A, C)) -> Self
    where
        A: AnyRap<SC> + 'static,
        C: Chip<(), CpuBackend<SC>>,
    {
        let air = Arc::new(air) as AirRef<SC>;
        self.load_periphery_ref((air, chip))
    }

    pub fn load_periphery_ref<C>(mut self, (air, chip): (AirRef<SC>, C)) -> Self
    where
        C: Chip<(), CpuBackend<SC>>,
    {
        let ctx = chip.generate_proving_ctx(());
        tracing::debug!("Generated air proving context for {}", air.name());
        self.air_ctxs.push((air, ctx));

        self
    }

    pub fn finalize(mut self) -> Self {
        if let Some(memory_tester) = self.memory.take() {
            let mut memory_controller = memory_tester.controller;
            let is_persistent = memory_controller.continuation_enabled();
            let mut memory = memory_tester.memory;
            let touched_memory = memory.finalize::<Val<SC>>(is_persistent);
            // Balance memory boundaries
            let range_checker = memory_controller.range_checker.clone();
            for mem_chip in memory_tester.chip_for_block.into_values() {
                self = self.load_periphery((mem_chip.air, mem_chip));
            }
            let mem_inventory = MemoryAirInventory::new(
                memory_controller.memory_bridge(),
                memory_controller.memory_config(),
                range_checker.bus(),
                is_persistent.then_some((
                    PermutationCheckBus::new(MEMORY_MERKLE_BUS),
                    PermutationCheckBus::new(POSEIDON2_DIRECT_BUS),
                )),
            );
            let ctxs = memory_controller
                .generate_proving_ctx(memory.access_adapter_records, touched_memory);
            for (air, ctx) in zip_eq(mem_inventory.into_airs(), ctxs)
                .filter(|(_, ctx)| ctx.main_trace_height() > 0)
            {
                self.air_ctxs.push((air, ctx));
            }
            if let Some(hasher_chip) = memory_controller.hasher_chip {
                let air: AirRef<SC> = match hasher_chip.as_ref() {
                    Poseidon2PeripheryChip::Register0(chip) => chip.air.clone(),
                    Poseidon2PeripheryChip::Register1(chip) => chip.air.clone(),
                };
                self = self.load_periphery_ref((air, hasher_chip));
            }
            // this must be last because other trace generation mutates its state
            self = self.load_periphery((range_checker.air, range_checker));
        }
        self
    }

    pub fn load_air_proving_ctx(
        mut self,
        air_proving_ctx: (AirRef<SC>, AirProvingContext<CpuBackend<SC>>),
    ) -> Self {
        self.air_ctxs.push(air_proving_ctx);
        self
    }

    pub fn load_and_prank_trace<E, A, C, P>(
        mut self,
        harness: TestChipHarness<Val<SC>, E, A, C, MatrixRecordArena<Val<SC>>>,
        modify_trace: P,
    ) -> Self
    where
        A: AnyRap<SC> + 'static,
        C: Chip<MatrixRecordArena<Val<SC>>, CpuBackend<SC>>,
        P: Fn(&mut RowMajorMatrix<Val<SC>>),
    {
        let arena = harness.arena;
        let mut ctx = harness.chip.generate_proving_ctx(arena);
        let trace: Arc<RowMajorMatrix<Val<SC>>> = Option::take(&mut ctx.common_main).unwrap();
        let mut trace = Arc::into_inner(trace).unwrap();
        modify_trace(&mut trace);
        ctx.common_main = Some(Arc::new(trace));
        self.air_ctxs.push((Arc::new(harness.air), ctx));
        self
    }

    /// Given a function to produce an engine from the max trace height,
    /// runs a simple test on that engine
    pub fn test<E, P: Fn() -> E>(
        self, // do no take ownership so it's easier to prank
        engine_provider: P,
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    {
        assert!(self.memory.is_none(), "Memory must be finalized");
        let (airs, ctxs): (Vec<_>, Vec<_>) = self.air_ctxs.into_iter().unzip();
        engine_provider().run_test_impl(airs, ctxs)
    }
}

impl VmChipTester<BabyBearPoseidon2Config> {
    pub fn simple_test(
        self,
    ) -> Result<VerificationData<BabyBearPoseidon2Config>, VerificationError> {
        self.test(|| BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1)))
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

impl VmChipTester<BabyBearBlake3Config> {
    pub fn simple_test(self) -> Result<VerificationData<BabyBearBlake3Config>, VerificationError> {
        self.test(|| BabyBearBlake3Engine::new(FriParameters::new_for_testing(1)))
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
