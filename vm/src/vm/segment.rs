use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::{
    range_tuple::{bus::RangeTupleCheckerBus, RangeTupleCheckerChip},
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
    xor::lookup::XorLookupChip,
};
use afs_stark_backend::rap::AnyRap;
use backtrace::Backtrace;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::Poseidon2Config;

use super::{
    connector::VmConnectorChip, cycle_tracker::CycleTracker, VirtualMachineState, VmConfig,
    VmCycleTracker, VmMetrics,
};
use crate::{
    arch::{
        bus::ExecutionBus,
        chips::{InstructionExecutor, InstructionExecutorVariant, MachineChip, MachineChipVariant},
        columns::ExecutionState,
        instructions::{
            Opcode, CORE_INSTRUCTIONS, FIELD_ARITHMETIC_INSTRUCTIONS, FIELD_EXTENSION_INSTRUCTIONS,
            UINT256_ARITHMETIC_INSTRUCTIONS,
        },
    },
    castf::CastFChip,
    cpu::{CpuChip, StreamsAndMetrics, BYTE_XOR_BUS, RANGE_CHECKER_BUS, RANGE_TUPLE_CHECKER_BUS},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    memory::{offline_checker::MemoryBus, MemoryChip, MemoryChipRef},
    modular_arithmetic::{
        ModularArithmeticChip, ModularArithmeticOp, SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME,
    },
    program::{ExecutionError, Program, ProgramChip},
    uint_arithmetic::UintArithmeticChip,
    uint_multiplication::UintMultiplicationChip,
};

#[derive(Debug)]
pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,

    pub executors: BTreeMap<Opcode, InstructionExecutorVariant<F>>,
    pub chips: Vec<MachineChipVariant<F>>,
    pub cpu_chip: Rc<RefCell<CpuChip<F>>>,
    pub program_chip: Rc<RefCell<ProgramChip<F>>>,
    pub memory_chip: MemoryChipRef<F>,
    pub connector_chip: VmConnectorChip<F>,

    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,

    pub cycle_tracker: VmCycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
}

pub struct SegmentResult<SC: StarkGenericConfig> {
    pub airs: Vec<Box<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    pub public_values: Vec<Vec<Val<SC>>>,

    pub metrics: VmMetrics,
}

impl<SC: StarkGenericConfig> SegmentResult<SC> {
    pub fn max_log_degree(&self) -> usize {
        self.traces
            .iter()
            .map(RowMajorMatrix::height)
            .map(log2_strict_usize)
            .max()
            .unwrap()
    }
}

impl<F: PrimeField32> ExecutionSegment<F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(config: VmConfig, program: Program<F>, state: VirtualMachineState<F>) -> Self {
        let execution_bus = ExecutionBus(0);
        let memory_bus = MemoryBus(1);
        let range_bus =
            VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, config.memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let memory_chip = Rc::new(RefCell::new(MemoryChip::with_volatile_memory(
            memory_bus,
            config.memory_config,
            range_checker.clone(),
        )));
        let cpu_chip = Rc::new(RefCell::new(CpuChip::from_state(
            config.cpu_options(),
            execution_bus,
            memory_chip.clone(),
            state.state,
        )));
        let program_chip = Rc::new(RefCell::new(ProgramChip::new(program)));

        let mut executors: BTreeMap<Opcode, InstructionExecutorVariant<F>> = BTreeMap::new();
        macro_rules! assign {
            ($opcodes: expr, $executor: expr) => {
                for opcode in $opcodes {
                    executors.insert(opcode, $executor.clone().into());
                }
            };
        }

        // NOTE: The order of entries in `chips` must be a linear extension of the dependency DAG.
        // That is, if chip A holds a strong reference to chip B, then A must precede B in `chips`.

        let mut chips = vec![
            MachineChipVariant::Cpu(cpu_chip.clone()),
            MachineChipVariant::Program(program_chip.clone()),
        ];

        for opcode in CORE_INSTRUCTIONS {
            executors.insert(opcode, cpu_chip.clone().into());
        }

        if config.field_arithmetic_enabled {
            let field_arithmetic_chip = Rc::new(RefCell::new(FieldArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            assign!(FIELD_ARITHMETIC_INSTRUCTIONS, field_arithmetic_chip);
            chips.push(MachineChipVariant::FieldArithmetic(field_arithmetic_chip));
        }
        if config.field_extension_enabled {
            let field_extension_chip = Rc::new(RefCell::new(FieldExtensionArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            assign!(FIELD_EXTENSION_INSTRUCTIONS, field_extension_chip);
            chips.push(MachineChipVariant::FieldExtension(field_extension_chip))
        }
        if config.perm_poseidon2_enabled || config.compress_poseidon2_enabled {
            let poseidon2_chip = Rc::new(RefCell::new(Poseidon2Chip::from_poseidon2_config(
                Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
                execution_bus,
                memory_chip.clone(),
            )));
            if config.perm_poseidon2_enabled {
                assign!([Opcode::PERM_POS2], poseidon2_chip);
            }
            if config.compress_poseidon2_enabled {
                assign!([Opcode::COMP_POS2], poseidon2_chip);
            }
            chips.push(MachineChipVariant::Poseidon2(poseidon2_chip.clone()));
        }
        if config.keccak_enabled {
            let byte_xor_chip = Arc::new(XorLookupChip::new(BYTE_XOR_BUS));
            let keccak_chip = Rc::new(RefCell::new(KeccakVmChip::new(
                execution_bus,
                memory_chip.clone(),
                byte_xor_chip.clone(),
            )));
            assign!([Opcode::KECCAK256], keccak_chip);
            chips.push(MachineChipVariant::Keccak256(keccak_chip));
            chips.push(MachineChipVariant::ByteXor(byte_xor_chip));
        }
        if config.modular_multiplication_enabled {
            let add_coord = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_COORD_PRIME.clone(),
                ModularArithmeticOp::Add,
            );
            let add_scalar = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_SCALAR_PRIME.clone(),
                ModularArithmeticOp::Add,
            );
            let sub_coord = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_COORD_PRIME.clone(),
                ModularArithmeticOp::Sub,
            );
            let sub_scalar = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_SCALAR_PRIME.clone(),
                ModularArithmeticOp::Sub,
            );
            let mul_coord = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_COORD_PRIME.clone(),
                ModularArithmeticOp::Mul,
            );
            let mul_scalar = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_SCALAR_PRIME.clone(),
                ModularArithmeticOp::Mul,
            );
            let div_coord = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_COORD_PRIME.clone(),
                ModularArithmeticOp::Div,
            );
            let div_scalar = ModularArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
                SECP256K1_SCALAR_PRIME.clone(),
                ModularArithmeticOp::Div,
            );
            assign!(
                [Opcode::SECP256K1_COORD_ADD],
                Rc::new(RefCell::new(add_coord.clone()))
            );
            assign!(
                [Opcode::SECP256K1_SCALAR_ADD],
                Rc::new(RefCell::new(add_scalar.clone()))
            );
            assign!(
                [Opcode::SECP256K1_COORD_SUB],
                Rc::new(RefCell::new(sub_coord.clone()))
            );
            assign!(
                [Opcode::SECP256K1_SCALAR_SUB],
                Rc::new(RefCell::new(sub_scalar.clone()))
            );
            assign!(
                [Opcode::SECP256K1_COORD_MUL],
                Rc::new(RefCell::new(mul_coord.clone()))
            );
            assign!(
                [Opcode::SECP256K1_SCALAR_MUL],
                Rc::new(RefCell::new(mul_scalar.clone()))
            );
            assign!(
                [Opcode::SECP256K1_COORD_DIV],
                Rc::new(RefCell::new(div_coord.clone()))
            );
            assign!(
                [Opcode::SECP256K1_SCALAR_DIV],
                Rc::new(RefCell::new(div_scalar.clone()))
            );
        }
        // Modular multiplication also depends on U256 arithmetic.
        if config.modular_multiplication_enabled || config.u256_arithmetic_enabled {
            let u256_chip = Rc::new(RefCell::new(UintArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            chips.push(MachineChipVariant::U256Arithmetic(u256_chip.clone()));
            assign!(UINT256_ARITHMETIC_INSTRUCTIONS, u256_chip);
        }
        if config.u256_multiplication_enabled {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, vec![(1 << 8), 32 * (1 << 8)]);
            let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));
            let u256_mult_chip = Rc::new(RefCell::new(UintMultiplicationChip::new(
                execution_bus,
                memory_chip.clone(),
                range_tuple_checker.clone(),
            )));
            assign!([Opcode::MUL256], u256_mult_chip);
            chips.push(MachineChipVariant::U256Multiplication(u256_mult_chip));
            chips.push(MachineChipVariant::RangeTupleChecker(range_tuple_checker));
        }
        if config.castf_enabled {
            let castf_chip = Rc::new(RefCell::new(CastFChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            assign!([Opcode::CASTF], castf_chip);
            chips.push(MachineChipVariant::CastF(castf_chip));
        }
        // Most chips have a reference to the memory chip, and the memory chip has a reference to
        // the range checker chip.
        chips.push(MachineChipVariant::Memory(memory_chip.clone()));
        chips.push(MachineChipVariant::RangeChecker(range_checker.clone()));

        let connector_chip = VmConnectorChip::new(execution_bus);

        Self {
            config,
            executors,
            chips,
            cpu_chip,
            program_chip,
            memory_chip,
            connector_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream.clone(),
            collected_metrics: Default::default(),
            cycle_tracker: CycleTracker::new(),
        }
    }

    /// Stopping is triggered by should_segment()
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        let mut timestamp: usize = self.cpu_chip.borrow().state.timestamp;
        let mut pc = F::from_canonical_usize(self.cpu_chip.borrow().state.pc);

        let mut collect_metrics = self.config.collect_metrics;
        // The backtrace for the previous instruction, if any.
        let mut prev_backtrace: Option<Backtrace> = None;

        self.cpu_chip.borrow_mut().streams_and_metrics = Some(StreamsAndMetrics {
            input_stream: self.input_stream.clone(),
            hint_stream: self.hint_stream.clone(),
            cycle_tracker: self.cycle_tracker.clone(),
            collected_metrics: self.collected_metrics.clone(),
        });

        self.connector_chip
            .begin(ExecutionState::new(pc, F::from_canonical_usize(timestamp)));

        loop {
            let pc_usize = pc.as_canonical_u64() as usize;

            let (instruction, debug_info) =
                RefCell::borrow_mut(&self.program_chip).get_instruction(pc_usize)?;
            tracing::trace!("pc: {pc_usize} | time: {timestamp} | {:?}", instruction);

            let dsl_instr = match &debug_info {
                Some(debug_info) => debug_info.dsl_instruction.to_string(),
                None => String::new(),
            };

            let opcode = instruction.opcode;

            let next_pc;

            let prev_trace_cells = self.current_trace_cells();

            if opcode == Opcode::FAIL {
                if let Some(mut backtrace) = prev_backtrace {
                    backtrace.resolve();
                    eprintln!("eDSL program failure; backtrace:\n{:?}", backtrace);
                } else {
                    eprintln!("eDSL program failure; no backtrace");
                }
                return Err(ExecutionError::Fail(pc_usize));
            }

            if self.executors.contains_key(&opcode) {
                let executor = self.executors.get_mut(&opcode).unwrap();
                match InstructionExecutor::execute(
                    executor,
                    instruction,
                    ExecutionState::new(pc_usize, timestamp),
                ) {
                    Ok(next_state) => {
                        next_pc = F::from_canonical_usize(next_state.pc);
                        timestamp = next_state.timestamp;
                    }
                    Err(e) => return Err(e),
                }
            } else {
                return Err(ExecutionError::DisabledOperation(pc_usize, opcode));
            }

            let now_trace_cells = self.current_trace_cells();
            let added_trace_cells = now_trace_cells - prev_trace_cells;

            if collect_metrics {
                let mut cpu_chip = self.cpu_chip.borrow_mut();
                let collected_metrics = &mut cpu_chip
                    .streams_and_metrics
                    .as_mut()
                    .unwrap()
                    .collected_metrics;
                collected_metrics
                    .opcode_counts
                    .entry(opcode.to_string())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                if !dsl_instr.is_empty() {
                    collected_metrics
                        .dsl_counts
                        .entry(dsl_instr)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }

                collected_metrics
                    .opcode_trace_cells
                    .entry(opcode.to_string())
                    .and_modify(|count| *count += added_trace_cells)
                    .or_insert(added_trace_cells);
            }

            prev_backtrace = debug_info.and_then(|debug_info| debug_info.trace);

            pc = next_pc;

            // clock_cycle += 1;
            if opcode == Opcode::TERMINATE && collect_metrics {
                self.update_chip_metrics();
                // Due to row padding, the padded rows will all have opcode TERMINATE, so stop metric collection after the first one
                collect_metrics = false;
            }
            if opcode == Opcode::TERMINATE
            // && vm
            //     .cpu_chip
            //     .borrow()
            //     .current_trace_height()
            //     .is_power_of_two()
            {
                // is_done = true;
                break;
            }
            if self.should_segment() {
                panic!("continuations not supported");
                // break
            }
        }

        self.connector_chip
            .end(ExecutionState::new(pc, F::from_canonical_usize(timestamp)));

        let streams_and_ct_refs = self
            .cpu_chip
            .borrow_mut()
            .streams_and_metrics
            .take()
            .unwrap();
        self.hint_stream = streams_and_ct_refs.hint_stream;
        self.input_stream = streams_and_ct_refs.input_stream;
        self.collected_metrics = streams_and_ct_refs.collected_metrics;
        self.cycle_tracker = streams_and_ct_refs.cycle_tracker;

        if collect_metrics {
            self.update_chip_metrics();
        }

        Ok(())
    }

    /// Compile the AIRs and trace generation outputs for the chips used in this segment
    /// Should be called after ::execute
    pub fn produce_result<SC: StarkGenericConfig>(self) -> SegmentResult<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut result = SegmentResult {
            airs: vec![],
            traces: vec![],
            public_values: vec![],
            metrics: self.collected_metrics,
        };

        // Drop all strong references to chips other than self.chips, which will be consumed next.
        drop(self.executors);
        drop(self.cpu_chip);
        drop(self.program_chip);
        drop(self.memory_chip);

        for mut chip in self.chips {
            if chip.current_trace_height() != 0 {
                result.airs.push(chip.air());
                result.public_values.push(chip.generate_public_values());
                result.traces.push(chip.generate_trace());
            }
        }
        let trace = self.connector_chip.generate_trace();
        result.airs.push(Box::new(self.connector_chip.air));
        result.public_values.push(vec![]);
        result.traces.push(trace);

        result
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of CPU trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    fn should_segment(&mut self) -> bool {
        self.chips
            .iter()
            .any(|chip| chip.current_trace_height() > self.config.max_segment_len)
    }

    fn current_trace_cells(&self) -> usize {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_cells())
            .sum()
    }

    pub(crate) fn update_chip_metrics(&mut self) {
        self.collected_metrics.chip_metrics = self.chip_metrics();
    }

    fn chip_metrics(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        for chip in self.chips.iter() {
            let chip_name: &'static str = chip.into();
            metrics.insert(chip_name.into(), chip.current_trace_height());
        }
        metrics
    }
}
