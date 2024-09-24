use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    mem,
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
use itertools::izip;
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
    alu::ArithmeticLogicChip,
    arch::{
        bus::ExecutionBus,
        chips::{InstructionExecutor, InstructionExecutorVariant, MachineChip, MachineChipVariant},
        columns::ExecutionState,
        instructions::{
            Opcode, ALU_256_INSTRUCTIONS, CORE_INSTRUCTIONS, FIELD_ARITHMETIC_INSTRUCTIONS,
            FIELD_EXTENSION_INSTRUCTIONS, SHIFT_256_INSTRUCTIONS, UI_32_INSTRUCTIONS,
        },
    },
    castf::CastFChip,
    core::{
        CoreChip, Streams, BYTE_XOR_BUS, RANGE_CHECKER_BUS, RANGE_TUPLE_CHECKER_BUS,
        READ_INSTRUCTION_BUS,
    },
    ecc::{EcAddUnequalChip, EcDoubleChip},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    memory::{offline_checker::MemoryBus, MemoryChip, MemoryChipRef},
    modular_addsub::{ModularAddSubChip, SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
    modular_multdiv::ModularMultDivChip,
    program::{bridge::ProgramBus, ExecutionError, Program, ProgramChip},
    shift::ShiftChip,
    ui::UiChip,
    uint_multiplication::UintMultiplicationChip,
};

#[derive(Debug)]
pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,

    pub executors: BTreeMap<Opcode, InstructionExecutorVariant<F>>,
    pub chips: Vec<MachineChipVariant<F>>,
    pub core_chip: Rc<RefCell<CoreChip<F>>>,
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
        let program_bus = ProgramBus(READ_INSTRUCTION_BUS);
        let memory_bus = MemoryBus(1);
        let range_bus =
            VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, config.memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let byte_xor_chip = Arc::new(XorLookupChip::new(BYTE_XOR_BUS));

        let memory_chip = Rc::new(RefCell::new(MemoryChip::with_volatile_memory(
            memory_bus,
            config.memory_config,
            range_checker.clone(),
        )));
        let core_chip = Rc::new(RefCell::new(CoreChip::from_state(
            config.core_options(),
            execution_bus,
            program_bus,
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
            MachineChipVariant::Core(core_chip.clone()),
            MachineChipVariant::Program(program_chip.clone()),
        ];

        for opcode in CORE_INSTRUCTIONS {
            executors.insert(opcode, core_chip.clone().into());
        }

        if config.field_arithmetic_enabled {
            let field_arithmetic_chip = Rc::new(RefCell::new(FieldArithmeticChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
            )));
            assign!(FIELD_ARITHMETIC_INSTRUCTIONS, field_arithmetic_chip);
            chips.push(MachineChipVariant::FieldArithmetic(field_arithmetic_chip));
        }
        if config.field_extension_enabled {
            let field_extension_chip = Rc::new(RefCell::new(FieldExtensionArithmeticChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
            )));
            assign!(FIELD_EXTENSION_INSTRUCTIONS, field_extension_chip);
            chips.push(MachineChipVariant::FieldExtension(field_extension_chip))
        }
        if config.perm_poseidon2_enabled || config.compress_poseidon2_enabled {
            let poseidon2_chip = Rc::new(RefCell::new(Poseidon2Chip::from_poseidon2_config(
                Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
                config
                    .poseidon2_max_constraint_degree
                    .expect("Poseidon2 is enabled but no max_constraint_degree provided"),
                execution_bus,
                program_bus,
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
            let keccak_chip = Rc::new(RefCell::new(KeccakVmChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                byte_xor_chip.clone(),
            )));
            assign!([Opcode::KECCAK256], keccak_chip);
            chips.push(MachineChipVariant::Keccak256(keccak_chip));
        }
        if config.modular_addsub_enabled {
            let mod_addsub_coord: ModularAddSubChip<F, 32, 8> = ModularAddSubChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                SECP256K1_COORD_PRIME.clone(),
            );
            let mod_addsub_scalar: ModularAddSubChip<F, 32, 8> = ModularAddSubChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                SECP256K1_SCALAR_PRIME.clone(),
            );
            assign!(
                [Opcode::SECP256K1_COORD_ADD, Opcode::SECP256K1_COORD_SUB],
                Rc::new(RefCell::new(mod_addsub_coord.clone()))
            );
            assign!(
                [Opcode::SECP256K1_SCALAR_ADD, Opcode::SECP256K1_SCALAR_SUB],
                Rc::new(RefCell::new(mod_addsub_scalar.clone()))
            );
        }
        if config.modular_multdiv_enabled {
            let mod_multdiv_coord: ModularMultDivChip<F, 63, 32, 8> = ModularMultDivChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                SECP256K1_COORD_PRIME.clone(),
            );
            let mod_multdiv_scalar: ModularMultDivChip<F, 63, 32, 8> = ModularMultDivChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                SECP256K1_SCALAR_PRIME.clone(),
            );
            assign!(
                [Opcode::SECP256K1_COORD_MUL, Opcode::SECP256K1_COORD_DIV],
                Rc::new(RefCell::new(mod_multdiv_coord.clone()))
            );
            assign!(
                [Opcode::SECP256K1_SCALAR_MUL, Opcode::SECP256K1_SCALAR_DIV],
                Rc::new(RefCell::new(mod_multdiv_scalar.clone()))
            );
        }
        // Modular multiplication also depends on U256 arithmetic.
        if config.modular_multdiv_enabled || config.u256_arithmetic_enabled {
            let u256_chip = Rc::new(RefCell::new(ArithmeticLogicChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                byte_xor_chip.clone(),
            )));
            chips.push(MachineChipVariant::ArithmeticLogicUnit256(
                u256_chip.clone(),
            ));
            assign!(ALU_256_INSTRUCTIONS, u256_chip);
        }
        if config.u256_multiplication_enabled {
            let range_tuple_bus =
                RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, vec![(1 << 8), 32 * (1 << 8)]);
            let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));
            let u256_mult_chip = Rc::new(RefCell::new(UintMultiplicationChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
                range_tuple_checker.clone(),
            )));
            assign!([Opcode::MUL256], u256_mult_chip);
            chips.push(MachineChipVariant::U256Multiplication(u256_mult_chip));
            chips.push(MachineChipVariant::RangeTupleChecker(range_tuple_checker));
        }
        if config.shift_256_enabled {
            let shift_chip = Rc::new(RefCell::new(ShiftChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            assign!(SHIFT_256_INSTRUCTIONS, shift_chip);
            chips.push(MachineChipVariant::Shift256(shift_chip));
        }
        if config.ui_32_enabled {
            let ui_chip = Rc::new(RefCell::new(UiChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
            )));
            assign!(UI_32_INSTRUCTIONS, ui_chip);
            chips.push(MachineChipVariant::Ui(ui_chip));
        }
        if config.castf_enabled {
            let castf_chip = Rc::new(RefCell::new(CastFChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
            )));
            assign!([Opcode::CASTF], castf_chip);
            chips.push(MachineChipVariant::CastF(castf_chip));
        }
        if config.secp256k1_enabled {
            let secp256k1_add_unequal_chip = Rc::new(RefCell::new(EcAddUnequalChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
            )));
            let secp256k1_double_chip = Rc::new(RefCell::new(EcDoubleChip::new(
                execution_bus,
                program_bus,
                memory_chip.clone(),
            )));
            assign!([Opcode::SECP256K1_EC_ADD_NE], secp256k1_add_unequal_chip);
            assign!([Opcode::SECP256K1_EC_DOUBLE], secp256k1_double_chip);
            chips.push(MachineChipVariant::Secp256k1AddUnequal(
                secp256k1_add_unequal_chip,
            ));
            chips.push(MachineChipVariant::Secp256k1Double(secp256k1_double_chip));
        }
        chips.push(MachineChipVariant::ByteXor(byte_xor_chip));
        // Most chips have a reference to the memory chip, and the memory chip has a reference to
        // the range checker chip.
        chips.push(MachineChipVariant::Memory(memory_chip.clone()));
        chips.push(MachineChipVariant::RangeChecker(range_checker.clone()));

        let connector_chip = VmConnectorChip::new(execution_bus);

        Self {
            config,
            executors,
            chips,
            core_chip,
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
        let mut timestamp: usize = self.core_chip.borrow().state.timestamp;
        let mut pc = F::from_canonical_usize(self.core_chip.borrow().state.pc);

        let mut collect_metrics = self.config.collect_metrics;
        // The backtrace for the previous instruction, if any.
        let mut prev_backtrace: Option<Backtrace> = None;

        self.core_chip.borrow_mut().streams = Streams {
            input_stream: self.input_stream.clone(),
            hint_stream: self.hint_stream.clone(),
        };

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

            // runtime only instruction handling
            match opcode {
                Opcode::CT_START => {
                    self.update_chip_metrics();
                    self.cycle_tracker
                        .start(instruction.debug.clone(), self.collected_metrics.clone())
                }
                Opcode::CT_END => {
                    self.update_chip_metrics();
                    self.cycle_tracker
                        .end(instruction.debug.clone(), self.collected_metrics.clone())
                }
                _ => {}
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
                self.collected_metrics
                    .opcode_counts
                    .entry(opcode.to_string())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                if !dsl_instr.is_empty() {
                    self.collected_metrics
                        .dsl_counts
                        .entry(dsl_instr)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }

                self.collected_metrics
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
                #[cfg(feature = "bench-metrics")]
                metrics::counter!("total_cells_used").absolute(self.current_trace_cells() as u64);
            }
            if opcode == Opcode::TERMINATE {
                break;
            }
            if self.should_segment() {
                panic!("continuations not supported");
            }
        }

        self.connector_chip
            .end(ExecutionState::new(pc, F::from_canonical_usize(timestamp)));

        let streams = mem::take(&mut self.core_chip.borrow_mut().streams);
        self.hint_stream = streams.hint_stream;
        self.input_stream = streams.input_stream;

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
        drop(self.core_chip);
        drop(self.program_chip);
        drop(self.memory_chip);

        for mut chip in self.chips {
            let heights = chip.current_trace_heights();
            let airs = chip.airs();
            let public_values = chip.generate_public_values_per_air();
            let traces = chip.generate_traces();

            for (height, air, public_values, trace) in izip!(heights, airs, public_values, traces) {
                if height != 0 {
                    result.airs.push(air);
                    result.public_values.push(public_values);
                    result.traces.push(trace);
                }
            }
        }
        let trace = self.connector_chip.generate_trace();
        result.airs.push(Box::new(self.connector_chip.air));
        result.public_values.push(vec![]);
        result.traces.push(trace);

        result
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of Core trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    fn should_segment(&mut self) -> bool {
        self.chips.iter().any(|chip| {
            chip.current_trace_heights()
                .iter()
                .any(|height| *height > self.config.max_segment_len)
        })
    }

    fn current_trace_cells(&self) -> usize {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_cells().into_iter().sum::<usize>())
            .sum()
    }

    pub(crate) fn update_chip_metrics(&mut self) {
        self.collected_metrics.chip_heights = self.chip_heights();
    }

    fn chip_heights(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        for chip in self.chips.iter() {
            let chip_name: &'static str = chip.into();
            for (i, height) in chip.current_trace_heights().iter().enumerate() {
                if i == 0 {
                    metrics.insert(chip_name.into(), *height);
                } else {
                    metrics.insert(format!("{} {}", chip_name, i + 1), *height);
                }
            }
        }
        metrics
    }
}
