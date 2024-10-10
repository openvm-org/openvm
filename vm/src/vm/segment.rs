use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    mem,
    ops::DerefMut,
    rc::Rc,
    sync::Arc,
};

use afs_primitives::{
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
    xor::lookup::XorLookupChip,
};
use afs_stark_backend::utils::AirInfo;
use backtrace::Backtrace;
use itertools::{izip, Itertools};
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use p3_uni_stark::{Domain, StarkGenericConfig};
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::Poseidon2Config;
use strum::EnumCount;

use super::{
    connector::VmConnectorChip, cycle_tracker::CycleTracker, VirtualMachineState, VmConfig,
    VmMetrics,
};
use crate::{
    alu::ArithmeticLogicChip,
    arch::{
        instructions::*, ExecutionBus, ExecutionState, ExecutorName, InstructionExecutor,
        InstructionExecutorVariant, MachineChip, MachineChipVariant, Rv32AluAdapter,
        Rv32BranchAdapter, Rv32LoadStoreAdapter, Rv32MultAdapter, Rv32RdWriteAdapter,
    },
    branch_eq::{BranchEqualIntegration, Rv32BranchEqualChip},
    branch_lt::{BranchLessThanIntegration, Rv32BranchLessThanChip},
    castf::CastFChip,
    core::{
        CoreChip, Streams, BYTE_XOR_BUS, RANGE_CHECKER_BUS, RANGE_TUPLE_CHECKER_BUS,
        READ_INSTRUCTION_BUS,
    },
    ecc::{EcAddUnequalChip, EcDoubleChip},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    loadstore::{LoadStoreIntegration, Rv32LoadStoreChip},
    memory::{offline_checker::MemoryBus, MemoryChip, MemoryChipRef},
    modular_addsub::ModularAddSubChip,
    modular_multdiv::ModularMultDivChip,
    new_alu::{ArithmeticLogicIntegration, Rv32ArithmeticLogicChip},
    new_divrem::{DivRemIntegration, Rv32DivRemChip},
    new_lt::{LessThanIntegration, Rv32LessThanChip},
    new_mul::{MultiplicationIntegration, Rv32MultiplicationChip},
    new_mulh::{MulHIntegration, Rv32MulHChip},
    new_shift::{Rv32ShiftChip, ShiftIntegration},
    program::{bridge::ProgramBus, DebugInfo, ExecutionError, Program, ProgramChip},
    rv32_auipc::{Rv32AuipcChip, Rv32AuipcIntegration},
    rv32_jal_lui::{Rv32JalLuiChip, Rv32JalLuiIntegration},
    shift::ShiftChip,
    ui::UiChip,
    uint_multiplication::UintMultiplicationChip,
    vm::config::PersistenceType,
};

#[derive(Debug)]
pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChipRef<F>,
    pub connector_chip: VmConnectorChip<F>,
    pub persistent_memory_hasher: Option<Rc<RefCell<Poseidon2Chip<F>>>>,

    pub executors: BTreeMap<usize, InstructionExecutorVariant<F>>,
    pub chips: Vec<MachineChipVariant<F>>,
    // FIXME: remove this
    pub core_chip: Rc<RefCell<CoreChip<F>>>,

    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,

    pub cycle_tracker: CycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
}

pub struct SegmentResult<SC: StarkGenericConfig> {
    pub air_infos: Vec<AirInfo<SC>>,
    pub metrics: VmMetrics,
}

impl<SC: StarkGenericConfig> SegmentResult<SC> {
    pub fn max_log_degree(&self) -> usize {
        self.air_infos
            .iter()
            .map(|air_info| air_info.common_trace.height())
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
        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, [(1 << 8), 32 * (1 << 8)]);
        let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));

        let memory_chip = Rc::new(RefCell::new(MemoryChip::new(
            memory_bus,
            config.memory_config.clone(),
            range_checker.clone(),
        )));
        let program_chip = ProgramChip::new(program);

        let mut executors: BTreeMap<usize, InstructionExecutorVariant<F>> = BTreeMap::new();

        // NOTE: The order of entries in `chips` must be a linear extension of the dependency DAG.
        // That is, if chip A holds a strong reference to chip B, then A must precede B in `chips`.

        let mut chips = vec![];

        let mut modular_muldiv_chips = vec![];
        let mut modular_addsub_chips = vec![];
        let mut core_chip = if config
            .executors
            .iter()
            .any(|(_, executor, _)| matches!(executor, ExecutorName::Core))
        {
            None
        } else {
            let offset = 0; // Core offset should be 0 by default; maybe it makes sense to make this always the case?
            let chip = Rc::new(RefCell::new(CoreChip::from_state(
                config.core_options(),
                execution_bus,
                program_bus,
                memory_chip.clone(),
                state.state,
                offset,
            )));
            let range = 0..CoreOpcode::COUNT;
            for opcode in range {
                executors.insert(offset + opcode, chip.clone().into());
            }
            chips.push(MachineChipVariant::Core(chip.clone()));
            Some(chip)
        };

        // We may not use this chip if the memory kind is volatile and there is no executor for Poseidon2.
        let poseidon_chip = {
            let offset = config
                .executors
                .iter()
                .find(|(_, name, _)| *name == ExecutorName::Poseidon2)
                .map(|(_, _, offset)| *offset)
                .unwrap_or(0); // If no Poseidon2 executor, offset doesn't matter.

            Rc::new(RefCell::new(Poseidon2Chip::from_poseidon2_config(
                Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
                config.poseidon2_max_constraint_degree,
                execution_bus,
                program_bus,
                memory_chip.clone(),
                offset,
            )))
        };

        for (range, executor, offset) in config.clone().executors {
            for opcode in range.clone() {
                if let Some(old_executor) = executors.get(&opcode) {
                    panic!(
                        "Attempting to override an executor for opcode {} ({:?} -> {:?})",
                        opcode, old_executor, executor
                    );
                }
            }
            match executor {
                ExecutorName::Core => {
                    assert!(core_chip.is_none(), "Core chip already initialized");
                    core_chip = Some(Rc::new(RefCell::new(CoreChip::from_state(
                        config.core_options(),
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        state.state,
                        offset,
                    ))));
                    for opcode in range {
                        executors.insert(opcode, core_chip.clone().unwrap().into());
                    }
                    chips.push(MachineChipVariant::Core(core_chip.clone().unwrap()));
                }
                ExecutorName::FieldArithmetic => {
                    let chip = Rc::new(RefCell::new(FieldArithmeticChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::FieldArithmetic(chip));
                }
                ExecutorName::FieldExtension => {
                    let chip = Rc::new(RefCell::new(FieldExtensionArithmeticChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::FieldExtension(chip));
                }
                ExecutorName::Poseidon2 => {
                    for opcode in range {
                        executors.insert(opcode, poseidon_chip.clone().into());
                    }
                    chips.push(MachineChipVariant::Poseidon2(poseidon_chip.clone()));
                }
                ExecutorName::Keccak256 => {
                    let chip = Rc::new(RefCell::new(KeccakVmChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        byte_xor_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::Keccak256(chip));
                }
                ExecutorName::ArithmeticLogicUnitRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32ArithmeticLogicChip::new(
                        Rv32AluAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        ArithmeticLogicIntegration::new(byte_xor_chip.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::ArithmeticLogicUnitRv32(chip));
                }
                ExecutorName::ArithmeticLogicUnit256 => {
                    // We probably must include this chip if we include any modular arithmetic,
                    // not sure if we need to enforce this here.
                    let chip = Rc::new(RefCell::new(ArithmeticLogicChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        byte_xor_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::ArithmeticLogicUnit256(chip));
                }
                ExecutorName::LessThanRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LessThanChip::new(
                        Rv32AluAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        LessThanIntegration::new(byte_xor_chip.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::LessThanRv32(chip));
                }
                ExecutorName::MultiplicationRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32MultiplicationChip::new(
                        Rv32MultAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        MultiplicationIntegration::new(range_tuple_checker.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::MultiplicationRv32(chip));
                }
                ExecutorName::MultiplicationHighRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32MulHChip::new(
                        Rv32MultAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        MulHIntegration::new(range_tuple_checker.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::MultiplicationHighRv32(chip));
                }
                ExecutorName::U256Multiplication => {
                    let chip = Rc::new(RefCell::new(UintMultiplicationChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        range_tuple_checker.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::U256Multiplication(chip));
                }
                ExecutorName::DivRemRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32DivRemChip::new(
                        Rv32MultAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        DivRemIntegration::new(range_tuple_checker.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::DivRemRv32(chip));
                }
                ExecutorName::ShiftRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32ShiftChip::new(
                        Rv32AluAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        ShiftIntegration::new(byte_xor_chip.clone(), range_checker.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::ShiftRv32(chip));
                }
                ExecutorName::Shift256 => {
                    let chip = Rc::new(RefCell::new(ShiftChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        byte_xor_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::Shift256(chip));
                }
                ExecutorName::LoadStoreRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LoadStoreChip::new(
                        Rv32LoadStoreAdapter::new(range_checker.clone(), offset),
                        LoadStoreIntegration::new(offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::LoadStoreRv32(chip));
                }
                ExecutorName::BranchEqualRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BranchEqualChip::new(
                        Rv32BranchAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        BranchEqualIntegration::new(offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::BranchEqualRv32(chip));
                }
                ExecutorName::BranchLessThanRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BranchLessThanChip::new(
                        Rv32BranchAdapter::new(execution_bus, program_bus, memory_chip.clone()),
                        BranchLessThanIntegration::new(byte_xor_chip.clone(), offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::BranchLessThanRv32(chip));
                }
                ExecutorName::JalLuiRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32JalLuiChip::new(
                        Rv32RdWriteAdapter::new(),
                        Rv32JalLuiIntegration::new(offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::JalLuiRv32(chip));
                }
                ExecutorName::AuipcRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32AuipcChip::new(
                        Rv32RdWriteAdapter::new(),
                        Rv32AuipcIntegration::new(offset),
                        memory_chip.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::AuipcRv32(chip));
                }
                ExecutorName::Ui => {
                    let chip = Rc::new(RefCell::new(UiChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::Ui(chip));
                }
                ExecutorName::CastF => {
                    let chip = Rc::new(RefCell::new(CastFChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::CastF(chip));
                }
                ExecutorName::Secp256k1AddUnequal => {
                    let chip = Rc::new(RefCell::new(EcAddUnequalChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::Secp256k1AddUnequal(chip));
                }
                ExecutorName::Secp256k1Double => {
                    let chip = Rc::new(RefCell::new(EcDoubleChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(MachineChipVariant::Secp256k1Double(chip));
                }
                ExecutorName::ModularAddSub | ExecutorName::ModularMultDiv => {
                    unreachable!("Modular executors should be handled differently")
                }
            }
        }

        for (range, executor, offset, modulus) in config.clone().modular_executors {
            for opcode in range.clone() {
                if let Some(old_executor) = executors.get(&opcode) {
                    panic!(
                        "Attempting to override an executor for opcode {} ({:?} -> {:?})",
                        opcode, old_executor, executor
                    );
                }
            }
            match executor {
                ExecutorName::ModularAddSub => {
                    let new_chip = Rc::new(RefCell::new(ModularAddSubChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        modulus,
                        offset,
                    )));
                    modular_addsub_chips.push(new_chip.clone());
                    for opcode in range {
                        executors.insert(opcode, new_chip.clone().into());
                    }
                }
                ExecutorName::ModularMultDiv => {
                    let new_chip = Rc::new(RefCell::new(ModularMultDivChip::new(
                        execution_bus,
                        program_bus,
                        memory_chip.clone(),
                        modulus,
                        offset,
                    )));
                    modular_muldiv_chips.push(new_chip.clone());
                    for opcode in range {
                        executors.insert(opcode, new_chip.clone().into());
                    }
                }
                _ => unreachable!(
                    "modular_executors should only contain ModularAddSub and ModularMultDiv"
                ),
            }
        }

        chips.push(MachineChipVariant::ByteXor(byte_xor_chip));
        chips.push(MachineChipVariant::RangeTupleChecker(range_tuple_checker));
        // Most chips have a reference to the memory chip, and the memory chip has a reference to
        // the range checker chip.
        chips.push(MachineChipVariant::Memory(memory_chip.clone()));
        chips.push(MachineChipVariant::RangeChecker(range_checker.clone()));

        let connector_chip = VmConnectorChip::new(execution_bus);

        let persistent_memory_hasher = match config.memory_config.persistence_type {
            PersistenceType::Persistent => Some(poseidon_chip),
            PersistenceType::Volatile => None,
        };

        Self {
            config,
            executors,
            chips,
            core_chip: core_chip.unwrap(),
            program_chip,
            memory_chip,
            persistent_memory_hasher,
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

            let (instruction, debug_info) = self.program_chip.get_instruction(pc_usize)?;
            tracing::trace!("pc: {pc_usize} | time: {timestamp} | {:?}", instruction);

            let (dsl_instr, trace) = debug_info.map_or(
                (None, None),
                |DebugInfo {
                     dsl_instruction,
                     trace,
                 }| (Some(dsl_instruction), trace),
            );

            let opcode = instruction.opcode;
            let prev_trace_cells = self.current_trace_cells();

            // runtime only instruction handling
            // FIXME: assumes CoreOpcode has offset 0:
            if opcode == CoreOpcode::FAIL as usize {
                if let Some(mut backtrace) = prev_backtrace {
                    backtrace.resolve();
                    eprintln!("eDSL program failure; backtrace:\n{:?}", backtrace);
                } else {
                    eprintln!("eDSL program failure; no backtrace");
                }
                return Err(ExecutionError::Fail(pc_usize));
            }
            if opcode == CoreOpcode::CT_START as usize {
                self.update_chip_metrics();
                self.cycle_tracker.start(instruction.debug.clone())
            }
            if opcode == CoreOpcode::CT_END as usize {
                self.update_chip_metrics();
                self.cycle_tracker.end(instruction.debug.clone())
            }
            prev_backtrace = trace;

            let mut opcode_name = None;
            if self.executors.contains_key(&opcode) {
                let executor = self.executors.get_mut(&opcode).unwrap();
                match InstructionExecutor::execute(
                    executor,
                    instruction,
                    ExecutionState::new(pc_usize, timestamp),
                ) {
                    Ok(next_state) => {
                        pc = F::from_canonical_usize(next_state.pc);
                        timestamp = next_state.timestamp;
                    }
                    Err(e) => return Err(e),
                }
                if collect_metrics {
                    opcode_name = Some(executor.get_opcode_name(opcode));
                }
            } else {
                return Err(ExecutionError::DisabledOperation(pc_usize, opcode));
            }

            if collect_metrics {
                let now_trace_cells = self.current_trace_cells();

                let opcode_name = opcode_name.unwrap_or(opcode.to_string());
                let key = (dsl_instr.clone(), opcode_name.clone());
                #[cfg(feature = "bench-metrics")]
                self.cycle_tracker.increment_opcode(&key);
                *self.collected_metrics.counts.entry(key).or_insert(0) += 1;

                for (air_name, now_value) in &now_trace_cells {
                    let prev_value = prev_trace_cells.get(air_name).unwrap_or(&0);
                    if prev_value != now_value {
                        let key = (dsl_instr.clone(), opcode_name.clone(), air_name.to_owned());
                        #[cfg(feature = "bench-metrics")]
                        self.cycle_tracker
                            .increment_cells_used(&key, now_value - prev_value);
                        *self.collected_metrics.trace_cells.entry(key).or_insert(0) +=
                            now_value - prev_value;
                    }
                }
                if opcode == CoreOpcode::TERMINATE as usize {
                    self.update_chip_metrics();
                    // Due to row padding, the padded rows will all have opcode TERMINATE, so stop metric collection after the first one
                    collect_metrics = false;
                    #[cfg(feature = "bench-metrics")]
                    metrics::counter!("total_cells_used")
                        .absolute(now_trace_cells.into_values().sum::<usize>() as u64);
                }
            }
            if opcode == CoreOpcode::TERMINATE as usize {
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
        // Finalize memory.
        if let Some(hasher) = &self.persistent_memory_hasher {
            let mut hasher = hasher.borrow_mut();
            self.memory_chip
                .borrow_mut()
                .finalize(Some(hasher.deref_mut()));
        } else {
            self.memory_chip
                .borrow_mut()
                .finalize(None::<&mut Poseidon2Chip<F>>);
        }

        // Drop all strong references to chips other than self.chips, which will be consumed next.
        drop(self.executors);
        drop(self.core_chip);
        drop(self.persistent_memory_hasher);
        drop(self.memory_chip);

        let mut result = SegmentResult {
            air_infos: vec![self.program_chip.into()],
            metrics: self.collected_metrics,
        };

        for mut chip in self.chips {
            let heights = chip.current_trace_heights();
            let airs = chip.airs();
            let public_values = chip.generate_public_values_per_air();
            let traces = chip.generate_traces();

            for (height, air, public_values, trace) in izip!(heights, airs, public_values, traces) {
                if height != 0 {
                    result
                        .air_infos
                        .push(AirInfo::simple(air, trace, public_values));
                }
            }
        }
        let trace = self.connector_chip.generate_trace();
        result.air_infos.push(AirInfo::simple_no_pis(
            Box::new(self.connector_chip.air),
            trace,
        ));

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

    fn current_trace_cells(&self) -> BTreeMap<String, usize> {
        self.chips
            .iter()
            .flat_map(|chip| {
                chip.air_names()
                    .into_iter()
                    .zip_eq(chip.current_trace_cells())
            })
            .collect()
    }

    pub(crate) fn update_chip_metrics(&mut self) {
        self.collected_metrics.chip_heights = self.chip_heights();
    }

    fn chip_heights(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        metrics.insert("ProgramChip".into(), self.program_chip.true_program_length);
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
