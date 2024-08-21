use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use poseidon2_air::poseidon2::Poseidon2Config;

use super::{VirtualMachineState, VmConfig, VmMetrics};
use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::{MachineChip, MachineChipVariant, OpCodeExecutorVariant},
        instructions::{OpCode, FIELD_ARITHMETIC_INSTRUCTIONS, FIELD_EXTENSION_INSTRUCTIONS},
    },
    cpu::{trace::ExecutionError, CpuChip, POSEIDON2_DIRECT_BUS, RANGE_CHECKER_BUS},
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    poseidon2::Poseidon2Chip,
    program::{Program, ProgramChip},
    vm::cycle_tracker::CycleTracker,
};

pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,

    pub executors: BTreeMap<OpCode, OpCodeExecutorVariant<F>>,
    pub chips: Vec<MachineChipVariant<F>>,
    pub cpu_chip: Rc<RefCell<CpuChip<1, F>>>,
    pub program_chip: Rc<RefCell<ProgramChip<F>>>,
    pub memory_chip: Rc<RefCell<MemoryChip<1, F>>>,

    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,

    pub cycle_tracker: CycleTracker,
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
            .max()
            .unwrap()
    }
}

impl<F: PrimeField32> ExecutionSegment<F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(config: VmConfig, program: Program<F>, state: VirtualMachineState<F>) -> Self {
        let execution_bus = ExecutionBus(0);

        let limb_bits = config.limb_bits;
        let decomp = config.decomp;
        let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << decomp));

        let cpu_chip = Rc::new(RefCell::new(CpuChip::from_state(
            config.cpu_options(),
            execution_bus,
            state.state,
        )));
        let program_chip = Rc::new(RefCell::new(ProgramChip::new(program)));
        let memory_chip = Rc::new(RefCell::new(MemoryChip::new(
            limb_bits,
            limb_bits,
            limb_bits,
            decomp,
            state.memory,
            range_checker.clone(),
        )));

        let mut executors = BTreeMap::new();
        macro_rules! assign {
            ($opcodes: expr, $executor: expr) => {
                for opcode in $opcodes {
                    executors.insert(opcode, $executor.clone().into());
                }
            };
        }

        let mut chips = vec![
            MachineChipVariant::Cpu(cpu_chip.clone()),
            MachineChipVariant::Program(program_chip.clone()),
            // TODO: Memory needs to appear before RangeChecker because Memory trace generation affects RangeChecker's trace. Should change so that RangeChecker is called only during execute.
            MachineChipVariant::Memory(memory_chip.clone()),
            MachineChipVariant::RangeChecker(range_checker.clone()),
        ];

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
                POSEIDON2_DIRECT_BUS,
                execution_bus,
                memory_chip.clone(),
            )));
            if config.perm_poseidon2_enabled {
                assign!([OpCode::PERM_POS2], poseidon2_chip);
            }
            if config.compress_poseidon2_enabled {
                assign!([OpCode::COMP_POS2], poseidon2_chip);
            }
            chips.push(MachineChipVariant::Poseidon2(poseidon2_chip.clone()));
        }

        Self {
            config,
            executors,
            chips,
            cpu_chip,
            program_chip,
            memory_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream,
            collected_metrics: Default::default(),
            cycle_tracker: CycleTracker::new(),
        }
    }

    /// Stopping is triggered by should_segment()
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        CpuChip::<1, _>::execute(self)
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

        for mut chip in self.chips {
            if chip.current_trace_height() != 0 {
                result.airs.push(chip.air());
                result.traces.push(chip.generate_trace());
                result.public_values.push(chip.get_public_values());
            }
        }

        result
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of CPU trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    ///
    /// Used by CpuChip::execute, should be private in the future
    pub fn should_segment(&mut self) -> bool {
        self.chips
            .iter()
            .any(|chip| chip.current_trace_height() > self.config.max_segment_len)
    }

    /// Used by CpuChip::execute, should be private in the future
    pub fn current_trace_cells(&self) -> usize {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_cells())
            .sum()
    }
}
