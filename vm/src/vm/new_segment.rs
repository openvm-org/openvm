use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
};

use p3_field::PrimeField32;
use p3_matrix::dense::DenseMatrix;

use poseidon2_air::poseidon2::Poseidon2Config;

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::{MachineChip, MachineChipVariant, OpCodeExecutorVariant},
    },
    cpu::{CpuChip, CpuOptions, POSEIDON2_BUS, trace::ExecutionError},
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    poseidon2::Poseidon2Chip,
    program::{Program, ProgramChip},
};

use super::{VirtualMachineState, VmConfig, VmMetrics};

pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,

    pub executors: Vec<OpCodeExecutorVariant<F>>,
    pub chips: Vec<MachineChipVariant<F>>,
    pub cpu_chip: CpuChip<1, F>,
    pub program_chip: ProgramChip<F>,

    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,

    pub public_values: Vec<Option<F>>,
    pub opcode_counts: BTreeMap<String, usize>,
    pub dsl_counts: BTreeMap<String, usize>,
    pub opcode_trace_cells: BTreeMap<String, usize>,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
}

impl<F: PrimeField32> ExecutionSegment<F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(config: VmConfig, state: VirtualMachineState<F>, program: Program<F>) -> Self {
        let opcode_counts = BTreeMap::new();
        let dsl_counts = BTreeMap::new();
        let opcode_trace_cells = BTreeMap::new();

        let execution_bus = ExecutionBus(0);

        let cpu_chip = CpuChip::from_state(config.cpu_options(), state.state);
        let program_chip = ProgramChip::new(program);
        let limb_bits = config.limb_bits;
        let decomp = config.decomp;
        let memory_chip = MemoryChip::new(limb_bits, limb_bits, limb_bits, decomp, state.memory);
        let memory_chip_ref = Rc::new(RefCell::new(memory_chip));
        let field_arithmetic_chip =
            FieldArithmeticChip::new(execution_bus, memory_chip_ref.clone());
        let field_extension_chip =
            FieldExtensionArithmeticChip::new(execution_bus, memory_chip_ref.clone());
        let poseidon2_chip = Poseidon2Chip::from_poseidon2_config(
            Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
            POSEIDON2_BUS,
            execution_bus,
            memory_chip_ref,
        );
        let chips = vec![
            MachineChipVariant::Cpu(cpu_chip),
            MachineChipVariant::Program(program_chip),
            MachineChipVariant::Memory(memory_chip),
            MachineChipVariant::FieldArithmetic(field_arithmetic_chip),
            MachineChipVariant::FieldExtension(field_extension_chip),
            MachineChipVariant::Poseidon2(poseidon2_chip),
        ];
        let executors = vec![OpCodeExecutorVariant::FieldArithmetic(
            field_arithmetic_chip,
        )];

        Self {
            config,
            public_values: vec![None; config.num_public_values],
            executors,
            chips,
            cpu_chip,
            program_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream,
            opcode_counts,
            dsl_counts,
            opcode_trace_cells,
            collected_metrics: Default::default(),
        }
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of CPU trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    pub fn should_segment(&mut self) -> bool {
        false
    }

    /// Stopping is triggered by should_segment()
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        CpuChip::execute(self)
    }

    /// Called by VM to generate traces for current segment. Includes empty traces.
    /// Should only be called after Self::execute
    pub fn generate_traces(&mut self) -> Vec<DenseMatrix<F>> {
        self.chips
            .iter_mut()
            .map(|chip| chip.generate_trace())
            .collect()
    }

    /// Generate Merkle proof/memory diff traces, and publish public values
    ///
    /// For now, only publishes program counter public values
    pub fn generate_commitments(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        // self.cpu_chip.generate_pvs();
        Ok(vec![])
    }

    // TODO[osama]: revisit this
    // [danny]: presumably fine now
    pub fn get_num_chips(&self) -> usize {
        self.chips.len()
    }

    // TODO[osama]: revisit this
    /// Returns public values for all chips in this segment
    pub fn get_pis(&mut self) -> Vec<Vec<F>> {
        self.chips
            .iter_mut()
            .map(|chip| chip.get_public_values())
            .collect()
    }

    pub fn metrics(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();

        metrics
    }
}
