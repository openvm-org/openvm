use std::sync::Arc;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use p3_field::PrimeField32;
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::Poseidon2Config;

pub enum Void {}

use crate::{
    cpu::{
        trace::{ExecutionError, Instruction},
        CpuAir, CpuOptions, POSEIDON2_BUS, RANGE_CHECKER_BUS,
    },
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    poseidon2::Poseidon2Chip,
    program::ProgramChip,
};

use self::config::VmParamsConfig;

pub mod config;

pub struct VirtualMachine<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmParamsConfig,
    pub program: Vec<Instruction<F>>,
    pub witness_stream: Vec<Vec<F>>,
    pub segments: Vec<Box<ExecutionSegment<WORD_SIZE, F>>>,
}

pub struct ExecutionSegment<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmParamsConfig,
    pub cpu_air: CpuAir<WORD_SIZE>,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChip<WORD_SIZE, F>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub poseidon2_chip: Poseidon2Chip<16, F>,
    pub witness_stream: Vec<Vec<F>>,

    traces: Vec<DenseMatrix<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> VirtualMachine<WORD_SIZE, F> {
    pub fn new(
        config: VmParamsConfig,
        program: Vec<Instruction<F>>,
        witness_stream: Vec<Vec<F>>,
    ) -> Self {
        let mut vm = Self {
            config,
            program,
            witness_stream,
            segments: vec![],
        };
        vm.new_segment();
        vm
    }

    pub fn new_segment(&mut self) {
        let program = self.program.clone();
        let witness_stream = self.witness_stream.clone();
        let segment = ExecutionSegment::new(self, program, witness_stream);
        self.segments.push(Box::new(segment));
    }

    pub fn next_segment(&mut self) {
        let pc = self.segments.last().unwrap().cpu_air.pc;
        let state = self.segments.last().unwrap().memory_chip.get_memory();

        self.new_segment();

        self.segments.last_mut().unwrap().cpu_air.pc = pc;
        self.segments
            .last_mut()
            .unwrap()
            .memory_chip
            .install_memory(state);
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    pub fn execute(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        let mut result = vec![];
        loop {
            result.extend(self.segments.last_mut().unwrap().generate_traces()?);
            if self.segments.last_mut().unwrap().cpu_air.is_done {
                break;
            }
            // Add additional traces for committing, if needed
            result.extend(self.segments.last_mut().unwrap().generate_commitments()?);
            self.next_segment();
        }
        Ok(result)
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> ExecutionSegment<WORD_SIZE, F> {
    pub fn new(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        program: Vec<Instruction<F>>,
        witness_stream: Vec<Vec<F>>,
    ) -> Self {
        let config = vm.config;
        let decomp = config.decomp;
        let limb_bits = config.limb_bits;

        let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << decomp));

        let cpu_air = CpuAir::new(config.cpu_options());
        let program_chip = ProgramChip::new(program.clone());
        let memory_chip = MemoryChip::new(limb_bits, limb_bits, limb_bits, decomp);
        let field_arithmetic_chip = FieldArithmeticChip::new();
        let field_extension_chip = FieldExtensionArithmeticChip::new();
        let poseidon2_chip = Poseidon2Chip::from_poseidon2_config(
            Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
            POSEIDON2_BUS,
        );

        Self {
            config,
            cpu_air,
            program_chip,
            memory_chip,
            field_arithmetic_chip,
            field_extension_chip,
            range_checker,
            poseidon2_chip,
            traces: vec![],
            witness_stream,
        }
    }

    pub fn switch_segments(&mut self) -> Result<bool, ExecutionError> {
        Ok(false)
    }

    /// Execution is determined by CPU trace generation, in turn determined by segment::continue_execution()
    /// which determines whether to continue execution or not
    fn generate_traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        let cpu_trace = CpuAir::generate_trace(self)?;
        let mut result = vec![
            cpu_trace,
            self.program_chip.generate_trace(),
            self.memory_chip.generate_trace(self.range_checker.clone()),
            self.range_checker.generate_trace(),
        ];
        if self.config.cpu_options().field_arithmetic_enabled {
            result.push(self.field_arithmetic_chip.generate_trace());
        }
        if self.config.cpu_options().field_extension_enabled {
            result.push(self.field_extension_chip.generate_trace());
        }
        if self.config.cpu_options().poseidon2_enabled() {
            result.push(self.poseidon2_chip.generate_trace());
        }
        Ok(result)
    }

    pub fn traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        if self.traces.is_empty() {
            self.traces = self.generate_traces()?;
        }
        Ok(self.traces.clone())
    }

    pub fn max_log_degree(&mut self) -> Result<usize, ExecutionError> {
        let mut checker_trace_degree = 0;
        for trace in self.traces()? {
            checker_trace_degree = std::cmp::max(checker_trace_degree, trace.height());
        }
        Ok(log2_strict_usize(checker_trace_degree))
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    /// Generate Merkle proof/memory diff traces
    pub fn generate_commitments(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        Ok(vec![])
    }
}

pub fn get_chips<const WORD_SIZE: usize, SC: StarkGenericConfig>(
    segment: &ExecutionSegment<WORD_SIZE, Val<SC>>,
) -> Vec<&dyn AnyRap<SC>>
where
    Val<SC>: PrimeField32,
{
    let mut result: Vec<&dyn AnyRap<SC>> = vec![
        &segment.cpu_air,
        &segment.program_chip.air,
        &segment.memory_chip.air,
        &segment.range_checker.air,
    ];
    if segment.config.cpu_options().field_arithmetic_enabled {
        result.push(&segment.field_arithmetic_chip.air as &dyn AnyRap<SC>);
    }
    if segment.config.cpu_options().field_extension_enabled {
        result.push(&segment.field_extension_chip.air as &dyn AnyRap<SC>);
    }
    if segment.config.cpu_options().poseidon2_enabled() {
        result.push(&segment.poseidon2_chip as &dyn AnyRap<SC>);
    }
    result
}
