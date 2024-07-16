use afs_stark_backend::rap::AnyRap;
use p3_field::PrimeField32;
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

mod segment;
pub use segment::{get_chips, ExecutionSegment};

pub enum Void {}

use crate::cpu::{
    trace::{ExecutionError, Instruction},
    CpuOptions,
};

use self::config::VmParamsConfig;

pub mod config;

pub const DEFAULT_MAX_LEN: usize = 1 << 20;

pub struct VirtualMachine<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmParamsConfig,
    pub program: Vec<Instruction<F>>,
    pub witness_stream: Vec<Vec<F>>,
    pub segments: Vec<Box<ExecutionSegment<WORD_SIZE, F>>>,
    pub traces: Vec<DenseMatrix<F>>,
    // NOT PUBLIC by design, adjust only for testing
    max_len: usize,
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
            traces: vec![],
            max_len: DEFAULT_MAX_LEN,
        };
        vm.new_segment();
        vm
    }

    pub fn set_test_segments(&mut self, max_len: usize) {
        self.max_len = max_len;
        self.segments[0].set_test_segments(max_len);
    }

    pub fn new_segment(&mut self) {
        let program = self.program.clone();
        let witness_stream = self.witness_stream.clone();
        let mut segment = ExecutionSegment::new(self, program, witness_stream);
        if self.max_len != DEFAULT_MAX_LEN {
            segment.set_test_segments(self.max_len);
        }
        self.segments.push(Box::new(segment));
    }

    pub fn next_segment(&mut self) {
        let mem_state = self.segments.last().unwrap().memory_chip.get_memory();
        let cpu_state = self.segments.last().unwrap().cpu_chip.get_state();

        self.new_segment();

        self.segments
            .last_mut()
            .unwrap()
            .cpu_chip
            .transfer_state(cpu_state);
        self.segments
            .last_mut()
            .unwrap()
            .memory_chip
            .install_memory(mem_state);
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        let mut result = vec![];
        loop {
            result.extend(self.segments.last_mut().unwrap().generate_traces()?);
            if self.segments.last_mut().unwrap().cpu_chip.is_done {
                break;
            }
            // Add additional traces for committing, if needed
            result.extend(self.segments.last_mut().unwrap().generate_commitments()?);
            self.next_segment();
        }
        self.traces = result;
        Ok(())
    }

    pub fn get_traces(&self) -> Vec<DenseMatrix<F>> {
        self.traces
            .clone()
            .into_iter()
            .filter(|trace| !trace.values.is_empty())
            .collect()
    }

    pub fn max_log_degree(&self) -> Result<usize, ExecutionError> {
        let mut checker_trace_degree = 0;
        for trace in &self.traces {
            checker_trace_degree = std::cmp::max(checker_trace_degree, trace.height());
        }
        Ok(log2_strict_usize(checker_trace_degree))
    }
}

// TODO: make into struct method
pub fn get_all_chips<const WORD_SIZE: usize, SC: StarkGenericConfig>(
    vm: &VirtualMachine<WORD_SIZE, Val<SC>>,
) -> Vec<&dyn AnyRap<SC>>
where
    Val<SC>: PrimeField32,
{
    let chips: Vec<_> = vm
        .segments
        .iter()
        .flat_map(|segment| get_chips::<WORD_SIZE, SC>(segment))
        .collect();

    let chips: Vec<_> = vm
        .traces
        .iter()
        .zip(chips)
        .filter(|(trace, _)| !trace.values.is_empty())
        .map(|(_, chip)| chip)
        .collect();
    chips
}
