use afs_stark_backend::config::{StarkGenericConfig, Val};
use afs_stark_backend::rap::AnyRap;
use std::sync::Arc;

use super::VirtualMachine;
use super::VmParamsConfig;
use crate::{
    cpu::{
        trace::{ExecutionError, Instruction},
        CpuChip, CpuOptions, POSEIDON2_BUS, RANGE_CHECKER_BUS,
    },
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    poseidon2::Poseidon2Chip,
    program::ProgramChip,
};
use afs_chips::range_gate::RangeCheckerGateChip;
use poseidon2_air::poseidon2::Poseidon2Config;

use p3_field::PrimeField32;
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_util::log2_strict_usize;

pub struct ExecutionSegment<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmParamsConfig,
    pub cpu_chip: CpuChip<WORD_SIZE, F>,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChip<WORD_SIZE, F>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub poseidon2_chip: Poseidon2Chip<16, F>,
    pub witness_stream: Vec<Vec<F>>,

    traces: Vec<DenseMatrix<F>>,
    max_len: usize,
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

        // TODO: reduce cloning
        let cpu_chip = CpuChip::new(config.cpu_options());
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
            cpu_chip,
            program_chip,
            memory_chip,
            field_arithmetic_chip,
            field_extension_chip,
            range_checker,
            poseidon2_chip,
            traces: vec![],
            witness_stream,
            max_len: 1 << 20,
        }
    }

    pub fn set_test_segments(&mut self, max_len: usize) {
        self.max_len = max_len;
    }

    pub fn switch_segments(&mut self) -> Result<bool, ExecutionError> {
        // let heights = [
        //     self.cpu_chip.current_height(),
        //     self.memory_chip.current_height(),
        //     self.field_arithmetic_chip.current_height(),
        //     self.field_extension_chip.current_height(),
        //     self.poseidon2_chip.current_height(),
        // ];
        // let max_height = *heights.iter().max().unwrap();
        // let maximum = (1 << 20) - 100;
        // let maximum = 4;
        Ok(self.cpu_chip.current_height() == self.max_len)
    }

    /// Execution is determined by CPU trace generation, in turn determined by segment::continue_execution()
    /// which determines whether to continue execution or not
    pub fn generate_traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        let cpu_trace = CpuChip::generate_trace(self)?;
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

    /// TODO: reduce cloning
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
        &segment.cpu_chip.air,
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
