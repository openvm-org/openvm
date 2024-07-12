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
    pub segments: Vec<ExecutionSegment<WORD_SIZE, F>>,
}

pub struct ExecutionSegment<const WORD_SIZE: usize, F: PrimeField32> {
    pub cpu_air: CpuAir<WORD_SIZE>,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChip<WORD_SIZE, F>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub poseidon2_chip: Poseidon2Chip<16, F>,
    pub witness_stream: Vec<Vec<F>>,
    pub vm: VirtualMachine<WORD_SIZE, F>,

    traces: Vec<DenseMatrix<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> VirtualMachine<WORD_SIZE, F> {
    pub fn new_segment(
        self,
        program: Vec<Instruction<F>>,
        witness_stream: Vec<Vec<F>>,
    ) -> ExecutionSegment<WORD_SIZE, F> {
        ExecutionSegment::new(self, program, witness_stream)
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> ExecutionSegment<WORD_SIZE, F> {
    pub fn new(
        vm: VirtualMachine<WORD_SIZE, F>,
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
            vm,
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

    fn generate_traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        let cpu_trace = CpuAir::generate_trace(&mut self.vm)?;
        let mut result = vec![
            cpu_trace,
            self.program_chip.generate_trace(),
            self.memory_chip.generate_trace(self.range_checker.clone()),
            self.range_checker.generate_trace(),
        ];
        if self.vm.options().field_arithmetic_enabled {
            result.push(self.field_arithmetic_chip.generate_trace());
        }
        if self.vm.options().field_extension_enabled {
            result.push(self.field_extension_chip.generate_trace());
        }
        if self.vm.options().poseidon2_enabled() {
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
    if segment.vm.options().field_arithmetic_enabled {
        result.push(&segment.field_arithmetic_chip.air as &dyn AnyRap<SC>);
    }
    if segment.vm.options().field_extension_enabled {
        result.push(&segment.field_extension_chip.air as &dyn AnyRap<SC>);
    }
    if segment.vm.options().poseidon2_enabled() {
        result.push(&segment.poseidon2_chip as &dyn AnyRap<SC>);
    }
    result
}
