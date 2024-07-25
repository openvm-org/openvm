use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use p3_field::PrimeField32;
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use poseidon2_air::poseidon2::Poseidon2Config;

pub mod cycle_tracker;

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

use self::config::VmConfig;

pub mod config;

pub struct VirtualMachine<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmConfig,

    pub cpu_air: CpuAir<WORD_SIZE>,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChip<WORD_SIZE, F>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub poseidon2_chip: Poseidon2Chip<16, F>,
    pub input_stream: VecDeque<Vec<F>>,
    pub public_values: Vec<Option<F>>,

    has_generation_happened: bool,
    traces: Vec<DenseMatrix<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> VirtualMachine<WORD_SIZE, F> {
    pub fn new(config: VmConfig, program: Vec<Instruction<F>>, input_stream: Vec<Vec<F>>) -> Self {
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
            has_generation_happened: false,
            traces: vec![],
            public_values: vec![None; config.num_public_values],
            input_stream: VecDeque::from(input_stream),
        }
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    fn generate_traces(&mut self) -> Result<(), ExecutionError> {
        if !self.has_generation_happened {
            self.has_generation_happened = true;

            let cpu_trace = CpuAir::generate_trace(self)?;
            self.traces = vec![
                cpu_trace,
                self.program_chip.generate_trace(),
                self.memory_chip.generate_trace(self.range_checker.clone()),
                self.range_checker.generate_trace(),
            ];
            if self.options().field_arithmetic_enabled {
                self.traces
                    .push(self.field_arithmetic_chip.generate_trace());
            }
            if self.options().field_extension_enabled {
                self.traces.push(self.field_extension_chip.generate_trace());
            }
            if self.options().poseidon2_enabled() {
                self.traces.push(self.poseidon2_chip.generate_trace());
            }
        }
        Ok(())
    }

    pub fn traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        self.generate_traces()?;
        Ok(self.traces.clone())
    }

    pub fn get_public_values(&mut self) -> Result<Vec<Vec<F>>, ExecutionError> {
        self.generate_traces()?;
        let cpu_public_values = self
            .public_values
            .iter()
            .map(|pi| pi.unwrap_or(F::zero()))
            .collect();
        let mut public_values = vec![vec![]; 4];
        public_values[0] = cpu_public_values;
        if self.options().field_arithmetic_enabled {
            public_values.push(vec![]);
        }
        if self.options().field_extension_enabled {
            public_values.push(vec![]);
        }
        if self.options().poseidon2_enabled() {
            public_values.push(vec![]);
        }
        Ok(public_values)
    }

    /*fn max_trace_heights(&self) -> Vec<usize> {
        let max_operations = self.config.max_operations;
        let max_program_length = self.config.max_program_length;
        let result = [
            max_operations,
            max_program_length,
            3 * max_operations,
            max_operations,
            max_operations,
        ];
        result
            .iter()
            .map(|height| height.next_power_of_two())
            .collect()
    }*/

    pub fn max_log_degree(&mut self) -> Result<usize, ExecutionError> {
        let mut checker_trace_degree = 0;
        for trace in self.traces()? {
            checker_trace_degree = std::cmp::max(checker_trace_degree, trace.height());
        }
        Ok(log2_strict_usize(checker_trace_degree))
    }

    pub fn metrics(&mut self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "memory_chip_accesses".to_string(),
            self.memory_chip.accesses.len(),
        );
        metrics.insert(
            "field_arithmetic_ops".to_string(),
            self.field_arithmetic_chip.operations.len(),
        );
        metrics.insert(
            "field_extension_ops".to_string(),
            self.field_extension_chip.operations.len(),
        );
        metrics.insert(
            "range_checker_count".to_string(),
            self.range_checker.count.len(),
        );
        metrics.insert(
            "poseidon2_chip_rows".to_string(),
            self.poseidon2_chip.rows.len(),
        );
        metrics.insert("input_stream_len".to_string(), self.input_stream.len());
        metrics
    }
}

pub fn get_chips<const WORD_SIZE: usize, SC: StarkGenericConfig>(
    vm: &VirtualMachine<WORD_SIZE, Val<SC>>,
) -> Vec<&dyn AnyRap<SC>>
where
    Val<SC>: PrimeField32,
{
    let mut result: Vec<&dyn AnyRap<SC>> = vec![
        &vm.cpu_air,
        &vm.program_chip.air,
        &vm.memory_chip.air,
        &vm.range_checker.air,
    ];
    if vm.options().field_arithmetic_enabled {
        result.push(&vm.field_arithmetic_chip.air as &dyn AnyRap<SC>);
    }
    if vm.options().field_extension_enabled {
        result.push(&vm.field_extension_chip.air as &dyn AnyRap<SC>);
    }
    if vm.options().poseidon2_enabled() {
        result.push(&vm.poseidon2_chip.air as &dyn AnyRap<SC>);
    }
    result
}
