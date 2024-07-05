use std::sync::Arc;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use p3_field::PrimeField32;
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

pub enum Void {}

use crate::{
    cpu::{
        trace::{ExecutionError, Instruction},
        CpuAir, CpuOptions, RANGE_CHECKER_BUS,
    },
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    program::ProgramChip,
};

use self::config::{VmConfig, VmParamsConfig};

pub mod config;

pub struct VirtualMachine<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmParamsConfig,

    pub cpu_air: CpuAir<WORD_SIZE>,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChip<WORD_SIZE, F>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,

    traces: Vec<DenseMatrix<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> VirtualMachine<WORD_SIZE, F> {
    pub fn new(config: VmConfig, program: Vec<Instruction<F>>) -> Self {
        let config = config.vm;
        let decomp = config.decomp;
        let limb_bits = config.limb_bits;

        let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << decomp));

        let cpu_air = CpuAir::new(config.cpu_options());
        let program_chip = ProgramChip::new(program.clone());
        let memory_chip = MemoryChip::new(limb_bits, limb_bits, limb_bits, decomp);
        let field_arithmetic_chip = FieldArithmeticChip::new();
        let field_extension_chip = FieldExtensionArithmeticChip::new();

        Self {
            config,
            cpu_air,
            program_chip,
            memory_chip,
            field_arithmetic_chip,
            field_extension_chip,
            range_checker,
            traces: vec![],
        }
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    fn generate_traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        let cpu_trace = CpuAir::generate_trace(self)?;
        let mut result = vec![
            cpu_trace,
            self.program_chip.generate_trace(),
            self.memory_chip.generate_trace(self.range_checker.clone()),
            self.range_checker.generate_trace(),
        ];
        if self.options().field_arithmetic_enabled {
            result.push(self.field_arithmetic_chip.generate_trace());
        }
        if self.options().field_extension_enabled {
            result.push(self.field_extension_chip.generate_trace());
        }
        Ok(result)
    }

    pub fn traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        if self.traces.is_empty() {
            self.traces = self.generate_traces()?;
        }
        Ok(self.traces.clone())
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
}

pub fn get_chips<const WORD_SIZE: usize, SC: StarkGenericConfig>(
    vm: &VirtualMachine<WORD_SIZE, Val<SC>>,
) -> Vec<&dyn AnyRap<SC>>
where
    Val<SC>: PrimeField32,
{
    if vm.options().field_extension_enabled {
        vec![
            &vm.cpu_air,
            &vm.program_chip.air,
            &vm.memory_chip.air,
            &vm.range_checker.air,
            &vm.field_arithmetic_chip.air,
            &vm.field_extension_chip.air,
        ]
    } else if vm.options().field_arithmetic_enabled {
        vec![
            &vm.cpu_air,
            &vm.program_chip.air,
            &vm.memory_chip.air,
            &vm.range_checker.air,
            &vm.field_arithmetic_chip.air,
        ]
    } else {
        vec![
            &vm.cpu_air,
            &vm.program_chip.air,
            &vm.memory_chip.air,
            &vm.range_checker.air,
        ]
    }
}
