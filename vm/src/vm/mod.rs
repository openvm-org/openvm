use std::sync::Arc;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_stark_backend::rap::AnyRap;
use p3_field::{PrimeField32, PrimeField64};
use p3_matrix::{dense::DenseMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

use crate::{
    cpu::{trace::Instruction, CpuAir, RANGE_CHECKER_BUS, WORD_SIZE},
    field_arithmetic::FieldArithmeticAir,
    memory::{offline_checker::OfflineChecker, MemoryAccess},
    program::ProgramAir,
};

use self::config::{VmConfig, VmParamsConfig};

pub mod config;

pub struct VirtualMachine<SC: StarkGenericConfig>
where
    Val<SC>: PrimeField64,
{
    pub config: VmParamsConfig,

    pub cpu_air: CpuAir,
    pub program_air: ProgramAir<Val<SC>>,
    pub memory_air: OfflineChecker,
    pub field_arithmetic_air: FieldArithmeticAir,
    pub range_checker: Arc<RangeCheckerGateChip>,

    pub cpu_trace: DenseMatrix<Val<SC>>,
    pub program_trace: DenseMatrix<Val<SC>>,
    pub memory_trace: DenseMatrix<Val<SC>>,
    pub field_arithmetic_trace: DenseMatrix<Val<SC>>,
    pub range_trace: DenseMatrix<Val<SC>>,
}

impl<SC: StarkGenericConfig> VirtualMachine<SC>
where
    Val<SC>: PrimeField64,
    Val<SC>: PrimeField32,
{
    pub fn new(config: VmConfig, program: Vec<Instruction<Val<SC>>>) -> Self {
        let config = config.vm;
        let decomp = config.decomp;
        let limb_bits = config.limb_bits;

        let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << decomp));

        let cpu_air = CpuAir::new(config.cpu_options());
        let program_air = ProgramAir::new(program.clone());
        let memory_air = OfflineChecker::new(WORD_SIZE, limb_bits, limb_bits, limb_bits, decomp);
        let field_arithmetic_air = FieldArithmeticAir::new();

        let execution = cpu_air.generate_program_execution(program_air.program.clone());
        let program_trace = program_air.generate_trace(&execution);

        let ops = execution
            .memory_accesses
            .iter()
            .map(|access| MemoryAccess {
                address: access.address,
                op_type: access.op_type,
                address_space: access.address_space,
                timestamp: access.timestamp,
                data: vec![access.data],
            })
            .collect::<Vec<_>>();
        let memory_trace_degree = execution.memory_accesses.len().next_power_of_two();
        let memory_trace =
            memory_air.generate_trace(ops, range_checker.clone(), memory_trace_degree);

        let range_trace: DenseMatrix<Val<SC>> = range_checker.generate_trace();

        let field_arithmetic_trace = field_arithmetic_air.generate_trace(&execution);

        Self {
            config,
            cpu_air,
            program_air,
            memory_air,
            field_arithmetic_air,
            range_checker,
            cpu_trace: execution.trace(),
            program_trace,
            memory_trace,
            field_arithmetic_trace,
            range_trace,
        }
    }

    pub fn chips(&self) -> Vec<&dyn AnyRap<SC>> {
        if self.config.field_arithmetic_enabled {
            vec![
                &self.cpu_air,
                &self.program_air,
                &self.memory_air,
                &self.field_arithmetic_air,
                &self.range_checker.air,
            ]
        } else {
            vec![
                &self.cpu_air,
                &self.program_air,
                &self.memory_air,
                &self.range_checker.air,
            ]
        }
    }

    pub fn traces(&self) -> Vec<DenseMatrix<Val<SC>>> {
        if self.config.field_arithmetic_enabled {
            vec![
                self.cpu_trace.clone(),
                self.program_trace.clone(),
                self.memory_trace.clone(),
                self.field_arithmetic_trace.clone(),
                self.range_trace.clone(),
            ]
        } else {
            vec![
                self.cpu_trace.clone(),
                self.program_trace.clone(),
                self.memory_trace.clone(),
                self.range_trace.clone(),
            ]
        }
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

    pub fn max_log_degree(&self) -> usize {
        let mut checker_trace_degree = 0;
        for trace in self.traces() {
            checker_trace_degree = std::cmp::max(checker_trace_degree, trace.height());
        }
        log2_strict_usize(checker_trace_degree)
    }
}
