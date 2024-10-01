use afs_stark_backend::rap::AnyRap;
use p3_air::BaseAir;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig};

use super::{columns::ProgramExecutionCols, ProgramChip};
use crate::arch::chips::MachineChip;

impl<F: PrimeField64> ProgramChip<F> {
    fn generate_cached_trace(&self) -> RowMajorMatrix<F> {
        let mut rows = vec![];
        for (pc, instruction) in self.air.program.instructions.iter().enumerate() {
            let exec_cols = ProgramExecutionCols {
                pc: F::from_canonical_usize(pc),
                opcode: F::from_canonical_usize(instruction.opcode as usize),
                op_a: instruction.op_a,
                op_b: instruction.op_b,
                op_c: instruction.op_c,
                as_b: instruction.d,
                as_c: instruction.e,
                op_f: instruction.op_f,
                op_g: instruction.op_g,
            };
            rows.extend(exec_cols.flatten());
        }

        RowMajorMatrix::new(rows, ProgramExecutionCols::<F>::width())
    }
}

impl<F: PrimeField64> MachineChip<F> for ProgramChip<F> {
    fn generate_trace(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(
            self.execution_frequencies
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect::<Vec<F>>(),
        )
    }

    fn generate_traces(self) -> Vec<RowMajorMatrix<F>> {
        let cached_trace = self.generate_cached_trace();
        let common_trace = self.generate_trace();

        vec![cached_trace, common_trace]
    }

    fn air<SC: StarkGenericConfig>(&self) -> Box<dyn AnyRap<SC>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        Box::new(self.air.clone())
    }

    fn current_trace_height(&self) -> usize {
        self.true_program_length
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}
