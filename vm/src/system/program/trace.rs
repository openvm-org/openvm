use std::borrow::BorrowMut;

use itertools::Itertools;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use super::{ProgramChip, ProgramExecutionCols};

impl<F: PrimeField64> ProgramChip<F> {
    pub fn generate_cached_trace(&self) -> RowMajorMatrix<F> {
        let width = ProgramExecutionCols::<F>::width();
        let instructions = self
            .program
            .instructions_and_debug_infos
            .iter()
            .sorted_by_key(|(pc, _)| *pc)
            .map(|(pc, (instruction, _))| (pc, instruction))
            .collect::<Vec<_>>();
        let mut rows = vec![F::zero(); instructions.len() * width];
        rows.par_chunks_mut(width)
            .zip(instructions)
            .for_each(|(row, (&pc, instruction))| {
                let row: &mut ProgramExecutionCols<F> = row.borrow_mut();
                *row = ProgramExecutionCols {
                    pc: F::from_canonical_u32(pc),
                    opcode: F::from_canonical_usize(instruction.opcode),
                    a: instruction.a,
                    b: instruction.b,
                    c: instruction.c,
                    d: instruction.d,
                    e: instruction.e,
                    f: instruction.f,
                    g: instruction.g,
                };
            });

        RowMajorMatrix::new(rows, width)
    }

    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new_col(
            self.execution_frequencies
                .iter()
                .map(|x| F::from_canonical_usize(*x))
                .collect::<Vec<F>>(),
        )
    }
}
