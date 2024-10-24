use std::{borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_commit::PolynomialSpace,
    prover::{
        helper::AirProofInputTestHelper,
        types::{AirProofInput, AirProofRawInput, CommittedTraceData, TraceCommitter},
    },
};
use itertools::Itertools;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use super::{Program, ProgramChip, ProgramExecutionCols};

/// A program with a committed cached trace.
pub struct CommittedProgram<SC: StarkGenericConfig> {
    pub committed_trace_data: CommittedTraceData<SC>,
    pub program: Program<Val<SC>>,
}

impl<F: PrimeField64> Program<F> {
    pub fn commit<SC: StarkGenericConfig>(&self, pcs: &SC::Pcs) -> CommittedProgram<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let cached_trace = generate_cached_trace(self);
        CommittedProgram {
            committed_trace_data: CommittedTraceData {
                raw_data: Arc::new(cached_trace.clone()),
                prover_data: TraceCommitter::new(pcs).commit(vec![cached_trace]),
            },
            program: self.clone(),
        }
    }
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn generate_air_proof_input<SC: StarkGenericConfig>(
        self,
        cached_trace: Option<CommittedTraceData<SC>>,
    ) -> AirProofInput<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let air = Arc::new(self.air);
        let common_trace = RowMajorMatrix::new_col(
            self.execution_frequencies
                .into_iter()
                .map(|x| F::from_canonical_usize(x))
                .collect::<Vec<F>>(),
        );
        if let Some(cached_trace) = cached_trace {
            AirProofInput {
                air,
                cached_mains_pdata: vec![cached_trace.prover_data],
                raw: AirProofRawInput {
                    cached_mains: vec![cached_trace.raw_data],
                    common_main: Some(common_trace),
                    public_values: vec![],
                },
            }
        } else {
            AirProofInput::cached_traces_no_pis(
                air,
                vec![generate_cached_trace(&self.program)],
                common_trace,
            )
        }
    }
}

fn generate_cached_trace<F: PrimeField64>(program: &Program<F>) -> RowMajorMatrix<F> {
    let width = ProgramExecutionCols::<F>::width();
    let instructions = program
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
