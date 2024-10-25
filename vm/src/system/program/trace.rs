use std::{borrow::BorrowMut, sync::Arc};

use afs_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_commit::PolynomialSpace,
    prover::{
        helper::AirProofInputTestHelper,
        types::{AirProofInput, AirProofRawInput, CommittedTraceData, TraceCommitter},
    },
};
use axvm_instructions::{TerminateOpcode, UsizeOpcode};
use itertools::Itertools;
use p3_field::{Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use super::{Instruction, Program, ProgramChip, ProgramExecutionCols, ProgramPvs, EXIT_CODE_FAIL};

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
        let mut public_values = vec![F::zero(); ProgramPvs::<F>::width()];
        *public_values.as_mut_slice().borrow_mut() = ProgramPvs {
            pc_start: F::from_canonical_u32(self.program.pc_start),
        };
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
                    public_values,
                },
            }
        } else {
            AirProofInput::cached_traces(
                air,
                vec![generate_cached_trace(&self.program)],
                common_trace,
                public_values,
            )
        }
    }
}

fn generate_cached_trace<F: PrimeField64>(program: &Program<F>) -> RowMajorMatrix<F> {
    let width = ProgramExecutionCols::<F>::width();
    let mut instructions = program
        .instructions_and_debug_infos
        .iter()
        .sorted_by_key(|(pc, _)| *pc)
        .map(|(&pc, (instruction, _))| (pc, instruction))
        .collect::<Vec<_>>();

    let padding = padding_instruction();
    while !instructions.len().is_power_of_two() {
        instructions.push((
            program.pc_base + instructions.len() as u32 * program.step,
            &padding,
        ));
    }

    let mut rows = vec![F::zero(); instructions.len() * width];
    rows.par_chunks_mut(width)
        .zip(instructions)
        .for_each(|(row, (pc, instruction))| {
            let row: &mut ProgramExecutionCols<F> = row.borrow_mut();
            *row = ProgramExecutionCols {
                pc_start: F::from_canonical_u32(program.pc_start),
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

pub(super) fn padding_instruction<F: Field>() -> Instruction<F> {
    Instruction::from_usize(
        TerminateOpcode::TERMINATE.with_default_offset(),
        [0, 0, EXIT_CODE_FAIL],
    )
}
