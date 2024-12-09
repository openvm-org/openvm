use std::{borrow::BorrowMut, sync::Arc};

use ax_stark_backend::{
    config::{Com, Domain, StarkGenericConfig, Val},
    p3_commit::PolynomialSpace,
    p3_field::{Field, PrimeField64},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::{
        helper::AirProofInputTestHelper,
        types::{AirProofInput, AirProofRawInput, CommittedTraceData, TraceCommitter},
    },
};
use axvm_instructions::{exe::AxVmExe, program::Program, AxVmOpcode, SystemOpcode};
use derivative::Derivative;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::{Instruction, ProgramChip, ProgramExecutionCols, EXIT_CODE_FAIL};

#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "AxVmExe<Val<SC>>: Serialize, CommittedTraceData<SC>: Serialize",
    deserialize = "AxVmExe<Val<SC>>: Deserialize<'de>, CommittedTraceData<SC>: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct AxVmCommittedExe<SC: StarkGenericConfig> {
    /// Raw executable.
    pub exe: AxVmExe<Val<SC>>,
    /// Committed program trace.
    pub committed_program: CommittedTraceData<SC>,
}

impl<SC: StarkGenericConfig> AxVmCommittedExe<SC>
where
    Val<SC>: PrimeField64,
{
    pub fn commit(exe: AxVmExe<Val<SC>>, pcs: &SC::Pcs) -> Self {
        let cached_trace = generate_cached_trace(&exe.program);
        Self {
            committed_program: CommittedTraceData {
                raw_data: Arc::new(cached_trace.clone()),
                prover_data: TraceCommitter::new(pcs).commit(vec![cached_trace]),
            },
            exe,
        }
    }
    pub fn get_program_commit(&self) -> Com<SC> {
        self.committed_program.prover_data.commit.clone()
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

pub(crate) fn generate_cached_trace<F: PrimeField64>(program: &Program<F>) -> RowMajorMatrix<F> {
    let width = ProgramExecutionCols::<F>::width();
    let mut instructions = program
        .enumerate_by_pc()
        .into_iter()
        .map(|(pc, instruction, _)| (pc, instruction))
        .collect_vec();

    let padding = padding_instruction();
    while !instructions.len().is_power_of_two() {
        instructions.push((
            program.pc_base + instructions.len() as u32 * program.step,
            padding.clone(),
        ));
    }

    let mut rows = F::zero_vec(instructions.len() * width);
    rows.par_chunks_mut(width)
        .zip(instructions)
        .for_each(|(row, (pc, instruction))| {
            let row: &mut ProgramExecutionCols<F> = row.borrow_mut();
            *row = ProgramExecutionCols {
                pc: F::from_canonical_u32(pc),
                opcode: instruction.opcode.to_field(),
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
        AxVmOpcode::with_default_offset(SystemOpcode::TERMINATE),
        [0, 0, EXIT_CODE_FAIL],
    )
}
