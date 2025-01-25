use std::{borrow::BorrowMut, sync::Arc};

use derivative::Derivative;
use itertools::Itertools;
use openvm_instructions::{exe::VmExe, program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::{
    config::{Com, Domain, StarkGenericConfig, Val},
    p3_commit::{Pcs, PolynomialSpace},
    p3_field::{Field, PrimeField64},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{
        helper::AirProofInputTestHelper,
        types::{AirProofInput, AirProofRawInput, CommittedTraceData},
    },
};
use serde::{Deserialize, Serialize};

use super::{Instruction, ProgramChip, ProgramExecutionCols, EXIT_CODE_FAIL};

#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "VmExe<Val<SC>>: Serialize, CommittedTraceData<SC>: Serialize",
    deserialize = "VmExe<Val<SC>>: Deserialize<'de>, CommittedTraceData<SC>: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct VmCommittedExe<SC: StarkGenericConfig> {
    /// Raw executable.
    pub exe: VmExe<Val<SC>>,
    /// Committed program trace.
    pub committed_program: CommittedTraceData<SC>,
}

impl<SC: StarkGenericConfig> VmCommittedExe<SC>
where
    Val<SC>: PrimeField64,
{
    pub fn commit(exe: VmExe<Val<SC>>, pcs: &SC::Pcs) -> Self {
        let cached_trace = generate_cached_trace(&exe.program);
        let domain = pcs.natural_domain_for_degree(cached_trace.height());
        let (commitment, pcs_data) = pcs.commit(vec![(domain, cached_trace.clone())]);
        Self {
            committed_program: CommittedTraceData {
                trace: Arc::new(cached_trace),
                commitment,
                pcs_data: Arc::new(pcs_data),
            },
            exe,
        }
    }
    pub fn get_program_commit(&self) -> Com<SC> {
        self.committed_program.commitment.clone()
    }
}

impl<F: PrimeField64> ProgramChip<F> {
    pub fn generate_air_proof_input<SC: StarkGenericConfig>(
        self,
        cached: Option<CommittedTraceData<SC>>,
    ) -> AirProofInput<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let common_trace = RowMajorMatrix::new_col(
            self.execution_frequencies
                .into_iter()
                .zip_eq(self.program.instructions_and_debug_infos.iter())
                .filter_map(|(frequency, option)| {
                    option.as_ref().map(|_| F::from_canonical_usize(frequency))
                })
                .collect::<Vec<F>>(),
        );
        if let Some(cached) = cached {
            AirProofInput {
                cached_mains_pdata: vec![(cached.commitment, cached.pcs_data)],
                raw: AirProofRawInput {
                    cached_mains: vec![cached.trace],
                    common_main: Some(common_trace),
                    public_values: vec![],
                },
            }
        } else {
            AirProofInput::cached_traces_no_pis(
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
        SystemOpcode::TERMINATE.global_opcode(),
        [0, 0, EXIT_CODE_FAIL],
    )
}
