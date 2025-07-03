use std::{borrow::BorrowMut, sync::Arc};

use derivative::Derivative;
use itertools::Itertools;
use openvm_circuit::arch::hasher::poseidon2::Poseidon2Hasher;
use openvm_instructions::{exe::VmExe, program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::{
    config::{Com, PcsProverData, StarkGenericConfig, Val},
    p3_commit::Pcs,
    p3_field::{Field, FieldAlgebra, PrimeField32, PrimeField64},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
};
use serde::{Deserialize, Serialize};

use super::{Instruction, ProgramExecutionCols, EXIT_CODE_FAIL};
use crate::{
    arch::{
        hasher::{poseidon2::vm_poseidon2_hasher, Hasher},
        MemoryConfig,
    },
    system::{
        memory::{merkle::MerkleTree, AddressMap, CHUNK},
        program::ProgramChip,
    },
};

/// **Note**: this struct stores the program ROM twice: once in [VmExe] and once as a cached trace
/// matrix `trace`.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "VmExe<Val<SC>>: Serialize, Com<SC>: Serialize, PcsProverData<SC>: Serialize",
    deserialize = "VmExe<Val<SC>>: Deserialize<'de>, Com<SC>: Deserialize<'de>, PcsProverData<SC>: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct VmCommittedExe<SC: StarkGenericConfig> {
    /// Raw executable.
    pub exe: VmExe<Val<SC>>,
    pub commitment: Com<SC>,
    /// Program ROM as cached trace matrix.
    pub trace: Arc<RowMajorMatrix<Val<SC>>>,
    pub prover_data: Arc<PcsProverData<SC>>,
}

impl<SC: StarkGenericConfig> VmCommittedExe<SC>
where
    Val<SC>: PrimeField32,
{
    /// Creates [VmCommittedExe] from [VmExe] by using `pcs` to commit to the
    /// program code as a _cached trace_ matrix.
    pub fn commit(exe: VmExe<Val<SC>>, pcs: &SC::Pcs) -> Self {
        let trace = generate_cached_trace(&exe.program);
        let domain = pcs.natural_domain_for_degree(trace.height());

        let (commitment, data) = pcs.commit(vec![(domain, trace.clone())]);
        Self {
            exe,
            commitment,
            trace: Arc::new(trace),
            prover_data: Arc::new(data),
        }
    }
    pub fn get_program_commit(&self) -> Com<SC> {
        self.commitment.clone()
    }

    /// Computes a commitment to [VmCommittedExe]. This is a Merklelized hash of:
    /// - Program code commitment (commitment of the cached trace)
    /// - Merkle root of the initial memory
    /// - Starting program counter (`pc_start`)
    ///
    /// The program code commitment is itself a commitment (via the proof system PCS) to
    /// the program code.
    ///
    /// The Merklelization uses Poseidon2 as a cryptographic hash function (for the leaves)
    /// and a cryptographic compression function (for internal nodes).
    ///
    /// **Note**: This function recomputes the Merkle tree for the initial memory image.
    pub fn compute_exe_commit(&self, memory_config: &MemoryConfig) -> Com<SC>
    where
        Com<SC>: AsRef<[Val<SC>; CHUNK]> + From<[Val<SC>; CHUNK]>,
    {
        let hasher = vm_poseidon2_hasher();
        let memory_dimensions = memory_config.memory_dimensions();
        let app_program_commit: &[Val<SC>; CHUNK] = self.commitment.as_ref();
        let mem_config = memory_config;
        let memory_image = AddressMap::from_sparse(
            mem_config.addr_space_sizes.clone(),
            self.exe.init_memory.clone(),
        );
        let init_memory_commit =
            MerkleTree::from_memory(&memory_image, &memory_dimensions, &hasher).root();
        Com::<SC>::from(compute_exe_commit(
            &hasher,
            app_program_commit,
            &init_memory_commit,
            Val::<SC>::from_canonical_u32(self.exe.pc_start),
        ))
    }
}

impl<SC: StarkGenericConfig> ProgramChip<SC> {
    pub fn generate_proving_ctx(self) -> AirProvingContext<CpuBackend<SC>> {
        assert_eq!(
            self.filtered_exec_frequencies.len(),
            self.cached.trace.height()
        );
        let common_trace = RowMajorMatrix::new_col(
            self.filtered_exec_frequencies
                .into_par_iter()
                .map(Val::<SC>::from_canonical_u32)
                .collect::<Vec<_>>(),
        );
        AirProvingContext {
            cached_mains: vec![self.cached],
            common_main: Some(Arc::new(common_trace)),
            public_values: vec![],
        }
    }
}

/// Computes a Merklelized hash of:
/// - Program code commitment (commitment of the cached trace)
/// - Merkle root of the initial memory
/// - Starting program counter (`pc_start`)
///
/// The Merklelization uses [Poseidon2Hasher] as a cryptographic hash function (for the leaves)
/// and a cryptographic compression function (for internal nodes).
pub fn compute_exe_commit<F: PrimeField32>(
    hasher: &Poseidon2Hasher<F>,
    program_commit: &[F; CHUNK],
    init_memory_root: &[F; CHUNK],
    pc_start: F,
) -> [F; CHUNK] {
    let mut padded_pc_start = [F::ZERO; CHUNK];
    padded_pc_start[0] = pc_start;
    let program_hash = hasher.hash(program_commit);
    let memory_hash = hasher.hash(init_memory_root);
    let pc_hash = hasher.hash(&padded_pc_start);
    hasher.compress(&hasher.compress(&program_hash, &memory_hash), &pc_hash)
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
