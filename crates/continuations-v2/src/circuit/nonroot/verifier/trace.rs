use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{NonRootVerifierPvs, VERIFIER_PVS_AIR_ID};

use crate::circuit::nonroot::{
    app::*,
    verifier::air::{VerifierChildLevel, VerifierPvsCols},
};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_app: bool,
    child_dag_commit: [F; DIGEST_SIZE],
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_proofs = proofs.len();
    let height = num_proofs.next_power_of_two();
    let width = VerifierPvsCols::<u8>::width();

    debug_assert!(num_proofs > 0);

    let mut trace = vec![F::ZERO; height * width];
    let mut child_level = VerifierChildLevel::App;

    for (proof_idx, (proof, chunk)) in proofs.iter().zip(trace.chunks_exact_mut(width)).enumerate()
    {
        let cols: &mut VerifierPvsCols<F> = chunk.borrow_mut();

        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_valid = F::ONE;
        cols.is_last = F::from_bool(proof_idx + 1 == num_proofs);

        // The child_vk may either be for the app layer or an aggregation layer
        if child_is_app {
            cols.child_pvs.program_commit = proof.trace_vdata[PROGRAM_AIR_ID]
                .as_ref()
                .unwrap()
                .cached_commitments[PROGRAM_CACHED_TRACE_INDEX];

            let &VmConnectorPvs {
                initial_pc,
                final_pc,
                exit_code,
                is_terminate,
            } = proof.public_values[CONNECTOR_AIR_ID].as_slice().borrow();
            cols.child_pvs.initial_pc = initial_pc;
            cols.child_pvs.final_pc = final_pc;
            cols.child_pvs.exit_code = exit_code;
            cols.child_pvs.is_terminate = is_terminate;

            let &MemoryMerklePvs::<_, DIGEST_SIZE> {
                initial_root,
                final_root,
            } = proof.public_values[MERKLE_AIR_ID].as_slice().borrow();
            cols.child_pvs.initial_root = initial_root;
            cols.child_pvs.final_root = final_root;
        } else {
            let child_pvs: &NonRootVerifierPvs<F> =
                proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
            cols.has_verifier_pvs = F::ONE;
            cols.child_pvs = *child_pvs;

            child_level = match child_pvs.internal_flag {
                F::ZERO => VerifierChildLevel::Leaf,
                F::ONE => VerifierChildLevel::InternalForLeaf,
                F::TWO => VerifierChildLevel::InternalRecursive,
                _ => unreachable!(),
            }
        }
    }

    let last_row: &VerifierPvsCols<F> =
        trace[(proofs.len() - 1) * width..proofs.len() * width].borrow();
    let mut pvs = last_row.child_pvs;

    let first_row: &VerifierPvsCols<F> = trace[..width].borrow();
    pvs.initial_pc = first_row.child_pvs.initial_pc;
    pvs.initial_root = first_row.child_pvs.initial_root;

    match child_level {
        VerifierChildLevel::App => {
            pvs.app_dag_commit = child_dag_commit;
        }
        VerifierChildLevel::Leaf => {
            pvs.leaf_dag_commit = child_dag_commit;
            pvs.internal_flag = F::ONE;
        }
        VerifierChildLevel::InternalForLeaf => {
            pvs.internal_for_leaf_dag_commit = child_dag_commit;
            pvs.internal_flag = F::TWO;
            pvs.recursion_flag = F::ONE;
        }
        VerifierChildLevel::InternalRecursive => {
            pvs.internal_recursive_dag_commit = child_dag_commit;
            pvs.internal_flag = F::TWO;
            pvs.recursion_flag = F::TWO;
        }
    }

    AirProvingContext {
        cached_mains: vec![],
        common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
        public_values: pvs.to_vec(),
    }
}
