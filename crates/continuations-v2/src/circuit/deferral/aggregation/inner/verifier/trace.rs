use std::borrow::{Borrow, BorrowMut};

use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{VerifierBasePvs, VERIFIER_PVS_AIR_ID};

use crate::circuit::deferral::aggregation::inner::verifier::air::{
    DeferralChildLevel, DeferralVerifierPvsCols,
};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_def: bool,
    child_dag_commit: [F; DIGEST_SIZE],
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_proofs = proofs.len();
    let height = num_proofs.next_power_of_two();
    let width = DeferralVerifierPvsCols::<u8>::width();

    debug_assert!(num_proofs > 0);

    let mut trace = vec![F::ZERO; height * width];
    let mut child_level = DeferralChildLevel::App;

    for (proof_idx, (proof, chunk)) in proofs.iter().zip(trace.chunks_exact_mut(width)).enumerate()
    {
        let cols: &mut DeferralVerifierPvsCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_valid = F::ONE;

        if child_is_def {
            cols.has_verifier_pvs = F::ONE;

            let child_pvs: &VerifierBasePvs<F> =
                proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
            cols.child_pvs = *child_pvs;

            child_level = match child_pvs.internal_flag {
                F::ZERO => DeferralChildLevel::Leaf,
                F::ONE => DeferralChildLevel::InternalForLeaf,
                F::TWO => DeferralChildLevel::InternalRecursive,
                _ => unreachable!(),
            };
        }
    }

    let last_row: &DeferralVerifierPvsCols<F> =
        trace[(num_proofs - 1) * width..num_proofs * width].borrow();
    let mut pvs = last_row.child_pvs;

    // Note app_dag_commit is def_dag_commit here
    match child_level {
        DeferralChildLevel::App => {
            pvs.app_dag_commit = child_dag_commit;
        }
        DeferralChildLevel::Leaf => {
            pvs.leaf_dag_commit = child_dag_commit;
            pvs.internal_flag = F::ONE;
        }
        DeferralChildLevel::InternalForLeaf => {
            pvs.internal_for_leaf_dag_commit = child_dag_commit;
            pvs.internal_flag = F::TWO;
            pvs.recursion_flag = F::ONE;
        }
        DeferralChildLevel::InternalRecursive => {
            pvs.internal_recursive_dag_commit = child_dag_commit;
            pvs.internal_flag = F::TWO;
            pvs.recursion_flag = F::TWO;
        }
    }

    AirProvingContext {
        cached_mains: vec![],
        common_main: RowMajorMatrix::new(trace, width),
        public_values: pvs.to_vec(),
    }
}
