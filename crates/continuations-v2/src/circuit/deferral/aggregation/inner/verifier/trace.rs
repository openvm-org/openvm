use std::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::VERIFIER_PVS_AIR_ID;

use crate::circuit::deferral::{
    aggregation::inner::verifier::air::{NonRootChildLevel, NonRootPvsCols},
    DeferralVerifierPvs,
};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_def: bool,
    child_dag_commit: [F; DIGEST_SIZE],
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_proofs = proofs.len();
    let height = num_proofs.next_power_of_two();
    let width = NonRootPvsCols::<u8>::width();

    debug_assert!(num_proofs > 0);

    let mut trace = vec![F::ZERO; height * width];
    let mut child_level = NonRootChildLevel::App;

    for (proof_idx, (proof, chunk)) in proofs.iter().zip(trace.chunks_exact_mut(width)).enumerate()
    {
        let cols: &mut NonRootPvsCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_valid = F::ONE;

        if child_is_def {
            cols.has_verifier_pvs = F::ONE;

            let child_pvs: &DeferralVerifierPvs<F> =
                proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
            cols.child_pvs = *child_pvs;

            child_level = match child_pvs.internal_flag {
                F::ZERO => NonRootChildLevel::Leaf,
                F::ONE => NonRootChildLevel::InternalForLeaf,
                F::TWO => NonRootChildLevel::InternalRecursive,
                _ => unreachable!(),
            };
        }
    }

    let last_row: &NonRootPvsCols<F> = trace[(num_proofs - 1) * width..num_proofs * width].borrow();
    let mut pvs = last_row.child_pvs;

    match child_level {
        NonRootChildLevel::App => {
            pvs.def_dag_commit = child_dag_commit;
        }
        NonRootChildLevel::Leaf => {
            pvs.leaf_dag_commit = child_dag_commit;
            pvs.internal_flag = F::ONE;
        }
        NonRootChildLevel::InternalForLeaf => {
            pvs.internal_for_leaf_dag_commit = child_dag_commit;
            pvs.internal_flag = F::TWO;
            pvs.recursion_flag = F::ONE;
        }
        NonRootChildLevel::InternalRecursive => {
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
