use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{DagCommit, VerifierBasePvs, VerifierDefPvs, VERIFIER_PVS_AIR_ID};

use crate::circuit::{
    inner::{
        verifier::air::{VerifierCombinedPvs, VerifierDeferralCols, VerifierPvsCols},
        ProofsType,
    },
    subair::hash_slice_trace,
    SingleAirTraceData,
};

#[derive(Copy, Clone)]
pub enum VerifierChildLevel {
    App,
    Leaf,
    InternalForLeaf,
    InternalRecursive,
}

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    proofs_type: ProofsType,
    child_is_app: bool,
    child_dag_commit: DagCommit<F>,
    deferral_enabled: bool,
) -> SingleAirTraceData<CpuBackend<BabyBearPoseidon2Config>> {
    let num_proofs = proofs.len();
    debug_assert!(num_proofs > 0);

    if !deferral_enabled {
        assert!(matches!(proofs_type, ProofsType::Vm))
    }

    let mut child_level = VerifierChildLevel::App;

    if !child_is_app {
        let proof = &proofs[0];
        let child_pvs: &VerifierBasePvs<F> = proof.public_values[VERIFIER_PVS_AIR_ID].as_slice()
            [0..VerifierBasePvs::<F>::width()]
            .borrow();
        child_level = match child_pvs.internal_flag {
            F::ZERO => VerifierChildLevel::Leaf,
            F::ONE => VerifierChildLevel::InternalForLeaf,
            F::TWO => VerifierChildLevel::InternalRecursive,
            _ => unreachable!(),
        };
    }

    let height = num_proofs.next_power_of_two();
    let base_width = VerifierPvsCols::<u8>::width();
    let def_width = if deferral_enabled {
        VerifierDeferralCols::<u8>::width()
    } else {
        0
    };
    let width = base_width + def_width;

    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);
    let mut poseidon2_compress_inputs = vec![];
    let mut poseidon2_permute_inputs = vec![];
    let mut trailing_deferral_flag = F::ZERO;

    for (proof_idx, proof) in proofs.iter().enumerate() {
        let chunk = chunks.next().unwrap();
        let (base_chunk, def_chunk) = chunk.split_at_mut(base_width);

        let cols: &mut VerifierPvsCols<F> = base_chunk.borrow_mut();
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_valid = F::ONE;

        if deferral_enabled {
            let def_cols: &mut VerifierDeferralCols<_> = def_chunk.borrow_mut();
            def_cols.is_last = F::from_bool(proof_idx + 1 == proofs.len());
            if matches!(proofs_type, ProofsType::Deferral) {
                def_cols.child_pvs.deferral_flag = F::ONE;
                trailing_deferral_flag = def_cols.child_pvs.deferral_flag;
            }
        }

        if !child_is_app {
            let pv_chunk = proof.public_values[VERIFIER_PVS_AIR_ID].as_slice();
            let (base_pv_chunk, def_pv_chunk) = pv_chunk.split_at(VerifierBasePvs::<u8>::width());

            let base_pvs: &VerifierBasePvs<_> = base_pv_chunk.borrow();
            cols.has_verifier_pvs = F::ONE;
            cols.child_pvs = *base_pvs;

            if deferral_enabled {
                let def_cols: &mut VerifierDeferralCols<_> = def_chunk.borrow_mut();
                let def_pvs: &VerifierDefPvs<_> = def_pv_chunk.borrow();
                def_cols.child_pvs = *def_pvs;
                trailing_deferral_flag = def_pvs.deferral_flag;
            }
        }
    }

    if deferral_enabled {
        for chunk in chunks {
            let (_, def_chunk) = chunk.split_at_mut(base_width);
            let def_cols: &mut VerifierDeferralCols<_> = def_chunk.borrow_mut();
            def_cols.child_pvs.deferral_flag = trailing_deferral_flag;
        }
    }

    let first_row: &VerifierPvsCols<F> = trace[..base_width].borrow();
    let mut base_pvs = first_row.child_pvs;

    match child_level {
        VerifierChildLevel::App => {
            base_pvs.app_dag_commit = child_dag_commit;
        }
        VerifierChildLevel::Leaf => {
            base_pvs.leaf_dag_commit = child_dag_commit;
            base_pvs.internal_flag = F::ONE;
        }
        VerifierChildLevel::InternalForLeaf => {
            base_pvs.internal_for_leaf_dag_commit = child_dag_commit;
            base_pvs.internal_flag = F::TWO;
            base_pvs.recursion_flag = F::ONE;
        }
        VerifierChildLevel::InternalRecursive => {
            base_pvs.internal_recursive_dag_commit = child_dag_commit;
            base_pvs.internal_flag = F::TWO;
            base_pvs.recursion_flag = F::TWO;
        }
    }

    let deferral_flag_pv = match proofs_type {
        ProofsType::Vm => F::ZERO,
        ProofsType::Deferral => F::ONE,
        ProofsType::Mix => {
            assert_eq!(num_proofs, 2);
            F::TWO
        }
        ProofsType::Combined => {
            assert_eq!(num_proofs, 1);
            F::TWO
        }
    };

    let mut def_hook_vk_commit = None;
    if deferral_enabled && deferral_flag_pv == F::ONE && base_pvs.internal_flag == F::TWO {
        let hash_elements = [
            base_pvs.app_dag_commit.cached_commit,
            base_pvs.app_dag_commit.vk_pre_hash,
            base_pvs.leaf_dag_commit.cached_commit,
            base_pvs.leaf_dag_commit.vk_pre_hash,
            base_pvs.internal_for_leaf_dag_commit.cached_commit,
            base_pvs.internal_for_leaf_dag_commit.vk_pre_hash,
        ];

        let mut row_compress_inputs = vec![];
        let mut row_permute_inputs = vec![];
        let (intermediate_states_vec, computed_def_hook_vk_commit) = hash_slice_trace(
            &hash_elements,
            Some(&mut row_permute_inputs),
            Some(&mut row_compress_inputs),
        );
        let intermediate_states: [[F; POSEIDON2_WIDTH]; 5] =
            intermediate_states_vec.try_into().unwrap();

        for chunk in trace.chunks_exact_mut(width) {
            let (_, def_chunk) = chunk.split_at_mut(base_width);
            let def_cols: &mut VerifierDeferralCols<_> = def_chunk.borrow_mut();
            def_cols.intermediate_states = intermediate_states;
        }

        for &input in &row_compress_inputs {
            poseidon2_compress_inputs.extend((0..height).map(|_| input));
        }
        for &input in &row_permute_inputs {
            poseidon2_permute_inputs.extend((0..height).map(|_| input));
        }
        def_hook_vk_commit = Some(computed_def_hook_vk_commit);
    }

    let public_values = if deferral_enabled {
        let last_row_def: &VerifierDeferralCols<F> =
            trace[(num_proofs - 1) * width + base_width..num_proofs * width].borrow();
        let mut def_pvs = last_row_def.child_pvs;
        def_pvs.deferral_flag = deferral_flag_pv;

        if let Some(def_hook_vk_commit) = def_hook_vk_commit {
            def_pvs.def_hook_vk_commit = def_hook_vk_commit;
        }

        let mut combined = vec![F::ZERO; VerifierCombinedPvs::<u8>::width()];
        let combined_pvs: &mut VerifierCombinedPvs<F> = combined.as_mut_slice().borrow_mut();
        combined_pvs.base = base_pvs;
        combined_pvs.def = def_pvs;
        combined
    } else {
        base_pvs.to_vec()
    };

    SingleAirTraceData {
        air_proving_ctx: AirProvingContext {
            cached_mains: vec![],
            common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
            public_values,
        },
        poseidon2_compress_inputs,
        poseidon2_permute_inputs,
    }
}
