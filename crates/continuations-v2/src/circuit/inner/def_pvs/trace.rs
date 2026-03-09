use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{DeferralPvs, DEF_PVS_AIR_ID};

use crate::{
    circuit::{
        deferral::DEF_HOOK_PVS_AIR_ID,
        inner::{def_pvs::air::DeferralPvsCols, ProofsType},
    },
    utils::digests_to_poseidon2_input,
};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    proofs_type: ProofsType,
    child_is_app: bool,
    absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    assert!(
        absent_trace_pvs.is_none()
            || (matches!(proofs_type, ProofsType::Deferral) && proofs.len() == 1),
        "absent_trace_pvs is only valid for single-proof deferral aggregation"
    );
    let mut proof_idxs = vec![];
    let (num_rows, def_flag) = match proofs_type {
        ProofsType::Vm => (1, 0),
        ProofsType::Deferral => {
            proof_idxs = (0..proofs.len()).collect_vec();
            (proofs.len() + absent_trace_pvs.is_some() as usize, 1)
        }
        ProofsType::Mix => {
            proof_idxs.push(1);
            (1, 1)
        }
        ProofsType::Combined => {
            proof_idxs.push(0);
            (1, 2)
        }
    };

    let width = DeferralPvsCols::<u8>::width();
    let mut trace = vec![F::ZERO; num_rows * width];
    let mut chunks = trace.chunks_exact_mut(width);

    let mut child_pvs_vec = vec![];
    let single_present_is_right = if let Some((_, is_right)) = absent_trace_pvs.as_ref() {
        *is_right
    } else {
        false
    };

    for (row_idx, proof_idx) in proof_idxs.iter().enumerate() {
        let proof = &proofs[*proof_idx];
        let chunk = chunks.next().unwrap();
        let cols: &mut DeferralPvsCols<F> = chunk.borrow_mut();
        cols.row_idx = F::from_usize(row_idx);
        cols.proof_idx = F::from_usize(*proof_idx);
        cols.is_present = F::ONE;
        cols.deferral_flag = F::from_usize(def_flag);
        cols.has_verifier_pvs = F::from_bool(!child_is_app);
        cols.single_present_is_right = F::from_bool(single_present_is_right);

        let air_id = if child_is_app {
            DEF_HOOK_PVS_AIR_ID
        } else {
            DEF_PVS_AIR_ID
        };
        let child_pvs: &DeferralPvs<_> = proof.public_values[air_id].as_slice().borrow();
        cols.child_pvs = *child_pvs;
        child_pvs_vec.push(cols.child_pvs);
    }

    if let Some((pvs, _)) = absent_trace_pvs {
        let chunk = chunks.next().unwrap();
        let cols: &mut DeferralPvsCols<F> = chunk.borrow_mut();
        cols.row_idx = F::ONE;
        cols.deferral_flag = F::from_usize(def_flag);
        cols.has_verifier_pvs = F::from_bool(!child_is_app);
        cols.single_present_is_right = F::from_bool(single_present_is_right);
        cols.child_pvs = pvs;
        child_pvs_vec.push(cols.child_pvs);
    }

    let mut poseidon2_inputs = vec![];
    let mut public_values = vec![F::ZERO; DeferralPvs::<u8>::width()];
    let pvs: &mut DeferralPvs<F> = public_values.as_mut_slice().borrow_mut();

    if child_pvs_vec.len() == 1 {
        *pvs = child_pvs_vec[0];
    } else if child_pvs_vec.len() == 2 {
        let first_child = child_pvs_vec[0];
        let second_child = child_pvs_vec[1];
        let (left_initial, right_initial, left_final, right_final) = if single_present_is_right {
            (
                second_child.initial_acc_hash,
                first_child.initial_acc_hash,
                second_child.final_acc_hash,
                first_child.final_acc_hash,
            )
        } else {
            (
                first_child.initial_acc_hash,
                second_child.initial_acc_hash,
                first_child.final_acc_hash,
                second_child.final_acc_hash,
            )
        };
        pvs.initial_acc_hash = poseidon2_compress_with_capacity(left_initial, right_initial).0;
        poseidon2_inputs.push(digests_to_poseidon2_input(left_initial, right_initial));
        pvs.final_acc_hash = poseidon2_compress_with_capacity(left_final, right_final).0;
        poseidon2_inputs.push(digests_to_poseidon2_input(left_final, right_final));
        pvs.depth = first_child.depth + F::ONE;
    }

    (
        AirProvingContext {
            cached_mains: vec![],
            common_main: RowMajorMatrix::new(trace, width),
            public_values,
        },
        poseidon2_inputs,
    )
}
