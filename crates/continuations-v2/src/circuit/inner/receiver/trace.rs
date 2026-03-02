use std::borrow::BorrowMut;

use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID};

use crate::circuit::inner::{app::*, receiver::air::UserPvsReceiverCols};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_app: bool,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    const APP_RESERVED_IDX: [usize; 3] = [PROGRAM_AIR_ID, CONNECTOR_AIR_ID, MERKLE_AIR_ID];
    const VERIFIER_RESERVED_IDX: [usize; 3] =
        [VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID, CONSTRAINT_EVAL_AIR_ID];

    let reserved_air_idx = if child_is_app {
        APP_RESERVED_IDX.as_slice()
    } else {
        VERIFIER_RESERVED_IDX.as_slice()
    };

    let num_pvs = proofs[0]
        .public_values
        .iter()
        .enumerate()
        .fold(0usize, |acc, (air_idx, pvs)| {
            if reserved_air_idx.contains(&air_idx) {
                acc
            } else {
                acc + pvs.len()
            }
        });

    let height = num_pvs.next_power_of_two();
    let width = UserPvsReceiverCols::<u8>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);

    for (proof_idx, proof) in proofs.iter().enumerate() {
        for (air_idx, pvs) in proof
            .public_values
            .iter()
            .enumerate()
            .filter(|(i, _)| !reserved_air_idx.contains(i))
        {
            for (pv_idx, value) in pvs.iter().enumerate() {
                let chunk = chunks.next().unwrap();
                let cols: &mut UserPvsReceiverCols<F> = chunk.borrow_mut();
                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.pv_bus_msg.air_idx = F::from_usize(air_idx);
                cols.pv_bus_msg.pv_idx = F::from_usize(pv_idx);
                cols.pv_bus_msg.value = *value;
            }
        }
    }

    AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&RowMajorMatrix::new(
        trace, width,
    )))
}
