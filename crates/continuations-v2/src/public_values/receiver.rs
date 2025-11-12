use std::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use recursion_circuit::bus::{PublicValuesBus, PublicValuesBusMessage};
use stark_backend_v2::{
    F,
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::public_values::{
    app::*,
    verifier::{CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UserPvsReceiverCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub pv_bus_msg: PublicValuesBusMessage<F>,
}

pub fn generate_proving_ctx(
    proofs: &[Proof],
    app_child: bool,
) -> AirProvingContextV2<CpuBackendV2> {
    const APP_RESERVED_IDX: [usize; 3] = [PROGRAM_AIR_ID, CONNECTOR_AIR_ID, MERKLE_AIR_ID];
    const VERIFIER_RESERVED_IDX: [usize; 2] = [VERIFIER_PVS_AIR_ID, CONSTRAINT_EVAL_AIR_ID];

    let reserved_air_idx = if app_child {
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
                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.pv_bus_msg.air_idx = F::from_canonical_usize(air_idx);
                cols.pv_bus_msg.pv_idx = F::from_canonical_usize(pv_idx);
                cols.pv_bus_msg.value = *value;
            }
        }
    }

    AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&RowMajorMatrix::new(
        trace, width,
    )))
}

pub struct UserPvsReceiverAir {
    pub public_values_bus: PublicValuesBus,
}

impl<F> BaseAir<F> for UserPvsReceiverAir {
    fn width(&self) -> usize {
        UserPvsReceiverCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsReceiverAir {}
impl<F> PartitionedBaseAir<F> for UserPvsReceiverAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UserPvsReceiverAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &UserPvsReceiverCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_valid);

        // TODO[INT-5302]: receive user public values from proof shape
    }
}
