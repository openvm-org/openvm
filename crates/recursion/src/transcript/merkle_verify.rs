use core::borrow::{Borrow, BorrowMut};

use crate::bus::{
    CommitmentsBus, CommitmentsBusMessage, MerkleVerifyBus, MerkleVerifyBusMessage, Poseidon2Bus,
};
use crate::system::Preflight;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::Matrix;
use stark_backend_v2::{DIGEST_SIZE, F, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

pub const CHUNK: usize = 8;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MerkleVerifyCols<T> {
    pub proof_idx: T,
    pub is_valid: T,
    pub merkle_verify_bus_msg: MerkleVerifyBusMessage<T>,
    pub commitment: [T; DIGEST_SIZE],
    // TODO: other columns to actually constrain the merkle proofs
}

pub struct MerkleVerifyAir {
    pub poseidon2_bus: Poseidon2Bus,
    pub merkle_verify_bus: MerkleVerifyBus,
    pub commitments_bus: CommitmentsBus,
}

impl<F: Field> BaseAir<F> for MerkleVerifyAir {
    fn width(&self) -> usize {
        MerkleVerifyCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MerkleVerifyAir {}
impl<F: Field> PartitionedBaseAir<F> for MerkleVerifyAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for MerkleVerifyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MerkleVerifyCols<AB::Var> = (*local).borrow();
        let _next: &MerkleVerifyCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // TODO: enable this when WHIR's OpenedValuesAir send
        // self.merkle_verify_bus.receive(
        //     builder,
        //     local.proof_idx,
        //     local.merkle_verify_bus_msg.clone(),
        //     local.is_valid,
        // );
        self.commitments_bus.receive(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: local.merkle_verify_bus_msg.commit_major,
                minor_idx: local.merkle_verify_bus_msg.commit_minor,
                commitment: local.commitment,
            },
            local.is_valid,
        );
    }
}

pub fn generate_trace(_proof: &Proof, preflight: &Preflight) -> Vec<F> {
    let width = MerkleVerifyCols::<F>::width();
    let num_valid_rows = preflight.merkle_verify_logs.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let mut trace = vec![F::ZERO; num_rows.next_power_of_two() * width];
    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut MerkleVerifyCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.merkle_verify_bus_msg = preflight.merkle_verify_logs[i].0.clone();
        cols.commitment = preflight.merkle_verify_logs[i].1;
    }

    trace
}
