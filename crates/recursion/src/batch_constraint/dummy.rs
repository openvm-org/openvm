use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage,
        BatchConstraintModuleBus, BatchConstraintModuleMessage, ColumnClaimsBus,
        ColumnClaimsMessage, ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus,
        StackingModuleBus, StackingModuleMessage, TranscriptBus, TranscriptBusMessage,
        XiRandomnessBus, XiRandomnessMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct BatchConstraintDummyCols<T> {
    proof_idx: T,
    stacking_module_msg: StackingModuleMessage<T>,
    has_stacking_module_msg: T,
    batch_constraint_module_msg: BatchConstraintModuleMessage<T>,
    has_batch_constraint_module_msg: T,
    xi_randomness_msg: XiRandomnessMessage<T>,
    has_xi_ranodmness_msg: T,
    constraint_sumcheck_rnd_msg: ConstraintSumcheckRandomness<T>,
    has_constraint_sumcheck_rnd: T,
    air_shape_bus_msg: AirShapeBusMessage<T>,
    has_air_shape_bus_msg: T,
    air_part_shape_bus_msg: AirPartShapeBusMessage<T>,
    has_air_part_shape_bus_msg: T,
    column_claims_bus_msg: ColumnClaimsMessage<T>,
    has_column_claims_bus_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_transcript_msg: T,
}

pub struct BatchConstraintDummyAir {
    pub bc_module_bus: BatchConstraintModuleBus,
    pub stacking_module_bus: StackingModuleBus,
    pub xi_randomness_bus: XiRandomnessBus,
    pub batch_constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for BatchConstraintDummyAir {}
impl PartitionedBaseAir<F> for BatchConstraintDummyAir {}

impl<F> BaseAir<F> for BatchConstraintDummyAir {
    fn width(&self) -> usize {
        BatchConstraintDummyCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for BatchConstraintDummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &BatchConstraintDummyCols<AB::Var> = (*local).borrow();

        self.bc_module_bus.receive(
            builder,
            local.proof_idx,
            local.batch_constraint_module_msg.clone(),
            local.has_batch_constraint_module_msg,
        );
        self.stacking_module_bus.send(
            builder,
            local.proof_idx,
            local.stacking_module_msg.clone(),
            local.has_stacking_module_msg,
        );
        self.xi_randomness_bus.receive(
            builder,
            local.proof_idx,
            local.xi_randomness_msg.clone(),
            local.has_xi_ranodmness_msg,
        );
        self.batch_constraint_randomness_bus.send(
            builder,
            local.proof_idx,
            local.constraint_sumcheck_rnd_msg.clone(),
            local.has_constraint_sumcheck_rnd,
        );
        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            local.air_shape_bus_msg.clone(),
            local.has_air_shape_bus_msg,
        );
        self.air_part_shape_bus.receive(
            builder,
            local.proof_idx,
            local.air_part_shape_bus_msg.clone(),
            local.has_air_part_shape_bus_msg,
        );
        self.column_claims_bus.send(
            builder,
            local.proof_idx,
            local.column_claims_bus_msg.clone(),
            local.has_column_claims_bus_msg,
        );
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            local.transcript_msg.clone(),
            local.has_transcript_msg,
        )
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let mut stacking_module_msgs = [StackingModuleMessage {
        tidx: F::from_canonical_usize(preflight.batch_constraint.post_tidx),
    }]
    .into_iter();

    let mut batch_constraint_module_msgs =
        preflight.batch_constraint_module_msgs(proof).into_iter();
    let mut xi_randomness_msgs = preflight.xi_randomness_messages().into_iter();
    let mut constraint_sc_msgs = preflight.batch_constraint_sumcheck_randomness().into_iter();
    let mut air_shape_bus_msgs = preflight.air_bus_msgs(vk).into_iter();
    let mut air_part_shape_bus_msgs = preflight.air_part_bus_msgs(vk).into_iter();
    let mut column_claims_bus_msgs = preflight.column_claims_messages(vk, proof).into_iter();
    let mut transcript_msgs = preflight
        // Sample alpha and beta
        .transcript_msgs(
            preflight.proof_shape.post_tidx + 2,
            preflight.proof_shape.post_tidx + 2 + 2 * D_EF,
        )
        .into_iter()
        .chain(preflight.transcript_msgs(
            preflight.gkr.post_tidx,
            preflight.batch_constraint.post_tidx,
        ))
        .collect::<Vec<_>>()
        .into_iter();

    let num_valid_rows = [
        stacking_module_msgs.len(),
        batch_constraint_module_msgs.len(),
        xi_randomness_msgs.len(),
        constraint_sc_msgs.len(),
        air_shape_bus_msgs.len(),
        air_part_shape_bus_msgs.len(),
        column_claims_bus_msgs.len(),
        transcript_msgs.len(),
    ]
    .into_iter()
    .max()
    .unwrap();

    let num_rows = num_valid_rows.next_power_of_two();
    let width = BatchConstraintDummyCols::<usize>::width();
    let mut trace = vec![F::ZERO; num_rows * width];

    for row in trace.chunks_mut(width).take(num_valid_rows) {
        let cols: &mut BatchConstraintDummyCols<F> = row.borrow_mut();
        if let Some(msg) = stacking_module_msgs.next() {
            cols.stacking_module_msg = msg;
            cols.has_stacking_module_msg = F::ONE;
        }
        if let Some(msg) = batch_constraint_module_msgs.next() {
            cols.batch_constraint_module_msg = msg;
            cols.has_batch_constraint_module_msg = F::ONE;
        }
        if let Some(msg) = xi_randomness_msgs.next() {
            cols.xi_randomness_msg = msg;
            cols.has_xi_ranodmness_msg = F::ONE;
        }
        if let Some(msg) = constraint_sc_msgs.next() {
            cols.constraint_sumcheck_rnd_msg = msg;
            cols.has_constraint_sumcheck_rnd = F::ONE;
        }
        if let Some(msg) = air_shape_bus_msgs.next() {
            cols.air_shape_bus_msg = msg;
            cols.has_air_shape_bus_msg = F::ONE;
        }
        if let Some(msg) = air_part_shape_bus_msgs.next() {
            cols.air_part_shape_bus_msg = msg;
            cols.has_air_part_shape_bus_msg = F::ONE;
        }
        if let Some(msg) = column_claims_bus_msgs.next() {
            cols.column_claims_bus_msg = msg;
            cols.has_column_claims_bus_msg = F::ONE;
        }
        if let Some(msg) = transcript_msgs.next() {
            cols.transcript_msg = msg;
            cols.has_transcript_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
