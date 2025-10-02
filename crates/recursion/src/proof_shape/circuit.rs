use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{DIGEST_SIZE, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage, GkrModuleBus,
        GkrModuleMessage, PublicValuesBus, StackingCommitmentsBus, StackingCommitmentsBusMessage,
        StackingWidthBusMessage, StackingWidthsBus, TranscriptBus, TranscriptBusMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyProofShapeCols<T> {
    is_summary: T,
    is_valid: T,
    tidx: T,
    air_idx: T,
    sorted_air_idx: T,
    n_logup: T,
    n_max: T,
    hypercube_dim: T,
    has_preprocessed: T,
    num_main_parts: T,
    num_interactions: T,
    commit: [T; DIGEST_SIZE],
    is_preprocessed: T,
    is_cached: T,
    part_idx: T,
    commit_idx: T,
    width: T,
    stacked_width: T,
}

pub(crate) struct DummyProofShapeAir {
    pub transcript_bus: TranscriptBus,
    pub gkr_bus: GkrModuleBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub stacking_commitments_bus: StackingCommitmentsBus,
    pub stacking_widths_bus: StackingWidthsBus,
    pub public_values_bus: PublicValuesBus,
}

impl<F> BaseAir<F> for DummyProofShapeAir {
    fn width(&self) -> usize {
        DummyProofShapeCols::<usize>::width()
    }
}

impl BaseAirWithPublicValues<F> for DummyProofShapeAir {}
impl PartitionedBaseAir<F> for DummyProofShapeAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyProofShapeAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyProofShapeCols<AB::Var> = (*local).borrow();
        let next: &DummyProofShapeCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        builder
            .when_last_row()
            .when(local.is_valid)
            .assert_one(local.is_summary);
        builder
            .when_transition()
            .when(local.is_valid)
            .when(AB::Expr::ONE - next.is_summary)
            .assert_one(local.is_summary);

        self.gkr_bus.send(
            builder,
            GkrModuleMessage {
                tidx: local.tidx,
                n_logup: local.n_logup,
                n_max: local.n_max,
            },
            local.is_summary,
        );

        self.air_shape_bus.send(
            builder,
            AirShapeBusMessage {
                sort_idx: local.sorted_air_idx,
                idx: local.air_idx,
                hypercube_dim: local.hypercube_dim,
                has_preprocessed: local.has_preprocessed,
                num_main_parts: local.num_main_parts,
                num_interactions: local.num_interactions,
            },
            AB::Expr::TWO
                * (AB::Expr::ONE - local.is_summary - local.is_preprocessed - local.is_cached),
        );
        // TODO: We should probably send common main with AirShapeBus and only use this for
        // preprocessed/cached?
        self.air_part_shape_bus.send(
            builder,
            AirPartShapeBusMessage {
                idx: local.air_idx,
                part: local.part_idx,
                width: local.width,
            },
            AB::Expr::TWO * (AB::Expr::ONE - local.is_summary),
        );
        self.stacking_commitments_bus.send(
            builder,
            StackingCommitmentsBusMessage {
                commit_idx: local.commit_idx,
                commitment: local.commit,
            },
            local.is_summary + local.is_preprocessed + local.is_cached,
        );
        self.stacking_widths_bus.send(
            builder,
            StackingWidthBusMessage {
                commit_idx: local.commit_idx,
                width: local.stacked_width,
            },
            AB::Expr::TWO * (local.is_summary + local.is_preprocessed + local.is_cached),
        );

        // Common main on summary row
        for i in 0..DIGEST_SIZE {
            self.transcript_bus.receive(
                builder,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i),
                    value: local.commit[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_summary,
            );
        }

        // TODO: public values
        // TODO: transcript
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let vk = &vk.inner;
    let preflight = &preflight.proof_shape;

    let num_parts_per_air: Vec<_> = preflight
        .sorted_trace_shapes
        .iter()
        .map(|&(air_id, _)| {
            let avk = &vk.per_air[air_id];
            1 + avk.num_cached_mains() + avk.preprocessed_data.is_some() as usize
        })
        .collect();

    let num_valid_rows = num_parts_per_air.iter().sum::<usize>() + 1;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyProofShapeCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    let stack_height = 1 << (vk.params.l_skip + vk.params.n_stack);

    let mut j = 0;
    let mut part_idx = 0;

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyProofShapeCols<F> = row.borrow_mut();

        cols.is_valid = F::ONE;
        cols.n_max = F::from_canonical_usize(preflight.n_max);

        // Summary row.
        if i == num_rows - 1 {
            cols.commit.copy_from_slice(&proof.common_main_commit);
            cols.is_summary = F::ONE;
            cols.tidx = F::from_canonical_usize(preflight.post_tidx);
            cols.stacked_width = F::from_canonical_usize(preflight.stacked_common_width);
            cols.n_logup = F::from_canonical_usize(preflight.n_logup);
        } else {
            let (air_id, shape) = &preflight.sorted_trace_shapes[j];
            let num_parts = num_parts_per_air[j];

            let avk = &vk.per_air[*air_id];

            cols.air_idx = F::from_canonical_usize(*air_id);
            cols.sorted_air_idx = F::from_canonical_usize(j);
            cols.has_preprocessed = F::from_bool(avk.preprocessed_data.is_some());
            cols.hypercube_dim = F::from_canonical_usize(shape.hypercube_dim);
            cols.num_main_parts = F::from_canonical_usize(avk.num_cached_mains() + 1);
            cols.num_interactions =
                F::from_canonical_usize(avk.symbolic_constraints.interactions.len());

            if part_idx == 1 && avk.preprocessed_data.is_some() {
                let data = avk.preprocessed_data.as_ref().unwrap();
                cols.commit = data.commit;
                cols.stacked_width = F::from_canonical_usize(data.stacking_width);
            } else if part_idx > 0 {
                let height = 1 << (shape.hypercube_dim + vk.params.l_skip);
                let cached_commit_idx = part_idx - 1 - avk.preprocessed_data.is_some() as usize;
                let num_cells = avk.params.width.cached_mains[cached_commit_idx] * height;
                cols.commit = shape.cached_commitments[cached_commit_idx];
                cols.stacked_width =
                    F::from_canonical_usize((num_cells + stack_height - 1) / stack_height);
            }
            cols.width = F::from_canonical_usize(avk.params.width.common_main);

            if part_idx == num_parts - 1 {
                j += 1;
                part_idx = 0;
            } else {
                part_idx += 1;
            }
        }
    }

    RowMajorMatrix::new(trace, width)
}
