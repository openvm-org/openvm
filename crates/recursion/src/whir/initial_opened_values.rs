use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{StackingIndexMessage, StackingIndicesBus},
    system::Preflight,
    whir::bus::{VerifyQueryBus, VerifyQueryBusMessage, WhirFoldingBus, WhirFoldingBusMessage},
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct InitialOpenedValuesCols<T> {
    is_valid: T,
    is_first_in_query: T,
    is_final_group: T,
    is_codeword_built: T,
    proof_idx: T,
    query_idx: T,
    is_query_zero: T,
    is_codeword_idx_zero: T,
    is_last_in_query: T,
    commit_idx: T,
    col_idx: T,
    coset_idx: T,
    codeword_value_acc: [T; 4],
    twiddle: T,
    zi_root: T,
    zi: T,
    yi: [T; D_EF],
    stacking_indices_bus_msg: StackingIndexMessage<T>,
    has_stacking_indices_bus_msg: T,
}

pub struct InitialOpenedValuesAir {
    pub stacking_indices_bus: StackingIndicesBus,
    pub verify_query_bus: VerifyQueryBus,
    pub folding_bus: WhirFoldingBus,
    pub k: usize,
}

impl BaseAirWithPublicValues<F> for InitialOpenedValuesAir {}
impl PartitionedBaseAir<F> for InitialOpenedValuesAir {}

impl<F> BaseAir<F> for InitialOpenedValuesAir {
    fn width(&self) -> usize {
        InitialOpenedValuesCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for InitialOpenedValuesAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &InitialOpenedValuesCols<AB::Var> = (*local).borrow();

        self.stacking_indices_bus.receive(
            builder,
            local.proof_idx,
            StackingIndexMessage {
                commit_idx: local.commit_idx,
                col_idx: local.col_idx,
            },
            // FIXME: this should be only `local.is_first_in_group`, but we need stackking module
            // to send this `num_queries` times.
            local.is_codeword_idx_zero * local.is_query_zero,
        );
        self.verify_query_bus.receive(
            builder,
            local.proof_idx,
            VerifyQueryBusMessage {
                whir_round: AB::Expr::ZERO,
                query_idx: local.query_idx.into(),
                zi_root: local.zi_root.into(),
                zi: local.zi.into(),
                yi: local.yi.map(Into::into),
            },
            local.is_first_in_query,
        );
        self.folding_bus.send(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: AB::Expr::ZERO,
                query_idx: local.query_idx.into(),
                height: AB::Expr::ZERO,
                coset_shift: local.zi_root.into(),
                coset_idx: local.coset_idx.into(),
                coset_size: AB::Expr::from_canonical_usize(1 << self.k),
                twiddle: local.twiddle.into(),
                value: local.codeword_value_acc.map(Into::into),
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            local.is_codeword_built,
        )
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let k_whir = vk.inner.params.k_whir;
    let num_valid_rows = proof
        .whir_proof
        .initial_round_opened_rows
        .iter()
        .flat_map(|opened_rows| opened_rows.iter().flatten())
        .count();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = InitialOpenedValuesCols::<F>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    // for query_idx in 0..num_queries {
    //     for commit_idx in num_commits {
    //         for col_idx in num_columns[commit_idx] {
    //             for codeword_idx in 0..1 << k_whir {
    //                  // generate row
    //             }
    //         }
    //     }
    // }

    let omega_k = F::two_adic_generator(k_whir);

    let mut query_idx = 0;
    let mut commit_idx = 0;
    let mut col_idx = 0;
    let mut codeword_idx = 0;

    for row in trace.chunks_mut(width).take(num_valid_rows) {
        let cols: &mut InitialOpenedValuesCols<F> = row.borrow_mut();

        cols.is_valid = F::ONE;
        cols.is_first_in_query = F::from_bool(commit_idx == 0 && col_idx == 0 && codeword_idx == 0);
        cols.is_query_zero = F::from_bool(query_idx == 0);
        cols.is_codeword_idx_zero = F::from_bool(codeword_idx == 0);
        cols.query_idx = F::from_canonical_usize(query_idx);
        cols.commit_idx = F::from_canonical_usize(commit_idx);
        cols.coset_idx = F::from_canonical_usize(codeword_idx);
        cols.twiddle = omega_k.exp_u64(codeword_idx as u64);
        cols.codeword_value_acc = preflight.whir.initial_round_coset_vals[query_idx][codeword_idx]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.col_idx = F::from_canonical_usize(col_idx);
        cols.zi_root = preflight.whir.zj_roots[0][query_idx];
        cols.zi = preflight.whir.zjs[0][query_idx];
        cols.yi = preflight.whir.yjs[0][query_idx]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.is_codeword_built = F::from_bool(
            (col_idx + 1) * (1 << k_whir)
                == proof.whir_proof.initial_round_opened_rows[commit_idx][0].len()
                && commit_idx == proof.whir_proof.initial_round_opened_rows.len() - 1,
        );

        codeword_idx += 1;
        if codeword_idx % (1 << k_whir) == 0 {
            col_idx += 1;
            codeword_idx = 0;
        }
        if col_idx * (1 << k_whir)
            == proof.whir_proof.initial_round_opened_rows[commit_idx][0].len()
        {
            commit_idx += 1;
            col_idx = 0;
        }
        if commit_idx == proof.whir_proof.initial_round_opened_rows.len() {
            commit_idx = 0;
            query_idx += 1;
        }
    }

    RowMajorMatrix::new(trace, width)
}
