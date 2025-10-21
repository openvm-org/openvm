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
    system::Preflight,
    whir::bus::{VerifyQueryBus, VerifyQueryBusMessage, WhirFoldingBus, WhirFoldingBusMessage},
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct NonInitialOpenedValuesCols<T> {
    is_valid: T,
    proof_idx: T,
    whir_round: T,
    query_idx: T,
    coset_idx: T,
    is_first_row_for_query: T,
    zi_root: T,
    zi: T,
    omega: T,
    opened_value: [T; D_EF],
    yi: [T; D_EF],
}

pub struct NonInitialOpenedValuesAir {
    pub verify_query_bus: VerifyQueryBus,
    pub folding_bus: WhirFoldingBus,
    pub k: usize,
}

impl BaseAirWithPublicValues<F> for NonInitialOpenedValuesAir {}
impl PartitionedBaseAir<F> for NonInitialOpenedValuesAir {}

impl<F> BaseAir<F> for NonInitialOpenedValuesAir {
    fn width(&self) -> usize {
        NonInitialOpenedValuesCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for NonInitialOpenedValuesAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &NonInitialOpenedValuesCols<AB::Var> = (*local).borrow();

        self.verify_query_bus.receive(
            builder,
            local.proof_idx,
            VerifyQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                zi_root: local.zi_root,
                zi: local.zi,
                yi: local.yi,
            },
            local.is_first_row_for_query,
        );
        self.folding_bus.send(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into(),
                height: AB::Expr::ZERO,
                coset_shift: local.zi_root.into(),
                coset_size: AB::Expr::from_canonical_usize(1 << self.k),
                coset_idx: local.coset_idx.into(),
                twiddle: local.omega.into(),
                value: local.opened_value.map(Into::into),
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            local.is_valid,
        )
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let num_rounds = preflight.whir.pow_samples.len();
    let num_queries = vk.inner.params.num_whir_queries;
    let k_whir = vk.inner.params.k_whir;
    let omega_k = F::two_adic_generator(k_whir);

    let num_valid_rows: usize = ((num_rounds - 1) * num_queries) << k_whir;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = NonInitialOpenedValuesCols::<F>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut NonInitialOpenedValuesCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;

        let coset_idx = i % (1 << k_whir);
        let query_idx = (i >> k_whir) % num_queries;
        let whir_round = ((i >> k_whir) / num_queries) + 1;

        cols.whir_round = F::from_canonical_usize(whir_round);
        cols.query_idx = F::from_canonical_usize(query_idx);
        cols.coset_idx = F::from_canonical_usize(coset_idx);
        cols.omega = omega_k.exp_u64(coset_idx as u64);
        cols.is_first_row_for_query = F::from_bool(coset_idx == 0);
        cols.zi_root = preflight.whir.zj_roots[whir_round][query_idx];
        cols.opened_value = proof.whir_proof.codeword_opened_rows[whir_round - 1][query_idx]
            [coset_idx]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.zi = preflight.whir.zjs[whir_round][query_idx];
        cols.yi = preflight.whir.yjs[whir_round][query_idx]
            .as_base_slice()
            .try_into()
            .unwrap();
    }

    RowMajorMatrix::new(trace, width)
}
