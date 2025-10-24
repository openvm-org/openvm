use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    system::Preflight,
    whir::{
        FoldRecord,
        bus::{WhirAlphaBus, WhirAlphaMessage, WhirFoldingBus, WhirFoldingBusMessage},
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct WhirFoldingCols<T> {
    is_valid: T,
    proof_idx: T,
    whir_round: T,
    query_idx: T,
    is_root: T,
    coset_shift: T,
    coset_idx: T,
    is_coset_idx_zero: T,
    /// Distance from the leaf layer in the folding tree.
    height: T,
    twiddle: T,
    coset_size: T,
    value: [T; 4],
    left_value: [T; 4],
    right_value: [T; 4],
    z_final: T,
    y_final: [T; 4],
    alpha: [T; 4],
}

pub struct WhirFoldingAir {
    pub alpha_bus: WhirAlphaBus,
    pub folding_bus: WhirFoldingBus,
    pub k: usize,
}

impl BaseAirWithPublicValues<F> for WhirFoldingAir {}
impl PartitionedBaseAir<F> for WhirFoldingAir {}

impl BaseAir<F> for WhirFoldingAir {
    fn width(&self) -> usize {
        WhirFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for WhirFoldingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &WhirFoldingCols<AB::Var> = (*local).borrow();

        self.alpha_bus.lookup_key(
            builder,
            local.proof_idx,
            WhirAlphaMessage {
                idx: local.whir_round * AB::Expr::from_canonical_usize(self.k) + local.height
                    - AB::Expr::ONE,
                challenge: local.alpha.map(Into::into),
            },
            local.is_valid,
        );
        self.folding_bus.receive(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into(),
                height: local.height - AB::Expr::ONE,
                coset_shift: local.coset_shift.into(),
                coset_size: AB::Expr::TWO * local.coset_size,
                coset_idx: local.coset_idx.into(),
                twiddle: local.twiddle.into(),
                value: local.left_value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.is_valid,
        );
        self.folding_bus.receive(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into(),
                height: local.height - AB::Expr::ONE,
                coset_shift: local.coset_shift.into(),
                coset_size: AB::Expr::TWO * local.coset_size,
                coset_idx: local.coset_idx + local.coset_size,
                twiddle: -local.twiddle.into(),
                value: local.right_value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.is_valid,
        );
        self.folding_bus.send(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into(),
                height: local.height.into(),
                coset_shift: local.coset_shift * local.coset_shift,
                coset_size: local.coset_size.into(),
                coset_idx: local.coset_idx.into(),
                twiddle: local.twiddle * local.twiddle,
                value: local.value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.is_valid - local.is_root,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    _proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let num_rounds = preflight.whir.pow_samples.len();
    let num_queries = vk.inner.params.num_whir_queries;
    let k_whir = vk.inner.params.k_whir;
    let internal_nodes = (1 << k_whir) - 1;

    let num_valid_rows: usize = num_rounds * num_queries * internal_nodes;
    assert_eq!(num_valid_rows, preflight.whir.fold_records.len());
    let num_rows = num_valid_rows.next_power_of_two();
    let width = WhirFoldingCols::<F>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let &FoldRecord {
            whir_round,
            query_idx,
            twiddle,
            coset_shift,
            coset_size,
            coset_idx,
            height,
            left_value,
            right_value,
            value,
        } = &preflight.whir.fold_records[i];

        let cols: &mut WhirFoldingCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_root = F::from_bool(coset_size == 1);
        cols.alpha = preflight.whir.alphas[whir_round * k_whir + height - 1]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.height = F::from_canonical_usize(height);
        cols.whir_round = F::from_canonical_usize(whir_round);
        cols.query_idx = F::from_canonical_usize(query_idx);
        cols.coset_idx = F::from_canonical_usize(coset_idx);
        cols.left_value = left_value.as_base_slice().try_into().unwrap();
        cols.right_value = right_value.as_base_slice().try_into().unwrap();
        cols.value = value.as_base_slice().try_into().unwrap();
        cols.twiddle = twiddle;
        cols.coset_shift = coset_shift;
        cols.coset_size = F::from_canonical_usize(coset_size);
        cols.z_final = preflight.whir.zjs[whir_round][query_idx];
        cols.y_final = preflight.whir.yjs[whir_round][query_idx]
            .as_base_slice()
            .try_into()
            .unwrap();
    }

    RowMajorMatrix::new(trace, width)
}
