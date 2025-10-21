use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    system::Preflight,
    whir::bus::{ExpBitsLenBus, ExpBitsLenMessage},
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct ExpBitsLenCols<T> {
    is_valid: T,
    base: T,
    bit_src: T,
    num_bits: T,
    result: T,
}

pub struct ExpBitsLenAir {
    pub exp_bits_len_bus: ExpBitsLenBus,
}

impl BaseAirWithPublicValues<F> for ExpBitsLenAir {}
impl PartitionedBaseAir<F> for ExpBitsLenAir {}

impl<F> BaseAir<F> for ExpBitsLenAir {
    fn width(&self) -> usize {
        ExpBitsLenCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpBitsLenAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &ExpBitsLenCols<AB::Var> = (*local).borrow();

        self.exp_bits_len_bus.add_key_with_lookups(
            builder,
            ExpBitsLenMessage {
                base: local.base,
                bit_src: local.bit_src,
                num_bits: local.num_bits,
                result: local.result,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    _proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let num_whir_rounds = preflight.whir.pow_samples.len();
    let num_queries = vk.inner.params.num_whir_queries;

    let num_valid_rows: usize = num_whir_rounds * (num_queries + 1);
    let num_rows = num_valid_rows.next_power_of_two();
    let width = ExpBitsLenCols::<F>::width();

    let m = vk.inner.params.n_stack + vk.inner.params.l_skip + vk.inner.params.log_blowup;

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();

        cols.is_valid = F::ONE;

        if i < preflight.whir.pow_samples.len() {
            cols.base = F::GENERATOR;
            cols.bit_src = preflight.whir.pow_samples[i];
            cols.num_bits = F::from_canonical_usize(vk.inner.params.logup_pow_bits);
            cols.result = F::ONE;
        } else {
            let j = i - preflight.whir.pow_samples.len();
            let whir_round = j / num_queries;
            cols.base = F::two_adic_generator(m - whir_round - vk.inner.params.k_whir);
            cols.bit_src = preflight.whir.queries[j];
            cols.num_bits = F::from_canonical_usize(m - whir_round - vk.inner.params.k_whir);
            cols.result = preflight.whir.zj_roots[whir_round][j % num_queries];
        }
    }

    RowMajorMatrix::new(trace, width)
}
