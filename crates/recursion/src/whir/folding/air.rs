use core::borrow::Borrow;

use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    utils::{base_to_ext, ext_field_multiply, ext_field_subtract},
    whir::bus::{WhirAlphaBus, WhirAlphaMessage, WhirFoldingBus, WhirFoldingBusMessage},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct WhirFoldingCols<T> {
    pub is_valid: T,
    pub proof_idx: T,
    pub whir_round: T,
    pub query_idx: T,
    pub is_root: T,
    pub coset_shift: T,
    pub coset_idx: T,
    /// Distance from the leaf layer in the folding tree.
    pub height: T,
    pub twiddle: T,
    pub coset_size: T,
    pub z_final: T,
    pub value: [T; 4],
    pub left_value: [T; 4],
    pub right_value: [T; 4],
    pub y_final: [T; 4],
    pub alpha: [T; 4],
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

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for WhirFoldingAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0).expect("window should have two elements");
        let local: &WhirFoldingCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_root);
        builder.when(local.is_root).assert_one(local.is_valid);
        builder.when(local.is_root).assert_one(local.twiddle);
        builder.when(local.is_root).assert_zero(local.coset_idx);
        builder
            .when(local.is_root)
            .assert_eq(local.height, AB::F::from_usize(self.k));
        builder
            .when(local.is_root)
            .assert_eq(local.z_final, local.coset_shift * local.coset_shift);
        assert_array_eq(&mut builder.when(local.is_root), local.value, local.y_final);

        let x = local.twiddle * local.coset_shift;

        let term = ext_field_multiply::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(local.alpha, base_to_ext::<AB::Expr>(x.clone())),
            ext_field_subtract::<AB::Expr>(local.left_value, local.right_value),
        );
        // value = left_value + term / (2x)
        assert_array_eq(
            builder,
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(local.value, local.left_value),
                base_to_ext::<AB::Expr>(x * AB::Expr::TWO),
            ),
            term,
        );

        self.alpha_bus.lookup_key(
            builder,
            local.proof_idx,
            WhirAlphaMessage {
                idx: local.whir_round * AB::Expr::from_usize(self.k) + local.height - AB::Expr::ONE,
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
