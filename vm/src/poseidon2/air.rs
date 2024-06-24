use super::columns::Poseidon2Cols;
use super::Poseidon2Air;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_field::AbstractField;
use p3_field::Field;
use p3_field::PrimeField;
use p3_matrix::Matrix;
use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
use p3_symmetric::Permutation;
use std::borrow::Borrow;

impl<const WIDTH: usize, F: PrimeField> BaseAir<F> for Poseidon2Air<WIDTH, F> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB, const WIDTH: usize, F: PrimeField> Air<AB> for Poseidon2Air<WIDTH, F>
where
    AB: AirBuilder,
    AB::Var: AbstractField<F = F>,
    F: PrimeField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let poseidon2_cols = Poseidon2Cols::from_slice(local, self);
        let Poseidon2Cols { io, aux } = poseidon2_cols;

        let external_layer = Poseidon2ExternalMatrixGeneral {};
        let internal_layer = DiffusionMatrixBabyBear {};

        let mut start_state: &mut [_; WIDTH] = io.input.clone().as_mut_slice().try_into().unwrap();
        external_layer.permute_mut(start_state);
        Self::ext_layer(start_state, &self.external_constants[0], &external_layer);
    }
}
