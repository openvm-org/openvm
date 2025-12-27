use std::{any::TypeId, marker::PhantomData};

use derivative::Derivative;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::{BabyBear, BabyBearInternalLayerParameters};
use p3_monty_31::InternalLayerBaseParameters;
use p3_poseidon2::{
    add_rc_and_sbox_generic, mds_light_permutation, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, GenericPoseidon2LinearLayers, InternalLayer,
    InternalLayerConstructor, MDSMat4,
};

use super::{babybear_internal_linear_layer, BABY_BEAR_POSEIDON2_SBOX_DEGREE};

const WIDTH: usize = crate::POSEIDON2_WIDTH;

pub trait Poseidon2MatrixConfig: Clone + Sync {
    fn int_diag_m1_matrix<F: FieldAlgebra>() -> [F; WIDTH];
}

/// This type needs to implement GenericPoseidon2LinearLayers generic in F so that our
/// Poseidon2SubAir can also be generic in F, but in reality each implementation of this struct's
/// functions should be field specific. To circumvent this, Poseidon2LinearLayers is generic in F
/// but **currently requires** that F is BabyBear.
#[derive(Debug, Clone)]
pub struct BabyBearPoseidon2LinearLayers;

// This is the same as the implementation for
// GenericPoseidon2LinearLayersMonty31<BabyBearParameters, BabyBearInternalLayerParameters> except
// that we drop the clause that FA needs be multipliable by BabyBear.
// TODO[jpw/stephen]: This is clearly not the best way to do this, but it would
// require some reworking in plonky3 to get around the generics.
impl<FA: FieldAlgebra> GenericPoseidon2LinearLayers<FA, WIDTH> for BabyBearPoseidon2LinearLayers {
    fn internal_linear_layer(state: &mut [FA; WIDTH]) {
        let diag_m1_matrix = &<BabyBearInternalLayerParameters as InternalLayerBaseParameters<
            _,
            16,
        >>::INTERNAL_DIAG_MONTY;
        assert_eq!(
            TypeId::of::<FA::F>(),
            TypeId::of::<BabyBear>(),
            "BabyBear is the only supported field type"
        );
        // SAFETY: TypeId check above ensures FA::F is BabyBear, so transmute is valid
        let diag_m1_matrix =
            unsafe { std::mem::transmute::<&[BabyBear; WIDTH], &[FA::F; WIDTH]>(diag_m1_matrix) };
        babybear_internal_linear_layer(state, diag_m1_matrix);
    }

    fn external_linear_layer(state: &mut [FA; WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
    }
}

// Below are generic implementations of the Poseidon2 Internal and External Layers
// generic in the field. These are currently used for the runtime poseidon2
// execution even though they are less optimized than the Monty31 specific
// implementations in Plonky3. We could use those more optimized implementations,
// but it would require many unsafe transmutes.

#[derive(Debug, Derivative)]
#[derivative(Clone)]
pub struct Poseidon2InternalLayer<F: FieldAlgebra, LinearLayers> {
    pub internal_constants: Vec<F>,
    _marker: PhantomData<LinearLayers>,
}

impl<AF: FieldAlgebra, LinearLayers> InternalLayerConstructor<AF>
    for Poseidon2InternalLayer<AF::F, LinearLayers>
{
    fn new_from_constants(internal_constants: Vec<AF::F>) -> Self {
        Self {
            internal_constants,
            _marker: PhantomData,
        }
    }
}

impl<FA: FieldAlgebra, LinearLayers, const WIDTH: usize>
    InternalLayer<FA, WIDTH, BABY_BEAR_POSEIDON2_SBOX_DEGREE>
    for Poseidon2InternalLayer<FA::F, LinearLayers>
where
    LinearLayers: GenericPoseidon2LinearLayers<FA, WIDTH>,
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [FA; WIDTH]) {
        self.internal_constants.iter().for_each(|&rc| {
            add_rc_and_sbox_generic::<_, BABY_BEAR_POSEIDON2_SBOX_DEGREE>(&mut state[0], rc);
            LinearLayers::internal_linear_layer(state);
        })
    }
}

#[derive(Debug, Derivative)]
#[derivative(Clone)]
pub struct Poseidon2ExternalLayer<F: FieldAlgebra, LinearLayers, const WIDTH: usize> {
    pub constants: ExternalLayerConstants<F, WIDTH>,
    _marker: PhantomData<LinearLayers>,
}

impl<FA: FieldAlgebra, LinearLayers, const WIDTH: usize> ExternalLayerConstructor<FA, WIDTH>
    for Poseidon2ExternalLayer<FA::F, LinearLayers, WIDTH>
{
    fn new_from_constants(external_layer_constants: ExternalLayerConstants<FA::F, WIDTH>) -> Self {
        Self {
            constants: external_layer_constants,
            _marker: PhantomData,
        }
    }
}

impl<FA: FieldAlgebra, LinearLayers, const WIDTH: usize>
    ExternalLayer<FA, WIDTH, BABY_BEAR_POSEIDON2_SBOX_DEGREE>
    for Poseidon2ExternalLayer<FA::F, LinearLayers, WIDTH>
where
    LinearLayers: GenericPoseidon2LinearLayers<FA, WIDTH>,
{
    fn permute_state_initial(&self, state: &mut [FA; WIDTH]) {
        LinearLayers::external_linear_layer(state);
        external_permute_state::<FA, LinearLayers, WIDTH>(
            state,
            self.constants.get_initial_constants(),
        );
    }

    fn permute_state_terminal(&self, state: &mut [FA; WIDTH]) {
        external_permute_state::<FA, LinearLayers, WIDTH>(
            state,
            self.constants.get_terminal_constants(),
        );
    }
}

fn external_permute_state<FA: FieldAlgebra, LinearLayers, const WIDTH: usize>(
    state: &mut [FA; WIDTH],
    constants: &[[FA::F; WIDTH]],
) where
    LinearLayers: GenericPoseidon2LinearLayers<FA, WIDTH>,
{
    for elem in constants.iter() {
        state.iter_mut().zip(elem.iter()).for_each(|(s, &rc)| {
            add_rc_and_sbox_generic::<_, BABY_BEAR_POSEIDON2_SBOX_DEGREE>(s, rc)
        });
        LinearLayers::external_linear_layer(state);
    }
}
