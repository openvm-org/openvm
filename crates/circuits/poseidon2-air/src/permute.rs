use std::marker::PhantomData;

use derivative::Derivative;
use openvm_stark_backend::p3_field::FieldAlgebra;
use p3_poseidon2::{
    add_rc_and_sbox_generic, ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor,
    GenericPoseidon2LinearLayers, InternalLayer, InternalLayerConstructor,
};

use crate::POSEIDON2_WIDTH;

pub trait Poseidon2MatrixConfig: Clone + Sync {
    fn int_diag_m1_matrix<F: FieldAlgebra>() -> [F; POSEIDON2_WIDTH];
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

impl<FA: FieldAlgebra, LinearLayers, const WIDTH: usize, const SBOX_DEGREE: u64>
    InternalLayer<FA, WIDTH, SBOX_DEGREE> for Poseidon2InternalLayer<FA::F, LinearLayers>
where
    LinearLayers: GenericPoseidon2LinearLayers<FA, WIDTH>,
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [FA; WIDTH]) {
        self.internal_constants.iter().for_each(|&rc| {
            add_rc_and_sbox_generic::<_, SBOX_DEGREE>(&mut state[0], rc);
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

impl<FA: FieldAlgebra, LinearLayers, const WIDTH: usize, const SBOX_DEGREE: u64>
    ExternalLayer<FA, WIDTH, SBOX_DEGREE> for Poseidon2ExternalLayer<FA::F, LinearLayers, WIDTH>
where
    LinearLayers: GenericPoseidon2LinearLayers<FA, WIDTH>,
{
    fn permute_state_initial(&self, state: &mut [FA; WIDTH]) {
        LinearLayers::external_linear_layer(state);
        external_permute_state::<FA, LinearLayers, WIDTH, SBOX_DEGREE>(
            state,
            self.constants.get_initial_constants(),
        );
    }

    fn permute_state_terminal(&self, state: &mut [FA; WIDTH]) {
        external_permute_state::<FA, LinearLayers, WIDTH, SBOX_DEGREE>(
            state,
            self.constants.get_terminal_constants(),
        );
    }
}

fn external_permute_state<
    FA: FieldAlgebra,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
>(
    state: &mut [FA; WIDTH],
    constants: &[[FA::F; WIDTH]],
) where
    LinearLayers: GenericPoseidon2LinearLayers<FA, WIDTH>,
{
    for elem in constants.iter() {
        state
            .iter_mut()
            .zip(elem.iter())
            .for_each(|(s, &rc)| add_rc_and_sbox_generic::<_, SBOX_DEGREE>(s, rc));
        LinearLayers::external_linear_layer(state);
    }
}
