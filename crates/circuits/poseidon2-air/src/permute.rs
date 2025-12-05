use std::marker::PhantomData;

use derivative::Derivative;
use openvm_stark_backend::p3_field::{Field, InjectiveMonomial, PrimeCharacteristicRing};
use openvm_stark_sdk::p3_baby_bear::{BabyBearInternalLayerParameters, BabyBearParameters};
use p3_monty_31::InternalLayerBaseParameters;
use p3_poseidon2::{
    add_rc_and_sbox_generic, mds_light_permutation, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, GenericPoseidon2LinearLayers, InternalLayer,
    InternalLayerConstructor, MDSMat4,
};

use super::BABY_BEAR_POSEIDON2_SBOX_DEGREE;

const WIDTH: usize = crate::POSEIDON2_WIDTH;

/// Linear layers for BabyBear Poseidon2 using the generic Plonky3 interfaces.
#[derive(Debug, Clone)]
pub struct BabyBearPoseidon2LinearLayers;

impl GenericPoseidon2LinearLayers<16> for BabyBearPoseidon2LinearLayers {
    fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; WIDTH]) {
        <BabyBearInternalLayerParameters as InternalLayerBaseParameters<
            BabyBearParameters,
            16,
        >>::generic_internal_linear_layer(state);
    }

    fn external_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; WIDTH]) {
        mds_light_permutation(state, &MDSMat4);
    }
}

// Generic implementations of the Poseidon2 internal and external layers over any
// PrimeCharacteristicRing.

#[derive(Debug, Derivative)]
#[derivative(Clone)]
pub struct Poseidon2InternalLayer<F: PrimeCharacteristicRing, LinearLayers> {
    pub internal_constants: Vec<F>,
    _marker: PhantomData<LinearLayers>,
}

impl<F: Field, LinearLayers> InternalLayerConstructor<F>
    for Poseidon2InternalLayer<F, LinearLayers>
{
    fn new_from_constants(internal_constants: Vec<F>) -> Self {
        Self {
            internal_constants,
            _marker: PhantomData,
        }
    }
}

impl<
        F: Field + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
        LinearLayers,
        const W: usize,
    > InternalLayer<F, W, BABY_BEAR_POSEIDON2_SBOX_DEGREE>
    for Poseidon2InternalLayer<F, LinearLayers>
where
    LinearLayers: GenericPoseidon2LinearLayers<W>,
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [F; W]) {
        self.internal_constants.iter().for_each(|&rc| {
            add_rc_and_sbox_generic::<_, _, BABY_BEAR_POSEIDON2_SBOX_DEGREE>(&mut state[0], rc);
            LinearLayers::internal_linear_layer(state);
        });
    }
}

#[derive(Debug, Derivative)]
#[derivative(Clone)]
pub struct Poseidon2ExternalLayer<F: PrimeCharacteristicRing, LinearLayers, const W: usize> {
    pub constants: ExternalLayerConstants<F, W>,
    _marker: PhantomData<LinearLayers>,
}

impl<F: Field, LinearLayers, const W: usize> ExternalLayerConstructor<F, W>
    for Poseidon2ExternalLayer<F, LinearLayers, W>
{
    fn new_from_constants(external_layer_constants: ExternalLayerConstants<F, W>) -> Self {
        Self {
            constants: external_layer_constants,
            _marker: PhantomData,
        }
    }
}

impl<
        F: Field + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
        LinearLayers,
        const W: usize,
    > ExternalLayer<F, W, BABY_BEAR_POSEIDON2_SBOX_DEGREE>
    for Poseidon2ExternalLayer<F, LinearLayers, W>
where
    LinearLayers: GenericPoseidon2LinearLayers<W>,
{
    fn permute_state_initial(&self, state: &mut [F; W]) {
        LinearLayers::external_linear_layer(state);
        external_permute_state::<F, LinearLayers, W>(state, self.constants.get_initial_constants());
    }

    fn permute_state_terminal(&self, state: &mut [F; W]) {
        external_permute_state::<F, LinearLayers, W>(
            state,
            self.constants.get_terminal_constants(),
        );
    }
}

fn external_permute_state<
    F: Field + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
    LinearLayers,
    const W: usize,
>(
    state: &mut [F; W],
    constants: &[[F; W]],
) where
    LinearLayers: GenericPoseidon2LinearLayers<W>,
{
    for elem in constants.iter() {
        state.iter_mut().zip(elem.iter()).for_each(|(s, &rc)| {
            add_rc_and_sbox_generic::<_, _, BABY_BEAR_POSEIDON2_SBOX_DEGREE>(s, rc)
        });
        LinearLayers::external_linear_layer(state);
    }
}
