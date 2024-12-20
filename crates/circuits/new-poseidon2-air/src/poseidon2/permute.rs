use std::{array::from_fn, marker::PhantomData};

use openvm_stark_backend::p3_field::{AbstractField, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBearInternalLayerParameters;
use p3_monty_31::InternalLayerBaseParameters;
use p3_poseidon2::{
    add_rc_and_sbox_generic, ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor,
    GenericPoseidon2LinearLayers, InternalLayer, InternalLayerConstructor,
};
use zkhash::{ark_ff::PrimeField, poseidon2::poseidon2_instance_babybear::MAT_DIAG16_M_1};

use super::{
    babybear_external_linear_layer, babybear_internal_linear_layer, POSEIDON2_SBOX_DEGREE,
};

pub trait Poseidon2MatrixConfig: Clone + Sync {
    fn mat_4<F: AbstractField>() -> [[F; 4]; 4];
    fn int_diag_m1_matrix<F: AbstractField>() -> [F; 16];
    fn reduction_factor<F: AbstractField>() -> F;
}

#[derive(Debug, Clone)]
pub struct MdsMatrixConfig;
#[derive(Debug, Clone)]
pub struct HlMdsMatrixConfig;

/// MDSMat4 from Plonky3
/// [ 2 3 1 1 ]
/// [ 1 2 3 1 ]
/// [ 1 1 2 3 ]
/// [ 3 1 1 2 ].
/// <https://github.com/plonky3/Plonky3/blob/64370174ba932beee44307c3003e1d787c101bb6/poseidon2/src/external.rs#L43>
pub const MDS_MAT_4: [[u32; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
pub const MDS_REDUCTION_FACTOR: u32 = 1;

impl Poseidon2MatrixConfig for MdsMatrixConfig {
    fn mat_4<F: AbstractField>() -> [[F; 4]; 4] {
        from_fn(|i| from_fn(|j| F::from_canonical_u32(MDS_MAT_4[i][j])))
    }

    fn int_diag_m1_matrix<F: AbstractField>() -> [F; 16] {
        let monty = <BabyBearInternalLayerParameters as InternalLayerBaseParameters<_, 16>>::INTERNAL_DIAG_MONTY;
        monty.map(|babybear| F::from_canonical_u32(babybear.as_canonical_u32()))
    }

    fn reduction_factor<F: AbstractField>() -> F {
        F::from_canonical_u32(MDS_REDUCTION_FACTOR)
    }
}

// Multiply a 4-element vector x by
// [ 5 7 1 3 ]
// [ 4 6 1 1 ]
// [ 1 3 5 7 ]
// [ 1 1 4 6 ].
// This uses the formula from the start of Appendix B in the Poseidon2 paper, with multiplications unrolled into additions.
// It is also the matrix used by the Horizon Labs implementation.
pub const HL_MDS_MAT_4: [[u32; 4]; 4] = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]];
pub const HL_MDS_REDUCTION_FACTOR: u32 = 1;

impl Poseidon2MatrixConfig for HlMdsMatrixConfig {
    fn mat_4<F: AbstractField>() -> [[F; 4]; 4] {
        from_fn(|i| from_fn(|j| F::from_canonical_u32(HL_MDS_MAT_4[i][j])))
    }

    fn int_diag_m1_matrix<F: AbstractField>() -> [F; 16] {
        from_fn(|i| F::from_canonical_u32(MAT_DIAG16_M_1[i].into_bigint().0[0] as u32))
    }

    fn reduction_factor<F: AbstractField>() -> F {
        F::from_canonical_u32(HL_MDS_REDUCTION_FACTOR)
    }
}

/// This type needs to implement GenericPoseidon2LinearLayers generic in F so that our Poseidon2SubAir can also
/// be generic in F, but in reality each implementation of this struct's functions should be field specific. To
/// circumvent this, Poseidon2LinearLayers is generic in F but **currently** asserts that F is BabyBear.
#[derive(Debug, Clone)]
pub struct Poseidon2LinearLayers<Config: Poseidon2MatrixConfig>(PhantomData<Config>);

impl<F: AbstractField, const WIDTH: usize, Config: Poseidon2MatrixConfig + Sync>
    GenericPoseidon2LinearLayers<F, WIDTH> for Poseidon2LinearLayers<Config>
{
    fn internal_linear_layer(state: &mut [F; WIDTH]) {
        assert!(
            std::any::type_name::<F>().contains("BabyBear"),
            "BabyBear is the only supported field type for Poseidon2LinearLayers"
        );
        babybear_internal_linear_layer(
            state,
            Config::int_diag_m1_matrix(),
            Config::reduction_factor(),
        );
    }

    fn external_linear_layer(state: &mut [F; WIDTH]) {
        assert!(
            std::any::type_name::<F>().contains("BabyBear"),
            "BabyBear is the only supported field type for Poseidon2LinearLayers"
        );
        babybear_external_linear_layer(state, Config::mat_4());
    }
}

#[derive(Debug, Clone)]
pub struct Poseidon2InternalLayer<F: AbstractField, Config: Poseidon2MatrixConfig> {
    pub internal_constants: Vec<F>,
    _marker: PhantomData<Config>,
}

impl<AF: AbstractField, Config: Poseidon2MatrixConfig> InternalLayerConstructor<AF>
    for Poseidon2InternalLayer<AF::F, Config>
{
    fn new_from_constants(internal_constants: Vec<AF::F>) -> Self {
        Self {
            internal_constants,
            _marker: PhantomData,
        }
    }
}

impl<AF: AbstractField, Config: Poseidon2MatrixConfig, const WIDTH: usize>
    InternalLayer<AF, WIDTH, POSEIDON2_SBOX_DEGREE> for Poseidon2InternalLayer<AF::F, Config>
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [AF; WIDTH]) {
        self.internal_constants.iter().for_each(|&rc| {
            add_rc_and_sbox_generic::<_, POSEIDON2_SBOX_DEGREE>(&mut state[0], rc);
            Poseidon2LinearLayers::<Config>::internal_linear_layer(state);
        })
    }
}

#[derive(Debug, Clone)]
pub struct Poseidon2ExternalLayer<
    F: AbstractField,
    Config: Poseidon2MatrixConfig,
    const WIDTH: usize,
> {
    pub constants: ExternalLayerConstants<F, WIDTH>,
    _marker: PhantomData<Config>,
}

impl<AF: AbstractField, Config: Poseidon2MatrixConfig, const WIDTH: usize>
    ExternalLayerConstructor<AF, WIDTH> for Poseidon2ExternalLayer<AF::F, Config, WIDTH>
{
    fn new_from_constants(external_layer_constants: ExternalLayerConstants<AF::F, WIDTH>) -> Self {
        Self {
            constants: external_layer_constants,
            _marker: PhantomData,
        }
    }
}

impl<AF: AbstractField, Config: Poseidon2MatrixConfig, const WIDTH: usize>
    ExternalLayer<AF, WIDTH, POSEIDON2_SBOX_DEGREE>
    for Poseidon2ExternalLayer<AF::F, Config, WIDTH>
{
    fn permute_state_initial(&self, state: &mut [AF; WIDTH]) {
        external_permute_state::<AF, Config, WIDTH>(state, self.constants.get_initial_constants());
    }

    fn permute_state_terminal(&self, state: &mut [AF; WIDTH]) {
        Poseidon2LinearLayers::<Config>::external_linear_layer(state);
        external_permute_state::<AF, Config, WIDTH>(state, self.constants.get_terminal_constants());
    }
}

fn external_permute_state<AF: AbstractField, Config: Poseidon2MatrixConfig, const WIDTH: usize>(
    state: &mut [AF; WIDTH],
    constants: &[[AF::F; WIDTH]],
) {
    for elem in constants.iter() {
        state
            .iter_mut()
            .zip(elem.iter())
            .for_each(|(s, &rc)| add_rc_and_sbox_generic::<_, POSEIDON2_SBOX_DEGREE>(s, rc));
        Poseidon2LinearLayers::<Config>::external_linear_layer(state);
    }
}
