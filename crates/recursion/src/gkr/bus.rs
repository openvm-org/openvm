use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::define_typed_per_proof_permutation_bus;

/// Message sent from GkrInputAir to GkrXiSamplerAir
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrXiSamplerInputMessage<T> {
    pub idx_start: T,
    pub num_challenges: T,
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(GkrXiSamplerInputBus, GkrXiSamplerInputMessage);

/// Message sent from GkrXiSamplerAir to GkrLayerAir
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrXiSamplerOutputMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(GkrXiSamplerOutputBus, GkrXiSamplerOutputMessage);

/// Message sent from GkrInputAir to GkrLayerAir
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrLayerInputMessage<T> {
    pub num_layers: T,
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(GkrLayerInputBus, GkrLayerInputMessage);

/// Message sent from GkrInputAir to GkrLayerAir
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrLayerOutputMessage<T> {
    pub tidx: T,
    pub input_layer_claim: [[T; D_EF]; 2],
}

define_typed_per_proof_permutation_bus!(GkrLayerOutputBus, GkrLayerOutputMessage);

/// Message sent from GkrLayerAir to GkrLayerSumcheckAir
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrSumcheckInputMessage<T> {
    /// GKR layer number
    pub layer: T,
    /// Transcript index for sumcheck
    pub tidx: T,
    /// Combined claim to verify
    pub claim: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(GkrSumcheckInputBus, GkrSumcheckInputMessage);

/// Message sent from GkrLayerSumcheckAir to GkrLayerAir
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrSumcheckOutputMessage<T> {
    /// GKR layer number
    pub layer: T,
    /// Transcript index after sumcheck
    pub tidx: T,
    /// New claim after sumcheck
    pub new_claim: [T; D_EF],
    /// Equality polynomial evaluation at r'
    pub eq_at_r_prime: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(GkrSumcheckOutputBus, GkrSumcheckOutputMessage);

/// Message for passing challenges between consecutive sumcheck sub-rounds
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrSumcheckChallengeMessage<T> {
    /// GKR layer number
    pub layer: T,
    /// Sumcheck round number
    pub sumcheck_round: T,
    /// The challenge value
    pub challenge: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(GkrSumcheckChallengeBus, GkrSumcheckChallengeMessage);
