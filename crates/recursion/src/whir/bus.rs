use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{define_typed_per_proof_lookup_bus, define_typed_per_proof_permutation_bus};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirAlphaMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(WhirAlphaBus, WhirAlphaMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirGammaMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirGammaBus, WhirGammaMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirQueryBusMessage<T> {
    pub whir_round: T,
    pub query_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirQueryBus, WhirQueryBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct VerifyQueriesBusMessage<T> {
    pub tidx: T,
    pub whir_round: T,
    pub num_queries: T,
    pub gamma: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(VerifyQueriesBus, VerifyQueriesBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct VerifyQueryBusMessage<T> {
    pub whir_round: T,
    pub query_idx: T,
    pub merkle_idx_bit_src: T,
    pub zi_root: T,
    pub zi: T,
    pub yi: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(VerifyQueryBus, VerifyQueryBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirSumcheckBusMessage<T> {
    pub tidx: T,
    pub sumcheck_idx: T,
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirSumcheckBus, WhirSumcheckBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirEqAlphaUMessage<T> {
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirEqAlphaUBus, WhirEqAlphaUMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirFoldingBusMessage<T> {
    pub whir_round: T,
    pub query_idx: T,
    pub height: T,
    pub coset_shift: T,
    pub coset_idx: T,
    pub coset_size: T,
    pub twiddle: T,
    pub value: [T; D_EF],
    pub z_final: T,
    pub y_final: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirFoldingBus, WhirFoldingBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FinalPolyMleEvalMessage<T> {
    pub tidx: T,
    pub num_whir_rounds: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(FinalPolyMleEvalBus, FinalPolyMleEvalMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FinalPolyFoldingMessage<T> {
    pub proof_idx: T,
    pub depth: T,
    pub node_idx: T,
    pub num_nodes_in_layer: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(FinalPolyFoldingBus, FinalPolyFoldingMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirFinalPolyBusMessage<T> {
    pub idx: T,
    pub coeff: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(WhirFinalPolyBus, WhirFinalPolyBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FinalPolyQueryEvalMessage<T> {
    pub last_whir_round: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(FinalPolyQueryEvalBus, FinalPolyQueryEvalMessage);
