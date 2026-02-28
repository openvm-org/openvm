pub mod bus;
pub mod def_pvs;
pub mod input;
pub mod verifier;

use std::{borrow::Borrow, sync::Arc};

use itertools::Itertools;
use openvm_circuit::arch::POSEIDON2_WIDTH;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
    AirRef,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use recursion_circuit::system::AggregationSubCircuit;

use crate::{
    circuit::{
        deferral::{DeferralAggregationPvs, DeferralCircuitPvs},
        root::digests_to_poseidon2_input,
    },
    prover::Circuit,
    SC,
};

const DEF_CIRCUIT_PVS_AIR_ID: usize = 0;
const DEF_AGG_PVS_AIR_ID: usize = 1;

#[derive(derive_new::new, Clone)]
pub struct DeferralNonRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for DeferralNonRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();
        let input_or_merkle_commit_bus = bus::InputOrMerkleCommitBus::new(next_bus_idx);
        let pv_air_consistency_bus = bus::PvAirConsistencyBus::new(next_bus_idx + 1);

        let verifier_pvs_air = verifier::NonRootPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            pv_air_consistency_bus,
        };

        let def_pvs_air = def_pvs::DeferralPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            input_or_merkle_commit_bus,
            pv_air_consistency_bus,
        };

        let input_commit_air = input::InputCommitAir {
            public_values_bus: bus_inventory.public_values_bus,
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            input_or_merkle_commit_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(def_pvs_air) as AirRef<SC>])
            .chain([Arc::new(input_commit_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
            .collect_vec()
    }
}

pub struct DeferralNonRootPreCtx<PB: ProverBackend> {
    pub verifier_pvs_ctx: AirProvingContext<PB>,
    pub def_pvs_ctx: AirProvingContext<PB>,
    pub input_ctx: AirProvingContext<PB>,
    pub poseidon2_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
}

fn fold_leaf_input_commit(
    proof: &Proof<BabyBearPoseidon2Config>,
    init: [F; DIGEST_SIZE],
) -> [F; DIGEST_SIZE] {
    proof
        .trace_vdata
        .iter()
        .flatten()
        .flat_map(|vdata| vdata.cached_commitments.iter().copied())
        .fold(init, |acc, cached_commit| {
            poseidon2_compress_with_capacity(acc, cached_commit).0
        })
}

fn child_merkle_commit(
    proof: &Proof<BabyBearPoseidon2Config>,
    child_is_def: bool,
) -> [F; DIGEST_SIZE] {
    if child_is_def {
        let child_pvs: &DeferralAggregationPvs<F> =
            proof.public_values[DEF_AGG_PVS_AIR_ID].as_slice().borrow();
        child_pvs.merkle_commit
    } else {
        let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
            .as_slice()
            .borrow();
        let folded_input_commit = fold_leaf_input_commit(proof, child_pvs.input_commit);
        poseidon2_compress_with_capacity(folded_input_commit, child_pvs.output_commit).0
    }
}

fn generate_poseidon2_inputs(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_def: bool,
    child_merkle_depth: Option<usize>,
) -> Vec<[F; POSEIDON2_WIDTH]> {
    let mut poseidon2_inputs: Vec<[F; POSEIDON2_WIDTH]> = Vec::new();

    for proof in proofs {
        if child_is_def {
            continue;
        }
        let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
            .as_slice()
            .borrow();

        // InputCommitAir: hash input_commit with each cached trace commitment.
        let mut current_commit = child_pvs.input_commit;
        for cached_commit in proof
            .trace_vdata
            .iter()
            .flatten()
            .flat_map(|vdata| vdata.cached_commitments.iter().copied())
        {
            poseidon2_inputs.push(digests_to_poseidon2_input(current_commit, cached_commit));
            current_commit = poseidon2_compress_with_capacity(current_commit, cached_commit).0;
        }

        // DeferralPvsAir (leaf): hash folded input_commit and output_commit into merkle_commit.
        poseidon2_inputs.push(digests_to_poseidon2_input(
            current_commit,
            child_pvs.output_commit,
        ));
    }

    // DeferralPvsAir: hash child merkle commits when this is not a wrapper.
    if let Some(depth) = child_merkle_depth {
        let left_merkle = child_merkle_commit(&proofs[0], child_is_def);
        let right_merkle = if proofs.len() == 2 {
            child_merkle_commit(&proofs[1], child_is_def)
        } else {
            def_pvs::zero_merkle_commit(depth)
        };
        poseidon2_inputs.push(digests_to_poseidon2_input(left_merkle, right_merkle));
    }

    poseidon2_inputs
}

// Trait used to remain generic in PB
pub trait DeferralNonRootTraceGen<PB: ProverBackend> {
    fn new() -> Self;
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: PB::Commitment,
        child_merkle_depth: Option<usize>,
    ) -> DeferralNonRootPreCtx<PB>;
}

pub struct DeferralNonRootTraceGenImpl;

impl DeferralNonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for DeferralNonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: [F; DIGEST_SIZE],
        child_merkle_depth: Option<usize>,
    ) -> DeferralNonRootPreCtx<CpuBackend<BabyBearPoseidon2Config>> {
        DeferralNonRootPreCtx {
            verifier_pvs_ctx: verifier::generate_proving_ctx(
                proofs,
                child_is_def,
                child_dag_commit,
            ),
            def_pvs_ctx: def_pvs::generate_proving_ctx(proofs, child_is_def, child_merkle_depth),
            input_ctx: input::generate_proving_ctx(proofs, child_is_def),
            poseidon2_inputs: generate_poseidon2_inputs(proofs, child_is_def, child_merkle_depth),
        }
    }
}

#[cfg(feature = "cuda")]
impl DeferralNonRootTraceGen<GpuBackend> for DeferralNonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: [F; DIGEST_SIZE],
        child_merkle_depth: Option<usize>,
    ) -> DeferralNonRootPreCtx<GpuBackend> {
        let DeferralNonRootPreCtx {
            verifier_pvs_ctx,
            def_pvs_ctx,
            input_ctx,
            poseidon2_inputs,
        } = <Self as DeferralNonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>>>::pre_verifier_subcircuit_tracegen(
            self,
            proofs,
            child_is_def,
            child_dag_commit,
            child_merkle_depth,
        );
        DeferralNonRootPreCtx {
            verifier_pvs_ctx: transport_air_proving_ctx_to_device(verifier_pvs_ctx),
            def_pvs_ctx: transport_air_proving_ctx_to_device(def_pvs_ctx),
            input_ctx: transport_air_proving_ctx_to_device(input_ctx),
            poseidon2_inputs,
        }
    }
}
