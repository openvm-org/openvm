use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{AirRef, StarkProtocolConfig};
use recursion_circuit::{prelude::F, system::AggregationSubCircuit};

use crate::circuit::Circuit;

pub mod bus;
pub mod def_pvs;
pub mod input;
pub mod verifier;

mod trace;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct DeferralInnerCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<SC: StarkProtocolConfig<F = F>, S: AggregationSubCircuit> Circuit<SC>
    for DeferralInnerCircuit<S>
{
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();
        let input_or_merkle_commit_bus = bus::InputOrMerkleCommitBus::new(next_bus_idx);
        let def_pvs_consistency_bus = bus::DefPvsConsistencyBus::new(next_bus_idx + 1);

        let verifier_pvs_air = verifier::DeferralVerifierPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            def_pvs_consistency_bus,
        };

        let def_pvs_air = def_pvs::DeferralAggPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            input_or_merkle_commit_bus,
            def_pvs_consistency_bus,
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
