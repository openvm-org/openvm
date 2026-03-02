use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::AirRef;
use recursion_circuit::system::AggregationSubCircuit;

use crate::{circuit::Circuit, SC};

pub mod bus;
pub mod def_pvs;
pub mod input;
pub mod verifier;

mod trace;
pub use trace::*;

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
