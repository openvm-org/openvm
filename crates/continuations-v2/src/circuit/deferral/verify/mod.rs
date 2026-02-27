use std::sync::Arc;

use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_stark_backend::AirRef;
use recursion_circuit::system::AggregationSubCircuit;

use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::verify::{
            bus::{OutputCommitBus, OutputValBus},
            output::DeferralOutputCommitAir,
            verifier::DeferredVerifyPvsAir,
        },
        root::bus::{MemoryMerkleCommitBus, UserPvsCommitBus, UserPvsCommitTreeBus},
        user_pvs::{commit::UserPvsCommitAir, memory::UserPvsInMemoryAir},
    },
    prover::Circuit,
    SC,
};

pub mod bus;
pub mod output;
pub mod verifier;

mod trace;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct DeferredVerifyCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    internal_recursive_dag_commit: CommitBytes,
    pub(crate) memory_dimensions: MemoryDimensions,
    pub(crate) num_user_pvs: usize,
}

impl<S: AggregationSubCircuit> Circuit for DeferredVerifyCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = MemoryMerkleCommitBus::new(next_bus_idx + 2);
        let output_val_bus = OutputValBus::new(next_bus_idx + 3);
        let output_commit_bus = OutputCommitBus::new(next_bus_idx + 4);

        let verifier_pvs_air = DeferredVerifyPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            memory_merkle_commit_bus,
            output_val_bus,
            output_commit_bus,
            final_state_bus: bus_inventory.final_state_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
        };
        let user_pvs_commit_air = UserPvsCommitAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            Some(output_val_bus),
            self.num_user_pvs,
        );
        let user_pvs_memory_air = UserPvsInMemoryAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            self.memory_dimensions,
            self.num_user_pvs,
        );
        let output_commit_air = DeferralOutputCommitAir {
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            range_bus: bus_inventory.range_checker_bus,
            output_val_bus,
            output_commit_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(user_pvs_commit_air) as AirRef<SC>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<SC>])
            .chain([Arc::new(output_commit_air) as AirRef<SC>])
            .collect()
    }
}
