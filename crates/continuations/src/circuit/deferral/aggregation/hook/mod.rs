use std::sync::Arc;

use itertools::Itertools;
use openvm_recursion_circuit::{prelude::F, system::AggregationSubCircuit};
use openvm_stark_backend::{AirRef, StarkProtocolConfig};

use crate::{
    bn254::DagCommitBytes,
    circuit::{
        subair::{HashSliceSubAir, MerkleRootBus, MerkleTreeInternalBus, MerkleTreeSubAir},
        Circuit,
    },
};

pub mod bus;
pub mod decommit;
pub mod onion;
pub mod verifier;

mod trace;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct DeferralHookCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub(crate) internal_recursive_dag_commit: DagCommitBytes,
}

impl<SC: StarkProtocolConfig<F = F>, S: AggregationSubCircuit> Circuit<SC>
    for DeferralHookCircuit<S>
{
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();
        let io_commit_bus = bus::IoCommitBus::new(next_bus_idx);
        let onion_res_bus = bus::OnionResultBus::new(next_bus_idx + 1);
        let def_vk_commit_bus = bus::DefVkCommitBus::new(next_bus_idx + 2);
        let merkle_root_bus = MerkleRootBus::new(next_bus_idx + 3);
        let merkle_tree_internal_bus = MerkleTreeInternalBus::new(next_bus_idx + 4);

        let verifier_pvs_air = verifier::DeferralHookPvsAir::new(
            bus_inventory.public_values_bus,
            bus_inventory.cached_commit_bus,
            bus_inventory.pre_hash_bus,
            bus_inventory.poseidon2_compress_bus,
            HashSliceSubAir {
                compress_bus: bus_inventory.poseidon2_compress_bus,
                permute_bus: bus_inventory.poseidon2_permute_bus,
            },
            def_vk_commit_bus,
            merkle_root_bus,
            onion_res_bus,
            self.internal_recursive_dag_commit,
        );

        let decommit_air = decommit::MerkleDecommitAir {
            subair: MerkleTreeSubAir::new(
                bus_inventory.poseidon2_compress_bus,
                merkle_root_bus,
                merkle_tree_internal_bus,
                0,
            ),
            io_commit_bus,
        };

        let onion_air = onion::OnionHashAir {
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            def_vk_commit_bus,
            io_commit_bus,
            onion_res_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(decommit_air) as AirRef<SC>])
            .chain([Arc::new(onion_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
            .collect_vec()
    }
}
