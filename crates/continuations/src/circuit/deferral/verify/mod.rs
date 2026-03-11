use std::sync::Arc;

use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_recursion_circuit::{prelude::F, system::AggregationSubCircuit};
use openvm_stark_backend::{AirRef, StarkProtocolConfig};

use crate::{
    bn254::{CommitBytes, DagCommitBytes},
    circuit::{
        deferral::verify::{
            bus::{OutputCommitBus, OutputValBus},
            commit::UserPvsCommitValuesAir,
            output::DeferralOutputCommitAir,
            verifier::DeferredVerifyPvsAir,
        },
        root::{
            bus::{
                DeferralAccPathBus, DeferralMerkleRootsBus, MemoryMerkleCommitBus,
                UserPvsCommitBus, UserPvsCommitTreeBus,
            },
            def_paths::DeferralAccMerklePathsAir,
            memory::UserPvsInMemoryAir,
        },
        subair::HashSliceSubAir,
        Circuit,
    },
};

pub mod bus;
pub mod commit;
pub mod output;
pub mod verifier;

mod trace;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct DeferredVerifyCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    internal_recursive_dag_commit: DagCommitBytes,
    def_hook_commit: Option<CommitBytes>,
    pub(crate) memory_dimensions: MemoryDimensions,
    pub(crate) num_user_pvs: usize,
}

impl<SC: StarkProtocolConfig<F = F>, S: AggregationSubCircuit> Circuit<SC>
    for DeferredVerifyCircuit<S>
{
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = MemoryMerkleCommitBus::new(next_bus_idx + 2);
        let output_val_bus = OutputValBus::new(next_bus_idx + 3);
        let output_commit_bus = OutputCommitBus::new(next_bus_idx + 4);
        let def_acc_paths_bus = DeferralAccPathBus::new(next_bus_idx + 5);
        let memory_merkle_roots_bus = DeferralMerkleRootsBus::new(next_bus_idx + 6);

        let verifier_pvs_air = DeferredVerifyPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            pre_hash_bus: bus_inventory.pre_hash_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            hash_slice_subair: HashSliceSubAir {
                compress_bus: bus_inventory.poseidon2_compress_bus,
                permute_bus: bus_inventory.poseidon2_permute_bus,
            },
            memory_merkle_commit_bus,
            output_val_bus,
            output_commit_bus,
            final_state_bus: bus_inventory.final_state_bus,
            def_acc_paths_bus,
            def_merkle_roots_bus: memory_merkle_roots_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
            expected_def_hook_commit: self.def_hook_commit,
        };
        let user_pvs_commit_air = UserPvsCommitValuesAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            output_val_bus,
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

        let acc_paths_air = self.def_hook_commit.map(|_| {
            Arc::new(DeferralAccMerklePathsAir::new(
                bus_inventory.poseidon2_compress_bus,
                def_acc_paths_bus,
                memory_merkle_roots_bus,
                self.memory_dimensions,
            )) as AirRef<SC>
        });

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(user_pvs_commit_air) as AirRef<SC>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(output_commit_air) as AirRef<SC>])
            .chain(acc_paths_air)
            .collect()
    }
}
