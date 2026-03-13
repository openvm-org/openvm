use std::sync::Arc;

use itertools::Itertools;
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_recursion_circuit::{prelude::F, system::AggregationSubCircuit};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{AirRef, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    bn254::{CommitBytes, DagCommitBytes},
    circuit::{
        root::bus::{DeferralAccPathBus, DeferralMerkleRootsBus},
        subair::HashSliceSubAir,
        Circuit,
    },
};

pub mod bus;
pub mod commit;
pub mod def_paths;
pub mod memory;
pub mod verifier;

mod trace;
pub use trace::*;

pub const NUM_DIGESTS_IN_VK_COMMIT: usize = 6;

#[derive(derive_new::new, Clone)]
pub struct RootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub(crate) internal_recursive_dag_commit: DagCommitBytes,
    pub(crate) def_hook_commit: Option<CommitBytes>,
    pub(crate) memory_dimensions: MemoryDimensions,
    pub(crate) num_user_pvs: usize,
}

impl<SC: StarkProtocolConfig<F = F>, S: AggregationSubCircuit> Circuit<SC> for RootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = bus::UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = bus::UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = bus::MemoryMerkleCommitBus::new(next_bus_idx + 2);
        let def_acc_paths_bus = DeferralAccPathBus::new(next_bus_idx + 3);
        let memory_merkle_roots_bus = DeferralMerkleRootsBus::new(next_bus_idx + 4);

        let verifier_pvs_air = verifier::RootVerifierPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            pre_hash_bus: bus_inventory.pre_hash_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            memory_merkle_commit_bus,
            def_acc_paths_bus,
            def_merkle_roots_bus: memory_merkle_roots_bus,
            hash_slice_subair: HashSliceSubAir {
                compress_bus: bus_inventory.poseidon2_compress_bus,
                permute_bus: bus_inventory.poseidon2_permute_bus,
            },
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
            expected_def_hook_commit: self.def_hook_commit,
        };
        let user_pvs_commit_air = commit::UserPvsCommitAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            self.num_user_pvs,
        );
        let user_pvs_memory_air = memory::UserPvsInMemoryAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            self.memory_dimensions,
            self.num_user_pvs,
        );
        let acc_paths_air = self.def_hook_commit.map(|_| {
            Arc::new(def_paths::DeferralAccMerklePathsAir::new(
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
            .chain(acc_paths_air)
            .collect_vec()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct RootVerifierPvs<F> {
    /// Hashed combination of the app-level ProgramAir cached trace, the Merkle root commit of
    /// the starting app memory state (i.e. initial_root), and the initial app program counter
    /// (i.e. initial_pc).
    pub app_exe_commit: [F; DIGEST_SIZE],
    /// Commit to the app-level verifying key, computed by hashing the cached_commit and
    /// vk_pre_hash components of the app, leaf, and internal-for-leaf DAG commits.
    pub app_vk_commit: [F; DIGEST_SIZE],
}
