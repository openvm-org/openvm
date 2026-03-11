use std::sync::Arc;

use itertools::Itertools;
use openvm_recursion_circuit::{prelude::F, system::AggregationSubCircuit};
use openvm_stark_backend::{AirRef, StarkProtocolConfig};
use openvm_verify_stark_host::pvs::{DeferralPvs, VmPvs, DEF_PVS_AIR_ID, VM_PVS_AIR_ID};

use crate::{
    bn254::CommitBytes,
    circuit::{
        inner::{
            bus::PvsAirConsistencyBus,
            def_pvs::DeferralPvsAir,
            unset::UnsetPvsAir,
            verifier::{VerifierDeferralConfig, VerifierPvsAir},
        },
        subair::HashSliceSubAir,
        Circuit,
    },
};

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
    };
}

pub mod bus;
pub mod def_pvs;
pub mod unset;
pub mod verifier;
pub mod vm_pvs;

mod trace;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct InnerCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub def_hook_commit: Option<CommitBytes>,
}

impl<SC: StarkProtocolConfig<F = F>, S: AggregationSubCircuit> Circuit<SC> for InnerCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let public_values_bus = bus_inventory.public_values_bus;
        let cached_commit_bus = bus_inventory.cached_commit_bus;
        let pre_hash_bus = bus_inventory.pre_hash_bus;
        let poseidon2_compress_bus = bus_inventory.poseidon2_compress_bus;
        let poseidon2_permute_bus = bus_inventory.poseidon2_permute_bus;
        let pvs_air_consistency_bus =
            PvsAirConsistencyBus::new(self.verifier_circuit.next_bus_idx());

        let deferral_enabled = self.def_hook_commit.is_some();

        let deferral_config = if deferral_enabled {
            VerifierDeferralConfig::Enabled {
                hash_slice_subair: HashSliceSubAir {
                    compress_bus: poseidon2_compress_bus,
                    permute_bus: poseidon2_permute_bus,
                },
            }
        } else {
            VerifierDeferralConfig::Disabled
        };

        let verifier_pvs_air = Arc::new(VerifierPvsAir {
            public_values_bus,
            cached_commit_bus,
            pre_hash_bus,
            pvs_air_consistency_bus,
            deferral_config,
        });

        let vm_pvs_air = Arc::new(vm_pvs::VmPvsAir {
            public_values_bus,
            cached_commit_bus,
            pvs_air_consistency_bus,
            deferral_enabled,
        });

        let (idx2_air, other_airs) = if deferral_enabled {
            let def_pvs_air = Arc::new(DeferralPvsAir {
                public_values_bus,
                cached_commit_bus,
                poseidon2_bus: poseidon2_compress_bus,
                pvs_air_consistency_bus,
                expected_def_hook_commit: self.def_hook_commit.unwrap(),
            }) as AirRef<SC>;
            let unset_vm_pvs_air = Arc::new(UnsetPvsAir {
                public_values_bus,
                pvs_air_consistency_bus,
                air_idx: VM_PVS_AIR_ID,
                num_pvs: VmPvs::<u8>::width(),
                def_flag: 1,
            }) as AirRef<SC>;
            let unset_def_pvs_air = Arc::new(UnsetPvsAir {
                public_values_bus,
                pvs_air_consistency_bus,
                air_idx: DEF_PVS_AIR_ID,
                num_pvs: DeferralPvs::<u8>::width(),
                def_flag: 0,
            }) as AirRef<SC>;
            (def_pvs_air, vec![unset_vm_pvs_air, unset_def_pvs_air])
        } else {
            let unset_dummy_air = Arc::new(UnsetPvsAir {
                public_values_bus,
                pvs_air_consistency_bus,
                air_idx: 0,
                num_pvs: 0,
                def_flag: 0,
            }) as AirRef<SC>;
            (unset_dummy_air, vec![])
        };

        [
            verifier_pvs_air as AirRef<SC>,
            vm_pvs_air as AirRef<SC>,
            idx2_air,
        ]
        .into_iter()
        .chain(self.verifier_circuit.airs())
        .chain(other_airs)
        .collect_vec()
    }
}
