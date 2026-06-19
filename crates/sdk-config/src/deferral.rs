use std::sync::Arc;

use openvm_continuations::CommitBytes;
use openvm_deferral_circuit::{DeferralExtension, DeferralFn};
use openvm_verify_stark_circuit::extension::verify_stark_deferral_fn;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SupportedDeferral {
    /// The built-in deferral circuit for recursively verifying OpenVM STARK proofs.
    VerifyStark,
    /// A custom deferral circuit.
    ///
    /// This variant can preserve the config shape through serialization, but the corresponding
    /// [`DeferralFn`] must be supplied manually after deserialization.
    Other(String),
}

/// A serializable deferral circuit entry in an [`SdkVmConfig`](crate::SdkVmConfig).
///
/// Entries are indexed by their position in [`DeferralConfig::circuits`]. Guest programs pass that
/// index to deferral guest APIs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeferralCircuitConfig {
    /// The supported deferral function this circuit computes.
    pub def_type: SupportedDeferral,
    /// The commitment to the deferral circuit proving key expected by the guest VM.
    pub commit: CommitBytes,
}

/// Serializable SDK configuration for the deferral extension.
///
/// The SDK converts this into the lower-level [`DeferralExtension`] when building execution,
/// transpiler, and prover components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeferralConfig {
    /// Ordered deferral circuits available to the VM.
    pub circuits: Vec<DeferralCircuitConfig>,
}

impl DeferralConfig {
    pub fn new(circuits: Vec<DeferralCircuitConfig>) -> Self {
        Self { circuits }
    }

    pub fn to_extension(&self) -> DeferralExtension {
        DeferralExtension {
            fns: self
                .circuits
                .iter()
                .map(|circuit| circuit.def_type.deferral_fn())
                .collect(),
            def_circuit_commits: self
                .circuits
                .iter()
                .map(|circuit| circuit.commit.to_field_le_bytes())
                .collect(),
        }
    }
}

impl From<DeferralConfig> for DeferralExtension {
    fn from(config: DeferralConfig) -> Self {
        config.to_extension()
    }
}

impl SupportedDeferral {
    fn deferral_fn(&self) -> Arc<DeferralFn> {
        match self {
            SupportedDeferral::VerifyStark => Arc::new(DeferralFn::new(verify_stark_deferral_fn)),
            SupportedDeferral::Other(name) => {
                let name = name.clone();
                Arc::new(DeferralFn::new(move |_| {
                    panic!("unsupported deferral function `{name}`")
                }))
            }
        }
    }
}
