use std::sync::Arc;

use openvm_continuations::CommitBytes;
use openvm_deferral_circuit::{DeferralExtension, DeferralFn};
use openvm_verify_stark_circuit::extension::verify_stark_deferral_fn;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SupportedDeferral {
    VerifyStark,
    Other(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeferralCircuitConfig {
    pub def_type: SupportedDeferral,
    pub commit: CommitBytes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeferralConfig {
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
