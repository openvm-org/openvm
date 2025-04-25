// Initial version copied from risc0 under Apache License 2.0

use std::path::{Path, PathBuf};

use cargo_metadata::Package;
use serde::{Deserialize, Serialize};

/// Options defining how to embed a guest package.
#[derive(Default, Clone)]
pub struct GuestOptions {
    /// Features for cargo to build the guest with.
    pub features: Vec<String>,
    /// Configuration flags to build the guest with.
    pub rustc_flags: Vec<String>,
    /// Cargo profile
    pub profile: Option<String>,
    /// Target directory
    pub target_dir: Option<PathBuf>,
    /// Custom options to pass as args to `cargo build`.
    pub options: Vec<String>,
}

impl GuestOptions {
    /// Add custom options to pass to `cargo build`.
    pub fn with_options<S: AsRef<str>>(mut self, options: impl IntoIterator<Item = S>) -> Self {
        self.options
            .extend(options.into_iter().map(|s| s.as_ref().to_string()));
        self
    }

    /// Add package features to pass to `cargo build`.
    pub fn with_features<S: AsRef<str>>(mut self, features: impl IntoIterator<Item = S>) -> Self {
        self.features
            .extend(features.into_iter().map(|s| s.as_ref().to_string()));
        self
    }

    /// Add rustc flags for building the guest.
    pub fn with_rustc_flags<S: AsRef<str>>(mut self, flags: impl IntoIterator<Item = S>) -> Self {
        self.rustc_flags
            .extend(flags.into_iter().map(|s| s.as_ref().to_string()));
        self
    }

    /// Set the cargo profile.
    pub fn with_profile(mut self, profile: String) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the target directory.
    pub fn with_target_dir<P: AsRef<Path>>(mut self, target_dir: P) -> Self {
        self.target_dir = Some(target_dir.as_ref().to_path_buf());
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_metadata(mut self, metadata: GuestMetadata) -> Self {
        self.rustc_flags = metadata.rustc_flags.unwrap_or_default();
        self
    }
}

/// Metadata defining options to build a guest
#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct GuestMetadata {
    /// Configuration flags to build the guest with.
    #[serde(rename = "rustc-flags")]
    pub(crate) rustc_flags: Option<Vec<String>>,
}

impl From<&Package> for GuestMetadata {
    fn from(value: &Package) -> Self {
        let Some(obj) = value.metadata.get("openvm") else {
            return Default::default();
        };
        serde_json::from_value(obj.clone()).unwrap()
    }
}
