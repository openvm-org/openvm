// Initial version copied from risc0 under Apache License 2.0

use std::path::PathBuf;

use cargo_metadata::Package;
use serde::{Deserialize, Serialize};

/// Options for configuring a docker build environment.
#[derive(Clone, Serialize, Deserialize)]
pub struct DockerOptions {
    /// Specify the root directory for docker builds.
    ///
    /// The current working directory is used if `None` is specified.
    pub root_dir: Option<PathBuf>,
}

/// Options defining how to embed a guest package in
/// [`crate::embed_methods_with_options`].
#[derive(Default, Clone)]
pub struct GuestOptions {
    /// Features for cargo to build the guest with.
    pub features: Vec<String>,
    /// Custom options to pass as args to `cargo build`.
    pub options: Vec<String>,
    /// Use a docker environment for building.
    pub use_docker: Option<DockerOptions>,
}

impl GuestOptions {
    /// Add custom options to pass to `cargo build`.
    pub fn with_options<S: AsRef<str>>(mut self, options: impl IntoIterator<Item = S>) -> Self {
        self.options
            .extend(options.into_iter().map(|s| s.as_ref().to_string()));
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
        let Some(obj) = value.metadata.get("risc0") else {
            return Default::default();
        };
        serde_json::from_value(obj.clone()).unwrap()
    }
}

/// Extended options defining how to embed a guest package in
/// [`crate::embed_methods_with_options`].
#[derive(Default, Clone)]
pub struct GuestBuildOptions {
    /// Features for cargo to build the guest with.
    pub(crate) features: Vec<String>,
    /// Custom options to pass as args to `cargo build`.
    pub(crate) options: Vec<String>,
    // Use a docker environment for building.
    // pub(crate) use_docker: Option<DockerOptions>,
    /// Configuration flags to build the guest with.
    pub(crate) rustc_flags: Vec<String>,
}

impl From<GuestOptions> for GuestBuildOptions {
    fn from(value: GuestOptions) -> Self {
        Self {
            features: value.features,
            options: value.options,
            // use_docker: value.use_docker,
            ..Default::default()
        }
    }
}

impl GuestBuildOptions {
    #[allow(dead_code)]
    pub(crate) fn with_metadata(mut self, metadata: GuestMetadata) -> Self {
        self.rustc_flags = metadata.rustc_flags.unwrap_or_default();
        self
    }
}
