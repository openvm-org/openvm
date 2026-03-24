use std::{
    collections::HashMap,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use openvm_static_verifier::{Halo2Params, Halo2ParamsReader};

/// Caching reader for Halo2 KZG parameters.
///
/// Reads SRS files from a directory and caches them in memory for reuse.
pub struct CacheHalo2ParamsReader {
    params_dir: PathBuf,
    cached: Mutex<HashMap<usize, Arc<Halo2Params>>>,
}

impl Halo2ParamsReader for CacheHalo2ParamsReader {
    fn read_params(&self, k: usize) -> Arc<Halo2Params> {
        self.read_params(k)
    }
}

impl CacheHalo2ParamsReader {
    pub fn new(params_dir: impl AsRef<Path>) -> Self {
        Self {
            params_dir: params_dir.as_ref().to_path_buf(),
            cached: Mutex::new(HashMap::new()),
        }
    }

    /// Create a reader using the default params directory: `~/.openvm/params/`.
    pub fn new_with_default_params_dir() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let params_dir = PathBuf::from(home).join(".openvm").join("params");
        Self::new(params_dir)
    }

    /// Read the KZG params for a given `k` value, caching the result.
    pub fn read_params(&self, k: usize) -> Arc<Halo2Params> {
        let mut cache = self.cached.lock().unwrap();
        if let Some(params) = cache.get(&k) {
            return params.clone();
        }
        let path = self.params_dir.join(format!("kzg_bn254_{k}.srs"));
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("Failed to open params file {}: {e}", path.display()));
        let mut reader = BufReader::new(file);

        // read_custom with RawBytes format
        let params =
            Halo2Params::read_custom(&mut reader, halo2_base::halo2_proofs::SerdeFormat::RawBytes)
                .unwrap_or_else(|e| panic!("Failed to read params from {}: {e}", path.display()));

        let params = Arc::new(params);
        cache.insert(k, params.clone());
        params
    }
}
