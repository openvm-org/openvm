use std::{marker::PhantomData, path::PathBuf};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkProvingKey,
};
use afs_test_utils::engine::StarkEngine;
use datafusion::arrow::error::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use super::{read_bytes, write_bytes};
use crate::node::AxdbNode;

pub struct PkUtil<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    _phantom: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> PkUtil<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub fn save_proving_key(
        node: &AxdbNode<SC, E>,
        idx_len: usize,
        data_len: usize,
        pk: MultiStarkProvingKey<SC>,
    ) -> Result<()> {
        let path = Self::proving_key_path(node, idx_len, data_len);
        std::fs::create_dir_all(path.parent().unwrap())?;

        let serialized_pk = bincode::serialize(&pk).unwrap();
        write_bytes(&serialized_pk, path.as_path())?;
        Ok(())
    }

    /// Attempts to find proving key on disk for a given AxdbNode type with parameters.
    /// Returns None if proving key is not found.
    pub fn find_proving_key(
        node: &AxdbNode<SC, E>,
        idx_len: usize,
        data_len: usize,
    ) -> Option<MultiStarkProvingKey<SC>> {
        let path = Self::proving_key_path(node, idx_len, data_len);
        let encoded_pk = read_bytes(path.as_path())?;
        bincode::deserialize(&encoded_pk).unwrap_or(None)
    }

    fn proving_key_path(node: &AxdbNode<SC, E>, idx_len: usize, data_len: usize) -> PathBuf {
        let mut path = PathBuf::new();
        path.push(".axiom");
        path.push("axdb");
        path.push("pk");
        path.push(format!("{}_{}_{}.pk.bin", node.name(), idx_len, data_len));
        path
    }
}
