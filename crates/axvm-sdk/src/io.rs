use ax_stark_sdk::p3_bn254_fr::Bn254Fr;
use serde::{Deserialize, Serialize};

use crate::F;

pub(crate) type Fr = Bn254Fr;

pub type StdIn = Vec<Vec<F>>;

#[derive(Clone, Deserialize, Serialize)]
pub struct EvmProof {
    pub instances: Vec<Vec<Fr>>,
    pub proof: Vec<u8>,
}
