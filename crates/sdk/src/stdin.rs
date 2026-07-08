use std::collections::VecDeque;

use itertools::Itertools;
use openvm_circuit::arch::{deferral::DeferralState, Streams};
use openvm_stark_backend::codec::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Program input. Both the input records and the derived hint stream are byte-backed.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct StdIn {
    pub buffer: VecDeque<Vec<u8>>,
    pub deferrals: Vec<DeferralState>,
}

impl StdIn {
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut ret = Self::default();
        ret.write_bytes(data);
        ret
    }

    pub fn read(&mut self) -> Option<Vec<u8>> {
        self.buffer.pop_front()
    }

    pub fn write<T: Serialize>(&mut self, data: &T) {
        let words = openvm::serde::to_vec(data).unwrap();
        let bytes: Vec<u8> = words.into_iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write_bytes(&bytes);
    }

    pub fn write_bytes(&mut self, data: &[u8]) {
        self.buffer.push_back(data.to_vec());
    }
}

impl From<StdIn> for Streams {
    fn from(mut std_in: StdIn) -> Self {
        let mut data = Vec::<Vec<u8>>::new();
        while let Some(input) = std_in.read() {
            data.push(input);
        }
        Streams {
            input_stream: data.into(),
            deferrals: std_in.deferrals,
            ..Default::default()
        }
    }
}

impl From<Vec<Vec<u8>>> for StdIn {
    fn from(inputs: Vec<Vec<u8>>) -> Self {
        let mut ret = StdIn::default();
        for input in inputs {
            ret.write_bytes(&input);
        }
        ret
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct DeferralInput {
    pub byte_vec: Vec<Vec<u8>>,
}

impl DeferralInput {
    pub fn is_empty(&self) -> bool {
        self.byte_vec.is_empty()
    }

    pub fn into_inputs<I: Decode>(self) -> Vec<I> {
        self.byte_vec
            .iter()
            .map(|input| I::decode_from_bytes(input).unwrap())
            .collect_vec()
    }

    pub fn from_inputs<I: Encode>(inputs: &[I]) -> Self {
        let byte_vec = inputs
            .iter()
            .map(|input| input.encode_to_vec().unwrap())
            .collect_vec();
        Self { byte_vec }
    }
}
