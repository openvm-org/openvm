use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use itertools::Itertools;
use openvm_circuit::arch::{deferral::DeferralState, Streams};
use openvm_stark_backend::{
    codec::{Decode, Encode},
    p3_field::Field,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct StdIn<F = crate::F> {
    pub buffer: VecDeque<Vec<F>>,
    pub kv_store: HashMap<Vec<u8>, Vec<u8>>,
    pub deferrals: Vec<DeferralState>,
}

impl<F: Field> StdIn<F> {
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut ret = Self::default();
        ret.write_bytes(data);
        ret
    }

    pub fn read(&mut self) -> Option<Vec<F>> {
        self.buffer.pop_front()
    }

    pub fn write<T: Serialize>(&mut self, data: &T) {
        let words = openvm::serde::to_vec(data).unwrap();
        let bytes: Vec<u8> = words.into_iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write_bytes(&bytes);
    }

    pub fn write_bytes(&mut self, data: &[u8]) {
        let field_data = data.iter().map(|b| F::from_u8(*b)).collect();
        self.buffer.push_back(field_data);
    }

    pub fn write_field(&mut self, data: &[F]) {
        self.buffer.push_back(data.to_vec());
    }
    pub fn add_key_value(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.kv_store.insert(key, value);
    }
}

impl<F: Field> From<StdIn<F>> for Streams<F> {
    fn from(mut std_in: StdIn<F>) -> Self {
        let mut data = Vec::<Vec<F>>::new();
        while let Some(input) = std_in.read() {
            data.push(input);
        }
        let mut ret = Streams::new(data);
        ret.kv_store = Arc::new(std_in.kv_store);
        ret.deferrals = std_in.deferrals;
        ret
    }
}

impl<F: Field> From<Vec<Vec<F>>> for StdIn<F> {
    fn from(inputs: Vec<Vec<F>>) -> Self {
        let mut ret = StdIn::<F>::default();
        for input in inputs {
            ret.write_field(&input);
        }
        ret
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct DeferralInput {
    pub byte_vec: Vec<Vec<u8>>,
}

impl DeferralInput {
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
