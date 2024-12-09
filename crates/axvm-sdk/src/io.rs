use std::collections::VecDeque;

use ax_stark_backend::p3_field::AbstractField;
use axvm_circuit::arch::Streams;
use serde::{Deserialize, Serialize};

use crate::F;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct StdIn {
    pub buffer: VecDeque<Vec<F>>,
}

impl StdIn {
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut ret = Self::default();
        ret.write_bytes(data);
        ret
    }

    pub fn read(&mut self) -> Option<Vec<F>> {
        self.buffer.pop_front()
    }

    pub fn write<T: Serialize>(&mut self, data: &T) {
        let bytes = bincode::serialize(data).unwrap();
        self.write_bytes(&bytes);
    }

    pub fn write_bytes(&mut self, data: &[u8]) {
        let field_data = data.iter().map(|b| F::from_canonical_u8(*b)).collect();
        self.buffer.push_back(field_data);
    }

    pub fn write_field(&mut self, data: &[F]) {
        self.buffer.push_back(data.to_vec());
    }
}

impl From<StdIn> for Streams<F> {
    fn from(mut std_in: StdIn) -> Self {
        let mut data = Vec::<Vec<F>>::new();
        while let Some(input) = std_in.read() {
            data.push(input);
        }
        Streams::new(data)
    }
}

impl From<Vec<Vec<F>>> for StdIn {
    fn from(inputs: Vec<Vec<F>>) -> Self {
        let mut ret = StdIn::default();
        for input in inputs {
            ret.write_field(&input);
        }
        ret
    }
}
