use std::collections::VecDeque;

use ax_stark_sdk::p3_bn254_fr::Bn254Fr;
use axvm_circuit::arch::Streams;
use p3_field::AbstractField;
use serde::{Deserialize, Serialize};

use crate::F;

pub(crate) type Fr = Bn254Fr;

#[derive(Clone, Deserialize, Serialize)]
pub struct EvmProof {
    pub instances: Vec<Vec<Fr>>,
    pub proof: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum InputType {
    Bytes = 0,
    Field,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct StdIn {
    pub buffer: VecDeque<Vec<u8>>,
    pub field_buffer: VecDeque<Vec<F>>,
    next: VecDeque<InputType>,
}

impl StdIn {
    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            buffer: VecDeque::from(vec![data.to_vec()]),
            field_buffer: VecDeque::new(),
            next: VecDeque::from(vec![InputType::Bytes]),
        }
    }

    pub fn read(&mut self) -> Option<Vec<F>> {
        if let Some(input_type) = self.next.pop_front() {
            if input_type == InputType::Bytes {
                let bytes = self.buffer.pop_front().unwrap();
                Some(bytes.iter().map(|b| F::from_canonical_u8(*b)).collect())
            } else {
                self.field_buffer.pop_front()
            }
        } else {
            None
        }
    }

    pub fn write<T: Serialize>(&mut self, data: &T) {
        let bytes = bincode::serialize(data).unwrap();
        self.buffer.push_back(bytes);
        self.next.push_back(InputType::Bytes);
    }

    pub fn write_bytes(&mut self, data: &[u8]) {
        self.buffer.push_back(data.to_vec());
        self.next.push_back(InputType::Bytes);
    }

    pub fn write_field(&mut self, data: &[F]) {
        self.field_buffer.push_back(data.to_vec());
        self.next.push_back(InputType::Field);
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
