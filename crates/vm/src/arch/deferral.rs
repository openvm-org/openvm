use std::collections::HashMap;

use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::Field;

pub type InputRaw = Vec<u8>;
pub type OutputRaw = Vec<u8>;
pub type InputCommit<F> = [F; DIGEST_SIZE];
pub type OutputCommit<F> = [F; DIGEST_SIZE];

#[derive(Clone, Debug)]
pub enum InputMapVal<F> {
    Raw(InputRaw),
    Output(OutputCommit<F>),
}

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralResult<F> {
    pub input: InputCommit<F>,
    pub output_commit: OutputCommit<F>,
    pub output_raw: OutputRaw,
}

#[derive(Clone, Debug)]
pub struct DeferralState<F> {
    input_map: HashMap<InputCommit<F>, InputMapVal<F>>,
    output_map: HashMap<OutputCommit<F>, OutputRaw>,
}

impl<F: Field> DeferralState<F> {
    pub fn new(generated: Vec<DeferralResult<F>>) -> Self {
        let (input_map, output_map) = generated
            .into_iter()
            .map(|res| {
                (
                    (res.input, InputMapVal::Output(res.output_commit)),
                    (res.output_commit, res.output_raw),
                )
            })
            .unzip();
        Self {
            input_map,
            output_map,
        }
    }

    pub fn store_input(&mut self, input_commit: InputCommit<F>, input_raw: InputRaw) {
        self.input_map
            .insert(input_commit, InputMapVal::Raw(input_raw));
    }

    pub fn store_output(
        &mut self,
        input_commit: &InputCommit<F>,
        output_commit: OutputCommit<F>,
        output_raw: OutputRaw,
    ) {
        *(self.input_map.get_mut(input_commit).unwrap()) = InputMapVal::Output(output_commit);
        self.output_map.insert(output_commit, output_raw);
    }

    pub fn get_input(&self, input_commit: &InputCommit<F>) -> &InputMapVal<F> {
        self.input_map.get(input_commit).unwrap()
    }

    pub fn get_output(&self, output_commit: &OutputCommit<F>) -> &OutputRaw {
        self.output_map.get(output_commit).unwrap()
    }
}
