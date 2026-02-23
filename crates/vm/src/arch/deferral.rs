use std::collections::HashMap;

pub type InputRaw = Vec<u8>;
pub type OutputRaw = Vec<u8>;
pub type InputCommit = Vec<u8>;
pub type OutputCommit = Vec<u8>;

#[derive(Clone, Debug)]
pub enum InputMapVal {
    Raw(InputRaw),
    Output(OutputCommit),
}

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralResult {
    pub input: InputCommit,
    pub output_commit: OutputCommit,
    pub output_raw: OutputRaw,
}

#[derive(Clone, Debug)]
pub struct DeferralState {
    input_map: HashMap<InputCommit, InputMapVal>,
    output_map: HashMap<OutputCommit, OutputRaw>,
}

impl DeferralState {
    pub fn new(generated: Vec<DeferralResult>) -> Self {
        let (input_map, output_map) = generated
            .into_iter()
            .map(|res| {
                (
                    (res.input, InputMapVal::Output(res.output_commit.clone())),
                    (res.output_commit, res.output_raw),
                )
            })
            .unzip();
        Self {
            input_map,
            output_map,
        }
    }

    pub fn store_input(&mut self, input_commit: InputCommit, input_raw: InputRaw) {
        self.input_map
            .insert(input_commit, InputMapVal::Raw(input_raw));
    }

    pub fn store_output(
        &mut self,
        input_commit: &InputCommit,
        output_commit: OutputCommit,
        output_raw: OutputRaw,
    ) {
        *(self.input_map.get_mut(input_commit).unwrap()) =
            InputMapVal::Output(output_commit.clone());
        self.output_map.insert(output_commit, output_raw);
    }

    pub fn get_input(&self, input_commit: &InputCommit) -> &InputMapVal {
        self.input_map.get(input_commit).unwrap()
    }

    pub fn get_output(&self, output_commit: &OutputCommit) -> &OutputRaw {
        self.output_map.get(output_commit).unwrap()
    }
}
