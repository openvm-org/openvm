use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::BabyBearPoseidon2Config,
    },
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_matrix::Matrix;
use stark_vm::vm::config::VMConfig;

use crate::isa::get_vm;

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.partial.pk and *.partial.vk.
#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[arg(
        long = "isa-file",
        short = 'f',
        help = "The .isa file input",
        required = true
    )]
    pub isa_file_path: String,
    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "keys"
    )]
    pub output_folder: String,
}

impl KeygenCommand {
    /// Execute the `keygen` command
    pub fn execute(self, config: VMConfig) -> Result<()> {
        let start = Instant::now();
        self.execute_helper(config)?;
        let duration = start.elapsed();
        println!("Generated keys in {:?}", duration);
        Ok(())
    }

    fn execute_helper(self, config: VMConfig) -> Result<()> {
        let vm = get_vm::<BabyBearPoseidon2Config>(config, &self.isa_file_path)?;
        let engine = config::baby_bear_poseidon2::default_engine(vm.max_log_degree());
        let mut keygen_builder = engine.keygen_builder(); // MultiStarkKeygenBuilder::new(&engine.config);

        let chips = vm.chips();
        let traces = vm.traces();

        for i in 0..chips.len() {
            keygen_builder.add_air(chips[i], traces[i].height(), 0);
        }

        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        let pk_path = self.output_folder.clone() + "/partial.pk";
        let vk_path = self.output_folder.clone() + "/partial.vk";
        fs::create_dir_all(self.output_folder).unwrap();
        write_bytes(&encoded_pk, pk_path).unwrap();
        write_bytes(&encoded_vk, vk_path).unwrap();
        Ok(())
    }
}

fn write_bytes(bytes: &[u8], path: String) -> Result<()> {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}
