use std::{
    fs::{self},
    path::Path,
    time::Instant,
};

use afs_test_utils::{
    config::{self},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, VirtualMachine},
};

use super::{write_bytes, WORD_SIZE};
use crate::asm::parse_asm_file;

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.pk and *.vk.
#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[arg(
        long = "asm-file",
        short = 'f',
        help = "The .asm file input",
        required = true
    )]
    pub asm_file_path: String,
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
    pub fn execute(self, config: VmConfig) -> Result<()> {
        let start = Instant::now();
        self.execute_helper(config)?;
        let duration = start.elapsed();
        println!("Generated keys in {:?}", duration);
        Ok(())
    }

    fn execute_helper(self, config: VmConfig) -> Result<()> {
        let instructions = parse_asm_file(Path::new(&self.asm_file_path.clone()))?;
        let program_len = instructions.len();
        let program = Program {
            instructions,
            debug_infos: vec![None; program_len],
        };
        let vm = VirtualMachine::<WORD_SIZE, _>::new(config, program, vec![]);
        let result = vm.execute()?;
        let engine = config::baby_bear_poseidon2::default_engine(result.max_log_degree);
        let mut keygen_builder = engine.keygen_builder();

        let chips = VirtualMachine::<WORD_SIZE, _>::get_chips(&result.nonempty_chips);

        for chip in chips {
            keygen_builder.add_air(chip, 0);
        }

        let pk = keygen_builder.generate_pk();
        let vk = pk.vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&vk)?;
        fs::create_dir_all(Path::new(&self.output_folder))?;
        let pk_path = Path::new(&self.output_folder).join("pk");
        let vk_path = Path::new(&self.output_folder).join("vk");
        fs::create_dir_all(self.output_folder)?;
        write_bytes(&encoded_pk, &pk_path)?;
        write_bytes(&encoded_vk, &vk_path)?;
        Ok(())
    }
}
