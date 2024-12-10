use std::{
    fs::{read, File},
    path::PathBuf,
};

use axvm_rv32im_transpiler::{Rv32ITranspilerExtension, Rv32MTranspilerExtension};
use axvm_sdk::Sdk;
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf, transpiler::Transpiler};
use clap::Parser;
use eyre::Result;

#[derive(Parser)]
#[command(name = "transpile", about = "Transpile an ELF into an axVM program")]
pub struct TranspileCmd {
    #[clap(long, action)]
    elf: PathBuf,
}

impl TranspileCmd {
    pub fn run(&self) -> Result<()> {
        let data = read(self.elf.clone())?;
        let elf = Elf::decode(&data, MEM_SIZE as u32)?;
        let exe = Sdk.transpile(
            elf,
            Transpiler::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension),
        )?;
        let path_name = self.elf.with_extension("axvmexe");
        let file = File::create(path_name.clone())?;
        serde_json::to_writer(file, &exe)?;
        eprintln!("Successfully transpiled to {}", path_name.display());
        Ok(())
    }
}
