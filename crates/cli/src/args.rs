use std::path::PathBuf;

use clap::Parser;
use openvm_circuit::arch::OPENVM_DEFAULT_INIT_FILE_NAME;

#[derive(Clone, Parser)]
pub struct OpenVmConfigArgs {
    #[arg(
        long,
        help = "Path to the OpenVM config .toml file that specifies the VM extensions, by default will search for the file at ${manifest_dir}/openvm.toml",
        help_heading = "OpenVM Options"
    )]
    pub config: Option<PathBuf>,

    #[arg(
        long,
        help = "Output directory that OpenVM proving artifacts will be copied to",
        help_heading = "OpenVM Options"
    )]
    pub output_dir: Option<PathBuf>,

    #[arg(
        long,
        default_value = OPENVM_DEFAULT_INIT_FILE_NAME,
        help = "Name of the init file",
        help_heading = "OpenVM Options"
    )]
    pub init_file_name: String,
}

impl Default for OpenVmConfigArgs {
    fn default() -> Self {
        Self {
            config: None,
            output_dir: None,
            init_file_name: OPENVM_DEFAULT_INIT_FILE_NAME.to_string(),
        }
    }
}

#[derive(Clone, Parser)]
pub struct ManifestArgs {
    #[arg(
        long,
        value_name = "DIR",
        help = "Directory for all generated artifacts and intermediate files",
        help_heading = "Output Options"
    )]
    pub target_dir: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PATH",
        help = "Path to the Cargo.toml file, by default searches for the file in the current or any parent directory",
        help_heading = "Manifest Options"
    )]
    pub manifest_path: Option<PathBuf>,
}

impl Default for ManifestArgs {
    fn default() -> Self {
        Self {
            target_dir: None,
            manifest_path: None,
        }
    }
}

#[derive(Clone, Parser)]
pub struct ProvingKeyArgs {
    #[arg(
        long,
        action,
        help = "Path to app proving key, by default will be ${openvm_dir}/app.pk",
        help_heading = "OpenVM Options"
    )]
    pub app_pk: Option<PathBuf>,

    #[arg(
        long,
        action,
        help = "Path to aggregation proving key, by default will be ${openvm_dir}/agg.pk",
        help_heading = "OpenVM Options"
    )]
    pub agg_pk: Option<PathBuf>,
}
