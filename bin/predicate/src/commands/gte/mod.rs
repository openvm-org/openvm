use afs_chips::single_page_index_scan::page_index_scan_input::Comp;
use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;

use super::common::{execute_predicate_common, CommonCommands};

#[derive(Debug, Parser)]
pub struct GteCommand {
    #[command(flatten)]
    pub args: CommonCommands,
}

/// `gte` command
impl GteCommand {
    /// Execute the `gte` command
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        println!(
            "Running `index >= {}` command on table: {}",
            self.args.value, self.args.table_id
        );
        execute_predicate_common(self.args, config, Comp::Gte)
    }
}
