use afs_page::page_btree::PageBTree;
use ax_sdk::page_config::MultitierPageConfig;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::utils::string_to_u16_vec;

use crate::commands::{BABYBEAR_COMMITMENT_LEN, LIMB_BITS};

#[derive(Debug, Parser)]
pub struct ReadCommand {
    #[arg(
        long = "table-id",
        short = 't',
        help = "The table ID",
        required = false
    )]
    pub table_id: Option<String>,

    #[arg(
        long = "db-folder",
        short = 'd',
        help = "Mock DB folder (default: new empty DB)",
        required = false,
        default_value = "multitier_mockdb"
    )]
    pub db_folder: String,

    #[arg(
        long = "index",
        short = 'i',
        help = "The index you want to query",
        required = true
    )]
    pub idx: String,
}

/// `mock read` subcommand
impl ReadCommand {
    /// Execute the `mock read` command
    pub fn execute(&self, config: &MultitierPageConfig) -> Result<()> {
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let mut db = if let Some(table_id) = &self.table_id {
            println!("db_file_path: {}/root/{}", self.db_folder, table_id);

            PageBTree::<BABYBEAR_COMMITMENT_LEN>::load(
                self.db_folder.clone(),
                table_id.to_owned(),
                "".to_owned(),
            )
            .unwrap()
        } else {
            PageBTree::new(
                LIMB_BITS,
                idx_len,
                data_len,
                config.page.leaf_height,
                config.page.internal_height,
                "".to_owned(),
            )
        };
        let idx_u16 = string_to_u16_vec(self.idx.clone(), idx_len);
        let data = db.search(&idx_u16).unwrap();
        println!(
            "Table ID: {}",
            self.table_id.clone().unwrap_or("".to_owned())
        );
        println!("{:?}: {:?}", self.idx, data);

        Ok(())
    }
}
