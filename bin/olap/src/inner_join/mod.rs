use std::marker::PhantomData;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::operations_file::AfsOperationsFile, afs_interface::AfsInterface, mock_db::MockDb,
    table::Table,
};
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use self::{
    keygen::InnerJoinKeygenCommand, prove::InnerJoinProveCommand, verify::InnerJoinVerifyCommand,
};

pub mod keygen;
pub mod prove;
pub mod verify;

#[derive(Debug, Parser)]
pub struct InnerJoinCli<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(subcommand)]
    pub command: InnerJoinCommand<SC, E>,

    #[command(flatten)]
    pub common: InnerJoinCommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

#[derive(Debug, Subcommand)]
pub enum InnerJoinCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[command(name = "keygen", about = "Generate keys for inner join")]
    Keygen(InnerJoinKeygenCommand<SC, E>),

    #[command(name = "prove", about = "Prove inner join")]
    Prove(prove::InnerJoinProveCommand<SC, E>),

    #[command(name = "verify", about = "Verify inner join")]
    Verify(verify::InnerJoinVerifyCommand<SC, E>),
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> InnerJoinCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &InnerJoinCommonCommands,
        command: &InnerJoinCommand<SC, E>,
    ) -> Result<()> {
        match command {
            InnerJoinCommand::Keygen(cmd) => {
                InnerJoinKeygenCommand::execute(config, engine, common).unwrap();
            }
            InnerJoinCommand::Prove(cmd) => {
                InnerJoinProveCommand::execute(config, engine, common).unwrap();
            }
            InnerJoinCommand::Verify(cmd) => {
                InnerJoinVerifyCommand::execute(config, engine, common).unwrap();
            }
        }
        Ok(())
    }
}

pub fn inner_join_setup(
    config: &PageConfig,
    common: &InnerJoinCommonCommands,
    db_path: String,
    afo_path: String,
) -> (Table, Table, usize, usize, usize) {
    let mut db = MockDb::from_file(&db_path);
    let afo = AfsOperationsFile::open(afo_path.clone());
    let op = afo.operations.get(0).unwrap();
    let interface_left = AfsInterface::new_with_table(op.table_id_left.to_string(), &mut db);
    let table_left = interface_left.current_table().unwrap();
    let page_left = table_left.to_page(
        table_left.metadata.index_bytes,
        table_left.metadata.data_bytes,
        height,
    );
    let index_len_left = (table_left.metadata.index_bytes + 1) / 2;
    let data_len_left = (table_left.metadata.data_bytes + 1) / 2;

    let interface_right = AfsInterface::new_with_table(op.table_id_right.to_string(), db);
    let table_right = interface_right.current_table().unwrap();
    let page_right = table_right.to_page(
        table_right.metadata.index_bytes,
        table_right.metadata.data_bytes,
        height,
    );
    let index_len_right = (table_right.metadata.index_bytes + 1) / 2;
    let data_len_right = (table_right.metadata.data_bytes + 1) / 2;

    (table_left, table_right, height, bits_per_fe, degree)
}
