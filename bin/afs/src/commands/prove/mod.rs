use std::{marker::PhantomData, sync::Arc, time::Instant};

use afs_chips::{
    execution_air::ExecutionAir,
    page_rw_checker::page_controller::{OpType, Operation, PageController},
};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkPartialProvingKey,
    prover::trace::{ProverTraceData, TraceCommitmentBuilder},
};
use afs_test_utils::{
    engine::StarkEngine,
    page_config::{PageConfig, PageMode},
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::{
        types::{AfsOperation, InputFileOp},
        AfsInputFile,
    },
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::codec::fixed_bytes::FixedBytesCodec,
    utils::{fixed_bytes_to_u16_vec, string_to_u8_vec},
};
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::de::DeserializeOwned;
use tracing::info_span;

use crate::commands::{read_from_path, write_bytes};

use super::create_prefix;

/// `afs prove` command
/// Uses information from config.toml to generate a proof of the changes made by a .afi file to a table
/// saves the proof in `output-folder` as */prove.bin.
#[derive(Debug, Parser)]
pub struct ProveCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "DB file input (default: new empty DB)",
        required = true
    )]
    pub db_file_path: String,

    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains keys",
        required = false,
        default_value = "keys"
    )]
    pub keys_folder: String,

    #[arg(
        long = "cache-folder",
        short = 'c',
        help = "The folder that contains cached traces",
        required = false,
        default_value = "cache"
    )]
    pub cache_folder: String,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> ProveCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    /// Execute the `prove` command
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        afi_file_path: String,
        db_file_path: String,
        keys_folder: String,
        cache_folder: String,
        silent: bool,
        // durations: Option<&mut (Duration, Duration)>,
    ) -> Result<()> {
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => Self::execute_rw(
                config,
                engine,
                prefix,
                afi_file_path,
                db_file_path,
                keys_folder,
                cache_folder,
                silent,
                // durations,
            )?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Proved table operations in {:?}", duration);

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_rw(
        config: &PageConfig,
        engine: &E,
        prefix: String,
        afi_file_path: String,
        db_file_path: String,
        keys_folder: String,
        cache_folder: String,
        silent: bool,
        // durations: Option<&mut (Duration, Duration)>,
    ) -> Result<()> {
        println!("Proving ops file: {}", afi_file_path);
        let instructions = AfsInputFile::open(&afi_file_path)?;
        let mut db = MockDb::from_file(&db_file_path);
        let idx_len = (config.page.index_bytes + 1) / 2;
        let data_len = (config.page.data_bytes + 1) / 2;
        let height = config.page.height;
        let codec = FixedBytesCodec::new(
            config.page.index_bytes,
            config.page.data_bytes,
            config.page.index_bytes,
            config.page.data_bytes,
        );
        let mut interface =
            AfsInterface::new(config.page.index_bytes, config.page.data_bytes, &mut db);
        let table_id = instructions.header.table_id;
        let page_init = interface.get_table(table_id.clone()).unwrap().to_page(
            config.page.index_bytes,
            config.page.data_bytes,
            height,
        );

        let zk_ops = instructions
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| afi_op_conv(op, table_id.clone(), &mut interface, i + 1, &codec))
            .collect::<Vec<_>>();

        assert!(height > 0);
        let page_bus_index = 0;
        let range_bus_index = 1;
        let ops_bus_index = 2;

        let checker_trace_degree = config.page.max_rw_ops * 4;
        let idx_limb_bits = config.page.bits_per_fe;
        let idx_decomp = 8;

        let mut page_controller: PageController<SC> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let init_prover_data_encoded =
            read_from_path(cache_folder.clone() + "/" + &table_id + ".cache.bin").unwrap();
        let init_prover_data: ProverTraceData<SC> =
            bincode::deserialize(&init_prover_data_encoded).unwrap();

        let (init_page_pdata, final_page_pdata) = page_controller.load_page_and_ops(
            &page_init,
            Some(Arc::new(init_prover_data)),
            None,
            zk_ops.clone(),
            checker_trace_degree,
            &mut trace_builder.committer,
        );

        // Generating trace for ops_sender and making sure it has height num_ops
        let trace_span = info_span!("Prove.generate_trace").entered();
        let ops_sender_trace = ops_sender.generate_trace(&zk_ops, config.page.max_rw_ops);
        trace_span.exit();

        let encoded_pk =
            read_from_path(keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<SC> =
            bincode::deserialize(&encoded_pk).unwrap();

        let proof = page_controller.prove(
            engine,
            &partial_pk,
            &mut trace_builder,
            init_page_pdata,
            final_page_pdata,
            &ops_sender,
            ops_sender_trace,
        );
        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let table = interface.get_table(table_id.clone()).unwrap();
        if !silent {
            println!("Table ID: {}", table_id);
            println!("{:?}", table.metadata);
            for (index, data) in table.body.iter() {
                println!("{:?}: {:?}", index, data);
            }
        }
        let proof_path = db_file_path.clone() + ".prove.bin";
        write_bytes(&encoded_proof, proof_path).unwrap();
        db.save_to_file(&(db_file_path.clone() + ".0"))?;
        Ok(())
    }
}

fn afi_op_conv(
    afi_op: &AfsOperation,
    table_id: String,
    interface: &mut AfsInterface,
    clk: usize,
    codec: &FixedBytesCodec,
) -> Operation {
    let idx_u8 = string_to_u8_vec(afi_op.args[0].clone(), codec.db.index_bytes);
    let idx_u16 = fixed_bytes_to_u16_vec(idx_u8.clone());
    let idx = codec.db_to_table_index_bytes(idx_u8.clone());
    match afi_op.operation {
        InputFileOp::Read => {
            assert!(afi_op.args.len() == 1);
            let data = interface
                .read(table_id, codec.db_to_table_index_bytes(idx_u8))
                .unwrap();
            let data_bytes = codec.table_to_db_data_bytes(data.clone());
            let data_u16 = fixed_bytes_to_u16_vec(data_bytes);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Read,
            }
        }
        InputFileOp::Insert => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_u8_vec(afi_op.args[1].clone(), codec.db.data_bytes);
            let data_u16 = fixed_bytes_to_u16_vec(data_u8.clone());
            let data = codec.db_to_table_data_bytes(data_u8);
            interface.insert(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        InputFileOp::Write => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_u8_vec(afi_op.args[1].clone(), codec.db.data_bytes);
            let data_u16 = fixed_bytes_to_u16_vec(data_u8.clone());
            let data = codec.db_to_table_data_bytes(data_u8);
            interface.write(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        InputFileOp::GroupBy => {
            panic!("GroupBy not supported yet")
        }
        InputFileOp::InnerJoin => {
            panic!("InnerJoin not supported yet")
        }
        InputFileOp::Where => {
            panic!("Where not supported yet")
        }
    }
}
