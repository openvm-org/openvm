use std::{sync::Arc, time::Instant};

use afs_chips::{
    execution_air::ExecutionAir,
    page_rw_checker::page_controller::{OpType, Operation, PageController},
};
use afs_stark_backend::{
    keygen::types::MultiStarkPartialProvingKey,
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder},
        MultiTraceStarkProver,
    },
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    page_config::{PageConfig, PageMode},
};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions, AfsOperation},
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::codec::fixed_bytes::FixedBytesCodec,
    utils::{fixed_bytes_to_field_vec, string_to_be_vec},
};
use p3_util::log2_strict_usize;

use crate::commands::{read_from_path, write_bytes};

use super::create_prefix;

/// `afs prove` command
/// Uses information from config.toml to generate a proof of the changes made by a .afi file to a table
/// saves the proof in `output-folder` as */prove.bin.
#[derive(Debug, Parser)]
pub struct ProveCommand {
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
}

impl ProveCommand {
    /// Execute the `prove` command
    pub fn execute(&self, config: &PageConfig) -> Result<()> {
        let start = Instant::now();
        let prefix = create_prefix(config);
        match config.page.mode {
            PageMode::ReadWrite => self.execute_rw(config, prefix)?,
            PageMode::ReadOnly => panic!(),
        }

        let duration = start.elapsed();
        println!("Proved table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_rw(&self, config: &PageConfig, prefix: String) -> Result<()> {
        println!("Proving ops file: {}", self.afi_file_path);
        let instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        let mut db = MockDb::from_file(&self.db_file_path);
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

        let max_log_degree = log2_strict_usize(checker_trace_degree)
            .max(log2_strict_usize(height))
            .max(8);

        let idx_decomp = 8;

        let mut page_controller: PageController<BabyBearPoseidon2Config> = PageController::new(
            page_bus_index,
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_limb_bits,
            idx_decomp,
        );
        let ops_sender = ExecutionAir::new(ops_bus_index, idx_len, data_len);
        let engine = config::baby_bear_poseidon2::default_engine(max_log_degree);
        let prover = MultiTraceStarkProver::new(&engine.config);
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let init_prover_data_encoded =
            read_from_path(self.cache_folder.clone() + "/" + &table_id + ".cache.bin").unwrap();
        let init_prover_data: ProverTraceData<BabyBearPoseidon2Config> =
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
        let ops_sender_trace =
            ops_sender.generate_trace_testing(&zk_ops, config.page.max_rw_ops, 1);

        let encoded_pk =
            read_from_path(self.keys_folder.clone() + "/" + &prefix + ".partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_pk).unwrap();
        let proof = page_controller.prove(
            &engine,
            &partial_pk,
            &mut trace_builder,
            init_page_pdata,
            final_page_pdata,
            &ops_sender,
            ops_sender_trace,
        );
        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let table = interface.get_table(table_id.clone()).unwrap();
        if !self.silent {
            println!("Table ID: {}", table_id);
            println!("{:?}", table.metadata);
            for (index, data) in table.body.iter() {
                println!("{:?}: {:?}", index, data);
            }
        }
        let proof_path = self.db_file_path.clone() + ".prove.bin";
        write_bytes(&encoded_proof, proof_path).unwrap();
        db.save_to_file(&(self.db_file_path.clone() + ".0"))?;
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
    let idx_u8 = string_to_be_vec(afi_op.args[0].clone(), codec.db.index_bytes);
    let idx_u16 = fixed_bytes_to_field_vec(idx_u8.clone());
    let idx = codec.db_to_table_index_bytes(idx_u8.clone());
    match afi_op.operation {
        InputFileBodyOperation::Read => {
            assert!(afi_op.args.len() == 1);
            let data = interface
                .read(table_id, codec.db_to_table_index_bytes(idx_u8))
                .unwrap();
            let data_bytes = codec.table_to_db_data_bytes(data.clone());
            let data_u16 = fixed_bytes_to_field_vec(data_bytes);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Read,
            }
        }
        InputFileBodyOperation::Insert => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_be_vec(afi_op.args[1].clone(), codec.db.data_bytes);
            let data_u16 = fixed_bytes_to_field_vec(data_u8.clone());
            let data = codec.db_to_table_data_bytes(data_u8);
            interface.insert(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        InputFileBodyOperation::Write => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_be_vec(afi_op.args[1].clone(), codec.db.data_bytes);
            let data_u16 = fixed_bytes_to_field_vec(data_u8.clone());
            let data = codec.db_to_table_data_bytes(data_u8);
            interface.write(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
    }
}
