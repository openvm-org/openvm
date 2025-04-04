use openvm as _;

use revm::{
    db::BenchmarkDB,
    primitives::{address, bytes, hex, Bytecode, Bytes, TxKind},
    Evm,
};

const BYTES: &str = include_str!("snailtracer.hex");

fn main() {
    let bytecode = Bytecode::new_raw(Bytes::from(hex::decode(BYTES).unwrap()));

    let mut evm = Evm::builder()
        .with_db(BenchmarkDB::new_bytecode(bytecode.clone()))
        .modify_tx_env(|tx| {
            tx.caller = address!("0000000000000000000000000000000000000001");
            tx.transact_to = TxKind::Call(address!("0000000000000000000000000000000000000000"));
            tx.data = bytes!("30627b7c");
            tx.gas_limit = 1_000_000_000;
        })
        .build();

    evm.transact().unwrap();
}
