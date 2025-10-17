//! Program to match the workload of <https://github.com/bluealloy/revm/blob/900409f134c1cbd4489d370a6b037f354afa4a5c/crates/revm/benches/bench.rs#L68>
//! We run 100 transfers to take the average
use alloy_primitives::{address, TxKind, U256};
#[allow(unused_imports, clippy::single_component_path_imports)]
use openvm_keccak256; // export native keccak
use revm::{db::BenchmarkDB, primitives::Bytecode, Evm};

// Necessary so the linker doesn't skip importing openvm crate
openvm::entry!(main);

fn main() {
    let mut evm = Evm::builder()
        .with_db(BenchmarkDB::new_bytecode(Bytecode::new()))
        .build();

    for i in 0..100 {
        evm = evm
            .modify()
            .modify_tx_env(|tx| {
                tx.caller = address!("0000000000000000000000000000000000000001");
                tx.transact_to = TxKind::Call(address!("0000000000000000000000000000000000000000"));
                tx.value = U256::from(10 + i);
            })
            .build();
        evm.transact().unwrap();
    }
}
