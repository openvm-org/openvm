#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(target_os = "zkvm", no_main)]

extern crate alloc;

use revm::{
    db::InMemoryDB,
    primitives::{hex, Bytes, ExecutionResult, Output, TxKind},
    Evm,
};

axvm::entry!(main);

fn main() {
    let bytecode: Bytes = axvm::io::read_vec().into();

    let mut evm = Evm::builder()
        .with_db(InMemoryDB::default())
        .modify_tx_env(|tx| {
            tx.transact_to = TxKind::Create;
            tx.data = bytecode.clone();
        })
        .build();

    tracing::info!("bytecode: {}", hex::encode(bytecode));
    let ref_tx = evm.transact_commit().unwrap();
    let ExecutionResult::Success {
        output: Output::Create(_, Some(address)),
        ..
    } = ref_tx
    else {
        panic!("Failed to create contract: {ref_tx:#?}");
    };

    tracing::info!("Created contract at {address}");
    evm = evm
        .modify()
        .modify_tx_env(|tx| {
            tx.transact_to = TxKind::Call(address);
            tx.data = Default::default();
            if let Some(nonce) = tx.nonce.as_mut() {
                *nonce += 1;
            } else {
                tx.nonce = Some(1);
            }
        })
        .build();

    let _result = evm.transact().unwrap();
    // let Some(storage0) = result
    //     .state
    //     .get(&address)
    //     .expect("Contract not found")
    //     .storage
    //     .get::<U256>(&Default::default())
    // else {
    //     panic!("Failed to write storage in the init code: {result:#?}");
    // };

    // tracing::info!("storage U256(0) at {address}:  {storage0:#?}");
    // assert_eq!(storage0.present_value(), U256::from(0x42), "{result:#?}");
}
