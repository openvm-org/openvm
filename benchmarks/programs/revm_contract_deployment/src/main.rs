#![no_std]
#![no_main]

extern crate alloc;
extern crate revm;

use revm::{
    db::InMemoryDB,
    interpreter::opcode,
    primitives::{hex, Bytes, ExecutionResult, Output, TxKind, U256},
    Evm,
};

/// Load number parameter and set to storage with slot 0
const INIT_CODE: &[u8] = &[
    opcode::PUSH1,
    0x01,
    opcode::PUSH1,
    0x17,
    opcode::PUSH1,
    0x1f,
    opcode::CODECOPY,
    opcode::PUSH0,
    opcode::MLOAD,
    opcode::PUSH0,
    opcode::SSTORE,
];

/// Copy runtime bytecode to memory and return
const RET: &[u8] = &[
    opcode::PUSH1,
    0x02,
    opcode::PUSH1,
    0x15,
    opcode::PUSH0,
    opcode::CODECOPY,
    opcode::PUSH1,
    0x02,
    opcode::PUSH0,
    opcode::RETURN,
];

/// Load storage from slot zero to memory
const RUNTIME_BYTECODE: &[u8] = &[opcode::PUSH0, opcode::SLOAD];

axvm::entry!(main);

fn main() {
    let param = 0x42;
    let bytecode: Bytes = [INIT_CODE, RET, RUNTIME_BYTECODE, &[param]].concat().into();
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
            *tx.nonce.as_mut().unwrap() += 1;
        })
        .build();

    let result = evm.transact().unwrap();
    let Some(storage0) = result
        .state
        .get(&address)
        .expect("Contract not found")
        .storage
        .get::<U256>(&Default::default())
    else {
        panic!("Failed to write storage in the init code: {result:#?}");
    };

    tracing::info!("storage U256(0) at {address}:  {storage0:#?}");
    assert_eq!(storage0.present_value(), U256::from(param), "{result:#?}");
}
