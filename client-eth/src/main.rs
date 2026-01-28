use openvm::io::{println, read, reveal_bytes32};
use openvm_client_executor::{io::ClientExecutorInput, ChainVariant, ClientExecutor};

openvm::init!();

pub fn main() {
    println("client-eth starting");
    // Read the input.
    let input: ClientExecutorInput = read();
    println("finished reading input");

    // Execute the block (crypto is installed inside executor).
    let executor = ClientExecutor;
    let header = executor.execute(ChainVariant::Mainnet, input).expect("failed to execute client");
    let block_hash = header.hash_slow();

    // Reveal the block hash.
    reveal_bytes32(*block_hash);
}
