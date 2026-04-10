use openvm as _;
use openvm_sha2::Sha256;

pub fn main() {
    let num_bytes: u32 = openvm::io::read();

    // Feed data to SHA-256 in 4 KB chunks to avoid a single huge allocation.
    const CHUNK_SIZE: usize = 4096;
    let chunk = vec![0xABu8; CHUNK_SIZE];

    let full_chunks = (num_bytes as usize) / CHUNK_SIZE;
    let remainder = (num_bytes as usize) % CHUNK_SIZE;

    let mut hasher = Sha256::new();
    for _ in 0..full_chunks {
        hasher.update(&chunk);
    }
    if remainder > 0 {
        hasher.update(&chunk[..remainder]);
    }
    hasher.finalize();
}
