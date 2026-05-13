pub mod poseidon2;

use openvm_stark_backend::p3_field::Field;

pub trait Hasher<const DIGEST_WIDTH: usize, F: Field> {
    /// Statelessly compresses two chunks of data into a single chunk.
    fn compress(&self, left: &[F; DIGEST_WIDTH], right: &[F; DIGEST_WIDTH]) -> [F; DIGEST_WIDTH];
    fn hash(&self, values: &[F; DIGEST_WIDTH]) -> [F; DIGEST_WIDTH] {
        self.compress(values, &[F::ZERO; DIGEST_WIDTH])
    }
    /// Chunk a list of fields. Use chunks as leaves to computes the root of the Merkle tree.
    /// Assumption: the number of public values is a power of two * DIGEST_WIDTH.
    fn merkle_root(&self, values: &[F]) -> [F; DIGEST_WIDTH] {
        let mut leaves: Vec<_> = chunk_public_values(values)
            .into_iter()
            .map(|c| self.hash(&c))
            .collect();
        while leaves.len() > 1 {
            leaves = leaves
                .chunks_exact(2)
                .map(|c| self.compress(&c[0], &c[1]))
                .collect();
        }
        leaves[0]
    }
}
pub trait HasherChip<const DIGEST_WIDTH: usize, F: Field>: Hasher<DIGEST_WIDTH, F> + Send + Sync {
    /// Stateful version of `hash` for recording the event in the chip.
    fn compress_and_record(&self, left: &[F; DIGEST_WIDTH], right: &[F; DIGEST_WIDTH]) -> [F; DIGEST_WIDTH];
    fn hash_and_record(&self, values: &[F; DIGEST_WIDTH]) -> [F; DIGEST_WIDTH] {
        self.compress_and_record(values, &[F::ZERO; DIGEST_WIDTH])
    }
}

fn chunk_public_values<const DIGEST_WIDTH: usize, F: Field>(public_values: &[F]) -> Vec<[F; DIGEST_WIDTH]> {
    public_values
        .chunks_exact(DIGEST_WIDTH)
        .map(|c| c.try_into().unwrap())
        .collect()
}
