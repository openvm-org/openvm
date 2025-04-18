use p3_baby_bear::BabyBear;
use util::HashTestChip;

use super::tree::MerkleTree;
use crate::{
    arch::MemoryConfig,
    system::memory::{controller::dimensions::MemoryDimensions, AddressMap},
};

mod util;

const CHUNK: usize = 4;
type F = BabyBear;

#[test]
fn test_merkle_tree_finalize() {
    let mem_config = MemoryConfig {
        pointer_max_bits: 10,
        ..Default::default()
    };
    let md = MemoryDimensions::from_config(&mem_config);
    let image = AddressMap::from_mem_config(&mem_config);
    let mut hash_test_chip = HashTestChip::new();
    let mut tree = MerkleTree::from_memory(&image, &md, &mut hash_test_chip);

    tree.finalize(&mut hash_test_chip, touched, &md);
}
