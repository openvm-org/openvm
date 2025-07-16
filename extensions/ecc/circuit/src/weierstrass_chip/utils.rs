use crypto_bigint::Encoding;
use k256::{
    elliptic_curve::{FieldBytesEncoding, PrimeField},
    U256,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CurveType {
    K256 = 0,
    Generic = 1,
}

#[inline(always)]
pub fn blocks_to_field_element(blocks: &[u8]) -> k256::FieldElement {
    let mut bytes = [0u8; 32];
    let len = blocks.len().min(32);
    bytes[..len].copy_from_slice(&blocks[..len]);

    let num = U256::from_le_bytes(bytes);
    k256::FieldElement::from_repr_vartime(num.encode_field_bytes()).unwrap()
}

#[inline(always)]
pub fn field_element_to_blocks<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_element: &k256::FieldElement,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
    start_block: usize,
) {
    let bytes = field_element.to_bytes();
    let mut byte_iter = bytes.iter().rev();
    for block in output.iter_mut().skip(start_block) {
        for byte in block.iter_mut() {
            *byte = byte_iter.next().copied().unwrap_or(0);
        }
    }
}
