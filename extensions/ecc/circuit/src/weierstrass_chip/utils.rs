#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CurveType {
    K256 = 0,
    Generic = 1,
}

#[inline(always)]
pub fn blocks_to_field_element<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    blocks: &[[u8; BLOCK_SIZE]],
) -> k256::FieldElement {
    let mut bytes = [0u8; 32];
    let mut idx = 0;

    for block in blocks {
        for &byte in block {
            if idx < 32 {
                bytes[idx] = byte;
                idx += 1;
            }
        }
    }

    bytes.reverse(); // Convert to big-endian for k256
    k256::FieldElement::from_bytes(&bytes.into()).unwrap()
}

#[inline(always)]
pub fn field_element_to_blocks<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_element: &k256::FieldElement,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
    start_block: usize,
) {
    let mut bytes = field_element.to_bytes();
    bytes.reverse(); // Convert to little-endian for blocks

    let mut idx = 0;
    for block in output
        .iter_mut()
        .skip(start_block)
        .take(BLOCKS.min(start_block + BLOCKS / 2) - start_block)
    {
        for byte in block.iter_mut().take(BLOCK_SIZE) {
            if idx < 32 {
                *byte = bytes[idx];
                idx += 1;
            }
        }
    }
}
