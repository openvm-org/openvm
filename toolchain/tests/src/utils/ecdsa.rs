use axvm::intrinsics::keccak256;

/// keystore_address = keccak256(salt, keccak256([eoa_addrs]), keccak256(vk))
pub fn generate_keystore_address(
    salt: [u8; 32],
    eoa_addrs: Vec<[u8; 20]>,
    vk_hash: [u8; 32],
) -> [u8; 32] {
    // concatenate EOAs
    let concat_eoas = eoa_addrs
        .iter()
        .fold(vec![], |acc, addr| [acc, addr.to_vec()].concat());
    let data_hash = keccak256(concat_eoas.as_slice());

    let concat_all = [salt.to_vec(), eoa_addrs, vk_hash.to_vec()].concat();
    keccak256(concat_all.as_slice())
}

pub fn generate_msg_hash(message: &str) -> [u8; 32] {
    keccak256(message.as_bytes())
}

pub fn generate_vk_hash(vk: &[u8]) -> [u8; 32] {
    hex!("0000000000000000000000000000000000000000000000000000000000000000")
}
