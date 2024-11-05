/// The keccak256 cryptographic hash function.
#[inline(always)]
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut output = [0u8; 32];
        set_keccak256(input, &mut output);
        output
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}

/// Sets `output` to the keccak256 hash of `input`.
#[inline(always)]
pub fn set_keccak256(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use tiny_keccak::Hasher;
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(input);
        hasher.finalize(output);
    }
    #[cfg(target_os = "zkvm")]
    {
        todo!()
    }
}
