// [!region imports]
use openvm as _;
// [!endregion imports]

// [!region main]

// todo: call the forked tiny keccak library once that is updated instead of directly calling the
// keccak256_guest native functions
pub fn main() {
    #[cfg(target_os = "zkvm")]
    {
        let mut buffer = [1u8; 200];
        let input = [2u8; 136];
        let output = [3u8; 136];
        let len: usize = 136;
        openvm_new_keccak256_guest::native_xorin(buffer.as_mut_ptr(), input.as_ptr(), len);
        assert_eq!(buffer[..136], output);
        openvm_new_keccak256_guest::native_keccakf(buffer.as_mut_ptr());
    }
}
// [!endregion main]
