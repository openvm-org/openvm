// [!region imports]
use openvm as _;
// [!endregion imports]

// [!region main]

// todo: call the forked tiny keccak library once that is updated instead of directly calling the
// keccak256_guest native functions
pub fn main() {
    #[cfg(target_os = "zkvm")]
    {
        let mut buffer = [0u8; 1000];
        let mut output = [0u8; 32];
        openvm_keccak256_guest::native_keccak256(output.as_mut_ptr(), 1000, buffer.as_mut_ptr());
        // let mut expected_output = [0u8; 32];
        // openvm_keccak256::keccak256(&buffer);
        // assert_eq!(output, expected_output);
    }
}
// [!endregion main]
