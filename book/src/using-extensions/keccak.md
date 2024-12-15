# Keccak Guest Library

## Functions for guest code

The library provides two functions for using in your guest code:

- `keccak256(input: &[u8]) -> [u8; 32]`: Computes the Keccak-256 hash of the input data and returns it as an array of 32 bytes.
- `set_keccak256(input: &[u8], output: &mut [u8; 32])`: Sets the output to the Keccak-256 hash of the input data into the provided output buffer.


To use the Keccak Guest Library, the `openvm-keccak-guest` crate must be imported in your program. Then you can use the above functions as in normal Rust code.

### Example:
```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use openvm::io::read_vec;
use openvm_keccak256_guest::keccak256;

openvm::entry!(main);

pub fn main() {
    let input = Vec::read_vec();
    let expected_output = Vec::read_vec();
    let output = keccak256(&input);
    if output != *expected_output {
        panic!();
    }
}
```

To actually run the above example guest program you would have something like this:

```rust
let elf = Elf::decode("your_path_to_elf", MEM_SIZE as u32)?;
let vm_config = SdkVmConfig::builder()
    .system(Default::default())
    .rv32im(Default::default())
    .io(Default::default())
    .keccak256(Default::default())
    .build();
// or Keccak256Rv32Config::default();
let transpiler = Transpiler::<F>::default()
    .with_extension(Rv32ITranspilerExtension)
    .with_extension(Rv32MTranspilerExtension)
    .with_extension(Rv32IoTranspilerExtension)
    .with_extension(Keccak256TranspilerExtension);
let executor = VmExecutor::<F, _>::new(vm_config);
let input = [].map(F::from_canonical_u8).to_vec();
let expected_output = [
        197, 210, 70, 1, 134, 247, 35, 60, 146, 126, 125, 178, 220, 199, 3, 192, 229, 0, 182, 83,
        202, 130, 39, 59, 123, 250, 216, 4, 93, 133, 164, 112,
].map(F::from_canonical_u8).to_vec();
executor.execute(openvm_exe, vec![input, expected_output])?;
```

## Native Keccak256

Keccak guest library provides another way to use the native Keccak-256 implementation. It provides a function that is meant to be linked to other external libraries. The external libraries can use this function as a hook for the Keccak-256 native implementation. Enabled only when the target is `zkvm`.

- `native_keccak256(input: *const u8, len: usize, output: *mut u8)`: This function has `C` ABI. Enabled only when the target is `zkvm` and takes in a pointer to the input, the length of the input, and a pointer to the output buffer.

In the external library, you can do the following:

```rust
extern "C" {
    fn native_keccak256(input: *const u8, len: usize, output: *mut u8);
}

fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(target_os = "zkvm")] {
    let mut output = [0u8; 32];
        unsafe {
            native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
        }
        output
    }
    #[cfg(not(target_os = "zkvm"))] {
        // Regular Keccak-256 implementation
    }
}
```

