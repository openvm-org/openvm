// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Functions for interacting with the host environment.
//!
//! The zkVM provides a set of functions to perform operations that manage
//! execution, I/O, and proof composition. The set of functions related to each
//! of these operations are described below.
//!
//! ## System State
//!
//! The guest has some control over the execution of the zkVM by pausing or
//! exiting the program explicitly. This can be achieved using the [pause] and
//! [exit] functions.
//!
//! ## Proof Verification
//!
//! The zkVM supports verification of RISC Zero [receipts] in a guest program,
//! enabling [proof composition]. This can be achieved using the [verify()] and
//! [verify_integrity] functions.
//!
//! ## Input and Output
//!
//! The zkVM provides a set of functions for handling input, public output, and
//! private output. This is useful when interacting with the host and committing
//! to some data publicly.
//!
//! The zkVM provides functions that automatically perform (de)serialization on
//! types and, for performance reasons, there is also a `_slice` variant that
//! works with raw slices of plain old data. Performing operations on slices is
//! more efficient, saving cycles during execution and consequently producing
//! smaller proofs that are faster to produce. However, the `_slice` variants
//! can be less ergonomic, so consider trade-offs when choosing between the two.
//! For more information about guest optimization, see RISC Zero's [instruction
//! on guest optimization][guest-optimization]
//!
//! Convenience functions to read and write to default file descriptors are
//! provided. See [read()], [write()], [self::commit] (and their `_slice`
//! variants) for more information.
//!
//! In order to access default file descriptors directly, see [stdin], [stdout],
//! [stderr] and [journal]. These file descriptors are either [FdReader] or
//! [FdWriter] instances, which can be used to read from or write to the host.
//! To read from or write into them, use the [Read] and [Write] traits.
//!
//! WARNING: Specifying a file descriptor with the same value of a default file
//! descriptor is not recommended and may lead to unexpected behavior. A list of
//! default file descriptors can be found in the [fileno] module.
//!
//! ## Utility
//!
//! The zkVM provides utility functions to log messages to the debug console and
//! to measure the number of processor cycles that have occurred since the guest
//! began. These can be achieved using the [log] and [cycle_count] functions.
//!
//! [receipts]: crate::Receipt
//! [proof composition]:https://www.risczero.com/blog/proof-composition
//! [guest-optimization]:
//!     https://dev.risczero.com/api/zkvm/optimization#when-reading-data-as-raw-bytes-use-envread_slice

extern crate alloc;

// mod read;
// mod write;

use alloc::alloc::{alloc, Layout};

use axvm_platform;
use bytemuck::Pod;
use serde::{de::DeserializeOwned, Serialize};

// pub use self::{
//     read::{FdReader, Read},
//     write::{FdWriter, Write},
// };

/// A random 16 byte value initialized to random data, provided by the host, on
/// guest start and upon resuming from a pause. Setting this value ensures that
/// the total memory image has at least 128 bits of entropy, preventing
/// information leakage through the post-state digest.
static mut MEMORY_IMAGE_ENTROPY: [u32; 4] = [0u32; 4];

/// Initialize globals before program main
pub(crate) fn init() {
    // unsafe {
    //     syscall::sys_rand(
    //         MEMORY_IMAGE_ENTROPY.as_mut_ptr(),
    //         MEMORY_IMAGE_ENTROPY.len(),
    //     )
    // }
}

/// Finalize execution
pub(crate) fn finalize() {}

/// Terminate execution of the zkVM.
///
/// Use an exit code of 0 to indicate success, and non-zero to indicate an error.
pub fn exit<const exit_code: u8>() -> ! {
    axvm_platform::rust_rt::terminate::<exit_code>();
    unreachable!();
}

/*
/// Serialize the given data and commit it to the journal.
///
/// Data in the journal is included in the receipt and is available to the
/// verifier. It is considered "public" data.
///
/// # Example
///
/// ```no_run
/// use risc0_zkvm::guest::env;
/// use std::collections::BTreeMap;
///
/// let data: BTreeMap<u64, bool> = BTreeMap::from([
///   (1, true),
///   (2, false),
/// ]);
///
/// env::commit(&data);
/// ```
///
/// More examples can be found in RISC Zero's [example page].
///
/// Additional explanation on I/O in the zkVM can be found in RISC Zero's [I/O documentation].
///
/// [example page]: https://dev.risczero.com/api/zkvm/examples
/// [I/O documentation]: https://dev.risczero.com/api/zkvm/tutorials/io
pub fn commit<T: Serialize>(data: &T) {
    journal().write(data)
}

/// Commit the given slice to the journal.
///
/// Data in the journal is included in the receipt and is available to the
/// verifier. It is considered "public" data.
///
/// This function reads a slice of [plain old data][bytemuck::Pod], not
/// incurring in deserialization overhead. Recommended for performance
/// optimizations. For more context on this, see RISC Zero's [instructions on
/// guest optimization].
///
/// # Example
///
/// ```no_run
/// use risc0_zkvm::guest::env;
///
/// let slice = [1u8, 2, 3, 4];
/// env::commit_slice(&slice);
/// ```
///
/// More examples can be found in RISC Zero's [example page].
///
/// Additional explanation on I/O in the zkVM can be found in RISC Zero's [I/O documentation].
///
/// [example page]: https://dev.risczero.com/api/zkvm/examples
/// [I/O documentation]: https://dev.risczero.com/api/zkvm/tutorials/io
/// [instructions on guest optimization]: https://dev.risczero.com/api/zkvm/optimization#when-reading-data-as-raw-bytes-use-envread_slice
pub fn commit_slice<T: Pod>(slice: &[T]) {
    journal().write_slice(slice);
}
*/
