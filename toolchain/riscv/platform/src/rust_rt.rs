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

//! This module contains the components required to link a Rust binary.
//!
//! In particular:
//! * It defines an entrypoint ensuring initialization and finalization are done
//!   properly.
//! * It includes a panic handler.
//! * It includes an allocator.

#[cfg(target_os = "zkvm")]
use crate::constants::CUSTOM_0;

extern crate alloc;

#[inline(always)]
pub fn terminate<const EXIT_CODE: u8>() {
    #[cfg(target_os = "zkvm")]
    crate::custom_insn_i!(CUSTOM_0, 0, "x0", "x0", EXIT_CODE);
    #[cfg(not(target_os = "zkvm"))]
    {
        unimplemented!()
    }
}
