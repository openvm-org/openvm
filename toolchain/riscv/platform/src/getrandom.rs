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

use getrandom::{register_custom_getrandom, Error};

/// This is a getrandom handler for the zkvm. It's intended to hook into a
/// getrandom crate or a dependent of the getrandom crate used by the guest code.
#[cfg(feature = "getrandom")]
pub fn zkvm_getrandom(dest: &mut [u8]) -> Result<(), Error> {
    todo!()
    // Randomness would come from the host
}

#[cfg(not(feature = "getrandom"))]
pub fn zkvm_getrandom(dest: &mut [u8]) -> Result<(), Error> {
    panic!("getrandom is not enabled in the current build");
}

register_custom_getrandom!(zkvm_getrandom);
