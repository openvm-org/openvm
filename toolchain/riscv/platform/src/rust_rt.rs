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
use core::arch::asm;

#[cfg(target_os = "zkvm")]
use crate::{constants::CUSTOM_0, custom_insn_i, custom_insn_r};

extern crate alloc;

#[inline(always)]
pub fn terminate<const EXIT_CODE: u8>() {
    #[cfg(target_os = "zkvm")]
    custom_insn_i!(CUSTOM_0, 0, "x0", "x0", EXIT_CODE);
    #[cfg(not(target_os = "zkvm"))]
    {
        unimplemented!()
    }
}

#[cfg(feature = "entrypoint")]
mod entrypoint {
    #[no_mangle]
    unsafe extern "C" fn __start() -> ! {
        // This definition of __start differs from risc0_zkvm::guest in that it does not initialize the
        // journal and will halt with empty output. It also assumes main follows the standard C
        // convention, and uses the returned i32 value as the user exit code for halt.
        let exit_code = {
            extern "C" {
                fn main(argc: i32, argv: *const *const u8) -> i32;
            }

            main(0, core::ptr::null())
        };

        terminate::<0>();
    }

    static STACK_TOP: u32 = crate::memory::STACK_TOP;

    // Entry point; sets up global pointer and stack pointer and passes
    // to __start.  TODO: when asm_const is stabilized, use that here
    // instead of defining a symbol and dereferencing it.
    core::arch::global_asm!(
        r#"
    .section .text._start
    .globl _start
    _start:
        .option push;
        .option norelax
        la gp, __global_pointer$
        .option pop
        la sp, {0}
        lw sp, 0(sp)
        call __start;
    "#,
        sym STACK_TOP
    );
}
