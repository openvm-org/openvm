//! # axVM

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![feature(asm_const)]

extern crate alloc;

pub mod intrinsics;
pub mod io;
pub mod process;
pub mod serde;

#[cfg(target_os = "zkvm")]
use core::arch::asm;

// always include rust_rt so the memory allocator is enabled
#[cfg(target_os = "zkvm")]
#[allow(unused_imports)]
use axvm_platform::rust_rt;

#[cfg(target_os = "zkvm")]
core::arch::global_asm!(include_str!("memset.s"));
#[cfg(target_os = "zkvm")]
core::arch::global_asm!(include_str!("memcpy.s"));

fn _fault() -> ! {
    #[cfg(target_os = "zkvm")]
    unsafe {
        asm!("sw x0, 1(x0)")
    };
    unreachable!();
}

// /// Aborts the guest with the given message.
// pub fn abort(msg: &str) -> ! {
//     // SAFETY: A compliant host should fault when it receives this syscall.
//     // sys_panic will issue an invalid instruction for non-compliant hosts.
//     unsafe {
//         sys_panic(msg.as_ptr(), msg.len());
//     }
// }

/// Used for defining the guest's entrypoint and main function.
///
/// When `#![no_main]` is used, the programs entrypoint and main function is left undefined. The
/// `entry` macro is required to indicate the main function and link it to an entrypoint provided
/// by the RISC Zero SDK.
///
/// When `std` is enabled, the entrypoint will be linked automatically and this macro is not
/// required.
///
/// # Example
///
/// ```ignore
/// #![no_main]
/// #![no_std]
///
/// risc0_zkvm::entry!(main);
///
/// fn main() { }
/// ```
#[macro_export]
macro_rules! entry {
    ($path:path) => {
        // Type check the given path
        const ZKVM_ENTRY: fn() = $path;

        // Include generated main in a module so we don't conflict
        // with any other definitions of "main" in this file.
        mod zkvm_generated_main {
            #[no_mangle]
            fn main() {
                super::ZKVM_ENTRY()
            }
        }
    };
}

#[cfg(target_os = "zkvm")]
#[no_mangle]
unsafe extern "C" fn __start() -> ! {
    #[cfg(feature = "heap-embedded-alloc")]
    axvm_platform::heap::embedded::init();

    {
        extern "C" {
            fn main();
        }
        main()
    }

    process::exit();
}

#[cfg(target_os = "zkvm")]
static STACK_TOP: u32 = axvm_platform::memory::STACK_TOP;

// Entry point; sets up global pointer and stack pointer and passes
// to zkvm_start.  TODO: when asm_const is stabilized, use that here
// instead of defining a symbol and dereferencing it.
#[cfg(target_os = "zkvm")]
core::arch::global_asm!(
    r#"
.section .text._start;
.globl _start;
_start:
    .option push;
    .option norelax;
    la gp, __global_pointer$;
    .option pop;
    la sp, {0};
    lw sp, 0(sp);
    call __start;
"#,
    sym STACK_TOP
);

/// Require that accesses to behind the given pointer before the memory
/// barrier don't get optimized away or reordered to after the memory
/// barrier.
#[allow(unused_variables)]
pub fn memory_barrier<T>(ptr: *const T) {
    // SAFETY: This passes a pointer in, but does nothing with it.
    #[cfg(target_os = "zkvm")]
    unsafe {
        asm!("/* {0} */", in(reg) (ptr))
    }
    #[cfg(not(target_os = "zkvm"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst)
}

// When std is not linked, register a panic handler here so the user does not
// have to. If std is linked, it will define the panic handler instead. This
// panic handler must not be included.
#[cfg(all(target_os = "zkvm", not(feature = "std")))]
#[panic_handler]
fn panic_impl(_panic_info: &core::panic::PanicInfo) -> ! {
    axvm_platform::rust_rt::terminate::<1>();
    unreachable!()
}
