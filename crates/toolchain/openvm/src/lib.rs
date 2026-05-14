//! # OpenVM standard library

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[macro_use]
extern crate alloc;

// always include rust_rt so the memory allocator is enabled
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
use core::arch::asm;

pub use openvm_platform as platform;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[allow(unused_imports)]
use openvm_platform::rust_rt;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use openvm_riscv_guest::*;

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
mod getrandom;
pub mod io;
#[cfg(all(feature = "std", target_os = "openvm"))]
pub mod pal_abi;
pub mod process;
pub mod serde;

#[cfg(not(any(openvm_intrinsics, target_os = "openvm")))]
pub mod utils;

#[cfg(not(any(openvm_intrinsics, target_os = "openvm")))]
pub mod host;

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
core::arch::global_asm!(include_str!("memset.s"));
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
core::arch::global_asm!(include_str!("memcpy.s"));

fn _fault() -> ! {
    #[cfg(any(openvm_intrinsics, target_os = "openvm"))]
    unsafe {
        asm!("sd x0, 1(x0)")
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
/// When `#![no_main]` is used, the program's entrypoint and main function are left undefined. The
/// `entry` macro is required to indicate the main function and link it to an entrypoint provided
/// by the `openvm` crate.
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
/// openvm::entry!(main);
///
/// fn main() { }
/// ```
#[cfg(all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")))]
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
/// This macro does nothing. You should name the function `main` so that the normal rust main
/// function setup is used.
#[cfg(any(feature = "std", not(any(openvm_intrinsics, target_os = "openvm"))))]
#[macro_export]
macro_rules! entry {
    ($path:path) => {};
}

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
#[no_mangle]
unsafe extern "C" fn __start() -> ! {
    #[cfg(feature = "heap-embedded-alloc")]
    openvm_platform::heap::embedded::init();

    {
        extern "C" {
            fn main();
        }
        main()
    }

    process::exit();
    unreachable!()
}

// Entry point; sets up global pointer and stack pointer and passes to `__start`.
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
core::arch::global_asm!(
    r#"
.section .text._start;
.globl _start;
_start:
    .option push;
    .option norelax;
    la gp, __global_pointer$;
    .option pop;
    li sp, {STACK_TOP};
    call __start;
"#,
    STACK_TOP = const openvm_platform::memory::STACK_TOP,
);

/// Require that accesses to behind the given pointer before the memory
/// barrier don't get optimized away or reordered to after the memory
/// barrier.
#[allow(unused_variables)]
pub fn memory_barrier<T>(ptr: *const T) {
    // SAFETY: This passes a pointer in, but does nothing with it.
    #[cfg(any(openvm_intrinsics, target_os = "openvm"))]
    unsafe {
        asm!("/* {0} */", in(reg) (ptr))
    }
    #[cfg(not(any(openvm_intrinsics, target_os = "openvm")))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst)
}

// When std is not linked, register a panic handler here so the user does not
// have to. If std is linked, it will define the panic handler instead. This
// panic handler must not be included.
#[cfg(all(any(openvm_intrinsics, target_os = "openvm"), not(feature = "std")))]
#[panic_handler]
fn panic_impl(panic_info: &core::panic::PanicInfo) -> ! {
    use core::fmt::Write;
    let mut writer = crate::io::Writer;
    let _ = write!(writer, "{}\n", panic_info);
    openvm_platform::rust_rt::terminate::<1>();
    unreachable!()
}

// Includes the openvm_init.rs file generated at build time
#[macro_export]
macro_rules! init {
    () => {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/openvm_init.rs"));
    };
    ($name:expr) => {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), concat!("/", $name)));
    };
}
