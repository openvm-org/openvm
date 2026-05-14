//! # OpenVM standard library

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[macro_use]
extern crate alloc;

// always include rust_rt so the memory allocator is enabled
#[cfg(openvm_intrinsics)]
use core::arch::asm;

pub use openvm_platform as platform;
#[cfg(openvm_intrinsics)]
#[allow(unused_imports)]
use openvm_platform::rust_rt;
#[cfg(openvm_intrinsics)]
pub use openvm_riscv_guest::*;

#[cfg(openvm_intrinsics)]
mod getrandom;
pub mod io;
#[cfg(openvm_intrinsics)]
mod pal_abi;
pub mod process;
pub mod serde;

#[cfg(not(openvm_intrinsics))]
pub mod utils;

#[cfg(not(openvm_intrinsics))]
pub mod host;

#[cfg(openvm_intrinsics)]
core::arch::global_asm!(include_str!("memset.s"));
#[cfg(openvm_intrinsics)]
core::arch::global_asm!(include_str!("memcpy.s"));

fn _fault() -> ! {
    #[cfg(openvm_intrinsics)]
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
/// `entry` macro is required to indicate the main function and link it to the entrypoint provided
/// by the `openvm` crate (a custom `_start` defined via `global_asm!`).
///
/// The macro is always required for guest builds (when `openvm_intrinsics` is set), independent of
/// whether the `std` feature is enabled — the custom `_start` replaces the one `std` would
/// otherwise provide.
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
#[cfg(openvm_intrinsics)]
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
/// This macro does nothing when not compiling for the guest target.
#[cfg(not(openvm_intrinsics))]
#[macro_export]
macro_rules! entry {
    ($path:path) => {};
}

#[cfg(openvm_intrinsics)]
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
// Overrides std's default `_start` — the openvm guest has no OS runtime to
// initialize, so we skip std's startup and jump straight into the user's `main`.
#[cfg(openvm_intrinsics)]
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
    #[cfg(openvm_intrinsics)]
    unsafe {
        asm!("/* {0} */", in(reg) (ptr))
    }
    #[cfg(not(openvm_intrinsics))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst)
}

// When std is not linked, register a panic handler here so the user does not
// have to. If std is linked, it will define the panic handler instead. This
// panic handler must not be included.
#[cfg(all(openvm_intrinsics, not(feature = "std")))]
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
