//! No-op `critical-section` implementation for the openvm guest.
//!
//! The guest is single-threaded and has no interrupts, so mutual exclusion is
//! vacuous. Registering this impl lets `portable-atomic` (used by `once_cell`,
//! `spin`, etc. to polyfill CAS on `riscv64im-unknown-none-elf`) satisfy its
//! `critical-section` backend without emitting CSR interrupt-disable
//! instructions that the openvm transpiler can't handle.
//!
//! SAFETY: sound while the guest stays single-threaded; revisit if that changes.

use critical_section::RawRestoreState;

struct SingleThreadedCriticalSection;
critical_section::set_impl!(SingleThreadedCriticalSection);

unsafe impl critical_section::Impl for SingleThreadedCriticalSection {
    unsafe fn acquire() -> RawRestoreState {}
    unsafe fn release(_token: RawRestoreState) {}
}
