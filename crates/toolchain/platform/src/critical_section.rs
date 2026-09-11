//! No-op `critical-section` implementation for the OpenVM guest.
//!
//! The guest is single-threaded and has no interrupts, so mutual exclusion is
//! vacuous. Registering the implementation at the platform level lets no_std
//! dependencies use `critical-section` independently of the selected heap
//! allocator.
//!
//! SAFETY: sound while the guest stays single-threaded; revisit if that changes.
//! A second `set_impl!` in the linked binary fails at link time due to duplicate
//! `_critical_section_1_0_{acquire,release}` symbols, so collisions are loud.

use critical_section::RawRestoreState;

struct SingleThreadedCriticalSection;
critical_section::set_impl!(SingleThreadedCriticalSection);

unsafe impl critical_section::Impl for SingleThreadedCriticalSection {
    unsafe fn acquire() -> RawRestoreState {}
    unsafe fn release(_token: RawRestoreState) {}
}
