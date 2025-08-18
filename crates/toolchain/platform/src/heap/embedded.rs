use critical_section::RawRestoreState;
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
pub static HEAP: Heap = Heap::empty();

struct CriticalSection;
critical_section::set_impl!(CriticalSection);

unsafe impl critical_section::Impl for CriticalSection {
    unsafe fn acquire() -> RawRestoreState {
        // this is a no-op. we're in a single-threaded, non-preemptive context
    }

    unsafe fn release(_token: RawRestoreState) {
        // this is a no-op. we're in a single-threaded, non-preemptive context
    }
}

pub fn init() {
    extern "C" {
        static _end: u8;
    }
    // SAFETY: _end is a valid symbol defined by the linker script that marks the end
    // of the data segment. It's safe to take its address as it's specifically placed
    // by the linker for this purpose.
    let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
    if heap_pos > crate::memory::GUEST_MAX_MEM {
        crate::print::println("Not enough memory for heap.");
        crate::rust_rt::terminate::<1>();
    }
    let heap_size: usize = crate::memory::GUEST_MAX_MEM - heap_pos;
    // SAFETY: heap_pos points to valid memory after the data segment (verified above),
    // and heap_size is calculated to fit within GUEST_MAX_MEM bounds. The HEAP is
    // a global static that's only initialized once in a single-threaded context.
    unsafe { HEAP.init(heap_pos, heap_size) }
}
