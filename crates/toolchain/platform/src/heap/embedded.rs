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
    // SAFETY: _end is a linker symbol marking the end of the data segment
    let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
    if heap_pos > crate::memory::GUEST_MAX_MEM {
        crate::print::println("Not enough memory for heap.");
        crate::rust_rt::terminate::<1>();
    }
    let heap_size: usize = crate::memory::GUEST_MAX_MEM - heap_pos;
    // SAFETY:
    // - heap_pos points to valid memory after data segment (verified above)
    // - heap_size is calculated to fit within GUEST_MAX_MEM bounds
    // - HEAP is initialized once in single-threaded context
    unsafe { HEAP.init(heap_pos, heap_size) }
}
