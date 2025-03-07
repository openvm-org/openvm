use super::WORD_SIZE;

pub const MEM_BITS: usize = 29;
pub const MEM_SIZE: usize = 1 << MEM_BITS;
pub const GUEST_MIN_MEM: usize = 0x0000_0400;
pub const GUEST_MAX_MEM: usize = MEM_SIZE;

/// Top of stack; stack grows down from this location.
pub const STACK_TOP: u32 = 0x0020_0400;
/// Program (text followed by data and then bss) gets loaded in
/// starting at this location.  HEAP begins right afterwards.
pub const TEXT_START: u32 = 0x0020_0800;

/// Returns whether `addr` is within guest memory bounds.
pub fn is_guest_memory(addr: u32) -> bool {
    GUEST_MIN_MEM <= (addr as usize) && (addr as usize) < GUEST_MAX_MEM
}

/// # Safety
///
/// This function should be safe to call, but clippy complains if it is not marked as `unsafe`.
#[cfg(feature = "rust-runtime")]
#[no_mangle]
pub unsafe extern "C" fn sys_alloc_aligned(bytes: usize, align: usize) -> *mut u8 {
    use crate::print::println;

    #[cfg(target_os = "zkvm")]
    extern "C" {
        // This symbol is defined by the loader and marks the end
        // of all elf sections, so this is where we start our
        // heap.
        //
        // This is generated automatically by the linker; see
        // https://lld.llvm.org/ELF/linker_script.html#sections-command
        static _end: u8;
    }

    // Pointer to next heap address to use, or 0 if the heap has not yet been
    // initialized.
    static mut HEAP_POS: usize = 0;

    // SAFETY: Single threaded, so nothing else can touch this while we're working.
    let mut heap_pos = unsafe { HEAP_POS };

    #[cfg(target_os = "zkvm")]
    if heap_pos == 0 {
        heap_pos = unsafe { (&_end) as *const u8 as usize };
    }

    // Honor requested alignment if larger than word size.
    // Note: align is typically a power of two.
    let align = usize::max(align, WORD_SIZE);

    let offset = heap_pos & (align - 1);
    if offset != 0 {
        heap_pos += align - offset;
    }

    match heap_pos.checked_add(bytes) {
        Some(new_heap_pos) if new_heap_pos <= GUEST_MAX_MEM => {
            // SAFETY: Single threaded, and non-preemptive so modification is safe.
            unsafe { HEAP_POS = new_heap_pos };
        }
        _ => {
            println("ERROR: Maximum memory exceeded, program terminating.");
            super::rust_rt::terminate::<1>();
        }
    }
    heap_pos as *mut u8
}
