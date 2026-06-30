use super::PAGE_SIZE;

/// Tracks which fixed-size pages of an address space's linear memory may contain non-zero data,
/// for the GPU host-to-device transfer. Pages that are *not* marked are guaranteed zero and are
/// skipped by the transport (which zero-fills the device buffer first).
/// Pages are [`PAGE_SIZE`] bytes, matching the mmap page size.
///
/// A freshly constructed set is empty: callers must [`mark_byte_range`](Self::mark_byte_range)
/// every page they write before the memory is transferred. Unmarked pages are transferred as
/// zero. `bits` is a little-endian bitset over pages: bit `i` set means page `i` may be non-zero.
#[derive(Debug, Clone)]
pub struct TouchedPages {
    bits: Box<[u64]>,
    num_pages: usize,
}

impl TouchedPages {
    /// No pages marked, with capacity for an address space of `num_bytes` bytes.
    #[inline]
    pub fn new(num_bytes: usize) -> Self {
        let num_pages = num_bytes.div_ceil(PAGE_SIZE);
        TouchedPages {
            bits: vec![0u64; num_pages.div_ceil(u64::BITS as usize)].into_boxed_slice(),
            num_pages,
        }
    }

    /// Marks every page overlapping the byte range `[start, start + len)` as possibly non-zero.
    #[inline]
    pub fn mark_byte_range(&mut self, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let first = start / PAGE_SIZE;
        let last = (start + len - 1) / PAGE_SIZE;
        debug_assert!(
            last < self.num_pages,
            "byte range out of address space bounds"
        );

        // Set whole `u64` words at once: each word covers 64 pages, so the fully-covered middle
        // becomes a single `memset` instead of one OR per page.
        let first_word = first / 64;
        let last_word = last / 64;

        if first_word == last_word {
            // All pages land in one word: set bits [first%64, last%64].
            let lo = first % 64;
            let hi = last % 64;
            self.bits[first_word] |= ((!0u64) >> (63 - (hi - lo))) << lo;
            return;
        }

        // Partial first word: bits [first%64, 63].
        self.bits[first_word] |= (!0u64) << (first % 64);
        // Fully-covered middle words: memset to all-ones.
        self.bits[first_word + 1..last_word].fill(!0u64);
        // Partial last word: bits [0, last%64].
        self.bits[last_word] |= (!0u64) >> (63 - (last % 64));
    }

    /// Yields the half-open **byte** ranges `[start, end)` of maximal runs of consecutive marked
    /// pages, clamped to `total_bytes`. Coalescing adjacent pages into runs minimizes the number
    /// of `cudaMemcpyAsync` calls.
    pub fn touched_byte_ranges(&self, total_bytes: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let is_set = |page: usize| self.bits[page / 64] >> (page % 64) & 1 == 1;
        let n = self.num_pages.min(total_bytes.div_ceil(PAGE_SIZE));
        let mut page = 0;
        while page < n {
            if is_set(page) {
                let run_start = page;
                while page < n && is_set(page) {
                    page += 1;
                }
                // Clamp the last run's end to total_bytes.
                runs.push((run_start * PAGE_SIZE, (page * PAGE_SIZE).min(total_bytes)));
            } else {
                page += 1;
            }
        }
        runs
    }
}

#[cfg(test)]
mod tests {
    use super::{TouchedPages, PAGE_SIZE};

    #[test]
    fn new_yields_no_runs() {
        let touched = TouchedPages::new(10 * PAGE_SIZE);
        assert_eq!(touched.touched_byte_ranges(10 * PAGE_SIZE), vec![]);
    }

    #[test]
    fn marking_full_range_yields_single_run() {
        let mut touched = TouchedPages::new(10 * PAGE_SIZE);
        touched.mark_byte_range(0, 10 * PAGE_SIZE);
        assert_eq!(
            touched.touched_byte_ranges(10 * PAGE_SIZE),
            vec![(0, 10 * PAGE_SIZE)]
        );
    }

    #[test]
    fn mark_byte_range_marks_overlapping_pages() {
        let mut touched = TouchedPages::new(10 * PAGE_SIZE);
        // A range spanning the boundary between page 1 and page 2 marks both.
        touched.mark_byte_range(2 * PAGE_SIZE - 1, 2);
        assert_eq!(
            touched.touched_byte_ranges(10 * PAGE_SIZE),
            vec![(PAGE_SIZE, 3 * PAGE_SIZE)]
        );
    }

    #[test]
    fn touched_byte_ranges_coalesces_adjacent_pages_and_separates_gaps() {
        let mut touched = TouchedPages::new(10 * PAGE_SIZE);
        // Pages 2, 3, 4 (contiguous) and page 7 (isolated).
        touched.mark_byte_range(2 * PAGE_SIZE, 3 * PAGE_SIZE);
        touched.mark_byte_range(7 * PAGE_SIZE + 100, 1);
        assert_eq!(
            touched.touched_byte_ranges(10 * PAGE_SIZE),
            vec![
                (2 * PAGE_SIZE, 5 * PAGE_SIZE),
                (7 * PAGE_SIZE, 8 * PAGE_SIZE)
            ]
        );
    }

    #[test]
    fn touched_byte_ranges_clamps_last_page_to_total_bytes() {
        // Address space of 3.5 pages; mark the final (partial) page.
        let num_bytes = 3 * PAGE_SIZE + PAGE_SIZE / 2;
        let mut touched = TouchedPages::new(num_bytes);
        touched.mark_byte_range(3 * PAGE_SIZE, 1);
        assert_eq!(
            touched.touched_byte_ranges(num_bytes),
            vec![(3 * PAGE_SIZE, num_bytes)]
        );
    }
}
