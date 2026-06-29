use super::PAGE_SIZE;

/// Tracks which fixed-size pages of an address space's linear memory may contain non-zero data,
/// for the GPU host-to-device transfer. Pages that are *not* marked are guaranteed zero and can be
/// skipped by the transport (which zero-fills the device buffer first), so this is always a
/// conservative **superset** of the truly non-zero pages: we may copy a still-zero page, but we
/// must never skip a non-zero one. Pages are [`PAGE_SIZE`] bytes, matching the mmap page size.
#[derive(Debug, Clone)]
pub enum TouchedPages {
    /// Conservative default: every page may be non-zero, so the transport copies everything.
    /// Always correct (used when no narrow page information is available).
    All,
    /// Only the marked page indices may be non-zero. `bits` is a little-endian bitset over pages.
    Marked { bits: Vec<u64>, num_pages: usize },
}

impl TouchedPages {
    /// Conservative default: all pages are considered possibly non-zero.
    #[inline]
    pub fn all() -> Self {
        TouchedPages::All
    }

    /// No pages marked, with capacity for an address space of `num_bytes` bytes.
    #[inline]
    pub fn none(num_bytes: usize) -> Self {
        let num_pages = num_bytes.div_ceil(PAGE_SIZE);
        TouchedPages::Marked {
            bits: vec![0u64; num_pages.div_ceil(u64::BITS as usize)],
            num_pages,
        }
    }

    /// Marks every page overlapping the byte range `[start, start + len)` as possibly non-zero.
    /// No-op for [`TouchedPages::All`] (which already covers every page).
    #[inline]
    pub fn mark_byte_range(&mut self, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        if let TouchedPages::Marked { bits, num_pages } = self {
            let first = start / PAGE_SIZE;
            let last = (start + len - 1) / PAGE_SIZE;
            debug_assert!(last < *num_pages, "byte range out of address space bounds");
            for page in first..=last {
                bits[page / 64] |= 1u64 << (page % 64);
            }
        }
    }

    /// Yields the half-open **byte** ranges `[start, end)` of maximal runs of consecutive marked
    /// pages, clamped to `total_bytes`. Coalescing adjacent pages into runs minimizes the number
    /// of `cudaMemcpyAsync` calls. [`TouchedPages::All`] yields a single full run.
    pub fn byte_runs(&self, total_bytes: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        match self {
            TouchedPages::All => {
                if total_bytes > 0 {
                    runs.push((0, total_bytes));
                }
            }
            TouchedPages::Marked { bits, num_pages } => {
                let is_set = |page: usize| bits[page / 64] >> (page % 64) & 1 == 1;
                let mut page = 0;
                while page < *num_pages {
                    if is_set(page) {
                        let run_start = page;
                        while page < *num_pages && is_set(page) {
                            page += 1;
                        }
                        let start_byte = run_start * PAGE_SIZE;
                        let end_byte = (page * PAGE_SIZE).min(total_bytes);
                        if start_byte < end_byte {
                            runs.push((start_byte, end_byte));
                        }
                    } else {
                        page += 1;
                    }
                }
            }
        }
        runs
    }
}

#[cfg(test)]
mod tests {
    use super::{TouchedPages, PAGE_SIZE};

    #[test]
    fn all_yields_single_full_run() {
        let touched = TouchedPages::all();
        assert_eq!(touched.byte_runs(10 * PAGE_SIZE), vec![(0, 10 * PAGE_SIZE)]);
        // Degenerate empty address space.
        assert_eq!(touched.byte_runs(0), vec![]);
    }

    #[test]
    fn none_yields_no_runs() {
        let touched = TouchedPages::none(10 * PAGE_SIZE);
        assert_eq!(touched.byte_runs(10 * PAGE_SIZE), vec![]);
    }

    #[test]
    fn mark_byte_range_marks_overlapping_pages() {
        let mut touched = TouchedPages::none(10 * PAGE_SIZE);
        // A range spanning the boundary between page 1 and page 2 marks both.
        touched.mark_byte_range(2 * PAGE_SIZE - 1, 2);
        assert_eq!(
            touched.byte_runs(10 * PAGE_SIZE),
            vec![(PAGE_SIZE, 3 * PAGE_SIZE)]
        );
    }

    #[test]
    fn byte_runs_coalesces_adjacent_pages_and_separates_gaps() {
        let mut touched = TouchedPages::none(10 * PAGE_SIZE);
        // Pages 2, 3, 4 (contiguous) and page 7 (isolated).
        touched.mark_byte_range(2 * PAGE_SIZE, 3 * PAGE_SIZE);
        touched.mark_byte_range(7 * PAGE_SIZE + 100, 1);
        assert_eq!(
            touched.byte_runs(10 * PAGE_SIZE),
            vec![
                (2 * PAGE_SIZE, 5 * PAGE_SIZE),
                (7 * PAGE_SIZE, 8 * PAGE_SIZE)
            ]
        );
    }

    #[test]
    fn byte_runs_clamps_last_page_to_total_bytes() {
        // Address space of 3.5 pages; mark the final (partial) page.
        let num_bytes = 3 * PAGE_SIZE + PAGE_SIZE / 2;
        let mut touched = TouchedPages::none(num_bytes);
        touched.mark_byte_range(3 * PAGE_SIZE, 1);
        assert_eq!(
            touched.byte_runs(num_bytes),
            vec![(3 * PAGE_SIZE, num_bytes)]
        );
    }

    #[test]
    fn mark_byte_range_is_noop_on_all() {
        let mut touched = TouchedPages::all();
        touched.mark_byte_range(0, PAGE_SIZE);
        assert_eq!(touched.byte_runs(PAGE_SIZE), vec![(0, PAGE_SIZE)]);
    }
}
