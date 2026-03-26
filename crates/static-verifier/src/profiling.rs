//! Cell-count profiling for the static verifier circuit.
//!
//! When the `cell-profiling` feature is enabled, [`CellProfiler`] tracks advice cell allocations
//! across pipeline stages and can generate flamegraph SVGs via `inferno`.
//!
//! When the feature is disabled, all methods are `#[inline(always)]` no-ops with zero overhead.

#[cfg(feature = "cell-profiling")]
mod enabled {
    use std::{fs::File, io::BufWriter};

    use crate::context_tree::ContextTree;

    pub struct CellProfiler {
        tree: ContextTree,
    }

    impl CellProfiler {
        pub fn new(label: &str, cell_count: usize) -> Self {
            Self {
                tree: ContextTree::with_name(label, cell_count),
            }
        }

        pub fn push(&mut self, name: &str, cell_count: usize) {
            self.tree.push(name, tracing::Level::DEBUG, cell_count);
        }

        pub fn pop(&mut self, cell_count: usize) {
            self.tree.pop(cell_count);
        }

        pub fn print(&self, cell_count: usize) {
            self.tree.print(cell_count);
        }

        pub fn write_flamegraph(&self, path: &str, title: &str, cell_count: usize) {
            let file = File::create(path).expect("Failed to create flamegraph file");
            let mut writer = BufWriter::new(file);
            self.tree
                .write_flamegraph(&mut writer, title, cell_count, false);
        }

        pub fn write_flamegraph_reversed(&self, path: &str, title: &str, cell_count: usize) {
            let file = File::create(path).expect("Failed to create flamegraph file");
            let mut writer = BufWriter::new(file);
            self.tree
                .write_flamegraph(&mut writer, title, cell_count, true);
        }
    }
}

#[cfg(not(feature = "cell-profiling"))]
mod disabled {
    pub struct CellProfiler;

    impl CellProfiler {
        #[inline(always)]
        pub fn new(_label: &str, _cell_count: usize) -> Self {
            Self
        }

        #[inline(always)]
        pub fn push(&mut self, _name: &str, _cell_count: usize) {}

        #[inline(always)]
        pub fn pop(&mut self, _cell_count: usize) {}

        #[inline(always)]
        pub fn print(&self, _cell_count: usize) {}

        #[inline(always)]
        pub fn write_flamegraph(&self, _path: &str, _title: &str, _cell_count: usize) {}

        #[inline(always)]
        pub fn write_flamegraph_reversed(&self, _path: &str, _title: &str, _cell_count: usize) {}
    }
}

#[cfg(not(feature = "cell-profiling"))]
pub use disabled::CellProfiler;
#[cfg(feature = "cell-profiling")]
pub use enabled::CellProfiler;
