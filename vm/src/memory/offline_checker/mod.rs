use afs_primitives::offline_checker::OfflineChecker;

pub mod bridge;
pub mod bus;
pub mod columns;
mod trace;

#[cfg(test)]
mod tests;

// TODO[osama]: to be deleted
pub struct MemoryOfflineChecker {
    pub offline_checker: OfflineChecker,
}

impl MemoryOfflineChecker {
    pub fn air_width(&self) -> usize {
        OfflineChecker::air_width(&self.offline_checker)
    }
}
