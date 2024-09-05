use p3_baby_bear::BabyBear;

use crate::memory::offline_checker::{
    bridge::MemoryOfflineChecker,
    bus::MemoryBus,
    columns::{MemoryReadAuxCols, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
};

#[test]
fn test_write_aux_cols_width() {
    type F = BabyBear;

    let mem_oc = MemoryOfflineChecker::new(MemoryBus(1), 29, 16);

    let disabled = MemoryWriteAuxCols::<1, F>::disabled(mem_oc);
    assert_eq!(
        disabled.flatten().len(),
        MemoryWriteAuxCols::<1, F>::width(&mem_oc)
    );

    let disabled = MemoryWriteAuxCols::<4, F>::disabled(mem_oc);
    assert_eq!(
        disabled.flatten().len(),
        MemoryWriteAuxCols::<4, F>::width(&mem_oc)
    );
}

#[test]
fn test_read_aux_cols_width() {
    type F = BabyBear;

    let mem_oc = MemoryOfflineChecker::new(MemoryBus(1), 29, 16);

    let disabled = MemoryReadAuxCols::<1, F>::disabled(mem_oc);
    assert_eq!(
        disabled.flatten().len(),
        MemoryReadAuxCols::<1, F>::width(&mem_oc)
    );

    let disabled = MemoryReadAuxCols::<4, F>::disabled(mem_oc);
    assert_eq!(
        disabled.flatten().len(),
        MemoryReadAuxCols::<4, F>::width(&mem_oc)
    );
}

#[test]
fn test_read_or_immediate_aux_cols_width() {
    type F = BabyBear;

    let mem_oc = MemoryOfflineChecker::new(MemoryBus(1), 29, 16);

    let disabled = MemoryReadOrImmediateAuxCols::<F>::disabled(mem_oc);
    assert_eq!(
        disabled.flatten().len(),
        MemoryReadOrImmediateAuxCols::<F>::width(&mem_oc)
    );
}
