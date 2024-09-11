use p3_baby_bear::BabyBear;

use crate::memory::offline_checker::columns::{
    MemoryReadAuxCols, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols,
};

#[test]
fn test_write_aux_cols_width() {
    type F = BabyBear;

    let disabled = MemoryWriteAuxCols::<1, F>::disabled();

    assert_eq!(
        disabled.flatten().len(),
        MemoryWriteAuxCols::<1, F>::width()
    );

    let disabled = MemoryWriteAuxCols::<4, F>::disabled();
    assert_eq!(
        disabled.flatten().len(),
        MemoryWriteAuxCols::<4, F>::width()
    );
}

#[test]
fn test_read_aux_cols_width() {
    type F = BabyBear;

    let disabled = MemoryReadAuxCols::<1, F>::disabled();
    assert_eq!(disabled.flatten().len(), MemoryReadAuxCols::<1, F>::width());

    let disabled = MemoryReadAuxCols::<4, F>::disabled();
    assert_eq!(disabled.flatten().len(), MemoryReadAuxCols::<4, F>::width());
}

#[test]
fn test_read_or_immediate_aux_cols_width() {
    type F = BabyBear;

    let disabled = MemoryReadOrImmediateAuxCols::<F>::disabled();
    assert_eq!(
        disabled.flatten().len(),
        MemoryReadOrImmediateAuxCols::<F>::width()
    );
}
