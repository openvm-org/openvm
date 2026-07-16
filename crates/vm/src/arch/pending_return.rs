use std::mem::ManuallyDrop;

/// Owns a value which must not be dropped until an asynchronous operation that
/// may reference it has been proven complete.
///
/// Dropping the wrapper intentionally leaks the value. This is the only safe
/// fallback if a cleanup worker exits or cannot prove the operation complete.
/// [`PendingReturn::release`] is the sole path that restores ordinary drop
/// semantics after the caller establishes that the value is idle.
pub(crate) struct PendingReturn<T> {
    value: ManuallyDrop<T>,
}

impl<T> PendingReturn<T> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            value: ManuallyDrop::new(value),
        }
    }

    pub(crate) fn release(mut self) -> T {
        // SAFETY: `self` is consumed, so this value can be taken only once.
        // `ManuallyDrop` prevents the field from being dropped a second time
        // when the wrapper goes out of scope.
        unsafe { ManuallyDrop::take(&mut self.value) }
    }
}

impl<T> Drop for PendingReturn<T> {
    fn drop(&mut self) {
        // Intentionally do not drop `value`: without an explicit release, the
        // asynchronous owner may still access it.
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::PendingReturn;

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct DropCounter;

    impl Drop for DropCounter {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn value_stays_live_until_return_is_explicitly_released() {
        DROPS.store(0, Ordering::SeqCst);

        drop(PendingReturn::new(DropCounter));
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);

        drop(PendingReturn::new(DropCounter).release());
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
    }
}
