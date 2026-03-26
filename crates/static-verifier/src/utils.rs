#[cfg(debug_assertions)]
thread_local! {
    pub(crate) static DEBUG_ASSERTS_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Suppress debug assertions for the duration of `f`, then restore the previous state.
#[cfg(all(test, debug_assertions))]
pub(crate) fn with_debug_asserts_disabled<R>(f: impl FnOnce() -> R) -> R {
    DEBUG_ASSERTS_ENABLED.with(|cell| {
        let prev = cell.get();
        cell.set(false);
        let result = f();
        cell.set(prev);
        result
    })
}

#[cfg(all(test, not(debug_assertions)))]
pub(crate) fn with_debug_asserts_disabled<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// Like `debug_assert_eq!`, but respects the thread-local disable flag.
#[cfg(debug_assertions)]
macro_rules! guarded_debug_assert_eq {
    ($left:expr, $right:expr $(,)?) => {
        $crate::utils::DEBUG_ASSERTS_ENABLED.with(|cell| {
            if cell.get() {
                assert_eq!($left, $right);
            }
        });
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        $crate::utils::DEBUG_ASSERTS_ENABLED.with(|cell| {
            if cell.get() {
                assert_eq!($left, $right, $($arg)+);
            }
        });
    };
}

#[cfg(not(debug_assertions))]
macro_rules! guarded_debug_assert_eq {
    ($($tt:tt)*) => {};
}

/// Like `debug_assert!`, but respects the thread-local disable flag.
#[cfg(debug_assertions)]
macro_rules! guarded_debug_assert {
    ($cond:expr $(,)?) => {
        $crate::utils::DEBUG_ASSERTS_ENABLED.with(|cell| {
            if cell.get() {
                assert!($cond);
            }
        });
    };
    ($cond:expr, $($arg:tt)+) => {
        $crate::utils::DEBUG_ASSERTS_ENABLED.with(|cell| {
            if cell.get() {
                assert!($cond, $($arg)+);
            }
        });
    };
}

#[cfg(not(debug_assertions))]
macro_rules! guarded_debug_assert {
    ($($tt:tt)*) => {};
}

pub(crate) use guarded_debug_assert;
pub(crate) use guarded_debug_assert_eq;
