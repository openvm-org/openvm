use std::{
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Mutex,
    },
    thread::JoinHandle,
    time::Duration,
};

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

/// Message protocol shared by cleanup workers whose inputs must be leaked
/// unless a consumer-specific synchronization step proves them idle.
pub(crate) enum PendingReturnMessage<T> {
    Return(PendingReturn<T>),
    /// Acknowledge after every return queued ahead of this message has been consumed.
    Barrier(mpsc::Sender<()>),
    Shutdown,
}

/// Run the batching and shutdown half of an asynchronous cleanup worker.
/// Receiving shutdown or channel closure with a partial batch drops only the
/// wrappers, which quarantines every underlying value without invoking
/// consumer work.
pub(crate) fn run_pending_return_worker<T>(
    receiver: mpsc::Receiver<PendingReturnMessage<T>>,
    shutting_down: &AtomicBool,
    work_gate: &Mutex<()>,
    idle_window: Duration,
    batch_limit: usize,
    mut consume: impl FnMut(Vec<PendingReturn<T>>, usize),
) {
    let mut batch_idx = 0usize;
    loop {
        let first = match receiver.recv() {
            Ok(PendingReturnMessage::Return(first)) => first,
            Ok(PendingReturnMessage::Barrier(acknowledge)) => {
                let _ = acknowledge.send(());
                continue;
            }
            Ok(PendingReturnMessage::Shutdown) | Err(_) => return,
        };
        let mut batch = vec![first];
        let mut acknowledge = None;
        let mut terminate = false;
        while batch.len() < batch_limit {
            match receiver.recv_timeout(idle_window) {
                Ok(PendingReturnMessage::Return(next)) => batch.push(next),
                Ok(PendingReturnMessage::Barrier(sender)) => {
                    acknowledge = Some(sender);
                    break;
                }
                Ok(PendingReturnMessage::Shutdown) => {
                    terminate = true;
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    terminate = true;
                    break;
                }
            }
        }
        if terminate {
            return;
        }

        let _work = work_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if shutting_down.load(Ordering::Acquire) {
            return;
        }
        consume(batch, batch_idx);
        if let Some(acknowledge) = acknowledge {
            let _ = acknowledge.send(());
        }
        batch_idx += 1;
    }
}

/// Stop a pending-return worker without allowing consumer work to start after
/// the shutdown transition. Existing work finishes behind `work_gate`; queued
/// values remain wrapped and are quarantined when the worker receives the
/// shutdown message.
pub(crate) fn shutdown_pending_return_worker<T>(
    shutting_down: &AtomicBool,
    lifecycle_gate: &Mutex<()>,
    work_gate: &Mutex<()>,
    sender: &Mutex<Option<mpsc::Sender<PendingReturnMessage<T>>>>,
    worker: &Mutex<Option<JoinHandle<()>>>,
) {
    let _lifecycle = lifecycle_gate
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if shutting_down.load(Ordering::Acquire) {
        return;
    }

    let work = work_gate
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    shutting_down.store(true, Ordering::Release);
    drop(work);

    if let Some(sender) = sender
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
    {
        let _ = sender.send(PendingReturnMessage::Shutdown);
        drop(sender);
    }
    if let Some(worker) = worker
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
    {
        let _ = worker.join();
    }
}

/// Intentionally keep a value alive until process exit without invoking any
/// consumer, allocator, or pool teardown path.
pub(crate) fn quarantine<T>(value: T) {
    drop(PendingReturn::new(value));
}

#[cfg(test)]
mod tests {
    use std::{
        process::Command,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc, Mutex,
        },
        time::Duration,
    };

    use super::{
        quarantine, run_pending_return_worker, shutdown_pending_return_worker, PendingReturn,
        PendingReturnMessage,
    };

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

    #[test]
    fn barrier_waits_for_prior_returns_but_not_later_returns() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let (acknowledge, acknowledged) = std::sync::mpsc::channel();
        let (later_started, observe_later_started) = std::sync::mpsc::channel();
        let (release_later, wait_for_release) = std::sync::mpsc::channel();
        let shutdown = AtomicBool::new(false);
        let work_gate = Mutex::new(());
        let processed = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let worker_shutdown = &shutdown;
            let worker_gate = &work_gate;
            let worker_processed = &processed;
            let worker = scope.spawn(move || {
                run_pending_return_worker(
                    receiver,
                    worker_shutdown,
                    worker_gate,
                    Duration::from_millis(1),
                    64,
                    |batch, _| {
                        for returned in batch {
                            let value = returned.release();
                            worker_processed.fetch_add(1, Ordering::SeqCst);
                            if value == 2 {
                                later_started.send(()).unwrap();
                                wait_for_release.recv().unwrap();
                            }
                        }
                    },
                );
            });

            sender
                .send(PendingReturnMessage::Return(PendingReturn::new(1)))
                .unwrap();
            sender
                .send(PendingReturnMessage::Barrier(acknowledge))
                .unwrap();
            sender
                .send(PendingReturnMessage::Return(PendingReturn::new(2)))
                .unwrap();

            acknowledged.recv_timeout(Duration::from_secs(1)).unwrap();
            assert!(processed.load(Ordering::SeqCst) >= 1);
            observe_later_started
                .recv_timeout(Duration::from_secs(1))
                .unwrap();
            release_later.send(()).unwrap();
            sender.send(PendingReturnMessage::Shutdown).unwrap();
            worker.join().unwrap();
        });
    }

    struct ReserveOwner {
        unused_registered: Option<DropCounter>,
        unused_fresh: Option<DropCounter>,
    }

    impl Drop for ReserveOwner {
        fn drop(&mut self) {
            quarantine(
                self.unused_registered
                    .take()
                    .expect("registered reserve entry already taken"),
            );
            drop(
                self.unused_fresh
                    .take()
                    .expect("fresh reserve entry already taken"),
            );
        }
    }

    const TEARDOWN_CHILD: &str = "OPENVM_PENDING_RETURN_TEARDOWN_TEST_CHILD";

    #[test]
    fn teardown_quarantines_pending_returns_and_unused_reserve() {
        if std::env::var_os(TEARDOWN_CHILD).is_some() {
            DROPS.store(0, Ordering::SeqCst);
            let (sender, receiver) = std::sync::mpsc::channel();
            let shutdown = Arc::new(AtomicBool::new(false));
            let lifecycle_gate = Mutex::new(());
            let work_gate = Arc::new(Mutex::new(()));
            let cleanup_calls = Arc::new(AtomicUsize::new(0));

            let worker_shutdown = Arc::clone(&shutdown);
            let worker_gate = Arc::clone(&work_gate);
            let worker_calls = Arc::clone(&cleanup_calls);
            let worker = std::thread::spawn(move || {
                run_pending_return_worker(
                    receiver,
                    &worker_shutdown,
                    &worker_gate,
                    Duration::from_secs(30),
                    64,
                    |_, _| {
                        worker_calls.fetch_add(1, Ordering::SeqCst);
                    },
                );
            });
            let sender = Mutex::new(Some(sender));
            let worker = Mutex::new(Some(worker));
            sender
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .send(PendingReturnMessage::Return(PendingReturn::new(
                    DropCounter,
                )))
                .unwrap();

            // Simulate the executor/pool owner dropping while an ordinary
            // return is still queued. A registered unused reserve must not be
            // sent to the worker or freed; a fresh entry can be freed without
            // touching consumer state.
            drop(ReserveOwner {
                unused_registered: Some(DropCounter),
                unused_fresh: Some(DropCounter),
            });
            shutdown_pending_return_worker(
                &shutdown,
                &lifecycle_gate,
                &work_gate,
                &sender,
                &worker,
            );
            assert_eq!(cleanup_calls.load(Ordering::SeqCst), 0);
            assert_eq!(DROPS.load(Ordering::SeqCst), 1);
            return;
        }

        let test_name = std::thread::current()
            .name()
            .expect("test thread has no name")
            .to_owned();
        let status =
            Command::new(std::env::current_exe().expect("test executable path unavailable"))
                .args(["--exact", &test_name, "--nocapture"])
                .env(TEARDOWN_CHILD, "1")
                .status()
                .expect("failed to start teardown regression child");
        assert!(
            status.success(),
            "teardown regression child failed: {status}"
        );
    }
}
