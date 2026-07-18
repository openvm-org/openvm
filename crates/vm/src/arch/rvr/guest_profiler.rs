//! Signal-sampled guest call-stack profiling for the RVR backend.
//!
//! Profiling is enabled explicitly for one RVR execution with a
//! [`GuestProfileConfig`](super::GuestProfileConfig). Raw output preserves
//! sample order for conversion to another profile format.

use std::{
    collections::HashMap,
    ffi::CStr,
    marker::PhantomData,
    mem::MaybeUninit,
    path::Path,
    rc::Rc,
    sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU64, AtomicUsize, Ordering},
};

use openvm_instructions::program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC};
use openvm_platform::memory::TEXT_START;
use rvr_state::RvState;

use super::{
    GuestProfileConfig, RawGuestProfile, RawGuestProfileSample, RawNativeFrame, RawNativeModule,
    RvrCompiled, RAW_GUEST_PROFILE_VERSION,
};

const MAX_STACK_DEPTH: usize = 128;
const GUEST_FP_REG: usize = 8;
const JITTER_SCALE: u64 = 1_000_000;
const JITTER_MIN: u64 = JITTER_SCALE / 2;

fn is_valid_guest_pc(pc: u64) -> bool {
    (TEXT_START..=u64::from(MAX_ALLOWED_PC)).contains(&pc)
        && pc.is_multiple_of(u64::from(DEFAULT_PC_STEP))
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StackSample {
    host_rip: u64,
    wall_time_ns: u64,
    cpu_time_ns: u64,
    pcs: [u32; MAX_STACK_DEPTH],
    depth: u16,
    stack_truncated: bool,
}

impl Default for StackSample {
    fn default() -> Self {
        Self {
            host_rip: 0,
            wall_time_ns: 0,
            cpu_time_ns: 0,
            pcs: [0; MAX_STACK_DEPTH],
            depth: 0,
            stack_truncated: false,
        }
    }
}

struct SignalContext {
    state_ptr: *const RvState,
    memory_base: *const u8,
    memory_size: usize,
    samples: *mut StackSample,
    capacity: usize,
    delivered_samples: AtomicUsize,
    write_idx: AtomicUsize,
    dropped_samples: AtomicUsize,
    timer_overruns: AtomicUsize,
    timer_arm_failures: AtomicUsize,
    clock_failures: AtomicUsize,
    handlers_in_flight: AtomicUsize,
    active: AtomicBool,
    owner_tid: i32,
    profile_signal: i32,
    timer_id: libc::timer_t,
    mean_interval_ns: u64,
    rng_state: AtomicU64,
}

// The context and its backing allocations outlive the armed timer. Access is
// restricted to the executing thread and SIGPROF handlers.
unsafe impl Send for SignalContext {}
unsafe impl Sync for SignalContext {}

static HANDLER_CTX: AtomicPtr<SignalContext> = AtomicPtr::new(std::ptr::null_mut());
static OWNER_TID: AtomicI32 = AtomicI32::new(0);
static PROFILE_SIGNAL: AtomicI32 = AtomicI32::new(0);

unsafe fn capture_sample(
    ctx: &SignalContext,
    host_rip: u64,
    wall_time_ns: u64,
    cpu_time_ns: u64,
) -> StackSample {
    // SAFETY: start() requires a live RvState until the timer is disarmed.
    let state = unsafe { &*ctx.state_ptr };
    let mut sample = StackSample {
        host_rip,
        wall_time_ns,
        cpu_time_ns,
        ..Default::default()
    };
    sample.pcs[0] = u32::try_from(state.pc).unwrap_or_default();
    sample.depth = 1;

    let mut fp = state.regs[GUEST_FP_REG];
    while usize::from(sample.depth) < MAX_STACK_DEPTH && fp != 0 {
        // RV64 frame-pointer layout with frame pointers enabled:
        // saved return address at fp - 8, parent frame pointer at fp - 16.
        let Some(ra_addr) = fp
            .checked_sub(8)
            .and_then(|addr| usize::try_from(addr).ok())
        else {
            break;
        };
        let Some(parent_addr) = fp
            .checked_sub(16)
            .and_then(|addr| usize::try_from(addr).ok())
        else {
            break;
        };
        if ra_addr
            .checked_add(8)
            .is_none_or(|end| end > ctx.memory_size)
            || parent_addr
                .checked_add(8)
                .is_none_or(|end| end > ctx.memory_size)
        {
            break;
        }

        // SAFETY: both eight-byte reads were bounds checked above.
        let ra = unsafe { (ctx.memory_base.add(ra_addr) as *const u64).read_unaligned() };
        // SAFETY: both eight-byte reads were bounds checked above.
        let parent_fp =
            unsafe { (ctx.memory_base.add(parent_addr) as *const u64).read_unaligned() };
        // A zero parent marks the ABI root frame. Its saved-RA slot is not a
        // caller and may contain stale stack data that happens to look like a
        // valid guest PC. Validate the chain before recording this frame.
        if parent_fp == 0 || parent_fp <= fp {
            break;
        }
        // A frame pointer may temporarily contain ordinary guest data in code
        // that does not establish a frame. Stop before recording data as a
        // return address: OpenVM PCs are aligned and live in the program-PC
        // range.
        if !is_valid_guest_pc(ra) {
            break;
        }
        sample.pcs[usize::from(sample.depth)] = u32::try_from(ra).unwrap_or_default();
        sample.depth += 1;
        fp = parent_fp;
    }
    sample.stack_truncated = usize::from(sample.depth) == MAX_STACK_DEPTH && fp != 0;
    sample
}

fn current_tid() -> i32 {
    // SAFETY: gettid takes no pointer arguments and cannot outlive local state.
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
}

fn next_interval_ns(ctx: &SignalContext) -> u64 {
    let mut current = ctx.rng_state.load(Ordering::Relaxed);
    loop {
        let mut next = current;
        next ^= next << 13;
        next ^= next >> 7;
        next ^= next << 17;
        match ctx.rng_state.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let jitter = JITTER_MIN + next % (JITTER_SCALE + 1);
                return ctx
                    .mean_interval_ns
                    .saturating_mul(jitter)
                    .div_ceil(JITTER_SCALE)
                    .max(1);
            }
            Err(observed) => current = observed,
        }
    }
}

fn arm_one_shot(timer_id: libc::timer_t, interval_ns: u64) -> libc::c_int {
    let timer = libc::itimerspec {
        it_interval: libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        it_value: libc::timespec {
            tv_sec: (interval_ns / 1_000_000_000) as libc::time_t,
            tv_nsec: (interval_ns % 1_000_000_000) as libc::c_long,
        },
    };
    // SAFETY: timer_id was returned by timer_create and timer is initialized.
    unsafe { libc::timer_settime(timer_id, 0, &timer, std::ptr::null_mut()) }
}

struct ErrnoGuard(libc::c_int);

impl ErrnoGuard {
    fn new() -> Self {
        // SAFETY: Linux exposes errno as thread-local storage through this
        // function. Profiling is compiled only on the supported Linux target.
        Self(unsafe { *libc::__errno_location() })
    }
}

impl Drop for ErrnoGuard {
    fn drop(&mut self) {
        // SAFETY: restore the interrupted thread's errno before sigreturn.
        unsafe { *libc::__errno_location() = self.0 };
    }
}

extern "C" fn sigprof_handler(
    _signal: libc::c_int,
    _info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let _errno = ErrnoGuard::new();
    if _signal != PROFILE_SIGNAL.load(Ordering::Acquire)
        || current_tid() != OWNER_TID.load(Ordering::Acquire)
    {
        return;
    }
    let ctx_ptr = HANDLER_CTX.load(Ordering::Acquire);
    if ctx_ptr.is_null() {
        return;
    }
    // SAFETY: the global pointer is cleared only after the timer is disarmed.
    let ctx = unsafe { &*ctx_ptr };
    ctx.handlers_in_flight.fetch_add(1, Ordering::Acquire);
    if !ctx.active.load(Ordering::Acquire) {
        ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
        return;
    }
    if _info.is_null() || unsafe { (*_info).si_code } != libc::SI_TIMER {
        ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
        return;
    }
    if unsafe { (*_info).si_value().sival_ptr } != ctx_ptr.cast::<libc::c_void>() {
        ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
        return;
    }
    ctx.delivered_samples.fetch_add(1, Ordering::Relaxed);

    // Rearm before capture so the requested wall-clock cadence includes the
    // time spent sampling. Overruns make an unsustainable rate explicit.
    let overruns = unsafe { libc::timer_getoverrun(ctx.timer_id) };
    if overruns > 0 {
        ctx.timer_overruns
            .fetch_add(overruns as usize, Ordering::Relaxed);
    }
    if ctx.active.load(Ordering::Acquire) && arm_one_shot(ctx.timer_id, next_interval_ns(ctx)) != 0
    {
        ctx.timer_arm_failures.fetch_add(1, Ordering::Relaxed);
    }
    if ctx.write_idx.load(Ordering::Relaxed) >= ctx.capacity {
        ctx.dropped_samples.fetch_add(1, Ordering::Relaxed);
        ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
        return;
    }

    let host_rip = if ucontext.is_null() {
        0
    } else {
        // SAFETY: SA_SIGINFO supplies a live ucontext for this handler call.
        let context = unsafe { &*(ucontext as *const libc::ucontext_t) };
        context.uc_mcontext.gregs[libc::REG_RIP as usize] as u64
    };
    let wall_time_ns = clock_time_ns(libc::CLOCK_MONOTONIC);
    let cpu_time_ns = clock_time_ns(libc::CLOCK_THREAD_CPUTIME_ID);
    if wall_time_ns == 0 || cpu_time_ns == 0 {
        ctx.clock_failures.fetch_add(1, Ordering::Relaxed);
        ctx.dropped_samples.fetch_add(1, Ordering::Relaxed);
        ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
        return;
    }
    // SAFETY: the active context owns live state and guest-memory pointers.
    let sample = unsafe { capture_sample(ctx, host_rip, wall_time_ns, cpu_time_ns) };
    let idx = ctx.write_idx.fetch_add(1, Ordering::Relaxed);
    if idx < ctx.capacity {
        // SAFETY: every accepted index is unique and inside the fixed buffer.
        unsafe { ctx.samples.add(idx).write(sample) };
    } else {
        ctx.dropped_samples.fetch_add(1, Ordering::Relaxed);
    }
    ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
}

#[inline]
fn clock_time_ns(clock: libc::clockid_t) -> u64 {
    // clock_gettime is async-signal-safe on supported Linux targets.
    let mut time: libc::timespec = unsafe { std::mem::zeroed() };
    // SAFETY: `time` points to initialized writable storage.
    if unsafe { libc::clock_gettime(clock, &mut time) } != 0 {
        return 0;
    }
    (time.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(time.tv_nsec as u64)
}

pub(super) struct GuestProfiler {
    config: GuestProfileConfig,
    buffer: Vec<MaybeUninit<StackSample>>,
    ctx: Box<SignalContext>,
    old_action: libc::sigaction,
    start_unix_time_ns: u64,
    start_wall_time_ns: u64,
    start_cpu_time_ns: u64,
    end_wall_time_ns: u64,
    end_cpu_time_ns: u64,
    started: bool,
    _not_send: PhantomData<Rc<()>>,
}

fn selected_profile_signal() -> Result<i32, String> {
    let signal = libc::SIGRTMIN() + 8;
    if signal > libc::SIGRTMAX() {
        return Err("no dedicated real-time signal is available for RVR profiling".to_string());
    }
    Ok(signal)
}

fn signal_set(signal: i32) -> libc::sigset_t {
    // SAFETY: sigset_t is valid when initialized through sigemptyset/sigaddset.
    let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, signal);
    }
    set
}

fn restore_signal_state(signal: i32, old_action: &libc::sigaction) {
    let set = signal_set(signal);
    // SAFETY: the action was captured when ours was installed. Only our
    // dedicated signal is unblocked, preserving unrelated mask changes made
    // by the execution.
    unsafe {
        libc::sigaction(signal, old_action, std::ptr::null_mut());
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
    }
}

impl GuestProfiler {
    /// # Safety
    ///
    /// `state` and its guest memory must stay at stable addresses until the
    /// returned profiler is finished or dropped on this same thread.
    pub(super) unsafe fn start<ModeState>(
        state: &RvState<ModeState>,
        config: &GuestProfileConfig,
    ) -> Result<Self, String> {
        if state.memory.is_null() {
            return Err("cannot profile RVR execution with null guest memory".to_string());
        }

        let owner_tid = current_tid();
        let profile_signal = selected_profile_signal()?;
        let set = signal_set(profile_signal);
        // SAFETY: old_mask points to writable storage and set is initialized.
        let mut old_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        if unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, &mut old_mask) } != 0 {
            return Err("failed to block the RVR profiling signal".to_string());
        }
        // Fail closed if the caller already had our dedicated signal blocked.
        if unsafe { libc::sigismember(&old_mask, profile_signal) } == 1 {
            // SAFETY: old_mask was captured above.
            unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut()) };
            return Err(format!(
                "RVR profiling signal {profile_signal} is already blocked on the execution thread"
            ));
        }

        let mut buffer = Vec::<MaybeUninit<StackSample>>::new();
        if let Err(error) = buffer.try_reserve_exact(config.max_samples()) {
            unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut()) };
            return Err(format!(
                "failed to reserve guest profile sample buffer: {error}"
            ));
        }
        // SAFETY: MaybeUninit does not require initialization. The signal
        // handler writes every field before an accepted slot is read.
        unsafe { buffer.set_len(config.max_samples()) };
        let mut ctx = Box::new(SignalContext {
            // `RvState` is `repr(C)` and all mode-specific state follows the
            // register, PC, and memory prefix sampled by the signal handler.
            state_ptr: std::ptr::from_ref(state).cast::<RvState>(),
            memory_base: state.memory.cast_const(),
            memory_size: openvm_platform::memory::MEM_SIZE,
            samples: buffer.as_mut_ptr().cast::<StackSample>(),
            capacity: buffer.len(),
            delivered_samples: AtomicUsize::new(0),
            write_idx: AtomicUsize::new(0),
            dropped_samples: AtomicUsize::new(0),
            timer_overruns: AtomicUsize::new(0),
            timer_arm_failures: AtomicUsize::new(0),
            clock_failures: AtomicUsize::new(0),
            handlers_in_flight: AtomicUsize::new(0),
            active: AtomicBool::new(false),
            owner_tid,
            profile_signal,
            timer_id: unsafe { std::mem::zeroed() },
            mean_interval_ns: 1_000_000_000u64.div_ceil(u64::from(config.sample_hz())),
            rng_state: AtomicU64::new(
                (clock_time_ns(libc::CLOCK_MONOTONIC) ^ (owner_tid as u64).rotate_left(17)).max(1),
            ),
        });

        let ctx_ptr = std::ptr::from_mut::<SignalContext>(&mut *ctx);
        HANDLER_CTX
            .compare_exchange(
                std::ptr::null_mut(),
                ctx_ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| {
                unsafe {
                    libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut());
                }
                "another RVR guest profiler is already active".to_string()
            })?;
        OWNER_TID.store(owner_tid, Ordering::Release);
        PROFILE_SIGNAL.store(profile_signal, Ordering::Release);

        // Atomically install our disposition and inspect the one it replaced.
        // This closes the query/install race with unrelated libraries.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut old_action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = sigprof_handler as *const () as usize;
        action.sa_flags = libc::SA_RESTART | libc::SA_SIGINFO;
        unsafe { libc::sigemptyset(&mut action.sa_mask) };
        if unsafe { libc::sigaction(profile_signal, &action, &mut old_action) } != 0 {
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            OWNER_TID.store(0, Ordering::Release);
            PROFILE_SIGNAL.store(0, Ordering::Release);
            unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut()) };
            return Err(format!(
                "failed to install RVR profiling signal handler: {}",
                std::io::Error::last_os_error()
            ));
        }
        if old_action.sa_sigaction != libc::SIG_DFL {
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            OWNER_TID.store(0, Ordering::Release);
            PROFILE_SIGNAL.store(0, Ordering::Release);
            restore_signal_state(profile_signal, &old_action);
            return Err(format!(
                "RVR profiling signal {profile_signal} already has a process handler"
            ));
        }

        let mut event: libc::sigevent = unsafe { std::mem::zeroed() };
        event.sigev_notify = libc::SIGEV_THREAD_ID;
        event.sigev_signo = profile_signal;
        event.sigev_value.sival_ptr = ctx_ptr.cast::<libc::c_void>();
        event.sigev_notify_thread_id = owner_tid;
        let mut timer_id: libc::timer_t = unsafe { std::mem::zeroed() };
        if unsafe { libc::timer_create(libc::CLOCK_MONOTONIC, &mut event, &mut timer_id) } != 0 {
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            OWNER_TID.store(0, Ordering::Release);
            PROFILE_SIGNAL.store(0, Ordering::Release);
            restore_signal_state(profile_signal, &old_action);
            return Err(format!(
                "failed to create thread-targeted RVR profiling timer: {}",
                std::io::Error::last_os_error()
            ));
        }
        ctx.timer_id = timer_id;
        let start_unix_time_ns = clock_time_ns(libc::CLOCK_REALTIME);
        let start_wall_time_ns = clock_time_ns(libc::CLOCK_MONOTONIC);
        let start_cpu_time_ns = clock_time_ns(libc::CLOCK_THREAD_CPUTIME_ID);
        ctx.active.store(true, Ordering::Release);
        if arm_one_shot(timer_id, next_interval_ns(&ctx)) != 0 {
            let error = std::io::Error::last_os_error();
            ctx.active.store(false, Ordering::Release);
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            OWNER_TID.store(0, Ordering::Release);
            PROFILE_SIGNAL.store(0, Ordering::Release);
            if unsafe { libc::timer_delete(timer_id) } != 0 {
                std::process::abort();
            }
            if drain_pending_signal(&set).is_err() {
                std::process::abort();
            }
            restore_signal_state(profile_signal, &old_action);
            return Err(format!("failed to arm RVR profiling timer: {error}"));
        }
        if unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut()) } != 0
        {
            ctx.active.store(false, Ordering::Release);
            if unsafe { libc::timer_delete(timer_id) } != 0 {
                std::process::abort();
            }
            if drain_pending_signal(&set).is_err() {
                std::process::abort();
            }
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            OWNER_TID.store(0, Ordering::Release);
            PROFILE_SIGNAL.store(0, Ordering::Release);
            restore_signal_state(profile_signal, &old_action);
            return Err("failed to unblock the RVR profiling signal".to_string());
        }

        Ok(Self {
            config: config.clone(),
            buffer,
            ctx,
            old_action,
            start_unix_time_ns,
            start_wall_time_ns,
            start_cpu_time_ns,
            end_wall_time_ns: 0,
            end_cpu_time_ns: 0,
            started: true,
            _not_send: PhantomData,
        })
    }

    pub(super) fn finish(mut self, compiled: &RvrCompiled) -> Result<(), String> {
        if let Err(error) = self.stop_sampling() {
            // A timer cleanup failure leaves handler-visible storage live.
            // Leak it rather than allowing Drop to free memory a kernel timer
            // may still reference.
            std::mem::forget(self);
            return Err(error);
        }
        let samples = self.samples();
        let generated_module = native_library_module(compiled)?;
        let output = emit_raw_profile(&samples, &generated_module, &self)?;
        std::fs::write(self.config.output(), output).map_err(|error| {
            format!(
                "failed to write guest profile {}: {error}",
                self.config.output().display()
            )
        })?;
        if let Some(output) = self.config.native_artifact_output() {
            compiled
                .save_artifact(output)
                .map_err(|error| format!("failed to preserve native profile artifact: {error}"))?;
        }

        let with_stack = samples.iter().filter(|sample| sample.depth > 1).count();
        let max_depth = samples.iter().map(|sample| sample.depth).max().unwrap_or(0);
        let total_depth: usize = samples.iter().map(|sample| usize::from(sample.depth)).sum();
        let average_depth = if samples.is_empty() {
            0.0
        } else {
            total_depth as f64 / samples.len() as f64
        };
        let delivered = self.ctx.delivered_samples.load(Ordering::Acquire);
        let dropped = self.ctx.dropped_samples.load(Ordering::Acquire);
        let overruns = self.ctx.timer_overruns.load(Ordering::Acquire);
        let arm_failures = self.ctx.timer_arm_failures.load(Ordering::Acquire);
        let clock_failures = self.ctx.clock_failures.load(Ordering::Acquire);
        let elapsed_ns = self
            .end_wall_time_ns
            .saturating_sub(self.start_wall_time_ns);
        let effective_hz = if elapsed_ns == 0 {
            0.0
        } else {
            delivered as f64 * 1_000_000_000.0 / elapsed_ns as f64
        };
        let truncated = samples
            .iter()
            .filter(|sample| sample.stack_truncated)
            .count();
        eprintln!(
            "[rvr-openvm] guest call profile: retained={}, delivered={delivered}, dropped={dropped}, overruns={overruns}, arm_failures={arm_failures}, clock_failures={clock_failures}, effective_hz={effective_hz:.1}, with_stack={}, truncated={truncated}, avg_depth={average_depth:.2}, max_depth={max_depth}, output={}",
            samples.len(),
            with_stack,
            self.config.output().display()
        );
        if dropped != 0 || overruns != 0 || arm_failures != 0 || clock_failures != 0 {
            return Err("RVR guest profile is incomplete; inspect the emitted profile diagnostics and lower the sampling rate".to_string());
        }
        Ok(())
    }

    fn stop_sampling(&mut self) -> Result<(), String> {
        if !self.started {
            return Ok(());
        }
        if current_tid() != self.ctx.owner_tid {
            return Err("RVR guest profiler must be finished on its execution thread".to_string());
        }

        let set = signal_set(self.ctx.profile_signal);
        if unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) } != 0 {
            return Err("failed to block the RVR profiling signal during teardown".to_string());
        }
        self.end_wall_time_ns = clock_time_ns(libc::CLOCK_MONOTONIC);
        self.end_cpu_time_ns = clock_time_ns(libc::CLOCK_THREAD_CPUTIME_ID);
        self.ctx.active.store(false, Ordering::Release);
        let zero: libc::itimerspec = unsafe { std::mem::zeroed() };
        let disarm_result =
            unsafe { libc::timer_settime(self.ctx.timer_id, 0, &zero, std::ptr::null_mut()) };
        let disarm_error = (disarm_result != 0).then(std::io::Error::last_os_error);
        if unsafe { libc::timer_delete(self.ctx.timer_id) } != 0 {
            return Err(format!(
                "failed to delete RVR profiling timer: {}",
                std::io::Error::last_os_error()
            ));
        }
        while self.ctx.handlers_in_flight.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }

        // Consume any timer signal that was queued before deletion while it is
        // still blocked, so it cannot reach the restored disposition.
        if drain_pending_signal(&set).is_err() {
            std::process::abort();
        }
        HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
        OWNER_TID.store(0, Ordering::Release);
        PROFILE_SIGNAL.store(0, Ordering::Release);
        restore_signal_state(self.ctx.profile_signal, &self.old_action);
        self.started = false;
        if let Some(error) = disarm_error {
            return Err(format!(
                "failed to disarm RVR profiling timer before deletion: {error}"
            ));
        }
        Ok(())
    }

    fn samples(&self) -> Vec<StackSample> {
        let count = self
            .ctx
            .write_idx
            .load(Ordering::Acquire)
            .min(self.ctx.capacity);
        (0..count)
            .map(|index| {
                // SAFETY: every retained index below write_idx was completely
                // initialized by the signal handler before sampling stopped.
                unsafe { *self.buffer[index].assume_init_ref() }
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct NativeModuleInfo {
    base: u64,
    path: String,
    name: String,
}

fn native_library_module(compiled: &RvrCompiled) -> Result<NativeModuleInfo, String> {
    // Use an exported function from the loaded artifact as an ASLR anchor.
    let execute: libloading::Symbol<unsafe extern "C" fn()> = unsafe {
        compiled
            .lib
            .get(b"rv_execute")
            .map_err(|error| format!("failed to locate rv_execute for profiling: {error}"))?
    };
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: the function pointer belongs to the loaded library and `info` is writable.
    if unsafe { libc::dladdr(*execute as *const () as *const libc::c_void, &mut info) } == 0
        || info.dli_fbase.is_null()
    {
        return Err("failed to determine native artifact load address".to_string());
    }
    native_module_from_dl_info(&info)
        .ok_or_else(|| "failed to inspect native artifact module".to_string())
}

impl Drop for GuestProfiler {
    fn drop(&mut self) {
        if self.stop_sampling().is_err() {
            std::process::abort();
        }
    }
}

fn drain_pending_signal(set: &libc::sigset_t) -> Result<(), String> {
    let timeout = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    loop {
        let result = unsafe { libc::sigtimedwait(set, std::ptr::null_mut(), &timeout) };
        if result >= 0 {
            continue;
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EAGAIN) => return Ok(()),
            _ => return Err(format!("failed to drain RVR profiling signal: {error}")),
        }
    }
}

fn emit_raw_profile(
    samples: &[StackSample],
    generated_module: &NativeModuleInfo,
    profiler: &GuestProfiler,
) -> Result<String, String> {
    let mut native_modules = vec![RawNativeModule {
        name: generated_module.name.clone(),
        path: generated_module.path.clone(),
        generated: true,
    }];
    let mut module_indices =
        HashMap::from([((generated_module.base, generated_module.path.clone()), 0u32)]);
    let samples = samples
        .iter()
        .map(|sample| {
            let resolved_module = (sample.host_rip != 0)
                .then(|| native_module_for_address(sample.host_rip))
                .flatten();
            let native_leaf = (sample.host_rip != 0).then(|| {
                if let Some(module) = resolved_module {
                    let key = (module.base, module.path.clone());
                    let module_index = *module_indices.entry(key).or_insert_with(|| {
                        let index = u32::try_from(native_modules.len()).unwrap_or(u32::MAX);
                        native_modules.push(RawNativeModule {
                            name: module.name,
                            path: module.path,
                            generated: false,
                        });
                        index
                    });
                    RawNativeFrame {
                        module_index: Some(module_index),
                        pc: sample.host_rip.saturating_sub(module.base),
                    }
                } else {
                    RawNativeFrame {
                        module_index: None,
                        pc: sample.host_rip,
                    }
                }
            });
            // Exact at profiled host-call boundaries, otherwise potentially
            // stale. The converter uses the resolved native symbol/range to
            // include this only for host-helper leaves, including helpers
            // linked into the generated shared object.
            let guest_callsite_pc =
                (native_leaf.is_some() && sample.depth != 0).then_some(u64::from(sample.pcs[0]));
            let guest_return_pcs = sample.pcs[1..usize::from(sample.depth)]
                .iter()
                .rev()
                .map(|&pc| u64::from(pc))
                .collect();
            RawGuestProfileSample {
                wall_time_ns: sample.wall_time_ns,
                cpu_time_ns: sample.cpu_time_ns,
                native_leaf,
                guest_callsite_pc,
                guest_return_pcs,
                stack_truncated: sample.stack_truncated,
            }
        })
        .collect();
    serde_json::to_string(&RawGuestProfile {
        version: RAW_GUEST_PROFILE_VERSION,
        requested_sample_hz: profiler.config.sample_hz(),
        owner_tid: profiler.ctx.owner_tid,
        start_unix_time_ns: profiler.start_unix_time_ns,
        start_wall_time_ns: profiler.start_wall_time_ns,
        end_wall_time_ns: profiler.end_wall_time_ns,
        start_cpu_time_ns: profiler.start_cpu_time_ns,
        end_cpu_time_ns: profiler.end_cpu_time_ns,
        delivered_samples: profiler.ctx.delivered_samples.load(Ordering::Acquire) as u64,
        dropped_samples: profiler.ctx.dropped_samples.load(Ordering::Acquire) as u64,
        timer_overruns: profiler.ctx.timer_overruns.load(Ordering::Acquire) as u64,
        timer_arm_failures: profiler.ctx.timer_arm_failures.load(Ordering::Acquire) as u64,
        clock_failures: profiler.ctx.clock_failures.load(Ordering::Acquire) as u64,
        native_modules,
        samples,
    })
    .map_err(|error| format!("failed to serialize raw guest profile: {error}"))
}

fn native_module_for_address(address: u64) -> Option<NativeModuleInfo> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr only inspects the supplied address and writes `info`.
    if unsafe { libc::dladdr(address as usize as *const libc::c_void, &mut info) } == 0
        || info.dli_fbase.is_null()
    {
        return None;
    }
    native_module_from_dl_info(&info)
}

fn native_module_from_dl_info(info: &libc::Dl_info) -> Option<NativeModuleInfo> {
    if info.dli_fbase.is_null() || info.dli_fname.is_null() {
        return None;
    }
    let path = unsafe { CStr::from_ptr(info.dli_fname) }
        .to_string_lossy()
        .into_owned();
    let name = Path::new(&path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "native-module".to_string());
    Some(NativeModuleInfo {
        base: info.dli_fbase as usize as u64,
        path,
        name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u64(memory: &mut [u8], address: usize, value: u64) {
        memory[address..address + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn test_context(state: &RvState, memory: &[u8]) -> SignalContext {
        SignalContext {
            state_ptr: std::ptr::from_ref(state),
            memory_base: memory.as_ptr(),
            memory_size: memory.len(),
            samples: std::ptr::null_mut(),
            capacity: 0,
            delivered_samples: AtomicUsize::new(0),
            write_idx: AtomicUsize::new(0),
            dropped_samples: AtomicUsize::new(0),
            timer_overruns: AtomicUsize::new(0),
            timer_arm_failures: AtomicUsize::new(0),
            clock_failures: AtomicUsize::new(0),
            handlers_in_flight: AtomicUsize::new(0),
            active: AtomicBool::new(false),
            owner_tid: current_tid(),
            profile_signal: 0,
            timer_id: unsafe { std::mem::zeroed() },
            mean_interval_ns: 1_000_000,
            rng_state: AtomicU64::new(1),
        }
    }

    #[test]
    fn walks_rv64_frame_chain() {
        let mut memory = vec![0u8; 256];
        put_u64(&mut memory, 120, TEXT_START + 0x100);
        put_u64(&mut memory, 112, 192);
        put_u64(&mut memory, 184, TEXT_START + 0x200);
        put_u64(&mut memory, 176, 224);
        put_u64(&mut memory, 216, TEXT_START + 0x300);
        put_u64(&mut memory, 208, 0);
        let mut state = RvState {
            pc: TEXT_START,
            ..Default::default()
        };
        state.regs[GUEST_FP_REG] = 128;
        let ctx = test_context(&state, &memory);

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[
                TEXT_START as u32,
                (TEXT_START + 0x100) as u32,
                (TEXT_START + 0x200) as u32,
            ]
        );
    }

    #[test]
    fn rejects_nonascending_parent_frame() {
        let mut memory = vec![0u8; 256];
        put_u64(&mut memory, 120, TEXT_START + 0x100);
        put_u64(&mut memory, 112, 64);
        let mut state = RvState {
            pc: TEXT_START,
            ..Default::default()
        };
        state.regs[GUEST_FP_REG] = 128;
        let ctx = test_context(&state, &memory);

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[TEXT_START as u32]
        );
    }

    #[test]
    fn ignores_valid_looking_return_address_in_root_frame() {
        let mut memory = vec![0u8; 256];
        put_u64(&mut memory, 120, TEXT_START + 0x100);
        put_u64(&mut memory, 112, 0);
        let mut state = RvState {
            pc: TEXT_START,
            ..Default::default()
        };
        state.regs[GUEST_FP_REG] = 128;
        let ctx = test_context(&state, &memory);

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[TEXT_START as u32]
        );
    }

    #[test]
    fn stops_before_guest_data_masquerading_as_a_return_address() {
        let mut memory = vec![0u8; 256];
        put_u64(&mut memory, 120, 0x20);
        put_u64(&mut memory, 112, 192);
        let mut state = RvState {
            pc: TEXT_START + 0x100,
            ..Default::default()
        };
        state.regs[GUEST_FP_REG] = 128;
        let ctx = test_context(&state, &memory);

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[(TEXT_START + 0x100) as u32]
        );
    }

    #[test]
    fn jitter_stays_within_half_to_one_and_a_half_intervals() {
        let state = RvState::default();
        let memory = vec![0u8; 32];
        let ctx = test_context(&state, &memory);
        for _ in 0..1_000 {
            let interval = next_interval_ns(&ctx);
            assert!((500_000..=1_500_000).contains(&interval));
        }
    }
}
