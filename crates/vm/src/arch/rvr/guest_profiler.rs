//! Signal-sampled guest call-stack profiling for the RVR backend.
//!
//! Profiling is enabled explicitly for one RVR execution with a
//! [`GuestProfileConfig`](super::GuestProfileConfig). Raw output preserves
//! sample order for conversion to another profile format.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};

use openvm_instructions::program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC};
use openvm_platform::memory::TEXT_START;
use rvr_state::RvState;

use super::{
    GuestProfileConfig, GuestProfileFormat, RawGuestProfile, RawGuestProfileSample, RvrCompiled,
    RAW_GUEST_PROFILE_VERSION,
};

const MAX_STACK_DEPTH: usize = 128;
const DEFAULT_BUFFER_CAPACITY: usize = 1 << 18;
const GUEST_FP_REG: usize = 8;

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
    pcs: [u64; MAX_STACK_DEPTH],
    depth: u16,
}

impl Default for StackSample {
    fn default() -> Self {
        Self {
            host_rip: 0,
            wall_time_ns: 0,
            cpu_time_ns: 0,
            pcs: [0; MAX_STACK_DEPTH],
            depth: 0,
        }
    }
}

struct SignalContext {
    state_ptr: *const RvState,
    memory_base: *const u8,
    memory_size: usize,
    samples: *mut StackSample,
    capacity: usize,
    write_idx: AtomicUsize,
    handlers_in_flight: AtomicUsize,
    active: AtomicBool,
}

// The context and its backing allocations outlive the armed timer. Access is
// restricted to the executing thread and SIGPROF handlers.
unsafe impl Send for SignalContext {}
unsafe impl Sync for SignalContext {}

static HANDLER_CTX: AtomicPtr<SignalContext> = AtomicPtr::new(std::ptr::null_mut());

unsafe extern "C" {
    fn setitimer(
        which: libc::c_int,
        new_value: *const libc::itimerval,
        old_value: *mut libc::itimerval,
    ) -> libc::c_int;
}

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
    sample.pcs[0] = state.pc;
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
        // A frame pointer may temporarily contain ordinary guest data in code
        // that does not establish a frame. Stop before recording data as a
        // return address: OpenVM PCs are aligned and live in the program-PC
        // range.
        if !is_valid_guest_pc(ra) {
            break;
        }
        sample.pcs[usize::from(sample.depth)] = ra;
        sample.depth += 1;

        // RISC-V stacks grow down, so each caller frame pointer must be above
        // the callee's. This also rejects cycles and corrupted chains.
        if parent_fp == 0 || parent_fp <= fp {
            break;
        }
        fp = parent_fp;
    }
    sample
}

extern "C" fn sigprof_handler(
    _signal: libc::c_int,
    _info: *mut libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
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

    let host_rip = if ucontext.is_null() {
        0
    } else {
        // SAFETY: SA_SIGINFO supplies a live ucontext for this handler call.
        let context = unsafe { &*(ucontext as *const libc::ucontext_t) };
        context.uc_mcontext.gregs[libc::REG_RIP as usize] as u64
    };
    let wall_time_ns = clock_time_ns(libc::CLOCK_MONOTONIC);
    let cpu_time_ns = clock_time_ns(libc::CLOCK_THREAD_CPUTIME_ID);
    // SAFETY: the active context owns live state and guest-memory pointers.
    let sample = unsafe { capture_sample(ctx, host_rip, wall_time_ns, cpu_time_ns) };
    let idx = ctx.write_idx.fetch_add(1, Ordering::Relaxed) % ctx.capacity;
    // SAFETY: capacity is nonzero and idx is reduced modulo capacity.
    unsafe { ctx.samples.add(idx).write(sample) };
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
    buffer: Vec<StackSample>,
    ctx: Box<SignalContext>,
    old_action: libc::sigaction,
    old_timer: libc::itimerval,
    started: bool,
}

impl GuestProfiler {
    pub(super) fn start<ModeState>(
        state: &RvState<ModeState>,
        config: &GuestProfileConfig,
    ) -> Result<Self, String> {
        if state.memory.is_null() {
            return Err("cannot profile RVR execution with null guest memory".to_string());
        }

        let mut buffer = vec![StackSample::default(); DEFAULT_BUFFER_CAPACITY];
        let mut ctx = Box::new(SignalContext {
            // `RvState` is `repr(C)` and all mode-specific state follows the
            // register, PC, and memory prefix sampled by the signal handler.
            state_ptr: std::ptr::from_ref(state).cast::<RvState>(),
            memory_base: state.memory.cast_const(),
            memory_size: openvm_platform::memory::MEM_SIZE,
            samples: buffer.as_mut_ptr(),
            capacity: buffer.len(),
            write_idx: AtomicUsize::new(0),
            handlers_in_flight: AtomicUsize::new(0),
            active: AtomicBool::new(false),
        });

        let ctx_ptr = std::ptr::from_mut::<SignalContext>(&mut *ctx);
        HANDLER_CTX
            .compare_exchange(
                std::ptr::null_mut(),
                ctx_ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| "another RVR guest profiler is already active".to_string())?;

        // SAFETY: sigaction is a plain C struct and all-zero is a valid base.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = sigprof_handler as *const () as usize;
        action.sa_flags = libc::SA_RESTART | libc::SA_SIGINFO;
        // SAFETY: action and old_action are valid sigaction objects.
        unsafe { libc::sigemptyset(&mut action.sa_mask) };
        // SAFETY: an all-zero sigaction is valid storage for the old action.
        let mut old_action: libc::sigaction = unsafe { std::mem::zeroed() };
        // SAFETY: pointers refer to initialized storage for this call.
        if unsafe { libc::sigaction(libc::SIGPROF, &action, &mut old_action) } != 0 {
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            return Err(format!(
                "failed to install SIGPROF handler: {}",
                std::io::Error::last_os_error()
            ));
        }

        let interval_us = 1_000_000u64.div_ceil(u64::from(config.sample_hz()));
        let timer = libc::itimerval {
            it_interval: libc::timeval {
                tv_sec: (interval_us / 1_000_000) as libc::time_t,
                tv_usec: (interval_us % 1_000_000) as libc::suseconds_t,
            },
            it_value: libc::timeval {
                tv_sec: (interval_us / 1_000_000) as libc::time_t,
                tv_usec: (interval_us % 1_000_000) as libc::suseconds_t,
            },
        };
        // SAFETY: an all-zero itimerval is valid storage for the old timer.
        let mut old_timer: libc::itimerval = unsafe { std::mem::zeroed() };
        ctx.active.store(true, Ordering::Release);
        // SAFETY: timer pointers refer to initialized storage.
        if unsafe { setitimer(libc::ITIMER_PROF, &timer, &mut old_timer) } != 0 {
            ctx.active.store(false, Ordering::Release);
            HANDLER_CTX.store(std::ptr::null_mut(), Ordering::Release);
            // SAFETY: restore the handler captured by successful sigaction.
            unsafe { libc::sigaction(libc::SIGPROF, &old_action, std::ptr::null_mut()) };
            return Err(format!(
                "failed to arm ITIMER_PROF: {}",
                std::io::Error::last_os_error()
            ));
        }

        Ok(Self {
            config: config.clone(),
            buffer,
            ctx,
            old_action,
            old_timer,
            started: true,
        })
    }

    pub(super) fn finish(mut self, compiled: &RvrCompiled) -> Result<(), String> {
        self.stop_sampling();
        let samples = self.samples();
        let native_base = native_library_base(compiled)?;
        let output = match self.config.format() {
            GuestProfileFormat::Folded => emit_folded_stacks(&samples),
            GuestProfileFormat::Raw => emit_raw_profile(&samples, native_base)?,
        };
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
        eprintln!(
            "[rvr-openvm] guest call profile: samples={}, with_stack={}, avg_depth={average_depth:.2}, max_depth={max_depth}, output={}",
            samples.len(),
            with_stack,
            self.config.output().display()
        );
        Ok(())
    }

    fn stop_sampling(&mut self) {
        if !self.started {
            return;
        }
        // Stop our timer before invalidating any handler-visible pointers.
        // SAFETY: zero disables ITIMER_PROF.
        let zero: libc::itimerval = unsafe { std::mem::zeroed() };
        // SAFETY: zero points to a valid timer value.
        unsafe { setitimer(libc::ITIMER_PROF, &zero, std::ptr::null_mut()) };
        self.ctx.active.store(false, Ordering::SeqCst);
        HANDLER_CTX.store(std::ptr::null_mut(), Ordering::SeqCst);
        while self.ctx.handlers_in_flight.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        // Restore process-global state even when execution or profile output
        // fails, so later work in this process is unaffected.
        // SAFETY: both values were captured before installing our state.
        unsafe {
            libc::sigaction(libc::SIGPROF, &self.old_action, std::ptr::null_mut());
            setitimer(libc::ITIMER_PROF, &self.old_timer, std::ptr::null_mut());
        }
        self.started = false;
    }

    fn samples(&self) -> Vec<StackSample> {
        let total = self.ctx.write_idx.load(Ordering::Acquire);
        let count = total.min(self.ctx.capacity);
        let start = if total > self.ctx.capacity {
            total % self.ctx.capacity
        } else {
            0
        };
        (0..count)
            .map(|offset| self.buffer[(start + offset) % self.ctx.capacity])
            .collect()
    }
}

fn native_library_base(compiled: &RvrCompiled) -> Result<u64, String> {
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
    Ok(info.dli_fbase as usize as u64)
}

impl Drop for GuestProfiler {
    fn drop(&mut self) {
        self.stop_sampling();
    }
}

fn emit_folded_stacks(samples: &[StackSample]) -> String {
    let mut counts = HashMap::<Vec<u64>, usize>::new();
    for sample in samples {
        let stack = sample.pcs[..usize::from(sample.depth)]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        *counts.entry(stack).or_default() += 1;
    }
    let mut stacks = counts.into_iter().collect::<Vec<_>>();
    stacks.sort_by(|(left_stack, left_count), (right_stack, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_stack.cmp(right_stack))
    });
    let mut output = String::new();
    for (stack, count) in stacks {
        use std::fmt::Write;
        for (idx, pc) in stack.iter().enumerate() {
            if idx != 0 {
                output.push(';');
            }
            let _ = write!(output, "{pc:#018x}");
        }
        let _ = writeln!(output, " {count}");
    }
    output
}

fn emit_raw_profile(samples: &[StackSample], native_base: u64) -> Result<String, String> {
    let samples = samples
        .iter()
        .map(|sample| {
            // Exclude state.pc when a current native IP was captured: state.pc
            // is only updated at selected RVR control-flow boundaries and is
            // not the interrupted instruction. The remaining entries are
            // frame-pointer-derived caller return addresses.
            let first_guest_pc = usize::from(sample.host_rip != 0);
            let guest_pcs = sample.pcs[first_guest_pc..usize::from(sample.depth)]
                .iter()
                .rev()
                .copied()
                .collect();
            RawGuestProfileSample {
                wall_time_ns: sample.wall_time_ns,
                cpu_time_ns: sample.cpu_time_ns,
                host_pc: sample.host_rip.checked_sub(native_base),
                guest_pcs,
            }
        })
        .collect();
    serde_json::to_string(&RawGuestProfile {
        version: RAW_GUEST_PROFILE_VERSION,
        samples,
    })
    .map_err(|error| format!("failed to serialize raw guest profile: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u64(memory: &mut [u8], address: usize, value: u64) {
        memory[address..address + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn walks_rv64_frame_chain() {
        let mut memory = vec![0u8; 256];
        put_u64(&mut memory, 120, TEXT_START + 0x100);
        put_u64(&mut memory, 112, 192);
        put_u64(&mut memory, 184, TEXT_START + 0x200);
        put_u64(&mut memory, 176, 0);
        let mut state = RvState {
            pc: TEXT_START,
            ..Default::default()
        };
        state.regs[GUEST_FP_REG] = 128;
        let ctx = SignalContext {
            state_ptr: std::ptr::from_ref(&state),
            memory_base: memory.as_ptr(),
            memory_size: memory.len(),
            samples: std::ptr::null_mut(),
            capacity: 0,
            write_idx: AtomicUsize::new(0),
            handlers_in_flight: AtomicUsize::new(0),
            active: AtomicBool::new(false),
        };

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[TEXT_START, TEXT_START + 0x100, TEXT_START + 0x200]
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
        let ctx = SignalContext {
            state_ptr: std::ptr::from_ref(&state),
            memory_base: memory.as_ptr(),
            memory_size: memory.len(),
            samples: std::ptr::null_mut(),
            capacity: 0,
            write_idx: AtomicUsize::new(0),
            handlers_in_flight: AtomicUsize::new(0),
            active: AtomicBool::new(false),
        };

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[TEXT_START, TEXT_START + 0x100]
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
        let ctx = SignalContext {
            state_ptr: std::ptr::from_ref(&state),
            memory_base: memory.as_ptr(),
            memory_size: memory.len(),
            samples: std::ptr::null_mut(),
            capacity: 0,
            write_idx: AtomicUsize::new(0),
            handlers_in_flight: AtomicUsize::new(0),
            active: AtomicBool::new(false),
        };

        // SAFETY: state and memory live for the duration of the call.
        let sample = unsafe { capture_sample(&ctx, 0, 10, 8) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[TEXT_START + 0x100]
        );
    }

    #[test]
    fn emits_folded_stacks_bottom_up() {
        let mut sample = StackSample::default();
        sample.pcs[..3].copy_from_slice(&[0x100, 0x200, 0x300]);
        sample.depth = 3;
        let folded = emit_folded_stacks(&[sample, sample]);
        assert_eq!(
            folded,
            "0x0000000000000300;0x0000000000000200;0x0000000000000100 2\n"
        );
    }

    #[test]
    fn emits_versioned_raw_samples_with_native_ip_and_real_clocks() {
        let mut first = StackSample {
            host_rip: 0x10_1234,
            wall_time_ns: 20_000,
            cpu_time_ns: 15_000,
            ..Default::default()
        };
        first.pcs[..3].copy_from_slice(&[0x100, 0x200, 0x300]);
        first.depth = 3;
        let mut second = StackSample {
            wall_time_ns: 30_000,
            cpu_time_ns: 25_000,
            ..Default::default()
        };
        second.pcs[..2].copy_from_slice(&[0x400, 0x500]);
        second.depth = 2;

        let raw: RawGuestProfile =
            serde_json::from_str(&emit_raw_profile(&[first, second], 0x10_0000).unwrap()).unwrap();
        assert_eq!(raw.version, RAW_GUEST_PROFILE_VERSION);
        assert_eq!(raw.samples.len(), 2);
        assert_eq!(raw.samples[0].host_pc, Some(0x1234));
        assert_eq!(raw.samples[0].guest_pcs, vec![0x300, 0x200]);
        assert_eq!(raw.samples[0].wall_time_ns, 20_000);
        assert_eq!(raw.samples[0].cpu_time_ns, 15_000);
        assert_eq!(raw.samples[1].host_pc, None);
        assert_eq!(raw.samples[1].guest_pcs, vec![0x500, 0x400]);
    }
}
