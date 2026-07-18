//! Signal-sampled guest call-stack profiling for the RVR backend.
//!
//! Set `OPENVM_RVR_GUEST_CALL_PROFILE` to a folded-stack output path to enable
//! profiling for RVR execution. The optional
//! `OPENVM_RVR_GUEST_CALL_PROFILE_HZ` variable controls the sampling rate and
//! defaults to 1 kHz. `OPENVM_RVR_GUEST_CALL_PROFILE_FORMAT=raw` preserves
//! sample order for conversion to another profile format.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};

use rvr_state::RvState;

const MAX_STACK_DEPTH: usize = 128;
const DEFAULT_BUFFER_CAPACITY: usize = 1 << 18;
const DEFAULT_SAMPLE_HZ: u32 = 1_000;
const GUEST_FP_REG: usize = 8;

#[derive(Clone, Copy)]
enum OutputFormat {
    Folded,
    Raw,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StackSample {
    pcs: [u64; MAX_STACK_DEPTH],
    depth: u16,
}

impl Default for StackSample {
    fn default() -> Self {
        Self {
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

unsafe fn capture_sample(ctx: &SignalContext) -> StackSample {
    // SAFETY: start() requires a live RvState until the timer is disarmed.
    let state = unsafe { &*ctx.state_ptr };
    let mut sample = StackSample::default();
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
        if ra != 0 {
            sample.pcs[usize::from(sample.depth)] = ra;
            sample.depth += 1;
        }

        // RISC-V stacks grow down, so each caller frame pointer must be above
        // the callee's. This also rejects cycles and corrupted chains.
        if parent_fp <= fp {
            break;
        }
        fp = parent_fp;
    }
    sample
}

extern "C" fn sigprof_handler(_signal: libc::c_int) {
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

    // SAFETY: the active context owns live state and guest-memory pointers.
    let sample = unsafe { capture_sample(ctx) };
    let idx = ctx.write_idx.fetch_add(1, Ordering::Relaxed) % ctx.capacity;
    // SAFETY: capacity is nonzero and idx is reduced modulo capacity.
    unsafe { ctx.samples.add(idx).write(sample) };
    ctx.handlers_in_flight.fetch_sub(1, Ordering::Release);
}

pub(super) struct GuestProfiler {
    output: PathBuf,
    output_format: OutputFormat,
    buffer: Vec<StackSample>,
    ctx: Box<SignalContext>,
    old_action: libc::sigaction,
    old_timer: libc::itimerval,
    started: bool,
}

impl GuestProfiler {
    pub(super) fn start_from_env<ModeState>(
        state: &RvState<ModeState>,
    ) -> Result<Option<Self>, String> {
        let Some(output) = std::env::var_os("OPENVM_RVR_GUEST_CALL_PROFILE") else {
            return Ok(None);
        };
        let sample_hz = match std::env::var("OPENVM_RVR_GUEST_CALL_PROFILE_HZ") {
            Ok(value) => value.parse::<u32>().map_err(|error| {
                format!("invalid OPENVM_RVR_GUEST_CALL_PROFILE_HZ={value:?}: {error}")
            })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_SAMPLE_HZ,
            Err(error) => {
                return Err(format!("invalid OPENVM_RVR_GUEST_CALL_PROFILE_HZ: {error}"));
            }
        };
        if sample_hz == 0 || sample_hz > 1_000_000 {
            return Err(format!(
                "OPENVM_RVR_GUEST_CALL_PROFILE_HZ must be in 1..=1000000, got {sample_hz}"
            ));
        }
        let output_format = match std::env::var("OPENVM_RVR_GUEST_CALL_PROFILE_FORMAT") {
            Ok(value) if value == "folded" => OutputFormat::Folded,
            Ok(value) if value == "raw" => OutputFormat::Raw,
            Ok(value) => {
                return Err(format!(
                    "OPENVM_RVR_GUEST_CALL_PROFILE_FORMAT must be `folded` or `raw`, got {value:?}"
                ));
            }
            Err(std::env::VarError::NotPresent) => OutputFormat::Folded,
            Err(error) => {
                return Err(format!(
                    "invalid OPENVM_RVR_GUEST_CALL_PROFILE_FORMAT: {error}"
                ));
            }
        };
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
        action.sa_flags = libc::SA_RESTART;
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

        let interval_us = 1_000_000u64.div_ceil(u64::from(sample_hz));
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

        Ok(Some(Self {
            output: output.into(),
            output_format,
            buffer,
            ctx,
            old_action,
            old_timer,
            started: true,
        }))
    }

    pub(super) fn finish(mut self) -> Result<(), String> {
        self.stop_sampling();
        let samples = self.samples();
        let output = match self.output_format {
            OutputFormat::Folded => emit_folded_stacks(&samples),
            OutputFormat::Raw => emit_raw_stacks(&samples),
        };
        std::fs::write(&self.output, output).map_err(|error| {
            format!(
                "failed to write guest profile {}: {error}",
                self.output.display()
            )
        })?;

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
            self.output.display()
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

fn emit_raw_stacks(samples: &[StackSample]) -> String {
    let mut output = String::new();
    for sample in samples {
        use std::fmt::Write;
        for (idx, pc) in sample.pcs[..usize::from(sample.depth)]
            .iter()
            .rev()
            .enumerate()
        {
            if idx != 0 {
                output.push(';');
            }
            let _ = write!(output, "{pc:#018x}");
        }
        output.push('\n');
    }
    output
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
        put_u64(&mut memory, 120, 0x200);
        put_u64(&mut memory, 112, 192);
        put_u64(&mut memory, 184, 0x300);
        put_u64(&mut memory, 176, 0);
        let mut state = RvState::default();
        state.pc = 0x100;
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
        let sample = unsafe { capture_sample(&ctx) };
        assert_eq!(
            &sample.pcs[..usize::from(sample.depth)],
            &[0x100, 0x200, 0x300]
        );
    }

    #[test]
    fn rejects_nonascending_parent_frame() {
        let mut memory = vec![0u8; 256];
        put_u64(&mut memory, 120, 0x200);
        put_u64(&mut memory, 112, 64);
        let mut state = RvState::default();
        state.pc = 0x100;
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
        let sample = unsafe { capture_sample(&ctx) };
        assert_eq!(&sample.pcs[..usize::from(sample.depth)], &[0x100, 0x200]);
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
    fn emits_raw_stacks_in_sample_order() {
        let mut first = StackSample::default();
        first.pcs[..3].copy_from_slice(&[0x100, 0x200, 0x300]);
        first.depth = 3;
        let mut second = StackSample::default();
        second.pcs[..2].copy_from_slice(&[0x400, 0x500]);
        second.depth = 2;
        assert_eq!(
            emit_raw_stacks(&[first, second]),
            "0x0000000000000300;0x0000000000000200;0x0000000000000100\n\
             0x0000000000000500;0x0000000000000400\n"
        );
    }
}
