use openvm_instructions::DEFERRAL_AS;

use super::*;
use crate::system::memory::{online::PAGE_SIZE, AddressMap};

#[test]
fn derives_a_safe_checkpoint_capacity() {
    let limits = PreflightLimits::new(1025, 0, 512).validated().unwrap();
    assert_eq!(limits.max_checkpoints, 3);
}

#[test]
fn rejects_zero_checkpoint_interval() {
    let error = PreflightLimits::new(1, 0, 0).validated().unwrap_err();
    assert!(error.contains("must be nonzero"));
}

#[test]
fn reused_buffers_grow_to_larger_limits() {
    let initial = PreflightBuffers::new(PreflightLimits::new(128, 8, 64)).unwrap();
    let transcript = PreflightTranscript {
        checkpoints: initial.checkpoints,
        replay_values: initial.replay_values,
    };
    let reused = PreflightBuffers::reuse(PreflightLimits::new(1024, 64, 64), transcript).unwrap();
    assert!(reused.checkpoints.capacity() >= 16);
    assert!(reused.replay_values.capacity() >= 64);
}

#[test]
fn finalization_rejects_reported_values_beyond_limits() {
    let mut retired = PreflightBuffers::new(PreflightLimits::new(8, 4, 4)).unwrap();
    let mut retired_dirty = PreflightDirtyPages::new(&AddressMap::default()).unwrap();
    let mut retired_ffi = retired.ffi_state(&mut retired_dirty);
    retired_ffi.retired = 9;
    let error = unsafe { retired.finish(&retired_ffi, 29, &retired_dirty) }.unwrap_err();
    assert!(error.contains("beyond its 8 instruction limit"));

    let mut buffers = PreflightBuffers::new(PreflightLimits::new(8, 4, 4)).unwrap();
    let mut dirty = PreflightDirtyPages::new(&AddressMap::default()).unwrap();
    let mut ffi = buffers.ffi_state(&mut dirty);
    ffi.replay_value_log_len = 5;
    let error = unsafe { buffers.finish(&ffi, 29, &dirty) }.unwrap_err();
    assert!(error.contains("out-of-bounds length"));
}

#[test]
fn finalization_enforces_timestamp_domain() {
    let mut buffers = PreflightBuffers::new(PreflightLimits::new(8, 0, 4)).unwrap();
    let mut dirty = PreflightDirtyPages::new(&AddressMap::default()).unwrap();
    let mut ffi = buffers.ffi_state(&mut dirty);
    ffi.timestamp = 4;
    let error = unsafe { buffers.finish(&ffi, 2, &dirty) }.unwrap_err();
    assert!(error.contains("outside the configured 2-bit domain"));
}

#[test]
fn segment_limits_use_metered_counts() {
    let segment = Segment::new(9, 17, 5, vec![]);
    assert_eq!(
        limits_for_segment(&segment).unwrap(),
        PreflightLimits::new(17, 5, DEFAULT_CHECKPOINT_INTERVAL)
    );
}

#[test]
fn segment_boundary_rejects_replay_value_count_mismatch() {
    let config = SystemConfig::default();
    let state = VmState::initial(&config, &Default::default(), 0, Streams::default());
    let execution = PreflightExecution {
        state,
        transcript: PreflightTranscript {
            checkpoints: vec![],
            replay_values: vec![1],
        },
        endpoint: PreflightEndpoint::Suspended,
        from_state: ExecutionState::new(0u32, 1u32),
        to_state: ExecutionState::new(0u32, 1u32),
        retired: 2,
    };
    let segment = Segment::new(0, 2, 2, vec![]);
    match require_segment_boundary(execution, &segment) {
        Err(ExecutionError::PreflightReplayValueCountMismatch { expected, actual }) => {
            assert_eq!((expected, actual), (2, 1));
        }
        _ => panic!("unexpected segment validation result"),
    }
}

#[test]
fn merges_executor_only_deferral_dirty_pages_into_initial_image_metadata() {
    let mut memory = AddressMap::default();
    assert!(memory.touched_pages[DEFERRAL_AS as usize]
        .touched_byte_ranges(PAGE_SIZE)
        .is_empty());

    let mut dirty = PreflightDirtyPages::new(&memory).unwrap();
    dirty.deferral[0] = 1;
    dirty.merge_into(&mut memory);

    assert_eq!(
        memory.touched_pages[DEFERRAL_AS as usize].touched_byte_ranges(PAGE_SIZE),
        vec![(0, PAGE_SIZE)]
    );
}
