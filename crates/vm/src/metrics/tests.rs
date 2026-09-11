use openvm_instructions::{exe::FnBound, SysPhantom};

use super::VmMetrics;

#[test]
fn replayed_cycle_tracker_span_crosses_segment_boundary() {
    let mut metrics = VmMetrics::default();
    metrics.record_replayed_instruction(
        "PHANTOM".to_string(),
        Some("CT-hash".to_string()),
        Some(SysPhantom::CtStart),
        4,
    );
    assert_eq!(metrics.cycle_tracker.get_full_name(), "hash");

    let mut next_segment = metrics.partial_take();
    next_segment.record_replayed_instruction("ADD".to_string(), None, None, 8);
    assert_eq!(next_segment.cycle_tracker.get_full_name(), "hash");

    next_segment.record_replayed_instruction(
        "PHANTOM".to_string(),
        Some("CT-hash".to_string()),
        Some(SysPhantom::CtEnd),
        12,
    );
    assert_eq!(next_segment.cycle_tracker.get_full_name(), "");
    assert_eq!(
        next_segment.counts.get(&(None, "ADD".to_string())).copied(),
        Some(1)
    );
}

#[test]
fn replayed_function_transition_preserves_caller_on_return() {
    let mut metrics = VmMetrics::default();
    metrics.fn_bounds.insert(
        0,
        FnBound {
            start: 0,
            end: 4,
            name: "caller".to_string(),
        },
    );
    metrics.fn_bounds.insert(
        8,
        FnBound {
            start: 8,
            end: 12,
            name: "callee".to_string(),
        },
    );
    metrics.current_fn = metrics.fn_bounds[&0].clone();
    metrics.cycle_tracker.start("caller".to_string());

    metrics.record_replayed_instruction("JAL".to_string(), None, None, 8);
    assert_eq!(metrics.cycle_tracker.get_full_name(), "caller;callee");

    metrics.record_replayed_instruction("JALR".to_string(), None, None, 4);
    assert_eq!(metrics.cycle_tracker.get_full_name(), "caller");
}

#[test]
fn first_replayed_instruction_uses_the_carried_cursor() {
    let mut metrics = VmMetrics::default();
    metrics.fn_bounds.insert(
        0,
        FnBound {
            start: 0,
            end: 8,
            name: "entry".to_string(),
        },
    );

    metrics.record_replayed_instruction("ADD".to_string(), Some("add".to_string()), None, 4);

    assert_eq!(metrics.cycle_tracker.get_full_name(), "");
    assert_eq!(
        metrics
            .counts
            .get(&(Some("add".to_string()), "ADD".to_string()))
            .copied(),
        Some(1)
    );
}
