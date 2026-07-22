//! CFG analysis for RVR IR.
//!
//! Multi-value variable tracking, worklist fixpoint propagation, call/return
//! analysis, and Duff's device scanning operate on `LiftedInstr` and `Block`.

use std::collections::{BTreeSet, HashMap, HashSet};

use openvm_instructions::{
    metering::MAX_METERED_BLOCK_INSNS,
    program::{DEFAULT_PC_STEP as INSTR_SIZE, MAX_ALLOWED_PC},
};
use rvr_openvm_ir::{
    Block, CfgEffect, CfgJumpKind, CfgOp, CfgOperand, CfgResultWidth, CfgTerm, InstrAt,
    LiftedInstr, Terminator, Variable,
};

const MAX_VALUES: usize = 16;
const MAX_ITERATIONS_MULTIPLIER: usize = 20;
const MAX_JUMP_TABLE_SCAN: usize = 256;

/// Error during basic-block (CFG) construction.
#[derive(Debug, thiserror::Error)]
pub enum CfgError {
    /// A statically computable control-flow target exceeds the PC address space.
    #[error("control-flow target {target:#x} from PC {pc:#x} exceeds PC address space")]
    PcOutOfBounds { pc: u64, target: u64 },
    /// A statically computable control-flow target does not name a lifted instruction.
    #[error("control-flow target {target:#x} from PC {pc:#x} does not point to an instruction")]
    PcNotInProgram { pc: u64, target: u64 },
}

// ── TrackedValue ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Unknown,
    Constant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrackedValue {
    kind: ValueKind,
    values: Vec<u64>,
}

impl TrackedValue {
    const fn unknown() -> Self {
        Self {
            kind: ValueKind::Unknown,
            values: Vec::new(),
        }
    }

    fn constant(value: u64) -> Self {
        Self {
            kind: ValueKind::Constant,
            values: vec![value],
        }
    }

    fn is_constant(&self) -> bool {
        self.kind == ValueKind::Constant
    }

    fn add_value(&mut self, value: u64) {
        if self.kind != ValueKind::Constant {
            return;
        }
        match self.values.binary_search(&value) {
            Ok(_) => {}
            Err(idx) => {
                if self.values.len() >= MAX_VALUES {
                    self.kind = ValueKind::Unknown;
                    self.values.clear();
                } else {
                    self.values.insert(idx, value);
                }
            }
        }
    }

    fn merge(&self, other: &Self) -> Self {
        if self.kind == ValueKind::Unknown || other.kind == ValueKind::Unknown {
            return Self::unknown();
        }

        let mut merged = Vec::with_capacity(self.values.len() + other.values.len());
        let (mut i, mut j) = (0, 0);

        while i < self.values.len() && j < other.values.len() {
            let a = self.values[i];
            let b = other.values[j];
            match a.cmp(&b) {
                std::cmp::Ordering::Equal => {
                    merged.push(a);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    merged.push(a);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(b);
                    j += 1;
                }
            }
            if merged.len() > MAX_VALUES {
                return Self::unknown();
            }
        }

        while i < self.values.len() {
            merged.push(self.values[i]);
            i += 1;
            if merged.len() > MAX_VALUES {
                return Self::unknown();
            }
        }

        while j < other.values.len() {
            merged.push(other.values[j]);
            j += 1;
            if merged.len() > MAX_VALUES {
                return Self::unknown();
            }
        }

        Self {
            kind: ValueKind::Constant,
            values: merged,
        }
    }
}

// ── VariableState ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct VariableState {
    values: HashMap<Variable, TrackedValue>,
}

impl VariableState {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    fn get(&self, var: Variable) -> TrackedValue {
        self.values
            .get(&var)
            .cloned()
            .unwrap_or_else(TrackedValue::unknown)
    }

    fn operand(&self, operand: CfgOperand) -> TrackedValue {
        match operand {
            CfgOperand::Var(var) => self.get(var),
            CfgOperand::Const(value) => TrackedValue::constant(value),
        }
    }

    fn set(&mut self, var: Variable, value: TrackedValue) {
        if value.is_constant() {
            self.values.insert(var, value);
        } else {
            self.values.remove(&var);
        }
    }

    fn set_unknown(&mut self, var: Variable) {
        self.values.remove(&var);
    }

    fn clear(&mut self) {
        self.values.clear();
    }

    fn merge(&mut self, other: &Self) -> bool {
        let mut changed = false;
        let vars: Vec<_> = self.values.keys().copied().collect();
        for var in vars {
            let current = self.get(var);
            let merged = current.merge(&other.get(var));
            if merged != current {
                self.set(var, merged);
                changed = true;
            }
        }
        changed
    }
}

// ── Abstract integer evaluation ───────────────────────────────────────────

fn compute_op_u64(op: CfgOp, left: u64, right: u64) -> u64 {
    match op {
        CfgOp::Add => left.wrapping_add(right),
        CfgOp::Sub => left.wrapping_sub(right),
        CfgOp::And => left & right,
        CfgOp::Or => left | right,
        CfgOp::Xor => left ^ right,
        CfgOp::ShiftLeft => left.wrapping_shl((right & 0x3f) as u32),
        CfgOp::ShiftRightLogical => left.wrapping_shr((right & 0x3f) as u32),
        CfgOp::ShiftRightArithmetic => ((left as i64) >> (right & 0x3f)) as u64,
        CfgOp::LessThanSigned => u64::from((left as i64) < (right as i64)),
        CfgOp::LessThanUnsigned => u64::from(left < right),
        CfgOp::Mul => left.wrapping_mul(right),
        CfgOp::MulHighSigned => (((left as i64 as i128) * (right as i64 as i128)) >> 64) as u64,
        CfgOp::MulHighSignedUnsigned => (((left as i64 as i128) * (right as i128)) >> 64) as u64,
        CfgOp::MulHighUnsigned => (((left as u128) * (right as u128)) >> 64) as u64,
        CfgOp::DivSigned => {
            if right == 0 {
                u64::MAX
            } else {
                (left as i64).wrapping_div(right as i64) as u64
            }
        }
        CfgOp::DivUnsigned => {
            if right == 0 {
                u64::MAX
            } else {
                left / right
            }
        }
        CfgOp::RemSigned => {
            if right == 0 {
                left
            } else {
                (left as i64).wrapping_rem(right as i64) as u64
            }
        }
        CfgOp::RemUnsigned => {
            if right == 0 {
                left
            } else {
                left % right
            }
        }
    }
}

fn compute_op_u32(op: CfgOp, left: u32, right: u32) -> u32 {
    match op {
        CfgOp::Add => left.wrapping_add(right),
        CfgOp::Sub => left.wrapping_sub(right),
        CfgOp::And => left & right,
        CfgOp::Or => left | right,
        CfgOp::Xor => left ^ right,
        CfgOp::ShiftLeft => left.wrapping_shl(right & 0x1f),
        CfgOp::ShiftRightLogical => left.wrapping_shr(right & 0x1f),
        CfgOp::ShiftRightArithmetic => ((left as i32) >> (right & 0x1f)) as u32,
        CfgOp::LessThanSigned => u32::from((left as i32) < (right as i32)),
        CfgOp::LessThanUnsigned => u32::from(left < right),
        CfgOp::Mul => left.wrapping_mul(right),
        CfgOp::MulHighSigned => (((left as i32 as i64) * (right as i32 as i64)) >> 32) as u32,
        CfgOp::MulHighSignedUnsigned => {
            (((left as i32 as i64) * (right as u64 as i64)) >> 32) as u32
        }
        CfgOp::MulHighUnsigned => (((left as u64) * (right as u64)) >> 32) as u32,
        CfgOp::DivSigned => {
            if right == 0 {
                u32::MAX
            } else {
                (left as i32).wrapping_div(right as i32) as u32
            }
        }
        CfgOp::DivUnsigned => {
            if right == 0 {
                u32::MAX
            } else {
                left / right
            }
        }
        CfgOp::RemSigned => {
            if right == 0 {
                left
            } else {
                (left as i32).wrapping_rem(right as i32) as u32
            }
        }
        CfgOp::RemUnsigned => {
            if right == 0 {
                left
            } else {
                left % right
            }
        }
    }
}

fn compute_op(op: CfgOp, result: CfgResultWidth, left: u64, right: u64) -> u64 {
    match result {
        CfgResultWidth::U32 => u64::from(compute_op_u32(op, left as u32, right as u32)),
        CfgResultWidth::U64 => compute_op_u64(op, left, right),
        CfgResultWidth::SignExtend32 => {
            compute_op_u32(op, left as u32, right as u32) as i32 as i64 as u64
        }
    }
}

/// Compute the cross-product of two tracked multi-value operands under a binary operation.
fn eval_op_multi(
    op: CfgOp,
    result_width: CfgResultWidth,
    lhs: &TrackedValue,
    rhs: &TrackedValue,
) -> TrackedValue {
    if !lhs.is_constant() || !rhs.is_constant() || lhs.values.is_empty() || rhs.values.is_empty() {
        return TrackedValue::unknown();
    }

    let mut result =
        TrackedValue::constant(compute_op(op, result_width, lhs.values[0], rhs.values[0]));
    for (left_idx, left) in lhs.values.iter().enumerate() {
        for (right_idx, right) in rhs.values.iter().enumerate() {
            if left_idx == 0 && right_idx == 0 {
                continue;
            }
            result.add_value(compute_op(op, result_width, *left, *right));
            if !result.is_constant() {
                return result;
            }
        }
    }
    result
}

// ── IR classification ─────────────────────────────────────────────────────

/// A call is a static or indirect jump whose CFG role is `Call`.
fn is_call(li: &LiftedInstr) -> bool {
    matches!(
        cfg_term_of(li),
        Some(
            CfgTerm::Jump {
                kind: CfgJumpKind::Call,
                ..
            } | CfgTerm::JumpIndirect {
                kind: CfgJumpKind::Call,
                ..
            }
        )
    )
}

/// A return is an indirect jump whose CFG role is `Return`.
fn is_return(li: &LiftedInstr) -> bool {
    matches!(
        cfg_term_of(li),
        Some(CfgTerm::JumpIndirect {
            kind: CfgJumpKind::Return,
            ..
        })
    )
}

/// An indirect jump has a computed target with CFG role `Jump`.
fn is_indirect_jump(li: &LiftedInstr) -> bool {
    matches!(
        cfg_term_of(li),
        Some(CfgTerm::JumpIndirect {
            kind: CfgJumpKind::Jump,
            ..
        })
    )
}

/// Whether this instruction ends its block with explicit control flow.
fn is_control_flow(li: &LiftedInstr) -> bool {
    cfg_term_of(li).is_some_and(|term| !matches!(term, CfgTerm::FallThrough))
}

/// Return a terminator's control-flow description.
fn cfg_term_of(li: &LiftedInstr) -> Option<CfgTerm> {
    match li {
        LiftedInstr::Term { pc, terminator, .. } => {
            Some(terminator.cfg_term(*pc, pc.wrapping_add(INSTR_SIZE as u64)))
        }
        LiftedInstr::Body(_) => None,
    }
}

// ── Simple variable tracking (phase 1, single-value) ──────────────────────

fn single_value(state: &VariableState, operand: CfgOperand) -> Option<u64> {
    let value = state.operand(operand);
    (value.values.len() == 1).then(|| value.values[0])
}

/// Evaluate one known indirect target as `(base + offset) & target_mask`.
fn simple_eval_indirect(
    state: &VariableState,
    base: CfgOperand,
    offset: i32,
    target_mask: u64,
) -> Option<u64> {
    single_value(state, base).and_then(|base| eval_indirect_target(base, offset, target_mask))
}

// ── Phase 1: collect_potential_targets ─────────────────────────────────────

fn extend_existing_pcs(
    pcs: &mut impl Extend<u64>,
    pc_to_idx: &HashMap<u64, usize>,
    candidates: impl IntoIterator<Item = u64>,
) {
    pcs.extend(
        candidates
            .into_iter()
            .filter(|pc| pc_to_idx.contains_key(pc)),
    );
}

fn collect_potential_targets(
    instructions: &[LiftedInstr],
    pc_to_idx: &HashMap<u64, usize>,
) -> (BTreeSet<u64>, BTreeSet<u64>, BTreeSet<u64>) {
    let mut function_entries = BTreeSet::new();
    let mut internal_targets = BTreeSet::new();
    let mut return_sites = BTreeSet::new();

    if let Some(first) = instructions.first() {
        function_entries.insert(first.pc());
    }

    let mut state = VariableState::new();

    for li in instructions {
        let pc = li.pc();

        match li {
            LiftedInstr::Body(InstrAt { instr, .. }) => {
                transfer_effect(instr.cfg_effect(), &mut state);
            }
            LiftedInstr::Term { terminator, .. } => {
                match terminator.cfg_term(pc, pc + INSTR_SIZE as u64) {
                    CfgTerm::FallThrough => {}
                    CfgTerm::Jump { kind, target, .. } => {
                        if pc_to_idx.contains_key(&target) {
                            if kind == CfgJumpKind::Call {
                                function_entries.insert(target);
                                let return_pc = pc + INSTR_SIZE as u64;
                                extend_existing_pcs(&mut return_sites, pc_to_idx, [return_pc]);
                            } else {
                                internal_targets.insert(target);
                            }
                        }
                    }
                    CfgTerm::JumpIndirect {
                        kind,
                        base_value,
                        offset,
                        target_mask,
                        ..
                    } => {
                        if let Some(target_pc) =
                            simple_eval_indirect(&state, base_value, offset, target_mask)
                        {
                            if pc_to_idx.contains_key(&target_pc) {
                                function_entries.insert(target_pc);
                            }
                        }
                        if kind == CfgJumpKind::Call {
                            let return_pc = pc + INSTR_SIZE as u64;
                            extend_existing_pcs(&mut return_sites, pc_to_idx, [return_pc]);
                        }
                    }
                    CfgTerm::Branch { target, .. } => {
                        let fallthrough = pc + INSTR_SIZE as u64;
                        extend_existing_pcs(
                            &mut internal_targets,
                            pc_to_idx,
                            [target, fallthrough],
                        );
                    }
                    CfgTerm::Opaque { successors } => {
                        extend_existing_pcs(&mut internal_targets, pc_to_idx, successors);
                    }
                    CfgTerm::Exit { .. } | CfgTerm::Trap { .. } => {}
                }
                state.clear();
            }
        }
    }

    (function_entries, internal_targets, return_sites)
}

// ── Phase 2: build_call_return_map ────────────────────────────────────────

fn build_call_return_map(
    instructions: &[LiftedInstr],
    pc_to_idx: &HashMap<u64, usize>,
) -> HashMap<u64, HashSet<u64>> {
    let mut map: HashMap<u64, HashSet<u64>> = HashMap::new();

    for li in instructions {
        if let LiftedInstr::Term { pc, terminator, .. } = li {
            if let CfgTerm::Jump {
                kind: CfgJumpKind::Call,
                target,
                ..
            } = terminator.cfg_term(*pc, pc + INSTR_SIZE as u64)
            {
                let return_site = pc + INSTR_SIZE as u64;
                if pc_to_idx.contains_key(&target) && pc_to_idx.contains_key(&return_site) {
                    map.entry(target).or_default().insert(return_site);
                }
            }
        }
    }

    map
}

// ── Phase 3: worklist ─────────────────────────────────────────────────────

struct WorklistContext<'a> {
    instructions: &'a [LiftedInstr],
    pc_to_idx: &'a HashMap<u64, usize>,
    function_entries: &'a BTreeSet<u64>,
    return_sites: &'a BTreeSet<u64>,
    sorted_function_entries: &'a [u64],
    func_internal_targets: &'a HashMap<u64, HashSet<u64>>,
    call_return_map: &'a HashMap<u64, HashSet<u64>>,
}

struct WorklistResult {
    successors: HashMap<u64, HashSet<u64>>,
    resolved_jumps: HashMap<u64, HashSet<u64>>,
}

fn worklist(
    ctx: &WorklistContext<'_>,
    function_entries: &BTreeSet<u64>,
    internal_targets: &BTreeSet<u64>,
) -> WorklistResult {
    let estimated_size = function_entries.len() + internal_targets.len();
    let mut states: HashMap<u64, VariableState> = HashMap::with_capacity(estimated_size);
    let mut work: Vec<u64> = Vec::with_capacity(estimated_size);
    let mut in_work: HashSet<u64> = HashSet::with_capacity(estimated_size);
    let mut successors: HashMap<u64, HashSet<u64>> = HashMap::with_capacity(estimated_size);
    let mut unresolved_dynamic_jumps: HashSet<u64> = HashSet::new();
    let mut resolved_jumps: HashMap<u64, HashSet<u64>> = HashMap::new();

    // Seed function entries with empty variable state. Internal targets receive
    // predecessor state, including constants used to resolve dynamic jumps.
    for addr in function_entries {
        if in_work.insert(*addr) {
            states.insert(*addr, VariableState::new());
            work.push(*addr);
        }
    }

    let max_iterations = ctx
        .instructions
        .len()
        .saturating_mul(MAX_ITERATIONS_MULTIPLIER);

    let mut wi = 0;
    while wi < work.len() {
        if wi > max_iterations {
            break;
        }

        let pc = work[wi];
        wi += 1;
        in_work.remove(&pc);

        let state = match states.get(&pc) {
            Some(s) => s.clone(),
            None => continue,
        };

        let Some(&instr_idx) = ctx.pc_to_idx.get(&pc) else {
            continue;
        };
        let li = &ctx.instructions[instr_idx];

        let succs = get_successors(
            li,
            &state,
            ctx,
            &mut unresolved_dynamic_jumps,
            &mut resolved_jumps,
        );

        let state_out = transfer(li, state);

        for target in &succs {
            if let Some(existing) = states.get_mut(target) {
                if existing.merge(&state_out) && in_work.insert(*target) {
                    work.push(*target);
                }
            } else {
                states.insert(*target, state_out.clone());
                if in_work.insert(*target) {
                    work.push(*target);
                }
            }
        }

        successors.entry(pc).or_default().extend(succs);
    }

    WorklistResult {
        successors,
        resolved_jumps,
    }
}

// ── Phase 4: transfer ─────────────────────────────────────────────────────

/// Process one LiftedInstr, updating the variable state for the output edge.
fn transfer(li: &LiftedInstr, mut state: VariableState) -> VariableState {
    match li {
        LiftedInstr::Body(InstrAt { instr, .. }) => {
            transfer_effect(instr.cfg_effect(), &mut state);
        }
        LiftedInstr::Term { pc, terminator, .. } => {
            if let Terminator::Instruction { node, .. } = terminator {
                transfer_effect(node.cfg_effect(), &mut state);
            }
            let fall_pc = pc + INSTR_SIZE as u64;
            match terminator.cfg_term(*pc, fall_pc) {
                CfgTerm::Jump {
                    link_dst: Some(dst),
                    ..
                }
                | CfgTerm::JumpIndirect {
                    link_dst: Some(dst),
                    ..
                } => state.set(dst, TrackedValue::constant(fall_pc)),
                _ => {}
            }
        }
    }
    state
}

fn transfer_effect(effect: CfgEffect, state: &mut VariableState) {
    match effect {
        CfgEffect::None => {}
        CfgEffect::WriteUnknown { dst } => state.set_unknown(dst),
        CfgEffect::WriteConst { dst, value } => {
            state.set(dst, TrackedValue::constant(value));
        }
        CfgEffect::WriteOp {
            dst,
            op,
            lhs,
            rhs,
            result,
        } => {
            let lhs = state.operand(lhs);
            let rhs = state.operand(rhs);
            state.set(dst, eval_op_multi(op, result, &lhs, &rhs));
        }
        CfgEffect::ClobberAll => state.clear(),
    }
}

// ── Phase 5: get_successors ───────────────────────────────────────────────

fn get_successors(
    li: &LiftedInstr,
    state: &VariableState,
    ctx: &WorklistContext<'_>,
    unresolved_dynamic_jumps: &mut HashSet<u64>,
    resolved_jumps: &mut HashMap<u64, HashSet<u64>>,
) -> HashSet<u64> {
    let mut result = HashSet::new();
    let pc = li.pc();
    let is_call_instr = is_call(li);

    match li {
        LiftedInstr::Body(_) => {
            // Body instructions fall through.
            let fallthrough = pc + INSTR_SIZE as u64;
            extend_existing_pcs(&mut result, ctx.pc_to_idx, [fallthrough]);
        }
        LiftedInstr::Term { terminator, .. } => {
            match terminator.cfg_term(pc, pc + INSTR_SIZE as u64) {
                CfgTerm::FallThrough => {
                    let fallthrough = pc + INSTR_SIZE as u64;
                    extend_existing_pcs(&mut result, ctx.pc_to_idx, [fallthrough]);
                }
                CfgTerm::Jump { target, .. } => {
                    if ctx.pc_to_idx.contains_key(&target) {
                        result.insert(target);
                        let return_pc = pc + INSTR_SIZE as u64;
                        if is_call_instr {
                            extend_existing_pcs(&mut result, ctx.pc_to_idx, [return_pc]);
                        }
                    }
                }
                CfgTerm::JumpIndirect {
                    base_value,
                    offset,
                    target_mask,
                    ..
                } => {
                    let addr_val = eval_indirect_multi(state, base_value, offset, target_mask);

                    if addr_val.is_constant() && !addr_val.values.is_empty() {
                        let mut targets: Vec<u64> = Vec::new();
                        for &t in &addr_val.values {
                            if ctx.pc_to_idx.contains_key(&t) {
                                targets.push(t);
                            }
                        }

                        if !targets.is_empty() {
                            result.extend(targets.iter().copied());
                            let entry = resolved_jumps.entry(pc).or_default();
                            entry.extend(targets.iter().copied());
                            if is_call_instr {
                                result.insert(pc + INSTR_SIZE as u64);
                            }
                        } else {
                            handle_unresolved_jump(
                                li,
                                pc,
                                is_call_instr,
                                ctx,
                                unresolved_dynamic_jumps,
                                &mut result,
                            );
                        }
                    } else {
                        handle_unresolved_jump(
                            li,
                            pc,
                            is_call_instr,
                            ctx,
                            unresolved_dynamic_jumps,
                            &mut result,
                        );
                    }
                }
                CfgTerm::Branch { target, .. } => {
                    let fallthrough = pc + INSTR_SIZE as u64;
                    extend_existing_pcs(&mut result, ctx.pc_to_idx, [fallthrough, target]);
                }
                CfgTerm::Opaque { successors } => {
                    extend_existing_pcs(&mut result, ctx.pc_to_idx, successors);
                }
                CfgTerm::Exit { .. } | CfgTerm::Trap { .. } => {}
            }
        }
    }

    result
}

/// Evaluate `(base + offset) & target_mask` and return a target inside the PC domain.
fn eval_indirect_target(base: u64, offset: i32, target_mask: u64) -> Option<u64> {
    let target = base.wrapping_add_signed(i64::from(offset)) & target_mask;
    (target <= u64::from(MAX_ALLOWED_PC)).then_some(target)
}

/// Evaluate an indirect target for every currently known value of the base operand.
fn eval_indirect_multi(
    state: &VariableState,
    base: CfgOperand,
    offset: i32,
    target_mask: u64,
) -> TrackedValue {
    let base = state.operand(base);
    if !base.is_constant() || base.values.is_empty() {
        return TrackedValue::unknown();
    }

    let first = match eval_indirect_target(base.values[0], offset, target_mask) {
        Some(t) => t,
        None => return TrackedValue::unknown(),
    };
    let mut result = TrackedValue::constant(first);
    for &v in base.values.iter().skip(1) {
        match eval_indirect_target(v, offset, target_mask) {
            Some(t) => {
                result.add_value(t);
                if !result.is_constant() {
                    return TrackedValue::unknown();
                }
            }
            None => return TrackedValue::unknown(),
        }
    }
    result
}

fn handle_unresolved_jump(
    li: &LiftedInstr,
    pc: u64,
    is_call_instr: bool,
    ctx: &WorklistContext<'_>,
    unresolved_dynamic_jumps: &mut HashSet<u64>,
    result: &mut HashSet<u64>,
) {
    if is_return(li) {
        if let Some(func_start) = binary_search_le(ctx.sorted_function_entries, pc) {
            if let Some(returns) = ctx.call_return_map.get(&func_start) {
                result.extend(returns.iter().copied());
            } else {
                result.extend(ctx.return_sites.iter().copied());
            }
        } else {
            result.extend(ctx.return_sites.iter().copied());
        }
    } else if is_call_instr {
        result.extend(ctx.function_entries.iter().copied());
        result.insert(pc + INSTR_SIZE as u64);
    } else if is_indirect_jump(li) {
        let duff_targets =
            scan_jump_table_targets(ctx.instructions, ctx.pc_to_idx, pc + INSTR_SIZE as u64);
        if !duff_targets.is_empty() {
            result.extend(duff_targets);
        }

        if let Some(func_start) = binary_search_le(ctx.sorted_function_entries, pc) {
            if let Some(targets) = ctx.func_internal_targets.get(&func_start) {
                result.extend(targets.iter().copied());
            }
        }

        if result.is_empty() {
            unresolved_dynamic_jumps.insert(pc);
            result.extend(ctx.function_entries.iter().copied());
        }
    }
}

// ── Phase 6: scan_jump_table_targets ──────────────────────────────────────

fn scan_jump_table_targets(
    instructions: &[LiftedInstr],
    pc_to_idx: &HashMap<u64, usize>,
    start_pc: u64,
) -> HashSet<u64> {
    let mut targets = HashSet::new();
    let mut pc = start_pc;
    let mut count = 0;

    while count < MAX_JUMP_TABLE_SCAN {
        let Some(&idx) = pc_to_idx.get(&pc) else {
            break;
        };
        let li = &instructions[idx];

        targets.insert(pc);

        if is_return(li) {
            break;
        }

        if let Some(CfgTerm::Jump { target, .. }) = cfg_term_of(li) {
            if !is_call(li) {
                targets.insert(target);
                break;
            }
        }

        if is_indirect_jump(li) {
            break;
        }

        pc = li.pc() + INSTR_SIZE as u64;
        count += 1;
    }

    targets
}

// ── Phase 7: compute_leaders ──────────────────────────────────────────────

fn compute_leaders(
    instructions: &[LiftedInstr],
    pc_to_idx: &HashMap<u64, usize>,
    successors: &HashMap<u64, HashSet<u64>>,
    function_entries: &BTreeSet<u64>,
    internal_targets: &BTreeSet<u64>,
    return_sites: &BTreeSet<u64>,
) -> BTreeSet<u64> {
    let mut leaders = BTreeSet::new();

    leaders.extend(function_entries.iter().copied());
    leaders.extend(internal_targets.iter().copied());
    leaders.extend(return_sites.iter().copied());

    for (&pc, succs) in successors {
        let Some(&idx) = pc_to_idx.get(&pc) else {
            continue;
        };
        let li = &instructions[idx];
        if is_control_flow(li) {
            leaders.extend(succs.iter().copied());
            let next_pc = li.pc() + INSTR_SIZE as u64;
            if pc_to_idx.contains_key(&next_pc) {
                leaders.insert(next_pc);
            }
        }
    }

    leaders
}

// ── binary_search_le ──────────────────────────────────────────────────────

fn binary_search_le(sorted: &[u64], target: u64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if sorted[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        None
    } else {
        Some(sorted[lo - 1])
    }
}

// ── Public API ────────────────────────────────────────────────────────────

/// Build basic blocks from a flat `LiftedInstr` sequence using multi-value
/// constant propagation with worklist fixpoint to resolve dynamic jump targets.
///
/// `extra_targets` provides additional block entry points discovered externally,
/// e.g. by scanning read-only data segments for code pointers (switch tables,
/// function pointer arrays).
///
/// Returns `Vec<Block>` with resolved indirect-jump targets filled in.
pub fn build_blocks(
    instructions: &[LiftedInstr],
    extra_targets: &[u64],
) -> Result<Vec<Block>, CfgError> {
    if instructions.is_empty() {
        return Ok(Vec::new());
    }

    // Build PC -> instruction index lookup.
    let pc_to_idx: HashMap<u64, usize> = instructions
        .iter()
        .enumerate()
        .map(|(i, li)| (li.pc(), i))
        .collect();

    // Phase 1: Discover potential targets.
    let (function_entries, mut internal_targets, return_sites) =
        collect_potential_targets(instructions, &pc_to_idx);

    // Incorporate externally discovered targets (e.g. from rodata scanning).
    for &target in extra_targets {
        if pc_to_idx.contains_key(&target) {
            internal_targets.insert(target);
        }
    }

    // Phase 2: Build call-return map.
    let call_return_map = build_call_return_map(instructions, &pc_to_idx);

    // Sorted function entries for binary_search_le.
    let sorted_function_entries: Vec<u64> = function_entries.iter().copied().collect();

    // Group internal targets by enclosing function.
    let mut func_internal_targets: HashMap<u64, HashSet<u64>> = HashMap::new();
    for target in &internal_targets {
        if let Some(func_start) = binary_search_le(&sorted_function_entries, *target) {
            func_internal_targets
                .entry(func_start)
                .or_default()
                .insert(*target);
        }
    }

    // Phase 3: Worklist fixpoint propagation.
    let ctx = WorklistContext {
        instructions,
        pc_to_idx: &pc_to_idx,
        function_entries: &function_entries,
        return_sites: &return_sites,
        sorted_function_entries: &sorted_function_entries,
        func_internal_targets: &func_internal_targets,
        call_return_map: &call_return_map,
    };

    let WorklistResult {
        successors,
        resolved_jumps,
    } = worklist(&ctx, &function_entries, &internal_targets);

    // Phase 7: Compute leaders.
    let leaders = compute_leaders(
        instructions,
        &pc_to_idx,
        &successors,
        &function_entries,
        &internal_targets,
        &return_sites,
    );

    // Build blocks by splitting at leaders and patching resolved indirect-jump targets.
    Ok(build_block_list(instructions, &leaders, &resolved_jumps))
}

/// Split the flat instruction list into `Block`s at leader boundaries,
/// filling in resolved indirect-jump targets.
fn build_block_list(
    instructions: &[LiftedInstr],
    leaders: &BTreeSet<u64>,
    resolved_jumps: &HashMap<u64, HashSet<u64>>,
) -> Vec<Block> {
    // Max block size; used to flush periodically so the segmentation check in
    // metered mode (which fires at block boundaries) stays granular enough.
    const MAX_BLOCK_INSNS: u32 = MAX_METERED_BLOCK_INSNS;
    let mut blocks: Vec<Block> = Vec::new();

    // Accumulate body instructions for the current block.
    let mut body: Vec<InstrAt> = Vec::new();
    let mut block_start_pc: Option<u64> = None;

    for li in instructions {
        let pc = li.pc();

        // If this PC is a leader and we already have accumulated body, flush the
        // previous block with a FallThrough terminator.
        if leaders.contains(&pc) && block_start_pc.is_some() {
            if !body.is_empty() {
                let start = block_start_pc.unwrap();
                let last_body_pc = body.last().unwrap().pc;
                blocks.push(Block {
                    start_pc: start,
                    end_pc: last_body_pc + INSTR_SIZE as u64,
                    instructions: std::mem::take(&mut body),
                    terminator: Terminator::FallThrough,
                    terminator_pc: last_body_pc,
                    terminator_source_loc: None,
                });
            }
            block_start_pc = None;
        }

        if block_start_pc.is_none() {
            block_start_pc = Some(pc);
        }

        match li {
            LiftedInstr::Body(instr_at) => {
                body.push(instr_at.clone());
                if body.len() >= MAX_BLOCK_INSNS as usize {
                    let start = block_start_pc.unwrap();
                    let last_body_pc = body.last().unwrap().pc;
                    blocks.push(Block {
                        start_pc: start,
                        end_pc: last_body_pc + INSTR_SIZE as u64,
                        instructions: std::mem::take(&mut body),
                        terminator: Terminator::FallThrough,
                        terminator_pc: last_body_pc,
                        terminator_source_loc: None,
                    });
                    block_start_pc = None;
                }
            }
            LiftedInstr::Term {
                terminator,
                source_loc,
                ..
            } => {
                let mut term = terminator.clone();

                // Store the indirect-jump targets found by CFG analysis.
                if let Some(targets) = resolved_jumps.get(&pc) {
                    if let Terminator::Instruction {
                        resolved_targets, ..
                    } = &mut term
                    {
                        let mut sorted_targets: Vec<u64> = targets.iter().copied().collect();
                        sorted_targets.sort_unstable();
                        *resolved_targets = sorted_targets;
                    }
                }

                let start = block_start_pc.unwrap();
                blocks.push(Block {
                    start_pc: start,
                    end_pc: pc + INSTR_SIZE as u64,
                    instructions: std::mem::take(&mut body),
                    terminator: term,
                    terminator_pc: pc,
                    terminator_source_loc: source_loc.clone(),
                });
                block_start_pc = None;
            }
        }
    }

    // Flush any trailing body instructions.
    if !body.is_empty() {
        if let Some(start) = block_start_pc {
            let last_body_pc = body.last().unwrap().pc;
            blocks.push(Block {
                start_pc: start,
                end_pc: last_body_pc + INSTR_SIZE as u64,
                instructions: std::mem::take(&mut body),
                terminator: Terminator::FallThrough,
                terminator_pc: last_body_pc,
                terminator_source_loc: None,
            });
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{CfgBranchCond, CfgIntWidth, ExtEmitCtx, ExtInstr};

    use super::*;

    fn term(pc: u64, terminator: Terminator) -> LiftedInstr {
        LiftedInstr::Term {
            pc,
            terminator,
            source_loc: None,
        }
    }

    fn body(pc: u64, instr: impl ExtInstr + 'static) -> LiftedInstr {
        LiftedInstr::Body(InstrAt {
            pc,
            instr: Box::new(instr),
            source_loc: None,
        })
    }

    #[derive(Debug, Clone)]
    struct TestInstr {
        effect: CfgEffect,
        term: Option<CfgTerm>,
    }

    impl ExtInstr for TestInstr {
        fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

        fn cfg_effect(&self) -> CfgEffect {
            self.effect.clone()
        }

        fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
            self.term.clone()
        }

        fn accesses_memory(&self) -> bool {
            false
        }

        fn clone_box(&self) -> Box<dyn ExtInstr> {
            Box::new(self.clone())
        }
    }

    fn instruction_term(pc: u64, cfg_term: CfgTerm) -> LiftedInstr {
        instruction_term_with_effect(pc, CfgEffect::None, cfg_term)
    }

    fn instruction_term_with_effect(pc: u64, effect: CfgEffect, cfg_term: CfgTerm) -> LiftedInstr {
        term(
            pc,
            Terminator::instruction(TestInstr {
                effect,
                term: Some(cfg_term),
            }),
        )
    }

    fn block_at(blocks: &[Block], start_pc: u64) -> &Block {
        blocks
            .iter()
            .find(|block| block.start_pc == start_pc)
            .unwrap_or_else(|| panic!("no block starts at {start_pc:#x}"))
    }

    const fn var(index: u32) -> Variable {
        Variable::new(index)
    }

    #[test]
    fn variable_state_merge_tracks_only_values_known_on_every_path() {
        let variable = var(1);
        let mut state = VariableState::new();
        state.set(variable, TrackedValue::constant(7));

        let mut same = VariableState::new();
        same.set(variable, TrackedValue::constant(7));
        assert!(!state.merge(&same));
        assert_eq!(state.get(variable), TrackedValue::constant(7));

        let mut different = VariableState::new();
        different.set(variable, TrackedValue::constant(9));
        assert!(state.merge(&different));
        assert_eq!(state.get(variable).values, vec![7, 9]);

        assert!(state.merge(&VariableState::new()));
        assert_eq!(state.get(variable), TrackedValue::unknown());

        let mut unknown = VariableState::new();
        assert!(!unknown.merge(&same));
        assert_eq!(unknown.get(variable), TrackedValue::unknown());
    }

    #[test]
    fn invalid_branch_target_preserves_valid_fallthrough() {
        let blocks = build_blocks(
            &[
                instruction_term(
                    0,
                    CfgTerm::Branch {
                        cond: CfgBranchCond::Eq,
                        width: CfgIntWidth::U64,
                        lhs: var(1),
                        rhs: var(2),
                        target: 8,
                        known: None,
                    },
                ),
                term(4, Terminator::Exit { code: 0 }),
            ],
            &[],
        )
        .expect("invalid branch edge should be handled at runtime");

        assert!(matches!(
            block_at(&blocks, 0).terminator.cfg_term(0, 4),
            CfgTerm::Branch { target: 8, .. }
        ));
        assert!(matches!(
            block_at(&blocks, 4).terminator,
            Terminator::Exit { code: 0 }
        ));
    }

    #[test]
    fn trailing_body_is_preserved_before_invalid_fallthrough() {
        let blocks = build_blocks(
            &[body(
                0,
                TestInstr {
                    effect: CfgEffect::WriteConst {
                        dst: var(5),
                        value: 1,
                    },
                    term: None,
                },
            )],
            &[],
        )
        .expect("invalid fallthrough should be handled at runtime");

        let block = block_at(&blocks, 0);
        assert_eq!(block.instructions.len(), 1);
        assert!(matches!(block.terminator, Terminator::FallThrough));
    }

    #[test]
    fn internal_target_keeps_predecessor_state_for_jump_resolution() {
        let blocks = build_blocks(
            &[
                body(
                    0,
                    TestInstr {
                        effect: CfgEffect::WriteConst {
                            dst: var(5),
                            value: 16,
                        },
                        term: None,
                    },
                ),
                instruction_term(
                    4,
                    CfgTerm::JumpIndirect {
                        kind: CfgJumpKind::Jump,
                        link_dst: None,
                        base_value: CfgOperand::Var(var(5)),
                        offset: 0,
                        target_mask: !1,
                    },
                ),
                body(
                    8,
                    TestInstr {
                        effect: CfgEffect::None,
                        term: None,
                    },
                ),
                body(
                    12,
                    TestInstr {
                        effect: CfgEffect::None,
                        term: None,
                    },
                ),
                term(16, Terminator::Exit { code: 0 }),
            ],
            &[4],
        )
        .expect("cfg build");

        assert_eq!(block_at(&blocks, 4).terminator.successors(4, 8), vec![16]);
    }

    #[test]
    fn clobber_all_invalidates_a_stale_indirect_target() {
        let blocks = build_blocks(
            &[
                body(
                    0,
                    TestInstr {
                        effect: CfgEffect::WriteConst {
                            dst: var(5),
                            value: 16,
                        },
                        term: None,
                    },
                ),
                body(
                    4,
                    TestInstr {
                        effect: CfgEffect::ClobberAll,
                        term: None,
                    },
                ),
                instruction_term(
                    8,
                    CfgTerm::JumpIndirect {
                        kind: CfgJumpKind::Jump,
                        link_dst: None,
                        base_value: CfgOperand::Var(var(5)),
                        offset: 0,
                        target_mask: !1,
                    },
                ),
                term(12, Terminator::Exit { code: 0 }),
                term(16, Terminator::Exit { code: 0 }),
            ],
            &[],
        )
        .expect("cfg build");

        assert!(block_at(&blocks, 0).terminator.successors(8, 12).is_empty());
    }

    #[test]
    fn terminator_effect_invalidates_a_stale_indirect_target() {
        let blocks = build_blocks(
            &[
                body(
                    0,
                    TestInstr {
                        effect: CfgEffect::WriteConst {
                            dst: var(5),
                            value: 16,
                        },
                        term: None,
                    },
                ),
                instruction_term_with_effect(
                    4,
                    CfgEffect::WriteUnknown { dst: var(5) },
                    CfgTerm::Opaque {
                        successors: vec![8],
                    },
                ),
                instruction_term(
                    8,
                    CfgTerm::JumpIndirect {
                        kind: CfgJumpKind::Jump,
                        link_dst: None,
                        base_value: CfgOperand::Var(var(5)),
                        offset: 0,
                        target_mask: !1,
                    },
                ),
                term(12, Terminator::Exit { code: 0 }),
                term(16, Terminator::Exit { code: 0 }),
            ],
            &[],
        )
        .expect("cfg build");

        assert!(block_at(&blocks, 8).terminator.successors(8, 12).is_empty());
    }

    #[test]
    fn u32_cfg_arithmetic_wraps_and_computes_high_multiply() {
        assert_eq!(compute_op(CfgOp::Add, CfgResultWidth::U32, u64::MAX, 1), 0);
        assert_eq!(
            compute_op(
                CfgOp::MulHighUnsigned,
                CfgResultWidth::U32,
                u64::from(u32::MAX),
                2,
            ),
            1
        );
        assert_eq!(
            compute_op(
                CfgOp::MulHighUnsigned,
                CfgResultWidth::SignExtend32,
                u64::from(u32::MAX),
                u64::from(u32::MAX),
            ),
            u64::MAX - 1
        );
    }
}
