//! CFG analysis for the new rvr-openvm-ir types.
//!
//! Multi-value register tracking, worklist fixpoint propagation,
//! call/return analysis, and Duff's device scanning -- all operating
//! on `LiftedInstr` / `Block` from `rvr_openvm_ir`.

use std::collections::{BTreeSet, HashMap, HashSet};

use rvr_openvm_ir::{AluOp, Block, Instr, InstrAt, LiftedInstr, MulDivOp, Terminator};

const NUM_REGS: usize = 32;
const MAX_VALUES: usize = 16;
const MAX_ITERATIONS_MULTIPLIER: usize = 20;
const MAX_JUMP_TABLE_SCAN: usize = 256;

/// Every OpenVM instruction is 4 bytes.
const INSTR_SIZE: u32 = 4;

// ── RegisterValue ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Unknown,
    Constant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RegisterValue {
    kind: ValueKind,
    values: Vec<u32>,
}

impl RegisterValue {
    const fn unknown() -> Self {
        Self {
            kind: ValueKind::Unknown,
            values: Vec::new(),
        }
    }

    fn constant(value: u32) -> Self {
        Self {
            kind: ValueKind::Constant,
            values: vec![value],
        }
    }

    fn is_constant(&self) -> bool {
        self.kind == ValueKind::Constant
    }

    fn add_value(&mut self, value: u32) {
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

// ── RegisterState ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct RegisterState {
    regs: [RegisterValue; NUM_REGS],
}

impl RegisterState {
    fn new() -> Self {
        let mut regs = std::array::from_fn(|_| RegisterValue::unknown());
        regs[0] = RegisterValue::constant(0);
        Self { regs }
    }

    fn get(&self, reg: u8) -> RegisterValue {
        let idx = reg as usize;
        if idx >= NUM_REGS {
            return RegisterValue::unknown();
        }
        if idx == 0 {
            return RegisterValue::constant(0);
        }
        self.regs[idx].clone()
    }

    fn set(&mut self, reg: u8, value: RegisterValue) {
        let idx = reg as usize;
        if idx == 0 || idx >= NUM_REGS {
            return;
        }
        self.regs[idx] = value;
    }

    fn set_unknown(&mut self, reg: u8) {
        self.set(reg, RegisterValue::unknown());
    }

    fn merge(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for idx in 1..NUM_REGS {
            let merged = self.regs[idx].merge(&other.regs[idx]);
            if merged != self.regs[idx] {
                self.regs[idx] = merged;
                changed = true;
            }
        }
        changed
    }
}

// ── Binary operation evaluation (RV32, u32) ───────────────────────────────

fn compute_binary_op(op: AluOp, left: u32, right: u32) -> u32 {
    match op {
        AluOp::Add => left.wrapping_add(right),
        AluOp::Sub => left.wrapping_sub(right),
        AluOp::And => left & right,
        AluOp::Or => left | right,
        AluOp::Xor => left ^ right,
        AluOp::Sll => left.wrapping_shl(right & 0x1f),
        AluOp::Srl => left.wrapping_shr(right & 0x1f),
        AluOp::Sra => ((left as i32).wrapping_shr(right & 0x1f)) as u32,
        AluOp::Slt => {
            if (left as i32) < (right as i32) {
                1
            } else {
                0
            }
        }
        AluOp::Sltu => {
            if left < right {
                1
            } else {
                0
            }
        }
    }
}

fn is_address_relevant_op(op: AluOp) -> bool {
    matches!(
        op,
        AluOp::Add
            | AluOp::Sub
            | AluOp::And
            | AluOp::Or
            | AluOp::Xor
            | AluOp::Sll
            | AluOp::Srl
            | AluOp::Sra
    )
}

/// Compute the cross-product of two multi-value register values under a binary op.
fn eval_binary_multi(op: AluOp, lv: &RegisterValue, rv: &RegisterValue) -> RegisterValue {
    if !lv.is_constant() || !rv.is_constant() || lv.values.is_empty() || rv.values.is_empty() {
        return RegisterValue::unknown();
    }

    if !is_address_relevant_op(op) {
        return RegisterValue::unknown();
    }

    let mut result = RegisterValue::constant(compute_binary_op(op, lv.values[0], rv.values[0]));
    'outer: for l in &lv.values {
        for r in &rv.values {
            if *l == lv.values[0] && *r == rv.values[0] {
                continue;
            }
            result.add_value(compute_binary_op(op, *l, *r));
            if !result.is_constant() {
                break 'outer;
            }
        }
    }
    result
}

/// Compute a mul/div op if both operands are known (single-value only for simplicity).
fn compute_muldiv_op(op: MulDivOp, left: u32, right: u32) -> u32 {
    match op {
        MulDivOp::Mul => left.wrapping_mul(right),
        MulDivOp::Mulh => {
            let a = left as i32 as i64;
            let b = right as i32 as i64;
            ((a.wrapping_mul(b)) >> 32) as u32
        }
        MulDivOp::Mulhsu => {
            let a = left as i32 as i64;
            let b = right as u64 as i64;
            ((a.wrapping_mul(b)) >> 32) as u32
        }
        MulDivOp::Mulhu => {
            let a = left as u64;
            let b = right as u64;
            ((a.wrapping_mul(b)) >> 32) as u32
        }
        MulDivOp::Div => {
            if right == 0 {
                u32::MAX // RISC-V div by zero
            } else {
                (left as i32).wrapping_div(right as i32) as u32
            }
        }
        MulDivOp::Divu => {
            if right == 0 {
                u32::MAX
            } else {
                left.wrapping_div(right)
            }
        }
        MulDivOp::Rem => {
            if right == 0 {
                left
            } else {
                (left as i32).wrapping_rem(right as i32) as u32
            }
        }
        MulDivOp::Remu => {
            if right == 0 {
                left
            } else {
                left.wrapping_rem(right)
            }
        }
    }
}

// ── IR classification ─────────────────────────────────────────────────────

/// A LiftedInstr is a call if it's a Term with Jump { link_rd: Some(_), .. }
/// or JumpDyn { link_rd: Some(_), .. }.
fn is_call(li: &LiftedInstr) -> bool {
    match li {
        LiftedInstr::Term {
            terminator: Terminator::Jump { link_rd, .. } | Terminator::JumpDyn { link_rd, .. },
            ..
        } => link_rd.is_some(),
        _ => false,
    }
}

/// A LiftedInstr is a return if it's a Term with JumpDyn { link_rd: None, rs1: 1, .. }.
fn is_return(li: &LiftedInstr) -> bool {
    match li {
        LiftedInstr::Term {
            terminator: Terminator::JumpDyn { link_rd, rs1, .. },
            ..
        } => link_rd.is_none() && *rs1 == 1,
        _ => false,
    }
}

/// A LiftedInstr is an indirect jump if it's a JumpDyn that is not a call and not a return.
fn is_indirect_jump(li: &LiftedInstr) -> bool {
    matches!(
        li,
        LiftedInstr::Term {
            terminator: Terminator::JumpDyn { .. },
            ..
        }
    ) && !is_call(li)
        && !is_return(li)
}

/// Returns true if this LiftedInstr is a control-flow terminator (not a body instruction
/// and not a FallThrough).
fn is_control_flow(li: &LiftedInstr) -> bool {
    match li {
        LiftedInstr::Term { terminator, .. } => terminator.is_block_end(),
        _ => false,
    }
}

// ── Simple register tracking (phase 1, single-value) ──────────────────────

/// Process a single instruction for simple single-value register tracking.
fn simple_process(li: &LiftedInstr, regs: &mut [Option<u32>; NUM_REGS]) {
    match li {
        LiftedInstr::Body(InstrAt { instr, .. }) => {
            simple_process_instr(instr, regs);
        }
        LiftedInstr::Term { terminator, .. } => {
            // Jump/JumpDyn may write link register
            match terminator {
                Terminator::Jump {
                    link_rd: Some(rd), ..
                }
                | Terminator::JumpDyn {
                    link_rd: Some(rd), ..
                } => {
                    // link register gets pc+4, which we don't track in simple mode
                    regs[*rd as usize] = None;
                }
                _ => {}
            }
        }
    }
}

fn simple_process_instr(instr: &Instr, regs: &mut [Option<u32>; NUM_REGS]) {
    match instr {
        Instr::AluReg { op, rd, rs1, rs2 } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = match (regs[*rs1 as usize], regs[*rs2 as usize]) {
                (Some(a), Some(b)) => Some(compute_binary_op(*op, a, b)),
                _ => None,
            };
            regs[rd as usize] = val;
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = regs[*rs1 as usize].map(|a| compute_binary_op(*op, a, *imm as u32));
            regs[rd as usize] = val;
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = regs[*rs1 as usize].map(|a| compute_binary_op(*op, a, u32::from(*shamt)));
            regs[rd as usize] = val;
        }
        Instr::Lui { rd, value } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            regs[rd as usize] = Some(*value);
        }
        Instr::Auipc { rd, value } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            regs[rd as usize] = Some(*value);
        }
        Instr::Load { rd, .. } => {
            let rd = *rd;
            if rd != 0 {
                regs[rd as usize] = None;
            }
        }
        Instr::MulDiv { op, rd, rs1, rs2 } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = match (regs[*rs1 as usize], regs[*rs2 as usize]) {
                (Some(a), Some(b)) => Some(compute_muldiv_op(*op, a, b)),
                _ => None,
            };
            regs[rd as usize] = val;
        }
        // IO / system instructions: most don't write general registers
        Instr::Store { .. }
        | Instr::Nop
        | Instr::HintInput
        | Instr::PrintStr { .. }
        | Instr::HintRandom { .. }
        | Instr::HintStoreW { .. }
        | Instr::HintBuffer { .. }
        | Instr::Reveal { .. }
        | Instr::Ext(_) => {}
    }
}

/// Evaluate the JumpDyn target address: (state[rs1] + imm) & !1.
fn simple_eval_jumpdyn(regs: &[Option<u32>; NUM_REGS], rs1: u8, imm: i32) -> Option<u32> {
    regs[rs1 as usize].map(|base| base.wrapping_add(imm as u32) & !1u32)
}

// ── Phase 1: collect_potential_targets ─────────────────────────────────────

fn collect_potential_targets(
    instructions: &[LiftedInstr],
    pc_to_idx: &HashMap<u32, usize>,
) -> (BTreeSet<u32>, BTreeSet<u32>, BTreeSet<u32>) {
    let mut function_entries = BTreeSet::new();
    let mut internal_targets = BTreeSet::new();
    let mut return_sites = BTreeSet::new();

    if let Some(first) = instructions.first() {
        function_entries.insert(first.pc());
    }

    let mut regs: [Option<u32>; NUM_REGS] = [None; NUM_REGS];
    regs[0] = Some(0);

    for li in instructions {
        let pc = li.pc();

        simple_process(li, &mut regs);

        match li {
            LiftedInstr::Body(_) => {
                // Body instructions don't end blocks.
            }
            LiftedInstr::Term { terminator, .. } => match terminator {
                Terminator::FallThrough => {}
                Terminator::Jump { target, .. } => {
                    if is_call(li) {
                        function_entries.insert(*target);
                        return_sites.insert(pc + INSTR_SIZE);
                    } else {
                        internal_targets.insert(*target);
                    }
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::JumpDyn { rs1, imm, .. } => {
                    if let Some(target_pc) = simple_eval_jumpdyn(&regs, *rs1, *imm) {
                        if pc_to_idx.contains_key(&target_pc) {
                            function_entries.insert(target_pc);
                        }
                    }
                    if is_call(li) {
                        return_sites.insert(pc + INSTR_SIZE);
                    }
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::Branch { target, .. } => {
                    internal_targets.insert(*target);
                    internal_targets.insert(pc + INSTR_SIZE);
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::Exit { .. } | Terminator::Trap { .. } => {
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::Extension(ext) => {
                    for target in ext.successors(pc + INSTR_SIZE) {
                        internal_targets.insert(target);
                    }
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
            },
        }
    }

    (function_entries, internal_targets, return_sites)
}

// ── Phase 2: build_call_return_map ────────────────────────────────────────

fn build_call_return_map(instructions: &[LiftedInstr]) -> HashMap<u32, HashSet<u32>> {
    let mut map: HashMap<u32, HashSet<u32>> = HashMap::new();

    for li in instructions {
        if let LiftedInstr::Term {
            pc,
            terminator:
                Terminator::Jump {
                    link_rd: Some(_),
                    target,
                },
            ..
        } = li
        {
            let return_site = pc + INSTR_SIZE;
            map.entry(*target).or_default().insert(return_site);
        }
    }

    map
}

// ── Phase 3: worklist ─────────────────────────────────────────────────────

struct WorklistContext<'a> {
    instructions: &'a [LiftedInstr],
    pc_to_idx: &'a HashMap<u32, usize>,
    function_entries: &'a BTreeSet<u32>,
    return_sites: &'a BTreeSet<u32>,
    sorted_function_entries: &'a [u32],
    func_internal_targets: &'a HashMap<u32, HashSet<u32>>,
    call_return_map: &'a HashMap<u32, HashSet<u32>>,
}

struct WorklistResult {
    successors: HashMap<u32, HashSet<u32>>,
    resolved_jumps: HashMap<u32, HashSet<u32>>,
}

fn worklist(
    ctx: &WorklistContext<'_>,
    function_entries: &BTreeSet<u32>,
    internal_targets: &BTreeSet<u32>,
) -> WorklistResult {
    let estimated_size = function_entries.len() + internal_targets.len();
    let mut states: HashMap<u32, RegisterState> = HashMap::with_capacity(estimated_size);
    let mut work: Vec<u32> = Vec::with_capacity(estimated_size);
    let mut in_work: HashSet<u32> = HashSet::with_capacity(estimated_size);
    let mut successors: HashMap<u32, HashSet<u32>> = HashMap::with_capacity(estimated_size);
    let mut unresolved_dynamic_jumps: HashSet<u32> = HashSet::new();
    let mut resolved_jumps: HashMap<u32, HashSet<u32>> = HashMap::new();

    for addr in function_entries.iter().chain(internal_targets.iter()) {
        if in_work.insert(*addr) {
            states.insert(*addr, RegisterState::new());
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

/// Process one LiftedInstr, updating the register state for the output edge.
fn transfer(li: &LiftedInstr, mut state: RegisterState) -> RegisterState {
    match li {
        LiftedInstr::Body(InstrAt { instr, .. }) => {
            transfer_instr(instr, &mut state);
        }
        LiftedInstr::Term { terminator, .. } => {
            // Jump / JumpDyn may write link register (rd = pc+4).
            // The value of pc+4 is known but we don't need to track it for
            // address resolution purposes -- link registers are typically
            // not used to compute jump targets. Set to unknown to be safe.
            match terminator {
                Terminator::Jump {
                    link_rd: Some(rd), ..
                }
                | Terminator::JumpDyn {
                    link_rd: Some(rd), ..
                } => {
                    state.set_unknown(*rd);
                }
                _ => {}
            }
        }
    }
    state
}

/// Update register state for a body instruction.
fn transfer_instr(instr: &Instr, state: &mut RegisterState) {
    match instr {
        Instr::AluReg { op, rd, rs1, rs2 } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = state.get(*rs2);
            let result = eval_binary_multi(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = RegisterValue::constant(*imm as u32);
            let result = eval_binary_multi(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = RegisterValue::constant(u32::from(*shamt));
            let result = eval_binary_multi(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::Lui { rd, value } => {
            if *rd == 0 {
                return;
            }
            state.set(*rd, RegisterValue::constant(*value));
        }
        Instr::Auipc { rd, value } => {
            if *rd == 0 {
                return;
            }
            state.set(*rd, RegisterValue::constant(*value));
        }
        Instr::Load { rd, .. } => {
            if *rd != 0 {
                state.set_unknown(*rd);
            }
        }
        Instr::MulDiv { op, rd, rs1, rs2 } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = state.get(*rs2);
            if lv.is_constant() && rv.is_constant() && lv.values.len() == 1 && rv.values.len() == 1
            {
                let result = compute_muldiv_op(*op, lv.values[0], rv.values[0]);
                state.set(*rd, RegisterValue::constant(result));
            } else {
                state.set_unknown(*rd);
            }
        }
        // IO / system instructions: most don't write general registers
        Instr::Store { .. }
        | Instr::Nop
        | Instr::HintInput
        | Instr::PrintStr { .. }
        | Instr::HintRandom { .. }
        | Instr::HintStoreW { .. }
        | Instr::HintBuffer { .. }
        | Instr::Reveal { .. }
        | Instr::Ext(_) => {}
    }
}

// ── Phase 5: get_successors ───────────────────────────────────────────────

fn get_successors(
    li: &LiftedInstr,
    state: &RegisterState,
    ctx: &WorklistContext<'_>,
    unresolved_dynamic_jumps: &mut HashSet<u32>,
    resolved_jumps: &mut HashMap<u32, HashSet<u32>>,
) -> HashSet<u32> {
    let mut result = HashSet::new();
    let pc = li.pc();
    let is_call_instr = is_call(li);

    match li {
        LiftedInstr::Body(_) => {
            // Body instructions fall through.
            result.insert(pc + INSTR_SIZE);
        }
        LiftedInstr::Term { terminator, .. } => {
            match terminator {
                Terminator::FallThrough => {
                    result.insert(pc + INSTR_SIZE);
                }
                Terminator::Jump { target, .. } => {
                    result.insert(*target);
                    if is_call_instr {
                        result.insert(pc + INSTR_SIZE);
                    }
                }
                Terminator::JumpDyn { rs1, imm, .. } => {
                    // Evaluate address: (state[rs1] + imm) & !1
                    let addr_val = eval_jumpdyn_multi(state, *rs1, *imm);

                    if addr_val.is_constant() && !addr_val.values.is_empty() {
                        let targets: Vec<u32> = addr_val
                            .values
                            .iter()
                            .filter(|t| ctx.pc_to_idx.contains_key(t))
                            .copied()
                            .collect();

                        if !targets.is_empty() {
                            result.extend(targets.iter().copied());
                            let entry = resolved_jumps.entry(pc).or_default();
                            entry.extend(targets.iter().copied());
                            if is_call_instr {
                                result.insert(pc + INSTR_SIZE);
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
                Terminator::Branch { target, .. } => {
                    result.insert(pc + INSTR_SIZE);
                    result.insert(*target);
                }
                Terminator::Exit { .. } | Terminator::Trap { .. } => {}
                Terminator::Extension(ext) => {
                    for target in ext.successors(pc + INSTR_SIZE) {
                        result.insert(target);
                    }
                }
            }
        }
    }

    result
}

/// Evaluate the JumpDyn target as a multi-value register value:
/// for each possible value of rs1, compute (val + imm) & !1.
fn eval_jumpdyn_multi(state: &RegisterState, rs1: u8, imm: i32) -> RegisterValue {
    let base = state.get(rs1);
    if !base.is_constant() || base.values.is_empty() {
        return RegisterValue::unknown();
    }

    let mut result = RegisterValue::constant(base.values[0].wrapping_add(imm as u32) & !1u32);
    for v in base.values.iter().skip(1) {
        result.add_value(v.wrapping_add(imm as u32) & !1u32);
        if !result.is_constant() {
            return RegisterValue::unknown();
        }
    }
    result
}

fn handle_unresolved_jump(
    li: &LiftedInstr,
    pc: u32,
    is_call_instr: bool,
    ctx: &WorklistContext<'_>,
    unresolved_dynamic_jumps: &mut HashSet<u32>,
    result: &mut HashSet<u32>,
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
        result.insert(pc + INSTR_SIZE);
    } else if is_indirect_jump(li) {
        let duff_targets =
            scan_jump_table_targets(ctx.instructions, ctx.pc_to_idx, pc + INSTR_SIZE);
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
    pc_to_idx: &HashMap<u32, usize>,
    start_pc: u32,
) -> HashSet<u32> {
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

        if let LiftedInstr::Term {
            terminator: Terminator::Jump { target, .. },
            ..
        } = li
        {
            if !is_call(li) {
                targets.insert(*target);
                break;
            }
        }

        if is_indirect_jump(li) {
            break;
        }

        pc = li.pc() + INSTR_SIZE;
        count += 1;
    }

    targets
}

// ── Phase 7: compute_leaders ──────────────────────────────────────────────

fn compute_leaders(
    instructions: &[LiftedInstr],
    pc_to_idx: &HashMap<u32, usize>,
    successors: &HashMap<u32, HashSet<u32>>,
    function_entries: &BTreeSet<u32>,
    internal_targets: &BTreeSet<u32>,
    return_sites: &BTreeSet<u32>,
) -> BTreeSet<u32> {
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
            let next_pc = li.pc() + INSTR_SIZE;
            if pc_to_idx.contains_key(&next_pc) {
                leaders.insert(next_pc);
            }
        }
    }

    leaders
}

// ── binary_search_le ──────────────────────────────────────────────────────

fn binary_search_le(sorted: &[u32], target: u32) -> Option<u32> {
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
/// Returns `Vec<Block>` with resolved `JumpDyn` targets filled in.
pub fn build_blocks(
    instructions: &[LiftedInstr],
    extra_targets: &[u32],
    max_block_insns: u32,
) -> Vec<Block> {
    if instructions.is_empty() {
        return Vec::new();
    }

    // Build PC -> instruction index lookup.
    let pc_to_idx: HashMap<u32, usize> = instructions
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
    let call_return_map = build_call_return_map(instructions);

    // Sorted function entries for binary_search_le.
    let sorted_function_entries: Vec<u32> = function_entries.iter().copied().collect();

    // Group internal targets by enclosing function.
    let mut func_internal_targets: HashMap<u32, HashSet<u32>> = HashMap::new();
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

    // Build blocks by splitting at leaders and patching resolved JumpDyn targets.
    build_block_list(instructions, &leaders, &resolved_jumps, max_block_insns)
}

/// Split the flat instruction list into `Block`s at leader boundaries,
/// filling in resolved JumpDyn targets.
fn build_block_list(
    instructions: &[LiftedInstr],
    leaders: &BTreeSet<u32>,
    resolved_jumps: &HashMap<u32, HashSet<u32>>,
    max_block_insns: u32,
) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();

    // Accumulate body instructions for the current block.
    let mut body: Vec<InstrAt> = Vec::new();
    let mut block_start_pc: Option<u32> = None;

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
                    end_pc: last_body_pc + INSTR_SIZE,
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
                if body.len() >= max_block_insns as usize {
                    let start = block_start_pc.unwrap();
                    let last_body_pc = body.last().unwrap().pc;
                    blocks.push(Block {
                        start_pc: start,
                        end_pc: last_body_pc + INSTR_SIZE,
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

                // Patch resolved targets into JumpDyn.
                if let Some(targets) = resolved_jumps.get(&pc) {
                    if let Terminator::JumpDyn { resolved, .. } = &mut term {
                        let mut sorted_targets: Vec<u32> = targets.iter().copied().collect();
                        sorted_targets.sort_unstable();
                        *resolved = sorted_targets;
                    }
                }

                let start = block_start_pc.unwrap();
                blocks.push(Block {
                    start_pc: start,
                    end_pc: pc + INSTR_SIZE,
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
                end_pc: last_body_pc + INSTR_SIZE,
                instructions: std::mem::take(&mut body),
                terminator: Terminator::FallThrough,
                terminator_pc: last_body_pc,
                terminator_source_loc: None,
            });
        }
    }

    blocks
}
