//! CFG analysis for the new rvr-openvm-ir types.
//!
//! Multi-value register tracking, worklist fixpoint propagation,
//! call/return analysis, and Duff's device scanning -- all operating
//! on `LiftedInstr` / `Block` from `rvr_openvm_ir`.

use std::collections::{BTreeSet, HashMap, HashSet};

use openvm_instructions::{
    metering::MAX_METERED_BLOCK_INSNS, program::DEFAULT_PC_STEP as INSTR_SIZE,
    riscv::RV64_NUM_REGISTERS as NUM_REGS,
};
use rvr_openvm_ir::{AluOp, Block, Instr, InstrAt, LiftedInstr, MulDivOp, Terminator};

use crate::helpers::{is_pc_in_bounds, sext32};

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

// ── RegisterValue ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Unknown,
    Constant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RegisterValue {
    kind: ValueKind,
    values: Vec<u64>,
}

impl RegisterValue {
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

// ── Binary operation evaluation (u64) ─────────────────────────────────────

fn compute_binary_op(op: AluOp, left: u64, right: u64) -> u64 {
    match op {
        AluOp::Add => left.wrapping_add(right),
        AluOp::Sub => left.wrapping_sub(right),
        AluOp::And => left & right,
        AluOp::Or => left | right,
        AluOp::Xor => left ^ right,
        AluOp::Sll => left.wrapping_shl((right & 0x3f) as u32),
        AluOp::Srl => left.wrapping_shr((right & 0x3f) as u32),
        AluOp::Sra => ((left as i64).wrapping_shr((right & 0x3f) as u32)) as u64,
        AluOp::Slt => {
            if (left as i64) < (right as i64) {
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

/// Same as `eval_binary_multi` but using W-suffix semantics (32-bit arithmetic, sign-extended).
fn eval_binary_multi_w(op: AluOp, lv: &RegisterValue, rv: &RegisterValue) -> RegisterValue {
    if !lv.is_constant() || !rv.is_constant() || lv.values.is_empty() || rv.values.is_empty() {
        return RegisterValue::unknown();
    }

    let mut result = RegisterValue::constant(compute_binary_op_w(op, lv.values[0], rv.values[0]));
    'outer: for l in &lv.values {
        for r in &rv.values {
            if *l == lv.values[0] && *r == rv.values[0] {
                continue;
            }
            result.add_value(compute_binary_op_w(op, *l, *r));
            if !result.is_constant() {
                break 'outer;
            }
        }
    }
    result
}

fn compute_binary_op_w(op: AluOp, left: u64, right: u64) -> u64 {
    let rs1 = left as u32;
    let rs2 = right as u32;
    let result: u32 = match op {
        AluOp::Add => rs1.wrapping_add(rs2),
        AluOp::Sub => rs1.wrapping_sub(rs2),
        AluOp::Sll => rs1 << (rs2 & 0x1f),
        AluOp::Srl => rs1 >> (rs2 & 0x1f),
        AluOp::Sra => ((rs1 as i32) >> (rs2 & 0x1f)) as u32,
        _ => unreachable!(),
    };
    sext32(result)
}

fn compute_muldiv_op_w(op: MulDivOp, left: u64, right: u64) -> u64 {
    let rs1 = left as u32;
    let rs2 = right as u32;
    let result: u32 = match op {
        MulDivOp::Mul => rs1.wrapping_mul(rs2),
        MulDivOp::Div => {
            let rs1_i32 = rs1 as i32;
            let rs2_i32 = rs2 as i32;
            match (rs1_i32, rs2_i32) {
                (_, 0) => u32::MAX,
                (i32::MIN, -1) => rs1,
                _ => (rs1_i32 / rs2_i32) as u32,
            }
        }
        MulDivOp::Divu => {
            if rs2 == 0 {
                u32::MAX
            } else {
                rs1 / rs2
            }
        }
        MulDivOp::Rem => {
            let rs1_i32 = rs1 as i32;
            let rs2_i32 = rs2 as i32;
            match (rs1_i32, rs2_i32) {
                (_, 0) => rs1,
                (i32::MIN, -1) => 0,
                _ => (rs1_i32 % rs2_i32) as u32,
            }
        }
        MulDivOp::Remu => {
            if rs2 == 0 {
                rs1
            } else {
                rs1 % rs2
            }
        }
        _ => unreachable!(),
    };
    sext32(result)
}

/// Compute a mul/div op if both operands are known (single-value only for simplicity).
fn compute_muldiv_op(op: MulDivOp, left: u64, right: u64) -> u64 {
    match op {
        MulDivOp::Mul => left.wrapping_mul(right),
        MulDivOp::Mulh => {
            let a = left as i64 as i128;
            let b = right as i64 as i128;
            ((a.wrapping_mul(b)) >> 64) as u64
        }
        MulDivOp::Mulhsu => {
            let a = left as i64 as i128;
            let b = right as i128;
            ((a.wrapping_mul(b)) >> 64) as u64
        }
        MulDivOp::Mulhu => {
            let a = left as u128;
            let b = right as u128;
            ((a.wrapping_mul(b)) >> 64) as u64
        }
        MulDivOp::Div => {
            if right == 0 {
                u64::MAX
            } else {
                (left as i64).wrapping_div(right as i64) as u64
            }
        }
        MulDivOp::Divu => {
            if right == 0 {
                u64::MAX
            } else {
                left.wrapping_div(right)
            }
        }
        MulDivOp::Rem => {
            if right == 0 {
                left
            } else {
                (left as i64).wrapping_rem(right as i64) as u64
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
fn simple_process(li: &LiftedInstr, regs: &mut [Option<u64>; NUM_REGS]) {
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

fn simple_process_instr(instr: &Instr, regs: &mut [Option<u64>; NUM_REGS]) {
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
        Instr::AluWReg { op, rd, rs1, rs2 } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = match (regs[*rs1 as usize], regs[*rs2 as usize]) {
                (Some(a), Some(b)) => Some(compute_binary_op_w(*op, a, b)),
                _ => None,
            };
            regs[rd as usize] = val;
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = regs[*rs1 as usize].map(|a| compute_binary_op(*op, a, *imm as i64 as u64));
            regs[rd as usize] = val;
        }
        Instr::AluWImm { op, rd, rs1, imm } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = regs[*rs1 as usize].map(|a| compute_binary_op_w(*op, a, *imm as i64 as u64));
            regs[rd as usize] = val;
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = regs[*rs1 as usize].map(|a| compute_binary_op(*op, a, u64::from(*shamt)));
            regs[rd as usize] = val;
        }
        Instr::ShiftWImm { op, rd, rs1, shamt } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = regs[*rs1 as usize].map(|a| compute_binary_op_w(*op, a, u64::from(*shamt)));
            regs[rd as usize] = val;
        }
        Instr::Lui { rd, value } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            regs[rd as usize] = Some(sext32(*value));
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
        Instr::MulDivW { op, rd, rs1, rs2 } => {
            let rd = *rd;
            if rd == 0 {
                return;
            }
            let val = match (regs[*rs1 as usize], regs[*rs2 as usize]) {
                (Some(a), Some(b)) => Some(compute_muldiv_op_w(*op, a, b)),
                _ => None,
            };
            regs[rd as usize] = val;
        }
        // IO / system instructions: most don't write general registers
        Instr::Store { .. } | Instr::Nop | Instr::Ext(_) => {}
    }
}

/// Evaluate the JumpDyn target address: (state[rs1] + imm) & !1.
fn simple_eval_jumpdyn(regs: &[Option<u64>; NUM_REGS], rs1: u8, imm: i32) -> Option<u64> {
    regs[rs1 as usize].and_then(|base| eval_jumpdyn_target(base, imm))
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

    let mut regs: [Option<u64>; NUM_REGS] = [None; NUM_REGS];
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
                    if pc_to_idx.contains_key(target) {
                        if is_call(li) {
                            function_entries.insert(*target);
                            let return_pc = pc + INSTR_SIZE as u64;
                            extend_existing_pcs(&mut return_sites, pc_to_idx, [return_pc]);
                        } else {
                            internal_targets.insert(*target);
                        }
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
                        let return_pc = pc + INSTR_SIZE as u64;
                        extend_existing_pcs(&mut return_sites, pc_to_idx, [return_pc]);
                    }
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::Branch { target, .. } => {
                    let fallthrough = pc + INSTR_SIZE as u64;
                    extend_existing_pcs(&mut internal_targets, pc_to_idx, [*target, fallthrough]);
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::Exit { .. } | Terminator::Trap { .. } => {
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
                Terminator::Extension(ext) => {
                    extend_existing_pcs(
                        &mut internal_targets,
                        pc_to_idx,
                        ext.successors(pc + INSTR_SIZE as u64),
                    );
                    regs = [None; NUM_REGS];
                    regs[0] = Some(0);
                }
            },
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
            let return_site = pc + INSTR_SIZE as u64;
            if pc_to_idx.contains_key(target) && pc_to_idx.contains_key(&return_site) {
                map.entry(*target).or_default().insert(return_site);
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
    let mut states: HashMap<u64, RegisterState> = HashMap::with_capacity(estimated_size);
    let mut work: Vec<u64> = Vec::with_capacity(estimated_size);
    let mut in_work: HashSet<u64> = HashSet::with_capacity(estimated_size);
    let mut successors: HashMap<u64, HashSet<u64>> = HashMap::with_capacity(estimated_size);
    let mut unresolved_dynamic_jumps: HashSet<u64> = HashSet::new();
    let mut resolved_jumps: HashMap<u64, HashSet<u64>> = HashMap::new();

    // Internal targets receive state from their real predecessors. An unknown seed would discard
    // constants used to resolve dynamic jumps.
    for addr in function_entries {
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
        Instr::AluWReg { op, rd, rs1, rs2 } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = state.get(*rs2);
            let result = eval_binary_multi_w(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::AluImm { op, rd, rs1, imm } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = RegisterValue::constant(*imm as u64);
            let result = eval_binary_multi(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::AluWImm { op, rd, rs1, imm } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = RegisterValue::constant(*imm as u64);
            let result = eval_binary_multi_w(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::ShiftImm { op, rd, rs1, shamt } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = RegisterValue::constant(u64::from(*shamt));
            let result = eval_binary_multi(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::ShiftWImm { op, rd, rs1, shamt } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = RegisterValue::constant(u64::from(*shamt));
            let result = eval_binary_multi_w(*op, &lv, &rv);
            state.set(*rd, result);
        }
        Instr::Lui { rd, value } => {
            if *rd == 0 {
                return;
            }
            state.set(*rd, RegisterValue::constant(sext32(*value)));
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
        Instr::MulDivW { op, rd, rs1, rs2 } => {
            if *rd == 0 {
                return;
            }
            let lv = state.get(*rs1);
            let rv = state.get(*rs2);
            if lv.is_constant() && rv.is_constant() && lv.values.len() == 1 && rv.values.len() == 1
            {
                let result = compute_muldiv_op_w(*op, lv.values[0], rv.values[0]);
                state.set(*rd, RegisterValue::constant(result));
            } else {
                state.set_unknown(*rd);
            }
        }
        // IO / system instructions: most don't write general registers
        Instr::Store { .. } | Instr::Nop | Instr::Ext(_) => {}
    }
}

// ── Phase 5: get_successors ───────────────────────────────────────────────

fn get_successors(
    li: &LiftedInstr,
    state: &RegisterState,
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
            match terminator {
                Terminator::FallThrough => {
                    let fallthrough = pc + INSTR_SIZE as u64;
                    extend_existing_pcs(&mut result, ctx.pc_to_idx, [fallthrough]);
                }
                Terminator::Jump { target, .. } => {
                    if ctx.pc_to_idx.contains_key(target) {
                        result.insert(*target);
                        let return_pc = pc + INSTR_SIZE as u64;
                        if is_call_instr {
                            extend_existing_pcs(&mut result, ctx.pc_to_idx, [return_pc]);
                        }
                    }
                }
                Terminator::JumpDyn { rs1, imm, .. } => {
                    // Evaluate address: (state[rs1] + imm) & !1
                    let addr_val = eval_jumpdyn_multi(state, *rs1, *imm);

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
                Terminator::Branch { target, .. } => {
                    let fallthrough = pc + INSTR_SIZE as u64;
                    extend_existing_pcs(&mut result, ctx.pc_to_idx, [fallthrough, *target]);
                }
                Terminator::Exit { .. } | Terminator::Trap { .. } => {}
                Terminator::Extension(ext) => {
                    extend_existing_pcs(
                        &mut result,
                        ctx.pc_to_idx,
                        ext.successors(pc + INSTR_SIZE as u64),
                    );
                }
            }
        }
    }

    result
}

/// Compute a single JumpDyn target: `(base + imm) & !1`, returning `None` if
/// the result exceeds the valid PC address space.
fn eval_jumpdyn_target(base: u64, imm: i32) -> Option<u64> {
    let target = base.wrapping_add(imm as i64 as u64) & !1u64;
    is_pc_in_bounds(target).then_some(target)
}

/// Evaluate the JumpDyn target as a multi-value register value:
/// for each possible value of rs1, compute (val + imm) & !1.
fn eval_jumpdyn_multi(state: &RegisterState, rs1: u8, imm: i32) -> RegisterValue {
    let base = state.get(rs1);
    if !base.is_constant() || base.values.is_empty() {
        return RegisterValue::unknown();
    }

    let first = match eval_jumpdyn_target(base.values[0], imm) {
        Some(t) => t,
        None => return RegisterValue::unknown(),
    };
    let mut result = RegisterValue::constant(first);
    for &v in base.values.iter().skip(1) {
        match eval_jumpdyn_target(v, imm) {
            Some(t) => {
                result.add_value(t);
                if !result.is_constant() {
                    return RegisterValue::unknown();
                }
            }
            None => return RegisterValue::unknown(),
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
/// Returns `Vec<Block>` with resolved `JumpDyn` targets filled in.
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

    // Build blocks by splitting at leaders and patching resolved JumpDyn targets.
    Ok(build_block_list(instructions, &leaders, &resolved_jumps))
}

/// Split the flat instruction list into `Block`s at leader boundaries,
/// filling in resolved JumpDyn targets.
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

                // Patch resolved targets into JumpDyn.
                if let Some(targets) = resolved_jumps.get(&pc) {
                    if let Terminator::JumpDyn { resolved, .. } = &mut term {
                        let mut sorted_targets: Vec<u64> = targets.iter().copied().collect();
                        sorted_targets.sort_unstable();
                        *resolved = sorted_targets;
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
    use rvr_openvm_ir::BranchCond;

    use super::*;

    fn term(pc: u64, terminator: Terminator) -> LiftedInstr {
        LiftedInstr::Term {
            pc,
            terminator,
            source_loc: None,
        }
    }

    fn body(pc: u64, instr: Instr) -> LiftedInstr {
        LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        })
    }

    fn block_at(blocks: &[Block], start_pc: u64) -> &Block {
        blocks
            .iter()
            .find(|block| block.start_pc == start_pc)
            .unwrap_or_else(|| panic!("no block starts at {start_pc:#x}"))
    }

    #[test]
    fn invalid_branch_target_preserves_valid_fallthrough() {
        let blocks = build_blocks(
            &[
                term(
                    0,
                    Terminator::Branch {
                        cond: BranchCond::Eq,
                        rs1: 1,
                        rs2: 2,
                        target: 8,
                    },
                ),
                term(4, Terminator::Exit { code: 0 }),
            ],
            &[],
        )
        .expect("invalid branch edge should be handled at runtime");

        assert!(matches!(
            block_at(&blocks, 0).terminator,
            Terminator::Branch { target: 8, .. }
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
                Instr::AluImm {
                    op: AluOp::Add,
                    rd: 5,
                    rs1: 0,
                    imm: 1,
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
                    Instr::AluImm {
                        op: AluOp::Add,
                        rd: 5,
                        rs1: 0,
                        imm: 16,
                    },
                ),
                term(
                    4,
                    Terminator::JumpDyn {
                        link_rd: None,
                        rs1: 5,
                        imm: 0,
                        resolved: Vec::new(),
                    },
                ),
                body(8, Instr::Nop),
                body(12, Instr::Nop),
                term(16, Terminator::Exit { code: 0 }),
            ],
            &[4],
        )
        .expect("cfg build");

        assert!(matches!(
            &block_at(&blocks, 4).terminator,
            Terminator::JumpDyn { resolved, .. } if resolved == &[16]
        ));
    }
}
