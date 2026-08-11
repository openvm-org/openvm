use std::sync::Arc;

use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerChip},
    ColumnsAir, SubAir, TraceSubRowGenerator,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::builder::{ExprBuilder, FieldExpr, FieldExprCols, FieldExpressionProgram};

fn normalize_opcode_flags(
    needs_setup: bool,
    local_opcode_idx: &[usize],
    opcode_flag_idx: Vec<usize>,
) -> Vec<usize> {
    let opcode_flag_idx = if opcode_flag_idx.is_empty() && needs_setup {
        vec![0]
    } else {
        opcode_flag_idx
    };
    assert_eq!(opcode_flag_idx.len(), local_opcode_idx.len() - 1);
    opcode_flag_idx
}

pub(crate) fn is_valid_opcode_flag_layout(
    needs_setup: bool,
    num_flags: usize,
    local_opcodes: &[usize],
    opcode_flags: &[usize],
) -> bool {
    let valid_shape = if needs_setup {
        !local_opcodes.is_empty() && opcode_flags.len() + 1 == local_opcodes.len()
    } else {
        local_opcodes.len() == 1 && opcode_flags.is_empty() && num_flags == 0
    };
    let local_opcodes_are_unique = local_opcodes
        .iter()
        .enumerate()
        .all(|(index, opcode)| !local_opcodes[..index].contains(opcode));
    let opcode_flags_are_valid = opcode_flags.iter().all(|&flag| flag < num_flags)
        && opcode_flags
            .iter()
            .enumerate()
            .all(|(index, flag)| !opcode_flags[..index].contains(flag));

    valid_shape && local_opcodes_are_unique && opcode_flags_are_valid
}

#[derive(Clone)]
pub struct FieldExpressionCoreAir {
    pub expr: FieldExpr,

    /// The global opcode offset.
    pub offset: usize,

    /// All the opcode indices (including setup) supported by this Air.
    /// The last one must be the setup opcode if it's a chip needs setup.
    pub local_opcode_idx: Vec<usize>,
    /// Opcode flag idx (indices from builder.new_flag()) for all except setup opcode. Empty if
    /// single op chip.
    pub opcode_flag_idx: Vec<usize>,
    // Example 1: 1-op chip EcAdd that needs setup
    //   local_opcode_idx = [0, 2], where 0 is EcAdd, 2 is setup
    //   opcode_flag_idx = [], not needed for single op chip.
    // Example 2: 1-op chip EvaluateLine that doesn't need setup
    //   local_opcode_idx = [2], the id within PairingOpcodeEnum
    //   opcode_flag_idx = [], not needed
    // Example 3: 2-op chip MulDiv that needs setup
    //   local_opcode_idx = [2, 3, 4], where 2 is Mul, 3 is Div, 4 is setup
    //   opcode_flag_idx = [0, 1], where 0 is mul_flag, 1 is div_flag, in the builder
    // We don't support 2-op chip that doesn't need setup right now.
}

// No columns provided: wraps `FieldExpr`, whose column layout is built dynamically.
impl ColumnsAir for FieldExpressionCoreAir {}

impl FieldExpressionCoreAir {
    pub fn new(
        expr: FieldExpr,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
    ) -> Self {
        let opcode_flag_idx = normalize_opcode_flags(
            expr.program().needs_setup(),
            &local_opcode_idx,
            opcode_flag_idx,
        );
        Self {
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.expr.program().num_inputs()
    }

    pub fn num_vars(&self) -> usize {
        self.expr.program().num_vars()
    }

    pub fn num_flags(&self) -> usize {
        self.expr.program().num_flags()
    }

    pub fn output_indices(&self) -> &[usize] {
        self.expr.program().output_indices()
    }
}

impl<F: Field> BaseAir<F> for FieldExpressionCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldExpressionCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for FieldExpressionCoreAir
where
    I: VmAdapterInterface<AB::Expr>,
    AdapterAirContext<AB::Expr, I>:
        From<AdapterAirContext<AB::Expr, DynAdapterInterface<AB::Expr>>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        assert_eq!(local.len(), BaseAir::<AB::F>::width(&self.expr));
        self.expr.eval(builder, local);
        let FieldExprCols {
            is_valid,
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local);
        assert_eq!(inputs.len(), self.num_inputs());
        assert_eq!(vars.len(), self.num_vars());
        assert_eq!(flags.len(), self.num_flags());
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = self
            .output_indices()
            .iter()
            .flat_map(|&i| vars[i].clone())
            .map(Into::into)
            .collect();

        let opcode_flags_except_last = self.opcode_flag_idx.iter().map(|&i| flags[i]).collect_vec();
        let last_opcode_flag = is_valid
            - opcode_flags_except_last
                .iter()
                .map(|&v| v.into())
                .sum::<AB::Expr>();
        builder.assert_bool(last_opcode_flag.clone());
        let opcode_flags = opcode_flags_except_last
            .into_iter()
            .map(Into::into)
            .chain(Some(last_opcode_flag));
        let expected_opcode = opcode_flags
            .zip(self.local_opcode_idx.iter().map(|&i| i + self.offset))
            .map(|(flag, global_idx)| flag * AB::Expr::from_usize(global_idx))
            .sum();

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: expected_opcode,
        };

        let ctx: AdapterAirContext<_, DynAdapterInterface<_>> = AdapterAirContext {
            to_pc: None,
            reads: reads.into(),
            writes: writes.into(),
            instruction: instruction.into(),
        };
        ctx.into()
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct FieldExpressionExecutor {
    program: FieldExpressionProgram,
    pub offset: usize,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    pub name: String,
}

impl FieldExpressionExecutor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program: FieldExpressionProgram,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        name: &str,
    ) -> Self {
        let opcode_flag_idx =
            normalize_opcode_flags(program.needs_setup(), &local_opcode_idx, opcode_flag_idx);
        tracing::debug!(
            "FieldExpressionCoreExecutor: opcode={name}, main_width={}",
            program.width()
        );
        Self {
            program,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            name: name.to_string(),
        }
    }

    pub fn program(&self) -> &FieldExpressionProgram {
        &self.program
    }
}

pub struct FieldExpressionFiller<A> {
    adapter: A,
    pub expr: FieldExpr,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub should_finalize: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldExpressionTraceError {
    InvalidLocalOpcode(usize),
    InvalidInputLength { expected: usize, actual: usize },
    InvalidFlagLayout,
    InvalidFlagIndex(usize),
    InvalidSetupInput,
    InvalidProgramOutput(usize),
    InvalidVariableCount { expected: usize, actual: usize },
    OutputMismatch,
    ProgramTooLarge,
    UnsupportedDeviceProgram(&'static str),
}

impl<A> FieldExpressionFiller<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter: A,
        expr: FieldExpr,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        range_checker: SharedVariableRangeCheckerChip,
        should_finalize: bool,
    ) -> Self {
        let opcode_flag_idx = normalize_opcode_flags(
            expr.program().needs_setup(),
            &local_opcode_idx,
            opcode_flag_idx,
        );
        Self {
            adapter,
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            should_finalize,
        }
    }
    pub fn num_inputs(&self) -> usize {
        self.expr.program().num_inputs()
    }

    pub fn num_flags(&self) -> usize {
        self.expr.program().num_flags()
    }

    pub fn adapter(&self) -> &A {
        &self.adapter
    }
}

impl<A> FieldExpressionFiller<A> {
    pub fn fill_dummy_core_row<F: PrimeField32>(&self, core_row: &mut [F]) {
        if !self.should_finalize {
            return;
        }

        let inputs = vec![BigUint::zero(); self.num_inputs()];
        let flags = vec![false; self.num_flags()];
        let range_checker = Arc::new(VariableRangeCheckerChip::new(self.range_checker.bus()));
        self.expr
            .generate_subrow((&range_checker, inputs, flags), core_row);
        core_row[0] = F::ZERO;
    }

    /// Replays one field expression directly from semantic execution values.
    ///
    /// When `logged_output` is present, it is checked before the range checker or
    /// trace row is mutated. Checkpoint callers can therefore fill against a
    /// temporary range checker and merge its counts only after every row passes.
    pub fn fill_trace_row_from_execution_data<F: PrimeField32 + Send + Sync + Clone>(
        &self,
        range_checker: &VariableRangeCheckerChip,
        local_opcode: usize,
        input_limbs: &[u8],
        logged_output: Option<&[u8]>,
        core_row: &mut [F],
    ) -> Result<(), FieldExpressionTraceError> {
        if !self.local_opcode_idx.contains(&local_opcode) {
            return Err(FieldExpressionTraceError::InvalidLocalOpcode(local_opcode));
        }
        let FieldExpressionRun {
            writes,
            inputs,
            flags,
            vars,
        } = run_field_expression_checked(
            self.expr.program(),
            &self.local_opcode_idx,
            &self.opcode_flag_idx,
            input_limbs,
            local_opcode,
        )?;
        if logged_output.is_some_and(|logged| logged != writes.0) {
            return Err(FieldExpressionTraceError::OutputMismatch);
        }
        self.expr
            .generate_subrow_from_vars(range_checker, inputs, flags, vars, core_row);
        Ok(())
    }
}

struct FieldExpressionRun {
    writes: DynArray<u8>,
    inputs: Vec<BigUint>,
    flags: Vec<bool>,
    vars: Vec<BigUint>,
}

fn run_field_expression_checked(
    program: &FieldExpressionProgram,
    local_opcode_flags: &[usize],
    opcode_flag_idx: &[usize],
    data: &[u8],
    local_opcode_idx: usize,
) -> Result<FieldExpressionRun, FieldExpressionTraceError> {
    let field_element_limbs = program.canonical_num_limbs();
    let expected_len = program
        .num_inputs()
        .checked_mul(field_element_limbs)
        .ok_or(FieldExpressionTraceError::InvalidInputLength {
            expected: usize::MAX,
            actual: data.len(),
        })?;
    if data.len() != expected_len {
        return Err(FieldExpressionTraceError::InvalidInputLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    let mut inputs = Vec::with_capacity(program.num_inputs());
    for i in 0..program.num_inputs() {
        let start = i * field_element_limbs;
        let end = start + field_element_limbs;
        let limb_slice = &data[start..end];
        let input = BigUint::from_bytes_le(limb_slice);
        inputs.push(input);
    }

    let mut flags = vec![];
    if program.needs_setup() {
        if !is_valid_opcode_flag_layout(
            true,
            program.num_flags(),
            local_opcode_flags,
            opcode_flag_idx,
        ) {
            return Err(FieldExpressionTraceError::InvalidFlagLayout);
        }
        flags = vec![false; program.num_flags()];

        // Find which opcode this is in our local_opcode_idx list
        if let Some(opcode_position) = local_opcode_flags
            .iter()
            .position(|&idx| idx == local_opcode_idx)
        {
            // If this is NOT the last opcode (setup), set the corresponding flag
            if opcode_position < opcode_flag_idx.len() {
                let flag_idx = opcode_flag_idx[opcode_position];
                if flag_idx >= flags.len() {
                    return Err(FieldExpressionTraceError::InvalidFlagIndex(flag_idx));
                }
                flags[flag_idx] = true;
            }
            // If opcode_position == step.opcode_flag_idx.len(), it's the setup operation
            // and all flags should remain false (which they already are)
        }
    }

    let is_setup = program.needs_setup() && flags.iter().all(|&flag| !flag);
    if is_setup && !program.setup_inputs_are_valid(&inputs) {
        return Err(FieldExpressionTraceError::InvalidSetupInput);
    }
    let vars = program.execute(&inputs, &flags);
    if vars.len() != program.num_vars() {
        return Err(FieldExpressionTraceError::InvalidVariableCount {
            expected: program.num_vars(),
            actual: vars.len(),
        });
    }

    // Write outputs directly to a pre-allocated buffer to avoid intermediate Vecs
    let num_outputs = program.output_indices().len();
    let total_output_bytes = num_outputs
        .checked_mul(field_element_limbs)
        .ok_or(FieldExpressionTraceError::InvalidProgramOutput(usize::MAX))?;
    let mut write_buffer = vec![0u8; total_output_bytes];
    for (i, &var_idx) in program.output_indices().iter().enumerate() {
        let Some(var) = vars.get(var_idx) else {
            return Err(FieldExpressionTraceError::InvalidProgramOutput(var_idx));
        };
        let start = i * field_element_limbs;
        let bytes = var.to_bytes_le();
        let copy_len = bytes.len().min(field_element_limbs);
        write_buffer[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
        // Remaining bytes are already zero from vec![0u8; ...]
    }
    let writes: DynArray<_> = write_buffer.into();

    Ok(FieldExpressionRun {
        writes,
        inputs,
        flags,
        vars,
    })
}

fn decode_precomputed_inputs<const NEEDS_SETUP: bool>(
    program: &FieldExpressionProgram,
    flag_idx: usize,
    data: &[u8],
) -> (Vec<BigUint>, Vec<bool>) {
    debug_assert_eq!(NEEDS_SETUP, program.needs_setup());
    let builder = program.builder();
    let field_element_limbs = builder.num_limbs;
    assert_eq!(data.len(), builder.num_input * field_element_limbs);

    let mut inputs = Vec::with_capacity(builder.num_input);
    for i in 0..builder.num_input {
        let start = i * field_element_limbs;
        let end = start + field_element_limbs;
        let limb_slice = &data[start..end];
        let input = BigUint::from_bytes_le(limb_slice);
        inputs.push(input);
    }

    let flags = if NEEDS_SETUP {
        let mut flags = vec![false; builder.num_flags];
        if flag_idx < builder.num_flags {
            flags[flag_idx] = true;
        }
        flags
    } else {
        vec![]
    };
    (inputs, flags)
}

fn encode_precomputed_outputs(builder: &ExprBuilder, vars: &[BigUint]) -> DynArray<u8> {
    assert_eq!(vars.len(), builder.num_variables);
    let total_output_bytes = builder.output_indices.len() * builder.num_limbs;
    let mut write_buffer = vec![0u8; total_output_bytes];
    for (i, &var_idx) in builder.output_indices.iter().enumerate() {
        let start = i * builder.num_limbs;
        let bytes = vars[var_idx].to_bytes_le();
        let copy_len = bytes.len().min(builder.num_limbs);
        write_buffer[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
    }
    write_buffer.into()
}

#[inline(always)]
pub fn run_field_expression_precomputed<const NEEDS_SETUP: bool>(
    program: &FieldExpressionProgram,
    flag_idx: usize,
    data: &[u8],
) -> DynArray<u8> {
    let (inputs, flags) = decode_precomputed_inputs::<NEEDS_SETUP>(program, flag_idx, data);
    let vars = program.execute(&inputs, &flags);
    encode_precomputed_outputs(program.builder(), &vars)
}
