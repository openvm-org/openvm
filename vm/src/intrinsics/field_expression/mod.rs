use std::sync::Arc;

use ax_circuit_primitives::{var_range::VariableRangeCheckerChip, SubAir, TraceSubRowGenerator};
use ax_ecc_primitives::field_expression::{FieldExpr, FieldExprCols};
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::instruction::Instruction;
use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray,
        MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
};
#[derive(Clone)]
pub struct FieldExpressionCoreAir {
    pub expr: FieldExpr,
    // global opcode offset
    pub offset: usize,

    // The opcodes handled by this air
    // Assumptions:
    // 1. local_opcode_indices is of size 1 or 2.
    // 2. When it's size 2, there is exactly one flag to tell us which opcode to use.
    //    and otherwise there is no flag.
    pub local_opcode_indices: Vec<usize>,
}

impl FieldExpressionCoreAir {
    pub fn new(expr: FieldExpr, offset: usize, local_opcode_indices: Vec<usize>) -> Self {
        Self {
            expr,
            offset,
            local_opcode_indices,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.expr.builder.num_input
    }

    pub fn num_vars(&self) -> usize {
        self.expr.builder.num_variables
    }

    pub fn num_flags(&self) -> usize {
        self.expr.builder.num_flags
    }

    pub fn output_indices(&self) -> &[usize] {
        &self.expr.builder.output_indices
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
            .map(|&i| vars[i].clone())
            .collect::<Vec<_>>()
            .concat()
            .iter()
            .map(|x| (*x).into())
            .collect();

        // no flags -> local_opcode_indices[0]
        // flag = 1 -> local_opcode_indices[0]
        // flag = 0 -> local_opcode_indices[1]
        let opcode_index_to_use = if flags.is_empty() {
            AB::Expr::zero()
        } else {
            AB::Expr::one() - flags[0]
        };
        let second_opcode = if self.local_opcode_indices.len() > 1 {
            self.local_opcode_indices[1]
        } else {
            0 // value doesn't matter as it will times a zero
        };
        let opcode_index = AB::Expr::from_canonical_usize(second_opcode)
            * opcode_index_to_use.clone()
            + AB::Expr::from_canonical_usize(self.local_opcode_indices[0])
                * (AB::Expr::one() - opcode_index_to_use);
        let expected_opcode = AB::Expr::from_canonical_usize(self.offset) + opcode_index;

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
}

pub struct FieldExpressionRecord {
    pub inputs: Vec<BigUint>,
}

pub struct FieldExpressionCoreChip {
    pub air: FieldExpressionCoreAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,

    pub name: String,
}

impl FieldExpressionCoreChip {
    pub fn new(
        expr: FieldExpr,
        offset: usize,
        local_opcode_indices: Vec<usize>,
        range_checker: Arc<VariableRangeCheckerChip>,
        name: &str,
    ) -> Self {
        let air = FieldExpressionCoreAir {
            expr,
            offset,
            local_opcode_indices,
        };
        Self {
            air,
            range_checker,
            name: name.to_string(),
        }
    }
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for FieldExpressionCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = FieldExpressionRecord;
    type Air = FieldExpressionCoreAir;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let field_element_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        let data: DynArray<_> = reads.into();
        let data = data.0;
        assert_eq!(data.len(), self.air.num_inputs() * field_element_limbs);
        let data_u32: Vec<u32> = data.iter().map(|x| x.as_canonical_u32()).collect();

        let mut inputs = vec![];
        for i in 0..self.air.num_inputs() {
            let start = i * field_element_limbs;
            let end = start + field_element_limbs;
            let limb_slice = &data_u32[start..end];
            let input = limbs_to_biguint(limb_slice, limb_bits);
            inputs.push(input);
        }

        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_index = opcode - self.air.offset;
        let flags = if self.air.local_opcode_indices.len() == 1 {
            vec![]
        } else {
            vec![local_opcode_index == self.air.local_opcode_indices[0]]
        };
        assert_eq!(flags.len(), self.air.num_flags());

        let vars = self.air.expr.execute(inputs.clone(), flags);
        assert_eq!(vars.len(), self.air.num_vars());

        let outputs: Vec<BigUint> = self
            .air
            .output_indices()
            .iter()
            .map(|&i| vars[i].clone())
            .collect();
        let writes: Vec<F> = outputs
            .iter()
            .map(|x| biguint_to_limbs_vec(x.clone(), limb_bits, field_element_limbs))
            .concat()
            .into_iter()
            .map(|x| F::from_canonical_u32(x))
            .collect();

        let ctx = AdapterRuntimeContext::<_, DynAdapterInterface<_>>::without_pc(writes);
        Ok((ctx.into(), FieldExpressionRecord { inputs }))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.name.clone()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        self.air
            .expr
            .generate_subrow((&self.range_checker, record.inputs, vec![]), row_slice);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
