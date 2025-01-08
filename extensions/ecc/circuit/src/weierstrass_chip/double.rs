use std::{cell::RefCell, iter, rc::Rc, sync::Arc};

use itertools::{zip_eq, Itertools};
use num_bigint_dig::BigUint;
use num_traits::One;
use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray, MinimalInstruction,
    Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_num_limbs,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::instruction::Instruction;
use openvm_mod_circuit_builder::{
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExprCols, FieldVariable,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    rap::BaseAirWithPublicValues,
};

// We do not use FieldExpressionCoreAir because EcDouble needs to do special constraints for
// its setup instruction.

#[derive(Clone)]
pub struct EcDoubleCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
    pub a_biguint: BigUint,
}

impl EcDoubleCoreAir {
    pub fn new(
        config: ExprBuilderConfig,
        range_bus: VariableRangeCheckerBus,
        a_biguint: BigUint,
        offset: usize,
    ) -> Self {
        config.check_valid();
        let builder = ExprBuilder::new(config, range_bus.range_max_bits);
        let builder = Rc::new(RefCell::new(builder));

        let mut x1 = ExprBuilder::new_input(builder.clone());
        let mut y1 = ExprBuilder::new_input(builder.clone());
        let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
        let is_double_flag = builder.borrow_mut().new_flag();
        // We need to prevent divide by zero when not double flag
        // (equivalently, when it is the setup opcode)
        let lambda_denom = FieldVariable::select(
            is_double_flag,
            &y1.int_mul(2),
            &ExprBuilder::new_const(builder.clone(), BigUint::one()),
        );
        let mut lambda = (x1.square().int_mul(3) + a) / lambda_denom;
        let mut x3 = lambda.square() - x1.int_mul(2);
        x3.save_output();
        let mut y3 = lambda * (x1 - x3.clone()) - y1;
        y3.save_output();

        let builder = builder.borrow().clone();
        let expr = FieldExpr::new(builder, range_bus, true);
        Self {
            expr,
            offset,
            a_biguint,
        }
    }

    pub fn output_indices(&self) -> &[usize] {
        &self.expr.output_indices
    }
}

impl<F: Field> BaseAir<F> for EcDoubleCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for EcDoubleCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for EcDoubleCoreAir
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
        assert_eq!(inputs.len(), 2);
        assert_eq!(vars.len(), 3); // x1^2, x3, y3
        assert_eq!(flags.len(), 1); // is_double_flag

        let reads: Vec<AB::Expr> = inputs.into_iter().flatten().map(Into::into).collect();
        let writes: Vec<AB::Expr> = self
            .output_indices()
            .iter()
            .flat_map(|&i| vars[i].clone())
            .map(Into::into)
            .collect();

        let is_setup = is_valid - flags[0];
        builder.assert_bool(is_setup.clone());
        let local_opcode_idx = flags[0]
            * AB::Expr::from_canonical_usize(Rv32WeierstrassOpcode::EC_DOUBLE as usize)
            + is_setup.clone()
                * AB::Expr::from_canonical_usize(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize);
        // when is_setup, assert `reads` equals `(modulus, a)`
        for (lhs, &rhs) in zip_eq(
            &reads,
            iter::empty()
                .chain(&self.expr.builder.prime_limbs)
                .chain(&big_uint_to_num_limbs(
                    &self.a_biguint,
                    self.expr.builder.limb_bits,
                    self.expr.builder.num_limbs,
                )),
        ) {
            builder
                .when(is_setup.clone())
                .assert_eq(lhs.clone(), AB::F::from_canonical_usize(rhs));
        }

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: local_opcode_idx + AB::Expr::from_canonical_usize(self.offset),
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

pub struct EcDoubleCoreChip {
    pub air: EcDoubleCoreAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl EcDoubleCoreChip {
    pub fn new(
        config: ExprBuilderConfig,
        range_checker: Arc<VariableRangeCheckerChip>,
        a_biguint: BigUint,
        offset: usize,
    ) -> Self {
        let air = EcDoubleCoreAir::new(config, range_checker.bus(), a_biguint, offset);
        Self { air, range_checker }
    }
}

#[derive(Clone)]
pub struct EcDoubleCoreRecord {
    pub x: BigUint,
    pub y: BigUint,
    pub is_double_flag: bool,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for EcDoubleCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = EcDoubleCoreRecord;
    type Air = EcDoubleCoreAir;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let num_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        let Instruction { opcode, .. } = instruction.clone();
        let local_opcode_idx = opcode.local_opcode_idx(self.air.offset);
        let data: DynArray<_> = reads.into();
        let data = data.0;
        debug_assert_eq!(data.len(), 2 * num_limbs);

        let x = data[..num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let y = data[num_limbs..]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();

        let x_biguint = limbs_to_biguint(&x, limb_bits);
        let y_biguint = limbs_to_biguint(&y, limb_bits);

        let is_double_flag = local_opcode_idx == Rv32WeierstrassOpcode::EC_DOUBLE as usize;

        let vars = self.air.expr.execute(
            vec![x_biguint.clone(), y_biguint.clone()],
            vec![is_double_flag],
        );
        assert_eq!(vars.len(), 3); // x1^2, x3, y3

        let writes = self
            .air
            .output_indices()
            .iter()
            .flat_map(|&i| {
                let limbs = biguint_to_limbs_vec(vars[i].clone(), limb_bits, num_limbs);
                limbs.into_iter().map(F::from_canonical_u32)
            })
            .collect_vec();

        let ctx = AdapterRuntimeContext::<_, DynAdapterInterface<_>>::without_pc(writes);

        Ok((
            ctx.into(),
            EcDoubleCoreRecord {
                x: x_biguint,
                y: y_biguint,
                is_double_flag,
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "EcDouble".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        self.air.expr.generate_subrow(
            (
                &self.range_checker,
                vec![record.x, record.y],
                vec![record.is_double_flag],
            ),
            row_slice,
        );
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }

    // We need finalize for double, as it might have a constant (a of y^2 = x^3 + ax + b)
    fn finalize(&self, trace: &mut RowMajorMatrix<F>, num_records: usize) {
        if num_records == 0 {
            return;
        }
        let core_width = <Self::Air as BaseAir<F>>::width(&self.air);
        let adapter_width = trace.width() - core_width;
        // We will be setting is_valid = 0. That forces is_double to be 0 (otherwise setup will be -1).
        // So the computation is like doing setup.
        // Thus we will copy over the first row (which is a setup row) and set is_valid = 0.
        let first_row = trace.rows().nth(0).unwrap().collect::<Vec<_>>();
        let first_row_core = first_row.split_at(adapter_width).1;
        for row in trace.rows_mut().skip(num_records) {
            let core_row = row.split_at_mut(adapter_width).1;
            core_row.copy_from_slice(first_row_core);
            core_row[0] = F::ZERO; // is_valid = 0
        }
    }
}
