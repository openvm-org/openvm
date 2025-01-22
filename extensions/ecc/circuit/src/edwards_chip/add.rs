use std::{cell::RefCell, iter, rc::Rc};

use itertools::{zip_eq, Itertools};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray, MinimalInstruction,
    Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_num_limbs,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    SubAir, TraceSubRowGenerator,
};
use openvm_ecc_transpiler::Rv32EdwardsOpcode;
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
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};

// We do not use FieldExpressionCoreAir because EcDouble needs to do special constraints for
// its setup instruction.

#[derive(Clone)]
pub struct TeEcAddCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
    pub a_biguint: BigUint,
    pub d_biguint: BigUint,
}

impl TeEcAddCoreAir {
    pub fn new(
        config: ExprBuilderConfig,
        range_bus: VariableRangeCheckerBus,
        a_biguint: BigUint,
        d_biguint: BigUint,
        offset: usize,
    ) -> Self {
        config.check_valid();
        let builder = ExprBuilder::new(config, range_bus.range_max_bits);
        let builder = Rc::new(RefCell::new(builder));

        let x1 = ExprBuilder::new_input(builder.clone());
        let y1 = ExprBuilder::new_input(builder.clone());
        let x2 = ExprBuilder::new_input(builder.clone());
        let y2 = ExprBuilder::new_input(builder.clone());
        let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
        let d = ExprBuilder::new_const(builder.clone(), d_biguint.clone());
        let one = ExprBuilder::new_const(builder.clone(), BigUint::one());

        let x1y2 = x1.clone() * y2.clone();
        let x2y1 = x2.clone() * y1.clone();
        let y1y2 = y1 * y2;
        let x1x2 = x1 * x2;
        let dx1x2y1y2 = d * x1x2.clone() * y1y2.clone();

        let mut x3 = (x1y2 + x2y1) / (one.clone() + dx1x2y1y2.clone());
        let mut y3 = (y1y2 - a * x1x2) / (one - dx1x2y1y2);

        x3.save_output();
        y3.save_output();

        let builder = builder.borrow().clone();

        let expr = FieldExpr::new(
            builder, range_bus, true,
            //vec![a_biguint.clone(), d_biguint.clone()],
        );
        Self {
            expr,
            offset,
            a_biguint,
            d_biguint,
        }
    }

    pub fn output_indices(&self) -> &[usize] {
        &self.expr.output_indices
    }
}

impl<F: Field> BaseAir<F> for TeEcAddCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for TeEcAddCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for TeEcAddCoreAir
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
        assert_eq!(inputs.len(), 4);
        assert_eq!(vars.len(), 12);
        assert_eq!(flags.len(), 1);

        let reads: Vec<AB::Expr> = inputs
            .clone()
            .into_iter()
            .flatten()
            .map(Into::into)
            .collect();
        let writes: Vec<AB::Expr> = self
            .output_indices()
            .iter()
            .flat_map(|&i| vars[i].clone())
            .map(Into::into)
            .collect();

        let is_setup = is_valid - flags[0];
        builder.assert_bool(is_setup.clone());
        let local_opcode_idx = flags[0]
            * AB::Expr::from_canonical_usize(Rv32EdwardsOpcode::EC_ADD as usize)
            + is_setup.clone()
                * AB::Expr::from_canonical_usize(Rv32EdwardsOpcode::SETUP_EC_ADD as usize);

        // when is_setup, assert `reads` equals `(modulus, a, d, 0)`
        for (lhs, &rhs) in zip_eq(
            &reads,
            iter::empty()
                .chain(&self.expr.builder.prime_limbs)
                .chain(&big_uint_to_num_limbs(
                    &self.a_biguint,
                    self.expr.builder.limb_bits,
                    self.expr.builder.num_limbs,
                ))
                .chain(&big_uint_to_num_limbs(
                    &self.d_biguint,
                    self.expr.builder.limb_bits,
                    self.expr.builder.num_limbs,
                ))
                .chain(&big_uint_to_num_limbs(
                    &BigUint::zero(),
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

    fn start_offset(&self) -> usize {
        self.offset
    }
}

pub struct TeEcAddCoreChip {
    pub air: TeEcAddCoreAir,
    pub range_checker: SharedVariableRangeCheckerChip,
}

impl TeEcAddCoreChip {
    pub fn new(
        config: ExprBuilderConfig,
        range_checker: SharedVariableRangeCheckerChip,
        a_biguint: BigUint,
        d_biguint: BigUint,
        offset: usize,
    ) -> Self {
        let air = TeEcAddCoreAir::new(config, range_checker.bus(), a_biguint, d_biguint, offset);
        Self { air, range_checker }
    }
}

#[serde_as]
#[derive(Clone, Serialize, Deserialize)]
pub struct TeEcAddCoreRecord {
    #[serde_as(as = "DisplayFromStr")]
    pub x1: BigUint,
    #[serde_as(as = "DisplayFromStr")]
    pub y1: BigUint,
    #[serde_as(as = "DisplayFromStr")]
    pub x2: BigUint,
    #[serde_as(as = "DisplayFromStr")]
    pub y2: BigUint,
    pub is_add_flag: bool,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for TeEcAddCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = TeEcAddCoreRecord;
    type Air = TeEcAddCoreAir;

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
        debug_assert_eq!(data.len(), 4 * num_limbs);

        let x1 = data[..num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let y1 = data[num_limbs..2 * num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let x2 = data[2 * num_limbs..3 * num_limbs]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();
        let y2 = data[3 * num_limbs..]
            .iter()
            .map(|x| x.as_canonical_u32())
            .collect_vec();

        let x1_biguint = limbs_to_biguint(&x1, limb_bits);
        let y1_biguint = limbs_to_biguint(&y1, limb_bits);
        let x2_biguint = limbs_to_biguint(&x2, limb_bits);
        let y2_biguint = limbs_to_biguint(&y2, limb_bits);

        let is_add_flag = local_opcode_idx == Rv32EdwardsOpcode::EC_ADD as usize;

        let vars = self.air.expr.execute(
            vec![
                x1_biguint.clone(),
                y1_biguint.clone(),
                x2_biguint.clone(),
                y2_biguint.clone(),
            ],
            vec![is_add_flag],
        );
        assert_eq!(vars.len(), 12);

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
            TeEcAddCoreRecord {
                x1: x1_biguint,
                y1: y1_biguint,
                x2: x2_biguint,
                y2: y2_biguint,
                is_add_flag,
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "TeEcAdd".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        self.air.expr.generate_subrow(
            (
                self.range_checker.as_ref(),
                vec![record.x1, record.y1, record.x2, record.y2],
                vec![record.is_add_flag],
            ),
            row_slice,
        );
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }

    // We need finalize for add, since it will have constants (a and d)
    fn finalize(&self, trace: &mut RowMajorMatrix<F>, num_records: usize) {
        if num_records == 0 {
            return;
        }
        let core_width = <Self::Air as BaseAir<F>>::width(&self.air);
        let adapter_width = trace.width() - core_width;
        let dummy_row = self.generate_dummy_trace_row(adapter_width, core_width);
        for row in trace.rows_mut().skip(num_records) {
            row.copy_from_slice(&dummy_row);
        }
    }
}

impl TeEcAddCoreChip {
    // We will be setting is_valid = 0. That forces is_add to be 0 (otherwise setup will be -1).
    // We generate a dummy row with is_add = 0, then we set is_valid = 0.
    fn generate_dummy_trace_row<F: PrimeField32>(
        &self,
        adapter_width: usize,
        core_width: usize,
    ) -> Vec<F> {
        let record = TeEcAddCoreRecord {
            x1: BigUint::zero(),
            y1: BigUint::zero(),
            x2: BigUint::zero(),
            y2: BigUint::zero(),
            is_add_flag: false,
        };
        let mut row = vec![F::ZERO; adapter_width + core_width];
        let core_row = &mut row[adapter_width..];
        // We **do not** want this trace row to update the range checker
        // so we must create a temporary range checker
        let tmp_range_checker = SharedVariableRangeCheckerChip::new(self.range_checker.bus());
        self.air.expr.generate_subrow(
            (
                tmp_range_checker.as_ref(),
                vec![record.x1, record.y1, record.x2, record.y2],
                vec![record.is_add_flag],
            ),
            core_row,
        );
        core_row[0] = F::ZERO; // is_valid = 0
        row
    }
}
