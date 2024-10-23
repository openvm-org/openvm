use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    sub_chip::{LocalTraceInstructions, SubAir},
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr, FieldExprCols},
    field_extension::{Fp12, Fp2},
};
use axvm_instructions::FP12Opcode;
use itertools::Itertools;
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::super::FIELD_ELEMENT_BITS;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray,
        MinimalInstruction, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    intrinsics::ecc::{Fp12BigUint, Fp2BigUint, FpBigUint},
    system::program::Instruction,
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
};

#[derive(Clone)]
pub struct Fp12MultiplyCoreAir {
    pub expr: FieldExpr,
    pub offset: usize,
}

impl Fp12MultiplyCoreAir {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        max_limb_bits: usize,
        range_bus: VariableRangeCheckerBus,
        offset: usize,
    ) -> Self {
        assert!(modulus.bits() <= num_limbs * limb_bits);
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            limb_bits,
            range_bus.index,
            range_bus.range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        let builder = ExprBuilder::new(
            modulus,
            limb_bits,
            num_limbs,
            range_bus.range_max_bits,
            max_limb_bits,
        );
        let builder = Rc::new(RefCell::new(builder));

        let mut x = Fp12::new(builder.clone());
        let mut y = Fp12::new(builder.clone());
        let mut xi = x.xi;
        let mut res = x.mul(&mut y, &mut xi);
        res.save();

        let builder = builder.borrow().clone();
        let expr = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
        };
        Self { expr, offset }
    }
}

impl<F: Field> BaseAir<F> for Fp12MultiplyCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Fp12MultiplyCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for Fp12MultiplyCoreAir
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
        SubAir::eval(&self.expr, builder, local.to_vec(), ());

        let FieldExprCols {
            is_valid,
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local);
        assert_eq!(inputs.len(), 12 + 12 + 2);
        assert_eq!(vars.len(), 34);
        assert_eq!(flags.len(), 0);
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = vars[vars.len() - 12..]
            .concat()
            .iter()
            .map(|x| (*x).into())
            .collect();

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: AB::Expr::from_canonical_usize(self.offset),
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

pub struct Fp12MultiplyCoreChip {
    pub air: Fp12MultiplyCoreAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl Fp12MultiplyCoreChip {
    pub fn new(
        modulus: BigUint,
        num_limbs: usize,
        limb_bits: usize,
        max_limb_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
        offset: usize,
    ) -> Self {
        let air = Fp12MultiplyCoreAir::new(
            modulus,
            num_limbs,
            limb_bits,
            max_limb_bits,
            range_checker.bus(),
            offset,
        );
        Self { air, range_checker }
    }
}

pub struct Fp12MultiplyCoreRecord {
    pub x: Fp12BigUint,
    pub y: Fp12BigUint,
    pub xi: Fp2BigUint,
}

impl<F: PrimeField32, I> VmCoreChip<F, I> for Fp12MultiplyCoreChip
where
    I: VmAdapterInterface<F>,
    I::Reads: Into<DynArray<F>>,
    I::Writes: From<DynArray<F>>,
    AdapterRuntimeContext<F, I>: From<AdapterRuntimeContext<F, DynAdapterInterface<F>>>,
{
    type Record = Fp12MultiplyCoreRecord;
    type Air = Fp12MultiplyCoreAir;

    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let num_limbs = self.air.expr.canonical_num_limbs();
        let limb_bits = self.air.expr.canonical_limb_bits();
        // let Instruction { opcode, .. } = instruction.clone();
        // let local_opcode_index = opcode - self.air.offset;
        // let local_opcode = FP12Opcode::from_usize(local_opcode_index);
        // let mul_flag = match local_opcode {
        //     FP12Opcode::MUL => true,
        //     _ => panic!("Unsupported opcode: {:?}", local_opcode),
        // };

        let data: DynArray<_> = reads.into();
        let data = data.0;
        let x = data[..num_limbs * 12]
            .chunks(num_limbs)
            .map(|x| x.iter().map(|y| y.as_canonical_u32()).collect_vec())
            .collect_vec();
        let y = data[num_limbs * 12..2 * num_limbs * 12]
            .chunks(num_limbs)
            .map(|x| x.iter().map(|y| y.as_canonical_u32()).collect_vec())
            .collect_vec();
        let xi = data[2 * num_limbs * 12..]
            .chunks(num_limbs)
            .map(|x| x.iter().map(|y| y.as_canonical_u32()).collect_vec())
            .collect_vec();
        let x_biguint = x
            .iter()
            .map(|x| limbs_to_biguint(x, limb_bits))
            .collect_vec();
        let y_biguint = y
            .iter()
            .map(|y| limbs_to_biguint(y, limb_bits))
            .collect_vec();
        let xi_biguint = xi
            .iter()
            .map(|xi| limbs_to_biguint(xi, limb_bits))
            .collect_vec();
        let input_vec = [x_biguint.clone(), y_biguint.clone(), xi_biguint.clone()].concat();

        let vars = self.air.expr.execute(input_vec, vec![]);
        assert_eq!(vars.len(), 34);
        let res_biguint_vec = vars[vars.len() - 12..].to_vec();
        tracing::trace!("FP12MultiplyOpcode | {res_biguint_vec:?} | {x_biguint:?} | {y_biguint:?}");
        let res_limbs = res_biguint_vec
            .iter()
            .flat_map(|x| biguint_to_limbs_vec(x.clone(), limb_bits, num_limbs))
            .collect_vec();
        let writes = res_limbs
            .into_iter()
            .map(F::from_canonical_u32)
            .collect_vec();
        let ctx = AdapterRuntimeContext::<_, DynAdapterInterface<_>>::without_pc(writes);

        Ok((
            ctx.into(),
            Fp12MultiplyCoreRecord {
                x: Fp12BigUint {
                    c0: Fp2BigUint {
                        c0: FpBigUint(x_biguint[0].clone()),
                        c1: FpBigUint(x_biguint[1].clone()),
                    },
                    c1: Fp2BigUint {
                        c0: FpBigUint(x_biguint[2].clone()),
                        c1: FpBigUint(x_biguint[3].clone()),
                    },
                    c2: Fp2BigUint {
                        c0: FpBigUint(x_biguint[4].clone()),
                        c1: FpBigUint(x_biguint[5].clone()),
                    },
                    c3: Fp2BigUint {
                        c0: FpBigUint(x_biguint[6].clone()),
                        c1: FpBigUint(x_biguint[7].clone()),
                    },
                    c4: Fp2BigUint {
                        c0: FpBigUint(x_biguint[8].clone()),
                        c1: FpBigUint(x_biguint[9].clone()),
                    },
                    c5: Fp2BigUint {
                        c0: FpBigUint(x_biguint[10].clone()),
                        c1: FpBigUint(x_biguint[11].clone()),
                    },
                },
                y: Fp12BigUint {
                    c0: Fp2BigUint {
                        c0: FpBigUint(y_biguint[0].clone()),
                        c1: FpBigUint(y_biguint[1].clone()),
                    },
                    c1: Fp2BigUint {
                        c0: FpBigUint(y_biguint[2].clone()),
                        c1: FpBigUint(y_biguint[3].clone()),
                    },
                    c2: Fp2BigUint {
                        c0: FpBigUint(y_biguint[4].clone()),
                        c1: FpBigUint(y_biguint[5].clone()),
                    },
                    c3: Fp2BigUint {
                        c0: FpBigUint(y_biguint[6].clone()),
                        c1: FpBigUint(y_biguint[7].clone()),
                    },
                    c4: Fp2BigUint {
                        c0: FpBigUint(y_biguint[8].clone()),
                        c1: FpBigUint(y_biguint[9].clone()),
                    },
                    c5: Fp2BigUint {
                        c0: FpBigUint(y_biguint[10].clone()),
                        c1: FpBigUint(y_biguint[11].clone()),
                    },
                },
                xi: Fp2BigUint {
                    c0: FpBigUint(xi_biguint[0].clone()),
                    c1: FpBigUint(xi_biguint[1].clone()),
                },
            },
        ))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        "Fp12Multiply".to_string()
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let input = (
            vec![
                record.x.c0.c0.0,
                record.x.c0.c1.0,
                record.x.c1.c0.0,
                record.x.c1.c1.0,
                record.x.c2.c0.0,
                record.x.c2.c1.0,
                record.x.c3.c0.0,
                record.x.c3.c1.0,
                record.x.c4.c0.0,
                record.x.c4.c1.0,
                record.x.c5.c0.0,
                record.x.c5.c1.0,
                record.y.c0.c0.0,
                record.y.c0.c1.0,
                record.y.c1.c0.0,
                record.y.c1.c1.0,
                record.y.c2.c0.0,
                record.y.c2.c1.0,
                record.y.c3.c0.0,
                record.y.c3.c1.0,
                record.y.c4.c0.0,
                record.y.c4.c1.0,
                record.y.c5.c0.0,
                record.y.c5.c1.0,
                BigUint::from(9u32),
                BigUint::from(1u32),
                // record.xi.c0.0,
                // record.xi.c1.0,
            ],
            self.range_checker.clone(),
            vec![],
        );
        let row = LocalTraceInstructions::<F>::generate_trace_row(&self.air.expr, input);
        for (i, element) in row.iter().enumerate() {
            row_slice[i] = *element;
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
