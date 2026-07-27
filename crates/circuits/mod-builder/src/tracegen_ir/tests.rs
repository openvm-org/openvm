use num_bigint::BigUint;

use super::*;
use crate::{ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionProgram};

#[test]
fn compiler_rejects_non_byte_limbs() {
    let range_bus = openvm_circuit_primitives::var_range::VariableRangeCheckerBus::new(1, 16);
    let builder = ExprBuilder::new(
        ExprBuilderConfig {
            modulus: BigUint::from(17u32),
            num_limbs: 2,
            limb_bits: 4,
        },
        range_bus.range_max_bits,
    );
    let expr = FieldExpr::new(FieldExpressionProgram::new(builder, false), range_bus);
    let error = compile_tracegen_ir(&expr, vec![0], vec![], 1).unwrap_err();
    assert_eq!(
        error.to_string(),
        "CUDA tracegen requires 8-bit limbs, got 4"
    );
}

mod tracegen_ir_dump {
    //! Dumps GPU validation vectors: blob, records, expected rows, rc counts.
    //! Run with TRACEGEN_IR_DUMP_DIR=/path cargo test ... -- --ignored dump_gpu_vectors
    use std::{fs, io::Write, sync::atomic::Ordering};

    use num_bigint::BigUint;
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime,
        var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
        TraceSubRowGenerator,
    };
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_field::{PrimeCharacteristicRing, PrimeField32},
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::super::*;
    use crate::{
        test_utils::*, utils::biguint_to_limbs_vec, ExprBuilder, FieldVariable, SymbolicExpr,
    };

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    fn write_u32s(path: &str, data: &[u32]) {
        let mut f = fs::File::create(path).unwrap();
        for &x in data {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dump_case(
        dir: &str,
        name: &str,
        expr: crate::FieldExpr,
        range_bus: VariableRangeCheckerBus,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        opcodes: &[usize],
        rows: usize,
    ) {
        let width = BaseAir::<BabyBear>::width(&expr);
        let prog = compile_tracegen_ir(
            &expr,
            local_opcode_idx.clone(),
            opcode_flag_idx.clone(),
            width,
        )
        .unwrap();
        let blob = prog.encode();

        let nl = expr.program().canonical_num_limbs();
        let prime = expr.program().prime().clone();
        let rec_stride = 1 + expr.program().num_inputs() * nl;
        let range_checker = std::sync::Arc::new(VariableRangeCheckerChip::new(range_bus));

        let mut state = 0x0123456789abcdefu64;
        let mut records = Vec::with_capacity(rows * rec_stride);
        let mut expected = Vec::with_capacity(rows * width);
        for r in 0..rows {
            let opcode = opcodes[r % opcodes.len()];
            let inputs: Vec<BigUint> = (0..expr.program().num_inputs())
                .map(|_| {
                    let bytes: Vec<u8> = (0..nl).map(|_| lcg(&mut state)).collect();
                    BigUint::from_bytes_le(&bytes) % &prime
                })
                .collect();
            records.push(opcode as u8);
            for x in &inputs {
                records.extend(biguint_to_limbs_vec(x, nl));
            }
            let mut flags = vec![false; expr.program().num_flags()];
            if expr.program().needs_setup() {
                if let Some(pos) = local_opcode_idx.iter().position(|&x| x == opcode) {
                    if pos < opcode_flag_idx.len() {
                        flags[opcode_flag_idx[pos]] = true;
                    }
                }
            }
            let mut row = BabyBear::zero_vec(width);
            expr.generate_subrow((range_checker.as_ref(), inputs, flags), &mut row);
            expected.extend(row.iter().map(|x| x.as_canonical_u32()));
        }
        let rc: Vec<u32> = range_checker
            .count
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect();

        write_u32s(&format!("{dir}/{name}.blob"), &blob);
        fs::write(format!("{dir}/{name}.records"), &records).unwrap();
        write_u32s(&format!("{dir}/{name}.expected"), &expected);
        write_u32s(&format!("{dir}/{name}.rc"), &rc);
        write_u32s(
            &format!("{dir}/{name}.meta"),
            &[rec_stride as u32, rows as u32, rc.len() as u32],
        );
        println!(
            "dumped {name}: width={width} rows={rows} rec_stride={rec_stride} rc_len={}",
            rc.len()
        );
    }

    #[test]
    #[ignore]
    fn dump_gpu_vectors() {
        let dir =
            std::env::var("TRACEGEN_IR_DUMP_DIR").unwrap_or("/tmp/tracegen_ir_vectors".into());
        fs::create_dir_all(&dir).unwrap();

        // EcAddNe shape
        {
            let prime = secp256k1_coord_prime();
            let (range_checker, builder) = setup(&prime);
            let x1 = ExprBuilder::new_input(builder.clone());
            let y1 = ExprBuilder::new_input(builder.clone());
            let x2 = ExprBuilder::new_input(builder.clone());
            let y2 = ExprBuilder::new_input(builder.clone());
            let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
            let mut x3 = lambda.square() - x1.clone() - x2;
            x3.save_output();
            let mut y3 = lambda * (x1 - x3.clone()) - y1;
            y3.save_output();
            let b = (*builder).borrow().clone();
            let program = crate::FieldExpressionProgram::new(b, true);
            let expr = crate::FieldExpr::new(program, range_checker.bus());
            dump_case(
                &dir,
                "ecaddne",
                expr,
                range_checker.bus(),
                vec![0, 2],
                vec![0],
                &[0],
                32768,
            );
        }
        // MulDiv with flags
        {
            let prime = secp256k1_coord_prime();
            let (range_checker, builder) = setup(&prime);
            let x = ExprBuilder::new_input(builder.clone());
            let y = ExprBuilder::new_input(builder.clone());
            let (z_idx, z) = (*builder).borrow_mut().new_var();
            let mut z = FieldVariable::from_var(builder.clone(), z);
            let is_mul_flag = (*builder).borrow_mut().new_flag();
            let is_div_flag = (*builder).borrow_mut().new_flag();
            let lvar = FieldVariable::select(is_mul_flag, &x, &z);
            let rvar = FieldVariable::select(is_mul_flag, &z, &x);
            let constraint = lvar * y.clone() - rvar;
            (*builder)
                .borrow_mut()
                .set_constraint(z_idx, constraint.expr);
            let compute = SymbolicExpr::Select(
                is_mul_flag,
                Box::new(x.expr.clone() * y.expr.clone()),
                Box::new(SymbolicExpr::Select(
                    is_div_flag,
                    Box::new(x.expr.clone() / y.expr.clone()),
                    Box::new(x.expr.clone()),
                )),
            );
            (*builder).borrow_mut().set_compute(z_idx, compute);
            z.save_output();
            let b = (*builder).borrow().clone();
            let program = crate::FieldExpressionProgram::new(b, true);
            let expr = crate::FieldExpr::new(program, range_checker.bus());
            dump_case(
                &dir,
                "muldiv",
                expr,
                range_checker.bus(),
                vec![2, 3, 4],
                vec![0, 1],
                &[2, 3],
                16384,
            );
        }
        // Int ops (EcDouble flavored)
        {
            let prime = secp256k1_coord_prime();
            let (range_checker, builder) = setup(&prime);
            let mut x1 = ExprBuilder::new_input(builder.clone());
            let mut y1 = ExprBuilder::new_input(builder.clone());
            let mut nom = x1.square().int_mul(3);
            let mut denom = y1.int_mul(2);
            let mut lambda = nom.div(&mut denom);
            let mut x3 = lambda.square() - x1.int_mul(2);
            x3.save_output();
            let mut y3 = lambda * (x1.clone() - x3.clone()) - y1.clone();
            y3.save_output();
            let mut w = x1.int_add(-7) + y1.int_add(11);
            w.save_output();
            let b = (*builder).borrow().clone();
            let program = crate::FieldExpressionProgram::new(b, true);
            let expr = crate::FieldExpr::new(program, range_checker.bus());
            dump_case(
                &dir,
                "intops",
                expr,
                range_checker.bus(),
                vec![0, 2],
                vec![0],
                &[0],
                16384,
            );
        }
    }
}
