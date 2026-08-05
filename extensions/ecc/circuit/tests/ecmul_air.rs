//! Reports the `EC_MUL` AIR's width and asserts a bound on its maximum constraint degree.
//!
//! `log_blowup` is chosen per application, not per chip, so an AIR whose degree exceeds the budget
//! raises the proving cost of every other chip alongside it. The assertion here is a guard against
//! that regressing unnoticed.
//!
//! This checks the constraint system's shape only. Satisfiability requires `air_test` against a
//! generated trace.

use num_bigint::BigUint;
use num_traits::Zero;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionBus},
    system::{
        memory::offline_checker::{MemoryBridge, MemoryBus},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::{
    bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime},
    var_range::VariableRangeCheckerBus,
};
use openvm_ecc_circuit::{
    ec_mul_step_expr, EcMulAir, ECC_BLOCKS_32, EC_MUL_TOTAL_ROWS, NUM_LIMBS_32,
};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_backend::{
    air_builders::symbolic::get_symbolic_builder, keygen::types::TraceWidth, p3_air::BaseAir,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;

const LIMB_BITS: usize = 8;
const RANGE_MAX_BITS: usize = 17;

/// `SystemConfig`'s default, and the bound every other AIR in the ECC configuration meets. Staying
/// at or below it keeps `log_blowup` at 1, which matters because the blowup is chosen per
/// application rather than per chip.
const DEGREE_BUDGET: usize = 3;

fn bus() -> VariableRangeCheckerBus {
    VariableRangeCheckerBus::new(1, RANGE_MAX_BITS)
}

/// Bus indices are arbitrary: only the symbolic constraint shape is under test.
fn air_for(modulus: BigUint, a: BigUint) -> EcMulAir<NUM_LIMBS_32, ECC_BLOCKS_32> {
    let config = ExprBuilderConfig {
        modulus,
        num_limbs: NUM_LIMBS_32,
        limb_bits: LIMB_BITS,
    };
    let expr = ec_mul_step_expr(config, bus(), a);
    EcMulAir::new(
        expr,
        ExecutionBridge::new(ExecutionBus::new(2), ProgramBus::new(3)),
        MemoryBridge::new(MemoryBus::new(4), 29, bus()),
        bus(),
        29,
        0,
    )
}

fn report(name: &str, air: &EcMulAir<NUM_LIMBS_32, ECC_BLOCKS_32>) -> usize {
    let width = <_ as BaseAir<F>>::width(air);
    let trace_width = TraceWidth {
        preprocessed: None,
        cached_mains: vec![],
        common_main: width,
    };
    let constraints = get_symbolic_builder::<F, _>(air, &trace_width).constraints();
    let degree = constraints.max_constraint_degree();

    // Separate the step expression's own contribution from the chip's row constraints, so it is
    // clear which one drives the total.
    let expr_width = <_ as BaseAir<F>>::width(&air.expr);
    let expr_degree = get_symbolic_builder::<F, _>(
        &air.expr,
        &TraceWidth {
            preprocessed: None,
            cached_mains: vec![],
            common_main: expr_width,
        },
    )
    .constraints()
    .max_constraint_degree();
    println!(
        "{:<12} field expression alone: width={expr_width} max_degree={expr_degree}",
        ""
    );

    println!(
        "{name:<12} width={width:<6} constraints={:<5} interactions={:<4} max_degree={degree}",
        constraints.constraints.len(),
        constraints.interactions.len(),
    );
    println!(
        "{:<12} cells / scalar-mul = {width} x {EC_MUL_TOTAL_ROWS} rows = {}",
        "",
        width * EC_MUL_TOTAL_ROWS
    );
    let header = openvm_ecc_circuit::ec_mul_header_width();
    let digest = openvm_ecc_circuit::ec_mul_digest_width::<NUM_LIMBS_32, ECC_BLOCKS_32>();
    println!(
        "{:<12} regions: header={header} expr={expr_width} digest={digest} (sum={})",
        "",
        header + expr_width + digest
    );
    println!(
        "{:<12} digest region idle on {} of {EC_MUL_TOTAL_ROWS} rows = {} cells ({:.1}% of chip)",
        "",
        EC_MUL_TOTAL_ROWS - 1,
        digest * (EC_MUL_TOTAL_ROWS - 1),
        100.0 * (digest * (EC_MUL_TOTAL_ROWS - 1)) as f64 / (width * EC_MUL_TOTAL_ROWS) as f64
    );
    degree
}

#[test]
fn ec_mul_air_width_and_degree() {
    let k1 = air_for(secp256k1_coord_prime(), BigUint::zero());
    let r1_p = secp256r1_coord_prime();
    let r1 = air_for(r1_p.clone(), &r1_p - BigUint::from(3u32));

    let d_k1 = report("secp256k1", &k1);
    let d_r1 = report("secp256r1", &r1);

    for (name, degree) in [("secp256k1", d_k1), ("secp256r1", d_r1)] {
        assert!(
            degree <= DEGREE_BUDGET,
            "{name}: max constraint degree {degree} exceeds budget {DEGREE_BUDGET}; a higher \
             degree forces a larger log_blowup on every chip in the application"
        );
    }
}
