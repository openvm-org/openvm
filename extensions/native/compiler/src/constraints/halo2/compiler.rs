use std::{
    collections::HashMap,
    fmt::Debug,
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use axvm_circuit::metrics::cycle_tracker::CycleTracker;
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::{ExtensionField, PrimeField};
use snark_verifier_sdk::snark_verifier::{
    halo2_base::{
        gates::{circuit::builder::BaseCircuitBuilder, GateInstructions, RangeChip},
        halo2_proofs::halo2curves::bn256::Fr,
        utils::{biguint_to_fe, ScalarField},
        Context,
    },
    util::arithmetic::PrimeField as _,
};

use super::stats::Halo2Stats;
use crate::{
    constraints::halo2::{
        baby_bear::{
            AssignedBabyBear, AssignedBabyBearExt4, BabyBearChip, BabyBearExt4, BabyBearExt4Chip,
        },
        poseidon2_perm::{Poseidon2Params, Poseidon2State},
    },
    ir::{Config, DslIr, TracedVec, Witness},
};

/// The backend for the Halo2 constraint compiler.
#[derive(Debug, Clone)]
pub struct Halo2ConstraintCompiler<C: Config> {
    pub num_public_values: usize,
    #[allow(unused_variables)]
    pub collect_metrics: bool,
    pub phantom: PhantomData<C>,
}

#[derive(Debug, Clone, Default)]
pub struct Halo2State<C: Config> {
    // halo2 stuff
    pub builder: BaseCircuitBuilder<Fr>,
    // Unassigned values: provided by the prover outside of constraint compiler
    // map from name as string to halo2 assigned value
    pub vars: HashMap<u32, Fr>,
    pub felts: HashMap<u32, C::F>,
    pub exts: HashMap<u32, C::EF>,
}

impl<C: Config> Halo2State<C> {
    pub fn load_witness(&mut self, witness: Witness<C>) {
        for (i, x) in witness.vars.iter().enumerate() {
            self.vars.insert(i as u32, convert_fr(x));
        }
        for (i, x) in witness.felts.into_iter().enumerate() {
            self.felts.insert(i as u32, x);
        }
        for (i, x) in witness.exts.into_iter().enumerate() {
            self.exts.insert(i as u32, x);
        }
    }
}

impl<C: Config + Debug> Halo2ConstraintCompiler<C> {
    pub fn new(num_public_values: usize) -> Self {
        Self {
            num_public_values,
            collect_metrics: false,
            phantom: PhantomData,
        }
    }
    pub fn with_collect_metrics(mut self) -> Self {
        self.collect_metrics = true;
        self
    }
    // Create halo2-lib constraints from a list of operations in the DSL.
    // Assume: C::N = C::F = C::EF is type Fr
    pub fn constrain_halo2(&self, halo2_state: &mut Halo2State<C>, operations: TracedVec<DslIr<C>>)
    where
        C: Config<N = Bn254Fr, F = BabyBear, EF = BabyBearExt4>,
    {
        let mut cell_tracker = CycleTracker::new();
        let range = Arc::new(halo2_state.builder.range_chip());
        let f_chip = Arc::new(BabyBearChip::new(range.clone()));
        let ext_chip = BabyBearExt4Chip::new(Arc::clone(&f_chip));
        let gate = f_chip.gate();
        let ctx = halo2_state.builder.main(0);
        let mut public_values = vec![ctx.load_zero(); self.num_public_values];

        // Local variables for referencing during the course of constraint building
        let mut vars = HashMap::new();
        let mut felts = HashMap::<u32, AssignedBabyBear>::new();
        let mut exts = HashMap::<u32, AssignedBabyBearExt4>::new();

        let mut vkey_hash = None;
        let mut committed_values_digest = None;
        #[cfg(feature = "bench-metrics")]
        let mut old_stats = stats_snapshot(ctx, range.clone());
        for (instruction, backtrace) in operations {
            #[cfg(feature = "bench-metrics")]
            if self.collect_metrics {
                old_stats = stats_snapshot(ctx, range.clone());
            }
            let res = catch_unwind(AssertUnwindSafe(|| {
                match instruction {
                    DslIr::ImmV(a, b) => {
                        let x = ctx.load_constant(convert_fr(&b));
                        vars.insert(a.0, x);
                    }
                    DslIr::ImmF(a, b) => {
                        let x = f_chip.load_constant(ctx, b);
                        felts.insert(a.0, x);
                    }
                    DslIr::ImmE(a, b) => {
                        let x = ext_chip.load_constant(ctx, b);
                        exts.insert(a.0, x);
                    }
                    DslIr::AddV(a, b, c) => {
                        let x = gate.add(ctx, vars[&b.0], vars[&c.0]);
                        vars.insert(a.0, x);
                    }
                    DslIr::AddVI(a, b, c) => {
                        let tmp = ctx.load_constant(convert_fr(&c));
                        let x = gate.add(ctx, vars[&b.0], tmp);
                        vars.insert(a.0, x);
                    }
                    DslIr::AddF(a, b, c) => {
                        let x = f_chip.add(ctx, felts[&b.0], felts[&c.0]);
                        felts.insert(a.0, x);
                    }
                    DslIr::AddFI(a, b, c) => {
                        let tmp = f_chip.load_constant(ctx, c);
                        let x = f_chip.add(ctx, felts[&b.0], tmp);
                        felts.insert(a.0, x);
                    }
                    DslIr::AddE(a, b, c) => {
                        let x = ext_chip.add(ctx, exts[&b.0], exts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::AddEF(a, b, c) => {
                        let mut x = exts[&b.0];
                        x.0[0] = f_chip.add(ctx, x.0[0], felts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::AddEFI(a, b, c) => {
                        let tmp = f_chip.load_constant(ctx, c);
                        let mut x = exts[&b.0];
                        x.0[0] = f_chip.add(ctx, x.0[0], tmp);
                        exts.insert(a.0, x);
                    }
                    DslIr::AddEI(a, b, c) => {
                        let tmp = ext_chip.load_constant(ctx, c);
                        let x = ext_chip.add(ctx, exts[&b.0], tmp);
                        exts.insert(a.0, x);
                    }
                    DslIr::AddEFFI(a, b, c) => {
                        let mut x = ext_chip.load_constant(ctx, c);
                        x.0[0] = f_chip.add(ctx, x.0[0], felts[&b.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::SubV(a, b, c) => {
                        let x = gate.sub(ctx, vars[&b.0], vars[&c.0]);
                        vars.insert(a.0, x);
                    }
                    DslIr::SubF(a, b, c) => {
                        let x = f_chip.sub(ctx, felts[&b.0], felts[&c.0]);
                        felts.insert(a.0, x);
                    }
                    DslIr::SubE(a, b, c) => {
                        let x = ext_chip.sub(ctx, exts[&b.0], exts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::SubEF(a, b, c) => {
                        let mut x = exts[&b.0];
                        x.0[0] = f_chip.sub(ctx, x.0[0], felts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::SubEI(a, b, c) => {
                        let tmp = ext_chip.load_constant(ctx, c);
                        let x = ext_chip.sub(ctx, exts[&b.0], tmp);
                        exts.insert(a.0, x);
                    }
                    DslIr::SubVIN(a, b, c) => {
                        let tmp = ctx.load_constant(convert_fr(&b));
                        let x = gate.sub(ctx, tmp, vars[&c.0]);
                        vars.insert(a.0, x);
                    }
                    DslIr::SubEIN(a, b, c) => {
                        let tmp = ext_chip.load_constant(ctx, b);
                        let x = ext_chip.sub(ctx, tmp, exts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::SubEFI(a, b, c) => {
                        let tmp = f_chip.load_constant(ctx, c);
                        let mut x = exts[&b.0];
                        x.0[0] = f_chip.sub(ctx, x.0[0], tmp);
                        exts.insert(a.0, x);
                    }
                    DslIr::MulV(a, b, c) => {
                        let x = gate.mul(ctx, vars[&b.0], vars[&c.0]);
                        vars.insert(a.0, x);
                    }
                    DslIr::MulVI(a, b, c) => {
                        let tmp = ctx.load_constant(convert_fr(&c));
                        let x = gate.mul(ctx, vars[&b.0], tmp);
                        vars.insert(a.0, x);
                    }
                    DslIr::MulF(a, b, c) => {
                        let x = f_chip.mul(ctx, felts[&b.0], felts[&c.0]);
                        felts.insert(a.0, x);
                    }
                    DslIr::MulFI(a, b, c) => {
                        let tmp = f_chip.load_constant(ctx, c);
                        let x = f_chip.mul(ctx, felts[&b.0], tmp);
                        felts.insert(a.0, x);
                    }
                    DslIr::MulE(a, b, c) => {
                        let x = ext_chip.mul(ctx, exts[&b.0], exts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::MulEI(a, b, c) => {
                        let tmp = ext_chip.load_constant(ctx, c);
                        let x = ext_chip.mul(ctx, exts[&b.0], tmp);
                        exts.insert(a.0, x);
                    }
                    DslIr::MulEF(a, b, c) => {
                        let x = ext_chip.scalar_mul(ctx, exts[&b.0], felts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::MulEFI(a, b, c) => {
                        let tmp = f_chip.load_constant(ctx, c);
                        let x = ext_chip.scalar_mul(ctx, exts[&b.0], tmp);
                        exts.insert(a.0, x);
                    }
                    DslIr::DivFIN(a, b, c) => {
                        // a = b / c
                        let tmp = f_chip.load_constant(ctx, b);
                        let x = f_chip.div(ctx, tmp, felts[&c.0]);
                        felts.insert(a.0, x);
                    }
                    DslIr::DivE(a, b, c) => {
                        let x = ext_chip.div(ctx, exts[&b.0], exts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::DivEIN(a, b, c) => {
                        let tmp = ext_chip.load_constant(ctx, b);
                        let x = ext_chip.div(ctx, tmp, exts[&c.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::NegE(a, b) => {
                        let x = ext_chip.neg(ctx, exts[&b.0]);
                        exts.insert(a.0, x);
                    }
                    DslIr::CastFV(a, b) => {
                        let felt = felts[&b.0];
                        let reduced_felt = f_chip.reduce(ctx, felt);
                        vars.insert(a.0, reduced_felt.value);
                    }
                    DslIr::CircuitNum2BitsV(value, bits, output) => {
                        let shortened_bits = bits.min(Fr::NUM_BITS as usize);
                        let mut x = gate.num_to_bits(ctx, vars[&value.0], shortened_bits);
                        let zero = ctx.load_zero();
                        x.resize(bits, zero);
                        for (o, x) in output.into_iter().zip_eq(x) {
                            vars.insert(o.0, x);
                        }
                    }
                    DslIr::CircuitNum2BitsF(value, output) => {
                        let val = f_chip.reduce(ctx, felts[&value.0]);
                        let x = gate.num_to_bits(ctx, val.value, 32); // C::F::bits());
                        assert!(output.len() <= x.len());
                        for (o, x) in output.into_iter().zip(x) {
                            vars.insert(o.0, x);
                        }
                    }
                    DslIr::CircuitPoseidon2Permute(state_vars) => {
                        use zkhash::{
                            ark_ff::{BigInteger, PrimeField as _},
                            fields::bn256::FpBN256 as ark_FpBN256,
                            poseidon2::poseidon2_instance_bn256::{MAT_DIAG3_M_1, RC3},
                        };

                        fn convert_fr(input: ark_FpBN256) -> Fr {
                            Fr::from_bytes_le(&input.into_bigint().to_bytes_le())
                        }
                        const T: usize = 3;
                        let rounds_f = 8;
                        let rounds_p = 56;
                        let mut round_constants: Vec<[Fr; T]> = RC3
                            .iter()
                            .map(|vec| {
                                vec.iter()
                                    .cloned()
                                    .map(convert_fr)
                                    .collect::<Vec<_>>()
                                    .try_into()
                                    .unwrap()
                            })
                            .collect();

                        let rounds_f_beginning = rounds_f / 2;
                        let p_end = rounds_f_beginning + rounds_p;
                        let internal_round_constants = round_constants
                            .drain(rounds_f_beginning..p_end)
                            .map(|vec| vec[0])
                            .collect::<Vec<_>>();
                        let external_round_constants = round_constants;
                        let params = Poseidon2Params {
                            rounds_f,
                            rounds_p,
                            mat_internal_diag_m_1: MAT_DIAG3_M_1
                                .iter()
                                .copied()
                                .map(convert_fr)
                                .collect_vec()
                                .try_into()
                                .unwrap(),
                            external_rc: external_round_constants,
                            internal_rc: internal_round_constants,
                        };

                        let mut state =
                            Poseidon2State::<Fr, T>::new(state_vars.map(|x| vars[&x.0]));
                        state.permutation(ctx, gate, &params);
                        for i in 0..T {
                            *vars.get_mut(&state_vars[i].0).unwrap() = state.s[i];
                        }
                    }
                    DslIr::CircuitSelectV(cond, a, b, out) => {
                        let x = gate.select(ctx, vars[&a.0], vars[&b.0], vars[&cond.0]);
                        vars.insert(out.0, x);
                    }
                    DslIr::CircuitSelectF(cond, a, b, out) => {
                        let x = f_chip.select(ctx, vars[&cond.0], felts[&a.0], felts[&b.0]);
                        felts.insert(out.0, x);
                    }
                    DslIr::CircuitSelectE(cond, a, b, out) => {
                        let x = ext_chip.select(ctx, vars[&cond.0], exts[&a.0], exts[&b.0]);
                        exts.insert(out.0, x);
                    }
                    DslIr::CircuitExt2Felt(a, b) => {
                        for (i, x) in a.iter().enumerate() {
                            felts.insert(x.0, exts[&b.0].0[i]);
                        }
                    }
                    DslIr::AssertEqV(a, b) => {
                        ctx.constrain_equal(&vars[&a.0], &vars[&b.0]);
                    }
                    DslIr::AssertEqVI(a, b) => {
                        gate.assert_is_const(ctx, &vars[&a.0], &convert_fr(&b));
                    }
                    DslIr::AssertEqF(a, b) => {
                        f_chip.assert_equal(ctx, felts[&a.0], felts[&b.0]);
                    }
                    DslIr::AssertEqFI(a, b) => {
                        let tmp = f_chip.load_constant(ctx, b);
                        f_chip.assert_equal(ctx, felts[&a.0], tmp);
                    }
                    DslIr::AssertEqE(a, b) => {
                        ext_chip.assert_equal(ctx, exts[&a.0], exts[&b.0]);
                    }
                    DslIr::AssertEqEI(a, b) => {
                        let tmp = ext_chip.load_constant(ctx, b);
                        ext_chip.assert_equal(ctx, exts[&a.0], tmp);
                    }
                    DslIr::PrintV(a) => {
                        println!("PrintV: {:?}", vars[&a.0].value());
                    }
                    DslIr::PrintF(a) => {
                        println!("PrintF: {:?}", felts[&a.0].to_baby_bear());
                    }
                    DslIr::PrintE(a) => {
                        println!("PrintE:");
                        for x in exts[&a.0].0.iter() {
                            println!("{:?}", x.to_baby_bear());
                        }
                    }
                    DslIr::WitnessVar(a, b) => {
                        let x = ctx.load_witness(halo2_state.vars[&b]);
                        vars.insert(a.0, x);
                    }
                    DslIr::WitnessFelt(a, b) => {
                        let x = f_chip.load_witness(ctx, halo2_state.felts[&b]);
                        felts.insert(a.0, x);
                    }
                    DslIr::WitnessExt(a, b) => {
                        let x = ext_chip.load_witness(ctx, halo2_state.exts[&b]);
                        exts.insert(a.0, x);
                    }
                    DslIr::CircuitCommitVkeyHash(a) => {
                        assert!(vkey_hash.is_none());
                        vkey_hash = Some(vars[&a.0]);
                    }
                    DslIr::CircuitCommitCommitedValuesDigest(a) => {
                        assert!(committed_values_digest.is_none());
                        committed_values_digest = Some(vars[&a.0]);
                    }
                    DslIr::CircuitFelts2Ext(a, b) => {
                        let x = AssignedBabyBearExt4(
                            a.iter()
                                .map(|a| felts[&a.0])
                                .collect_vec()
                                .try_into()
                                .unwrap(),
                        );
                        exts.insert(b.0, x);
                    }
                    DslIr::CycleTrackerStart(name) => {
                        cell_tracker.start(name);
                    }
                    DslIr::CycleTrackerEnd(name) => {
                        cell_tracker.end(name);
                    }
                    DslIr::CircuitPublish(val, index) => {
                        public_values[index] = vars[&val.0];
                    }
                    _ => panic!("unsupported {:?}", instruction),
                }
            }));
            if res.is_err() {
                if let Some(mut backtrace) = backtrace {
                    backtrace.resolve();
                    eprintln!("axvm circuit failure; backtrace:\n{:?}", backtrace);
                }
                res.unwrap();
            }
            #[cfg(feature = "bench-metrics")]
            if self.collect_metrics {
                let mut new_stats = stats_snapshot(ctx, range.clone());
                new_stats.diff(&old_stats);
                new_stats.increment(cell_tracker.get_full_name());
            }
        }

        halo2_state.builder.assigned_instances = vec![public_values];
    }
}

/// Assumes F is Bn254 Fr and converts to halo2 Fr type
pub fn convert_fr<F: PrimeField>(a: &F) -> Fr {
    biguint_to_fe(&a.as_canonical_biguint())
}

#[allow(dead_code)]
pub fn convert_efr<F: PrimeField, EF: ExtensionField<F>>(a: &EF) -> Vec<Fr> {
    let slc = a.as_base_slice();
    slc.iter()
        .map(|x| biguint_to_fe(&x.as_canonical_biguint()))
        .collect()
}

// Unfortunately `builder.statistics()` cannot be called when `ctx` exists.
#[allow(dead_code)] // used only in bench-metrics
fn stats_snapshot(ctx: &Context<Fr>, range_chip: Arc<RangeChip<Fr>>) -> Halo2Stats {
    Halo2Stats {
        total_gate_cell: ctx.advice.len(),
        // FIXME: this is inaccurate because of duplicated constants. But it's too slow if we always
        // check for duplicates.
        total_fixed: ctx.copy_manager.lock().unwrap().constant_equalities.len(),
        total_lookup_cell: range_chip.lookup_manager()[0].total_rows(),
    }
}

#[allow(dead_code)]
fn is_babybear_ir<C: Config>(ir: &DslIr<C>) -> bool {
    matches!(
        ir,
        DslIr::ImmF(_, _)
            | DslIr::AddF(_, _, _)
            | DslIr::AddFI(_, _, _)
            | DslIr::SubF(_, _, _)
            | DslIr::MulF(_, _, _)
            | DslIr::MulFI(_, _, _)
            | DslIr::DivFIN(_, _, _)
            | DslIr::CircuitSelectF(_, _, _, _)
            | DslIr::AssertEqF(_, _)
            | DslIr::AssertEqFI(_, _)
            | DslIr::WitnessFelt(_, _)
            | DslIr::CircuitFelts2Ext(_, _)
            | DslIr::ImmE(_, _)
            | DslIr::AddE(_, _, _)
            | DslIr::AddEF(_, _, _)
            | DslIr::AddEFI(_, _, _)
            | DslIr::AddEI(_, _, _)
            | DslIr::AddEFFI(_, _, _)
            | DslIr::SubE(_, _, _)
            | DslIr::SubEF(_, _, _)
            | DslIr::SubEI(_, _, _)
            | DslIr::SubEIN(_, _, _)
            | DslIr::SubEFI(_, _, _)
            | DslIr::MulE(_, _, _)
            | DslIr::MulEI(_, _, _)
            | DslIr::MulEF(_, _, _)
            | DslIr::MulEFI(_, _, _)
            | DslIr::DivE(_, _, _)
            | DslIr::DivEIN(_, _, _)
            | DslIr::NegE(_, _)
            | DslIr::CircuitSelectE(_, _, _, _)
            | DslIr::AssertEqE(_, _)
            | DslIr::AssertEqEI(_, _)
            | DslIr::WitnessExt(_, _)
            | DslIr::CastFV(_, _)
    )
}

#[allow(dead_code)]
fn is_num2bits_ir<C: Config>(ir: &DslIr<C>) -> bool {
    matches!(
        ir,
        DslIr::CircuitNum2BitsV(_, _, _) | DslIr::CircuitNum2BitsF(_, _)
    )
}
