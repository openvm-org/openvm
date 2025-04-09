use std::env::set_var;
use std::io::Write;

use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_sdk::config::baby_bear_blake3::BabyBearBlake3Engine;
use openvm_stark_sdk::config::FriParameters;
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::all::compile;
use crate::transpiled_merkle::TLFunction_main;
use crate::{
    core::stage1::stage1,
    parser::parser::parse_program_source,
    transpiled_fibonacci::{isize_to_field_elem, TLFunction_fibonacci},
};

pub mod air;
pub mod all;
pub mod core;
pub mod parser;
mod transpilation;
pub mod transpiled_fibonacci;
mod transpiled_merkle;

fn main() {
    println!("Hello, world!");
    unsafe {
        set_var("RUST_BACKTRACE", "0");
    }
    // parse_and_compile_and_transpile_fibonacci();
    // test_fibonacci();

    /*let x = true;
    let y = false;
    let z = x | y;*/

    //let mut x = Box::new(Some(vec![]));
    //x.as_mut().as_mut().unwrap().push(1);

    // parse_and_compile_and_transpile_merkle();
    test_merkle(false);
}

fn test_fibonacci() {
    let mut tracker = transpiled_fibonacci::Tracker::default();
    let mut fibonacci = TLFunction_fibonacci::default();
    fibonacci.n = isize_to_field_elem(12);
    println!("calculating {}th fibonacci number", fibonacci.n);
    fibonacci.stage_0(&mut tracker);
    assert_eq!(fibonacci.a, isize_to_field_elem(144));
    println!("success: {}", fibonacci.a)
}

fn parse_and_compile_and_transpile_fibonacci() {
    let fibonacci_program_source = std::fs::read_to_string("fibonacci.txt").unwrap();
    let program = parse_program_source(&fibonacci_program_source).unwrap();

    let stage2_program = stage1(program).unwrap();
    //println!("{:?}", stage2_program);
    let transpiled = stage2_program.transpile_execution();
    println!("{}", transpiled);
}

fn parse_and_compile_and_transpile_merkle() {
    let source = std::fs::read_to_string("merkle.txt").unwrap();
    let result = compile(&source);
    let file = std::fs::File::create("src/transpiled_merkle.rs").unwrap();
    let mut file = std::io::BufWriter::new(file);
    file.write_all(result.as_bytes()).unwrap();
}

fn test_merkle(should_fail: bool) {
    let mut tracker = transpiled_merkle::Tracker::default();
    let mut main = TLFunction_main::default();

    main.materialized = true;
    main.should_fail = should_fail;
    main.stage_0(&mut tracker);
    println!(
        "commit = {:?}",
        main.callee_0.as_ref().as_ref().unwrap().commit
    );
    println!("tracker = {:?}", tracker);
    let mut trace_set = transpiled_merkle::TraceSet::new(&tracker);
    main.generate_trace(&tracker, &mut trace_set);
    println!("trace_set = {:?}", trace_set);
    println!(
        "trace_set.merkle_verify_trace length = {}",
        trace_set.merkle_verify_trace.len()
    );
    println!(
        "trace_set.main_trace length = {}",
        trace_set.main_trace.len()
    );
    let proof_input = transpiled_merkle::ProofInput::new(trace_set);
    let engine = BabyBearBlake3Engine::new(FriParameters::new_for_testing(1));
    let result = engine.run_test_impl(proof_input.airs, proof_input.inputs);
    result.expect("Verification failed");
    println!("success");
}
