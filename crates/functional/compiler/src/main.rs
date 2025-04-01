use crate::{
    core::stage1::stage1,
    parser::parser::parse_program_source,
    transpiled_fibonacci::{isize_to_field_elem, TLFunction_fibonacci},
    transpiled_merkle::TLFunction_main,
};

pub mod air;
pub mod core;
pub mod parser;
mod transpilation;
pub mod transpiled_fibonacci;
pub mod transpiled_merkle;

fn main() {
    println!("Hello, world!");
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
    let program = parse_program_source(&source).unwrap();
    let stage2_program = stage1(program).unwrap();
    let execution = stage2_program.transpile_execution();
    println!("{}", execution);
    let air_set = stage2_program.construct_airs();
    let trace_generation = stage2_program.transpile_trace_generation(&air_set);
    println!("{}", trace_generation);
}

fn test_merkle(should_fail: bool) {
    let mut tracker = transpiled_merkle::Tracker::default();
    let mut main = TLFunction_main::default();
    main.should_fail = should_fail;
    main.stage_0(&mut tracker);
    println!(
        "commit = {:?}",
        main.callee_0.as_ref().as_ref().unwrap().commit
    );
}
