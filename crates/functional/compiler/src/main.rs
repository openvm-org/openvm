use crate::{
    core::stage1::stage1,
    parser::parser::parse_program_source,
    transpiled_fibonacci::{isize_to_field_elem, TLFunction_fibonacci, Tracker},
};

pub mod air;
pub mod core;
pub mod execution;
pub mod parser;
pub mod transpiled_fibonacci;

fn main() {
    println!("Hello, world!");
    parse_and_compile_and_transpile_fibonacci();
    // test_fibonacci();

    /*let x = true;
    let y = false;
    let z = x | y;*/

    //let mut x = Box::new(Some(vec![]));
    //x.as_mut().as_mut().unwrap().push(1);
}

fn test_fibonacci() {
    let mut tracker = Tracker::default();
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
    let transpiled = stage2_program.transpile();
    println!("{}", transpiled);
}
