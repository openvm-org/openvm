use crate::core::stage1::stage1;
use crate::parser::parser::parse_program_source;
use crate::transpilation::air::proof_input::rust_proof_input;

pub fn compile(source: &str) -> String {
    let program = parse_program_source(&source).unwrap();
    let stage2_program = stage1(program).unwrap();
    let execution = stage2_program.transpile_execution();
    let air_set = stage2_program.construct_airs();
    let trace_generation = stage2_program.transpile_trace_generation(&air_set);
    let airs = air_set.transpile_airs();
    let proof_input = rust_proof_input(&stage2_program);

    let mut result = String::new();
    result.push_str(&execution.to_string());
    result.push_str(&trace_generation.to_string());
    result.push_str(&airs.to_string());
    result.push_str(&proof_input.to_string());
    result
}
