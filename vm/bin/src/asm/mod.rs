use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use color_eyre::eyre::{Report, Result};
use p3_field::PrimeField64;
use stark_vm::program::Instruction;

pub fn parse_asm_file<F: PrimeField64>(path: &Path) -> Result<Vec<Instruction<F>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut result = vec![];
    for line in reader.lines() {
        if let Some(instruction) = instruction_from_line::<F>(&line?)? {
            result.push(instruction);
        }
    }

    Ok(result)
}

fn instruction_from_line<F: PrimeField64>(line: &str) -> Result<Option<Instruction<F>>> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(None);
    }
    if parts[0].starts_with('#') {
        return Ok(None);
    }
    if parts.len() != 8 {
        return Err(Report::msg(
            "Instruction should have opcode followed by 7 arguments",
        ));
    }
    let opcode = parts[0]
        .parse()
        .map_err(|_| Report::msg("Invalid opcode"))?;
    let mut ints = vec![];
    for part in parts.iter().skip(1) {
        ints.push(
            part.parse::<isize>()
                .map_err(|_| Report::msg("Opcode argument should be int"))?,
        );
    }

    Ok(Some(Instruction::large_from_isize(
        opcode, ints[0], ints[1], ints[2], ints[3], ints[4], ints[5], ints[6],
    )))
}
