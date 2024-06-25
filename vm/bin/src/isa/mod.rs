use std::fs::File;

use std::io::{BufRead, BufReader};

use p3_field::{PrimeField32, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, Val};
use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::OpCode::{self, *};
use stark_vm::vm::config::VMConfig;
use stark_vm::vm::VM;

pub fn get_vm<AC: StarkGenericConfig>(
    config: VMConfig,
    file_path: &str,
) -> Result<VM<AC>, std::io::Error>
where
    Val<AC>: PrimeField64,
    Val<AC>: PrimeField32,
{
    let instructions = parse_isa_file::<Val<AC>>(file_path)?;
    let vm = VM::new(config, instructions);
    Ok(vm)
}

pub fn parse_isa_file<F: PrimeField64>(
    file_path: &str,
) -> Result<Vec<Instruction<F>>, std::io::Error> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<String>, _>>()?;

    let mut result = vec![];
    for line in lines {
        if let Some(instruction) = instruction_from_line::<F>(&line)? {
            result.push(instruction);
        }
    }

    Ok(result)
}

fn instruction_from_line<F: PrimeField64>(
    line: &str,
) -> Result<Option<Instruction<F>>, std::io::Error> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(None);
    }
    if parts[0].starts_with('#') {
        return Ok(None);
    }
    if parts.len() != 6 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Instruction should have opcode followed by 5 arguments",
        ));
    }
    let opcode = opcode_from_string(parts[0])
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid opcode"))?;
    let mut ints = vec![];
    for part in parts.iter().skip(1) {
        let try_int = part.parse::<isize>();
        ints.push(match try_int {
            Ok(int) => int,
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Opcode argument should be int",
                ))
            }
        });
    }

    //println!("Instruction::from_isize({}, {}, {}, {}, {}, {}),", parts[0], ints[0], ints[1], ints[2], ints[3], ints[4]);

    Ok(Some(Instruction::from_isize(
        opcode, ints[0], ints[1], ints[2], ints[3], ints[4],
    )))
}

fn opcode_from_string(opcode: &str) -> Result<OpCode, ()> {
    match opcode {
        "LOADW" => Ok(LOADW),
        "STOREW" => Ok(STOREW),
        "JAL" => Ok(JAL),
        "BEQ" => Ok(BEQ),
        "BNE" => Ok(BNE),
        "TERMINATE" => Ok(TERMINATE),
        "FADD" => Ok(FADD),
        "FSUB" => Ok(FSUB),
        "FMUL" => Ok(FMUL),
        "FDIV" => Ok(FDIV),
        _ => Err(()),
    }
}
