use std::fs::File;

use std::io::{BufRead, BufReader};

use p3_field::{PrimeField32, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, Val};
use stark_vm::cpu::trace::Instruction;
use stark_vm::vm::config::VmConfig;
use stark_vm::vm::VirtualMachine;

pub fn get_vm<AC: StarkGenericConfig>(
    config: VmConfig,
    file_path: &str,
) -> Result<VirtualMachine<AC>, std::io::Error>
where
    Val<AC>: PrimeField32,
{
    let instructions = parse_isa_file::<Val<AC>>(file_path)?;
    let vm = VirtualMachine::new(config, instructions);
    Ok(vm)
}

pub fn parse_isa_file<F: PrimeField64>(
    file_path: &str,
) -> Result<Vec<Instruction<F>>, std::io::Error> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut result = vec![];
    for line in reader.lines() {
        if let Some(instruction) = instruction_from_line::<F>(&line?)? {
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
    let opcode = parts[0]
        .parse()
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

    Ok(Some(Instruction::from_isize(
        opcode, ints[0], ints[1], ints[2], ints[3], ints[4],
    )))
}
