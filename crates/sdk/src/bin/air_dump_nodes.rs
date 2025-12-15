use std::{env, error::Error, path::Path};

use openvm_sdk::{fs::read_object_from_file, keygen::AppVerifyingKey};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        eprintln!("usage: air_dump_nodes <vk_path> <air_idx>");
        std::process::exit(1);
    }
    let vk_path = &args[0];
    let air_idx: usize = args[1].parse()?;

    let vk = read_object_from_file::<AppVerifyingKey, _>(Path::new(vk_path))?;
    let air = vk
        .vk
        .inner
        .per_air
        .get(air_idx)
        .ok_or_else(|| format!("air idx {air_idx} out of range"))?;

    println!(
        "air #{air_idx} nodes len={}",
        air.symbolic_constraints.constraints.nodes.len()
    );
    for (i, node) in air
        .symbolic_constraints
        .constraints
        .nodes
        .iter()
        .enumerate()
    {
        println!("{i}: {node:?}");
    }
    Ok(())
}
