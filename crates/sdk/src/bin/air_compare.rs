use std::{env, error::Error, path::Path};

use bitcode::serialize;
use openvm_sdk::{fs::read_object_from_file, keygen::AppVerifyingKey};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("usage: air_compare <vk_a> <vk_b> [air_idx]");
        std::process::exit(1);
    }
    let air_idx: usize = if args.len() == 3 {
        args.pop().unwrap().parse()?
    } else {
        26
    };
    let path_b = args.pop().unwrap();
    let path_a = args.pop().unwrap();

    let vk_a = read_object_from_file::<AppVerifyingKey, _>(Path::new(&path_a))?;
    let vk_b = read_object_from_file::<AppVerifyingKey, _>(Path::new(&path_b))?;

    let air_a = vk_a
        .vk
        .inner
        .per_air
        .get(air_idx)
        .ok_or_else(|| format!("air idx {air_idx} out of range for A"))?;
    let air_b = vk_b
        .vk
        .inner
        .per_air
        .get(air_idx)
        .ok_or_else(|| format!("air idx {air_idx} out of range for B"))?;

    println!("Comparing air #{air_idx}");
    cmp_bytes(
        "air",
        serialize(air_a)?.as_slice(),
        serialize(air_b)?.as_slice(),
    );
    cmp_bytes(
        "symbolic",
        serialize(&air_a.symbolic_constraints)?.as_slice(),
        serialize(&air_b.symbolic_constraints)?.as_slice(),
    );

    let nodes_a = &air_a.symbolic_constraints.constraints.nodes;
    let nodes_b = &air_b.symbolic_constraints.constraints.nodes;
    println!("nodes len A={} B={}", nodes_a.len(), nodes_b.len());
    let min_nodes = nodes_a.len().min(nodes_b.len());
    for i in 0..min_nodes {
        if nodes_a[i] != nodes_b[i] {
            println!("first node diff at idx {i}");
            println!("  A: {:?}", nodes_a[i]);
            println!("  B: {:?}", nodes_b[i]);
            break;
        }
    }
    if nodes_a.len() != nodes_b.len() {
        println!(
            "node lens differ (A={}, B={})",
            nodes_a.len(),
            nodes_b.len()
        );
    }

    Ok(())
}

fn cmp_bytes(label: &str, a: &[u8], b: &[u8]) {
    println!("{label}: len A={} B={}", a.len(), b.len());
    let min = a.len().min(b.len());
    for i in 0..min {
        if a[i] != b[i] {
            let start = i.saturating_sub(8);
            let end = (i + 8).min(min);
            println!(
                "  first byte diff at offset {i}: A=0x{:02x} B=0x{:02x}",
                a[i], b[i]
            );
            println!("  A bytes[{start}:{end}]: {:?}", &a[start..end]);
            println!("  B bytes[{start}:{end}]: {:?}", &b[start..end]);
            return;
        }
    }
    if a.len() != b.len() {
        println!("  prefixes equal, lengths differ");
    } else {
        println!("  bytes identical");
    }
}
