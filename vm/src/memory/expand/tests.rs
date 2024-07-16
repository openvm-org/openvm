use std::collections::{HashMap, HashSet};

use p3_field::{PrimeField32, PrimeField64};

use crate::cpu::OpCode::*;
use crate::memory::expand::columns::ExpandCols;
use crate::memory::expand::ExpandAir;
use crate::memory::tree::trees_from_full_memory;

const TEST_CHUNK: usize = 8;

#[test]
fn test_flatten_fromslice_roundtrip() {
    let num_cols = ExpandCols::<TEST_CHUNK, usize>::get_width();
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ExpandCols::<TEST_CHUNK, _>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

fn test<const CHUNK: usize, F: PrimeField32>(
    height: usize,
    initial_memory: &HashMap<(F, F), F>,
    touched_addresses: HashSet<(F, F)>,
    final_memory: &HashMap<(F, F), F>,
) {
    // checking validity of test data
    for (address, value) in final_memory {
        assert!((address.0.as_canonical_u64() as usize) < (1 << height));
        if initial_memory.get(address) != Some(value) {
            assert!(touched_addresses.contains(address));
        }
    }
    for (address, _) in initial_memory {
        assert!(final_memory.contains_key(&address));
    }
    for address in touched_addresses.iter() {
        assert!(final_memory.contains_key(&address));
    }

    let initial_trees = trees_from_full_memory(height, initial_memory);
    let final_trees_check = trees_from_full_memory(height, final_memory);

    let air = ExpandAir { height };
    let trace_degree = (height * touched_addresses.len()).next_power_of_two();
    let (trace, final_trees) = air.generate_trace_and_final_tree(
        initial_trees,
        touched_addresses,
        final_memory,
        trace_degree,
    );

    assert_eq!(final_trees, final_trees_check);
}
