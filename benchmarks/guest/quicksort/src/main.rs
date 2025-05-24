use core::hint::black_box;
use openvm as _;

const ARRAY_SIZE: usize = 3_500;

fn quicksort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }

    let pivot_index = partition(arr);

    // Sort left part
    quicksort(&mut arr[0..pivot_index]);
    // Sort right part
    quicksort(&mut arr[pivot_index + 1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    if len <= 1 {
        return 0;
    }

    // Choose pivot (middle element)
    let pivot_index = len / 2;

    // Move pivot to the end
    arr.swap(pivot_index, len - 1);

    // Partition
    let mut store_index = 0;
    for i in 0..len - 1 {
        if arr[i] < arr[len - 1] {
            arr.swap(i, store_index);
            store_index += 1;
        }
    }

    // Move pivot to its final place
    arr.swap(store_index, len - 1);
    store_index
}

pub fn main() {
    // Generate array of random-like values
    let mut arr = Vec::with_capacity(ARRAY_SIZE);

    // Initialize with pseudo-random values
    let mut val = 1;
    for _ in 0..ARRAY_SIZE {
        arr.push(val);
        val = ((val * 8191) << 7) ^ val;
    }

    // Prevent compiler from optimizing away the computation
    let mut input = black_box(arr);

    // Sort the array
    quicksort(&mut input);

    // Prevent compiler from optimizing away the computation
    black_box(&input);
}
