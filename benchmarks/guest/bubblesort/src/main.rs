use core::hint::black_box;
use openvm as _;

const ARRAY_SIZE: usize = 1_000;

fn bubblesort<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    for i in 0..len {
        let mut swapped = false;
        for j in 0..len - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                swapped = true;
            }
        }
        // If no swapping occurred in this pass, array is sorted
        if !swapped {
            break;
        }
    }
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
    bubblesort(&mut input);

    // Prevent compiler from optimizing away the computation
    black_box(&input);
}
