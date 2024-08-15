use std::ops::Range;

use p3_field::Field;

use crate::poseidon2::Poseidon2Air;

/// Composed of IO and Aux columns, which are disjoint
/// Aux columns composed of Vec<Vec<T>>, one for each phase
#[derive(Clone, Debug)]
pub struct Poseidon2Cols<const WIDTH: usize, T> {
    pub io: Poseidon2IoCols<WIDTH, T>,
    pub aux: Poseidon2AuxCols<WIDTH, T>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2IoCols<const WIDTH: usize, T> {
    pub input: [T; WIDTH],
    pub output: [T; WIDTH],
}

#[derive(Clone, Debug)]
pub struct Poseidon2AuxCols<const WIDTH: usize, T> {
    // contains one state (array of length WIDTH) for each round of phase1, of which there are `rounds_f/2`
    pub phase1: Vec<[T; WIDTH]>,
    // contains one state (array of length WIDTH) for each round of phase2, of which there are `rounds_p`
    pub phase2: Vec<[T; WIDTH]>,
    // contains one state (array of length WIDTH) for each round of phase3, of which there are `rounds_f - rounds_f/2`
    pub phase3: Vec<[T; WIDTH]>,
}

/// Index map for columns
pub struct Poseidon2ColsIndexMap<const WIDTH: usize> {
    pub input: Range<usize>,
    pub output: Range<usize>,
    pub phase1: Vec<Range<usize>>,
    pub phase2: Vec<Range<usize>>,
    pub phase3: Vec<Range<usize>>,
}

impl<const WIDTH: usize, T: Clone> Poseidon2Cols<WIDTH, T> {
    pub fn get_width<F: Clone>(poseidon2_air: &Poseidon2Air<WIDTH, F>) -> usize {
        let io_width = Poseidon2IoCols::<WIDTH, T>::get_width();
        let aux_width = Poseidon2AuxCols::<WIDTH, T>::get_width(poseidon2_air);
        io_width + aux_width
    }

    pub fn from_slice(slice: &[T], index_map: &Poseidon2ColsIndexMap<WIDTH>) -> Self {
        assert_eq!(slice.len(), index_map.output.end);

        let input = core::array::from_fn(|i| slice[index_map.input.start + i].clone());
        let output = core::array::from_fn(|i| slice[index_map.output.start + i].clone());
        // SAFETY: each element of phase1, phase2, phase3 is a range of length WIDTH
        let phase1: Vec<[T; WIDTH]> = index_map
            .phase1
            .iter()
            .map(|r| core::array::from_fn(|i| slice[r.start + i].clone()))
            .collect();
        let phase2: Vec<[T; WIDTH]> = index_map
            .phase2
            .iter()
            .map(|r| core::array::from_fn(|i| slice[r.start + i].clone()))
            .collect();
        let phase3 = index_map
            .phase3
            .iter()
            .map(|r| core::array::from_fn(|i| slice[r.start + i].clone()))
            .collect();
        Self {
            io: Poseidon2IoCols { input, output },
            aux: Poseidon2AuxCols {
                phase1,
                phase2,
                phase3,
            },
        }
    }

    pub fn index_map(poseidon2_air: &Poseidon2Air<WIDTH, T>) -> Poseidon2ColsIndexMap<WIDTH> {
        let phase1_len = poseidon2_air.rounds_f / 2;
        let phase2_len = poseidon2_air.rounds_p;
        let phase3_len = poseidon2_air.rounds_f - phase1_len;

        let input = 0..WIDTH;
        let phase1: Vec<_> = (0..phase1_len)
            .map(|i| input.end + i * WIDTH..input.end + (i + 1) * WIDTH)
            .collect();
        let phase2: Vec<_> = (0..phase2_len)
            .map(|i| {
                phase1.last().unwrap().end + i * WIDTH..phase1.last().unwrap().end + (i + 1) * WIDTH
            })
            .collect();
        let phase3: Vec<_> = (0..phase3_len)
            .map(|i| {
                phase2.last().unwrap().end + i * WIDTH..phase2.last().unwrap().end + (i + 1) * WIDTH
            })
            .collect();
        let output = phase3.last().unwrap().end..phase3.last().unwrap().end + WIDTH;
        Poseidon2ColsIndexMap {
            input,
            output,
            phase1,
            phase2,
            phase3,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.input.to_vec();
        flattened.extend(self.aux.flatten());
        flattened.extend(self.io.output.to_vec());
        flattened
    }
}

impl<const WIDTH: usize, T: Field> Poseidon2Cols<WIDTH, T> {
    pub fn blank_row(poseidon2_air: &Poseidon2Air<WIDTH, T>) -> Self {
        let zero_row = [T::zero(); WIDTH];
        Poseidon2Cols::from_slice(
            poseidon2_air.generate_local_trace(zero_row).as_slice(),
            &Poseidon2Cols::<WIDTH, T>::index_map(poseidon2_air),
        )
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2IoCols<WIDTH, T> {
    pub fn get_width() -> usize {
        2 * WIDTH
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.input.to_vec();
        flattened.extend(self.output.to_vec());
        flattened
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2AuxCols<WIDTH, T> {
    pub fn get_width<F: Clone>(poseidon2_air: &Poseidon2Air<WIDTH, F>) -> usize {
        (poseidon2_air.rounds_f + poseidon2_air.rounds_p) * WIDTH
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened: Vec<T> = self.phase1.iter().flat_map(|s| s.to_vec()).collect();
        flattened.extend(self.phase2.iter().flat_map(|s| s.to_vec()));
        flattened.extend(self.phase3.iter().flat_map(|s| s.to_vec()));
        flattened
    }
}
