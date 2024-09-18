use p3_air::AirBuilder;
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
    pub phase1: Vec<Poseidon2ExternalRoundCols<WIDTH, T>>,
    // contains one state (array of length WIDTH) for each round of phase2, of which there are `rounds_p`
    pub phase2: Vec<Poseidon2InternalRoundCols<WIDTH, T>>,
    // contains one state (array of length WIDTH) for each round of phase3, of which there are `rounds_f - rounds_f/2`
    pub phase3: Vec<Poseidon2ExternalRoundCols<WIDTH, T>>,
}

#[derive(Clone, Debug)]
pub struct Poseidon2ExternalRoundCols<const WIDTH: usize, T> {
    pub intermediate_sbox_powers: [Option<T>; WIDTH],
    pub round_output: [T; WIDTH],
}

#[derive(Clone, Debug)]
pub struct Poseidon2InternalRoundCols<const WIDTH: usize, T> {
    pub intermediate_sbox_power: Option<T>,
    pub round_output: [T; WIDTH],
}

impl<const WIDTH: usize, T: Field> Poseidon2Cols<WIDTH, T> {
    pub fn blank_row(p2_air: &Poseidon2Air<WIDTH, T>) -> Self {
        let zero_row = [T::zero(); WIDTH];
        p2_air.generate_local_trace(zero_row)
    }
}

fn need_intermediate_sbox_powers<const WIDTH: usize, T>(p2_air: &Poseidon2Air<WIDTH, T>) -> bool {
    p2_air.max_constraint_degree < 7
}

// Straightforward implementation for the functions from_slice, flatten, and width, into_expr below

impl<const WIDTH: usize, T: Clone> Poseidon2ExternalRoundCols<WIDTH, T> {
    fn from_slice<F>(slice: &[T], p2_air: &Poseidon2Air<WIDTH, F>) -> Self {
        assert!(slice.len() == Poseidon2ExternalRoundCols::<WIDTH, T>::width(p2_air));

        if need_intermediate_sbox_powers(p2_air) {
            Self {
                intermediate_sbox_powers: core::array::from_fn(|i| Some(slice[i].clone())),
                round_output: core::array::from_fn(|i| slice[WIDTH + i].clone()),
            }
        } else {
            Self {
                intermediate_sbox_powers: core::array::from_fn(|_| None),
                round_output: core::array::from_fn(|i| slice[i].clone()),
            }
        }
    }
}

impl<const WIDTH: usize, T> Poseidon2ExternalRoundCols<WIDTH, T> {
    fn flatten(self) -> Vec<T> {
        self.intermediate_sbox_powers
            .into_iter()
            .flatten()
            .chain(self.round_output)
            .collect()
    }

    fn width<F>(p2_air: &Poseidon2Air<WIDTH, F>) -> usize {
        if need_intermediate_sbox_powers(p2_air) {
            2 * WIDTH
        } else {
            WIDTH
        }
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2InternalRoundCols<WIDTH, T> {
    fn from_slice<F>(slice: &[T], p2_air: &Poseidon2Air<WIDTH, F>) -> Self {
        if need_intermediate_sbox_powers(p2_air) {
            Self {
                intermediate_sbox_power: Some(slice[0].clone()),
                round_output: core::array::from_fn(|i| slice[1 + i].clone()),
            }
        } else {
            Self {
                intermediate_sbox_power: None,
                round_output: core::array::from_fn(|i| slice[i].clone()),
            }
        }
    }

    fn flatten(self) -> Vec<T> {
        self.intermediate_sbox_power
            .into_iter()
            .chain(self.round_output)
            .collect()
    }

    fn width<F>(p2_air: &Poseidon2Air<WIDTH, F>) -> usize {
        if need_intermediate_sbox_powers(p2_air) {
            1 + WIDTH
        } else {
            WIDTH
        }
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2Cols<WIDTH, T> {
    pub fn width<F: Clone>(poseidon2_air: &Poseidon2Air<WIDTH, F>) -> usize {
        Poseidon2IoCols::<WIDTH, T>::width() + Poseidon2AuxCols::<WIDTH, T>::width(poseidon2_air)
    }

    pub fn from_slice<F>(slice: &[T], p2_air: &Poseidon2Air<WIDTH, F>) -> Self {
        Self {
            io: Poseidon2IoCols::from_slice(&slice[0..2 * WIDTH]),
            aux: Poseidon2AuxCols::from_slice(&slice[2 * WIDTH..], p2_air),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2IoCols<WIDTH, T> {
    fn from_slice(slice: &[T]) -> Self {
        Self {
            input: core::array::from_fn(|i| slice[i].clone()),
            output: core::array::from_fn(|i| slice[WIDTH + i].clone()),
        }
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2AuxCols<WIDTH, T> {
    fn from_slice<F>(slice: &[T], p2_air: &Poseidon2Air<WIDTH, F>) -> Self {
        let external_round_width = Poseidon2ExternalRoundCols::<WIDTH, T>::width(p2_air);
        let internal_round_width = Poseidon2InternalRoundCols::<WIDTH, T>::width(p2_air);

        let mut phase1 = vec![];
        let mut phase2 = vec![];
        let mut phase3 = vec![];

        let mut start = 0;
        let mut end = start;

        for _ in 0..p2_air.rounds_f / 2 {
            end += external_round_width;
            phase1.push(Poseidon2ExternalRoundCols::from_slice(
                &slice[start..end],
                p2_air,
            ));
            start = end;
        }

        for _ in 0..p2_air.rounds_p {
            end += internal_round_width;
            phase2.push(Poseidon2InternalRoundCols::from_slice(
                &slice[start..end],
                p2_air,
            ));
            start = end;
        }

        for _ in 0..p2_air.rounds_f - p2_air.rounds_f / 2 {
            end += external_round_width;
            phase3.push(Poseidon2ExternalRoundCols::from_slice(
                &slice[start..end],
                p2_air,
            ));
            start = end;
        }

        Self {
            phase1,
            phase2,
            phase3,
        }
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2IoCols<WIDTH, T> {
    pub fn width() -> usize {
        2 * WIDTH
    }

    pub fn flatten(self) -> Vec<T> {
        self.input.into_iter().chain(self.output).collect()
    }
}

impl<const WIDTH: usize, T: Clone> Poseidon2AuxCols<WIDTH, T> {
    pub fn width<F>(p2_air: &Poseidon2Air<WIDTH, F>) -> usize {
        p2_air.rounds_f * Poseidon2ExternalRoundCols::<WIDTH, T>::width(p2_air)
            + p2_air.rounds_p * Poseidon2InternalRoundCols::<WIDTH, T>::width(p2_air)
    }

    pub fn flatten(self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.extend(self.phase1.into_iter().flat_map(|s| s.flatten()));
        flattened.extend(self.phase2.into_iter().flat_map(|s| s.flatten()));
        flattened.extend(self.phase3.into_iter().flat_map(|s| s.flatten()));
        flattened
    }
}

impl<const WIDTH: usize, T> Poseidon2InternalRoundCols<WIDTH, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> Poseidon2InternalRoundCols<WIDTH, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        Poseidon2InternalRoundCols {
            intermediate_sbox_power: self.intermediate_sbox_power.map(Into::into),
            round_output: self.round_output.map(Into::into),
        }
    }
}

impl<const WIDTH: usize, T> Poseidon2ExternalRoundCols<WIDTH, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> Poseidon2ExternalRoundCols<WIDTH, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        Poseidon2ExternalRoundCols {
            intermediate_sbox_powers: self.intermediate_sbox_powers.map(|op| op.map(Into::into)),
            round_output: self.round_output.map(Into::into),
        }
    }
}

impl<const WIDTH: usize, T> Poseidon2AuxCols<WIDTH, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> Poseidon2AuxCols<WIDTH, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        Poseidon2AuxCols {
            phase1: self
                .phase1
                .into_iter()
                .map(|p| p.into_expr::<AB>())
                .collect(),
            phase2: self
                .phase2
                .into_iter()
                .map(|p| p.into_expr::<AB>())
                .collect(),
            phase3: self
                .phase3
                .into_iter()
                .map(|p| p.into_expr::<AB>())
                .collect(),
        }
    }
}
