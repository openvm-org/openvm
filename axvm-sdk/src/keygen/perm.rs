use std::cmp::Reverse;

use axvm_circuit::arch::{CONNECTOR_AIR_ID, PROGRAM_AIR_ID, PUBLIC_VALUES_AIR_ID};

// FIXME: is there a good crate to replace this?
pub struct AirIdPermutation {
    pub perm: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct SpecialAirIds {
    pub program_air_id: usize,
    pub connector_air_id: usize,
    pub public_values_air_id: usize,
}

impl AirIdPermutation {
    pub fn compute(heights: &[usize]) -> AirIdPermutation {
        let mut height_with_air_id: Vec<_> = heights.iter().copied().enumerate().collect();
        height_with_air_id.sort_by_key(|(_, h)| Reverse(*h));
        AirIdPermutation {
            perm: height_with_air_id
                .into_iter()
                .map(|(a_id, _)| a_id)
                .collect(),
        }
    }
    #[allow(dead_code)]
    pub fn get_special_air_ids(&self) -> SpecialAirIds {
        let perm_len = self.perm.len();
        let mut ret = SpecialAirIds {
            program_air_id: perm_len,
            connector_air_id: perm_len,
            public_values_air_id: perm_len,
        };
        for (i, &air_id) in self.perm.iter().enumerate() {
            if air_id == PROGRAM_AIR_ID {
                ret.program_air_id = i;
            } else if air_id == CONNECTOR_AIR_ID {
                ret.connector_air_id = i;
            } else if air_id == PUBLIC_VALUES_AIR_ID {
                ret.public_values_air_id = i;
            }
        }
        ret
    }
    /// arr[i] <- arr[perm[i]]
    pub(crate) fn permute<T>(&self, arr: &mut [T]) {
        debug_assert_eq!(arr.len(), self.perm.len());
        let mut perm = self.perm.clone();
        for i in 0..perm.len() {
            if perm[i] != i {
                let mut curr = i;
                loop {
                    let target = perm[curr];
                    perm[curr] = curr;
                    if perm[target] == target {
                        break;
                    }
                    arr.swap(curr, target);
                    curr = target;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::keygen::perm::AirIdPermutation;

    #[test]
    fn test_air_id_permutation() {
        {
            let perm = AirIdPermutation {
                perm: vec![2, 0, 1, 3],
            };
            let mut arr = vec![0, 100, 200, 300];
            perm.permute(&mut arr);
            assert_eq!(arr, vec![200, 0, 100, 300]);
        }
        {
            let perm = AirIdPermutation {
                perm: vec![0, 1, 2, 3],
            };
            let mut arr = vec![0, 100, 200, 300];
            perm.permute(&mut arr);
            assert_eq!(arr, vec![0, 100, 200, 300]);
        }
    }
}
