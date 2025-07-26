use std::cmp::Reverse;

#[cfg(feature = "evm-prove")]
use openvm_continuations::verifier::common::types::SpecialAirIds;

/// Permutation of the AIR IDs to order them by forced trace heights.
pub(crate) struct AirIdPermutation {
    pub perm: Vec<usize>,
}

impl AirIdPermutation {
    pub fn compute(heights: &[u32]) -> AirIdPermutation {
        let mut height_with_air_id: Vec<_> = heights.iter().copied().enumerate().collect();
        height_with_air_id.sort_by_key(|(_, h)| Reverse(*h));
        AirIdPermutation {
            perm: height_with_air_id
                .into_iter()
                .map(|(a_id, _)| a_id)
                .collect(),
        }
    }
    #[cfg(feature = "evm-prove")]
    pub fn get_special_air_ids(&self) -> SpecialAirIds {
        use openvm_circuit::arch::{CONNECTOR_AIR_ID, PROGRAM_AIR_ID, PUBLIC_VALUES_AIR_ID};

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
        debug_assert_ne!(ret.program_air_id, perm_len, "Program AIR not found");
        debug_assert_ne!(ret.connector_air_id, perm_len, "Connector AIR not found");
        debug_assert_ne!(
            ret.public_values_air_id, perm_len,
            "Public Values AIR not found"
        );
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
