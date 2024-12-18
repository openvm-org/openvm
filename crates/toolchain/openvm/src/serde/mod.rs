//! Custom serialization and deserialization library that works on
//! `serde::Serialize` and `serde::Deserialize` traits.

// Initial version copied from <https://github.com/risc0/risc0/blob/9a10467f897b9e4a54f3cdf35c3d88367bfd9028/risc0/zkvm/src/serde/mod.rs#L1> under Apache License.

mod deserializer;
mod err;
mod serializer;

pub use deserializer::{from_slice, Deserializer, WordRead};
pub use err::{Error, Result};
pub use serializer::{to_vec, to_vec_with_capacity, Serializer, WordWrite};

#[cfg(test)]
mod tests {
    use alloc::{collections::BTreeMap, string::String, vec, vec::Vec};

    use chrono::NaiveDate;

    use crate::serde::{from_slice, to_vec};

    #[test]
    fn test_vec_round_trip() {
        let input: Vec<u64> = vec![1, 2, 3];
        let data = to_vec(&input).unwrap();
        let output: Vec<u64> = from_slice(data.as_slice()).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_map_round_trip() {
        let input: BTreeMap<String, u32> =
            BTreeMap::from([("foo".into(), 1), ("bar".into(), 2), ("baz".into(), 3)]);
        let data = to_vec(&input).unwrap();
        let output: BTreeMap<String, u32> = from_slice(data.as_slice()).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_tuple_round_trip() {
        let input: (u32, u64) = (1, 2);
        let data = to_vec(&input).unwrap();
        let output: (u32, u64) = from_slice(data.as_slice()).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn naive_date_round_trip() {
        let input: NaiveDate = NaiveDate::parse_from_str("2015-09-05", "%Y-%m-%d").unwrap();
        let date_vec = to_vec(&input).unwrap();
        let output: NaiveDate = from_slice(date_vec.as_slice()).unwrap();
        assert_eq!(input, output);
    }
}
