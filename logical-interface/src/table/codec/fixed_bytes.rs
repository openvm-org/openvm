use std::{error::Error, fmt::Debug};

use alloy_primitives::FixedBytes;
use num_traits::ToBytes;
use serde_derive::{Deserialize, Serialize};

use crate::mock_db::utils::{bytes_to_fixed_bytes_be, string_to_fixed_bytes_be};

#[derive(Debug, Deserialize, Serialize)]
pub struct FixedBytesCodec<T, U, const INDEX_BYTES: usize, const DATA_BYTES: usize>
where
    T: Sized + ToBytes + PartialEq + Clone + for<'a> TryFrom<&'a [u8], Error = Box<dyn Error>>,
    U: Sized + ToBytes + Clone + for<'a> TryFrom<&'a [u8], Error = Box<dyn Error>>,
{
    _phantom: std::marker::PhantomData<(T, U)>,
}

impl<T, U, const INDEX_BYTES: usize, const DATA_BYTES: usize>
    FixedBytesCodec<T, U, INDEX_BYTES, DATA_BYTES>
where
    T: Sized + ToBytes + PartialEq + Clone + for<'a> TryFrom<&'a [u8], Error = Box<dyn Error>>,
    U: Sized + ToBytes + Clone + for<'a> TryFrom<&'a [u8], Error = Box<dyn Error>>,
{
    pub fn index_to_fixed_bytes(index: T) -> FixedBytes<INDEX_BYTES> {
        let index_bytes = index.to_be_bytes();
        let index_bytes = index_bytes.as_ref();
        // let serialized = bincode::serialize(&index).unwrap();
        // let index_slice = serialized.as_slice();
        // let index_arr: [u8; INDEX_BYTES] =
        //     index_slice.try_into().expect("Slice with incorrect length");
        FixedBytes::<INDEX_BYTES>::left_padding_from(index_bytes)
    }

    pub fn data_to_fixed_bytes(data: U) -> FixedBytes<DATA_BYTES> {
        let data_bytes = data.to_be_bytes();
        let data_bytes = data_bytes.as_ref();
        // let serialized = bincode::serialize(&data).unwrap();
        // let data_slice = serialized.as_slice();
        // let data_arr: [u8; DATA_BYTES] =
        //     data_slice.try_into().expect("Slice with incorrect length");
        FixedBytes::<DATA_BYTES>::left_padding_from(data_bytes)
    }

    pub fn fixed_bytes_to_index(index_fb: FixedBytes<INDEX_BYTES>) -> T {
        // let x: u32 = 5;
        // let b: FixedBytes<32> = FixedBytes::new([0; 32]);
        // let mm = b.try_into().unwrap();
        // let a = u128::from(1);
        // a.to_be_bytes();

        // let _conv = u32::try_from(b).

        let index_bytes: &[u8] = index_fb.as_ref();
        T::try_from(index_bytes).expect("Failed to convert index bytes")
    }

    pub fn fixed_bytes_to_data(data_fb: FixedBytes<DATA_BYTES>) -> U {
        let data_bytes: &[u8] = data_fb.as_ref();
        U::try_from(data_bytes).expect("Failed to convert data bytes")
    }
}
