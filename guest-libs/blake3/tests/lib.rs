//! Integration tests for openvm-blake3
//!
//! These tests verify BLAKE3 works correctly when compiled and executed on OpenVM.

use openvm_blake3::blake3;

#[test]
fn test_blake3_basic() {
    let hash = blake3(b"hello world");
    let expected =
        hex::decode("d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24").unwrap();
    assert_eq!(hash.as_slice(), expected.as_slice());
}

#[test]
fn test_blake3_empty() {
    let hash = blake3(b"");
    let expected =
        hex::decode("af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262").unwrap();
    assert_eq!(hash.as_slice(), expected.as_slice());
}

#[test]
fn test_blake3_longer_input() {
    // Test with 1KB input
    let input = vec![0u8; 1024];
    let hash = blake3(&input);
    // BLAKE3 hash of 1024 zero bytes (verified against blake3 reference implementation)
    let expected =
        hex::decode("d6fd9de5bccf223f523b316c9cd1cf9a9d87ea42473d68e011dad13f09bf8917").unwrap();
    assert_eq!(hash.as_slice(), expected.as_slice());
}
