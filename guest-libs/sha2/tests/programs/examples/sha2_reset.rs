#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_sha2::{Digest, Sha256, Sha384, Sha512};

openvm::entry!(main);

macro_rules! assert_reset_contract {
    ($ty:ty) => {{
        let mut hasher = <$ty>::new();
        hasher.update(b"prefix");
        hasher.reset();
        assert_eq!(hasher.finalize().as_slice(), <$ty>::digest(b"").as_slice());

        let mut hasher = <$ty>::new();
        hasher.update(b"prefix");
        let mut output = <$ty>::digest(b"");
        hasher.finalize_into_reset(&mut output);
        assert_eq!(output.as_slice(), <$ty>::digest(b"prefix").as_slice());

        hasher.update(b"suffix");
        assert_eq!(
            hasher.finalize().as_slice(),
            <$ty>::digest(b"suffix").as_slice()
        );
    }};
}

pub fn main() {
    assert_reset_contract!(Sha256);
    assert_reset_contract!(Sha384);
    assert_reset_contract!(Sha512);
}
