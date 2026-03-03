#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "getrandom")]
compile_error!("this program is not compatible with getrandom v0.2");
use getrandom_v02::Error;
// For this to work, need to enable the "custom" feature in getrandom-v02
#[cfg(all(feature = "getrandom-v02", not(feature = "getrandom-unsupported")))]
getrandom_v02::register_custom_getrandom!(__getrandom_v02_custom);

fn get_random_u128() -> Result<u128, Error> {
    let mut buf = [0u8; 16];
    getrandom_v02::getrandom(&mut buf)?;
    Ok(u128::from_ne_bytes(buf))
}

openvm::entry!(main);

pub fn main() {
    // do unrelated stuff
    let mut c = core::hint::black_box(0);
    for _ in 0..10 {
        c += 1;
    }

    #[cfg(not(feature = "getrandom-unsupported"))]
    {
        // not a good random function!
        assert_eq!(get_random_u128(), Ok(0));
    }
    #[cfg(feature = "getrandom-unsupported")]
    {
        assert!(get_random_u128().is_err());
    }
}

// custom user-specified getrandom
#[cfg(all(feature = "getrandom-v02", not(feature = "getrandom-unsupported")))]
pub fn __getrandom_v02_custom(dest: &mut [u8]) -> Result<(), Error> {
    for byte in dest {
        *byte = 0u8;
    }
    Ok(())
}
