#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "getrandom-v02")]
compile_error!("this program is not compatible with getrandom v0.2");

use getrandom::Error;

fn get_random_u128() -> Result<u128, Error> {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf)?;
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
#[cfg(all(feature = "getrandom", not(feature = "getrandom-unsupported")))]
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), Error> {
    for i in 0..len {
        *dest.add(i) = 0u8;
    }
    Ok(())
}
