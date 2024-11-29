#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::mem::transmute;

use regex_automata::{
    dfa::{dense::DFA, regex::Regex},
    util::{lazy::Lazy, wire::AlignAs},
    HalfMatch, Input,
};

// This crate provides its own "lazy" type, kind of like
// lazy_static! or once_cell::sync::Lazy. But it works in no-alloc
// no-std environments and let's us write this using completely
// safe code.
static FWD_DFA: Lazy<DFA<&'static [u32]>> = Lazy::new(|| {
    // This assignment is made possible (implicitly) via the
    // CoerceUnsized trait. This is what guarantees that our
    // bytes are stored in memory on a 4 byte boundary. You
    // *must* do this or something equivalent for correct
    // deserialization.
    static ALIGNED: &AlignAs<[u8], u32> = &AlignAs {
        _align: [],
        #[cfg(target_endian = "big")]
        bytes: *include_bytes!("../email.bigendian.fwd"),
        #[cfg(target_endian = "little")]
        bytes: *include_bytes!("../email.littleendian.fwd"),
    };

    let (dfa, _) = DFA::from_bytes(&ALIGNED.bytes).expect("serialized DFA should be valid");
    dfa
});
static REV_DFA: Lazy<DFA<&'static [u32]>> = Lazy::new(|| {
    // This assignment is made possible (implicitly) via the
    // CoerceUnsized trait. This is what guarantees that our
    // bytes are stored in memory on a 4 byte boundary. You
    // *must* do this or something equivalent for correct
    // deserialization.
    static ALIGNED: &AlignAs<[u8], u32> = &AlignAs {
        _align: [],
        #[cfg(target_endian = "big")]
        bytes: *include_bytes!("../email.bigendian.rev"),
        #[cfg(target_endian = "little")]
        bytes: *include_bytes!("../email.littleendian.rev"),
    };

    let (dfa, _) = DFA::from_bytes(&ALIGNED.bytes).expect("serialized DFA should be valid");
    dfa
});

axvm::entry!(main);

pub fn main() {
    let data = axvm::io::read_vec();
    let data = core::str::from_utf8(&data).expect("Invalid UTF-8");

    let re = Regex::builder().build_from_dfas(FWD_DFA.as_ref(), REV_DFA.as_ref());

    let caps = re.captures(data).expect("No match found.");
    let email = caps.name("email").expect("No email found.");
    let email_hash = axvm::intrinsics::keccak256(email.as_str().as_bytes());

    let email_hash = unsafe { transmute::<[u8; 32], [u32; 8]>(email_hash) };

    email_hash
        .into_iter()
        .enumerate()
        .for_each(|(i, x)| axvm::io::reveal(x, i));
}
