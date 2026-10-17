#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use elliptic_curve::{point::AffineCoordinates, sec1::ToEncodedPoint};
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{weierstrass::WeierstrassPoint, CyclicGroup};
use openvm_p256::{P256Coord as Coord, P256Point as Point};

openvm::init!("openvm_init_point_observations.rs");
openvm::entry!(main);

pub fn main() {
    let case = openvm::io::read_vec()[0];
    let mut point = Point::GENERATOR;
    if case != 0 {
        let coordinate = if matches!(case, 1 | 3 | 7) {
            point.x_mut()
        } else {
            point.y_mut()
        };
        *coordinate = Coord::from_repr(Coord::MODULUS);
    }
    match case {
        0 => {
            assert_eq!(point.as_le_bytes(), point.as_le_bytes_unchecked());
            let x = point.x_be_bytes();
            let y = point.y_be_bytes();
            assert_eq!(&AffineCoordinates::x(&point)[..], x);
            assert_eq!(point.y_is_odd().unwrap_u8(), y[31] & 1);
            for compress in [false, true] {
                let encoded = point.to_encoded_point(compress);
                assert_eq!(&encoded.as_bytes()[1..33], x);
                if compress {
                    assert_eq!(encoded.as_bytes()[0], 2 | (y[31] & 1));
                } else {
                    assert_eq!(&encoded.as_bytes()[33..], y);
                }
            }
        }
        1 => {
            black_box(point.x_be_bytes());
        }
        2 => {
            black_box(point.y_be_bytes());
        }
        3 => {
            black_box(AffineCoordinates::x(&point));
        }
        4 => {
            black_box(point.y_is_odd());
        }
        // Keep x reduced and nonzero so is_identity() skips the y comparison.
        5 | 6 => {
            black_box(point.to_encoded_point(case == 5));
        }
        7 | 8 => {
            black_box(point.as_le_bytes());
        }
        9 => {
            black_box(openvm::serde::to_vec(point.y()).unwrap());
        }
        _ => panic!("unknown case"),
    }
}
