#![no_std]
#![no_main]
use core::panic::PanicInfo;

mod io;
use io::{exit_qemu, print_str, print_u64, EXIT_FAILURE, EXIT_SUCCESS};

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    print_str("PANIC!\n");
    exit_qemu(EXIT_FAILURE);
}

fn assert_eq(actual: u64, expected: u64, label: &str) {
    if actual != expected {
        print_str(label);
        print_str(": FAIL expected ");
        print_u64(expected);
        print_str(" got ");
        print_u64(actual);
        print_str("\n");
        exit_qemu(EXIT_FAILURE);
    }
}

// --- 64-bit arithmetic ---

fn test_64bit_arith() {
    print_str("64-bit arithmetic...\n");

    let a: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    let b: u64 = 1;
    assert_eq(a.wrapping_add(b), 0, "u64 overflow add");

    let x: i64 = -1;
    let y: i64 = -2;
    assert_eq((x.wrapping_mul(y)) as u64, 2, "i64 neg mul");

    let large: u64 = 1u64 << 48;
    assert_eq(large >> 16, 1u64 << 32, "u64 srl");
    assert_eq(large << 8, 1u64 << 56, "u64 sll");

    let neg: i64 = -(1i64 << 48);
    assert_eq((neg >> 16) as u64, (-(1i64 << 32)) as u64, "i64 sra");

    // division / remainder
    assert_eq(100u64 / 7, 14, "u64 div");
    assert_eq(100u64 % 7, 2, "u64 rem");

    let d: i64 = -100;
    assert_eq((d / 7) as u64, (-14i64) as u64, "i64 div neg");
    assert_eq((d % 7) as u64, (-2i64) as u64, "i64 rem neg");
}

// --- 32-bit W-type operations ---

fn test_word_ops() {
    print_str("32-bit W ops...\n");

    // ADDW sign-extends the 32-bit result to 64 bits
    let a: u32 = 0x7FFF_FFFF;
    let b: u32 = 1;
    let result = a.wrapping_add(b); // 0x8000_0000 as u32
    // When sign-extended to 64 bits, this is negative
    assert_eq(result as i32 as i64 as u64, (-2147483648i64) as u64, "addw sign ext");

    // MULW
    let x: u32 = 100_000;
    let y: u32 = 40_000;
    let product = x.wrapping_mul(y); // 4_000_000_000 wraps in u32 to -294967296
    assert_eq(product as u64, 4_000_000_000u64, "mulw");

    // DIVW / REMW
    let p: i32 = -100;
    let q: i32 = 7;
    assert_eq((p / q) as u64, (-14i64) as u64, "divw neg");
    assert_eq((p % q) as u64, (-2i64) as u64, "remw neg");
}

// --- Memory: byte/half/word/double loads/stores via arrays ---

fn test_memory() {
    print_str("Memory access...\n");

    let mut buf = [0u8; 32];
    buf[0] = 0xAB;
    buf[1] = 0xCD;
    buf[2] = 0xEF;
    buf[3] = 0x01;
    buf[4] = 0x23;
    buf[5] = 0x45;
    buf[6] = 0x67;
    buf[7] = 0x89;

    // byte load
    assert_eq(buf[0] as u64, 0xAB, "lb[0]");
    assert_eq(buf[7] as u64, 0x89, "lb[7]");

    // reinterpret as u16 (half-word load)
    let h: u16 = u16::from_le_bytes([buf[0], buf[1]]);
    assert_eq(h as u64, 0xCDAB, "lh");

    // reinterpret as u32 (word load)
    let w: u32 = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq(w as u64, 0x01EF_CDAB, "lw");

    // reinterpret as u64 (double-word load)
    let d: u64 = u64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]);
    assert_eq(d, 0x8967_4523_01EF_CDAB, "ld");

    // sign-extended word load (LW)
    let signed_w = w as i32; // 0x01EFCDAB is positive
    assert_eq(signed_w as i64 as u64, 0x01EF_CDAB, "lw sign ext pos");

    // write a negative i32 and check sign extension
    let neg_bytes = (-1i32).to_le_bytes();
    buf[8..12].copy_from_slice(&neg_bytes);
    let neg_w = i32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    assert_eq(neg_w as i64 as u64, u64::MAX, "lw sign ext neg");
}

// --- Sorting (exercises branches, comparisons, memory writes) ---

fn insertion_sort(arr: &mut [u64]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

fn test_sorting() {
    print_str("Sorting...\n");

    let mut arr = [
        0xDEAD_BEEF_CAFE_BABEu64,
        42,
        0xFFFF_FFFF_FFFF_FFFF,
        0,
        1_000_000_000_000,
        7,
        999,
        0x0000_0001_0000_0000,
    ];
    insertion_sort(&mut arr);

    assert_eq(arr[0], 0, "sort[0]");
    assert_eq(arr[1], 7, "sort[1]");
    assert_eq(arr[2], 42, "sort[2]");
    assert_eq(arr[3], 999, "sort[3]");
    assert_eq(arr[4], 0x0000_0001_0000_0000, "sort[4]");
    assert_eq(arr[5], 1_000_000_000_000, "sort[5]");
    assert_eq(arr[6], 0xDEAD_BEEF_CAFE_BABE, "sort[6]");
    assert_eq(arr[7], 0xFFFF_FFFF_FFFF_FFFF, "sort[7]");
}

// --- Bitwise operations ---

fn popcount(mut x: u64) -> u64 {
    let mut count = 0u64;
    while x != 0 {
        count += x & 1;
        x >>= 1;
    }
    count
}

fn test_bitwise() {
    print_str("Bitwise...\n");

    assert_eq(0xAAAA_AAAA_AAAA_AAAAu64 & 0x5555_5555_5555_5555, 0, "and");
    assert_eq(0xAAAA_AAAA_AAAA_AAAAu64 | 0x5555_5555_5555_5555, u64::MAX, "or");
    assert_eq(0xFFu64 ^ 0xFFu64, 0, "xor same");
    assert_eq(0xFFu64 ^ 0x00u64, 0xFF, "xor zero");

    assert_eq(popcount(0), 0, "popcount 0");
    assert_eq(popcount(1), 1, "popcount 1");
    assert_eq(popcount(u64::MAX), 64, "popcount max");
    assert_eq(popcount(0xAAAA_AAAA_AAAA_AAAA), 32, "popcount alt");
}

// --- Recursive GCD ---

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn test_recursion() {
    print_str("Recursion (GCD)...\n");

    assert_eq(gcd(48, 18), 6, "gcd(48,18)");
    assert_eq(gcd(1_000_000_007, 1_000_000_009), 1, "gcd primes");
    assert_eq(gcd(0, 42), 42, "gcd(0,42)");
    assert_eq(gcd(1u64 << 32, 1u64 << 16), 1u64 << 16, "gcd powers of 2");
}

// --- SLT / SLTU ---

fn test_comparisons() {
    print_str("Comparisons...\n");

    // Unsigned comparisons
    let a: u64 = 5;
    let b: u64 = 10;
    assert_eq(if a < b { 1 } else { 0 }, 1, "sltu 5<10");
    assert_eq(if b < a { 1 } else { 0 }, 0, "sltu 10<5");

    // Signed comparisons with negative numbers
    let neg: i64 = -1;
    let pos: i64 = 1;
    assert_eq(if neg < pos { 1 } else { 0 }, 1, "slt -1<1");
    assert_eq(if pos < neg { 1 } else { 0 }, 0, "slt 1<-1");

    // Large unsigned: -1 as u64 is MAX, which is > 1
    assert_eq(
        if (neg as u64) < (pos as u64) { 1 } else { 0 },
        0,
        "sltu MAX<1",
    );
}

fn main() {
    print_str("=== RV64IM stress test ===\n");

    test_64bit_arith();
    test_word_ops();
    test_memory();
    test_sorting();
    test_bitwise();
    test_recursion();
    test_comparisons();

    print_str("=== ALL TESTS PASSED ===\n");
    exit_qemu(EXIT_SUCCESS);
}
