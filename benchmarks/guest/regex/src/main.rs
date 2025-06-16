use regex::Regex;

const PATTERN: &str = r"(?m)(\r\n|^)From:([^\r\n]+<)?(?P<email>[^<>]+)>?";

pub fn main() {
    let data = openvm::io::read_vec();
    let data = core::str::from_utf8(&data).expect("Invalid UTF-8");

    // Compile the regex
    let re = Regex::new(PATTERN).expect("Invalid regex");

    let caps = re.captures(data).expect("No match found.");
    let email = caps.name("email").expect("No email found.");
    let email_hash = openvm_keccak256::keccak256(email.as_str().as_bytes());

    openvm::io::reveal_bytes32(email_hash);
}
