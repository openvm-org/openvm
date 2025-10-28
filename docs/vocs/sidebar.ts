import { SidebarItem } from "vocs";

export const specsSidebar: SidebarItem[] = [
    {
        text: "OpenVM Design",
        items: [
            {
                text: "Overview",
                link: "/specs/openvm/overview"
            },
            {
                text: "Modular ISA Design",
                link: "/specs/openvm/isa"
            },
        ]
    },
    {
        text: "VM Architecture",
        items: [
            {
                text: "Circuit Architecture",
                link: "/specs/architecture/circuit-architecture"
            },
            {
                text: "Memory Design",
                link: "/specs/architecture/memory"
            },
            {
                text: "Continuations Design",
                link: "/specs/architecture/continuations"
            },
            {
                text: "Distributed Proving",
                link: "/specs/architecture/distributed-proving"
            }
        ]
    },
    {
        text: "Security",
        items: [
            {
                text: "Security Model",
                link: "/specs/security/security-model"
            },
            {
                text: "Audits",
                link: "/specs/security/audits"
            },
            {
                text: "Reporting Vulnerabilities",
                link: "/specs/security/reporting-vulnerabilities"
            }
        ]
    },
    {
        text: "OpenVM Reference",
        items: [
            {
                text: "Instruction Reference",
                link: "/specs/reference/instruction-reference"
            },
            {
                text: "RISC-V Custom Code",
                link: "/specs/reference/riscv-custom-code"
            },
            {
                text: "RISC-V Transpiler",
                link: "/specs/reference/transpiler"
            },
            {
                text: "Rust Frontend",
                link: "/specs/reference/rust-frontend"
            }
        ]
    }
]

export const bookSidebar: SidebarItem[] = [
    {
        text: "Introduction",
        link: "/book/getting-started/introduction"
    },
    {
        text: "Install",
        link: "/book/getting-started/install"
    },
    {
        text: "Quickstart",
        link: "/book/getting-started/quickstart"
    },
    {
        text: "Writing Apps",
        items: [
            {
                text: "Overview",
                link: "/book/writing-apps/overview"
            },
            {
                text: "Writing a Program",
                link: "/book/writing-apps/writing-a-program"
            },
            {
                text: "Compiling a Program",
                link: "/book/writing-apps/compiling-a-program"
            },
            {
                text: "Running a Program",
                link: "/book/writing-apps/running-a-program"
            },
            {
                text: "Generating Proofs",
                link: "/book/writing-apps/generating-proofs"
            },
            {
                text: "Verifying Proofs",
                link: "/book/writing-apps/verifying-proofs"
            },
            {
                text: "Solidity SDK",
                link: "/book/writing-apps/solidity-sdk"
            }
        ]
    },
    {
        text: "Acceleration Using Extensions",
        items: [
            {
                text: "Overview",
                link: "/book/acceleration-using-extensions/overview"
            },
            {
                text: "Keccak",
                link: "/book/acceleration-using-extensions/keccak"
            },
            {
                text: "SHA-256",
                link: "/book/acceleration-using-extensions/sha-256"
            },
            {
                text: "Big Integer",
                link: "/book/acceleration-using-extensions/big-integer"
            },
            {
                text: "Algebra (Modular Arithmetic)",
                link: "/book/acceleration-using-extensions/algebra"
            },
            {
                text: "Elliptic Curve Cryptography",
                link: "/book/acceleration-using-extensions/elliptic-curve-cryptography"
            },
            {
                text: "Elliptic Curve Pairing",
                link: "/book/acceleration-using-extensions/elliptic-curve-pairing"
            }
        ]
    },
    {
        text: "Guest Libraries",
        items: [
            {
                text: "Keccak256",
                link: "/book/guest-libraries/keccak256"
            },
            {
                text: "SHA2",
                link: "/book/guest-libraries/sha2"
            },
            {
                text: "Ruint",
                link: "/book/guest-libraries/ruint"
            },
            {
                text: "K256",
                link: "/book/guest-libraries/k256"
            },
            {
                text: "P256",
                link: "/book/guest-libraries/p256"
            },
            {
                text: "Pairing",
                link: "/book/guest-libraries/pairing"
            },
            {
                text: "Verify STARK",
                link: "/book/guest-libraries/verify-stark"
            }
        ]
    },
    {
        text: "Advanced Usage",
        items: [
            {
                text: "SDK",
                link: "/book/advanced-usage/sdk"
            },
            {
                text: "Creating a New Extension",
                link: "/book/advanced-usage/creating-a-new-extension"
            },
            {
                text: "Recursive Verification",
                link: "/book/advanced-usage/recursive-verification"
            }
        ]
    },
]