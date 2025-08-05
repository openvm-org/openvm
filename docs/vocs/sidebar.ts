import { SidebarItem } from "vocs";

export const sidebar: SidebarItem[] = [
    {
        text: "Getting Started",
        items: [
            {
                text: "Install",
                link: "/getting-started/install"
            },
            {
                text: "Quickstart",
                link: "/getting-started/quickstart"
            }
        ]
    },
    {
        text: "Writing Apps",
        items: [
            {
                text: "Overview",
                link: "/writing-apps/overview"
            },
            {
                text: "Writing a Program",
                link: "/writing-apps/writing-a-program"
            },
            {
                text: "Compiling",
                link: "/writing-apps/compiling"
            },
            {
                text: "Running a Program",
                link: "/writing-apps/running-a-program"
            },
            {
                text: "Generating Proofs",
                link: "/writing-apps/generating-proofs"
            },
            {
                text: "Verifying Proofs",
                link: "/writing-apps/verifying-proofs"
            },
            {
                text: "Solidity SDK",
                link: "/writing-apps/solidity-sdk"
            }
        ]
    },
    {
        text: "Acceleration Using Extensions",
        items: [
            {
                text: "Overview",
                link: "/acceleration-using-extensions/overview"
            },
            {
                text: "Keccak",
                link: "/acceleration-using-extensions/keccak"
            },
            {
                text: "SHA-256",
                link: "/acceleration-using-extensions/sha-256"
            },
            {
                text: "Big Integer",
                link: "/acceleration-using-extensions/big-integer"
            },
            {
                text: "Algebra (Modular Arithmetic)",
                link: "/acceleration-using-extensions/algebra"
            },
            {
                text: "Elliptic Curve Cryptography",
                link: "/acceleration-using-extensions/elliptic-curve-cryptography"
            },
            {
                text: "Elliptic Curve Pairing",
                link: "/acceleration-using-extensions/elliptic-curve-pairing"
            }
        ]
    },
    {
        text: "Guest Libraries",
        items: [
            {
                text: "Keccak256",
                link: "/guest-libraries/keccak256"
            },
            {
                text: "SHA2",
                link: "/guest-libraries/sha2"
            },
            {
                text: "Ruint",
                link: "/guest-libraries/ruint"
            },
            {
                text: "K256",
                link: "/guest-libraries/k256"
            },
            {
                text: "P256",
                link: "/guest-libraries/p256"
            },
            {
                text: "Pairing",
                link: "/guest-libraries/pairing"
            },
            {
                text: "Verify STARK",
                link: "/guest-libraries/verify-stark"
            }
        ]
    },
    {
        text: "Advanced Usage",
        items: [
            {
                text: "SDK",
                link: "/advanced-usage/sdk"
            },
            {
                text: "Creating a New Extension",
                link: "/advanced-usage/creating-a-new-extension"
            },
            {
                text: "Recursive Verification",
                link: "/advanced-usage/recursive-verification"
            }
        ]
    },
]