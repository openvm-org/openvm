| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 410 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 8,537 |  14,365,133 |  1,503 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 4,217 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 585 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 222 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 284 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-41b5ded8f27b7ab8cf33deab4593a2206faacceb.md) | 1,940 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/41b5ded8f27b7ab8cf33deab4593a2206faacceb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29658421761)
