| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/fibonacci-38792216df00300054f151299cc86091b60aa622.md) | 477 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/keccak-38792216df00300054f151299cc86091b60aa622.md) | 7,635 |  14,365,133 |  1,638 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/sha2_bench-38792216df00300054f151299cc86091b60aa622.md) | 4,407 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/regex-38792216df00300054f151299cc86091b60aa622.md) | 772 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/ecrecover-38792216df00300054f151299cc86091b60aa622.md) | 207 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/pairing-38792216df00300054f151299cc86091b60aa622.md) | 252 |  592,827 |  176 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/kitchen_sink-38792216df00300054f151299cc86091b60aa622.md) | 2,250 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/38792216df00300054f151299cc86091b60aa622

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33808357257)
