| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 475 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 7,286 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 4,771 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 685 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 227 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 276 |  592,827 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-cdb6430c60307e04634e5976b13c3dcd0290f2d4.md) | 2,687 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cdb6430c60307e04634e5976b13c3dcd0290f2d4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29751274659)
