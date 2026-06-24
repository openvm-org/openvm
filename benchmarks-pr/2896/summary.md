| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 1,040 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 16,192 |  14,365,133 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 8,364 |  11,167,961 |  1,024 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 1,205 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 435 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 602 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-fb8e03c51884fdd0a1d67a7521769681b05fda3b.md) | 3,885 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fb8e03c51884fdd0a1d67a7521769681b05fda3b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28112261314)
