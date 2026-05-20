| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/fibonacci-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 1,574 |  4,000,051 |  447 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/keccak-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 14,008 |  14,365,133 |  2,401 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/sha2_bench-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 9,301 |  11,167,961 |  1,412 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/regex-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 1,458 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/ecrecover-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 476 |  112,210 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/pairing-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 595 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/kitchen_sink-46617d63bccb1afd11bab978ba7b13e8b0b675c1.md) | 2,161 |  1,979,971 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/46617d63bccb1afd11bab978ba7b13e8b0b675c1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26192896712)
