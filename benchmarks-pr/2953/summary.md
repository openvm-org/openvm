| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 980 |  4,000,051 |  400 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 15,473 |  14,365,133 |  2,968 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 8,225 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 1,205 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 438 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 579 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-f7d9ed0946ce5f5ec37f4274583e43f45ff2087c.md) | 3,837 |  1,979,971 |  866 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f7d9ed0946ce5f5ec37f4274583e43f45ff2087c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28951988826)
