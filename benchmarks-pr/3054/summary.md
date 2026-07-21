| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 466 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 7,236 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 4,679 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 672 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 231 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 315 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-5a4dc61247e0d792699a2fb9bafdccc7717c0f23.md) | 2,684 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5a4dc61247e0d792699a2fb9bafdccc7717c0f23

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29865650544)
