| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 474 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 7,452 |  14,365,133 |  1,607 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 4,237 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 757 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 209 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 252 |  592,827 |  174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-b5becfd8ada194d2edf08e3f385e6398be0ebc16.md) | 2,220 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b5becfd8ada194d2edf08e3f385e6398be0ebc16

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38413377715)
