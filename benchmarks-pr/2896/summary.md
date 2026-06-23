| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/fibonacci-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 1,017 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/keccak-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 16,111 |  14,365,133 |  3,014 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/sha2_bench-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 8,336 |  11,167,961 |  1,023 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/regex-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 1,176 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/ecrecover-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 435 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/pairing-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 604 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2896/kitchen_sink-dce127c8d62c9ccc12de810c9b2e3dbc0574d124.md) | 3,857 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dce127c8d62c9ccc12de810c9b2e3dbc0574d124

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28048661131)
