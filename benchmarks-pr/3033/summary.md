| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/fibonacci-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 410 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/keccak-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 8,693 |  14,365,133 |  1,554 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/sha2_bench-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 4,261 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/regex-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 574 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/ecrecover-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 222 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/pairing-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 288 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3033/kitchen_sink-c508f53ff3d981ce9c4897eb85207faf7e8f230c.md) | 1,938 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c508f53ff3d981ce9c4897eb85207faf7e8f230c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29584639340)
