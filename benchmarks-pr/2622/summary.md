| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-2401d3a424612e185c0f47381c5ae5b7902f2924.md) | 4,176 |  12,000,265 |  1,360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-2401d3a424612e185c0f47381c5ae5b7902f2924.md) | 19,273 |  1,235,218 |  3,384 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-2401d3a424612e185c0f47381c5ae5b7902f2924.md) | 1,602 |  4,136,694 |  523 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-2401d3a424612e185c0f47381c5ae5b7902f2924.md) | 648 |  122,348 |  339 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-2401d3a424612e185c0f47381c5ae5b7902f2924.md) | 1,064 |  1,745,757 |  345 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-2401d3a424612e185c0f47381c5ae5b7902f2924.md) | 3,303 |  154,763 |  724 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2401d3a424612e185c0f47381c5ae5b7902f2924

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23565790804)
