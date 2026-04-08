| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/fibonacci-b356a2202089a19d345c8160e2ec7bad4d6e68bd.md) | 3,825 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/keccak-b356a2202089a19d345c8160e2ec7bad4d6e68bd.md) | 15,856 |  1,235,218 |  2,227 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/regex-b356a2202089a19d345c8160e2ec7bad4d6e68bd.md) | 1,418 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/ecrecover-b356a2202089a19d345c8160e2ec7bad4d6e68bd.md) | 637 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/pairing-b356a2202089a19d345c8160e2ec7bad4d6e68bd.md) | 922 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/kitchen_sink-b356a2202089a19d345c8160e2ec7bad4d6e68bd.md) | 2,431 |  154,763 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b356a2202089a19d345c8160e2ec7bad4d6e68bd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24142202617)
