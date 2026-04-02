| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/fibonacci-c97840dda454bdca16e01a65b88ad765a574790e.md) | 3,787 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/keccak-c97840dda454bdca16e01a65b88ad765a574790e.md) | 15,650 |  1,235,218 |  2,197 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/regex-c97840dda454bdca16e01a65b88ad765a574790e.md) | 1,409 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/ecrecover-c97840dda454bdca16e01a65b88ad765a574790e.md) | 642 |  122,348 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/pairing-c97840dda454bdca16e01a65b88ad765a574790e.md) | 911 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/kitchen_sink-c97840dda454bdca16e01a65b88ad765a574790e.md) | 2,363 |  154,763 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c97840dda454bdca16e01a65b88ad765a574790e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23914653662)
