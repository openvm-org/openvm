| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2638/fibonacci-d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0.md) | 3,824 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2638/keccak-d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0.md) | 15,503 |  1,235,218 |  2,161 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2638/regex-d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0.md) | 1,421 |  4,136,694 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2638/ecrecover-d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0.md) | 634 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2638/pairing-d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0.md) | 917 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2638/kitchen_sink-d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0.md) | 2,364 |  154,763 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d37be81ae8b2ce1e5b2fe48fcca5f7c833b687f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23816190371)
