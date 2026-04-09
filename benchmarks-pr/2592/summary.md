| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-5b4514f2b42431e691bb1c0e5393fc36d7d04e76.md) | 3,799 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-5b4514f2b42431e691bb1c0e5393fc36d7d04e76.md) | 18,557 |  18,655,329 |  3,314 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-5b4514f2b42431e691bb1c0e5393fc36d7d04e76.md) | 1,431 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-5b4514f2b42431e691bb1c0e5393fc36d7d04e76.md) | 646 |  123,583 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-5b4514f2b42431e691bb1c0e5393fc36d7d04e76.md) | 908 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-5b4514f2b42431e691bb1c0e5393fc36d7d04e76.md) | 2,152 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5b4514f2b42431e691bb1c0e5393fc36d7d04e76

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24205609348)
