| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-dec4b7c0e14b281394e28e0a302e74beb16a8a6b.md) | 3,785 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-dec4b7c0e14b281394e28e0a302e74beb16a8a6b.md) | 15,619 |  1,235,218 |  2,166 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-dec4b7c0e14b281394e28e0a302e74beb16a8a6b.md) | 1,417 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-dec4b7c0e14b281394e28e0a302e74beb16a8a6b.md) | 635 |  122,348 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-dec4b7c0e14b281394e28e0a302e74beb16a8a6b.md) | 928 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-dec4b7c0e14b281394e28e0a302e74beb16a8a6b.md) | 2,359 |  154,763 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dec4b7c0e14b281394e28e0a302e74beb16a8a6b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23612446245)
