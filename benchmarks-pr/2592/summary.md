| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-a2b01fbc3bae7300691c9a891e7c8d6c6c438157.md) | 3,851 |  12,000,265 |  966 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-a2b01fbc3bae7300691c9a891e7c8d6c6c438157.md) | 18,434 |  18,655,329 |  3,301 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-a2b01fbc3bae7300691c9a891e7c8d6c6c438157.md) | 1,426 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-a2b01fbc3bae7300691c9a891e7c8d6c6c438157.md) | 649 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-a2b01fbc3bae7300691c9a891e7c8d6c6c438157.md) | 903 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-a2b01fbc3bae7300691c9a891e7c8d6c6c438157.md) | 2,281 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a2b01fbc3bae7300691c9a891e7c8d6c6c438157

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24049421816)
