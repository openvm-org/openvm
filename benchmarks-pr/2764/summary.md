| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/fibonacci-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 3,805 |  12,000,265 |  966 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/keccak-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 19,087 |  18,655,329 |  3,355 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/sha2_bench-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 8,996 |  14,793,960 |  1,384 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/regex-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 1,428 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/ecrecover-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 651 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/pairing-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 903 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/kitchen_sink-c7c9d042b8ed0534e6c8a80085d0664ff07e89fd.md) | 2,039 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c7c9d042b8ed0534e6c8a80085d0664ff07e89fd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25184376283)
