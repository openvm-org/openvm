| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/fibonacci-7cba12f9fd1140139a73223dfbaa5541c1786a5d.md) | 3,844 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/keccak-7cba12f9fd1140139a73223dfbaa5541c1786a5d.md) | 18,629 |  18,655,329 |  3,328 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/regex-7cba12f9fd1140139a73223dfbaa5541c1786a5d.md) | 1,444 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/ecrecover-7cba12f9fd1140139a73223dfbaa5541c1786a5d.md) | 735 |  317,792 |  355 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/pairing-7cba12f9fd1140139a73223dfbaa5541c1786a5d.md) | 907 |  1,745,757 |  313 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/kitchen_sink-7cba12f9fd1140139a73223dfbaa5541c1786a5d.md) | 2,499 |  2,580,026 |  543 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7cba12f9fd1140139a73223dfbaa5541c1786a5d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23864198496)
