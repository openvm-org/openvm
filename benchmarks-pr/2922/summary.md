| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 1,027 |  4,000,051 |  401 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 15,226 |  14,365,133 |  3,001 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 7,835 |  11,167,961 |  997 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 1,160 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 433 |  112,210 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 555 |  592,827 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-0672b159accb6a8a87dae24635dec3347a8ca34f.md) | 3,760 |  1,979,971 |  855 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0672b159accb6a8a87dae24635dec3347a8ca34f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28097754697)
