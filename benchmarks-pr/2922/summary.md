| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-d3c14f09540826125e12ed76c372d308238a4da8.md) | 1,016 |  4,000,051 |  386 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-d3c14f09540826125e12ed76c372d308238a4da8.md) | 15,223 |  14,365,133 |  2,991 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-d3c14f09540826125e12ed76c372d308238a4da8.md) | 7,746 |  11,167,961 |  993 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-d3c14f09540826125e12ed76c372d308238a4da8.md) | 1,152 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-d3c14f09540826125e12ed76c372d308238a4da8.md) | 436 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-d3c14f09540826125e12ed76c372d308238a4da8.md) | 557 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-d3c14f09540826125e12ed76c372d308238a4da8.md) | 3,781 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d3c14f09540826125e12ed76c372d308238a4da8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28098773316)
