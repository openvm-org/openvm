| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 462 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 7,361 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 4,163 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 656 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 222 |  112,210 |  195 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 234 |  592,827 |  201 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb.md) | 2,025 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6b439b1c42e6fd40b846cdd05def1a7b2dcb68cb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31537112787)
