| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 1,024 |  4,000,051 |  386 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 15,693 |  14,365,133 |  3,015 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 8,359 |  11,167,961 |  1,016 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 1,192 |  4,090,656 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 434 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 594 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-7b16496bd45d11d07179302e2889540bab6d4f70.md) | 3,876 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7b16496bd45d11d07179302e2889540bab6d4f70

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28392008504)
