| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-d74395cace54521e6402673f920d221953ee89fb.md) | 1,033 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-d74395cace54521e6402673f920d221953ee89fb.md) | 15,837 |  14,365,133 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-d74395cace54521e6402673f920d221953ee89fb.md) | 8,147 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-d74395cace54521e6402673f920d221953ee89fb.md) | 1,173 |  4,090,656 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-d74395cace54521e6402673f920d221953ee89fb.md) | 431 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-d74395cace54521e6402673f920d221953ee89fb.md) | 594 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-d74395cace54521e6402673f920d221953ee89fb.md) | 3,884 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d74395cace54521e6402673f920d221953ee89fb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28394458256)
