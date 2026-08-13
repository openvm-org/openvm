| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 451 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 7,460 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 4,163 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 666 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 197 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 240 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d.md) | 2,049 |  1,979,971 |  528 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/95413d6b8dd1e7a364573ceda9ff6e591a0e6f9d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31677707412)
