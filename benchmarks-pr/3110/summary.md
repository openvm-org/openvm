| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/fibonacci-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 482 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/keccak-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 7,301 |  14,365,133 |  1,503 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/sha2_bench-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 4,111 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/regex-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 647 |  4,090,656 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/ecrecover-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 221 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/pairing-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 237 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/kitchen_sink-1e385dfce807f32fa5408a3929470324f0c18c89.md) | 2,041 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1e385dfce807f32fa5408a3929470324f0c18c89

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31041528198)
