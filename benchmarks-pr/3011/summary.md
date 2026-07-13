| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/fibonacci-e795c692c117f6792775922086690470e23c030d.md) | 878 |  4,000,051 |  393 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/keccak-e795c692c117f6792775922086690470e23c030d.md) | 15,448 |  14,365,133 |  3,042 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/sha2_bench-e795c692c117f6792775922086690470e23c030d.md) | 8,019 |  11,167,961 |  1,010 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/regex-e795c692c117f6792775922086690470e23c030d.md) | 1,026 |  4,090,656 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/ecrecover-e795c692c117f6792775922086690470e23c030d.md) | 301 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/pairing-e795c692c117f6792775922086690470e23c030d.md) | 447 |  592,827 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3011/kitchen_sink-e795c692c117f6792775922086690470e23c030d.md) | 3,739 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e795c692c117f6792775922086690470e23c030d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29279152883)
