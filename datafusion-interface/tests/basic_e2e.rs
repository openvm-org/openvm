use std::sync::Arc;

use afs_datafusion_interface::{
    afs_exec::AfsExec, afs_node::AfsNode, committed_page, PCS_LOG_DEGREE,
};
use afs_page::common::{page::Page, page_cols::PageCols};
use afs_test_utils::config::baby_bear_poseidon2::{
    default_engine, BabyBearPoseidon2Config, BabyBearPoseidon2Engine,
};
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    execution::{context::SessionContext, options::CsvReadOptions},
    logical_expr::{col, lit, table_scan, Expr, LogicalPlan},
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
};

// #[test]
// pub fn gen_schema() {
//     let fields = vec![
//         Field::new("a", DataType::UInt32, false),
//         Field::new("b", DataType::UInt32, false),
//         Field::new("c", DataType::UInt32, false),
//         Field::new("d", DataType::UInt32, false),
//     ];
//     let schema = Schema::new(fields);
//     let serialized_schema = bincode::serialize(&schema).unwrap();
//     std::fs::write("tests/data/example.schema.bin", serialized_schema).unwrap();

//     let schema_file = std::fs::read("tests/data/example.schema.bin").unwrap();
//     let schema_new = bincode::deserialize(&schema_file).unwrap();
//     assert_eq!(schema, schema_new);
// }

// #[test]
// pub fn gen_page() {
//     let page = Page::from_page_cols(vec![
//         PageCols::<u32> {
//             is_alloc: 1,
//             idx: vec![2],
//             data: vec![1, 0, 4],
//         },
//         PageCols::<u32> {
//             is_alloc: 1,
//             idx: vec![4],
//             data: vec![2, 0, 8],
//         },
//         PageCols::<u32> {
//             is_alloc: 1,
//             idx: vec![8],
//             data: vec![3, 0, 16],
//         },
//         PageCols::<u32> {
//             is_alloc: 1,
//             idx: vec![16],
//             data: vec![4, 0, 32],
//         },
//     ]);
//     let serialized = bincode::serialize(&page).unwrap();
//     std::fs::write("tests/data/example.page.bin", serialized).unwrap();
// }

#[tokio::test]
pub async fn test_basic_e2e() {
    let ctx = SessionContext::new();

    // ctx.register_csv("example", "tests/data/example.csv", CsvReadOptions::new())
    //     .await
    //     .unwrap();

    let cp = committed_page!(
        "example",
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
        BabyBearPoseidon2Config
    );
    let schema = cp.schema.clone();
    ctx.register_table("example", Arc::new(cp)).unwrap();

    // let sql = "SELECT a FROM example WHERE a <= b GROUP BY a";
    // let sql = "SELECT a FROM example WHERE a <= 10";
    // let sql = "SELECT a FROM example";
    // let logical = ctx.state().create_logical_plan(sql).await.unwrap();
    let logical = table_scan(Some("example"), &schema, None)
        .unwrap()
        // .filter(col("a").lt(lit(10)))
        // .unwrap()
        // .filter(col("b").lt(lit(20)))
        // .unwrap()
        .build()
        .unwrap();
    println!("{:#?}", logical.clone());

    let engine = default_engine(PCS_LOG_DEGREE);
    let mut afs = AfsExec::new(ctx, logical, engine).await;
    println!("Flattened AFS execution plan: {:?}", afs.afs_execution_plan);

    let output = afs.execute().await.unwrap();
    println!("Output page: {:?}", output.page);

    afs.keygen().await.unwrap();
    let end_node = afs.afs_execution_plan.last().unwrap();
    match end_node {
        AfsNode::PageScan(page_scan) => {
            let pk = page_scan.pk.as_ref().unwrap();
            println!(
                "Proving key interaction chunk size: {:?}",
                pk.interaction_chunk_size
            );
        }
        _ => unreachable!(),
    }
}
