use std::sync::Arc;

use afs_page::common::page::Page;
use afs_test_utils::config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config};
use axdb_interface::{committed_page, exec::AxdbExec, PCS_LOG_DEGREE};
use datafusion::{arrow::datatypes::Schema, execution::context::SessionContext};

// #[test]
// pub fn gen_schema() {
//     use datafusion::arrow::datatypes::{DataType, Field};
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
//     use afs_page::common::page_cols::PageCols;
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

    // use datafusion::execution::options::CsvReadOptions;
    // let page_id = "example";
    // ctx.register_csv(page_id, "tests/data/example.csv", CsvReadOptions::new())
    //     .await
    //     .unwrap();

    let cp = committed_page!(
        "example",
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
        BabyBearPoseidon2Config
    );
    let page_id = cp.page_id.clone();
    ctx.register_table(page_id.clone(), Arc::new(cp.clone()))
        .unwrap();

    // let sql = format!("SELECT a FROM {} WHERE a <= b GROUP BY a", page_id);
    // let sql = format!("SELECT a FROM {} WHERE a <= 10", page_id);
    // let sql = format!("SELECT a FROM {}", page_id);
    // let logical = ctx.state().create_logical_plan(sql.as_str()).await.unwrap();

    use datafusion::logical_expr::{col, lit, table_scan};
    let schema = cp.schema.clone();
    let logical = table_scan(Some(page_id), &schema, None)
        .unwrap()
        .filter(col("a").lt(lit(10)))
        .unwrap()
        .filter(col("a").lt(lit(11)))
        .unwrap()
        .build()
        .unwrap();
    println!("{:#?}", logical.clone());

    let engine = default_engine(PCS_LOG_DEGREE);
    let mut afs = AxdbExec::new(ctx, logical, engine).await;
    println!(
        "Flattened Axdb execution plan: {:?}",
        afs.axdb_execution_plan
    );

    afs.execute().await.unwrap();
    let last_node = afs.last_node().await.unwrap();
    let output = last_node.lock().await.output().clone().unwrap();
    println!("Output page: {:?}", output.page);

    afs.keygen().await.unwrap();
    afs.prove().await.unwrap();
    afs.verify().await.unwrap();
}
