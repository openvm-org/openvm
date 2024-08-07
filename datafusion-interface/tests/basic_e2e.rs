use std::sync::Arc;

use afs_datafusion_interface::{afs_exec::AfsExec, committed_page::CommittedPage};
use afs_page::common::page::Page;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    execution::{context::SessionContext, options::CsvReadOptions},
    logical_expr::LogicalPlan,
};

#[tokio::test]
pub async fn test_basic_e2e() {
    let ctx = SessionContext::new();
    // ctx.register_table(table_ref, provider);

    // let df = ctx
    //     .read_csv("tests/data/example.csv", CsvReadOptions::new())
    //     .await
    //     .unwrap();
    // let results = df.collect().await.unwrap();
    // println!("{:?}", results);

    // ctx.register_csv("example", "tests/data/example.csv", CsvReadOptions::new())
    //     .await
    //     .unwrap();

    let schema_path = std::fs::read("tests/data/example.schema.bin").unwrap();
    let schema: Schema = bincode::deserialize(&schema_path).unwrap();
    let page_path = std::fs::read("tests/data/example.page.bin").unwrap();
    let page: Page = bincode::deserialize(&page_path).unwrap();
    let cp =
        CommittedPage::<BabyBearPoseidon2Config>::new("example".to_string(), schema, page, None);
    ctx.register_table("example", Arc::new(cp)).unwrap();

    // let sql = "SELECT a FROM example WHERE a <= b GROUP BY a";
    let sql = "SELECT a FROM example";
    let logical = ctx.state().create_logical_plan(sql).await.unwrap();
    // let execution = ctx.state().create_physical_plan(&logical).await.unwrap();
    println!("{:?}", logical);
    let afs = AfsExec::new(ctx, logical);
    // let output = afs.execute().unwrap();
    // println!("{:?}", output);
}

#[test]
pub fn gen_schema() {
    let fields = vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
        Field::new("c", DataType::UInt32, false),
        Field::new("d", DataType::UInt32, false),
    ];
    let schema = Schema::new(fields);
    let serialized_schema = bincode::serialize(&schema).unwrap();
    std::fs::write("tests/data/example.schema.bin", serialized_schema).unwrap();

    let schema_file = std::fs::read("tests/data/example.schema.bin").unwrap();
    let schema_new = bincode::deserialize(&schema_file).unwrap();
    assert_eq!(schema, schema_new);
}
