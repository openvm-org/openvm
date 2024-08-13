use std::sync::Arc;

use afs_datafusion_interface::{afs_exec::AfsExec, committed_page, committed_page::CommittedPage};
use afs_page::common::{page::Page, page_cols::PageCols};
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    datasource::DefaultTableSource,
    execution::{context::SessionContext, options::CsvReadOptions},
    logical_expr::{col, lit, table_scan, Expr, LogicalPlan},
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
};

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

#[test]
pub fn gen_page() {
    let page = Page::from_page_cols(vec![
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![],
            data: vec![0, 1, 0, 2],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![],
            data: vec![0, 2, 0, 4],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![],
            data: vec![0, 3, 0, 8],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![],
            data: vec![0, 4, 0, 16],
        },
    ]);
    let serialized = bincode::serialize(&page).unwrap();
    std::fs::write("tests/data/example.page.bin", serialized).unwrap();
}

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

    // let table_source = DefaultTableSource::new(Arc::new(cp));

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

    // let t = ctx.table("example").await.unwrap();
    // t.show().await.unwrap();
    // let record_batches = t.collect().await.unwrap();
    // for batch in record_batches {
    //     println!("{:?}", batch);
    // }
    // println!("table example {:#?}", t.show());

    // let default_planner = DefaultPhysicalPlanner::default();
    // let physical = default_planner
    //     .create_physical_plan(&logical, &ctx.state())
    //     .await
    //     .unwrap();
    // let execution = ctx.state().create_physical_plan(&logical).await.unwrap();
    // println!("{:#?}", physical);
    // let res = physical.execute(0, ctx.task_ctx()).unwrap();
    // while let Some(batch) = res.as_ref().next().await {
    //     println!("{:?}", batch);
    // }

    let mut afs = AfsExec::<BabyBearPoseidon2Config>::new(ctx, logical).await;
    println!("{:?}", afs.afs_execution_plan);

    let output = afs.execute().await.unwrap();
    println!("{:?}", output.page);
}
