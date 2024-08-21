use std::sync::Arc;

use afs_page::common::page::Page;
use afs_test_utils::config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config};
use axdb_interface::{committed_page, exec::AxdbExec, PCS_LOG_DEGREE};
use datafusion::{
    arrow::datatypes::Schema,
    execution::context::SessionContext,
    logical_expr::{col, lit, table_scan},
};

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
    let sql = format!("SELECT a FROM {} WHERE a <= 10", page_id);
    // let sql = format!("SELECT a FROM {}", page_id);
    let logical = ctx.state().create_logical_plan(sql.as_str()).await.unwrap();

    let engine = default_engine(PCS_LOG_DEGREE);
    let mut afs = AxdbExec::new(ctx, logical, engine).await;
    println!(
        "Flattened Axdb execution plan: {:?}",
        afs.axdb_execution_plan
    );

    afs.execute().await.unwrap();

    // After running keygen once, you will not need to run it again for the same LogicalPlan
    afs.keygen().await.unwrap();

    afs.prove().await.unwrap();
    afs.verify().await.unwrap();

    let output = afs.output().await.unwrap();
    println!("Output RecordBatch: {:?}", output);
}

#[tokio::test]
pub async fn test_page_scan_with_filter() {
    let ctx = SessionContext::new();

    let cp = committed_page!(
        "example",
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
        BabyBearPoseidon2Config
    );
    let page_id = cp.page_id.clone();
    ctx.register_table(page_id.clone(), Arc::new(cp.clone()))
        .unwrap();

    let schema = cp.schema.clone();

    // Builds a LogicalPlan with two filters inside a TableScan node
    let logical = table_scan(Some(page_id), &schema, None)
        .unwrap()
        .filter(col("a").lt(lit(10)))
        .unwrap()
        .filter(col("a").gt(lit(3)))
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

    // After running keygen once, you will not need to run it again for the same LogicalPlan
    afs.keygen().await.unwrap();

    afs.prove().await.unwrap();
    afs.verify().await.unwrap();

    let output = afs.output().await.unwrap();
    println!("Output RecordBatch: {:?}", output);
}
