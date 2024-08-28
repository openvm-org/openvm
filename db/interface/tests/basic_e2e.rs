use std::sync::Arc;

use afs_page::common::page::Page;
use afs_test_utils::config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config};
use axdb_interface::{
    committed_page, committed_page::CommittedPage, controller::AxdbController, NUM_IDX_COLS,
    PCS_LOG_DEGREE,
};
use datafusion::{
    arrow::{
        array::UInt32Array,
        datatypes::{DataType, Field, Schema},
    },
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
    let mut axdb = AxdbController::new(ctx, logical, engine).await;
    println!(
        "Flattened Axdb execution plan: {:?}",
        axdb.axdb_execution_plan
    );

    axdb.execute().await.unwrap();

    // After running keygen once, you will not need to run it again for the same LogicalPlan
    axdb.keygen().await.unwrap();

    axdb.prove().await.unwrap();
    axdb.verify().await.unwrap();

    let output = axdb.output().await.unwrap();
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
    let mut axdb = AxdbController::new(ctx, logical, engine).await;
    println!(
        "Flattened Axdb execution plan: {:?}",
        axdb.axdb_execution_plan
    );

    axdb.execute().await.unwrap();

    // After running keygen once, you will not need to run it again for the same LogicalPlan
    axdb.keygen().await.unwrap();

    axdb.prove().await.unwrap();
    axdb.verify().await.unwrap();

    let output = axdb.output().await.unwrap();
    println!("Output RecordBatch: {:?}", output);
}

#[tokio::test]
pub async fn test_validate_ingestion() {
    let cp_file = committed_page!(
        "example",
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
        BabyBearPoseidon2Config
    );

    let cp = CommittedPage::<BabyBearPoseidon2Config>::from_cols(
        vec![
            (
                Field::new("a", DataType::UInt32, false),
                Arc::new(UInt32Array::from(vec![2, 4, 8, 16])),
            ),
            (
                Field::new("b", DataType::UInt32, false),
                Arc::new(UInt32Array::from(vec![1, 2, 3, 4])),
            ),
            (
                Field::new("c", DataType::UInt32, false),
                Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
            ),
            (
                Field::new("d", DataType::UInt32, false),
                Arc::new(UInt32Array::from(vec![4, 8, 16, 32])),
            ),
        ],
        NUM_IDX_COLS,
    );

    assert_eq!(cp_file.schema, cp.schema);
    assert_eq!(cp_file.page, cp.page);
}
