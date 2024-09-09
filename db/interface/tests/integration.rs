use std::sync::Arc;

use ax_sdk::config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config};
use axdb_interface::{
    common::{
        committed_page::CommittedPage, cryptographic_object::CryptographicObject,
        cryptographic_schema::CryptographicSchema,
    },
    controller::AxdbController,
    NUM_IDX_COLS, PCS_LOG_DEGREE,
};
use datafusion::{
    arrow::{
        array::UInt32Array,
        datatypes::{DataType, Field, Schema},
    },
    execution::context::SessionContext,
    logical_expr::{col, lit, table_scan},
};

/// Runs keygen for a given schema (Page in CommittedPage is an empty page)
pub async fn run_keygen() {
    let ctx = SessionContext::new();

    let schema = std::fs::read("tests/data/example.schema.bin").unwrap();
    let schema: Schema = bincode::deserialize(&schema).unwrap();

    let cs = CryptographicSchema::new(schema, NUM_IDX_COLS);

    let table_id = cs.id.clone();
    let co = CryptographicObject::<BabyBearPoseidon2Config>::CryptographicSchema(cs);
    ctx.register_table(table_id.clone(), Arc::new(co)).unwrap();

    let sql = format!("SELECT a FROM {} WHERE a <= 10", table_id);
    let logical = ctx.state().create_logical_plan(sql.as_str()).await.unwrap();

    let engine = default_engine(PCS_LOG_DEGREE);
    let mut axdb = AxdbController::new(ctx, logical, engine).await;
    println!(
        "Flattened Axdb execution plan: {:?}",
        axdb.axdb_execution_plan
    );

    // After running keygen once, you will not need to run it again for the same LogicalPlan
    axdb.keygen().await.unwrap();
}

/// Runs execute, prove, and verify on a CommittedPage (Page contains concrete data)
pub async fn run_execute_prove_verify() {
    let ctx = SessionContext::new();

    // use datafusion::execution::options::CsvReadOptions;
    // let page_id = "example";
    // ctx.register_csv(page_id, "tests/data/example.csv", CsvReadOptions::new())
    //     .await
    //     .unwrap();

    let cp = CommittedPage::<BabyBearPoseidon2Config>::new_from_paths(
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
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
    axdb.prove().await.unwrap();
    axdb.verify().await.unwrap();

    let output = axdb.output().await.unwrap();
    println!("Output RecordBatch: {:?}", output);
}

#[tokio::test]
pub async fn test_basic_e2e() {
    run_keygen().await;
    run_execute_prove_verify().await;
}

#[tokio::test]
#[ignore]
pub async fn test_keygen() {
    run_keygen().await;
}

#[tokio::test]
#[ignore]
pub async fn test_execute() {
    run_execute_prove_verify().await;
}

#[tokio::test]
pub async fn test_page_scan_with_filter() {
    let ctx = SessionContext::new();

    let cp = CommittedPage::<BabyBearPoseidon2Config>::new_from_paths(
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
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

    // After running keygen once, you will not need to run it again for the same LogicalPlan
    axdb.keygen().await.unwrap();

    axdb.execute().await.unwrap();
    axdb.prove().await.unwrap();
    axdb.verify().await.unwrap();

    let output = axdb.output().await.unwrap();
    println!("Output RecordBatch: {:?}", output);
}

#[tokio::test]
pub async fn test_validate_ingestion() {
    let cp_file = CommittedPage::<BabyBearPoseidon2Config>::new_from_paths(
        "tests/data/example.page.bin",
        "tests/data/example.schema.bin",
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
