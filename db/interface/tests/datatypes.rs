use std::sync::Arc;

use afs_page::common::page::Page;
use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use ax_sdk::config::baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config};
use axdb_interface::{
    common::committed_page::CommittedPage, controller::AxdbController, PCS_LOG_DEGREE,
};
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    execution::context::SessionContext,
};
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

fn create_page<SC: StarkGenericConfig>() -> CommittedPage<SC>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    let page = Page::from_2d_vec(
        &[
            vec![1, 1, 2, 0, 1, 0, 2, 4, 4, 16],
            vec![1, 1, 4, 0, 2, 0, 2, 4, 8, 64],
            vec![1, 1, 8, 0, 3, 0, 2, 4, 16, 256],
            vec![1, 1, 16, 0, 4, 0, 2, 4, 32, 1024],
        ],
        2,
        7,
    );
    let schema = Schema::new(vec![
        Field::new("idx", DataType::UInt32, false),
        Field::new("d0", DataType::UInt32, false),
        Field::new("d1", DataType::UInt64, false),
        Field::new("d2", DataType::UInt16, false),
    ]);
    CommittedPage::<SC>::new(schema, page)
}

#[tokio::test]
pub async fn test_wide_types() {
    let ctx = SessionContext::new();

    let cp = create_page::<BabyBearPoseidon2Config>();
    let page_id = cp.page_id.clone();
    ctx.register_table(page_id.clone(), Arc::new(cp)).unwrap();

    let sql = format!("SELECT idx FROM {} WHERE idx <= 259", page_id);
    let logical = ctx.state().create_logical_plan(sql.as_str()).await.unwrap();
    println!("{:#?}", logical.clone());

    let engine = default_engine(PCS_LOG_DEGREE);
    let mut afs = AxdbController::new(ctx, logical, engine).await;
    println!(
        "Flattened Axdb execution plan: {:?}",
        afs.axdb_execution_plan
    );

    afs.execute().await.unwrap();

    afs.keygen().await.unwrap();
    println!("Keygen completed");

    afs.prove().await.unwrap();
    println!("STARK proof generated");

    afs.verify().await.unwrap();
    println!("STARK proof verified");

    let output = afs.output().await.unwrap();
    println!("Output RecordBatch: {:?}", output);
}
