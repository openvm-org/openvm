#[cfg(feature = "evm-prove")]
use std::path::Path;
use std::{
    marker::PhantomData,
    sync::{Arc, OnceLock},
};

use eyre::eyre;
use openvm_circuit::arch::{VmBuilder, VmExecutionConfig, VmExecutor};
use openvm_continuations::RootSC;
use openvm_sdk_config::TranspilerConfig;
use openvm_stark_backend::{keygen::types::MultiStarkProvingKey, StarkEngine, SystemParams};
use openvm_stark_sdk::config::root_params_with_100_bits_security;
#[cfg(feature = "evm-prove")]
use openvm_static_verifier::StaticVerifierShape;
use openvm_transpiler::transpiler::Transpiler;

#[cfg(feature = "evm-prove")]
use crate::halo2_params::CacheHalo2ParamsReader;
#[cfg(feature = "evm-prove")]
use crate::{config::Halo2Config, keygen::Halo2ProvingKey, prover::Halo2Prover};
use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig, AppConfig},
    keygen::{AggProvingKey, AppProvingKey},
    prover::{AggProver, DeferralPathProver, DeferralProver, RootProver},
    GenericSdk, SdkError, F, SC,
};

enum AppSource<VC> {
    Config(AppConfig<VC>),
    Pk(AppProvingKey<VC>),
}

#[allow(clippy::large_enum_variant)]
enum AggSource {
    Params(AggregationSystemParams),
    Pk(AggProvingKey),
}

struct PrebuiltRootPk {
    pk: Arc<MultiStarkProvingKey<RootSC>>,
    trace_heights: Vec<usize>,
}

enum RootSource {
    Params(SystemParams),
    Pk(PrebuiltRootPk),
}

#[cfg(feature = "evm-prove")]
enum Halo2Source {
    Config {
        shape: StaticVerifierShape,
        config: Halo2Config,
    },
    Pk(Halo2ProvingKey),
}

/// Construction-only API for [`GenericSdk`].
///
/// Each proving layer has one source of truth: either user-supplied config/params or a
/// pre-generated proving key. `build()` normalizes those sources into an immutable [`GenericSdk`].
pub struct GenericSdkBuilder<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    app_source: Option<AppSource<VB::VmConfig>>,
    agg_source: Option<AggSource>,
    root_source: Option<RootSource>,
    agg_tree_config: Option<AggregationTreeConfig>,
    transpiler: Option<Transpiler<F>>,
    deferral_prover: Option<DeferralProver>,
    #[cfg(feature = "evm-prove")]
    halo2_source: Option<Halo2Source>,
    #[cfg(feature = "evm-prove")]
    halo2_params_reader: Option<CacheHalo2ParamsReader>,
    _phantom: PhantomData<E>,
}

impl<E, VB> GenericSdkBuilder<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    pub fn new() -> Self {
        Self::default()
    }

    fn set_once<T>(slot: &mut Option<T>, field_name: &str, value: T) {
        assert!(slot.is_none(), "{field_name} already set");
        *slot = Some(value);
    }

    fn init_once_lock<T>(value: Option<T>, field_name: &str) -> OnceLock<T> {
        let lock = OnceLock::new();
        if let Some(value) = value {
            assert!(
                lock.set(value).is_ok(),
                "{field_name} should only be initialized once"
            );
        }
        lock
    }

    fn agg_config_from_pk(agg_pk: &AggProvingKey) -> AggregationConfig {
        AggregationConfig {
            params: AggregationSystemParams {
                leaf: agg_pk.leaf_pk.params.clone(),
                internal: agg_pk.internal_recursive_pk.params.clone(),
            },
        }
    }

    #[cfg(feature = "evm-prove")]
    fn halo2_shape_from_pk(halo2_pk: &Halo2ProvingKey) -> StaticVerifierShape {
        halo2_pk.verifier.shape
    }

    #[cfg(feature = "evm-prove")]
    fn halo2_config_from_pk(halo2_pk: &Halo2ProvingKey) -> Halo2Config {
        Halo2Config {
            wrapper_k: Some(halo2_pk.wrapper.pinning.metadata.config_params.k),
            profiling: halo2_pk.profiling,
        }
    }

    fn build_deferral_path_prover(
        agg_config: &AggregationConfig,
        deferral_prover: DeferralProver,
    ) -> Arc<DeferralPathProver> {
        let deferral_tree_config = AggregationTreeConfig {
            num_children_leaf: 2,
            num_children_internal: 2,
        };
        let agg_prover = AggProver::new(
            deferral_prover.def_hook_prover.get_vk(),
            agg_config.clone(),
            deferral_tree_config,
            Some(deferral_prover.def_hook_prover.get_cached_commit()),
        );
        Arc::new(DeferralPathProver {
            deferral_prover: Arc::new(deferral_prover),
            agg_prover: Arc::new(agg_prover),
        })
    }

    fn require_dependency(
        has_value: bool,
        value_name: &str,
        has_dependency: bool,
        dependency_name: &str,
    ) -> Result<(), SdkError> {
        if has_value && !has_dependency {
            return Err(SdkError::Other(eyre!(
                "`{value_name}` requires `{dependency_name}` to also be set"
            )));
        }
        Ok(())
    }

    fn normalize_app_source(
        app_source: AppSource<VB::VmConfig>,
    ) -> (AppConfig<VB::VmConfig>, Option<AppProvingKey<VB::VmConfig>>) {
        match app_source {
            AppSource::Config(app_config) => (app_config, None),
            AppSource::Pk(app_pk) => {
                let app_config = app_pk.app_config();
                (app_config, Some(app_pk))
            }
        }
    }

    fn normalize_agg_source(agg_source: AggSource) -> (AggregationConfig, Option<AggProvingKey>) {
        match agg_source {
            AggSource::Params(agg_params) => (AggregationConfig { params: agg_params }, None),
            AggSource::Pk(agg_pk) => {
                let agg_config = Self::agg_config_from_pk(&agg_pk);
                (agg_config, Some(agg_pk))
            }
        }
    }

    fn normalize_root_source(root_source: RootSource) -> (SystemParams, Option<PrebuiltRootPk>) {
        match root_source {
            RootSource::Params(root_params) => (root_params, None),
            RootSource::Pk(root_pk) => (root_pk.pk.params.clone(), Some(root_pk)),
        }
    }

    #[cfg(feature = "evm-prove")]
    fn normalize_halo2_source(
        halo2_source: Halo2Source,
    ) -> (StaticVerifierShape, Halo2Config, Option<Halo2ProvingKey>) {
        match halo2_source {
            Halo2Source::Config { shape, config } => (shape, config, None),
            Halo2Source::Pk(halo2_pk) => (
                Self::halo2_shape_from_pk(&halo2_pk),
                Self::halo2_config_from_pk(&halo2_pk),
                Some(halo2_pk),
            ),
        }
    }

    pub fn app_config(mut self, app_config: AppConfig<VB::VmConfig>) -> Self {
        Self::set_once(
            &mut self.app_source,
            "app_source",
            AppSource::Config(app_config),
        );
        self
    }

    pub fn app_pk(mut self, app_pk: AppProvingKey<VB::VmConfig>) -> Self {
        Self::set_once(&mut self.app_source, "app_source", AppSource::Pk(app_pk));
        self
    }

    pub fn agg_params(mut self, agg_params: AggregationSystemParams) -> Self {
        Self::set_once(
            &mut self.agg_source,
            "agg_source",
            AggSource::Params(agg_params),
        );
        self
    }

    pub fn agg_pk(mut self, agg_pk: AggProvingKey) -> Self {
        Self::set_once(&mut self.agg_source, "agg_source", AggSource::Pk(agg_pk));
        self
    }

    pub fn root_params(mut self, root_params: SystemParams) -> Self {
        Self::set_once(
            &mut self.root_source,
            "root_source",
            RootSource::Params(root_params),
        );
        self
    }

    pub fn root_pk(
        mut self,
        root_pk: Arc<MultiStarkProvingKey<RootSC>>,
        trace_heights: Vec<usize>,
    ) -> Self {
        Self::set_once(
            &mut self.root_source,
            "root_source",
            RootSource::Pk(PrebuiltRootPk {
                pk: root_pk,
                trace_heights,
            }),
        );
        self
    }

    pub fn agg_tree_config(mut self, agg_tree_config: AggregationTreeConfig) -> Self {
        Self::set_once(
            &mut self.agg_tree_config,
            "agg_tree_config",
            agg_tree_config,
        );
        self
    }

    pub fn transpiler(mut self, transpiler: Transpiler<F>) -> Self {
        Self::set_once(&mut self.transpiler, "transpiler", transpiler);
        self
    }

    pub fn default_transpiler(mut self) -> Self
    where
        VB::VmConfig: TranspilerConfig<F>,
    {
        let transpiler = match self.app_source.as_ref() {
            Some(AppSource::Config(app_config)) => app_config.app_vm_config.transpiler(),
            Some(AppSource::Pk(app_pk)) => app_pk.app_vm_pk.vm_config.transpiler(),
            None => panic!("default_transpiler requires app_source to be set first"),
        };
        Self::set_once(&mut self.transpiler, "transpiler", transpiler);
        self
    }

    pub fn deferral_prover(mut self, deferral_prover: DeferralProver) -> Self {
        Self::set_once(
            &mut self.deferral_prover,
            "deferral_prover",
            deferral_prover,
        );
        self
    }

    #[cfg(feature = "evm-prove")]
    pub fn halo2_config(mut self, shape: StaticVerifierShape, config: Halo2Config) -> Self {
        Self::set_once(
            &mut self.halo2_source,
            "halo2_source",
            Halo2Source::Config { shape, config },
        );
        self
    }

    #[cfg(feature = "evm-prove")]
    pub fn halo2_pk(mut self, halo2_pk: Halo2ProvingKey) -> Self {
        Self::set_once(
            &mut self.halo2_source,
            "halo2_source",
            Halo2Source::Pk(halo2_pk),
        );
        self
    }

    #[cfg(feature = "evm-prove")]
    pub fn halo2_params_dir(mut self, params_dir: impl AsRef<Path>) -> Self {
        Self::set_once(
            &mut self.halo2_params_reader,
            "halo2_params_reader",
            CacheHalo2ParamsReader::new(params_dir),
        );
        self
    }

    pub fn build(self) -> Result<GenericSdk<E, VB>, SdkError>
    where
        VB: Default,
    {
        let app_source = self
            .app_source
            .ok_or_else(|| SdkError::Other(eyre!("`app_config` or `app_pk` must be set")))?;
        let agg_source = self
            .agg_source
            .ok_or_else(|| SdkError::Other(eyre!("`agg_params` or `agg_pk` must be set")))?;
        let root_source = self
            .root_source
            .unwrap_or_else(|| RootSource::Params(root_params_with_100_bits_security()));
        #[cfg(feature = "evm-prove")]
        let halo2_source = self.halo2_source.unwrap_or(Halo2Source::Config {
            shape: StaticVerifierShape::default(),
            config: Halo2Config {
                wrapper_k: None,
                profiling: false,
            },
        });

        #[cfg(feature = "evm-prove")]
        Self::require_dependency(
            matches!(halo2_source, Halo2Source::Pk(_)),
            "halo2_pk",
            matches!(root_source, RootSource::Pk(_)),
            "root_pk",
        )?;
        Self::require_dependency(
            matches!(root_source, RootSource::Pk(_)),
            "root_pk",
            matches!(agg_source, AggSource::Pk(_)),
            "agg_pk",
        )?;
        Self::require_dependency(
            matches!(agg_source, AggSource::Pk(_)),
            "agg_pk",
            matches!(app_source, AppSource::Pk(_)),
            "app_pk",
        )?;

        let Self {
            app_source: _,
            agg_source: _,
            root_source: _,
            agg_tree_config,
            transpiler,
            deferral_prover,
            #[cfg(feature = "evm-prove")]
                halo2_source: _,
            #[cfg(feature = "evm-prove")]
            halo2_params_reader,
            _phantom: _,
        } = self;

        let (app_config, app_pk_seed) = Self::normalize_app_source(app_source);
        let (agg_config, agg_pk_seed) = Self::normalize_agg_source(agg_source);
        let (root_params, root_pk_seed) = Self::normalize_root_source(root_source);

        let executor = VmExecutor::new(app_config.app_vm_config.clone())
            .map_err(|e| SdkError::Vm(e.into()))?;
        let agg_tree_config = agg_tree_config.unwrap_or_default();

        let def_path_prover = deferral_prover
            .map(|deferral_prover| Self::build_deferral_path_prover(&agg_config, deferral_prover));
        let def_hook_cached_commit = def_path_prover
            .as_ref()
            .map(|def_path_prover| def_path_prover.def_hook_cached_commit());
        let def_hook_vk_commit = def_path_prover
            .as_ref()
            .map(|def_path_prover| def_path_prover.def_hook_vk_commit());

        let app_vm_vk = app_pk_seed
            .as_ref()
            .map(|app_pk| Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()));
        let app_pk = Self::init_once_lock(app_pk_seed, "app_pk");

        let agg_prover_seed = agg_pk_seed.map(|agg_pk| {
            let app_vm_vk = app_vm_vk.expect("validated `agg_pk` dependency on `app_pk`");
            Arc::new(AggProver::from_pk(
                app_vm_vk,
                agg_pk,
                agg_tree_config,
                def_hook_cached_commit,
            ))
        });

        let root_prover_seed = root_pk_seed.map(|root_pk| {
            let agg_prover = agg_prover_seed
                .as_ref()
                .expect("validated `root_pk` dependency on `agg_pk`");
            let system_config = app_config.app_vm_config.as_ref();
            let memory_dimensions = system_config.memory_config.memory_dimensions();
            let num_user_pvs = system_config.num_public_values;
            let internal_recursive_dag_commit = agg_prover
                .internal_recursive_prover
                .get_self_vk_pcs_data()
                .unwrap()
                .commitment
                .into();

            Arc::new(RootProver::from_pk(
                agg_prover.internal_recursive_prover.get_vk(),
                internal_recursive_dag_commit,
                root_pk.pk,
                memory_dimensions,
                num_user_pvs,
                def_hook_vk_commit,
                Some(root_pk.trace_heights),
            ))
        });

        #[cfg(feature = "evm-prove")]
        let halo2_params_reader =
            halo2_params_reader.unwrap_or_else(CacheHalo2ParamsReader::new_with_default_params_dir);
        #[cfg(feature = "evm-prove")]
        let (halo2_shape, halo2_config, halo2_pk_seed) = Self::normalize_halo2_source(halo2_source);
        #[cfg(feature = "evm-prove")]
        let halo2_prover_seed =
            halo2_pk_seed.map(|halo2_pk| Halo2Prover::new(&halo2_params_reader, halo2_pk));

        Ok(GenericSdk {
            app_config,
            agg_config,
            agg_tree_config,
            root_params,
            #[cfg(feature = "evm-prove")]
            halo2_shape,
            #[cfg(feature = "evm-prove")]
            halo2_config,
            app_vm_builder: Default::default(),
            transpiler,
            executor,
            app_pk,
            agg_prover: Self::init_once_lock(agg_prover_seed, "agg_prover"),
            root_prover: Self::init_once_lock(root_prover_seed, "root_prover"),
            def_path_prover,
            #[cfg(feature = "evm-prove")]
            halo2_params_reader,
            #[cfg(feature = "evm-prove")]
            halo2_prover: Self::init_once_lock(halo2_prover_seed, "halo2_prover"),
            _phantom: PhantomData,
        })
    }
}

impl<E, VB> Default for GenericSdkBuilder<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    fn default() -> Self {
        Self {
            app_source: None,
            agg_source: None,
            root_source: None,
            agg_tree_config: None,
            transpiler: None,
            deferral_prover: None,
            #[cfg(feature = "evm-prove")]
            halo2_source: None,
            #[cfg(feature = "evm-prove")]
            halo2_params_reader: None,
            _phantom: PhantomData,
        }
    }
}

impl<E, VB> GenericSdk<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F>,
{
    pub fn builder() -> GenericSdkBuilder<E, VB> {
        GenericSdkBuilder::new()
    }
}
