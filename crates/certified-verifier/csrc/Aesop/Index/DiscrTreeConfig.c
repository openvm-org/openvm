// Lean compiler output
// Module: Aesop.Index.DiscrTreeConfig
// Imports: public import Init public meta import Init public import Lean.Meta.Basic public import Lean.Meta.DiscrTree.Types import Lean.Meta.DiscrTree.Main
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* l_Lean_Meta_Config_toConfigWithKey(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mkPath(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_indexConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 2, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_indexConfig___closed__0 = (const lean_object*)&lp_aesop_Aesop_indexConfig___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_indexConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_indexConfig___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_indexConfig;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_indexConfig___closed__1(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = ((lean_object*)(lp_aesop_Aesop_indexConfig___closed__0));
v___x_8_ = l_Lean_Meta_Config_toConfigWithKey(v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_indexConfig(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_indexConfig___closed__1, &lp_aesop_Aesop_indexConfig___closed__1_once, _init_lp_aesop_Aesop_indexConfig___closed__1);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object* v_e_10_, lean_object* v_a_11_, lean_object* v_a_12_, lean_object* v_a_13_, lean_object* v_a_14_){
_start:
{
lean_object* v___x_16_; lean_object* v_config_17_; uint8_t v_trackZetaDelta_18_; lean_object* v_zetaDeltaSet_19_; lean_object* v_lctx_20_; lean_object* v_localInstances_21_; lean_object* v_defEqCtx_x3f_22_; lean_object* v_synthPendingDepth_23_; lean_object* v_customCanUnfoldPredicate_x3f_24_; uint8_t v_univApprox_25_; uint8_t v_inTypeClassResolution_26_; uint8_t v_cacheInferType_27_; uint64_t v___x_28_; uint8_t v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_16_ = lp_aesop_Aesop_indexConfig;
v_config_17_ = lean_ctor_get(v___x_16_, 0);
v_trackZetaDelta_18_ = lean_ctor_get_uint8(v_a_11_, sizeof(void*)*7);
v_zetaDeltaSet_19_ = lean_ctor_get(v_a_11_, 1);
v_lctx_20_ = lean_ctor_get(v_a_11_, 2);
v_localInstances_21_ = lean_ctor_get(v_a_11_, 3);
v_defEqCtx_x3f_22_ = lean_ctor_get(v_a_11_, 4);
v_synthPendingDepth_23_ = lean_ctor_get(v_a_11_, 5);
v_customCanUnfoldPredicate_x3f_24_ = lean_ctor_get(v_a_11_, 6);
v_univApprox_25_ = lean_ctor_get_uint8(v_a_11_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_26_ = lean_ctor_get_uint8(v_a_11_, sizeof(void*)*7 + 2);
v_cacheInferType_27_ = lean_ctor_get_uint8(v_a_11_, sizeof(void*)*7 + 3);
v___x_28_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_config_17_);
v___x_29_ = 0;
lean_inc_ref(v_config_17_);
v___x_30_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_30_, 0, v_config_17_);
lean_ctor_set_uint64(v___x_30_, sizeof(void*)*1, v___x_28_);
lean_inc(v_customCanUnfoldPredicate_x3f_24_);
lean_inc(v_synthPendingDepth_23_);
lean_inc(v_defEqCtx_x3f_22_);
lean_inc_ref(v_localInstances_21_);
lean_inc_ref(v_lctx_20_);
lean_inc(v_zetaDeltaSet_19_);
v___x_31_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_31_, 0, v___x_30_);
lean_ctor_set(v___x_31_, 1, v_zetaDeltaSet_19_);
lean_ctor_set(v___x_31_, 2, v_lctx_20_);
lean_ctor_set(v___x_31_, 3, v_localInstances_21_);
lean_ctor_set(v___x_31_, 4, v_defEqCtx_x3f_22_);
lean_ctor_set(v___x_31_, 5, v_synthPendingDepth_23_);
lean_ctor_set(v___x_31_, 6, v_customCanUnfoldPredicate_x3f_24_);
lean_ctor_set_uint8(v___x_31_, sizeof(void*)*7, v_trackZetaDelta_18_);
lean_ctor_set_uint8(v___x_31_, sizeof(void*)*7 + 1, v_univApprox_25_);
lean_ctor_set_uint8(v___x_31_, sizeof(void*)*7 + 2, v_inTypeClassResolution_26_);
lean_ctor_set_uint8(v___x_31_, sizeof(void*)*7 + 3, v_cacheInferType_27_);
v___x_32_ = l_Lean_Meta_DiscrTree_mkPath(v_e_10_, v___x_29_, v___x_31_, v_a_12_, v_a_13_, v_a_14_);
lean_dec_ref_known(v___x_31_, 7);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkDiscrTreePath___boxed(lean_object* v_e_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_, lean_object* v_a_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_aesop_Aesop_mkDiscrTreePath(v_e_33_, v_a_34_, v_a_35_, v_a_36_, v_a_37_);
lean_dec(v_a_37_);
lean_dec_ref(v_a_36_);
lean_dec(v_a_35_);
lean_dec_ref(v_a_34_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg(lean_object* v_t_40_, lean_object* v_e_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_47_; lean_object* v_config_48_; uint8_t v_trackZetaDelta_49_; lean_object* v_zetaDeltaSet_50_; lean_object* v_lctx_51_; lean_object* v_localInstances_52_; lean_object* v_defEqCtx_x3f_53_; lean_object* v_synthPendingDepth_54_; lean_object* v_customCanUnfoldPredicate_x3f_55_; uint8_t v_univApprox_56_; uint8_t v_inTypeClassResolution_57_; uint8_t v_cacheInferType_58_; uint64_t v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_47_ = lp_aesop_Aesop_indexConfig;
v_config_48_ = lean_ctor_get(v___x_47_, 0);
v_trackZetaDelta_49_ = lean_ctor_get_uint8(v_a_42_, sizeof(void*)*7);
v_zetaDeltaSet_50_ = lean_ctor_get(v_a_42_, 1);
v_lctx_51_ = lean_ctor_get(v_a_42_, 2);
v_localInstances_52_ = lean_ctor_get(v_a_42_, 3);
v_defEqCtx_x3f_53_ = lean_ctor_get(v_a_42_, 4);
v_synthPendingDepth_54_ = lean_ctor_get(v_a_42_, 5);
v_customCanUnfoldPredicate_x3f_55_ = lean_ctor_get(v_a_42_, 6);
v_univApprox_56_ = lean_ctor_get_uint8(v_a_42_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_57_ = lean_ctor_get_uint8(v_a_42_, sizeof(void*)*7 + 2);
v_cacheInferType_58_ = lean_ctor_get_uint8(v_a_42_, sizeof(void*)*7 + 3);
v___x_59_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_config_48_);
lean_inc_ref(v_config_48_);
v___x_60_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_60_, 0, v_config_48_);
lean_ctor_set_uint64(v___x_60_, sizeof(void*)*1, v___x_59_);
lean_inc(v_customCanUnfoldPredicate_x3f_55_);
lean_inc(v_synthPendingDepth_54_);
lean_inc(v_defEqCtx_x3f_53_);
lean_inc_ref(v_localInstances_52_);
lean_inc_ref(v_lctx_51_);
lean_inc(v_zetaDeltaSet_50_);
v___x_61_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v_zetaDeltaSet_50_);
lean_ctor_set(v___x_61_, 2, v_lctx_51_);
lean_ctor_set(v___x_61_, 3, v_localInstances_52_);
lean_ctor_set(v___x_61_, 4, v_defEqCtx_x3f_53_);
lean_ctor_set(v___x_61_, 5, v_synthPendingDepth_54_);
lean_ctor_set(v___x_61_, 6, v_customCanUnfoldPredicate_x3f_55_);
lean_ctor_set_uint8(v___x_61_, sizeof(void*)*7, v_trackZetaDelta_49_);
lean_ctor_set_uint8(v___x_61_, sizeof(void*)*7 + 1, v_univApprox_56_);
lean_ctor_set_uint8(v___x_61_, sizeof(void*)*7 + 2, v_inTypeClassResolution_57_);
lean_ctor_set_uint8(v___x_61_, sizeof(void*)*7 + 3, v_cacheInferType_58_);
v___x_62_ = l_Lean_Meta_DiscrTree_getUnify___redArg(v_t_40_, v_e_41_, v___x_61_, v_a_43_, v_a_44_, v_a_45_);
lean_dec_ref_known(v___x_61_, 7);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___redArg___boxed(lean_object* v_t_63_, lean_object* v_e_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_aesop_Aesop_getUnify___redArg(v_t_63_, v_e_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_);
lean_dec(v_a_68_);
lean_dec_ref(v_a_67_);
lean_dec(v_a_66_);
lean_dec_ref(v_a_65_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify(lean_object* v_00_u03b1_71_, lean_object* v_t_72_, lean_object* v_e_73_, lean_object* v_a_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_aesop_Aesop_getUnify___redArg(v_t_72_, v_e_73_, v_a_74_, v_a_75_, v_a_76_, v_a_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getUnify___boxed(lean_object* v_00_u03b1_80_, lean_object* v_t_81_, lean_object* v_e_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_, lean_object* v_a_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_aesop_Aesop_getUnify(v_00_u03b1_80_, v_t_81_, v_e_82_, v_a_83_, v_a_84_, v_a_85_, v_a_86_);
lean_dec(v_a_86_);
lean_dec_ref(v_a_85_);
lean_dec(v_a_84_);
lean_dec_ref(v_a_83_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg(lean_object* v_t_89_, lean_object* v_e_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_){
_start:
{
lean_object* v___x_96_; lean_object* v_config_97_; uint8_t v_trackZetaDelta_98_; lean_object* v_zetaDeltaSet_99_; lean_object* v_lctx_100_; lean_object* v_localInstances_101_; lean_object* v_defEqCtx_x3f_102_; lean_object* v_synthPendingDepth_103_; lean_object* v_customCanUnfoldPredicate_x3f_104_; uint8_t v_univApprox_105_; uint8_t v_inTypeClassResolution_106_; uint8_t v_cacheInferType_107_; uint64_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_96_ = lp_aesop_Aesop_indexConfig;
v_config_97_ = lean_ctor_get(v___x_96_, 0);
v_trackZetaDelta_98_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7);
v_zetaDeltaSet_99_ = lean_ctor_get(v_a_91_, 1);
v_lctx_100_ = lean_ctor_get(v_a_91_, 2);
v_localInstances_101_ = lean_ctor_get(v_a_91_, 3);
v_defEqCtx_x3f_102_ = lean_ctor_get(v_a_91_, 4);
v_synthPendingDepth_103_ = lean_ctor_get(v_a_91_, 5);
v_customCanUnfoldPredicate_x3f_104_ = lean_ctor_get(v_a_91_, 6);
v_univApprox_105_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_106_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7 + 2);
v_cacheInferType_107_ = lean_ctor_get_uint8(v_a_91_, sizeof(void*)*7 + 3);
v___x_108_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_config_97_);
lean_inc_ref(v_config_97_);
v___x_109_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_109_, 0, v_config_97_);
lean_ctor_set_uint64(v___x_109_, sizeof(void*)*1, v___x_108_);
lean_inc(v_customCanUnfoldPredicate_x3f_104_);
lean_inc(v_synthPendingDepth_103_);
lean_inc(v_defEqCtx_x3f_102_);
lean_inc_ref(v_localInstances_101_);
lean_inc_ref(v_lctx_100_);
lean_inc(v_zetaDeltaSet_99_);
v___x_110_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_zetaDeltaSet_99_);
lean_ctor_set(v___x_110_, 2, v_lctx_100_);
lean_ctor_set(v___x_110_, 3, v_localInstances_101_);
lean_ctor_set(v___x_110_, 4, v_defEqCtx_x3f_102_);
lean_ctor_set(v___x_110_, 5, v_synthPendingDepth_103_);
lean_ctor_set(v___x_110_, 6, v_customCanUnfoldPredicate_x3f_104_);
lean_ctor_set_uint8(v___x_110_, sizeof(void*)*7, v_trackZetaDelta_98_);
lean_ctor_set_uint8(v___x_110_, sizeof(void*)*7 + 1, v_univApprox_105_);
lean_ctor_set_uint8(v___x_110_, sizeof(void*)*7 + 2, v_inTypeClassResolution_106_);
lean_ctor_set_uint8(v___x_110_, sizeof(void*)*7 + 3, v_cacheInferType_107_);
v___x_111_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v_t_89_, v_e_90_, v___x_110_, v_a_92_, v_a_93_, v_a_94_);
lean_dec_ref_known(v___x_110_, 7);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___redArg___boxed(lean_object* v_t_112_, lean_object* v_e_113_, lean_object* v_a_114_, lean_object* v_a_115_, lean_object* v_a_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_aesop_Aesop_getMatch___redArg(v_t_112_, v_e_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_);
lean_dec(v_a_117_);
lean_dec_ref(v_a_116_);
lean_dec(v_a_115_);
lean_dec_ref(v_a_114_);
lean_dec_ref(v_t_112_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch(lean_object* v_00_u03b1_120_, lean_object* v_t_121_, lean_object* v_e_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_aesop_Aesop_getMatch___redArg(v_t_121_, v_e_122_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getMatch___boxed(lean_object* v_00_u03b1_129_, lean_object* v_t_130_, lean_object* v_e_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_aesop_Aesop_getMatch(v_00_u03b1_129_, v_t_130_, v_e_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_);
lean_dec(v_a_135_);
lean_dec_ref(v_a_134_);
lean_dec(v_a_133_);
lean_dec_ref(v_a_132_);
lean_dec_ref(v_t_130_);
return v_res_137_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree_Types(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree_Main(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_indexConfig = _init_lp_aesop_Aesop_indexConfig();
lean_mark_persistent(lp_aesop_Aesop_indexConfig);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree_Types(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree_Main(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
}
#ifdef __cplusplus
}
#endif
