// Lean compiler output
// Module: Mathlib.Lean.Meta.RefinedDiscrTree
// Imports: public import Init public meta import Init public import Mathlib.Lean.Meta.RefinedDiscrTree.Lookup public import Mathlib.Lean.Meta.RefinedDiscrTree.Initialize
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_getMatch___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_getDiag(lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_createImportedDiscrTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_createModuleTreeRef___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_0__Lean_Meta_RefinedDiscrTree_withTreeCtx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "RefinedDiscrTree import initialization"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "RefinedDiscrTree local search"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_0__Lean_Meta_RefinedDiscrTree_withTreeCtx(lean_object* v_ctx_1_){
_start:
{
lean_object* v_fileName_2_; lean_object* v_fileMap_3_; lean_object* v_options_4_; lean_object* v_currRecDepth_5_; lean_object* v_maxRecDepth_6_; lean_object* v_ref_7_; lean_object* v_currNamespace_8_; lean_object* v_openDecls_9_; lean_object* v_initHeartbeats_10_; lean_object* v_quotContext_11_; lean_object* v_currMacroScope_12_; lean_object* v_cancelTk_x3f_13_; uint8_t v_suppressElabErrors_14_; lean_object* v_inheritedTraceOptions_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v_fileName_2_ = lean_ctor_get(v_ctx_1_, 0);
v_fileMap_3_ = lean_ctor_get(v_ctx_1_, 1);
v_options_4_ = lean_ctor_get(v_ctx_1_, 2);
v_currRecDepth_5_ = lean_ctor_get(v_ctx_1_, 3);
v_maxRecDepth_6_ = lean_ctor_get(v_ctx_1_, 4);
v_ref_7_ = lean_ctor_get(v_ctx_1_, 5);
v_currNamespace_8_ = lean_ctor_get(v_ctx_1_, 6);
v_openDecls_9_ = lean_ctor_get(v_ctx_1_, 7);
v_initHeartbeats_10_ = lean_ctor_get(v_ctx_1_, 8);
v_quotContext_11_ = lean_ctor_get(v_ctx_1_, 10);
v_currMacroScope_12_ = lean_ctor_get(v_ctx_1_, 11);
v_cancelTk_x3f_13_ = lean_ctor_get(v_ctx_1_, 12);
v_suppressElabErrors_14_ = lean_ctor_get_uint8(v_ctx_1_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_15_ = lean_ctor_get(v_ctx_1_, 13);
v_isSharedCheck_24_ = !lean_is_exclusive(v_ctx_1_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v_ctx_1_, 9);
lean_dec(v_unused_25_);
v___x_17_ = v_ctx_1_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_inheritedTraceOptions_15_);
lean_inc(v_cancelTk_x3f_13_);
lean_inc(v_currMacroScope_12_);
lean_inc(v_quotContext_11_);
lean_inc(v_initHeartbeats_10_);
lean_inc(v_openDecls_9_);
lean_inc(v_currNamespace_8_);
lean_inc(v_ref_7_);
lean_inc(v_maxRecDepth_6_);
lean_inc(v_currRecDepth_5_);
lean_inc(v_options_4_);
lean_inc(v_fileMap_3_);
lean_inc(v_fileName_2_);
lean_dec(v_ctx_1_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; uint8_t v___x_20_; lean_object* v___x_22_; 
v___x_19_ = lean_unsigned_to_nat(0u);
v___x_20_ = l_Lean_getDiag(v_options_4_);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 9, v___x_19_);
v___x_22_ = v___x_17_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_fileName_2_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_fileMap_3_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_options_4_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_currRecDepth_5_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_maxRecDepth_6_);
lean_ctor_set(v_reuseFailAlloc_23_, 5, v_ref_7_);
lean_ctor_set(v_reuseFailAlloc_23_, 6, v_currNamespace_8_);
lean_ctor_set(v_reuseFailAlloc_23_, 7, v_openDecls_9_);
lean_ctor_set(v_reuseFailAlloc_23_, 8, v_initHeartbeats_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 9, v___x_19_);
lean_ctor_set(v_reuseFailAlloc_23_, 10, v_quotContext_11_);
lean_ctor_set(v_reuseFailAlloc_23_, 11, v_currMacroScope_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 12, v_cancelTk_x3f_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 13, v_inheritedTraceOptions_15_);
lean_ctor_set_uint8(v_reuseFailAlloc_23_, sizeof(void*)*14 + 1, v_suppressElabErrors_14_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
lean_ctor_set_uint8(v___x_22_, sizeof(void*)*14, v___x_20_);
return v___x_22_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg(lean_object* v_category_26_, lean_object* v_opts_27_, lean_object* v_act_28_, lean_object* v_decl_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
lean_inc(v___y_33_);
lean_inc_ref(v___y_32_);
lean_inc(v___y_31_);
lean_inc_ref(v___y_30_);
v___x_35_ = lean_apply_4(v_act_28_, v___y_30_, v___y_31_, v___y_32_, v___y_33_);
v___x_36_ = l_Lean_profileitIOUnsafe___redArg(v_category_26_, v_opts_27_, v___x_35_, v_decl_29_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg___boxed(lean_object* v_category_37_, lean_object* v_opts_38_, lean_object* v_act_39_, lean_object* v_decl_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg(v_category_37_, v_opts_38_, v_act_39_, v_decl_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
lean_dec_ref(v_opts_38_);
lean_dec_ref(v_category_37_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0(lean_object* v_00_u03b1_47_, lean_object* v_category_48_, lean_object* v_opts_49_, lean_object* v_act_50_, lean_object* v_decl_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg(v_category_48_, v_opts_49_, v_act_50_, v_decl_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___boxed(lean_object* v_00_u03b1_58_, lean_object* v_category_59_, lean_object* v_opts_60_, lean_object* v_act_61_, lean_object* v_decl_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0(v_00_u03b1_58_, v_category_59_, v_opts_60_, v_act_61_, v_decl_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
lean_dec_ref(v_opts_60_);
lean_dec_ref(v_category_59_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___lam__0(lean_object* v___x_69_, lean_object* v_env_70_, lean_object* v_addEntry_71_, lean_object* v_constantsPerTask_72_, lean_object* v_capacityPerTask_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
lean_inc_ref(v___y_76_);
v___x_79_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_0__Lean_Meta_RefinedDiscrTree_withTreeCtx(v___y_76_);
v___x_80_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_createImportedDiscrTree___redArg(v___x_69_, v_env_70_, v_addEntry_71_, v_constantsPerTask_72_, v_capacityPerTask_73_, v___x_79_, v___y_77_);
lean_dec_ref(v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___lam__0___boxed(lean_object* v___x_81_, lean_object* v_env_82_, lean_object* v_addEntry_83_, lean_object* v_constantsPerTask_84_, lean_object* v_capacityPerTask_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___lam__0(v___x_81_, v_env_82_, v_addEntry_83_, v_constantsPerTask_84_, v_capacityPerTask_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
lean_dec(v_constantsPerTask_84_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg(lean_object* v_ext_93_, lean_object* v_addEntry_94_, lean_object* v_ty_95_, lean_object* v_constantsPerTask_96_, lean_object* v_capacityPerTask_97_, lean_object* v_a_98_, lean_object* v_a_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v___x_103_; lean_object* v_ngen_104_; lean_object* v_namePrefix_105_; lean_object* v_idx_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_183_; 
v___x_103_ = lean_st_ref_get(v_a_101_);
v_ngen_104_ = lean_ctor_get(v___x_103_, 2);
lean_inc_ref(v_ngen_104_);
lean_dec(v___x_103_);
v_namePrefix_105_ = lean_ctor_get(v_ngen_104_, 0);
v_idx_106_ = lean_ctor_get(v_ngen_104_, 1);
v_isSharedCheck_183_ = !lean_is_exclusive(v_ngen_104_);
if (v_isSharedCheck_183_ == 0)
{
v___x_108_ = v_ngen_104_;
v_isShared_109_ = v_isSharedCheck_183_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_idx_106_);
lean_inc(v_namePrefix_105_);
lean_dec(v_ngen_104_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_183_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v___x_110_; lean_object* v_env_111_; lean_object* v_nextMacroScope_112_; lean_object* v_auxDeclNGen_113_; lean_object* v_traceState_114_; lean_object* v_cache_115_; lean_object* v_messages_116_; lean_object* v_infoState_117_; lean_object* v_snapshotTasks_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_181_; 
v___x_110_ = lean_st_ref_take(v_a_101_);
v_env_111_ = lean_ctor_get(v___x_110_, 0);
v_nextMacroScope_112_ = lean_ctor_get(v___x_110_, 1);
v_auxDeclNGen_113_ = lean_ctor_get(v___x_110_, 3);
v_traceState_114_ = lean_ctor_get(v___x_110_, 4);
v_cache_115_ = lean_ctor_get(v___x_110_, 5);
v_messages_116_ = lean_ctor_get(v___x_110_, 6);
v_infoState_117_ = lean_ctor_get(v___x_110_, 7);
v_snapshotTasks_118_ = lean_ctor_get(v___x_110_, 8);
v_isSharedCheck_181_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_181_ == 0)
{
lean_object* v_unused_182_; 
v_unused_182_ = lean_ctor_get(v___x_110_, 2);
lean_dec(v_unused_182_);
v___x_120_ = v___x_110_;
v_isShared_121_ = v_isSharedCheck_181_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_snapshotTasks_118_);
lean_inc(v_infoState_117_);
lean_inc(v_messages_116_);
lean_inc(v_cache_115_);
lean_inc(v_traceState_114_);
lean_inc(v_auxDeclNGen_113_);
lean_inc(v_nextMacroScope_112_);
lean_inc(v_env_111_);
lean_dec(v___x_110_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_181_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_126_; 
lean_inc(v_idx_106_);
lean_inc(v_namePrefix_105_);
v___x_122_ = l_Lean_Name_num___override(v_namePrefix_105_, v_idx_106_);
v___x_123_ = lean_unsigned_to_nat(1u);
v___x_124_ = lean_nat_add(v_idx_106_, v___x_123_);
lean_dec(v_idx_106_);
if (v_isShared_109_ == 0)
{
lean_ctor_set(v___x_108_, 1, v___x_124_);
v___x_126_ = v___x_108_;
goto v_reusejp_125_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_namePrefix_105_);
lean_ctor_set(v_reuseFailAlloc_180_, 1, v___x_124_);
v___x_126_ = v_reuseFailAlloc_180_;
goto v_reusejp_125_;
}
v_reusejp_125_:
{
lean_object* v___x_128_; 
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 2, v___x_126_);
v___x_128_ = v___x_120_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_env_111_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v_nextMacroScope_112_);
lean_ctor_set(v_reuseFailAlloc_179_, 2, v___x_126_);
lean_ctor_set(v_reuseFailAlloc_179_, 3, v_auxDeclNGen_113_);
lean_ctor_set(v_reuseFailAlloc_179_, 4, v_traceState_114_);
lean_ctor_set(v_reuseFailAlloc_179_, 5, v_cache_115_);
lean_ctor_set(v_reuseFailAlloc_179_, 6, v_messages_116_);
lean_ctor_set(v_reuseFailAlloc_179_, 7, v_infoState_117_);
lean_ctor_set(v_reuseFailAlloc_179_, 8, v_snapshotTasks_118_);
v___x_128_ = v_reuseFailAlloc_179_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v_env_133_; lean_object* v_asyncMode_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_a_138_; lean_object* v___x_161_; 
v___x_129_ = lean_st_ref_set(v_a_101_, v___x_128_);
v___x_130_ = lean_box(0);
v___x_131_ = lean_st_mk_ref(v___x_130_);
v___x_132_ = lean_st_ref_get(v_a_101_);
v_env_133_ = lean_ctor_get(v___x_132_, 0);
lean_inc_ref(v_env_133_);
lean_dec(v___x_132_);
v_asyncMode_134_ = lean_ctor_get(v_ext_93_, 2);
v___x_135_ = lean_box(0);
v___x_136_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_131_, v_ext_93_, v_env_133_, v_asyncMode_134_, v___x_135_);
lean_dec(v___x_131_);
v___x_161_ = lean_st_ref_get(v___x_136_);
if (lean_obj_tag(v___x_161_) == 0)
{
lean_object* v___x_162_; lean_object* v_options_163_; lean_object* v_env_164_; lean_object* v___x_165_; lean_object* v___f_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_162_ = lean_st_ref_get(v_a_101_);
v_options_163_ = lean_ctor_get(v_a_100_, 2);
v_env_164_ = lean_ctor_get(v___x_162_, 0);
lean_inc_ref(v_env_164_);
lean_dec(v___x_162_);
v___x_165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_122_);
lean_ctor_set(v___x_165_, 1, v___x_123_);
v___f_166_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_166_, 0, v___x_165_);
lean_closure_set(v___f_166_, 1, v_env_164_);
lean_closure_set(v___f_166_, 2, v_addEntry_94_);
lean_closure_set(v___f_166_, 3, v_constantsPerTask_96_);
lean_closure_set(v___f_166_, 4, v_capacityPerTask_97_);
v___x_167_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___closed__0));
v___x_168_ = lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg(v___x_167_, v_options_163_, v___f_166_, v___x_135_, v_a_98_, v_a_99_, v_a_100_, v_a_101_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v_a_169_; 
v_a_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_a_169_);
lean_dec_ref_known(v___x_168_, 1);
v_a_138_ = v_a_169_;
goto v___jp_137_;
}
else
{
lean_object* v_a_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_177_; 
lean_dec(v___x_136_);
lean_dec_ref(v_ty_95_);
v_a_170_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_177_ == 0)
{
v___x_172_ = v___x_168_;
v_isShared_173_ = v_isSharedCheck_177_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_a_170_);
lean_dec(v___x_168_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_177_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___x_175_; 
if (v_isShared_173_ == 0)
{
v___x_175_ = v___x_172_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v_a_170_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
else
{
lean_object* v_val_178_; 
lean_dec(v___x_122_);
lean_dec(v_capacityPerTask_97_);
lean_dec(v_constantsPerTask_96_);
lean_dec_ref(v_addEntry_94_);
v_val_178_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_val_178_);
lean_dec_ref_known(v___x_161_, 1);
v_a_138_ = v_val_178_;
goto v___jp_137_;
}
v___jp_137_:
{
uint8_t v___x_139_; lean_object* v___x_140_; 
v___x_139_ = 0;
v___x_140_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_getMatch___redArg(v_a_138_, v_ty_95_, v___x_139_, v___x_139_, v_a_98_, v_a_99_, v_a_100_, v_a_101_);
if (lean_obj_tag(v___x_140_) == 0)
{
lean_object* v_a_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_152_; 
v_a_141_ = lean_ctor_get(v___x_140_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_140_);
if (v_isSharedCheck_152_ == 0)
{
v___x_143_ = v___x_140_;
v_isShared_144_ = v_isSharedCheck_152_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_a_141_);
lean_dec(v___x_140_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_152_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v_fst_145_; lean_object* v_snd_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_150_; 
v_fst_145_ = lean_ctor_get(v_a_141_, 0);
lean_inc(v_fst_145_);
v_snd_146_ = lean_ctor_get(v_a_141_, 1);
lean_inc(v_snd_146_);
lean_dec(v_a_141_);
v___x_147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_147_, 0, v_snd_146_);
v___x_148_ = lean_st_ref_set(v___x_136_, v___x_147_);
lean_dec(v___x_136_);
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 0, v_fst_145_);
v___x_150_ = v___x_143_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_fst_145_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
else
{
lean_object* v_a_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_160_; 
lean_dec(v___x_136_);
v_a_153_ = lean_ctor_get(v___x_140_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_140_);
if (v_isSharedCheck_160_ == 0)
{
v___x_155_ = v___x_140_;
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_a_153_);
lean_dec(v___x_140_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v___x_158_; 
if (v_isShared_156_ == 0)
{
v___x_158_ = v___x_155_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v_a_153_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg___boxed(lean_object* v_ext_184_, lean_object* v_addEntry_185_, lean_object* v_ty_186_, lean_object* v_constantsPerTask_187_, lean_object* v_capacityPerTask_188_, lean_object* v_a_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg(v_ext_184_, v_addEntry_185_, v_ty_186_, v_constantsPerTask_187_, v_capacityPerTask_188_, v_a_189_, v_a_190_, v_a_191_, v_a_192_);
lean_dec(v_a_192_);
lean_dec_ref(v_a_191_);
lean_dec(v_a_190_);
lean_dec_ref(v_a_189_);
lean_dec_ref(v_ext_184_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches(lean_object* v_00_u03b1_195_, lean_object* v_ext_196_, lean_object* v_addEntry_197_, lean_object* v_ty_198_, lean_object* v_constantsPerTask_199_, lean_object* v_capacityPerTask_200_, lean_object* v_a_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg(v_ext_196_, v_addEntry_197_, v_ty_198_, v_constantsPerTask_199_, v_capacityPerTask_200_, v_a_201_, v_a_202_, v_a_203_, v_a_204_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___boxed(lean_object* v_00_u03b1_207_, lean_object* v_ext_208_, lean_object* v_addEntry_209_, lean_object* v_ty_210_, lean_object* v_constantsPerTask_211_, lean_object* v_capacityPerTask_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches(v_00_u03b1_207_, v_ext_208_, v_addEntry_209_, v_ty_210_, v_constantsPerTask_211_, v_capacityPerTask_212_, v_a_213_, v_a_214_, v_a_215_, v_a_216_);
lean_dec(v_a_216_);
lean_dec_ref(v_a_215_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_213_);
lean_dec_ref(v_ext_208_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___lam__0(lean_object* v_moduleRef_219_, lean_object* v_ty_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
lean_object* v___x_226_; uint8_t v___x_227_; lean_object* v___x_228_; 
v___x_226_ = lean_st_ref_get(v_moduleRef_219_);
v___x_227_ = 0;
v___x_228_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_getMatch___redArg(v___x_226_, v_ty_220_, v___x_227_, v___x_227_, v___y_221_, v___y_222_, v___y_223_, v___y_224_);
if (lean_obj_tag(v___x_228_) == 0)
{
lean_object* v_a_229_; lean_object* v___x_231_; uint8_t v_isShared_232_; uint8_t v_isSharedCheck_239_; 
v_a_229_ = lean_ctor_get(v___x_228_, 0);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_228_);
if (v_isSharedCheck_239_ == 0)
{
v___x_231_ = v___x_228_;
v_isShared_232_ = v_isSharedCheck_239_;
goto v_resetjp_230_;
}
else
{
lean_inc(v_a_229_);
lean_dec(v___x_228_);
v___x_231_ = lean_box(0);
v_isShared_232_ = v_isSharedCheck_239_;
goto v_resetjp_230_;
}
v_resetjp_230_:
{
lean_object* v_fst_233_; lean_object* v_snd_234_; lean_object* v___x_235_; lean_object* v___x_237_; 
v_fst_233_ = lean_ctor_get(v_a_229_, 0);
lean_inc(v_fst_233_);
v_snd_234_ = lean_ctor_get(v_a_229_, 1);
lean_inc(v_snd_234_);
lean_dec(v_a_229_);
v___x_235_ = lean_st_ref_set(v_moduleRef_219_, v_snd_234_);
if (v_isShared_232_ == 0)
{
lean_ctor_set(v___x_231_, 0, v_fst_233_);
v___x_237_ = v___x_231_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v_fst_233_);
v___x_237_ = v_reuseFailAlloc_238_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
return v___x_237_;
}
}
}
else
{
lean_object* v_a_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
v_a_240_ = lean_ctor_get(v___x_228_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_228_);
if (v_isSharedCheck_247_ == 0)
{
v___x_242_ = v___x_228_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_a_240_);
lean_dec(v___x_228_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_a_240_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___lam__0___boxed(lean_object* v_moduleRef_248_, lean_object* v_ty_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___lam__0(v_moduleRef_248_, v_ty_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_);
lean_dec(v___y_253_);
lean_dec_ref(v___y_252_);
lean_dec(v___y_251_);
lean_dec_ref(v___y_250_);
lean_dec(v_moduleRef_248_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg(lean_object* v_moduleRef_257_, lean_object* v_ty_258_, lean_object* v_a_259_, lean_object* v_a_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v_options_264_; lean_object* v___f_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; 
v_options_264_ = lean_ctor_get(v_a_261_, 2);
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_265_, 0, v_moduleRef_257_);
lean_closure_set(v___f_265_, 1, v_ty_258_);
v___x_266_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___closed__0));
v___x_267_ = lean_box(0);
v___x_268_ = lp_mathlib_Lean_profileitM___at___00Lean_Meta_RefinedDiscrTree_findImportMatches_spec__0___redArg(v___x_266_, v_options_264_, v___f_265_, v___x_267_, v_a_259_, v_a_260_, v_a_261_, v_a_262_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg___boxed(lean_object* v_moduleRef_269_, lean_object* v_ty_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg(v_moduleRef_269_, v_ty_270_, v_a_271_, v_a_272_, v_a_273_, v_a_274_);
lean_dec(v_a_274_);
lean_dec_ref(v_a_273_);
lean_dec(v_a_272_);
lean_dec_ref(v_a_271_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches(lean_object* v_00_u03b1_277_, lean_object* v_moduleRef_278_, lean_object* v_ty_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg(v_moduleRef_278_, v_ty_279_, v_a_280_, v_a_281_, v_a_282_, v_a_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___boxed(lean_object* v_00_u03b1_286_, lean_object* v_moduleRef_287_, lean_object* v_ty_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches(v_00_u03b1_286_, v_moduleRef_287_, v_ty_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_);
lean_dec(v_a_292_);
lean_dec_ref(v_a_291_);
lean_dec(v_a_290_);
lean_dec_ref(v_a_289_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___redArg(lean_object* v_ext_295_, lean_object* v_addEntry_296_, lean_object* v_ty_297_, lean_object* v_constantsPerTask_298_, lean_object* v_capacityPerTask_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_){
_start:
{
lean_object* v___x_305_; 
lean_inc_ref(v_addEntry_296_);
v___x_305_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_createModuleTreeRef___redArg(v_addEntry_296_, v_a_300_, v_a_301_, v_a_302_, v_a_303_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v_a_306_; lean_object* v___x_307_; 
v_a_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_a_306_);
lean_dec_ref_known(v___x_305_, 1);
lean_inc_ref(v_ty_297_);
v___x_307_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findModuleMatches___redArg(v_a_306_, v_ty_297_, v_a_300_, v_a_301_, v_a_302_, v_a_303_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v_a_308_; lean_object* v___x_309_; 
v_a_308_ = lean_ctor_get(v___x_307_, 0);
lean_inc(v_a_308_);
lean_dec_ref_known(v___x_307_, 1);
v___x_309_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findImportMatches___redArg(v_ext_295_, v_addEntry_296_, v_ty_297_, v_constantsPerTask_298_, v_capacityPerTask_299_, v_a_300_, v_a_301_, v_a_302_, v_a_303_);
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v_a_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_318_; 
v_a_310_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_318_ == 0)
{
v___x_312_ = v___x_309_;
v_isShared_313_ = v_isSharedCheck_318_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_a_310_);
lean_dec(v___x_309_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_318_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___x_314_; lean_object* v___x_316_; 
v___x_314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_314_, 0, v_a_308_);
lean_ctor_set(v___x_314_, 1, v_a_310_);
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 0, v___x_314_);
v___x_316_ = v___x_312_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_317_; 
v_reuseFailAlloc_317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_317_, 0, v___x_314_);
v___x_316_ = v_reuseFailAlloc_317_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
return v___x_316_;
}
}
}
else
{
lean_object* v_a_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_326_; 
lean_dec(v_a_308_);
v_a_319_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_326_ == 0)
{
v___x_321_ = v___x_309_;
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___x_309_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_324_; 
if (v_isShared_322_ == 0)
{
v___x_324_ = v___x_321_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_a_319_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
else
{
lean_object* v_a_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_334_; 
lean_dec(v_capacityPerTask_299_);
lean_dec(v_constantsPerTask_298_);
lean_dec_ref(v_ty_297_);
lean_dec_ref(v_addEntry_296_);
v_a_327_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_334_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_334_ == 0)
{
v___x_329_ = v___x_307_;
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_a_327_);
lean_dec(v___x_307_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_332_; 
if (v_isShared_330_ == 0)
{
v___x_332_ = v___x_329_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_a_327_);
v___x_332_ = v_reuseFailAlloc_333_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
return v___x_332_;
}
}
}
}
else
{
lean_object* v_a_335_; lean_object* v___x_337_; uint8_t v_isShared_338_; uint8_t v_isSharedCheck_342_; 
lean_dec(v_capacityPerTask_299_);
lean_dec(v_constantsPerTask_298_);
lean_dec_ref(v_ty_297_);
lean_dec_ref(v_addEntry_296_);
v_a_335_ = lean_ctor_get(v___x_305_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_305_);
if (v_isSharedCheck_342_ == 0)
{
v___x_337_ = v___x_305_;
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
else
{
lean_inc(v_a_335_);
lean_dec(v___x_305_);
v___x_337_ = lean_box(0);
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
v_resetjp_336_:
{
lean_object* v___x_340_; 
if (v_isShared_338_ == 0)
{
v___x_340_ = v___x_337_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_a_335_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___redArg___boxed(lean_object* v_ext_343_, lean_object* v_addEntry_344_, lean_object* v_ty_345_, lean_object* v_constantsPerTask_346_, lean_object* v_capacityPerTask_347_, lean_object* v_a_348_, lean_object* v_a_349_, lean_object* v_a_350_, lean_object* v_a_351_, lean_object* v_a_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___redArg(v_ext_343_, v_addEntry_344_, v_ty_345_, v_constantsPerTask_346_, v_capacityPerTask_347_, v_a_348_, v_a_349_, v_a_350_, v_a_351_);
lean_dec(v_a_351_);
lean_dec_ref(v_a_350_);
lean_dec(v_a_349_);
lean_dec_ref(v_a_348_);
lean_dec_ref(v_ext_343_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches(lean_object* v_00_u03b1_354_, lean_object* v_ext_355_, lean_object* v_addEntry_356_, lean_object* v_ty_357_, lean_object* v_constantsPerTask_358_, lean_object* v_capacityPerTask_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___redArg(v_ext_355_, v_addEntry_356_, v_ty_357_, v_constantsPerTask_358_, v_capacityPerTask_359_, v_a_360_, v_a_361_, v_a_362_, v_a_363_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches___boxed(lean_object* v_00_u03b1_366_, lean_object* v_ext_367_, lean_object* v_addEntry_368_, lean_object* v_ty_369_, lean_object* v_constantsPerTask_370_, lean_object* v_capacityPerTask_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_findMatches(v_00_u03b1_366_, v_ext_367_, v_addEntry_368_, v_ty_369_, v_constantsPerTask_370_, v_capacityPerTask_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_);
lean_dec(v_a_375_);
lean_dec_ref(v_a_374_);
lean_dec(v_a_373_);
lean_dec_ref(v_a_372_);
lean_dec_ref(v_ext_367_);
return v_res_377_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Lookup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Initialize(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree(builtin);
}
#ifdef __cplusplus
}
#endif
