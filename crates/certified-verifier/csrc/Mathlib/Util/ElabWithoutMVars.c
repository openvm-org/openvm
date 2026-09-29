// Lean compiler output
// Module: Mathlib.Util.ElabWithoutMVars
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermWithHoles(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Argument passed to "};
static const lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__0 = (const lean_object*)&lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__1;
static const lean_string_object lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = " has metavariables:"};
static const lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__2 = (const lean_object*)&lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg(lean_object* v_a_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_, lean_object* v___y_9_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
lean_inc(v___y_3_);
lean_inc_ref(v___y_2_);
v___x_11_ = lean_apply_2(v_a_1_, v___y_2_, v___y_3_);
v___x_12_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___x_11_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, v___y_8_, v___y_9_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg___boxed(lean_object* v_a_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg(v_a_13_, v___y_14_, v___y_15_, v___y_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_, v___y_21_);
lean_dec(v___y_21_);
lean_dec_ref(v___y_20_);
lean_dec(v___y_19_);
lean_dec_ref(v___y_18_);
lean_dec(v___y_17_);
lean_dec_ref(v___y_16_);
lean_dec(v___y_15_);
lean_dec_ref(v___y_14_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1(lean_object* v_00_u03b1_24_, lean_object* v_a_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg(v_a_25_, v___y_26_, v___y_27_, v___y_28_, v___y_29_, v___y_30_, v___y_31_, v___y_32_, v___y_33_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___boxed(lean_object* v_00_u03b1_36_, lean_object* v_a_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1(v_00_u03b1_36_, v_a_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0_spec__2(lean_object* v_msgData_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_){
_start:
{
lean_object* v___x_54_; lean_object* v_env_55_; lean_object* v___x_56_; lean_object* v_mctx_57_; lean_object* v_lctx_58_; lean_object* v_options_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_54_ = lean_st_ref_get(v___y_52_);
v_env_55_ = lean_ctor_get(v___x_54_, 0);
lean_inc_ref(v_env_55_);
lean_dec(v___x_54_);
v___x_56_ = lean_st_ref_get(v___y_50_);
v_mctx_57_ = lean_ctor_get(v___x_56_, 0);
lean_inc_ref(v_mctx_57_);
lean_dec(v___x_56_);
v_lctx_58_ = lean_ctor_get(v___y_49_, 2);
v_options_59_ = lean_ctor_get(v___y_51_, 2);
lean_inc_ref(v_options_59_);
lean_inc_ref(v_lctx_58_);
v___x_60_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_60_, 0, v_env_55_);
lean_ctor_set(v___x_60_, 1, v_mctx_57_);
lean_ctor_set(v___x_60_, 2, v_lctx_58_);
lean_ctor_set(v___x_60_, 3, v_options_59_);
v___x_61_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v_msgData_48_);
v___x_62_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0_spec__2___boxed(lean_object* v_msgData_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0_spec__2(v_msgData_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg(lean_object* v_msg_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_ref_76_; lean_object* v___x_77_; lean_object* v_a_78_; lean_object* v___x_80_; uint8_t v_isShared_81_; uint8_t v_isSharedCheck_86_; 
v_ref_76_ = lean_ctor_get(v___y_73_, 5);
v___x_77_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0_spec__2(v_msg_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_);
v_a_78_ = lean_ctor_get(v___x_77_, 0);
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_86_ == 0)
{
v___x_80_ = v___x_77_;
v_isShared_81_ = v_isSharedCheck_86_;
goto v_resetjp_79_;
}
else
{
lean_inc(v_a_78_);
lean_dec(v___x_77_);
v___x_80_ = lean_box(0);
v_isShared_81_ = v_isSharedCheck_86_;
goto v_resetjp_79_;
}
v_resetjp_79_:
{
lean_object* v___x_82_; lean_object* v___x_84_; 
lean_inc(v_ref_76_);
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, v_ref_76_);
lean_ctor_set(v___x_82_, 1, v_a_78_);
if (v_isShared_81_ == 0)
{
lean_ctor_set_tag(v___x_80_, 1);
lean_ctor_set(v___x_80_, 0, v___x_82_);
v___x_84_ = v___x_80_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v___x_82_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg___boxed(lean_object* v_msg_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg(v_msg_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg(lean_object* v_ref_94_, lean_object* v_msg_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v_fileName_105_; lean_object* v_fileMap_106_; lean_object* v_options_107_; lean_object* v_currRecDepth_108_; lean_object* v_maxRecDepth_109_; lean_object* v_ref_110_; lean_object* v_currNamespace_111_; lean_object* v_openDecls_112_; lean_object* v_initHeartbeats_113_; lean_object* v_maxHeartbeats_114_; lean_object* v_quotContext_115_; lean_object* v_currMacroScope_116_; uint8_t v_diag_117_; lean_object* v_cancelTk_x3f_118_; uint8_t v_suppressElabErrors_119_; lean_object* v_inheritedTraceOptions_120_; lean_object* v_ref_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v_fileName_105_ = lean_ctor_get(v___y_102_, 0);
v_fileMap_106_ = lean_ctor_get(v___y_102_, 1);
v_options_107_ = lean_ctor_get(v___y_102_, 2);
v_currRecDepth_108_ = lean_ctor_get(v___y_102_, 3);
v_maxRecDepth_109_ = lean_ctor_get(v___y_102_, 4);
v_ref_110_ = lean_ctor_get(v___y_102_, 5);
v_currNamespace_111_ = lean_ctor_get(v___y_102_, 6);
v_openDecls_112_ = lean_ctor_get(v___y_102_, 7);
v_initHeartbeats_113_ = lean_ctor_get(v___y_102_, 8);
v_maxHeartbeats_114_ = lean_ctor_get(v___y_102_, 9);
v_quotContext_115_ = lean_ctor_get(v___y_102_, 10);
v_currMacroScope_116_ = lean_ctor_get(v___y_102_, 11);
v_diag_117_ = lean_ctor_get_uint8(v___y_102_, sizeof(void*)*14);
v_cancelTk_x3f_118_ = lean_ctor_get(v___y_102_, 12);
v_suppressElabErrors_119_ = lean_ctor_get_uint8(v___y_102_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_120_ = lean_ctor_get(v___y_102_, 13);
v_ref_121_ = l_Lean_replaceRef(v_ref_94_, v_ref_110_);
lean_inc_ref(v_inheritedTraceOptions_120_);
lean_inc(v_cancelTk_x3f_118_);
lean_inc(v_currMacroScope_116_);
lean_inc(v_quotContext_115_);
lean_inc(v_maxHeartbeats_114_);
lean_inc(v_initHeartbeats_113_);
lean_inc(v_openDecls_112_);
lean_inc(v_currNamespace_111_);
lean_inc(v_maxRecDepth_109_);
lean_inc(v_currRecDepth_108_);
lean_inc_ref(v_options_107_);
lean_inc_ref(v_fileMap_106_);
lean_inc_ref(v_fileName_105_);
v___x_122_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_122_, 0, v_fileName_105_);
lean_ctor_set(v___x_122_, 1, v_fileMap_106_);
lean_ctor_set(v___x_122_, 2, v_options_107_);
lean_ctor_set(v___x_122_, 3, v_currRecDepth_108_);
lean_ctor_set(v___x_122_, 4, v_maxRecDepth_109_);
lean_ctor_set(v___x_122_, 5, v_ref_121_);
lean_ctor_set(v___x_122_, 6, v_currNamespace_111_);
lean_ctor_set(v___x_122_, 7, v_openDecls_112_);
lean_ctor_set(v___x_122_, 8, v_initHeartbeats_113_);
lean_ctor_set(v___x_122_, 9, v_maxHeartbeats_114_);
lean_ctor_set(v___x_122_, 10, v_quotContext_115_);
lean_ctor_set(v___x_122_, 11, v_currMacroScope_116_);
lean_ctor_set(v___x_122_, 12, v_cancelTk_x3f_118_);
lean_ctor_set(v___x_122_, 13, v_inheritedTraceOptions_120_);
lean_ctor_set_uint8(v___x_122_, sizeof(void*)*14, v_diag_117_);
lean_ctor_set_uint8(v___x_122_, sizeof(void*)*14 + 1, v_suppressElabErrors_119_);
v___x_123_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg(v_msg_95_, v___y_100_, v___y_101_, v___x_122_, v___y_103_);
lean_dec_ref_known(v___x_122_, 14);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg___boxed(lean_object* v_ref_124_, lean_object* v_msg_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg(v_ref_124_, v_msg_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v_ref_124_);
return v_res_135_;
}
}
static lean_object* _init_lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__1(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = ((lean_object*)(lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__0));
v___x_138_ = l_Lean_stringToMessageData(v___x_137_);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__3(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = ((lean_object*)(lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__2));
v___x_141_ = l_Lean_stringToMessageData(v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0(lean_object* v_t_142_, lean_object* v___x_143_, lean_object* v_tactic_144_, uint8_t v___x_145_, lean_object* v___x_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_){
_start:
{
lean_object* v___x_156_; 
lean_inc(v_tactic_144_);
lean_inc(v_t_142_);
v___x_156_ = l_Lean_Elab_Tactic_elabTermWithHoles(v_t_142_, v___x_143_, v_tactic_144_, v___x_145_, v___x_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_);
if (lean_obj_tag(v___x_156_) == 0)
{
lean_object* v_a_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_190_; 
v_a_157_ = lean_ctor_get(v___x_156_, 0);
v_isSharedCheck_190_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_190_ == 0)
{
v___x_159_ = v___x_156_;
v_isShared_160_ = v_isSharedCheck_190_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_a_157_);
lean_dec(v___x_156_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_190_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v_fst_161_; lean_object* v_snd_162_; lean_object* v___x_164_; uint8_t v_isShared_165_; uint8_t v_isSharedCheck_189_; 
v_fst_161_ = lean_ctor_get(v_a_157_, 0);
v_snd_162_ = lean_ctor_get(v_a_157_, 1);
v_isSharedCheck_189_ = !lean_is_exclusive(v_a_157_);
if (v_isSharedCheck_189_ == 0)
{
v___x_164_ = v_a_157_;
v_isShared_165_ = v_isSharedCheck_189_;
goto v_resetjp_163_;
}
else
{
lean_inc(v_snd_162_);
lean_inc(v_fst_161_);
lean_dec(v_a_157_);
v___x_164_ = lean_box(0);
v_isShared_165_ = v_isSharedCheck_189_;
goto v_resetjp_163_;
}
v_resetjp_163_:
{
uint8_t v___x_166_; 
v___x_166_ = l_List_isEmpty___redArg(v_snd_162_);
lean_dec(v_snd_162_);
if (v___x_166_ == 0)
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_170_; 
lean_del_object(v___x_159_);
v___x_167_ = lean_obj_once(&lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__1, &lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__1_once, _init_lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__1);
v___x_168_ = l_Lean_MessageData_ofName(v_tactic_144_);
if (v_isShared_165_ == 0)
{
lean_ctor_set_tag(v___x_164_, 7);
lean_ctor_set(v___x_164_, 1, v___x_168_);
lean_ctor_set(v___x_164_, 0, v___x_167_);
v___x_170_ = v___x_164_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_167_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v___x_168_);
v___x_170_ = v_reuseFailAlloc_185_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v_a_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_184_; 
v___x_171_ = lean_obj_once(&lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__3, &lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__3_once, _init_lp_mathlib_elabTermWithoutNewMVars___lam__0___closed__3);
v___x_172_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_170_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = l_Lean_MessageData_ofExpr(v_fst_161_);
v___x_174_ = l_Lean_indentD(v___x_173_);
v___x_175_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_172_);
lean_ctor_set(v___x_175_, 1, v___x_174_);
v___x_176_ = lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg(v_t_142_, v___x_175_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_);
lean_dec(v_t_142_);
v_a_177_ = lean_ctor_get(v___x_176_, 0);
v_isSharedCheck_184_ = !lean_is_exclusive(v___x_176_);
if (v_isSharedCheck_184_ == 0)
{
v___x_179_ = v___x_176_;
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_a_177_);
lean_dec(v___x_176_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_184_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___x_182_; 
if (v_isShared_180_ == 0)
{
v___x_182_ = v___x_179_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v_a_177_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
}
}
else
{
lean_object* v___x_187_; 
lean_del_object(v___x_164_);
lean_dec(v_tactic_144_);
lean_dec(v_t_142_);
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 0, v_fst_161_);
v___x_187_ = v___x_159_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_fst_161_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
}
}
}
else
{
lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_198_; 
lean_dec(v_tactic_144_);
lean_dec(v_t_142_);
v_a_191_ = lean_ctor_get(v___x_156_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_198_ == 0)
{
v___x_193_ = v___x_156_;
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_156_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_196_; 
if (v_isShared_194_ == 0)
{
v___x_196_ = v___x_193_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_a_191_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars___lam__0___boxed(lean_object* v_t_199_, lean_object* v___x_200_, lean_object* v_tactic_201_, lean_object* v___x_202_, lean_object* v___x_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
uint8_t v___x_4180__boxed_213_; lean_object* v_res_214_; 
v___x_4180__boxed_213_ = lean_unbox(v___x_202_);
v_res_214_ = lp_mathlib_elabTermWithoutNewMVars___lam__0(v_t_199_, v___x_200_, v_tactic_201_, v___x_4180__boxed_213_, v___x_203_, v___y_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
lean_dec(v___y_209_);
lean_dec_ref(v___y_208_);
lean_dec(v___y_207_);
lean_dec_ref(v___y_206_);
lean_dec(v___y_205_);
lean_dec_ref(v___y_204_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars(lean_object* v_tactic_215_, lean_object* v_t_216_, lean_object* v_a_217_, lean_object* v_a_218_, lean_object* v_a_219_, lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_, lean_object* v_a_224_){
_start:
{
lean_object* v___x_226_; uint8_t v___x_227_; lean_object* v___x_228_; lean_object* v___f_229_; lean_object* v___x_230_; 
v___x_226_ = lean_box(0);
v___x_227_ = 0;
v___x_228_ = lean_box(v___x_227_);
v___f_229_ = lean_alloc_closure((void*)(lp_mathlib_elabTermWithoutNewMVars___lam__0___boxed), 14, 5);
lean_closure_set(v___f_229_, 0, v_t_216_);
lean_closure_set(v___f_229_, 1, v___x_226_);
lean_closure_set(v___f_229_, 2, v_tactic_215_);
lean_closure_set(v___f_229_, 3, v___x_228_);
lean_closure_set(v___f_229_, 4, v___x_226_);
v___x_230_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00elabTermWithoutNewMVars_spec__1___redArg(v___f_229_, v_a_217_, v_a_218_, v_a_219_, v_a_220_, v_a_221_, v_a_222_, v_a_223_, v_a_224_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabTermWithoutNewMVars___boxed(lean_object* v_tactic_231_, lean_object* v_t_232_, lean_object* v_a_233_, lean_object* v_a_234_, lean_object* v_a_235_, lean_object* v_a_236_, lean_object* v_a_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_elabTermWithoutNewMVars(v_tactic_231_, v_t_232_, v_a_233_, v_a_234_, v_a_235_, v_a_236_, v_a_237_, v_a_238_, v_a_239_, v_a_240_);
lean_dec(v_a_240_);
lean_dec_ref(v_a_239_);
lean_dec(v_a_238_);
lean_dec_ref(v_a_237_);
lean_dec(v_a_236_);
lean_dec_ref(v_a_235_);
lean_dec(v_a_234_);
lean_dec_ref(v_a_233_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0(lean_object* v_00_u03b1_243_, lean_object* v_ref_244_, lean_object* v_msg_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___redArg(v_ref_244_, v_msg_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0___boxed(lean_object* v_00_u03b1_256_, lean_object* v_ref_257_, lean_object* v_msg_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0(v_00_u03b1_256_, v_ref_257_, v_msg_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec(v_ref_257_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0(lean_object* v_00_u03b1_269_, lean_object* v_msg_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___redArg(v_msg_270_, v___y_275_, v___y_276_, v___y_277_, v___y_278_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0___boxed(lean_object* v_00_u03b1_281_, lean_object* v_msg_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00elabTermWithoutNewMVars_spec__0_spec__0(v_00_u03b1_281_, v_msg_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
return v_res_292_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_ElabWithoutMVars(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_ElabWithoutMVars(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_ElabWithoutMVars(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_ElabWithoutMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_ElabWithoutMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_ElabWithoutMVars(builtin);
}
#ifdef __cplusplus
}
#endif
