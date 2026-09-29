// Lean compiler output
// Module: Mathlib.Util.SynthesizeUsing
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Basic public import Qq public import Qq.Typ
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_goalsToMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_synthesizeUsing___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_synthesizeUsing___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_synthesizeUsing___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_synthesizeUsing___redArg___closed__0 = (const lean_object*)&lp_mathlib_synthesizeUsing___redArg___closed__0_value;
static const lean_array_object lp_mathlib_synthesizeUsing___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_synthesizeUsing___redArg___closed__1 = (const lean_object*)&lp_mathlib_synthesizeUsing___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_synthesizeUsing___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_synthesizeUsing___redArg___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_synthesizeUsing___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_synthesizeUsing___redArg___closed__2 = (const lean_object*)&lp_mathlib_synthesizeUsing___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_synthesizeUsing___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_synthesizeUsing___redArg___closed__3 = (const lean_object*)&lp_mathlib_synthesizeUsing___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00synthesizeUsing_x27_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00synthesizeUsing_x27_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_synthesizeUsing_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "synthesizeUsing': unsolved goals\n"};
static const lean_object* lp_mathlib_synthesizeUsing_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_synthesizeUsing_x27___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_synthesizeUsing_x27___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_synthesizeUsing_x27___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___redArg(lean_object* v_a_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1_, v___y_2_, v___y_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___redArg___boxed(lean_object* v_a_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___redArg(v_a_10_, v___y_11_, v___y_12_, v___y_13_, v___y_14_, v___y_15_, v___y_16_);
lean_dec(v___y_16_);
lean_dec_ref(v___y_15_);
lean_dec(v___y_14_);
lean_dec_ref(v___y_13_);
lean_dec(v___y_12_);
lean_dec_ref(v___y_11_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0(lean_object* v_00_u03b1_19_, lean_object* v_a_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_20_, v___y_21_, v___y_22_, v___y_23_, v___y_24_, v___y_25_, v___y_26_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___boxed(lean_object* v_00_u03b1_29_, lean_object* v_a_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0(v_00_u03b1_29_, v_a_30_, v___y_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_);
lean_dec(v___y_36_);
lean_dec_ref(v___y_35_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
lean_dec(v___y_32_);
lean_dec_ref(v___y_31_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg(lean_object* v_e_39_, lean_object* v___y_40_){
_start:
{
uint8_t v___x_42_; 
v___x_42_ = l_Lean_Expr_hasMVar(v_e_39_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_43_, 0, v_e_39_);
return v___x_43_;
}
else
{
lean_object* v___x_44_; lean_object* v_mctx_45_; lean_object* v___x_46_; lean_object* v_fst_47_; lean_object* v_snd_48_; lean_object* v___x_49_; lean_object* v_cache_50_; lean_object* v_zetaDeltaFVarIds_51_; lean_object* v_postponed_52_; lean_object* v_diag_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_62_; 
v___x_44_ = lean_st_ref_get(v___y_40_);
v_mctx_45_ = lean_ctor_get(v___x_44_, 0);
lean_inc_ref(v_mctx_45_);
lean_dec(v___x_44_);
v___x_46_ = l_Lean_instantiateMVarsCore(v_mctx_45_, v_e_39_);
v_fst_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc(v_fst_47_);
v_snd_48_ = lean_ctor_get(v___x_46_, 1);
lean_inc(v_snd_48_);
lean_dec_ref(v___x_46_);
v___x_49_ = lean_st_ref_take(v___y_40_);
v_cache_50_ = lean_ctor_get(v___x_49_, 1);
v_zetaDeltaFVarIds_51_ = lean_ctor_get(v___x_49_, 2);
v_postponed_52_ = lean_ctor_get(v___x_49_, 3);
v_diag_53_ = lean_ctor_get(v___x_49_, 4);
v_isSharedCheck_62_ = !lean_is_exclusive(v___x_49_);
if (v_isSharedCheck_62_ == 0)
{
lean_object* v_unused_63_; 
v_unused_63_ = lean_ctor_get(v___x_49_, 0);
lean_dec(v_unused_63_);
v___x_55_ = v___x_49_;
v_isShared_56_ = v_isSharedCheck_62_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_diag_53_);
lean_inc(v_postponed_52_);
lean_inc(v_zetaDeltaFVarIds_51_);
lean_inc(v_cache_50_);
lean_dec(v___x_49_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_62_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_58_; 
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 0, v_snd_48_);
v___x_58_ = v___x_55_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_61_; 
v_reuseFailAlloc_61_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_61_, 0, v_snd_48_);
lean_ctor_set(v_reuseFailAlloc_61_, 1, v_cache_50_);
lean_ctor_set(v_reuseFailAlloc_61_, 2, v_zetaDeltaFVarIds_51_);
lean_ctor_set(v_reuseFailAlloc_61_, 3, v_postponed_52_);
lean_ctor_set(v_reuseFailAlloc_61_, 4, v_diag_53_);
v___x_58_ = v_reuseFailAlloc_61_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_st_ref_set(v___y_40_, v___x_58_);
v___x_60_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_60_, 0, v_fst_47_);
return v___x_60_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg___boxed(lean_object* v_e_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg(v_e_64_, v___y_65_);
lean_dec(v___y_65_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1(lean_object* v_e_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg(v_e_68_, v___y_70_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___boxed(lean_object* v_e_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1(v_e_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
return v_res_81_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_synthesizeUsing___redArg___lam__0(lean_object* v_x_82_){
_start:
{
uint8_t v___x_83_; 
v___x_83_ = 0;
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___redArg___lam__0___boxed(lean_object* v_x_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_mathlib_synthesizeUsing___redArg___lam__0(v_x_84_);
lean_dec(v_x_84_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___redArg(lean_object* v_type_101_, lean_object* v_tac_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v___x_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_108_, 0, v_type_101_);
v___x_109_ = 0;
v___x_110_ = lean_box(0);
v___x_111_ = l_Lean_Meta_mkFreshExprMVar(v___x_108_, v___x_109_, v___x_110_, v_a_103_, v_a_104_, v_a_105_, v_a_106_);
if (lean_obj_tag(v___x_111_) == 0)
{
lean_object* v_a_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v_a_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_a_112_);
lean_dec_ref_known(v___x_111_, 1);
v___x_113_ = l_Lean_Expr_mvarId_x21(v_a_112_);
v___x_114_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_114_, 0, v___x_113_);
lean_closure_set(v___x_114_, 1, v_tac_102_);
v___x_115_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00synthesizeUsing_spec__0___boxed), 9, 2);
lean_closure_set(v___x_115_, 0, lean_box(0));
lean_closure_set(v___x_115_, 1, v___x_114_);
v___x_116_ = ((lean_object*)(lp_mathlib_synthesizeUsing___redArg___closed__2));
v___x_117_ = ((lean_object*)(lp_mathlib_synthesizeUsing___redArg___closed__3));
v___x_118_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_115_, v___x_116_, v___x_117_, v_a_103_, v_a_104_, v_a_105_, v_a_106_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_object* v_a_119_; lean_object* v___x_120_; lean_object* v_a_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_137_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
lean_inc(v_a_119_);
lean_dec_ref_known(v___x_118_, 1);
v___x_120_ = lp_mathlib_Lean_instantiateMVars___at___00synthesizeUsing_spec__1___redArg(v_a_112_, v_a_104_);
v_a_121_ = lean_ctor_get(v___x_120_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_137_ == 0)
{
v___x_123_ = v___x_120_;
v_isShared_124_ = v_isSharedCheck_137_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_a_121_);
lean_dec(v___x_120_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_137_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v_fst_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_135_; 
v_fst_125_ = lean_ctor_get(v_a_119_, 0);
v_isSharedCheck_135_ = !lean_is_exclusive(v_a_119_);
if (v_isSharedCheck_135_ == 0)
{
lean_object* v_unused_136_; 
v_unused_136_ = lean_ctor_get(v_a_119_, 1);
lean_dec(v_unused_136_);
v___x_127_ = v_a_119_;
v_isShared_128_ = v_isSharedCheck_135_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_fst_125_);
lean_dec(v_a_119_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_135_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 1, v_a_121_);
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v_fst_125_);
lean_ctor_set(v_reuseFailAlloc_134_, 1, v_a_121_);
v___x_130_ = v_reuseFailAlloc_134_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
lean_object* v___x_132_; 
if (v_isShared_124_ == 0)
{
lean_ctor_set(v___x_123_, 0, v___x_130_);
v___x_132_ = v___x_123_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_130_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
}
}
}
else
{
lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
lean_dec(v_a_112_);
v_a_138_ = lean_ctor_get(v___x_118_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_145_ == 0)
{
v___x_140_ = v___x_118_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_118_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_a_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
}
else
{
lean_object* v_a_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_153_; 
lean_dec_ref(v_tac_102_);
v_a_146_ = lean_ctor_get(v___x_111_, 0);
v_isSharedCheck_153_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_153_ == 0)
{
v___x_148_ = v___x_111_;
v_isShared_149_ = v_isSharedCheck_153_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_a_146_);
lean_dec(v___x_111_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_153_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_151_; 
if (v_isShared_149_ == 0)
{
v___x_151_ = v___x_148_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v_a_146_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___redArg___boxed(lean_object* v_type_154_, lean_object* v_tac_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_synthesizeUsing___redArg(v_type_154_, v_tac_155_, v_a_156_, v_a_157_, v_a_158_, v_a_159_);
lean_dec(v_a_159_);
lean_dec_ref(v_a_158_);
lean_dec(v_a_157_);
lean_dec_ref(v_a_156_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing(lean_object* v_u_162_, lean_object* v_type_163_, lean_object* v_tac_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_synthesizeUsing___redArg(v_type_163_, v_tac_164_, v_a_165_, v_a_166_, v_a_167_, v_a_168_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing___boxed(lean_object* v_u_171_, lean_object* v_type_172_, lean_object* v_tac_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_synthesizeUsing(v_u_171_, v_type_172_, v_tac_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
lean_dec(v_a_177_);
lean_dec_ref(v_a_176_);
lean_dec(v_a_175_);
lean_dec_ref(v_a_174_);
lean_dec(v_u_171_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00synthesizeUsing_x27_spec__0_spec__0(lean_object* v_msgData_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
lean_object* v___x_186_; lean_object* v_env_187_; lean_object* v___x_188_; lean_object* v_mctx_189_; lean_object* v_lctx_190_; lean_object* v_options_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_186_ = lean_st_ref_get(v___y_184_);
v_env_187_ = lean_ctor_get(v___x_186_, 0);
lean_inc_ref(v_env_187_);
lean_dec(v___x_186_);
v___x_188_ = lean_st_ref_get(v___y_182_);
v_mctx_189_ = lean_ctor_get(v___x_188_, 0);
lean_inc_ref(v_mctx_189_);
lean_dec(v___x_188_);
v_lctx_190_ = lean_ctor_get(v___y_181_, 2);
v_options_191_ = lean_ctor_get(v___y_183_, 2);
lean_inc_ref(v_options_191_);
lean_inc_ref(v_lctx_190_);
v___x_192_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_192_, 0, v_env_187_);
lean_ctor_set(v___x_192_, 1, v_mctx_189_);
lean_ctor_set(v___x_192_, 2, v_lctx_190_);
lean_ctor_set(v___x_192_, 3, v_options_191_);
v___x_193_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_192_);
lean_ctor_set(v___x_193_, 1, v_msgData_180_);
v___x_194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00synthesizeUsing_x27_spec__0_spec__0___boxed(lean_object* v_msgData_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00synthesizeUsing_x27_spec__0_spec__0(v_msgData_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg(lean_object* v_msg_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_){
_start:
{
lean_object* v_ref_208_; lean_object* v___x_209_; lean_object* v_a_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_218_; 
v_ref_208_ = lean_ctor_get(v___y_205_, 5);
v___x_209_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00synthesizeUsing_x27_spec__0_spec__0(v_msg_202_, v___y_203_, v___y_204_, v___y_205_, v___y_206_);
v_a_210_ = lean_ctor_get(v___x_209_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v___x_209_);
if (v_isSharedCheck_218_ == 0)
{
v___x_212_ = v___x_209_;
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_a_210_);
lean_dec(v___x_209_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_214_; lean_object* v___x_216_; 
lean_inc(v_ref_208_);
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v_ref_208_);
lean_ctor_set(v___x_214_, 1, v_a_210_);
if (v_isShared_213_ == 0)
{
lean_ctor_set_tag(v___x_212_, 1);
lean_ctor_set(v___x_212_, 0, v___x_214_);
v___x_216_ = v___x_212_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v___x_214_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg___boxed(lean_object* v_msg_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg(v_msg_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
return v_res_225_;
}
}
static lean_object* _init_lp_mathlib_synthesizeUsing_x27___redArg___closed__1(void){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_227_ = ((lean_object*)(lp_mathlib_synthesizeUsing_x27___redArg___closed__0));
v___x_228_ = l_Lean_stringToMessageData(v___x_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27___redArg(lean_object* v_type_229_, lean_object* v_tac_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib_synthesizeUsing___redArg(v_type_229_, v_tac_230_, v_a_231_, v_a_232_, v_a_233_, v_a_234_);
if (lean_obj_tag(v___x_236_) == 0)
{
lean_object* v_a_237_; lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_265_; 
v_a_237_ = lean_ctor_get(v___x_236_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_236_);
if (v_isSharedCheck_265_ == 0)
{
v___x_239_ = v___x_236_;
v_isShared_240_ = v_isSharedCheck_265_;
goto v_resetjp_238_;
}
else
{
lean_inc(v_a_237_);
lean_dec(v___x_236_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_265_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v_fst_241_; lean_object* v_snd_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_264_; 
v_fst_241_ = lean_ctor_get(v_a_237_, 0);
v_snd_242_ = lean_ctor_get(v_a_237_, 1);
v_isSharedCheck_264_ = !lean_is_exclusive(v_a_237_);
if (v_isSharedCheck_264_ == 0)
{
v___x_244_ = v_a_237_;
v_isShared_245_ = v_isSharedCheck_264_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_snd_242_);
lean_inc(v_fst_241_);
lean_dec(v_a_237_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_264_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
uint8_t v___x_246_; 
v___x_246_ = l_List_isEmpty___redArg(v_fst_241_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_250_; 
lean_dec(v_snd_242_);
lean_del_object(v___x_239_);
v___x_247_ = lean_obj_once(&lp_mathlib_synthesizeUsing_x27___redArg___closed__1, &lp_mathlib_synthesizeUsing_x27___redArg___closed__1_once, _init_lp_mathlib_synthesizeUsing_x27___redArg___closed__1);
v___x_248_ = l_Lean_Elab_goalsToMessageData(v_fst_241_);
if (v_isShared_245_ == 0)
{
lean_ctor_set_tag(v___x_244_, 7);
lean_ctor_set(v___x_244_, 1, v___x_248_);
lean_ctor_set(v___x_244_, 0, v___x_247_);
v___x_250_ = v___x_244_;
goto v_reusejp_249_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v___x_247_);
lean_ctor_set(v_reuseFailAlloc_260_, 1, v___x_248_);
v___x_250_ = v_reuseFailAlloc_260_;
goto v_reusejp_249_;
}
v_reusejp_249_:
{
lean_object* v___x_251_; lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
v___x_251_ = lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg(v___x_250_, v_a_231_, v_a_232_, v_a_233_, v_a_234_);
v_a_252_ = lean_ctor_get(v___x_251_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_251_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_251_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_251_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
else
{
lean_object* v___x_262_; 
lean_del_object(v___x_244_);
lean_dec(v_fst_241_);
if (v_isShared_240_ == 0)
{
lean_ctor_set(v___x_239_, 0, v_snd_242_);
v___x_262_ = v___x_239_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_snd_242_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
}
else
{
lean_object* v_a_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_273_; 
v_a_266_ = lean_ctor_get(v___x_236_, 0);
v_isSharedCheck_273_ = !lean_is_exclusive(v___x_236_);
if (v_isSharedCheck_273_ == 0)
{
v___x_268_ = v___x_236_;
v_isShared_269_ = v_isSharedCheck_273_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_a_266_);
lean_dec(v___x_236_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_273_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_271_; 
if (v_isShared_269_ == 0)
{
v___x_271_ = v___x_268_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v_a_266_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
return v___x_271_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27___redArg___boxed(lean_object* v_type_274_, lean_object* v_tac_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_synthesizeUsing_x27___redArg(v_type_274_, v_tac_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
lean_dec(v_a_279_);
lean_dec_ref(v_a_278_);
lean_dec(v_a_277_);
lean_dec_ref(v_a_276_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27(lean_object* v_u_282_, lean_object* v_type_283_, lean_object* v_tac_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lp_mathlib_synthesizeUsing_x27___redArg(v_type_283_, v_tac_284_, v_a_285_, v_a_286_, v_a_287_, v_a_288_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsing_x27___boxed(lean_object* v_u_291_, lean_object* v_type_292_, lean_object* v_tac_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_synthesizeUsing_x27(v_u_291_, v_type_292_, v_tac_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_);
lean_dec(v_a_297_);
lean_dec_ref(v_a_296_);
lean_dec(v_a_295_);
lean_dec_ref(v_a_294_);
lean_dec(v_u_291_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0(lean_object* v_00_u03b1_300_, lean_object* v_msg_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___redArg(v_msg_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0___boxed(lean_object* v_00_u03b1_308_, lean_object* v_msg_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_Lean_throwError___at___00synthesizeUsing_x27_spec__0(v_00_u03b1_308_, v_msg_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic___redArg(lean_object* v_type_316_, lean_object* v_tac_317_, lean_object* v_a_318_, lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v_a_321_){
_start:
{
lean_object* v___x_323_; lean_object* v___x_324_; 
v___x_323_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_323_, 0, v_tac_317_);
v___x_324_ = lp_mathlib_synthesizeUsing___redArg(v_type_316_, v___x_323_, v_a_318_, v_a_319_, v_a_320_, v_a_321_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic___redArg___boxed(lean_object* v_type_325_, lean_object* v_tac_326_, lean_object* v_a_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_, lean_object* v_a_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_mathlib_synthesizeUsingTactic___redArg(v_type_325_, v_tac_326_, v_a_327_, v_a_328_, v_a_329_, v_a_330_);
lean_dec(v_a_330_);
lean_dec_ref(v_a_329_);
lean_dec(v_a_328_);
lean_dec_ref(v_a_327_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic(lean_object* v_u_333_, lean_object* v_type_334_, lean_object* v_tac_335_, lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_synthesizeUsingTactic___redArg(v_type_334_, v_tac_335_, v_a_336_, v_a_337_, v_a_338_, v_a_339_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic___boxed(lean_object* v_u_342_, lean_object* v_type_343_, lean_object* v_tac_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_, lean_object* v_a_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_synthesizeUsingTactic(v_u_342_, v_type_343_, v_tac_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
lean_dec(v_a_348_);
lean_dec_ref(v_a_347_);
lean_dec(v_a_346_);
lean_dec_ref(v_a_345_);
lean_dec(v_u_342_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg(lean_object* v_type_351_, lean_object* v_tac_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_358_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_358_, 0, v_tac_352_);
v___x_359_ = lp_mathlib_synthesizeUsing_x27___redArg(v_type_351_, v___x_358_, v_a_353_, v_a_354_, v_a_355_, v_a_356_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg___boxed(lean_object* v_type_360_, lean_object* v_tac_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_synthesizeUsingTactic_x27___redArg(v_type_360_, v_tac_361_, v_a_362_, v_a_363_, v_a_364_, v_a_365_);
lean_dec(v_a_365_);
lean_dec_ref(v_a_364_);
lean_dec(v_a_363_);
lean_dec_ref(v_a_362_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27(lean_object* v_u_368_, lean_object* v_type_369_, lean_object* v_tac_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_mathlib_synthesizeUsingTactic_x27___redArg(v_type_369_, v_tac_370_, v_a_371_, v_a_372_, v_a_373_, v_a_374_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_synthesizeUsingTactic_x27___boxed(lean_object* v_u_377_, lean_object* v_type_378_, lean_object* v_tac_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_synthesizeUsingTactic_x27(v_u_377_, v_type_378_, v_tac_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
lean_dec(v_a_383_);
lean_dec_ref(v_a_382_);
lean_dec(v_a_381_);
lean_dec_ref(v_a_380_);
lean_dec(v_u_377_);
return v_res_385_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin) {
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
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_Qq_Qq_Typ(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
}
#ifdef __cplusplus
}
#endif
