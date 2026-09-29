// Lean compiler output
// Module: Mathlib.Lean.Elab.Term
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.Elab.Term
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeLightImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg(v_e_30_, v___y_34_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___boxed(lean_object* v_e_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0(v_e_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern___lam__0(lean_object* v_patt_48_, lean_object* v_expectedType_x3f_49_, uint8_t v___x_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = l_Lean_Elab_Term_elabTerm(v_patt_48_, v_expectedType_x3f_49_, v___x_50_, v___x_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_);
if (lean_obj_tag(v___x_58_) == 0)
{
lean_object* v_a_59_; uint8_t v___x_60_; lean_object* v___x_61_; 
v_a_59_ = lean_ctor_get(v___x_58_, 0);
lean_inc(v_a_59_);
lean_dec_ref_known(v___x_58_, 1);
v___x_60_ = 1;
v___x_61_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_60_, v___x_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_);
if (lean_obj_tag(v___x_61_) == 0)
{
lean_object* v___x_62_; 
lean_dec_ref_known(v___x_61_, 1);
v___x_62_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_elabPattern_spec__0___redArg(v_a_59_, v___y_54_);
return v___x_62_;
}
else
{
lean_object* v_a_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_70_; 
lean_dec(v_a_59_);
v_a_63_ = lean_ctor_get(v___x_61_, 0);
v_isSharedCheck_70_ = !lean_is_exclusive(v___x_61_);
if (v_isSharedCheck_70_ == 0)
{
v___x_65_ = v___x_61_;
v_isShared_66_ = v_isSharedCheck_70_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_a_63_);
lean_dec(v___x_61_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_70_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v___x_68_; 
if (v_isShared_66_ == 0)
{
v___x_68_ = v___x_65_;
goto v_reusejp_67_;
}
else
{
lean_object* v_reuseFailAlloc_69_; 
v_reuseFailAlloc_69_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_69_, 0, v_a_63_);
v___x_68_ = v_reuseFailAlloc_69_;
goto v_reusejp_67_;
}
v_reusejp_67_:
{
return v___x_68_;
}
}
}
}
else
{
return v___x_58_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern___lam__0___boxed(lean_object* v_patt_71_, lean_object* v_expectedType_x3f_72_, lean_object* v___x_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
uint8_t v___x_871__boxed_81_; lean_object* v_res_82_; 
v___x_871__boxed_81_ = lean_unbox(v___x_73_);
v_res_82_ = lp_mathlib_Lean_Elab_Term_elabPattern___lam__0(v_patt_71_, v_expectedType_x3f_72_, v___x_871__boxed_81_, v___y_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern(lean_object* v_patt_83_, lean_object* v_expectedType_x3f_84_, lean_object* v_a_85_, lean_object* v_a_86_, lean_object* v_a_87_, lean_object* v_a_88_, lean_object* v_a_89_, lean_object* v_a_90_){
_start:
{
lean_object* v_declName_x3f_92_; lean_object* v_macroStack_93_; uint8_t v_mayPostpone_94_; lean_object* v_autoBoundImplicitContext_95_; lean_object* v_autoBoundImplicitForbidden_96_; lean_object* v_sectionVars_97_; lean_object* v_sectionFVars_98_; uint8_t v_implicitLambda_99_; uint8_t v_heedElabAsElim_100_; uint8_t v_isNoncomputableSection_101_; uint8_t v_isMetaSection_102_; uint8_t v_inPattern_103_; lean_object* v_tacSnap_x3f_104_; uint8_t v_saveRecAppSyntax_105_; uint8_t v_holesAsSyntheticOpaque_106_; uint8_t v_checkDeprecated_107_; lean_object* v_fixedTermElabs_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___f_111_; uint8_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v_declName_x3f_92_ = lean_ctor_get(v_a_85_, 0);
v_macroStack_93_ = lean_ctor_get(v_a_85_, 1);
v_mayPostpone_94_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8);
v_autoBoundImplicitContext_95_ = lean_ctor_get(v_a_85_, 2);
v_autoBoundImplicitForbidden_96_ = lean_ctor_get(v_a_85_, 3);
v_sectionVars_97_ = lean_ctor_get(v_a_85_, 4);
v_sectionFVars_98_ = lean_ctor_get(v_a_85_, 5);
v_implicitLambda_99_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 2);
v_heedElabAsElim_100_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 3);
v_isNoncomputableSection_101_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 4);
v_isMetaSection_102_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 5);
v_inPattern_103_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 7);
v_tacSnap_x3f_104_ = lean_ctor_get(v_a_85_, 6);
v_saveRecAppSyntax_105_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 8);
v_holesAsSyntheticOpaque_106_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 9);
v_checkDeprecated_107_ = lean_ctor_get_uint8(v_a_85_, sizeof(void*)*8 + 10);
v_fixedTermElabs_108_ = lean_ctor_get(v_a_85_, 7);
v___x_109_ = 1;
v___x_110_ = lean_box(v___x_109_);
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_elabPattern___lam__0___boxed), 10, 3);
lean_closure_set(v___f_111_, 0, v_patt_83_);
lean_closure_set(v___f_111_, 1, v_expectedType_x3f_84_);
lean_closure_set(v___f_111_, 2, v___x_110_);
v___x_112_ = 0;
lean_inc_ref(v_fixedTermElabs_108_);
lean_inc(v_tacSnap_x3f_104_);
lean_inc(v_sectionFVars_98_);
lean_inc(v_sectionVars_97_);
lean_inc_ref(v_autoBoundImplicitForbidden_96_);
lean_inc(v_autoBoundImplicitContext_95_);
lean_inc(v_macroStack_93_);
lean_inc(v_declName_x3f_92_);
v___x_113_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_113_, 0, v_declName_x3f_92_);
lean_ctor_set(v___x_113_, 1, v_macroStack_93_);
lean_ctor_set(v___x_113_, 2, v_autoBoundImplicitContext_95_);
lean_ctor_set(v___x_113_, 3, v_autoBoundImplicitForbidden_96_);
lean_ctor_set(v___x_113_, 4, v_sectionVars_97_);
lean_ctor_set(v___x_113_, 5, v_sectionFVars_98_);
lean_ctor_set(v___x_113_, 6, v_tacSnap_x3f_104_);
lean_ctor_set(v___x_113_, 7, v_fixedTermElabs_108_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8, v_mayPostpone_94_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 1, v___x_112_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 2, v_implicitLambda_99_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 3, v_heedElabAsElim_100_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 4, v_isNoncomputableSection_101_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 5, v_isMetaSection_102_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 6, v___x_109_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 7, v_inPattern_103_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 8, v_saveRecAppSyntax_105_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 9, v_holesAsSyntheticOpaque_106_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*8 + 10, v_checkDeprecated_107_);
v___x_114_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeLightImp(lean_box(0), v___f_111_, v___x_113_, v_a_86_, v_a_87_, v_a_88_, v_a_89_, v_a_90_);
lean_dec_ref_known(v___x_113_, 8);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_elabPattern___boxed(lean_object* v_patt_115_, lean_object* v_expectedType_x3f_116_, lean_object* v_a_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Lean_Elab_Term_elabPattern(v_patt_115_, v_expectedType_x3f_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_, v_a_121_, v_a_122_);
lean_dec(v_a_122_);
lean_dec_ref(v_a_121_);
lean_dec(v_a_120_);
lean_dec_ref(v_a_119_);
lean_dec(v_a_118_);
lean_dec_ref(v_a_117_);
return v_res_124_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Term(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Elab_Term(uint8_t builtin) {
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
res = runtime_initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Elab_Term(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Term(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Elab_Term(uint8_t builtin) {
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
res = initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Elab_Term(builtin);
}
#ifdef __cplusplus
}
#endif
