// Lean compiler output
// Module: Aesop.RuleTac
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Apply public import Aesop.RuleTac.Basic public import Aesop.RuleTac.Cases public import Aesop.RuleTac.Forward public import Aesop.RuleTac.Preprocess public import Aesop.RuleTac.Tactic public import Aesop.RuleTac.Descr
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
lean_object* lp_aesop_Aesop_RuleTac_apply(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_applyConsts(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_forward(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_cases(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_ruleTacImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_singleRuleTacImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_tacticStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_preprocess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_forwardMatches(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run(lean_object* v_x_1_, lean_object* v_a_2_, lean_object* v_a_3_, lean_object* v_a_4_, lean_object* v_a_5_, lean_object* v_a_6_, lean_object* v_a_7_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v_term_9_; uint8_t v_md_10_; lean_object* v___x_11_; 
v_term_9_ = lean_ctor_get(v_x_1_, 0);
lean_inc_ref(v_term_9_);
v_md_10_ = lean_ctor_get_uint8(v_x_1_, sizeof(void*)*1);
lean_dec_ref_known(v_x_1_, 1);
v___x_11_ = lp_aesop_Aesop_RuleTac_apply(v_term_9_, v_md_10_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_11_;
}
case 1:
{
lean_object* v_constructorNames_12_; uint8_t v_md_13_; lean_object* v___x_14_; 
v_constructorNames_12_ = lean_ctor_get(v_x_1_, 0);
lean_inc_ref(v_constructorNames_12_);
v_md_13_ = lean_ctor_get_uint8(v_x_1_, sizeof(void*)*1);
lean_dec_ref_known(v_x_1_, 1);
v___x_14_ = lp_aesop_Aesop_RuleTac_applyConsts(v_constructorNames_12_, v_md_13_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_14_;
}
case 2:
{
lean_object* v_term_15_; lean_object* v_immediate_16_; uint8_t v_isDestruct_17_; lean_object* v___x_18_; 
v_term_15_ = lean_ctor_get(v_x_1_, 0);
lean_inc_ref(v_term_15_);
v_immediate_16_ = lean_ctor_get(v_x_1_, 1);
lean_inc_ref(v_immediate_16_);
v_isDestruct_17_ = lean_ctor_get_uint8(v_x_1_, sizeof(void*)*2);
lean_dec_ref_known(v_x_1_, 2);
v___x_18_ = lp_aesop_Aesop_RuleTac_forward(v_term_15_, v_immediate_16_, v_isDestruct_17_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_18_;
}
case 3:
{
lean_object* v_target_19_; uint8_t v_md_20_; uint8_t v_isRecursiveType_21_; lean_object* v_ctorNames_22_; lean_object* v___x_23_; 
v_target_19_ = lean_ctor_get(v_x_1_, 0);
lean_inc_ref(v_target_19_);
v_md_20_ = lean_ctor_get_uint8(v_x_1_, sizeof(void*)*2);
v_isRecursiveType_21_ = lean_ctor_get_uint8(v_x_1_, sizeof(void*)*2 + 1);
v_ctorNames_22_ = lean_ctor_get(v_x_1_, 1);
lean_inc_ref(v_ctorNames_22_);
lean_dec_ref_known(v_x_1_, 2);
v___x_23_ = lp_aesop_Aesop_RuleTac_cases(v_target_19_, v_md_20_, v_isRecursiveType_21_, v_ctorNames_22_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_23_;
}
case 4:
{
lean_object* v_decl_24_; lean_object* v___x_25_; 
v_decl_24_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_decl_24_);
lean_dec_ref_known(v_x_1_, 1);
v___x_25_ = lp_aesop_Aesop_RuleTac_tacticMImpl(v_decl_24_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_25_;
}
case 5:
{
lean_object* v_decl_26_; lean_object* v___x_27_; 
v_decl_26_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_decl_26_);
lean_dec_ref_known(v_x_1_, 1);
v___x_27_ = lp_aesop_Aesop_RuleTac_ruleTacImpl(v_decl_26_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_27_;
}
case 6:
{
lean_object* v_decl_28_; lean_object* v___x_29_; 
v_decl_28_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_decl_28_);
lean_dec_ref_known(v_x_1_, 1);
v___x_29_ = lp_aesop_Aesop_RuleTac_tacGenImpl(v_decl_28_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_29_;
}
case 7:
{
lean_object* v_decl_30_; lean_object* v___x_31_; 
v_decl_30_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_decl_30_);
lean_dec_ref_known(v_x_1_, 1);
v___x_31_ = lp_aesop_Aesop_RuleTac_singleRuleTacImpl(v_decl_30_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_31_;
}
case 8:
{
lean_object* v_stx_32_; lean_object* v___x_33_; 
v_stx_32_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_stx_32_);
lean_dec_ref_known(v_x_1_, 1);
v___x_33_ = lp_aesop_Aesop_RuleTac_tacticStx(v_stx_32_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_33_;
}
case 9:
{
lean_object* v___x_34_; 
v___x_34_ = lp_aesop_Aesop_RuleTac_preprocess(v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_34_;
}
default: 
{
lean_object* v_ms_35_; lean_object* v___x_36_; 
v_ms_35_ = lean_ctor_get(v_x_1_, 0);
lean_inc_ref(v_ms_35_);
lean_dec_ref_known(v_x_1_, 1);
v___x_36_ = lp_aesop_Aesop_RuleTac_forwardMatches(v_ms_35_, v_a_2_, v_a_3_, v_a_4_, v_a_5_, v_a_6_, v_a_7_);
return v___x_36_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object* v_x_37_, lean_object* v_a_38_, lean_object* v_a_39_, lean_object* v_a_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_aesop_Aesop_RuleTacDescr_run(v_x_37_, v_a_38_, v_a_39_, v_a_40_, v_a_41_, v_a_42_, v_a_43_);
lean_dec(v_a_43_);
lean_dec_ref(v_a_42_);
lean_dec(v_a_41_);
lean_dec_ref(v_a_40_);
lean_dec(v_a_39_);
return v_res_45_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Apply(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Forward(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Tactic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Descr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Preprocess(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Descr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_Apply(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Forward(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Preprocess(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Descr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Preprocess(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Descr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac(builtin);
}
#ifdef __cplusplus
}
#endif
