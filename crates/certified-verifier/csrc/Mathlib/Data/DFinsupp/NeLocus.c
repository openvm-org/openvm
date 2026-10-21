// Lean compiler output
// Module: Mathlib.Data.DFinsupp.NeLocus
// Imports: public import Init public meta import Init public import Mathlib.Data.DFinsupp.Defs
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
lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_ndunion___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_neLocus___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_neLocus___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_neLocus___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_neLocus___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_neLocus___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_neLocus___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg___lam__0(lean_object* v_f_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v_f_1_, 0);
lean_inc(v_toFun_3_);
lean_dec_ref(v_f_1_);
v___x_4_ = lean_apply_1(v_toFun_3_, v___y_2_);
return v___x_4_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_neLocus___redArg___lam__1(lean_object* v___f_5_, lean_object* v_f_6_, lean_object* v_g_7_, lean_object* v_inst_8_, lean_object* v_a_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; uint8_t v___x_13_; 
lean_inc(v___f_5_);
lean_inc_n(v_a_9_, 2);
v___x_10_ = lean_apply_2(v___f_5_, v_f_6_, v_a_9_);
v___x_11_ = lean_apply_2(v___f_5_, v_g_7_, v_a_9_);
v___x_12_ = lean_apply_3(v_inst_8_, v_a_9_, v___x_10_, v___x_11_);
v___x_13_ = lean_unbox(v___x_12_);
if (v___x_13_ == 0)
{
uint8_t v___x_14_; 
v___x_14_ = 1;
return v___x_14_;
}
else
{
uint8_t v___x_15_; 
v___x_15_ = 0;
return v___x_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg___lam__1___boxed(lean_object* v___f_16_, lean_object* v_f_17_, lean_object* v_g_18_, lean_object* v_inst_19_, lean_object* v_a_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_mathlib_DFinsupp_neLocus___redArg___lam__1(v___f_16_, v_f_17_, v_g_18_, v_inst_19_, v_a_20_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_neLocus___redArg___lam__2(lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_i_25_, lean_object* v_x_26_){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; uint8_t v___x_29_; 
lean_inc(v_i_25_);
v___x_27_ = lean_apply_1(v_inst_23_, v_i_25_);
v___x_28_ = lean_apply_3(v_inst_24_, v_i_25_, v_x_26_, v___x_27_);
v___x_29_ = lean_unbox(v___x_28_);
if (v___x_29_ == 0)
{
uint8_t v___x_30_; 
v___x_30_ = 1;
return v___x_30_;
}
else
{
uint8_t v___x_31_; 
v___x_31_ = 0;
return v___x_31_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg___lam__2___boxed(lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_i_34_, lean_object* v_x_35_){
_start:
{
uint8_t v_res_36_; lean_object* v_r_37_; 
v_res_36_ = lp_mathlib_DFinsupp_neLocus___redArg___lam__2(v_inst_32_, v_inst_33_, v_i_34_, v_x_35_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_f_42_, lean_object* v_g_43_){
_start:
{
lean_object* v___f_44_; lean_object* v___f_45_; lean_object* v___f_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___f_44_ = ((lean_object*)(lp_mathlib_DFinsupp_neLocus___redArg___closed__0));
lean_inc_ref(v_inst_40_);
lean_inc_ref(v_g_43_);
lean_inc_ref(v_f_42_);
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_neLocus___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_45_, 0, v___f_44_);
lean_closure_set(v___f_45_, 1, v_f_42_);
lean_closure_set(v___f_45_, 2, v_g_43_);
lean_closure_set(v___f_45_, 3, v_inst_40_);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_neLocus___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_46_, 0, v_inst_41_);
lean_closure_set(v___f_46_, 1, v_inst_40_);
lean_inc_ref(v___f_46_);
lean_inc_ref_n(v_inst_39_, 2);
v___x_47_ = lp_mathlib_DFinsupp_support___redArg(v_inst_39_, v___f_46_, v_f_42_);
v___x_48_ = lp_mathlib_DFinsupp_support___redArg(v_inst_39_, v___f_46_, v_g_43_);
v___x_49_ = lp_mathlib_Multiset_ndunion___redArg(v_inst_39_, v___x_47_, v___x_48_);
v___x_50_ = lp_mathlib_Multiset_filter___redArg(v___f_45_, v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_neLocus(lean_object* v_00_u03b1_51_, lean_object* v_N_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_f_56_, lean_object* v_g_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_DFinsupp_neLocus___redArg(v_inst_53_, v_inst_54_, v_inst_55_, v_f_56_, v_g_57_);
return v___x_58_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_NeLocus(builtin);
}
#ifdef __cplusplus
}
#endif
