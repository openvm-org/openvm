// Lean compiler output
// Module: Mathlib.Data.Multiset.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.ZeroCons
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
lean_object* lp_mathlib_Multiset_ofList___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_List_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongInductionOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongInductionOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongInductionOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInductionOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInductionOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInductionOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_chooseX(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_choose___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_subsingletonEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_subsingletonEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Multiset_subsingletonEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_subsingletonEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_subsingletonEquiv___closed__0 = (const lean_object*)&lp_mathlib_Multiset_subsingletonEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Multiset_subsingletonEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_ofList___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_subsingletonEquiv___closed__1 = (const lean_object*)&lp_mathlib_Multiset_subsingletonEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Multiset_subsingletonEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_subsingletonEquiv___closed__1_value),((lean_object*)&lp_mathlib_Multiset_subsingletonEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Multiset_subsingletonEquiv___closed__2 = (const lean_object*)&lp_mathlib_Multiset_subsingletonEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_subsingletonEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongInductionOn___redArg(lean_object* v_s_1_, lean_object* v_ih_2_){
_start:
{
lean_object* v___f_3_; lean_object* v___x_4_; 
lean_inc(v_ih_2_);
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_strongInductionOn___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3_, 0, v_ih_2_);
v___x_4_ = lean_apply_2(v_ih_2_, v_s_1_, v___f_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongInductionOn___redArg___lam__0(lean_object* v_ih_5_, lean_object* v_t_6_, lean_object* v___h_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiset_strongInductionOn___redArg(v_t_6_, v_ih_5_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongInductionOn(lean_object* v_00_u03b1_9_, lean_object* v_p_10_, lean_object* v_s_11_, lean_object* v_ih_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Multiset_strongInductionOn___redArg(v_s_11_, v_ih_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction___redArg(lean_object* v_H_14_, lean_object* v_s_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___x_17_; 
lean_inc(v_H_14_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_strongDownwardInduction___redArg___lam__0), 4, 1);
lean_closure_set(v___f_16_, 0, v_H_14_);
v___x_17_ = lean_apply_3(v_H_14_, v_s_15_, v___f_16_, lean_box(0));
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction___redArg___lam__0(lean_object* v_H_18_, lean_object* v_t_19_, lean_object* v_ht_20_, lean_object* v___h_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Multiset_strongDownwardInduction___redArg(v_H_18_, v_t_19_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction(lean_object* v_00_u03b1_23_, lean_object* v_p_24_, lean_object* v_n_25_, lean_object* v_H_26_, lean_object* v_s_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Multiset_strongDownwardInduction___redArg(v_H_26_, v_s_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInduction___boxed(lean_object* v_00_u03b1_30_, lean_object* v_p_31_, lean_object* v_n_32_, lean_object* v_H_33_, lean_object* v_s_34_, lean_object* v_a_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Multiset_strongDownwardInduction(v_00_u03b1_30_, v_p_31_, v_n_32_, v_H_33_, v_s_34_, v_a_35_);
lean_dec(v_n_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInductionOn___redArg(lean_object* v_s_37_, lean_object* v_H_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Multiset_strongDownwardInduction___redArg(v_H_38_, v_s_37_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInductionOn(lean_object* v_00_u03b1_40_, lean_object* v_p_41_, lean_object* v_n_42_, lean_object* v_s_43_, lean_object* v_H_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_Multiset_strongDownwardInduction___redArg(v_H_44_, v_s_43_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_strongDownwardInductionOn___boxed(lean_object* v_00_u03b1_47_, lean_object* v_p_48_, lean_object* v_n_49_, lean_object* v_s_50_, lean_object* v_H_51_, lean_object* v_a_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Multiset_strongDownwardInductionOn(v_00_u03b1_47_, v_p_48_, v_n_49_, v_s_50_, v_H_51_, v_a_52_);
lean_dec(v_n_49_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_chooseX___redArg(lean_object* v_inst_54_, lean_object* v_l_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_List_chooseX___redArg(v_inst_54_, v_l_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_chooseX(lean_object* v_00_u03b1_57_, lean_object* v_p_58_, lean_object* v_inst_59_, lean_object* v_l_60_, lean_object* v___hp_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_List_chooseX___redArg(v_inst_59_, v_l_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_choose___redArg(lean_object* v_inst_63_, lean_object* v_l_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_List_chooseX___redArg(v_inst_63_, v_l_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_choose(lean_object* v_00_u03b1_66_, lean_object* v_p_67_, lean_object* v_inst_68_, lean_object* v_l_69_, lean_object* v_hp_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_List_chooseX___redArg(v_inst_68_, v_l_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_subsingletonEquiv___lam__0(lean_object* v_a_72_){
_start:
{
lean_inc(v_a_72_);
return v_a_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_subsingletonEquiv___lam__0___boxed(lean_object* v_a_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Multiset_subsingletonEquiv___lam__0(v_a_73_);
lean_dec(v_a_73_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_subsingletonEquiv(lean_object* v_00_u03b1_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = ((lean_object*)(lp_mathlib_Multiset_subsingletonEquiv___closed__2));
return v___x_82_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
