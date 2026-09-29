// Lean compiler output
// Module: Mathlib.Data.Multiset.Bind
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Multiset.Basic public import Mathlib.Data.Multiset.Fold
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
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_appendTR___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_instSProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_product, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_instSProd___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instSProd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(lean_object* v_s_2_){
_start:
{
lean_object* v___f_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___f_3_ = ((lean_object*)(lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg___closed__0));
v___x_4_ = lean_box(0);
v___x_5_ = l_List_foldrTR___redArg(v___f_3_, v___x_4_, v_s_2_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0(lean_object* v_00_u03b1_6_, lean_object* v_s_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(v_s_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join___redArg(lean_object* v_a_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(v_a_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_join(lean_object* v_00_u03b1_11_, lean_object* v_a_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(v_a_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind___redArg(lean_object* v_s_14_, lean_object* v_f_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_16_ = lp_mathlib_Multiset_map___redArg(v_f_15_, v_s_14_);
v___x_17_ = lp_mathlib_Multiset_sum___at___00Multiset_join_spec__0___redArg(v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_bind(lean_object* v_00_u03b1_18_, lean_object* v_00_u03b2_19_, lean_object* v_s_20_, lean_object* v_f_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Multiset_bind___redArg(v_s_20_, v_f_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__0(lean_object* v_a_23_, lean_object* v_snd_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_25_, 0, v_a_23_);
lean_ctor_set(v___x_25_, 1, v_snd_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg___lam__1(lean_object* v_t_26_, lean_object* v_a_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_product___redArg___lam__0), 2, 1);
lean_closure_set(v___f_28_, 0, v_a_27_);
v___x_29_ = lp_mathlib_Multiset_map___redArg(v___f_28_, v_t_26_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product___redArg(lean_object* v_s_30_, lean_object* v_t_31_){
_start:
{
lean_object* v___f_32_; lean_object* v___x_33_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_product___redArg___lam__1), 2, 1);
lean_closure_set(v___f_32_, 0, v_t_31_);
v___x_33_ = lp_mathlib_Multiset_bind___redArg(v_s_30_, v___f_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_product(lean_object* v_00_u03b1_34_, lean_object* v_00_u03b2_35_, lean_object* v_s_36_, lean_object* v_t_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Multiset_product___redArg(v_s_36_, v_t_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSProd(lean_object* v_00_u03b1_40_, lean_object* v_00_u03b2_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = ((lean_object*)(lp_mathlib_Multiset_instSProd___closed__0));
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__0(lean_object* v_a_43_, lean_object* v_snd_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v_a_43_);
lean_ctor_set(v___x_45_, 1, v_snd_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg___lam__1(lean_object* v_t_46_, lean_object* v_a_47_){
_start:
{
lean_object* v___f_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
lean_inc(v_a_47_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sigma___redArg___lam__0), 2, 1);
lean_closure_set(v___f_48_, 0, v_a_47_);
v___x_49_ = lean_apply_1(v_t_46_, v_a_47_);
v___x_50_ = lp_mathlib_Multiset_map___redArg(v___f_48_, v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma___redArg(lean_object* v_s_51_, lean_object* v_t_52_){
_start:
{
lean_object* v___f_53_; lean_object* v___x_54_; 
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sigma___redArg___lam__1), 2, 1);
lean_closure_set(v___f_53_, 0, v_t_52_);
v___x_54_ = lp_mathlib_Multiset_bind___redArg(v_s_51_, v___f_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sigma(lean_object* v_00_u03b1_55_, lean_object* v_00_u03c3_56_, lean_object* v_s_57_, lean_object* v_t_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Multiset_sigma___redArg(v_s_57_, v_t_58_);
return v___x_59_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
}
#ifdef __cplusplus
}
#endif
