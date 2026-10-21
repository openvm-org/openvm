// Lean compiler output
// Module: Mathlib.Data.Multiset.AddSub
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.Count public import Mathlib.Data.List.Count
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
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_List_diff___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_instAdd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_add, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_instAdd___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instAdd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAdd(lean_object*);
static const lean_array_object lp_mathlib_Multiset_erase___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Multiset_erase___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_erase___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add___redArg(lean_object* v_s_u2081_1_, lean_object* v_s_u2082_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = l_List_appendTR___redArg(v_s_u2081_1_, v_s_u2082_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_add(lean_object* v_00_u03b1_4_, lean_object* v_s_u2081_5_, lean_object* v_s_u2082_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_List_appendTR___redArg(v_s_u2081_5_, v_s_u2082_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instAdd(lean_object* v_00_u03b1_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = ((lean_object*)(lp_mathlib_Multiset_instAdd___closed__0));
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase___redArg(lean_object* v_inst_13_, lean_object* v_s_14_, lean_object* v_a_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___f_16_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_16_, 0, v_inst_13_);
v___x_17_ = ((lean_object*)(lp_mathlib_Multiset_erase___redArg___closed__0));
lean_inc(v_s_14_);
v___x_18_ = l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_box(0), v___f_16_, v_s_14_, v_a_15_, v_s_14_, v___x_17_);
lean_dec(v_s_14_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_erase(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_s_21_, lean_object* v_a_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_Multiset_erase___redArg(v_inst_20_, v_s_21_, v_a_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub___redArg(lean_object* v_inst_24_, lean_object* v_s_25_, lean_object* v_t_26_){
_start:
{
lean_object* v___f_27_; lean_object* v___x_28_; 
v___f_27_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_27_, 0, v_inst_24_);
v___x_28_ = lp_batteries_List_diff___redArg(v___f_27_, v_s_25_, v_t_26_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sub(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_s_31_, lean_object* v_t_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Multiset_sub___redArg(v_inst_30_, v_s_31_, v_t_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub___redArg(lean_object* v_inst_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sub), 4, 2);
lean_closure_set(v___x_35_, 0, lean_box(0));
lean_closure_set(v___x_35_, 1, v_inst_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSub(lean_object* v_00_u03b1_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sub), 4, 2);
lean_closure_set(v___x_38_, 0, lean_box(0));
lean_closure_set(v___x_38_, 1, v_inst_37_);
return v___x_38_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Count(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_AddSub(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_AddSub(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Count(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_AddSub(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_AddSub(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_AddSub(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_AddSub(builtin);
}
#ifdef __cplusplus
}
#endif
