// Lean compiler output
// Module: Mathlib.Data.Finset.Disjoint
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Insert
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
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDisjoint___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDisjoint___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDisjoint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDisjoint___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDisjoint(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDisjoint___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjUnion___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjUnion(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDisjoint___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_V_2_, lean_object* v_a_3_, lean_object* v_h_4_){
_start:
{
uint8_t v___x_5_; 
v___x_5_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_1_, v_a_3_, v_V_2_);
if (v___x_5_ == 0)
{
uint8_t v___x_6_; 
v___x_6_ = 1;
return v___x_6_;
}
else
{
uint8_t v___x_7_; 
v___x_7_ = 0;
return v___x_7_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDisjoint___redArg___lam__0___boxed(lean_object* v_inst_8_, lean_object* v_V_9_, lean_object* v_a_10_, lean_object* v_h_11_){
_start:
{
uint8_t v_res_12_; lean_object* v_r_13_; 
v_res_12_ = lp_mathlib_Finset_decidableDisjoint___redArg___lam__0(v_inst_8_, v_V_9_, v_a_10_, v_h_11_);
v_r_13_ = lean_box(v_res_12_);
return v_r_13_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDisjoint___redArg(lean_object* v_inst_14_, lean_object* v_U_15_, lean_object* v_V_16_){
_start:
{
lean_object* v___f_17_; uint8_t v___x_18_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableDisjoint___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_17_, 0, v_inst_14_);
lean_closure_set(v___f_17_, 1, v_V_16_);
v___x_18_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_U_15_, v___f_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDisjoint___redArg___boxed(lean_object* v_inst_19_, lean_object* v_U_20_, lean_object* v_V_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mathlib_Finset_decidableDisjoint___redArg(v_inst_19_, v_U_20_, v_V_21_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDisjoint(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_U_26_, lean_object* v_V_27_){
_start:
{
uint8_t v___x_28_; 
v___x_28_ = lp_mathlib_Finset_decidableDisjoint___redArg(v_inst_25_, v_U_26_, v_V_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDisjoint___boxed(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_U_31_, lean_object* v_V_32_){
_start:
{
uint8_t v_res_33_; lean_object* v_r_34_; 
v_res_33_ = lp_mathlib_Finset_decidableDisjoint(v_00_u03b1_29_, v_inst_30_, v_U_31_, v_V_32_);
v_r_34_ = lean_box(v_res_33_);
return v_r_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjUnion___redArg(lean_object* v_s_35_, lean_object* v_t_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = l_List_appendTR___redArg(v_s_35_, v_t_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjUnion(lean_object* v_00_u03b1_38_, lean_object* v_s_39_, lean_object* v_t_40_, lean_object* v_h_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = l_List_appendTR___redArg(v_s_39_, v_t_40_);
return v___x_42_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Insert(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Disjoint(builtin);
}
#ifdef __cplusplus
}
#endif
