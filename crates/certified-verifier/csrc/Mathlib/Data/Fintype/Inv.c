// Lean compiler output
// Module: Mathlib.Data.Fintype.Inv
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Basic public import Mathlib.Data.Fintype.Defs
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
lean_object* lp_mathlib_List_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOfMemRange___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOfMemRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_invOfMemRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_invOfMemRange___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_invOfMemRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_chooseX(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_choose___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bijInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bijInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0(lean_object* v_f_1_, lean_object* v_inst_2_, lean_object* v_b_3_, lean_object* v_a_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_5_ = lean_apply_1(v_f_1_, v_a_4_);
v___x_6_ = lean_apply_2(v_inst_2_, v___x_5_, v_b_3_);
v___x_7_ = lean_unbox(v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0___boxed(lean_object* v_f_8_, lean_object* v_inst_9_, lean_object* v_b_10_, lean_object* v_a_11_){
_start:
{
uint8_t v_res_12_; lean_object* v_r_13_; 
v_res_12_ = lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0(v_f_8_, v_inst_9_, v_b_10_, v_a_11_);
v_r_13_ = lean_box(v_res_12_);
return v_r_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOfMemRange___redArg(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_, lean_object* v_b_17_){
_start:
{
lean_object* v___f_18_; lean_object* v___x_19_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_18_, 0, v_f_16_);
lean_closure_set(v___f_18_, 1, v_inst_15_);
lean_closure_set(v___f_18_, 2, v_b_17_);
v___x_19_ = lp_mathlib_List_chooseX___redArg(v___f_18_, v_inst_14_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOfMemRange(lean_object* v_00_u03b1_20_, lean_object* v_00_u03b2_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_f_24_, lean_object* v_hf_25_, lean_object* v_b_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_Function_Injective_invOfMemRange___redArg(v_inst_22_, v_inst_23_, v_f_24_, v_b_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_invOfMemRange___redArg___lam__0(lean_object* v_f_28_, lean_object* v___y_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_apply_1(v_f_28_, v___y_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_invOfMemRange___redArg(lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_f_33_, lean_object* v_b_34_){
_start:
{
lean_object* v___f_35_; lean_object* v___x_36_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_invOfMemRange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_35_, 0, v_f_33_);
v___x_36_ = lp_mathlib_Function_Injective_invOfMemRange___redArg(v_inst_31_, v_inst_32_, v___f_35_, v_b_34_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_invOfMemRange(lean_object* v_00_u03b1_37_, lean_object* v_00_u03b2_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_f_41_, lean_object* v_b_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Function_Embedding_invOfMemRange___redArg(v_inst_39_, v_inst_40_, v_f_41_, v_b_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_chooseX___redArg(lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_List_chooseX___redArg(v_inst_45_, v_inst_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_chooseX(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_p_49_, lean_object* v_inst_50_, lean_object* v_hp_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_List_chooseX___redArg(v_inst_50_, v_inst_48_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_choose___redArg(lean_object* v_inst_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_List_chooseX___redArg(v_inst_54_, v_inst_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_choose(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_p_58_, lean_object* v_inst_59_, lean_object* v_hp_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_List_chooseX___redArg(v_inst_59_, v_inst_57_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bijInv___redArg(lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_f_64_, lean_object* v_b_65_){
_start:
{
lean_object* v___f_66_; lean_object* v___x_67_; 
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_invOfMemRange___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_66_, 0, v_f_64_);
lean_closure_set(v___f_66_, 1, v_inst_63_);
lean_closure_set(v___f_66_, 2, v_b_65_);
v___x_67_ = lp_mathlib_List_chooseX___redArg(v___f_66_, v_inst_62_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_bijInv(lean_object* v_00_u03b1_68_, lean_object* v_00_u03b2_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_f_72_, lean_object* v_f__bij_73_, lean_object* v_b_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Fintype_bijInv___redArg(v_inst_70_, v_inst_71_, v_f_72_, v_b_74_);
return v___x_75_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Inv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Inv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Inv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Inv(builtin);
}
#ifdef __cplusplus
}
#endif
