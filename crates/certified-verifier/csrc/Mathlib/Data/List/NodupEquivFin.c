// Lean compiler output
// Module: Mathlib.Data.List.NodupEquivFin
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Duplicate public import Mathlib.Data.List.Sort
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
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_idxOf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getEquivOfForallCountEqOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_getEquivOfForallCountEqOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_SortedLT_getIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_SortedLT_getIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_SortedLT_getIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0(lean_object* v_l_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = l_List_get___redArg(v_l_1_, v_i_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed(lean_object* v_l_4_, lean_object* v_i_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0(v_l_4_, v_i_5_);
lean_dec(v_l_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg(lean_object* v_l_7_){
_start:
{
lean_object* v___f_8_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_8_, 0, v_l_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList(lean_object* v_00_u03b1_9_, lean_object* v_l_10_, lean_object* v_nd_11_, lean_object* v_h_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_13_, 0, v_l_10_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquiv___redArg___lam__1(lean_object* v___f_14_, lean_object* v_l_15_, lean_object* v_x_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = l_List_idxOf___redArg(v___f_14_, v_x_16_, v_l_15_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquiv___redArg(lean_object* v_inst_18_, lean_object* v_l_19_){
_start:
{
lean_object* v___f_20_; lean_object* v___f_21_; lean_object* v___f_22_; lean_object* v___x_23_; 
lean_inc(v_l_19_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_20_, 0, v_l_19_);
v___f_21_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_21_, 0, v_inst_18_);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getEquiv___redArg___lam__1), 3, 2);
lean_closure_set(v___f_22_, 0, v___f_21_);
lean_closure_set(v___f_22_, 1, v_l_19_);
v___x_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_23_, 0, v___f_20_);
lean_ctor_set(v___x_23_, 1, v___f_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquiv(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_l_26_, lean_object* v_H_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_List_Nodup_getEquiv___redArg(v_inst_25_, v_l_26_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg___lam__1(lean_object* v___f_29_, lean_object* v_l_30_, lean_object* v_a_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = l_List_idxOf___redArg(v___f_29_, v_a_31_, v_l_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(lean_object* v_inst_33_, lean_object* v_l_34_){
_start:
{
lean_object* v___f_35_; lean_object* v___f_36_; lean_object* v___f_37_; lean_object* v___x_38_; 
lean_inc(v_l_34_);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_35_, 0, v_l_34_);
v___f_36_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_36_, 0, v_inst_33_);
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg___lam__1), 3, 2);
lean_closure_set(v___f_37_, 0, v___f_36_);
lean_closure_set(v___f_37_, 1, v_l_34_);
v___x_38_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_38_, 0, v___f_35_);
lean_ctor_set(v___x_38_, 1, v___f_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList(lean_object* v_00_u03b1_39_, lean_object* v_inst_40_, lean_object* v_l_41_, lean_object* v_nd_42_, lean_object* v_h_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(v_inst_40_, v_l_41_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getEquivOfForallCountEqOne___redArg(lean_object* v_inst_45_, lean_object* v_l_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(v_inst_45_, v_l_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_getEquivOfForallCountEqOne(lean_object* v_00_u03b1_48_, lean_object* v_inst_49_, lean_object* v_l_50_, lean_object* v_h_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(v_inst_49_, v_l_50_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_SortedLT_getIso___redArg(lean_object* v_inst_53_, lean_object* v_l_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_List_Nodup_getEquiv___redArg(v_inst_53_, v_l_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_SortedLT_getIso(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_l_59_, lean_object* v_H_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_List_Nodup_getEquiv___redArg(v_inst_58_, v_l_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_SortedLT_getIso___boxed(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_l_65_, lean_object* v_H_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_List_SortedLT_getIso(v_00_u03b1_62_, v_inst_63_, v_inst_64_, v_l_65_, v_H_66_);
lean_dec_ref(v_inst_63_);
return v_res_67_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_NodupEquivFin(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_NodupEquivFin(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_NodupEquivFin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_NodupEquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_NodupEquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_NodupEquivFin(builtin);
}
#ifdef __cplusplus
}
#endif
