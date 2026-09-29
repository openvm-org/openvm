// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Group.List.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Monoid public import Batteries.Data.List.Lemmas
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
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_x_4_){
_start:
{
if (lean_obj_tag(v_x_4_) == 0)
{
lean_dec(v_inst_3_);
lean_dec(v_inst_2_);
lean_inc(v_inst_1_);
return v_inst_1_;
}
else
{
lean_object* v_tail_5_; 
v_tail_5_ = lean_ctor_get(v_x_4_, 1);
if (lean_obj_tag(v_tail_5_) == 0)
{
lean_object* v_head_6_; 
lean_dec(v_inst_3_);
lean_dec(v_inst_2_);
v_head_6_ = lean_ctor_get(v_x_4_, 0);
lean_inc(v_head_6_);
lean_dec_ref_known(v_x_4_, 2);
return v_head_6_;
}
else
{
lean_object* v_head_7_; lean_object* v_head_8_; lean_object* v_tail_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
lean_inc_ref(v_tail_5_);
v_head_7_ = lean_ctor_get(v_x_4_, 0);
lean_inc(v_head_7_);
lean_dec_ref_known(v_x_4_, 2);
v_head_8_ = lean_ctor_get(v_tail_5_, 0);
lean_inc(v_head_8_);
v_tail_9_ = lean_ctor_get(v_tail_5_, 1);
lean_inc(v_tail_9_);
lean_dec_ref_known(v_tail_5_, 2);
lean_inc(v_inst_3_);
v___x_10_ = lean_apply_1(v_inst_3_, v_head_8_);
lean_inc_n(v_inst_2_, 2);
v___x_11_ = lean_apply_2(v_inst_2_, v_head_7_, v___x_10_);
v___x_12_ = lp_mathlib_List_alternatingSum___redArg(v_inst_1_, v_inst_2_, v_inst_3_, v_tail_9_);
v___x_13_ = lean_apply_2(v_inst_2_, v___x_11_, v___x_12_);
return v___x_13_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum___redArg___boxed(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_x_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_List_alternatingSum___redArg(v_inst_14_, v_inst_15_, v_inst_16_, v_x_17_);
lean_dec(v_inst_14_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum(lean_object* v_G_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_List_alternatingSum___redArg(v_inst_20_, v_inst_21_, v_inst_22_, v_x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingSum___boxed(lean_object* v_G_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_x_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_List_alternatingSum(v_G_25_, v_inst_26_, v_inst_27_, v_inst_28_, v_x_29_);
lean_dec(v_inst_26_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd___redArg(lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_x_34_){
_start:
{
if (lean_obj_tag(v_x_34_) == 0)
{
lean_dec(v_inst_33_);
lean_dec(v_inst_32_);
lean_inc(v_inst_31_);
return v_inst_31_;
}
else
{
lean_object* v_tail_35_; 
v_tail_35_ = lean_ctor_get(v_x_34_, 1);
if (lean_obj_tag(v_tail_35_) == 0)
{
lean_object* v_head_36_; 
lean_dec(v_inst_33_);
lean_dec(v_inst_32_);
v_head_36_ = lean_ctor_get(v_x_34_, 0);
lean_inc(v_head_36_);
lean_dec_ref_known(v_x_34_, 2);
return v_head_36_;
}
else
{
lean_object* v_head_37_; lean_object* v_head_38_; lean_object* v_tail_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
lean_inc_ref(v_tail_35_);
v_head_37_ = lean_ctor_get(v_x_34_, 0);
lean_inc(v_head_37_);
lean_dec_ref_known(v_x_34_, 2);
v_head_38_ = lean_ctor_get(v_tail_35_, 0);
lean_inc(v_head_38_);
v_tail_39_ = lean_ctor_get(v_tail_35_, 1);
lean_inc(v_tail_39_);
lean_dec_ref_known(v_tail_35_, 2);
lean_inc(v_inst_33_);
v___x_40_ = lean_apply_1(v_inst_33_, v_head_38_);
lean_inc_n(v_inst_32_, 2);
v___x_41_ = lean_apply_2(v_inst_32_, v_head_37_, v___x_40_);
v___x_42_ = lp_mathlib_List_alternatingProd___redArg(v_inst_31_, v_inst_32_, v_inst_33_, v_tail_39_);
v___x_43_ = lean_apply_2(v_inst_32_, v___x_41_, v___x_42_);
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd___redArg___boxed(lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_x_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_List_alternatingProd___redArg(v_inst_44_, v_inst_45_, v_inst_46_, v_x_47_);
lean_dec(v_inst_44_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd(lean_object* v_G_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_x_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_List_alternatingProd___redArg(v_inst_50_, v_inst_51_, v_inst_52_, v_x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_alternatingProd___boxed(lean_object* v_G_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_x_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_List_alternatingProd(v_G_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_x_59_);
lean_dec(v_inst_56_);
return v_res_60_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
