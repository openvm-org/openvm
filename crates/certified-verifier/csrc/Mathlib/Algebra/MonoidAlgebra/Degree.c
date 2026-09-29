// Lean compiler output
// Module: Mathlib.Algebra.MonoidAlgebra.Degree
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subsemigroup.Operations public import Mathlib.Algebra.MonoidAlgebra.Support public import Mathlib.Order.Filter.Extr
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
lean_object* lp_mathlib_Finset_inf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_supDegree___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_supDegree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_supDegree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_infDegree___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_infDegree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_infDegree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_supDegree___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_D_3_, lean_object* v_f_4_){
_start:
{
lean_object* v_support_5_; lean_object* v___x_6_; 
v_support_5_ = lean_ctor_get(v_f_4_, 0);
lean_inc(v_support_5_);
lean_dec_ref(v_f_4_);
v___x_6_ = lp_mathlib_Finset_sup___redArg(v_inst_1_, v_inst_2_, v_support_5_, v_D_3_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_supDegree(lean_object* v_R_7_, lean_object* v_A_8_, lean_object* v_B_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_D_13_, lean_object* v_f_14_){
_start:
{
lean_object* v_support_15_; lean_object* v___x_16_; 
v_support_15_ = lean_ctor_get(v_f_14_, 0);
lean_inc(v_support_15_);
lean_dec_ref(v_f_14_);
v___x_16_ = lp_mathlib_Finset_sup___redArg(v_inst_11_, v_inst_12_, v_support_15_, v_D_13_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_supDegree___boxed(lean_object* v_R_17_, lean_object* v_A_18_, lean_object* v_B_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_D_23_, lean_object* v_f_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_AddMonoidAlgebra_supDegree(v_R_17_, v_A_18_, v_B_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_D_23_, v_f_24_);
lean_dec_ref(v_inst_20_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_infDegree___redArg(lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_D_28_, lean_object* v_f_29_){
_start:
{
lean_object* v_support_30_; lean_object* v___x_31_; 
v_support_30_ = lean_ctor_get(v_f_29_, 0);
lean_inc(v_support_30_);
lean_dec_ref(v_f_29_);
v___x_31_ = lp_mathlib_Finset_inf___redArg(v_inst_26_, v_inst_27_, v_support_30_, v_D_28_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_infDegree(lean_object* v_R_32_, lean_object* v_A_33_, lean_object* v_T_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_D_38_, lean_object* v_f_39_){
_start:
{
lean_object* v_support_40_; lean_object* v___x_41_; 
v_support_40_ = lean_ctor_get(v_f_39_, 0);
lean_inc(v_support_40_);
lean_dec_ref(v_f_39_);
v___x_41_ = lp_mathlib_Finset_inf___redArg(v_inst_36_, v_inst_37_, v_support_40_, v_D_38_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_infDegree___boxed(lean_object* v_R_42_, lean_object* v_A_43_, lean_object* v_T_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_D_48_, lean_object* v_f_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_AddMonoidAlgebra_infDegree(v_R_42_, v_A_43_, v_T_44_, v_inst_45_, v_inst_46_, v_inst_47_, v_D_48_, v_f_49_);
lean_dec_ref(v_inst_45_);
return v_res_50_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Extr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Extr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Extr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Extr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Degree(builtin);
}
#ifdef __cplusplus
}
#endif
