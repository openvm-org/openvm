// Lean compiler output
// Module: Mathlib.RingTheory.Localization.FractionRing
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Equiv public import Mathlib.Algebra.Field.Subfield.Basic public import Mathlib.Algebra.Order.GroupWithZero.Submonoid public import Mathlib.Algebra.Order.Ring.Int public import Mathlib.Algebra.Ring.CompTypeclasses public import Mathlib.GroupTheory.GroupAction.FixingSubgroup public import Mathlib.RingTheory.Localization.Basic public import Mathlib.RingTheory.SimpleRing.Basic
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
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Localization_instUniqueLocalization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_FractionRing_instDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_instDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_FractionRing_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_FractionRing_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toSemiring_2_; lean_object* v___x_3_; 
v_toSemiring_2_ = lean_ctor_get(v_inst_1_, 0);
v___x_3_ = lp_mathlib_Localization_instUniqueLocalization___redArg(v_toSemiring_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg___boxed(lean_object* v_inst_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_FractionRing_unique___redArg(v_inst_4_);
lean_dec_ref(v_inst_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique(lean_object* v_R_6_, lean_object* v_inst_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_FractionRing_unique___redArg(v_inst_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___boxed(lean_object* v_R_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_FractionRing_unique(v_R_10_, v_inst_11_, v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_13_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_FractionRing_instDecidableEq___redArg___lam__0(lean_object* v_toMul_14_, lean_object* v_inst_15_, lean_object* v_a_16_, lean_object* v_c_17_, lean_object* v_b_18_, lean_object* v_d_19_){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; uint8_t v___x_23_; 
lean_inc(v_toMul_14_);
v___x_20_ = lean_apply_2(v_toMul_14_, v_a_16_, v_d_19_);
v___x_21_ = lean_apply_2(v_toMul_14_, v_b_18_, v_c_17_);
v___x_22_ = lean_apply_2(v_inst_15_, v___x_20_, v___x_21_);
v___x_23_ = lean_unbox(v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_instDecidableEq___redArg___lam__0___boxed(lean_object* v_toMul_24_, lean_object* v_inst_25_, lean_object* v_a_26_, lean_object* v_c_27_, lean_object* v_b_28_, lean_object* v_d_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_FractionRing_instDecidableEq___redArg___lam__0(v_toMul_24_, v_inst_25_, v_a_26_, v_c_27_, v_b_28_, v_d_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_FractionRing_instDecidableEq___redArg(lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_x_34_, lean_object* v_y_35_){
_start:
{
lean_object* v_toSemiring_36_; lean_object* v___x_37_; lean_object* v_toMul_38_; lean_object* v___f_39_; lean_object* v___x_40_; uint8_t v___x_41_; 
v_toSemiring_36_ = lean_ctor_get(v_inst_32_, 0);
lean_inc_ref(v_toSemiring_36_);
lean_dec_ref(v_inst_32_);
v___x_37_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_36_);
v_toMul_38_ = lean_ctor_get(v___x_37_, 0);
lean_inc(v_toMul_38_);
lean_dec_ref(v___x_37_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_FractionRing_instDecidableEq___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_39_, 0, v_toMul_38_);
lean_closure_set(v___f_39_, 1, v_inst_33_);
v___x_40_ = lp_mathlib_Localization_recOnSubsingleton_u2082___redArg(v_x_34_, v_y_35_, v___f_39_);
v___x_41_ = lean_unbox(v___x_40_);
lean_dec(v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_instDecidableEq___redArg___boxed(lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_x_44_, lean_object* v_y_45_){
_start:
{
uint8_t v_res_46_; lean_object* v_r_47_; 
v_res_46_ = lp_mathlib_FractionRing_instDecidableEq___redArg(v_inst_42_, v_inst_43_, v_x_44_, v_y_45_);
v_r_47_ = lean_box(v_res_46_);
return v_r_47_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_FractionRing_instDecidableEq(lean_object* v_R_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_x_51_, lean_object* v_y_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lp_mathlib_FractionRing_instDecidableEq___redArg(v_inst_49_, v_inst_50_, v_x_51_, v_y_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_instDecidableEq___boxed(lean_object* v_R_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_x_57_, lean_object* v_y_58_){
_start:
{
uint8_t v_res_59_; lean_object* v_r_60_; 
v_res_59_ = lp_mathlib_FractionRing_instDecidableEq(v_R_54_, v_inst_55_, v_inst_56_, v_x_57_, v_y_58_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Submonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_FixingSubgroup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_FixingSubgroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Submonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_FixingSubgroup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_FixingSubgroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
}
#ifdef __cplusplus
}
#endif
