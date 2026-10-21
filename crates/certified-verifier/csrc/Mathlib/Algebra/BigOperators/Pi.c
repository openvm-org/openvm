// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Pi
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Lemmas public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise public import Mathlib.Algebra.BigOperators.GroupWithZero.Finset public import Mathlib.Algebra.Group.Action.Pi public import Mathlib.Algebra.Notation.Indicator public import Mathlib.Algebra.Ring.Pi public import Mathlib.Data.Fintype.Basic public import Mathlib.Data.FunLike.IsApply
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_evalMulHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_single___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_instCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Finset_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_mulSingle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_i_2_);
v___x_4_ = lp_mathlib_Monoid_toMulOneClass___redArg(v___x_3_);
lean_dec_ref(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1(lean_object* v_00_u03c6_5_, lean_object* v_i_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_8_; lean_object* v___f_9_; lean_object* v___x_10_; 
lean_inc(v_i_6_);
v___x_8_ = lean_apply_1(v_00_u03c6_5_, v_i_6_);
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_9_, 0, v_i_6_);
v___x_10_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_9_, v___x_8_, v___y_7_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__2(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_00_u03c6_13_, lean_object* v___y_14_){
_start:
{
lean_object* v___f_15_; lean_object* v___x_16_; lean_object* v___x_89__overap_17_; lean_object* v___x_18_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1), 3, 1);
lean_closure_set(v___f_15_, 0, v_00_u03c6_13_);
v___x_16_ = lp_mathlib_MonoidHom_instCommMonoid___redArg(v_inst_11_);
v___x_89__overap_17_ = lp_mathlib_Finset_prod___redArg(v___x_16_, v_inst_12_, v___f_15_);
lean_dec_ref(v___x_16_);
v___x_18_ = lean_apply_1(v___x_89__overap_17_, v___y_14_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3(lean_object* v_inst_19_, lean_object* v___f_20_, lean_object* v_00_u03c6_21_, lean_object* v_i_22_, lean_object* v___y_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lp_mathlib_MonoidHom_mulSingle___redArg(v_inst_19_, v___f_20_, v_i_22_);
v___x_25_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_24_, v_00_u03c6_21_, v___y_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv___redArg(lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; lean_object* v___f_31_; lean_object* v___f_32_; lean_object* v___x_33_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_30_, 0, v_inst_28_);
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__2), 4, 2);
lean_closure_set(v___f_31_, 0, v_inst_29_);
lean_closure_set(v___f_31_, 1, v_inst_26_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__3), 5, 2);
lean_closure_set(v___f_32_, 0, v_inst_27_);
lean_closure_set(v___f_32_, 1, v___f_30_);
v___x_33_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_33_, 0, v___f_32_);
lean_ctor_set(v___x_33_, 1, v___f_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHomMulEquiv(lean_object* v_00_u03b9_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_M_37_, lean_object* v_inst_38_, lean_object* v_M_x27_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Pi_monoidHomMulEquiv___redArg(v_inst_35_, v_inst_36_, v_inst_38_, v_inst_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__0(lean_object* v_inst_42_, lean_object* v_i_43_){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_44_ = lean_apply_1(v_inst_42_, v_i_43_);
v___x_45_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_44_);
lean_dec_ref(v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__2(lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_00_u03c6_48_, lean_object* v___y_49_){
_start:
{
lean_object* v___f_50_; lean_object* v___x_51_; lean_object* v___x_89__overap_52_; lean_object* v___x_53_; 
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidHomMulEquiv___redArg___lam__1), 3, 1);
lean_closure_set(v___f_50_, 0, v_00_u03c6_48_);
v___x_51_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v_inst_46_);
v___x_89__overap_52_ = lp_mathlib_Finset_sum___redArg(v___x_51_, v_inst_47_, v___f_50_);
lean_dec_ref(v___x_51_);
v___x_53_ = lean_apply_1(v___x_89__overap_52_, v___y_49_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__1(lean_object* v_inst_54_, lean_object* v___f_55_, lean_object* v_00_u03c6_56_, lean_object* v_i_57_, lean_object* v___y_58_){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lp_mathlib_AddMonoidHom_single___redArg(v_inst_54_, v___f_55_, v_i_57_);
v___x_60_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_59_, v_00_u03c6_56_, v___y_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv___redArg(lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___f_65_; lean_object* v___f_66_; lean_object* v___f_67_; lean_object* v___x_68_; 
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_65_, 0, v_inst_63_);
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__2), 4, 2);
lean_closure_set(v___f_66_, 0, v_inst_64_);
lean_closure_set(v___f_66_, 1, v_inst_61_);
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoidHomAddEquiv___redArg___lam__1), 5, 2);
lean_closure_set(v___f_67_, 0, v_inst_62_);
lean_closure_set(v___f_67_, 1, v___f_65_);
v___x_68_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_68_, 0, v___f_67_);
lean_ctor_set(v___x_68_, 1, v___f_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHomAddEquiv(lean_object* v_00_u03b9_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_M_72_, lean_object* v_inst_73_, lean_object* v_M_x27_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_Pi_addMonoidHomAddEquiv___redArg(v_inst_70_, v_inst_71_, v_inst_73_, v_inst_75_);
return v___x_76_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Indicator(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_FunLike_IsApply(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_FunLike_IsApply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Indicator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_FunLike_IsApply(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_FunLike_IsApply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
