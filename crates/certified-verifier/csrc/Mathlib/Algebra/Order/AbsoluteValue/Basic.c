// Lean compiler output
// Module: Mathlib.Algebra.Order.AbsoluteValue.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Regular public import Mathlib.Algebra.GroupWithZero.Units.Lemmas public import Mathlib.Algebra.Order.Hom.Basic public import Mathlib.Algebra.Order.Ring.Abs public import Mathlib.Tactic.Positivity.Core
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
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
uint8_t l_Lean_Expr_isFVar(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_abs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "IsAbsoluteValue"};
static const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__0 = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "abv_nonneg"};
static const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__1 = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(145, 27, 61, 149, 64, 200, 31, 171)}};
static const lean_ctor_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(26, 205, 224, 209, 53, 159, 183, 39)}};
static const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__2 = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "abv: function is not a variable"};
static const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__3 = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__4;
static const lean_string_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 9, .m_data = "not abv ·"};
static const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__5 = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___closed__0 = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv = (const lean_object*)&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_Simps_apply___redArg(lean_object* v_f_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_1_, v_a_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_Simps_apply(lean_object* v_R_4_, lean_object* v_S_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_f_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_apply_1(v_f_9_, v_a_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_Simps_apply___boxed(lean_object* v_R_12_, lean_object* v_S_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_, lean_object* v_a_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_AbsoluteValue_Simps_apply(v_R_12_, v_S_13_, v_inst_14_, v_inst_15_, v_inst_16_, v_f_17_, v_a_18_);
lean_dec_ref(v_inst_16_);
lean_dec_ref(v_inst_15_);
lean_dec_ref(v_inst_14_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0(lean_object* v_abv_20_, lean_object* v___y_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_apply_1(v_abv_20_, v___y_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg(lean_object* v_abv_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_24_, 0, v_abv_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom(lean_object* v_R_25_, lean_object* v_S_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_abv_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_33_, 0, v_abv_30_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___boxed(lean_object* v_R_34_, lean_object* v_S_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_abv_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_AbsoluteValue_toMonoidWithZeroHom(v_R_34_, v_S_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_abv_39_, v_inst_40_, v_inst_41_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
lean_dec_ref(v_inst_36_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidHom___redArg(lean_object* v_abv_43_){
_start:
{
lean_object* v___f_44_; 
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_44_, 0, v_abv_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidHom(lean_object* v_R_45_, lean_object* v_S_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_abv_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___f_53_; 
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_53_, 0, v_abv_50_);
return v___f_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_toMonoidHom___boxed(lean_object* v_R_54_, lean_object* v_S_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_abv_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_AbsoluteValue_toMonoidHom(v_R_54_, v_S_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_abv_59_, v_inst_60_, v_inst_61_);
lean_dec_ref(v_inst_58_);
lean_dec_ref(v_inst_57_);
lean_dec_ref(v_inst_56_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs___redArg(lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_65_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_64_);
v___x_66_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_63_);
v___x_67_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_66_);
lean_dec_ref(v___x_66_);
v___x_68_ = lean_alloc_closure((void*)(lp_mathlib_abs___boxed), 4, 3);
lean_closure_set(v___x_68_, 0, lean_box(0));
lean_closure_set(v___x_68_, 1, v___x_65_);
lean_closure_set(v___x_68_, 2, v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs___redArg___boxed(lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_AbsoluteValue_abs___redArg(v_inst_69_, v_inst_70_);
lean_dec_ref(v_inst_70_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs(lean_object* v_S_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_AbsoluteValue_abs___redArg(v_inst_73_, v_inst_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_abs___boxed(lean_object* v_S_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_AbsoluteValue_abs(v_S_77_, v_inst_78_, v_inst_79_, v_inst_80_);
lean_dec_ref(v_inst_79_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited___redArg(lean_object* v_inst_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_AbsoluteValue_abs___redArg(v_inst_82_, v_inst_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited___redArg___boxed(lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_AbsoluteValue_instInhabited___redArg(v_inst_85_, v_inst_86_);
lean_dec_ref(v_inst_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited(lean_object* v_S_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_AbsoluteValue_abs___redArg(v_inst_89_, v_inst_90_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_instInhabited___boxed(lean_object* v_S_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_AbsoluteValue_instInhabited(v_S_93_, v_inst_94_, v_inst_95_, v_inst_96_);
lean_dec_ref(v_inst_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial___redArg___lam__0(lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_x_100_){
_start:
{
lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_101_ = lean_apply_1(v_inst_98_, v_x_100_);
v___x_102_ = lean_unbox(v___x_101_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v_toOne_105_; 
v___x_103_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_99_);
v___x_104_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_103_);
v_toOne_105_ = lean_ctor_get(v___x_104_, 2);
lean_inc(v_toOne_105_);
lean_dec_ref(v___x_104_);
return v_toOne_105_;
}
else
{
lean_object* v___x_106_; lean_object* v_toZero_107_; 
v___x_106_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_99_);
v_toZero_107_ = lean_ctor_get(v___x_106_, 1);
lean_inc(v_toZero_107_);
lean_dec_ref(v___x_106_);
return v_toZero_107_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial___redArg(lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___f_110_; 
v___f_110_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_trivial___redArg___lam__0), 3, 2);
lean_closure_set(v___f_110_, 0, v_inst_108_);
lean_closure_set(v___f_110_, 1, v_inst_109_);
return v___f_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial(lean_object* v_R_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_S_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___f_120_; 
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_trivial___redArg___lam__0), 3, 2);
lean_closure_set(v___f_120_, 0, v_inst_113_);
lean_closure_set(v___f_120_, 1, v_inst_116_);
return v___f_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AbsoluteValue_trivial___boxed(lean_object* v_R_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_S_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_AbsoluteValue_trivial(v_R_121_, v_inst_122_, v_inst_123_, v_inst_124_, v_S_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_122_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0_spec__0(lean_object* v_msgData_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_){
_start:
{
lean_object* v___x_137_; lean_object* v_env_138_; lean_object* v___x_139_; lean_object* v_mctx_140_; lean_object* v_lctx_141_; lean_object* v_options_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_137_ = lean_st_ref_get(v___y_135_);
v_env_138_ = lean_ctor_get(v___x_137_, 0);
lean_inc_ref(v_env_138_);
lean_dec(v___x_137_);
v___x_139_ = lean_st_ref_get(v___y_133_);
v_mctx_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc_ref(v_mctx_140_);
lean_dec(v___x_139_);
v_lctx_141_ = lean_ctor_get(v___y_132_, 2);
v_options_142_ = lean_ctor_get(v___y_134_, 2);
lean_inc_ref(v_options_142_);
lean_inc_ref(v_lctx_141_);
v___x_143_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_143_, 0, v_env_138_);
lean_ctor_set(v___x_143_, 1, v_mctx_140_);
lean_ctor_set(v___x_143_, 2, v_lctx_141_);
lean_ctor_set(v___x_143_, 3, v_options_142_);
v___x_144_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set(v___x_144_, 1, v_msgData_131_);
v___x_145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0_spec__0___boxed(lean_object* v_msgData_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0_spec__0(v_msgData_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg(lean_object* v_msg_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v_ref_159_; lean_object* v___x_160_; lean_object* v_a_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_169_; 
v_ref_159_ = lean_ctor_get(v___y_156_, 5);
v___x_160_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0_spec__0(v_msg_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
v_a_161_ = lean_ctor_get(v___x_160_, 0);
v_isSharedCheck_169_ = !lean_is_exclusive(v___x_160_);
if (v_isSharedCheck_169_ == 0)
{
v___x_163_ = v___x_160_;
v_isShared_164_ = v_isSharedCheck_169_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_a_161_);
lean_dec(v___x_160_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_169_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v___x_165_; lean_object* v___x_167_; 
lean_inc(v_ref_159_);
v___x_165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_165_, 0, v_ref_159_);
lean_ctor_set(v___x_165_, 1, v_a_161_);
if (v_isShared_164_ == 0)
{
lean_ctor_set_tag(v___x_163_, 1);
lean_ctor_set(v___x_163_, 0, v___x_165_);
v___x_167_ = v___x_163_;
goto v_reusejp_166_;
}
else
{
lean_object* v_reuseFailAlloc_168_; 
v_reuseFailAlloc_168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_168_, 0, v___x_165_);
v___x_167_ = v_reuseFailAlloc_168_;
goto v_reusejp_166_;
}
v_reusejp_166_:
{
return v___x_167_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg___boxed(lean_object* v_msg_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg(v_msg_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
return v_res_176_;
}
}
static lean_object* _init_lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__4(void){
_start:
{
lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_183_ = ((lean_object*)(lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__3));
v___x_184_ = l_Lean_stringToMessageData(v___x_183_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__6(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_186_ = ((lean_object*)(lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__5));
v___x_187_ = l_Lean_stringToMessageData(v___x_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0(lean_object* v_x_188_, lean_object* v_00___u03b1_189_, lean_object* v___z_u03b1_190_, lean_object* v_p_u03b1_x3f_191_, lean_object* v_e_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
if (lean_obj_tag(v_p_u03b1_x3f_191_) == 0)
{
lean_object* v___x_198_; lean_object* v___x_199_; 
lean_dec_ref(v_e_192_);
v___x_198_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_198_, 0, v_p_u03b1_x3f_191_);
v___x_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
return v___x_199_;
}
else
{
lean_object* v_val_200_; lean_object* v___x_201_; 
v_val_200_ = lean_ctor_get(v_p_u03b1_x3f_191_, 0);
lean_inc(v_val_200_);
lean_dec_ref_known(v_p_u03b1_x3f_191_, 1);
v___x_201_ = l_Lean_Meta_whnfR(v_e_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_);
if (lean_obj_tag(v___x_201_) == 0)
{
lean_object* v_a_202_; 
v_a_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc(v_a_202_);
lean_dec_ref_known(v___x_201_, 1);
if (lean_obj_tag(v_a_202_) == 5)
{
lean_object* v_fn_203_; lean_object* v_arg_204_; lean_object* v___y_206_; lean_object* v___y_207_; lean_object* v___y_208_; lean_object* v___y_209_; lean_object* v___x_233_; uint8_t v___x_234_; 
v_fn_203_ = lean_ctor_get(v_a_202_, 0);
lean_inc_ref(v_fn_203_);
v_arg_204_ = lean_ctor_get(v_a_202_, 1);
lean_inc_ref(v_arg_204_);
lean_dec_ref_known(v_a_202_, 2);
v___x_233_ = l_Lean_Expr_getAppFn(v_fn_203_);
v___x_234_ = l_Lean_Expr_isFVar(v___x_233_);
lean_dec_ref(v___x_233_);
if (v___x_234_ == 0)
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v_a_237_; lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_244_; 
lean_dec_ref(v_arg_204_);
lean_dec_ref(v_fn_203_);
lean_dec(v_val_200_);
v___x_235_ = lean_obj_once(&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__4, &lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__4_once, _init_lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__4);
v___x_236_ = lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg(v___x_235_, v___y_193_, v___y_194_, v___y_195_, v___y_196_);
v_a_237_ = lean_ctor_get(v___x_236_, 0);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_236_);
if (v_isSharedCheck_244_ == 0)
{
v___x_239_ = v___x_236_;
v_isShared_240_ = v_isSharedCheck_244_;
goto v_resetjp_238_;
}
else
{
lean_inc(v_a_237_);
lean_dec(v___x_236_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_244_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v___x_242_; 
if (v_isShared_240_ == 0)
{
v___x_242_ = v___x_239_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_a_237_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
else
{
v___y_206_ = v___y_193_;
v___y_207_ = v___y_194_;
v___y_208_ = v___y_195_;
v___y_209_ = v___y_196_;
goto v___jp_205_;
}
v___jp_205_:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_210_ = ((lean_object*)(lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__2));
v___x_211_ = lean_unsigned_to_nat(2u);
v___x_212_ = lean_mk_empty_array_with_capacity(v___x_211_);
v___x_213_ = lean_array_push(v___x_212_, v_fn_203_);
v___x_214_ = lean_array_push(v___x_213_, v_arg_204_);
v___x_215_ = l_Lean_Meta_mkAppM(v___x_210_, v___x_214_, v___y_206_, v___y_207_, v___y_208_, v___y_209_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_224_; 
v_a_216_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_224_ == 0)
{
v___x_218_ = v___x_215_;
v_isShared_219_ = v_isSharedCheck_224_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_224_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_220_; lean_object* v___x_222_; 
v___x_220_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_220_, 0, v_val_200_);
lean_ctor_set(v___x_220_, 1, v_a_216_);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_220_);
v___x_222_ = v___x_218_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_220_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
else
{
lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_232_; 
lean_dec(v_val_200_);
v_a_225_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_232_ == 0)
{
v___x_227_ = v___x_215_;
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_215_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_230_; 
if (v_isShared_228_ == 0)
{
v___x_230_ = v___x_227_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v_a_225_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
}
else
{
lean_object* v___x_245_; lean_object* v___x_246_; 
lean_dec(v_a_202_);
lean_dec(v_val_200_);
v___x_245_ = lean_obj_once(&lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__6, &lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__6_once, _init_lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___closed__6);
v___x_246_ = lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg(v___x_245_, v___y_193_, v___y_194_, v___y_195_, v___y_196_);
return v___x_246_;
}
}
else
{
lean_object* v_a_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_254_; 
lean_dec(v_val_200_);
v_a_247_ = lean_ctor_get(v___x_201_, 0);
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_201_);
if (v_isSharedCheck_254_ == 0)
{
v___x_249_ = v___x_201_;
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_a_247_);
lean_dec(v___x_201_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v___x_252_; 
if (v_isShared_250_ == 0)
{
v___x_252_ = v___x_249_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v_a_247_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0___boxed(lean_object* v_x_255_, lean_object* v_00___u03b1_256_, lean_object* v___z_u03b1_257_, lean_object* v_p_u03b1_x3f_258_, lean_object* v_e_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv___lam__0(v_x_255_, v_00___u03b1_256_, v___z_u03b1_257_, v_p_u03b1_x3f_258_, v_e_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec_ref(v___z_u03b1_257_);
lean_dec_ref(v_00___u03b1_256_);
lean_dec(v_x_255_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0(lean_object* v_00_u03b1_268_, lean_object* v_msg_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___redArg(v_msg_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0___boxed(lean_object* v_00_u03b1_276_, lean_object* v_msg_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_Lean_throwError___at___00IsAbsoluteValue_Mathlib_Meta_Positivity_evalAbv_spec__0(v_00_u03b1_276_, v_msg_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
lean_dec_ref(v___y_278_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue___redArg(lean_object* v_abv_284_){
_start:
{
lean_inc(v_abv_284_);
return v_abv_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue___redArg___boxed(lean_object* v_abv_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_IsAbsoluteValue_toAbsoluteValue___redArg(v_abv_285_);
lean_dec(v_abv_285_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue(lean_object* v_S_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_R_290_, lean_object* v_inst_291_, lean_object* v_abv_292_, lean_object* v_inst_293_){
_start:
{
lean_inc(v_abv_292_);
return v_abv_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_toAbsoluteValue___boxed(lean_object* v_S_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_R_297_, lean_object* v_inst_298_, lean_object* v_abv_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_IsAbsoluteValue_toAbsoluteValue(v_S_294_, v_inst_295_, v_inst_296_, v_R_297_, v_inst_298_, v_abv_299_, v_inst_300_);
lean_dec(v_abv_299_);
lean_dec_ref(v_inst_298_);
lean_dec_ref(v_inst_296_);
lean_dec_ref(v_inst_295_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom___redArg(lean_object* v_abv_302_){
_start:
{
lean_object* v___f_303_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_303_, 0, v_abv_302_);
return v___f_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom(lean_object* v_S_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_R_307_, lean_object* v_inst_308_, lean_object* v_abv_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_){
_start:
{
lean_object* v___f_313_; 
v___f_313_ = lean_alloc_closure((void*)(lp_mathlib_AbsoluteValue_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_313_, 0, v_abv_309_);
return v___f_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom___boxed(lean_object* v_S_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_R_317_, lean_object* v_inst_318_, lean_object* v_abv_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_IsAbsoluteValue_abvHom(v_S_314_, v_inst_315_, v_inst_316_, v_R_317_, v_inst_318_, v_abv_319_, v_inst_320_, v_inst_321_, v_inst_322_);
lean_dec_ref(v_inst_318_);
lean_dec_ref(v_inst_316_);
lean_dec_ref(v_inst_315_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27___redArg(lean_object* v_abv_324_){
_start:
{
lean_inc(v_abv_324_);
return v_abv_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27___redArg___boxed(lean_object* v_abv_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_IsAbsoluteValue_abvHom_x27___redArg(v_abv_325_);
lean_dec(v_abv_325_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27(lean_object* v_S_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_R_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_abv_334_, lean_object* v_inst_335_){
_start:
{
lean_inc(v_abv_334_);
return v_abv_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAbsoluteValue_abvHom_x27___boxed(lean_object* v_S_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_R_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_abv_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_IsAbsoluteValue_abvHom_x27(v_S_336_, v_inst_337_, v_inst_338_, v_inst_339_, v_R_340_, v_inst_341_, v_inst_342_, v_abv_343_, v_inst_344_);
lean_dec(v_abv_343_);
lean_dec_ref(v_inst_341_);
lean_dec_ref(v_inst_338_);
lean_dec_ref(v_inst_337_);
return v_res_345_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Positivity_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Positivity_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Positivity_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Positivity_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_AbsoluteValue_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
