// Lean compiler output
// Module: Mathlib.Algebra.Order.Nonneg.Field
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Basic public import Mathlib.Algebra.Order.Field.Canonical public import Mathlib.Algebra.Order.Nonneg.Ring public import Mathlib.Algebra.Order.Positive.Ring public import Mathlib.Data.Nat.Cast.Order.Ring
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
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Nonneg_add___redArg(lean_object*);
lean_object* lp_mathlib_Nonneg_mul___redArg(lean_object*);
lean_object* lp_mathlib_Nonneg_nsmul___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Nonneg_natCast___redArg(lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_Nonneg_linearOrderedCommMonoidWithZero___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Nonneg_semiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nonneg_unitsEquivPos___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___closed__0 = (const lean_object*)&lp_mathlib_Nonneg_unitsEquivPos___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__0(lean_object* v_r_1_){
_start:
{
lean_object* v_val_2_; 
v_val_2_ = lean_ctor_get(v_r_1_, 0);
lean_inc(v_val_2_);
return v_val_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__0___boxed(lean_object* v_r_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__0(v_r_3_);
lean_dec_ref(v_r_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__1(lean_object* v_toInv_5_, lean_object* v_r_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
lean_inc(v_r_6_);
v___x_7_ = lean_apply_1(v_toInv_5_, v_r_6_);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v_r_6_);
lean_ctor_set(v___x_8_, 1, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v_toInv_14_; lean_object* v___x_16_; uint8_t v_isShared_17_; uint8_t v_isSharedCheck_23_; 
v___x_11_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_10_);
v___x_12_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_11_);
v___x_13_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_12_);
lean_dec_ref(v___x_12_);
v_toInv_14_ = lean_ctor_get(v___x_13_, 1);
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_13_);
if (v_isSharedCheck_23_ == 0)
{
lean_object* v_unused_24_; 
v_unused_24_ = lean_ctor_get(v___x_13_, 0);
lean_dec(v_unused_24_);
v___x_16_ = v___x_13_;
v_isShared_17_ = v_isSharedCheck_23_;
goto v_resetjp_15_;
}
else
{
lean_inc(v_toInv_14_);
lean_dec(v___x_13_);
v___x_16_ = lean_box(0);
v_isShared_17_ = v_isSharedCheck_23_;
goto v_resetjp_15_;
}
v_resetjp_15_:
{
lean_object* v___f_18_; lean_object* v___f_19_; lean_object* v___x_21_; 
v___f_18_ = ((lean_object*)(lp_mathlib_Nonneg_unitsEquivPos___redArg___closed__0));
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_unitsEquivPos___redArg___lam__1), 2, 1);
lean_closure_set(v___f_19_, 0, v_toInv_14_);
if (v_isShared_17_ == 0)
{
lean_ctor_set(v___x_16_, 1, v___f_19_);
lean_ctor_set(v___x_16_, 0, v___f_18_);
v___x_21_ = v___x_16_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___f_18_);
lean_ctor_set(v_reuseFailAlloc_22_, 1, v___f_19_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___redArg___boxed(lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Nonneg_unitsEquivPos___redArg(v_inst_25_);
lean_dec_ref(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos(lean_object* v_R_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Nonneg_unitsEquivPos___redArg(v_inst_28_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_unitsEquivPos___boxed(lean_object* v_R_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Nonneg_unitsEquivPos(v_R_33_, v_inst_34_, v_inst_35_, v_inst_36_, v_inst_37_);
lean_dec_ref(v_inst_35_);
lean_dec_ref(v_inst_34_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___redArg___lam__0(lean_object* v_toInv_39_, lean_object* v_x_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_apply_1(v_toInv_39_, v_x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___redArg(lean_object* v_inst_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v_toInv_46_; lean_object* v___f_47_; 
v___x_43_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_42_);
v___x_44_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_43_);
v___x_45_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_44_);
lean_dec_ref(v___x_44_);
v_toInv_46_ = lean_ctor_get(v___x_45_, 1);
lean_inc(v_toInv_46_);
lean_dec_ref(v___x_45_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_47_, 0, v_toInv_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___redArg___boxed(lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Nonneg_inv___redArg(v_inst_48_);
lean_dec_ref(v_inst_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv(lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Nonneg_inv___redArg(v_inst_51_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_inv___boxed(lean_object* v_00_u03b1_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Nonneg_inv(v_00_u03b1_55_, v_inst_56_, v_inst_57_, v_inst_58_);
lean_dec_ref(v_inst_57_);
lean_dec_ref(v_inst_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___redArg___lam__0(lean_object* v_toDiv_60_, lean_object* v_x_61_, lean_object* v_y_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lean_apply_2(v_toDiv_60_, v_x_61_, v_y_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___redArg(lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v_toDiv_67_; lean_object* v___f_68_; 
v___x_65_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_64_);
v___x_66_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_65_);
v_toDiv_67_ = lean_ctor_get(v___x_66_, 2);
lean_inc(v_toDiv_67_);
lean_dec_ref(v___x_66_);
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_div___redArg___lam__0), 3, 1);
lean_closure_set(v___f_68_, 0, v_toDiv_67_);
return v___f_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___redArg___boxed(lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_Nonneg_div___redArg(v_inst_69_);
lean_dec_ref(v_inst_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div(lean_object* v_00_u03b1_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Nonneg_div___redArg(v_inst_72_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_div___boxed(lean_object* v_00_u03b1_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_Nonneg_div(v_00_u03b1_76_, v_inst_77_, v_inst_78_, v_inst_79_);
lean_dec_ref(v_inst_78_);
lean_dec_ref(v_inst_77_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___redArg___lam__0(lean_object* v_toZPow_81_, lean_object* v_a_82_, lean_object* v_n_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lean_apply_2(v_toZPow_81_, v_n_83_, v_a_82_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___redArg(lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v_toZPow_88_; lean_object* v___f_89_; 
v___x_86_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_85_);
v___x_87_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_86_);
v_toZPow_88_ = lean_ctor_get(v___x_87_, 3);
lean_inc(v_toZPow_88_);
lean_dec_ref(v___x_87_);
v___f_89_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_zpow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_89_, 0, v_toZPow_88_);
return v___f_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___redArg___boxed(lean_object* v_inst_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_Nonneg_zpow___redArg(v_inst_90_);
lean_dec_ref(v_inst_90_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow(lean_object* v_00_u03b1_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_Nonneg_zpow___redArg(v_inst_93_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_zpow___boxed(lean_object* v_00_u03b1_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Nonneg_zpow(v_00_u03b1_97_, v_inst_98_, v_inst_99_, v_inst_100_);
lean_dec_ref(v_inst_99_);
lean_dec_ref(v_inst_98_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast___redArg___lam__0(lean_object* v_toNNRatCast_102_, lean_object* v_q_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_apply_1(v_toNNRatCast_102_, v_q_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast___redArg(lean_object* v_inst_105_){
_start:
{
lean_object* v_toNNRatCast_106_; lean_object* v___f_107_; 
v_toNNRatCast_106_ = lean_ctor_get(v_inst_105_, 4);
lean_inc(v_toNNRatCast_106_);
lean_dec_ref(v_inst_105_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instNNRatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_107_, 0, v_toNNRatCast_106_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast(lean_object* v_00_u03b1_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_Nonneg_instNNRatCast___redArg(v_inst_109_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatCast___boxed(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Nonneg_instNNRatCast(v_00_u03b1_113_, v_inst_114_, v_inst_115_, v_inst_116_);
lean_dec_ref(v_inst_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul___redArg___lam__0(lean_object* v_inst_118_, lean_object* v_q_119_, lean_object* v_a_120_){
_start:
{
lean_object* v_nnqsmul_121_; lean_object* v___x_122_; 
v_nnqsmul_121_ = lean_ctor_get(v_inst_118_, 5);
lean_inc(v_nnqsmul_121_);
lean_dec_ref(v_inst_118_);
v___x_122_ = lean_apply_2(v_nnqsmul_121_, v_q_119_, v_a_120_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul___redArg(lean_object* v_inst_123_){
_start:
{
lean_object* v___f_124_; 
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instNNRatSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_124_, 0, v_inst_123_);
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___f_129_; 
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_instNNRatSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_129_, 0, v_inst_126_);
return v___f_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_instNNRatSMul___boxed(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Nonneg_instNNRatSMul(v_00_u03b1_130_, v_inst_131_, v_inst_132_, v_inst_133_);
lean_dec_ref(v_inst_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__0(lean_object* v_inst_135_, lean_object* v_n_136_, lean_object* v_x_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v_toZPow_140_; lean_object* v___x_141_; 
v___x_138_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_135_);
v___x_139_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_138_);
v_toZPow_140_ = lean_ctor_get(v___x_139_, 3);
lean_inc(v_toZPow_140_);
lean_dec_ref(v___x_139_);
v___x_141_ = lean_apply_2(v_toZPow_140_, v_n_136_, v_x_137_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__0___boxed(lean_object* v_inst_142_, lean_object* v_n_143_, lean_object* v_x_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_Nonneg_divisionSemiring___redArg___lam__0(v_inst_142_, v_n_143_, v_x_144_);
lean_dec_ref(v_inst_142_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__1(lean_object* v_nnqsmul_146_, lean_object* v_x1_147_, lean_object* v_x2_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lean_apply_2(v_nnqsmul_146_, v_x1_147_, v_x2_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg___lam__2(lean_object* v___x_150_, lean_object* v_n_151_, lean_object* v_x_152_){
_start:
{
lean_object* v_toMonoid_153_; lean_object* v_toNPow_154_; lean_object* v___x_155_; 
v_toMonoid_153_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref(v_toMonoid_153_);
lean_dec_ref(v___x_150_);
v_toNPow_154_ = lean_ctor_get(v_toMonoid_153_, 2);
lean_inc(v_toNPow_154_);
lean_dec_ref(v_toMonoid_153_);
v___x_155_ = lean_apply_2(v_toNPow_154_, v_n_151_, v_x_152_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___redArg(lean_object* v_inst_156_){
_start:
{
lean_object* v_toSemiring_157_; lean_object* v_nnqsmul_158_; lean_object* v___x_159_; lean_object* v_toZero_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v_toAddMonoid_163_; lean_object* v_toOne_164_; lean_object* v___x_165_; lean_object* v_toNonUnitalNonAssocSemiring_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_189_; 
v_toSemiring_157_ = lean_ctor_get(v_inst_156_, 0);
v_nnqsmul_158_ = lean_ctor_get(v_inst_156_, 5);
lean_inc_ref_n(v_toSemiring_157_, 2);
v___x_159_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_157_);
v_toZero_160_ = lean_ctor_get(v___x_159_, 1);
lean_inc(v_toZero_160_);
v___x_161_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_toSemiring_157_);
lean_inc_ref(v___x_161_);
v___x_162_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_161_);
v_toAddMonoid_163_ = lean_ctor_get(v___x_162_, 1);
lean_inc_ref(v_toAddMonoid_163_);
v_toOne_164_ = lean_ctor_get(v___x_162_, 2);
lean_inc(v_toOne_164_);
v___x_165_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_163_);
lean_dec_ref(v_toAddMonoid_163_);
v_toNonUnitalNonAssocSemiring_166_ = lean_ctor_get(v___x_161_, 0);
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_161_);
if (v_isSharedCheck_189_ == 0)
{
lean_object* v_unused_190_; lean_object* v_unused_191_; 
v_unused_190_ = lean_ctor_get(v___x_161_, 2);
lean_dec(v_unused_190_);
v_unused_191_ = lean_ctor_get(v___x_161_, 1);
lean_dec(v_unused_191_);
v___x_168_ = v___x_161_;
v_isShared_169_ = v_isSharedCheck_189_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_toNonUnitalNonAssocSemiring_166_);
lean_dec(v___x_161_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_189_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v_toAddCommMonoid_170_; lean_object* v___f_171_; lean_object* v___f_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___f_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_185_; 
v_toAddCommMonoid_170_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_166_, 0);
lean_inc_ref(v_toAddCommMonoid_170_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_166_);
lean_inc_ref(v_inst_156_);
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_divisionSemiring___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_171_, 0, v_inst_156_);
lean_inc(v_nnqsmul_158_);
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_172_, 0, v_nnqsmul_158_);
v___x_173_ = lp_mathlib_Nonneg_add___redArg(v___x_165_);
v___x_174_ = lp_mathlib_Nonneg_mul___redArg(v___x_159_);
v___x_175_ = lp_mathlib_Nonneg_inv___redArg(v_inst_156_);
v___x_176_ = lp_mathlib_Nonneg_div___redArg(v_inst_156_);
v___x_177_ = lp_mathlib_Nonneg_nsmul___redArg(v_toAddCommMonoid_170_);
v___x_178_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_toSemiring_157_);
v___f_179_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_179_, 0, v___x_178_);
v___x_180_ = lp_mathlib_Nonneg_natCast___redArg(v___x_162_);
v___x_181_ = lp_mathlib_Nonneg_instNNRatCast___redArg(v_inst_156_);
v___x_182_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_182_, 0, lean_box(0));
lean_closure_set(v___x_182_, 1, v___x_180_);
v___x_183_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___x_173_, v_toZero_160_, v___x_177_);
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 2, v___f_179_);
lean_ctor_set(v___x_168_, 1, v___x_174_);
lean_ctor_set(v___x_168_, 0, v_toOne_164_);
v___x_185_ = v___x_168_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_toOne_164_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v___x_174_);
lean_ctor_set(v_reuseFailAlloc_188_, 2, v___f_179_);
v___x_185_ = v_reuseFailAlloc_188_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_186_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_186_, 0, v___x_183_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
lean_ctor_set(v___x_186_, 2, v___x_182_);
v___x_187_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_187_, 0, v___x_186_);
lean_ctor_set(v___x_187_, 1, v___x_175_);
lean_ctor_set(v___x_187_, 2, v___x_176_);
lean_ctor_set(v___x_187_, 3, v___f_171_);
lean_ctor_set(v___x_187_, 4, v___x_181_);
lean_ctor_set(v___x_187_, 5, v___f_172_);
return v___x_187_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib_Nonneg_divisionSemiring___redArg(v_inst_193_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_divisionSemiring___boxed(lean_object* v_00_u03b1_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Nonneg_divisionSemiring(v_00_u03b1_197_, v_inst_198_, v_inst_199_, v_inst_200_);
lean_dec_ref(v_inst_199_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield___redArg___lam__0(lean_object* v_nnqsmul_202_, lean_object* v_a_203_, lean_object* v_a_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lean_apply_2(v_nnqsmul_202_, v_a_203_, v_a_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield___redArg(lean_object* v_inst_206_){
_start:
{
lean_object* v_toCommSemiring_207_; lean_object* v_nnqsmul_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v_toZPow_214_; lean_object* v___f_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_toCommSemiring_207_ = lean_ctor_get(v_inst_206_, 0);
v_nnqsmul_208_ = lean_ctor_get(v_inst_206_, 5);
lean_inc(v_nnqsmul_208_);
lean_inc_ref(v_toCommSemiring_207_);
v___x_209_ = lp_mathlib_Nonneg_semiring___redArg(v_toCommSemiring_207_);
v___x_210_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v_inst_206_);
lean_inc_ref(v___x_210_);
v___x_211_ = lp_mathlib_Nonneg_divisionSemiring___redArg(v___x_210_);
v___x_212_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_211_);
lean_dec_ref(v___x_211_);
v___x_213_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_212_);
v_toZPow_214_ = lean_ctor_get(v___x_213_, 3);
lean_inc(v_toZPow_214_);
lean_dec_ref(v___x_213_);
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_Nonneg_semifield___redArg___lam__0), 3, 1);
lean_closure_set(v___f_215_, 0, v_nnqsmul_208_);
v___x_216_ = lp_mathlib_Nonneg_inv___redArg(v___x_210_);
v___x_217_ = lp_mathlib_Nonneg_div___redArg(v___x_210_);
v___x_218_ = lp_mathlib_Nonneg_instNNRatCast___redArg(v___x_210_);
v___x_219_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_219_, 0, v___x_209_);
lean_ctor_set(v___x_219_, 1, v___x_216_);
lean_ctor_set(v___x_219_, 2, v___x_217_);
lean_ctor_set(v___x_219_, 3, v_toZPow_214_);
lean_ctor_set(v___x_219_, 4, v___x_218_);
lean_ctor_set(v___x_219_, 5, v___f_215_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield(lean_object* v_00_u03b1_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lp_mathlib_Nonneg_semifield___redArg(v_inst_221_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_semifield___boxed(lean_object* v_00_u03b1_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Nonneg_semifield(v_00_u03b1_225_, v_inst_226_, v_inst_227_, v_inst_228_);
lean_dec_ref(v_inst_227_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg(lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; lean_object* v_toCommSemiring_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v_toZPow_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_248_; 
v___x_232_ = lp_mathlib_Field_toSemifield___redArg(v_inst_230_);
v_toCommSemiring_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc_ref(v_toCommSemiring_233_);
v___x_234_ = lp_mathlib_Nonneg_linearOrderedCommMonoidWithZero___redArg(v_toCommSemiring_233_, v_inst_231_);
v___x_235_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v___x_232_);
lean_inc_ref(v___x_235_);
v___x_236_ = lp_mathlib_Nonneg_divisionSemiring___redArg(v___x_235_);
v___x_237_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_236_);
lean_dec_ref(v___x_236_);
v___x_238_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_237_);
v_toZPow_239_ = lean_ctor_get(v___x_238_, 3);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_248_ == 0)
{
lean_object* v_unused_249_; lean_object* v_unused_250_; lean_object* v_unused_251_; 
v_unused_249_ = lean_ctor_get(v___x_238_, 2);
lean_dec(v_unused_249_);
v_unused_250_ = lean_ctor_get(v___x_238_, 1);
lean_dec(v_unused_250_);
v_unused_251_ = lean_ctor_get(v___x_238_, 0);
lean_dec(v_unused_251_);
v___x_241_ = v___x_238_;
v_isShared_242_ = v_isSharedCheck_248_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_toZPow_239_);
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_248_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_246_; 
v___x_243_ = lp_mathlib_Nonneg_inv___redArg(v___x_235_);
v___x_244_ = lp_mathlib_Nonneg_div___redArg(v___x_235_);
lean_dec_ref(v___x_235_);
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 2, v___x_244_);
lean_ctor_set(v___x_241_, 1, v___x_243_);
lean_ctor_set(v___x_241_, 0, v___x_234_);
v___x_246_ = v___x_241_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v___x_234_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_247_, 2, v___x_244_);
lean_ctor_set(v_reuseFailAlloc_247_, 3, v_toZPow_239_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg___boxed(lean_object* v_inst_252_, lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg(v_inst_252_, v_inst_253_);
lean_dec_ref(v_inst_252_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero(lean_object* v_00_u03b1_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___redArg(v_inst_256_, v_inst_257_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_linearOrderedCommGroupWithZero___boxed(lean_object* v_00_u03b1_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Nonneg_linearOrderedCommGroupWithZero(v_00_u03b1_260_, v_inst_261_, v_inst_262_, v_inst_263_);
lean_dec_ref(v_inst_261_);
return v_res_264_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(builtin);
}
#ifdef __cplusplus
}
#endif
