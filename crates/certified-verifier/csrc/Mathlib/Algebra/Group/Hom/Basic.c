// Lean compiler output
// Module: Mathlib.Algebra.Group.Hom.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.Group.Hom.Defs
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulAddMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowGroupHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowGroupHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowGroupHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulAddGroupHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulAddGroupHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zsmulAddGroupHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_instAdd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_instAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instDiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instSub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powMonoidHom___redArg___lam__0(lean_object* v_toNPow_1_, lean_object* v_n_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toNPow_1_, v_n_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powMonoidHom___redArg(lean_object* v_inst_5_, lean_object* v_n_6_){
_start:
{
lean_object* v_toNPow_7_; lean_object* v___f_8_; 
v_toNPow_7_ = lean_ctor_get(v_inst_5_, 2);
lean_inc(v_toNPow_7_);
lean_dec_ref(v_inst_5_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_powMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_8_, 0, v_toNPow_7_);
lean_closure_set(v___f_8_, 1, v_n_6_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powMonoidHom(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_n_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_powMonoidHom___redArg(v_inst_10_, v_n_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulAddMonoidHom___redArg___lam__0(lean_object* v_toNSMul_13_, lean_object* v_n_14_, lean_object* v_x_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_apply_2(v_toNSMul_13_, v_n_14_, v_x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulAddMonoidHom___redArg(lean_object* v_inst_17_, lean_object* v_n_18_){
_start:
{
lean_object* v_toNSMul_19_; lean_object* v___f_20_; 
v_toNSMul_19_ = lean_ctor_get(v_inst_17_, 2);
lean_inc(v_toNSMul_19_);
lean_dec_ref(v_inst_17_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_nsmulAddMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_20_, 0, v_toNSMul_19_);
lean_closure_set(v___f_20_, 1, v_n_18_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulAddMonoidHom(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_n_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_nsmulAddMonoidHom___redArg(v_inst_22_, v_n_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowGroupHom___redArg___lam__0(lean_object* v_toZPow_25_, lean_object* v_n_26_, lean_object* v_x_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_apply_2(v_toZPow_25_, v_n_26_, v_x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowGroupHom___redArg(lean_object* v_inst_29_, lean_object* v_n_30_){
_start:
{
lean_object* v_toZPow_31_; lean_object* v___f_32_; 
v_toZPow_31_ = lean_ctor_get(v_inst_29_, 3);
lean_inc(v_toZPow_31_);
lean_dec_ref(v_inst_29_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_zpowGroupHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_32_, 0, v_toZPow_31_);
lean_closure_set(v___f_32_, 1, v_n_30_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowGroupHom(lean_object* v_00_u03b1_33_, lean_object* v_inst_34_, lean_object* v_n_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_zpowGroupHom___redArg(v_inst_34_, v_n_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulAddGroupHom___redArg___lam__0(lean_object* v_toZSMul_37_, lean_object* v_n_38_, lean_object* v_x_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_apply_2(v_toZSMul_37_, v_n_38_, v_x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulAddGroupHom___redArg(lean_object* v_inst_41_, lean_object* v_n_42_){
_start:
{
lean_object* v_toZSMul_43_; lean_object* v___f_44_; 
v_toZSMul_43_ = lean_ctor_get(v_inst_41_, 3);
lean_inc(v_toZSMul_43_);
lean_dec_ref(v_inst_41_);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_zsmulAddGroupHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_44_, 0, v_toZSMul_43_);
lean_closure_set(v___f_44_, 1, v_n_42_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zsmulAddGroupHom(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_, lean_object* v_n_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_zsmulAddGroupHom___redArg(v_inst_46_, v_n_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom___redArg(lean_object* v_inst_49_){
_start:
{
lean_object* v___x_50_; lean_object* v_toInv_51_; 
v___x_50_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_49_);
v_toInv_51_ = lean_ctor_get(v___x_50_, 1);
lean_inc(v_toInv_51_);
lean_dec_ref(v___x_50_);
return v_toInv_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom___redArg___boxed(lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_invMonoidHom___redArg(v_inst_52_);
lean_dec_ref(v_inst_52_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom(lean_object* v_00_u03b1_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_invMonoidHom___redArg(v_inst_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invMonoidHom___boxed(lean_object* v_00_u03b1_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_invMonoidHom(v_00_u03b1_57_, v_inst_58_);
lean_dec_ref(v_inst_58_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom___redArg(lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; lean_object* v_toNeg_62_; 
v___x_61_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_60_);
v_toNeg_62_ = lean_ctor_get(v___x_61_, 1);
lean_inc(v_toNeg_62_);
lean_dec_ref(v___x_61_);
return v_toNeg_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom___redArg___boxed(lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_negAddMonoidHom___redArg(v_inst_63_);
lean_dec_ref(v_inst_63_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_negAddMonoidHom___redArg(v_inst_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_negAddMonoidHom___boxed(lean_object* v_00_u03b1_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_negAddMonoidHom(v_00_u03b1_68_, v_inst_69_);
lean_dec_ref(v_inst_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul___redArg___lam__0(lean_object* v_toMul_71_, lean_object* v_f_72_, lean_object* v_g_73_, lean_object* v___y_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
lean_inc(v___y_74_);
v___x_75_ = lean_apply_1(v_f_72_, v___y_74_);
v___x_76_ = lean_apply_1(v_g_73_, v___y_74_);
v___x_77_ = lean_apply_2(v_toMul_71_, v___x_75_, v___x_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul___redArg(lean_object* v_inst_78_){
_start:
{
lean_object* v___x_79_; lean_object* v_toMul_80_; lean_object* v___f_81_; 
v___x_79_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_78_);
v_toMul_80_ = lean_ctor_get(v___x_79_, 1);
lean_inc(v_toMul_80_);
lean_dec_ref(v___x_79_);
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_81_, 0, v_toMul_80_);
return v___f_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul(lean_object* v_M_82_, lean_object* v_N_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_OneHom_instMul___redArg(v_inst_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMul___boxed(lean_object* v_M_87_, lean_object* v_N_88_, lean_object* v_inst_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_OneHom_instMul(v_M_87_, v_N_88_, v_inst_89_, v_inst_90_);
lean_dec(v_inst_89_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd___redArg___lam__0(lean_object* v_toAdd_92_, lean_object* v_f_93_, lean_object* v_g_94_, lean_object* v___y_95_){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
lean_inc(v___y_95_);
v___x_96_ = lean_apply_1(v_f_93_, v___y_95_);
v___x_97_ = lean_apply_1(v_g_94_, v___y_95_);
v___x_98_ = lean_apply_2(v_toAdd_92_, v___x_96_, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd___redArg(lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; lean_object* v_toAdd_101_; lean_object* v___f_102_; 
v___x_100_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_99_);
v_toAdd_101_ = lean_ctor_get(v___x_100_, 1);
lean_inc(v_toAdd_101_);
lean_dec_ref(v___x_100_);
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instAdd___redArg___lam__0), 4, 1);
lean_closure_set(v___f_102_, 0, v_toAdd_101_);
return v___f_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd(lean_object* v_M_103_, lean_object* v_N_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_ZeroHom_instAdd___redArg(v_inst_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAdd___boxed(lean_object* v_M_108_, lean_object* v_N_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_ZeroHom_instAdd(v_M_108_, v_N_109_, v_inst_110_, v_inst_111_);
lean_dec(v_inst_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv___redArg___lam__0(lean_object* v_toInv_113_, lean_object* v_f_114_, lean_object* v___y_115_){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = lean_apply_1(v_f_114_, v___y_115_);
v___x_117_ = lean_apply_1(v_toInv_113_, v___x_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv___redArg(lean_object* v_inst_118_){
_start:
{
lean_object* v_toInv_119_; lean_object* v___f_120_; 
v_toInv_119_ = lean_ctor_get(v_inst_118_, 1);
lean_inc(v_toInv_119_);
lean_dec_ref(v_inst_118_);
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_120_, 0, v_toInv_119_);
return v___f_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv(lean_object* v_M_121_, lean_object* v_N_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_mathlib_OneHom_instInv___redArg(v_inst_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instInv___boxed(lean_object* v_M_126_, lean_object* v_N_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_OneHom_instInv(v_M_126_, v_N_127_, v_inst_128_, v_inst_129_);
lean_dec(v_inst_128_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg___redArg___lam__0(lean_object* v_toNeg_131_, lean_object* v_f_132_, lean_object* v___y_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = lean_apply_1(v_f_132_, v___y_133_);
v___x_135_ = lean_apply_1(v_toNeg_131_, v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg___redArg(lean_object* v_inst_136_){
_start:
{
lean_object* v_toNeg_137_; lean_object* v___f_138_; 
v_toNeg_137_ = lean_ctor_get(v_inst_136_, 1);
lean_inc(v_toNeg_137_);
lean_dec_ref(v_inst_136_);
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instNeg___redArg___lam__0), 3, 1);
lean_closure_set(v___f_138_, 0, v_toNeg_137_);
return v___f_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg(lean_object* v_M_139_, lean_object* v_N_140_, lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_mathlib_ZeroHom_instNeg___redArg(v_inst_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNeg___boxed(lean_object* v_M_144_, lean_object* v_N_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_ZeroHom_instNeg(v_M_144_, v_N_145_, v_inst_146_, v_inst_147_);
lean_dec(v_inst_146_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv___redArg___lam__0(lean_object* v_toDiv_149_, lean_object* v_f_150_, lean_object* v_g_151_, lean_object* v___y_152_){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
lean_inc(v___y_152_);
v___x_153_ = lean_apply_1(v_f_150_, v___y_152_);
v___x_154_ = lean_apply_1(v_g_151_, v___y_152_);
v___x_155_ = lean_apply_2(v_toDiv_149_, v___x_153_, v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv___redArg(lean_object* v_inst_156_){
_start:
{
lean_object* v_toDiv_157_; lean_object* v___f_158_; 
v_toDiv_157_ = lean_ctor_get(v_inst_156_, 2);
lean_inc(v_toDiv_157_);
lean_dec_ref(v_inst_156_);
v___f_158_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instDiv___redArg___lam__0), 4, 1);
lean_closure_set(v___f_158_, 0, v_toDiv_157_);
return v___f_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv(lean_object* v_M_159_, lean_object* v_N_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_OneHom_instDiv___redArg(v_inst_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instDiv___boxed(lean_object* v_M_164_, lean_object* v_N_165_, lean_object* v_inst_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_OneHom_instDiv(v_M_164_, v_N_165_, v_inst_166_, v_inst_167_);
lean_dec(v_inst_166_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub___redArg___lam__0(lean_object* v_toSub_169_, lean_object* v_f_170_, lean_object* v_g_171_, lean_object* v___y_172_){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
lean_inc(v___y_172_);
v___x_173_ = lean_apply_1(v_f_170_, v___y_172_);
v___x_174_ = lean_apply_1(v_g_171_, v___y_172_);
v___x_175_ = lean_apply_2(v_toSub_169_, v___x_173_, v___x_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub___redArg(lean_object* v_inst_176_){
_start:
{
lean_object* v_toSub_177_; lean_object* v___f_178_; 
v_toSub_177_ = lean_ctor_get(v_inst_176_, 2);
lean_inc(v_toSub_177_);
lean_dec_ref(v_inst_176_);
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instSub___redArg___lam__0), 4, 1);
lean_closure_set(v___f_178_, 0, v_toSub_177_);
return v___f_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub(lean_object* v_M_179_, lean_object* v_N_180_, lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_mathlib_ZeroHom_instSub___redArg(v_inst_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instSub___boxed(lean_object* v_M_184_, lean_object* v_N_185_, lean_object* v_inst_186_, lean_object* v_inst_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_ZeroHom_instSub(v_M_184_, v_N_185_, v_inst_186_, v_inst_187_);
lean_dec(v_inst_186_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul___redArg___lam__0(lean_object* v_inst_189_, lean_object* v_f_190_, lean_object* v_g_191_, lean_object* v___y_192_){
_start:
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
lean_inc(v___y_192_);
v___x_193_ = lean_apply_1(v_f_190_, v___y_192_);
v___x_194_ = lean_apply_1(v_g_191_, v___y_192_);
v___x_195_ = lean_apply_2(v_inst_189_, v___x_193_, v___x_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul___redArg(lean_object* v_inst_196_){
_start:
{
lean_object* v___f_197_; 
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_197_, 0, v_inst_196_);
return v___f_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul(lean_object* v_M_198_, lean_object* v_N_199_, lean_object* v_inst_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v___f_202_; 
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_202_, 0, v_inst_201_);
return v___f_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_instMul___boxed(lean_object* v_M_203_, lean_object* v_N_204_, lean_object* v_inst_205_, lean_object* v_inst_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_MulHom_instMul(v_M_203_, v_N_204_, v_inst_205_, v_inst_206_);
lean_dec(v_inst_205_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_instAdd___redArg(lean_object* v_inst_208_){
_start:
{
lean_object* v___f_209_; 
v___f_209_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_209_, 0, v_inst_208_);
return v___f_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_instAdd(lean_object* v_M_210_, lean_object* v_N_211_, lean_object* v_inst_212_, lean_object* v_inst_213_){
_start:
{
lean_object* v___f_214_; 
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_214_, 0, v_inst_213_);
return v___f_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_instAdd___boxed(lean_object* v_M_215_, lean_object* v_N_216_, lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_AddHom_instAdd(v_M_215_, v_N_216_, v_inst_217_, v_inst_218_);
lean_dec(v_inst_217_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv___redArg(lean_object* v_f_220_){
_start:
{
lean_inc(v_f_220_);
return v_f_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv___redArg___boxed(lean_object* v_f_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_MonoidHom_ofMapMulInv___redArg(v_f_221_);
lean_dec(v_f_221_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv(lean_object* v_G_223_, lean_object* v_inst_224_, lean_object* v_H_225_, lean_object* v_inst_226_, lean_object* v_f_227_, lean_object* v_map__div_228_){
_start:
{
lean_inc(v_f_227_);
return v_f_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapMulInv___boxed(lean_object* v_G_229_, lean_object* v_inst_230_, lean_object* v_H_231_, lean_object* v_inst_232_, lean_object* v_f_233_, lean_object* v_map__div_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_MonoidHom_ofMapMulInv(v_G_229_, v_inst_230_, v_H_231_, v_inst_232_, v_f_233_, v_map__div_234_);
lean_dec(v_f_233_);
lean_dec_ref(v_inst_232_);
lean_dec_ref(v_inst_230_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg___redArg(lean_object* v_f_236_){
_start:
{
lean_inc(v_f_236_);
return v_f_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg___redArg___boxed(lean_object* v_f_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_AddMonoidHom_ofMapAddNeg___redArg(v_f_237_);
lean_dec(v_f_237_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg(lean_object* v_G_239_, lean_object* v_inst_240_, lean_object* v_H_241_, lean_object* v_inst_242_, lean_object* v_f_243_, lean_object* v_map__div_244_){
_start:
{
lean_inc(v_f_243_);
return v_f_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapAddNeg___boxed(lean_object* v_G_245_, lean_object* v_inst_246_, lean_object* v_H_247_, lean_object* v_inst_248_, lean_object* v_f_249_, lean_object* v_map__div_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_AddMonoidHom_ofMapAddNeg(v_G_245_, v_inst_246_, v_H_247_, v_inst_248_, v_f_249_, v_map__div_250_);
lean_dec(v_f_249_);
lean_dec_ref(v_inst_248_);
lean_dec_ref(v_inst_246_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv___redArg(lean_object* v_f_252_){
_start:
{
lean_inc(v_f_252_);
return v_f_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv___redArg___boxed(lean_object* v_f_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_MonoidHom_ofMapDiv___redArg(v_f_253_);
lean_dec(v_f_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv(lean_object* v_G_255_, lean_object* v_inst_256_, lean_object* v_H_257_, lean_object* v_inst_258_, lean_object* v_f_259_, lean_object* v_hf_260_){
_start:
{
lean_inc(v_f_259_);
return v_f_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofMapDiv___boxed(lean_object* v_G_261_, lean_object* v_inst_262_, lean_object* v_H_263_, lean_object* v_inst_264_, lean_object* v_f_265_, lean_object* v_hf_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_MonoidHom_ofMapDiv(v_G_261_, v_inst_262_, v_H_263_, v_inst_264_, v_f_265_, v_hf_266_);
lean_dec(v_f_265_);
lean_dec_ref(v_inst_264_);
lean_dec_ref(v_inst_262_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub___redArg(lean_object* v_f_268_){
_start:
{
lean_inc(v_f_268_);
return v_f_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub___redArg___boxed(lean_object* v_f_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_AddMonoidHom_ofMapSub___redArg(v_f_269_);
lean_dec(v_f_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub(lean_object* v_G_271_, lean_object* v_inst_272_, lean_object* v_H_273_, lean_object* v_inst_274_, lean_object* v_f_275_, lean_object* v_hf_276_){
_start:
{
lean_inc(v_f_275_);
return v_f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofMapSub___boxed(lean_object* v_G_277_, lean_object* v_inst_278_, lean_object* v_H_279_, lean_object* v_inst_280_, lean_object* v_f_281_, lean_object* v_hf_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_AddMonoidHom_ofMapSub(v_G_277_, v_inst_278_, v_H_279_, v_inst_280_, v_f_281_, v_hf_282_);
lean_dec(v_f_281_);
lean_dec_ref(v_inst_280_);
lean_dec_ref(v_inst_278_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul___redArg(lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v_toMul_287_; lean_object* v___f_288_; 
v___x_285_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_284_);
v___x_286_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_285_);
v_toMul_287_ = lean_ctor_get(v___x_286_, 1);
lean_inc(v_toMul_287_);
lean_dec_ref(v___x_286_);
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_288_, 0, v_toMul_287_);
return v___f_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul___redArg___boxed(lean_object* v_inst_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_MonoidHom_mul___redArg(v_inst_289_);
lean_dec_ref(v_inst_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul(lean_object* v_M_291_, lean_object* v_N_292_, lean_object* v_inst_293_, lean_object* v_inst_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lp_mathlib_MonoidHom_mul___redArg(v_inst_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mul___boxed(lean_object* v_M_296_, lean_object* v_N_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_MonoidHom_mul(v_M_296_, v_N_297_, v_inst_298_, v_inst_299_);
lean_dec_ref(v_inst_299_);
lean_dec_ref(v_inst_298_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add___redArg(lean_object* v_inst_301_){
_start:
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v_toAdd_304_; lean_object* v___f_305_; 
v___x_302_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_301_);
v___x_303_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_302_);
v_toAdd_304_ = lean_ctor_get(v___x_303_, 1);
lean_inc(v_toAdd_304_);
lean_dec_ref(v___x_303_);
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instAdd___redArg___lam__0), 4, 1);
lean_closure_set(v___f_305_, 0, v_toAdd_304_);
return v___f_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add___redArg___boxed(lean_object* v_inst_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_AddMonoidHom_add___redArg(v_inst_306_);
lean_dec_ref(v_inst_306_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add(lean_object* v_M_308_, lean_object* v_N_309_, lean_object* v_inst_310_, lean_object* v_inst_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lp_mathlib_AddMonoidHom_add___redArg(v_inst_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_add___boxed(lean_object* v_M_313_, lean_object* v_N_314_, lean_object* v_inst_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_AddMonoidHom_add(v_M_313_, v_N_314_, v_inst_315_, v_inst_316_);
lean_dec_ref(v_inst_316_);
lean_dec_ref(v_inst_315_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv___redArg(lean_object* v_inst_318_){
_start:
{
lean_object* v___x_319_; lean_object* v_toInv_320_; lean_object* v___f_321_; 
v___x_319_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_318_);
v_toInv_320_ = lean_ctor_get(v___x_319_, 1);
lean_inc(v_toInv_320_);
lean_dec_ref(v___x_319_);
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_321_, 0, v_toInv_320_);
return v___f_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv___redArg___boxed(lean_object* v_inst_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_MonoidHom_instInv___redArg(v_inst_322_);
lean_dec_ref(v_inst_322_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv(lean_object* v_M_324_, lean_object* v_G_325_, lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_MonoidHom_instInv___redArg(v_inst_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instInv___boxed(lean_object* v_M_329_, lean_object* v_G_330_, lean_object* v_inst_331_, lean_object* v_inst_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_MonoidHom_instInv(v_M_329_, v_G_330_, v_inst_331_, v_inst_332_);
lean_dec_ref(v_inst_332_);
lean_dec_ref(v_inst_331_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg___redArg(lean_object* v_inst_334_){
_start:
{
lean_object* v___x_335_; lean_object* v_toNeg_336_; lean_object* v___f_337_; 
v___x_335_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_334_);
v_toNeg_336_ = lean_ctor_get(v___x_335_, 1);
lean_inc(v_toNeg_336_);
lean_dec_ref(v___x_335_);
v___f_337_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instNeg___redArg___lam__0), 3, 1);
lean_closure_set(v___f_337_, 0, v_toNeg_336_);
return v___f_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg___redArg___boxed(lean_object* v_inst_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_AddMonoidHom_instNeg___redArg(v_inst_338_);
lean_dec_ref(v_inst_338_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg(lean_object* v_M_340_, lean_object* v_G_341_, lean_object* v_inst_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lp_mathlib_AddMonoidHom_instNeg___redArg(v_inst_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNeg___boxed(lean_object* v_M_345_, lean_object* v_G_346_, lean_object* v_inst_347_, lean_object* v_inst_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_AddMonoidHom_instNeg(v_M_345_, v_G_346_, v_inst_347_, v_inst_348_);
lean_dec_ref(v_inst_348_);
lean_dec_ref(v_inst_347_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instDiv___redArg(lean_object* v_inst_350_){
_start:
{
lean_object* v_toDiv_351_; lean_object* v___f_352_; 
v_toDiv_351_ = lean_ctor_get(v_inst_350_, 2);
lean_inc(v_toDiv_351_);
lean_dec_ref(v_inst_350_);
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instDiv___redArg___lam__0), 4, 1);
lean_closure_set(v___f_352_, 0, v_toDiv_351_);
return v___f_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instDiv(lean_object* v_M_353_, lean_object* v_G_354_, lean_object* v_inst_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_MonoidHom_instDiv___redArg(v_inst_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instDiv___boxed(lean_object* v_M_358_, lean_object* v_G_359_, lean_object* v_inst_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_MonoidHom_instDiv(v_M_358_, v_G_359_, v_inst_360_, v_inst_361_);
lean_dec_ref(v_inst_360_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instSub___redArg(lean_object* v_inst_363_){
_start:
{
lean_object* v_toSub_364_; lean_object* v___f_365_; 
v_toSub_364_ = lean_ctor_get(v_inst_363_, 2);
lean_inc(v_toSub_364_);
lean_dec_ref(v_inst_363_);
v___f_365_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instSub___redArg___lam__0), 4, 1);
lean_closure_set(v___f_365_, 0, v_toSub_364_);
return v___f_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instSub(lean_object* v_M_366_, lean_object* v_G_367_, lean_object* v_inst_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lp_mathlib_AddMonoidHom_instSub___redArg(v_inst_369_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instSub___boxed(lean_object* v_M_371_, lean_object* v_G_372_, lean_object* v_inst_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib_AddMonoidHom_instSub(v_M_371_, v_G_372_, v_inst_373_, v_inst_374_);
lean_dec_ref(v_inst_373_);
return v_res_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective___redArg(lean_object* v_inst_376_){
_start:
{
lean_inc_ref(v_inst_376_);
return v_inst_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective___redArg___boxed(lean_object* v_inst_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_mathlib_MonoidHom_commGroupOfInjective___redArg(v_inst_377_);
lean_dec_ref(v_inst_377_);
return v_res_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective(lean_object* v_G_379_, lean_object* v_H_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_f_383_, lean_object* v_hf_384_){
_start:
{
lean_inc_ref(v_inst_381_);
return v_inst_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfInjective___boxed(lean_object* v_G_385_, lean_object* v_H_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_f_389_, lean_object* v_hf_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_MonoidHom_commGroupOfInjective(v_G_385_, v_H_386_, v_inst_387_, v_inst_388_, v_f_389_, v_hf_390_);
lean_dec(v_f_389_);
lean_dec_ref(v_inst_388_);
lean_dec_ref(v_inst_387_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective___redArg(lean_object* v_inst_392_){
_start:
{
lean_inc_ref(v_inst_392_);
return v_inst_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective___redArg___boxed(lean_object* v_inst_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_MonoidHom_commGroupOfSurjective___redArg(v_inst_393_);
lean_dec_ref(v_inst_393_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective(lean_object* v_G_395_, lean_object* v_H_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_f_399_, lean_object* v_hf_400_){
_start:
{
lean_inc_ref(v_inst_398_);
return v_inst_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_commGroupOfSurjective___boxed(lean_object* v_G_401_, lean_object* v_H_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_f_405_, lean_object* v_hf_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_MonoidHom_commGroupOfSurjective(v_G_401_, v_H_402_, v_inst_403_, v_inst_404_, v_f_405_, v_hf_406_);
lean_dec(v_f_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
return v_res_407_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
