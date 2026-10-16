// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Opposite public import Mathlib.Algebra.GroupWithZero.Hom public import Mathlib.Algebra.GroupWithZero.Opposite public import Mathlib.Algebra.Notation.Pi.Basic
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
lean_object* lp_mathlib_SMul_comp_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_compFun___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_compFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_compFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toSMulWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toSMulWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toOppositeSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toOppositeSMulWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_natSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_natSMulWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_intSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_intSMulWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulActionWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_compHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_compFun___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_compFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_compFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Function_Injective_smulZeroClass___redArg(v_inst_2_);
lean_dec(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass(lean_object* v_M_4_, lean_object* v_A_5_, lean_object* v_B_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_f_11_, lean_object* v_hf_12_, lean_object* v_smul_13_){
_start:
{
lean_inc(v_inst_10_);
return v_inst_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulZeroClass___boxed(lean_object* v_M_14_, lean_object* v_A_15_, lean_object* v_B_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_f_21_, lean_object* v_hf_22_, lean_object* v_smul_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Function_Injective_smulZeroClass(v_M_14_, v_A_15_, v_B_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_inst_20_, v_f_21_, v_hf_22_, v_smul_23_);
lean_dec(v_f_21_);
lean_dec(v_inst_20_);
lean_dec(v_inst_19_);
lean_dec(v_inst_18_);
lean_dec(v_inst_17_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass___redArg(lean_object* v_inst_25_){
_start:
{
lean_inc(v_inst_25_);
return v_inst_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass___redArg___boxed(lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_ZeroHom_smulZeroClass___redArg(v_inst_26_);
lean_dec(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass(lean_object* v_M_28_, lean_object* v_A_29_, lean_object* v_B_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_f_35_, lean_object* v_smul_36_){
_start:
{
lean_inc(v_inst_34_);
return v_inst_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_smulZeroClass___boxed(lean_object* v_M_37_, lean_object* v_A_38_, lean_object* v_B_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_f_44_, lean_object* v_smul_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_ZeroHom_smulZeroClass(v_M_37_, v_A_38_, v_B_39_, v_inst_40_, v_inst_41_, v_inst_42_, v_inst_43_, v_f_44_, v_smul_45_);
lean_dec(v_f_44_);
lean_dec(v_inst_43_);
lean_dec(v_inst_42_);
lean_dec(v_inst_41_);
lean_dec(v_inst_40_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft___redArg(lean_object* v_inst_47_){
_start:
{
lean_inc(v_inst_47_);
return v_inst_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft___redArg___boxed(lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Function_Surjective_smulZeroClassLeft___redArg(v_inst_48_);
lean_dec(v_inst_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft(lean_object* v_R_50_, lean_object* v_S_51_, lean_object* v_M_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_f_56_, lean_object* v_hf_57_, lean_object* v_hsmul_58_){
_start:
{
lean_inc(v_inst_55_);
return v_inst_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulZeroClassLeft___boxed(lean_object* v_R_59_, lean_object* v_S_60_, lean_object* v_M_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_f_65_, lean_object* v_hf_66_, lean_object* v_hsmul_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Function_Surjective_smulZeroClassLeft(v_R_59_, v_S_60_, v_M_61_, v_inst_62_, v_inst_63_, v_inst_64_, v_f_65_, v_hf_66_, v_hsmul_67_);
lean_dec(v_f_65_);
lean_dec(v_inst_64_);
lean_dec(v_inst_63_);
lean_dec(v_inst_62_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_compFun___redArg(lean_object* v_inst_69_, lean_object* v_f_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_71_, 0, lean_box(0));
lean_closure_set(v___x_71_, 1, lean_box(0));
lean_closure_set(v___x_71_, 2, lean_box(0));
lean_closure_set(v___x_71_, 3, v_inst_69_);
lean_closure_set(v___x_71_, 4, v_f_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_compFun(lean_object* v_M_72_, lean_object* v_N_73_, lean_object* v_A_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_f_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_78_, 0, lean_box(0));
lean_closure_set(v___x_78_, 1, lean_box(0));
lean_closure_set(v___x_78_, 2, lean_box(0));
lean_closure_set(v___x_78_, 3, v_inst_76_);
lean_closure_set(v___x_78_, 4, v_f_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_compFun___boxed(lean_object* v_M_79_, lean_object* v_N_80_, lean_object* v_A_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_f_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_SMulZeroClass_compFun(v_M_79_, v_N_80_, v_A_81_, v_inst_82_, v_inst_83_, v_f_84_);
lean_dec(v_inst_82_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0(lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v_x_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_apply_2(v_inst_86_, v_x_87_, v_x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg(lean_object* v_inst_90_, lean_object* v_x_91_){
_start:
{
lean_object* v___f_92_; 
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_92_, 0, v_inst_90_);
lean_closure_set(v___f_92_, 1, v_x_91_);
return v___f_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom(lean_object* v_M_93_, lean_object* v_A_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_x_97_){
_start:
{
lean_object* v___f_98_; 
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_98_, 0, v_inst_96_);
lean_closure_set(v___f_98_, 1, v_x_97_);
return v___f_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulZeroClass_toZeroHom___boxed(lean_object* v_M_99_, lean_object* v_A_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_x_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_SMulZeroClass_toZeroHom(v_M_99_, v_A_100_, v_inst_101_, v_inst_102_, v_x_103_);
lean_dec(v_inst_101_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toSMulWithZero___redArg___lam__0(lean_object* v_toMul_105_, lean_object* v_x1_106_, lean_object* v_x2_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lean_apply_2(v_toMul_105_, v_x1_106_, v_x2_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toSMulWithZero___redArg(lean_object* v_inst_109_){
_start:
{
lean_object* v_toMul_110_; lean_object* v___f_111_; 
v_toMul_110_ = lean_ctor_get(v_inst_109_, 0);
lean_inc(v_toMul_110_);
lean_dec_ref(v_inst_109_);
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_MulZeroClass_toSMulWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_111_, 0, v_toMul_110_);
return v___f_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toSMulWithZero(lean_object* v_M_u2080_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_MulZeroClass_toSMulWithZero___redArg(v_inst_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toOppositeSMulWithZero___redArg(lean_object* v_inst_115_){
_start:
{
lean_object* v_toMul_116_; lean_object* v___f_117_; 
v_toMul_116_ = lean_ctor_get(v_inst_115_, 0);
lean_inc(v_toMul_116_);
lean_dec_ref(v_inst_115_);
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_117_, 0, v_toMul_116_);
return v___f_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_toOppositeSMulWithZero(lean_object* v_M_u2080_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_MulZeroClass_toOppositeSMulWithZero___redArg(v_inst_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero___redArg(lean_object* v_inst_121_){
_start:
{
lean_inc(v_inst_121_);
return v_inst_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero___redArg___boxed(lean_object* v_inst_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Function_Injective_smulWithZero___redArg(v_inst_122_);
lean_dec(v_inst_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero(lean_object* v_M_u2080_124_, lean_object* v_A_125_, lean_object* v_A_x27_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_f_132_, lean_object* v_hf_133_, lean_object* v_smul_134_){
_start:
{
lean_inc(v_inst_131_);
return v_inst_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_smulWithZero___boxed(lean_object* v_M_u2080_135_, lean_object* v_A_136_, lean_object* v_A_x27_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_f_143_, lean_object* v_hf_144_, lean_object* v_smul_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Function_Injective_smulWithZero(v_M_u2080_135_, v_A_136_, v_A_x27_137_, v_inst_138_, v_inst_139_, v_inst_140_, v_inst_141_, v_inst_142_, v_f_143_, v_hf_144_, v_smul_145_);
lean_dec(v_f_143_);
lean_dec(v_inst_142_);
lean_dec(v_inst_141_);
lean_dec(v_inst_140_);
lean_dec(v_inst_139_);
lean_dec(v_inst_138_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero___redArg(lean_object* v_inst_147_){
_start:
{
lean_inc(v_inst_147_);
return v_inst_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero___redArg___boxed(lean_object* v_inst_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Function_Surjective_smulWithZero___redArg(v_inst_148_);
lean_dec(v_inst_148_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero(lean_object* v_M_u2080_150_, lean_object* v_A_151_, lean_object* v_A_x27_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_f_158_, lean_object* v_hf_159_, lean_object* v_smul_160_){
_start:
{
lean_inc(v_inst_157_);
return v_inst_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_smulWithZero___boxed(lean_object* v_M_u2080_161_, lean_object* v_A_162_, lean_object* v_A_x27_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_f_169_, lean_object* v_hf_170_, lean_object* v_smul_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Function_Surjective_smulWithZero(v_M_u2080_161_, v_A_162_, v_A_x27_163_, v_inst_164_, v_inst_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_f_169_, v_hf_170_, v_smul_171_);
lean_dec(v_f_169_);
lean_dec(v_inst_168_);
lean_dec(v_inst_167_);
lean_dec(v_inst_166_);
lean_dec(v_inst_165_);
lean_dec(v_inst_164_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom___redArg___lam__0(lean_object* v_f_173_, lean_object* v_inst_174_, lean_object* v_x1_175_, lean_object* v_x2_176_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lean_apply_1(v_f_173_, v_x1_175_);
v___x_178_ = lean_apply_2(v_inst_174_, v___x_177_, v_x2_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom___redArg(lean_object* v_inst_179_, lean_object* v_f_180_){
_start:
{
lean_object* v___f_181_; 
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_SMulWithZero_compHom___redArg___lam__0), 4, 2);
lean_closure_set(v___f_181_, 0, v_f_180_);
lean_closure_set(v___f_181_, 1, v_inst_179_);
return v___f_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom(lean_object* v_M_u2080_182_, lean_object* v_M_u2080_x27_183_, lean_object* v_A_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_f_189_){
_start:
{
lean_object* v___f_190_; 
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_SMulWithZero_compHom___redArg___lam__0), 4, 2);
lean_closure_set(v___f_190_, 0, v_f_189_);
lean_closure_set(v___f_190_, 1, v_inst_187_);
return v___f_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulWithZero_compHom___boxed(lean_object* v_M_u2080_191_, lean_object* v_M_u2080_x27_192_, lean_object* v_A_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_f_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_SMulWithZero_compHom(v_M_u2080_191_, v_M_u2080_x27_192_, v_A_193_, v_inst_194_, v_inst_195_, v_inst_196_, v_inst_197_, v_f_198_);
lean_dec(v_inst_197_);
lean_dec(v_inst_195_);
lean_dec(v_inst_194_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_natSMulWithZero___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v_toNSMul_201_; lean_object* v___f_202_; 
v_toNSMul_201_ = lean_ctor_get(v_inst_200_, 2);
lean_inc(v_toNSMul_201_);
lean_dec_ref(v_inst_200_);
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_202_, 0, v_toNSMul_201_);
return v___f_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_natSMulWithZero(lean_object* v_A_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_AddMonoid_natSMulWithZero___redArg(v_inst_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_intSMulWithZero___redArg(lean_object* v_inst_206_){
_start:
{
lean_object* v_toZSMul_207_; lean_object* v___f_208_; 
v_toZSMul_207_ = lean_ctor_get(v_inst_206_, 3);
lean_inc(v_toZSMul_207_);
lean_dec_ref(v_inst_206_);
v___f_208_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_208_, 0, v_toZSMul_207_);
return v___f_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGroup_intSMulWithZero(lean_object* v_A_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_AddGroup_intSMulWithZero___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero___redArg(lean_object* v_m_212_){
_start:
{
lean_inc(v_m_212_);
return v_m_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero___redArg___boxed(lean_object* v_m_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_MulActionWithZero_toSMulWithZero___redArg(v_m_213_);
lean_dec(v_m_213_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero(lean_object* v_M_u2080_215_, lean_object* v_A_216_, lean_object* v_x_217_, lean_object* v_x_218_, lean_object* v_m_219_){
_start:
{
lean_inc(v_m_219_);
return v_m_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_toSMulWithZero___boxed(lean_object* v_M_u2080_220_, lean_object* v_A_221_, lean_object* v_x_222_, lean_object* v_x_223_, lean_object* v_m_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_MulActionWithZero_toSMulWithZero(v_M_u2080_220_, v_A_221_, v_x_222_, v_x_223_, v_m_224_);
lean_dec(v_m_224_);
lean_dec(v_x_223_);
lean_dec_ref(v_x_222_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulActionWithZero___redArg(lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_227_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_226_);
v___x_228_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_227_);
v___x_229_ = lp_mathlib_MulZeroClass_toSMulWithZero___redArg(v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulActionWithZero(lean_object* v_M_u2080_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_MonoidWithZero_toMulActionWithZero___redArg(v_inst_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero___redArg(lean_object* v_inst_233_){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_234_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_233_);
v___x_235_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_234_);
v___x_236_ = lp_mathlib_MulZeroClass_toOppositeSMulWithZero___redArg(v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero(lean_object* v_M_u2080_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_MonoidWithZero_toOppositeMulActionWithZero___redArg(v_inst_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero___redArg(lean_object* v_inst_240_){
_start:
{
lean_inc(v_inst_240_);
return v_inst_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero___redArg___boxed(lean_object* v_inst_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_Function_Injective_mulActionWithZero___redArg(v_inst_241_);
lean_dec(v_inst_241_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero(lean_object* v_M_u2080_243_, lean_object* v_A_244_, lean_object* v_A_x27_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_f_251_, lean_object* v_hf_252_, lean_object* v_smul_253_){
_start:
{
lean_inc(v_inst_250_);
return v_inst_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulActionWithZero___boxed(lean_object* v_M_u2080_254_, lean_object* v_A_255_, lean_object* v_A_x27_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_f_262_, lean_object* v_hf_263_, lean_object* v_smul_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_Function_Injective_mulActionWithZero(v_M_u2080_254_, v_A_255_, v_A_x27_256_, v_inst_257_, v_inst_258_, v_inst_259_, v_inst_260_, v_inst_261_, v_f_262_, v_hf_263_, v_smul_264_);
lean_dec(v_f_262_);
lean_dec(v_inst_261_);
lean_dec(v_inst_260_);
lean_dec(v_inst_259_);
lean_dec(v_inst_258_);
lean_dec_ref(v_inst_257_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero___redArg(lean_object* v_inst_266_){
_start:
{
lean_inc(v_inst_266_);
return v_inst_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero___redArg___boxed(lean_object* v_inst_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Function_Surjective_mulActionWithZero___redArg(v_inst_267_);
lean_dec(v_inst_267_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero(lean_object* v_M_u2080_269_, lean_object* v_A_270_, lean_object* v_A_x27_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_f_277_, lean_object* v_hf_278_, lean_object* v_smul_279_){
_start:
{
lean_inc(v_inst_276_);
return v_inst_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionWithZero___boxed(lean_object* v_M_u2080_280_, lean_object* v_A_281_, lean_object* v_A_x27_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_f_288_, lean_object* v_hf_289_, lean_object* v_smul_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Function_Surjective_mulActionWithZero(v_M_u2080_280_, v_A_281_, v_A_x27_282_, v_inst_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_f_288_, v_hf_289_, v_smul_290_);
lean_dec(v_f_288_);
lean_dec(v_inst_287_);
lean_dec(v_inst_286_);
lean_dec(v_inst_285_);
lean_dec(v_inst_284_);
lean_dec_ref(v_inst_283_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_compHom___redArg(lean_object* v_inst_292_, lean_object* v_f_293_){
_start:
{
lean_object* v___f_294_; 
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_SMulWithZero_compHom___redArg___lam__0), 4, 2);
lean_closure_set(v___f_294_, 0, v_f_293_);
lean_closure_set(v___f_294_, 1, v_inst_292_);
return v___f_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_compHom(lean_object* v_M_u2080_295_, lean_object* v_M_u2080_x27_296_, lean_object* v_A_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_f_302_){
_start:
{
lean_object* v___f_303_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_SMulWithZero_compHom___redArg___lam__0), 4, 2);
lean_closure_set(v___f_303_, 0, v_f_302_);
lean_closure_set(v___f_303_, 1, v_inst_301_);
return v___f_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionWithZero_compHom___boxed(lean_object* v_M_u2080_304_, lean_object* v_M_u2080_x27_305_, lean_object* v_A_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_f_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_MulActionWithZero_compHom(v_M_u2080_304_, v_M_u2080_x27_305_, v_A_306_, v_inst_307_, v_inst_308_, v_inst_309_, v_inst_310_, v_f_311_);
lean_dec(v_inst_309_);
lean_dec_ref(v_inst_308_);
lean_dec_ref(v_inst_307_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul___redArg(lean_object* v_inst_313_){
_start:
{
lean_inc(v_inst_313_);
return v_inst_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul___redArg___boxed(lean_object* v_inst_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_Function_Injective_distribSMul___redArg(v_inst_314_);
lean_dec(v_inst_314_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul(lean_object* v_M_316_, lean_object* v_A_317_, lean_object* v_B_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_f_323_, lean_object* v_hf_324_, lean_object* v_smul_325_){
_start:
{
lean_inc(v_inst_322_);
return v_inst_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribSMul___boxed(lean_object* v_M_326_, lean_object* v_A_327_, lean_object* v_B_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_f_333_, lean_object* v_hf_334_, lean_object* v_smul_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Function_Injective_distribSMul(v_M_326_, v_A_327_, v_B_328_, v_inst_329_, v_inst_330_, v_inst_331_, v_inst_332_, v_f_333_, v_hf_334_, v_smul_335_);
lean_dec(v_f_333_);
lean_dec(v_inst_332_);
lean_dec_ref(v_inst_331_);
lean_dec(v_inst_330_);
lean_dec_ref(v_inst_329_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul___redArg(lean_object* v_inst_337_){
_start:
{
lean_inc(v_inst_337_);
return v_inst_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul___redArg___boxed(lean_object* v_inst_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_Function_Surjective_distribSMul___redArg(v_inst_338_);
lean_dec(v_inst_338_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul(lean_object* v_M_340_, lean_object* v_A_341_, lean_object* v_B_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_f_347_, lean_object* v_hf_348_, lean_object* v_smul_349_){
_start:
{
lean_inc(v_inst_346_);
return v_inst_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMul___boxed(lean_object* v_M_350_, lean_object* v_A_351_, lean_object* v_B_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_f_357_, lean_object* v_hf_358_, lean_object* v_smul_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_Function_Surjective_distribSMul(v_M_350_, v_A_351_, v_B_352_, v_inst_353_, v_inst_354_, v_inst_355_, v_inst_356_, v_f_357_, v_hf_358_, v_smul_359_);
lean_dec(v_f_357_);
lean_dec(v_inst_356_);
lean_dec_ref(v_inst_355_);
lean_dec(v_inst_354_);
lean_dec_ref(v_inst_353_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft___redArg(lean_object* v_inst_361_){
_start:
{
lean_inc(v_inst_361_);
return v_inst_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft___redArg___boxed(lean_object* v_inst_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_Function_Surjective_distribSMulLeft___redArg(v_inst_362_);
lean_dec(v_inst_362_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft(lean_object* v_R_364_, lean_object* v_S_365_, lean_object* v_M_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_f_370_, lean_object* v_hf_371_, lean_object* v_hsmul_372_){
_start:
{
lean_inc(v_inst_369_);
return v_inst_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribSMulLeft___boxed(lean_object* v_R_373_, lean_object* v_S_374_, lean_object* v_M_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_f_379_, lean_object* v_hf_380_, lean_object* v_hsmul_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_Function_Surjective_distribSMulLeft(v_R_373_, v_S_374_, v_M_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_f_379_, v_hf_380_, v_hsmul_381_);
lean_dec(v_f_379_);
lean_dec(v_inst_378_);
lean_dec(v_inst_377_);
lean_dec_ref(v_inst_376_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_compFun___redArg(lean_object* v_inst_383_, lean_object* v_f_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_385_, 0, lean_box(0));
lean_closure_set(v___x_385_, 1, lean_box(0));
lean_closure_set(v___x_385_, 2, lean_box(0));
lean_closure_set(v___x_385_, 3, v_inst_383_);
lean_closure_set(v___x_385_, 4, v_f_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_compFun(lean_object* v_M_386_, lean_object* v_N_387_, lean_object* v_A_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_f_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_392_, 0, lean_box(0));
lean_closure_set(v___x_392_, 1, lean_box(0));
lean_closure_set(v___x_392_, 2, lean_box(0));
lean_closure_set(v___x_392_, 3, v_inst_390_);
lean_closure_set(v___x_392_, 4, v_f_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_compFun___boxed(lean_object* v_M_393_, lean_object* v_N_394_, lean_object* v_A_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_f_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_mathlib_DistribSMul_compFun(v_M_393_, v_N_394_, v_A_395_, v_inst_396_, v_inst_397_, v_f_398_);
lean_dec_ref(v_inst_396_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toAddMonoidHom___redArg(lean_object* v_inst_400_, lean_object* v_x_401_){
_start:
{
lean_object* v___f_402_; 
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_402_, 0, v_inst_400_);
lean_closure_set(v___f_402_, 1, v_x_401_);
return v___f_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toAddMonoidHom(lean_object* v_M_403_, lean_object* v_A_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_x_407_){
_start:
{
lean_object* v___f_408_; 
v___f_408_ = lean_alloc_closure((void*)(lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_408_, 0, v_inst_406_);
lean_closure_set(v___f_408_, 1, v_x_407_);
return v___f_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribSMul_toAddMonoidHom___boxed(lean_object* v_M_409_, lean_object* v_A_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_x_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_DistribSMul_toAddMonoidHom(v_M_409_, v_A_410_, v_inst_411_, v_inst_412_, v_x_413_);
lean_dec_ref(v_inst_411_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul___redArg(lean_object* v_inst_415_){
_start:
{
lean_inc(v_inst_415_);
return v_inst_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul___redArg___boxed(lean_object* v_inst_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_mathlib_DistribMulAction_toDistribSMul___redArg(v_inst_416_);
lean_dec(v_inst_416_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul(lean_object* v_M_418_, lean_object* v_A_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_inst_422_){
_start:
{
lean_inc(v_inst_422_);
return v_inst_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toDistribSMul___boxed(lean_object* v_M_423_, lean_object* v_A_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_mathlib_DistribMulAction_toDistribSMul(v_M_423_, v_A_424_, v_inst_425_, v_inst_426_, v_inst_427_);
lean_dec(v_inst_427_);
lean_dec_ref(v_inst_426_);
lean_dec_ref(v_inst_425_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction___redArg(lean_object* v_inst_429_){
_start:
{
lean_inc(v_inst_429_);
return v_inst_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction___redArg___boxed(lean_object* v_inst_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_mathlib_Function_Injective_distribMulAction___redArg(v_inst_430_);
lean_dec(v_inst_430_);
return v_res_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction(lean_object* v_M_432_, lean_object* v_A_433_, lean_object* v_B_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_f_440_, lean_object* v_hf_441_, lean_object* v_smul_442_){
_start:
{
lean_inc(v_inst_439_);
return v_inst_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribMulAction___boxed(lean_object* v_M_443_, lean_object* v_A_444_, lean_object* v_B_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_f_451_, lean_object* v_hf_452_, lean_object* v_smul_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_mathlib_Function_Injective_distribMulAction(v_M_443_, v_A_444_, v_B_445_, v_inst_446_, v_inst_447_, v_inst_448_, v_inst_449_, v_inst_450_, v_f_451_, v_hf_452_, v_smul_453_);
lean_dec(v_f_451_);
lean_dec(v_inst_450_);
lean_dec_ref(v_inst_449_);
lean_dec(v_inst_448_);
lean_dec_ref(v_inst_447_);
lean_dec_ref(v_inst_446_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction___redArg(lean_object* v_inst_455_){
_start:
{
lean_inc(v_inst_455_);
return v_inst_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction___redArg___boxed(lean_object* v_inst_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_Function_Surjective_distribMulAction___redArg(v_inst_456_);
lean_dec(v_inst_456_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction(lean_object* v_M_458_, lean_object* v_A_459_, lean_object* v_B_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_f_466_, lean_object* v_hf_467_, lean_object* v_smul_468_){
_start:
{
lean_inc(v_inst_465_);
return v_inst_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distribMulAction___boxed(lean_object* v_M_469_, lean_object* v_A_470_, lean_object* v_B_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_f_477_, lean_object* v_hf_478_, lean_object* v_smul_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_Function_Surjective_distribMulAction(v_M_469_, v_A_470_, v_B_471_, v_inst_472_, v_inst_473_, v_inst_474_, v_inst_475_, v_inst_476_, v_f_477_, v_hf_478_, v_smul_479_);
lean_dec(v_f_477_);
lean_dec(v_inst_476_);
lean_dec_ref(v_inst_475_);
lean_dec(v_inst_474_);
lean_dec_ref(v_inst_473_);
lean_dec_ref(v_inst_472_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(lean_object* v_inst_481_, lean_object* v_inst_482_){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_481_);
v___x_484_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toAddMonoidHom___boxed), 5, 4);
lean_closure_set(v___x_484_, 0, lean_box(0));
lean_closure_set(v___x_484_, 1, lean_box(0));
lean_closure_set(v___x_484_, 2, v___x_483_);
lean_closure_set(v___x_484_, 3, v_inst_482_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg___boxed(lean_object* v_inst_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(v_inst_485_, v_inst_486_);
lean_dec_ref(v_inst_485_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd(lean_object* v_M_488_, lean_object* v_A_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_inst_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_DistribMulAction_toAddMonoidEnd___redArg(v_inst_491_, v_inst_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddMonoidEnd___boxed(lean_object* v_M_494_, lean_object* v_A_495_, lean_object* v_inst_496_, lean_object* v_inst_497_, lean_object* v_inst_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_DistribMulAction_toAddMonoidEnd(v_M_494_, v_A_495_, v_inst_496_, v_inst_497_, v_inst_498_);
lean_dec_ref(v_inst_497_);
lean_dec_ref(v_inst_496_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass___redArg(lean_object* v_inst_500_){
_start:
{
lean_inc(v_inst_500_);
return v_inst_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass___redArg___boxed(lean_object* v_inst_501_){
_start:
{
lean_object* v_res_502_; 
v_res_502_ = lp_mathlib_instSMulZeroClass___redArg(v_inst_501_);
lean_dec(v_inst_501_);
return v_res_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass(lean_object* v_00_u03b1_503_, lean_object* v_00_u03b2_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_){
_start:
{
lean_inc(v_inst_507_);
return v_inst_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZeroClass___boxed(lean_object* v_00_u03b1_508_, lean_object* v_00_u03b2_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_){
_start:
{
lean_object* v_res_513_; 
v_res_513_ = lp_mathlib_instSMulZeroClass(v_00_u03b1_508_, v_00_u03b2_509_, v_inst_510_, v_inst_511_, v_inst_512_);
lean_dec(v_inst_512_);
lean_dec_ref(v_inst_511_);
lean_dec_ref(v_inst_510_);
return v_res_513_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
