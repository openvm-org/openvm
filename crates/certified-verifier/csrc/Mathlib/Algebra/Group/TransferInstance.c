// Lean compiler output
// Module: Mathlib.Algebra.Group.TransferInstance
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Data.Fintype.Basic
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
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_instance__transfer__via__equivalence;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_LibraryNote_instance__transfer__via__equivalence(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one___redArg(lean_object* v_e_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v_invFun_4_; lean_object* v___x_5_; 
v_invFun_4_ = lean_ctor_get(v_e_2_, 1);
lean_inc(v_invFun_4_);
lean_dec_ref(v_e_2_);
v___x_5_ = lean_apply_1(v_invFun_4_, v_inst_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_one(lean_object* v_00_u03b1_6_, lean_object* v_00_u03b2_7_, lean_object* v_e_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_invFun_10_; lean_object* v___x_11_; 
v_invFun_10_ = lean_ctor_get(v_e_8_, 1);
lean_inc(v_invFun_10_);
lean_dec_ref(v_e_8_);
v___x_11_ = lean_apply_1(v_invFun_10_, v_inst_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero___redArg(lean_object* v_e_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v_invFun_14_; lean_object* v___x_15_; 
v_invFun_14_ = lean_ctor_get(v_e_12_, 1);
lean_inc(v_invFun_14_);
lean_dec_ref(v_e_12_);
v___x_15_ = lean_apply_1(v_invFun_14_, v_inst_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_zero(lean_object* v_00_u03b1_16_, lean_object* v_00_u03b2_17_, lean_object* v_e_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Equiv_zero___redArg(v_e_18_, v_inst_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg___lam__0(lean_object* v_e_21_, lean_object* v_inst_22_, lean_object* v_x_23_, lean_object* v_y_24_){
_start:
{
lean_object* v_toFun_25_; lean_object* v_invFun_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v_toFun_25_ = lean_ctor_get(v_e_21_, 0);
lean_inc_n(v_toFun_25_, 2);
v_invFun_26_ = lean_ctor_get(v_e_21_, 1);
lean_inc(v_invFun_26_);
lean_dec_ref(v_e_21_);
v___x_27_ = lean_apply_1(v_toFun_25_, v_x_23_);
v___x_28_ = lean_apply_1(v_toFun_25_, v_y_24_);
v___x_29_ = lean_apply_2(v_inst_22_, v___x_27_, v___x_28_);
v___x_30_ = lean_apply_1(v_invFun_26_, v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul___redArg(lean_object* v_e_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_33_, 0, v_e_31_);
lean_closure_set(v___f_33_, 1, v_inst_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mul(lean_object* v_00_u03b1_34_, lean_object* v_00_u03b2_35_, lean_object* v_e_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_38_, 0, v_e_36_);
lean_closure_set(v___f_38_, 1, v_inst_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add___redArg(lean_object* v_e_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___f_41_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_41_, 0, v_e_39_);
lean_closure_set(v___f_41_, 1, v_inst_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_add(lean_object* v_00_u03b1_42_, lean_object* v_00_u03b2_43_, lean_object* v_e_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___f_46_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_46_, 0, v_e_44_);
lean_closure_set(v___f_46_, 1, v_inst_45_);
return v___f_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div___redArg(lean_object* v_e_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___f_49_; 
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_49_, 0, v_e_47_);
lean_closure_set(v___f_49_, 1, v_inst_48_);
return v___f_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_div(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_, lean_object* v_e_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___f_54_; 
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_54_, 0, v_e_52_);
lean_closure_set(v___f_54_, 1, v_inst_53_);
return v___f_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub___redArg(lean_object* v_e_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___f_57_; 
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_57_, 0, v_e_55_);
lean_closure_set(v___f_57_, 1, v_inst_56_);
return v___f_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sub(lean_object* v_00_u03b1_58_, lean_object* v_00_u03b2_59_, lean_object* v_e_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___f_62_; 
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_62_, 0, v_e_60_);
lean_closure_set(v___f_62_, 1, v_inst_61_);
return v___f_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg___lam__0(lean_object* v_e_63_, lean_object* v_inst_64_, lean_object* v_x_65_){
_start:
{
lean_object* v_toFun_66_; lean_object* v_invFun_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v_toFun_66_ = lean_ctor_get(v_e_63_, 0);
lean_inc(v_toFun_66_);
v_invFun_67_ = lean_ctor_get(v_e_63_, 1);
lean_inc(v_invFun_67_);
lean_dec_ref(v_e_63_);
v___x_68_ = lean_apply_1(v_toFun_66_, v_x_65_);
v___x_69_ = lean_apply_1(v_inst_64_, v___x_68_);
v___x_70_ = lean_apply_1(v_invFun_67_, v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv___redArg(lean_object* v_e_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___f_73_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_73_, 0, v_e_71_);
lean_closure_set(v___f_73_, 1, v_inst_72_);
return v___f_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Inv(lean_object* v_00_u03b1_74_, lean_object* v_00_u03b2_75_, lean_object* v_e_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_78_, 0, v_e_76_);
lean_closure_set(v___f_78_, 1, v_inst_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg___redArg(lean_object* v_e_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___f_81_; 
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_81_, 0, v_e_79_);
lean_closure_set(v___f_81_, 1, v_inst_80_);
return v___f_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Neg(lean_object* v_00_u03b1_82_, lean_object* v_00_u03b2_83_, lean_object* v_e_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___f_86_; 
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_86_, 0, v_e_84_);
lean_closure_set(v___f_86_, 1, v_inst_85_);
return v___f_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg___lam__0(lean_object* v_e_87_, lean_object* v_inst_88_, lean_object* v_x_89_, lean_object* v_n_90_){
_start:
{
lean_object* v_toFun_91_; lean_object* v_invFun_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v_toFun_91_ = lean_ctor_get(v_e_87_, 0);
lean_inc(v_toFun_91_);
v_invFun_92_ = lean_ctor_get(v_e_87_, 1);
lean_inc(v_invFun_92_);
lean_dec_ref(v_e_87_);
v___x_93_ = lean_apply_1(v_toFun_91_, v_x_89_);
v___x_94_ = lean_apply_2(v_inst_88_, v___x_93_, v_n_90_);
v___x_95_ = lean_apply_1(v_invFun_92_, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow___redArg(lean_object* v_e_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___f_98_; 
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_pow___redArg___lam__0), 4, 2);
lean_closure_set(v___f_98_, 0, v_e_96_);
lean_closure_set(v___f_98_, 1, v_inst_97_);
return v___f_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pow(lean_object* v_M_99_, lean_object* v_00_u03b1_100_, lean_object* v_00_u03b2_101_, lean_object* v_e_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; 
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_pow___redArg___lam__0), 4, 2);
lean_closure_set(v___f_104_, 0, v_e_102_);
lean_closure_set(v___f_104_, 1, v_inst_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg___lam__0(lean_object* v_e_105_, lean_object* v_inst_106_, lean_object* v_n_107_, lean_object* v_x_108_){
_start:
{
lean_object* v_toFun_109_; lean_object* v_invFun_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v_toFun_109_ = lean_ctor_get(v_e_105_, 0);
lean_inc(v_toFun_109_);
v_invFun_110_ = lean_ctor_get(v_e_105_, 1);
lean_inc(v_invFun_110_);
lean_dec_ref(v_e_105_);
v___x_111_ = lean_apply_1(v_toFun_109_, v_x_108_);
v___x_112_ = lean_apply_2(v_inst_106_, v_n_107_, v___x_111_);
v___x_113_ = lean_apply_1(v_invFun_110_, v___x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul___redArg(lean_object* v_e_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___f_116_; 
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_116_, 0, v_e_114_);
lean_closure_set(v___f_116_, 1, v_inst_115_);
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_smul(lean_object* v_M_117_, lean_object* v_00_u03b1_118_, lean_object* v_00_u03b2_119_, lean_object* v_e_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___f_122_; 
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_122_, 0, v_e_120_);
lean_closure_set(v___f_122_, 1, v_inst_121_);
return v___f_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd___redArg(lean_object* v_e_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___f_125_; 
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_125_, 0, v_e_123_);
lean_closure_set(v___f_125_, 1, v_inst_124_);
return v___f_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_vadd(lean_object* v_M_126_, lean_object* v_00_u03b1_127_, lean_object* v_00_u03b2_128_, lean_object* v_e_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v___f_131_; 
v___f_131_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_131_, 0, v_e_129_);
lean_closure_set(v___f_131_, 1, v_inst_130_);
return v___f_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg(lean_object* v_e_132_){
_start:
{
lean_inc_ref(v_e_132_);
return v_e_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___redArg___boxed(lean_object* v_e_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Equiv_mulEquiv___redArg(v_e_133_);
lean_dec_ref(v_e_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv(lean_object* v_00_u03b1_135_, lean_object* v_00_u03b2_136_, lean_object* v_e_137_, lean_object* v_inst_138_){
_start:
{
lean_inc_ref(v_e_137_);
return v_e_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulEquiv___boxed(lean_object* v_00_u03b1_139_, lean_object* v_00_u03b2_140_, lean_object* v_e_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Equiv_mulEquiv(v_00_u03b1_139_, v_00_u03b2_140_, v_e_141_, v_inst_142_);
lean_dec(v_inst_142_);
lean_dec_ref(v_e_141_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg(lean_object* v_e_144_){
_start:
{
lean_inc_ref(v_e_144_);
return v_e_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___redArg___boxed(lean_object* v_e_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Equiv_addEquiv___redArg(v_e_145_);
lean_dec_ref(v_e_145_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv(lean_object* v_00_u03b1_147_, lean_object* v_00_u03b2_148_, lean_object* v_e_149_, lean_object* v_inst_150_){
_start:
{
lean_inc_ref(v_e_149_);
return v_e_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addEquiv___boxed(lean_object* v_00_u03b1_151_, lean_object* v_00_u03b2_152_, lean_object* v_e_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_Equiv_addEquiv(v_00_u03b1_151_, v_00_u03b2_152_, v_e_153_, v_inst_154_);
lean_dec(v_inst_154_);
lean_dec_ref(v_e_153_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup___redArg(lean_object* v_e_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v_mul_158_; 
v_mul_158_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_158_, 0, v_e_156_);
lean_closure_set(v_mul_158_, 1, v_inst_157_);
return v_mul_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semigroup(lean_object* v_00_u03b1_159_, lean_object* v_00_u03b2_160_, lean_object* v_e_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v_mul_163_; 
v_mul_163_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_163_, 0, v_e_161_);
lean_closure_set(v_mul_163_, 1, v_inst_162_);
return v_mul_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup___redArg(lean_object* v_e_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v_mul_166_; 
v_mul_166_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_166_, 0, v_e_164_);
lean_closure_set(v_mul_166_, 1, v_inst_165_);
return v_mul_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addSemigroup(lean_object* v_00_u03b1_167_, lean_object* v_00_u03b2_168_, lean_object* v_e_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v_mul_171_; 
v_mul_171_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_171_, 0, v_e_169_);
lean_closure_set(v_mul_171_, 1, v_inst_170_);
return v_mul_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup___redArg(lean_object* v_e_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_mul_174_; 
v_mul_174_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_174_, 0, v_e_172_);
lean_closure_set(v_mul_174_, 1, v_inst_173_);
return v_mul_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commSemigroup(lean_object* v_00_u03b1_175_, lean_object* v_00_u03b2_176_, lean_object* v_e_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v_mul_179_; 
v_mul_179_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_179_, 0, v_e_177_);
lean_closure_set(v_mul_179_, 1, v_inst_178_);
return v_mul_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup___redArg(lean_object* v_e_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_mul_182_; 
v_mul_182_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_182_, 0, v_e_180_);
lean_closure_set(v_mul_182_, 1, v_inst_181_);
return v_mul_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommSemigroup(lean_object* v_00_u03b1_183_, lean_object* v_00_u03b2_184_, lean_object* v_e_185_, lean_object* v_inst_186_){
_start:
{
lean_object* v_mul_187_; 
v_mul_187_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_187_, 0, v_e_185_);
lean_closure_set(v_mul_187_, 1, v_inst_186_);
return v_mul_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg___lam__0(lean_object* v_toFun_188_, lean_object* v_toMul_189_, lean_object* v_invFun_190_, lean_object* v_x_191_, lean_object* v_y_192_){
_start:
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
lean_inc(v_toFun_188_);
v___x_193_ = lean_apply_1(v_toFun_188_, v_x_191_);
v___x_194_ = lean_apply_1(v_toFun_188_, v_y_192_);
v___x_195_ = lean_apply_2(v_toMul_189_, v___x_193_, v___x_194_);
v___x_196_ = lean_apply_1(v_invFun_190_, v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass___redArg(lean_object* v_e_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; lean_object* v_toOne_200_; lean_object* v_toMul_201_; lean_object* v_toFun_202_; lean_object* v_invFun_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_212_; 
v___x_199_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_198_);
v_toOne_200_ = lean_ctor_get(v___x_199_, 0);
lean_inc(v_toOne_200_);
v_toMul_201_ = lean_ctor_get(v___x_199_, 1);
lean_inc(v_toMul_201_);
lean_dec_ref(v___x_199_);
v_toFun_202_ = lean_ctor_get(v_e_197_, 0);
v_invFun_203_ = lean_ctor_get(v_e_197_, 1);
v_isSharedCheck_212_ = !lean_is_exclusive(v_e_197_);
if (v_isSharedCheck_212_ == 0)
{
v___x_205_ = v_e_197_;
v_isShared_206_ = v_isSharedCheck_212_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_invFun_203_);
lean_inc(v_toFun_202_);
lean_dec(v_e_197_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_212_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v_mul_207_; lean_object* v_one_208_; lean_object* v___x_210_; 
lean_inc(v_invFun_203_);
v_mul_207_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_207_, 0, v_toFun_202_);
lean_closure_set(v_mul_207_, 1, v_toMul_201_);
lean_closure_set(v_mul_207_, 2, v_invFun_203_);
v_one_208_ = lean_apply_1(v_invFun_203_, v_toOne_200_);
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 1, v_mul_207_);
lean_ctor_set(v___x_205_, 0, v_one_208_);
v___x_210_ = v___x_205_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_one_208_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v_mul_207_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulOneClass(lean_object* v_00_u03b1_213_, lean_object* v_00_u03b2_214_, lean_object* v_e_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; lean_object* v_toOne_218_; lean_object* v_toMul_219_; lean_object* v_toFun_220_; lean_object* v_invFun_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_230_; 
v___x_217_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_216_);
v_toOne_218_ = lean_ctor_get(v___x_217_, 0);
lean_inc(v_toOne_218_);
v_toMul_219_ = lean_ctor_get(v___x_217_, 1);
lean_inc(v_toMul_219_);
lean_dec_ref(v___x_217_);
v_toFun_220_ = lean_ctor_get(v_e_215_, 0);
v_invFun_221_ = lean_ctor_get(v_e_215_, 1);
v_isSharedCheck_230_ = !lean_is_exclusive(v_e_215_);
if (v_isSharedCheck_230_ == 0)
{
v___x_223_ = v_e_215_;
v_isShared_224_ = v_isSharedCheck_230_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_invFun_221_);
lean_inc(v_toFun_220_);
lean_dec(v_e_215_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_230_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v_mul_225_; lean_object* v_one_226_; lean_object* v___x_228_; 
lean_inc(v_invFun_221_);
v_mul_225_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_225_, 0, v_toFun_220_);
lean_closure_set(v_mul_225_, 1, v_toMul_219_);
lean_closure_set(v_mul_225_, 2, v_invFun_221_);
v_one_226_ = lean_apply_1(v_invFun_221_, v_toOne_218_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 1, v_mul_225_);
lean_ctor_set(v___x_223_, 0, v_one_226_);
v___x_228_ = v___x_223_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_one_226_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v_mul_225_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass___redArg(lean_object* v_e_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; lean_object* v_toZero_234_; lean_object* v_toAdd_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_244_; 
v___x_233_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_232_);
v_toZero_234_ = lean_ctor_get(v___x_233_, 0);
v_toAdd_235_ = lean_ctor_get(v___x_233_, 1);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_244_ == 0)
{
v___x_237_ = v___x_233_;
v_isShared_238_ = v_isSharedCheck_244_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_toAdd_235_);
lean_inc(v_toZero_234_);
lean_dec(v___x_233_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_244_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v_one_239_; lean_object* v_mul_240_; lean_object* v___x_242_; 
lean_inc_ref(v_e_231_);
v_one_239_ = lp_mathlib_Equiv_zero___redArg(v_e_231_, v_toZero_234_);
v_mul_240_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_240_, 0, v_e_231_);
lean_closure_set(v_mul_240_, 1, v_toAdd_235_);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 1, v_mul_240_);
lean_ctor_set(v___x_237_, 0, v_one_239_);
v___x_242_ = v___x_237_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_one_239_);
lean_ctor_set(v_reuseFailAlloc_243_, 1, v_mul_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addZeroClass(lean_object* v_00_u03b1_245_, lean_object* v_00_u03b2_246_, lean_object* v_e_247_, lean_object* v_inst_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_Equiv_addZeroClass___redArg(v_e_247_, v_inst_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg___lam__1(lean_object* v_toFun_250_, lean_object* v_toNPow_251_, lean_object* v_invFun_252_, lean_object* v_n_253_, lean_object* v_x_254_){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_255_ = lean_apply_1(v_toFun_250_, v_x_254_);
v___x_256_ = lean_apply_2(v_toNPow_251_, v_n_253_, v___x_255_);
v___x_257_ = lean_apply_1(v_invFun_252_, v___x_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid___redArg(lean_object* v_e_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v_toOne_262_; lean_object* v_toMul_263_; lean_object* v_toFun_264_; lean_object* v_invFun_265_; lean_object* v_toNPow_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_276_; 
v___x_260_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_259_);
v___x_261_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_260_);
v_toOne_262_ = lean_ctor_get(v___x_261_, 0);
lean_inc(v_toOne_262_);
v_toMul_263_ = lean_ctor_get(v___x_261_, 1);
lean_inc(v_toMul_263_);
lean_dec_ref(v___x_261_);
v_toFun_264_ = lean_ctor_get(v_e_258_, 0);
lean_inc(v_toFun_264_);
v_invFun_265_ = lean_ctor_get(v_e_258_, 1);
lean_inc(v_invFun_265_);
lean_dec_ref(v_e_258_);
v_toNPow_266_ = lean_ctor_get(v_inst_259_, 2);
v_isSharedCheck_276_ = !lean_is_exclusive(v_inst_259_);
if (v_isSharedCheck_276_ == 0)
{
lean_object* v_unused_277_; lean_object* v_unused_278_; 
v_unused_277_ = lean_ctor_get(v_inst_259_, 1);
lean_dec(v_unused_277_);
v_unused_278_ = lean_ctor_get(v_inst_259_, 0);
lean_dec(v_unused_278_);
v___x_268_ = v_inst_259_;
v_isShared_269_ = v_isSharedCheck_276_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_toNPow_266_);
lean_dec(v_inst_259_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_276_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v_mul_270_; lean_object* v_one_271_; lean_object* v___f_272_; lean_object* v___x_274_; 
lean_inc_n(v_invFun_265_, 2);
lean_inc(v_toFun_264_);
v_mul_270_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_270_, 0, v_toFun_264_);
lean_closure_set(v_mul_270_, 1, v_toMul_263_);
lean_closure_set(v_mul_270_, 2, v_invFun_265_);
v_one_271_ = lean_apply_1(v_invFun_265_, v_toOne_262_);
v___f_272_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_272_, 0, v_toFun_264_);
lean_closure_set(v___f_272_, 1, v_toNPow_266_);
lean_closure_set(v___f_272_, 2, v_invFun_265_);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 2, v___f_272_);
lean_ctor_set(v___x_268_, 1, v_mul_270_);
lean_ctor_set(v___x_268_, 0, v_one_271_);
v___x_274_ = v___x_268_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v_one_271_);
lean_ctor_set(v_reuseFailAlloc_275_, 1, v_mul_270_);
lean_ctor_set(v_reuseFailAlloc_275_, 2, v___f_272_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
return v___x_274_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_monoid(lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_e_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v_toOne_285_; lean_object* v_toMul_286_; lean_object* v_toFun_287_; lean_object* v_invFun_288_; lean_object* v_toNPow_289_; lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_299_; 
v___x_283_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_282_);
v___x_284_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_283_);
v_toOne_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_toOne_285_);
v_toMul_286_ = lean_ctor_get(v___x_284_, 1);
lean_inc(v_toMul_286_);
lean_dec_ref(v___x_284_);
v_toFun_287_ = lean_ctor_get(v_e_281_, 0);
lean_inc(v_toFun_287_);
v_invFun_288_ = lean_ctor_get(v_e_281_, 1);
lean_inc(v_invFun_288_);
lean_dec_ref(v_e_281_);
v_toNPow_289_ = lean_ctor_get(v_inst_282_, 2);
v_isSharedCheck_299_ = !lean_is_exclusive(v_inst_282_);
if (v_isSharedCheck_299_ == 0)
{
lean_object* v_unused_300_; lean_object* v_unused_301_; 
v_unused_300_ = lean_ctor_get(v_inst_282_, 1);
lean_dec(v_unused_300_);
v_unused_301_ = lean_ctor_get(v_inst_282_, 0);
lean_dec(v_unused_301_);
v___x_291_ = v_inst_282_;
v_isShared_292_ = v_isSharedCheck_299_;
goto v_resetjp_290_;
}
else
{
lean_inc(v_toNPow_289_);
lean_dec(v_inst_282_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_299_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
lean_object* v_mul_293_; lean_object* v_one_294_; lean_object* v___f_295_; lean_object* v___x_297_; 
lean_inc_n(v_invFun_288_, 2);
lean_inc(v_toFun_287_);
v_mul_293_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_293_, 0, v_toFun_287_);
lean_closure_set(v_mul_293_, 1, v_toMul_286_);
lean_closure_set(v_mul_293_, 2, v_invFun_288_);
v_one_294_ = lean_apply_1(v_invFun_288_, v_toOne_285_);
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_295_, 0, v_toFun_287_);
lean_closure_set(v___f_295_, 1, v_toNPow_289_);
lean_closure_set(v___f_295_, 2, v_invFun_288_);
if (v_isShared_292_ == 0)
{
lean_ctor_set(v___x_291_, 2, v___f_295_);
lean_ctor_set(v___x_291_, 1, v_mul_293_);
lean_ctor_set(v___x_291_, 0, v_one_294_);
v___x_297_ = v___x_291_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v_one_294_);
lean_ctor_set(v_reuseFailAlloc_298_, 1, v_mul_293_);
lean_ctor_set(v_reuseFailAlloc_298_, 2, v___f_295_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
return v___x_297_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid___redArg(lean_object* v_e_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v_toZero_306_; lean_object* v_toAdd_307_; lean_object* v_toNSMul_308_; lean_object* v_one_309_; lean_object* v_mul_310_; lean_object* v___f_311_; lean_object* v_pow_312_; lean_object* v___x_313_; 
v___x_304_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_303_);
v___x_305_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_304_);
v_toZero_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_toZero_306_);
v_toAdd_307_ = lean_ctor_get(v___x_305_, 1);
lean_inc(v_toAdd_307_);
lean_dec_ref(v___x_305_);
v_toNSMul_308_ = lean_ctor_get(v_inst_303_, 2);
lean_inc(v_toNSMul_308_);
lean_dec_ref(v_inst_303_);
lean_inc_ref_n(v_e_302_, 2);
v_one_309_ = lp_mathlib_Equiv_zero___redArg(v_e_302_, v_toZero_306_);
v_mul_310_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_310_, 0, v_e_302_);
lean_closure_set(v_mul_310_, 1, v_toAdd_307_);
v___f_311_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_311_, 0, v_toNSMul_308_);
v_pow_312_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v_pow_312_, 0, v_e_302_);
lean_closure_set(v_pow_312_, 1, v___f_311_);
v___x_313_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_mul_310_, v_one_309_, v_pow_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addMonoid(lean_object* v_00_u03b1_314_, lean_object* v_00_u03b2_315_, lean_object* v_e_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Equiv_addMonoid___redArg(v_e_316_, v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid___redArg(lean_object* v_e_319_, lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v_toOne_323_; lean_object* v_toMul_324_; lean_object* v_toFun_325_; lean_object* v_invFun_326_; lean_object* v_toNPow_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_337_; 
v___x_321_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_320_);
v___x_322_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_321_);
v_toOne_323_ = lean_ctor_get(v___x_322_, 0);
lean_inc(v_toOne_323_);
v_toMul_324_ = lean_ctor_get(v___x_322_, 1);
lean_inc(v_toMul_324_);
lean_dec_ref(v___x_322_);
v_toFun_325_ = lean_ctor_get(v_e_319_, 0);
lean_inc(v_toFun_325_);
v_invFun_326_ = lean_ctor_get(v_e_319_, 1);
lean_inc(v_invFun_326_);
lean_dec_ref(v_e_319_);
v_toNPow_327_ = lean_ctor_get(v_inst_320_, 2);
v_isSharedCheck_337_ = !lean_is_exclusive(v_inst_320_);
if (v_isSharedCheck_337_ == 0)
{
lean_object* v_unused_338_; lean_object* v_unused_339_; 
v_unused_338_ = lean_ctor_get(v_inst_320_, 1);
lean_dec(v_unused_338_);
v_unused_339_ = lean_ctor_get(v_inst_320_, 0);
lean_dec(v_unused_339_);
v___x_329_ = v_inst_320_;
v_isShared_330_ = v_isSharedCheck_337_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_toNPow_327_);
lean_dec(v_inst_320_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_337_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v_mul_331_; lean_object* v_one_332_; lean_object* v___f_333_; lean_object* v___x_335_; 
lean_inc_n(v_invFun_326_, 2);
lean_inc(v_toFun_325_);
v_mul_331_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_331_, 0, v_toFun_325_);
lean_closure_set(v_mul_331_, 1, v_toMul_324_);
lean_closure_set(v_mul_331_, 2, v_invFun_326_);
v_one_332_ = lean_apply_1(v_invFun_326_, v_toOne_323_);
v___f_333_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_333_, 0, v_toFun_325_);
lean_closure_set(v___f_333_, 1, v_toNPow_327_);
lean_closure_set(v___f_333_, 2, v_invFun_326_);
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 2, v___f_333_);
lean_ctor_set(v___x_329_, 1, v_mul_331_);
lean_ctor_set(v___x_329_, 0, v_one_332_);
v___x_335_ = v___x_329_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v_one_332_);
lean_ctor_set(v_reuseFailAlloc_336_, 1, v_mul_331_);
lean_ctor_set(v_reuseFailAlloc_336_, 2, v___f_333_);
v___x_335_ = v_reuseFailAlloc_336_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
return v___x_335_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commMonoid(lean_object* v_00_u03b1_340_, lean_object* v_00_u03b2_341_, lean_object* v_e_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v_toOne_346_; lean_object* v_toMul_347_; lean_object* v_toFun_348_; lean_object* v_invFun_349_; lean_object* v_toNPow_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_360_; 
v___x_344_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_343_);
v___x_345_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_344_);
v_toOne_346_ = lean_ctor_get(v___x_345_, 0);
lean_inc(v_toOne_346_);
v_toMul_347_ = lean_ctor_get(v___x_345_, 1);
lean_inc(v_toMul_347_);
lean_dec_ref(v___x_345_);
v_toFun_348_ = lean_ctor_get(v_e_342_, 0);
lean_inc(v_toFun_348_);
v_invFun_349_ = lean_ctor_get(v_e_342_, 1);
lean_inc(v_invFun_349_);
lean_dec_ref(v_e_342_);
v_toNPow_350_ = lean_ctor_get(v_inst_343_, 2);
v_isSharedCheck_360_ = !lean_is_exclusive(v_inst_343_);
if (v_isSharedCheck_360_ == 0)
{
lean_object* v_unused_361_; lean_object* v_unused_362_; 
v_unused_361_ = lean_ctor_get(v_inst_343_, 1);
lean_dec(v_unused_361_);
v_unused_362_ = lean_ctor_get(v_inst_343_, 0);
lean_dec(v_unused_362_);
v___x_352_ = v_inst_343_;
v_isShared_353_ = v_isSharedCheck_360_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_toNPow_350_);
lean_dec(v_inst_343_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_360_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
lean_object* v_mul_354_; lean_object* v_one_355_; lean_object* v___f_356_; lean_object* v___x_358_; 
lean_inc_n(v_invFun_349_, 2);
lean_inc(v_toFun_348_);
v_mul_354_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_354_, 0, v_toFun_348_);
lean_closure_set(v_mul_354_, 1, v_toMul_347_);
lean_closure_set(v_mul_354_, 2, v_invFun_349_);
v_one_355_ = lean_apply_1(v_invFun_349_, v_toOne_346_);
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_356_, 0, v_toFun_348_);
lean_closure_set(v___f_356_, 1, v_toNPow_350_);
lean_closure_set(v___f_356_, 2, v_invFun_349_);
if (v_isShared_353_ == 0)
{
lean_ctor_set(v___x_352_, 2, v___f_356_);
lean_ctor_set(v___x_352_, 1, v_mul_354_);
lean_ctor_set(v___x_352_, 0, v_one_355_);
v___x_358_ = v___x_352_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v_one_355_);
lean_ctor_set(v_reuseFailAlloc_359_, 1, v_mul_354_);
lean_ctor_set(v_reuseFailAlloc_359_, 2, v___f_356_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid___redArg(lean_object* v_e_363_, lean_object* v_inst_364_){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v_toZero_367_; lean_object* v_toAdd_368_; lean_object* v_toNSMul_369_; lean_object* v_one_370_; lean_object* v_mul_371_; lean_object* v___f_372_; lean_object* v_pow_373_; lean_object* v___x_374_; 
v___x_365_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_364_);
v___x_366_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_365_);
v_toZero_367_ = lean_ctor_get(v___x_366_, 0);
lean_inc(v_toZero_367_);
v_toAdd_368_ = lean_ctor_get(v___x_366_, 1);
lean_inc(v_toAdd_368_);
lean_dec_ref(v___x_366_);
v_toNSMul_369_ = lean_ctor_get(v_inst_364_, 2);
lean_inc(v_toNSMul_369_);
lean_dec_ref(v_inst_364_);
lean_inc_ref_n(v_e_363_, 2);
v_one_370_ = lp_mathlib_Equiv_zero___redArg(v_e_363_, v_toZero_367_);
v_mul_371_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_371_, 0, v_e_363_);
lean_closure_set(v_mul_371_, 1, v_toAdd_368_);
v___f_372_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_372_, 0, v_toNSMul_369_);
v_pow_373_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v_pow_373_, 0, v_e_363_);
lean_closure_set(v_pow_373_, 1, v___f_372_);
v___x_374_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_mul_371_, v_one_370_, v_pow_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommMonoid(lean_object* v_00_u03b1_375_, lean_object* v_00_u03b2_376_, lean_object* v_e_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_Equiv_addCommMonoid___redArg(v_e_377_, v_inst_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__1(lean_object* v_toFun_380_, lean_object* v_toInv_381_, lean_object* v_invFun_382_, lean_object* v_x_383_){
_start:
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_384_ = lean_apply_1(v_toFun_380_, v_x_383_);
v___x_385_ = lean_apply_1(v_toInv_381_, v___x_384_);
v___x_386_ = lean_apply_1(v_invFun_382_, v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__0(lean_object* v_toFun_387_, lean_object* v_toDiv_388_, lean_object* v_invFun_389_, lean_object* v_x_390_, lean_object* v_y_391_){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
lean_inc(v_toFun_387_);
v___x_392_ = lean_apply_1(v_toFun_387_, v_x_390_);
v___x_393_ = lean_apply_1(v_toFun_387_, v_y_391_);
v___x_394_ = lean_apply_2(v_toDiv_388_, v___x_392_, v___x_393_);
v___x_395_ = lean_apply_1(v_invFun_389_, v___x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg___lam__2(lean_object* v_toFun_396_, lean_object* v_toZPow_397_, lean_object* v_invFun_398_, lean_object* v_n_399_, lean_object* v_x_400_){
_start:
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_401_ = lean_apply_1(v_toFun_396_, v_x_400_);
v___x_402_ = lean_apply_2(v_toZPow_397_, v_n_399_, v___x_401_);
v___x_403_ = lean_apply_1(v_invFun_398_, v___x_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group___redArg(lean_object* v_e_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v_toMonoid_406_; lean_object* v_toInv_407_; lean_object* v_toDiv_408_; lean_object* v_toZPow_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_438_; 
v_toMonoid_406_ = lean_ctor_get(v_inst_405_, 0);
v_toInv_407_ = lean_ctor_get(v_inst_405_, 1);
v_toDiv_408_ = lean_ctor_get(v_inst_405_, 2);
v_toZPow_409_ = lean_ctor_get(v_inst_405_, 3);
v_isSharedCheck_438_ = !lean_is_exclusive(v_inst_405_);
if (v_isSharedCheck_438_ == 0)
{
v___x_411_ = v_inst_405_;
v_isShared_412_ = v_isSharedCheck_438_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_toZPow_409_);
lean_inc(v_toDiv_408_);
lean_inc(v_toInv_407_);
lean_inc(v_toMonoid_406_);
lean_dec(v_inst_405_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_438_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v_toOne_415_; lean_object* v_toMul_416_; lean_object* v_toFun_417_; lean_object* v_invFun_418_; lean_object* v_toNPow_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_435_; 
v___x_413_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_406_);
v___x_414_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_413_);
v_toOne_415_ = lean_ctor_get(v___x_414_, 0);
lean_inc(v_toOne_415_);
v_toMul_416_ = lean_ctor_get(v___x_414_, 1);
lean_inc(v_toMul_416_);
lean_dec_ref(v___x_414_);
v_toFun_417_ = lean_ctor_get(v_e_404_, 0);
lean_inc(v_toFun_417_);
v_invFun_418_ = lean_ctor_get(v_e_404_, 1);
lean_inc(v_invFun_418_);
lean_dec_ref(v_e_404_);
v_toNPow_419_ = lean_ctor_get(v_toMonoid_406_, 2);
v_isSharedCheck_435_ = !lean_is_exclusive(v_toMonoid_406_);
if (v_isSharedCheck_435_ == 0)
{
lean_object* v_unused_436_; lean_object* v_unused_437_; 
v_unused_436_ = lean_ctor_get(v_toMonoid_406_, 1);
lean_dec(v_unused_436_);
v_unused_437_ = lean_ctor_get(v_toMonoid_406_, 0);
lean_dec(v_unused_437_);
v___x_421_ = v_toMonoid_406_;
v_isShared_422_ = v_isSharedCheck_435_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_toNPow_419_);
lean_dec(v_toMonoid_406_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_435_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v_mul_423_; lean_object* v_inv_424_; lean_object* v_div_425_; lean_object* v___f_426_; lean_object* v_one_427_; lean_object* v___f_428_; lean_object* v___x_430_; 
lean_inc_n(v_invFun_418_, 5);
lean_inc_n(v_toFun_417_, 4);
v_mul_423_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_423_, 0, v_toFun_417_);
lean_closure_set(v_mul_423_, 1, v_toMul_416_);
lean_closure_set(v_mul_423_, 2, v_invFun_418_);
v_inv_424_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 3);
lean_closure_set(v_inv_424_, 0, v_toFun_417_);
lean_closure_set(v_inv_424_, 1, v_toInv_407_);
lean_closure_set(v_inv_424_, 2, v_invFun_418_);
v_div_425_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 5, 3);
lean_closure_set(v_div_425_, 0, v_toFun_417_);
lean_closure_set(v_div_425_, 1, v_toDiv_408_);
lean_closure_set(v_div_425_, 2, v_invFun_418_);
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(v___f_426_, 0, v_toFun_417_);
lean_closure_set(v___f_426_, 1, v_toZPow_409_);
lean_closure_set(v___f_426_, 2, v_invFun_418_);
v_one_427_ = lean_apply_1(v_invFun_418_, v_toOne_415_);
v___f_428_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_428_, 0, v_toFun_417_);
lean_closure_set(v___f_428_, 1, v_toNPow_419_);
lean_closure_set(v___f_428_, 2, v_invFun_418_);
if (v_isShared_422_ == 0)
{
lean_ctor_set(v___x_421_, 2, v___f_428_);
lean_ctor_set(v___x_421_, 1, v_mul_423_);
lean_ctor_set(v___x_421_, 0, v_one_427_);
v___x_430_ = v___x_421_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_one_427_);
lean_ctor_set(v_reuseFailAlloc_434_, 1, v_mul_423_);
lean_ctor_set(v_reuseFailAlloc_434_, 2, v___f_428_);
v___x_430_ = v_reuseFailAlloc_434_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
lean_object* v___x_432_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 3, v___f_426_);
lean_ctor_set(v___x_411_, 2, v_div_425_);
lean_ctor_set(v___x_411_, 1, v_inv_424_);
lean_ctor_set(v___x_411_, 0, v___x_430_);
v___x_432_ = v___x_411_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v___x_430_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v_inv_424_);
lean_ctor_set(v_reuseFailAlloc_433_, 2, v_div_425_);
lean_ctor_set(v_reuseFailAlloc_433_, 3, v___f_426_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_group(lean_object* v_00_u03b1_439_, lean_object* v_00_u03b2_440_, lean_object* v_e_441_, lean_object* v_inst_442_){
_start:
{
lean_object* v_toMonoid_443_; lean_object* v_toInv_444_; lean_object* v_toDiv_445_; lean_object* v_toZPow_446_; lean_object* v___x_448_; uint8_t v_isShared_449_; uint8_t v_isSharedCheck_475_; 
v_toMonoid_443_ = lean_ctor_get(v_inst_442_, 0);
v_toInv_444_ = lean_ctor_get(v_inst_442_, 1);
v_toDiv_445_ = lean_ctor_get(v_inst_442_, 2);
v_toZPow_446_ = lean_ctor_get(v_inst_442_, 3);
v_isSharedCheck_475_ = !lean_is_exclusive(v_inst_442_);
if (v_isSharedCheck_475_ == 0)
{
v___x_448_ = v_inst_442_;
v_isShared_449_ = v_isSharedCheck_475_;
goto v_resetjp_447_;
}
else
{
lean_inc(v_toZPow_446_);
lean_inc(v_toDiv_445_);
lean_inc(v_toInv_444_);
lean_inc(v_toMonoid_443_);
lean_dec(v_inst_442_);
v___x_448_ = lean_box(0);
v_isShared_449_ = v_isSharedCheck_475_;
goto v_resetjp_447_;
}
v_resetjp_447_:
{
lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v_toOne_452_; lean_object* v_toMul_453_; lean_object* v_toFun_454_; lean_object* v_invFun_455_; lean_object* v_toNPow_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_472_; 
v___x_450_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_443_);
v___x_451_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_450_);
v_toOne_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_toOne_452_);
v_toMul_453_ = lean_ctor_get(v___x_451_, 1);
lean_inc(v_toMul_453_);
lean_dec_ref(v___x_451_);
v_toFun_454_ = lean_ctor_get(v_e_441_, 0);
lean_inc(v_toFun_454_);
v_invFun_455_ = lean_ctor_get(v_e_441_, 1);
lean_inc(v_invFun_455_);
lean_dec_ref(v_e_441_);
v_toNPow_456_ = lean_ctor_get(v_toMonoid_443_, 2);
v_isSharedCheck_472_ = !lean_is_exclusive(v_toMonoid_443_);
if (v_isSharedCheck_472_ == 0)
{
lean_object* v_unused_473_; lean_object* v_unused_474_; 
v_unused_473_ = lean_ctor_get(v_toMonoid_443_, 1);
lean_dec(v_unused_473_);
v_unused_474_ = lean_ctor_get(v_toMonoid_443_, 0);
lean_dec(v_unused_474_);
v___x_458_ = v_toMonoid_443_;
v_isShared_459_ = v_isSharedCheck_472_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_toNPow_456_);
lean_dec(v_toMonoid_443_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_472_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v_mul_460_; lean_object* v_inv_461_; lean_object* v_div_462_; lean_object* v___f_463_; lean_object* v_one_464_; lean_object* v___f_465_; lean_object* v___x_467_; 
lean_inc_n(v_invFun_455_, 5);
lean_inc_n(v_toFun_454_, 4);
v_mul_460_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_460_, 0, v_toFun_454_);
lean_closure_set(v_mul_460_, 1, v_toMul_453_);
lean_closure_set(v_mul_460_, 2, v_invFun_455_);
v_inv_461_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 3);
lean_closure_set(v_inv_461_, 0, v_toFun_454_);
lean_closure_set(v_inv_461_, 1, v_toInv_444_);
lean_closure_set(v_inv_461_, 2, v_invFun_455_);
v_div_462_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 5, 3);
lean_closure_set(v_div_462_, 0, v_toFun_454_);
lean_closure_set(v_div_462_, 1, v_toDiv_445_);
lean_closure_set(v_div_462_, 2, v_invFun_455_);
v___f_463_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(v___f_463_, 0, v_toFun_454_);
lean_closure_set(v___f_463_, 1, v_toZPow_446_);
lean_closure_set(v___f_463_, 2, v_invFun_455_);
v_one_464_ = lean_apply_1(v_invFun_455_, v_toOne_452_);
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_465_, 0, v_toFun_454_);
lean_closure_set(v___f_465_, 1, v_toNPow_456_);
lean_closure_set(v___f_465_, 2, v_invFun_455_);
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 2, v___f_465_);
lean_ctor_set(v___x_458_, 1, v_mul_460_);
lean_ctor_set(v___x_458_, 0, v_one_464_);
v___x_467_ = v___x_458_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_one_464_);
lean_ctor_set(v_reuseFailAlloc_471_, 1, v_mul_460_);
lean_ctor_set(v_reuseFailAlloc_471_, 2, v___f_465_);
v___x_467_ = v_reuseFailAlloc_471_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
lean_object* v___x_469_; 
if (v_isShared_449_ == 0)
{
lean_ctor_set(v___x_448_, 3, v___f_463_);
lean_ctor_set(v___x_448_, 2, v_div_462_);
lean_ctor_set(v___x_448_, 1, v_inv_461_);
lean_ctor_set(v___x_448_, 0, v___x_467_);
v___x_469_ = v___x_448_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_467_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v_inv_461_);
lean_ctor_set(v_reuseFailAlloc_470_, 2, v_div_462_);
lean_ctor_set(v_reuseFailAlloc_470_, 3, v___f_463_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
return v___x_469_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup___redArg(lean_object* v_e_476_, lean_object* v_inst_477_){
_start:
{
lean_object* v_toAddMonoid_478_; lean_object* v_toNeg_479_; lean_object* v_toSub_480_; lean_object* v_toZSMul_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v_toZero_484_; lean_object* v_toAdd_485_; lean_object* v_one_486_; lean_object* v_mul_487_; lean_object* v_toNSMul_488_; lean_object* v_inv_489_; lean_object* v_div_490_; lean_object* v___f_491_; lean_object* v_npow_492_; lean_object* v___f_493_; lean_object* v_zpow_494_; lean_object* v___x_495_; 
v_toAddMonoid_478_ = lean_ctor_get(v_inst_477_, 0);
lean_inc_ref(v_toAddMonoid_478_);
v_toNeg_479_ = lean_ctor_get(v_inst_477_, 1);
lean_inc(v_toNeg_479_);
v_toSub_480_ = lean_ctor_get(v_inst_477_, 2);
lean_inc(v_toSub_480_);
v_toZSMul_481_ = lean_ctor_get(v_inst_477_, 3);
lean_inc(v_toZSMul_481_);
lean_dec_ref(v_inst_477_);
v___x_482_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_478_);
v___x_483_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_482_);
v_toZero_484_ = lean_ctor_get(v___x_483_, 0);
lean_inc(v_toZero_484_);
v_toAdd_485_ = lean_ctor_get(v___x_483_, 1);
lean_inc(v_toAdd_485_);
lean_dec_ref(v___x_483_);
lean_inc_ref_n(v_e_476_, 5);
v_one_486_ = lp_mathlib_Equiv_zero___redArg(v_e_476_, v_toZero_484_);
v_mul_487_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_487_, 0, v_e_476_);
lean_closure_set(v_mul_487_, 1, v_toAdd_485_);
v_toNSMul_488_ = lean_ctor_get(v_toAddMonoid_478_, 2);
lean_inc(v_toNSMul_488_);
lean_dec_ref(v_toAddMonoid_478_);
v_inv_489_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(v_inv_489_, 0, v_e_476_);
lean_closure_set(v_inv_489_, 1, v_toNeg_479_);
v_div_490_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_div_490_, 0, v_e_476_);
lean_closure_set(v_div_490_, 1, v_toSub_480_);
v___f_491_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_491_, 0, v_toNSMul_488_);
v_npow_492_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v_npow_492_, 0, v_e_476_);
lean_closure_set(v_npow_492_, 1, v___f_491_);
v___f_493_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_493_, 0, v_toZSMul_481_);
v_zpow_494_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v_zpow_494_, 0, v_e_476_);
lean_closure_set(v_zpow_494_, 1, v___f_493_);
v___x_495_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_mul_487_, v_one_486_, v_npow_492_, v_inv_489_, v_div_490_, v_zpow_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addGroup(lean_object* v_00_u03b1_496_, lean_object* v_00_u03b2_497_, lean_object* v_e_498_, lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lp_mathlib_Equiv_addGroup___redArg(v_e_498_, v_inst_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup___redArg(lean_object* v_e_501_, lean_object* v_inst_502_){
_start:
{
lean_object* v_toMonoid_503_; lean_object* v_toInv_504_; lean_object* v_toDiv_505_; lean_object* v_toZPow_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_535_; 
v_toMonoid_503_ = lean_ctor_get(v_inst_502_, 0);
v_toInv_504_ = lean_ctor_get(v_inst_502_, 1);
v_toDiv_505_ = lean_ctor_get(v_inst_502_, 2);
v_toZPow_506_ = lean_ctor_get(v_inst_502_, 3);
v_isSharedCheck_535_ = !lean_is_exclusive(v_inst_502_);
if (v_isSharedCheck_535_ == 0)
{
v___x_508_ = v_inst_502_;
v_isShared_509_ = v_isSharedCheck_535_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_toZPow_506_);
lean_inc(v_toDiv_505_);
lean_inc(v_toInv_504_);
lean_inc(v_toMonoid_503_);
lean_dec(v_inst_502_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_535_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v_toOne_512_; lean_object* v_toMul_513_; lean_object* v_toFun_514_; lean_object* v_invFun_515_; lean_object* v_toNPow_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_532_; 
v___x_510_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_503_);
v___x_511_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_510_);
v_toOne_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_toOne_512_);
v_toMul_513_ = lean_ctor_get(v___x_511_, 1);
lean_inc(v_toMul_513_);
lean_dec_ref(v___x_511_);
v_toFun_514_ = lean_ctor_get(v_e_501_, 0);
lean_inc(v_toFun_514_);
v_invFun_515_ = lean_ctor_get(v_e_501_, 1);
lean_inc(v_invFun_515_);
lean_dec_ref(v_e_501_);
v_toNPow_516_ = lean_ctor_get(v_toMonoid_503_, 2);
v_isSharedCheck_532_ = !lean_is_exclusive(v_toMonoid_503_);
if (v_isSharedCheck_532_ == 0)
{
lean_object* v_unused_533_; lean_object* v_unused_534_; 
v_unused_533_ = lean_ctor_get(v_toMonoid_503_, 1);
lean_dec(v_unused_533_);
v_unused_534_ = lean_ctor_get(v_toMonoid_503_, 0);
lean_dec(v_unused_534_);
v___x_518_ = v_toMonoid_503_;
v_isShared_519_ = v_isSharedCheck_532_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_toNPow_516_);
lean_dec(v_toMonoid_503_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_532_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v_mul_520_; lean_object* v_inv_521_; lean_object* v_div_522_; lean_object* v___f_523_; lean_object* v_one_524_; lean_object* v___f_525_; lean_object* v___x_527_; 
lean_inc_n(v_invFun_515_, 5);
lean_inc_n(v_toFun_514_, 4);
v_mul_520_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_520_, 0, v_toFun_514_);
lean_closure_set(v_mul_520_, 1, v_toMul_513_);
lean_closure_set(v_mul_520_, 2, v_invFun_515_);
v_inv_521_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 3);
lean_closure_set(v_inv_521_, 0, v_toFun_514_);
lean_closure_set(v_inv_521_, 1, v_toInv_504_);
lean_closure_set(v_inv_521_, 2, v_invFun_515_);
v_div_522_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 5, 3);
lean_closure_set(v_div_522_, 0, v_toFun_514_);
lean_closure_set(v_div_522_, 1, v_toDiv_505_);
lean_closure_set(v_div_522_, 2, v_invFun_515_);
v___f_523_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(v___f_523_, 0, v_toFun_514_);
lean_closure_set(v___f_523_, 1, v_toZPow_506_);
lean_closure_set(v___f_523_, 2, v_invFun_515_);
v_one_524_ = lean_apply_1(v_invFun_515_, v_toOne_512_);
v___f_525_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_525_, 0, v_toFun_514_);
lean_closure_set(v___f_525_, 1, v_toNPow_516_);
lean_closure_set(v___f_525_, 2, v_invFun_515_);
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 2, v___f_525_);
lean_ctor_set(v___x_518_, 1, v_mul_520_);
lean_ctor_set(v___x_518_, 0, v_one_524_);
v___x_527_ = v___x_518_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_one_524_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v_mul_520_);
lean_ctor_set(v_reuseFailAlloc_531_, 2, v___f_525_);
v___x_527_ = v_reuseFailAlloc_531_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
lean_object* v___x_529_; 
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 3, v___f_523_);
lean_ctor_set(v___x_508_, 2, v_div_522_);
lean_ctor_set(v___x_508_, 1, v_inv_521_);
lean_ctor_set(v___x_508_, 0, v___x_527_);
v___x_529_ = v___x_508_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___x_527_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v_inv_521_);
lean_ctor_set(v_reuseFailAlloc_530_, 2, v_div_522_);
lean_ctor_set(v_reuseFailAlloc_530_, 3, v___f_523_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_commGroup(lean_object* v_00_u03b1_536_, lean_object* v_00_u03b2_537_, lean_object* v_e_538_, lean_object* v_inst_539_){
_start:
{
lean_object* v_toMonoid_540_; lean_object* v_toInv_541_; lean_object* v_toDiv_542_; lean_object* v_toZPow_543_; lean_object* v___x_545_; uint8_t v_isShared_546_; uint8_t v_isSharedCheck_572_; 
v_toMonoid_540_ = lean_ctor_get(v_inst_539_, 0);
v_toInv_541_ = lean_ctor_get(v_inst_539_, 1);
v_toDiv_542_ = lean_ctor_get(v_inst_539_, 2);
v_toZPow_543_ = lean_ctor_get(v_inst_539_, 3);
v_isSharedCheck_572_ = !lean_is_exclusive(v_inst_539_);
if (v_isSharedCheck_572_ == 0)
{
v___x_545_ = v_inst_539_;
v_isShared_546_ = v_isSharedCheck_572_;
goto v_resetjp_544_;
}
else
{
lean_inc(v_toZPow_543_);
lean_inc(v_toDiv_542_);
lean_inc(v_toInv_541_);
lean_inc(v_toMonoid_540_);
lean_dec(v_inst_539_);
v___x_545_ = lean_box(0);
v_isShared_546_ = v_isSharedCheck_572_;
goto v_resetjp_544_;
}
v_resetjp_544_:
{
lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v_toOne_549_; lean_object* v_toMul_550_; lean_object* v_toFun_551_; lean_object* v_invFun_552_; lean_object* v_toNPow_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_569_; 
v___x_547_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_540_);
v___x_548_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_547_);
v_toOne_549_ = lean_ctor_get(v___x_548_, 0);
lean_inc(v_toOne_549_);
v_toMul_550_ = lean_ctor_get(v___x_548_, 1);
lean_inc(v_toMul_550_);
lean_dec_ref(v___x_548_);
v_toFun_551_ = lean_ctor_get(v_e_538_, 0);
lean_inc(v_toFun_551_);
v_invFun_552_ = lean_ctor_get(v_e_538_, 1);
lean_inc(v_invFun_552_);
lean_dec_ref(v_e_538_);
v_toNPow_553_ = lean_ctor_get(v_toMonoid_540_, 2);
v_isSharedCheck_569_ = !lean_is_exclusive(v_toMonoid_540_);
if (v_isSharedCheck_569_ == 0)
{
lean_object* v_unused_570_; lean_object* v_unused_571_; 
v_unused_570_ = lean_ctor_get(v_toMonoid_540_, 1);
lean_dec(v_unused_570_);
v_unused_571_ = lean_ctor_get(v_toMonoid_540_, 0);
lean_dec(v_unused_571_);
v___x_555_ = v_toMonoid_540_;
v_isShared_556_ = v_isSharedCheck_569_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_toNPow_553_);
lean_dec(v_toMonoid_540_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_569_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v_mul_557_; lean_object* v_inv_558_; lean_object* v_div_559_; lean_object* v___f_560_; lean_object* v_one_561_; lean_object* v___f_562_; lean_object* v___x_564_; 
lean_inc_n(v_invFun_552_, 5);
lean_inc_n(v_toFun_551_, 4);
v_mul_557_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mulOneClass___redArg___lam__0), 5, 3);
lean_closure_set(v_mul_557_, 0, v_toFun_551_);
lean_closure_set(v_mul_557_, 1, v_toMul_550_);
lean_closure_set(v_mul_557_, 2, v_invFun_552_);
v_inv_558_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__1), 4, 3);
lean_closure_set(v_inv_558_, 0, v_toFun_551_);
lean_closure_set(v_inv_558_, 1, v_toInv_541_);
lean_closure_set(v_inv_558_, 2, v_invFun_552_);
v_div_559_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__0), 5, 3);
lean_closure_set(v_div_559_, 0, v_toFun_551_);
lean_closure_set(v_div_559_, 1, v_toDiv_542_);
lean_closure_set(v_div_559_, 2, v_invFun_552_);
v___f_560_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_group___redArg___lam__2), 5, 3);
lean_closure_set(v___f_560_, 0, v_toFun_551_);
lean_closure_set(v___f_560_, 1, v_toZPow_543_);
lean_closure_set(v___f_560_, 2, v_invFun_552_);
v_one_561_ = lean_apply_1(v_invFun_552_, v_toOne_549_);
v___f_562_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_monoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_562_, 0, v_toFun_551_);
lean_closure_set(v___f_562_, 1, v_toNPow_553_);
lean_closure_set(v___f_562_, 2, v_invFun_552_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 2, v___f_562_);
lean_ctor_set(v___x_555_, 1, v_mul_557_);
lean_ctor_set(v___x_555_, 0, v_one_561_);
v___x_564_ = v___x_555_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_568_; 
v_reuseFailAlloc_568_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_568_, 0, v_one_561_);
lean_ctor_set(v_reuseFailAlloc_568_, 1, v_mul_557_);
lean_ctor_set(v_reuseFailAlloc_568_, 2, v___f_562_);
v___x_564_ = v_reuseFailAlloc_568_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
lean_object* v___x_566_; 
if (v_isShared_546_ == 0)
{
lean_ctor_set(v___x_545_, 3, v___f_560_);
lean_ctor_set(v___x_545_, 2, v_div_559_);
lean_ctor_set(v___x_545_, 1, v_inv_558_);
lean_ctor_set(v___x_545_, 0, v___x_564_);
v___x_566_ = v___x_545_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v___x_564_);
lean_ctor_set(v_reuseFailAlloc_567_, 1, v_inv_558_);
lean_ctor_set(v_reuseFailAlloc_567_, 2, v_div_559_);
lean_ctor_set(v_reuseFailAlloc_567_, 3, v___f_560_);
v___x_566_ = v_reuseFailAlloc_567_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
return v___x_566_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup___redArg(lean_object* v_e_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v_toAddMonoid_575_; lean_object* v_toNeg_576_; lean_object* v_toSub_577_; lean_object* v_toZSMul_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v_toZero_581_; lean_object* v_toAdd_582_; lean_object* v_one_583_; lean_object* v_mul_584_; lean_object* v_toNSMul_585_; lean_object* v_inv_586_; lean_object* v_div_587_; lean_object* v___f_588_; lean_object* v_npow_589_; lean_object* v___f_590_; lean_object* v_zpow_591_; lean_object* v___x_592_; 
v_toAddMonoid_575_ = lean_ctor_get(v_inst_574_, 0);
lean_inc_ref(v_toAddMonoid_575_);
v_toNeg_576_ = lean_ctor_get(v_inst_574_, 1);
lean_inc(v_toNeg_576_);
v_toSub_577_ = lean_ctor_get(v_inst_574_, 2);
lean_inc(v_toSub_577_);
v_toZSMul_578_ = lean_ctor_get(v_inst_574_, 3);
lean_inc(v_toZSMul_578_);
lean_dec_ref(v_inst_574_);
v___x_579_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_575_);
v___x_580_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_579_);
v_toZero_581_ = lean_ctor_get(v___x_580_, 0);
lean_inc(v_toZero_581_);
v_toAdd_582_ = lean_ctor_get(v___x_580_, 1);
lean_inc(v_toAdd_582_);
lean_dec_ref(v___x_580_);
lean_inc_ref_n(v_e_573_, 5);
v_one_583_ = lp_mathlib_Equiv_zero___redArg(v_e_573_, v_toZero_581_);
v_mul_584_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_mul_584_, 0, v_e_573_);
lean_closure_set(v_mul_584_, 1, v_toAdd_582_);
v_toNSMul_585_ = lean_ctor_get(v_toAddMonoid_575_, 2);
lean_inc(v_toNSMul_585_);
lean_dec_ref(v_toAddMonoid_575_);
v_inv_586_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Inv___redArg___lam__0), 3, 2);
lean_closure_set(v_inv_586_, 0, v_e_573_);
lean_closure_set(v_inv_586_, 1, v_toNeg_576_);
v_div_587_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mul___redArg___lam__0), 4, 2);
lean_closure_set(v_div_587_, 0, v_e_573_);
lean_closure_set(v_div_587_, 1, v_toSub_577_);
v___f_588_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_588_, 0, v_toNSMul_585_);
v_npow_589_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v_npow_589_, 0, v_e_573_);
lean_closure_set(v_npow_589_, 1, v___f_588_);
v___f_590_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_590_, 0, v_toZSMul_578_);
v_zpow_591_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v_zpow_591_, 0, v_e_573_);
lean_closure_set(v_zpow_591_, 1, v___f_590_);
v___x_592_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_mul_584_, v_one_583_, v_npow_589_, v_inv_586_, v_div_587_, v_zpow_591_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addCommGroup(lean_object* v_00_u03b1_593_, lean_object* v_00_u03b2_594_, lean_object* v_e_595_, lean_object* v_inst_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_mathlib_Equiv_addCommGroup___redArg(v_e_595_, v_inst_596_);
return v___x_597_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_instance__transfer__via__equivalence = _init_lp_mathlib_LibraryNote_instance__transfer__via__equivalence();
lean_mark_persistent(lp_mathlib_LibraryNote_instance__transfer__via__equivalence);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_TransferInstance(builtin);
}
#ifdef __cplusplus
}
#endif
