// Lean compiler output
// Module: Mathlib.Algebra.Group.Pointwise.Set.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Basic public import Mathlib.Algebra.Group.Prod public import Mathlib.Algebra.Order.Monoid.Unbundled.Pow public import Mathlib.Data.Set.NAry
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
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_pointwise__nat__action;
LEAN_EXPORT lean_object* lp_mathlib_Set_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_one___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_zero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_inv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_neg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_div___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sub___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Set_monoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_monoid___closed__0 = (const lean_object*)&lp_mathlib_Set_monoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Set_addMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_addMonoid___closed__0 = (const lean_object*)&lp_mathlib_Set_addMonoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_LibraryNote_pointwise__nat__action(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_one(lean_object* v_00_u03b1_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_one___boxed(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Set_one(v_00_u03b1_5_, v_inst_6_);
lean_dec(v_inst_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_zero(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_box(0);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_zero___boxed(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Set_zero(v_00_u03b1_11_, v_inst_12_);
lean_dec(v_inst_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_box(0);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonOneHom___boxed(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Set_singletonOneHom(v_00_u03b1_17_, v_inst_18_);
lean_dec(v_inst_18_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_box(0);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonZeroHom___boxed(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Set_singletonZeroHom(v_00_u03b1_23_, v_inst_24_);
lean_dec(v_inst_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_inv(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_inv___boxed(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Set_inv(v_00_u03b1_29_, v_inst_30_);
lean_dec(v_inst_30_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_neg(lean_object* v_00_u03b1_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lean_box(0);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_neg___boxed(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_Set_neg(v_00_u03b1_35_, v_inst_36_);
lean_dec(v_inst_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_box(0);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveInv___boxed(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Set_involutiveInv(v_00_u03b1_41_, v_inst_42_);
lean_dec(v_inst_42_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg(lean_object* v_00_u03b1_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_involutiveNeg___boxed(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Set_involutiveNeg(v_00_u03b1_47_, v_inst_48_);
lean_dec(v_inst_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mul(lean_object* v_00_u03b1_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_box(0);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mul___boxed(lean_object* v_00_u03b1_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Set_mul(v_00_u03b1_53_, v_inst_54_);
lean_dec(v_inst_54_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_add(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_box(0);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_add___boxed(lean_object* v_00_u03b1_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Set_add(v_00_u03b1_59_, v_inst_60_);
lean_dec(v_inst_60_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lean_box(0);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMulHom___boxed(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Set_singletonMulHom(v_00_u03b1_65_, v_inst_66_);
lean_dec(v_inst_66_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom(lean_object* v_00_u03b1_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_box(0);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddHom___boxed(lean_object* v_00_u03b1_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_Set_singletonAddHom(v_00_u03b1_71_, v_inst_72_);
lean_dec(v_inst_72_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_div(lean_object* v_00_u03b1_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lean_box(0);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_div___boxed(lean_object* v_00_u03b1_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_Set_div(v_00_u03b1_77_, v_inst_78_);
lean_dec(v_inst_78_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sub(lean_object* v_00_u03b1_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lean_box(0);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sub___boxed(lean_object* v_00_u03b1_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Set_sub(v_00_u03b1_83_, v_inst_84_);
lean_dec(v_inst_84_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow(lean_object* v_00_u03b1_86_, lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_box(0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NPow___boxed(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_Set_NPow(v_00_u03b1_90_, v_inst_91_, v_inst_92_);
lean_dec(v_inst_92_);
lean_dec(v_inst_91_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul(lean_object* v_00_u03b1_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_box(0);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_NSMul___boxed(lean_object* v_00_u03b1_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Set_NSMul(v_00_u03b1_98_, v_inst_99_, v_inst_100_);
lean_dec(v_inst_100_);
lean_dec(v_inst_99_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lean_box(0);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZPow___boxed(lean_object* v_00_u03b1_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Set_ZPow(v_00_u03b1_107_, v_inst_108_, v_inst_109_, v_inst_110_);
lean_dec(v_inst_110_);
lean_dec(v_inst_109_);
lean_dec(v_inst_108_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul(lean_object* v_00_u03b1_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lean_box(0);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_ZSMul___boxed(lean_object* v_00_u03b1_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Set_ZSMul(v_00_u03b1_117_, v_inst_118_, v_inst_119_, v_inst_120_);
lean_dec(v_inst_120_);
lean_dec(v_inst_119_);
lean_dec(v_inst_118_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup(lean_object* v_00_u03b1_122_, lean_object* v_inst_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lean_box(0);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_semigroup___boxed(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_Set_semigroup(v_00_u03b1_125_, v_inst_126_);
lean_dec(v_inst_126_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup(lean_object* v_00_u03b1_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_box(0);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addSemigroup___boxed(lean_object* v_00_u03b1_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_Set_addSemigroup(v_00_u03b1_131_, v_inst_132_);
lean_dec(v_inst_132_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup(lean_object* v_00_u03b1_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_box(0);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commSemigroup___boxed(lean_object* v_00_u03b1_137_, lean_object* v_inst_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Set_commSemigroup(v_00_u03b1_137_, v_inst_138_);
lean_dec(v_inst_138_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup(lean_object* v_00_u03b1_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lean_box(0);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommSemigroup___boxed(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_Set_addCommSemigroup(v_00_u03b1_143_, v_inst_144_);
lean_dec(v_inst_144_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_148_, 0, lean_box(0));
lean_ctor_set(v___x_148_, 1, lean_box(0));
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_mulOneClass___boxed(lean_object* v_00_u03b1_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_Set_mulOneClass(v_00_u03b1_149_, v_inst_150_);
lean_dec_ref(v_inst_150_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass(lean_object* v_00_u03b1_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_154_, 0, lean_box(0));
lean_ctor_set(v___x_154_, 1, lean_box(0));
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addZeroClass___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Set_addZeroClass(v_00_u03b1_155_, v_inst_156_);
lean_dec_ref(v_inst_156_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom(lean_object* v_00_u03b1_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_box(0);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonMonoidHom___boxed(lean_object* v_00_u03b1_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_Set_singletonMonoidHom(v_00_u03b1_161_, v_inst_162_);
lean_dec_ref(v_inst_162_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_box(0);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_singletonAddMonoidHom___boxed(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Set_singletonAddMonoidHom(v_00_u03b1_167_, v_inst_168_);
lean_dec_ref(v_inst_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid(lean_object* v_00_u03b1_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = ((lean_object*)(lp_mathlib_Set_monoid___closed__0));
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_monoid___boxed(lean_object* v_00_u03b1_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Set_monoid(v_00_u03b1_174_, v_inst_175_);
lean_dec_ref(v_inst_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid(lean_object* v_00_u03b1_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = ((lean_object*)(lp_mathlib_Set_addMonoid___closed__0));
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addMonoid___boxed(lean_object* v_00_u03b1_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_Set_addMonoid(v_00_u03b1_181_, v_inst_182_);
lean_dec_ref(v_inst_182_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg(lean_object* v_inst_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Set_monoid(lean_box(0), v_inst_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___redArg___boxed(lean_object* v_inst_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Set_commMonoid___redArg(v_inst_186_);
lean_dec_ref(v_inst_186_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid(lean_object* v_00_u03b1_188_, lean_object* v_inst_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_Set_monoid(lean_box(0), v_inst_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_commMonoid___boxed(lean_object* v_00_u03b1_191_, lean_object* v_inst_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_Set_commMonoid(v_00_u03b1_191_, v_inst_192_);
lean_dec_ref(v_inst_192_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg(lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_Set_addMonoid(lean_box(0), v_inst_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___redArg___boxed(lean_object* v_inst_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Set_addCommMonoid___redArg(v_inst_196_);
lean_dec_ref(v_inst_196_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_mathlib_Set_addMonoid(lean_box(0), v_inst_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_addCommMonoid___boxed(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_Set_addCommMonoid(v_00_u03b1_201_, v_inst_202_);
lean_dec_ref(v_inst_202_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid___redArg(lean_object* v_inst_204_){
_start:
{
lean_object* v_toMonoid_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_213_; 
v_toMonoid_205_ = lean_ctor_get(v_inst_204_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v_inst_204_);
if (v_isSharedCheck_213_ == 0)
{
lean_object* v_unused_214_; lean_object* v_unused_215_; lean_object* v_unused_216_; 
v_unused_214_ = lean_ctor_get(v_inst_204_, 3);
lean_dec(v_unused_214_);
v_unused_215_ = lean_ctor_get(v_inst_204_, 2);
lean_dec(v_unused_215_);
v_unused_216_ = lean_ctor_get(v_inst_204_, 1);
lean_dec(v_unused_216_);
v___x_207_ = v_inst_204_;
v_isShared_208_ = v_isSharedCheck_213_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_toMonoid_205_);
lean_dec(v_inst_204_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_213_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_209_ = lp_mathlib_Set_monoid(lean_box(0), v_toMonoid_205_);
lean_dec_ref(v_toMonoid_205_);
if (v_isShared_208_ == 0)
{
lean_ctor_set(v___x_207_, 3, lean_box(0));
lean_ctor_set(v___x_207_, 2, lean_box(0));
lean_ctor_set(v___x_207_, 1, lean_box(0));
lean_ctor_set(v___x_207_, 0, v___x_209_);
v___x_211_ = v___x_207_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_212_, 1, lean_box(0));
lean_ctor_set(v_reuseFailAlloc_212_, 2, lean_box(0));
lean_ctor_set(v_reuseFailAlloc_212_, 3, lean_box(0));
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionMonoid(lean_object* v_00_u03b1_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_Set_divisionMonoid___redArg(v_inst_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid___redArg(lean_object* v_inst_220_){
_start:
{
lean_object* v_toAddMonoid_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_229_; 
v_toAddMonoid_221_ = lean_ctor_get(v_inst_220_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v_inst_220_);
if (v_isSharedCheck_229_ == 0)
{
lean_object* v_unused_230_; lean_object* v_unused_231_; lean_object* v_unused_232_; 
v_unused_230_ = lean_ctor_get(v_inst_220_, 3);
lean_dec(v_unused_230_);
v_unused_231_ = lean_ctor_get(v_inst_220_, 2);
lean_dec(v_unused_231_);
v_unused_232_ = lean_ctor_get(v_inst_220_, 1);
lean_dec(v_unused_232_);
v___x_223_ = v_inst_220_;
v_isShared_224_ = v_isSharedCheck_229_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_toAddMonoid_221_);
lean_dec(v_inst_220_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_229_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_227_; 
v___x_225_ = lp_mathlib_Set_addMonoid(lean_box(0), v_toAddMonoid_221_);
lean_dec_ref(v_toAddMonoid_221_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 3, lean_box(0));
lean_ctor_set(v___x_223_, 2, lean_box(0));
lean_ctor_set(v___x_223_, 1, lean_box(0));
lean_ctor_set(v___x_223_, 0, v___x_225_);
v___x_227_ = v___x_223_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v___x_225_);
lean_ctor_set(v_reuseFailAlloc_228_, 1, lean_box(0));
lean_ctor_set(v_reuseFailAlloc_228_, 2, lean_box(0));
lean_ctor_set(v_reuseFailAlloc_228_, 3, lean_box(0));
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionMonoid(lean_object* v_00_u03b1_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lp_mathlib_Set_subtractionMonoid___redArg(v_inst_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid___redArg(lean_object* v_inst_236_){
_start:
{
lean_object* v___x_237_; 
v___x_237_ = lp_mathlib_Set_divisionMonoid___redArg(v_inst_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_divisionCommMonoid(lean_object* v_00_u03b1_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lp_mathlib_Set_divisionMonoid___redArg(v_inst_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid___redArg(lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_Set_subtractionMonoid___redArg(v_inst_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_subtractionCommMonoid(lean_object* v_00_u03b1_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_Set_subtractionMonoid___redArg(v_inst_244_);
return v___x_245_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_pointwise__nat__action = _init_lp_mathlib_LibraryNote_pointwise__nat__action();
lean_mark_persistent(lp_mathlib_LibraryNote_pointwise__nat__action);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
