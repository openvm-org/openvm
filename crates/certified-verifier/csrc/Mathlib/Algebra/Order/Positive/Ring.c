// Lean compiler output
// Module: Mathlib.Algebra.Order.Positive.Ring
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Ring.Defs public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addCommSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addLeftCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addLeftCancelSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addRightCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_addRightCancelSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instSemigroupSubtypeLtOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instSemigroupSubtypeLtOfNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instSemigroupSubtypeLtOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_commMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_commMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg___lam__0(lean_object* v_toAdd_1_, lean_object* v_x_2_, lean_object* v_y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toAdd_1_, v_x_2_, v_y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v_toAdd_6_; lean_object* v___f_7_; 
v_toAdd_6_ = lean_ctor_get(v_inst_5_, 1);
lean_inc(v_toAdd_6_);
lean_dec_ref(v_inst_5_);
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_toAdd_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib(lean_object* v_M_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_9_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___boxed(lean_object* v_M_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib(v_M_13_, v_inst_14_, v_inst_15_, v_inst_16_);
lean_dec_ref(v_inst_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addSemigroup___redArg(lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addSemigroup(lean_object* v_M_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_21_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addSemigroup___boxed(lean_object* v_M_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Positive_addSemigroup(v_M_25_, v_inst_26_, v_inst_27_, v_inst_28_);
lean_dec_ref(v_inst_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addCommSemigroup___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addCommSemigroup(lean_object* v_M_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_33_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addCommSemigroup___boxed(lean_object* v_M_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Positive_addCommSemigroup(v_M_37_, v_inst_38_, v_inst_39_, v_inst_40_);
lean_dec_ref(v_inst_39_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addLeftCancelSemigroup___redArg(lean_object* v_inst_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addLeftCancelSemigroup(lean_object* v_M_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_45_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addLeftCancelSemigroup___boxed(lean_object* v_M_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Positive_addLeftCancelSemigroup(v_M_49_, v_inst_50_, v_inst_51_, v_inst_52_);
lean_dec_ref(v_inst_51_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addRightCancelSemigroup___redArg(lean_object* v_inst_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addRightCancelSemigroup(lean_object* v_M_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_inst_57_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_addRightCancelSemigroup___boxed(lean_object* v_M_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Positive_addRightCancelSemigroup(v_M_61_, v_inst_62_, v_inst_63_, v_inst_64_);
lean_dec_ref(v_inst_63_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg___lam__0(lean_object* v_toMul_66_, lean_object* v_x_67_, lean_object* v_y_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lean_apply_2(v_toMul_66_, v_x_67_, v_y_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; lean_object* v_toMul_72_; lean_object* v___f_73_; 
v___x_71_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_70_);
v_toMul_72_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_toMul_72_);
lean_dec_ref(v___x_71_);
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_73_, 0, v_toMul_72_);
return v___f_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib(lean_object* v_R_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(v_inst_75_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___boxed(lean_object* v_R_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib(v_R_79_, v_inst_80_, v_inst_81_, v_inst_82_);
lean_dec_ref(v_inst_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg___lam__0(lean_object* v_toNPow_84_, lean_object* v_x_85_, lean_object* v_n_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lean_apply_2(v_toNPow_84_, v_n_86_, v_x_85_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg(lean_object* v_inst_88_){
_start:
{
lean_object* v_toMonoid_89_; lean_object* v_toNPow_90_; lean_object* v___f_91_; 
v_toMonoid_89_ = lean_ctor_get(v_inst_88_, 1);
lean_inc_ref(v_toMonoid_89_);
lean_dec_ref(v_inst_88_);
v_toNPow_90_ = lean_ctor_get(v_toMonoid_89_, 2);
lean_inc(v_toNPow_90_);
lean_dec_ref(v_toMonoid_89_);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_91_, 0, v_toNPow_90_);
return v___f_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib(lean_object* v_R_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg(v_inst_93_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___boxed(lean_object* v_R_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib(v_R_97_, v_inst_98_, v_inst_99_, v_inst_100_);
lean_dec_ref(v_inst_99_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instSemigroupSubtypeLtOfNat___redArg(lean_object* v_inst_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(v_inst_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instSemigroupSubtypeLtOfNat(lean_object* v_R_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(v_inst_105_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instSemigroupSubtypeLtOfNat___boxed(lean_object* v_R_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Positive_instSemigroupSubtypeLtOfNat(v_R_109_, v_inst_110_, v_inst_111_, v_inst_112_);
lean_dec_ref(v_inst_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat___redArg(lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; lean_object* v_toNonUnitalNonAssocSemiring_116_; lean_object* v_toAddCommMonoid_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_126_; 
lean_inc_ref(v_inst_114_);
v___x_115_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_114_);
v_toNonUnitalNonAssocSemiring_116_ = lean_ctor_get(v___x_115_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_116_);
lean_dec_ref(v___x_115_);
v_toAddCommMonoid_117_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_116_, 0);
v_isSharedCheck_126_ = !lean_is_exclusive(v_toNonUnitalNonAssocSemiring_116_);
if (v_isSharedCheck_126_ == 0)
{
lean_object* v_unused_127_; 
v_unused_127_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_116_, 1);
lean_dec(v_unused_127_);
v___x_119_ = v_toNonUnitalNonAssocSemiring_116_;
v_isShared_120_ = v_isSharedCheck_126_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_toAddCommMonoid_117_);
lean_dec(v_toNonUnitalNonAssocSemiring_116_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_126_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_124_; 
v___x_121_ = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(v_inst_114_);
v___x_122_ = lp_mathlib_Positive_instAddSubtypeLtOfNat__mathlib___redArg(v_toAddCommMonoid_117_);
if (v_isShared_120_ == 0)
{
lean_ctor_set(v___x_119_, 1, v___x_122_);
lean_ctor_set(v___x_119_, 0, v___x_121_);
v___x_124_ = v___x_119_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v___x_121_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v___x_122_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat(lean_object* v_R_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_Positive_instDistribSubtypeLtOfNat___redArg(v_inst_129_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instDistribSubtypeLtOfNat___boxed(lean_object* v_R_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_Positive_instDistribSubtypeLtOfNat(v_R_133_, v_inst_134_, v_inst_135_, v_inst_136_);
lean_dec_ref(v_inst_135_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib___redArg(lean_object* v_inst_138_){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v_toOne_141_; 
v___x_139_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_138_);
v___x_140_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_139_);
v_toOne_141_ = lean_ctor_get(v___x_140_, 2);
lean_inc(v_toOne_141_);
lean_dec_ref(v___x_140_);
return v_toOne_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib(lean_object* v_R_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib___redArg(v_inst_143_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib___boxed(lean_object* v_R_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib(v_R_147_, v_inst_148_, v_inst_149_, v_inst_150_);
lean_dec_ref(v_inst_149_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(lean_object* v_inst_152_){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___f_156_; lean_object* v___x_157_; 
lean_inc_ref_n(v_inst_152_, 2);
v___x_153_ = lp_mathlib_Positive_instOneSubtypeLtOfNat__mathlib___redArg(v_inst_152_);
v___x_154_ = lp_mathlib_Positive_instMulSubtypeLtOfNat__mathlib___redArg(v_inst_152_);
v___x_155_ = lp_mathlib_Positive_instPowSubtypeLtOfNatNat__mathlib___redArg(v_inst_152_);
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_156_, 0, v___x_155_);
v___x_157_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_157_, 0, v___x_153_);
lean_ctor_set(v___x_157_, 1, v___x_154_);
lean_ctor_set(v___x_157_, 2, v___f_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat(lean_object* v_R_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(v_inst_159_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_instMonoidSubtypeLtOfNat___boxed(lean_object* v_R_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Positive_instMonoidSubtypeLtOfNat(v_R_163_, v_inst_164_, v_inst_165_, v_inst_166_);
lean_dec_ref(v_inst_165_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_commMonoid___redArg(lean_object* v_inst_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(v_inst_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_commMonoid(lean_object* v_R_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_Positive_instMonoidSubtypeLtOfNat___redArg(v_inst_171_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Positive_commMonoid___boxed(lean_object* v_R_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Positive_commMonoid(v_R_175_, v_inst_176_, v_inst_177_, v_inst_178_);
lean_dec_ref(v_inst_177_);
return v_res_179_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
}
#ifdef __cplusplus
}
#endif
