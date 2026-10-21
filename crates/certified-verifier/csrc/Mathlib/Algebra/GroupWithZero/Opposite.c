// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Opposite public import Mathlib.Algebra.GroupWithZero.InjSurj public import Mathlib.Algebra.GroupWithZero.NeZero
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
lean_object* lp_mathlib_AddOpposite_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toMul_2_; lean_object* v_toZero_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_11_; 
v_toMul_2_ = lean_ctor_get(v_inst_1_, 0);
v_toZero_3_ = lean_ctor_get(v_inst_1_, 1);
v_isSharedCheck_11_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_11_ == 0)
{
v___x_5_ = v_inst_1_;
v_isShared_6_ = v_isSharedCheck_11_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toZero_3_);
lean_inc(v_toMul_2_);
lean_dec(v_inst_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_11_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___f_7_; lean_object* v___x_9_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_toMul_2_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 0, v___f_7_);
v___x_9_ = v___x_5_;
goto v_reusejp_8_;
}
else
{
lean_object* v_reuseFailAlloc_10_; 
v_reuseFailAlloc_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_10_, 0, v___f_7_);
lean_ctor_set(v_reuseFailAlloc_10_, 1, v_toZero_3_);
v___x_9_ = v_reuseFailAlloc_10_;
goto v_reusejp_8_;
}
v_reusejp_8_:
{
return v___x_9_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroClass(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_MulOpposite_instMulZeroClass___redArg(v_inst_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v_toMulOneClass_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v_toZero_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_27_; 
v_toMulOneClass_16_ = lean_ctor_get(v_inst_15_, 0);
lean_inc_ref(v_toMulOneClass_16_);
v___x_17_ = lp_mathlib_MulOpposite_instMulOneClass___redArg(v_toMulOneClass_16_);
v___x_18_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_15_);
v___x_19_ = lp_mathlib_MulOpposite_instMulZeroClass___redArg(v___x_18_);
v_toZero_20_ = lean_ctor_get(v___x_19_, 1);
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_27_ == 0)
{
lean_object* v_unused_28_; 
v_unused_28_ = lean_ctor_get(v___x_19_, 0);
lean_dec(v_unused_28_);
v___x_22_ = v___x_19_;
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_toZero_20_);
lean_dec(v___x_19_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_25_; 
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 0, v___x_17_);
v___x_25_ = v___x_22_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_17_);
lean_ctor_set(v_reuseFailAlloc_26_, 1, v_toZero_20_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulZeroOneClass(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(v_inst_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v_toSemigroup_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v_toZero_36_; lean_object* v___x_38_; uint8_t v_isShared_39_; uint8_t v_isSharedCheck_44_; 
v_toSemigroup_33_ = lean_ctor_get(v_inst_32_, 0);
lean_inc(v_toSemigroup_33_);
v___x_34_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_32_);
v___x_35_ = lp_mathlib_MulOpposite_instMulZeroClass___redArg(v___x_34_);
v_toZero_36_ = lean_ctor_get(v___x_35_, 1);
v_isSharedCheck_44_ = !lean_is_exclusive(v___x_35_);
if (v_isSharedCheck_44_ == 0)
{
lean_object* v_unused_45_; 
v_unused_45_ = lean_ctor_get(v___x_35_, 0);
lean_dec(v_unused_45_);
v___x_38_ = v___x_35_;
v_isShared_39_ = v_isSharedCheck_44_;
goto v_resetjp_37_;
}
else
{
lean_inc(v_toZero_36_);
lean_dec(v___x_35_);
v___x_38_ = lean_box(0);
v_isShared_39_ = v_isSharedCheck_44_;
goto v_resetjp_37_;
}
v_resetjp_37_:
{
lean_object* v___f_40_; lean_object* v___x_42_; 
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_40_, 0, v_toSemigroup_33_);
if (v_isShared_39_ == 0)
{
lean_ctor_set(v___x_38_, 0, v___f_40_);
v___x_42_ = v___x_38_;
goto v_reusejp_41_;
}
else
{
lean_object* v_reuseFailAlloc_43_; 
v_reuseFailAlloc_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_43_, 0, v___f_40_);
lean_ctor_set(v_reuseFailAlloc_43_, 1, v_toZero_36_);
v___x_42_ = v_reuseFailAlloc_43_;
goto v_reusejp_41_;
}
v_reusejp_41_:
{
return v___x_42_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroupWithZero(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_MulOpposite_instSemigroupWithZero___redArg(v_inst_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero___redArg(lean_object* v_inst_49_){
_start:
{
lean_object* v_toMonoid_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v_toZero_54_; lean_object* v___x_56_; uint8_t v_isShared_57_; uint8_t v_isSharedCheck_61_; 
v_toMonoid_50_ = lean_ctor_get(v_inst_49_, 0);
lean_inc_ref(v_toMonoid_50_);
v___x_51_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_toMonoid_50_);
v___x_52_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_49_);
v___x_53_ = lp_mathlib_MulOpposite_instMulZeroOneClass___redArg(v___x_52_);
v_toZero_54_ = lean_ctor_get(v___x_53_, 1);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_53_);
if (v_isSharedCheck_61_ == 0)
{
lean_object* v_unused_62_; 
v_unused_62_ = lean_ctor_get(v___x_53_, 0);
lean_dec(v_unused_62_);
v___x_56_ = v___x_53_;
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
else
{
lean_inc(v_toZero_54_);
lean_dec(v___x_53_);
v___x_56_ = lean_box(0);
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
v_resetjp_55_:
{
lean_object* v___x_59_; 
if (v_isShared_57_ == 0)
{
lean_ctor_set(v___x_56_, 0, v___x_51_);
v___x_59_ = v___x_56_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v___x_51_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v_toZero_54_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoidWithZero(lean_object* v_00_u03b1_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_MulOpposite_instMonoidWithZero___redArg(v_inst_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero___redArg(lean_object* v_inst_66_){
_start:
{
lean_object* v_toMonoidWithZero_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v_toInv_71_; lean_object* v_toDiv_72_; lean_object* v_toZPow_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_80_; 
v_toMonoidWithZero_67_ = lean_ctor_get(v_inst_66_, 0);
lean_inc_ref(v_toMonoidWithZero_67_);
v___x_68_ = lp_mathlib_MulOpposite_instMonoidWithZero___redArg(v_toMonoidWithZero_67_);
v___x_69_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_66_);
v___x_70_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v___x_69_);
v_toInv_71_ = lean_ctor_get(v___x_70_, 1);
v_toDiv_72_ = lean_ctor_get(v___x_70_, 2);
v_toZPow_73_ = lean_ctor_get(v___x_70_, 3);
v_isSharedCheck_80_ = !lean_is_exclusive(v___x_70_);
if (v_isSharedCheck_80_ == 0)
{
lean_object* v_unused_81_; 
v_unused_81_ = lean_ctor_get(v___x_70_, 0);
lean_dec(v_unused_81_);
v___x_75_ = v___x_70_;
v_isShared_76_ = v_isSharedCheck_80_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_toZPow_73_);
lean_inc(v_toDiv_72_);
lean_inc(v_toInv_71_);
lean_dec(v___x_70_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_80_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___x_78_; 
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 0, v___x_68_);
v___x_78_ = v___x_75_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_79_; 
v_reuseFailAlloc_79_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_79_, 0, v___x_68_);
lean_ctor_set(v_reuseFailAlloc_79_, 1, v_toInv_71_);
lean_ctor_set(v_reuseFailAlloc_79_, 2, v_toDiv_72_);
lean_ctor_set(v_reuseFailAlloc_79_, 3, v_toZPow_73_);
v___x_78_ = v_reuseFailAlloc_79_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
return v___x_78_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroupWithZero(lean_object* v_00_u03b1_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_MulOpposite_instGroupWithZero___redArg(v_inst_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass___redArg(lean_object* v_inst_85_){
_start:
{
lean_object* v_toMul_86_; lean_object* v_toZero_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_95_; 
v_toMul_86_ = lean_ctor_get(v_inst_85_, 0);
v_toZero_87_ = lean_ctor_get(v_inst_85_, 1);
v_isSharedCheck_95_ = !lean_is_exclusive(v_inst_85_);
if (v_isSharedCheck_95_ == 0)
{
v___x_89_ = v_inst_85_;
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_toZero_87_);
lean_inc(v_toMul_86_);
lean_dec(v_inst_85_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___f_91_; lean_object* v___x_93_; 
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_91_, 0, v_toMul_86_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 0, v___f_91_);
v___x_93_ = v___x_89_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___f_91_);
lean_ctor_set(v_reuseFailAlloc_94_, 1, v_toZero_87_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroClass(lean_object* v_00_u03b1_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_AddOpposite_instMulZeroClass___redArg(v_inst_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(lean_object* v_inst_99_){
_start:
{
lean_object* v_toMulOneClass_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v_toZero_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_111_; 
v_toMulOneClass_100_ = lean_ctor_get(v_inst_99_, 0);
lean_inc_ref(v_toMulOneClass_100_);
v___x_101_ = lp_mathlib_AddOpposite_instMulOneClass___redArg(v_toMulOneClass_100_);
v___x_102_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_99_);
v___x_103_ = lp_mathlib_AddOpposite_instMulZeroClass___redArg(v___x_102_);
v_toZero_104_ = lean_ctor_get(v___x_103_, 1);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_103_);
if (v_isSharedCheck_111_ == 0)
{
lean_object* v_unused_112_; 
v_unused_112_ = lean_ctor_get(v___x_103_, 0);
lean_dec(v_unused_112_);
v___x_106_ = v___x_103_;
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_toZero_104_);
lean_dec(v___x_103_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_109_; 
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 0, v___x_101_);
v___x_109_ = v___x_106_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v___x_101_);
lean_ctor_set(v_reuseFailAlloc_110_, 1, v_toZero_104_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulZeroOneClass(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(v_inst_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero___redArg(lean_object* v_inst_116_){
_start:
{
lean_object* v_toSemigroup_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v_toZero_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_128_; 
v_toSemigroup_117_ = lean_ctor_get(v_inst_116_, 0);
lean_inc(v_toSemigroup_117_);
v___x_118_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_116_);
v___x_119_ = lp_mathlib_AddOpposite_instMulZeroClass___redArg(v___x_118_);
v_toZero_120_ = lean_ctor_get(v___x_119_, 1);
v_isSharedCheck_128_ = !lean_is_exclusive(v___x_119_);
if (v_isSharedCheck_128_ == 0)
{
lean_object* v_unused_129_; 
v_unused_129_ = lean_ctor_get(v___x_119_, 0);
lean_dec(v_unused_129_);
v___x_122_ = v___x_119_;
v_isShared_123_ = v_isSharedCheck_128_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_toZero_120_);
lean_dec(v___x_119_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_128_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___f_124_; lean_object* v___x_126_; 
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_124_, 0, v_toSemigroup_117_);
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 0, v___f_124_);
v___x_126_ = v___x_122_;
goto v_reusejp_125_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v___f_124_);
lean_ctor_set(v_reuseFailAlloc_127_, 1, v_toZero_120_);
v___x_126_ = v_reuseFailAlloc_127_;
goto v_reusejp_125_;
}
v_reusejp_125_:
{
return v___x_126_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroupWithZero(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_AddOpposite_instSemigroupWithZero___redArg(v_inst_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero___redArg(lean_object* v_inst_133_){
_start:
{
lean_object* v_toMonoid_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v_toZero_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v_toMonoid_134_ = lean_ctor_get(v_inst_133_, 0);
lean_inc_ref(v_toMonoid_134_);
v___x_135_ = lp_mathlib_AddOpposite_instMonoid___redArg(v_toMonoid_134_);
v___x_136_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_133_);
v___x_137_ = lp_mathlib_AddOpposite_instMulZeroOneClass___redArg(v___x_136_);
v_toZero_138_ = lean_ctor_get(v___x_137_, 1);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_137_);
if (v_isSharedCheck_145_ == 0)
{
lean_object* v_unused_146_; 
v_unused_146_ = lean_ctor_get(v___x_137_, 0);
lean_dec(v_unused_146_);
v___x_140_ = v___x_137_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_toZero_138_);
lean_dec(v___x_137_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
lean_ctor_set(v___x_140_, 0, v___x_135_);
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v___x_135_);
lean_ctor_set(v_reuseFailAlloc_144_, 1, v_toZero_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoidWithZero(lean_object* v_00_u03b1_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_AddOpposite_instMonoidWithZero___redArg(v_inst_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero___redArg(lean_object* v_inst_150_){
_start:
{
lean_object* v_toMonoidWithZero_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v_toInv_155_; lean_object* v_toDiv_156_; lean_object* v_toZPow_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_164_; 
v_toMonoidWithZero_151_ = lean_ctor_get(v_inst_150_, 0);
lean_inc_ref(v_toMonoidWithZero_151_);
v___x_152_ = lp_mathlib_AddOpposite_instMonoidWithZero___redArg(v_toMonoidWithZero_151_);
v___x_153_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_150_);
v___x_154_ = lp_mathlib_AddOpposite_instDivInvMonoid___redArg(v___x_153_);
v_toInv_155_ = lean_ctor_get(v___x_154_, 1);
v_toDiv_156_ = lean_ctor_get(v___x_154_, 2);
v_toZPow_157_ = lean_ctor_get(v___x_154_, 3);
v_isSharedCheck_164_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_164_ == 0)
{
lean_object* v_unused_165_; 
v_unused_165_ = lean_ctor_get(v___x_154_, 0);
lean_dec(v_unused_165_);
v___x_159_ = v___x_154_;
v_isShared_160_ = v_isSharedCheck_164_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_toZPow_157_);
lean_inc(v_toDiv_156_);
lean_inc(v_toInv_155_);
lean_dec(v___x_154_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_164_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___x_162_; 
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 0, v___x_152_);
v___x_162_ = v___x_159_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_163_; 
v_reuseFailAlloc_163_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_163_, 0, v___x_152_);
lean_ctor_set(v_reuseFailAlloc_163_, 1, v_toInv_155_);
lean_ctor_set(v_reuseFailAlloc_163_, 2, v_toDiv_156_);
lean_ctor_set(v_reuseFailAlloc_163_, 3, v_toZPow_157_);
v___x_162_ = v_reuseFailAlloc_163_;
goto v_reusejp_161_;
}
v_reusejp_161_:
{
return v___x_162_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroupWithZero(lean_object* v_00_u03b1_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_mathlib_AddOpposite_instGroupWithZero___redArg(v_inst_167_);
return v___x_168_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
