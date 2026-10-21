// Lean compiler output
// Module: Mathlib.Algebra.Order.GroupWithZero.Synonym
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Algebra.Order.Group.Synonym
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
lean_object* lp_mathlib_OrderDual_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Lex_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Lex_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Lex_instMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroClass___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toMul_2_; lean_object* v_toZero_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_toMul_2_ = lean_ctor_get(v_inst_1_, 0);
v_toZero_3_ = lean_ctor_get(v_inst_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_inst_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toZero_3_);
lean_inc(v_toMul_2_);
lean_dec(v_inst_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_8_; 
if (v_isShared_6_ == 0)
{
v___x_8_ = v___x_5_;
goto v_reusejp_7_;
}
else
{
lean_object* v_reuseFailAlloc_9_; 
v_reuseFailAlloc_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_toMul_2_);
lean_ctor_set(v_reuseFailAlloc_9_, 1, v_toZero_3_);
v___x_8_ = v_reuseFailAlloc_9_;
goto v_reusejp_7_;
}
v_reusejp_7_:
{
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroClass(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_OrderDual_instMulZeroClass___redArg(v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroOneClass___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v_toMulOneClass_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v_toZero_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_25_; 
v_toMulOneClass_15_ = lean_ctor_get(v_inst_14_, 0);
lean_inc_ref(v_toMulOneClass_15_);
v___x_16_ = lp_mathlib_OrderDual_instMulOneClass___redArg(v_toMulOneClass_15_);
v___x_17_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_14_);
v_toZero_18_ = lean_ctor_get(v___x_17_, 1);
v_isSharedCheck_25_ = !lean_is_exclusive(v___x_17_);
if (v_isSharedCheck_25_ == 0)
{
lean_object* v_unused_26_; 
v_unused_26_ = lean_ctor_get(v___x_17_, 0);
lean_dec(v_unused_26_);
v___x_20_ = v___x_17_;
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_toZero_18_);
lean_dec(v___x_17_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_23_; 
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 0, v___x_16_);
v___x_23_ = v___x_20_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v___x_16_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v_toZero_18_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMulZeroOneClass(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_OrderDual_instMulZeroOneClass___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemigroupWithZero___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v_toSemigroup_31_; lean_object* v___x_32_; lean_object* v_toZero_33_; lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_40_; 
v_toSemigroup_31_ = lean_ctor_get(v_inst_30_, 0);
lean_inc(v_toSemigroup_31_);
v___x_32_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_30_);
v_toZero_33_ = lean_ctor_get(v___x_32_, 1);
v_isSharedCheck_40_ = !lean_is_exclusive(v___x_32_);
if (v_isSharedCheck_40_ == 0)
{
lean_object* v_unused_41_; 
v_unused_41_ = lean_ctor_get(v___x_32_, 0);
lean_dec(v_unused_41_);
v___x_35_ = v___x_32_;
v_isShared_36_ = v_isSharedCheck_40_;
goto v_resetjp_34_;
}
else
{
lean_inc(v_toZero_33_);
lean_dec(v___x_32_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_40_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v___x_38_; 
if (v_isShared_36_ == 0)
{
lean_ctor_set(v___x_35_, 0, v_toSemigroup_31_);
v___x_38_ = v___x_35_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v_toSemigroup_31_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v_toZero_33_);
v___x_38_ = v_reuseFailAlloc_39_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
return v___x_38_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemigroupWithZero(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_OrderDual_instSemigroupWithZero___redArg(v_inst_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMonoidWithZero___redArg(lean_object* v_inst_45_){
_start:
{
lean_object* v_toMonoid_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v_toZero_50_; lean_object* v___x_52_; uint8_t v_isShared_53_; uint8_t v_isSharedCheck_57_; 
v_toMonoid_46_ = lean_ctor_get(v_inst_45_, 0);
lean_inc_ref(v_toMonoid_46_);
v___x_47_ = lp_mathlib_OrderDual_instMonoid___redArg(v_toMonoid_46_);
v___x_48_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_45_);
v___x_49_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_48_);
v_toZero_50_ = lean_ctor_get(v___x_49_, 1);
v_isSharedCheck_57_ = !lean_is_exclusive(v___x_49_);
if (v_isSharedCheck_57_ == 0)
{
lean_object* v_unused_58_; 
v_unused_58_ = lean_ctor_get(v___x_49_, 0);
lean_dec(v_unused_58_);
v___x_52_ = v___x_49_;
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
else
{
lean_inc(v_toZero_50_);
lean_dec(v___x_49_);
v___x_52_ = lean_box(0);
v_isShared_53_ = v_isSharedCheck_57_;
goto v_resetjp_51_;
}
v_resetjp_51_:
{
lean_object* v___x_55_; 
if (v_isShared_53_ == 0)
{
lean_ctor_set(v___x_52_, 0, v___x_47_);
v___x_55_ = v___x_52_;
goto v_reusejp_54_;
}
else
{
lean_object* v_reuseFailAlloc_56_; 
v_reuseFailAlloc_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_56_, 0, v___x_47_);
lean_ctor_set(v_reuseFailAlloc_56_, 1, v_toZero_50_);
v___x_55_ = v_reuseFailAlloc_56_;
goto v_reusejp_54_;
}
v_reusejp_54_:
{
return v___x_55_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMonoidWithZero(lean_object* v_00_u03b1_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_OrderDual_instMonoidWithZero___redArg(v_inst_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommMonoidWithZero___redArg(lean_object* v_inst_62_){
_start:
{
lean_object* v_toCommMonoid_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v_toZero_68_; lean_object* v___x_70_; uint8_t v_isShared_71_; uint8_t v_isSharedCheck_75_; 
v_toCommMonoid_63_ = lean_ctor_get(v_inst_62_, 0);
lean_inc_ref(v_toCommMonoid_63_);
v___x_64_ = lp_mathlib_OrderDual_instMonoid___redArg(v_toCommMonoid_63_);
v___x_65_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_62_);
v___x_66_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_65_);
v___x_67_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_66_);
v_toZero_68_ = lean_ctor_get(v___x_67_, 1);
v_isSharedCheck_75_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_75_ == 0)
{
lean_object* v_unused_76_; 
v_unused_76_ = lean_ctor_get(v___x_67_, 0);
lean_dec(v_unused_76_);
v___x_70_ = v___x_67_;
v_isShared_71_ = v_isSharedCheck_75_;
goto v_resetjp_69_;
}
else
{
lean_inc(v_toZero_68_);
lean_dec(v___x_67_);
v___x_70_ = lean_box(0);
v_isShared_71_ = v_isSharedCheck_75_;
goto v_resetjp_69_;
}
v_resetjp_69_:
{
lean_object* v___x_73_; 
if (v_isShared_71_ == 0)
{
lean_ctor_set(v___x_70_, 0, v___x_64_);
v___x_73_ = v___x_70_;
goto v_reusejp_72_;
}
else
{
lean_object* v_reuseFailAlloc_74_; 
v_reuseFailAlloc_74_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_74_, 0, v___x_64_);
lean_ctor_set(v_reuseFailAlloc_74_, 1, v_toZero_68_);
v___x_73_ = v_reuseFailAlloc_74_;
goto v_reusejp_72_;
}
v_reusejp_72_:
{
return v___x_73_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommMonoidWithZero(lean_object* v_00_u03b1_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_OrderDual_instCommMonoidWithZero___redArg(v_inst_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGroupWithZero___redArg(lean_object* v_inst_80_){
_start:
{
lean_object* v_toMonoidWithZero_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v_toInv_84_; lean_object* v_toDiv_85_; lean_object* v___x_86_; lean_object* v_toZPow_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_94_; 
v_toMonoidWithZero_81_ = lean_ctor_get(v_inst_80_, 0);
lean_inc_ref(v_toMonoidWithZero_81_);
v___x_82_ = lp_mathlib_OrderDual_instMonoidWithZero___redArg(v_toMonoidWithZero_81_);
v___x_83_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_80_);
v_toInv_84_ = lean_ctor_get(v___x_83_, 1);
lean_inc(v_toInv_84_);
v_toDiv_85_ = lean_ctor_get(v___x_83_, 2);
lean_inc(v_toDiv_85_);
v___x_86_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_83_);
v_toZPow_87_ = lean_ctor_get(v___x_86_, 3);
v_isSharedCheck_94_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_94_ == 0)
{
lean_object* v_unused_95_; lean_object* v_unused_96_; lean_object* v_unused_97_; 
v_unused_95_ = lean_ctor_get(v___x_86_, 2);
lean_dec(v_unused_95_);
v_unused_96_ = lean_ctor_get(v___x_86_, 1);
lean_dec(v_unused_96_);
v_unused_97_ = lean_ctor_get(v___x_86_, 0);
lean_dec(v_unused_97_);
v___x_89_ = v___x_86_;
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_toZPow_87_);
lean_dec(v___x_86_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_92_; 
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 2, v_toDiv_85_);
lean_ctor_set(v___x_89_, 1, v_toInv_84_);
lean_ctor_set(v___x_89_, 0, v___x_82_);
v___x_92_ = v___x_89_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v___x_82_);
lean_ctor_set(v_reuseFailAlloc_93_, 1, v_toInv_84_);
lean_ctor_set(v_reuseFailAlloc_93_, 2, v_toDiv_85_);
lean_ctor_set(v_reuseFailAlloc_93_, 3, v_toZPow_87_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGroupWithZero(lean_object* v_00_u03b1_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_OrderDual_instGroupWithZero___redArg(v_inst_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommGroupWithZero___redArg(lean_object* v_inst_101_){
_start:
{
lean_object* v_toCommMonoidWithZero_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v_toInv_106_; lean_object* v_toDiv_107_; lean_object* v___x_108_; lean_object* v_toZPow_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_116_; 
v_toCommMonoidWithZero_102_ = lean_ctor_get(v_inst_101_, 0);
lean_inc_ref(v_toCommMonoidWithZero_102_);
v___x_103_ = lp_mathlib_OrderDual_instCommMonoidWithZero___redArg(v_toCommMonoidWithZero_102_);
v___x_104_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v_inst_101_);
v___x_105_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_104_);
v_toInv_106_ = lean_ctor_get(v___x_105_, 1);
lean_inc(v_toInv_106_);
v_toDiv_107_ = lean_ctor_get(v___x_105_, 2);
lean_inc(v_toDiv_107_);
v___x_108_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_105_);
v_toZPow_109_ = lean_ctor_get(v___x_108_, 3);
v_isSharedCheck_116_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_116_ == 0)
{
lean_object* v_unused_117_; lean_object* v_unused_118_; lean_object* v_unused_119_; 
v_unused_117_ = lean_ctor_get(v___x_108_, 2);
lean_dec(v_unused_117_);
v_unused_118_ = lean_ctor_get(v___x_108_, 1);
lean_dec(v_unused_118_);
v_unused_119_ = lean_ctor_get(v___x_108_, 0);
lean_dec(v_unused_119_);
v___x_111_ = v___x_108_;
v_isShared_112_ = v_isSharedCheck_116_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_toZPow_109_);
lean_dec(v___x_108_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_116_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v___x_114_; 
if (v_isShared_112_ == 0)
{
lean_ctor_set(v___x_111_, 2, v_toDiv_107_);
lean_ctor_set(v___x_111_, 1, v_toInv_106_);
lean_ctor_set(v___x_111_, 0, v___x_103_);
v___x_114_ = v___x_111_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v___x_103_);
lean_ctor_set(v_reuseFailAlloc_115_, 1, v_toInv_106_);
lean_ctor_set(v_reuseFailAlloc_115_, 2, v_toDiv_107_);
lean_ctor_set(v_reuseFailAlloc_115_, 3, v_toZPow_109_);
v___x_114_ = v_reuseFailAlloc_115_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
return v___x_114_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCommGroupWithZero(lean_object* v_00_u03b1_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_OrderDual_instCommGroupWithZero___redArg(v_inst_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroClass___redArg(lean_object* v_inst_123_){
_start:
{
lean_object* v_toMul_124_; lean_object* v_toZero_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_132_; 
v_toMul_124_ = lean_ctor_get(v_inst_123_, 0);
v_toZero_125_ = lean_ctor_get(v_inst_123_, 1);
v_isSharedCheck_132_ = !lean_is_exclusive(v_inst_123_);
if (v_isSharedCheck_132_ == 0)
{
v___x_127_ = v_inst_123_;
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_toZero_125_);
lean_inc(v_toMul_124_);
lean_dec(v_inst_123_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_toMul_124_);
lean_ctor_set(v_reuseFailAlloc_131_, 1, v_toZero_125_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
return v___x_130_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroClass(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_Lex_instMulZeroClass___redArg(v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroOneClass___redArg(lean_object* v_inst_136_){
_start:
{
lean_object* v_toMulOneClass_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v_toZero_140_; lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_147_; 
v_toMulOneClass_137_ = lean_ctor_get(v_inst_136_, 0);
lean_inc_ref(v_toMulOneClass_137_);
v___x_138_ = lp_mathlib_Lex_instMulOneClass___redArg(v_toMulOneClass_137_);
v___x_139_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_136_);
v_toZero_140_ = lean_ctor_get(v___x_139_, 1);
v_isSharedCheck_147_ = !lean_is_exclusive(v___x_139_);
if (v_isSharedCheck_147_ == 0)
{
lean_object* v_unused_148_; 
v_unused_148_ = lean_ctor_get(v___x_139_, 0);
lean_dec(v_unused_148_);
v___x_142_ = v___x_139_;
v_isShared_143_ = v_isSharedCheck_147_;
goto v_resetjp_141_;
}
else
{
lean_inc(v_toZero_140_);
lean_dec(v___x_139_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_147_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
lean_object* v___x_145_; 
if (v_isShared_143_ == 0)
{
lean_ctor_set(v___x_142_, 0, v___x_138_);
v___x_145_ = v___x_142_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v___x_138_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v_toZero_140_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMulZeroOneClass(lean_object* v_00_u03b1_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_Lex_instMulZeroOneClass___redArg(v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemigroupWithZero___redArg(lean_object* v_inst_152_){
_start:
{
lean_object* v_toSemigroup_153_; lean_object* v___x_154_; lean_object* v_toZero_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_162_; 
v_toSemigroup_153_ = lean_ctor_get(v_inst_152_, 0);
lean_inc(v_toSemigroup_153_);
v___x_154_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_152_);
v_toZero_155_ = lean_ctor_get(v___x_154_, 1);
v_isSharedCheck_162_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_162_ == 0)
{
lean_object* v_unused_163_; 
v_unused_163_ = lean_ctor_get(v___x_154_, 0);
lean_dec(v_unused_163_);
v___x_157_ = v___x_154_;
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_toZero_155_);
lean_dec(v___x_154_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_160_; 
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 0, v_toSemigroup_153_);
v___x_160_ = v___x_157_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_toSemigroup_153_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_toZero_155_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemigroupWithZero(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_Lex_instSemigroupWithZero___redArg(v_inst_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMonoidWithZero___redArg(lean_object* v_inst_167_){
_start:
{
lean_object* v_toMonoid_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v_toZero_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
v_toMonoid_168_ = lean_ctor_get(v_inst_167_, 0);
lean_inc_ref(v_toMonoid_168_);
v___x_169_ = lp_mathlib_Lex_instMonoid___redArg(v_toMonoid_168_);
v___x_170_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_167_);
v___x_171_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_170_);
v_toZero_172_ = lean_ctor_get(v___x_171_, 1);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_179_ == 0)
{
lean_object* v_unused_180_; 
v_unused_180_ = lean_ctor_get(v___x_171_, 0);
lean_dec(v_unused_180_);
v___x_174_ = v___x_171_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_toZero_172_);
lean_dec(v___x_171_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 0, v___x_169_);
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v___x_169_);
lean_ctor_set(v_reuseFailAlloc_178_, 1, v_toZero_172_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instMonoidWithZero(lean_object* v_00_u03b1_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_mathlib_Lex_instMonoidWithZero___redArg(v_inst_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommMonoidWithZero___redArg(lean_object* v_inst_184_){
_start:
{
lean_object* v_toCommMonoid_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v_toZero_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_197_; 
v_toCommMonoid_185_ = lean_ctor_get(v_inst_184_, 0);
lean_inc_ref(v_toCommMonoid_185_);
v___x_186_ = lp_mathlib_Lex_instMonoid___redArg(v_toCommMonoid_185_);
v___x_187_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_184_);
v___x_188_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_187_);
v___x_189_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_188_);
v_toZero_190_ = lean_ctor_get(v___x_189_, 1);
v_isSharedCheck_197_ = !lean_is_exclusive(v___x_189_);
if (v_isSharedCheck_197_ == 0)
{
lean_object* v_unused_198_; 
v_unused_198_ = lean_ctor_get(v___x_189_, 0);
lean_dec(v_unused_198_);
v___x_192_ = v___x_189_;
v_isShared_193_ = v_isSharedCheck_197_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_toZero_190_);
lean_dec(v___x_189_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_197_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_195_; 
if (v_isShared_193_ == 0)
{
lean_ctor_set(v___x_192_, 0, v___x_186_);
v___x_195_ = v___x_192_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_186_);
lean_ctor_set(v_reuseFailAlloc_196_, 1, v_toZero_190_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommMonoidWithZero(lean_object* v_00_u03b1_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_Lex_instCommMonoidWithZero___redArg(v_inst_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instGroupWithZero___redArg(lean_object* v_inst_202_){
_start:
{
lean_object* v_toMonoidWithZero_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v_toInv_206_; lean_object* v_toDiv_207_; lean_object* v___x_208_; lean_object* v_toZPow_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
v_toMonoidWithZero_203_ = lean_ctor_get(v_inst_202_, 0);
lean_inc_ref(v_toMonoidWithZero_203_);
v___x_204_ = lp_mathlib_Lex_instMonoidWithZero___redArg(v_toMonoidWithZero_203_);
v___x_205_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_202_);
v_toInv_206_ = lean_ctor_get(v___x_205_, 1);
lean_inc(v_toInv_206_);
v_toDiv_207_ = lean_ctor_get(v___x_205_, 2);
lean_inc(v_toDiv_207_);
v___x_208_ = lp_mathlib_Lex_instDivInvMonoid___redArg(v___x_205_);
v_toZPow_209_ = lean_ctor_get(v___x_208_, 3);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_208_);
if (v_isSharedCheck_216_ == 0)
{
lean_object* v_unused_217_; lean_object* v_unused_218_; lean_object* v_unused_219_; 
v_unused_217_ = lean_ctor_get(v___x_208_, 2);
lean_dec(v_unused_217_);
v_unused_218_ = lean_ctor_get(v___x_208_, 1);
lean_dec(v_unused_218_);
v_unused_219_ = lean_ctor_get(v___x_208_, 0);
lean_dec(v_unused_219_);
v___x_211_ = v___x_208_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_toZPow_209_);
lean_dec(v___x_208_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
lean_ctor_set(v___x_211_, 2, v_toDiv_207_);
lean_ctor_set(v___x_211_, 1, v_toInv_206_);
lean_ctor_set(v___x_211_, 0, v___x_204_);
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v___x_204_);
lean_ctor_set(v_reuseFailAlloc_215_, 1, v_toInv_206_);
lean_ctor_set(v_reuseFailAlloc_215_, 2, v_toDiv_207_);
lean_ctor_set(v_reuseFailAlloc_215_, 3, v_toZPow_209_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instGroupWithZero(lean_object* v_00_u03b1_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_Lex_instGroupWithZero___redArg(v_inst_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommGroupWithZero___redArg(lean_object* v_inst_223_){
_start:
{
lean_object* v_toCommMonoidWithZero_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v_toInv_228_; lean_object* v_toDiv_229_; lean_object* v___x_230_; lean_object* v_toZPow_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_238_; 
v_toCommMonoidWithZero_224_ = lean_ctor_get(v_inst_223_, 0);
lean_inc_ref(v_toCommMonoidWithZero_224_);
v___x_225_ = lp_mathlib_Lex_instCommMonoidWithZero___redArg(v_toCommMonoidWithZero_224_);
v___x_226_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v_inst_223_);
v___x_227_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_226_);
v_toInv_228_ = lean_ctor_get(v___x_227_, 1);
lean_inc(v_toInv_228_);
v_toDiv_229_ = lean_ctor_get(v___x_227_, 2);
lean_inc(v_toDiv_229_);
v___x_230_ = lp_mathlib_Lex_instDivInvMonoid___redArg(v___x_227_);
v_toZPow_231_ = lean_ctor_get(v___x_230_, 3);
v_isSharedCheck_238_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_238_ == 0)
{
lean_object* v_unused_239_; lean_object* v_unused_240_; lean_object* v_unused_241_; 
v_unused_239_ = lean_ctor_get(v___x_230_, 2);
lean_dec(v_unused_239_);
v_unused_240_ = lean_ctor_get(v___x_230_, 1);
lean_dec(v_unused_240_);
v_unused_241_ = lean_ctor_get(v___x_230_, 0);
lean_dec(v_unused_241_);
v___x_233_ = v___x_230_;
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_toZPow_231_);
lean_dec(v___x_230_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_236_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 2, v_toDiv_229_);
lean_ctor_set(v___x_233_, 1, v_toInv_228_);
lean_ctor_set(v___x_233_, 0, v___x_225_);
v___x_236_ = v___x_233_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v___x_225_);
lean_ctor_set(v_reuseFailAlloc_237_, 1, v_toInv_228_);
lean_ctor_set(v_reuseFailAlloc_237_, 2, v_toDiv_229_);
lean_ctor_set(v_reuseFailAlloc_237_, 3, v_toZPow_231_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instCommGroupWithZero(lean_object* v_00_u03b1_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lp_mathlib_Lex_instCommGroupWithZero___redArg(v_inst_243_);
return v___x_244_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Synonym(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Synonym(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Synonym(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Synonym(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Synonym(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Synonym(builtin);
}
#ifdef __cplusplus
}
#endif
