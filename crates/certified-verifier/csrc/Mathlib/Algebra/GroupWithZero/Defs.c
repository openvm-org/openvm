// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.DivInvMonoid public import Mathlib.Basic.Nontrivial.Defs public import Mathlib.Basic.Logic.Basic public import Batteries.Tactic.SeqFocus
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
LEAN_EXPORT lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toSemigroup_2_; lean_object* v_toZero_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_toSemigroup_2_ = lean_ctor_get(v_self_1_, 0);
v_toZero_3_ = lean_ctor_get(v_self_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_self_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toZero_3_);
lean_inc(v_toSemigroup_2_);
lean_dec(v_self_1_);
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
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_toSemigroup_2_);
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
LEAN_EXPORT lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass(lean_object* v_S_u2080_11_, lean_object* v_self_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_self_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object* v_self_14_){
_start:
{
lean_object* v_toMulOneClass_15_; lean_object* v_toZero_16_; lean_object* v_toMul_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_24_; 
v_toMulOneClass_15_ = lean_ctor_get(v_self_14_, 0);
lean_inc_ref(v_toMulOneClass_15_);
v_toZero_16_ = lean_ctor_get(v_self_14_, 1);
lean_inc(v_toZero_16_);
lean_dec_ref(v_self_14_);
v_toMul_17_ = lean_ctor_get(v_toMulOneClass_15_, 1);
v_isSharedCheck_24_ = !lean_is_exclusive(v_toMulOneClass_15_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v_toMulOneClass_15_, 0);
lean_dec(v_unused_25_);
v___x_19_ = v_toMulOneClass_15_;
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_toMul_17_);
lean_dec(v_toMulOneClass_15_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_22_; 
if (v_isShared_20_ == 0)
{
lean_ctor_set(v___x_19_, 1, v_toZero_16_);
lean_ctor_set(v___x_19_, 0, v_toMul_17_);
v___x_22_ = v___x_19_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_toMul_17_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_toZero_16_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass(lean_object* v_M_u2080_26_, lean_object* v_self_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_self_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object* v_self_29_){
_start:
{
lean_object* v_toMonoid_30_; lean_object* v_toZero_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_41_; 
v_toMonoid_30_ = lean_ctor_get(v_self_29_, 0);
v_toZero_31_ = lean_ctor_get(v_self_29_, 1);
v_isSharedCheck_41_ = !lean_is_exclusive(v_self_29_);
if (v_isSharedCheck_41_ == 0)
{
v___x_33_ = v_self_29_;
v_isShared_34_ = v_isSharedCheck_41_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_toZero_31_);
lean_inc(v_toMonoid_30_);
lean_dec(v_self_29_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_41_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v_toOne_35_; lean_object* v_toMul_36_; lean_object* v___x_38_; 
v_toOne_35_ = lean_ctor_get(v_toMonoid_30_, 0);
lean_inc(v_toOne_35_);
v_toMul_36_ = lean_ctor_get(v_toMonoid_30_, 1);
lean_inc(v_toMul_36_);
lean_dec_ref(v_toMonoid_30_);
if (v_isShared_34_ == 0)
{
lean_ctor_set(v___x_33_, 1, v_toMul_36_);
lean_ctor_set(v___x_33_, 0, v_toOne_35_);
v___x_38_ = v___x_33_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v_toOne_35_);
lean_ctor_set(v_reuseFailAlloc_40_, 1, v_toMul_36_);
v___x_38_ = v_reuseFailAlloc_40_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
lean_object* v___x_39_; 
v___x_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
lean_ctor_set(v___x_39_, 1, v_toZero_31_);
return v___x_39_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass(lean_object* v_M_u2080_42_, lean_object* v_self_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_self_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toSemigroupWithZero___redArg(lean_object* v_self_45_){
_start:
{
lean_object* v_toMonoid_46_; lean_object* v_toZero_47_; lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_55_; 
v_toMonoid_46_ = lean_ctor_get(v_self_45_, 0);
v_toZero_47_ = lean_ctor_get(v_self_45_, 1);
v_isSharedCheck_55_ = !lean_is_exclusive(v_self_45_);
if (v_isSharedCheck_55_ == 0)
{
v___x_49_ = v_self_45_;
v_isShared_50_ = v_isSharedCheck_55_;
goto v_resetjp_48_;
}
else
{
lean_inc(v_toZero_47_);
lean_inc(v_toMonoid_46_);
lean_dec(v_self_45_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_55_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v_toMul_51_; lean_object* v___x_53_; 
v_toMul_51_ = lean_ctor_get(v_toMonoid_46_, 1);
lean_inc(v_toMul_51_);
lean_dec_ref(v_toMonoid_46_);
if (v_isShared_50_ == 0)
{
lean_ctor_set(v___x_49_, 0, v_toMul_51_);
v___x_53_ = v___x_49_;
goto v_reusejp_52_;
}
else
{
lean_object* v_reuseFailAlloc_54_; 
v_reuseFailAlloc_54_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_54_, 0, v_toMul_51_);
lean_ctor_set(v_reuseFailAlloc_54_, 1, v_toZero_47_);
v___x_53_ = v_reuseFailAlloc_54_;
goto v_reusejp_52_;
}
v_reusejp_52_:
{
return v___x_53_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZero_toSemigroupWithZero(lean_object* v_M_u2080_56_, lean_object* v_self_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_MonoidWithZero_toSemigroupWithZero___redArg(v_self_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object* v_self_59_){
_start:
{
lean_object* v_toCommMonoid_60_; lean_object* v_toZero_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_68_; 
v_toCommMonoid_60_ = lean_ctor_get(v_self_59_, 0);
v_toZero_61_ = lean_ctor_get(v_self_59_, 1);
v_isSharedCheck_68_ = !lean_is_exclusive(v_self_59_);
if (v_isSharedCheck_68_ == 0)
{
v___x_63_ = v_self_59_;
v_isShared_64_ = v_isSharedCheck_68_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_toZero_61_);
lean_inc(v_toCommMonoid_60_);
lean_dec(v_self_59_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_68_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v___x_66_; 
if (v_isShared_64_ == 0)
{
v___x_66_ = v___x_63_;
goto v_reusejp_65_;
}
else
{
lean_object* v_reuseFailAlloc_67_; 
v_reuseFailAlloc_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_67_, 0, v_toCommMonoid_60_);
lean_ctor_set(v_reuseFailAlloc_67_, 1, v_toZero_61_);
v___x_66_ = v_reuseFailAlloc_67_;
goto v_reusejp_65_;
}
v_reusejp_65_:
{
return v___x_66_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero(lean_object* v_M_u2080_69_, lean_object* v_self_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_self_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object* v_self_72_){
_start:
{
lean_object* v_toMonoidWithZero_73_; lean_object* v_toInv_74_; lean_object* v_toDiv_75_; lean_object* v_toZPow_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_84_; 
v_toMonoidWithZero_73_ = lean_ctor_get(v_self_72_, 0);
v_toInv_74_ = lean_ctor_get(v_self_72_, 1);
v_toDiv_75_ = lean_ctor_get(v_self_72_, 2);
v_toZPow_76_ = lean_ctor_get(v_self_72_, 3);
v_isSharedCheck_84_ = !lean_is_exclusive(v_self_72_);
if (v_isSharedCheck_84_ == 0)
{
v___x_78_ = v_self_72_;
v_isShared_79_ = v_isSharedCheck_84_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_toZPow_76_);
lean_inc(v_toDiv_75_);
lean_inc(v_toInv_74_);
lean_inc(v_toMonoidWithZero_73_);
lean_dec(v_self_72_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_84_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v_toMonoid_80_; lean_object* v___x_82_; 
v_toMonoid_80_ = lean_ctor_get(v_toMonoidWithZero_73_, 0);
lean_inc_ref(v_toMonoid_80_);
lean_dec_ref(v_toMonoidWithZero_73_);
if (v_isShared_79_ == 0)
{
lean_ctor_set(v___x_78_, 0, v_toMonoid_80_);
v___x_82_ = v___x_78_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v_toMonoid_80_);
lean_ctor_set(v_reuseFailAlloc_83_, 1, v_toInv_74_);
lean_ctor_set(v_reuseFailAlloc_83_, 2, v_toDiv_75_);
lean_ctor_set(v_reuseFailAlloc_83_, 3, v_toZPow_76_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid(lean_object* v_G_u2080_85_, lean_object* v_self_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_self_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object* v_self_88_){
_start:
{
lean_object* v_toCommMonoidWithZero_89_; lean_object* v_toInv_90_; lean_object* v_toDiv_91_; lean_object* v_toZPow_92_; lean_object* v___x_94_; uint8_t v_isShared_95_; uint8_t v_isSharedCheck_108_; 
v_toCommMonoidWithZero_89_ = lean_ctor_get(v_self_88_, 0);
v_toInv_90_ = lean_ctor_get(v_self_88_, 1);
v_toDiv_91_ = lean_ctor_get(v_self_88_, 2);
v_toZPow_92_ = lean_ctor_get(v_self_88_, 3);
v_isSharedCheck_108_ = !lean_is_exclusive(v_self_88_);
if (v_isSharedCheck_108_ == 0)
{
v___x_94_ = v_self_88_;
v_isShared_95_ = v_isSharedCheck_108_;
goto v_resetjp_93_;
}
else
{
lean_inc(v_toZPow_92_);
lean_inc(v_toDiv_91_);
lean_inc(v_toInv_90_);
lean_inc(v_toCommMonoidWithZero_89_);
lean_dec(v_self_88_);
v___x_94_ = lean_box(0);
v_isShared_95_ = v_isSharedCheck_108_;
goto v_resetjp_93_;
}
v_resetjp_93_:
{
lean_object* v_toCommMonoid_96_; lean_object* v_toZero_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_107_; 
v_toCommMonoid_96_ = lean_ctor_get(v_toCommMonoidWithZero_89_, 0);
v_toZero_97_ = lean_ctor_get(v_toCommMonoidWithZero_89_, 1);
v_isSharedCheck_107_ = !lean_is_exclusive(v_toCommMonoidWithZero_89_);
if (v_isSharedCheck_107_ == 0)
{
v___x_99_ = v_toCommMonoidWithZero_89_;
v_isShared_100_ = v_isSharedCheck_107_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_toZero_97_);
lean_inc(v_toCommMonoid_96_);
lean_dec(v_toCommMonoidWithZero_89_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_107_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_102_; 
if (v_isShared_100_ == 0)
{
v___x_102_ = v___x_99_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v_toCommMonoid_96_);
lean_ctor_set(v_reuseFailAlloc_106_, 1, v_toZero_97_);
v___x_102_ = v_reuseFailAlloc_106_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
lean_object* v___x_104_; 
if (v_isShared_95_ == 0)
{
lean_ctor_set(v___x_94_, 0, v___x_102_);
v___x_104_ = v___x_94_;
goto v_reusejp_103_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v___x_102_);
lean_ctor_set(v_reuseFailAlloc_105_, 1, v_toInv_90_);
lean_ctor_set(v_reuseFailAlloc_105_, 2, v_toDiv_91_);
lean_ctor_set(v_reuseFailAlloc_105_, 3, v_toZPow_92_);
v___x_104_ = v_reuseFailAlloc_105_;
goto v_reusejp_103_;
}
v_reusejp_103_:
{
return v___x_104_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero(lean_object* v_G_u2080_109_, lean_object* v_self_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v_self_110_);
return v___x_111_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
