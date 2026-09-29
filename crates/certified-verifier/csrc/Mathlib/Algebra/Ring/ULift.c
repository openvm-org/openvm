// Lean compiler output
// Module: Mathlib.Algebra.Ring.ULift
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.ULift public import Mathlib.Algebra.Ring.Equiv public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Tactic.PPWithUniv
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
lean_object* lp_mathlib_ULift_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_ULift_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_ULift_addGroup___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_ULift_addCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_ULift_monoid___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distrib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNatCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instIntCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instIntCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instIntCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_ULift_ringEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ULift_ringEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ULift_ringEquiv___closed__0 = (const lean_object*)&lp_mathlib_ULift_ringEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_ULift_ringEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ULift_ringEquiv___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ULift_ringEquiv___closed__1 = (const lean_object*)&lp_mathlib_ULift_ringEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_ULift_ringEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_ULift_ringEquiv___closed__0_value),((lean_object*)&lp_mathlib_ULift_ringEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_ULift_ringEquiv___closed__2 = (const lean_object*)&lp_mathlib_ULift_ringEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_ring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroClass___redArg(lean_object* v_inst_1_){
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
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroClass(lean_object* v_M_u2080_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_ULift_mulZeroClass___redArg(v_inst_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distrib___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v_toMul_16_; lean_object* v_toAdd_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_26_; 
v_toMul_16_ = lean_ctor_get(v_inst_15_, 0);
v_toAdd_17_ = lean_ctor_get(v_inst_15_, 1);
v_isSharedCheck_26_ = !lean_is_exclusive(v_inst_15_);
if (v_isSharedCheck_26_ == 0)
{
v___x_19_ = v_inst_15_;
v_isShared_20_ = v_isSharedCheck_26_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_toAdd_17_);
lean_inc(v_toMul_16_);
lean_dec(v_inst_15_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_26_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___f_21_; lean_object* v___f_22_; lean_object* v___x_24_; 
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_21_, 0, v_toMul_16_);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v_toAdd_17_);
if (v_isShared_20_ == 0)
{
lean_ctor_set(v___x_19_, 1, v___f_22_);
lean_ctor_set(v___x_19_, 0, v___f_21_);
v___x_24_ = v___x_19_;
goto v_reusejp_23_;
}
else
{
lean_object* v_reuseFailAlloc_25_; 
v_reuseFailAlloc_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_25_, 0, v___f_21_);
lean_ctor_set(v_reuseFailAlloc_25_, 1, v___f_22_);
v___x_24_ = v_reuseFailAlloc_25_;
goto v_reusejp_23_;
}
v_reusejp_23_:
{
return v___x_24_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distrib(lean_object* v_R_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_ULift_distrib___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNatCast___redArg___lam__0(lean_object* v_inst_30_, lean_object* v_x_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_apply_1(v_inst_30_, v_x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNatCast___redArg(lean_object* v_inst_33_){
_start:
{
lean_object* v___f_34_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_34_, 0, v_inst_33_);
return v___f_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instNatCast(lean_object* v_R_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___f_37_; 
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_37_, 0, v_inst_36_);
return v___f_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instIntCast___redArg___lam__0(lean_object* v_inst_38_, lean_object* v_x_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_apply_1(v_inst_38_, v_x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instIntCast___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instIntCast(lean_object* v_R_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_45_, 0, v_inst_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoidWithOne___redArg(lean_object* v_inst_46_){
_start:
{
lean_object* v_toNatCast_47_; lean_object* v_toAddMonoid_48_; lean_object* v_toOne_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_58_; 
v_toNatCast_47_ = lean_ctor_get(v_inst_46_, 0);
v_toAddMonoid_48_ = lean_ctor_get(v_inst_46_, 1);
v_toOne_49_ = lean_ctor_get(v_inst_46_, 2);
v_isSharedCheck_58_ = !lean_is_exclusive(v_inst_46_);
if (v_isSharedCheck_58_ == 0)
{
v___x_51_ = v_inst_46_;
v_isShared_52_ = v_isSharedCheck_58_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_toOne_49_);
lean_inc(v_toAddMonoid_48_);
lean_inc(v_toNatCast_47_);
lean_dec(v_inst_46_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_58_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___f_53_; lean_object* v___x_54_; lean_object* v___x_56_; 
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instNatCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_53_, 0, v_toNatCast_47_);
v___x_54_ = lp_mathlib_ULift_addMonoid___redArg(v_toAddMonoid_48_);
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 1, v___x_54_);
lean_ctor_set(v___x_51_, 0, v___f_53_);
v___x_56_ = v___x_51_;
goto v_reusejp_55_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v___f_53_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v___x_54_);
lean_ctor_set(v_reuseFailAlloc_57_, 2, v_toOne_49_);
v___x_56_ = v_reuseFailAlloc_57_;
goto v_reusejp_55_;
}
v_reusejp_55_:
{
return v___x_56_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoidWithOne(lean_object* v_R_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_ULift_addMonoidWithOne___redArg(v_inst_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoidWithOne___redArg(lean_object* v_inst_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_ULift_addMonoidWithOne___redArg(v_inst_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoidWithOne(lean_object* v_R_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_ULift_addMonoidWithOne___redArg(v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroupWithOne___redArg(lean_object* v_inst_67_){
_start:
{
lean_object* v_toIntCast_68_; lean_object* v_toAddMonoidWithOne_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_83_; 
v_toIntCast_68_ = lean_ctor_get(v_inst_67_, 0);
lean_inc(v_toIntCast_68_);
v_toAddMonoidWithOne_69_ = lean_ctor_get(v_inst_67_, 1);
lean_inc_ref(v_toAddMonoidWithOne_69_);
v___x_70_ = lp_mathlib_ULift_addMonoidWithOne___redArg(v_toAddMonoidWithOne_69_);
v___x_71_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_inst_67_);
v_isSharedCheck_83_ = !lean_is_exclusive(v_inst_67_);
if (v_isSharedCheck_83_ == 0)
{
lean_object* v_unused_84_; lean_object* v_unused_85_; lean_object* v_unused_86_; lean_object* v_unused_87_; lean_object* v_unused_88_; 
v_unused_84_ = lean_ctor_get(v_inst_67_, 4);
lean_dec(v_unused_84_);
v_unused_85_ = lean_ctor_get(v_inst_67_, 3);
lean_dec(v_unused_85_);
v_unused_86_ = lean_ctor_get(v_inst_67_, 2);
lean_dec(v_unused_86_);
v_unused_87_ = lean_ctor_get(v_inst_67_, 1);
lean_dec(v_unused_87_);
v_unused_88_ = lean_ctor_get(v_inst_67_, 0);
lean_dec(v_unused_88_);
v___x_73_ = v_inst_67_;
v_isShared_74_ = v_isSharedCheck_83_;
goto v_resetjp_72_;
}
else
{
lean_dec(v_inst_67_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_83_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
lean_object* v___x_75_; lean_object* v_toNeg_76_; lean_object* v_toSub_77_; lean_object* v_toZSMul_78_; lean_object* v___f_79_; lean_object* v___x_81_; 
v___x_75_ = lp_mathlib_ULift_addGroup___redArg(v___x_71_);
v_toNeg_76_ = lean_ctor_get(v___x_75_, 1);
lean_inc(v_toNeg_76_);
v_toSub_77_ = lean_ctor_get(v___x_75_, 2);
lean_inc(v_toSub_77_);
v_toZSMul_78_ = lean_ctor_get(v___x_75_, 3);
lean_inc(v_toZSMul_78_);
lean_dec_ref(v___x_75_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_79_, 0, v_toIntCast_68_);
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 4, v_toZSMul_78_);
lean_ctor_set(v___x_73_, 3, v_toSub_77_);
lean_ctor_set(v___x_73_, 2, v_toNeg_76_);
lean_ctor_set(v___x_73_, 1, v___x_70_);
lean_ctor_set(v___x_73_, 0, v___f_79_);
v___x_81_ = v___x_73_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v___f_79_);
lean_ctor_set(v_reuseFailAlloc_82_, 1, v___x_70_);
lean_ctor_set(v_reuseFailAlloc_82_, 2, v_toNeg_76_);
lean_ctor_set(v_reuseFailAlloc_82_, 3, v_toSub_77_);
lean_ctor_set(v_reuseFailAlloc_82_, 4, v_toZSMul_78_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroupWithOne(lean_object* v_R_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_mathlib_ULift_addGroupWithOne___redArg(v_inst_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroupWithOne___redArg(lean_object* v_inst_92_){
_start:
{
lean_object* v_toAddCommGroup_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_107_; 
v_toAddCommGroup_93_ = lean_ctor_get(v_inst_92_, 0);
lean_inc_ref(v_toAddCommGroup_93_);
v___x_94_ = lp_mathlib_ULift_addCommGroup___redArg(v_toAddCommGroup_93_);
v___x_95_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v_inst_92_);
v_isSharedCheck_107_ = !lean_is_exclusive(v_inst_92_);
if (v_isSharedCheck_107_ == 0)
{
lean_object* v_unused_108_; lean_object* v_unused_109_; lean_object* v_unused_110_; lean_object* v_unused_111_; 
v_unused_108_ = lean_ctor_get(v_inst_92_, 3);
lean_dec(v_unused_108_);
v_unused_109_ = lean_ctor_get(v_inst_92_, 2);
lean_dec(v_unused_109_);
v_unused_110_ = lean_ctor_get(v_inst_92_, 1);
lean_dec(v_unused_110_);
v_unused_111_ = lean_ctor_get(v_inst_92_, 0);
lean_dec(v_unused_111_);
v___x_97_ = v_inst_92_;
v_isShared_98_ = v_isSharedCheck_107_;
goto v_resetjp_96_;
}
else
{
lean_dec(v_inst_92_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_107_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_99_; lean_object* v_toAddMonoidWithOne_100_; lean_object* v_toIntCast_101_; lean_object* v_toNatCast_102_; lean_object* v_toOne_103_; lean_object* v___x_105_; 
v___x_99_ = lp_mathlib_ULift_addGroupWithOne___redArg(v___x_95_);
v_toAddMonoidWithOne_100_ = lean_ctor_get(v___x_99_, 1);
lean_inc_ref(v_toAddMonoidWithOne_100_);
v_toIntCast_101_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_toIntCast_101_);
lean_dec_ref(v___x_99_);
v_toNatCast_102_ = lean_ctor_get(v_toAddMonoidWithOne_100_, 0);
lean_inc(v_toNatCast_102_);
v_toOne_103_ = lean_ctor_get(v_toAddMonoidWithOne_100_, 2);
lean_inc(v_toOne_103_);
lean_dec_ref(v_toAddMonoidWithOne_100_);
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 3, v_toOne_103_);
lean_ctor_set(v___x_97_, 2, v_toNatCast_102_);
lean_ctor_set(v___x_97_, 1, v_toIntCast_101_);
lean_ctor_set(v___x_97_, 0, v___x_94_);
v___x_105_ = v___x_97_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v___x_94_);
lean_ctor_set(v_reuseFailAlloc_106_, 1, v_toIntCast_101_);
lean_ctor_set(v_reuseFailAlloc_106_, 2, v_toNatCast_102_);
lean_ctor_set(v_reuseFailAlloc_106_, 3, v_toOne_103_);
v___x_105_ = v_reuseFailAlloc_106_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
return v___x_105_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroupWithOne(lean_object* v_R_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_ULift_addCommGroupWithOne___redArg(v_inst_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(lean_object* v_inst_115_){
_start:
{
lean_object* v_toAddCommMonoid_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v_toMul_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_127_; 
v_toAddCommMonoid_116_ = lean_ctor_get(v_inst_115_, 0);
lean_inc_ref(v_toAddCommMonoid_116_);
v___x_117_ = lp_mathlib_ULift_addCommMonoid___redArg(v_toAddCommMonoid_116_);
v___x_118_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_115_);
v___x_119_ = lp_mathlib_ULift_distrib___redArg(v___x_118_);
v_toMul_120_ = lean_ctor_get(v___x_119_, 0);
v_isSharedCheck_127_ = !lean_is_exclusive(v___x_119_);
if (v_isSharedCheck_127_ == 0)
{
lean_object* v_unused_128_; 
v_unused_128_ = lean_ctor_get(v___x_119_, 1);
lean_dec(v_unused_128_);
v___x_122_ = v___x_119_;
v_isShared_123_ = v_isSharedCheck_127_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_toMul_120_);
lean_dec(v___x_119_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_127_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_125_; 
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 1, v_toMul_120_);
lean_ctor_set(v___x_122_, 0, v___x_117_);
v___x_125_ = v___x_122_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v___x_117_);
lean_ctor_set(v_reuseFailAlloc_126_, 1, v_toMul_120_);
v___x_125_ = v_reuseFailAlloc_126_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
return v___x_125_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocSemiring(lean_object* v_R_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v_inst_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocSemiring___redArg(lean_object* v_inst_132_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_toNatCast_137_; lean_object* v_toOne_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v_toNonUnitalNonAssocSemiring_133_ = lean_ctor_get(v_inst_132_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_133_);
v___x_134_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_133_);
v___x_135_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_132_);
v___x_136_ = lp_mathlib_ULift_addMonoidWithOne___redArg(v___x_135_);
v_toNatCast_137_ = lean_ctor_get(v___x_136_, 0);
v_toOne_138_ = lean_ctor_get(v___x_136_, 2);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_145_ == 0)
{
lean_object* v_unused_146_; 
v_unused_146_ = lean_ctor_get(v___x_136_, 1);
lean_dec(v_unused_146_);
v___x_140_ = v___x_136_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_toOne_138_);
lean_inc(v_toNatCast_137_);
lean_dec(v___x_136_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
lean_ctor_set(v___x_140_, 2, v_toNatCast_137_);
lean_ctor_set(v___x_140_, 1, v_toOne_138_);
lean_ctor_set(v___x_140_, 0, v___x_134_);
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_144_, 1, v_toOne_138_);
lean_ctor_set(v_reuseFailAlloc_144_, 2, v_toNatCast_137_);
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocSemiring(lean_object* v_R_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_ULift_nonAssocSemiring___redArg(v_inst_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalSemiring___redArg(lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalSemiring(lean_object* v_R_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v_inst_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semiring___redArg(lean_object* v_inst_155_){
_start:
{
lean_object* v_toAddCommMonoid_156_; lean_object* v_toMonoid_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v_toOne_161_; lean_object* v_toNatCast_162_; lean_object* v___x_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_183_; 
v_toAddCommMonoid_156_ = lean_ctor_get(v_inst_155_, 0);
v_toMonoid_157_ = lean_ctor_get(v_inst_155_, 1);
lean_inc_ref(v_toMonoid_157_);
lean_inc_ref(v_toAddCommMonoid_156_);
v___x_158_ = lp_mathlib_ULift_addCommMonoid___redArg(v_toAddCommMonoid_156_);
lean_inc_ref(v_inst_155_);
v___x_159_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_155_);
v___x_160_ = lp_mathlib_ULift_nonAssocSemiring___redArg(v___x_159_);
v_toOne_161_ = lean_ctor_get(v___x_160_, 1);
lean_inc(v_toOne_161_);
v_toNatCast_162_ = lean_ctor_get(v___x_160_, 2);
lean_inc(v_toNatCast_162_);
lean_dec_ref(v___x_160_);
v___x_163_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_155_);
v_isSharedCheck_183_ = !lean_is_exclusive(v_inst_155_);
if (v_isSharedCheck_183_ == 0)
{
lean_object* v_unused_184_; lean_object* v_unused_185_; lean_object* v_unused_186_; 
v_unused_184_ = lean_ctor_get(v_inst_155_, 2);
lean_dec(v_unused_184_);
v_unused_185_ = lean_ctor_get(v_inst_155_, 1);
lean_dec(v_unused_185_);
v_unused_186_ = lean_ctor_get(v_inst_155_, 0);
lean_dec(v_unused_186_);
v___x_165_ = v_inst_155_;
v_isShared_166_ = v_isSharedCheck_183_;
goto v_resetjp_164_;
}
else
{
lean_dec(v_inst_155_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_183_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_167_; lean_object* v_toMul_168_; lean_object* v___x_169_; lean_object* v_toNPow_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_180_; 
v___x_167_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v___x_163_);
v_toMul_168_ = lean_ctor_get(v___x_167_, 1);
lean_inc(v_toMul_168_);
lean_dec_ref(v___x_167_);
v___x_169_ = lp_mathlib_ULift_monoid___redArg(v_toMonoid_157_);
v_toNPow_170_ = lean_ctor_get(v___x_169_, 2);
v_isSharedCheck_180_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_180_ == 0)
{
lean_object* v_unused_181_; lean_object* v_unused_182_; 
v_unused_181_ = lean_ctor_get(v___x_169_, 1);
lean_dec(v_unused_181_);
v_unused_182_ = lean_ctor_get(v___x_169_, 0);
lean_dec(v_unused_182_);
v___x_172_ = v___x_169_;
v_isShared_173_ = v_isSharedCheck_180_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_toNPow_170_);
lean_dec(v___x_169_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_180_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___x_175_; 
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 1, v_toMul_168_);
lean_ctor_set(v___x_172_, 0, v_toOne_161_);
v___x_175_ = v___x_172_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_toOne_161_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v_toMul_168_);
lean_ctor_set(v_reuseFailAlloc_179_, 2, v_toNPow_170_);
v___x_175_ = v_reuseFailAlloc_179_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
lean_object* v___x_177_; 
if (v_isShared_166_ == 0)
{
lean_ctor_set(v___x_165_, 2, v_toNatCast_162_);
lean_ctor_set(v___x_165_, 1, v___x_175_);
lean_ctor_set(v___x_165_, 0, v___x_158_);
v___x_177_ = v___x_165_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v___x_158_);
lean_ctor_set(v_reuseFailAlloc_178_, 1, v___x_175_);
lean_ctor_set(v_reuseFailAlloc_178_, 2, v_toNatCast_162_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semiring(lean_object* v_R_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_ULift_semiring___redArg(v_inst_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__0(lean_object* v_self_190_){
_start:
{
lean_inc(v_self_190_);
return v_self_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__0___boxed(lean_object* v_self_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_ULift_ringEquiv___lam__0(v_self_191_);
lean_dec(v_self_191_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__1(lean_object* v_down_193_){
_start:
{
lean_inc(v_down_193_);
return v_down_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___lam__1___boxed(lean_object* v_down_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_ULift_ringEquiv___lam__1(v_down_194_);
lean_dec(v_down_194_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv(lean_object* v_R_201_, lean_object* v_inst_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = ((lean_object*)(lp_mathlib_ULift_ringEquiv___closed__2));
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ringEquiv___boxed(lean_object* v_R_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_ULift_ringEquiv(v_R_204_, v_inst_205_);
lean_dec_ref(v_inst_205_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommSemiring___redArg(lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v_inst_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommSemiring(lean_object* v_R_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemiring___redArg(lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_ULift_semiring___redArg(v_inst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemiring(lean_object* v_R_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib_ULift_semiring___redArg(v_inst_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v_toAddCommGroup_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v_toMul_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_229_; 
v_toAddCommGroup_218_ = lean_ctor_get(v_inst_217_, 0);
lean_inc_ref(v_toAddCommGroup_218_);
v___x_219_ = lp_mathlib_ULift_addCommGroup___redArg(v_toAddCommGroup_218_);
v___x_220_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_217_);
v___x_221_ = lp_mathlib_ULift_nonUnitalNonAssocSemiring___redArg(v___x_220_);
v_toMul_222_ = lean_ctor_get(v___x_221_, 1);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_221_);
if (v_isSharedCheck_229_ == 0)
{
lean_object* v_unused_230_; 
v_unused_230_ = lean_ctor_get(v___x_221_, 0);
lean_dec(v_unused_230_);
v___x_224_ = v___x_221_;
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_toMul_222_);
lean_dec(v___x_221_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
if (v_isShared_225_ == 0)
{
lean_ctor_set(v___x_224_, 0, v___x_219_);
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v___x_219_);
lean_ctor_set(v_reuseFailAlloc_228_, 1, v_toMul_222_);
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalNonAssocRing(lean_object* v_R_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(v_inst_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalRing___redArg(lean_object* v_inst_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(v_inst_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalRing(lean_object* v_R_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(v_inst_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocRing___redArg(lean_object* v_inst_239_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v_toOne_244_; lean_object* v_toNatCast_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v_toIntCast_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_255_; 
v_toNonUnitalNonAssocRing_240_ = lean_ctor_get(v_inst_239_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_240_);
v___x_241_ = lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(v_toNonUnitalNonAssocRing_240_);
lean_inc_ref(v_inst_239_);
v___x_242_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_239_);
v___x_243_ = lp_mathlib_ULift_nonAssocSemiring___redArg(v___x_242_);
v_toOne_244_ = lean_ctor_get(v___x_243_, 1);
lean_inc(v_toOne_244_);
v_toNatCast_245_ = lean_ctor_get(v___x_243_, 2);
lean_inc(v_toNatCast_245_);
lean_dec_ref(v___x_243_);
v___x_246_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_239_);
v___x_247_ = lp_mathlib_ULift_addCommGroupWithOne___redArg(v___x_246_);
v_toIntCast_248_ = lean_ctor_get(v___x_247_, 1);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_247_);
if (v_isSharedCheck_255_ == 0)
{
lean_object* v_unused_256_; lean_object* v_unused_257_; lean_object* v_unused_258_; 
v_unused_256_ = lean_ctor_get(v___x_247_, 3);
lean_dec(v_unused_256_);
v_unused_257_ = lean_ctor_get(v___x_247_, 2);
lean_dec(v_unused_257_);
v_unused_258_ = lean_ctor_get(v___x_247_, 0);
lean_dec(v_unused_258_);
v___x_250_ = v___x_247_;
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_toIntCast_248_);
lean_dec(v___x_247_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 3, v_toIntCast_248_);
lean_ctor_set(v___x_250_, 2, v_toNatCast_245_);
lean_ctor_set(v___x_250_, 1, v_toOne_244_);
lean_ctor_set(v___x_250_, 0, v___x_241_);
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v___x_241_);
lean_ctor_set(v_reuseFailAlloc_254_, 1, v_toOne_244_);
lean_ctor_set(v_reuseFailAlloc_254_, 2, v_toNatCast_245_);
lean_ctor_set(v_reuseFailAlloc_254_, 3, v_toIntCast_248_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonAssocRing(lean_object* v_R_259_, lean_object* v_inst_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_mathlib_ULift_nonAssocRing___redArg(v_inst_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ring___redArg(lean_object* v_inst_262_){
_start:
{
lean_object* v_toSemiring_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v_toNeg_267_; lean_object* v_toSub_268_; lean_object* v_toZSMul_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v_toIntCast_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_279_; 
v_toSemiring_263_ = lean_ctor_get(v_inst_262_, 0);
lean_inc_ref(v_toSemiring_263_);
v___x_264_ = lp_mathlib_ULift_semiring___redArg(v_toSemiring_263_);
v___x_265_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_262_);
v___x_266_ = lp_mathlib_ULift_addCommGroup___redArg(v___x_265_);
v_toNeg_267_ = lean_ctor_get(v___x_266_, 1);
lean_inc(v_toNeg_267_);
v_toSub_268_ = lean_ctor_get(v___x_266_, 2);
lean_inc(v_toSub_268_);
v_toZSMul_269_ = lean_ctor_get(v___x_266_, 3);
lean_inc(v_toZSMul_269_);
lean_dec_ref(v___x_266_);
v___x_270_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_262_);
v___x_271_ = lp_mathlib_ULift_addGroupWithOne___redArg(v___x_270_);
v_toIntCast_272_ = lean_ctor_get(v___x_271_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_279_ == 0)
{
lean_object* v_unused_280_; lean_object* v_unused_281_; lean_object* v_unused_282_; lean_object* v_unused_283_; 
v_unused_280_ = lean_ctor_get(v___x_271_, 4);
lean_dec(v_unused_280_);
v_unused_281_ = lean_ctor_get(v___x_271_, 3);
lean_dec(v_unused_281_);
v_unused_282_ = lean_ctor_get(v___x_271_, 2);
lean_dec(v_unused_282_);
v_unused_283_ = lean_ctor_get(v___x_271_, 1);
lean_dec(v_unused_283_);
v___x_274_ = v___x_271_;
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_toIntCast_272_);
lean_dec(v___x_271_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_277_; 
if (v_isShared_275_ == 0)
{
lean_ctor_set(v___x_274_, 4, v_toIntCast_272_);
lean_ctor_set(v___x_274_, 3, v_toZSMul_269_);
lean_ctor_set(v___x_274_, 2, v_toSub_268_);
lean_ctor_set(v___x_274_, 1, v_toNeg_267_);
lean_ctor_set(v___x_274_, 0, v___x_264_);
v___x_277_ = v___x_274_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v_toNeg_267_);
lean_ctor_set(v_reuseFailAlloc_278_, 2, v_toSub_268_);
lean_ctor_set(v_reuseFailAlloc_278_, 3, v_toZSMul_269_);
lean_ctor_set(v_reuseFailAlloc_278_, 4, v_toIntCast_272_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_ring(lean_object* v_R_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_mathlib_ULift_ring___redArg(v_inst_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommRing___redArg(lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(v_inst_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_nonUnitalCommRing(lean_object* v_R_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_ULift_nonUnitalNonAssocRing___redArg(v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commRing___redArg(lean_object* v_inst_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_ULift_ring___redArg(v_inst_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commRing(lean_object* v_R_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_ULift_ring___redArg(v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift___redArg(lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_f_299_){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v_toFun_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v_toFun_308_; lean_object* v___f_309_; lean_object* v___f_310_; 
v___x_300_ = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(v_inst_298_);
v___x_301_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v___x_300_);
v___x_302_ = lp_mathlib_ULift_ringEquiv(lean_box(0), v___x_301_);
lean_dec_ref(v___x_301_);
v___x_303_ = lp_mathlib_Equiv_symm___redArg(v___x_302_);
v_toFun_304_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_toFun_304_);
lean_dec_ref(v___x_303_);
v___x_305_ = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(v_inst_297_);
v___x_306_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v___x_305_);
v___x_307_ = lp_mathlib_ULift_ringEquiv(lean_box(0), v___x_306_);
lean_dec_ref(v___x_306_);
v_toFun_308_ = lean_ctor_get(v___x_307_, 0);
lean_inc(v_toFun_308_);
lean_dec_ref(v___x_307_);
v___f_309_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_309_, 0, v_toFun_308_);
lean_closure_set(v___f_309_, 1, v_f_299_);
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_310_, 0, v___f_309_);
lean_closure_set(v___f_310_, 1, v_toFun_304_);
return v___f_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift___redArg___boxed(lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_f_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_RingHom_ulift___redArg(v_inst_311_, v_inst_312_, v_f_313_);
lean_dec_ref(v_inst_312_);
lean_dec_ref(v_inst_311_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift(lean_object* v_R_315_, lean_object* v_S_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_f_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_mathlib_RingHom_ulift___redArg(v_inst_317_, v_inst_318_, v_f_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ulift___boxed(lean_object* v_R_321_, lean_object* v_S_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_f_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_RingHom_ulift(v_R_321_, v_S_322_, v_inst_323_, v_inst_324_, v_f_325_);
lean_dec_ref(v_inst_324_);
lean_dec_ref(v_inst_323_);
return v_res_326_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_ULift(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_ULift(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_ULift(builtin);
}
#ifdef __cplusplus
}
#endif
