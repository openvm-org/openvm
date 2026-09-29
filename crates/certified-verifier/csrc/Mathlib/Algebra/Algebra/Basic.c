// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Module.Equiv.Basic public import Mathlib.Algebra.Module.Submodule.Ker public import Mathlib.Algebra.Module.Submodule.RestrictScalars public import Mathlib.Algebra.Module.ULift public import Mathlib.Algebra.Ring.CharZero public import Mathlib.Algebra.Ring.Subring.Basic public import Mathlib.Data.Nat.Cast.Order.Basic public import Mathlib.Data.Int.CharZero import Mathlib.Algebra.Ring.Hom.InjSurj
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
lean_object* lp_mathlib_ULift_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_Int_castRingHom___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_restrictScalars___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_ULift_ringEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_smulLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubsemiringClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PUnit_smul___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_castRingHom___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_PUnit_algebra___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_algebra___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_algebra___closed__0 = (const lean_object*)&lp_mathlib_PUnit_algebra___closed__0_value;
static const lean_closure_object lp_mathlib_PUnit_algebra___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_smul___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_algebra___closed__1 = (const lean_object*)&lp_mathlib_PUnit_algebra___closed__1_value;
static const lean_ctor_object lp_mathlib_PUnit_algebra___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_PUnit_algebra___closed__1_value),((lean_object*)&lp_mathlib_PUnit_algebra___closed__0_value)}};
static const lean_object* lp_mathlib_PUnit_algebra___closed__2 = (const lean_object*)&lp_mathlib_PUnit_algebra___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_ofSubsemiring___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubsemiringClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_ofSubsemiring___redArg___closed__0 = (const lean_object*)&lp_mathlib_Algebra_ofSubsemiring___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofSubsemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofSubsemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofSubsemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algebraMapSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algebraMapSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___closed__0 = (const lean_object*)&lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNatAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNatAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toIntAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toIntAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__1 = (const lean_object*)&lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__1_value),((lean_object*)&lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__2 = (const lean_object*)&lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_extendScalarsOfSurjective___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_extendScalarsOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_extendScalarsOfSurjective___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra___lam__0(lean_object* v_x_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra___lam__0___boxed(lean_object* v_x_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_PUnit_algebra___lam__0(v_x_3_);
lean_dec(v_x_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra(lean_object* v_R_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = ((lean_object*)(lp_mathlib_PUnit_algebra___closed__2));
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_algebra___boxed(lean_object* v_R_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_PUnit_algebra(v_R_13_, v_inst_14_);
lean_dec_ref(v_inst_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra___redArg___lam__0(lean_object* v_algebraMap_16_, lean_object* v_r_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_apply_1(v_algebraMap_16_, v_r_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v_toSMul_20_; lean_object* v_algebraMap_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_30_; 
v_toSMul_20_ = lean_ctor_get(v_inst_19_, 0);
v_algebraMap_21_ = lean_ctor_get(v_inst_19_, 1);
v_isSharedCheck_30_ = !lean_is_exclusive(v_inst_19_);
if (v_isSharedCheck_30_ == 0)
{
v___x_23_ = v_inst_19_;
v_isShared_24_ = v_isSharedCheck_30_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_algebraMap_21_);
lean_inc(v_toSMul_20_);
lean_dec(v_inst_19_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_30_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___f_25_; lean_object* v___f_26_; lean_object* v___x_28_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_ULift_algebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_25_, 0, v_algebraMap_21_);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_26_, 0, v_toSMul_20_);
if (v_isShared_24_ == 0)
{
lean_ctor_set(v___x_23_, 1, v___f_25_);
lean_ctor_set(v___x_23_, 0, v___f_26_);
v___x_28_ = v___x_23_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v___f_26_);
lean_ctor_set(v_reuseFailAlloc_29_, 1, v___f_25_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra(lean_object* v_R_31_, lean_object* v_A_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_ULift_algebra___redArg(v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra___boxed(lean_object* v_R_37_, lean_object* v_A_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_ULift_algebra(v_R_37_, v_A_38_, v_inst_39_, v_inst_40_, v_inst_41_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra_x27___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v_toSMul_45_; lean_object* v_algebraMap_46_; lean_object* v___x_48_; uint8_t v_isShared_49_; uint8_t v_isSharedCheck_59_; 
v_toSMul_45_ = lean_ctor_get(v_inst_44_, 0);
v_algebraMap_46_ = lean_ctor_get(v_inst_44_, 1);
v_isSharedCheck_59_ = !lean_is_exclusive(v_inst_44_);
if (v_isSharedCheck_59_ == 0)
{
v___x_48_ = v_inst_44_;
v_isShared_49_ = v_isSharedCheck_59_;
goto v_resetjp_47_;
}
else
{
lean_inc(v_algebraMap_46_);
lean_inc(v_toSMul_45_);
lean_dec(v_inst_44_);
v___x_48_ = lean_box(0);
v_isShared_49_ = v_isSharedCheck_59_;
goto v_resetjp_47_;
}
v_resetjp_47_:
{
lean_object* v___x_50_; lean_object* v_toNonUnitalNonAssocSemiring_51_; lean_object* v___x_52_; lean_object* v_toFun_53_; lean_object* v___f_54_; lean_object* v___f_55_; lean_object* v___x_57_; 
v___x_50_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_43_);
v_toNonUnitalNonAssocSemiring_51_ = lean_ctor_get(v___x_50_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_51_);
lean_dec_ref(v___x_50_);
v___x_52_ = lp_mathlib_ULift_ringEquiv(lean_box(0), v_toNonUnitalNonAssocSemiring_51_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_51_);
v_toFun_53_ = lean_ctor_get(v___x_52_, 0);
lean_inc(v_toFun_53_);
lean_dec_ref(v___x_52_);
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_54_, 0, v_toSMul_45_);
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_55_, 0, v_toFun_53_);
lean_closure_set(v___f_55_, 1, v_algebraMap_46_);
if (v_isShared_49_ == 0)
{
lean_ctor_set(v___x_48_, 1, v___f_55_);
lean_ctor_set(v___x_48_, 0, v___f_54_);
v___x_57_ = v___x_48_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v___f_54_);
lean_ctor_set(v_reuseFailAlloc_58_, 1, v___f_55_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra_x27(lean_object* v_R_60_, lean_object* v_A_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_ULift_algebra_x27___redArg(v_inst_62_, v_inst_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algebra_x27___boxed(lean_object* v_R_66_, lean_object* v_A_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_ULift_algebra_x27(v_R_66_, v_A_67_, v_inst_68_, v_inst_69_, v_inst_70_);
lean_dec_ref(v_inst_69_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofSubsemiring___redArg(lean_object* v_inst_73_){
_start:
{
lean_object* v_toSMul_74_; lean_object* v_algebraMap_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_85_; 
v_toSMul_74_ = lean_ctor_get(v_inst_73_, 0);
v_algebraMap_75_ = lean_ctor_get(v_inst_73_, 1);
v_isSharedCheck_85_ = !lean_is_exclusive(v_inst_73_);
if (v_isSharedCheck_85_ == 0)
{
v___x_77_ = v_inst_73_;
v_isShared_78_ = v_isSharedCheck_85_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_algebraMap_75_);
lean_inc(v_toSMul_74_);
lean_dec(v_inst_73_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_85_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v___f_79_; lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___x_83_; 
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_79_, 0, v_toSMul_74_);
v___f_80_ = ((lean_object*)(lp_mathlib_Algebra_ofSubsemiring___redArg___closed__0));
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_81_, 0, v___f_80_);
lean_closure_set(v___f_81_, 1, v_algebraMap_75_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 1, v___f_81_);
lean_ctor_set(v___x_77_, 0, v___f_79_);
v___x_83_ = v___x_77_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v___f_79_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v___f_81_);
v___x_83_ = v_reuseFailAlloc_84_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofSubsemiring(lean_object* v_R_86_, lean_object* v_A_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_C_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_S_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lp_mathlib_Algebra_ofSubsemiring___redArg(v_inst_90_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofSubsemiring___boxed(lean_object* v_R_96_, lean_object* v_A_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_C_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_S_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_Algebra_ofSubsemiring(v_R_96_, v_A_97_, v_inst_98_, v_inst_99_, v_inst_100_, v_C_101_, v_inst_102_, v_inst_103_, v_S_104_);
lean_dec(v_S_104_);
lean_dec_ref(v_inst_99_);
lean_dec_ref(v_inst_98_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algebraMapSubmonoid(lean_object* v_R_106_, lean_object* v_inst_107_, lean_object* v_S_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_M_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lean_box(0);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_algebraMapSubmonoid___boxed(lean_object* v_R_113_, lean_object* v_inst_114_, lean_object* v_S_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_M_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Algebra_algebraMapSubmonoid(v_R_113_, v_inst_114_, v_S_115_, v_inst_116_, v_inst_117_, v_M_118_);
lean_dec_ref(v_inst_117_);
lean_dec_ref(v_inst_116_);
lean_dec_ref(v_inst_114_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg___lam__0(lean_object* v_toIntCast_120_, lean_object* v_toSMul_121_, lean_object* v_z_122_, lean_object* v_a_123_){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = lean_apply_1(v_toIntCast_120_, v_z_122_);
v___x_125_ = lean_apply_2(v_toSMul_121_, v___x_124_, v_a_123_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg___lam__1(lean_object* v_toIntCast_126_, lean_object* v_algebraMap_127_, lean_object* v_z_128_){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = lean_apply_1(v_toIntCast_126_, v_z_128_);
v___x_130_ = lean_apply_1(v_algebraMap_127_, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg___lam__2(lean_object* v_toNeg_131_, lean_object* v_toOne_132_, lean_object* v_toSMul_133_, lean_object* v_a_134_){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = lean_apply_1(v_toNeg_131_, v_toOne_132_);
v___x_136_ = lean_apply_2(v_toSMul_133_, v___x_135_, v_a_134_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing___redArg(lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v_toAddCommMonoid_140_; lean_object* v_toSMul_141_; lean_object* v_algebraMap_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v_toNeg_145_; lean_object* v___x_146_; lean_object* v_toAddMonoidWithOne_147_; lean_object* v_toIntCast_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_160_; 
v_toAddCommMonoid_140_ = lean_ctor_get(v_inst_138_, 0);
v_toSMul_141_ = lean_ctor_get(v_inst_139_, 0);
lean_inc(v_toSMul_141_);
v_algebraMap_142_ = lean_ctor_get(v_inst_139_, 1);
lean_inc(v_algebraMap_142_);
lean_dec_ref(v_inst_139_);
v___x_143_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_137_);
v___x_144_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_143_);
lean_dec_ref(v___x_143_);
v_toNeg_145_ = lean_ctor_get(v___x_144_, 1);
lean_inc(v_toNeg_145_);
lean_dec_ref(v___x_144_);
v___x_146_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_137_);
v_toAddMonoidWithOne_147_ = lean_ctor_get(v___x_146_, 1);
v_toIntCast_148_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_160_ == 0)
{
lean_object* v_unused_161_; lean_object* v_unused_162_; lean_object* v_unused_163_; 
v_unused_161_ = lean_ctor_get(v___x_146_, 4);
lean_dec(v_unused_161_);
v_unused_162_ = lean_ctor_get(v___x_146_, 3);
lean_dec(v_unused_162_);
v_unused_163_ = lean_ctor_get(v___x_146_, 2);
lean_dec(v_unused_163_);
v___x_150_ = v___x_146_;
v_isShared_151_ = v_isSharedCheck_160_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_toAddMonoidWithOne_147_);
lean_inc(v_toIntCast_148_);
lean_dec(v___x_146_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_160_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v_toOne_152_; lean_object* v___f_153_; lean_object* v___f_154_; lean_object* v___f_155_; lean_object* v___x_156_; lean_object* v___x_158_; 
v_toOne_152_ = lean_ctor_get(v_toAddMonoidWithOne_147_, 2);
lean_inc(v_toOne_152_);
lean_dec_ref(v_toAddMonoidWithOne_147_);
lean_inc(v_toSMul_141_);
lean_inc(v_toIntCast_148_);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_semiringToRing___redArg___lam__0), 4, 2);
lean_closure_set(v___f_153_, 0, v_toIntCast_148_);
lean_closure_set(v___f_153_, 1, v_toSMul_141_);
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_semiringToRing___redArg___lam__1), 3, 2);
lean_closure_set(v___f_154_, 0, v_toIntCast_148_);
lean_closure_set(v___f_154_, 1, v_algebraMap_142_);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_semiringToRing___redArg___lam__2), 4, 3);
lean_closure_set(v___f_155_, 0, v_toNeg_145_);
lean_closure_set(v___f_155_, 1, v_toOne_152_);
lean_closure_set(v___f_155_, 2, v_toSMul_141_);
lean_inc_ref(v___f_155_);
lean_inc_ref(v_toAddCommMonoid_140_);
v___x_156_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_156_, 0, lean_box(0));
lean_closure_set(v___x_156_, 1, v_toAddCommMonoid_140_);
lean_closure_set(v___x_156_, 2, v___f_155_);
if (v_isShared_151_ == 0)
{
lean_ctor_set(v___x_150_, 4, v___f_154_);
lean_ctor_set(v___x_150_, 3, v___f_153_);
lean_ctor_set(v___x_150_, 2, v___x_156_);
lean_ctor_set(v___x_150_, 1, v___f_155_);
lean_ctor_set(v___x_150_, 0, v_inst_138_);
v___x_158_ = v___x_150_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v_inst_138_);
lean_ctor_set(v_reuseFailAlloc_159_, 1, v___f_155_);
lean_ctor_set(v_reuseFailAlloc_159_, 2, v___x_156_);
lean_ctor_set(v_reuseFailAlloc_159_, 3, v___f_153_);
lean_ctor_set(v_reuseFailAlloc_159_, 4, v___f_154_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_semiringToRing(lean_object* v_A_164_, lean_object* v_R_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v_toAddCommMonoid_169_; lean_object* v_toSMul_170_; lean_object* v_algebraMap_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v_toNeg_174_; lean_object* v___x_175_; lean_object* v_toAddMonoidWithOne_176_; lean_object* v_toIntCast_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_189_; 
v_toAddCommMonoid_169_ = lean_ctor_get(v_inst_167_, 0);
v_toSMul_170_ = lean_ctor_get(v_inst_168_, 0);
lean_inc(v_toSMul_170_);
v_algebraMap_171_ = lean_ctor_get(v_inst_168_, 1);
lean_inc(v_algebraMap_171_);
lean_dec_ref(v_inst_168_);
v___x_172_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_166_);
v___x_173_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_172_);
lean_dec_ref(v___x_172_);
v_toNeg_174_ = lean_ctor_get(v___x_173_, 1);
lean_inc(v_toNeg_174_);
lean_dec_ref(v___x_173_);
v___x_175_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_166_);
v_toAddMonoidWithOne_176_ = lean_ctor_get(v___x_175_, 1);
v_toIntCast_177_ = lean_ctor_get(v___x_175_, 0);
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_189_ == 0)
{
lean_object* v_unused_190_; lean_object* v_unused_191_; lean_object* v_unused_192_; 
v_unused_190_ = lean_ctor_get(v___x_175_, 4);
lean_dec(v_unused_190_);
v_unused_191_ = lean_ctor_get(v___x_175_, 3);
lean_dec(v_unused_191_);
v_unused_192_ = lean_ctor_get(v___x_175_, 2);
lean_dec(v_unused_192_);
v___x_179_ = v___x_175_;
v_isShared_180_ = v_isSharedCheck_189_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_toAddMonoidWithOne_176_);
lean_inc(v_toIntCast_177_);
lean_dec(v___x_175_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_189_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v_toOne_181_; lean_object* v___f_182_; lean_object* v___f_183_; lean_object* v___f_184_; lean_object* v___x_185_; lean_object* v___x_187_; 
v_toOne_181_ = lean_ctor_get(v_toAddMonoidWithOne_176_, 2);
lean_inc(v_toOne_181_);
lean_dec_ref(v_toAddMonoidWithOne_176_);
lean_inc(v_toSMul_170_);
lean_inc(v_toIntCast_177_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_semiringToRing___redArg___lam__0), 4, 2);
lean_closure_set(v___f_182_, 0, v_toIntCast_177_);
lean_closure_set(v___f_182_, 1, v_toSMul_170_);
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_semiringToRing___redArg___lam__1), 3, 2);
lean_closure_set(v___f_183_, 0, v_toIntCast_177_);
lean_closure_set(v___f_183_, 1, v_algebraMap_171_);
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_semiringToRing___redArg___lam__2), 4, 3);
lean_closure_set(v___f_184_, 0, v_toNeg_174_);
lean_closure_set(v___f_184_, 1, v_toOne_181_);
lean_closure_set(v___f_184_, 2, v_toSMul_170_);
lean_inc_ref(v___f_184_);
lean_inc_ref(v_toAddCommMonoid_169_);
v___x_185_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_185_, 0, lean_box(0));
lean_closure_set(v___x_185_, 1, v_toAddCommMonoid_169_);
lean_closure_set(v___x_185_, 2, v___f_184_);
if (v_isShared_180_ == 0)
{
lean_ctor_set(v___x_179_, 4, v___f_183_);
lean_ctor_set(v___x_179_, 3, v___f_182_);
lean_ctor_set(v___x_179_, 2, v___x_185_);
lean_ctor_set(v___x_179_, 1, v___f_184_);
lean_ctor_set(v___x_179_, 0, v_inst_167_);
v___x_187_ = v___x_179_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_inst_167_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v___f_184_);
lean_ctor_set(v_reuseFailAlloc_188_, 2, v___x_185_);
lean_ctor_set(v_reuseFailAlloc_188_, 3, v___f_182_);
lean_ctor_set(v_reuseFailAlloc_188_, 4, v___f_183_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__0(lean_object* v_00_u03c6_193_, lean_object* v_toMul_194_, lean_object* v_c_195_, lean_object* v_x_196_){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_197_ = lean_apply_1(v_00_u03c6_193_, v_c_195_);
v___x_198_ = lean_apply_2(v_toMul_194_, v___x_197_, v_x_196_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__1(lean_object* v_toIntCast_199_, lean_object* v_00_u03c6_200_, lean_object* v_z_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_202_ = lean_apply_1(v_toIntCast_199_, v_z_201_);
v___x_203_ = lean_apply_1(v_00_u03c6_200_, v___x_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__2(lean_object* v_toIntCast_204_, lean_object* v___f_205_, lean_object* v_z_206_, lean_object* v_a_207_){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_208_ = lean_apply_1(v_toIntCast_204_, v_z_206_);
v___x_209_ = lean_apply_2(v___f_205_, v___x_208_, v_a_207_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__3(lean_object* v_toNeg_210_, lean_object* v_toOne_211_, lean_object* v___f_212_, lean_object* v_a_213_){
_start:
{
lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_214_ = lean_apply_1(v_toNeg_210_, v_toOne_211_);
v___x_215_ = lean_apply_2(v___f_212_, v___x_214_, v_a_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing___redArg(lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_00_u03c6_218_){
_start:
{
lean_object* v___x_219_; lean_object* v_toMul_220_; lean_object* v_toAddCommMonoid_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v_toNeg_224_; lean_object* v___x_225_; lean_object* v_toAddMonoidWithOne_226_; lean_object* v_toIntCast_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_240_; 
lean_inc_ref(v_inst_217_);
v___x_219_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_217_);
v_toMul_220_ = lean_ctor_get(v___x_219_, 0);
lean_inc(v_toMul_220_);
lean_dec_ref(v___x_219_);
v_toAddCommMonoid_221_ = lean_ctor_get(v_inst_217_, 0);
v___x_222_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_216_);
v___x_223_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_222_);
lean_dec_ref(v___x_222_);
v_toNeg_224_ = lean_ctor_get(v___x_223_, 1);
lean_inc(v_toNeg_224_);
lean_dec_ref(v___x_223_);
v___x_225_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_216_);
v_toAddMonoidWithOne_226_ = lean_ctor_get(v___x_225_, 1);
v_toIntCast_227_ = lean_ctor_get(v___x_225_, 0);
v_isSharedCheck_240_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_240_ == 0)
{
lean_object* v_unused_241_; lean_object* v_unused_242_; lean_object* v_unused_243_; 
v_unused_241_ = lean_ctor_get(v___x_225_, 4);
lean_dec(v_unused_241_);
v_unused_242_ = lean_ctor_get(v___x_225_, 3);
lean_dec(v_unused_242_);
v_unused_243_ = lean_ctor_get(v___x_225_, 2);
lean_dec(v_unused_243_);
v___x_229_ = v___x_225_;
v_isShared_230_ = v_isSharedCheck_240_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_toAddMonoidWithOne_226_);
lean_inc(v_toIntCast_227_);
lean_dec(v___x_225_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_240_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v_toOne_231_; lean_object* v___f_232_; lean_object* v___f_233_; lean_object* v___f_234_; lean_object* v___f_235_; lean_object* v___x_236_; lean_object* v___x_238_; 
v_toOne_231_ = lean_ctor_get(v_toAddMonoidWithOne_226_, 2);
lean_inc(v_toOne_231_);
lean_dec_ref(v_toAddMonoidWithOne_226_);
lean_inc(v_00_u03c6_218_);
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__0), 4, 2);
lean_closure_set(v___f_232_, 0, v_00_u03c6_218_);
lean_closure_set(v___f_232_, 1, v_toMul_220_);
lean_inc(v_toIntCast_227_);
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__1), 3, 2);
lean_closure_set(v___f_233_, 0, v_toIntCast_227_);
lean_closure_set(v___f_233_, 1, v_00_u03c6_218_);
lean_inc_ref(v___f_232_);
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__2), 4, 2);
lean_closure_set(v___f_234_, 0, v_toIntCast_227_);
lean_closure_set(v___f_234_, 1, v___f_232_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__3), 4, 3);
lean_closure_set(v___f_235_, 0, v_toNeg_224_);
lean_closure_set(v___f_235_, 1, v_toOne_231_);
lean_closure_set(v___f_235_, 2, v___f_232_);
lean_inc_ref(v___f_235_);
lean_inc_ref(v_toAddCommMonoid_221_);
v___x_236_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_236_, 0, lean_box(0));
lean_closure_set(v___x_236_, 1, v_toAddCommMonoid_221_);
lean_closure_set(v___x_236_, 2, v___f_235_);
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 4, v___f_233_);
lean_ctor_set(v___x_229_, 3, v___f_234_);
lean_ctor_set(v___x_229_, 2, v___x_236_);
lean_ctor_set(v___x_229_, 1, v___f_235_);
lean_ctor_set(v___x_229_, 0, v_inst_217_);
v___x_238_ = v___x_229_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v_inst_217_);
lean_ctor_set(v_reuseFailAlloc_239_, 1, v___f_235_);
lean_ctor_set(v_reuseFailAlloc_239_, 2, v___x_236_);
lean_ctor_set(v_reuseFailAlloc_239_, 3, v___f_234_);
lean_ctor_set(v_reuseFailAlloc_239_, 4, v___f_233_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_commSemiringToCommRing(lean_object* v_R_244_, lean_object* v_A_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_00_u03c6_248_){
_start:
{
lean_object* v___x_249_; lean_object* v_toMul_250_; lean_object* v_toAddCommMonoid_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v_toNeg_254_; lean_object* v___x_255_; lean_object* v_toAddMonoidWithOne_256_; lean_object* v_toIntCast_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_270_; 
lean_inc_ref(v_inst_247_);
v___x_249_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_247_);
v_toMul_250_ = lean_ctor_get(v___x_249_, 0);
lean_inc(v_toMul_250_);
lean_dec_ref(v___x_249_);
v_toAddCommMonoid_251_ = lean_ctor_get(v_inst_247_, 0);
v___x_252_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_246_);
v___x_253_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_252_);
lean_dec_ref(v___x_252_);
v_toNeg_254_ = lean_ctor_get(v___x_253_, 1);
lean_inc(v_toNeg_254_);
lean_dec_ref(v___x_253_);
v___x_255_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_246_);
v_toAddMonoidWithOne_256_ = lean_ctor_get(v___x_255_, 1);
v_toIntCast_257_ = lean_ctor_get(v___x_255_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_270_ == 0)
{
lean_object* v_unused_271_; lean_object* v_unused_272_; lean_object* v_unused_273_; 
v_unused_271_ = lean_ctor_get(v___x_255_, 4);
lean_dec(v_unused_271_);
v_unused_272_ = lean_ctor_get(v___x_255_, 3);
lean_dec(v_unused_272_);
v_unused_273_ = lean_ctor_get(v___x_255_, 2);
lean_dec(v_unused_273_);
v___x_259_ = v___x_255_;
v_isShared_260_ = v_isSharedCheck_270_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_toAddMonoidWithOne_256_);
lean_inc(v_toIntCast_257_);
lean_dec(v___x_255_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_270_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v_toOne_261_; lean_object* v___f_262_; lean_object* v___f_263_; lean_object* v___f_264_; lean_object* v___f_265_; lean_object* v___x_266_; lean_object* v___x_268_; 
v_toOne_261_ = lean_ctor_get(v_toAddMonoidWithOne_256_, 2);
lean_inc(v_toOne_261_);
lean_dec_ref(v_toAddMonoidWithOne_256_);
lean_inc(v_00_u03c6_248_);
v___f_262_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__0), 4, 2);
lean_closure_set(v___f_262_, 0, v_00_u03c6_248_);
lean_closure_set(v___f_262_, 1, v_toMul_250_);
lean_inc(v_toIntCast_257_);
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__1), 3, 2);
lean_closure_set(v___f_263_, 0, v_toIntCast_257_);
lean_closure_set(v___f_263_, 1, v_00_u03c6_248_);
lean_inc_ref(v___f_262_);
v___f_264_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__2), 4, 2);
lean_closure_set(v___f_264_, 0, v_toIntCast_257_);
lean_closure_set(v___f_264_, 1, v___f_262_);
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_commSemiringToCommRing___redArg___lam__3), 4, 3);
lean_closure_set(v___f_265_, 0, v_toNeg_254_);
lean_closure_set(v___f_265_, 1, v_toOne_261_);
lean_closure_set(v___f_265_, 2, v___f_262_);
lean_inc_ref(v___f_265_);
lean_inc_ref(v_toAddCommMonoid_251_);
v___x_266_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_266_, 0, lean_box(0));
lean_closure_set(v___x_266_, 1, v_toAddCommMonoid_251_);
lean_closure_set(v___x_266_, 2, v___f_265_);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 4, v___f_263_);
lean_ctor_set(v___x_259_, 3, v___f_264_);
lean_ctor_set(v___x_259_, 2, v___x_266_);
lean_ctor_set(v___x_259_, 1, v___f_265_);
lean_ctor_set(v___x_259_, 0, v_inst_247_);
v___x_268_ = v___x_259_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_inst_247_);
lean_ctor_set(v_reuseFailAlloc_269_, 1, v___f_265_);
lean_ctor_set(v_reuseFailAlloc_269_, 2, v___x_266_);
lean_ctor_set(v_reuseFailAlloc_269_, 3, v___f_264_);
lean_ctor_set(v_reuseFailAlloc_269_, 4, v___f_263_);
v___x_268_ = v_reuseFailAlloc_269_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
return v___x_268_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___lam__0(lean_object* v_self_274_){
_start:
{
lean_inc(v_self_274_);
return v_self_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___lam__0___boxed(lean_object* v_self_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___lam__0(v_self_275_);
lean_dec(v_self_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg(lean_object* v_inst_278_){
_start:
{
lean_object* v_toSemiring_279_; lean_object* v___f_280_; lean_object* v___x_281_; lean_object* v___f_282_; lean_object* v___x_283_; 
v_toSemiring_279_ = lean_ctor_get(v_inst_278_, 0);
v___f_280_ = ((lean_object*)(lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___closed__0));
v___x_281_ = lp_mathlib_Semiring_toModule___redArg(v_toSemiring_279_);
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_282_, 0, v___x_281_);
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v___f_282_);
lean_ctor_set(v___x_283_, 1, v___f_280_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg___boxed(lean_object* v_inst_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg(v_inst_284_);
lean_dec_ref(v_inst_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter(lean_object* v_R_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_mathlib_Algebra_instSubtypeMemSubringCenter___redArg(v_inst_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubringCenter___boxed(lean_object* v_R_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Algebra_instSubtypeMemSubringCenter(v_R_289_, v_inst_290_);
lean_dec_ref(v_inst_290_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___redArg___lam__0(lean_object* v_inst_292_, lean_object* v_r_293_, lean_object* v___y_294_){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_295_ = lp_mathlib_LinearMap_id___lam__0(v___y_294_);
v___x_296_ = lean_apply_2(v_inst_292_, v_r_293_, v___x_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___redArg___lam__0___boxed(lean_object* v_inst_297_, lean_object* v_r_298_, lean_object* v___y_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Module_End_instAlgebra___redArg___lam__0(v_inst_297_, v_r_298_, v___y_299_);
lean_dec(v___y_299_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___redArg(lean_object* v_inst_301_){
_start:
{
lean_object* v___f_302_; lean_object* v___f_303_; lean_object* v___x_304_; 
lean_inc(v_inst_301_);
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_Module_End_instAlgebra___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_302_, 0, v_inst_301_);
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_303_, 0, v_inst_301_);
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v___f_303_);
lean_ctor_set(v___x_304_, 1, v___f_302_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra(lean_object* v_R_305_, lean_object* v_S_306_, lean_object* v_M_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_Module_End_instAlgebra___redArg(v_inst_311_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_End_instAlgebra___boxed(lean_object* v_R_317_, lean_object* v_S_318_, lean_object* v_M_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_Module_End_instAlgebra(v_R_317_, v_S_318_, v_M_319_, v_inst_320_, v_inst_321_, v_inst_322_, v_inst_323_, v_inst_324_, v_inst_325_, v_inst_326_, v_inst_327_);
lean_dec(v_inst_326_);
lean_dec(v_inst_324_);
lean_dec_ref(v_inst_322_);
lean_dec_ref(v_inst_321_);
lean_dec_ref(v_inst_320_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNatAlgebra___redArg(lean_object* v_inst_329_){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v_toAddMonoid_332_; lean_object* v_toNSMul_333_; lean_object* v___f_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_330_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_329_);
lean_inc_ref(v___x_330_);
v___x_331_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_330_);
v_toAddMonoid_332_ = lean_ctor_get(v___x_331_, 1);
lean_inc_ref(v_toAddMonoid_332_);
lean_dec_ref(v___x_331_);
v_toNSMul_333_ = lean_ctor_get(v_toAddMonoid_332_, 2);
lean_inc(v_toNSMul_333_);
lean_dec_ref(v_toAddMonoid_332_);
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_334_, 0, v_toNSMul_333_);
v___x_335_ = lp_mathlib_Nat_castRingHom___redArg(v___x_330_);
v___x_336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_336_, 0, v___f_334_);
lean_ctor_set(v___x_336_, 1, v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNatAlgebra(lean_object* v_R_337_, lean_object* v_inst_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_Semiring_toNatAlgebra___redArg(v_inst_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toIntAlgebra___redArg(lean_object* v_inst_340_){
_start:
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v_toZSMul_343_; lean_object* v___f_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
lean_inc_ref(v_inst_340_);
v___x_341_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_340_);
v___x_342_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_341_);
lean_dec_ref(v___x_341_);
v_toZSMul_343_ = lean_ctor_get(v___x_342_, 3);
lean_inc(v_toZSMul_343_);
lean_dec_ref(v___x_342_);
v___f_344_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_344_, 0, v_toZSMul_343_);
v___x_345_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_340_);
lean_dec_ref(v_inst_340_);
v___x_346_ = lp_mathlib_Int_castRingHom___redArg(v___x_345_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v___f_344_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toIntAlgebra(lean_object* v_R_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_Ring_toIntAlgebra___redArg(v_inst_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap___redArg(lean_object* v_f_351_, lean_object* v_h_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lean_apply_1(v_f_351_, v_h_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap(lean_object* v_R_354_, lean_object* v_A_355_, lean_object* v_B_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_f_362_, lean_object* v_hf_363_, lean_object* v_r_364_, lean_object* v_h_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_apply_1(v_f_362_, v_h_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap___boxed(lean_object* v_R_367_, lean_object* v_A_368_, lean_object* v_B_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_f_375_, lean_object* v_hf_376_, lean_object* v_r_377_, lean_object* v_h_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_Invertible_algebraMapOfInvertibleAlgebraMap(v_R_367_, v_A_368_, v_B_369_, v_inst_370_, v_inst_371_, v_inst_372_, v_inst_373_, v_inst_374_, v_f_375_, v_hf_376_, v_r_377_, v_h_378_);
lean_dec(v_r_377_);
lean_dec_ref(v_inst_374_);
lean_dec_ref(v_inst_373_);
lean_dec_ref(v_inst_372_);
lean_dec_ref(v_inst_371_);
lean_dec_ref(v_inst_370_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___lam__0(lean_object* v_f_380_, lean_object* v___y_381_){
_start:
{
lean_object* v___f_382_; lean_object* v___x_383_; 
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_382_, 0, v_f_380_);
v___x_383_ = lp_mathlib_LinearMap_restrictScalars___redArg___lam__0(v___f_382_, v___y_381_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___lam__1(lean_object* v_f_384_, lean_object* v___y_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lean_apply_1(v_f_384_, v___y_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv(lean_object* v_R_392_, lean_object* v_S_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_M_397_, lean_object* v_N_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_h_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = ((lean_object*)(lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___closed__2));
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv___boxed(lean_object* v_R_409_, lean_object* v_S_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_M_414_, lean_object* v_N_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_h_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv(v_R_409_, v_S_410_, v_inst_411_, v_inst_412_, v_inst_413_, v_M_414_, v_N_415_, v_inst_416_, v_inst_417_, v_inst_418_, v_inst_419_, v_inst_420_, v_inst_421_, v_inst_422_, v_inst_423_, v_h_424_);
lean_dec(v_inst_422_);
lean_dec(v_inst_421_);
lean_dec(v_inst_419_);
lean_dec(v_inst_418_);
lean_dec_ref(v_inst_417_);
lean_dec_ref(v_inst_416_);
lean_dec_ref(v_inst_413_);
lean_dec_ref(v_inst_412_);
lean_dec_ref(v_inst_411_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective___redArg(lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_l_435_){
_start:
{
lean_object* v___x_436_; lean_object* v_toLinearMap_437_; lean_object* v___x_438_; 
v___x_436_ = lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv(lean_box(0), lean_box(0), v_inst_426_, v_inst_427_, v_inst_428_, lean_box(0), lean_box(0), v_inst_429_, v_inst_430_, v_inst_431_, v_inst_432_, lean_box(0), v_inst_433_, v_inst_434_, lean_box(0), lean_box(0));
v_toLinearMap_437_ = lean_ctor_get(v___x_436_, 0);
lean_inc(v_toLinearMap_437_);
lean_dec_ref(v___x_436_);
v___x_438_ = lean_apply_1(v_toLinearMap_437_, v_l_435_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective___redArg___boxed(lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_l_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_mathlib_LinearMap_extendScalarsOfSurjective___redArg(v_inst_439_, v_inst_440_, v_inst_441_, v_inst_442_, v_inst_443_, v_inst_444_, v_inst_445_, v_inst_446_, v_inst_447_, v_l_448_);
lean_dec(v_inst_447_);
lean_dec(v_inst_446_);
lean_dec(v_inst_445_);
lean_dec(v_inst_444_);
lean_dec_ref(v_inst_443_);
lean_dec_ref(v_inst_442_);
lean_dec_ref(v_inst_441_);
lean_dec_ref(v_inst_440_);
lean_dec_ref(v_inst_439_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective(lean_object* v_R_450_, lean_object* v_S_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_M_455_, lean_object* v_N_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_h_465_, lean_object* v_l_466_){
_start:
{
lean_object* v___x_467_; lean_object* v_toLinearMap_468_; lean_object* v___x_469_; 
v___x_467_ = lp_mathlib_LinearMap_extendScalarsOfSurjectiveEquiv(lean_box(0), lean_box(0), v_inst_452_, v_inst_453_, v_inst_454_, lean_box(0), lean_box(0), v_inst_457_, v_inst_458_, v_inst_459_, v_inst_460_, lean_box(0), v_inst_462_, v_inst_463_, lean_box(0), lean_box(0));
v_toLinearMap_468_ = lean_ctor_get(v___x_467_, 0);
lean_inc(v_toLinearMap_468_);
lean_dec_ref(v___x_467_);
v___x_469_ = lean_apply_1(v_toLinearMap_468_, v_l_466_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_extendScalarsOfSurjective___boxed(lean_object** _args){
lean_object* v_R_470_ = _args[0];
lean_object* v_S_471_ = _args[1];
lean_object* v_inst_472_ = _args[2];
lean_object* v_inst_473_ = _args[3];
lean_object* v_inst_474_ = _args[4];
lean_object* v_M_475_ = _args[5];
lean_object* v_N_476_ = _args[6];
lean_object* v_inst_477_ = _args[7];
lean_object* v_inst_478_ = _args[8];
lean_object* v_inst_479_ = _args[9];
lean_object* v_inst_480_ = _args[10];
lean_object* v_inst_481_ = _args[11];
lean_object* v_inst_482_ = _args[12];
lean_object* v_inst_483_ = _args[13];
lean_object* v_inst_484_ = _args[14];
lean_object* v_h_485_ = _args[15];
lean_object* v_l_486_ = _args[16];
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_LinearMap_extendScalarsOfSurjective(v_R_470_, v_S_471_, v_inst_472_, v_inst_473_, v_inst_474_, v_M_475_, v_N_476_, v_inst_477_, v_inst_478_, v_inst_479_, v_inst_480_, v_inst_481_, v_inst_482_, v_inst_483_, v_inst_484_, v_h_485_, v_l_486_);
lean_dec(v_inst_483_);
lean_dec(v_inst_482_);
lean_dec(v_inst_480_);
lean_dec(v_inst_479_);
lean_dec_ref(v_inst_478_);
lean_dec_ref(v_inst_477_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_473_);
lean_dec_ref(v_inst_472_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_extendScalarsOfSurjective___redArg(lean_object* v_f_488_){
_start:
{
lean_object* v_toLinearMap_489_; lean_object* v_invFun_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_497_; 
v_toLinearMap_489_ = lean_ctor_get(v_f_488_, 0);
v_invFun_490_ = lean_ctor_get(v_f_488_, 1);
v_isSharedCheck_497_ = !lean_is_exclusive(v_f_488_);
if (v_isSharedCheck_497_ == 0)
{
v___x_492_ = v_f_488_;
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_invFun_490_);
lean_inc(v_toLinearMap_489_);
lean_dec(v_f_488_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_495_; 
if (v_isShared_493_ == 0)
{
v___x_495_ = v___x_492_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_toLinearMap_489_);
lean_ctor_set(v_reuseFailAlloc_496_, 1, v_invFun_490_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_extendScalarsOfSurjective(lean_object* v_R_498_, lean_object* v_S_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_M_503_, lean_object* v_N_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_h_513_, lean_object* v_f_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_mathlib_LinearEquiv_extendScalarsOfSurjective___redArg(v_f_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_extendScalarsOfSurjective___boxed(lean_object** _args){
lean_object* v_R_516_ = _args[0];
lean_object* v_S_517_ = _args[1];
lean_object* v_inst_518_ = _args[2];
lean_object* v_inst_519_ = _args[3];
lean_object* v_inst_520_ = _args[4];
lean_object* v_M_521_ = _args[5];
lean_object* v_N_522_ = _args[6];
lean_object* v_inst_523_ = _args[7];
lean_object* v_inst_524_ = _args[8];
lean_object* v_inst_525_ = _args[9];
lean_object* v_inst_526_ = _args[10];
lean_object* v_inst_527_ = _args[11];
lean_object* v_inst_528_ = _args[12];
lean_object* v_inst_529_ = _args[13];
lean_object* v_inst_530_ = _args[14];
lean_object* v_h_531_ = _args[15];
lean_object* v_f_532_ = _args[16];
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_mathlib_LinearEquiv_extendScalarsOfSurjective(v_R_516_, v_S_517_, v_inst_518_, v_inst_519_, v_inst_520_, v_M_521_, v_N_522_, v_inst_523_, v_inst_524_, v_inst_525_, v_inst_526_, v_inst_527_, v_inst_528_, v_inst_529_, v_inst_530_, v_h_531_, v_f_532_);
lean_dec(v_inst_529_);
lean_dec(v_inst_528_);
lean_dec(v_inst_526_);
lean_dec(v_inst_525_);
lean_dec_ref(v_inst_524_);
lean_dec_ref(v_inst_523_);
lean_dec_ref(v_inst_520_);
lean_dec_ref(v_inst_519_);
lean_dec_ref(v_inst_518_);
return v_res_533_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_RestrictScalars(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CharZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_CharZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_InjSurj(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_RestrictScalars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_RestrictScalars(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CharZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_CharZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_InjSurj(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_RestrictScalars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
