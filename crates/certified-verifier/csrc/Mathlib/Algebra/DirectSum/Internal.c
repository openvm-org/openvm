// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Internal
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Operations public import Mathlib.Algebra.Algebra.Subalgebra.Basic public import Mathlib.Algebra.DirectSum.Algebra public import Mathlib.Algebra.Order.Antidiag.Prod
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
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DirectSum_toAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_SubsemiringClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* l_instSMulOfMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_SetLike_gMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_SetLike_gMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_InvMemClass_inv___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_sub___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_DirectSum_toSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gnonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gnonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gnonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_coeRingHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_coeRingHom___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_coeRingHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_galgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_galgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_galgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_coeAlgHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_coeAlgHom___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_coeAlgHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subsemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subsemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___aux__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___aux__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___aux__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__0 = (const lean_object*)&lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__0_value;
static const lean_closure_object lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubsemiringClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__1 = (const lean_object*)&lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__1_value;
static const lean_closure_object lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_comp___redArg___lam__0, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__1_value),((lean_object*)&lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__0_value)} };
static const lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__2 = (const lean_object*)&lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gnonUnitalNonAssocSemiring___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v_toMul_3_; lean_object* v___f_4_; 
v___x_2_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_1_);
v_toMul_3_ = lean_ctor_get(v___x_2_, 0);
lean_inc(v_toMul_3_);
lean_dec_ref(v___x_2_);
v___f_4_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_4_, 0, v_toMul_3_);
return v___f_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gnonUnitalNonAssocSemiring(lean_object* v_00_u03b9_5_, lean_object* v_00_u03c3_6_, lean_object* v_R_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_A_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_SetLike_gnonUnitalNonAssocSemiring___redArg(v_inst_9_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gnonUnitalNonAssocSemiring___boxed(lean_object* v_00_u03b9_15_, lean_object* v_00_u03c3_16_, lean_object* v_R_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_A_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_SetLike_gnonUnitalNonAssocSemiring(v_00_u03b9_15_, v_00_u03c3_16_, v_R_17_, v_inst_18_, v_inst_19_, v_inst_20_, v_inst_21_, v_A_22_, v_inst_23_);
lean_dec(v_A_22_);
lean_dec(v_inst_18_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg___lam__0(lean_object* v_toNatCast_25_, lean_object* v_n_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_apply_1(v_toNatCast_25_, v_n_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg___lam__1(lean_object* v_toNPow_28_, lean_object* v_n_29_, lean_object* v_x_30_, lean_object* v_a_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_apply_2(v_toNPow_28_, v_n_29_, v_a_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg___lam__1___boxed(lean_object* v_toNPow_33_, lean_object* v_n_34_, lean_object* v_x_35_, lean_object* v_a_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_SetLike_gsemiring___redArg___lam__1(v_toNPow_33_, v_n_34_, v_x_35_, v_a_36_);
lean_dec(v_x_35_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___redArg(lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; lean_object* v_toNonUnitalNonAssocSemiring_40_; lean_object* v_toMonoid_41_; lean_object* v___x_42_; lean_object* v_toNatCast_43_; lean_object* v___x_44_; lean_object* v_toGOne_45_; lean_object* v_toNPow_46_; lean_object* v___f_47_; lean_object* v___x_48_; lean_object* v___f_49_; lean_object* v___x_50_; 
lean_inc_ref(v_inst_38_);
v___x_39_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_38_);
v_toNonUnitalNonAssocSemiring_40_ = lean_ctor_get(v___x_39_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_40_);
v_toMonoid_41_ = lean_ctor_get(v_inst_38_, 1);
lean_inc_ref_n(v_toMonoid_41_, 2);
lean_dec_ref(v_inst_38_);
v___x_42_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_39_);
v_toNatCast_43_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_toNatCast_43_);
lean_dec_ref(v___x_42_);
v___x_44_ = lp_mathlib_SetLike_gMonoid___redArg(v_toMonoid_41_);
v_toGOne_45_ = lean_ctor_get(v___x_44_, 1);
lean_inc(v_toGOne_45_);
lean_dec_ref(v___x_44_);
v_toNPow_46_ = lean_ctor_get(v_toMonoid_41_, 2);
lean_inc(v_toNPow_46_);
lean_dec_ref(v_toMonoid_41_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gsemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_47_, 0, v_toNatCast_43_);
v___x_48_ = lp_mathlib_SetLike_gnonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_40_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gsemiring___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_49_, 0, v_toNPow_46_);
v___x_50_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_50_, 0, v___x_48_);
lean_ctor_set(v___x_50_, 1, v_toGOne_45_);
lean_ctor_set(v___x_50_, 2, v___f_49_);
lean_ctor_set(v___x_50_, 3, v___f_47_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring(lean_object* v_00_u03b9_51_, lean_object* v_00_u03c3_52_, lean_object* v_R_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_A_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_SetLike_gsemiring___redArg(v_inst_55_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gsemiring___boxed(lean_object* v_00_u03b9_61_, lean_object* v_00_u03c3_62_, lean_object* v_R_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_A_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_SetLike_gsemiring(v_00_u03b9_61_, v_00_u03c3_62_, v_R_63_, v_inst_64_, v_inst_65_, v_inst_66_, v_inst_67_, v_A_68_, v_inst_69_);
lean_dec(v_A_68_);
lean_dec_ref(v_inst_64_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommSemiring___redArg(lean_object* v_inst_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_SetLike_gsemiring___redArg(v_inst_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommSemiring(lean_object* v_00_u03b9_73_, lean_object* v_00_u03c3_74_, lean_object* v_R_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_A_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_SetLike_gsemiring___redArg(v_inst_77_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommSemiring___boxed(lean_object* v_00_u03b9_83_, lean_object* v_00_u03c3_84_, lean_object* v_R_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_A_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_SetLike_gcommSemiring(v_00_u03b9_83_, v_00_u03c3_84_, v_R_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_A_90_, v_inst_91_);
lean_dec(v_A_90_);
lean_dec_ref(v_inst_86_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring___redArg___lam__0(lean_object* v_toIntCast_93_, lean_object* v_z_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_apply_1(v_toIntCast_93_, v_z_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring___redArg(lean_object* v_inst_96_){
_start:
{
lean_object* v_toSemiring_97_; lean_object* v___x_98_; lean_object* v_toIntCast_99_; lean_object* v___f_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v_toSemiring_97_ = lean_ctor_get(v_inst_96_, 0);
lean_inc_ref(v_toSemiring_97_);
v___x_98_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_96_);
v_toIntCast_99_ = lean_ctor_get(v___x_98_, 0);
lean_inc(v_toIntCast_99_);
lean_dec_ref(v___x_98_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_100_, 0, v_toIntCast_99_);
v___x_101_ = lp_mathlib_SetLike_gsemiring___redArg(v_toSemiring_97_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___f_100_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring(lean_object* v_00_u03b9_103_, lean_object* v_00_u03c3_104_, lean_object* v_R_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_A_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_SetLike_gring___redArg(v_inst_107_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gring___boxed(lean_object* v_00_u03b9_113_, lean_object* v_00_u03c3_114_, lean_object* v_R_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_A_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_SetLike_gring(v_00_u03b9_113_, v_00_u03c3_114_, v_R_115_, v_inst_116_, v_inst_117_, v_inst_118_, v_inst_119_, v_A_120_, v_inst_121_);
lean_dec(v_A_120_);
lean_dec_ref(v_inst_116_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommRing___redArg(lean_object* v_inst_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_SetLike_gring___redArg(v_inst_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommRing(lean_object* v_00_u03b9_125_, lean_object* v_00_u03c3_126_, lean_object* v_R_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_A_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_SetLike_gring___redArg(v_inst_129_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gcommRing___boxed(lean_object* v_00_u03b9_135_, lean_object* v_00_u03c3_136_, lean_object* v_R_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_A_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_SetLike_gcommRing(v_00_u03b9_135_, v_00_u03c3_136_, v_R_137_, v_inst_138_, v_inst_139_, v_inst_140_, v_inst_141_, v_A_142_, v_inst_143_);
lean_dec(v_A_142_);
lean_dec_ref(v_inst_138_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__0(lean_object* v_i_145_, lean_object* v___y_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_SubmonoidClass_subtype___lam__0(v___y_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__0___boxed(lean_object* v_i_148_, lean_object* v___y_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_DirectSum_coeRingHom___redArg___lam__0(v_i_148_, v___y_149_);
lean_dec(v___y_149_);
lean_dec(v_i_148_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__1(lean_object* v_toAddCommMonoid_151_, lean_object* v_i_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_toAddCommMonoid_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg___lam__1___boxed(lean_object* v_toAddCommMonoid_154_, lean_object* v_i_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_DirectSum_coeRingHom___redArg___lam__1(v_toAddCommMonoid_154_, v_i_155_);
lean_dec(v_i_155_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___redArg(lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_toAddCommMonoid_160_; lean_object* v___f_161_; lean_object* v___f_162_; lean_object* v___x_163_; 
v_toAddCommMonoid_160_ = lean_ctor_get(v_inst_159_, 0);
v___f_161_ = ((lean_object*)(lp_mathlib_DirectSum_coeRingHom___redArg___closed__0));
lean_inc_ref(v_toAddCommMonoid_160_);
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_coeRingHom___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_162_, 0, v_toAddCommMonoid_160_);
v___x_163_ = lp_mathlib_DirectSum_toSemiring___redArg(v_inst_158_, v___f_162_, v_inst_159_, v___f_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom(lean_object* v_00_u03b9_164_, lean_object* v_00_u03c3_165_, lean_object* v_R_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_A_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_DirectSum_coeRingHom___redArg(v_inst_167_, v_inst_168_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeRingHom___boxed(lean_object* v_00_u03b9_175_, lean_object* v_00_u03c3_176_, lean_object* v_R_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_A_182_, lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_DirectSum_coeRingHom(v_00_u03b9_175_, v_00_u03c3_176_, v_R_177_, v_inst_178_, v_inst_179_, v_inst_180_, v_inst_181_, v_A_182_, v_inst_183_, v_inst_184_);
lean_dec_ref(v_inst_183_);
lean_dec(v_A_182_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_galgebra___redArg(lean_object* v_inst_186_){
_start:
{
lean_object* v_algebraMap_187_; lean_object* v___f_188_; lean_object* v___f_189_; 
v_algebraMap_187_ = lean_ctor_get(v_inst_186_, 1);
lean_inc(v_algebraMap_187_);
lean_dec_ref(v_inst_186_);
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_188_, 0, v_algebraMap_187_);
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_189_, 0, v___f_188_);
return v___f_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_galgebra(lean_object* v_00_u03b9_190_, lean_object* v_S_191_, lean_object* v_R_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_A_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_Submodule_galgebra___redArg(v_inst_196_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_galgebra___boxed(lean_object* v_00_u03b9_200_, lean_object* v_S_201_, lean_object* v_R_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_A_207_, lean_object* v_inst_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_Submodule_galgebra(v_00_u03b9_200_, v_S_201_, v_R_202_, v_inst_203_, v_inst_204_, v_inst_205_, v_inst_206_, v_A_207_, v_inst_208_);
lean_dec_ref(v_A_207_);
lean_dec_ref(v_inst_205_);
lean_dec_ref(v_inst_204_);
lean_dec_ref(v_inst_203_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg___lam__0(lean_object* v_i_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lp_mathlib_SMulMemClass_subtype___lam__0(v___y_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg___lam__0___boxed(lean_object* v_i_213_, lean_object* v___y_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_DirectSum_coeAlgHom___redArg___lam__0(v_i_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec(v_i_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___redArg(lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v_toAddCommMonoid_219_; lean_object* v___f_220_; lean_object* v___f_221_; lean_object* v___x_222_; 
v_toAddCommMonoid_219_ = lean_ctor_get(v_inst_218_, 0);
v___f_220_ = ((lean_object*)(lp_mathlib_DirectSum_coeAlgHom___redArg___closed__0));
lean_inc_ref(v_toAddCommMonoid_219_);
v___f_221_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_coeRingHom___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_221_, 0, v_toAddCommMonoid_219_);
v___x_222_ = lp_mathlib_DirectSum_toAlgebra___redArg(v___f_221_, v_inst_218_, v_inst_217_, v___f_220_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom(lean_object* v_00_u03b9_223_, lean_object* v_S_224_, lean_object* v_R_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_A_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_DirectSum_coeAlgHom___redArg(v_inst_226_, v_inst_229_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAlgHom___boxed(lean_object* v_00_u03b9_234_, lean_object* v_S_235_, lean_object* v_R_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_A_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_DirectSum_coeAlgHom(v_00_u03b9_234_, v_S_235_, v_R_236_, v_inst_237_, v_inst_238_, v_inst_239_, v_inst_240_, v_inst_241_, v_A_242_, v_inst_243_);
lean_dec_ref(v_A_242_);
lean_dec_ref(v_inst_241_);
lean_dec_ref(v_inst_239_);
lean_dec_ref(v_inst_238_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subsemiring(lean_object* v_00_u03b9_245_, lean_object* v_00_u03c3_246_, lean_object* v_R_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_A_252_, lean_object* v_inst_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lean_box(0);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subsemiring___boxed(lean_object* v_00_u03b9_255_, lean_object* v_00_u03c3_256_, lean_object* v_R_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_A_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_SetLike_GradeZero_subsemiring(v_00_u03b9_255_, v_00_u03c3_256_, v_R_257_, v_inst_258_, v_inst_259_, v_inst_260_, v_inst_261_, v_A_262_, v_inst_263_);
lean_dec(v_A_262_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_258_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___aux__5___redArg(lean_object* v_inst_265_, lean_object* v_n_266_){
_start:
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v_toNatCast_269_; lean_object* v___x_270_; 
v___x_267_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_265_);
v___x_268_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_267_);
v_toNatCast_269_ = lean_ctor_get(v___x_268_, 0);
lean_inc(v_toNatCast_269_);
lean_dec_ref(v___x_268_);
v___x_270_ = lean_apply_1(v_toNatCast_269_, v_n_266_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___aux__5(lean_object* v_00_u03b9_271_, lean_object* v_00_u03c3_272_, lean_object* v_R_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_A_278_, lean_object* v_inst_279_, lean_object* v_n_280_){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v_toNatCast_283_; lean_object* v___x_284_; 
v___x_281_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_274_);
v___x_282_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_281_);
v_toNatCast_283_ = lean_ctor_get(v___x_282_, 0);
lean_inc(v_toNatCast_283_);
lean_dec_ref(v___x_282_);
v___x_284_ = lean_apply_1(v_toNatCast_283_, v_n_280_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___aux__5___boxed(lean_object* v_00_u03b9_285_, lean_object* v_00_u03c3_286_, lean_object* v_R_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_A_292_, lean_object* v_inst_293_, lean_object* v_n_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_SetLike_GradeZero_instSemiring___aux__5(v_00_u03b9_285_, v_00_u03c3_286_, v_R_287_, v_inst_288_, v_inst_289_, v_inst_290_, v_inst_291_, v_A_292_, v_inst_293_, v_n_294_);
lean_dec(v_A_292_);
lean_dec_ref(v_inst_289_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring___redArg(lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_A_299_){
_start:
{
lean_object* v_toAddCommMonoid_300_; lean_object* v_toMonoid_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v_toAddCommMonoid_300_ = lean_ctor_get(v_inst_296_, 0);
v_toMonoid_301_ = lean_ctor_get(v_inst_296_, 1);
lean_inc_ref(v_toAddCommMonoid_300_);
v___x_302_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_toAddCommMonoid_300_);
lean_inc(v_A_299_);
lean_inc_ref(v_inst_297_);
lean_inc_ref(v_toMonoid_301_);
v___x_303_ = lp_mathlib_SetLike_GradeZero_instMonoid___redArg(v_inst_298_, v_toMonoid_301_, v_inst_297_, v_A_299_);
v___x_304_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_GradeZero_instSemiring___aux__5___boxed), 10, 9);
lean_closure_set(v___x_304_, 0, lean_box(0));
lean_closure_set(v___x_304_, 1, lean_box(0));
lean_closure_set(v___x_304_, 2, lean_box(0));
lean_closure_set(v___x_304_, 3, v_inst_296_);
lean_closure_set(v___x_304_, 4, v_inst_297_);
lean_closure_set(v___x_304_, 5, v_inst_298_);
lean_closure_set(v___x_304_, 6, lean_box(0));
lean_closure_set(v___x_304_, 7, v_A_299_);
lean_closure_set(v___x_304_, 8, lean_box(0));
v___x_305_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_305_, 0, v___x_302_);
lean_ctor_set(v___x_305_, 1, v___x_303_);
lean_ctor_set(v___x_305_, 2, v___x_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instSemiring(lean_object* v_00_u03b9_306_, lean_object* v_00_u03c3_307_, lean_object* v_R_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_A_313_, lean_object* v_inst_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_SetLike_GradeZero_instSemiring___redArg(v_inst_309_, v_inst_310_, v_inst_311_, v_A_313_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommSemiring___redArg(lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_A_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_mathlib_SetLike_GradeZero_instSemiring___redArg(v_inst_316_, v_inst_317_, v_inst_318_, v_A_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommSemiring(lean_object* v_00_u03b9_321_, lean_object* v_00_u03c3_322_, lean_object* v_R_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_A_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_SetLike_GradeZero_instSemiring___redArg(v_inst_324_, v_inst_325_, v_inst_326_, v_A_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1(lean_object* v_00_u03b9_336_, lean_object* v_00_u03c3_337_, lean_object* v_R_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_A_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___f_345_; 
v___f_345_ = ((lean_object*)(lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__2));
return v___f_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___boxed(lean_object* v_00_u03b9_346_, lean_object* v_00_u03c3_347_, lean_object* v_R_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_A_353_, lean_object* v_inst_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1(v_00_u03b9_346_, v_00_u03c3_347_, v_R_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_inst_352_, v_A_353_, v_inst_354_);
lean_dec(v_A_353_);
lean_dec_ref(v_inst_350_);
lean_dec_ref(v_inst_349_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___redArg(lean_object* v_inst_356_){
_start:
{
lean_object* v___x_357_; lean_object* v_toMul_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_368_; 
v___x_357_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_356_);
v_toMul_358_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_368_ == 0)
{
lean_object* v_unused_369_; 
v_unused_369_ = lean_ctor_get(v___x_357_, 1);
lean_dec(v_unused_369_);
v___x_360_ = v___x_357_;
v_isShared_361_ = v_isSharedCheck_368_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_toMul_358_);
lean_dec(v___x_357_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_368_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v___f_362_; lean_object* v___f_363_; lean_object* v___f_364_; lean_object* v___x_366_; 
v___f_362_ = lean_alloc_closure((void*)(l_instSMulOfMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_362_, 0, v_toMul_358_);
v___f_363_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_363_, 0, v___f_362_);
v___f_364_ = ((lean_object*)(lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___aux__1___closed__2));
if (v_isShared_361_ == 0)
{
lean_ctor_set(v___x_360_, 1, v___f_364_);
lean_ctor_set(v___x_360_, 0, v___f_363_);
v___x_366_ = v___x_360_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v___f_363_);
lean_ctor_set(v_reuseFailAlloc_367_, 1, v___f_364_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
return v___x_366_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat(lean_object* v_00_u03b9_370_, lean_object* v_00_u03c3_371_, lean_object* v_R_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_A_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___redArg(v_inst_373_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat___boxed(lean_object* v_00_u03b9_380_, lean_object* v_00_u03c3_381_, lean_object* v_R_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_A_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_SetLike_GradeZero_instAlgebraSubtypeMemOfNat(v_00_u03b9_380_, v_00_u03c3_381_, v_R_382_, v_inst_383_, v_inst_384_, v_inst_385_, v_inst_386_, v_A_387_, v_inst_388_);
lean_dec(v_A_387_);
lean_dec_ref(v_inst_384_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subring(lean_object* v_00_u03b9_390_, lean_object* v_00_u03c3_391_, lean_object* v_R_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_A_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lean_box(0);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subring___boxed(lean_object* v_00_u03b9_400_, lean_object* v_00_u03c3_401_, lean_object* v_R_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_A_407_, lean_object* v_inst_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_SetLike_GradeZero_subring(v_00_u03b9_400_, v_00_u03c3_401_, v_R_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_A_407_, v_inst_408_);
lean_dec(v_A_407_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6___redArg(lean_object* v_inst_410_, lean_object* v_n_411_){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v_toIntCast_415_; lean_object* v___x_416_; 
v___x_412_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_410_);
v___x_413_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_412_);
v___x_414_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_413_);
lean_dec_ref(v___x_413_);
v_toIntCast_415_ = lean_ctor_get(v___x_414_, 0);
lean_inc(v_toIntCast_415_);
lean_dec_ref(v___x_414_);
v___x_416_ = lean_apply_1(v_toIntCast_415_, v_n_411_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6___redArg___boxed(lean_object* v_inst_417_, lean_object* v_n_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib_SetLike_GradeZero_instRing___aux__6___redArg(v_inst_417_, v_n_418_);
lean_dec_ref(v_inst_417_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6(lean_object* v_00_u03b9_420_, lean_object* v_00_u03c3_421_, lean_object* v_R_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_A_427_, lean_object* v_inst_428_, lean_object* v_n_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v_toIntCast_433_; lean_object* v___x_434_; 
v___x_430_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_423_);
v___x_431_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_430_);
v___x_432_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_431_);
lean_dec_ref(v___x_431_);
v_toIntCast_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc(v_toIntCast_433_);
lean_dec_ref(v___x_432_);
v___x_434_ = lean_apply_1(v_toIntCast_433_, v_n_429_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___aux__6___boxed(lean_object* v_00_u03b9_435_, lean_object* v_00_u03c3_436_, lean_object* v_R_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_A_442_, lean_object* v_inst_443_, lean_object* v_n_444_){
_start:
{
lean_object* v_res_445_; 
v_res_445_ = lp_mathlib_SetLike_GradeZero_instRing___aux__6(v_00_u03b9_435_, v_00_u03c3_436_, v_R_437_, v_inst_438_, v_inst_439_, v_inst_440_, v_inst_441_, v_A_442_, v_inst_443_, v_n_444_);
lean_dec(v_A_442_);
lean_dec_ref(v_inst_439_);
lean_dec_ref(v_inst_438_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing___redArg(lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_A_449_){
_start:
{
lean_object* v_toSemiring_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v_toNeg_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v_toZSMul_458_; lean_object* v___f_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
v_toSemiring_450_ = lean_ctor_get(v_inst_446_, 0);
lean_inc(v_A_449_);
lean_inc_ref(v_inst_447_);
lean_inc_ref(v_toSemiring_450_);
v___x_451_ = lp_mathlib_SetLike_GradeZero_instSemiring___redArg(v_toSemiring_450_, v_inst_447_, v_inst_448_, v_A_449_);
v___x_452_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_446_);
v___x_453_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_452_);
v_toNeg_454_ = lean_ctor_get(v___x_453_, 1);
lean_inc(v_toNeg_454_);
lean_dec_ref(v___x_453_);
lean_inc_ref(v_inst_446_);
v___x_455_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_446_);
v___x_456_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_455_);
lean_dec_ref(v___x_455_);
v___x_457_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v___x_452_);
v_toZSMul_458_ = lean_ctor_get(v___x_457_, 3);
lean_inc(v_toZSMul_458_);
lean_dec_ref(v___x_457_);
v___f_459_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_459_, 0, v_toNeg_454_);
v___x_460_ = lp_mathlib_AddSubgroupClass_sub___redArg(v___x_456_);
v___x_461_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_GradeZero_instRing___aux__6___boxed), 10, 9);
lean_closure_set(v___x_461_, 0, lean_box(0));
lean_closure_set(v___x_461_, 1, lean_box(0));
lean_closure_set(v___x_461_, 2, lean_box(0));
lean_closure_set(v___x_461_, 3, v_inst_446_);
lean_closure_set(v___x_461_, 4, v_inst_447_);
lean_closure_set(v___x_461_, 5, v_inst_448_);
lean_closure_set(v___x_461_, 6, lean_box(0));
lean_closure_set(v___x_461_, 7, v_A_449_);
lean_closure_set(v___x_461_, 8, lean_box(0));
v___x_462_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_462_, 0, v___x_451_);
lean_ctor_set(v___x_462_, 1, v___f_459_);
lean_ctor_set(v___x_462_, 2, v___x_460_);
lean_ctor_set(v___x_462_, 3, v_toZSMul_458_);
lean_ctor_set(v___x_462_, 4, v___x_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instRing(lean_object* v_00_u03b9_463_, lean_object* v_00_u03c3_464_, lean_object* v_R_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_A_470_, lean_object* v_inst_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lp_mathlib_SetLike_GradeZero_instRing___redArg(v_inst_466_, v_inst_467_, v_inst_468_, v_A_470_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommRing___redArg(lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_A_476_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = lp_mathlib_SetLike_GradeZero_instRing___redArg(v_inst_473_, v_inst_474_, v_inst_475_, v_A_476_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommRing(lean_object* v_00_u03b9_478_, lean_object* v_00_u03c3_479_, lean_object* v_R_480_, lean_object* v_inst_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_A_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_SetLike_GradeZero_instRing___redArg(v_inst_481_, v_inst_482_, v_inst_483_, v_A_485_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subalgebra(lean_object* v_00_u03b9_488_, lean_object* v_S_489_, lean_object* v_R_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_A_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lean_box(0);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_subalgebra___boxed(lean_object* v_00_u03b9_498_, lean_object* v_S_499_, lean_object* v_R_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_A_505_, lean_object* v_inst_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_SetLike_GradeZero_subalgebra(v_00_u03b9_498_, v_S_499_, v_R_500_, v_inst_501_, v_inst_502_, v_inst_503_, v_inst_504_, v_A_505_, v_inst_506_);
lean_dec_ref(v_A_505_);
lean_dec_ref(v_inst_504_);
lean_dec_ref(v_inst_503_);
lean_dec_ref(v_inst_502_);
lean_dec_ref(v_inst_501_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1___redArg(lean_object* v_inst_508_){
_start:
{
lean_object* v_algebraMap_509_; lean_object* v___f_510_; 
v_algebraMap_509_ = lean_ctor_get(v_inst_508_, 1);
lean_inc(v_algebraMap_509_);
lean_dec_ref(v_inst_508_);
v___f_510_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_510_, 0, v_algebraMap_509_);
return v___f_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1(lean_object* v_00_u03b9_511_, lean_object* v_S_512_, lean_object* v_R_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_A_518_, lean_object* v_inst_519_){
_start:
{
lean_object* v_algebraMap_520_; lean_object* v___f_521_; 
v_algebraMap_520_ = lean_ctor_get(v_inst_516_, 1);
lean_inc(v_algebraMap_520_);
lean_dec_ref(v_inst_516_);
v___f_521_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_521_, 0, v_algebraMap_520_);
return v___f_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1___boxed(lean_object* v_00_u03b9_522_, lean_object* v_S_523_, lean_object* v_R_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_A_529_, lean_object* v_inst_530_){
_start:
{
lean_object* v_res_531_; 
v_res_531_ = lp_mathlib_SetLike_GradeZero_instAlgebra___aux__1(v_00_u03b9_522_, v_S_523_, v_R_524_, v_inst_525_, v_inst_526_, v_inst_527_, v_inst_528_, v_A_529_, v_inst_530_);
lean_dec_ref(v_A_529_);
lean_dec_ref(v_inst_528_);
lean_dec_ref(v_inst_526_);
lean_dec_ref(v_inst_525_);
return v_res_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___redArg(lean_object* v_inst_532_){
_start:
{
lean_object* v_toSMul_533_; lean_object* v_algebraMap_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_543_; 
v_toSMul_533_ = lean_ctor_get(v_inst_532_, 0);
v_algebraMap_534_ = lean_ctor_get(v_inst_532_, 1);
v_isSharedCheck_543_ = !lean_is_exclusive(v_inst_532_);
if (v_isSharedCheck_543_ == 0)
{
v___x_536_ = v_inst_532_;
v_isShared_537_ = v_isSharedCheck_543_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_algebraMap_534_);
lean_inc(v_toSMul_533_);
lean_dec(v_inst_532_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_543_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v___f_538_; lean_object* v___f_539_; lean_object* v___x_541_; 
v___f_538_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_538_, 0, v_toSMul_533_);
v___f_539_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_539_, 0, v_algebraMap_534_);
if (v_isShared_537_ == 0)
{
lean_ctor_set(v___x_536_, 1, v___f_539_);
lean_ctor_set(v___x_536_, 0, v___f_538_);
v___x_541_ = v___x_536_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v___f_538_);
lean_ctor_set(v_reuseFailAlloc_542_, 1, v___f_539_);
v___x_541_ = v_reuseFailAlloc_542_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
return v___x_541_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra(lean_object* v_00_u03b9_544_, lean_object* v_S_545_, lean_object* v_R_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_A_551_, lean_object* v_inst_552_){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_mathlib_SetLike_GradeZero_instAlgebra___redArg(v_inst_549_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instAlgebra___boxed(lean_object* v_00_u03b9_554_, lean_object* v_S_555_, lean_object* v_R_556_, lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_inst_560_, lean_object* v_A_561_, lean_object* v_inst_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_SetLike_GradeZero_instAlgebra(v_00_u03b9_554_, v_S_555_, v_R_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_inst_560_, v_A_561_, v_inst_562_);
lean_dec_ref(v_A_561_);
lean_dec_ref(v_inst_560_);
lean_dec_ref(v_inst_558_);
lean_dec_ref(v_inst_557_);
return v_res_563_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(builtin);
}
#ifdef __cplusplus
}
#endif
