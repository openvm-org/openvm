// Lean compiler output
// Module: Mathlib.Algebra.Ring.Prod
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.Cast.Prod public import Mathlib.Algebra.GroupWithZero.Prod public import Mathlib.Algebra.Ring.CompTypeclasses public import Mathlib.Algebra.Ring.Equiv
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMulZeroOneClass___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMonoidWithZero___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistrib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistrib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommRing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalRingHom_fst___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalRingHom_snd___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_prodComm___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_prodComm___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_prodProdProdComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_prodProdProdComm___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_prodProdProdComm___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_prodProdProdComm___closed__0_value;
static const lean_closure_object lp_mathlib_RingEquiv_prodProdProdComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_prodProdProdComm___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_prodProdProdComm___closed__1 = (const lean_object*)&lp_mathlib_RingEquiv_prodProdProdComm___closed__1_value;
static const lean_ctor_object lp_mathlib_RingEquiv_prodProdProdComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RingEquiv_prodProdProdComm___closed__0_value),((lean_object*)&lp_mathlib_RingEquiv_prodProdProdComm___closed__1_value)}};
static const lean_object* lp_mathlib_RingEquiv_prodProdProdComm___closed__2 = (const lean_object*)&lp_mathlib_RingEquiv_prodProdProdComm___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistrib___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v_toMul_3_; lean_object* v_toAdd_4_; lean_object* v_toMul_5_; lean_object* v_toAdd_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_15_; 
v_toMul_3_ = lean_ctor_get(v_inst_1_, 0);
lean_inc(v_toMul_3_);
v_toAdd_4_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_toAdd_4_);
lean_dec_ref(v_inst_1_);
v_toMul_5_ = lean_ctor_get(v_inst_2_, 0);
v_toAdd_6_ = lean_ctor_get(v_inst_2_, 1);
v_isSharedCheck_15_ = !lean_is_exclusive(v_inst_2_);
if (v_isSharedCheck_15_ == 0)
{
v___x_8_ = v_inst_2_;
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_toAdd_6_);
lean_inc(v_toMul_5_);
lean_dec(v_inst_2_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___f_10_; lean_object* v___f_11_; lean_object* v___x_13_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_10_, 0, v_toMul_3_);
lean_closure_set(v___f_10_, 1, v_toMul_5_);
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_11_, 0, v_toAdd_4_);
lean_closure_set(v___f_11_, 1, v_toAdd_6_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___f_11_);
lean_ctor_set(v___x_8_, 0, v___f_10_);
v___x_13_ = v___x_8_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_14_; 
v_reuseFailAlloc_14_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_14_, 0, v___f_10_);
lean_ctor_set(v_reuseFailAlloc_14_, 1, v___f_11_);
v___x_13_ = v_reuseFailAlloc_14_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
return v___x_13_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistrib(lean_object* v_R_16_, lean_object* v_S_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Prod_instDistrib___redArg(v_inst_18_, v_inst_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_toAddCommMonoid_23_; lean_object* v_toAddCommMonoid_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v_toMul_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_36_; 
v_toAddCommMonoid_23_ = lean_ctor_get(v_inst_21_, 0);
v_toAddCommMonoid_24_ = lean_ctor_get(v_inst_22_, 0);
lean_inc_ref(v_toAddCommMonoid_24_);
lean_inc_ref(v_toAddCommMonoid_23_);
v___x_25_ = lp_mathlib_Prod_instAddMonoid___redArg(v_toAddCommMonoid_23_, v_toAddCommMonoid_24_);
v___x_26_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_21_);
v___x_27_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_22_);
v___x_28_ = lp_mathlib_Prod_instDistrib___redArg(v___x_26_, v___x_27_);
v_toMul_29_ = lean_ctor_get(v___x_28_, 0);
v_isSharedCheck_36_ = !lean_is_exclusive(v___x_28_);
if (v_isSharedCheck_36_ == 0)
{
lean_object* v_unused_37_; 
v_unused_37_ = lean_ctor_get(v___x_28_, 1);
lean_dec(v_unused_37_);
v___x_31_ = v___x_28_;
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_toMul_29_);
lean_dec(v___x_28_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
lean_object* v___x_34_; 
if (v_isShared_32_ == 0)
{
lean_ctor_set(v___x_31_, 1, v_toMul_29_);
lean_ctor_set(v___x_31_, 0, v___x_25_);
v___x_34_ = v___x_31_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v___x_25_);
lean_ctor_set(v_reuseFailAlloc_35_, 1, v_toMul_29_);
v___x_34_ = v_reuseFailAlloc_35_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
return v___x_34_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocSemiring(lean_object* v_R_38_, lean_object* v_S_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v_inst_40_, v_inst_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalSemiring___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v_inst_43_, v_inst_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalSemiring(lean_object* v_R_46_, lean_object* v_S_47_, lean_object* v_inst_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v_inst_48_, v_inst_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocSemiring___redArg(lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_53_; lean_object* v_toNonUnitalNonAssocSemiring_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v_toMulOneClass_62_; lean_object* v_toOne_63_; lean_object* v_toNatCast_64_; lean_object* v___x_66_; uint8_t v_isShared_67_; uint8_t v_isSharedCheck_71_; 
v_toNonUnitalNonAssocSemiring_53_ = lean_ctor_get(v_inst_51_, 0);
v_toNonUnitalNonAssocSemiring_54_ = lean_ctor_get(v_inst_52_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_54_);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_53_);
v___x_55_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_53_, v_toNonUnitalNonAssocSemiring_54_);
lean_inc_ref(v_inst_51_);
v___x_56_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_51_);
lean_inc_ref(v_inst_52_);
v___x_57_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_52_);
v___x_58_ = lp_mathlib_Prod_instMulZeroOneClass___redArg(v___x_56_, v___x_57_);
v___x_59_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_51_);
v___x_60_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_52_);
v___x_61_ = lp_mathlib_Prod_instAddMonoidWithOne___redArg(v___x_59_, v___x_60_);
v_toMulOneClass_62_ = lean_ctor_get(v___x_58_, 0);
lean_inc_ref(v_toMulOneClass_62_);
lean_dec_ref(v___x_58_);
v_toOne_63_ = lean_ctor_get(v_toMulOneClass_62_, 0);
lean_inc(v_toOne_63_);
lean_dec_ref(v_toMulOneClass_62_);
v_toNatCast_64_ = lean_ctor_get(v___x_61_, 0);
v_isSharedCheck_71_ = !lean_is_exclusive(v___x_61_);
if (v_isSharedCheck_71_ == 0)
{
lean_object* v_unused_72_; lean_object* v_unused_73_; 
v_unused_72_ = lean_ctor_get(v___x_61_, 2);
lean_dec(v_unused_72_);
v_unused_73_ = lean_ctor_get(v___x_61_, 1);
lean_dec(v_unused_73_);
v___x_66_ = v___x_61_;
v_isShared_67_ = v_isSharedCheck_71_;
goto v_resetjp_65_;
}
else
{
lean_inc(v_toNatCast_64_);
lean_dec(v___x_61_);
v___x_66_ = lean_box(0);
v_isShared_67_ = v_isSharedCheck_71_;
goto v_resetjp_65_;
}
v_resetjp_65_:
{
lean_object* v___x_69_; 
if (v_isShared_67_ == 0)
{
lean_ctor_set(v___x_66_, 2, v_toNatCast_64_);
lean_ctor_set(v___x_66_, 1, v_toOne_63_);
lean_ctor_set(v___x_66_, 0, v___x_55_);
v___x_69_ = v___x_66_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v___x_55_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_toOne_63_);
lean_ctor_set(v_reuseFailAlloc_70_, 2, v_toNatCast_64_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
return v___x_69_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocSemiring(lean_object* v_R_74_, lean_object* v_S_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Prod_instNonAssocSemiring___redArg(v_inst_76_, v_inst_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemiring___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v_toMonoid_90_; lean_object* v_toAddCommMonoid_91_; lean_object* v_toMul_92_; lean_object* v_toOne_93_; lean_object* v_toNatCast_94_; lean_object* v___x_96_; uint8_t v_isShared_97_; uint8_t v_isSharedCheck_111_; 
v___x_81_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_79_);
v___x_82_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_80_);
v___x_83_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v___x_81_, v___x_82_);
lean_inc_ref(v_inst_79_);
v___x_84_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_79_);
lean_inc_ref(v_inst_80_);
v___x_85_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_80_);
v___x_86_ = lp_mathlib_Prod_instNonAssocSemiring___redArg(v___x_84_, v___x_85_);
v___x_87_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_79_);
lean_dec_ref(v_inst_79_);
v___x_88_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_80_);
lean_dec_ref(v_inst_80_);
v___x_89_ = lp_mathlib_Prod_instMonoidWithZero___redArg(v___x_87_, v___x_88_);
v_toMonoid_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc_ref(v_toMonoid_90_);
lean_dec_ref(v___x_89_);
v_toAddCommMonoid_91_ = lean_ctor_get(v___x_83_, 0);
lean_inc_ref(v_toAddCommMonoid_91_);
v_toMul_92_ = lean_ctor_get(v___x_83_, 1);
lean_inc(v_toMul_92_);
lean_dec_ref(v___x_83_);
v_toOne_93_ = lean_ctor_get(v___x_86_, 1);
v_toNatCast_94_ = lean_ctor_get(v___x_86_, 2);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_111_ == 0)
{
lean_object* v_unused_112_; 
v_unused_112_ = lean_ctor_get(v___x_86_, 0);
lean_dec(v_unused_112_);
v___x_96_ = v___x_86_;
v_isShared_97_ = v_isSharedCheck_111_;
goto v_resetjp_95_;
}
else
{
lean_inc(v_toNatCast_94_);
lean_inc(v_toOne_93_);
lean_dec(v___x_86_);
v___x_96_ = lean_box(0);
v_isShared_97_ = v_isSharedCheck_111_;
goto v_resetjp_95_;
}
v_resetjp_95_:
{
lean_object* v_toNPow_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_108_; 
v_toNPow_98_ = lean_ctor_get(v_toMonoid_90_, 2);
v_isSharedCheck_108_ = !lean_is_exclusive(v_toMonoid_90_);
if (v_isSharedCheck_108_ == 0)
{
lean_object* v_unused_109_; lean_object* v_unused_110_; 
v_unused_109_ = lean_ctor_get(v_toMonoid_90_, 1);
lean_dec(v_unused_109_);
v_unused_110_ = lean_ctor_get(v_toMonoid_90_, 0);
lean_dec(v_unused_110_);
v___x_100_ = v_toMonoid_90_;
v_isShared_101_ = v_isSharedCheck_108_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_toNPow_98_);
lean_dec(v_toMonoid_90_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_108_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 1, v_toMul_92_);
lean_ctor_set(v___x_100_, 0, v_toOne_93_);
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v_toOne_93_);
lean_ctor_set(v_reuseFailAlloc_107_, 1, v_toMul_92_);
lean_ctor_set(v_reuseFailAlloc_107_, 2, v_toNPow_98_);
v___x_103_ = v_reuseFailAlloc_107_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
lean_object* v___x_105_; 
if (v_isShared_97_ == 0)
{
lean_ctor_set(v___x_96_, 1, v___x_103_);
lean_ctor_set(v___x_96_, 0, v_toAddCommMonoid_91_);
v___x_105_ = v___x_96_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v_toAddCommMonoid_91_);
lean_ctor_set(v_reuseFailAlloc_106_, 1, v___x_103_);
lean_ctor_set(v_reuseFailAlloc_106_, 2, v_toNatCast_94_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemiring(lean_object* v_R_113_, lean_object* v_S_114_, lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_Prod_instSemiring___redArg(v_inst_115_, v_inst_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommSemiring___redArg(lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v_inst_118_, v_inst_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommSemiring(lean_object* v_R_121_, lean_object* v_S_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v_inst_123_, v_inst_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemiring___redArg(lean_object* v_inst_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_Prod_instSemiring___redArg(v_inst_126_, v_inst_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemiring(lean_object* v_R_129_, lean_object* v_S_130_, lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_Prod_instSemiring___redArg(v_inst_131_, v_inst_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(lean_object* v_inst_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v_toAddCommGroup_136_; lean_object* v_toAddCommGroup_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v_toMul_142_; lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_149_; 
v_toAddCommGroup_136_ = lean_ctor_get(v_inst_134_, 0);
v_toAddCommGroup_137_ = lean_ctor_get(v_inst_135_, 0);
lean_inc_ref(v_toAddCommGroup_137_);
lean_inc_ref(v_toAddCommGroup_136_);
v___x_138_ = lp_mathlib_Prod_subNegMonoid___redArg(v_toAddCommGroup_136_, v_toAddCommGroup_137_);
v___x_139_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_134_);
v___x_140_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_135_);
v___x_141_ = lp_mathlib_Prod_instNonUnitalNonAssocSemiring___redArg(v___x_139_, v___x_140_);
v_toMul_142_ = lean_ctor_get(v___x_141_, 1);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_149_ == 0)
{
lean_object* v_unused_150_; 
v_unused_150_ = lean_ctor_get(v___x_141_, 0);
lean_dec(v_unused_150_);
v___x_144_ = v___x_141_;
v_isShared_145_ = v_isSharedCheck_149_;
goto v_resetjp_143_;
}
else
{
lean_inc(v_toMul_142_);
lean_dec(v___x_141_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_149_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
lean_object* v___x_147_; 
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_138_);
v___x_147_ = v___x_144_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_138_);
lean_ctor_set(v_reuseFailAlloc_148_, 1, v_toMul_142_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalNonAssocRing(lean_object* v_R_151_, lean_object* v_S_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(v_inst_153_, v_inst_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalRing___redArg(lean_object* v_inst_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(v_inst_156_, v_inst_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalRing(lean_object* v_R_159_, lean_object* v_S_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(v_inst_161_, v_inst_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocRing___redArg(lean_object* v_inst_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_166_; lean_object* v_toNonUnitalNonAssocRing_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v_toOne_177_; lean_object* v_toNatCast_178_; lean_object* v_toIntCast_179_; lean_object* v___x_180_; 
v_toNonUnitalNonAssocRing_166_ = lean_ctor_get(v_inst_164_, 0);
v_toNonUnitalNonAssocRing_167_ = lean_ctor_get(v_inst_165_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_167_);
lean_inc_ref(v_toNonUnitalNonAssocRing_166_);
v___x_168_ = lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(v_toNonUnitalNonAssocRing_166_, v_toNonUnitalNonAssocRing_167_);
lean_inc_ref(v_inst_164_);
v___x_169_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_164_);
lean_inc_ref(v_inst_165_);
v___x_170_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_165_);
v___x_171_ = lp_mathlib_Prod_instNonAssocSemiring___redArg(v___x_169_, v___x_170_);
v___x_172_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_164_);
v___x_173_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_172_);
lean_dec_ref(v___x_172_);
v___x_174_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_165_);
v___x_175_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_174_);
lean_dec_ref(v___x_174_);
v___x_176_ = lp_mathlib_Prod_instAddGroupWithOne___redArg(v___x_173_, v___x_175_);
v_toOne_177_ = lean_ctor_get(v___x_171_, 1);
lean_inc(v_toOne_177_);
v_toNatCast_178_ = lean_ctor_get(v___x_171_, 2);
lean_inc(v_toNatCast_178_);
lean_dec_ref(v___x_171_);
v_toIntCast_179_ = lean_ctor_get(v___x_176_, 0);
lean_inc(v_toIntCast_179_);
lean_dec_ref(v___x_176_);
v___x_180_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_180_, 0, v___x_168_);
lean_ctor_set(v___x_180_, 1, v_toOne_177_);
lean_ctor_set(v___x_180_, 2, v_toNatCast_178_);
lean_ctor_set(v___x_180_, 3, v_toIntCast_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonAssocRing(lean_object* v_R_181_, lean_object* v_S_182_, lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Prod_instNonAssocRing___redArg(v_inst_183_, v_inst_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRing___redArg(lean_object* v_inst_186_, lean_object* v_inst_187_){
_start:
{
lean_object* v_toSemiring_188_; lean_object* v_toSemiring_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v_toNeg_197_; lean_object* v_toSub_198_; lean_object* v_toZSMul_199_; lean_object* v_toIntCast_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_207_; 
v_toSemiring_188_ = lean_ctor_get(v_inst_186_, 0);
v_toSemiring_189_ = lean_ctor_get(v_inst_187_, 0);
lean_inc_ref(v_toSemiring_189_);
lean_inc_ref(v_toSemiring_188_);
v___x_190_ = lp_mathlib_Prod_instSemiring___redArg(v_toSemiring_188_, v_toSemiring_189_);
v___x_191_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_186_);
v___x_192_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_187_);
v___x_193_ = lp_mathlib_Prod_subNegMonoid___redArg(v___x_191_, v___x_192_);
v___x_194_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_186_);
v___x_195_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_187_);
v___x_196_ = lp_mathlib_Prod_instAddGroupWithOne___redArg(v___x_194_, v___x_195_);
v_toNeg_197_ = lean_ctor_get(v___x_193_, 1);
lean_inc(v_toNeg_197_);
v_toSub_198_ = lean_ctor_get(v___x_193_, 2);
lean_inc(v_toSub_198_);
v_toZSMul_199_ = lean_ctor_get(v___x_193_, 3);
lean_inc(v_toZSMul_199_);
lean_dec_ref(v___x_193_);
v_toIntCast_200_ = lean_ctor_get(v___x_196_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_207_ == 0)
{
lean_object* v_unused_208_; lean_object* v_unused_209_; lean_object* v_unused_210_; lean_object* v_unused_211_; 
v_unused_208_ = lean_ctor_get(v___x_196_, 4);
lean_dec(v_unused_208_);
v_unused_209_ = lean_ctor_get(v___x_196_, 3);
lean_dec(v_unused_209_);
v_unused_210_ = lean_ctor_get(v___x_196_, 2);
lean_dec(v_unused_210_);
v_unused_211_ = lean_ctor_get(v___x_196_, 1);
lean_dec(v_unused_211_);
v___x_202_ = v___x_196_;
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_toIntCast_200_);
lean_dec(v___x_196_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 4, v_toIntCast_200_);
lean_ctor_set(v___x_202_, 3, v_toZSMul_199_);
lean_ctor_set(v___x_202_, 2, v_toSub_198_);
lean_ctor_set(v___x_202_, 1, v_toNeg_197_);
lean_ctor_set(v___x_202_, 0, v___x_190_);
v___x_205_ = v___x_202_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v_toNeg_197_);
lean_ctor_set(v_reuseFailAlloc_206_, 2, v_toSub_198_);
lean_ctor_set(v_reuseFailAlloc_206_, 3, v_toZSMul_199_);
lean_ctor_set(v_reuseFailAlloc_206_, 4, v_toIntCast_200_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRing(lean_object* v_R_212_, lean_object* v_S_213_, lean_object* v_inst_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib_Prod_instRing___redArg(v_inst_214_, v_inst_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommRing___redArg(lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(v_inst_217_, v_inst_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instNonUnitalCommRing(lean_object* v_R_220_, lean_object* v_S_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lp_mathlib_Prod_instNonUnitalNonAssocRing___redArg(v_inst_222_, v_inst_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommRing___redArg(lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_Prod_instRing___redArg(v_inst_225_, v_inst_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommRing(lean_object* v_R_228_, lean_object* v_S_229_, lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_Prod_instRing___redArg(v_inst_230_, v_inst_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst___lam__0(lean_object* v_self_233_){
_start:
{
lean_object* v_fst_234_; 
v_fst_234_ = lean_ctor_get(v_self_233_, 0);
lean_inc(v_fst_234_);
return v_fst_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst___lam__0___boxed(lean_object* v_self_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_NonUnitalRingHom_fst___lam__0(v_self_235_);
lean_dec_ref(v_self_235_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst(lean_object* v_R_238_, lean_object* v_S_239_, lean_object* v_inst_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v___f_242_; 
v___f_242_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fst___closed__0));
return v___f_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fst___boxed(lean_object* v_R_243_, lean_object* v_S_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_NonUnitalRingHom_fst(v_R_243_, v_S_244_, v_inst_245_, v_inst_246_);
lean_dec_ref(v_inst_246_);
lean_dec_ref(v_inst_245_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd___lam__0(lean_object* v_self_248_){
_start:
{
lean_object* v_snd_249_; 
v_snd_249_ = lean_ctor_get(v_self_248_, 1);
lean_inc(v_snd_249_);
return v_snd_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd___lam__0___boxed(lean_object* v_self_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_NonUnitalRingHom_snd___lam__0(v_self_250_);
lean_dec_ref(v_self_250_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd(lean_object* v_R_253_, lean_object* v_S_254_, lean_object* v_inst_255_, lean_object* v_inst_256_){
_start:
{
lean_object* v___f_257_; 
v___f_257_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_snd___closed__0));
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_snd___boxed(lean_object* v_R_258_, lean_object* v_S_259_, lean_object* v_inst_260_, lean_object* v_inst_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_NonUnitalRingHom_snd(v_R_258_, v_S_259_, v_inst_260_, v_inst_261_);
lean_dec_ref(v_inst_261_);
lean_dec_ref(v_inst_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0(lean_object* v_f_263_, lean_object* v_g_264_, lean_object* v_x_265_){
_start:
{
lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; 
lean_inc(v_x_265_);
v___x_266_ = lean_apply_1(v_f_263_, v_x_265_);
v___x_267_ = lean_apply_1(v_g_264_, v_x_265_);
v___x_268_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_266_);
lean_ctor_set(v___x_268_, 1, v___x_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod___redArg(lean_object* v_f_269_, lean_object* v_g_270_){
_start:
{
lean_object* v___f_271_; 
v___f_271_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_271_, 0, v_f_269_);
lean_closure_set(v___f_271_, 1, v_g_270_);
return v___f_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod(lean_object* v_R_272_, lean_object* v_S_273_, lean_object* v_T_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_f_278_, lean_object* v_g_279_){
_start:
{
lean_object* v___f_280_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_280_, 0, v_f_278_);
lean_closure_set(v___f_280_, 1, v_g_279_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prod___boxed(lean_object* v_R_281_, lean_object* v_S_282_, lean_object* v_T_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_f_287_, lean_object* v_g_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_NonUnitalRingHom_prod(v_R_281_, v_S_282_, v_T_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_f_287_, v_g_288_);
lean_dec_ref(v_inst_286_);
lean_dec_ref(v_inst_285_);
lean_dec_ref(v_inst_284_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prodMap___redArg(lean_object* v_f_290_, lean_object* v_g_291_){
_start:
{
lean_object* v___f_292_; lean_object* v___f_293_; lean_object* v___f_294_; lean_object* v___f_295_; lean_object* v___f_296_; 
v___f_292_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fst___closed__0));
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_293_, 0, v___f_292_);
lean_closure_set(v___f_293_, 1, v_f_290_);
v___f_294_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_snd___closed__0));
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_295_, 0, v___f_294_);
lean_closure_set(v___f_295_, 1, v_g_291_);
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_296_, 0, v___f_293_);
lean_closure_set(v___f_296_, 1, v___f_295_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prodMap(lean_object* v_R_297_, lean_object* v_R_x27_298_, lean_object* v_S_299_, lean_object* v_S_x27_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_f_305_, lean_object* v_g_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_NonUnitalRingHom_prodMap___redArg(v_f_305_, v_g_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_prodMap___boxed(lean_object* v_R_308_, lean_object* v_R_x27_309_, lean_object* v_S_310_, lean_object* v_S_x27_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_f_316_, lean_object* v_g_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_mathlib_NonUnitalRingHom_prodMap(v_R_308_, v_R_x27_309_, v_S_310_, v_S_x27_311_, v_inst_312_, v_inst_313_, v_inst_314_, v_inst_315_, v_f_316_, v_g_317_);
lean_dec_ref(v_inst_315_);
lean_dec_ref(v_inst_314_);
lean_dec_ref(v_inst_313_);
lean_dec_ref(v_inst_312_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst(lean_object* v_R_319_, lean_object* v_S_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v___f_323_; 
v___f_323_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fst___closed__0));
return v___f_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fst___boxed(lean_object* v_R_324_, lean_object* v_S_325_, lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_RingHom_fst(v_R_324_, v_S_325_, v_inst_326_, v_inst_327_);
lean_dec_ref(v_inst_327_);
lean_dec_ref(v_inst_326_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd(lean_object* v_R_329_, lean_object* v_S_330_, lean_object* v_inst_331_, lean_object* v_inst_332_){
_start:
{
lean_object* v___f_333_; 
v___f_333_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_snd___closed__0));
return v___f_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_snd___boxed(lean_object* v_R_334_, lean_object* v_S_335_, lean_object* v_inst_336_, lean_object* v_inst_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_RingHom_snd(v_R_334_, v_S_335_, v_inst_336_, v_inst_337_);
lean_dec_ref(v_inst_337_);
lean_dec_ref(v_inst_336_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prod___redArg(lean_object* v_f_339_, lean_object* v_g_340_){
_start:
{
lean_object* v___f_341_; 
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_341_, 0, v_f_339_);
lean_closure_set(v___f_341_, 1, v_g_340_);
return v___f_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prod(lean_object* v_R_342_, lean_object* v_S_343_, lean_object* v_T_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_f_348_, lean_object* v_g_349_){
_start:
{
lean_object* v___f_350_; 
v___f_350_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_350_, 0, v_f_348_);
lean_closure_set(v___f_350_, 1, v_g_349_);
return v___f_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prod___boxed(lean_object* v_R_351_, lean_object* v_S_352_, lean_object* v_T_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_f_357_, lean_object* v_g_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_RingHom_prod(v_R_351_, v_S_352_, v_T_353_, v_inst_354_, v_inst_355_, v_inst_356_, v_f_357_, v_g_358_);
lean_dec_ref(v_inst_356_);
lean_dec_ref(v_inst_355_);
lean_dec_ref(v_inst_354_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prodMap___redArg(lean_object* v_f_360_, lean_object* v_g_361_){
_start:
{
lean_object* v___f_362_; lean_object* v___f_363_; lean_object* v___f_364_; lean_object* v___f_365_; lean_object* v___f_366_; 
v___f_362_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fst___closed__0));
v___f_363_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_363_, 0, v___f_362_);
lean_closure_set(v___f_363_, 1, v_f_360_);
v___f_364_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_snd___closed__0));
v___f_365_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_365_, 0, v___f_364_);
lean_closure_set(v___f_365_, 1, v_g_361_);
v___f_366_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_366_, 0, v___f_363_);
lean_closure_set(v___f_366_, 1, v___f_365_);
return v___f_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prodMap(lean_object* v_R_367_, lean_object* v_R_x27_368_, lean_object* v_S_369_, lean_object* v_S_x27_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_f_375_, lean_object* v_g_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_RingHom_prodMap___redArg(v_f_375_, v_g_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_prodMap___boxed(lean_object* v_R_378_, lean_object* v_R_x27_379_, lean_object* v_S_380_, lean_object* v_S_x27_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_f_386_, lean_object* v_g_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_mathlib_RingHom_prodMap(v_R_378_, v_R_x27_379_, v_S_380_, v_S_x27_381_, v_inst_382_, v_inst_383_, v_inst_384_, v_inst_385_, v_f_386_, v_g_387_);
lean_dec_ref(v_inst_385_);
lean_dec_ref(v_inst_384_);
lean_dec_ref(v_inst_383_);
lean_dec_ref(v_inst_382_);
return v_res_388_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_prodComm___closed__0(void){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodComm(lean_object* v_R_390_, lean_object* v_S_391_, lean_object* v_inst_392_, lean_object* v_inst_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lean_obj_once(&lp_mathlib_RingEquiv_prodComm___closed__0, &lp_mathlib_RingEquiv_prodComm___closed__0_once, _init_lp_mathlib_RingEquiv_prodComm___closed__0);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodComm___boxed(lean_object* v_R_395_, lean_object* v_S_396_, lean_object* v_inst_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_mathlib_RingEquiv_prodComm(v_R_395_, v_S_396_, v_inst_397_, v_inst_398_);
lean_dec_ref(v_inst_398_);
lean_dec_ref(v_inst_397_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm___lam__0(lean_object* v_rrss_400_){
_start:
{
lean_object* v_fst_401_; lean_object* v_snd_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_427_; 
v_fst_401_ = lean_ctor_get(v_rrss_400_, 0);
v_snd_402_ = lean_ctor_get(v_rrss_400_, 1);
v_isSharedCheck_427_ = !lean_is_exclusive(v_rrss_400_);
if (v_isSharedCheck_427_ == 0)
{
v___x_404_ = v_rrss_400_;
v_isShared_405_ = v_isSharedCheck_427_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_snd_402_);
lean_inc(v_fst_401_);
lean_dec(v_rrss_400_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_427_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v_fst_406_; lean_object* v_snd_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_426_; 
v_fst_406_ = lean_ctor_get(v_fst_401_, 0);
v_snd_407_ = lean_ctor_get(v_fst_401_, 1);
v_isSharedCheck_426_ = !lean_is_exclusive(v_fst_401_);
if (v_isSharedCheck_426_ == 0)
{
v___x_409_ = v_fst_401_;
v_isShared_410_ = v_isSharedCheck_426_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_snd_407_);
lean_inc(v_fst_406_);
lean_dec(v_fst_401_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_426_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v_fst_411_; lean_object* v_snd_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_425_; 
v_fst_411_ = lean_ctor_get(v_snd_402_, 0);
v_snd_412_ = lean_ctor_get(v_snd_402_, 1);
v_isSharedCheck_425_ = !lean_is_exclusive(v_snd_402_);
if (v_isSharedCheck_425_ == 0)
{
v___x_414_ = v_snd_402_;
v_isShared_415_ = v_isSharedCheck_425_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_snd_412_);
lean_inc(v_fst_411_);
lean_dec(v_snd_402_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_425_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_417_; 
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 1, v_fst_411_);
lean_ctor_set(v___x_414_, 0, v_fst_406_);
v___x_417_ = v___x_414_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_424_; 
v_reuseFailAlloc_424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_424_, 0, v_fst_406_);
lean_ctor_set(v_reuseFailAlloc_424_, 1, v_fst_411_);
v___x_417_ = v_reuseFailAlloc_424_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
lean_object* v___x_419_; 
if (v_isShared_410_ == 0)
{
lean_ctor_set(v___x_409_, 1, v_snd_412_);
lean_ctor_set(v___x_409_, 0, v_snd_407_);
v___x_419_ = v___x_409_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_snd_407_);
lean_ctor_set(v_reuseFailAlloc_423_, 1, v_snd_412_);
v___x_419_ = v_reuseFailAlloc_423_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
lean_object* v___x_421_; 
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 1, v___x_419_);
lean_ctor_set(v___x_404_, 0, v___x_417_);
v___x_421_ = v___x_404_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_417_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v___x_419_);
v___x_421_ = v_reuseFailAlloc_422_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
return v___x_421_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm___lam__1(lean_object* v_rsrs_428_){
_start:
{
lean_object* v_fst_429_; lean_object* v_snd_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_455_; 
v_fst_429_ = lean_ctor_get(v_rsrs_428_, 0);
v_snd_430_ = lean_ctor_get(v_rsrs_428_, 1);
v_isSharedCheck_455_ = !lean_is_exclusive(v_rsrs_428_);
if (v_isSharedCheck_455_ == 0)
{
v___x_432_ = v_rsrs_428_;
v_isShared_433_ = v_isSharedCheck_455_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_snd_430_);
lean_inc(v_fst_429_);
lean_dec(v_rsrs_428_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_455_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v_fst_434_; lean_object* v_snd_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_454_; 
v_fst_434_ = lean_ctor_get(v_fst_429_, 0);
v_snd_435_ = lean_ctor_get(v_fst_429_, 1);
v_isSharedCheck_454_ = !lean_is_exclusive(v_fst_429_);
if (v_isSharedCheck_454_ == 0)
{
v___x_437_ = v_fst_429_;
v_isShared_438_ = v_isSharedCheck_454_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_snd_435_);
lean_inc(v_fst_434_);
lean_dec(v_fst_429_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_454_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v_fst_439_; lean_object* v_snd_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_453_; 
v_fst_439_ = lean_ctor_get(v_snd_430_, 0);
v_snd_440_ = lean_ctor_get(v_snd_430_, 1);
v_isSharedCheck_453_ = !lean_is_exclusive(v_snd_430_);
if (v_isSharedCheck_453_ == 0)
{
v___x_442_ = v_snd_430_;
v_isShared_443_ = v_isSharedCheck_453_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_snd_440_);
lean_inc(v_fst_439_);
lean_dec(v_snd_430_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_453_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
lean_ctor_set(v___x_442_, 1, v_fst_439_);
lean_ctor_set(v___x_442_, 0, v_fst_434_);
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_452_; 
v_reuseFailAlloc_452_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_452_, 0, v_fst_434_);
lean_ctor_set(v_reuseFailAlloc_452_, 1, v_fst_439_);
v___x_445_ = v_reuseFailAlloc_452_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
lean_object* v___x_447_; 
if (v_isShared_438_ == 0)
{
lean_ctor_set(v___x_437_, 1, v_snd_440_);
lean_ctor_set(v___x_437_, 0, v_snd_435_);
v___x_447_ = v___x_437_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_snd_435_);
lean_ctor_set(v_reuseFailAlloc_451_, 1, v_snd_440_);
v___x_447_ = v_reuseFailAlloc_451_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
lean_object* v___x_449_; 
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 1, v___x_447_);
lean_ctor_set(v___x_432_, 0, v___x_445_);
v___x_449_ = v___x_432_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v___x_445_);
lean_ctor_set(v_reuseFailAlloc_450_, 1, v___x_447_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm(lean_object* v_R_461_, lean_object* v_R_x27_462_, lean_object* v_S_463_, lean_object* v_S_x27_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = ((lean_object*)(lp_mathlib_RingEquiv_prodProdProdComm___closed__2));
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodProdProdComm___boxed(lean_object* v_R_470_, lean_object* v_R_x27_471_, lean_object* v_S_472_, lean_object* v_S_x27_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_RingEquiv_prodProdProdComm(v_R_470_, v_R_x27_471_, v_S_472_, v_S_x27_473_, v_inst_474_, v_inst_475_, v_inst_476_, v_inst_477_);
lean_dec_ref(v_inst_477_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
lean_dec_ref(v_inst_474_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing___redArg___lam__1(lean_object* v_toZero_479_, lean_object* v_x_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_481_, 0, v_x_480_);
lean_ctor_set(v___x_481_, 1, v_toZero_479_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing___redArg(lean_object* v_inst_482_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_483_; lean_object* v___x_484_; lean_object* v_toZero_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_494_; 
v_toNonUnitalNonAssocSemiring_483_ = lean_ctor_get(v_inst_482_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_483_);
lean_dec_ref(v_inst_482_);
v___x_484_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_toNonUnitalNonAssocSemiring_483_);
v_toZero_485_ = lean_ctor_get(v___x_484_, 1);
v_isSharedCheck_494_ = !lean_is_exclusive(v___x_484_);
if (v_isSharedCheck_494_ == 0)
{
lean_object* v_unused_495_; 
v_unused_495_ = lean_ctor_get(v___x_484_, 0);
lean_dec(v_unused_495_);
v___x_487_ = v___x_484_;
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_toZero_485_);
lean_dec(v___x_484_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_494_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___f_489_; lean_object* v___f_490_; lean_object* v___x_492_; 
v___f_489_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_fst___closed__0));
v___f_490_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_prodZeroRing___redArg___lam__1), 2, 1);
lean_closure_set(v___f_490_, 0, v_toZero_485_);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 1, v___f_489_);
lean_ctor_set(v___x_487_, 0, v___f_490_);
v___x_492_ = v___x_487_;
goto v_reusejp_491_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v___f_490_);
lean_ctor_set(v_reuseFailAlloc_493_, 1, v___f_489_);
v___x_492_ = v_reuseFailAlloc_493_;
goto v_reusejp_491_;
}
v_reusejp_491_:
{
return v___x_492_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing(lean_object* v_R_496_, lean_object* v_S_497_, lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_inst_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lp_mathlib_RingEquiv_prodZeroRing___redArg(v_inst_499_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodZeroRing___boxed(lean_object* v_R_502_, lean_object* v_S_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_RingEquiv_prodZeroRing(v_R_502_, v_S_503_, v_inst_504_, v_inst_505_, v_inst_506_);
lean_dec_ref(v_inst_504_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd___redArg___lam__1(lean_object* v_toZero_508_, lean_object* v_x_509_){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_510_, 0, v_toZero_508_);
lean_ctor_set(v___x_510_, 1, v_x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd___redArg(lean_object* v_inst_511_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_512_; lean_object* v___x_513_; lean_object* v_toZero_514_; lean_object* v___x_516_; uint8_t v_isShared_517_; uint8_t v_isSharedCheck_523_; 
v_toNonUnitalNonAssocSemiring_512_ = lean_ctor_get(v_inst_511_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_512_);
lean_dec_ref(v_inst_511_);
v___x_513_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_toNonUnitalNonAssocSemiring_512_);
v_toZero_514_ = lean_ctor_get(v___x_513_, 1);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_523_ == 0)
{
lean_object* v_unused_524_; 
v_unused_524_ = lean_ctor_get(v___x_513_, 0);
lean_dec(v_unused_524_);
v___x_516_ = v___x_513_;
v_isShared_517_ = v_isSharedCheck_523_;
goto v_resetjp_515_;
}
else
{
lean_inc(v_toZero_514_);
lean_dec(v___x_513_);
v___x_516_ = lean_box(0);
v_isShared_517_ = v_isSharedCheck_523_;
goto v_resetjp_515_;
}
v_resetjp_515_:
{
lean_object* v___f_518_; lean_object* v___f_519_; lean_object* v___x_521_; 
v___f_518_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_snd___closed__0));
v___f_519_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_zeroRingProd___redArg___lam__1), 2, 1);
lean_closure_set(v___f_519_, 0, v_toZero_514_);
if (v_isShared_517_ == 0)
{
lean_ctor_set(v___x_516_, 1, v___f_518_);
lean_ctor_set(v___x_516_, 0, v___f_519_);
v___x_521_ = v___x_516_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___f_519_);
lean_ctor_set(v_reuseFailAlloc_522_, 1, v___f_518_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd(lean_object* v_R_525_, lean_object* v_S_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_inst_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lp_mathlib_RingEquiv_zeroRingProd___redArg(v_inst_528_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_zeroRingProd___boxed(lean_object* v_R_531_, lean_object* v_S_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_){
_start:
{
lean_object* v_res_536_; 
v_res_536_ = lp_mathlib_RingEquiv_zeroRingProd(v_R_531_, v_S_532_, v_inst_533_, v_inst_534_, v_inst_535_);
lean_dec_ref(v_inst_533_);
return v_res_536_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
