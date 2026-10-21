// Lean compiler output
// Module: Mathlib.RingTheory.NonUnitalSubsemiring.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Membership public import Mathlib.Algebra.Group.Subsemigroup.Membership public import Mathlib.Algebra.Group.Subsemigroup.Operations public import Mathlib.Algebra.GroupWithZero.Center public import Mathlib.Algebra.Ring.Center public import Mathlib.Algebra.Ring.Centralizer public import Mathlib.Algebra.Ring.Opposite public import Mathlib.Algebra.Ring.Prod public import Mathlib.Algebra.Ring.Submonoid.Basic public import Mathlib.Data.Set.Finite.Range public import Mathlib.GroupTheory.Submonoid.Center public import Mathlib.GroupTheory.Subsemigroup.Centralizer public import Mathlib.RingTheory.NonUnitalSubsemiring.Defs
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_MulMemClass_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemigroup_topEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center_instNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center_instNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__1 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__0_value),((lean_object*)&lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__2 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubsemiring_decidableMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closureNonUnitalCommSemiringOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closureNonUnitalCommSemiringOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_nonUnitalSubsemiringClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_nonUnitalSubsemiringClosure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_gi___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_NonUnitalSubsemiring_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_NonUnitalSubsemiring_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topEquiv___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v_toMul_3_; lean_object* v___x_4_; 
v___x_2_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_1_);
v_toMul_3_ = lean_ctor_get(v___x_2_, 0);
lean_inc(v_toMul_3_);
lean_dec_ref(v___x_2_);
v___x_4_ = lp_mathlib_Subsemigroup_topEquiv(lean_box(0), v_toMul_3_);
lean_dec(v_toMul_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_topEquiv(lean_object* v_R_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_NonUnitalSubsemiring_topEquiv___redArg(v_inst_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_comap(lean_object* v_R_8_, lean_object* v_S_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_f_12_, lean_object* v_s_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_box(0);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_comap___boxed(lean_object* v_R_15_, lean_object* v_S_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_f_19_, lean_object* v_s_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_NonUnitalSubsemiring_comap(v_R_15_, v_S_16_, v_inst_17_, v_inst_18_, v_f_19_, v_s_20_);
lean_dec(v_f_19_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_map(lean_object* v_R_22_, lean_object* v_S_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_f_26_, lean_object* v_s_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_map___boxed(lean_object* v_R_29_, lean_object* v_S_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_f_33_, lean_object* v_s_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_NonUnitalSubsemiring_map(v_R_29_, v_S_30_, v_inst_31_, v_inst_32_, v_f_33_, v_s_34_);
lean_dec(v_f_33_);
lean_dec_ref(v_inst_32_);
lean_dec_ref(v_inst_31_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srange(lean_object* v_R_36_, lean_object* v_S_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_f_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_box(0);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srange___boxed(lean_object* v_R_42_, lean_object* v_S_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_f_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_NonUnitalRingHom_srange(v_R_42_, v_S_43_, v_inst_44_, v_inst_45_, v_f_46_);
lean_dec(v_f_46_);
lean_dec_ref(v_inst_45_);
lean_dec_ref(v_inst_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet___lam__0(lean_object* v_s_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet(lean_object* v_R_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___f_53_; 
v___f_53_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_instInfSet___closed__0));
return v___f_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInfSet___boxed(lean_object* v_R_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_NonUnitalSubsemiring_instInfSet(v_R_54_, v_inst_55_);
lean_dec_ref(v_inst_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___lam__0(lean_object* v_x1_57_, lean_object* v_x2_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_box(0);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg(lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___f_65_; lean_object* v___x_66_; lean_object* v_toLattice_67_; lean_object* v_toSupSet_68_; lean_object* v_toInfSet_69_; lean_object* v___x_71_; uint8_t v_isShared_72_; uint8_t v_isSharedCheck_87_; 
v___x_64_ = lp_mathlib_NonUnitalSubsemiring_instPartialOrder(lean_box(0), v_inst_63_);
v___f_65_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_instInfSet___closed__0));
v___x_66_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_64_, v___f_65_);
v_toLattice_67_ = lean_ctor_get(v___x_66_, 0);
v_toSupSet_68_ = lean_ctor_get(v___x_66_, 1);
v_toInfSet_69_ = lean_ctor_get(v___x_66_, 2);
v_isSharedCheck_87_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_87_ == 0)
{
lean_object* v_unused_88_; 
v_unused_88_ = lean_ctor_get(v___x_66_, 3);
lean_dec(v_unused_88_);
v___x_71_ = v___x_66_;
v_isShared_72_ = v_isSharedCheck_87_;
goto v_resetjp_70_;
}
else
{
lean_inc(v_toInfSet_69_);
lean_inc(v_toSupSet_68_);
lean_inc(v_toLattice_67_);
lean_dec(v___x_66_);
v___x_71_ = lean_box(0);
v_isShared_72_ = v_isSharedCheck_87_;
goto v_resetjp_70_;
}
v_resetjp_70_:
{
lean_object* v_toSemilatticeSup_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_85_; 
v_toSemilatticeSup_73_ = lean_ctor_get(v_toLattice_67_, 0);
v_isSharedCheck_85_ = !lean_is_exclusive(v_toLattice_67_);
if (v_isSharedCheck_85_ == 0)
{
lean_object* v_unused_86_; 
v_unused_86_ = lean_ctor_get(v_toLattice_67_, 1);
lean_dec(v_unused_86_);
v___x_75_ = v_toLattice_67_;
v_isShared_76_ = v_isSharedCheck_85_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_toSemilatticeSup_73_);
lean_dec(v_toLattice_67_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_85_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___f_77_; lean_object* v___x_79_; 
v___f_77_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__0));
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 1, v___f_77_);
v___x_79_ = v___x_75_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v_toSemilatticeSup_73_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v___f_77_);
v___x_79_ = v_reuseFailAlloc_84_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
lean_object* v___x_80_; lean_object* v___x_82_; 
v___x_80_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___closed__1));
if (v_isShared_72_ == 0)
{
lean_ctor_set(v___x_71_, 3, v___x_80_);
lean_ctor_set(v___x_71_, 0, v___x_79_);
v___x_82_ = v___x_71_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v___x_79_);
lean_ctor_set(v_reuseFailAlloc_83_, 1, v_toSupSet_68_);
lean_ctor_set(v_reuseFailAlloc_83_, 2, v_toInfSet_69_);
lean_ctor_set(v_reuseFailAlloc_83_, 3, v___x_80_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg___boxed(lean_object* v_inst_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg(v_inst_89_);
lean_dec_ref(v_inst_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice(lean_object* v_R_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___redArg(v_inst_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instCompleteLattice___boxed(lean_object* v_R_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_NonUnitalSubsemiring_instCompleteLattice(v_R_94_, v_inst_95_);
lean_dec_ref(v_inst_95_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center(lean_object* v_R_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lean_box(0);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center___boxed(lean_object* v_R_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_NonUnitalSubsemiring_center(v_R_100_, v_inst_101_);
lean_dec_ref(v_inst_101_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center_instNonUnitalCommSemiring___redArg(lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; lean_object* v_toMul_105_; lean_object* v___x_106_; lean_object* v_toAddCommMonoid_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_115_; 
lean_inc_ref(v_inst_103_);
v___x_104_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_103_);
v_toMul_105_ = lean_ctor_get(v___x_104_, 0);
lean_inc(v_toMul_105_);
lean_dec_ref(v___x_104_);
v___x_106_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_103_);
v_toAddCommMonoid_107_ = lean_ctor_get(v___x_106_, 0);
v_isSharedCheck_115_ = !lean_is_exclusive(v___x_106_);
if (v_isSharedCheck_115_ == 0)
{
lean_object* v_unused_116_; 
v_unused_116_ = lean_ctor_get(v___x_106_, 1);
lean_dec(v_unused_116_);
v___x_109_ = v___x_106_;
v_isShared_110_ = v_isSharedCheck_115_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_toAddCommMonoid_107_);
lean_dec(v___x_106_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_115_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___f_111_; lean_object* v___x_113_; 
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_111_, 0, v_toMul_105_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 1, v___f_111_);
v___x_113_ = v___x_109_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v_toAddCommMonoid_107_);
lean_ctor_set(v_reuseFailAlloc_114_, 1, v___f_111_);
v___x_113_ = v_reuseFailAlloc_114_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
return v___x_113_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_center_instNonUnitalCommSemiring(lean_object* v_R_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_NonUnitalSubsemiring_center_instNonUnitalCommSemiring___redArg(v_inst_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___lam__0(lean_object* v_f_120_, lean_object* v___y_121_){
_start:
{
lean_object* v_toFun_122_; lean_object* v___x_123_; 
v_toFun_122_ = lean_ctor_get(v_f_120_, 0);
lean_inc(v_toFun_122_);
lean_dec_ref(v_f_120_);
v___x_123_ = lean_apply_1(v_toFun_122_, v___y_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___lam__1(lean_object* v_f_124_, lean_object* v___y_125_){
_start:
{
lean_object* v_invFun_126_; lean_object* v___x_127_; 
v_invFun_126_ = lean_ctor_get(v_f_124_, 1);
lean_inc(v_invFun_126_);
lean_dec_ref(v_f_124_);
v___x_127_ = lean_apply_1(v_invFun_126_, v___y_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(lean_object* v_e_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_134_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg___closed__2));
v___x_135_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_134_, v_e_133_);
v___x_136_ = lp_mathlib_Subsemigroup_centerCongr___redArg(v___x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr(lean_object* v_R_137_, lean_object* v_S_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_e_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___boxed(lean_object* v_R_143_, lean_object* v_S_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_e_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_NonUnitalSubsemiring_centerCongr(v_R_143_, v_S_144_, v_inst_145_, v_inst_146_, v_e_147_);
lean_dec_ref(v_inst_146_);
lean_dec_ref(v_inst_145_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(lean_object* v_inst_149_){
_start:
{
lean_object* v___x_150_; lean_object* v_toMul_151_; lean_object* v___x_152_; 
v___x_150_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_149_);
v_toMul_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc(v_toMul_151_);
lean_dec_ref(v___x_150_);
v___x_152_ = lp_mathlib_Subsemigroup_centerToMulOpposite(lean_box(0), v_toMul_151_);
lean_dec(v_toMul_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite(lean_object* v_R_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(v_inst_154_);
return v___x_155_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___lam__0(lean_object* v_toMul_156_, lean_object* v_x_157_, lean_object* v_inst_158_, lean_object* v_a_159_){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; uint8_t v___x_163_; 
lean_inc(v_toMul_156_);
lean_inc(v_x_157_);
lean_inc(v_a_159_);
v___x_160_ = lean_apply_2(v_toMul_156_, v_a_159_, v_x_157_);
v___x_161_ = lean_apply_2(v_toMul_156_, v_x_157_, v_a_159_);
v___x_162_ = lean_apply_2(v_inst_158_, v___x_160_, v___x_161_);
v___x_163_ = lean_unbox(v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___lam__0___boxed(lean_object* v_toMul_164_, lean_object* v_x_165_, lean_object* v_inst_166_, lean_object* v_a_167_){
_start:
{
uint8_t v_res_168_; lean_object* v_r_169_; 
v_res_168_ = lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___lam__0(v_toMul_164_, v_x_165_, v_inst_166_, v_a_167_);
v_r_169_ = lean_box(v_res_168_);
return v_r_169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg(lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_x_173_){
_start:
{
lean_object* v___x_174_; lean_object* v_toMul_175_; lean_object* v___f_176_; uint8_t v___x_177_; 
v___x_174_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_170_);
v_toMul_175_ = lean_ctor_get(v___x_174_, 0);
lean_inc(v_toMul_175_);
lean_dec_ref(v___x_174_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_176_, 0, v_toMul_175_);
lean_closure_set(v___f_176_, 1, v_x_173_);
lean_closure_set(v___f_176_, 2, v_inst_171_);
v___x_177_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_176_, v_inst_172_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg___boxed(lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_x_181_){
_start:
{
uint8_t v_res_182_; lean_object* v_r_183_; 
v_res_182_ = lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg(v_inst_178_, v_inst_179_, v_inst_180_, v_x_181_);
v_r_183_ = lean_box(v_res_182_);
return v_r_183_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubsemiring_decidableMemCenter(lean_object* v_R_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_x_188_){
_start:
{
uint8_t v___x_189_; 
v___x_189_ = lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___redArg(v_inst_185_, v_inst_186_, v_inst_187_, v_x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_decidableMemCenter___boxed(lean_object* v_R_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_x_194_){
_start:
{
uint8_t v_res_195_; lean_object* v_r_196_; 
v_res_195_ = lp_mathlib_NonUnitalSubsemiring_decidableMemCenter(v_R_190_, v_inst_191_, v_inst_192_, v_inst_193_, v_x_194_);
v_r_196_ = lean_box(v_res_195_);
return v_r_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centralizer(lean_object* v_R_197_, lean_object* v_inst_198_, lean_object* v_s_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lean_box(0);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_centralizer___boxed(lean_object* v_R_201_, lean_object* v_inst_202_, lean_object* v_s_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_NonUnitalSubsemiring_centralizer(v_R_201_, v_inst_202_, v_s_203_);
lean_dec_ref(v_inst_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closure(lean_object* v_R_205_, lean_object* v_inst_206_, lean_object* v_s_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lean_box(0);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closure___boxed(lean_object* v_R_209_, lean_object* v_inst_210_, lean_object* v_s_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_NonUnitalSubsemiring_closure(v_R_209_, v_inst_210_, v_s_211_);
lean_dec_ref(v_inst_210_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closureNonUnitalCommSemiringOfComm___redArg(lean_object* v_inst_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_closureNonUnitalCommSemiringOfComm(lean_object* v_R_215_, lean_object* v_inst_216_, lean_object* v_s_217_, lean_object* v_hcomm_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_216_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_nonUnitalSubsemiringClosure(lean_object* v_R_220_, lean_object* v_inst_221_, lean_object* v_M_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_box(0);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_nonUnitalSubsemiringClosure___boxed(lean_object* v_R_224_, lean_object* v_inst_225_, lean_object* v_M_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_Subsemigroup_nonUnitalSubsemiringClosure(v_R_224_, v_inst_225_, v_M_226_);
lean_dec_ref(v_inst_225_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_gi___lam__0(lean_object* v_s_228_, lean_object* v_x_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lean_box(0);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_gi(lean_object* v_R_232_, lean_object* v_inst_233_){
_start:
{
lean_object* v___f_234_; 
v___f_234_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_gi___closed__0));
return v___f_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_gi___boxed(lean_object* v_R_235_, lean_object* v_inst_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_NonUnitalSubsemiring_gi(v_R_235_, v_inst_236_);
lean_dec_ref(v_inst_236_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prod(lean_object* v_R_238_, lean_object* v_S_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_s_242_, lean_object* v_t_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_box(0);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prod___boxed(lean_object* v_R_245_, lean_object* v_S_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_s_249_, lean_object* v_t_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_NonUnitalSubsemiring_prod(v_R_245_, v_S_246_, v_inst_247_, v_inst_248_, v_s_249_, v_t_250_);
lean_dec_ref(v_inst_248_);
lean_dec_ref(v_inst_247_);
return v_res_251_;
}
}
static lean_object* _init_lp_mathlib_NonUnitalSubsemiring_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prodEquiv(lean_object* v_R_253_, lean_object* v_S_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_s_257_, lean_object* v_t_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lean_obj_once(&lp_mathlib_NonUnitalSubsemiring_prodEquiv___closed__0, &lp_mathlib_NonUnitalSubsemiring_prodEquiv___closed__0_once, _init_lp_mathlib_NonUnitalSubsemiring_prodEquiv___closed__0);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_prodEquiv___boxed(lean_object* v_R_260_, lean_object* v_S_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_s_264_, lean_object* v_t_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_NonUnitalSubsemiring_prodEquiv(v_R_260_, v_S_261_, v_inst_262_, v_inst_263_, v_s_264_, v_t_265_);
lean_dec_ref(v_inst_263_);
lean_dec_ref(v_inst_262_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___lam__0(lean_object* v_f_267_, lean_object* v___y_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lean_apply_1(v_f_267_, v___y_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg(lean_object* v_f_271_){
_start:
{
lean_object* v___f_272_; lean_object* v___f_273_; 
v___f_272_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg___closed__0));
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0), 3, 2);
lean_closure_set(v___f_273_, 0, v___f_272_);
lean_closure_set(v___f_273_, 1, v_f_271_);
return v___f_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict(lean_object* v_R_274_, lean_object* v_S_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_f_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg(v_f_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_srangeRestrict___boxed(lean_object* v_R_280_, lean_object* v_S_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_f_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_NonUnitalRingHom_srangeRestrict(v_R_280_, v_S_281_, v_inst_282_, v_inst_283_, v_f_284_);
lean_dec_ref(v_inst_283_);
lean_dec_ref(v_inst_282_);
return v_res_285_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___closed__0(void){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr(lean_object* v_R_287_, lean_object* v_inst_288_, lean_object* v_s_289_, lean_object* v_t_290_, lean_object* v_h_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lean_obj_once(&lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___closed__0, &lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___closed__0_once, _init_lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___closed__0);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr___boxed(lean_object* v_R_293_, lean_object* v_inst_294_, lean_object* v_s_295_, lean_object* v_t_296_, lean_object* v_h_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_RingEquiv_nonUnitalSubsemiringCongr(v_R_293_, v_inst_294_, v_s_295_, v_t_296_, v_h_297_);
lean_dec_ref(v_inst_294_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__0(lean_object* v_f_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___x_54__overap_301_; lean_object* v___x_302_; 
v___x_54__overap_301_ = lp_mathlib_NonUnitalRingHom_srangeRestrict___redArg(v_f_299_);
v___x_302_ = lean_apply_1(v___x_54__overap_301_, v___y_300_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__1(lean_object* v_g_303_, lean_object* v_x_304_){
_start:
{
lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_305_ = lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0(v_x_304_);
v___x_306_ = lean_apply_1(v_g_303_, v___x_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__1___boxed(lean_object* v_g_307_, lean_object* v_x_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__1(v_g_307_, v_x_308_);
lean_dec(v_x_308_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg(lean_object* v_g_310_, lean_object* v_f_311_){
_start:
{
lean_object* v___f_312_; lean_object* v___f_313_; lean_object* v___x_314_; 
v___f_312_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_312_, 0, v_f_311_);
v___f_313_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_313_, 0, v_g_310_);
v___x_314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_314_, 0, v___f_312_);
lean_ctor_set(v___x_314_, 1, v___f_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27(lean_object* v_R_315_, lean_object* v_S_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_g_319_, lean_object* v_f_320_, lean_object* v_h_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_RingEquiv_sofLeftInverse_x27___redArg(v_g_319_, v_f_320_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sofLeftInverse_x27___boxed(lean_object* v_R_323_, lean_object* v_S_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_g_327_, lean_object* v_f_328_, lean_object* v_h_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_RingEquiv_sofLeftInverse_x27(v_R_323_, v_S_324_, v_inst_325_, v_inst_326_, v_g_327_, v_f_328_, v_h_329_);
lean_dec_ref(v_inst_326_);
lean_dec_ref(v_inst_325_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringMap___redArg(lean_object* v_e_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringMap(lean_object* v_R_333_, lean_object* v_S_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_e_337_, lean_object* v_s_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_337_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubsemiringMap___boxed(lean_object* v_R_340_, lean_object* v_S_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_e_344_, lean_object* v_s_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_mathlib_RingEquiv_nonUnitalSubsemiringMap(v_R_340_, v_S_341_, v_inst_342_, v_inst_343_, v_e_344_, v_s_345_);
lean_dec_ref(v_inst_343_);
lean_dec_ref(v_inst_342_);
return v_res_346_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Membership(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Center(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Center(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Centralizer(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Membership(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Centralizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Submonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
