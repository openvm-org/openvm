// Lean compiler output
// Module: Mathlib.GroupTheory.Submonoid.Center
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.GroupTheory.Subsemigroup.Center
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_AddUnits_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCenter___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCenter___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCenter(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCenter___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCenter___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCenter(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_unitsCenterToCenterUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubmonoidClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsCenterToCenterUnits___closed__0 = (const lean_object*)&lp_mathlib_unitsCenterToCenterUnits___closed__0_value;
static const lean_closure_object lp_mathlib_unitsCenterToCenterUnits___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_unitsCenterToCenterUnits___closed__0_value)} };
static const lean_object* lp_mathlib_unitsCenterToCenterUnits___closed__1 = (const lean_object*)&lp_mathlib_unitsCenterToCenterUnits___closed__1_value;
static const lean_closure_object lp_mathlib_unitsCenterToCenterUnits___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_unitsCenterToCenterUnits___closed__1_value)} };
static const lean_object* lp_mathlib_unitsCenterToCenterUnits___closed__2 = (const lean_object*)&lp_mathlib_unitsCenterToCenterUnits___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_unitsCenterToCenterUnits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCenterToCenterUnits___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_addUnitsCenterToCenterAddUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddUnits_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_unitsCenterToCenterUnits___closed__0_value)} };
static const lean_object* lp_mathlib_addUnitsCenterToCenterAddUnits___closed__0 = (const lean_object*)&lp_mathlib_addUnitsCenterToCenterAddUnits___closed__0_value;
static const lean_closure_object lp_mathlib_addUnitsCenterToCenterAddUnits___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_addUnitsCenterToCenterAddUnits___closed__0_value)} };
static const lean_object* lp_mathlib_addUnitsCenterToCenterAddUnits___closed__1 = (const lean_object*)&lp_mathlib_addUnitsCenterToCenterAddUnits___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_addUnitsCenterToCenterAddUnits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addUnitsCenterToCenterAddUnits___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subsemigroup_centerToMulOpposite___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subsemigroup_centerToMulOpposite___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___closed__0 = (const lean_object*)&lp_mathlib_Subsemigroup_centerToMulOpposite___closed__0_value;
static const lean_ctor_object lp_mathlib_Subsemigroup_centerToMulOpposite___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subsemigroup_centerToMulOpposite___closed__0_value),((lean_object*)&lp_mathlib_Subsemigroup_centerToMulOpposite___closed__0_value)}};
static const lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___closed__1 = (const lean_object*)&lp_mathlib_Subsemigroup_centerToMulOpposite___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerToAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerToAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerToMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerToAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerToAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center(lean_object* v_M_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center___boxed(lean_object* v_M_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Submonoid_center(v_M_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center(lean_object* v_M_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center___boxed(lean_object* v_M_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_AddSubmonoid_center(v_M_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid_x27___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; lean_object* v_toOne_15_; lean_object* v_toMul_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_14_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_inst_13_);
v_toOne_15_ = lean_ctor_get(v___x_14_, 0);
lean_inc_n(v_toOne_15_, 2);
v_toMul_16_ = lean_ctor_get(v___x_14_, 1);
lean_inc_n(v_toMul_16_, 2);
lean_dec_ref(v___x_14_);
v___x_17_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_17_, 0, lean_box(0));
lean_closure_set(v___x_17_, 1, v_toMul_16_);
lean_closure_set(v___x_17_, 2, v_toOne_15_);
v___x_18_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_18_, 0, v_toOne_15_);
lean_ctor_set(v___x_18_, 1, v_toMul_16_);
lean_ctor_set(v___x_18_, 2, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid_x27(lean_object* v_M_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; lean_object* v_toOne_22_; lean_object* v_toMul_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_21_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_inst_20_);
v_toOne_22_ = lean_ctor_get(v___x_21_, 0);
lean_inc_n(v_toOne_22_, 2);
v_toMul_23_ = lean_ctor_get(v___x_21_, 1);
lean_inc_n(v_toMul_23_, 2);
lean_dec_ref(v___x_21_);
v___x_24_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_24_, 0, lean_box(0));
lean_closure_set(v___x_24_, 1, v_toMul_23_);
lean_closure_set(v___x_24_, 2, v_toOne_22_);
v___x_25_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_25_, 0, v_toOne_22_);
lean_ctor_set(v___x_25_, 1, v_toMul_23_);
lean_ctor_set(v___x_25_, 2, v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid_x27___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; lean_object* v_toZero_28_; lean_object* v_toAdd_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_27_ = lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(v_inst_26_);
v_toZero_28_ = lean_ctor_get(v___x_27_, 0);
lean_inc_n(v_toZero_28_, 2);
v_toAdd_29_ = lean_ctor_get(v___x_27_, 1);
lean_inc_n(v_toAdd_29_, 2);
lean_dec_ref(v___x_27_);
v___x_30_ = lean_alloc_closure((void*)(lp_mathlib_nsmulBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_30_, 0, lean_box(0));
lean_closure_set(v___x_30_, 1, v_toAdd_29_);
lean_closure_set(v___x_30_, 2, v_toZero_28_);
v___x_31_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_31_, 0, v_toZero_28_);
lean_ctor_set(v___x_31_, 1, v_toAdd_29_);
lean_ctor_set(v___x_31_, 2, v___x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid_x27(lean_object* v_M_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_AddSubmonoid_center_addCommMonoid_x27___redArg(v_inst_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid___redArg(lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_center_commMonoid(lean_object* v_M_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_center_addCommMonoid(lean_object* v_M_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_43_);
return v___x_44_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCenter___redArg(uint8_t v_inst_45_){
_start:
{
return v_inst_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCenter___redArg___boxed(lean_object* v_inst_46_){
_start:
{
uint8_t v_inst_10__boxed_47_; uint8_t v_res_48_; lean_object* v_r_49_; 
v_inst_10__boxed_47_ = lean_unbox(v_inst_46_);
v_res_48_ = lp_mathlib_Submonoid_decidableMemCenter___redArg(v_inst_10__boxed_47_);
v_r_49_ = lean_box(v_res_48_);
return v_r_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_decidableMemCenter(lean_object* v_M_50_, lean_object* v_inst_51_, lean_object* v_a_52_, uint8_t v_inst_53_){
_start:
{
return v_inst_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_decidableMemCenter___boxed(lean_object* v_M_54_, lean_object* v_inst_55_, lean_object* v_a_56_, lean_object* v_inst_57_){
_start:
{
uint8_t v_inst_14__boxed_58_; uint8_t v_res_59_; lean_object* v_r_60_; 
v_inst_14__boxed_58_ = lean_unbox(v_inst_57_);
v_res_59_ = lp_mathlib_Submonoid_decidableMemCenter(v_M_54_, v_inst_55_, v_a_56_, v_inst_14__boxed_58_);
lean_dec(v_a_56_);
lean_dec_ref(v_inst_55_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCenter___redArg(uint8_t v_inst_61_){
_start:
{
return v_inst_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCenter___redArg___boxed(lean_object* v_inst_62_){
_start:
{
uint8_t v_inst_10__boxed_63_; uint8_t v_res_64_; lean_object* v_r_65_; 
v_inst_10__boxed_63_ = lean_unbox(v_inst_62_);
v_res_64_ = lp_mathlib_AddSubmonoid_decidableMemCenter___redArg(v_inst_10__boxed_63_);
v_r_65_ = lean_box(v_res_64_);
return v_r_65_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddSubmonoid_decidableMemCenter(lean_object* v_M_66_, lean_object* v_inst_67_, lean_object* v_a_68_, uint8_t v_inst_69_){
_start:
{
return v_inst_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_decidableMemCenter___boxed(lean_object* v_M_70_, lean_object* v_inst_71_, lean_object* v_a_72_, lean_object* v_inst_73_){
_start:
{
uint8_t v_inst_14__boxed_74_; uint8_t v_res_75_; lean_object* v_r_76_; 
v_inst_14__boxed_74_ = lean_unbox(v_inst_73_);
v_res_75_ = lp_mathlib_AddSubmonoid_decidableMemCenter(v_M_70_, v_inst_71_, v_a_72_, v_inst_14__boxed_74_);
lean_dec(v_a_72_);
lean_dec_ref(v_inst_71_);
v_r_76_ = lean_box(v_res_75_);
return v_r_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCenterToCenterUnits(lean_object* v_M_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v___f_84_; 
v___f_84_ = ((lean_object*)(lp_mathlib_unitsCenterToCenterUnits___closed__2));
return v___f_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCenterToCenterUnits___boxed(lean_object* v_M_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_unitsCenterToCenterUnits(v_M_85_, v_inst_86_);
lean_dec_ref(v_inst_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addUnitsCenterToCenterAddUnits(lean_object* v_M_92_, lean_object* v_inst_93_){
_start:
{
lean_object* v___f_94_; 
v___f_94_ = ((lean_object*)(lp_mathlib_addUnitsCenterToCenterAddUnits___closed__1));
return v___f_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addUnitsCenterToCenterAddUnits___boxed(lean_object* v_M_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_addUnitsCenterToCenterAddUnits(v_M_95_, v_inst_96_);
lean_dec_ref(v_inst_96_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg___lam__0(lean_object* v_e_98_, lean_object* v_r_99_){
_start:
{
lean_object* v_toFun_100_; lean_object* v___x_101_; 
v_toFun_100_ = lean_ctor_get(v_e_98_, 0);
lean_inc(v_toFun_100_);
lean_dec_ref(v_e_98_);
v___x_101_ = lean_apply_1(v_toFun_100_, v_r_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg___lam__1(lean_object* v_e_102_, lean_object* v_s_103_){
_start:
{
lean_object* v___x_104_; lean_object* v_toFun_105_; lean_object* v___x_106_; 
v___x_104_ = lp_mathlib_Equiv_symm___redArg(v_e_102_);
v_toFun_105_ = lean_ctor_get(v___x_104_, 0);
lean_inc(v_toFun_105_);
lean_dec_ref(v___x_104_);
v___x_106_ = lean_apply_1(v_toFun_105_, v_s_103_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___redArg(lean_object* v_e_107_){
_start:
{
lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_110_; 
lean_inc_ref(v_e_107_);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib_Subsemigroup_centerCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_108_, 0, v_e_107_);
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_Subsemigroup_centerCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_109_, 0, v_e_107_);
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v___f_108_);
lean_ctor_set(v___x_110_, 1, v___f_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr(lean_object* v_M_111_, lean_object* v_N_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_e_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_Subsemigroup_centerCongr___redArg(v_e_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerCongr___boxed(lean_object* v_M_117_, lean_object* v_N_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_e_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_Subsemigroup_centerCongr(v_M_117_, v_N_118_, v_inst_119_, v_inst_120_, v_e_121_);
lean_dec(v_inst_120_);
lean_dec(v_inst_119_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerCongr___redArg(lean_object* v_e_123_){
_start:
{
lean_object* v___f_124_; lean_object* v___f_125_; lean_object* v___x_126_; 
lean_inc_ref(v_e_123_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Subsemigroup_centerCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_124_, 0, v_e_123_);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_Subsemigroup_centerCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_125_, 0, v_e_123_);
v___x_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_126_, 0, v___f_124_);
lean_ctor_set(v___x_126_, 1, v___f_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerCongr(lean_object* v_M_127_, lean_object* v_N_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_e_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_AddSubsemigroup_centerCongr___redArg(v_e_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerCongr___boxed(lean_object* v_M_133_, lean_object* v_N_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_e_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_AddSubsemigroup_centerCongr(v_M_133_, v_N_134_, v_inst_135_, v_inst_136_, v_e_137_);
lean_dec(v_inst_136_);
lean_dec(v_inst_135_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerCongr___redArg(lean_object* v_e_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_Subsemigroup_centerCongr___redArg(v_e_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerCongr(lean_object* v_M_141_, lean_object* v_N_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_e_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_Subsemigroup_centerCongr___redArg(v_e_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerCongr___boxed(lean_object* v_M_147_, lean_object* v_N_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_e_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_Submonoid_centerCongr(v_M_147_, v_N_148_, v_inst_149_, v_inst_150_, v_e_151_);
lean_dec_ref(v_inst_150_);
lean_dec_ref(v_inst_149_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerCongr___redArg(lean_object* v_e_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_AddSubsemigroup_centerCongr___redArg(v_e_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerCongr(lean_object* v_M_155_, lean_object* v_N_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_e_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_AddSubsemigroup_centerCongr___redArg(v_e_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerCongr___boxed(lean_object* v_M_161_, lean_object* v_N_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_e_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_AddSubmonoid_centerCongr(v_M_161_, v_N_162_, v_inst_163_, v_inst_164_, v_e_165_);
lean_dec_ref(v_inst_164_);
lean_dec_ref(v_inst_163_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___lam__0(lean_object* v_r_167_){
_start:
{
lean_inc(v_r_167_);
return v_r_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___lam__0___boxed(lean_object* v_r_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Subsemigroup_centerToMulOpposite___lam__0(v_r_168_);
lean_dec(v_r_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite(lean_object* v_M_173_, lean_object* v_inst_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = ((lean_object*)(lp_mathlib_Subsemigroup_centerToMulOpposite___closed__1));
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_centerToMulOpposite___boxed(lean_object* v_M_176_, lean_object* v_inst_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_Subsemigroup_centerToMulOpposite(v_M_176_, v_inst_177_);
lean_dec(v_inst_177_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerToAddOpposite(lean_object* v_M_179_, lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = ((lean_object*)(lp_mathlib_Subsemigroup_centerToMulOpposite___closed__1));
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_centerToAddOpposite___boxed(lean_object* v_M_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_AddSubsemigroup_centerToAddOpposite(v_M_182_, v_inst_183_);
lean_dec(v_inst_183_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerToMulOpposite___redArg(lean_object* v_inst_185_){
_start:
{
lean_object* v___x_186_; lean_object* v_toMul_187_; lean_object* v___x_188_; 
v___x_186_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_185_);
v_toMul_187_ = lean_ctor_get(v___x_186_, 1);
lean_inc(v_toMul_187_);
lean_dec_ref(v___x_186_);
v___x_188_ = lp_mathlib_Subsemigroup_centerToMulOpposite(lean_box(0), v_toMul_187_);
lean_dec(v_toMul_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_centerToMulOpposite(lean_object* v_M_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_Submonoid_centerToMulOpposite___redArg(v_inst_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerToAddOpposite___redArg(lean_object* v_inst_192_){
_start:
{
lean_object* v___x_193_; lean_object* v_toAdd_194_; lean_object* v___x_195_; 
v___x_193_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_192_);
v_toAdd_194_ = lean_ctor_get(v___x_193_, 1);
lean_inc(v_toAdd_194_);
lean_dec_ref(v___x_193_);
v___x_195_ = lp_mathlib_AddSubsemigroup_centerToAddOpposite(lean_box(0), v_toAdd_194_);
lean_dec(v_toAdd_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_centerToAddOpposite(lean_object* v_M_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_AddSubmonoid_centerToAddOpposite___redArg(v_inst_197_);
return v___x_198_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Submonoid_Center(builtin);
}
#ifdef __cplusplus
}
#endif
