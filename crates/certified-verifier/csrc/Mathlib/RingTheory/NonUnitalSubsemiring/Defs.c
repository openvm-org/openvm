// Lean compiler output
// Module: Mathlib.RingTheory.NonUnitalSubsemiring.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Hom.Defs public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.Algebra.Group.Submonoid.Defs public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_MulMemClass_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiringClass_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiringClass_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_toSubsemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_toSubsemigroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_NonUnitalSubsemiring_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_NonUnitalSubsemiring_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_instMin___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqSlocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqSlocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubsemiring_inclusion___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_inclusion___closed__0_value;
static const lean_closure_object lp_mathlib_NonUnitalSubsemiring_inclusion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_NonUnitalSubsemiring_inclusion___closed__0_value),((lean_object*)&lp_mathlib_NonUnitalSubsemiringClass_subtype___closed__0_value)} };
static const lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion___closed__1 = (const lean_object*)&lp_mathlib_NonUnitalSubsemiring_inclusion___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toAddCommMonoid_2_; lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v_toMul_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_13_; 
v_toAddCommMonoid_2_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_toAddCommMonoid_2_);
v___x_3_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_toAddCommMonoid_2_);
v___x_4_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_1_);
v_toMul_5_ = lean_ctor_get(v___x_4_, 0);
v_isSharedCheck_13_ = !lean_is_exclusive(v___x_4_);
if (v_isSharedCheck_13_ == 0)
{
lean_object* v_unused_14_; 
v_unused_14_ = lean_ctor_get(v___x_4_, 1);
lean_dec(v_unused_14_);
v___x_7_ = v___x_4_;
v_isShared_8_ = v_isSharedCheck_13_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_toMul_5_);
lean_dec(v___x_4_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_13_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v___f_9_; lean_object* v___x_11_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_toMul_5_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 1, v___f_9_);
lean_ctor_set(v___x_7_, 0, v___x_3_);
v___x_11_ = v___x_7_;
goto v_reusejp_10_;
}
else
{
lean_object* v_reuseFailAlloc_12_; 
v_reuseFailAlloc_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_12_, 0, v___x_3_);
lean_ctor_set(v_reuseFailAlloc_12_, 1, v___f_9_);
v___x_11_ = v_reuseFailAlloc_12_;
goto v_reusejp_10_;
}
v_reusejp_10_:
{
return v___x_11_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring(lean_object* v_R_15_, lean_object* v_S_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_s_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_17_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___boxed(lean_object* v_R_22_, lean_object* v_S_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_s_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring(v_R_22_, v_S_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_s_27_);
lean_dec(v_s_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring(lean_object* v_S_31_, lean_object* v_s_32_, lean_object* v_R_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_34_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring___boxed(lean_object* v_S_38_, lean_object* v_s_39_, lean_object* v_R_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocCommSemiring(v_S_38_, v_s_39_, v_R_40_, v_inst_41_, v_inst_42_, v_inst_43_);
lean_dec(v_s_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0(lean_object* v_self_45_){
_start:
{
lean_inc(v_self_45_);
return v_self_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0___boxed(lean_object* v_self_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_NonUnitalSubsemiringClass_subtype___lam__0(v_self_46_);
lean_dec(v_self_46_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype(lean_object* v_R_49_, lean_object* v_S_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_s_54_){
_start:
{
lean_object* v___f_55_; 
v___f_55_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiringClass_subtype___closed__0));
return v___f_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_subtype___boxed(lean_object* v_R_56_, lean_object* v_S_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_s_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_NonUnitalSubsemiringClass_subtype(v_R_56_, v_S_57_, v_inst_58_, v_inst_59_, v_inst_60_, v_s_61_);
lean_dec(v_s_61_);
lean_dec_ref(v_inst_58_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring___redArg(lean_object* v_inst_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring(lean_object* v_S_65_, lean_object* v_s_66_, lean_object* v_R_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_68_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring___boxed(lean_object* v_S_72_, lean_object* v_s_73_, lean_object* v_R_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalSemiring(v_S_72_, v_s_73_, v_R_74_, v_inst_75_, v_inst_76_, v_inst_77_);
lean_dec(v_s_73_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring___redArg(lean_object* v_inst_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring(lean_object* v_S_81_, lean_object* v_s_82_, lean_object* v_R_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalNonAssocSemiring___redArg(v_inst_84_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring___boxed(lean_object* v_S_88_, lean_object* v_s_89_, lean_object* v_R_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_NonUnitalSubsemiringClass_toNonUnitalCommSemiring(v_S_88_, v_s_89_, v_R_90_, v_inst_91_, v_inst_92_, v_inst_93_);
lean_dec(v_s_89_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_toSubsemigroup(lean_object* v_R_95_, lean_object* v_inst_96_, lean_object* v_self_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_box(0);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_toSubsemigroup___boxed(lean_object* v_R_99_, lean_object* v_inst_100_, lean_object* v_self_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_NonUnitalSubsemiring_toSubsemigroup(v_R_99_, v_inst_100_, v_self_101_);
lean_dec_ref(v_inst_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instSetLike(lean_object* v_R_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lean_box(0);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instSetLike___boxed(lean_object* v_R_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_NonUnitalSubsemiring_instSetLike(v_R_106_, v_inst_107_);
lean_dec_ref(v_inst_107_);
return v_res_108_;
}
}
static lean_object* _init_lp_mathlib_NonUnitalSubsemiring_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = lean_box(0);
v___x_110_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instPartialOrder(lean_object* v_R_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lean_obj_once(&lp_mathlib_NonUnitalSubsemiring_instPartialOrder___closed__0, &lp_mathlib_NonUnitalSubsemiring_instPartialOrder___closed__0_once, _init_lp_mathlib_NonUnitalSubsemiring_instPartialOrder___closed__0);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instPartialOrder___boxed(lean_object* v_R_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_NonUnitalSubsemiring_instPartialOrder(v_R_114_, v_inst_115_);
lean_dec_ref(v_inst_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_ofClass(lean_object* v_S_117_, lean_object* v_R_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_s_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lean_box(0);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_ofClass___boxed(lean_object* v_S_124_, lean_object* v_R_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_s_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_NonUnitalSubsemiring_ofClass(v_S_124_, v_R_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_s_129_);
lean_dec(v_s_129_);
lean_dec_ref(v_inst_126_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_copy(lean_object* v_R_131_, lean_object* v_inst_132_, lean_object* v_S_133_, lean_object* v_s_134_, lean_object* v_hs_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_box(0);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_copy___boxed(lean_object* v_R_137_, lean_object* v_inst_138_, lean_object* v_S_139_, lean_object* v_s_140_, lean_object* v_hs_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_NonUnitalSubsemiring_copy(v_R_137_, v_inst_138_, v_S_139_, v_s_140_, v_hs_141_);
lean_dec_ref(v_inst_138_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_mk_x27(lean_object* v_R_143_, lean_object* v_inst_144_, lean_object* v_s_145_, lean_object* v_sg_146_, lean_object* v_hg_147_, lean_object* v_sa_148_, lean_object* v_ha_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lean_box(0);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_mk_x27___boxed(lean_object* v_R_151_, lean_object* v_inst_152_, lean_object* v_s_153_, lean_object* v_sg_154_, lean_object* v_hg_155_, lean_object* v_sa_156_, lean_object* v_ha_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_NonUnitalSubsemiring_mk_x27(v_R_151_, v_inst_152_, v_s_153_, v_sg_154_, v_hg_155_, v_sa_156_, v_ha_157_);
lean_dec_ref(v_inst_152_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instTop(lean_object* v_R_159_, lean_object* v_inst_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lean_box(0);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instTop___boxed(lean_object* v_R_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_NonUnitalSubsemiring_instTop(v_R_162_, v_inst_163_);
lean_dec_ref(v_inst_163_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instBot(lean_object* v_R_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_box(0);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instBot___boxed(lean_object* v_R_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_NonUnitalSubsemiring_instBot(v_R_168_, v_inst_169_);
lean_dec_ref(v_inst_169_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInhabited(lean_object* v_R_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lean_box(0);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instInhabited___boxed(lean_object* v_R_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_NonUnitalSubsemiring_instInhabited(v_R_174_, v_inst_175_);
lean_dec_ref(v_inst_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instMin___lam__0(lean_object* v_s_177_, lean_object* v_t_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_box(0);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instMin(lean_object* v_R_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v___f_183_; 
v___f_183_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_instMin___closed__0));
return v___f_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_instMin___boxed(lean_object* v_R_184_, lean_object* v_inst_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_NonUnitalSubsemiring_instMin(v_R_184_, v_inst_185_);
lean_dec_ref(v_inst_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0(lean_object* v_inst_187_, lean_object* v_f_188_, lean_object* v_n_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lean_apply_2(v_inst_187_, v_f_188_, v_n_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg(lean_object* v_inst_191_, lean_object* v_f_192_){
_start:
{
lean_object* v___f_193_; 
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0), 3, 2);
lean_closure_set(v___f_193_, 0, v_inst_191_);
lean_closure_set(v___f_193_, 1, v_f_192_);
return v___f_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict(lean_object* v_R_194_, lean_object* v_S_195_, lean_object* v_inst_196_, lean_object* v_F_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_S_x27_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_f_204_, lean_object* v_s_205_, lean_object* v_h_206_){
_start:
{
lean_object* v___f_207_; 
v___f_207_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0), 3, 2);
lean_closure_set(v___f_207_, 0, v_inst_198_);
lean_closure_set(v___f_207_, 1, v_f_204_);
return v___f_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___boxed(lean_object* v_R_208_, lean_object* v_S_209_, lean_object* v_inst_210_, lean_object* v_F_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_S_x27_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_f_218_, lean_object* v_s_219_, lean_object* v_h_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_NonUnitalRingHom_codRestrict(v_R_208_, v_S_209_, v_inst_210_, v_F_211_, v_inst_212_, v_inst_213_, v_inst_214_, v_S_x27_215_, v_inst_216_, v_inst_217_, v_f_218_, v_s_219_, v_h_220_);
lean_dec(v_s_219_);
lean_dec_ref(v_inst_213_);
lean_dec_ref(v_inst_210_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqSlocus(lean_object* v_R_222_, lean_object* v_S_223_, lean_object* v_inst_224_, lean_object* v_F_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_f_229_, lean_object* v_g_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lean_box(0);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqSlocus___boxed(lean_object* v_R_232_, lean_object* v_S_233_, lean_object* v_inst_234_, lean_object* v_F_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_f_239_, lean_object* v_g_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_NonUnitalRingHom_eqSlocus(v_R_232_, v_S_233_, v_inst_234_, v_F_235_, v_inst_236_, v_inst_237_, v_inst_238_, v_f_239_, v_g_240_);
lean_dec(v_g_240_);
lean_dec(v_f_239_);
lean_dec_ref(v_inst_237_);
lean_dec(v_inst_236_);
lean_dec_ref(v_inst_234_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion___lam__0(lean_object* v_f_242_, lean_object* v___y_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_apply_1(v_f_242_, v___y_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion(lean_object* v_R_249_, lean_object* v_inst_250_, lean_object* v_S_251_, lean_object* v_T_252_, lean_object* v_h_253_){
_start:
{
lean_object* v___f_254_; 
v___f_254_ = ((lean_object*)(lp_mathlib_NonUnitalSubsemiring_inclusion___closed__1));
return v___f_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubsemiring_inclusion___boxed(lean_object* v_R_255_, lean_object* v_inst_256_, lean_object* v_S_257_, lean_object* v_T_258_, lean_object* v_h_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_NonUnitalSubsemiring_inclusion(v_R_255_, v_inst_256_, v_S_257_, v_T_258_, v_h_259_);
lean_dec_ref(v_inst_256_);
return v_res_260_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
