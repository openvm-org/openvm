// Lean compiler output
// Module: Mathlib.GroupTheory.MonoidLocalization.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Basic public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Algebra.Regular.Basic public import Mathlib.GroupTheory.Congruence.Hom public import Mathlib.GroupTheory.OreLocalization.Basic
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddOreLocalization_addOreSetComm(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOreLocalization_instAddMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_oreSetComm(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_instMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_r(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_r___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_r_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_r_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_rec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_rec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_recOnSubsingleton_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_recOnSubsingleton_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_recOnSubsingleton_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkHom___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Localization_mkHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Localization_mkHom___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Localization_mkHom___closed__0 = (const lean_object*)&lp_mathlib_Localization_mkHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mkHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mkHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submonoid_LocalizationMap_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submonoid_LocalizationMap_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_Submonoid_LocalizationMap_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Localization_decidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_decidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Localization_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Localization_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddLocalization_decidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_decidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddLocalization_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddLocalization_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instCancelCommMonoidLocalization___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instCancelCommMonoidLocalization(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instAddCancelCommMonoidLocalization___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instAddCancelCommMonoidLocalization(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_r(lean_object* v_M_1_, lean_object* v_inst_2_, lean_object* v_S_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_r___boxed(lean_object* v_M_5_, lean_object* v_inst_6_, lean_object* v_S_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Localization_r(v_M_5_, v_inst_6_, v_S_7_);
lean_dec_ref(v_inst_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r(lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_S_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r___boxed(lean_object* v_M_13_, lean_object* v_inst_14_, lean_object* v_S_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_AddLocalization_r(v_M_13_, v_inst_14_, v_S_15_);
lean_dec_ref(v_inst_14_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_r_x27(lean_object* v_M_17_, lean_object* v_inst_18_, lean_object* v_S_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_box(0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_r_x27___boxed(lean_object* v_M_21_, lean_object* v_inst_22_, lean_object* v_S_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Localization_r_x27(v_M_21_, v_inst_22_, v_S_23_);
lean_dec_ref(v_inst_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r_x27(lean_object* v_M_25_, lean_object* v_inst_26_, lean_object* v_S_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_r_x27___boxed(lean_object* v_M_29_, lean_object* v_inst_30_, lean_object* v_S_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_AddLocalization_r_x27(v_M_29_, v_inst_30_, v_S_31_);
lean_dec_ref(v_inst_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mk___redArg(lean_object* v_x_33_, lean_object* v_y_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_x_33_);
lean_ctor_set(v___x_35_, 1, v_y_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mk(lean_object* v_M_36_, lean_object* v_inst_37_, lean_object* v_S_38_, lean_object* v_x_39_, lean_object* v_y_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_41_, 0, v_x_39_);
lean_ctor_set(v___x_41_, 1, v_y_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mk___boxed(lean_object* v_M_42_, lean_object* v_inst_43_, lean_object* v_S_44_, lean_object* v_x_45_, lean_object* v_y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Localization_mk(v_M_42_, v_inst_43_, v_S_44_, v_x_45_, v_y_46_);
lean_dec_ref(v_inst_43_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mk___redArg(lean_object* v_x_48_, lean_object* v_y_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_50_, 0, v_x_48_);
lean_ctor_set(v___x_50_, 1, v_y_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mk(lean_object* v_M_51_, lean_object* v_inst_52_, lean_object* v_S_53_, lean_object* v_x_54_, lean_object* v_y_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v_x_54_);
lean_ctor_set(v___x_56_, 1, v_y_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mk___boxed(lean_object* v_M_57_, lean_object* v_inst_58_, lean_object* v_S_59_, lean_object* v_x_60_, lean_object* v_y_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_AddLocalization_mk(v_M_57_, v_inst_58_, v_S_59_, v_x_60_, v_y_61_);
lean_dec_ref(v_inst_58_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_rec___redArg(lean_object* v_f_63_, lean_object* v_x_64_){
_start:
{
lean_object* v_fst_65_; lean_object* v_snd_66_; lean_object* v___x_67_; 
v_fst_65_ = lean_ctor_get(v_x_64_, 0);
lean_inc(v_fst_65_);
v_snd_66_ = lean_ctor_get(v_x_64_, 1);
lean_inc(v_snd_66_);
lean_dec(v_x_64_);
v___x_67_ = lean_apply_2(v_f_63_, v_fst_65_, v_snd_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_rec(lean_object* v_M_68_, lean_object* v_inst_69_, lean_object* v_S_70_, lean_object* v_p_71_, lean_object* v_f_72_, lean_object* v_H_73_, lean_object* v_x_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Localization_rec___redArg(v_f_72_, v_x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_rec___boxed(lean_object* v_M_76_, lean_object* v_inst_77_, lean_object* v_S_78_, lean_object* v_p_79_, lean_object* v_f_80_, lean_object* v_H_81_, lean_object* v_x_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Localization_rec(v_M_76_, v_inst_77_, v_S_78_, v_p_79_, v_f_80_, v_H_81_, v_x_82_);
lean_dec_ref(v_inst_77_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_rec___redArg(lean_object* v_f_84_, lean_object* v_x_85_){
_start:
{
lean_object* v_fst_86_; lean_object* v_snd_87_; lean_object* v___x_88_; 
v_fst_86_ = lean_ctor_get(v_x_85_, 0);
lean_inc(v_fst_86_);
v_snd_87_ = lean_ctor_get(v_x_85_, 1);
lean_inc(v_snd_87_);
lean_dec(v_x_85_);
v___x_88_ = lean_apply_2(v_f_84_, v_fst_86_, v_snd_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_rec(lean_object* v_M_89_, lean_object* v_inst_90_, lean_object* v_S_91_, lean_object* v_p_92_, lean_object* v_f_93_, lean_object* v_H_94_, lean_object* v_x_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_AddLocalization_rec___redArg(v_f_93_, v_x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_rec___boxed(lean_object* v_M_97_, lean_object* v_inst_98_, lean_object* v_S_99_, lean_object* v_p_100_, lean_object* v_f_101_, lean_object* v_H_102_, lean_object* v_x_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_AddLocalization_rec(v_M_97_, v_inst_98_, v_S_99_, v_p_100_, v_f_101_, v_H_102_, v_x_103_);
lean_dec_ref(v_inst_98_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__0(lean_object* v_f_105_, lean_object* v_x_106_, lean_object* v_x_107_, lean_object* v_x_108_, lean_object* v_x_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_apply_4(v_f_105_, v_x_106_, v_x_108_, v_x_107_, v_x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__1(lean_object* v_f_111_, lean_object* v_x_112_, lean_object* v_x_113_, lean_object* v_t_114_){
_start:
{
lean_object* v___f_115_; lean_object* v___x_116_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__0), 5, 3);
lean_closure_set(v___f_115_, 0, v_f_111_);
lean_closure_set(v___f_115_, 1, v_x_112_);
lean_closure_set(v___f_115_, 2, v_x_113_);
v___x_116_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v___f_115_, v_t_114_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___redArg(lean_object* v_x_117_, lean_object* v_y_118_, lean_object* v_f_119_){
_start:
{
lean_object* v___f_120_; lean_object* v___x_22__overap_121_; lean_object* v___x_122_; 
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__1), 4, 1);
lean_closure_set(v___f_120_, 0, v_f_119_);
v___x_22__overap_121_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v___f_120_, v_x_117_);
v___x_122_ = lean_apply_1(v___x_22__overap_121_, v_y_118_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082(lean_object* v_M_123_, lean_object* v_inst_124_, lean_object* v_S_125_, lean_object* v_r_126_, lean_object* v_h_127_, lean_object* v_x_128_, lean_object* v_y_129_, lean_object* v_f_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_Localization_recOnSubsingleton_u2082___redArg(v_x_128_, v_y_129_, v_f_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_recOnSubsingleton_u2082___boxed(lean_object* v_M_132_, lean_object* v_inst_133_, lean_object* v_S_134_, lean_object* v_r_135_, lean_object* v_h_136_, lean_object* v_x_137_, lean_object* v_y_138_, lean_object* v_f_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_Localization_recOnSubsingleton_u2082(v_M_132_, v_inst_133_, v_S_134_, v_r_135_, v_h_136_, v_x_137_, v_y_138_, v_f_139_);
lean_dec_ref(v_inst_133_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_recOnSubsingleton_u2082___redArg(lean_object* v_x_141_, lean_object* v_y_142_, lean_object* v_f_143_){
_start:
{
lean_object* v___f_144_; lean_object* v___x_22__overap_145_; lean_object* v___x_146_; 
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_Localization_recOnSubsingleton_u2082___redArg___lam__1), 4, 1);
lean_closure_set(v___f_144_, 0, v_f_143_);
v___x_22__overap_145_ = lp_mathlib_Prod_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3167448894____hygCtx___hyg_3_(v___f_144_, v_x_141_);
v___x_146_ = lean_apply_1(v___x_22__overap_145_, v_y_142_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_recOnSubsingleton_u2082(lean_object* v_M_147_, lean_object* v_inst_148_, lean_object* v_S_149_, lean_object* v_r_150_, lean_object* v_h_151_, lean_object* v_x_152_, lean_object* v_y_153_, lean_object* v_f_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_AddLocalization_recOnSubsingleton_u2082___redArg(v_x_152_, v_y_153_, v_f_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_recOnSubsingleton_u2082___boxed(lean_object* v_M_156_, lean_object* v_inst_157_, lean_object* v_S_158_, lean_object* v_r_159_, lean_object* v_h_160_, lean_object* v_x_161_, lean_object* v_y_162_, lean_object* v_f_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_AddLocalization_recOnSubsingleton_u2082(v_M_156_, v_inst_157_, v_S_158_, v_r_159_, v_h_160_, v_x_161_, v_y_162_, v_f_163_);
lean_dec_ref(v_inst_157_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn___redArg(lean_object* v_x_165_, lean_object* v_f_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_Localization_rec___redArg(v_f_166_, v_x_165_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn(lean_object* v_M_168_, lean_object* v_inst_169_, lean_object* v_S_170_, lean_object* v_p_171_, lean_object* v_x_172_, lean_object* v_f_173_, lean_object* v_H_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_Localization_rec___redArg(v_f_173_, v_x_172_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn___boxed(lean_object* v_M_176_, lean_object* v_inst_177_, lean_object* v_S_178_, lean_object* v_p_179_, lean_object* v_x_180_, lean_object* v_f_181_, lean_object* v_H_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_Localization_liftOn(v_M_176_, v_inst_177_, v_S_178_, v_p_179_, v_x_180_, v_f_181_, v_H_182_);
lean_dec_ref(v_inst_177_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn___redArg(lean_object* v_x_184_, lean_object* v_f_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_AddLocalization_rec___redArg(v_f_185_, v_x_184_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn(lean_object* v_M_187_, lean_object* v_inst_188_, lean_object* v_S_189_, lean_object* v_p_190_, lean_object* v_x_191_, lean_object* v_f_192_, lean_object* v_H_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_AddLocalization_rec___redArg(v_f_192_, v_x_191_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn___boxed(lean_object* v_M_195_, lean_object* v_inst_196_, lean_object* v_S_197_, lean_object* v_p_198_, lean_object* v_x_199_, lean_object* v_f_200_, lean_object* v_H_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_AddLocalization_liftOn(v_M_195_, v_inst_196_, v_S_197_, v_p_198_, v_x_199_, v_f_200_, v_H_201_);
lean_dec_ref(v_inst_196_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082___redArg___lam__0(lean_object* v_f_203_, lean_object* v_y_204_, lean_object* v_a_205_, lean_object* v_b_206_){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_207_ = lean_apply_2(v_f_203_, v_a_205_, v_b_206_);
v___x_208_ = lp_mathlib_Localization_rec___redArg(v___x_207_, v_y_204_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082___redArg(lean_object* v_x_209_, lean_object* v_y_210_, lean_object* v_f_211_){
_start:
{
lean_object* v___f_212_; lean_object* v___x_213_; 
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_Localization_liftOn_u2082___redArg___lam__0), 4, 2);
lean_closure_set(v___f_212_, 0, v_f_211_);
lean_closure_set(v___f_212_, 1, v_y_210_);
v___x_213_ = lp_mathlib_Localization_rec___redArg(v___f_212_, v_x_209_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082(lean_object* v_M_214_, lean_object* v_inst_215_, lean_object* v_S_216_, lean_object* v_p_217_, lean_object* v_x_218_, lean_object* v_y_219_, lean_object* v_f_220_, lean_object* v_H_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_Localization_liftOn_u2082___redArg(v_x_218_, v_y_219_, v_f_220_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_liftOn_u2082___boxed(lean_object* v_M_223_, lean_object* v_inst_224_, lean_object* v_S_225_, lean_object* v_p_226_, lean_object* v_x_227_, lean_object* v_y_228_, lean_object* v_f_229_, lean_object* v_H_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_Localization_liftOn_u2082(v_M_223_, v_inst_224_, v_S_225_, v_p_226_, v_x_227_, v_y_228_, v_f_229_, v_H_230_);
lean_dec_ref(v_inst_224_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082___redArg___lam__0(lean_object* v_f_232_, lean_object* v_y_233_, lean_object* v_a_234_, lean_object* v_b_235_){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = lean_apply_2(v_f_232_, v_a_234_, v_b_235_);
v___x_237_ = lp_mathlib_AddLocalization_rec___redArg(v___x_236_, v_y_233_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082___redArg(lean_object* v_x_238_, lean_object* v_y_239_, lean_object* v_f_240_){
_start:
{
lean_object* v___f_241_; lean_object* v___x_242_; 
v___f_241_ = lean_alloc_closure((void*)(lp_mathlib_AddLocalization_liftOn_u2082___redArg___lam__0), 4, 2);
lean_closure_set(v___f_241_, 0, v_f_240_);
lean_closure_set(v___f_241_, 1, v_y_239_);
v___x_242_ = lp_mathlib_AddLocalization_rec___redArg(v___f_241_, v_x_238_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082(lean_object* v_M_243_, lean_object* v_inst_244_, lean_object* v_S_245_, lean_object* v_p_246_, lean_object* v_x_247_, lean_object* v_y_248_, lean_object* v_f_249_, lean_object* v_H_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lp_mathlib_AddLocalization_liftOn_u2082___redArg(v_x_247_, v_y_248_, v_f_249_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_liftOn_u2082___boxed(lean_object* v_M_252_, lean_object* v_inst_253_, lean_object* v_S_254_, lean_object* v_p_255_, lean_object* v_x_256_, lean_object* v_y_257_, lean_object* v_f_258_, lean_object* v_H_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_AddLocalization_liftOn_u2082(v_M_252_, v_inst_253_, v_S_254_, v_p_255_, v_x_256_, v_y_257_, v_f_258_, v_H_259_);
lean_dec_ref(v_inst_253_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkHom___lam__0(lean_object* v_x_261_){
_start:
{
lean_object* v_fst_262_; lean_object* v_snd_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_270_; 
v_fst_262_ = lean_ctor_get(v_x_261_, 0);
v_snd_263_ = lean_ctor_get(v_x_261_, 1);
v_isSharedCheck_270_ = !lean_is_exclusive(v_x_261_);
if (v_isSharedCheck_270_ == 0)
{
v___x_265_ = v_x_261_;
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_snd_263_);
lean_inc(v_fst_262_);
lean_dec(v_x_261_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_268_; 
if (v_isShared_266_ == 0)
{
v___x_268_ = v___x_265_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_fst_262_);
lean_ctor_set(v_reuseFailAlloc_269_, 1, v_snd_263_);
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
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkHom(lean_object* v_M_272_, lean_object* v_inst_273_, lean_object* v_S_274_){
_start:
{
lean_object* v___f_275_; 
v___f_275_ = ((lean_object*)(lp_mathlib_Localization_mkHom___closed__0));
return v___f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_mkHom___boxed(lean_object* v_M_276_, lean_object* v_inst_277_, lean_object* v_S_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Localization_mkHom(v_M_276_, v_inst_277_, v_S_278_);
lean_dec_ref(v_inst_277_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mkHom(lean_object* v_M_280_, lean_object* v_inst_281_, lean_object* v_S_282_){
_start:
{
lean_object* v___f_283_; 
v___f_283_ = ((lean_object*)(lp_mathlib_Localization_mkHom___closed__0));
return v___f_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_mkHom___boxed(lean_object* v_M_284_, lean_object* v_inst_285_, lean_object* v_S_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_AddLocalization_mkHom(v_M_284_, v_inst_285_, v_S_286_);
lean_dec_ref(v_inst_285_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap___redArg(lean_object* v_f_288_){
_start:
{
lean_inc(v_f_288_);
return v_f_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap___redArg___boxed(lean_object* v_f_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_MonoidHom_toLocalizationMap___redArg(v_f_289_);
lean_dec(v_f_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap(lean_object* v_M_291_, lean_object* v_inst_292_, lean_object* v_S_293_, lean_object* v_N_294_, lean_object* v_inst_295_, lean_object* v_f_296_, lean_object* v_H1_297_, lean_object* v_H2_298_, lean_object* v_H3_299_){
_start:
{
lean_inc(v_f_296_);
return v_f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toLocalizationMap___boxed(lean_object* v_M_300_, lean_object* v_inst_301_, lean_object* v_S_302_, lean_object* v_N_303_, lean_object* v_inst_304_, lean_object* v_f_305_, lean_object* v_H1_306_, lean_object* v_H2_307_, lean_object* v_H3_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_MonoidHom_toLocalizationMap(v_M_300_, v_inst_301_, v_S_302_, v_N_303_, v_inst_304_, v_f_305_, v_H1_306_, v_H2_307_, v_H3_308_);
lean_dec(v_f_305_);
lean_dec_ref(v_inst_304_);
lean_dec_ref(v_inst_301_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap___redArg(lean_object* v_f_310_){
_start:
{
lean_inc(v_f_310_);
return v_f_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap___redArg___boxed(lean_object* v_f_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_AddMonoidHom_toLocalizationMap___redArg(v_f_311_);
lean_dec(v_f_311_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap(lean_object* v_M_313_, lean_object* v_inst_314_, lean_object* v_S_315_, lean_object* v_N_316_, lean_object* v_inst_317_, lean_object* v_f_318_, lean_object* v_H1_319_, lean_object* v_H2_320_, lean_object* v_H3_321_){
_start:
{
lean_inc(v_f_318_);
return v_f_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toLocalizationMap___boxed(lean_object* v_M_322_, lean_object* v_inst_323_, lean_object* v_S_324_, lean_object* v_N_325_, lean_object* v_inst_326_, lean_object* v_f_327_, lean_object* v_H1_328_, lean_object* v_H2_329_, lean_object* v_H3_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_AddMonoidHom_toLocalizationMap(v_M_322_, v_inst_323_, v_S_324_, v_N_325_, v_inst_326_, v_f_327_, v_H1_328_, v_H2_329_, v_H3_330_);
lean_dec(v_f_327_);
lean_dec_ref(v_inst_326_);
lean_dec_ref(v_inst_323_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___redArg(lean_object* v_f_332_){
_start:
{
lean_inc(v_f_332_);
return v_f_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___redArg___boxed(lean_object* v_f_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___redArg(v_f_333_);
lean_dec(v_f_333_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom(lean_object* v_M_335_, lean_object* v_inst_336_, lean_object* v_S_337_, lean_object* v_N_338_, lean_object* v_inst_339_, lean_object* v_f_340_){
_start:
{
lean_inc(v_f_340_);
return v_f_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidHom___boxed(lean_object* v_M_341_, lean_object* v_inst_342_, lean_object* v_S_343_, lean_object* v_N_344_, lean_object* v_inst_345_, lean_object* v_f_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_Submonoid_LocalizationMap_toMonoidHom(v_M_341_, v_inst_342_, v_S_343_, v_N_344_, v_inst_345_, v_f_346_);
lean_dec(v_f_346_);
lean_dec_ref(v_inst_345_);
lean_dec_ref(v_inst_342_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___redArg(lean_object* v_f_348_){
_start:
{
lean_inc(v_f_348_);
return v_f_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___redArg___boxed(lean_object* v_f_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___redArg(v_f_349_);
lean_dec(v_f_349_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom(lean_object* v_M_351_, lean_object* v_inst_352_, lean_object* v_S_353_, lean_object* v_N_354_, lean_object* v_inst_355_, lean_object* v_f_356_){
_start:
{
lean_inc(v_f_356_);
return v_f_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom___boxed(lean_object* v_M_357_, lean_object* v_inst_358_, lean_object* v_S_359_, lean_object* v_N_360_, lean_object* v_inst_361_, lean_object* v_f_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_AddSubmonoid_LocalizationMap_toAddMonoidHom(v_M_357_, v_inst_358_, v_S_359_, v_N_360_, v_inst_361_, v_f_362_);
lean_dec(v_f_362_);
lean_dec_ref(v_inst_361_);
lean_dec_ref(v_inst_358_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike___lam__0(lean_object* v_f_364_, lean_object* v___y_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_apply_1(v_f_364_, v___y_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike(lean_object* v_M_368_, lean_object* v_inst_369_, lean_object* v_S_370_, lean_object* v_N_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v___f_373_; 
v___f_373_ = ((lean_object*)(lp_mathlib_Submonoid_LocalizationMap_instFunLike___closed__0));
return v___f_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instFunLike___boxed(lean_object* v_M_374_, lean_object* v_inst_375_, lean_object* v_S_376_, lean_object* v_N_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_Submonoid_LocalizationMap_instFunLike(v_M_374_, v_inst_375_, v_S_376_, v_N_377_, v_inst_378_);
lean_dec_ref(v_inst_378_);
lean_dec_ref(v_inst_375_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instFunLike(lean_object* v_M_380_, lean_object* v_inst_381_, lean_object* v_S_382_, lean_object* v_N_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___f_385_; 
v___f_385_ = ((lean_object*)(lp_mathlib_Submonoid_LocalizationMap_instFunLike___closed__0));
return v___f_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instFunLike___boxed(lean_object* v_M_386_, lean_object* v_inst_387_, lean_object* v_S_388_, lean_object* v_N_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_AddSubmonoid_LocalizationMap_instFunLike(v_M_386_, v_inst_387_, v_S_388_, v_N_389_, v_inst_390_);
lean_dec_ref(v_inst_390_);
lean_dec_ref(v_inst_387_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___redArg___lam__0(lean_object* v___x_392_, lean_object* v_x_393_){
_start:
{
lean_object* v___x_394_; lean_object* v_toOne_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_402_; 
v___x_394_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_392_);
v_toOne_395_ = lean_ctor_get(v___x_394_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_394_);
if (v_isSharedCheck_402_ == 0)
{
lean_object* v_unused_403_; 
v_unused_403_ = lean_ctor_get(v___x_394_, 1);
lean_dec(v_unused_403_);
v___x_397_ = v___x_394_;
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_toOne_395_);
lean_dec(v___x_394_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_400_; 
if (v_isShared_398_ == 0)
{
lean_ctor_set(v___x_397_, 1, v_toOne_395_);
lean_ctor_set(v___x_397_, 0, v_x_393_);
v___x_400_ = v___x_397_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_x_393_);
lean_ctor_set(v_reuseFailAlloc_401_, 1, v_toOne_395_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___redArg(lean_object* v_inst_404_){
_start:
{
lean_object* v___x_405_; lean_object* v___f_406_; 
v___x_405_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_404_);
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_Localization_monoidOf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_406_, 0, v___x_405_);
return v___f_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___redArg___boxed(lean_object* v_inst_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_Localization_monoidOf___redArg(v_inst_407_);
lean_dec_ref(v_inst_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf(lean_object* v_M_409_, lean_object* v_inst_410_, lean_object* v_S_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_Localization_monoidOf___redArg(v_inst_410_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_monoidOf___boxed(lean_object* v_M_413_, lean_object* v_inst_414_, lean_object* v_S_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Localization_monoidOf(v_M_413_, v_inst_414_, v_S_415_);
lean_dec_ref(v_inst_414_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg___lam__0(lean_object* v___x_417_, lean_object* v_x_418_){
_start:
{
lean_object* v___x_419_; lean_object* v_toZero_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_427_; 
v___x_419_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_417_);
v_toZero_420_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_427_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_427_ == 0)
{
lean_object* v_unused_428_; 
v_unused_428_ = lean_ctor_get(v___x_419_, 1);
lean_dec(v_unused_428_);
v___x_422_ = v___x_419_;
v_isShared_423_ = v_isSharedCheck_427_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_toZero_420_);
lean_dec(v___x_419_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_427_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v___x_425_; 
if (v_isShared_423_ == 0)
{
lean_ctor_set(v___x_422_, 1, v_toZero_420_);
lean_ctor_set(v___x_422_, 0, v_x_418_);
v___x_425_ = v___x_422_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_x_418_);
lean_ctor_set(v_reuseFailAlloc_426_, 1, v_toZero_420_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg(lean_object* v_inst_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___f_431_; 
v___x_430_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_429_);
v___f_431_ = lean_alloc_closure((void*)(lp_mathlib_AddLocalization_addMonoidOf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_431_, 0, v___x_430_);
return v___f_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___redArg___boxed(lean_object* v_inst_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_mathlib_AddLocalization_addMonoidOf___redArg(v_inst_432_);
lean_dec_ref(v_inst_432_);
return v_res_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf(lean_object* v_M_434_, lean_object* v_inst_435_, lean_object* v_S_436_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lp_mathlib_AddLocalization_addMonoidOf___redArg(v_inst_435_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_addMonoidOf___boxed(lean_object* v_M_438_, lean_object* v_inst_439_, lean_object* v_S_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_AddLocalization_addMonoidOf(v_M_438_, v_inst_439_, v_S_440_);
lean_dec_ref(v_inst_439_);
return v_res_441_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Localization_decidableEq___redArg___lam__0(lean_object* v_toMul_442_, lean_object* v_inst_443_, lean_object* v_x_444_, lean_object* v_x_445_, lean_object* v_x_446_, lean_object* v_x_447_){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; uint8_t v___x_451_; 
lean_inc(v_toMul_442_);
v___x_448_ = lean_apply_2(v_toMul_442_, v_x_447_, v_x_444_);
v___x_449_ = lean_apply_2(v_toMul_442_, v_x_446_, v_x_445_);
v___x_450_ = lean_apply_2(v_inst_443_, v___x_448_, v___x_449_);
v___x_451_ = lean_unbox(v___x_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_decidableEq___redArg___lam__0___boxed(lean_object* v_toMul_452_, lean_object* v_inst_453_, lean_object* v_x_454_, lean_object* v_x_455_, lean_object* v_x_456_, lean_object* v_x_457_){
_start:
{
uint8_t v_res_458_; lean_object* v_r_459_; 
v_res_458_ = lp_mathlib_Localization_decidableEq___redArg___lam__0(v_toMul_452_, v_inst_453_, v_x_454_, v_x_455_, v_x_456_, v_x_457_);
v_r_459_ = lean_box(v_res_458_);
return v_r_459_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Localization_decidableEq___redArg(lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_a_462_, lean_object* v_b_463_){
_start:
{
lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v_toMul_466_; lean_object* v___f_467_; lean_object* v___x_468_; uint8_t v___x_469_; 
v___x_464_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_460_);
v___x_465_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_464_);
v_toMul_466_ = lean_ctor_get(v___x_465_, 1);
lean_inc(v_toMul_466_);
lean_dec_ref(v___x_465_);
v___f_467_ = lean_alloc_closure((void*)(lp_mathlib_Localization_decidableEq___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_467_, 0, v_toMul_466_);
lean_closure_set(v___f_467_, 1, v_inst_461_);
v___x_468_ = lp_mathlib_Localization_recOnSubsingleton_u2082___redArg(v_a_462_, v_b_463_, v___f_467_);
v___x_469_ = lean_unbox(v___x_468_);
lean_dec(v___x_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_decidableEq___redArg___boxed(lean_object* v_inst_470_, lean_object* v_inst_471_, lean_object* v_a_472_, lean_object* v_b_473_){
_start:
{
uint8_t v_res_474_; lean_object* v_r_475_; 
v_res_474_ = lp_mathlib_Localization_decidableEq___redArg(v_inst_470_, v_inst_471_, v_a_472_, v_b_473_);
lean_dec_ref(v_inst_470_);
v_r_475_ = lean_box(v_res_474_);
return v_r_475_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Localization_decidableEq(lean_object* v_00_u03b1_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_s_479_, lean_object* v_inst_480_, lean_object* v_a_481_, lean_object* v_b_482_){
_start:
{
uint8_t v___x_483_; 
v___x_483_ = lp_mathlib_Localization_decidableEq___redArg(v_inst_477_, v_inst_480_, v_a_481_, v_b_482_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_decidableEq___boxed(lean_object* v_00_u03b1_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_s_487_, lean_object* v_inst_488_, lean_object* v_a_489_, lean_object* v_b_490_){
_start:
{
uint8_t v_res_491_; lean_object* v_r_492_; 
v_res_491_ = lp_mathlib_Localization_decidableEq(v_00_u03b1_484_, v_inst_485_, v_inst_486_, v_s_487_, v_inst_488_, v_a_489_, v_b_490_);
lean_dec_ref(v_inst_485_);
v_r_492_ = lean_box(v_res_491_);
return v_r_492_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddLocalization_decidableEq___redArg___lam__0(lean_object* v_toAdd_493_, lean_object* v_inst_494_, lean_object* v_x_495_, lean_object* v_x_496_, lean_object* v_x_497_, lean_object* v_x_498_){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; uint8_t v___x_502_; 
lean_inc(v_toAdd_493_);
v___x_499_ = lean_apply_2(v_toAdd_493_, v_x_498_, v_x_495_);
v___x_500_ = lean_apply_2(v_toAdd_493_, v_x_497_, v_x_496_);
v___x_501_ = lean_apply_2(v_inst_494_, v___x_499_, v___x_500_);
v___x_502_ = lean_unbox(v___x_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_decidableEq___redArg___lam__0___boxed(lean_object* v_toAdd_503_, lean_object* v_inst_504_, lean_object* v_x_505_, lean_object* v_x_506_, lean_object* v_x_507_, lean_object* v_x_508_){
_start:
{
uint8_t v_res_509_; lean_object* v_r_510_; 
v_res_509_ = lp_mathlib_AddLocalization_decidableEq___redArg___lam__0(v_toAdd_503_, v_inst_504_, v_x_505_, v_x_506_, v_x_507_, v_x_508_);
v_r_510_ = lean_box(v_res_509_);
return v_r_510_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddLocalization_decidableEq___redArg(lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_a_513_, lean_object* v_b_514_){
_start:
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v_toAdd_517_; lean_object* v___f_518_; lean_object* v___x_519_; uint8_t v___x_520_; 
v___x_515_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_511_);
v___x_516_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_515_);
v_toAdd_517_ = lean_ctor_get(v___x_516_, 1);
lean_inc(v_toAdd_517_);
lean_dec_ref(v___x_516_);
v___f_518_ = lean_alloc_closure((void*)(lp_mathlib_AddLocalization_decidableEq___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_518_, 0, v_toAdd_517_);
lean_closure_set(v___f_518_, 1, v_inst_512_);
v___x_519_ = lp_mathlib_AddLocalization_recOnSubsingleton_u2082___redArg(v_a_513_, v_b_514_, v___f_518_);
v___x_520_ = lean_unbox(v___x_519_);
lean_dec(v___x_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_decidableEq___redArg___boxed(lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_a_523_, lean_object* v_b_524_){
_start:
{
uint8_t v_res_525_; lean_object* v_r_526_; 
v_res_525_ = lp_mathlib_AddLocalization_decidableEq___redArg(v_inst_521_, v_inst_522_, v_a_523_, v_b_524_);
lean_dec_ref(v_inst_521_);
v_r_526_ = lean_box(v_res_525_);
return v_r_526_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddLocalization_decidableEq(lean_object* v_00_u03b1_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_s_530_, lean_object* v_inst_531_, lean_object* v_a_532_, lean_object* v_b_533_){
_start:
{
uint8_t v___x_534_; 
v___x_534_ = lp_mathlib_AddLocalization_decidableEq___redArg(v_inst_528_, v_inst_531_, v_a_532_, v_b_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddLocalization_decidableEq___boxed(lean_object* v_00_u03b1_535_, lean_object* v_inst_536_, lean_object* v_inst_537_, lean_object* v_s_538_, lean_object* v_inst_539_, lean_object* v_a_540_, lean_object* v_b_541_){
_start:
{
uint8_t v_res_542_; lean_object* v_r_543_; 
v_res_542_ = lp_mathlib_AddLocalization_decidableEq(v_00_u03b1_535_, v_inst_536_, v_inst_537_, v_s_538_, v_inst_539_, v_a_540_, v_b_541_);
lean_dec_ref(v_inst_536_);
v_r_543_ = lean_box(v_res_542_);
return v_r_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap___redArg(lean_object* v_inst_544_){
_start:
{
lean_object* v___x_545_; 
v___x_545_ = lp_mathlib_Localization_monoidOf___redArg(v_inst_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap___redArg___boxed(lean_object* v_inst_546_){
_start:
{
lean_object* v_res_547_; 
v_res_547_ = lp_mathlib_OreLocalization_localizationMap___redArg(v_inst_546_);
lean_dec_ref(v_inst_546_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap(lean_object* v_R_548_, lean_object* v_inst_549_, lean_object* v_S_550_){
_start:
{
lean_object* v___x_551_; 
v___x_551_ = lp_mathlib_Localization_monoidOf___redArg(v_inst_549_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_localizationMap___boxed(lean_object* v_R_552_, lean_object* v_inst_553_, lean_object* v_S_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_OreLocalization_localizationMap(v_R_552_, v_inst_553_, v_S_554_);
lean_dec_ref(v_inst_553_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___redArg(lean_object* v_inst_556_){
_start:
{
lean_inc_ref(v_inst_556_);
return v_inst_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___redArg___boxed(lean_object* v_inst_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___redArg(v_inst_557_);
lean_dec_ref(v_inst_557_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid(lean_object* v_M_559_, lean_object* v_N_560_, lean_object* v_inst_561_, lean_object* v_S_562_, lean_object* v_inst_563_, lean_object* v_f_564_){
_start:
{
lean_inc_ref(v_inst_563_);
return v_inst_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid___boxed(lean_object* v_M_565_, lean_object* v_N_566_, lean_object* v_inst_567_, lean_object* v_S_568_, lean_object* v_inst_569_, lean_object* v_f_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib_Submonoid_LocalizationMap_cancelCommMonoid(v_M_565_, v_N_566_, v_inst_567_, v_S_568_, v_inst_569_, v_f_570_);
lean_dec(v_f_570_);
lean_dec_ref(v_inst_569_);
lean_dec_ref(v_inst_567_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___redArg(lean_object* v_inst_572_){
_start:
{
lean_inc_ref(v_inst_572_);
return v_inst_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___redArg___boxed(lean_object* v_inst_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___redArg(v_inst_573_);
lean_dec_ref(v_inst_573_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid(lean_object* v_M_575_, lean_object* v_N_576_, lean_object* v_inst_577_, lean_object* v_S_578_, lean_object* v_inst_579_, lean_object* v_f_580_){
_start:
{
lean_inc_ref(v_inst_579_);
return v_inst_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid___boxed(lean_object* v_M_581_, lean_object* v_N_582_, lean_object* v_inst_583_, lean_object* v_S_584_, lean_object* v_inst_585_, lean_object* v_f_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib_AddSubmonoid_LocalizationMap_addCancelCommMonoid(v_M_581_, v_N_582_, v_inst_583_, v_S_584_, v_inst_585_, v_f_586_);
lean_dec(v_f_586_);
lean_dec_ref(v_inst_585_);
lean_dec_ref(v_inst_583_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instCancelCommMonoidLocalization___redArg(lean_object* v_inst_588_, lean_object* v_S_589_){
_start:
{
lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_590_ = lp_mathlib_OreLocalization_oreSetComm(lean_box(0), v_inst_588_, v_S_589_);
v___x_591_ = lp_mathlib_OreLocalization_instMonoid___redArg(v_inst_588_, v_S_589_, v___x_590_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_instCancelCommMonoidLocalization(lean_object* v_M_592_, lean_object* v_inst_593_, lean_object* v_S_594_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lp_mathlib_Submonoid_LocalizationMap_instCancelCommMonoidLocalization___redArg(v_inst_593_, v_S_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instAddCancelCommMonoidLocalization___redArg(lean_object* v_inst_596_, lean_object* v_S_597_){
_start:
{
lean_object* v___x_598_; lean_object* v___x_599_; 
v___x_598_ = lp_mathlib_AddOreLocalization_addOreSetComm(lean_box(0), v_inst_596_, v_S_597_);
v___x_599_ = lp_mathlib_AddOreLocalization_instAddMonoid___redArg(v_inst_596_, v_S_597_, v___x_598_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_LocalizationMap_instAddCancelCommMonoidLocalization(lean_object* v_M_600_, lean_object* v_inst_601_, lean_object* v_S_602_){
_start:
{
lean_object* v___x_603_; 
v___x_603_ = lp_mathlib_AddSubmonoid_LocalizationMap_instAddCancelCommMonoidLocalization___redArg(v_inst_601_, v_S_602_);
return v___x_603_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
