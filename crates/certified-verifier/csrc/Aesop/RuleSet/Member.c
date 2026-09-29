// Lean compiler output
// Module: Aesop.RuleSet.Member
// Imports: public import Init public meta import Init public import Aesop.Rule
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
extern lean_object* lp_aesop_Aesop_instInhabitedNormRuleInfo_default;
lean_object* lp_aesop_Aesop_instInhabitedRule_default___redArg(lean_object*);
lean_object* lp_aesop_Aesop_UnfoldRule_name(lean_object*);
lean_object* lp_aesop_Aesop_LocalNormSimpRule_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unfoldRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unfoldRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normForwardRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normForwardRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeForwardRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeForwardRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeForwardRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeForwardRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSetMember;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_name___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_base_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_base_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_normSimpRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_normSimpRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedGlobalRuleSetMember;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_global_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_global_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_localNormSimpRule_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_localNormSimpRule_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedLocalRuleSetMember;
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_name___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_toGlobalRuleSetMember_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
case 3:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
case 4:
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(4u);
return v___x_6_;
}
case 5:
{
lean_object* v___x_7_; 
v___x_7_ = lean_unsigned_to_nat(5u);
return v___x_7_;
}
default: 
{
lean_object* v___x_8_; 
v___x_8_ = lean_unsigned_to_nat(6u);
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorIdx___boxed(lean_object* v_x_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_aesop_Aesop_BaseRuleSetMember_ctorIdx(v_x_9_);
lean_dec_ref(v_x_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(lean_object* v_t_11_, lean_object* v_k_12_){
_start:
{
switch(lean_obj_tag(v_t_11_))
{
case 4:
{
lean_object* v_r_u2081_13_; lean_object* v_r_u2082_14_; lean_object* v___x_15_; 
v_r_u2081_13_ = lean_ctor_get(v_t_11_, 0);
lean_inc_ref(v_r_u2081_13_);
v_r_u2082_14_ = lean_ctor_get(v_t_11_, 1);
lean_inc_ref(v_r_u2082_14_);
lean_dec_ref_known(v_t_11_, 2);
v___x_15_ = lean_apply_2(v_k_12_, v_r_u2081_13_, v_r_u2082_14_);
return v___x_15_;
}
case 5:
{
lean_object* v_r_u2081_16_; lean_object* v_r_u2082_17_; lean_object* v___x_18_; 
v_r_u2081_16_ = lean_ctor_get(v_t_11_, 0);
lean_inc_ref(v_r_u2081_16_);
v_r_u2082_17_ = lean_ctor_get(v_t_11_, 1);
lean_inc_ref(v_r_u2082_17_);
lean_dec_ref_known(v_t_11_, 2);
v___x_18_ = lean_apply_2(v_k_12_, v_r_u2081_16_, v_r_u2082_17_);
return v___x_18_;
}
case 6:
{
lean_object* v_r_u2081_19_; lean_object* v_r_u2082_20_; lean_object* v___x_21_; 
v_r_u2081_19_ = lean_ctor_get(v_t_11_, 0);
lean_inc_ref(v_r_u2081_19_);
v_r_u2082_20_ = lean_ctor_get(v_t_11_, 1);
lean_inc_ref(v_r_u2082_20_);
lean_dec_ref_known(v_t_11_, 2);
v___x_21_ = lean_apply_2(v_k_12_, v_r_u2081_19_, v_r_u2082_20_);
return v___x_21_;
}
default: 
{
lean_object* v_r_22_; lean_object* v___x_23_; 
v_r_22_ = lean_ctor_get(v_t_11_, 0);
lean_inc_ref(v_r_22_);
lean_dec_ref(v_t_11_);
v___x_23_ = lean_apply_1(v_k_12_, v_r_22_);
return v___x_23_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorElim(lean_object* v_motive_24_, lean_object* v_ctorIdx_25_, lean_object* v_t_26_, lean_object* v_h_27_, lean_object* v_k_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_26_, v_k_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_ctorElim___boxed(lean_object* v_motive_30_, lean_object* v_ctorIdx_31_, lean_object* v_t_32_, lean_object* v_h_33_, lean_object* v_k_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim(v_motive_30_, v_ctorIdx_31_, v_t_32_, v_h_33_, v_k_34_);
lean_dec(v_ctorIdx_31_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normRule_elim___redArg(lean_object* v_t_36_, lean_object* v_normRule_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_36_, v_normRule_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normRule_elim(lean_object* v_motive_39_, lean_object* v_t_40_, lean_object* v_h_41_, lean_object* v_normRule_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_40_, v_normRule_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeRule_elim___redArg(lean_object* v_t_44_, lean_object* v_unsafeRule_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_44_, v_unsafeRule_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeRule_elim(lean_object* v_motive_47_, lean_object* v_t_48_, lean_object* v_h_49_, lean_object* v_unsafeRule_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_48_, v_unsafeRule_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeRule_elim___redArg(lean_object* v_t_52_, lean_object* v_safeRule_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_52_, v_safeRule_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeRule_elim(lean_object* v_motive_55_, lean_object* v_t_56_, lean_object* v_h_57_, lean_object* v_safeRule_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_56_, v_safeRule_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unfoldRule_elim___redArg(lean_object* v_t_60_, lean_object* v_unfoldRule_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_60_, v_unfoldRule_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unfoldRule_elim(lean_object* v_motive_63_, lean_object* v_t_64_, lean_object* v_h_65_, lean_object* v_unfoldRule_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_64_, v_unfoldRule_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normForwardRule_elim___redArg(lean_object* v_t_68_, lean_object* v_normForwardRule_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_68_, v_normForwardRule_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_normForwardRule_elim(lean_object* v_motive_71_, lean_object* v_t_72_, lean_object* v_h_73_, lean_object* v_normForwardRule_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_72_, v_normForwardRule_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeForwardRule_elim___redArg(lean_object* v_t_76_, lean_object* v_unsafeForwardRule_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_76_, v_unsafeForwardRule_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_unsafeForwardRule_elim(lean_object* v_motive_79_, lean_object* v_t_80_, lean_object* v_h_81_, lean_object* v_unsafeForwardRule_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_80_, v_unsafeForwardRule_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeForwardRule_elim___redArg(lean_object* v_t_84_, lean_object* v_safeForwardRule_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_84_, v_safeForwardRule_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_safeForwardRule_elim(lean_object* v_motive_87_, lean_object* v_t_88_, lean_object* v_h_89_, lean_object* v_safeForwardRule_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_aesop_Aesop_BaseRuleSetMember_ctorElim___redArg(v_t_88_, v_safeForwardRule_90_);
return v___x_91_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__0(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = lp_aesop_Aesop_instInhabitedNormRuleInfo_default;
v___x_93_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v___x_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__1(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__0, &lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__0);
v___x_95_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
return v___x_95_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default(void){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__1, &lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default___closed__1);
return v___x_96_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember(void){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default;
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_name(lean_object* v_x_98_){
_start:
{
switch(lean_obj_tag(v_x_98_))
{
case 3:
{
lean_object* v_r_99_; lean_object* v___x_100_; 
v_r_99_ = lean_ctor_get(v_x_98_, 0);
v___x_100_ = lp_aesop_Aesop_UnfoldRule_name(v_r_99_);
return v___x_100_;
}
case 4:
{
lean_object* v_r_u2081_101_; lean_object* v_name_102_; 
v_r_u2081_101_ = lean_ctor_get(v_x_98_, 0);
v_name_102_ = lean_ctor_get(v_r_u2081_101_, 1);
lean_inc_ref(v_name_102_);
return v_name_102_;
}
case 5:
{
lean_object* v_r_u2081_103_; lean_object* v_name_104_; 
v_r_u2081_103_ = lean_ctor_get(v_x_98_, 0);
v_name_104_ = lean_ctor_get(v_r_u2081_103_, 1);
lean_inc_ref(v_name_104_);
return v_name_104_;
}
case 6:
{
lean_object* v_r_u2081_105_; lean_object* v_name_106_; 
v_r_u2081_105_ = lean_ctor_get(v_x_98_, 0);
v_name_106_ = lean_ctor_get(v_r_u2081_105_, 1);
lean_inc_ref(v_name_106_);
return v_name_106_;
}
default: 
{
lean_object* v_r_107_; lean_object* v_name_108_; 
v_r_107_ = lean_ctor_get(v_x_98_, 0);
v_name_108_ = lean_ctor_get(v_r_107_, 0);
lean_inc_ref(v_name_108_);
return v_name_108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseRuleSetMember_name___boxed(lean_object* v_x_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_aesop_Aesop_BaseRuleSetMember_name(v_x_109_);
lean_dec_ref(v_x_109_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorIdx(lean_object* v_x_111_){
_start:
{
if (lean_obj_tag(v_x_111_) == 0)
{
lean_object* v___x_112_; 
v___x_112_ = lean_unsigned_to_nat(0u);
return v___x_112_;
}
else
{
lean_object* v___x_113_; 
v___x_113_ = lean_unsigned_to_nat(1u);
return v___x_113_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorIdx___boxed(lean_object* v_x_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorIdx(v_x_114_);
lean_dec_ref(v_x_114_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(lean_object* v_t_116_, lean_object* v_k_117_){
_start:
{
lean_object* v_m_118_; lean_object* v___x_119_; 
v_m_118_ = lean_ctor_get(v_t_116_, 0);
lean_inc_ref(v_m_118_);
lean_dec_ref(v_t_116_);
v___x_119_ = lean_apply_1(v_k_117_, v_m_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorElim(lean_object* v_motive_120_, lean_object* v_ctorIdx_121_, lean_object* v_t_122_, lean_object* v_h_123_, lean_object* v_k_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(v_t_122_, v_k_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___boxed(lean_object* v_motive_126_, lean_object* v_ctorIdx_127_, lean_object* v_t_128_, lean_object* v_h_129_, lean_object* v_k_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorElim(v_motive_126_, v_ctorIdx_127_, v_t_128_, v_h_129_, v_k_130_);
lean_dec(v_ctorIdx_127_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_base_elim___redArg(lean_object* v_t_132_, lean_object* v_base_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(v_t_132_, v_base_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_base_elim(lean_object* v_motive_135_, lean_object* v_t_136_, lean_object* v_h_137_, lean_object* v_base_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(v_t_136_, v_base_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_normSimpRule_elim___redArg(lean_object* v_t_140_, lean_object* v_normSimpRule_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(v_t_140_, v_normSimpRule_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_normSimpRule_elim(lean_object* v_motive_143_, lean_object* v_t_144_, lean_object* v_h_145_, lean_object* v_normSimpRule_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_aesop_Aesop_GlobalRuleSetMember_ctorElim___redArg(v_t_144_, v_normSimpRule_146_);
return v___x_147_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default___closed__0(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_148_ = lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default;
v___x_149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
return v___x_149_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default(void){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default___closed__0, &lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default___closed__0);
return v___x_150_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedGlobalRuleSetMember(void){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default;
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name(lean_object* v_x_152_){
_start:
{
if (lean_obj_tag(v_x_152_) == 0)
{
lean_object* v_m_153_; lean_object* v___x_154_; 
v_m_153_ = lean_ctor_get(v_x_152_, 0);
v___x_154_ = lp_aesop_Aesop_BaseRuleSetMember_name(v_m_153_);
return v___x_154_;
}
else
{
lean_object* v_e_155_; lean_object* v_name_156_; 
v_e_155_ = lean_ctor_get(v_x_152_, 0);
v_name_156_ = lean_ctor_get(v_e_155_, 0);
lean_inc_ref(v_name_156_);
return v_name_156_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name___boxed(lean_object* v_x_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_aesop_Aesop_GlobalRuleSetMember_name(v_x_157_);
lean_dec_ref(v_x_157_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorIdx(lean_object* v_x_159_){
_start:
{
if (lean_obj_tag(v_x_159_) == 0)
{
lean_object* v___x_160_; 
v___x_160_ = lean_unsigned_to_nat(0u);
return v___x_160_;
}
else
{
lean_object* v___x_161_; 
v___x_161_ = lean_unsigned_to_nat(1u);
return v___x_161_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorIdx___boxed(lean_object* v_x_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_aesop_Aesop_LocalRuleSetMember_ctorIdx(v_x_162_);
lean_dec_ref(v_x_162_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(lean_object* v_t_164_, lean_object* v_k_165_){
_start:
{
lean_object* v_m_166_; lean_object* v___x_167_; 
v_m_166_ = lean_ctor_get(v_t_164_, 0);
lean_inc_ref(v_m_166_);
lean_dec_ref(v_t_164_);
v___x_167_ = lean_apply_1(v_k_165_, v_m_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorElim(lean_object* v_motive_168_, lean_object* v_ctorIdx_169_, lean_object* v_t_170_, lean_object* v_h_171_, lean_object* v_k_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(v_t_170_, v_k_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_ctorElim___boxed(lean_object* v_motive_174_, lean_object* v_ctorIdx_175_, lean_object* v_t_176_, lean_object* v_h_177_, lean_object* v_k_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_aesop_Aesop_LocalRuleSetMember_ctorElim(v_motive_174_, v_ctorIdx_175_, v_t_176_, v_h_177_, v_k_178_);
lean_dec(v_ctorIdx_175_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_global_elim___redArg(lean_object* v_t_180_, lean_object* v_global_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(v_t_180_, v_global_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_global_elim(lean_object* v_motive_183_, lean_object* v_t_184_, lean_object* v_h_185_, lean_object* v_global_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(v_t_184_, v_global_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_localNormSimpRule_elim___redArg(lean_object* v_t_188_, lean_object* v_localNormSimpRule_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(v_t_188_, v_localNormSimpRule_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_localNormSimpRule_elim(lean_object* v_motive_191_, lean_object* v_t_192_, lean_object* v_h_193_, lean_object* v_localNormSimpRule_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_aesop_Aesop_LocalRuleSetMember_ctorElim___redArg(v_t_192_, v_localNormSimpRule_194_);
return v___x_195_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default___closed__0(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_196_ = lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default;
v___x_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
return v___x_197_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default(void){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default___closed__0, &lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default___closed__0);
return v___x_198_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedLocalRuleSetMember(void){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default;
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_name(lean_object* v_x_200_){
_start:
{
if (lean_obj_tag(v_x_200_) == 0)
{
lean_object* v_m_201_; lean_object* v___x_202_; 
v_m_201_ = lean_ctor_get(v_x_200_, 0);
v___x_202_ = lp_aesop_Aesop_GlobalRuleSetMember_name(v_m_201_);
return v___x_202_;
}
else
{
lean_object* v_r_203_; lean_object* v___x_204_; 
v_r_203_ = lean_ctor_get(v_x_200_, 0);
v___x_204_ = lp_aesop_Aesop_LocalNormSimpRule_name(v_r_203_);
return v___x_204_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_name___boxed(lean_object* v_x_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_aesop_Aesop_LocalRuleSetMember_name(v_x_205_);
lean_dec_ref(v_x_205_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSetMember_toGlobalRuleSetMember_x3f(lean_object* v_x_207_){
_start:
{
if (lean_obj_tag(v_x_207_) == 0)
{
lean_object* v_m_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_215_; 
v_m_208_ = lean_ctor_get(v_x_207_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v_x_207_);
if (v_isSharedCheck_215_ == 0)
{
v___x_210_ = v_x_207_;
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_m_208_);
lean_dec(v_x_207_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_213_; 
if (v_isShared_211_ == 0)
{
lean_ctor_set_tag(v___x_210_, 1);
v___x_213_ = v___x_210_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_214_; 
v_reuseFailAlloc_214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_214_, 0, v_m_208_);
v___x_213_ = v_reuseFailAlloc_214_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
return v___x_213_;
}
}
}
else
{
lean_object* v___x_216_; 
lean_dec_ref(v_x_207_);
v___x_216_ = lean_box(0);
return v___x_216_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleSet_Member(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default = _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default);
lp_aesop_Aesop_instInhabitedBaseRuleSetMember = _init_lp_aesop_Aesop_instInhabitedBaseRuleSetMember();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedBaseRuleSetMember);
lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default = _init_lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGlobalRuleSetMember_default);
lp_aesop_Aesop_instInhabitedGlobalRuleSetMember = _init_lp_aesop_Aesop_instInhabitedGlobalRuleSetMember();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedGlobalRuleSetMember);
lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default = _init_lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedLocalRuleSetMember_default);
lp_aesop_Aesop_instInhabitedLocalRuleSetMember = _init_lp_aesop_Aesop_instInhabitedLocalRuleSetMember();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedLocalRuleSetMember);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleSet_Member(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Rule(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleSet_Member(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet_Member(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleSet_Member(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleSet_Member(builtin);
}
#ifdef __cplusplus
}
#endif
