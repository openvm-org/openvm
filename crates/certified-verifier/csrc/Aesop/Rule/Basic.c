// Lean compiler output
// Module: Aesop.Rule.Basic
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Descr
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
uint8_t lp_aesop_Aesop_RuleName_compare(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedRuleName_default;
extern lean_object* lp_aesop_Aesop_instInhabitedRuleTacDescr_default;
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule_default___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule_default(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_instBEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rule_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rule_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rule_instBEq___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instBEq(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_instOrd___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rule_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rule_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rule_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rule_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instOrd(lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_Rule_instHashable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instHashable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Rule_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Rule_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_Rule_instHashable___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instHashable(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriority___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriority___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriority(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriority___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByName___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByName___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByName(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByName___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriorityThenName(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriorityThenName___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_mapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_mapM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_mapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule_default___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_2_ = lp_aesop_Aesop_instInhabitedRuleName_default;
v___x_3_ = lean_box(0);
v___x_4_ = lean_box(0);
v___x_5_ = lp_aesop_Aesop_instInhabitedRuleTacDescr_default;
v___x_6_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_6_, 0, v___x_2_);
lean_ctor_set(v___x_6_, 1, v___x_3_);
lean_ctor_set(v___x_6_, 2, v___x_4_);
lean_ctor_set(v___x_6_, 3, v_inst_1_);
lean_ctor_set(v___x_6_, 4, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule_default(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v_inst_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v_inst_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRule(lean_object* v_a_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_aesop_Aesop_instInhabitedRule_default___redArg(v_inst_13_);
return v___x_14_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_instBEq___lam__0(lean_object* v_r_15_, lean_object* v_s_16_){
_start:
{
lean_object* v_name_17_; lean_object* v_name_18_; lean_object* v_name_19_; uint8_t v_builder_20_; uint8_t v_phase_21_; uint8_t v_scope_22_; uint64_t v_hash_23_; lean_object* v_name_24_; uint8_t v_builder_25_; uint8_t v_phase_26_; uint8_t v_scope_27_; uint64_t v_hash_28_; uint8_t v___x_29_; 
v_name_17_ = lean_ctor_get(v_r_15_, 0);
v_name_18_ = lean_ctor_get(v_s_16_, 0);
v_name_19_ = lean_ctor_get(v_name_17_, 0);
v_builder_20_ = lean_ctor_get_uint8(v_name_17_, sizeof(void*)*1 + 8);
v_phase_21_ = lean_ctor_get_uint8(v_name_17_, sizeof(void*)*1 + 9);
v_scope_22_ = lean_ctor_get_uint8(v_name_17_, sizeof(void*)*1 + 10);
v_hash_23_ = lean_ctor_get_uint64(v_name_17_, sizeof(void*)*1);
v_name_24_ = lean_ctor_get(v_name_18_, 0);
v_builder_25_ = lean_ctor_get_uint8(v_name_18_, sizeof(void*)*1 + 8);
v_phase_26_ = lean_ctor_get_uint8(v_name_18_, sizeof(void*)*1 + 9);
v_scope_27_ = lean_ctor_get_uint8(v_name_18_, sizeof(void*)*1 + 10);
v_hash_28_ = lean_ctor_get_uint64(v_name_18_, sizeof(void*)*1);
v___x_29_ = lean_uint64_dec_eq(v_hash_23_, v_hash_28_);
if (v___x_29_ == 0)
{
return v___x_29_;
}
else
{
uint8_t v___x_30_; 
v___x_30_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_20_, v_builder_25_);
if (v___x_30_ == 0)
{
return v___x_30_;
}
else
{
uint8_t v___x_31_; 
v___x_31_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_21_, v_phase_26_);
if (v___x_31_ == 0)
{
return v___x_31_;
}
else
{
uint8_t v___x_32_; 
v___x_32_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_22_, v_scope_27_);
if (v___x_32_ == 0)
{
return v___x_32_;
}
else
{
uint8_t v___x_33_; 
v___x_33_ = lean_name_eq(v_name_19_, v_name_24_);
return v___x_33_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instBEq___lam__0___boxed(lean_object* v_r_34_, lean_object* v_s_35_){
_start:
{
uint8_t v_res_36_; lean_object* v_r_37_; 
v_res_36_ = lp_aesop_Aesop_Rule_instBEq___lam__0(v_r_34_, v_s_35_);
lean_dec_ref(v_s_35_);
lean_dec_ref(v_r_34_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instBEq(lean_object* v_00_u03b1_39_){
_start:
{
lean_object* v___f_40_; 
v___f_40_ = ((lean_object*)(lp_aesop_Aesop_Rule_instBEq___closed__0));
return v___f_40_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_instOrd___lam__0(lean_object* v_r_41_, lean_object* v_s_42_){
_start:
{
lean_object* v_name_43_; lean_object* v_name_44_; uint8_t v___x_45_; 
v_name_43_ = lean_ctor_get(v_r_41_, 0);
v_name_44_ = lean_ctor_get(v_s_42_, 0);
v___x_45_ = lp_aesop_Aesop_RuleName_compare(v_name_43_, v_name_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instOrd___lam__0___boxed(lean_object* v_r_46_, lean_object* v_s_47_){
_start:
{
uint8_t v_res_48_; lean_object* v_r_49_; 
v_res_48_ = lp_aesop_Aesop_Rule_instOrd___lam__0(v_r_46_, v_s_47_);
lean_dec_ref(v_s_47_);
lean_dec_ref(v_r_46_);
v_r_49_ = lean_box(v_res_48_);
return v_r_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instOrd(lean_object* v_00_u03b1_51_){
_start:
{
lean_object* v___f_52_; 
v___f_52_ = ((lean_object*)(lp_aesop_Aesop_Rule_instOrd___closed__0));
return v___f_52_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Rule_instHashable___lam__0(lean_object* v_r_53_){
_start:
{
lean_object* v_name_54_; uint64_t v_hash_55_; 
v_name_54_ = lean_ctor_get(v_r_53_, 0);
v_hash_55_ = lean_ctor_get_uint64(v_name_54_, sizeof(void*)*1);
return v_hash_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instHashable___lam__0___boxed(lean_object* v_r_56_){
_start:
{
uint64_t v_res_57_; lean_object* v_r_58_; 
v_res_57_ = lp_aesop_Aesop_Rule_instHashable___lam__0(v_r_56_);
lean_dec_ref(v_r_56_);
v_r_58_ = lean_box_uint64(v_res_57_);
return v_r_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_instHashable(lean_object* v_00_u03b1_60_){
_start:
{
lean_object* v___f_61_; 
v___f_61_ = ((lean_object*)(lp_aesop_Aesop_Rule_instHashable___closed__0));
return v___f_61_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriority___redArg(lean_object* v_inst_62_, lean_object* v_r_63_, lean_object* v_s_64_){
_start:
{
lean_object* v_extra_65_; lean_object* v_extra_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v_extra_65_ = lean_ctor_get(v_r_63_, 3);
lean_inc(v_extra_65_);
lean_dec_ref(v_r_63_);
v_extra_66_ = lean_ctor_get(v_s_64_, 3);
lean_inc(v_extra_66_);
lean_dec_ref(v_s_64_);
v___x_67_ = lean_apply_2(v_inst_62_, v_extra_65_, v_extra_66_);
v___x_68_ = lean_unbox(v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriority___redArg___boxed(lean_object* v_inst_69_, lean_object* v_r_70_, lean_object* v_s_71_){
_start:
{
uint8_t v_res_72_; lean_object* v_r_73_; 
v_res_72_ = lp_aesop_Aesop_Rule_compareByPriority___redArg(v_inst_69_, v_r_70_, v_s_71_);
v_r_73_ = lean_box(v_res_72_);
return v_r_73_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriority(lean_object* v_00_u03b1_74_, lean_object* v_inst_75_, lean_object* v_r_76_, lean_object* v_s_77_){
_start:
{
uint8_t v___x_78_; 
v___x_78_ = lp_aesop_Aesop_Rule_compareByPriority___redArg(v_inst_75_, v_r_76_, v_s_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriority___boxed(lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_r_81_, lean_object* v_s_82_){
_start:
{
uint8_t v_res_83_; lean_object* v_r_84_; 
v_res_83_ = lp_aesop_Aesop_Rule_compareByPriority(v_00_u03b1_79_, v_inst_80_, v_r_81_, v_s_82_);
v_r_84_ = lean_box(v_res_83_);
return v_r_84_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByName___redArg(lean_object* v_r_85_, lean_object* v_s_86_){
_start:
{
lean_object* v_name_87_; lean_object* v_name_88_; uint8_t v___x_89_; 
v_name_87_ = lean_ctor_get(v_r_85_, 0);
v_name_88_ = lean_ctor_get(v_s_86_, 0);
v___x_89_ = lp_aesop_Aesop_RuleName_compare(v_name_87_, v_name_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByName___redArg___boxed(lean_object* v_r_90_, lean_object* v_s_91_){
_start:
{
uint8_t v_res_92_; lean_object* v_r_93_; 
v_res_92_ = lp_aesop_Aesop_Rule_compareByName___redArg(v_r_90_, v_s_91_);
lean_dec_ref(v_s_91_);
lean_dec_ref(v_r_90_);
v_r_93_ = lean_box(v_res_92_);
return v_r_93_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByName(lean_object* v_00_u03b1_94_, lean_object* v_r_95_, lean_object* v_s_96_){
_start:
{
uint8_t v___x_97_; 
v___x_97_ = lp_aesop_Aesop_Rule_compareByName___redArg(v_r_95_, v_s_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByName___boxed(lean_object* v_00_u03b1_98_, lean_object* v_r_99_, lean_object* v_s_100_){
_start:
{
uint8_t v_res_101_; lean_object* v_r_102_; 
v_res_101_ = lp_aesop_Aesop_Rule_compareByName(v_00_u03b1_98_, v_r_99_, v_s_100_);
lean_dec_ref(v_s_100_);
lean_dec_ref(v_r_99_);
v_r_102_ = lean_box(v_res_101_);
return v_r_102_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(lean_object* v_inst_103_, lean_object* v_r_104_, lean_object* v_s_105_){
_start:
{
uint8_t v___x_106_; 
lean_inc_ref(v_s_105_);
lean_inc_ref(v_r_104_);
v___x_106_ = lp_aesop_Aesop_Rule_compareByPriority___redArg(v_inst_103_, v_r_104_, v_s_105_);
if (v___x_106_ == 1)
{
uint8_t v___x_107_; 
v___x_107_ = lp_aesop_Aesop_Rule_compareByName___redArg(v_r_104_, v_s_105_);
lean_dec_ref(v_s_105_);
lean_dec_ref(v_r_104_);
return v___x_107_;
}
else
{
lean_dec_ref(v_s_105_);
lean_dec_ref(v_r_104_);
return v___x_106_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg___boxed(lean_object* v_inst_108_, lean_object* v_r_109_, lean_object* v_s_110_){
_start:
{
uint8_t v_res_111_; lean_object* v_r_112_; 
v_res_111_ = lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(v_inst_108_, v_r_109_, v_s_110_);
v_r_112_ = lean_box(v_res_111_);
return v_r_112_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rule_compareByPriorityThenName(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_r_115_, lean_object* v_s_116_){
_start:
{
uint8_t v___x_117_; 
v___x_117_ = lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(v_inst_114_, v_r_115_, v_s_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_compareByPriorityThenName___boxed(lean_object* v_00_u03b1_118_, lean_object* v_inst_119_, lean_object* v_r_120_, lean_object* v_s_121_){
_start:
{
uint8_t v_res_122_; lean_object* v_r_123_; 
v_res_122_ = lp_aesop_Aesop_Rule_compareByPriorityThenName(v_00_u03b1_118_, v_inst_119_, v_r_120_, v_s_121_);
v_r_123_ = lean_box(v_res_122_);
return v_r_123_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_map___redArg(lean_object* v_f_124_, lean_object* v_r_125_){
_start:
{
lean_object* v_name_126_; lean_object* v_indexingMode_127_; lean_object* v_pattern_x3f_128_; lean_object* v_extra_129_; lean_object* v_tac_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_138_; 
v_name_126_ = lean_ctor_get(v_r_125_, 0);
v_indexingMode_127_ = lean_ctor_get(v_r_125_, 1);
v_pattern_x3f_128_ = lean_ctor_get(v_r_125_, 2);
v_extra_129_ = lean_ctor_get(v_r_125_, 3);
v_tac_130_ = lean_ctor_get(v_r_125_, 4);
v_isSharedCheck_138_ = !lean_is_exclusive(v_r_125_);
if (v_isSharedCheck_138_ == 0)
{
v___x_132_ = v_r_125_;
v_isShared_133_ = v_isSharedCheck_138_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_tac_130_);
lean_inc(v_extra_129_);
lean_inc(v_pattern_x3f_128_);
lean_inc(v_indexingMode_127_);
lean_inc(v_name_126_);
lean_dec(v_r_125_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_138_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_134_; lean_object* v___x_136_; 
v___x_134_ = lean_apply_1(v_f_124_, v_extra_129_);
if (v_isShared_133_ == 0)
{
lean_ctor_set(v___x_132_, 3, v___x_134_);
v___x_136_ = v___x_132_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_name_126_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_indexingMode_127_);
lean_ctor_set(v_reuseFailAlloc_137_, 2, v_pattern_x3f_128_);
lean_ctor_set(v_reuseFailAlloc_137_, 3, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_137_, 4, v_tac_130_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
return v___x_136_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_map(lean_object* v_00_u03b1_139_, lean_object* v_00_u03b2_140_, lean_object* v_f_141_, lean_object* v_r_142_){
_start:
{
lean_object* v_name_143_; lean_object* v_indexingMode_144_; lean_object* v_pattern_x3f_145_; lean_object* v_extra_146_; lean_object* v_tac_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_155_; 
v_name_143_ = lean_ctor_get(v_r_142_, 0);
v_indexingMode_144_ = lean_ctor_get(v_r_142_, 1);
v_pattern_x3f_145_ = lean_ctor_get(v_r_142_, 2);
v_extra_146_ = lean_ctor_get(v_r_142_, 3);
v_tac_147_ = lean_ctor_get(v_r_142_, 4);
v_isSharedCheck_155_ = !lean_is_exclusive(v_r_142_);
if (v_isSharedCheck_155_ == 0)
{
v___x_149_ = v_r_142_;
v_isShared_150_ = v_isSharedCheck_155_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_tac_147_);
lean_inc(v_extra_146_);
lean_inc(v_pattern_x3f_145_);
lean_inc(v_indexingMode_144_);
lean_inc(v_name_143_);
lean_dec(v_r_142_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_155_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_151_; lean_object* v___x_153_; 
v___x_151_ = lean_apply_1(v_f_141_, v_extra_146_);
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 3, v___x_151_);
v___x_153_ = v___x_149_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_name_143_);
lean_ctor_set(v_reuseFailAlloc_154_, 1, v_indexingMode_144_);
lean_ctor_set(v_reuseFailAlloc_154_, 2, v_pattern_x3f_145_);
lean_ctor_set(v_reuseFailAlloc_154_, 3, v___x_151_);
lean_ctor_set(v_reuseFailAlloc_154_, 4, v_tac_147_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_mapM___redArg___lam__0(lean_object* v_name_156_, lean_object* v_indexingMode_157_, lean_object* v_pattern_x3f_158_, lean_object* v_tac_159_, lean_object* v_toPure_160_, lean_object* v_____do__lift_161_){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_162_, 0, v_name_156_);
lean_ctor_set(v___x_162_, 1, v_indexingMode_157_);
lean_ctor_set(v___x_162_, 2, v_pattern_x3f_158_);
lean_ctor_set(v___x_162_, 3, v_____do__lift_161_);
lean_ctor_set(v___x_162_, 4, v_tac_159_);
v___x_163_ = lean_apply_2(v_toPure_160_, lean_box(0), v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_mapM___redArg(lean_object* v_inst_164_, lean_object* v_f_165_, lean_object* v_r_166_){
_start:
{
lean_object* v_toApplicative_167_; lean_object* v_toBind_168_; lean_object* v_name_169_; lean_object* v_indexingMode_170_; lean_object* v_pattern_x3f_171_; lean_object* v_extra_172_; lean_object* v_tac_173_; lean_object* v_toPure_174_; lean_object* v___x_175_; lean_object* v___f_176_; lean_object* v___x_177_; 
v_toApplicative_167_ = lean_ctor_get(v_inst_164_, 0);
lean_inc_ref(v_toApplicative_167_);
v_toBind_168_ = lean_ctor_get(v_inst_164_, 1);
lean_inc(v_toBind_168_);
lean_dec_ref(v_inst_164_);
v_name_169_ = lean_ctor_get(v_r_166_, 0);
lean_inc_ref(v_name_169_);
v_indexingMode_170_ = lean_ctor_get(v_r_166_, 1);
lean_inc(v_indexingMode_170_);
v_pattern_x3f_171_ = lean_ctor_get(v_r_166_, 2);
lean_inc(v_pattern_x3f_171_);
v_extra_172_ = lean_ctor_get(v_r_166_, 3);
lean_inc(v_extra_172_);
v_tac_173_ = lean_ctor_get(v_r_166_, 4);
lean_inc(v_tac_173_);
lean_dec_ref(v_r_166_);
v_toPure_174_ = lean_ctor_get(v_toApplicative_167_, 1);
lean_inc(v_toPure_174_);
lean_dec_ref(v_toApplicative_167_);
v___x_175_ = lean_apply_1(v_f_165_, v_extra_172_);
v___f_176_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rule_mapM___redArg___lam__0), 6, 5);
lean_closure_set(v___f_176_, 0, v_name_169_);
lean_closure_set(v___f_176_, 1, v_indexingMode_170_);
lean_closure_set(v___f_176_, 2, v_pattern_x3f_171_);
lean_closure_set(v___f_176_, 3, v_tac_173_);
lean_closure_set(v___f_176_, 4, v_toPure_174_);
v___x_177_ = lean_apply_4(v_toBind_168_, lean_box(0), lean_box(0), v___x_175_, v___f_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rule_mapM(lean_object* v_m_178_, lean_object* v_00_u03b1_179_, lean_object* v_00_u03b2_180_, lean_object* v_inst_181_, lean_object* v_f_182_, lean_object* v_r_183_){
_start:
{
lean_object* v_toApplicative_184_; lean_object* v_toBind_185_; lean_object* v_name_186_; lean_object* v_indexingMode_187_; lean_object* v_pattern_x3f_188_; lean_object* v_extra_189_; lean_object* v_tac_190_; lean_object* v_toPure_191_; lean_object* v___x_192_; lean_object* v___f_193_; lean_object* v___x_194_; 
v_toApplicative_184_ = lean_ctor_get(v_inst_181_, 0);
lean_inc_ref(v_toApplicative_184_);
v_toBind_185_ = lean_ctor_get(v_inst_181_, 1);
lean_inc(v_toBind_185_);
lean_dec_ref(v_inst_181_);
v_name_186_ = lean_ctor_get(v_r_183_, 0);
lean_inc_ref(v_name_186_);
v_indexingMode_187_ = lean_ctor_get(v_r_183_, 1);
lean_inc(v_indexingMode_187_);
v_pattern_x3f_188_ = lean_ctor_get(v_r_183_, 2);
lean_inc(v_pattern_x3f_188_);
v_extra_189_ = lean_ctor_get(v_r_183_, 3);
lean_inc(v_extra_189_);
v_tac_190_ = lean_ctor_get(v_r_183_, 4);
lean_inc(v_tac_190_);
lean_dec_ref(v_r_183_);
v_toPure_191_ = lean_ctor_get(v_toApplicative_184_, 1);
lean_inc(v_toPure_191_);
lean_dec_ref(v_toApplicative_184_);
v___x_192_ = lean_apply_1(v_f_182_, v_extra_189_);
v___f_193_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rule_mapM___redArg___lam__0), 6, 5);
lean_closure_set(v___f_193_, 0, v_name_186_);
lean_closure_set(v___f_193_, 1, v_indexingMode_187_);
lean_closure_set(v___f_193_, 2, v_pattern_x3f_188_);
lean_closure_set(v___f_193_, 3, v_tac_190_);
lean_closure_set(v___f_193_, 4, v_toPure_191_);
v___x_194_ = lean_apply_4(v_toBind_185_, lean_box(0), lean_box(0), v___x_192_, v___f_193_);
return v___x_194_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Descr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Rule_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Descr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Rule_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_Descr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Rule_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Descr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Rule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Rule_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
