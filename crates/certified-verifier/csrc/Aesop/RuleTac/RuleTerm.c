// Lean compiler output
// Module: Aesop.RuleTac.RuleTerm
// Imports: public import Init public meta import Init public import Aesop.Rule.Name
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
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* lp_aesop_Aesop_getRuleNameForExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedRuleTerm_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedRuleTerm = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleTerm_default___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToMessageDataRuleTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToMessageDataRuleTerm___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToMessageDataRuleTerm___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm = (const lean_object*)&lp_aesop_Aesop_instToMessageDataRuleTerm___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedElabRuleTerm_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedElabRuleTerm = (const lean_object*)&lp_aesop_Aesop_instInhabitedElabRuleTerm_default___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ElabRuleTerm_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ElabRuleTerm_instToMessageData___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_ElabRuleTerm_instToMessageData___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData = (const lean_object*)&lp_aesop_Aesop_ElabRuleTerm_instToMessageData___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_ElabRuleTerm_scope(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_scope___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_aesop_Aesop_RuleTerm_ctorIdx(v_x_4_);
lean_dec_ref(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___redArg(lean_object* v_t_6_, lean_object* v_k_7_){
_start:
{
lean_object* v_decl_8_; lean_object* v___x_9_; 
v_decl_8_ = lean_ctor_get(v_t_6_, 0);
lean_inc(v_decl_8_);
lean_dec_ref(v_t_6_);
v___x_9_ = lean_apply_1(v_k_7_, v_decl_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim(lean_object* v_motive_10_, lean_object* v_ctorIdx_11_, lean_object* v_t_12_, lean_object* v_h_13_, lean_object* v_k_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(v_t_12_, v_k_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_aesop_Aesop_RuleTerm_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_18_, v_h_19_, v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim___redArg(lean_object* v_t_22_, lean_object* v_const_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(v_t_22_, v_const_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_const_elim(lean_object* v_motive_25_, lean_object* v_t_26_, lean_object* v_h_27_, lean_object* v_const_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(v_t_26_, v_const_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim___redArg(lean_object* v_t_30_, lean_object* v_term_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(v_t_30_, v_term_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTerm_term_elim(lean_object* v_motive_33_, lean_object* v_t_34_, lean_object* v_h_35_, lean_object* v_term_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_aesop_Aesop_RuleTerm_ctorElim___redArg(v_t_34_, v_term_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToMessageDataRuleTerm___lam__0(lean_object* v_x_42_){
_start:
{
if (lean_obj_tag(v_x_42_) == 0)
{
lean_object* v_decl_43_; lean_object* v___x_44_; 
v_decl_43_ = lean_ctor_get(v_x_42_, 0);
lean_inc(v_decl_43_);
lean_dec_ref_known(v_x_42_, 1);
v___x_44_ = l_Lean_MessageData_ofName(v_decl_43_);
return v___x_44_;
}
else
{
lean_object* v_term_45_; lean_object* v___x_46_; 
v_term_45_ = lean_ctor_get(v_x_42_, 0);
lean_inc(v_term_45_);
lean_dec_ref_known(v_x_42_, 1);
v___x_46_ = l_Lean_MessageData_ofSyntax(v_term_45_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx(lean_object* v_x_49_){
_start:
{
if (lean_obj_tag(v_x_49_) == 0)
{
lean_object* v___x_50_; 
v___x_50_ = lean_unsigned_to_nat(0u);
return v___x_50_;
}
else
{
lean_object* v___x_51_; 
v___x_51_ = lean_unsigned_to_nat(1u);
return v___x_51_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorIdx___boxed(lean_object* v_x_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_aesop_Aesop_ElabRuleTerm_ctorIdx(v_x_52_);
lean_dec_ref(v_x_52_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(lean_object* v_t_54_, lean_object* v_k_55_){
_start:
{
if (lean_obj_tag(v_t_54_) == 0)
{
lean_object* v_decl_56_; lean_object* v___x_57_; 
v_decl_56_ = lean_ctor_get(v_t_54_, 0);
lean_inc(v_decl_56_);
lean_dec_ref_known(v_t_54_, 1);
v___x_57_ = lean_apply_1(v_k_55_, v_decl_56_);
return v___x_57_;
}
else
{
lean_object* v_term_58_; lean_object* v_expr_59_; lean_object* v___x_60_; 
v_term_58_ = lean_ctor_get(v_t_54_, 0);
lean_inc(v_term_58_);
v_expr_59_ = lean_ctor_get(v_t_54_, 1);
lean_inc_ref(v_expr_59_);
lean_dec_ref_known(v_t_54_, 2);
v___x_60_ = lean_apply_2(v_k_55_, v_term_58_, v_expr_59_);
return v___x_60_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim(lean_object* v_motive_61_, lean_object* v_ctorIdx_62_, lean_object* v_t_63_, lean_object* v_h_64_, lean_object* v_k_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(v_t_63_, v_k_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ctorElim___boxed(lean_object* v_motive_67_, lean_object* v_ctorIdx_68_, lean_object* v_t_69_, lean_object* v_h_70_, lean_object* v_k_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_aesop_Aesop_ElabRuleTerm_ctorElim(v_motive_67_, v_ctorIdx_68_, v_t_69_, v_h_70_, v_k_71_);
lean_dec(v_ctorIdx_68_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim___redArg(lean_object* v_t_73_, lean_object* v_const_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(v_t_73_, v_const_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_const_elim(lean_object* v_motive_76_, lean_object* v_t_77_, lean_object* v_h_78_, lean_object* v_const_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(v_t_77_, v_const_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim___redArg(lean_object* v_t_81_, lean_object* v_term_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(v_t_81_, v_term_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_term_elim(lean_object* v_motive_84_, lean_object* v_t_85_, lean_object* v_h_86_, lean_object* v_term_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_aesop_Aesop_ElabRuleTerm_ctorElim___redArg(v_t_85_, v_term_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_instToMessageData___lam__0(lean_object* v_x_93_){
_start:
{
if (lean_obj_tag(v_x_93_) == 0)
{
lean_object* v_decl_94_; lean_object* v___x_95_; 
v_decl_94_ = lean_ctor_get(v_x_93_, 0);
lean_inc(v_decl_94_);
lean_dec_ref_known(v_x_93_, 1);
v___x_95_ = l_Lean_MessageData_ofName(v_decl_94_);
return v___x_95_;
}
else
{
lean_object* v_term_96_; lean_object* v___x_97_; 
v_term_96_ = lean_ctor_get(v_x_93_, 0);
lean_inc(v_term_96_);
lean_dec_ref_known(v_x_93_, 2);
v___x_97_ = l_Lean_MessageData_ofSyntax(v_term_96_);
return v___x_97_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr(lean_object* v_x_100_, lean_object* v_a_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_){
_start:
{
if (lean_obj_tag(v_x_100_) == 0)
{
lean_object* v_decl_106_; lean_object* v___x_107_; 
v_decl_106_ = lean_ctor_get(v_x_100_, 0);
lean_inc(v_decl_106_);
lean_dec_ref_known(v_x_100_, 1);
v___x_107_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_106_, v_a_101_, v_a_102_, v_a_103_, v_a_104_);
return v___x_107_;
}
else
{
lean_object* v_expr_108_; lean_object* v___x_109_; 
v_expr_108_ = lean_ctor_get(v_x_100_, 1);
lean_inc_ref(v_expr_108_);
lean_dec_ref_known(v_x_100_, 2);
v___x_109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_109_, 0, v_expr_108_);
return v___x_109_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_expr___boxed(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_aesop_Aesop_ElabRuleTerm_expr(v_x_110_, v_a_111_, v_a_112_, v_a_113_, v_a_114_);
lean_dec(v_a_114_);
lean_dec_ref(v_a_113_);
lean_dec(v_a_112_);
lean_dec_ref(v_a_111_);
return v_res_116_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ElabRuleTerm_scope(lean_object* v_x_117_){
_start:
{
if (lean_obj_tag(v_x_117_) == 0)
{
uint8_t v___x_118_; 
v___x_118_ = 0;
return v___x_118_;
}
else
{
uint8_t v___x_119_; 
v___x_119_ = 1;
return v___x_119_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_scope___boxed(lean_object* v_x_120_){
_start:
{
uint8_t v_res_121_; lean_object* v_r_122_; 
v_res_121_ = lp_aesop_Aesop_ElabRuleTerm_scope(v_x_120_);
lean_dec_ref(v_x_120_);
v_r_122_ = lean_box(v_res_121_);
return v_r_122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name(lean_object* v_x_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_, lean_object* v_a_127_){
_start:
{
if (lean_obj_tag(v_x_123_) == 0)
{
lean_object* v_decl_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_136_; 
v_decl_129_ = lean_ctor_get(v_x_123_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v_x_123_);
if (v_isSharedCheck_136_ == 0)
{
v___x_131_ = v_x_123_;
v_isShared_132_ = v_isSharedCheck_136_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_decl_129_);
lean_dec(v_x_123_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_136_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v___x_134_; 
if (v_isShared_132_ == 0)
{
v___x_134_ = v___x_131_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v_decl_129_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
else
{
lean_object* v_expr_137_; lean_object* v___x_138_; 
v_expr_137_ = lean_ctor_get(v_x_123_, 1);
lean_inc_ref(v_expr_137_);
lean_dec_ref_known(v_x_123_, 2);
v___x_138_ = lp_aesop_Aesop_getRuleNameForExpr(v_expr_137_, v_a_124_, v_a_125_, v_a_126_, v_a_127_);
return v___x_138_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_name___boxed(lean_object* v_x_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_aesop_Aesop_ElabRuleTerm_name(v_x_139_, v_a_140_, v_a_141_, v_a_142_, v_a_143_);
lean_dec(v_a_143_);
lean_dec_ref(v_a_142_);
lean_dec(v_a_141_);
lean_dec_ref(v_a_140_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(lean_object* v_x_146_){
_start:
{
if (lean_obj_tag(v_x_146_) == 0)
{
lean_object* v_decl_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
v_decl_147_ = lean_ctor_get(v_x_146_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v_x_146_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v_x_146_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_decl_147_);
lean_dec(v_x_146_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_decl_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
else
{
lean_object* v_term_155_; lean_object* v___x_156_; 
v_term_155_ = lean_ctor_get(v_x_146_, 0);
lean_inc(v_term_155_);
lean_dec_ref_known(v_x_146_, 2);
v___x_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_156_, 0, v_term_155_);
return v___x_156_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(lean_object* v_tm_157_, lean_object* v_expr_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = l_Lean_Expr_constName_x3f(v_expr_158_);
if (lean_obj_tag(v___x_159_) == 1)
{
lean_object* v_val_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_167_; 
lean_dec_ref(v_expr_158_);
lean_dec(v_tm_157_);
v_val_160_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_167_ == 0)
{
v___x_162_ = v___x_159_;
v_isShared_163_ = v_isSharedCheck_167_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_val_160_);
lean_dec(v___x_159_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_167_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_165_; 
if (v_isShared_163_ == 0)
{
lean_ctor_set_tag(v___x_162_, 0);
v___x_165_ = v___x_162_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v_val_160_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
else
{
lean_object* v___x_168_; 
lean_dec(v___x_159_);
v___x_168_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_168_, 0, v_tm_157_);
lean_ctor_set(v___x_168_, 1, v_expr_158_);
return v___x_168_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
}
#ifdef __cplusplus
}
#endif
