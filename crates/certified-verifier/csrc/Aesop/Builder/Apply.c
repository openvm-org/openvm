// Lean compiler output
// Module: Aesop.Builder.Apply
// Imports: public import Init public meta import Init public import Aesop.Builder.Basic import Batteries.Lean.Expr
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
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Meta_instBEqTransparencyMode_beq(uint8_t, uint8_t);
lean_object* lp_aesop_Aesop_IndexingMode_targetMatchingConclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_name(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_ElabRuleTerm_scope(lean_object*);
lean_object* lp_aesop_Aesop_PhaseSpec_toRule(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_expr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_elabRuleTermForApplyLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_warn_applyIff;
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lp_batteries_Lean_Expr_isAppOf_x27(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RulePattern_elab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_applyTransparency(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_applyTransparency___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_applyIndexTransparency(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_applyIndexTransparency___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getApplyIndexingMode(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getApplyIndexingMode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__0 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__1 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__2 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__3 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__3_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__4 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__5 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__5_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__6 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__6_value;
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__7 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___closed__0 = (const lean_object*)&lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 234, .m_capacity = 234, .m_length = 231, .m_data = "Apply builder was used for a theorem with conclusion A ↔ B.\nYou probably want to use the simp builder or create an alias that applies the theorem in one direction.\nUse `set_option aesop.warn.applyIff false` to disable this warning."};
static const lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_applyCore(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_applyCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_applyTransparency(lean_object* v_opts_1_){
_start:
{
lean_object* v_transparency_x3f_2_; 
v_transparency_x3f_2_ = lean_ctor_get(v_opts_1_, 4);
if (lean_obj_tag(v_transparency_x3f_2_) == 0)
{
uint8_t v___x_3_; 
v___x_3_ = 1;
return v___x_3_;
}
else
{
lean_object* v_val_4_; uint8_t v___x_5_; 
v_val_4_ = lean_ctor_get(v_transparency_x3f_2_, 0);
v___x_5_ = lean_unbox(v_val_4_);
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_applyTransparency___boxed(lean_object* v_opts_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_aesop_Aesop_RuleBuilderOptions_applyTransparency(v_opts_6_);
lean_dec_ref(v_opts_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_applyIndexTransparency(lean_object* v_opts_9_){
_start:
{
lean_object* v_indexTransparency_x3f_10_; 
v_indexTransparency_x3f_10_ = lean_ctor_get(v_opts_9_, 5);
if (lean_obj_tag(v_indexTransparency_x3f_10_) == 0)
{
uint8_t v___x_11_; 
v___x_11_ = 2;
return v___x_11_;
}
else
{
lean_object* v_val_12_; uint8_t v___x_13_; 
v_val_12_ = lean_ctor_get(v_indexTransparency_x3f_10_, 0);
v___x_13_ = lean_unbox(v_val_12_);
return v___x_13_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_applyIndexTransparency___boxed(lean_object* v_opts_14_){
_start:
{
uint8_t v_res_15_; lean_object* v_r_16_; 
v_res_15_ = lp_aesop_Aesop_RuleBuilderOptions_applyIndexTransparency(v_opts_14_);
lean_dec_ref(v_opts_14_);
v_r_16_ = lean_box(v_res_15_);
return v_r_16_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getApplyIndexingMode(uint8_t v_indexMd_17_, lean_object* v_type_18_, lean_object* v_a_19_, lean_object* v_a_20_, lean_object* v_a_21_, lean_object* v_a_22_){
_start:
{
uint8_t v___x_24_; uint8_t v___x_25_; 
v___x_24_ = 2;
v___x_25_ = l_Lean_Meta_instBEqTransparencyMode_beq(v_indexMd_17_, v___x_24_);
if (v___x_25_ == 0)
{
lean_object* v___x_26_; lean_object* v___x_27_; 
lean_dec_ref(v_type_18_);
v___x_26_ = lean_box(0);
v___x_27_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
return v___x_27_;
}
else
{
lean_object* v___x_28_; 
v___x_28_ = lp_aesop_Aesop_IndexingMode_targetMatchingConclusion(v_type_18_, v_a_19_, v_a_20_, v_a_21_, v_a_22_);
return v___x_28_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getApplyIndexingMode___boxed(lean_object* v_indexMd_29_, lean_object* v_type_30_, lean_object* v_a_31_, lean_object* v_a_32_, lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_){
_start:
{
uint8_t v_indexMd_boxed_36_; lean_object* v_res_37_; 
v_indexMd_boxed_36_ = lean_unbox(v_indexMd_29_);
v_res_37_ = lp_aesop_Aesop_RuleBuilder_getApplyIndexingMode(v_indexMd_boxed_36_, v_type_30_, v_a_31_, v_a_32_, v_a_33_, v_a_34_);
lean_dec(v_a_34_);
lean_dec_ref(v_a_33_);
lean_dec(v_a_32_);
lean_dec_ref(v_a_31_);
return v_res_37_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1(lean_object* v_opts_38_, lean_object* v_opt_39_){
_start:
{
lean_object* v_name_40_; lean_object* v_defValue_41_; lean_object* v_map_42_; lean_object* v___x_43_; 
v_name_40_ = lean_ctor_get(v_opt_39_, 0);
v_defValue_41_ = lean_ctor_get(v_opt_39_, 1);
v_map_42_ = lean_ctor_get(v_opts_38_, 0);
v___x_43_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_42_, v_name_40_);
if (lean_obj_tag(v___x_43_) == 0)
{
uint8_t v___x_44_; 
v___x_44_ = lean_unbox(v_defValue_41_);
return v___x_44_;
}
else
{
lean_object* v_val_45_; 
v_val_45_ = lean_ctor_get(v___x_43_, 0);
lean_inc(v_val_45_);
lean_dec_ref_known(v___x_43_, 1);
if (lean_obj_tag(v_val_45_) == 1)
{
uint8_t v_v_46_; 
v_v_46_ = lean_ctor_get_uint8(v_val_45_, 0);
lean_dec_ref_known(v_val_45_, 0);
return v_v_46_;
}
else
{
uint8_t v___x_47_; 
lean_dec(v_val_45_);
v___x_47_ = lean_unbox(v_defValue_41_);
return v___x_47_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1___boxed(lean_object* v_opts_48_, lean_object* v_opt_49_){
_start:
{
uint8_t v_res_50_; lean_object* v_r_51_; 
v_res_50_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1(v_opts_48_, v_opt_49_);
lean_dec_ref(v_opt_49_);
lean_dec_ref(v_opts_48_);
v_r_51_ = lean_box(v_res_50_);
return v_r_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___lam__0(lean_object* v_k_52_, lean_object* v_b_53_, lean_object* v_c_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
lean_object* v___x_60_; 
lean_inc(v___y_58_);
lean_inc_ref(v___y_57_);
lean_inc(v___y_56_);
lean_inc_ref(v___y_55_);
v___x_60_ = lean_apply_7(v_k_52_, v_b_53_, v_c_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_, lean_box(0));
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___lam__0___boxed(lean_object* v_k_61_, lean_object* v_b_62_, lean_object* v_c_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___lam__0(v_k_61_, v_b_62_, v_c_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg(lean_object* v_type_70_, lean_object* v_k_71_, uint8_t v_cleanupAnnotations_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___f_78_; uint8_t v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___f_78_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_78_, 0, v_k_71_);
v___x_79_ = 0;
v___x_80_ = lean_box(0);
v___x_81_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_79_, v___x_80_, v_type_70_, v___f_78_, v_cleanupAnnotations_72_, v___x_79_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
if (lean_obj_tag(v___x_81_) == 0)
{
lean_object* v_a_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
v_a_82_ = lean_ctor_get(v___x_81_, 0);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_81_);
if (v_isSharedCheck_89_ == 0)
{
v___x_84_ = v___x_81_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_a_82_);
lean_dec(v___x_81_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_82_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
else
{
lean_object* v_a_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_97_; 
v_a_90_ = lean_ctor_get(v___x_81_, 0);
v_isSharedCheck_97_ = !lean_is_exclusive(v___x_81_);
if (v_isSharedCheck_97_ == 0)
{
v___x_92_ = v___x_81_;
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_a_90_);
lean_dec(v___x_81_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_95_; 
if (v_isShared_93_ == 0)
{
v___x_95_ = v___x_92_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_a_90_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg___boxed(lean_object* v_type_98_, lean_object* v_k_99_, lean_object* v_cleanupAnnotations_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_106_; lean_object* v_res_107_; 
v_cleanupAnnotations_boxed_106_ = lean_unbox(v_cleanupAnnotations_100_);
v_res_107_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg(v_type_98_, v_k_99_, v_cleanupAnnotations_boxed_106_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
lean_dec(v___y_102_);
lean_dec_ref(v___y_101_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2(lean_object* v_00_u03b1_108_, lean_object* v_type_109_, lean_object* v_k_110_, uint8_t v_cleanupAnnotations_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg(v_type_109_, v_k_110_, v_cleanupAnnotations_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___boxed(lean_object* v_00_u03b1_118_, lean_object* v_type_119_, lean_object* v_k_120_, lean_object* v_cleanupAnnotations_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_127_; lean_object* v_res_128_; 
v_cleanupAnnotations_boxed_127_ = lean_unbox(v_cleanupAnnotations_121_);
v_res_128_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2(v_00_u03b1_118_, v_type_119_, v_k_120_, v_cleanupAnnotations_boxed_127_, v___y_122_, v___y_123_, v___y_124_, v___y_125_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
lean_dec(v___y_123_);
lean_dec_ref(v___y_122_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0(lean_object* v_e_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v___x_138_; uint8_t v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_138_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___closed__1));
v___x_139_ = lp_batteries_Lean_Expr_isAppOf_x27(v_e_132_, v___x_138_);
v___x_140_ = lean_box(v___x_139_);
v___x_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0___boxed(lean_object* v_e_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__0(v_e_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
lean_dec_ref(v_e_142_);
return v_res_148_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0(uint8_t v___y_157_, uint8_t v_suppressElabErrors_158_, lean_object* v_x_159_){
_start:
{
if (lean_obj_tag(v_x_159_) == 1)
{
lean_object* v_pre_160_; 
v_pre_160_ = lean_ctor_get(v_x_159_, 0);
switch(lean_obj_tag(v_pre_160_))
{
case 1:
{
lean_object* v_pre_161_; 
v_pre_161_ = lean_ctor_get(v_pre_160_, 0);
switch(lean_obj_tag(v_pre_161_))
{
case 0:
{
lean_object* v_str_162_; lean_object* v_str_163_; lean_object* v___x_164_; uint8_t v___x_165_; 
v_str_162_ = lean_ctor_get(v_x_159_, 1);
v_str_163_ = lean_ctor_get(v_pre_160_, 1);
v___x_164_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__0));
v___x_165_ = lean_string_dec_eq(v_str_163_, v___x_164_);
if (v___x_165_ == 0)
{
lean_object* v___x_166_; uint8_t v___x_167_; 
v___x_166_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__1));
v___x_167_ = lean_string_dec_eq(v_str_163_, v___x_166_);
if (v___x_167_ == 0)
{
return v___y_157_;
}
else
{
lean_object* v___x_168_; uint8_t v___x_169_; 
v___x_168_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__2));
v___x_169_ = lean_string_dec_eq(v_str_162_, v___x_168_);
if (v___x_169_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
}
else
{
lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_170_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__3));
v___x_171_ = lean_string_dec_eq(v_str_162_, v___x_170_);
if (v___x_171_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
}
case 1:
{
lean_object* v_pre_172_; 
v_pre_172_ = lean_ctor_get(v_pre_161_, 0);
if (lean_obj_tag(v_pre_172_) == 0)
{
lean_object* v_str_173_; lean_object* v_str_174_; lean_object* v_str_175_; lean_object* v___x_176_; uint8_t v___x_177_; 
v_str_173_ = lean_ctor_get(v_x_159_, 1);
v_str_174_ = lean_ctor_get(v_pre_160_, 1);
v_str_175_ = lean_ctor_get(v_pre_161_, 1);
v___x_176_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__4));
v___x_177_ = lean_string_dec_eq(v_str_175_, v___x_176_);
if (v___x_177_ == 0)
{
return v___y_157_;
}
else
{
lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_178_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__5));
v___x_179_ = lean_string_dec_eq(v_str_174_, v___x_178_);
if (v___x_179_ == 0)
{
return v___y_157_;
}
else
{
lean_object* v___x_180_; uint8_t v___x_181_; 
v___x_180_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__6));
v___x_181_ = lean_string_dec_eq(v_str_173_, v___x_180_);
if (v___x_181_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
}
}
else
{
return v___y_157_;
}
}
default: 
{
return v___y_157_;
}
}
}
case 0:
{
lean_object* v_str_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v_str_182_ = lean_ctor_get(v_x_159_, 1);
v___x_183_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___closed__7));
v___x_184_ = lean_string_dec_eq(v_str_182_, v___x_183_);
if (v___x_184_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
default: 
{
return v___y_157_;
}
}
}
else
{
return v___y_157_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___boxed(lean_object* v___y_185_, lean_object* v_suppressElabErrors_186_, lean_object* v_x_187_){
_start:
{
uint8_t v___y_4439__boxed_188_; uint8_t v_suppressElabErrors_boxed_189_; uint8_t v_res_190_; lean_object* v_r_191_; 
v___y_4439__boxed_188_ = lean_unbox(v___y_185_);
v_suppressElabErrors_boxed_189_ = lean_unbox(v_suppressElabErrors_186_);
v_res_190_ = lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0(v___y_4439__boxed_188_, v_suppressElabErrors_boxed_189_, v_x_187_);
lean_dec(v_x_187_);
v_r_191_ = lean_box(v_res_190_);
return v_r_191_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3_spec__4(lean_object* v_msgData_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v___x_198_; lean_object* v_env_199_; lean_object* v___x_200_; lean_object* v_mctx_201_; lean_object* v_lctx_202_; lean_object* v_options_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_198_ = lean_st_ref_get(v___y_196_);
v_env_199_ = lean_ctor_get(v___x_198_, 0);
lean_inc_ref(v_env_199_);
lean_dec(v___x_198_);
v___x_200_ = lean_st_ref_get(v___y_194_);
v_mctx_201_ = lean_ctor_get(v___x_200_, 0);
lean_inc_ref(v_mctx_201_);
lean_dec(v___x_200_);
v_lctx_202_ = lean_ctor_get(v___y_193_, 2);
v_options_203_ = lean_ctor_get(v___y_195_, 2);
lean_inc_ref(v_options_203_);
lean_inc_ref(v_lctx_202_);
v___x_204_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_204_, 0, v_env_199_);
lean_ctor_set(v___x_204_, 1, v_mctx_201_);
lean_ctor_set(v___x_204_, 2, v_lctx_202_);
lean_ctor_set(v___x_204_, 3, v_options_203_);
v___x_205_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
lean_ctor_set(v___x_205_, 1, v_msgData_192_);
v___x_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3_spec__4___boxed(lean_object* v_msgData_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3_spec__4(v_msgData_207_, v___y_208_, v___y_209_, v___y_210_, v___y_211_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
lean_dec(v___y_209_);
lean_dec_ref(v___y_208_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3(lean_object* v_ref_215_, lean_object* v_msgData_216_, uint8_t v_severity_217_, uint8_t v_isSilent_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
uint8_t v___y_225_; lean_object* v___y_226_; lean_object* v___y_227_; lean_object* v___y_228_; uint8_t v___y_229_; lean_object* v___y_230_; lean_object* v___y_231_; lean_object* v___y_232_; lean_object* v___y_233_; lean_object* v___y_261_; uint8_t v___y_262_; lean_object* v___y_263_; lean_object* v___y_264_; uint8_t v___y_265_; uint8_t v___y_266_; lean_object* v___y_267_; lean_object* v___y_268_; lean_object* v___y_286_; uint8_t v___y_287_; lean_object* v___y_288_; lean_object* v___y_289_; uint8_t v___y_290_; uint8_t v___y_291_; lean_object* v___y_292_; lean_object* v___y_293_; lean_object* v___y_297_; uint8_t v___y_298_; lean_object* v___y_299_; lean_object* v___y_300_; uint8_t v___y_301_; lean_object* v___y_302_; uint8_t v___y_303_; uint8_t v___x_308_; lean_object* v___y_310_; lean_object* v___y_311_; lean_object* v___y_312_; uint8_t v___y_313_; lean_object* v___y_314_; uint8_t v___y_315_; uint8_t v___y_316_; uint8_t v___y_318_; uint8_t v___x_333_; 
v___x_308_ = 2;
v___x_333_ = l_Lean_instBEqMessageSeverity_beq(v_severity_217_, v___x_308_);
if (v___x_333_ == 0)
{
v___y_318_ = v___x_333_;
goto v___jp_317_;
}
else
{
uint8_t v___x_334_; 
lean_inc_ref(v_msgData_216_);
v___x_334_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_216_);
v___y_318_ = v___x_334_;
goto v___jp_317_;
}
v___jp_224_:
{
lean_object* v___x_234_; lean_object* v_currNamespace_235_; lean_object* v_openDecls_236_; lean_object* v_env_237_; lean_object* v_nextMacroScope_238_; lean_object* v_ngen_239_; lean_object* v_auxDeclNGen_240_; lean_object* v_traceState_241_; lean_object* v_cache_242_; lean_object* v_messages_243_; lean_object* v_infoState_244_; lean_object* v_snapshotTasks_245_; lean_object* v___x_247_; uint8_t v_isShared_248_; uint8_t v_isSharedCheck_259_; 
v___x_234_ = lean_st_ref_take(v___y_233_);
v_currNamespace_235_ = lean_ctor_get(v___y_232_, 6);
v_openDecls_236_ = lean_ctor_get(v___y_232_, 7);
v_env_237_ = lean_ctor_get(v___x_234_, 0);
v_nextMacroScope_238_ = lean_ctor_get(v___x_234_, 1);
v_ngen_239_ = lean_ctor_get(v___x_234_, 2);
v_auxDeclNGen_240_ = lean_ctor_get(v___x_234_, 3);
v_traceState_241_ = lean_ctor_get(v___x_234_, 4);
v_cache_242_ = lean_ctor_get(v___x_234_, 5);
v_messages_243_ = lean_ctor_get(v___x_234_, 6);
v_infoState_244_ = lean_ctor_get(v___x_234_, 7);
v_snapshotTasks_245_ = lean_ctor_get(v___x_234_, 8);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_259_ == 0)
{
v___x_247_ = v___x_234_;
v_isShared_248_ = v_isSharedCheck_259_;
goto v_resetjp_246_;
}
else
{
lean_inc(v_snapshotTasks_245_);
lean_inc(v_infoState_244_);
lean_inc(v_messages_243_);
lean_inc(v_cache_242_);
lean_inc(v_traceState_241_);
lean_inc(v_auxDeclNGen_240_);
lean_inc(v_ngen_239_);
lean_inc(v_nextMacroScope_238_);
lean_inc(v_env_237_);
lean_dec(v___x_234_);
v___x_247_ = lean_box(0);
v_isShared_248_ = v_isSharedCheck_259_;
goto v_resetjp_246_;
}
v_resetjp_246_:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_254_; 
lean_inc(v_openDecls_236_);
lean_inc(v_currNamespace_235_);
v___x_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_249_, 0, v_currNamespace_235_);
lean_ctor_set(v___x_249_, 1, v_openDecls_236_);
v___x_250_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v___y_231_);
lean_inc_ref(v___y_230_);
lean_inc_ref(v___y_228_);
v___x_251_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_251_, 0, v___y_228_);
lean_ctor_set(v___x_251_, 1, v___y_226_);
lean_ctor_set(v___x_251_, 2, v___y_227_);
lean_ctor_set(v___x_251_, 3, v___y_230_);
lean_ctor_set(v___x_251_, 4, v___x_250_);
lean_ctor_set_uint8(v___x_251_, sizeof(void*)*5, v___y_225_);
lean_ctor_set_uint8(v___x_251_, sizeof(void*)*5 + 1, v___y_229_);
lean_ctor_set_uint8(v___x_251_, sizeof(void*)*5 + 2, v_isSilent_218_);
v___x_252_ = l_Lean_MessageLog_add(v___x_251_, v_messages_243_);
if (v_isShared_248_ == 0)
{
lean_ctor_set(v___x_247_, 6, v___x_252_);
v___x_254_ = v___x_247_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_env_237_);
lean_ctor_set(v_reuseFailAlloc_258_, 1, v_nextMacroScope_238_);
lean_ctor_set(v_reuseFailAlloc_258_, 2, v_ngen_239_);
lean_ctor_set(v_reuseFailAlloc_258_, 3, v_auxDeclNGen_240_);
lean_ctor_set(v_reuseFailAlloc_258_, 4, v_traceState_241_);
lean_ctor_set(v_reuseFailAlloc_258_, 5, v_cache_242_);
lean_ctor_set(v_reuseFailAlloc_258_, 6, v___x_252_);
lean_ctor_set(v_reuseFailAlloc_258_, 7, v_infoState_244_);
lean_ctor_set(v_reuseFailAlloc_258_, 8, v_snapshotTasks_245_);
v___x_254_ = v_reuseFailAlloc_258_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_255_ = lean_st_ref_set(v___y_233_, v___x_254_);
v___x_256_ = lean_box(0);
v___x_257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
return v___x_257_;
}
}
}
v___jp_260_:
{
lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_284_; 
v___x_269_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_216_);
v___x_270_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3_spec__4(v___x_269_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
v_a_271_ = lean_ctor_get(v___x_270_, 0);
v_isSharedCheck_284_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_284_ == 0)
{
v___x_273_ = v___x_270_;
v_isShared_274_ = v_isSharedCheck_284_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_270_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_284_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
lean_inc_ref_n(v___y_267_, 2);
v___x_275_ = l_Lean_FileMap_toPosition(v___y_267_, v___y_263_);
lean_dec(v___y_263_);
v___x_276_ = l_Lean_FileMap_toPosition(v___y_267_, v___y_268_);
lean_dec(v___y_268_);
v___x_277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
v___x_278_ = ((lean_object*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___closed__0));
if (v___y_266_ == 0)
{
lean_del_object(v___x_273_);
lean_dec_ref(v___y_261_);
v___y_225_ = v___y_262_;
v___y_226_ = v___x_275_;
v___y_227_ = v___x_277_;
v___y_228_ = v___y_264_;
v___y_229_ = v___y_265_;
v___y_230_ = v___x_278_;
v___y_231_ = v_a_271_;
v___y_232_ = v___y_221_;
v___y_233_ = v___y_222_;
goto v___jp_224_;
}
else
{
uint8_t v___x_279_; 
lean_inc(v_a_271_);
v___x_279_ = l_Lean_MessageData_hasTag(v___y_261_, v_a_271_);
if (v___x_279_ == 0)
{
lean_object* v___x_280_; lean_object* v___x_282_; 
lean_dec_ref_known(v___x_277_, 1);
lean_dec_ref(v___x_275_);
lean_dec(v_a_271_);
v___x_280_ = lean_box(0);
if (v_isShared_274_ == 0)
{
lean_ctor_set(v___x_273_, 0, v___x_280_);
v___x_282_ = v___x_273_;
goto v_reusejp_281_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v___x_280_);
v___x_282_ = v_reuseFailAlloc_283_;
goto v_reusejp_281_;
}
v_reusejp_281_:
{
return v___x_282_;
}
}
else
{
lean_del_object(v___x_273_);
v___y_225_ = v___y_262_;
v___y_226_ = v___x_275_;
v___y_227_ = v___x_277_;
v___y_228_ = v___y_264_;
v___y_229_ = v___y_265_;
v___y_230_ = v___x_278_;
v___y_231_ = v_a_271_;
v___y_232_ = v___y_221_;
v___y_233_ = v___y_222_;
goto v___jp_224_;
}
}
}
}
v___jp_285_:
{
lean_object* v___x_294_; 
v___x_294_ = l_Lean_Syntax_getTailPos_x3f(v___y_288_, v___y_287_);
lean_dec(v___y_288_);
if (lean_obj_tag(v___x_294_) == 0)
{
lean_inc(v___y_293_);
v___y_261_ = v___y_286_;
v___y_262_ = v___y_287_;
v___y_263_ = v___y_293_;
v___y_264_ = v___y_289_;
v___y_265_ = v___y_290_;
v___y_266_ = v___y_291_;
v___y_267_ = v___y_292_;
v___y_268_ = v___y_293_;
goto v___jp_260_;
}
else
{
lean_object* v_val_295_; 
v_val_295_ = lean_ctor_get(v___x_294_, 0);
lean_inc(v_val_295_);
lean_dec_ref_known(v___x_294_, 1);
v___y_261_ = v___y_286_;
v___y_262_ = v___y_287_;
v___y_263_ = v___y_293_;
v___y_264_ = v___y_289_;
v___y_265_ = v___y_290_;
v___y_266_ = v___y_291_;
v___y_267_ = v___y_292_;
v___y_268_ = v_val_295_;
goto v___jp_260_;
}
}
v___jp_296_:
{
lean_object* v_ref_304_; lean_object* v___x_305_; 
v_ref_304_ = l_Lean_replaceRef(v_ref_215_, v___y_300_);
v___x_305_ = l_Lean_Syntax_getPos_x3f(v_ref_304_, v___y_298_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v___x_306_; 
v___x_306_ = lean_unsigned_to_nat(0u);
v___y_286_ = v___y_297_;
v___y_287_ = v___y_298_;
v___y_288_ = v_ref_304_;
v___y_289_ = v___y_299_;
v___y_290_ = v___y_303_;
v___y_291_ = v___y_301_;
v___y_292_ = v___y_302_;
v___y_293_ = v___x_306_;
goto v___jp_285_;
}
else
{
lean_object* v_val_307_; 
v_val_307_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_val_307_);
lean_dec_ref_known(v___x_305_, 1);
v___y_286_ = v___y_297_;
v___y_287_ = v___y_298_;
v___y_288_ = v_ref_304_;
v___y_289_ = v___y_299_;
v___y_290_ = v___y_303_;
v___y_291_ = v___y_301_;
v___y_292_ = v___y_302_;
v___y_293_ = v_val_307_;
goto v___jp_285_;
}
}
v___jp_309_:
{
if (v___y_316_ == 0)
{
v___y_297_ = v___y_310_;
v___y_298_ = v___y_315_;
v___y_299_ = v___y_311_;
v___y_300_ = v___y_312_;
v___y_301_ = v___y_313_;
v___y_302_ = v___y_314_;
v___y_303_ = v_severity_217_;
goto v___jp_296_;
}
else
{
v___y_297_ = v___y_310_;
v___y_298_ = v___y_315_;
v___y_299_ = v___y_311_;
v___y_300_ = v___y_312_;
v___y_301_ = v___y_313_;
v___y_302_ = v___y_314_;
v___y_303_ = v___x_308_;
goto v___jp_296_;
}
}
v___jp_317_:
{
if (v___y_318_ == 0)
{
lean_object* v_fileName_319_; lean_object* v_fileMap_320_; lean_object* v_options_321_; lean_object* v_ref_322_; uint8_t v_suppressElabErrors_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___f_326_; uint8_t v___x_327_; uint8_t v___x_328_; 
v_fileName_319_ = lean_ctor_get(v___y_221_, 0);
v_fileMap_320_ = lean_ctor_get(v___y_221_, 1);
v_options_321_ = lean_ctor_get(v___y_221_, 2);
v_ref_322_ = lean_ctor_get(v___y_221_, 5);
v_suppressElabErrors_323_ = lean_ctor_get_uint8(v___y_221_, sizeof(void*)*14 + 1);
v___x_324_ = lean_box(v___y_318_);
v___x_325_ = lean_box(v_suppressElabErrors_323_);
v___f_326_ = lean_alloc_closure((void*)(lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_326_, 0, v___x_324_);
lean_closure_set(v___f_326_, 1, v___x_325_);
v___x_327_ = 1;
v___x_328_ = l_Lean_instBEqMessageSeverity_beq(v_severity_217_, v___x_327_);
if (v___x_328_ == 0)
{
v___y_310_ = v___f_326_;
v___y_311_ = v_fileName_319_;
v___y_312_ = v_ref_322_;
v___y_313_ = v_suppressElabErrors_323_;
v___y_314_ = v_fileMap_320_;
v___y_315_ = v___y_318_;
v___y_316_ = v___x_328_;
goto v___jp_309_;
}
else
{
lean_object* v___x_329_; uint8_t v___x_330_; 
v___x_329_ = l_Lean_warningAsError;
v___x_330_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1(v_options_321_, v___x_329_);
v___y_310_ = v___f_326_;
v___y_311_ = v_fileName_319_;
v___y_312_ = v_ref_322_;
v___y_313_ = v_suppressElabErrors_323_;
v___y_314_ = v_fileMap_320_;
v___y_315_ = v___y_318_;
v___y_316_ = v___x_330_;
goto v___jp_309_;
}
}
else
{
lean_object* v___x_331_; lean_object* v___x_332_; 
lean_dec_ref(v_msgData_216_);
v___x_331_ = lean_box(0);
v___x_332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
return v___x_332_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3___boxed(lean_object* v_ref_335_, lean_object* v_msgData_336_, lean_object* v_severity_337_, lean_object* v_isSilent_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_){
_start:
{
uint8_t v_severity_boxed_344_; uint8_t v_isSilent_boxed_345_; lean_object* v_res_346_; 
v_severity_boxed_344_ = lean_unbox(v_severity_337_);
v_isSilent_boxed_345_ = lean_unbox(v_isSilent_338_);
v_res_346_ = lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3(v_ref_335_, v_msgData_336_, v_severity_boxed_344_, v_isSilent_boxed_345_, v___y_339_, v___y_340_, v___y_341_, v___y_342_);
lean_dec(v___y_342_);
lean_dec_ref(v___y_341_);
lean_dec(v___y_340_);
lean_dec_ref(v___y_339_);
lean_dec(v_ref_335_);
return v_res_346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0(lean_object* v_msgData_347_, uint8_t v_severity_348_, uint8_t v_isSilent_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_){
_start:
{
lean_object* v_ref_355_; lean_object* v___x_356_; 
v_ref_355_ = lean_ctor_get(v___y_352_, 5);
v___x_356_ = lp_aesop_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0_spec__3(v_ref_355_, v_msgData_347_, v_severity_348_, v_isSilent_349_, v___y_350_, v___y_351_, v___y_352_, v___y_353_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0___boxed(lean_object* v_msgData_357_, lean_object* v_severity_358_, lean_object* v_isSilent_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_){
_start:
{
uint8_t v_severity_boxed_365_; uint8_t v_isSilent_boxed_366_; lean_object* v_res_367_; 
v_severity_boxed_365_ = lean_unbox(v_severity_358_);
v_isSilent_boxed_366_ = lean_unbox(v_isSilent_359_);
v_res_367_ = lp_aesop_Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0(v_msgData_357_, v_severity_boxed_365_, v_isSilent_boxed_366_, v___y_360_, v___y_361_, v___y_362_, v___y_363_);
lean_dec(v___y_363_);
lean_dec_ref(v___y_362_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0(lean_object* v_msgData_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_){
_start:
{
uint8_t v___x_374_; uint8_t v___x_375_; lean_object* v___x_376_; 
v___x_374_ = 1;
v___x_375_ = 0;
v___x_376_ = lp_aesop_Lean_log___at___00Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0_spec__0(v_msgData_368_, v___x_374_, v___x_375_, v___y_369_, v___y_370_, v___y_371_, v___y_372_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0___boxed(lean_object* v_msgData_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_aesop_Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0(v_msgData_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
return v_res_383_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__1(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_385_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__0));
v___x_386_ = l_Lean_stringToMessageData(v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1(lean_object* v___f_387_, lean_object* v_x_388_, lean_object* v_conclusion_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
lean_object* v___x_398_; 
lean_inc_ref(v___f_387_);
lean_inc(v___y_393_);
lean_inc_ref(v___y_392_);
lean_inc(v___y_391_);
lean_inc_ref(v___y_390_);
lean_inc_ref(v_conclusion_389_);
v___x_398_ = lean_apply_6(v___f_387_, v_conclusion_389_, v___y_390_, v___y_391_, v___y_392_, v___y_393_, lean_box(0));
if (lean_obj_tag(v___x_398_) == 0)
{
lean_object* v_a_399_; uint8_t v___x_400_; 
v_a_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc(v_a_399_);
lean_dec_ref_known(v___x_398_, 1);
v___x_400_ = lean_unbox(v_a_399_);
lean_dec(v_a_399_);
if (v___x_400_ == 0)
{
lean_object* v___x_401_; 
lean_inc(v___y_393_);
lean_inc_ref(v___y_392_);
lean_inc(v___y_391_);
lean_inc_ref(v___y_390_);
v___x_401_ = lean_whnf(v_conclusion_389_, v___y_390_, v___y_391_, v___y_392_, v___y_393_);
if (lean_obj_tag(v___x_401_) == 0)
{
lean_object* v_a_402_; lean_object* v___x_403_; 
v_a_402_ = lean_ctor_get(v___x_401_, 0);
lean_inc(v_a_402_);
lean_dec_ref_known(v___x_401_, 1);
lean_inc(v___y_393_);
lean_inc_ref(v___y_392_);
lean_inc(v___y_391_);
lean_inc_ref(v___y_390_);
v___x_403_ = lean_apply_6(v___f_387_, v_a_402_, v___y_390_, v___y_391_, v___y_392_, v___y_393_, lean_box(0));
if (lean_obj_tag(v___x_403_) == 0)
{
lean_object* v_a_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_413_; 
v_a_404_ = lean_ctor_get(v___x_403_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_403_);
if (v_isSharedCheck_413_ == 0)
{
v___x_406_ = v___x_403_;
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_a_404_);
lean_dec(v___x_403_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
uint8_t v___x_408_; 
v___x_408_ = lean_unbox(v_a_404_);
lean_dec(v_a_404_);
if (v___x_408_ == 0)
{
lean_object* v___x_409_; lean_object* v___x_411_; 
v___x_409_ = lean_box(0);
if (v_isShared_407_ == 0)
{
lean_ctor_set(v___x_406_, 0, v___x_409_);
v___x_411_ = v___x_406_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_409_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
else
{
lean_del_object(v___x_406_);
goto v___jp_395_;
}
}
}
else
{
lean_object* v_a_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_421_; 
v_a_414_ = lean_ctor_get(v___x_403_, 0);
v_isSharedCheck_421_ = !lean_is_exclusive(v___x_403_);
if (v_isSharedCheck_421_ == 0)
{
v___x_416_ = v___x_403_;
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_a_414_);
lean_dec(v___x_403_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_419_; 
if (v_isShared_417_ == 0)
{
v___x_419_ = v___x_416_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v_a_414_);
v___x_419_ = v_reuseFailAlloc_420_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
return v___x_419_;
}
}
}
}
else
{
lean_object* v_a_422_; lean_object* v___x_424_; uint8_t v_isShared_425_; uint8_t v_isSharedCheck_429_; 
lean_dec_ref(v___f_387_);
v_a_422_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_429_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_429_ == 0)
{
v___x_424_ = v___x_401_;
v_isShared_425_ = v_isSharedCheck_429_;
goto v_resetjp_423_;
}
else
{
lean_inc(v_a_422_);
lean_dec(v___x_401_);
v___x_424_ = lean_box(0);
v_isShared_425_ = v_isSharedCheck_429_;
goto v_resetjp_423_;
}
v_resetjp_423_:
{
lean_object* v___x_427_; 
if (v_isShared_425_ == 0)
{
v___x_427_ = v___x_424_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v_a_422_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
}
else
{
lean_dec_ref(v_conclusion_389_);
lean_dec_ref(v___f_387_);
goto v___jp_395_;
}
}
else
{
lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
lean_dec_ref(v_conclusion_389_);
lean_dec_ref(v___f_387_);
v_a_430_ = lean_ctor_get(v___x_398_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_437_ == 0)
{
v___x_432_ = v___x_398_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___x_398_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_433_ == 0)
{
v___x_435_ = v___x_432_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_a_430_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
v___jp_395_:
{
lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_396_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__1, &lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__1_once, _init_lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___closed__1);
v___x_397_ = lp_aesop_Lean_logWarning___at___00Aesop_RuleBuilder_checkNoIff_spec__0(v___x_396_, v___y_390_, v___y_391_, v___y_392_, v___y_393_);
return v___x_397_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1___boxed(lean_object* v___f_438_, lean_object* v_x_439_, lean_object* v_conclusion_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_aesop_Aesop_RuleBuilder_checkNoIff___lam__1(v___f_438_, v_x_439_, v_conclusion_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec_ref(v_x_439_);
return v_res_446_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff(lean_object* v_type_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v_options_456_; lean_object* v___x_457_; uint8_t v___x_458_; 
v_options_456_ = lean_ctor_get(v_a_453_, 2);
v___x_457_ = lp_aesop_Aesop_aesop_warn_applyIff;
v___x_458_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_checkNoIff_spec__1(v_options_456_, v___x_457_);
if (v___x_458_ == 0)
{
lean_object* v___x_459_; lean_object* v___x_460_; 
lean_dec_ref(v_type_450_);
v___x_459_ = lean_box(0);
v___x_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
return v___x_460_;
}
else
{
lean_object* v___f_461_; uint8_t v___x_462_; lean_object* v___x_463_; 
v___f_461_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_checkNoIff___closed__1));
v___x_462_ = 0;
v___x_463_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_RuleBuilder_checkNoIff_spec__2___redArg(v_type_450_, v___f_461_, v___x_462_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
return v___x_463_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_checkNoIff___boxed(lean_object* v_type_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_, lean_object* v_a_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_aesop_Aesop_RuleBuilder_checkNoIff(v_type_464_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
lean_dec(v_a_468_);
lean_dec_ref(v_a_467_);
lean_dec(v_a_466_);
lean_dec_ref(v_a_465_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_applyCore(lean_object* v_t_471_, lean_object* v_pat_x3f_472_, lean_object* v_imode_x3f_473_, uint8_t v_md_474_, uint8_t v_indexMd_475_, lean_object* v_phase_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_){
_start:
{
lean_object* v_a_483_; lean_object* v___x_508_; 
lean_inc_ref(v_t_471_);
v___x_508_ = lp_aesop_Aesop_ElabRuleTerm_expr(v_t_471_, v_a_477_, v_a_478_, v_a_479_, v_a_480_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v_a_509_; lean_object* v___x_510_; 
v_a_509_ = lean_ctor_get(v___x_508_, 0);
lean_inc(v_a_509_);
lean_dec_ref_known(v___x_508_, 1);
lean_inc(v_a_480_);
lean_inc_ref(v_a_479_);
lean_inc(v_a_478_);
lean_inc_ref(v_a_477_);
v___x_510_ = lean_infer_type(v_a_509_, v_a_477_, v_a_478_, v_a_479_, v_a_480_);
if (lean_obj_tag(v___x_510_) == 0)
{
if (lean_obj_tag(v_imode_x3f_473_) == 0)
{
lean_object* v_a_511_; lean_object* v___x_512_; 
v_a_511_ = lean_ctor_get(v___x_510_, 0);
lean_inc(v_a_511_);
lean_dec_ref_known(v___x_510_, 1);
v___x_512_ = lp_aesop_Aesop_RuleBuilder_getApplyIndexingMode(v_indexMd_475_, v_a_511_, v_a_477_, v_a_478_, v_a_479_, v_a_480_);
if (lean_obj_tag(v___x_512_) == 0)
{
lean_object* v_a_513_; 
v_a_513_ = lean_ctor_get(v___x_512_, 0);
lean_inc(v_a_513_);
lean_dec_ref_known(v___x_512_, 1);
v_a_483_ = v_a_513_;
goto v___jp_482_;
}
else
{
lean_object* v_a_514_; lean_object* v___x_516_; uint8_t v_isShared_517_; uint8_t v_isSharedCheck_521_; 
lean_dec_ref(v_phase_476_);
lean_dec(v_pat_x3f_472_);
lean_dec_ref(v_t_471_);
v_a_514_ = lean_ctor_get(v___x_512_, 0);
v_isSharedCheck_521_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_521_ == 0)
{
v___x_516_ = v___x_512_;
v_isShared_517_ = v_isSharedCheck_521_;
goto v_resetjp_515_;
}
else
{
lean_inc(v_a_514_);
lean_dec(v___x_512_);
v___x_516_ = lean_box(0);
v_isShared_517_ = v_isSharedCheck_521_;
goto v_resetjp_515_;
}
v_resetjp_515_:
{
lean_object* v___x_519_; 
if (v_isShared_517_ == 0)
{
v___x_519_ = v___x_516_;
goto v_reusejp_518_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v_a_514_);
v___x_519_ = v_reuseFailAlloc_520_;
goto v_reusejp_518_;
}
v_reusejp_518_:
{
return v___x_519_;
}
}
}
}
else
{
lean_object* v_val_522_; 
lean_dec_ref_known(v___x_510_, 1);
v_val_522_ = lean_ctor_get(v_imode_x3f_473_, 0);
lean_inc(v_val_522_);
lean_dec_ref_known(v_imode_x3f_473_, 1);
v_a_483_ = v_val_522_;
goto v___jp_482_;
}
}
else
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_530_; 
lean_dec_ref(v_phase_476_);
lean_dec(v_imode_x3f_473_);
lean_dec(v_pat_x3f_472_);
lean_dec_ref(v_t_471_);
v_a_523_ = lean_ctor_get(v___x_510_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v___x_510_);
if (v_isSharedCheck_530_ == 0)
{
v___x_525_ = v___x_510_;
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_510_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v_a_523_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
else
{
lean_object* v_a_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_538_; 
lean_dec_ref(v_phase_476_);
lean_dec(v_imode_x3f_473_);
lean_dec(v_pat_x3f_472_);
lean_dec_ref(v_t_471_);
v_a_531_ = lean_ctor_get(v___x_508_, 0);
v_isSharedCheck_538_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_538_ == 0)
{
v___x_533_ = v___x_508_;
v_isShared_534_ = v_isSharedCheck_538_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_a_531_);
lean_dec(v___x_508_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_538_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___x_536_; 
if (v_isShared_534_ == 0)
{
v___x_536_ = v___x_533_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v_a_531_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
}
v___jp_482_:
{
lean_object* v___x_484_; lean_object* v___x_485_; 
lean_inc_ref_n(v_t_471_, 2);
v___x_484_ = lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(v_t_471_);
v___x_485_ = lp_aesop_Aesop_ElabRuleTerm_name(v_t_471_, v_a_477_, v_a_478_, v_a_479_, v_a_480_);
if (lean_obj_tag(v___x_485_) == 0)
{
lean_object* v_a_486_; lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_499_; 
v_a_486_ = lean_ctor_get(v___x_485_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_499_ == 0)
{
v___x_488_ = v___x_485_;
v_isShared_489_ = v_isSharedCheck_499_;
goto v_resetjp_487_;
}
else
{
lean_inc(v_a_486_);
lean_dec(v___x_485_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_499_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v___x_490_; uint8_t v___x_491_; uint8_t v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_497_; 
v___x_490_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_490_, 0, v___x_484_);
lean_ctor_set_uint8(v___x_490_, sizeof(void*)*1, v_md_474_);
v___x_491_ = 0;
v___x_492_ = lp_aesop_Aesop_ElabRuleTerm_scope(v_t_471_);
lean_dec_ref(v_t_471_);
v___x_493_ = lp_aesop_Aesop_PhaseSpec_toRule(v_phase_476_, v_a_486_, v___x_491_, v___x_492_, v___x_490_, v_a_483_, v_pat_x3f_472_);
v___x_494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_494_, 0, v___x_493_);
v___x_495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_495_, 0, v___x_494_);
if (v_isShared_489_ == 0)
{
lean_ctor_set(v___x_488_, 0, v___x_495_);
v___x_497_ = v___x_488_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_495_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
else
{
lean_object* v_a_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_507_; 
lean_dec_ref(v___x_484_);
lean_dec(v_a_483_);
lean_dec_ref(v_phase_476_);
lean_dec(v_pat_x3f_472_);
lean_dec_ref(v_t_471_);
v_a_500_ = lean_ctor_get(v___x_485_, 0);
v_isSharedCheck_507_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_507_ == 0)
{
v___x_502_ = v___x_485_;
v_isShared_503_ = v_isSharedCheck_507_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_a_500_);
lean_dec(v___x_485_);
v___x_502_ = lean_box(0);
v_isShared_503_ = v_isSharedCheck_507_;
goto v_resetjp_501_;
}
v_resetjp_501_:
{
lean_object* v___x_505_; 
if (v_isShared_503_ == 0)
{
v___x_505_ = v___x_502_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v_a_500_);
v___x_505_ = v_reuseFailAlloc_506_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
return v___x_505_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_applyCore___boxed(lean_object* v_t_539_, lean_object* v_pat_x3f_540_, lean_object* v_imode_x3f_541_, lean_object* v_md_542_, lean_object* v_indexMd_543_, lean_object* v_phase_544_, lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_, lean_object* v_a_548_, lean_object* v_a_549_){
_start:
{
uint8_t v_md_boxed_550_; uint8_t v_indexMd_boxed_551_; lean_object* v_res_552_; 
v_md_boxed_550_ = lean_unbox(v_md_542_);
v_indexMd_boxed_551_ = lean_unbox(v_indexMd_543_);
v_res_552_ = lp_aesop_Aesop_RuleBuilder_applyCore(v_t_539_, v_pat_x3f_540_, v_imode_x3f_541_, v_md_boxed_550_, v_indexMd_boxed_551_, v_phase_544_, v_a_545_, v_a_546_, v_a_547_, v_a_548_);
lean_dec(v_a_548_);
lean_dec_ref(v_a_547_);
lean_dec(v_a_546_);
lean_dec_ref(v_a_545_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_apply(lean_object* v_input_553_, lean_object* v_a_554_, lean_object* v_a_555_, lean_object* v_a_556_, lean_object* v_a_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_){
_start:
{
lean_object* v_term_562_; lean_object* v_options_563_; lean_object* v_phase_564_; lean_object* v___x_565_; 
v_term_562_ = lean_ctor_get(v_input_553_, 0);
lean_inc_n(v_term_562_, 2);
v_options_563_ = lean_ctor_get(v_input_553_, 1);
lean_inc_ref(v_options_563_);
v_phase_564_ = lean_ctor_get(v_input_553_, 2);
lean_inc_ref(v_phase_564_);
lean_dec_ref(v_input_553_);
v___x_565_ = lp_aesop_Aesop_elabRuleTermForApplyLike(v_term_562_, v_a_554_, v_a_555_, v_a_556_, v_a_557_, v_a_558_, v_a_559_, v_a_560_);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v_a_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
v_a_566_ = lean_ctor_get(v___x_565_, 0);
lean_inc_n(v_a_566_, 3);
lean_dec_ref_known(v___x_565_, 1);
v___x_567_ = lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(v_term_562_, v_a_566_);
lean_inc(v_a_560_);
lean_inc_ref(v_a_559_);
lean_inc(v_a_558_);
lean_inc_ref(v_a_557_);
v___x_568_ = lean_infer_type(v_a_566_, v_a_557_, v_a_558_, v_a_559_, v_a_560_);
if (lean_obj_tag(v___x_568_) == 0)
{
lean_object* v_a_569_; lean_object* v___x_570_; 
v_a_569_ = lean_ctor_get(v___x_568_, 0);
lean_inc(v_a_569_);
lean_dec_ref_known(v___x_568_, 1);
v___x_570_ = lp_aesop_Aesop_RuleBuilder_checkNoIff(v_a_569_, v_a_557_, v_a_558_, v_a_559_, v_a_560_);
if (lean_obj_tag(v___x_570_) == 0)
{
lean_object* v_indexingMode_x3f_571_; lean_object* v_pattern_x3f_572_; lean_object* v_a_574_; 
lean_dec_ref_known(v___x_570_, 1);
v_indexingMode_x3f_571_ = lean_ctor_get(v_options_563_, 1);
lean_inc(v_indexingMode_x3f_571_);
v_pattern_x3f_572_ = lean_ctor_get(v_options_563_, 3);
lean_inc(v_pattern_x3f_572_);
if (lean_obj_tag(v_pattern_x3f_572_) == 0)
{
lean_object* v___x_578_; 
lean_dec(v_a_566_);
v___x_578_ = lean_box(0);
v_a_574_ = v___x_578_;
goto v___jp_573_;
}
else
{
lean_object* v_val_579_; lean_object* v___x_581_; uint8_t v_isShared_582_; uint8_t v_isSharedCheck_596_; 
v_val_579_ = lean_ctor_get(v_pattern_x3f_572_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v_pattern_x3f_572_);
if (v_isSharedCheck_596_ == 0)
{
v___x_581_ = v_pattern_x3f_572_;
v_isShared_582_ = v_isSharedCheck_596_;
goto v_resetjp_580_;
}
else
{
lean_inc(v_val_579_);
lean_dec(v_pattern_x3f_572_);
v___x_581_ = lean_box(0);
v_isShared_582_ = v_isSharedCheck_596_;
goto v_resetjp_580_;
}
v_resetjp_580_:
{
lean_object* v___x_583_; 
v___x_583_ = lp_aesop_Aesop_RulePattern_elab(v_val_579_, v_a_566_, v_a_555_, v_a_556_, v_a_557_, v_a_558_, v_a_559_, v_a_560_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v_a_584_; lean_object* v___x_586_; 
v_a_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_584_);
lean_dec_ref_known(v___x_583_, 1);
if (v_isShared_582_ == 0)
{
lean_ctor_set(v___x_581_, 0, v_a_584_);
v___x_586_ = v___x_581_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_587_; 
v_reuseFailAlloc_587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_587_, 0, v_a_584_);
v___x_586_ = v_reuseFailAlloc_587_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
v_a_574_ = v___x_586_;
goto v___jp_573_;
}
}
else
{
lean_object* v_a_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_595_; 
lean_del_object(v___x_581_);
lean_dec(v_indexingMode_x3f_571_);
lean_dec_ref(v___x_567_);
lean_dec_ref(v_phase_564_);
lean_dec_ref(v_options_563_);
v_a_588_ = lean_ctor_get(v___x_583_, 0);
v_isSharedCheck_595_ = !lean_is_exclusive(v___x_583_);
if (v_isSharedCheck_595_ == 0)
{
v___x_590_ = v___x_583_;
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_a_588_);
lean_dec(v___x_583_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___x_593_; 
if (v_isShared_591_ == 0)
{
v___x_593_ = v___x_590_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v_a_588_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
}
}
}
v___jp_573_:
{
uint8_t v___x_575_; uint8_t v___x_576_; lean_object* v___x_577_; 
v___x_575_ = lp_aesop_Aesop_RuleBuilderOptions_applyTransparency(v_options_563_);
v___x_576_ = lp_aesop_Aesop_RuleBuilderOptions_applyIndexTransparency(v_options_563_);
lean_dec_ref(v_options_563_);
v___x_577_ = lp_aesop_Aesop_RuleBuilder_applyCore(v___x_567_, v_a_574_, v_indexingMode_x3f_571_, v___x_575_, v___x_576_, v_phase_564_, v_a_557_, v_a_558_, v_a_559_, v_a_560_);
return v___x_577_;
}
}
else
{
lean_object* v_a_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_604_; 
lean_dec_ref(v___x_567_);
lean_dec(v_a_566_);
lean_dec_ref(v_phase_564_);
lean_dec_ref(v_options_563_);
v_a_597_ = lean_ctor_get(v___x_570_, 0);
v_isSharedCheck_604_ = !lean_is_exclusive(v___x_570_);
if (v_isSharedCheck_604_ == 0)
{
v___x_599_ = v___x_570_;
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_a_597_);
lean_dec(v___x_570_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_602_; 
if (v_isShared_600_ == 0)
{
v___x_602_ = v___x_599_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_603_; 
v_reuseFailAlloc_603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_603_, 0, v_a_597_);
v___x_602_ = v_reuseFailAlloc_603_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
return v___x_602_;
}
}
}
}
else
{
lean_object* v_a_605_; lean_object* v___x_607_; uint8_t v_isShared_608_; uint8_t v_isSharedCheck_612_; 
lean_dec_ref(v___x_567_);
lean_dec(v_a_566_);
lean_dec_ref(v_phase_564_);
lean_dec_ref(v_options_563_);
v_a_605_ = lean_ctor_get(v___x_568_, 0);
v_isSharedCheck_612_ = !lean_is_exclusive(v___x_568_);
if (v_isSharedCheck_612_ == 0)
{
v___x_607_ = v___x_568_;
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
else
{
lean_inc(v_a_605_);
lean_dec(v___x_568_);
v___x_607_ = lean_box(0);
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
v_resetjp_606_:
{
lean_object* v___x_610_; 
if (v_isShared_608_ == 0)
{
v___x_610_ = v___x_607_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_a_605_);
v___x_610_ = v_reuseFailAlloc_611_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
return v___x_610_;
}
}
}
}
else
{
lean_object* v_a_613_; lean_object* v___x_615_; uint8_t v_isShared_616_; uint8_t v_isSharedCheck_620_; 
lean_dec_ref(v_phase_564_);
lean_dec_ref(v_options_563_);
lean_dec(v_term_562_);
v_a_613_ = lean_ctor_get(v___x_565_, 0);
v_isSharedCheck_620_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_620_ == 0)
{
v___x_615_ = v___x_565_;
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
else
{
lean_inc(v_a_613_);
lean_dec(v___x_565_);
v___x_615_ = lean_box(0);
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
v_resetjp_614_:
{
lean_object* v___x_618_; 
if (v_isShared_616_ == 0)
{
v___x_618_ = v___x_615_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_a_613_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_apply___boxed(lean_object* v_input_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_aesop_Aesop_RuleBuilder_apply(v_input_621_, v_a_622_, v_a_623_, v_a_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_);
lean_dec(v_a_628_);
lean_dec_ref(v_a_627_);
lean_dec(v_a_626_);
lean_dec_ref(v_a_625_);
lean_dec(v_a_624_);
lean_dec_ref(v_a_623_);
lean_dec_ref(v_a_622_);
return v_res_630_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Expr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Builder_Apply(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Builder_Apply(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Builder_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Expr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Builder_Apply(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Builder_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Builder_Apply(builtin);
}
#ifdef __cplusplus
}
#endif
