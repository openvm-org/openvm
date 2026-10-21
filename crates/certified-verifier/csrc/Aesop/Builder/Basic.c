// Lean compiler output
// Module: Aesop.Builder.Basic
// Imports: public import Init public meta import Init public import Aesop.RuleSet.Member public import Aesop.RuleTac.ElabRuleTerm
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
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lp_aesop_Aesop_elabInductiveRuleIdent_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
uint64_t lp_aesop_Aesop_instHashableBuilderName_hash(uint8_t);
uint64_t lp_aesop_Aesop_instHashablePhaseName_hash(uint8_t);
uint64_t lp_aesop_Aesop_instHashableScopeName_hash(uint8_t);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
static const lean_ctor_object lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedRuleBuilderOptions = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleBuilderOptions_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleBuilderOptions_instEmptyCollection = (const lean_object*)&lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_safe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_safe_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_norm_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_norm_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_unsafe_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_unsafe_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedPhaseSpec_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedPhaseSpec_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPhaseSpec_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedPhaseSpec;
LEAN_EXPORT uint8_t lp_aesop_Aesop_PhaseSpec_phase(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_phase___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_toRule(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_toRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleBuilderInput_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRuleBuilderInput_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleBuilderInput_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleBuilderInput;
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderInput_phaseName(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderInput_phaseName___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0;
static const lean_string_object lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__1 = (const lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__1_value;
static const lean_ctor_object lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__1_value)}};
static const lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__2 = (const lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__2_value;
static lean_once_cell_t lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "aesop: "};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_elabGlobalRuleIdent___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__1;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = " builder: expected '"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__2 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_elabGlobalRuleIdent___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__3;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "' to be an unambiguous global constant"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__4 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_elabGlobalRuleIdent___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__5;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__6 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__6_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__7 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__7_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__8 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__8_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__9 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__9_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__10 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__10_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__11 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__11_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__12 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__12_value;
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___closed__13 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent___closed__13_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_elabInductiveRuleIdent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 87, .m_capacity = 87, .m_length = 86, .m_data = "' to be an inductive type or structure (or to reduce to one at the given transparency)"};
static const lean_object* lp_aesop_Aesop_elabInductiveRuleIdent___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabInductiveRuleIdent___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_elabInductiveRuleIdent___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_elabInductiveRuleIdent___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorIdx(lean_object* v_x_7_){
_start:
{
switch(lean_obj_tag(v_x_7_))
{
case 0:
{
lean_object* v___x_8_; 
v___x_8_ = lean_unsigned_to_nat(0u);
return v___x_8_;
}
case 1:
{
lean_object* v___x_9_; 
v___x_9_ = lean_unsigned_to_nat(1u);
return v___x_9_;
}
default: 
{
lean_object* v___x_10_; 
v___x_10_ = lean_unsigned_to_nat(2u);
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorIdx___boxed(lean_object* v_x_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_aesop_Aesop_PhaseSpec_ctorIdx(v_x_11_);
lean_dec_ref(v_x_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(lean_object* v_t_13_, lean_object* v_k_14_){
_start:
{
switch(lean_obj_tag(v_t_13_))
{
case 0:
{
lean_object* v_info_15_; lean_object* v___x_16_; 
v_info_15_ = lean_ctor_get(v_t_13_, 0);
lean_inc_ref(v_info_15_);
lean_dec_ref_known(v_t_13_, 1);
v___x_16_ = lean_apply_1(v_k_14_, v_info_15_);
return v___x_16_;
}
case 1:
{
lean_object* v_info_17_; lean_object* v___x_18_; 
v_info_17_ = lean_ctor_get(v_t_13_, 0);
lean_inc(v_info_17_);
lean_dec_ref_known(v_t_13_, 1);
v___x_18_ = lean_apply_1(v_k_14_, v_info_17_);
return v___x_18_;
}
default: 
{
double v_info_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_info_19_ = lean_ctor_get_float(v_t_13_, 0);
lean_dec_ref_known(v_t_13_, 0);
v___x_20_ = lean_box_float(v_info_19_);
v___x_21_ = lean_apply_1(v_k_14_, v___x_20_);
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorElim(lean_object* v_motive_22_, lean_object* v_ctorIdx_23_, lean_object* v_t_24_, lean_object* v_h_25_, lean_object* v_k_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_24_, v_k_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_ctorElim___boxed(lean_object* v_motive_28_, lean_object* v_ctorIdx_29_, lean_object* v_t_30_, lean_object* v_h_31_, lean_object* v_k_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_aesop_Aesop_PhaseSpec_ctorElim(v_motive_28_, v_ctorIdx_29_, v_t_30_, v_h_31_, v_k_32_);
lean_dec(v_ctorIdx_29_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_safe_elim___redArg(lean_object* v_t_34_, lean_object* v_safe_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_34_, v_safe_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_safe_elim(lean_object* v_motive_37_, lean_object* v_t_38_, lean_object* v_h_39_, lean_object* v_safe_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_38_, v_safe_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_norm_elim___redArg(lean_object* v_t_42_, lean_object* v_norm_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_42_, v_norm_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_norm_elim(lean_object* v_motive_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_norm_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_46_, v_norm_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_unsafe_elim___redArg(lean_object* v_t_50_, lean_object* v_unsafe_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_50_, v_unsafe_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_unsafe_elim(lean_object* v_motive_53_, lean_object* v_t_54_, lean_object* v_h_55_, lean_object* v_unsafe_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_aesop_Aesop_PhaseSpec_ctorElim___redArg(v_t_54_, v_unsafe_56_);
return v___x_57_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPhaseSpec_default___closed__0(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = lp_aesop_Aesop_instInhabitedSafeRuleInfo_default;
v___x_59_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
return v___x_59_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPhaseSpec_default(void){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedPhaseSpec_default___closed__0, &lp_aesop_Aesop_instInhabitedPhaseSpec_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedPhaseSpec_default___closed__0);
return v___x_60_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedPhaseSpec(void){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_aesop_Aesop_instInhabitedPhaseSpec_default;
return v___x_61_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_PhaseSpec_phase(lean_object* v_x_62_){
_start:
{
switch(lean_obj_tag(v_x_62_))
{
case 0:
{
uint8_t v___x_63_; 
v___x_63_ = 1;
return v___x_63_;
}
case 1:
{
uint8_t v___x_64_; 
v___x_64_ = 0;
return v___x_64_;
}
default: 
{
uint8_t v___x_65_; 
v___x_65_ = 2;
return v___x_65_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_phase___boxed(lean_object* v_x_66_){
_start:
{
uint8_t v_res_67_; lean_object* v_r_68_; 
v_res_67_ = lp_aesop_Aesop_PhaseSpec_phase(v_x_66_);
lean_dec_ref(v_x_66_);
v_r_68_ = lean_box(v_res_67_);
return v_r_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_toRule(lean_object* v_phase_69_, lean_object* v_ruleExprName_70_, uint8_t v_builder_71_, uint8_t v_scope_72_, lean_object* v_tac_73_, lean_object* v_indexingMode_74_, lean_object* v_pattern_x3f_75_){
_start:
{
uint8_t v___x_76_; uint64_t v___y_78_; 
v___x_76_ = lp_aesop_Aesop_PhaseSpec_phase(v_phase_69_);
if (lean_obj_tag(v_ruleExprName_70_) == 0)
{
uint64_t v___x_108_; 
v___x_108_ = 1723ULL;
v___y_78_ = v___x_108_;
goto v___jp_77_;
}
else
{
uint64_t v_hash_109_; 
v_hash_109_ = lean_ctor_get_uint64(v_ruleExprName_70_, sizeof(void*)*2);
v___y_78_ = v_hash_109_;
goto v___jp_77_;
}
v___jp_77_:
{
uint64_t v___x_79_; uint64_t v___x_80_; uint64_t v___x_81_; uint64_t v___x_82_; uint64_t v___x_83_; uint64_t v___x_84_; lean_object* v_name_85_; 
v___x_79_ = lp_aesop_Aesop_instHashableBuilderName_hash(v_builder_71_);
v___x_80_ = lp_aesop_Aesop_instHashablePhaseName_hash(v___x_76_);
v___x_81_ = lp_aesop_Aesop_instHashableScopeName_hash(v_scope_72_);
v___x_82_ = lean_uint64_mix_hash(v___x_80_, v___x_81_);
v___x_83_ = lean_uint64_mix_hash(v___x_79_, v___x_82_);
v___x_84_ = lean_uint64_mix_hash(v___y_78_, v___x_83_);
v_name_85_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v_name_85_, 0, v_ruleExprName_70_);
lean_ctor_set_uint8(v_name_85_, sizeof(void*)*1 + 8, v_builder_71_);
lean_ctor_set_uint8(v_name_85_, sizeof(void*)*1 + 9, v___x_76_);
lean_ctor_set_uint8(v_name_85_, sizeof(void*)*1 + 10, v_scope_72_);
lean_ctor_set_uint64(v_name_85_, sizeof(void*)*1, v___x_84_);
switch(lean_obj_tag(v_phase_69_))
{
case 0:
{
lean_object* v_info_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_94_; 
v_info_86_ = lean_ctor_get(v_phase_69_, 0);
v_isSharedCheck_94_ = !lean_is_exclusive(v_phase_69_);
if (v_isSharedCheck_94_ == 0)
{
v___x_88_ = v_phase_69_;
v_isShared_89_ = v_isSharedCheck_94_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_info_86_);
lean_dec(v_phase_69_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_94_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___x_90_; lean_object* v___x_92_; 
v___x_90_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_90_, 0, v_name_85_);
lean_ctor_set(v___x_90_, 1, v_indexingMode_74_);
lean_ctor_set(v___x_90_, 2, v_pattern_x3f_75_);
lean_ctor_set(v___x_90_, 3, v_info_86_);
lean_ctor_set(v___x_90_, 4, v_tac_73_);
if (v_isShared_89_ == 0)
{
lean_ctor_set_tag(v___x_88_, 2);
lean_ctor_set(v___x_88_, 0, v___x_90_);
v___x_92_ = v___x_88_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v___x_90_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
case 1:
{
lean_object* v_info_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_103_; 
v_info_95_ = lean_ctor_get(v_phase_69_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v_phase_69_);
if (v_isSharedCheck_103_ == 0)
{
v___x_97_ = v_phase_69_;
v_isShared_98_ = v_isSharedCheck_103_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_info_95_);
lean_dec(v_phase_69_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_103_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_99_; lean_object* v___x_101_; 
v___x_99_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_99_, 0, v_name_85_);
lean_ctor_set(v___x_99_, 1, v_indexingMode_74_);
lean_ctor_set(v___x_99_, 2, v_pattern_x3f_75_);
lean_ctor_set(v___x_99_, 3, v_info_95_);
lean_ctor_set(v___x_99_, 4, v_tac_73_);
if (v_isShared_98_ == 0)
{
lean_ctor_set_tag(v___x_97_, 0);
lean_ctor_set(v___x_97_, 0, v___x_99_);
v___x_101_ = v___x_97_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v___x_99_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
default: 
{
double v_info_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_info_104_ = lean_ctor_get_float(v_phase_69_, 0);
lean_dec_ref_known(v_phase_69_, 0);
v___x_105_ = lean_box_float(v_info_104_);
v___x_106_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_106_, 0, v_name_85_);
lean_ctor_set(v___x_106_, 1, v_indexingMode_74_);
lean_ctor_set(v___x_106_, 2, v_pattern_x3f_75_);
lean_ctor_set(v___x_106_, 3, v___x_105_);
lean_ctor_set(v___x_106_, 4, v_tac_73_);
v___x_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
return v___x_107_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_PhaseSpec_toRule___boxed(lean_object* v_phase_110_, lean_object* v_ruleExprName_111_, lean_object* v_builder_112_, lean_object* v_scope_113_, lean_object* v_tac_114_, lean_object* v_indexingMode_115_, lean_object* v_pattern_x3f_116_){
_start:
{
uint8_t v_builder_boxed_117_; uint8_t v_scope_boxed_118_; lean_object* v_res_119_; 
v_builder_boxed_117_ = lean_unbox(v_builder_112_);
v_scope_boxed_118_ = lean_unbox(v_scope_113_);
v_res_119_ = lp_aesop_Aesop_PhaseSpec_toRule(v_phase_110_, v_ruleExprName_111_, v_builder_boxed_117_, v_scope_boxed_118_, v_tac_114_, v_indexingMode_115_, v_pattern_x3f_116_);
return v_res_119_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleBuilderInput_default___closed__0(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lp_aesop_Aesop_instInhabitedPhaseSpec_default;
v___x_121_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedRuleBuilderOptions_default));
v___x_122_ = lean_box(0);
v___x_123_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleBuilderInput_default(void){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRuleBuilderInput_default___closed__0, &lp_aesop_Aesop_instInhabitedRuleBuilderInput_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedRuleBuilderInput_default___closed__0);
return v___x_124_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleBuilderInput(void){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_aesop_Aesop_instInhabitedRuleBuilderInput_default;
return v___x_125_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderInput_phaseName(lean_object* v_input_126_){
_start:
{
lean_object* v_phase_127_; uint8_t v___x_128_; 
v_phase_127_ = lean_ctor_get(v_input_126_, 2);
v___x_128_ = lp_aesop_Aesop_PhaseSpec_phase(v_phase_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderInput_phaseName___boxed(lean_object* v_input_129_){
_start:
{
uint8_t v_res_130_; lean_object* v_r_131_; 
v_res_130_ = lp_aesop_Aesop_RuleBuilderInput_phaseName(v_input_129_);
lean_dec_ref(v_input_129_);
v_r_131_ = lean_box(v_res_130_);
return v_r_131_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__0(lean_object* v_msgData_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v___x_138_; lean_object* v_env_139_; lean_object* v___x_140_; lean_object* v_mctx_141_; lean_object* v_lctx_142_; lean_object* v_options_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_138_ = lean_st_ref_get(v___y_136_);
v_env_139_ = lean_ctor_get(v___x_138_, 0);
lean_inc_ref(v_env_139_);
lean_dec(v___x_138_);
v___x_140_ = lean_st_ref_get(v___y_134_);
v_mctx_141_ = lean_ctor_get(v___x_140_, 0);
lean_inc_ref(v_mctx_141_);
lean_dec(v___x_140_);
v_lctx_142_ = lean_ctor_get(v___y_133_, 2);
v_options_143_ = lean_ctor_get(v___y_135_, 2);
lean_inc_ref(v_options_143_);
lean_inc_ref(v_lctx_142_);
v___x_144_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_144_, 0, v_env_139_);
lean_ctor_set(v___x_144_, 1, v_mctx_141_);
lean_ctor_set(v___x_144_, 2, v_lctx_142_);
lean_ctor_set(v___x_144_, 3, v_options_143_);
v___x_145_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
lean_ctor_set(v___x_145_, 1, v_msgData_132_);
v___x_146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__0___boxed(lean_object* v_msgData_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__0(v_msgData_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
lean_dec(v___y_151_);
lean_dec_ref(v___y_150_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
return v_res_153_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__2(lean_object* v_opts_154_, lean_object* v_opt_155_){
_start:
{
lean_object* v_name_156_; lean_object* v_defValue_157_; lean_object* v_map_158_; lean_object* v___x_159_; 
v_name_156_ = lean_ctor_get(v_opt_155_, 0);
v_defValue_157_ = lean_ctor_get(v_opt_155_, 1);
v_map_158_ = lean_ctor_get(v_opts_154_, 0);
v___x_159_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_158_, v_name_156_);
if (lean_obj_tag(v___x_159_) == 0)
{
uint8_t v___x_160_; 
v___x_160_ = lean_unbox(v_defValue_157_);
return v___x_160_;
}
else
{
lean_object* v_val_161_; 
v_val_161_ = lean_ctor_get(v___x_159_, 0);
lean_inc(v_val_161_);
lean_dec_ref_known(v___x_159_, 1);
if (lean_obj_tag(v_val_161_) == 1)
{
uint8_t v_v_162_; 
v_v_162_ = lean_ctor_get_uint8(v_val_161_, 0);
lean_dec_ref_known(v_val_161_, 0);
return v_v_162_;
}
else
{
uint8_t v___x_163_; 
lean_dec(v_val_161_);
v___x_163_ = lean_unbox(v_defValue_157_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__2___boxed(lean_object* v_opts_164_, lean_object* v_opt_165_){
_start:
{
uint8_t v_res_166_; lean_object* v_r_167_; 
v_res_166_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__2(v_opts_164_, v_opt_165_);
lean_dec_ref(v_opt_165_);
lean_dec_ref(v_opts_164_);
v_r_167_ = lean_box(v_res_166_);
return v_r_167_;
}
}
static lean_object* _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = lean_box(1);
v___x_169_ = l_Lean_MessageData_ofFormat(v___x_168_);
return v___x_169_;
}
}
static lean_object* _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__3(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_173_ = ((lean_object*)(lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__2));
v___x_174_ = l_Lean_MessageData_ofFormat(v___x_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3(lean_object* v_x_175_, lean_object* v_x_176_){
_start:
{
if (lean_obj_tag(v_x_176_) == 0)
{
return v_x_175_;
}
else
{
lean_object* v_head_177_; lean_object* v_tail_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_200_; 
v_head_177_ = lean_ctor_get(v_x_176_, 0);
v_tail_178_ = lean_ctor_get(v_x_176_, 1);
v_isSharedCheck_200_ = !lean_is_exclusive(v_x_176_);
if (v_isSharedCheck_200_ == 0)
{
v___x_180_ = v_x_176_;
v_isShared_181_ = v_isSharedCheck_200_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_tail_178_);
lean_inc(v_head_177_);
lean_dec(v_x_176_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_200_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v_before_182_; lean_object* v___x_184_; uint8_t v_isShared_185_; uint8_t v_isSharedCheck_198_; 
v_before_182_ = lean_ctor_get(v_head_177_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v_head_177_);
if (v_isSharedCheck_198_ == 0)
{
lean_object* v_unused_199_; 
v_unused_199_ = lean_ctor_get(v_head_177_, 1);
lean_dec(v_unused_199_);
v___x_184_ = v_head_177_;
v_isShared_185_ = v_isSharedCheck_198_;
goto v_resetjp_183_;
}
else
{
lean_inc(v_before_182_);
lean_dec(v_head_177_);
v___x_184_ = lean_box(0);
v_isShared_185_ = v_isSharedCheck_198_;
goto v_resetjp_183_;
}
v_resetjp_183_:
{
lean_object* v___x_186_; lean_object* v___x_188_; 
v___x_186_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_185_ == 0)
{
lean_ctor_set_tag(v___x_184_, 7);
lean_ctor_set(v___x_184_, 1, v___x_186_);
lean_ctor_set(v___x_184_, 0, v_x_175_);
v___x_188_ = v___x_184_;
goto v_reusejp_187_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_x_175_);
lean_ctor_set(v_reuseFailAlloc_197_, 1, v___x_186_);
v___x_188_ = v_reuseFailAlloc_197_;
goto v_reusejp_187_;
}
v_reusejp_187_:
{
lean_object* v___x_189_; lean_object* v___x_191_; 
v___x_189_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__3, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__3_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__3);
if (v_isShared_181_ == 0)
{
lean_ctor_set_tag(v___x_180_, 7);
lean_ctor_set(v___x_180_, 1, v___x_189_);
lean_ctor_set(v___x_180_, 0, v___x_188_);
v___x_191_ = v___x_180_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_188_);
lean_ctor_set(v_reuseFailAlloc_196_, 1, v___x_189_);
v___x_191_ = v_reuseFailAlloc_196_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_192_ = l_Lean_MessageData_ofSyntax(v_before_182_);
v___x_193_ = l_Lean_indentD(v___x_192_);
v___x_194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_191_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v_x_175_ = v___x_194_;
v_x_176_ = v_tail_178_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_204_ = ((lean_object*)(lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__1));
v___x_205_ = l_Lean_MessageData_ofFormat(v___x_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg(lean_object* v_msgData_206_, lean_object* v_macroStack_207_, lean_object* v___y_208_){
_start:
{
lean_object* v_options_210_; lean_object* v___x_211_; uint8_t v___x_212_; 
v_options_210_ = lean_ctor_get(v___y_208_, 2);
v___x_211_ = l_Lean_Elab_pp_macroStack;
v___x_212_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__2(v_options_210_, v___x_211_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; 
lean_dec(v_macroStack_207_);
v___x_213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_213_, 0, v_msgData_206_);
return v___x_213_;
}
else
{
if (lean_obj_tag(v_macroStack_207_) == 0)
{
lean_object* v___x_214_; 
v___x_214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_214_, 0, v_msgData_206_);
return v___x_214_;
}
else
{
lean_object* v_head_215_; lean_object* v_after_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_231_; 
v_head_215_ = lean_ctor_get(v_macroStack_207_, 0);
lean_inc(v_head_215_);
v_after_216_ = lean_ctor_get(v_head_215_, 1);
v_isSharedCheck_231_ = !lean_is_exclusive(v_head_215_);
if (v_isSharedCheck_231_ == 0)
{
lean_object* v_unused_232_; 
v_unused_232_ = lean_ctor_get(v_head_215_, 0);
lean_dec(v_unused_232_);
v___x_218_ = v_head_215_;
v_isShared_219_ = v_isSharedCheck_231_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_after_216_);
lean_dec(v_head_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_231_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_220_; lean_object* v___x_222_; 
v___x_220_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_219_ == 0)
{
lean_ctor_set_tag(v___x_218_, 7);
lean_ctor_set(v___x_218_, 1, v___x_220_);
lean_ctor_set(v___x_218_, 0, v_msgData_206_);
v___x_222_ = v___x_218_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_msgData_206_);
lean_ctor_set(v_reuseFailAlloc_230_, 1, v___x_220_);
v___x_222_ = v_reuseFailAlloc_230_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v_msgData_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_223_ = lean_obj_once(&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__2, &lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__2_once, _init_lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___closed__2);
v___x_224_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_222_);
lean_ctor_set(v___x_224_, 1, v___x_223_);
v___x_225_ = l_Lean_MessageData_ofSyntax(v_after_216_);
v___x_226_ = l_Lean_indentD(v___x_225_);
v_msgData_227_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_227_, 0, v___x_224_);
lean_ctor_set(v_msgData_227_, 1, v___x_226_);
v___x_228_ = lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1_spec__3(v_msgData_227_, v_macroStack_207_);
v___x_229_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
return v___x_229_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg___boxed(lean_object* v_msgData_233_, lean_object* v_macroStack_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg(v_msgData_233_, v_macroStack_234_, v___y_235_);
lean_dec_ref(v___y_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg(lean_object* v_msg_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
lean_object* v_ref_246_; lean_object* v___x_247_; lean_object* v_a_248_; lean_object* v_macroStack_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_260_; 
v_ref_246_ = lean_ctor_get(v___y_243_, 5);
v___x_247_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__0(v_msg_238_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
v_a_248_ = lean_ctor_get(v___x_247_, 0);
lean_inc(v_a_248_);
lean_dec_ref(v___x_247_);
v_macroStack_249_ = lean_ctor_get(v___y_239_, 1);
v___x_250_ = l_Lean_Elab_getBetterRef(v_ref_246_, v_macroStack_249_);
lean_inc(v_macroStack_249_);
v___x_251_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg(v_a_248_, v_macroStack_249_, v___y_243_);
v_a_252_ = lean_ctor_get(v___x_251_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v___x_251_);
if (v_isSharedCheck_260_ == 0)
{
v___x_254_ = v___x_251_;
v_isShared_255_ = v_isSharedCheck_260_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_251_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_260_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_256_; lean_object* v___x_258_; 
v___x_256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_250_);
lean_ctor_set(v___x_256_, 1, v_a_252_);
if (v_isShared_255_ == 0)
{
lean_ctor_set_tag(v___x_254_, 1);
lean_ctor_set(v___x_254_, 0, v___x_256_);
v___x_258_ = v___x_254_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v___x_256_);
v___x_258_ = v_reuseFailAlloc_259_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
return v___x_258_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg___boxed(lean_object* v_msg_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg(v_msg_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
return v_res_269_;
}
}
static lean_object* _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__1(void){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_271_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__0));
v___x_272_ = l_Lean_stringToMessageData(v___x_271_);
return v___x_272_;
}
}
static lean_object* _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__3(void){
_start:
{
lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_274_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__2));
v___x_275_ = l_Lean_stringToMessageData(v___x_274_);
return v___x_275_;
}
}
static lean_object* _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__5(void){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__4));
v___x_278_ = l_Lean_stringToMessageData(v___x_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent(uint8_t v_builderName_287_, lean_object* v_term_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_, lean_object* v_a_294_){
_start:
{
lean_object* v___x_296_; 
lean_inc(v_term_288_);
v___x_296_ = lp_aesop_Aesop_elabGlobalRuleIdent_x3f(v_term_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v_a_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_326_; 
v_a_297_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_326_ == 0)
{
v___x_299_ = v___x_296_;
v_isShared_300_ = v_isSharedCheck_326_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_a_297_);
lean_dec(v___x_296_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_326_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
if (lean_obj_tag(v_a_297_) == 1)
{
lean_object* v_val_301_; lean_object* v___x_303_; 
lean_dec(v_term_288_);
v_val_301_ = lean_ctor_get(v_a_297_, 0);
lean_inc(v_val_301_);
lean_dec_ref_known(v_a_297_, 1);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 0, v_val_301_);
v___x_303_ = v___x_299_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_val_301_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
else
{
lean_object* v___x_305_; lean_object* v___y_307_; 
lean_del_object(v___x_299_);
lean_dec(v_a_297_);
v___x_305_ = lean_obj_once(&lp_aesop_Aesop_elabGlobalRuleIdent___closed__1, &lp_aesop_Aesop_elabGlobalRuleIdent___closed__1_once, _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__1);
switch(v_builderName_287_)
{
case 0:
{
lean_object* v___x_318_; 
v___x_318_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__6));
v___y_307_ = v___x_318_;
goto v___jp_306_;
}
case 1:
{
lean_object* v___x_319_; 
v___x_319_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__7));
v___y_307_ = v___x_319_;
goto v___jp_306_;
}
case 2:
{
lean_object* v___x_320_; 
v___x_320_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__8));
v___y_307_ = v___x_320_;
goto v___jp_306_;
}
case 3:
{
lean_object* v___x_321_; 
v___x_321_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__9));
v___y_307_ = v___x_321_;
goto v___jp_306_;
}
case 4:
{
lean_object* v___x_322_; 
v___x_322_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__10));
v___y_307_ = v___x_322_;
goto v___jp_306_;
}
case 5:
{
lean_object* v___x_323_; 
v___x_323_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__11));
v___y_307_ = v___x_323_;
goto v___jp_306_;
}
case 6:
{
lean_object* v___x_324_; 
v___x_324_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__12));
v___y_307_ = v___x_324_;
goto v___jp_306_;
}
default: 
{
lean_object* v___x_325_; 
v___x_325_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__13));
v___y_307_ = v___x_325_;
goto v___jp_306_;
}
}
v___jp_306_:
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
lean_inc_ref(v___y_307_);
v___x_308_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_308_, 0, v___y_307_);
v___x_309_ = l_Lean_MessageData_ofFormat(v___x_308_);
v___x_310_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_305_);
lean_ctor_set(v___x_310_, 1, v___x_309_);
v___x_311_ = lean_obj_once(&lp_aesop_Aesop_elabGlobalRuleIdent___closed__3, &lp_aesop_Aesop_elabGlobalRuleIdent___closed__3_once, _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__3);
v___x_312_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_310_);
lean_ctor_set(v___x_312_, 1, v___x_311_);
v___x_313_ = l_Lean_MessageData_ofSyntax(v_term_288_);
v___x_314_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_314_, 0, v___x_312_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
v___x_315_ = lean_obj_once(&lp_aesop_Aesop_elabGlobalRuleIdent___closed__5, &lp_aesop_Aesop_elabGlobalRuleIdent___closed__5_once, _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__5);
v___x_316_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_314_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg(v___x_316_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
return v___x_317_;
}
}
}
}
else
{
lean_object* v_a_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_334_; 
lean_dec(v_term_288_);
v_a_327_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_334_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_334_ == 0)
{
v___x_329_ = v___x_296_;
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_a_327_);
lean_dec(v___x_296_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_334_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_332_; 
if (v_isShared_330_ == 0)
{
v___x_332_ = v___x_329_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_a_327_);
v___x_332_ = v_reuseFailAlloc_333_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
return v___x_332_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent___boxed(lean_object* v_builderName_335_, lean_object* v_term_336_, lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_){
_start:
{
uint8_t v_builderName_boxed_344_; lean_object* v_res_345_; 
v_builderName_boxed_344_ = lean_unbox(v_builderName_335_);
v_res_345_ = lp_aesop_Aesop_elabGlobalRuleIdent(v_builderName_boxed_344_, v_term_336_, v_a_337_, v_a_338_, v_a_339_, v_a_340_, v_a_341_, v_a_342_);
lean_dec(v_a_342_);
lean_dec_ref(v_a_341_);
lean_dec(v_a_340_);
lean_dec_ref(v_a_339_);
lean_dec(v_a_338_);
lean_dec_ref(v_a_337_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0(lean_object* v_00_u03b1_346_, lean_object* v_msg_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_){
_start:
{
lean_object* v___x_355_; 
v___x_355_ = lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg(v_msg_347_, v___y_348_, v___y_349_, v___y_350_, v___y_351_, v___y_352_, v___y_353_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___boxed(lean_object* v_00_u03b1_356_, lean_object* v_msg_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0(v_00_u03b1_356_, v_msg_357_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_);
lean_dec(v___y_363_);
lean_dec_ref(v___y_362_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1(lean_object* v_msgData_366_, lean_object* v_macroStack_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___redArg(v_msgData_366_, v_macroStack_367_, v___y_372_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1___boxed(lean_object* v_msgData_376_, lean_object* v_macroStack_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0_spec__1(v_msgData_376_, v_macroStack_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
return v_res_385_;
}
}
static lean_object* _init_lp_aesop_Aesop_elabInductiveRuleIdent___closed__1(void){
_start:
{
lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_387_ = ((lean_object*)(lp_aesop_Aesop_elabInductiveRuleIdent___closed__0));
v___x_388_ = l_Lean_stringToMessageData(v___x_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent(uint8_t v_builderName_389_, lean_object* v_term_390_, uint8_t v_md_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
lean_object* v___x_399_; 
lean_inc(v_term_390_);
v___x_399_ = lp_aesop_Aesop_elabInductiveRuleIdent_x3f(v_term_390_, v_md_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_429_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_429_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_429_ == 0)
{
v___x_402_ = v___x_399_;
v_isShared_403_ = v_isSharedCheck_429_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_399_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_429_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
if (lean_obj_tag(v_a_400_) == 1)
{
lean_object* v_val_404_; lean_object* v___x_406_; 
lean_dec(v_term_390_);
v_val_404_ = lean_ctor_get(v_a_400_, 0);
lean_inc(v_val_404_);
lean_dec_ref_known(v_a_400_, 1);
if (v_isShared_403_ == 0)
{
lean_ctor_set(v___x_402_, 0, v_val_404_);
v___x_406_ = v___x_402_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_407_; 
v_reuseFailAlloc_407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_407_, 0, v_val_404_);
v___x_406_ = v_reuseFailAlloc_407_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
return v___x_406_;
}
}
else
{
lean_object* v___x_408_; lean_object* v___y_410_; 
lean_del_object(v___x_402_);
lean_dec(v_a_400_);
v___x_408_ = lean_obj_once(&lp_aesop_Aesop_elabGlobalRuleIdent___closed__1, &lp_aesop_Aesop_elabGlobalRuleIdent___closed__1_once, _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__1);
switch(v_builderName_389_)
{
case 0:
{
lean_object* v___x_421_; 
v___x_421_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__6));
v___y_410_ = v___x_421_;
goto v___jp_409_;
}
case 1:
{
lean_object* v___x_422_; 
v___x_422_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__7));
v___y_410_ = v___x_422_;
goto v___jp_409_;
}
case 2:
{
lean_object* v___x_423_; 
v___x_423_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__8));
v___y_410_ = v___x_423_;
goto v___jp_409_;
}
case 3:
{
lean_object* v___x_424_; 
v___x_424_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__9));
v___y_410_ = v___x_424_;
goto v___jp_409_;
}
case 4:
{
lean_object* v___x_425_; 
v___x_425_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__10));
v___y_410_ = v___x_425_;
goto v___jp_409_;
}
case 5:
{
lean_object* v___x_426_; 
v___x_426_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__11));
v___y_410_ = v___x_426_;
goto v___jp_409_;
}
case 6:
{
lean_object* v___x_427_; 
v___x_427_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__12));
v___y_410_ = v___x_427_;
goto v___jp_409_;
}
default: 
{
lean_object* v___x_428_; 
v___x_428_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent___closed__13));
v___y_410_ = v___x_428_;
goto v___jp_409_;
}
}
v___jp_409_:
{
lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
lean_inc_ref(v___y_410_);
v___x_411_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_411_, 0, v___y_410_);
v___x_412_ = l_Lean_MessageData_ofFormat(v___x_411_);
v___x_413_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_413_, 0, v___x_408_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
v___x_414_ = lean_obj_once(&lp_aesop_Aesop_elabGlobalRuleIdent___closed__3, &lp_aesop_Aesop_elabGlobalRuleIdent___closed__3_once, _init_lp_aesop_Aesop_elabGlobalRuleIdent___closed__3);
v___x_415_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_413_);
lean_ctor_set(v___x_415_, 1, v___x_414_);
v___x_416_ = l_Lean_MessageData_ofSyntax(v_term_390_);
v___x_417_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_417_, 0, v___x_415_);
lean_ctor_set(v___x_417_, 1, v___x_416_);
v___x_418_ = lean_obj_once(&lp_aesop_Aesop_elabInductiveRuleIdent___closed__1, &lp_aesop_Aesop_elabInductiveRuleIdent___closed__1_once, _init_lp_aesop_Aesop_elabInductiveRuleIdent___closed__1);
v___x_419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_417_);
lean_ctor_set(v___x_419_, 1, v___x_418_);
v___x_420_ = lp_aesop_Lean_throwError___at___00Aesop_elabGlobalRuleIdent_spec__0___redArg(v___x_419_, v_a_392_, v_a_393_, v_a_394_, v_a_395_, v_a_396_, v_a_397_);
return v___x_420_;
}
}
}
}
else
{
lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
lean_dec(v_term_390_);
v_a_430_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_437_ == 0)
{
v___x_432_ = v___x_399_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___x_399_);
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
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent___boxed(lean_object* v_builderName_438_, lean_object* v_term_439_, lean_object* v_md_440_, lean_object* v_a_441_, lean_object* v_a_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_){
_start:
{
uint8_t v_builderName_boxed_448_; uint8_t v_md_boxed_449_; lean_object* v_res_450_; 
v_builderName_boxed_448_ = lean_unbox(v_builderName_438_);
v_md_boxed_449_ = lean_unbox(v_md_440_);
v_res_450_ = lp_aesop_Aesop_elabInductiveRuleIdent(v_builderName_boxed_448_, v_term_439_, v_md_boxed_449_, v_a_441_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_);
lean_dec(v_a_446_);
lean_dec_ref(v_a_445_);
lean_dec(v_a_444_);
lean_dec_ref(v_a_443_);
lean_dec(v_a_442_);
lean_dec_ref(v_a_441_);
return v_res_450_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet_Member(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Builder_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet_Member(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedPhaseSpec_default = _init_lp_aesop_Aesop_instInhabitedPhaseSpec_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPhaseSpec_default);
lp_aesop_Aesop_instInhabitedPhaseSpec = _init_lp_aesop_Aesop_instInhabitedPhaseSpec();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedPhaseSpec);
lp_aesop_Aesop_instInhabitedRuleBuilderInput_default = _init_lp_aesop_Aesop_instInhabitedRuleBuilderInput_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleBuilderInput_default);
lp_aesop_Aesop_instInhabitedRuleBuilderInput = _init_lp_aesop_Aesop_instInhabitedRuleBuilderInput();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleBuilderInput);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Builder_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleSet_Member(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Builder_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet_Member(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Builder_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
