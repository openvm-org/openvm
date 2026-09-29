// Lean compiler output
// Module: Aesop.RuleTac.ElabRuleTerm
// Imports: public import Init public meta import Init public import Lean.Elab.Tactic.Basic public import Lean.Meta.Tactic.Simp.Simproc public import Aesop.ElabM import Lean.Elab.Tactic.Simp
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
lean_object* l_Lean_Elab_Term_withoutAutoBoundImplicit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabSimpArgs(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_isInductiveCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermForApply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t l_Lean_Syntax_isIdent(lean_object*);
lean_object* l_Lean_Elab_Term_resolveId_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn_x27(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint8_t l_Lean_Meta_TransparencyMode_lt(uint8_t, uint8_t);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_elabGlobalRuleIdent_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabGlobalRuleIdent_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__0 = (const lean_object*)&lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__0_value;
static lean_once_cell_t lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1;
static const lean_string_object lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is not an inductive type"};
static const lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__2 = (const lean_object*)&lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0;
static const lean_string_object lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__1 = (const lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__1_value;
static const lean_ctor_object lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__1_value)}};
static const lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__2 = (const lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__2_value;
static lean_once_cell_t lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_runTacticMAsTermElabM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runTacticMAsTermElabM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__0_value;
static const lean_array_object lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__1 = (const lean_object*)&lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__2 = (const lean_object*)&lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__3 = (const lean_object*)&lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_elabSimpTheorems_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_elabSimpTheorems_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "aesop: simp builder currently does not support wildcard '*'"};
static const lean_object* lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__0 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_mkSimpArgs___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkSimpArgs___closed__1;
static lean_once_cell_t lp_aesop_Aesop_mkSimpArgs___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkSimpArgs___closed__2;
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__3 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__3_value;
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__4 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__4_value;
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__5 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__5_value;
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__6 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_mkSimpArgs___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_mkSimpArgs___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__7_value_aux_0),((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_mkSimpArgs___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__7_value_aux_1),((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_mkSimpArgs___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__7_value_aux_2),((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__6_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__7 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__7_value;
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__8 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_mkSimpArgs___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__9 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__9_value;
static lean_once_cell_t lp_aesop_Aesop_mkSimpArgs___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkSimpArgs___closed__10;
static lean_once_cell_t lp_aesop_Aesop_mkSimpArgs___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkSimpArgs___closed__11;
static const lean_string_object lp_aesop_Aesop_mkSimpArgs___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_Aesop_mkSimpArgs___closed__12 = (const lean_object*)&lp_aesop_Aesop_mkSimpArgs___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_mkSimpArgs___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkSimpArgs___closed__13;
static lean_once_cell_t lp_aesop_Aesop_mkSimpArgs___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkSimpArgs___closed__14;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkSimpArgs(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0(lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1(lean_object*);
static const lean_ctor_object lp_aesop_Aesop_checkElabRuleTermForSimp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__2;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__4;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__6;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__7;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__8;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__9;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__10;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__11;
static lean_once_cell_t lp_aesop_Aesop_checkElabRuleTermForSimp___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpMetaM(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpMetaM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0(lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_, lean_object* v___y_9_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0___closed__0));
v___x_12_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_12_, 0, v___x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0___boxed(lean_object* v_x_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0(v_x_13_, v___y_14_, v___y_15_, v___y_16_, v___y_17_, v___y_18_, v___y_19_);
lean_dec(v___y_19_);
lean_dec_ref(v___y_18_);
lean_dec(v___y_17_);
lean_dec_ref(v___y_16_);
lean_dec(v___y_15_);
lean_dec_ref(v___y_14_);
lean_dec(v_x_13_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f(lean_object* v_stx_23_, lean_object* v_a_24_, lean_object* v_a_25_, lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_, lean_object* v_a_29_){
_start:
{
lean_object* v___y_32_; uint8_t v___y_33_; lean_object* v_a_38_; lean_object* v___y_42_; uint8_t v___x_52_; 
v___x_52_ = l_Lean_Syntax_isIdent(v_stx_23_);
if (v___x_52_ == 0)
{
lean_object* v___x_53_; lean_object* v___x_54_; 
lean_dec(v_stx_23_);
v___x_53_ = lean_box(0);
v___x_54_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
return v___x_54_;
}
else
{
lean_object* v___x_55_; uint8_t v___x_56_; lean_object* v___x_57_; 
v___x_55_ = ((lean_object*)(lp_aesop_Aesop_elabGlobalRuleIdent_x3f___closed__0));
v___x_56_ = 0;
v___x_57_ = l_Lean_Elab_Term_resolveId_x3f(v_stx_23_, v___x_55_, v___x_56_, v_a_24_, v_a_25_, v_a_26_, v_a_27_, v_a_28_, v_a_29_);
if (lean_obj_tag(v___x_57_) == 0)
{
lean_object* v_a_58_; lean_object* v___x_60_; uint8_t v_isShared_61_; uint8_t v_isSharedCheck_77_; 
v_a_58_ = lean_ctor_get(v___x_57_, 0);
v_isSharedCheck_77_ = !lean_is_exclusive(v___x_57_);
if (v_isSharedCheck_77_ == 0)
{
v___x_60_ = v___x_57_;
v_isShared_61_ = v_isSharedCheck_77_;
goto v_resetjp_59_;
}
else
{
lean_inc(v_a_58_);
lean_dec(v___x_57_);
v___x_60_ = lean_box(0);
v_isShared_61_ = v_isSharedCheck_77_;
goto v_resetjp_59_;
}
v_resetjp_59_:
{
if (lean_obj_tag(v_a_58_) == 1)
{
lean_object* v_val_62_; 
v_val_62_ = lean_ctor_get(v_a_58_, 0);
if (lean_obj_tag(v_val_62_) == 4)
{
lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_73_; 
lean_inc_ref(v_val_62_);
v_isSharedCheck_73_ = !lean_is_exclusive(v_a_58_);
if (v_isSharedCheck_73_ == 0)
{
lean_object* v_unused_74_; 
v_unused_74_ = lean_ctor_get(v_a_58_, 0);
lean_dec(v_unused_74_);
v___x_64_ = v_a_58_;
v_isShared_65_ = v_isSharedCheck_73_;
goto v_resetjp_63_;
}
else
{
lean_dec(v_a_58_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_73_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v_declName_66_; lean_object* v___x_68_; 
v_declName_66_ = lean_ctor_get(v_val_62_, 0);
lean_inc(v_declName_66_);
lean_dec_ref_known(v_val_62_, 2);
if (v_isShared_65_ == 0)
{
lean_ctor_set(v___x_64_, 0, v_declName_66_);
v___x_68_ = v___x_64_;
goto v_reusejp_67_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v_declName_66_);
v___x_68_ = v_reuseFailAlloc_72_;
goto v_reusejp_67_;
}
v_reusejp_67_:
{
lean_object* v___x_70_; 
if (v_isShared_61_ == 0)
{
lean_ctor_set(v___x_60_, 0, v___x_68_);
v___x_70_ = v___x_60_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_71_; 
v_reuseFailAlloc_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_71_, 0, v___x_68_);
v___x_70_ = v_reuseFailAlloc_71_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
return v___x_70_;
}
}
}
}
else
{
lean_object* v___x_75_; 
lean_del_object(v___x_60_);
v___x_75_ = lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0(v_a_58_, v_a_24_, v_a_25_, v_a_26_, v_a_27_, v_a_28_, v_a_29_);
lean_dec_ref_known(v_a_58_, 1);
v___y_42_ = v___x_75_;
goto v___jp_41_;
}
}
else
{
lean_object* v___x_76_; 
lean_del_object(v___x_60_);
v___x_76_ = lp_aesop_Aesop_elabGlobalRuleIdent_x3f___lam__0(v_a_58_, v_a_24_, v_a_25_, v_a_26_, v_a_27_, v_a_28_, v_a_29_);
lean_dec(v_a_58_);
v___y_42_ = v___x_76_;
goto v___jp_41_;
}
}
}
else
{
lean_object* v_a_78_; 
v_a_78_ = lean_ctor_get(v___x_57_, 0);
lean_inc(v_a_78_);
lean_dec_ref_known(v___x_57_, 1);
v_a_38_ = v_a_78_;
goto v___jp_37_;
}
}
v___jp_31_:
{
if (v___y_33_ == 0)
{
lean_object* v___x_34_; lean_object* v___x_35_; 
lean_dec_ref(v___y_32_);
v___x_34_ = lean_box(0);
v___x_35_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_35_, 0, v___x_34_);
return v___x_35_;
}
else
{
lean_object* v___x_36_; 
v___x_36_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_36_, 0, v___y_32_);
return v___x_36_;
}
}
v___jp_37_:
{
uint8_t v___x_39_; 
v___x_39_ = l_Lean_Exception_isInterrupt(v_a_38_);
if (v___x_39_ == 0)
{
uint8_t v___x_40_; 
lean_inc_ref(v_a_38_);
v___x_40_ = l_Lean_Exception_isRuntime(v_a_38_);
v___y_32_ = v_a_38_;
v___y_33_ = v___x_40_;
goto v___jp_31_;
}
else
{
v___y_32_ = v_a_38_;
v___y_33_ = v___x_39_;
goto v___jp_31_;
}
}
v___jp_41_:
{
lean_object* v_a_43_; lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_51_; 
v_a_43_ = lean_ctor_get(v___y_42_, 0);
v_isSharedCheck_51_ = !lean_is_exclusive(v___y_42_);
if (v_isSharedCheck_51_ == 0)
{
v___x_45_ = v___y_42_;
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
else
{
lean_inc(v_a_43_);
lean_dec(v___y_42_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_51_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v_a_47_; lean_object* v___x_49_; 
v_a_47_ = lean_ctor_get(v_a_43_, 0);
lean_inc(v_a_47_);
lean_dec(v_a_43_);
if (v_isShared_46_ == 0)
{
lean_ctor_set(v___x_45_, 0, v_a_47_);
v___x_49_ = v___x_45_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v_a_47_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabGlobalRuleIdent_x3f___boxed(lean_object* v_stx_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_aesop_Aesop_elabGlobalRuleIdent_x3f(v_stx_79_, v_a_80_, v_a_81_, v_a_82_, v_a_83_, v_a_84_, v_a_85_);
lean_dec(v_a_85_);
lean_dec_ref(v_a_84_);
lean_dec(v_a_83_);
lean_dec_ref(v_a_82_);
lean_dec(v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___lam__0(lean_object* v_k_88_, lean_object* v_b_89_, lean_object* v_c_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v___x_96_; 
lean_inc(v___y_94_);
lean_inc_ref(v___y_93_);
lean_inc(v___y_92_);
lean_inc_ref(v___y_91_);
v___x_96_ = lean_apply_7(v_k_88_, v_b_89_, v_c_90_, v___y_91_, v___y_92_, v___y_93_, v___y_94_, lean_box(0));
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___lam__0___boxed(lean_object* v_k_97_, lean_object* v_b_98_, lean_object* v_c_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___lam__0(v_k_97_, v_b_98_, v_c_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_);
lean_dec(v___y_103_);
lean_dec_ref(v___y_102_);
lean_dec(v___y_101_);
lean_dec_ref(v___y_100_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg(lean_object* v_type_106_, lean_object* v_k_107_, uint8_t v_cleanupAnnotations_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v___f_114_; uint8_t v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___f_114_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_114_, 0, v_k_107_);
v___x_115_ = 0;
v___x_116_ = lean_box(0);
v___x_117_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_115_, v___x_116_, v_type_106_, v___f_114_, v_cleanupAnnotations_108_, v___x_115_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
if (lean_obj_tag(v___x_117_) == 0)
{
lean_object* v_a_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_125_; 
v_a_118_ = lean_ctor_get(v___x_117_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_117_);
if (v_isSharedCheck_125_ == 0)
{
v___x_120_ = v___x_117_;
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_a_118_);
lean_dec(v___x_117_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v_a_118_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
else
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_133_; 
v_a_126_ = lean_ctor_get(v___x_117_, 0);
v_isSharedCheck_133_ = !lean_is_exclusive(v___x_117_);
if (v_isSharedCheck_133_ == 0)
{
v___x_128_ = v___x_117_;
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_117_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_131_; 
if (v_isShared_129_ == 0)
{
v___x_131_ = v___x_128_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_a_126_);
v___x_131_ = v_reuseFailAlloc_132_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
return v___x_131_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg___boxed(lean_object* v_type_134_, lean_object* v_k_135_, lean_object* v_cleanupAnnotations_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_142_; lean_object* v_res_143_; 
v_cleanupAnnotations_boxed_142_ = lean_unbox(v_cleanupAnnotations_136_);
v_res_143_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg(v_type_134_, v_k_135_, v_cleanupAnnotations_boxed_142_, v___y_137_, v___y_138_, v___y_139_, v___y_140_);
lean_dec(v___y_140_);
lean_dec_ref(v___y_139_);
lean_dec(v___y_138_);
lean_dec_ref(v___y_137_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1(lean_object* v_00_u03b1_144_, lean_object* v_type_145_, lean_object* v_k_146_, uint8_t v_cleanupAnnotations_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg(v_type_145_, v_k_146_, v_cleanupAnnotations_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___boxed(lean_object* v_00_u03b1_154_, lean_object* v_type_155_, lean_object* v_k_156_, lean_object* v_cleanupAnnotations_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_163_; lean_object* v_res_164_; 
v_cleanupAnnotations_boxed_163_ = lean_unbox(v_cleanupAnnotations_157_);
v_res_164_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1(v_00_u03b1_154_, v_type_155_, v_k_156_, v_cleanupAnnotations_boxed_163_, v___y_158_, v___y_159_, v___y_160_, v___y_161_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_158_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg(lean_object* v_x_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = l_Lean_Meta_saveState___redArg(v___y_167_, v___y_169_);
if (lean_obj_tag(v___x_171_) == 0)
{
lean_object* v_a_172_; lean_object* v_r_173_; 
v_a_172_ = lean_ctor_get(v___x_171_, 0);
lean_inc(v_a_172_);
lean_dec_ref_known(v___x_171_, 1);
lean_inc(v___y_169_);
lean_inc_ref(v___y_168_);
lean_inc(v___y_167_);
lean_inc_ref(v___y_166_);
v_r_173_ = lean_apply_5(v_x_165_, v___y_166_, v___y_167_, v___y_168_, v___y_169_, lean_box(0));
if (lean_obj_tag(v_r_173_) == 0)
{
lean_object* v_a_174_; lean_object* v___x_175_; 
v_a_174_ = lean_ctor_get(v_r_173_, 0);
lean_inc(v_a_174_);
lean_dec_ref_known(v_r_173_, 1);
v___x_175_ = l_Lean_Meta_SavedState_restore___redArg(v_a_172_, v___y_167_, v___y_169_);
lean_dec(v_a_172_);
if (lean_obj_tag(v___x_175_) == 0)
{
lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_182_; 
v_isSharedCheck_182_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_182_ == 0)
{
lean_object* v_unused_183_; 
v_unused_183_ = lean_ctor_get(v___x_175_, 0);
lean_dec(v_unused_183_);
v___x_177_ = v___x_175_;
v_isShared_178_ = v_isSharedCheck_182_;
goto v_resetjp_176_;
}
else
{
lean_dec(v___x_175_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_182_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___x_180_; 
if (v_isShared_178_ == 0)
{
lean_ctor_set(v___x_177_, 0, v_a_174_);
v___x_180_ = v___x_177_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_181_; 
v_reuseFailAlloc_181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_181_, 0, v_a_174_);
v___x_180_ = v_reuseFailAlloc_181_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
return v___x_180_;
}
}
}
else
{
lean_object* v_a_184_; lean_object* v___x_186_; uint8_t v_isShared_187_; uint8_t v_isSharedCheck_191_; 
lean_dec(v_a_174_);
v_a_184_ = lean_ctor_get(v___x_175_, 0);
v_isSharedCheck_191_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_191_ == 0)
{
v___x_186_ = v___x_175_;
v_isShared_187_ = v_isSharedCheck_191_;
goto v_resetjp_185_;
}
else
{
lean_inc(v_a_184_);
lean_dec(v___x_175_);
v___x_186_ = lean_box(0);
v_isShared_187_ = v_isSharedCheck_191_;
goto v_resetjp_185_;
}
v_resetjp_185_:
{
lean_object* v___x_189_; 
if (v_isShared_187_ == 0)
{
v___x_189_ = v___x_186_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_190_; 
v_reuseFailAlloc_190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_190_, 0, v_a_184_);
v___x_189_ = v_reuseFailAlloc_190_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
return v___x_189_;
}
}
}
}
else
{
lean_object* v_a_192_; lean_object* v___x_193_; 
v_a_192_ = lean_ctor_get(v_r_173_, 0);
lean_inc(v_a_192_);
lean_dec_ref_known(v_r_173_, 1);
v___x_193_ = l_Lean_Meta_SavedState_restore___redArg(v_a_172_, v___y_167_, v___y_169_);
lean_dec(v_a_172_);
if (lean_obj_tag(v___x_193_) == 0)
{
lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_200_; 
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_200_ == 0)
{
lean_object* v_unused_201_; 
v_unused_201_ = lean_ctor_get(v___x_193_, 0);
lean_dec(v_unused_201_);
v___x_195_ = v___x_193_;
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
else
{
lean_dec(v___x_193_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_198_; 
if (v_isShared_196_ == 0)
{
lean_ctor_set_tag(v___x_195_, 1);
lean_ctor_set(v___x_195_, 0, v_a_192_);
v___x_198_ = v___x_195_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v_a_192_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
else
{
lean_object* v_a_202_; lean_object* v___x_204_; uint8_t v_isShared_205_; uint8_t v_isSharedCheck_209_; 
lean_dec(v_a_192_);
v_a_202_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_209_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_209_ == 0)
{
v___x_204_ = v___x_193_;
v_isShared_205_ = v_isSharedCheck_209_;
goto v_resetjp_203_;
}
else
{
lean_inc(v_a_202_);
lean_dec(v___x_193_);
v___x_204_ = lean_box(0);
v_isShared_205_ = v_isSharedCheck_209_;
goto v_resetjp_203_;
}
v_resetjp_203_:
{
lean_object* v___x_207_; 
if (v_isShared_205_ == 0)
{
v___x_207_ = v___x_204_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v_a_202_);
v___x_207_ = v_reuseFailAlloc_208_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
return v___x_207_;
}
}
}
}
}
else
{
lean_object* v_a_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_217_; 
lean_dec_ref(v_x_165_);
v_a_210_ = lean_ctor_get(v___x_171_, 0);
v_isSharedCheck_217_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_217_ == 0)
{
v___x_212_ = v___x_171_;
v_isShared_213_ = v_isSharedCheck_217_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_a_210_);
lean_dec(v___x_171_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_217_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_215_; 
if (v_isShared_213_ == 0)
{
v___x_215_ = v___x_212_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v_a_210_);
v___x_215_ = v_reuseFailAlloc_216_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
return v___x_215_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg___boxed(lean_object* v_x_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg(v_x_218_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
lean_dec(v___y_222_);
lean_dec_ref(v___y_221_);
lean_dec(v___y_220_);
lean_dec_ref(v___y_219_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2(lean_object* v_00_u03b1_225_, lean_object* v_x_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg(v_x_226_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___boxed(lean_object* v_00_u03b1_233_, lean_object* v_x_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2(v_00_u03b1_233_, v_x_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3(lean_object* v_msgData_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v___x_247_; lean_object* v_env_248_; lean_object* v___x_249_; lean_object* v_mctx_250_; lean_object* v_lctx_251_; lean_object* v_options_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_247_ = lean_st_ref_get(v___y_245_);
v_env_248_ = lean_ctor_get(v___x_247_, 0);
lean_inc_ref(v_env_248_);
lean_dec(v___x_247_);
v___x_249_ = lean_st_ref_get(v___y_243_);
v_mctx_250_ = lean_ctor_get(v___x_249_, 0);
lean_inc_ref(v_mctx_250_);
lean_dec(v___x_249_);
v_lctx_251_ = lean_ctor_get(v___y_242_, 2);
v_options_252_ = lean_ctor_get(v___y_244_, 2);
lean_inc_ref(v_options_252_);
lean_inc_ref(v_lctx_251_);
v___x_253_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_253_, 0, v_env_248_);
lean_ctor_set(v___x_253_, 1, v_mctx_250_);
lean_ctor_set(v___x_253_, 2, v_lctx_251_);
lean_ctor_set(v___x_253_, 3, v_options_252_);
v___x_254_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v_msgData_241_);
v___x_255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3___boxed(lean_object* v_msgData_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3(v_msgData_256_, v___y_257_, v___y_258_, v___y_259_, v___y_260_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg(lean_object* v_msg_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_ref_269_; lean_object* v___x_270_; lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_279_; 
v_ref_269_ = lean_ctor_get(v___y_266_, 5);
v___x_270_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3(v_msg_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
v_a_271_ = lean_ctor_get(v___x_270_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_279_ == 0)
{
v___x_273_ = v___x_270_;
v_isShared_274_ = v_isSharedCheck_279_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_270_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_279_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_275_; lean_object* v___x_277_; 
lean_inc(v_ref_269_);
v___x_275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_275_, 0, v_ref_269_);
lean_ctor_set(v___x_275_, 1, v_a_271_);
if (v_isShared_274_ == 0)
{
lean_ctor_set_tag(v___x_273_, 1);
lean_ctor_set(v___x_273_, 0, v___x_275_);
v___x_277_ = v___x_273_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_275_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_msg_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg(v_msg_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
return v_res_286_;
}
}
static lean_object* _init_lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_288_ = ((lean_object*)(lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__0));
v___x_289_ = l_Lean_stringToMessageData(v___x_288_);
return v___x_289_;
}
}
static lean_object* _init_lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = ((lean_object*)(lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__2));
v___x_292_ = l_Lean_stringToMessageData(v___x_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0(lean_object* v_constName_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_){
_start:
{
lean_object* v___x_299_; lean_object* v_env_300_; lean_object* v___x_301_; 
v___x_299_ = lean_st_ref_get(v___y_297_);
v_env_300_ = lean_ctor_get(v___x_299_, 0);
lean_inc_ref(v_env_300_);
lean_dec(v___x_299_);
lean_inc(v_constName_293_);
v___x_301_ = l_Lean_isInductiveCore_x3f(v_env_300_, v_constName_293_);
if (lean_obj_tag(v___x_301_) == 0)
{
lean_object* v___x_302_; uint8_t v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; 
v___x_302_ = lean_obj_once(&lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1, &lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1_once, _init_lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1);
v___x_303_ = 0;
v___x_304_ = l_Lean_MessageData_ofConstName(v_constName_293_, v___x_303_);
v___x_305_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_302_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = lean_obj_once(&lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3, &lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3_once, _init_lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3);
v___x_307_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_305_);
lean_ctor_set(v___x_307_, 1, v___x_306_);
v___x_308_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg(v___x_307_, v___y_294_, v___y_295_, v___y_296_, v___y_297_);
return v___x_308_;
}
else
{
lean_object* v_val_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_316_; 
lean_dec(v_constName_293_);
v_val_309_ = lean_ctor_get(v___x_301_, 0);
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_301_);
if (v_isSharedCheck_316_ == 0)
{
v___x_311_ = v___x_301_;
v_isShared_312_ = v_isSharedCheck_316_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_val_309_);
lean_dec(v___x_301_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_316_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_314_; 
if (v_isShared_312_ == 0)
{
lean_ctor_set_tag(v___x_311_, 0);
v___x_314_ = v___x_311_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v_val_309_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___boxed(lean_object* v_constName_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0(v_constName_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__0(lean_object* v_a_324_, lean_object* v_args_325_, lean_object* v_x_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = l_Lean_mkAppN(v_a_324_, v_args_325_);
lean_inc(v___y_330_);
lean_inc_ref(v___y_329_);
lean_inc(v___y_328_);
lean_inc_ref(v___y_327_);
v___x_333_ = lean_whnf(v___x_332_, v___y_327_, v___y_328_, v___y_329_, v___y_330_);
if (lean_obj_tag(v___x_333_) == 0)
{
lean_object* v_a_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_370_; 
v_a_334_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_370_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_370_ == 0)
{
v___x_336_ = v___x_333_;
v_isShared_337_ = v_isSharedCheck_370_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_a_334_);
lean_dec(v___x_333_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_370_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v___x_338_; 
v___x_338_ = l_Lean_Expr_getAppFn_x27(v_a_334_);
lean_dec(v_a_334_);
if (lean_obj_tag(v___x_338_) == 4)
{
lean_object* v_declName_339_; lean_object* v___x_340_; 
lean_del_object(v___x_336_);
v_declName_339_ = lean_ctor_get(v___x_338_, 0);
lean_inc(v_declName_339_);
lean_dec_ref_known(v___x_338_, 2);
v___x_340_ = lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0(v_declName_339_, v___y_327_, v___y_328_, v___y_329_, v___y_330_);
if (lean_obj_tag(v___x_340_) == 0)
{
lean_object* v_a_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_349_; 
v_a_341_ = lean_ctor_get(v___x_340_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v___x_340_);
if (v_isSharedCheck_349_ == 0)
{
v___x_343_ = v___x_340_;
v_isShared_344_ = v_isSharedCheck_349_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_a_341_);
lean_dec(v___x_340_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_349_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___x_345_; lean_object* v___x_347_; 
v___x_345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_345_, 0, v_a_341_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_345_);
v___x_347_ = v___x_343_;
goto v_reusejp_346_;
}
else
{
lean_object* v_reuseFailAlloc_348_; 
v_reuseFailAlloc_348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_348_, 0, v___x_345_);
v___x_347_ = v_reuseFailAlloc_348_;
goto v_reusejp_346_;
}
v_reusejp_346_:
{
return v___x_347_;
}
}
}
else
{
lean_object* v_a_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_365_; 
v_a_350_ = lean_ctor_get(v___x_340_, 0);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_340_);
if (v_isSharedCheck_365_ == 0)
{
v___x_352_ = v___x_340_;
v_isShared_353_ = v_isSharedCheck_365_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_a_350_);
lean_dec(v___x_340_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_365_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
uint8_t v___y_355_; uint8_t v___x_363_; 
v___x_363_ = l_Lean_Exception_isInterrupt(v_a_350_);
if (v___x_363_ == 0)
{
uint8_t v___x_364_; 
lean_inc(v_a_350_);
v___x_364_ = l_Lean_Exception_isRuntime(v_a_350_);
v___y_355_ = v___x_364_;
goto v___jp_354_;
}
else
{
v___y_355_ = v___x_363_;
goto v___jp_354_;
}
v___jp_354_:
{
if (v___y_355_ == 0)
{
lean_object* v___x_356_; lean_object* v___x_358_; 
lean_dec(v_a_350_);
v___x_356_ = lean_box(0);
if (v_isShared_353_ == 0)
{
lean_ctor_set_tag(v___x_352_, 0);
lean_ctor_set(v___x_352_, 0, v___x_356_);
v___x_358_ = v___x_352_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v___x_356_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
else
{
lean_object* v___x_361_; 
if (v_isShared_353_ == 0)
{
v___x_361_ = v___x_352_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_a_350_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
}
}
else
{
lean_object* v___x_366_; lean_object* v___x_368_; 
lean_dec_ref(v___x_338_);
v___x_366_ = lean_box(0);
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 0, v___x_366_);
v___x_368_ = v___x_336_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_369_; 
v_reuseFailAlloc_369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_369_, 0, v___x_366_);
v___x_368_ = v_reuseFailAlloc_369_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
return v___x_368_;
}
}
}
}
else
{
lean_object* v_a_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_378_; 
v_a_371_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_378_ == 0)
{
v___x_373_ = v___x_333_;
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_a_371_);
lean_dec(v___x_333_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_376_; 
if (v_isShared_374_ == 0)
{
v___x_376_ = v___x_373_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_a_371_);
v___x_376_ = v_reuseFailAlloc_377_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
return v___x_376_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__0___boxed(lean_object* v_a_379_, lean_object* v_args_380_, lean_object* v_x_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__0(v_a_379_, v_args_380_, v_x_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
lean_dec_ref(v_x_381_);
lean_dec_ref(v_args_380_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__1(lean_object* v_decl_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
if (lean_obj_tag(v___x_394_) == 0)
{
lean_object* v_a_395_; lean_object* v___x_396_; 
v_a_395_ = lean_ctor_get(v___x_394_, 0);
lean_inc_n(v_a_395_, 2);
lean_dec_ref_known(v___x_394_, 1);
lean_inc(v___y_392_);
lean_inc_ref(v___y_391_);
lean_inc(v___y_390_);
lean_inc_ref(v___y_389_);
v___x_396_ = lean_infer_type(v_a_395_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v___f_398_; uint8_t v___x_399_; lean_object* v___x_400_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
lean_inc(v_a_397_);
lean_dec_ref_known(v___x_396_, 1);
v___f_398_ = lean_alloc_closure((void*)(lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__0___boxed), 8, 1);
lean_closure_set(v___f_398_, 0, v_a_395_);
v___x_399_ = 0;
v___x_400_ = lp_aesop_Lean_Meta_forallTelescope___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__1___redArg(v_a_397_, v___f_398_, v___x_399_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
return v___x_400_;
}
else
{
lean_object* v_a_401_; lean_object* v___x_403_; uint8_t v_isShared_404_; uint8_t v_isSharedCheck_408_; 
lean_dec(v_a_395_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
v_a_401_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_408_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_408_ == 0)
{
v___x_403_ = v___x_396_;
v_isShared_404_ = v_isSharedCheck_408_;
goto v_resetjp_402_;
}
else
{
lean_inc(v_a_401_);
lean_dec(v___x_396_);
v___x_403_ = lean_box(0);
v_isShared_404_ = v_isSharedCheck_408_;
goto v_resetjp_402_;
}
v_resetjp_402_:
{
lean_object* v___x_406_; 
if (v_isShared_404_ == 0)
{
v___x_406_ = v___x_403_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_407_; 
v_reuseFailAlloc_407_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_407_, 0, v_a_401_);
v___x_406_ = v_reuseFailAlloc_407_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
return v___x_406_;
}
}
}
}
else
{
lean_object* v_a_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_416_; 
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
v_a_409_ = lean_ctor_get(v___x_394_, 0);
v_isSharedCheck_416_ = !lean_is_exclusive(v___x_394_);
if (v_isSharedCheck_416_ == 0)
{
v___x_411_ = v___x_394_;
v_isShared_412_ = v_isSharedCheck_416_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_a_409_);
lean_dec(v___x_394_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_416_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_414_; 
if (v_isShared_412_ == 0)
{
v___x_414_ = v___x_411_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_a_409_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__1___boxed(lean_object* v_decl_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__1(v_decl_417_, v___y_418_, v___y_419_, v___y_420_, v___y_421_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f(lean_object* v_decl_424_, lean_object* v_a_425_, lean_object* v_a_426_, lean_object* v_a_427_, lean_object* v_a_428_){
_start:
{
lean_object* v___f_430_; lean_object* v___x_431_; 
v___f_430_ = lean_alloc_closure((void*)(lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___lam__1___boxed), 6, 1);
lean_closure_set(v___f_430_, 0, v_decl_424_);
v___x_431_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__2___redArg(v___f_430_, v_a_425_, v_a_426_, v_a_427_, v_a_428_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_matchInductiveTypeSynonym_x3f___boxed(lean_object* v_decl_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_aesop_Aesop_matchInductiveTypeSynonym_x3f(v_decl_432_, v_a_433_, v_a_434_, v_a_435_, v_a_436_);
lean_dec(v_a_436_);
lean_dec_ref(v_a_435_);
lean_dec(v_a_434_);
lean_dec_ref(v_a_433_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0(lean_object* v_00_u03b1_439_, lean_object* v_msg_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___redArg(v_msg_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0___boxed(lean_object* v_00_u03b1_447_, lean_object* v_msg_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0(v_00_u03b1_447_, v_msg_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
lean_dec(v___y_452_);
lean_dec_ref(v___y_451_);
lean_dec(v___y_450_);
lean_dec_ref(v___y_449_);
return v_res_454_;
}
}
static lean_object* _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_455_ = lean_box(1);
v___x_456_ = l_Lean_MessageData_ofFormat(v___x_455_);
return v___x_456_;
}
}
static lean_object* _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__3(void){
_start:
{
lean_object* v___x_460_; lean_object* v___x_461_; 
v___x_460_ = ((lean_object*)(lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__2));
v___x_461_ = l_Lean_MessageData_ofFormat(v___x_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3(lean_object* v_x_462_, lean_object* v_x_463_){
_start:
{
if (lean_obj_tag(v_x_463_) == 0)
{
return v_x_462_;
}
else
{
lean_object* v_head_464_; lean_object* v_tail_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_487_; 
v_head_464_ = lean_ctor_get(v_x_463_, 0);
v_tail_465_ = lean_ctor_get(v_x_463_, 1);
v_isSharedCheck_487_ = !lean_is_exclusive(v_x_463_);
if (v_isSharedCheck_487_ == 0)
{
v___x_467_ = v_x_463_;
v_isShared_468_ = v_isSharedCheck_487_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_tail_465_);
lean_inc(v_head_464_);
lean_dec(v_x_463_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_487_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v_before_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_485_; 
v_before_469_ = lean_ctor_get(v_head_464_, 0);
v_isSharedCheck_485_ = !lean_is_exclusive(v_head_464_);
if (v_isSharedCheck_485_ == 0)
{
lean_object* v_unused_486_; 
v_unused_486_ = lean_ctor_get(v_head_464_, 1);
lean_dec(v_unused_486_);
v___x_471_ = v_head_464_;
v_isShared_472_ = v_isSharedCheck_485_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_before_469_);
lean_dec(v_head_464_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_485_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_473_; lean_object* v___x_475_; 
v___x_473_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_472_ == 0)
{
lean_ctor_set_tag(v___x_471_, 7);
lean_ctor_set(v___x_471_, 1, v___x_473_);
lean_ctor_set(v___x_471_, 0, v_x_462_);
v___x_475_ = v___x_471_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_x_462_);
lean_ctor_set(v_reuseFailAlloc_484_, 1, v___x_473_);
v___x_475_ = v_reuseFailAlloc_484_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
lean_object* v___x_476_; lean_object* v___x_478_; 
v___x_476_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__3, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__3_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__3);
if (v_isShared_468_ == 0)
{
lean_ctor_set_tag(v___x_467_, 7);
lean_ctor_set(v___x_467_, 1, v___x_476_);
lean_ctor_set(v___x_467_, 0, v___x_475_);
v___x_478_ = v___x_467_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v___x_475_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v___x_476_);
v___x_478_ = v_reuseFailAlloc_483_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_479_ = l_Lean_MessageData_ofSyntax(v_before_469_);
v___x_480_ = l_Lean_indentD(v___x_479_);
v___x_481_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_478_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
v_x_462_ = v___x_481_;
v_x_463_ = v_tail_465_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__2(lean_object* v_opts_488_, lean_object* v_opt_489_){
_start:
{
lean_object* v_name_490_; lean_object* v_defValue_491_; lean_object* v_map_492_; lean_object* v___x_493_; 
v_name_490_ = lean_ctor_get(v_opt_489_, 0);
v_defValue_491_ = lean_ctor_get(v_opt_489_, 1);
v_map_492_ = lean_ctor_get(v_opts_488_, 0);
v___x_493_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_492_, v_name_490_);
if (lean_obj_tag(v___x_493_) == 0)
{
uint8_t v___x_494_; 
v___x_494_ = lean_unbox(v_defValue_491_);
return v___x_494_;
}
else
{
lean_object* v_val_495_; 
v_val_495_ = lean_ctor_get(v___x_493_, 0);
lean_inc(v_val_495_);
lean_dec_ref_known(v___x_493_, 1);
if (lean_obj_tag(v_val_495_) == 1)
{
uint8_t v_v_496_; 
v_v_496_ = lean_ctor_get_uint8(v_val_495_, 0);
lean_dec_ref_known(v_val_495_, 0);
return v_v_496_;
}
else
{
uint8_t v___x_497_; 
lean_dec(v_val_495_);
v___x_497_ = lean_unbox(v_defValue_491_);
return v___x_497_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_opts_498_, lean_object* v_opt_499_){
_start:
{
uint8_t v_res_500_; lean_object* v_r_501_; 
v_res_500_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__2(v_opts_498_, v_opt_499_);
lean_dec_ref(v_opt_499_);
lean_dec_ref(v_opts_498_);
v_r_501_ = lean_box(v_res_500_);
return v_r_501_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_505_; lean_object* v___x_506_; 
v___x_505_ = ((lean_object*)(lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__1));
v___x_506_ = l_Lean_MessageData_ofFormat(v___x_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg(lean_object* v_msgData_507_, lean_object* v_macroStack_508_, lean_object* v___y_509_){
_start:
{
lean_object* v_options_511_; lean_object* v___x_512_; uint8_t v___x_513_; 
v_options_511_ = lean_ctor_get(v___y_509_, 2);
v___x_512_ = l_Lean_Elab_pp_macroStack;
v___x_513_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__2(v_options_511_, v___x_512_);
if (v___x_513_ == 0)
{
lean_object* v___x_514_; 
lean_dec(v_macroStack_508_);
v___x_514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_514_, 0, v_msgData_507_);
return v___x_514_;
}
else
{
if (lean_obj_tag(v_macroStack_508_) == 0)
{
lean_object* v___x_515_; 
v___x_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_515_, 0, v_msgData_507_);
return v___x_515_;
}
else
{
lean_object* v_head_516_; lean_object* v_after_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_532_; 
v_head_516_ = lean_ctor_get(v_macroStack_508_, 0);
lean_inc(v_head_516_);
v_after_517_ = lean_ctor_get(v_head_516_, 1);
v_isSharedCheck_532_ = !lean_is_exclusive(v_head_516_);
if (v_isSharedCheck_532_ == 0)
{
lean_object* v_unused_533_; 
v_unused_533_ = lean_ctor_get(v_head_516_, 0);
lean_dec(v_unused_533_);
v___x_519_ = v_head_516_;
v_isShared_520_ = v_isSharedCheck_532_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_after_517_);
lean_dec(v_head_516_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_532_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_521_; lean_object* v___x_523_; 
v___x_521_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_520_ == 0)
{
lean_ctor_set_tag(v___x_519_, 7);
lean_ctor_set(v___x_519_, 1, v___x_521_);
lean_ctor_set(v___x_519_, 0, v_msgData_507_);
v___x_523_ = v___x_519_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_msgData_507_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v___x_521_);
v___x_523_ = v_reuseFailAlloc_531_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v_msgData_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_524_ = lean_obj_once(&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__2, &lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__2_once, _init_lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___closed__2);
v___x_525_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_523_);
lean_ctor_set(v___x_525_, 1, v___x_524_);
v___x_526_ = l_Lean_MessageData_ofSyntax(v_after_517_);
v___x_527_ = l_Lean_indentD(v___x_526_);
v_msgData_528_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_528_, 0, v___x_525_);
lean_ctor_set(v_msgData_528_, 1, v___x_527_);
v___x_529_ = lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1_spec__3(v_msgData_528_, v_macroStack_508_);
v___x_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_530_, 0, v___x_529_);
return v___x_530_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_msgData_534_, lean_object* v_macroStack_535_, lean_object* v___y_536_, lean_object* v___y_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg(v_msgData_534_, v_macroStack_535_, v___y_536_);
lean_dec_ref(v___y_536_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg(lean_object* v_msg_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
lean_object* v_ref_547_; lean_object* v___x_548_; lean_object* v_a_549_; lean_object* v_macroStack_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_561_; 
v_ref_547_ = lean_ctor_get(v___y_544_, 5);
v___x_548_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3(v_msg_539_, v___y_542_, v___y_543_, v___y_544_, v___y_545_);
v_a_549_ = lean_ctor_get(v___x_548_, 0);
lean_inc(v_a_549_);
lean_dec_ref(v___x_548_);
v_macroStack_550_ = lean_ctor_get(v___y_540_, 1);
v___x_551_ = l_Lean_Elab_getBetterRef(v_ref_547_, v_macroStack_550_);
lean_inc(v_macroStack_550_);
v___x_552_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg(v_a_549_, v_macroStack_550_, v___y_544_);
v_a_553_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_561_ == 0)
{
v___x_555_ = v___x_552_;
v_isShared_556_ = v_isSharedCheck_561_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_552_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_561_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; lean_object* v___x_559_; 
v___x_557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_557_, 0, v___x_551_);
lean_ctor_set(v___x_557_, 1, v_a_553_);
if (v_isShared_556_ == 0)
{
lean_ctor_set_tag(v___x_555_, 1);
lean_ctor_set(v___x_555_, 0, v___x_557_);
v___x_559_ = v___x_555_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v___x_557_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_msg_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg(v_msg_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
lean_dec(v___y_564_);
lean_dec_ref(v___y_563_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0(lean_object* v_constName_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_){
_start:
{
lean_object* v___x_579_; lean_object* v_env_580_; lean_object* v___x_581_; 
v___x_579_ = lean_st_ref_get(v___y_577_);
v_env_580_ = lean_ctor_get(v___x_579_, 0);
lean_inc_ref(v_env_580_);
lean_dec(v___x_579_);
lean_inc(v_constName_571_);
v___x_581_ = l_Lean_isInductiveCore_x3f(v_env_580_, v_constName_571_);
if (lean_obj_tag(v___x_581_) == 0)
{
lean_object* v___x_582_; uint8_t v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_582_ = lean_obj_once(&lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1, &lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1_once, _init_lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__1);
v___x_583_ = 0;
v___x_584_ = l_Lean_MessageData_ofConstName(v_constName_571_, v___x_583_);
v___x_585_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_585_, 0, v___x_582_);
lean_ctor_set(v___x_585_, 1, v___x_584_);
v___x_586_ = lean_obj_once(&lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3, &lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3_once, _init_lp_aesop_Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0___closed__3);
v___x_587_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_587_, 0, v___x_585_);
lean_ctor_set(v___x_587_, 1, v___x_586_);
v___x_588_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg(v___x_587_, v___y_572_, v___y_573_, v___y_574_, v___y_575_, v___y_576_, v___y_577_);
return v___x_588_;
}
else
{
lean_object* v_val_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_596_; 
lean_dec(v_constName_571_);
v_val_589_ = lean_ctor_get(v___x_581_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_581_);
if (v_isSharedCheck_596_ == 0)
{
v___x_591_ = v___x_581_;
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_val_589_);
lean_dec(v___x_581_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_594_; 
if (v_isShared_592_ == 0)
{
lean_ctor_set_tag(v___x_591_, 0);
v___x_594_ = v___x_591_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v_val_589_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0___boxed(lean_object* v_constName_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_aesop_Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0(v_constName_597_, v___y_598_, v___y_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
lean_dec(v___y_603_);
lean_dec_ref(v___y_602_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
lean_dec(v___y_599_);
lean_dec_ref(v___y_598_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent_x3f(lean_object* v_stx_606_, uint8_t v_md_607_, lean_object* v_a_608_, lean_object* v_a_609_, lean_object* v_a_610_, lean_object* v_a_611_, lean_object* v_a_612_, lean_object* v_a_613_){
_start:
{
lean_object* v___x_615_; 
v___x_615_ = lp_aesop_Aesop_elabGlobalRuleIdent_x3f(v_stx_606_, v_a_608_, v_a_609_, v_a_610_, v_a_611_, v_a_612_, v_a_613_);
if (lean_obj_tag(v___x_615_) == 0)
{
lean_object* v_a_616_; lean_object* v___x_618_; uint8_t v_isShared_619_; uint8_t v_isSharedCheck_689_; 
v_a_616_ = lean_ctor_get(v___x_615_, 0);
v_isSharedCheck_689_ = !lean_is_exclusive(v___x_615_);
if (v_isSharedCheck_689_ == 0)
{
v___x_618_ = v___x_615_;
v_isShared_619_ = v_isSharedCheck_689_;
goto v_resetjp_617_;
}
else
{
lean_inc(v_a_616_);
lean_dec(v___x_615_);
v___x_618_ = lean_box(0);
v_isShared_619_ = v_isSharedCheck_689_;
goto v_resetjp_617_;
}
v_resetjp_617_:
{
if (lean_obj_tag(v_a_616_) == 1)
{
lean_object* v_val_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_684_; 
v_val_620_ = lean_ctor_get(v_a_616_, 0);
v_isSharedCheck_684_ = !lean_is_exclusive(v_a_616_);
if (v_isSharedCheck_684_ == 0)
{
v___x_622_ = v_a_616_;
v_isShared_623_ = v_isSharedCheck_684_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_val_620_);
lean_dec(v_a_616_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_684_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v_val_625_; lean_object* v___x_633_; 
lean_inc(v_val_620_);
v___x_633_ = lp_aesop_Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0(v_val_620_, v_a_608_, v_a_609_, v_a_610_, v_a_611_, v_a_612_, v_a_613_);
if (lean_obj_tag(v___x_633_) == 0)
{
lean_object* v_a_634_; 
v_a_634_ = lean_ctor_get(v___x_633_, 0);
lean_inc(v_a_634_);
lean_dec_ref_known(v___x_633_, 1);
v_val_625_ = v_a_634_;
goto v___jp_624_;
}
else
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_683_; 
v_a_635_ = lean_ctor_get(v___x_633_, 0);
v_isSharedCheck_683_ = !lean_is_exclusive(v___x_633_);
if (v_isSharedCheck_683_ == 0)
{
v___x_637_ = v___x_633_;
v_isShared_638_ = v_isSharedCheck_683_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_633_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_683_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
uint8_t v___y_640_; uint8_t v___y_674_; uint8_t v___x_681_; 
v___x_681_ = l_Lean_Exception_isInterrupt(v_a_635_);
if (v___x_681_ == 0)
{
uint8_t v___x_682_; 
lean_inc(v_a_635_);
v___x_682_ = l_Lean_Exception_isRuntime(v_a_635_);
v___y_674_ = v___x_682_;
goto v___jp_673_;
}
else
{
v___y_674_ = v___x_681_;
goto v___jp_673_;
}
v___jp_639_:
{
lean_object* v_keyedConfig_641_; uint8_t v_trackZetaDelta_642_; lean_object* v_zetaDeltaSet_643_; lean_object* v_lctx_644_; lean_object* v_localInstances_645_; lean_object* v_defEqCtx_x3f_646_; lean_object* v_synthPendingDepth_647_; lean_object* v_customCanUnfoldPredicate_x3f_648_; uint8_t v_univApprox_649_; uint8_t v_inTypeClassResolution_650_; uint8_t v_cacheInferType_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v_keyedConfig_641_ = lean_ctor_get(v_a_610_, 0);
v_trackZetaDelta_642_ = lean_ctor_get_uint8(v_a_610_, sizeof(void*)*7);
v_zetaDeltaSet_643_ = lean_ctor_get(v_a_610_, 1);
v_lctx_644_ = lean_ctor_get(v_a_610_, 2);
v_localInstances_645_ = lean_ctor_get(v_a_610_, 3);
v_defEqCtx_x3f_646_ = lean_ctor_get(v_a_610_, 4);
v_synthPendingDepth_647_ = lean_ctor_get(v_a_610_, 5);
v_customCanUnfoldPredicate_x3f_648_ = lean_ctor_get(v_a_610_, 6);
v_univApprox_649_ = lean_ctor_get_uint8(v_a_610_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_650_ = lean_ctor_get_uint8(v_a_610_, sizeof(void*)*7 + 2);
v_cacheInferType_651_ = lean_ctor_get_uint8(v_a_610_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_641_);
v___x_652_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___y_640_, v_keyedConfig_641_);
lean_inc(v_customCanUnfoldPredicate_x3f_648_);
lean_inc(v_synthPendingDepth_647_);
lean_inc(v_defEqCtx_x3f_646_);
lean_inc_ref(v_localInstances_645_);
lean_inc_ref(v_lctx_644_);
lean_inc(v_zetaDeltaSet_643_);
v___x_653_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_653_, 0, v___x_652_);
lean_ctor_set(v___x_653_, 1, v_zetaDeltaSet_643_);
lean_ctor_set(v___x_653_, 2, v_lctx_644_);
lean_ctor_set(v___x_653_, 3, v_localInstances_645_);
lean_ctor_set(v___x_653_, 4, v_defEqCtx_x3f_646_);
lean_ctor_set(v___x_653_, 5, v_synthPendingDepth_647_);
lean_ctor_set(v___x_653_, 6, v_customCanUnfoldPredicate_x3f_648_);
lean_ctor_set_uint8(v___x_653_, sizeof(void*)*7, v_trackZetaDelta_642_);
lean_ctor_set_uint8(v___x_653_, sizeof(void*)*7 + 1, v_univApprox_649_);
lean_ctor_set_uint8(v___x_653_, sizeof(void*)*7 + 2, v_inTypeClassResolution_650_);
lean_ctor_set_uint8(v___x_653_, sizeof(void*)*7 + 3, v_cacheInferType_651_);
lean_inc(v_val_620_);
v___x_654_ = lp_aesop_Aesop_matchInductiveTypeSynonym_x3f(v_val_620_, v___x_653_, v_a_611_, v_a_612_, v_a_613_);
lean_dec_ref_known(v___x_653_, 7);
if (lean_obj_tag(v___x_654_) == 0)
{
lean_object* v_a_655_; lean_object* v___x_657_; uint8_t v_isShared_658_; uint8_t v_isSharedCheck_664_; 
v_a_655_ = lean_ctor_get(v___x_654_, 0);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_654_);
if (v_isSharedCheck_664_ == 0)
{
v___x_657_ = v___x_654_;
v_isShared_658_ = v_isSharedCheck_664_;
goto v_resetjp_656_;
}
else
{
lean_inc(v_a_655_);
lean_dec(v___x_654_);
v___x_657_ = lean_box(0);
v_isShared_658_ = v_isSharedCheck_664_;
goto v_resetjp_656_;
}
v_resetjp_656_:
{
if (lean_obj_tag(v_a_655_) == 0)
{
lean_object* v___x_659_; lean_object* v___x_661_; 
lean_del_object(v___x_622_);
lean_dec(v_val_620_);
lean_del_object(v___x_618_);
v___x_659_ = lean_box(0);
if (v_isShared_658_ == 0)
{
lean_ctor_set(v___x_657_, 0, v___x_659_);
v___x_661_ = v___x_657_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v___x_659_);
v___x_661_ = v_reuseFailAlloc_662_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
return v___x_661_;
}
}
else
{
lean_object* v_val_663_; 
lean_del_object(v___x_657_);
v_val_663_ = lean_ctor_get(v_a_655_, 0);
lean_inc(v_val_663_);
lean_dec_ref_known(v_a_655_, 1);
v_val_625_ = v_val_663_;
goto v___jp_624_;
}
}
}
else
{
lean_object* v_a_665_; lean_object* v___x_667_; uint8_t v_isShared_668_; uint8_t v_isSharedCheck_672_; 
lean_del_object(v___x_622_);
lean_dec(v_val_620_);
lean_del_object(v___x_618_);
v_a_665_ = lean_ctor_get(v___x_654_, 0);
v_isSharedCheck_672_ = !lean_is_exclusive(v___x_654_);
if (v_isSharedCheck_672_ == 0)
{
v___x_667_ = v___x_654_;
v_isShared_668_ = v_isSharedCheck_672_;
goto v_resetjp_666_;
}
else
{
lean_inc(v_a_665_);
lean_dec(v___x_654_);
v___x_667_ = lean_box(0);
v_isShared_668_ = v_isSharedCheck_672_;
goto v_resetjp_666_;
}
v_resetjp_666_:
{
lean_object* v___x_670_; 
if (v_isShared_668_ == 0)
{
v___x_670_ = v___x_667_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_671_; 
v_reuseFailAlloc_671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_671_, 0, v_a_665_);
v___x_670_ = v_reuseFailAlloc_671_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
return v___x_670_;
}
}
}
}
v___jp_673_:
{
if (v___y_674_ == 0)
{
lean_object* v___x_675_; uint8_t v_transparency_676_; uint8_t v___x_677_; 
lean_del_object(v___x_637_);
lean_dec(v_a_635_);
v___x_675_ = l_Lean_Meta_Context_config(v_a_610_);
v_transparency_676_ = lean_ctor_get_uint8(v___x_675_, 9);
lean_dec_ref(v___x_675_);
v___x_677_ = l_Lean_Meta_TransparencyMode_lt(v_transparency_676_, v_md_607_);
if (v___x_677_ == 0)
{
v___y_640_ = v_transparency_676_;
goto v___jp_639_;
}
else
{
v___y_640_ = v_md_607_;
goto v___jp_639_;
}
}
else
{
lean_object* v___x_679_; 
lean_del_object(v___x_622_);
lean_dec(v_val_620_);
lean_del_object(v___x_618_);
if (v_isShared_638_ == 0)
{
v___x_679_ = v___x_637_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_680_; 
v_reuseFailAlloc_680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_680_, 0, v_a_635_);
v___x_679_ = v_reuseFailAlloc_680_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
return v___x_679_;
}
}
}
}
}
v___jp_624_:
{
lean_object* v___x_626_; lean_object* v___x_628_; 
v___x_626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_626_, 0, v_val_620_);
lean_ctor_set(v___x_626_, 1, v_val_625_);
if (v_isShared_623_ == 0)
{
lean_ctor_set(v___x_622_, 0, v___x_626_);
v___x_628_ = v___x_622_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_632_; 
v_reuseFailAlloc_632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_632_, 0, v___x_626_);
v___x_628_ = v_reuseFailAlloc_632_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
lean_object* v___x_630_; 
if (v_isShared_619_ == 0)
{
lean_ctor_set(v___x_618_, 0, v___x_628_);
v___x_630_ = v___x_618_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v___x_628_);
v___x_630_ = v_reuseFailAlloc_631_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
return v___x_630_;
}
}
}
}
}
else
{
lean_object* v___x_685_; lean_object* v___x_687_; 
lean_dec(v_a_616_);
v___x_685_ = lean_box(0);
if (v_isShared_619_ == 0)
{
lean_ctor_set(v___x_618_, 0, v___x_685_);
v___x_687_ = v___x_618_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v___x_685_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
}
}
else
{
lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
v_a_690_ = lean_ctor_get(v___x_615_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_615_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_615_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_615_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_695_; 
if (v_isShared_693_ == 0)
{
v___x_695_ = v___x_692_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_a_690_);
v___x_695_ = v_reuseFailAlloc_696_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
return v___x_695_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabInductiveRuleIdent_x3f___boxed(lean_object* v_stx_698_, lean_object* v_md_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_, lean_object* v_a_703_, lean_object* v_a_704_, lean_object* v_a_705_, lean_object* v_a_706_){
_start:
{
uint8_t v_md_boxed_707_; lean_object* v_res_708_; 
v_md_boxed_707_ = lean_unbox(v_md_699_);
v_res_708_ = lp_aesop_Aesop_elabInductiveRuleIdent_x3f(v_stx_698_, v_md_boxed_707_, v_a_700_, v_a_701_, v_a_702_, v_a_703_, v_a_704_, v_a_705_);
lean_dec(v_a_705_);
lean_dec_ref(v_a_704_);
lean_dec(v_a_703_);
lean_dec_ref(v_a_702_);
lean_dec(v_a_701_);
lean_dec_ref(v_a_700_);
return v_res_708_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0(lean_object* v_00_u03b1_709_, lean_object* v_msg_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_){
_start:
{
lean_object* v___x_718_; 
v___x_718_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___redArg(v_msg_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_, v___y_715_, v___y_716_);
return v___x_718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0___boxed(lean_object* v_00_u03b1_719_, lean_object* v_msg_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0(v_00_u03b1_719_, v_msg_720_, v___y_721_, v___y_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec(v___y_724_);
lean_dec_ref(v___y_723_);
lean_dec(v___y_722_);
lean_dec_ref(v___y_721_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1(lean_object* v_msgData_729_, lean_object* v_macroStack_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_){
_start:
{
lean_object* v___x_738_; 
v___x_738_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___redArg(v_msgData_729_, v_macroStack_730_, v___y_735_);
return v___x_738_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1___boxed(lean_object* v_msgData_739_, lean_object* v_macroStack_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
lean_object* v_res_748_; 
v_res_748_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_elabInductiveRuleIdent_x3f_spec__0_spec__0_spec__1(v_msgData_739_, v_macroStack_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
lean_dec(v___y_746_);
lean_dec_ref(v___y_745_);
lean_dec(v___y_744_);
lean_dec_ref(v___y_743_);
lean_dec(v___y_742_);
lean_dec_ref(v___y_741_);
return v_res_748_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___redArg(lean_object* v_goal_752_, lean_object* v_x_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v_a_757_, lean_object* v_a_758_, lean_object* v_a_759_){
_start:
{
lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; 
v___x_761_ = lean_box(0);
v___x_762_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_762_, 0, v_goal_752_);
lean_ctor_set(v___x_762_, 1, v___x_761_);
v___x_763_ = lean_st_mk_ref(v___x_762_);
v___x_764_ = ((lean_object*)(lp_aesop_Aesop_runTacticMAsTermElabM___redArg___closed__0));
lean_inc(v_a_759_);
lean_inc_ref(v_a_758_);
lean_inc(v_a_757_);
lean_inc_ref(v_a_756_);
lean_inc(v_a_755_);
lean_inc_ref(v_a_754_);
lean_inc(v___x_763_);
v___x_765_ = lean_apply_9(v_x_753_, v___x_764_, v___x_763_, v_a_754_, v_a_755_, v_a_756_, v_a_757_, v_a_758_, v_a_759_, lean_box(0));
if (lean_obj_tag(v___x_765_) == 0)
{
lean_object* v_a_766_; lean_object* v___x_768_; uint8_t v_isShared_769_; uint8_t v_isSharedCheck_774_; 
v_a_766_ = lean_ctor_get(v___x_765_, 0);
v_isSharedCheck_774_ = !lean_is_exclusive(v___x_765_);
if (v_isSharedCheck_774_ == 0)
{
v___x_768_ = v___x_765_;
v_isShared_769_ = v_isSharedCheck_774_;
goto v_resetjp_767_;
}
else
{
lean_inc(v_a_766_);
lean_dec(v___x_765_);
v___x_768_ = lean_box(0);
v_isShared_769_ = v_isSharedCheck_774_;
goto v_resetjp_767_;
}
v_resetjp_767_:
{
lean_object* v___x_770_; lean_object* v___x_772_; 
v___x_770_ = lean_st_ref_get(v___x_763_);
lean_dec(v___x_763_);
lean_dec(v___x_770_);
if (v_isShared_769_ == 0)
{
v___x_772_ = v___x_768_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v_a_766_);
v___x_772_ = v_reuseFailAlloc_773_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
return v___x_772_;
}
}
}
else
{
lean_dec(v___x_763_);
return v___x_765_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___redArg___boxed(lean_object* v_goal_775_, lean_object* v_x_776_, lean_object* v_a_777_, lean_object* v_a_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v_a_782_, lean_object* v_a_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_aesop_Aesop_runTacticMAsTermElabM___redArg(v_goal_775_, v_x_776_, v_a_777_, v_a_778_, v_a_779_, v_a_780_, v_a_781_, v_a_782_);
lean_dec(v_a_782_);
lean_dec_ref(v_a_781_);
lean_dec(v_a_780_);
lean_dec_ref(v_a_779_);
lean_dec(v_a_778_);
lean_dec_ref(v_a_777_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM(lean_object* v_00_u03b1_785_, lean_object* v_goal_786_, lean_object* v_x_787_, lean_object* v_a_788_, lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_){
_start:
{
lean_object* v___x_795_; 
v___x_795_ = lp_aesop_Aesop_runTacticMAsTermElabM___redArg(v_goal_786_, v_x_787_, v_a_788_, v_a_789_, v_a_790_, v_a_791_, v_a_792_, v_a_793_);
return v___x_795_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsTermElabM___boxed(lean_object* v_00_u03b1_796_, lean_object* v_goal_797_, lean_object* v_x_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_aesop_Aesop_runTacticMAsTermElabM(v_00_u03b1_796_, v_goal_797_, v_x_798_, v_a_799_, v_a_800_, v_a_801_, v_a_802_, v_a_803_, v_a_804_);
lean_dec(v_a_804_);
lean_dec_ref(v_a_803_);
lean_dec(v_a_802_);
lean_dec_ref(v_a_801_);
lean_dec(v_a_800_);
lean_dec_ref(v_a_799_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM___redArg(lean_object* v_x_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_, lean_object* v_a_812_, lean_object* v_a_813_, lean_object* v_a_814_){
_start:
{
lean_object* v_goal_816_; lean_object* v___x_817_; 
v_goal_816_ = lean_ctor_get(v_a_808_, 0);
lean_inc(v_goal_816_);
v___x_817_ = lp_aesop_Aesop_runTacticMAsTermElabM___redArg(v_goal_816_, v_x_807_, v_a_809_, v_a_810_, v_a_811_, v_a_812_, v_a_813_, v_a_814_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM___redArg___boxed(lean_object* v_x_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_, lean_object* v_a_822_, lean_object* v_a_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_){
_start:
{
lean_object* v_res_827_; 
v_res_827_ = lp_aesop_Aesop_runTacticMAsElabM___redArg(v_x_818_, v_a_819_, v_a_820_, v_a_821_, v_a_822_, v_a_823_, v_a_824_, v_a_825_);
lean_dec(v_a_825_);
lean_dec_ref(v_a_824_);
lean_dec(v_a_823_);
lean_dec_ref(v_a_822_);
lean_dec(v_a_821_);
lean_dec_ref(v_a_820_);
lean_dec_ref(v_a_819_);
return v_res_827_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM(lean_object* v_00_u03b1_828_, lean_object* v_x_829_, lean_object* v_a_830_, lean_object* v_a_831_, lean_object* v_a_832_, lean_object* v_a_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_){
_start:
{
lean_object* v___x_838_; 
v___x_838_ = lp_aesop_Aesop_runTacticMAsElabM___redArg(v_x_829_, v_a_830_, v_a_831_, v_a_832_, v_a_833_, v_a_834_, v_a_835_, v_a_836_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runTacticMAsElabM___boxed(lean_object* v_00_u03b1_839_, lean_object* v_x_840_, lean_object* v_a_841_, lean_object* v_a_842_, lean_object* v_a_843_, lean_object* v_a_844_, lean_object* v_a_845_, lean_object* v_a_846_, lean_object* v_a_847_, lean_object* v_a_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_aesop_Aesop_runTacticMAsElabM(v_00_u03b1_839_, v_x_840_, v_a_841_, v_a_842_, v_a_843_, v_a_844_, v_a_845_, v_a_846_, v_a_847_);
lean_dec(v_a_847_);
lean_dec_ref(v_a_846_);
lean_dec(v_a_845_);
lean_dec_ref(v_a_844_);
lean_dec(v_a_843_);
lean_dec_ref(v_a_842_);
lean_dec_ref(v_a_841_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___redArg(lean_object* v_a_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_){
_start:
{
lean_object* v___x_858_; 
v___x_858_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_850_, v___y_851_, v___y_852_, v___y_853_, v___y_854_, v___y_855_, v___y_856_);
return v___x_858_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___redArg___boxed(lean_object* v_a_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
lean_object* v_res_867_; 
v_res_867_ = lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___redArg(v_a_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
lean_dec(v___y_861_);
lean_dec_ref(v___y_860_);
return v_res_867_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0(lean_object* v_00_u03b1_868_, lean_object* v_a_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_){
_start:
{
lean_object* v___x_877_; 
v___x_877_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, v___y_875_);
return v___x_877_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___boxed(lean_object* v_00_u03b1_878_, lean_object* v_a_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_){
_start:
{
lean_object* v_res_887_; 
v_res_887_ = lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0(v_00_u03b1_878_, v_a_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, v___y_885_);
lean_dec(v___y_885_);
lean_dec_ref(v___y_884_);
lean_dec(v___y_883_);
lean_dec_ref(v___y_882_);
lean_dec(v___y_881_);
lean_dec_ref(v___y_880_);
return v_res_887_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration___redArg(lean_object* v_x_888_, lean_object* v_a_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_, lean_object* v_a_894_){
_start:
{
lean_object* v___x_896_; lean_object* v___x_897_; uint8_t v___x_898_; lean_object* v___x_899_; 
v___x_896_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_withoutAutoBoundImplicit___boxed), 9, 2);
lean_closure_set(v___x_896_, 0, lean_box(0));
lean_closure_set(v___x_896_, 1, v_x_888_);
v___x_897_ = lean_alloc_closure((void*)(lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_withFullElaboration_spec__0___boxed), 9, 2);
lean_closure_set(v___x_897_, 0, lean_box(0));
lean_closure_set(v___x_897_, 1, v___x_896_);
v___x_898_ = 1;
v___x_899_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_897_, v___x_898_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_);
return v___x_899_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration___redArg___boxed(lean_object* v_x_900_, lean_object* v_a_901_, lean_object* v_a_902_, lean_object* v_a_903_, lean_object* v_a_904_, lean_object* v_a_905_, lean_object* v_a_906_, lean_object* v_a_907_){
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_aesop_Aesop_withFullElaboration___redArg(v_x_900_, v_a_901_, v_a_902_, v_a_903_, v_a_904_, v_a_905_, v_a_906_);
lean_dec(v_a_906_);
lean_dec_ref(v_a_905_);
lean_dec(v_a_904_);
lean_dec_ref(v_a_903_);
lean_dec(v_a_902_);
lean_dec_ref(v_a_901_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration(lean_object* v_00_u03b1_909_, lean_object* v_x_910_, lean_object* v_a_911_, lean_object* v_a_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_, lean_object* v_a_916_){
_start:
{
lean_object* v___x_918_; 
v___x_918_ = lp_aesop_Aesop_withFullElaboration___redArg(v_x_910_, v_a_911_, v_a_912_, v_a_913_, v_a_914_, v_a_915_, v_a_916_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withFullElaboration___boxed(lean_object* v_00_u03b1_919_, lean_object* v_x_920_, lean_object* v_a_921_, lean_object* v_a_922_, lean_object* v_a_923_, lean_object* v_a_924_, lean_object* v_a_925_, lean_object* v_a_926_, lean_object* v_a_927_){
_start:
{
lean_object* v_res_928_; 
v_res_928_ = lp_aesop_Aesop_withFullElaboration(v_00_u03b1_919_, v_x_920_, v_a_921_, v_a_922_, v_a_923_, v_a_924_, v_a_925_, v_a_926_);
lean_dec(v_a_926_);
lean_dec_ref(v_a_925_);
lean_dec(v_a_924_);
lean_dec_ref(v_a_923_);
lean_dec(v_a_922_);
lean_dec_ref(v_a_921_);
return v_res_928_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeCore(lean_object* v_goal_929_, lean_object* v_stx_930_, lean_object* v_a_931_, lean_object* v_a_932_, lean_object* v_a_933_, lean_object* v_a_934_, lean_object* v_a_935_, lean_object* v_a_936_){
_start:
{
uint8_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; 
v___x_938_ = 0;
v___x_939_ = lean_box(v___x_938_);
v___x_940_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_elabTermForApply___boxed), 11, 2);
lean_closure_set(v___x_940_, 0, v_stx_930_);
lean_closure_set(v___x_940_, 1, v___x_939_);
v___x_941_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runTacticMAsTermElabM___boxed), 10, 3);
lean_closure_set(v___x_941_, 0, lean_box(0));
lean_closure_set(v___x_941_, 1, v_goal_929_);
lean_closure_set(v___x_941_, 2, v___x_940_);
v___x_942_ = lp_aesop_Aesop_withFullElaboration___redArg(v___x_941_, v_a_931_, v_a_932_, v_a_933_, v_a_934_, v_a_935_, v_a_936_);
return v___x_942_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeCore___boxed(lean_object* v_goal_943_, lean_object* v_stx_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_aesop_Aesop_elabRuleTermForApplyLikeCore(v_goal_943_, v_stx_944_, v_a_945_, v_a_946_, v_a_947_, v_a_948_, v_a_949_, v_a_950_);
lean_dec(v_a_950_);
lean_dec_ref(v_a_949_);
lean_dec(v_a_948_);
lean_dec_ref(v_a_947_);
lean_dec(v_a_946_);
lean_dec_ref(v_a_945_);
return v_res_952_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___lam__0(lean_object* v_x_953_){
_start:
{
uint8_t v___x_954_; 
v___x_954_ = 0;
return v___x_954_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___lam__0___boxed(lean_object* v_x_955_){
_start:
{
uint8_t v_res_956_; lean_object* v_r_957_; 
v_res_956_ = lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___lam__0(v_x_955_);
lean_dec(v_x_955_);
v_r_957_ = lean_box(v_res_956_);
return v_r_957_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(lean_object* v_goal_972_, lean_object* v_stx_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; 
v___x_979_ = lean_alloc_closure((void*)(lp_aesop_Aesop_elabRuleTermForApplyLikeCore___boxed), 9, 2);
lean_closure_set(v___x_979_, 0, v_goal_972_);
lean_closure_set(v___x_979_, 1, v_stx_973_);
v___x_980_ = ((lean_object*)(lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__2));
v___x_981_ = ((lean_object*)(lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__3));
v___x_982_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_979_, v___x_980_, v___x_981_, v_a_974_, v_a_975_, v_a_976_, v_a_977_);
if (lean_obj_tag(v___x_982_) == 0)
{
lean_object* v_a_983_; lean_object* v___x_985_; uint8_t v_isShared_986_; uint8_t v_isSharedCheck_991_; 
v_a_983_ = lean_ctor_get(v___x_982_, 0);
v_isSharedCheck_991_ = !lean_is_exclusive(v___x_982_);
if (v_isSharedCheck_991_ == 0)
{
v___x_985_ = v___x_982_;
v_isShared_986_ = v_isSharedCheck_991_;
goto v_resetjp_984_;
}
else
{
lean_inc(v_a_983_);
lean_dec(v___x_982_);
v___x_985_ = lean_box(0);
v_isShared_986_ = v_isSharedCheck_991_;
goto v_resetjp_984_;
}
v_resetjp_984_:
{
lean_object* v_fst_987_; lean_object* v___x_989_; 
v_fst_987_ = lean_ctor_get(v_a_983_, 0);
lean_inc(v_fst_987_);
lean_dec(v_a_983_);
if (v_isShared_986_ == 0)
{
lean_ctor_set(v___x_985_, 0, v_fst_987_);
v___x_989_ = v___x_985_;
goto v_reusejp_988_;
}
else
{
lean_object* v_reuseFailAlloc_990_; 
v_reuseFailAlloc_990_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_990_, 0, v_fst_987_);
v___x_989_ = v_reuseFailAlloc_990_;
goto v_reusejp_988_;
}
v_reusejp_988_:
{
return v___x_989_;
}
}
}
else
{
lean_object* v_a_992_; lean_object* v___x_994_; uint8_t v_isShared_995_; uint8_t v_isSharedCheck_999_; 
v_a_992_ = lean_ctor_get(v___x_982_, 0);
v_isSharedCheck_999_ = !lean_is_exclusive(v___x_982_);
if (v_isSharedCheck_999_ == 0)
{
v___x_994_ = v___x_982_;
v_isShared_995_ = v_isSharedCheck_999_;
goto v_resetjp_993_;
}
else
{
lean_inc(v_a_992_);
lean_dec(v___x_982_);
v___x_994_ = lean_box(0);
v_isShared_995_ = v_isSharedCheck_999_;
goto v_resetjp_993_;
}
v_resetjp_993_:
{
lean_object* v___x_997_; 
if (v_isShared_995_ == 0)
{
v___x_997_ = v___x_994_;
goto v_reusejp_996_;
}
else
{
lean_object* v_reuseFailAlloc_998_; 
v_reuseFailAlloc_998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_998_, 0, v_a_992_);
v___x_997_ = v_reuseFailAlloc_998_;
goto v_reusejp_996_;
}
v_reusejp_996_:
{
return v___x_997_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___boxed(lean_object* v_goal_1000_, lean_object* v_stx_1001_, lean_object* v_a_1002_, lean_object* v_a_1003_, lean_object* v_a_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_){
_start:
{
lean_object* v_res_1007_; 
v_res_1007_ = lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(v_goal_1000_, v_stx_1001_, v_a_1002_, v_a_1003_, v_a_1004_, v_a_1005_);
lean_dec(v_a_1005_);
lean_dec_ref(v_a_1004_);
lean_dec(v_a_1003_);
lean_dec_ref(v_a_1002_);
return v_res_1007_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLike(lean_object* v_stx_1008_, lean_object* v_a_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_){
_start:
{
lean_object* v_goal_1017_; lean_object* v___x_1018_; 
v_goal_1017_ = lean_ctor_get(v_a_1009_, 0);
lean_inc(v_goal_1017_);
v___x_1018_ = lp_aesop_Aesop_elabRuleTermForApplyLikeCore(v_goal_1017_, v_stx_1008_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_, v_a_1014_, v_a_1015_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForApplyLike___boxed(lean_object* v_stx_1019_, lean_object* v_a_1020_, lean_object* v_a_1021_, lean_object* v_a_1022_, lean_object* v_a_1023_, lean_object* v_a_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_){
_start:
{
lean_object* v_res_1028_; 
v_res_1028_ = lp_aesop_Aesop_elabRuleTermForApplyLike(v_stx_1019_, v_a_1020_, v_a_1021_, v_a_1022_, v_a_1023_, v_a_1024_, v_a_1025_, v_a_1026_);
lean_dec(v_a_1026_);
lean_dec_ref(v_a_1025_);
lean_dec(v_a_1024_);
lean_dec_ref(v_a_1023_);
lean_dec(v_a_1022_);
lean_dec_ref(v_a_1021_);
lean_dec_ref(v_a_1020_);
return v_res_1028_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_elabSimpTheorems_spec__0(lean_object* v_as_1029_, size_t v_i_1030_, size_t v_stop_1031_){
_start:
{
uint8_t v___x_1032_; 
v___x_1032_ = lean_usize_dec_eq(v_i_1030_, v_stop_1031_);
if (v___x_1032_ == 0)
{
lean_object* v___x_1033_; lean_object* v_snd_1034_; 
v___x_1033_ = lean_array_uget_borrowed(v_as_1029_, v_i_1030_);
v_snd_1034_ = lean_ctor_get(v___x_1033_, 1);
if (lean_obj_tag(v_snd_1034_) == 6)
{
uint8_t v___x_1035_; 
v___x_1035_ = 1;
return v___x_1035_;
}
else
{
size_t v___x_1036_; size_t v___x_1037_; 
v___x_1036_ = ((size_t)1ULL);
v___x_1037_ = lean_usize_add(v_i_1030_, v___x_1036_);
v_i_1030_ = v___x_1037_;
goto _start;
}
}
else
{
uint8_t v___x_1039_; 
v___x_1039_ = 0;
return v___x_1039_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_elabSimpTheorems_spec__0___boxed(lean_object* v_as_1040_, lean_object* v_i_1041_, lean_object* v_stop_1042_){
_start:
{
size_t v_i_boxed_1043_; size_t v_stop_boxed_1044_; uint8_t v_res_1045_; lean_object* v_r_1046_; 
v_i_boxed_1043_ = lean_unbox_usize(v_i_1041_);
lean_dec(v_i_1041_);
v_stop_boxed_1044_ = lean_unbox_usize(v_stop_1042_);
lean_dec(v_stop_1042_);
v_res_1045_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_elabSimpTheorems_spec__0(v_as_1040_, v_i_boxed_1043_, v_stop_boxed_1044_);
lean_dec_ref(v_as_1040_);
v_r_1046_ = lean_box(v_res_1045_);
return v_r_1046_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg(lean_object* v_msg_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_){
_start:
{
lean_object* v_ref_1053_; lean_object* v___x_1054_; lean_object* v_a_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1063_; 
v_ref_1053_ = lean_ctor_get(v___y_1050_, 5);
v___x_1054_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoInduct___at___00Aesop_matchInductiveTypeSynonym_x3f_spec__0_spec__0_spec__3(v_msg_1047_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_);
v_a_1055_ = lean_ctor_get(v___x_1054_, 0);
v_isSharedCheck_1063_ = !lean_is_exclusive(v___x_1054_);
if (v_isSharedCheck_1063_ == 0)
{
v___x_1057_ = v___x_1054_;
v_isShared_1058_ = v_isSharedCheck_1063_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_a_1055_);
lean_dec(v___x_1054_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1063_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
lean_object* v___x_1059_; lean_object* v___x_1061_; 
lean_inc(v_ref_1053_);
v___x_1059_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1059_, 0, v_ref_1053_);
lean_ctor_set(v___x_1059_, 1, v_a_1055_);
if (v_isShared_1058_ == 0)
{
lean_ctor_set_tag(v___x_1057_, 1);
lean_ctor_set(v___x_1057_, 0, v___x_1059_);
v___x_1061_ = v___x_1057_;
goto v_reusejp_1060_;
}
else
{
lean_object* v_reuseFailAlloc_1062_; 
v_reuseFailAlloc_1062_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1062_, 0, v___x_1059_);
v___x_1061_ = v_reuseFailAlloc_1062_;
goto v_reusejp_1060_;
}
v_reusejp_1060_:
{
return v___x_1061_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg___boxed(lean_object* v_msg_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_){
_start:
{
lean_object* v_res_1070_; 
v_res_1070_ = lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg(v_msg_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
lean_dec(v___y_1068_);
lean_dec_ref(v___y_1067_);
lean_dec(v___y_1066_);
lean_dec_ref(v___y_1065_);
return v_res_1070_;
}
}
static lean_object* _init_lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1072_; lean_object* v___x_1073_; 
v___x_1072_ = ((lean_object*)(lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__0));
v___x_1073_ = l_Lean_stringToMessageData(v___x_1072_);
return v___x_1073_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems___lam__0(lean_object* v_stx_1074_, lean_object* v_ctx_1075_, lean_object* v_simprocs_1076_, uint8_t v___x_1077_, uint8_t v___y_1078_, uint8_t v___x_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v___x_1089_; 
v___x_1089_ = l_Lean_Elab_Tactic_elabSimpArgs(v_stx_1074_, v_ctx_1075_, v_simprocs_1076_, v___x_1077_, v___y_1078_, v___x_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_, v___y_1084_, v___y_1085_, v___y_1086_, v___y_1087_);
if (lean_obj_tag(v___x_1089_) == 0)
{
lean_object* v_a_1090_; lean_object* v___x_1092_; uint8_t v_isShared_1093_; uint8_t v_isSharedCheck_1118_; 
v_a_1090_ = lean_ctor_get(v___x_1089_, 0);
v_isSharedCheck_1118_ = !lean_is_exclusive(v___x_1089_);
if (v_isSharedCheck_1118_ == 0)
{
v___x_1092_ = v___x_1089_;
v_isShared_1093_ = v_isSharedCheck_1118_;
goto v_resetjp_1091_;
}
else
{
lean_inc(v_a_1090_);
lean_dec(v___x_1089_);
v___x_1092_ = lean_box(0);
v_isShared_1093_ = v_isSharedCheck_1118_;
goto v_resetjp_1091_;
}
v_resetjp_1091_:
{
lean_object* v_ctx_1094_; lean_object* v_simprocs_1095_; lean_object* v_simpArgs_1096_; lean_object* v___x_1102_; lean_object* v___x_1103_; uint8_t v___x_1104_; 
v_ctx_1094_ = lean_ctor_get(v_a_1090_, 0);
lean_inc_ref(v_ctx_1094_);
v_simprocs_1095_ = lean_ctor_get(v_a_1090_, 1);
lean_inc_ref(v_simprocs_1095_);
v_simpArgs_1096_ = lean_ctor_get(v_a_1090_, 2);
lean_inc_ref(v_simpArgs_1096_);
lean_dec(v_a_1090_);
v___x_1102_ = lean_unsigned_to_nat(0u);
v___x_1103_ = lean_array_get_size(v_simpArgs_1096_);
v___x_1104_ = lean_nat_dec_lt(v___x_1102_, v___x_1103_);
if (v___x_1104_ == 0)
{
lean_dec_ref(v_simpArgs_1096_);
goto v___jp_1097_;
}
else
{
if (v___x_1104_ == 0)
{
lean_dec_ref(v_simpArgs_1096_);
goto v___jp_1097_;
}
else
{
size_t v___x_1105_; size_t v___x_1106_; uint8_t v___x_1107_; 
v___x_1105_ = ((size_t)0ULL);
v___x_1106_ = lean_usize_of_nat(v___x_1103_);
v___x_1107_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_elabSimpTheorems_spec__0(v_simpArgs_1096_, v___x_1105_, v___x_1106_);
lean_dec_ref(v_simpArgs_1096_);
if (v___x_1107_ == 0)
{
goto v___jp_1097_;
}
else
{
lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
lean_dec_ref(v_simprocs_1095_);
lean_dec_ref(v_ctx_1094_);
lean_del_object(v___x_1092_);
v___x_1108_ = lean_obj_once(&lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__1, &lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__1_once, _init_lp_aesop_Aesop_elabSimpTheorems___lam__0___closed__1);
v___x_1109_ = lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg(v___x_1108_, v___y_1084_, v___y_1085_, v___y_1086_, v___y_1087_);
v_a_1110_ = lean_ctor_get(v___x_1109_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1109_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1109_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1109_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
}
}
v___jp_1097_:
{
lean_object* v___x_1098_; lean_object* v___x_1100_; 
v___x_1098_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1098_, 0, v_ctx_1094_);
lean_ctor_set(v___x_1098_, 1, v_simprocs_1095_);
if (v_isShared_1093_ == 0)
{
lean_ctor_set(v___x_1092_, 0, v___x_1098_);
v___x_1100_ = v___x_1092_;
goto v_reusejp_1099_;
}
else
{
lean_object* v_reuseFailAlloc_1101_; 
v_reuseFailAlloc_1101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1101_, 0, v___x_1098_);
v___x_1100_ = v_reuseFailAlloc_1101_;
goto v_reusejp_1099_;
}
v_reusejp_1099_:
{
return v___x_1100_;
}
}
}
}
else
{
lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1126_; 
v_a_1119_ = lean_ctor_get(v___x_1089_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1089_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1121_ = v___x_1089_;
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1089_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___x_1124_; 
if (v_isShared_1122_ == 0)
{
v___x_1124_ = v___x_1121_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v_a_1119_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems___lam__0___boxed(lean_object* v_stx_1127_, lean_object* v_ctx_1128_, lean_object* v_simprocs_1129_, lean_object* v___x_1130_, lean_object* v___y_1131_, lean_object* v___x_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
uint8_t v___x_1978__boxed_1142_; uint8_t v___y_1979__boxed_1143_; uint8_t v___x_1980__boxed_1144_; lean_object* v_res_1145_; 
v___x_1978__boxed_1142_ = lean_unbox(v___x_1130_);
v___y_1979__boxed_1143_ = lean_unbox(v___y_1131_);
v___x_1980__boxed_1144_ = lean_unbox(v___x_1132_);
v_res_1145_ = lp_aesop_Aesop_elabSimpTheorems___lam__0(v_stx_1127_, v_ctx_1128_, v_simprocs_1129_, v___x_1978__boxed_1142_, v___y_1979__boxed_1143_, v___x_1980__boxed_1144_, v___y_1133_, v___y_1134_, v___y_1135_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_);
lean_dec(v___y_1140_);
lean_dec_ref(v___y_1139_);
lean_dec(v___y_1138_);
lean_dec_ref(v___y_1137_);
lean_dec(v___y_1136_);
lean_dec_ref(v___y_1135_);
lean_dec(v___y_1134_);
lean_dec_ref(v___y_1133_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems(lean_object* v_stx_1146_, lean_object* v_ctx_1147_, lean_object* v_simprocs_1148_, uint8_t v_isSimpAll_1149_, lean_object* v_a_1150_, lean_object* v_a_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_, lean_object* v_a_1156_, lean_object* v_a_1157_){
_start:
{
uint8_t v___x_1159_; uint8_t v___y_1161_; 
v___x_1159_ = 1;
if (v_isSimpAll_1149_ == 0)
{
uint8_t v___x_1168_; 
v___x_1168_ = 0;
v___y_1161_ = v___x_1168_;
goto v___jp_1160_;
}
else
{
uint8_t v___x_1169_; 
v___x_1169_ = 1;
v___y_1161_ = v___x_1169_;
goto v___jp_1160_;
}
v___jp_1160_:
{
uint8_t v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___f_1166_; lean_object* v___x_1167_; 
v___x_1162_ = 0;
v___x_1163_ = lean_box(v___x_1159_);
v___x_1164_ = lean_box(v___y_1161_);
v___x_1165_ = lean_box(v___x_1162_);
v___f_1166_ = lean_alloc_closure((void*)(lp_aesop_Aesop_elabSimpTheorems___lam__0___boxed), 15, 6);
lean_closure_set(v___f_1166_, 0, v_stx_1146_);
lean_closure_set(v___f_1166_, 1, v_ctx_1147_);
lean_closure_set(v___f_1166_, 2, v_simprocs_1148_);
lean_closure_set(v___f_1166_, 3, v___x_1163_);
lean_closure_set(v___f_1166_, 4, v___x_1164_);
lean_closure_set(v___f_1166_, 5, v___x_1165_);
v___x_1167_ = l_Lean_Elab_Tactic_withoutRecover___redArg(v___f_1166_, v_a_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_, v_a_1155_, v_a_1156_, v_a_1157_);
return v___x_1167_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabSimpTheorems___boxed(lean_object* v_stx_1170_, lean_object* v_ctx_1171_, lean_object* v_simprocs_1172_, lean_object* v_isSimpAll_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_){
_start:
{
uint8_t v_isSimpAll_boxed_1183_; lean_object* v_res_1184_; 
v_isSimpAll_boxed_1183_ = lean_unbox(v_isSimpAll_1173_);
v_res_1184_ = lp_aesop_Aesop_elabSimpTheorems(v_stx_1170_, v_ctx_1171_, v_simprocs_1172_, v_isSimpAll_boxed_1183_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_);
lean_dec(v_a_1181_);
lean_dec_ref(v_a_1180_);
lean_dec(v_a_1179_);
lean_dec_ref(v_a_1178_);
lean_dec(v_a_1177_);
lean_dec_ref(v_a_1176_);
lean_dec(v_a_1175_);
lean_dec_ref(v_a_1174_);
return v_res_1184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1(lean_object* v_00_u03b1_1185_, lean_object* v_msg_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_){
_start:
{
lean_object* v___x_1196_; 
v___x_1196_ = lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___redArg(v_msg_1186_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_);
return v___x_1196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1___boxed(lean_object* v_00_u03b1_1197_, lean_object* v_msg_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v_res_1208_; 
v_res_1208_ = lp_aesop_Lean_throwError___at___00Aesop_elabSimpTheorems_spec__1(v_00_u03b1_1197_, v_msg_1198_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_);
lean_dec(v___y_1206_);
lean_dec_ref(v___y_1205_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
return v_res_1208_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkSimpArgs___closed__1(void){
_start:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1210_ = ((lean_object*)(lp_aesop_Aesop_mkSimpArgs___closed__0));
v___x_1211_ = l_Lean_mkAtom(v___x_1210_);
return v___x_1211_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkSimpArgs___closed__2(void){
_start:
{
uint8_t v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; 
v___x_1212_ = 0;
v___x_1213_ = lean_box(0);
v___x_1214_ = l_Lean_SourceInfo_fromRef(v___x_1213_, v___x_1212_);
return v___x_1214_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkSimpArgs___closed__10(void){
_start:
{
lean_object* v___x_1227_; 
v___x_1227_ = l_Array_mkArray0(lean_box(0));
return v___x_1227_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkSimpArgs___closed__11(void){
_start:
{
lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; 
v___x_1228_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__10, &lp_aesop_Aesop_mkSimpArgs___closed__10_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__10);
v___x_1229_ = ((lean_object*)(lp_aesop_Aesop_mkSimpArgs___closed__9));
v___x_1230_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__2, &lp_aesop_Aesop_mkSimpArgs___closed__2_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__2);
v___x_1231_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1230_);
lean_ctor_set(v___x_1231_, 1, v___x_1229_);
lean_ctor_set(v___x_1231_, 2, v___x_1228_);
return v___x_1231_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkSimpArgs___closed__13(void){
_start:
{
lean_object* v___x_1233_; lean_object* v___x_1234_; 
v___x_1233_ = ((lean_object*)(lp_aesop_Aesop_mkSimpArgs___closed__12));
v___x_1234_ = l_Lean_mkAtom(v___x_1233_);
return v___x_1234_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkSimpArgs___closed__14(void){
_start:
{
lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; 
v___x_1235_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__1, &lp_aesop_Aesop_mkSimpArgs___closed__1_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__1);
v___x_1236_ = lean_unsigned_to_nat(3u);
v___x_1237_ = lean_mk_empty_array_with_capacity(v___x_1236_);
v___x_1238_ = lean_array_push(v___x_1237_, v___x_1235_);
return v___x_1238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkSimpArgs(lean_object* v_simpTheorem_1239_){
_start:
{
lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; 
v___x_1240_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__2, &lp_aesop_Aesop_mkSimpArgs___closed__2_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__2);
v___x_1241_ = ((lean_object*)(lp_aesop_Aesop_mkSimpArgs___closed__7));
v___x_1242_ = ((lean_object*)(lp_aesop_Aesop_mkSimpArgs___closed__9));
v___x_1243_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__11, &lp_aesop_Aesop_mkSimpArgs___closed__11_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__11);
v___x_1244_ = l_Lean_Syntax_node3(v___x_1240_, v___x_1241_, v___x_1243_, v___x_1243_, v_simpTheorem_1239_);
v___x_1245_ = lean_unsigned_to_nat(1u);
v___x_1246_ = lean_mk_empty_array_with_capacity(v___x_1245_);
v___x_1247_ = lean_array_push(v___x_1246_, v___x_1244_);
v___x_1248_ = lean_box(2);
v___x_1249_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1248_);
lean_ctor_set(v___x_1249_, 1, v___x_1242_);
lean_ctor_set(v___x_1249_, 2, v___x_1247_);
v___x_1250_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__13, &lp_aesop_Aesop_mkSimpArgs___closed__13_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__13);
v___x_1251_ = lean_obj_once(&lp_aesop_Aesop_mkSimpArgs___closed__14, &lp_aesop_Aesop_mkSimpArgs___closed__14_once, _init_lp_aesop_Aesop_mkSimpArgs___closed__14);
v___x_1252_ = lean_array_push(v___x_1251_, v___x_1249_);
v___x_1253_ = lean_array_push(v___x_1252_, v___x_1250_);
v___x_1254_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1254_, 0, v___x_1248_);
lean_ctor_set(v___x_1254_, 1, v___x_1242_);
lean_ctor_set(v___x_1254_, 2, v___x_1253_);
return v___x_1254_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpCore(lean_object* v_goal_1255_, lean_object* v_term_1256_, lean_object* v_ctx_1257_, lean_object* v_simprocs_1258_, uint8_t v_isSimpAll_1259_, lean_object* v_a_1260_, lean_object* v_a_1261_, lean_object* v_a_1262_, lean_object* v_a_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_){
_start:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; 
v___x_1267_ = lp_aesop_Aesop_mkSimpArgs(v_term_1256_);
v___x_1268_ = lean_box(v_isSimpAll_1259_);
v___x_1269_ = lean_alloc_closure((void*)(lp_aesop_Aesop_elabSimpTheorems___boxed), 13, 4);
lean_closure_set(v___x_1269_, 0, v___x_1267_);
lean_closure_set(v___x_1269_, 1, v_ctx_1257_);
lean_closure_set(v___x_1269_, 2, v_simprocs_1258_);
lean_closure_set(v___x_1269_, 3, v___x_1268_);
v___x_1270_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runTacticMAsTermElabM___boxed), 10, 3);
lean_closure_set(v___x_1270_, 0, lean_box(0));
lean_closure_set(v___x_1270_, 1, v_goal_1255_);
lean_closure_set(v___x_1270_, 2, v___x_1269_);
v___x_1271_ = lp_aesop_Aesop_withFullElaboration___redArg(v___x_1270_, v_a_1260_, v_a_1261_, v_a_1262_, v_a_1263_, v_a_1264_, v_a_1265_);
return v___x_1271_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpCore___boxed(lean_object* v_goal_1272_, lean_object* v_term_1273_, lean_object* v_ctx_1274_, lean_object* v_simprocs_1275_, lean_object* v_isSimpAll_1276_, lean_object* v_a_1277_, lean_object* v_a_1278_, lean_object* v_a_1279_, lean_object* v_a_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_){
_start:
{
uint8_t v_isSimpAll_boxed_1284_; lean_object* v_res_1285_; 
v_isSimpAll_boxed_1284_ = lean_unbox(v_isSimpAll_1276_);
v_res_1285_ = lp_aesop_Aesop_elabRuleTermForSimpCore(v_goal_1272_, v_term_1273_, v_ctx_1274_, v_simprocs_1275_, v_isSimpAll_boxed_1284_, v_a_1277_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_);
lean_dec(v_a_1282_);
lean_dec_ref(v_a_1281_);
lean_dec(v_a_1280_);
lean_dec_ref(v_a_1279_);
lean_dec(v_a_1278_);
lean_dec_ref(v_a_1277_);
return v_res_1285_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1286_; 
v___x_1286_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1286_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1287_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__0);
v___x_1288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1287_);
return v___x_1288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0(lean_object* v_00_u03b2_1289_){
_start:
{
lean_object* v___x_1290_; 
v___x_1290_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0___closed__1);
return v___x_1290_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__0(void){
_start:
{
lean_object* v___x_1291_; 
v___x_1291_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1291_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1292_; lean_object* v___x_1293_; 
v___x_1292_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__0);
v___x_1293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1293_, 0, v___x_1292_);
return v___x_1293_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1(lean_object* v_00_u03b2_1294_){
_start:
{
lean_object* v___x_1295_; 
v___x_1295_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1___closed__1);
return v___x_1295_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1(void){
_start:
{
lean_object* v___x_1303_; 
v___x_1303_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_1303_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__2(void){
_start:
{
lean_object* v___x_1304_; 
v___x_1304_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__0(lean_box(0));
return v___x_1304_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3(void){
_start:
{
lean_object* v___x_1305_; 
v___x_1305_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_checkElabRuleTermForSimp_spec__1(lean_box(0));
return v___x_1305_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__4(void){
_start:
{
lean_object* v___x_1306_; 
v___x_1306_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1306_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5(void){
_start:
{
lean_object* v___x_1307_; lean_object* v___x_1308_; 
v___x_1307_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__4, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__4_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__4);
v___x_1308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1308_, 0, v___x_1307_);
return v___x_1308_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__6(void){
_start:
{
lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; 
v___x_1309_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5);
v___x_1310_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3);
v___x_1311_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__2, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__2_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__2);
v___x_1312_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1);
v___x_1313_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1313_, 0, v___x_1312_);
lean_ctor_set(v___x_1313_, 1, v___x_1312_);
lean_ctor_set(v___x_1313_, 2, v___x_1311_);
lean_ctor_set(v___x_1313_, 3, v___x_1310_);
lean_ctor_set(v___x_1313_, 4, v___x_1311_);
lean_ctor_set(v___x_1313_, 5, v___x_1309_);
return v___x_1313_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__7(void){
_start:
{
lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; 
v___x_1314_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__6, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__6_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__6);
v___x_1315_ = lean_unsigned_to_nat(1u);
v___x_1316_ = lean_mk_empty_array_with_capacity(v___x_1315_);
v___x_1317_ = lean_array_push(v___x_1316_, v___x_1314_);
return v___x_1317_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__8(void){
_start:
{
lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; 
v___x_1318_ = lean_box(0);
v___x_1319_ = lean_unsigned_to_nat(16u);
v___x_1320_ = lean_mk_array(v___x_1319_, v___x_1318_);
return v___x_1320_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__9(void){
_start:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; 
v___x_1321_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__8, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__8_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__8);
v___x_1322_ = lean_unsigned_to_nat(0u);
v___x_1323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1323_, 0, v___x_1322_);
lean_ctor_set(v___x_1323_, 1, v___x_1321_);
return v___x_1323_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__10(void){
_start:
{
lean_object* v___x_1324_; lean_object* v___x_1325_; uint8_t v___x_1326_; lean_object* v___x_1327_; 
v___x_1324_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__5);
v___x_1325_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__9, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__9_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__9);
v___x_1326_ = 1;
v___x_1327_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1327_, 0, v___x_1325_);
lean_ctor_set(v___x_1327_, 1, v___x_1324_);
lean_ctor_set_uint8(v___x_1327_, sizeof(void*)*2, v___x_1326_);
return v___x_1327_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__11(void){
_start:
{
lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; 
v___x_1328_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__3);
v___x_1329_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__1);
v___x_1330_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1329_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
lean_ctor_set(v___x_1330_, 2, v___x_1328_);
lean_ctor_set(v___x_1330_, 3, v___x_1328_);
return v___x_1330_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__12(void){
_start:
{
lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1331_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__11, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__11_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__11);
v___x_1332_ = lean_unsigned_to_nat(1u);
v___x_1333_ = lean_mk_empty_array_with_capacity(v___x_1332_);
v___x_1334_ = lean_array_push(v___x_1333_, v___x_1331_);
return v___x_1334_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp(lean_object* v_term_1335_, uint8_t v_isSimpAll_1336_, lean_object* v_a_1337_, lean_object* v_a_1338_, lean_object* v_a_1339_, lean_object* v_a_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_){
_start:
{
lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1345_ = ((lean_object*)(lp_aesop_Aesop_checkElabRuleTermForSimp___closed__0));
v___x_1346_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__7, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__7_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__7);
v___x_1347_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__10, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__10_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__10);
v___x_1348_ = l_Lean_Options_empty;
v___x_1349_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_1345_, v___x_1346_, v___x_1347_, v___x_1348_, v_a_1340_, v_a_1342_, v_a_1343_);
if (lean_obj_tag(v___x_1349_) == 0)
{
lean_object* v_a_1350_; lean_object* v_goal_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; 
v_a_1350_ = lean_ctor_get(v___x_1349_, 0);
lean_inc(v_a_1350_);
lean_dec_ref_known(v___x_1349_, 1);
v_goal_1351_ = lean_ctor_get(v_a_1337_, 0);
v___x_1352_ = lean_obj_once(&lp_aesop_Aesop_checkElabRuleTermForSimp___closed__12, &lp_aesop_Aesop_checkElabRuleTermForSimp___closed__12_once, _init_lp_aesop_Aesop_checkElabRuleTermForSimp___closed__12);
lean_inc(v_goal_1351_);
v___x_1353_ = lp_aesop_Aesop_elabRuleTermForSimpCore(v_goal_1351_, v_term_1335_, v_a_1350_, v___x_1352_, v_isSimpAll_1336_, v_a_1338_, v_a_1339_, v_a_1340_, v_a_1341_, v_a_1342_, v_a_1343_);
if (lean_obj_tag(v___x_1353_) == 0)
{
lean_object* v___x_1355_; uint8_t v_isShared_1356_; uint8_t v_isSharedCheck_1361_; 
v_isSharedCheck_1361_ = !lean_is_exclusive(v___x_1353_);
if (v_isSharedCheck_1361_ == 0)
{
lean_object* v_unused_1362_; 
v_unused_1362_ = lean_ctor_get(v___x_1353_, 0);
lean_dec(v_unused_1362_);
v___x_1355_ = v___x_1353_;
v_isShared_1356_ = v_isSharedCheck_1361_;
goto v_resetjp_1354_;
}
else
{
lean_dec(v___x_1353_);
v___x_1355_ = lean_box(0);
v_isShared_1356_ = v_isSharedCheck_1361_;
goto v_resetjp_1354_;
}
v_resetjp_1354_:
{
lean_object* v___x_1357_; lean_object* v___x_1359_; 
v___x_1357_ = lean_box(0);
if (v_isShared_1356_ == 0)
{
lean_ctor_set(v___x_1355_, 0, v___x_1357_);
v___x_1359_ = v___x_1355_;
goto v_reusejp_1358_;
}
else
{
lean_object* v_reuseFailAlloc_1360_; 
v_reuseFailAlloc_1360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1360_, 0, v___x_1357_);
v___x_1359_ = v_reuseFailAlloc_1360_;
goto v_reusejp_1358_;
}
v_reusejp_1358_:
{
return v___x_1359_;
}
}
}
else
{
lean_object* v_a_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1370_; 
v_a_1363_ = lean_ctor_get(v___x_1353_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1353_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1365_ = v___x_1353_;
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_a_1363_);
lean_dec(v___x_1353_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v___x_1368_; 
if (v_isShared_1366_ == 0)
{
v___x_1368_ = v___x_1365_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_a_1363_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
else
{
lean_object* v_a_1371_; lean_object* v___x_1373_; uint8_t v_isShared_1374_; uint8_t v_isSharedCheck_1378_; 
lean_dec(v_term_1335_);
v_a_1371_ = lean_ctor_get(v___x_1349_, 0);
v_isSharedCheck_1378_ = !lean_is_exclusive(v___x_1349_);
if (v_isSharedCheck_1378_ == 0)
{
v___x_1373_ = v___x_1349_;
v_isShared_1374_ = v_isSharedCheck_1378_;
goto v_resetjp_1372_;
}
else
{
lean_inc(v_a_1371_);
lean_dec(v___x_1349_);
v___x_1373_ = lean_box(0);
v_isShared_1374_ = v_isSharedCheck_1378_;
goto v_resetjp_1372_;
}
v_resetjp_1372_:
{
lean_object* v___x_1376_; 
if (v_isShared_1374_ == 0)
{
v___x_1376_ = v___x_1373_;
goto v_reusejp_1375_;
}
else
{
lean_object* v_reuseFailAlloc_1377_; 
v_reuseFailAlloc_1377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1377_, 0, v_a_1371_);
v___x_1376_ = v_reuseFailAlloc_1377_;
goto v_reusejp_1375_;
}
v_reusejp_1375_:
{
return v___x_1376_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkElabRuleTermForSimp___boxed(lean_object* v_term_1379_, lean_object* v_isSimpAll_1380_, lean_object* v_a_1381_, lean_object* v_a_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_, lean_object* v_a_1388_){
_start:
{
uint8_t v_isSimpAll_boxed_1389_; lean_object* v_res_1390_; 
v_isSimpAll_boxed_1389_ = lean_unbox(v_isSimpAll_1380_);
v_res_1390_ = lp_aesop_Aesop_checkElabRuleTermForSimp(v_term_1379_, v_isSimpAll_boxed_1389_, v_a_1381_, v_a_1382_, v_a_1383_, v_a_1384_, v_a_1385_, v_a_1386_, v_a_1387_);
lean_dec(v_a_1387_);
lean_dec_ref(v_a_1386_);
lean_dec(v_a_1385_);
lean_dec_ref(v_a_1384_);
lean_dec(v_a_1383_);
lean_dec_ref(v_a_1382_);
lean_dec_ref(v_a_1381_);
return v_res_1390_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpMetaM(lean_object* v_goal_1391_, lean_object* v_term_1392_, lean_object* v_ctx_1393_, lean_object* v_simprocs_1394_, uint8_t v_isSimpAll_1395_, lean_object* v_a_1396_, lean_object* v_a_1397_, lean_object* v_a_1398_, lean_object* v_a_1399_){
_start:
{
lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; 
v___x_1401_ = lean_box(v_isSimpAll_1395_);
v___x_1402_ = lean_alloc_closure((void*)(lp_aesop_Aesop_elabRuleTermForSimpCore___boxed), 12, 5);
lean_closure_set(v___x_1402_, 0, v_goal_1391_);
lean_closure_set(v___x_1402_, 1, v_term_1392_);
lean_closure_set(v___x_1402_, 2, v_ctx_1393_);
lean_closure_set(v___x_1402_, 3, v_simprocs_1394_);
lean_closure_set(v___x_1402_, 4, v___x_1401_);
v___x_1403_ = ((lean_object*)(lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__2));
v___x_1404_ = ((lean_object*)(lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM___closed__3));
v___x_1405_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_1402_, v___x_1403_, v___x_1404_, v_a_1396_, v_a_1397_, v_a_1398_, v_a_1399_);
if (lean_obj_tag(v___x_1405_) == 0)
{
lean_object* v_a_1406_; lean_object* v___x_1408_; uint8_t v_isShared_1409_; uint8_t v_isSharedCheck_1414_; 
v_a_1406_ = lean_ctor_get(v___x_1405_, 0);
v_isSharedCheck_1414_ = !lean_is_exclusive(v___x_1405_);
if (v_isSharedCheck_1414_ == 0)
{
v___x_1408_ = v___x_1405_;
v_isShared_1409_ = v_isSharedCheck_1414_;
goto v_resetjp_1407_;
}
else
{
lean_inc(v_a_1406_);
lean_dec(v___x_1405_);
v___x_1408_ = lean_box(0);
v_isShared_1409_ = v_isSharedCheck_1414_;
goto v_resetjp_1407_;
}
v_resetjp_1407_:
{
lean_object* v_fst_1410_; lean_object* v___x_1412_; 
v_fst_1410_ = lean_ctor_get(v_a_1406_, 0);
lean_inc(v_fst_1410_);
lean_dec(v_a_1406_);
if (v_isShared_1409_ == 0)
{
lean_ctor_set(v___x_1408_, 0, v_fst_1410_);
v___x_1412_ = v___x_1408_;
goto v_reusejp_1411_;
}
else
{
lean_object* v_reuseFailAlloc_1413_; 
v_reuseFailAlloc_1413_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1413_, 0, v_fst_1410_);
v___x_1412_ = v_reuseFailAlloc_1413_;
goto v_reusejp_1411_;
}
v_reusejp_1411_:
{
return v___x_1412_;
}
}
}
else
{
lean_object* v_a_1415_; lean_object* v___x_1417_; uint8_t v_isShared_1418_; uint8_t v_isSharedCheck_1422_; 
v_a_1415_ = lean_ctor_get(v___x_1405_, 0);
v_isSharedCheck_1422_ = !lean_is_exclusive(v___x_1405_);
if (v_isSharedCheck_1422_ == 0)
{
v___x_1417_ = v___x_1405_;
v_isShared_1418_ = v_isSharedCheck_1422_;
goto v_resetjp_1416_;
}
else
{
lean_inc(v_a_1415_);
lean_dec(v___x_1405_);
v___x_1417_ = lean_box(0);
v_isShared_1418_ = v_isSharedCheck_1422_;
goto v_resetjp_1416_;
}
v_resetjp_1416_:
{
lean_object* v___x_1420_; 
if (v_isShared_1418_ == 0)
{
v___x_1420_ = v___x_1417_;
goto v_reusejp_1419_;
}
else
{
lean_object* v_reuseFailAlloc_1421_; 
v_reuseFailAlloc_1421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1421_, 0, v_a_1415_);
v___x_1420_ = v_reuseFailAlloc_1421_;
goto v_reusejp_1419_;
}
v_reusejp_1419_:
{
return v___x_1420_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabRuleTermForSimpMetaM___boxed(lean_object* v_goal_1423_, lean_object* v_term_1424_, lean_object* v_ctx_1425_, lean_object* v_simprocs_1426_, lean_object* v_isSimpAll_1427_, lean_object* v_a_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_, lean_object* v_a_1431_, lean_object* v_a_1432_){
_start:
{
uint8_t v_isSimpAll_boxed_1433_; lean_object* v_res_1434_; 
v_isSimpAll_boxed_1433_ = lean_unbox(v_isSimpAll_1427_);
v_res_1434_ = lp_aesop_Aesop_elabRuleTermForSimpMetaM(v_goal_1423_, v_term_1424_, v_ctx_1425_, v_simprocs_1426_, v_isSimpAll_boxed_1433_, v_a_1428_, v_a_1429_, v_a_1430_, v_a_1431_);
lean_dec(v_a_1431_);
lean_dec_ref(v_a_1430_);
lean_dec(v_a_1429_);
lean_dec_ref(v_a_1428_);
return v_res_1434_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Simproc(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_ElabM(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Simproc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_ElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Simproc(uint8_t builtin);
lean_object* initialize_aesop_Aesop_ElabM(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Simproc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_ElabM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
}
#ifdef __cplusplus
}
#endif
