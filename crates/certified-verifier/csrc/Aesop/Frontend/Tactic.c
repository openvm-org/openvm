// Lean compiler output
// Module: Aesop.Frontend.Tactic
// Imports: public import Init public meta import Init public import Aesop.Frontend.RuleExpr public import Aesop.RuleSet import Aesop.Frontend.Extension import Batteries.Linter.UnreachableTactic import Lean.Elab.SyntheticMVars import Lean.Meta.Eval
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedFileMap_default;
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Elab_InfoTree_substitute(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_Term_TermElabM_0__Lean_Elab_Term_withoutModifyingStateWithInfoAndMessagesImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_evalExpr_x27___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleSetName_elab(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabM_Context_forErasing(lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleExpr_elab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalRules(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lp_aesop_Aesop_getDefaultRuleSetNames();
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_generateScript;
extern lean_object* lp_aesop_Aesop_Check_script;
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Check_script_steps;
lean_object* lp_aesop_Aesop_LocalRuleSet_add(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleExpr_toLocalRuleFilters(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_erase(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleExpr_buildAdditionalLocalRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkLocalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 163, 173, 41, 168, 168, 65, 81)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tactic_clause"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__6_value),LEAN_SCALAR_PTR_LITERAL(155, 100, 241, 83, 163, 176, 80, 201)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(161, 46, 151, 87, 121, 28, 117, 117)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__8_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "`(tactic_clause| "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__12_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__6_value),LEAN_SCALAR_PTR_LITERAL(155, 100, 241, 83, 163, 176, 80, 201)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__13_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__13_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__11_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__17 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__7_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__17_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__18 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__19 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__19_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__19_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Parser_Category_Aesop_tactic__clause;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ruleSetSpec"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 32, 74, 104, 45, 149, 1, 73)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__3_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__7_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__0_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__12_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_ruleSetSpec = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tactic_clause(Add_)"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(191, 185, 249, 38, 205, 193, 185, 174)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "add "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__6_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rule_expr"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__8_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__7_value),LEAN_SCALAR_PTR_LITERAL(99, 17, 168, 180, 164, 44, 144, 28)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__10_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__10_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__12_value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__14_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__16_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__16_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "tactic_clause(Erase_)"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(189, 18, 116, 19, 126, 120, 125, 141)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "erase "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__5_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__7_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "tactic_clause(Rule_sets:=[_])"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 132, 153, 102, 236, 217, 165, 41)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rule_sets"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__7_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__12_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__10_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__12_value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__10_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__12_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__15_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__17 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__17_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__17_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "tactic_clause(Config:=_)"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 251, 221, 65, 202, 52, 203, 84)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__5_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__11_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__11_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "tactic_clause(Simp_config:=_)"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(47, 11, 106, 227, 85, 245, 7, 77)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "simp_config"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__5_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__8_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "aesopTactic"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(54, 142, 162, 195, 161, 101, 248, 175)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__6_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__9_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__8_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__12_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__5_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__16_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__16_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "aesopTactic\?"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 245, 87, 84, 72, 165, 203, 79)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "aesop\?"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__5_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_Parser_initFn_00___x40_Aesop_Frontend_Tactic_1696840279____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_Parser_initFn_00___x40_Aesop_Frontend_Tactic_1696840279____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9_spec__10(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__0;
static lean_once_cell_t lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__4;
static const lean_array_object lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Options"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(246, 158, 67, 126, 40, 141, 32, 135)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabOptions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabOptions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Simp"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 38, 229, 237, 143, 62, 212, 6)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 221, 251, 144, 155, 142, 49, 30)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfig(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "ConfigCtx"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_tactic__clause_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 38, 229, 237, 143, 62, 212, 6)}};
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 124, 165, 154, 200, 99, 195, 142)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfigCtx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfigCtx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "aesop: rule set '"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "' is already active"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__3;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "aesop: trying to deactivate rule set '"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__4_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__5;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "', but it is not active"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__6_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__1_value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg();
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_parse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_parse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0;
static const lean_string_object lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__1 = (const lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__1_value;
static const lean_ctor_object lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__1_value)}};
static const lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__2 = (const lean_object*)&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__2_value;
static lean_once_cell_t lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "aesop: '"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "' is not registered (with the given features) in any rule set."};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_updateRuleSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_updateRuleSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Lean_Parser_Category_Aesop_tactic__clause(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_box(0);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_Parser_initFn_00___x40_Aesop_Frontend_Tactic_1696840279____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_336_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1));
v___x_337_ = lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(v___x_336_);
if (lean_obj_tag(v___x_337_) == 0)
{
lean_object* v___x_338_; lean_object* v___x_339_; 
lean_dec_ref_known(v___x_337_, 1);
v___x_338_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1));
v___x_339_ = lp_batteries_Batteries_Linter_UnreachableTactic_addIgnoreTacticKind(v___x_338_);
return v___x_339_;
}
else
{
return v___x_337_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_Parser_initFn_00___x40_Aesop_Frontend_Tactic_1696840279____hygCtx___hyg_2____boxed(lean_object* v_a_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_Parser_initFn_00___x40_Aesop_Frontend_Tactic_1696840279____hygCtx___hyg_2_();
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg(lean_object* v_e_342_, lean_object* v___y_343_){
_start:
{
uint8_t v___x_345_; 
v___x_345_ = l_Lean_Expr_hasMVar(v_e_342_);
if (v___x_345_ == 0)
{
lean_object* v___x_346_; 
v___x_346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_346_, 0, v_e_342_);
return v___x_346_;
}
else
{
lean_object* v___x_347_; lean_object* v_mctx_348_; lean_object* v___x_349_; lean_object* v_fst_350_; lean_object* v_snd_351_; lean_object* v___x_352_; lean_object* v_cache_353_; lean_object* v_zetaDeltaFVarIds_354_; lean_object* v_postponed_355_; lean_object* v_diag_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_365_; 
v___x_347_ = lean_st_ref_get(v___y_343_);
v_mctx_348_ = lean_ctor_get(v___x_347_, 0);
lean_inc_ref(v_mctx_348_);
lean_dec(v___x_347_);
v___x_349_ = l_Lean_instantiateMVarsCore(v_mctx_348_, v_e_342_);
v_fst_350_ = lean_ctor_get(v___x_349_, 0);
lean_inc(v_fst_350_);
v_snd_351_ = lean_ctor_get(v___x_349_, 1);
lean_inc(v_snd_351_);
lean_dec_ref(v___x_349_);
v___x_352_ = lean_st_ref_take(v___y_343_);
v_cache_353_ = lean_ctor_get(v___x_352_, 1);
v_zetaDeltaFVarIds_354_ = lean_ctor_get(v___x_352_, 2);
v_postponed_355_ = lean_ctor_get(v___x_352_, 3);
v_diag_356_ = lean_ctor_get(v___x_352_, 4);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_352_);
if (v_isSharedCheck_365_ == 0)
{
lean_object* v_unused_366_; 
v_unused_366_ = lean_ctor_get(v___x_352_, 0);
lean_dec(v_unused_366_);
v___x_358_ = v___x_352_;
v_isShared_359_ = v_isSharedCheck_365_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_diag_356_);
lean_inc(v_postponed_355_);
lean_inc(v_zetaDeltaFVarIds_354_);
lean_inc(v_cache_353_);
lean_dec(v___x_352_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_365_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_361_; 
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 0, v_snd_351_);
v___x_361_ = v___x_358_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v_snd_351_);
lean_ctor_set(v_reuseFailAlloc_364_, 1, v_cache_353_);
lean_ctor_set(v_reuseFailAlloc_364_, 2, v_zetaDeltaFVarIds_354_);
lean_ctor_set(v_reuseFailAlloc_364_, 3, v_postponed_355_);
lean_ctor_set(v_reuseFailAlloc_364_, 4, v_diag_356_);
v___x_361_ = v_reuseFailAlloc_364_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_362_ = lean_st_ref_set(v___y_343_, v___x_361_);
v___x_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_363_, 0, v_fst_350_);
return v___x_363_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg___boxed(lean_object* v_e_367_, lean_object* v___y_368_, lean_object* v___y_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg(v_e_367_, v___y_368_);
lean_dec(v___y_368_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0(lean_object* v_e_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg(v_e_371_, v___y_375_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___boxed(lean_object* v_e_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0(v_e_380_, v___y_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec(v___y_384_);
lean_dec_ref(v___y_383_);
lean_dec(v___y_382_);
lean_dec_ref(v___y_381_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___redArg(lean_object* v_a_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_389_, v___y_390_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___redArg___boxed(lean_object* v_a_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___redArg(v_a_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
lean_dec(v___y_400_);
lean_dec_ref(v___y_399_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1(lean_object* v_00_u03b1_407_, lean_object* v_a_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___boxed(lean_object* v_00_u03b1_417_, lean_object* v_a_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1(v_00_u03b1_417_, v_a_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0(lean_object* v_x_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v___x_435_; 
lean_inc(v___y_429_);
lean_inc_ref(v___y_428_);
v___x_435_ = lean_apply_7(v_x_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_, lean_box(0));
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0___boxed(lean_object* v_x_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0(v_x_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg(lean_object* v_lctx_445_, lean_object* v_localInsts_446_, lean_object* v_x_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v___f_455_; lean_object* v___x_456_; 
lean_inc(v___y_449_);
lean_inc_ref(v___y_448_);
v___f_455_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_455_, 0, v_x_447_);
lean_closure_set(v___f_455_, 1, v___y_448_);
lean_closure_set(v___f_455_, 2, v___y_449_);
v___x_456_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_445_, v_localInsts_446_, v___f_455_, v___y_450_, v___y_451_, v___y_452_, v___y_453_);
if (lean_obj_tag(v___x_456_) == 0)
{
return v___x_456_;
}
else
{
lean_object* v_a_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_464_; 
v_a_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_464_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_464_ == 0)
{
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_a_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_462_; 
if (v_isShared_460_ == 0)
{
v___x_462_ = v___x_459_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v_a_457_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___boxed(lean_object* v_lctx_465_, lean_object* v_localInsts_466_, lean_object* v_x_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v_res_475_; 
v_res_475_ = lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg(v_lctx_465_, v_localInsts_466_, v_x_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3(lean_object* v_00_u03b1_476_, lean_object* v_lctx_477_, lean_object* v_localInsts_478_, lean_object* v_x_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg(v_lctx_477_, v_localInsts_478_, v_x_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_, v___y_484_, v___y_485_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___boxed(lean_object* v_00_u03b1_488_, lean_object* v_lctx_489_, lean_object* v_localInsts_490_, lean_object* v_x_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3(v_00_u03b1_488_, v_lctx_489_, v_localInsts_490_, v_x_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_);
lean_dec(v___y_497_);
lean_dec_ref(v___y_496_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg(lean_object* v_x_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = l___private_Lean_Elab_Term_TermElabM_0__Lean_Elab_Term_withoutModifyingStateWithInfoAndMessagesImpl(lean_box(0), v_x_500_, v___y_501_, v___y_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v_a_509_; lean_object* v___x_511_; uint8_t v_isShared_512_; uint8_t v_isSharedCheck_516_; 
v_a_509_ = lean_ctor_get(v___x_508_, 0);
v_isSharedCheck_516_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_516_ == 0)
{
v___x_511_ = v___x_508_;
v_isShared_512_ = v_isSharedCheck_516_;
goto v_resetjp_510_;
}
else
{
lean_inc(v_a_509_);
lean_dec(v___x_508_);
v___x_511_ = lean_box(0);
v_isShared_512_ = v_isSharedCheck_516_;
goto v_resetjp_510_;
}
v_resetjp_510_:
{
lean_object* v___x_514_; 
if (v_isShared_512_ == 0)
{
v___x_514_ = v___x_511_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_a_509_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
}
else
{
lean_object* v_a_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_524_; 
v_a_517_ = lean_ctor_get(v___x_508_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_524_ == 0)
{
v___x_519_ = v___x_508_;
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_a_517_);
lean_dec(v___x_508_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_522_; 
if (v_isShared_520_ == 0)
{
v___x_522_ = v___x_519_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v_a_517_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
return v___x_522_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg___boxed(lean_object* v_x_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg(v_x_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_);
lean_dec(v___y_531_);
lean_dec_ref(v___y_530_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4(lean_object* v_00_u03b1_534_, lean_object* v_x_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_){
_start:
{
lean_object* v___x_543_; 
v___x_543_ = lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg(v_x_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___boxed(lean_object* v_00_u03b1_544_, lean_object* v_x_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_){
_start:
{
lean_object* v_res_553_; 
v_res_553_ = lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4(v_00_u03b1_544_, v_x_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
return v_res_553_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__0(lean_object* v_stx_554_, lean_object* v___x_555_, uint8_t v___x_556_, lean_object* v___x_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_){
_start:
{
lean_object* v___x_565_; 
v___x_565_ = l_Lean_Elab_Term_elabTermEnsuringType(v_stx_554_, v___x_555_, v___x_556_, v___x_556_, v___x_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v_a_566_; uint8_t v___x_567_; lean_object* v___x_568_; 
v_a_566_ = lean_ctor_get(v___x_565_, 0);
lean_inc(v_a_566_);
lean_dec_ref_known(v___x_565_, 1);
v___x_567_ = 0;
v___x_568_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_567_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
if (lean_obj_tag(v___x_568_) == 0)
{
lean_object* v___x_569_; 
lean_dec_ref_known(v___x_568_, 1);
v___x_569_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_Frontend_elabConfigUnsafe_spec__0___redArg(v_a_566_, v___y_561_);
return v___x_569_;
}
else
{
lean_object* v_a_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_577_; 
lean_dec(v_a_566_);
v_a_570_ = lean_ctor_get(v___x_568_, 0);
v_isSharedCheck_577_ = !lean_is_exclusive(v___x_568_);
if (v_isSharedCheck_577_ == 0)
{
v___x_572_ = v___x_568_;
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_a_570_);
lean_dec(v___x_568_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_575_; 
if (v_isShared_573_ == 0)
{
v___x_575_ = v___x_572_;
goto v_reusejp_574_;
}
else
{
lean_object* v_reuseFailAlloc_576_; 
v_reuseFailAlloc_576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_576_, 0, v_a_570_);
v___x_575_ = v_reuseFailAlloc_576_;
goto v_reusejp_574_;
}
v_reusejp_574_:
{
return v___x_575_;
}
}
}
}
else
{
return v___x_565_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__0___boxed(lean_object* v_stx_578_, lean_object* v___x_579_, lean_object* v___x_580_, lean_object* v___x_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
uint8_t v___x_11541__boxed_589_; lean_object* v_res_590_; 
v___x_11541__boxed_589_ = lean_unbox(v___x_580_);
v_res_590_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__0(v_stx_578_, v___x_579_, v___x_11541__boxed_589_, v___x_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_);
lean_dec(v___y_587_);
lean_dec_ref(v___y_586_);
lean_dec(v___y_585_);
lean_dec_ref(v___y_584_);
lean_dec(v___y_583_);
lean_dec_ref(v___y_582_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__1(lean_object* v___x_591_, uint8_t v___x_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_){
_start:
{
lean_object* v___x_600_; 
v___x_600_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_591_, v___x_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_);
return v___x_600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__1___boxed(lean_object* v___x_601_, lean_object* v___x_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_){
_start:
{
uint8_t v___x_11599__boxed_610_; lean_object* v_res_611_; 
v___x_11599__boxed_610_ = lean_unbox(v___x_602_);
v_res_611_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__1(v___x_601_, v___x_11599__boxed_610_, v___y_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_);
lean_dec(v___y_608_);
lean_dec_ref(v___y_607_);
lean_dec(v___y_606_);
lean_dec_ref(v___y_605_);
lean_dec(v___y_604_);
lean_dec_ref(v___y_603_);
return v_res_611_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10(lean_object* v___x_612_, lean_object* v_ctx_x3f_613_, size_t v_sz_614_, size_t v_i_615_, lean_object* v_bs_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
uint8_t v___x_624_; 
v___x_624_ = lean_usize_dec_lt(v_i_615_, v_sz_614_);
if (v___x_624_ == 0)
{
lean_object* v___x_625_; 
lean_dec_ref(v_ctx_x3f_613_);
v___x_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_625_, 0, v_bs_616_);
return v___x_625_;
}
else
{
lean_object* v_assignment_626_; lean_object* v___x_627_; 
v_assignment_626_ = lean_ctor_get(v___x_612_, 0);
lean_inc_ref(v_ctx_x3f_613_);
lean_inc(v___y_622_);
lean_inc_ref(v___y_621_);
lean_inc(v___y_620_);
lean_inc_ref(v___y_619_);
lean_inc(v___y_618_);
lean_inc_ref(v___y_617_);
v___x_627_ = lean_apply_7(v_ctx_x3f_613_, v___y_617_, v___y_618_, v___y_619_, v___y_620_, v___y_621_, v___y_622_, lean_box(0));
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v_v_629_; lean_object* v___x_630_; lean_object* v_bs_x27_631_; lean_object* v_a_633_; lean_object* v_tree_638_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_a_628_);
lean_dec_ref_known(v___x_627_, 1);
v_v_629_ = lean_array_uget(v_bs_616_, v_i_615_);
v___x_630_ = lean_unsigned_to_nat(0u);
v_bs_x27_631_ = lean_array_uset(v_bs_616_, v_i_615_, v___x_630_);
v_tree_638_ = l_Lean_Elab_InfoTree_substitute(v_v_629_, v_assignment_626_);
if (lean_obj_tag(v_a_628_) == 0)
{
v_a_633_ = v_tree_638_;
goto v___jp_632_;
}
else
{
lean_object* v_val_639_; lean_object* v___x_640_; 
v_val_639_ = lean_ctor_get(v_a_628_, 0);
lean_inc(v_val_639_);
lean_dec_ref_known(v_a_628_, 1);
v___x_640_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_640_, 0, v_val_639_);
lean_ctor_set(v___x_640_, 1, v_tree_638_);
v_a_633_ = v___x_640_;
goto v___jp_632_;
}
v___jp_632_:
{
size_t v___x_634_; size_t v___x_635_; lean_object* v___x_636_; 
v___x_634_ = ((size_t)1ULL);
v___x_635_ = lean_usize_add(v_i_615_, v___x_634_);
v___x_636_ = lean_array_uset(v_bs_x27_631_, v_i_615_, v_a_633_);
v_i_615_ = v___x_635_;
v_bs_616_ = v___x_636_;
goto _start;
}
}
else
{
lean_object* v_a_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_648_; 
lean_dec_ref(v_bs_616_);
lean_dec_ref(v_ctx_x3f_613_);
v_a_641_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_648_ == 0)
{
v___x_643_ = v___x_627_;
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_a_641_);
lean_dec(v___x_627_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
v_resetjp_642_:
{
lean_object* v___x_646_; 
if (v_isShared_644_ == 0)
{
v___x_646_ = v___x_643_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v_a_641_);
v___x_646_ = v_reuseFailAlloc_647_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
return v___x_646_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10___boxed(lean_object* v___x_649_, lean_object* v_ctx_x3f_650_, lean_object* v_sz_651_, lean_object* v_i_652_, lean_object* v_bs_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_){
_start:
{
size_t v_sz_boxed_661_; size_t v_i_boxed_662_; lean_object* v_res_663_; 
v_sz_boxed_661_ = lean_unbox_usize(v_sz_651_);
lean_dec(v_sz_651_);
v_i_boxed_662_ = lean_unbox_usize(v_i_652_);
lean_dec(v_i_652_);
v_res_663_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10(v___x_649_, v_ctx_x3f_650_, v_sz_boxed_661_, v_i_boxed_662_, v_bs_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
lean_dec_ref(v___x_649_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9(lean_object* v___x_664_, lean_object* v_ctx_x3f_665_, lean_object* v_x_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
if (lean_obj_tag(v_x_666_) == 0)
{
lean_object* v_cs_674_; lean_object* v___x_676_; uint8_t v_isShared_677_; uint8_t v_isSharedCheck_700_; 
v_cs_674_ = lean_ctor_get(v_x_666_, 0);
v_isSharedCheck_700_ = !lean_is_exclusive(v_x_666_);
if (v_isSharedCheck_700_ == 0)
{
v___x_676_ = v_x_666_;
v_isShared_677_ = v_isSharedCheck_700_;
goto v_resetjp_675_;
}
else
{
lean_inc(v_cs_674_);
lean_dec(v_x_666_);
v___x_676_ = lean_box(0);
v_isShared_677_ = v_isSharedCheck_700_;
goto v_resetjp_675_;
}
v_resetjp_675_:
{
size_t v_sz_678_; size_t v___x_679_; lean_object* v___x_680_; 
v_sz_678_ = lean_array_size(v_cs_674_);
v___x_679_ = ((size_t)0ULL);
v___x_680_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9_spec__10(v___x_664_, v_ctx_x3f_665_, v_sz_678_, v___x_679_, v_cs_674_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_);
if (lean_obj_tag(v___x_680_) == 0)
{
lean_object* v_a_681_; lean_object* v___x_683_; uint8_t v_isShared_684_; uint8_t v_isSharedCheck_691_; 
v_a_681_ = lean_ctor_get(v___x_680_, 0);
v_isSharedCheck_691_ = !lean_is_exclusive(v___x_680_);
if (v_isSharedCheck_691_ == 0)
{
v___x_683_ = v___x_680_;
v_isShared_684_ = v_isSharedCheck_691_;
goto v_resetjp_682_;
}
else
{
lean_inc(v_a_681_);
lean_dec(v___x_680_);
v___x_683_ = lean_box(0);
v_isShared_684_ = v_isSharedCheck_691_;
goto v_resetjp_682_;
}
v_resetjp_682_:
{
lean_object* v___x_686_; 
if (v_isShared_677_ == 0)
{
lean_ctor_set(v___x_676_, 0, v_a_681_);
v___x_686_ = v___x_676_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_690_; 
v_reuseFailAlloc_690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_690_, 0, v_a_681_);
v___x_686_ = v_reuseFailAlloc_690_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
lean_object* v___x_688_; 
if (v_isShared_684_ == 0)
{
lean_ctor_set(v___x_683_, 0, v___x_686_);
v___x_688_ = v___x_683_;
goto v_reusejp_687_;
}
else
{
lean_object* v_reuseFailAlloc_689_; 
v_reuseFailAlloc_689_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_689_, 0, v___x_686_);
v___x_688_ = v_reuseFailAlloc_689_;
goto v_reusejp_687_;
}
v_reusejp_687_:
{
return v___x_688_;
}
}
}
}
else
{
lean_object* v_a_692_; lean_object* v___x_694_; uint8_t v_isShared_695_; uint8_t v_isSharedCheck_699_; 
lean_del_object(v___x_676_);
v_a_692_ = lean_ctor_get(v___x_680_, 0);
v_isSharedCheck_699_ = !lean_is_exclusive(v___x_680_);
if (v_isSharedCheck_699_ == 0)
{
v___x_694_ = v___x_680_;
v_isShared_695_ = v_isSharedCheck_699_;
goto v_resetjp_693_;
}
else
{
lean_inc(v_a_692_);
lean_dec(v___x_680_);
v___x_694_ = lean_box(0);
v_isShared_695_ = v_isSharedCheck_699_;
goto v_resetjp_693_;
}
v_resetjp_693_:
{
lean_object* v___x_697_; 
if (v_isShared_695_ == 0)
{
v___x_697_ = v___x_694_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v_a_692_);
v___x_697_ = v_reuseFailAlloc_698_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
return v___x_697_;
}
}
}
}
}
else
{
lean_object* v_vs_701_; lean_object* v___x_703_; uint8_t v_isShared_704_; uint8_t v_isSharedCheck_727_; 
v_vs_701_ = lean_ctor_get(v_x_666_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v_x_666_);
if (v_isSharedCheck_727_ == 0)
{
v___x_703_ = v_x_666_;
v_isShared_704_ = v_isSharedCheck_727_;
goto v_resetjp_702_;
}
else
{
lean_inc(v_vs_701_);
lean_dec(v_x_666_);
v___x_703_ = lean_box(0);
v_isShared_704_ = v_isSharedCheck_727_;
goto v_resetjp_702_;
}
v_resetjp_702_:
{
size_t v_sz_705_; size_t v___x_706_; lean_object* v___x_707_; 
v_sz_705_ = lean_array_size(v_vs_701_);
v___x_706_ = ((size_t)0ULL);
v___x_707_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10(v___x_664_, v_ctx_x3f_665_, v_sz_705_, v___x_706_, v_vs_701_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_);
if (lean_obj_tag(v___x_707_) == 0)
{
lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_718_; 
v_a_708_ = lean_ctor_get(v___x_707_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_718_ == 0)
{
v___x_710_ = v___x_707_;
v_isShared_711_ = v_isSharedCheck_718_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_dec(v___x_707_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_718_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_713_; 
if (v_isShared_704_ == 0)
{
lean_ctor_set(v___x_703_, 0, v_a_708_);
v___x_713_ = v___x_703_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_a_708_);
v___x_713_ = v_reuseFailAlloc_717_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
lean_object* v___x_715_; 
if (v_isShared_711_ == 0)
{
lean_ctor_set(v___x_710_, 0, v___x_713_);
v___x_715_ = v___x_710_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v___x_713_);
v___x_715_ = v_reuseFailAlloc_716_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
return v___x_715_;
}
}
}
}
else
{
lean_object* v_a_719_; lean_object* v___x_721_; uint8_t v_isShared_722_; uint8_t v_isSharedCheck_726_; 
lean_del_object(v___x_703_);
v_a_719_ = lean_ctor_get(v___x_707_, 0);
v_isSharedCheck_726_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_726_ == 0)
{
v___x_721_ = v___x_707_;
v_isShared_722_ = v_isSharedCheck_726_;
goto v_resetjp_720_;
}
else
{
lean_inc(v_a_719_);
lean_dec(v___x_707_);
v___x_721_ = lean_box(0);
v_isShared_722_ = v_isSharedCheck_726_;
goto v_resetjp_720_;
}
v_resetjp_720_:
{
lean_object* v___x_724_; 
if (v_isShared_722_ == 0)
{
v___x_724_ = v___x_721_;
goto v_reusejp_723_;
}
else
{
lean_object* v_reuseFailAlloc_725_; 
v_reuseFailAlloc_725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_725_, 0, v_a_719_);
v___x_724_ = v_reuseFailAlloc_725_;
goto v_reusejp_723_;
}
v_reusejp_723_:
{
return v___x_724_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9_spec__10(lean_object* v___x_728_, lean_object* v_ctx_x3f_729_, size_t v_sz_730_, size_t v_i_731_, lean_object* v_bs_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
uint8_t v___x_740_; 
v___x_740_ = lean_usize_dec_lt(v_i_731_, v_sz_730_);
if (v___x_740_ == 0)
{
lean_object* v___x_741_; 
lean_dec_ref(v_ctx_x3f_729_);
v___x_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_741_, 0, v_bs_732_);
return v___x_741_;
}
else
{
lean_object* v_v_742_; lean_object* v___x_743_; 
v_v_742_ = lean_array_uget_borrowed(v_bs_732_, v_i_731_);
lean_inc(v_v_742_);
lean_inc_ref(v_ctx_x3f_729_);
v___x_743_ = lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9(v___x_728_, v_ctx_x3f_729_, v_v_742_, v___y_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_743_) == 0)
{
lean_object* v_a_744_; lean_object* v___x_745_; lean_object* v_bs_x27_746_; size_t v___x_747_; size_t v___x_748_; lean_object* v___x_749_; 
v_a_744_ = lean_ctor_get(v___x_743_, 0);
lean_inc(v_a_744_);
lean_dec_ref_known(v___x_743_, 1);
v___x_745_ = lean_unsigned_to_nat(0u);
v_bs_x27_746_ = lean_array_uset(v_bs_732_, v_i_731_, v___x_745_);
v___x_747_ = ((size_t)1ULL);
v___x_748_ = lean_usize_add(v_i_731_, v___x_747_);
v___x_749_ = lean_array_uset(v_bs_x27_746_, v_i_731_, v_a_744_);
v_i_731_ = v___x_748_;
v_bs_732_ = v___x_749_;
goto _start;
}
else
{
lean_object* v_a_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_758_; 
lean_dec_ref(v_bs_732_);
lean_dec_ref(v_ctx_x3f_729_);
v_a_751_ = lean_ctor_get(v___x_743_, 0);
v_isSharedCheck_758_ = !lean_is_exclusive(v___x_743_);
if (v_isSharedCheck_758_ == 0)
{
v___x_753_ = v___x_743_;
v_isShared_754_ = v_isSharedCheck_758_;
goto v_resetjp_752_;
}
else
{
lean_inc(v_a_751_);
lean_dec(v___x_743_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_758_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
lean_object* v___x_756_; 
if (v_isShared_754_ == 0)
{
v___x_756_ = v___x_753_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v_a_751_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9_spec__10___boxed(lean_object* v___x_759_, lean_object* v_ctx_x3f_760_, lean_object* v_sz_761_, lean_object* v_i_762_, lean_object* v_bs_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_){
_start:
{
size_t v_sz_boxed_771_; size_t v_i_boxed_772_; lean_object* v_res_773_; 
v_sz_boxed_771_ = lean_unbox_usize(v_sz_761_);
lean_dec(v_sz_761_);
v_i_boxed_772_ = lean_unbox_usize(v_i_762_);
lean_dec(v_i_762_);
v_res_773_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9_spec__10(v___x_759_, v_ctx_x3f_760_, v_sz_boxed_771_, v_i_boxed_772_, v_bs_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_, v___y_769_);
lean_dec(v___y_769_);
lean_dec_ref(v___y_768_);
lean_dec(v___y_767_);
lean_dec_ref(v___y_766_);
lean_dec(v___y_765_);
lean_dec_ref(v___y_764_);
lean_dec_ref(v___x_759_);
return v_res_773_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9___boxed(lean_object* v___x_774_, lean_object* v_ctx_x3f_775_, lean_object* v_x_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9(v___x_774_, v_ctx_x3f_775_, v_x_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
lean_dec(v___y_782_);
lean_dec_ref(v___y_781_);
lean_dec(v___y_780_);
lean_dec_ref(v___y_779_);
lean_dec(v___y_778_);
lean_dec_ref(v___y_777_);
lean_dec_ref(v___x_774_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8(lean_object* v___x_785_, lean_object* v_ctx_x3f_786_, lean_object* v_t_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v_root_795_; lean_object* v_tail_796_; lean_object* v_size_797_; size_t v_shift_798_; lean_object* v_tailOff_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_835_; 
v_root_795_ = lean_ctor_get(v_t_787_, 0);
v_tail_796_ = lean_ctor_get(v_t_787_, 1);
v_size_797_ = lean_ctor_get(v_t_787_, 2);
v_shift_798_ = lean_ctor_get_usize(v_t_787_, 4);
v_tailOff_799_ = lean_ctor_get(v_t_787_, 3);
v_isSharedCheck_835_ = !lean_is_exclusive(v_t_787_);
if (v_isSharedCheck_835_ == 0)
{
v___x_801_ = v_t_787_;
v_isShared_802_ = v_isSharedCheck_835_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_tailOff_799_);
lean_inc(v_size_797_);
lean_inc(v_tail_796_);
lean_inc(v_root_795_);
lean_dec(v_t_787_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_835_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v___x_803_; 
lean_inc_ref(v_ctx_x3f_786_);
v___x_803_ = lp_aesop_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__9(v___x_785_, v_ctx_x3f_786_, v_root_795_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_);
if (lean_obj_tag(v___x_803_) == 0)
{
lean_object* v_a_804_; size_t v_sz_805_; size_t v___x_806_; lean_object* v___x_807_; 
v_a_804_ = lean_ctor_get(v___x_803_, 0);
lean_inc(v_a_804_);
lean_dec_ref_known(v___x_803_, 1);
v_sz_805_ = lean_array_size(v_tail_796_);
v___x_806_ = ((size_t)0ULL);
v___x_807_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8_spec__10(v___x_785_, v_ctx_x3f_786_, v_sz_805_, v___x_806_, v_tail_796_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_);
if (lean_obj_tag(v___x_807_) == 0)
{
lean_object* v_a_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_818_; 
v_a_808_ = lean_ctor_get(v___x_807_, 0);
v_isSharedCheck_818_ = !lean_is_exclusive(v___x_807_);
if (v_isSharedCheck_818_ == 0)
{
v___x_810_ = v___x_807_;
v_isShared_811_ = v_isSharedCheck_818_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_a_808_);
lean_dec(v___x_807_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_818_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___x_813_; 
if (v_isShared_802_ == 0)
{
lean_ctor_set(v___x_801_, 1, v_a_808_);
lean_ctor_set(v___x_801_, 0, v_a_804_);
v___x_813_ = v___x_801_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_817_; 
v_reuseFailAlloc_817_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v_reuseFailAlloc_817_, 0, v_a_804_);
lean_ctor_set(v_reuseFailAlloc_817_, 1, v_a_808_);
lean_ctor_set(v_reuseFailAlloc_817_, 2, v_size_797_);
lean_ctor_set(v_reuseFailAlloc_817_, 3, v_tailOff_799_);
lean_ctor_set_usize(v_reuseFailAlloc_817_, 4, v_shift_798_);
v___x_813_ = v_reuseFailAlloc_817_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
lean_object* v___x_815_; 
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 0, v___x_813_);
v___x_815_ = v___x_810_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_816_; 
v_reuseFailAlloc_816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_816_, 0, v___x_813_);
v___x_815_ = v_reuseFailAlloc_816_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
return v___x_815_;
}
}
}
}
else
{
lean_object* v_a_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_826_; 
lean_dec(v_a_804_);
lean_del_object(v___x_801_);
lean_dec(v_tailOff_799_);
lean_dec(v_size_797_);
v_a_819_ = lean_ctor_get(v___x_807_, 0);
v_isSharedCheck_826_ = !lean_is_exclusive(v___x_807_);
if (v_isSharedCheck_826_ == 0)
{
v___x_821_ = v___x_807_;
v_isShared_822_ = v_isSharedCheck_826_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_a_819_);
lean_dec(v___x_807_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_826_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v___x_824_; 
if (v_isShared_822_ == 0)
{
v___x_824_ = v___x_821_;
goto v_reusejp_823_;
}
else
{
lean_object* v_reuseFailAlloc_825_; 
v_reuseFailAlloc_825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_825_, 0, v_a_819_);
v___x_824_ = v_reuseFailAlloc_825_;
goto v_reusejp_823_;
}
v_reusejp_823_:
{
return v___x_824_;
}
}
}
}
else
{
lean_object* v_a_827_; lean_object* v___x_829_; uint8_t v_isShared_830_; uint8_t v_isSharedCheck_834_; 
lean_del_object(v___x_801_);
lean_dec(v_tailOff_799_);
lean_dec(v_size_797_);
lean_dec_ref(v_tail_796_);
lean_dec_ref(v_ctx_x3f_786_);
v_a_827_ = lean_ctor_get(v___x_803_, 0);
v_isSharedCheck_834_ = !lean_is_exclusive(v___x_803_);
if (v_isSharedCheck_834_ == 0)
{
v___x_829_ = v___x_803_;
v_isShared_830_ = v_isSharedCheck_834_;
goto v_resetjp_828_;
}
else
{
lean_inc(v_a_827_);
lean_dec(v___x_803_);
v___x_829_ = lean_box(0);
v_isShared_830_ = v_isSharedCheck_834_;
goto v_resetjp_828_;
}
v_resetjp_828_:
{
lean_object* v___x_832_; 
if (v_isShared_830_ == 0)
{
v___x_832_ = v___x_829_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v_a_827_);
v___x_832_ = v_reuseFailAlloc_833_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
return v___x_832_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8___boxed(lean_object* v___x_836_, lean_object* v_ctx_x3f_837_, lean_object* v_t_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_){
_start:
{
lean_object* v_res_846_; 
v_res_846_ = lp_aesop_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8(v___x_836_, v_ctx_x3f_837_, v_t_838_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_, v___y_844_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
lean_dec(v___y_842_);
lean_dec_ref(v___y_841_);
lean_dec(v___y_840_);
lean_dec_ref(v___y_839_);
lean_dec_ref(v___x_836_);
return v_res_846_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0(lean_object* v___y_847_, lean_object* v_ctx_x3f_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v_a_854_, lean_object* v_a_x3f_855_){
_start:
{
lean_object* v___x_857_; lean_object* v_infoState_858_; lean_object* v_trees_859_; lean_object* v___x_860_; 
v___x_857_ = lean_st_ref_get(v___y_847_);
v_infoState_858_ = lean_ctor_get(v___x_857_, 7);
lean_inc_ref(v_infoState_858_);
lean_dec(v___x_857_);
v_trees_859_ = lean_ctor_get(v_infoState_858_, 2);
lean_inc_ref(v_trees_859_);
v___x_860_ = lp_aesop_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__8(v_infoState_858_, v_ctx_x3f_848_, v_trees_859_, v___y_849_, v___y_850_, v___y_851_, v___y_852_, v___y_853_, v___y_847_);
lean_dec_ref(v_infoState_858_);
if (lean_obj_tag(v___x_860_) == 0)
{
lean_object* v_a_861_; lean_object* v___x_863_; uint8_t v_isShared_864_; uint8_t v_isSharedCheck_899_; 
v_a_861_ = lean_ctor_get(v___x_860_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_899_ == 0)
{
v___x_863_ = v___x_860_;
v_isShared_864_ = v_isSharedCheck_899_;
goto v_resetjp_862_;
}
else
{
lean_inc(v_a_861_);
lean_dec(v___x_860_);
v___x_863_ = lean_box(0);
v_isShared_864_ = v_isSharedCheck_899_;
goto v_resetjp_862_;
}
v_resetjp_862_:
{
lean_object* v___x_865_; lean_object* v_infoState_866_; lean_object* v_env_867_; lean_object* v_nextMacroScope_868_; lean_object* v_ngen_869_; lean_object* v_auxDeclNGen_870_; lean_object* v_traceState_871_; lean_object* v_cache_872_; lean_object* v_messages_873_; lean_object* v_snapshotTasks_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_898_; 
v___x_865_ = lean_st_ref_take(v___y_847_);
v_infoState_866_ = lean_ctor_get(v___x_865_, 7);
v_env_867_ = lean_ctor_get(v___x_865_, 0);
v_nextMacroScope_868_ = lean_ctor_get(v___x_865_, 1);
v_ngen_869_ = lean_ctor_get(v___x_865_, 2);
v_auxDeclNGen_870_ = lean_ctor_get(v___x_865_, 3);
v_traceState_871_ = lean_ctor_get(v___x_865_, 4);
v_cache_872_ = lean_ctor_get(v___x_865_, 5);
v_messages_873_ = lean_ctor_get(v___x_865_, 6);
v_snapshotTasks_874_ = lean_ctor_get(v___x_865_, 8);
v_isSharedCheck_898_ = !lean_is_exclusive(v___x_865_);
if (v_isSharedCheck_898_ == 0)
{
v___x_876_ = v___x_865_;
v_isShared_877_ = v_isSharedCheck_898_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_snapshotTasks_874_);
lean_inc(v_infoState_866_);
lean_inc(v_messages_873_);
lean_inc(v_cache_872_);
lean_inc(v_traceState_871_);
lean_inc(v_auxDeclNGen_870_);
lean_inc(v_ngen_869_);
lean_inc(v_nextMacroScope_868_);
lean_inc(v_env_867_);
lean_dec(v___x_865_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_898_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
uint8_t v_enabled_878_; lean_object* v_assignment_879_; lean_object* v_lazyAssignment_880_; lean_object* v___x_882_; uint8_t v_isShared_883_; uint8_t v_isSharedCheck_896_; 
v_enabled_878_ = lean_ctor_get_uint8(v_infoState_866_, sizeof(void*)*3);
v_assignment_879_ = lean_ctor_get(v_infoState_866_, 0);
v_lazyAssignment_880_ = lean_ctor_get(v_infoState_866_, 1);
v_isSharedCheck_896_ = !lean_is_exclusive(v_infoState_866_);
if (v_isSharedCheck_896_ == 0)
{
lean_object* v_unused_897_; 
v_unused_897_ = lean_ctor_get(v_infoState_866_, 2);
lean_dec(v_unused_897_);
v___x_882_ = v_infoState_866_;
v_isShared_883_ = v_isSharedCheck_896_;
goto v_resetjp_881_;
}
else
{
lean_inc(v_lazyAssignment_880_);
lean_inc(v_assignment_879_);
lean_dec(v_infoState_866_);
v___x_882_ = lean_box(0);
v_isShared_883_ = v_isSharedCheck_896_;
goto v_resetjp_881_;
}
v_resetjp_881_:
{
lean_object* v___x_884_; lean_object* v___x_886_; 
v___x_884_ = l_Lean_PersistentArray_append___redArg(v_a_854_, v_a_861_);
lean_dec(v_a_861_);
if (v_isShared_883_ == 0)
{
lean_ctor_set(v___x_882_, 2, v___x_884_);
v___x_886_ = v___x_882_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v_assignment_879_);
lean_ctor_set(v_reuseFailAlloc_895_, 1, v_lazyAssignment_880_);
lean_ctor_set(v_reuseFailAlloc_895_, 2, v___x_884_);
lean_ctor_set_uint8(v_reuseFailAlloc_895_, sizeof(void*)*3, v_enabled_878_);
v___x_886_ = v_reuseFailAlloc_895_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
lean_object* v___x_888_; 
if (v_isShared_877_ == 0)
{
lean_ctor_set(v___x_876_, 7, v___x_886_);
v___x_888_ = v___x_876_;
goto v_reusejp_887_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v_env_867_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v_nextMacroScope_868_);
lean_ctor_set(v_reuseFailAlloc_894_, 2, v_ngen_869_);
lean_ctor_set(v_reuseFailAlloc_894_, 3, v_auxDeclNGen_870_);
lean_ctor_set(v_reuseFailAlloc_894_, 4, v_traceState_871_);
lean_ctor_set(v_reuseFailAlloc_894_, 5, v_cache_872_);
lean_ctor_set(v_reuseFailAlloc_894_, 6, v_messages_873_);
lean_ctor_set(v_reuseFailAlloc_894_, 7, v___x_886_);
lean_ctor_set(v_reuseFailAlloc_894_, 8, v_snapshotTasks_874_);
v___x_888_ = v_reuseFailAlloc_894_;
goto v_reusejp_887_;
}
v_reusejp_887_:
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_892_; 
v___x_889_ = lean_st_ref_set(v___y_847_, v___x_888_);
v___x_890_ = lean_box(0);
if (v_isShared_864_ == 0)
{
lean_ctor_set(v___x_863_, 0, v___x_890_);
v___x_892_ = v___x_863_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v___x_890_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_907_; 
lean_dec_ref(v_a_854_);
v_a_900_ = lean_ctor_get(v___x_860_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_907_ == 0)
{
v___x_902_ = v___x_860_;
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_860_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_a_900_);
v___x_905_ = v_reuseFailAlloc_906_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
return v___x_905_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v___y_908_, lean_object* v_ctx_x3f_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v_a_915_, lean_object* v_a_x3f_916_, lean_object* v___y_917_){
_start:
{
lean_object* v_res_918_; 
v_res_918_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0(v___y_908_, v_ctx_x3f_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_, v_a_915_, v_a_x3f_916_);
lean_dec(v_a_x3f_916_);
lean_dec_ref(v___y_914_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
lean_dec(v___y_908_);
return v_res_918_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_919_ = lean_unsigned_to_nat(32u);
v___x_920_ = lean_mk_empty_array_with_capacity(v___x_919_);
v___x_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_921_, 0, v___x_920_);
return v___x_921_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__1(void){
_start:
{
size_t v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_922_ = ((size_t)5ULL);
v___x_923_ = lean_unsigned_to_nat(0u);
v___x_924_ = lean_unsigned_to_nat(32u);
v___x_925_ = lean_mk_empty_array_with_capacity(v___x_924_);
v___x_926_ = lean_obj_once(&lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__0, &lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__0);
v___x_927_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_927_, 0, v___x_926_);
lean_ctor_set(v___x_927_, 1, v___x_925_);
lean_ctor_set(v___x_927_, 2, v___x_923_);
lean_ctor_set(v___x_927_, 3, v___x_923_);
lean_ctor_set_usize(v___x_927_, 4, v___x_922_);
return v___x_927_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg(lean_object* v___y_928_){
_start:
{
lean_object* v___x_930_; lean_object* v_infoState_931_; lean_object* v_trees_932_; lean_object* v___x_933_; lean_object* v_infoState_934_; lean_object* v_env_935_; lean_object* v_nextMacroScope_936_; lean_object* v_ngen_937_; lean_object* v_auxDeclNGen_938_; lean_object* v_traceState_939_; lean_object* v_cache_940_; lean_object* v_messages_941_; lean_object* v_snapshotTasks_942_; lean_object* v___x_944_; uint8_t v_isShared_945_; uint8_t v_isSharedCheck_963_; 
v___x_930_ = lean_st_ref_get(v___y_928_);
v_infoState_931_ = lean_ctor_get(v___x_930_, 7);
lean_inc_ref(v_infoState_931_);
lean_dec(v___x_930_);
v_trees_932_ = lean_ctor_get(v_infoState_931_, 2);
lean_inc_ref(v_trees_932_);
lean_dec_ref(v_infoState_931_);
v___x_933_ = lean_st_ref_take(v___y_928_);
v_infoState_934_ = lean_ctor_get(v___x_933_, 7);
v_env_935_ = lean_ctor_get(v___x_933_, 0);
v_nextMacroScope_936_ = lean_ctor_get(v___x_933_, 1);
v_ngen_937_ = lean_ctor_get(v___x_933_, 2);
v_auxDeclNGen_938_ = lean_ctor_get(v___x_933_, 3);
v_traceState_939_ = lean_ctor_get(v___x_933_, 4);
v_cache_940_ = lean_ctor_get(v___x_933_, 5);
v_messages_941_ = lean_ctor_get(v___x_933_, 6);
v_snapshotTasks_942_ = lean_ctor_get(v___x_933_, 8);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_933_);
if (v_isSharedCheck_963_ == 0)
{
v___x_944_ = v___x_933_;
v_isShared_945_ = v_isSharedCheck_963_;
goto v_resetjp_943_;
}
else
{
lean_inc(v_snapshotTasks_942_);
lean_inc(v_infoState_934_);
lean_inc(v_messages_941_);
lean_inc(v_cache_940_);
lean_inc(v_traceState_939_);
lean_inc(v_auxDeclNGen_938_);
lean_inc(v_ngen_937_);
lean_inc(v_nextMacroScope_936_);
lean_inc(v_env_935_);
lean_dec(v___x_933_);
v___x_944_ = lean_box(0);
v_isShared_945_ = v_isSharedCheck_963_;
goto v_resetjp_943_;
}
v_resetjp_943_:
{
uint8_t v_enabled_946_; lean_object* v_assignment_947_; lean_object* v_lazyAssignment_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_961_; 
v_enabled_946_ = lean_ctor_get_uint8(v_infoState_934_, sizeof(void*)*3);
v_assignment_947_ = lean_ctor_get(v_infoState_934_, 0);
v_lazyAssignment_948_ = lean_ctor_get(v_infoState_934_, 1);
v_isSharedCheck_961_ = !lean_is_exclusive(v_infoState_934_);
if (v_isSharedCheck_961_ == 0)
{
lean_object* v_unused_962_; 
v_unused_962_ = lean_ctor_get(v_infoState_934_, 2);
lean_dec(v_unused_962_);
v___x_950_ = v_infoState_934_;
v_isShared_951_ = v_isSharedCheck_961_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_lazyAssignment_948_);
lean_inc(v_assignment_947_);
lean_dec(v_infoState_934_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_961_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_952_; lean_object* v___x_954_; 
v___x_952_ = lean_obj_once(&lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__1, &lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__1_once, _init_lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___closed__1);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 2, v___x_952_);
v___x_954_ = v___x_950_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_assignment_947_);
lean_ctor_set(v_reuseFailAlloc_960_, 1, v_lazyAssignment_948_);
lean_ctor_set(v_reuseFailAlloc_960_, 2, v___x_952_);
lean_ctor_set_uint8(v_reuseFailAlloc_960_, sizeof(void*)*3, v_enabled_946_);
v___x_954_ = v_reuseFailAlloc_960_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
lean_object* v___x_956_; 
if (v_isShared_945_ == 0)
{
lean_ctor_set(v___x_944_, 7, v___x_954_);
v___x_956_ = v___x_944_;
goto v_reusejp_955_;
}
else
{
lean_object* v_reuseFailAlloc_959_; 
v_reuseFailAlloc_959_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_959_, 0, v_env_935_);
lean_ctor_set(v_reuseFailAlloc_959_, 1, v_nextMacroScope_936_);
lean_ctor_set(v_reuseFailAlloc_959_, 2, v_ngen_937_);
lean_ctor_set(v_reuseFailAlloc_959_, 3, v_auxDeclNGen_938_);
lean_ctor_set(v_reuseFailAlloc_959_, 4, v_traceState_939_);
lean_ctor_set(v_reuseFailAlloc_959_, 5, v_cache_940_);
lean_ctor_set(v_reuseFailAlloc_959_, 6, v_messages_941_);
lean_ctor_set(v_reuseFailAlloc_959_, 7, v___x_954_);
lean_ctor_set(v_reuseFailAlloc_959_, 8, v_snapshotTasks_942_);
v___x_956_ = v_reuseFailAlloc_959_;
goto v_reusejp_955_;
}
v_reusejp_955_:
{
lean_object* v___x_957_; lean_object* v___x_958_; 
v___x_957_ = lean_st_ref_set(v___y_928_, v___x_956_);
v___x_958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_958_, 0, v_trees_932_);
return v___x_958_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg___boxed(lean_object* v___y_964_, lean_object* v___y_965_){
_start:
{
lean_object* v_res_966_; 
v_res_966_ = lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg(v___y_964_);
lean_dec(v___y_964_);
return v_res_966_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg(lean_object* v_x_967_, lean_object* v_ctx_x3f_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_){
_start:
{
lean_object* v___x_976_; lean_object* v_infoState_977_; uint8_t v_enabled_978_; 
v___x_976_ = lean_st_ref_get(v___y_974_);
v_infoState_977_ = lean_ctor_get(v___x_976_, 7);
lean_inc_ref(v_infoState_977_);
lean_dec(v___x_976_);
v_enabled_978_ = lean_ctor_get_uint8(v_infoState_977_, sizeof(void*)*3);
lean_dec_ref(v_infoState_977_);
if (v_enabled_978_ == 0)
{
lean_object* v___x_979_; 
lean_dec_ref(v_ctx_x3f_968_);
lean_inc(v___y_974_);
lean_inc_ref(v___y_973_);
lean_inc(v___y_972_);
lean_inc_ref(v___y_971_);
lean_inc(v___y_970_);
lean_inc_ref(v___y_969_);
v___x_979_ = lean_apply_7(v_x_967_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, lean_box(0));
return v___x_979_;
}
else
{
lean_object* v___x_980_; lean_object* v_a_981_; lean_object* v_r_982_; 
v___x_980_ = lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg(v___y_974_);
v_a_981_ = lean_ctor_get(v___x_980_, 0);
lean_inc(v_a_981_);
lean_dec_ref(v___x_980_);
lean_inc(v___y_974_);
lean_inc_ref(v___y_973_);
lean_inc(v___y_972_);
lean_inc_ref(v___y_971_);
lean_inc(v___y_970_);
lean_inc_ref(v___y_969_);
v_r_982_ = lean_apply_7(v_x_967_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, lean_box(0));
if (lean_obj_tag(v_r_982_) == 0)
{
lean_object* v_a_983_; lean_object* v___x_985_; uint8_t v_isShared_986_; uint8_t v_isSharedCheck_1007_; 
v_a_983_ = lean_ctor_get(v_r_982_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v_r_982_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_985_ = v_r_982_;
v_isShared_986_ = v_isSharedCheck_1007_;
goto v_resetjp_984_;
}
else
{
lean_inc(v_a_983_);
lean_dec(v_r_982_);
v___x_985_ = lean_box(0);
v_isShared_986_ = v_isSharedCheck_1007_;
goto v_resetjp_984_;
}
v_resetjp_984_:
{
lean_object* v___x_988_; 
lean_inc(v_a_983_);
if (v_isShared_986_ == 0)
{
lean_ctor_set_tag(v___x_985_, 1);
v___x_988_ = v___x_985_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v_a_983_);
v___x_988_ = v_reuseFailAlloc_1006_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
lean_object* v___x_989_; 
v___x_989_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0(v___y_974_, v_ctx_x3f_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v_a_981_, v___x_988_);
lean_dec_ref(v___x_988_);
if (lean_obj_tag(v___x_989_) == 0)
{
lean_object* v___x_991_; uint8_t v_isShared_992_; uint8_t v_isSharedCheck_996_; 
v_isSharedCheck_996_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_996_ == 0)
{
lean_object* v_unused_997_; 
v_unused_997_ = lean_ctor_get(v___x_989_, 0);
lean_dec(v_unused_997_);
v___x_991_ = v___x_989_;
v_isShared_992_ = v_isSharedCheck_996_;
goto v_resetjp_990_;
}
else
{
lean_dec(v___x_989_);
v___x_991_ = lean_box(0);
v_isShared_992_ = v_isSharedCheck_996_;
goto v_resetjp_990_;
}
v_resetjp_990_:
{
lean_object* v___x_994_; 
if (v_isShared_992_ == 0)
{
lean_ctor_set(v___x_991_, 0, v_a_983_);
v___x_994_ = v___x_991_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_995_; 
v_reuseFailAlloc_995_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_995_, 0, v_a_983_);
v___x_994_ = v_reuseFailAlloc_995_;
goto v_reusejp_993_;
}
v_reusejp_993_:
{
return v___x_994_;
}
}
}
else
{
lean_object* v_a_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1005_; 
lean_dec(v_a_983_);
v_a_998_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1005_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1005_ == 0)
{
v___x_1000_ = v___x_989_;
v_isShared_1001_ = v_isSharedCheck_1005_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_a_998_);
lean_dec(v___x_989_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1005_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1003_; 
if (v_isShared_1001_ == 0)
{
v___x_1003_ = v___x_1000_;
goto v_reusejp_1002_;
}
else
{
lean_object* v_reuseFailAlloc_1004_; 
v_reuseFailAlloc_1004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1004_, 0, v_a_998_);
v___x_1003_ = v_reuseFailAlloc_1004_;
goto v_reusejp_1002_;
}
v_reusejp_1002_:
{
return v___x_1003_;
}
}
}
}
}
}
else
{
lean_object* v_a_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v_a_1008_ = lean_ctor_get(v_r_982_, 0);
lean_inc(v_a_1008_);
lean_dec_ref_known(v_r_982_, 1);
v___x_1009_ = lean_box(0);
v___x_1010_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___lam__0(v___y_974_, v_ctx_x3f_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v_a_981_, v___x_1009_);
if (lean_obj_tag(v___x_1010_) == 0)
{
lean_object* v___x_1012_; uint8_t v_isShared_1013_; uint8_t v_isSharedCheck_1017_; 
v_isSharedCheck_1017_ = !lean_is_exclusive(v___x_1010_);
if (v_isSharedCheck_1017_ == 0)
{
lean_object* v_unused_1018_; 
v_unused_1018_ = lean_ctor_get(v___x_1010_, 0);
lean_dec(v_unused_1018_);
v___x_1012_ = v___x_1010_;
v_isShared_1013_ = v_isSharedCheck_1017_;
goto v_resetjp_1011_;
}
else
{
lean_dec(v___x_1010_);
v___x_1012_ = lean_box(0);
v_isShared_1013_ = v_isSharedCheck_1017_;
goto v_resetjp_1011_;
}
v_resetjp_1011_:
{
lean_object* v___x_1015_; 
if (v_isShared_1013_ == 0)
{
lean_ctor_set_tag(v___x_1012_, 1);
lean_ctor_set(v___x_1012_, 0, v_a_1008_);
v___x_1015_ = v___x_1012_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v_a_1008_);
v___x_1015_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
return v___x_1015_;
}
}
}
else
{
lean_object* v_a_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1026_; 
lean_dec(v_a_1008_);
v_a_1019_ = lean_ctor_get(v___x_1010_, 0);
v_isSharedCheck_1026_ = !lean_is_exclusive(v___x_1010_);
if (v_isSharedCheck_1026_ == 0)
{
v___x_1021_ = v___x_1010_;
v_isShared_1022_ = v_isSharedCheck_1026_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_a_1019_);
lean_dec(v___x_1010_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1026_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
lean_object* v___x_1024_; 
if (v_isShared_1022_ == 0)
{
v___x_1024_ = v___x_1021_;
goto v_reusejp_1023_;
}
else
{
lean_object* v_reuseFailAlloc_1025_; 
v_reuseFailAlloc_1025_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1025_, 0, v_a_1019_);
v___x_1024_ = v_reuseFailAlloc_1025_;
goto v_reusejp_1023_;
}
v_reusejp_1023_:
{
return v___x_1024_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg___boxed(lean_object* v_x_1027_, lean_object* v_ctx_x3f_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v_res_1036_; 
v_res_1036_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg(v_x_1027_, v_ctx_x3f_1028_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_);
lean_dec(v___y_1034_);
lean_dec_ref(v___y_1033_);
lean_dec(v___y_1032_);
lean_dec_ref(v___y_1031_);
lean_dec(v___y_1030_);
lean_dec_ref(v___y_1029_);
return v_res_1036_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg(lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_){
_start:
{
lean_object* v___x_1041_; lean_object* v_env_1042_; lean_object* v___x_1043_; lean_object* v_mctx_1044_; lean_object* v_options_1045_; lean_object* v_currNamespace_1046_; lean_object* v_openDecls_1047_; lean_object* v___x_1048_; lean_object* v_ngen_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1041_ = lean_st_ref_get(v___y_1039_);
v_env_1042_ = lean_ctor_get(v___x_1041_, 0);
lean_inc_ref(v_env_1042_);
lean_dec(v___x_1041_);
v___x_1043_ = lean_st_ref_get(v___y_1037_);
v_mctx_1044_ = lean_ctor_get(v___x_1043_, 0);
lean_inc_ref(v_mctx_1044_);
lean_dec(v___x_1043_);
v_options_1045_ = lean_ctor_get(v___y_1038_, 2);
v_currNamespace_1046_ = lean_ctor_get(v___y_1038_, 6);
v_openDecls_1047_ = lean_ctor_get(v___y_1038_, 7);
v___x_1048_ = lean_st_ref_get(v___y_1039_);
v_ngen_1049_ = lean_ctor_get(v___x_1048_, 2);
lean_inc_ref(v_ngen_1049_);
lean_dec(v___x_1048_);
v___x_1050_ = lean_box(0);
v___x_1051_ = l_Lean_instInhabitedFileMap_default;
lean_inc(v_openDecls_1047_);
lean_inc(v_currNamespace_1046_);
lean_inc_ref(v_options_1045_);
v___x_1052_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_1052_, 0, v_env_1042_);
lean_ctor_set(v___x_1052_, 1, v___x_1050_);
lean_ctor_set(v___x_1052_, 2, v___x_1051_);
lean_ctor_set(v___x_1052_, 3, v_mctx_1044_);
lean_ctor_set(v___x_1052_, 4, v_options_1045_);
lean_ctor_set(v___x_1052_, 5, v_currNamespace_1046_);
lean_ctor_set(v___x_1052_, 6, v_openDecls_1047_);
lean_ctor_set(v___x_1052_, 7, v_ngen_1049_);
v___x_1053_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg___boxed(lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_){
_start:
{
lean_object* v_res_1058_; 
v_res_1058_ = lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg(v___y_1054_, v___y_1055_, v___y_1056_);
lean_dec(v___y_1056_);
lean_dec_ref(v___y_1055_);
lean_dec(v___y_1054_);
return v_res_1058_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2(lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_){
_start:
{
lean_object* v___x_1066_; lean_object* v_a_1067_; lean_object* v___x_1069_; uint8_t v_isShared_1070_; uint8_t v_isSharedCheck_1091_; 
v___x_1066_ = lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg(v___y_1062_, v___y_1063_, v___y_1064_);
v_a_1067_ = lean_ctor_get(v___x_1066_, 0);
v_isSharedCheck_1091_ = !lean_is_exclusive(v___x_1066_);
if (v_isSharedCheck_1091_ == 0)
{
v___x_1069_ = v___x_1066_;
v_isShared_1070_ = v_isSharedCheck_1091_;
goto v_resetjp_1068_;
}
else
{
lean_inc(v_a_1067_);
lean_dec(v___x_1066_);
v___x_1069_ = lean_box(0);
v_isShared_1070_ = v_isSharedCheck_1091_;
goto v_resetjp_1068_;
}
v_resetjp_1068_:
{
lean_object* v_fileMap_1071_; lean_object* v_env_1072_; lean_object* v_mctx_1073_; lean_object* v_options_1074_; lean_object* v_currNamespace_1075_; lean_object* v_openDecls_1076_; lean_object* v_ngen_1077_; lean_object* v___x_1079_; uint8_t v_isShared_1080_; uint8_t v_isSharedCheck_1088_; 
v_fileMap_1071_ = lean_ctor_get(v___y_1063_, 1);
v_env_1072_ = lean_ctor_get(v_a_1067_, 0);
v_mctx_1073_ = lean_ctor_get(v_a_1067_, 3);
v_options_1074_ = lean_ctor_get(v_a_1067_, 4);
v_currNamespace_1075_ = lean_ctor_get(v_a_1067_, 5);
v_openDecls_1076_ = lean_ctor_get(v_a_1067_, 6);
v_ngen_1077_ = lean_ctor_get(v_a_1067_, 7);
v_isSharedCheck_1088_ = !lean_is_exclusive(v_a_1067_);
if (v_isSharedCheck_1088_ == 0)
{
lean_object* v_unused_1089_; lean_object* v_unused_1090_; 
v_unused_1089_ = lean_ctor_get(v_a_1067_, 2);
lean_dec(v_unused_1089_);
v_unused_1090_ = lean_ctor_get(v_a_1067_, 1);
lean_dec(v_unused_1090_);
v___x_1079_ = v_a_1067_;
v_isShared_1080_ = v_isSharedCheck_1088_;
goto v_resetjp_1078_;
}
else
{
lean_inc(v_ngen_1077_);
lean_inc(v_openDecls_1076_);
lean_inc(v_currNamespace_1075_);
lean_inc(v_options_1074_);
lean_inc(v_mctx_1073_);
lean_inc(v_env_1072_);
lean_dec(v_a_1067_);
v___x_1079_ = lean_box(0);
v_isShared_1080_ = v_isSharedCheck_1088_;
goto v_resetjp_1078_;
}
v_resetjp_1078_:
{
lean_object* v___x_1081_; lean_object* v___x_1083_; 
v___x_1081_ = lean_box(0);
lean_inc_ref(v_fileMap_1071_);
if (v_isShared_1080_ == 0)
{
lean_ctor_set(v___x_1079_, 2, v_fileMap_1071_);
lean_ctor_set(v___x_1079_, 1, v___x_1081_);
v___x_1083_ = v___x_1079_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1087_; 
v_reuseFailAlloc_1087_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_1087_, 0, v_env_1072_);
lean_ctor_set(v_reuseFailAlloc_1087_, 1, v___x_1081_);
lean_ctor_set(v_reuseFailAlloc_1087_, 2, v_fileMap_1071_);
lean_ctor_set(v_reuseFailAlloc_1087_, 3, v_mctx_1073_);
lean_ctor_set(v_reuseFailAlloc_1087_, 4, v_options_1074_);
lean_ctor_set(v_reuseFailAlloc_1087_, 5, v_currNamespace_1075_);
lean_ctor_set(v_reuseFailAlloc_1087_, 6, v_openDecls_1076_);
lean_ctor_set(v_reuseFailAlloc_1087_, 7, v_ngen_1077_);
v___x_1083_ = v_reuseFailAlloc_1087_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
lean_object* v___x_1085_; 
if (v_isShared_1070_ == 0)
{
lean_ctor_set(v___x_1069_, 0, v___x_1083_);
v___x_1085_ = v___x_1069_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v___x_1083_);
v___x_1085_ = v_reuseFailAlloc_1086_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
return v___x_1085_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2___boxed(lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_){
_start:
{
lean_object* v_res_1099_; 
v_res_1099_ = lp_aesop_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2(v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_, v___y_1097_);
lean_dec(v___y_1097_);
lean_dec_ref(v___y_1096_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
lean_dec(v___y_1093_);
lean_dec_ref(v___y_1092_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___lam__0(lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
lean_object* v___x_1107_; lean_object* v_a_1108_; lean_object* v___x_1110_; uint8_t v_isShared_1111_; uint8_t v_isSharedCheck_1117_; 
v___x_1107_ = lp_aesop_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2(v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_, v___y_1105_);
v_a_1108_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1110_ = v___x_1107_;
v_isShared_1111_ = v_isSharedCheck_1117_;
goto v_resetjp_1109_;
}
else
{
lean_inc(v_a_1108_);
lean_dec(v___x_1107_);
v___x_1110_ = lean_box(0);
v_isShared_1111_ = v_isSharedCheck_1117_;
goto v_resetjp_1109_;
}
v_resetjp_1109_:
{
lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1115_; 
v___x_1112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1112_, 0, v_a_1108_);
v___x_1113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1113_, 0, v___x_1112_);
if (v_isShared_1111_ == 0)
{
lean_ctor_set(v___x_1110_, 0, v___x_1113_);
v___x_1115_ = v___x_1110_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v___x_1113_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___lam__0___boxed(lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_){
_start:
{
lean_object* v_res_1125_; 
v_res_1125_ = lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___lam__0(v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_);
lean_dec(v___y_1123_);
lean_dec_ref(v___y_1122_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
lean_dec(v___y_1119_);
lean_dec_ref(v___y_1118_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg(lean_object* v_x_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_){
_start:
{
lean_object* v___f_1135_; lean_object* v___x_1136_; 
v___f_1135_ = ((lean_object*)(lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___closed__0));
v___x_1136_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg(v_x_1127_, v___f_1135_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg___boxed(lean_object* v_x_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg(v_x_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
lean_dec(v___y_1139_);
lean_dec_ref(v___y_1138_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2(lean_object* v_00_u03b1_1146_, lean_object* v_x_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_){
_start:
{
lean_object* v___x_1155_; 
v___x_1155_ = lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___redArg(v_x_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_);
return v___x_1155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___boxed(lean_object* v_00_u03b1_1156_, lean_object* v_x_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_){
_start:
{
lean_object* v_res_1165_; 
v_res_1165_ = lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2(v_00_u03b1_1156_, v_x_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
lean_dec(v___y_1161_);
lean_dec_ref(v___y_1160_);
lean_dec(v___y_1159_);
lean_dec_ref(v___y_1158_);
return v_res_1165_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__0(void){
_start:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1166_ = lean_unsigned_to_nat(32u);
v___x_1167_ = lean_mk_empty_array_with_capacity(v___x_1166_);
v___x_1168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1168_, 0, v___x_1167_);
return v___x_1168_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__1(void){
_start:
{
size_t v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v___x_1169_ = ((size_t)5ULL);
v___x_1170_ = lean_unsigned_to_nat(0u);
v___x_1171_ = lean_unsigned_to_nat(32u);
v___x_1172_ = lean_mk_empty_array_with_capacity(v___x_1171_);
v___x_1173_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__0, &lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__0_once, _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__0);
v___x_1174_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1174_, 0, v___x_1173_);
lean_ctor_set(v___x_1174_, 1, v___x_1172_);
lean_ctor_set(v___x_1174_, 2, v___x_1170_);
lean_ctor_set(v___x_1174_, 3, v___x_1170_);
lean_ctor_set_usize(v___x_1174_, 4, v___x_1169_);
return v___x_1174_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__2(void){
_start:
{
lean_object* v___x_1175_; 
v___x_1175_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1175_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__3(void){
_start:
{
lean_object* v___x_1176_; lean_object* v___x_1177_; 
v___x_1176_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__2, &lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__2_once, _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__2);
v___x_1177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1177_, 0, v___x_1176_);
return v___x_1177_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__4(void){
_start:
{
lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; 
v___x_1178_ = lean_box(1);
v___x_1179_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__1, &lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__1_once, _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__1);
v___x_1180_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__3, &lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__3_once, _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__3);
v___x_1181_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1181_, 0, v___x_1180_);
lean_ctor_set(v___x_1181_, 1, v___x_1179_);
lean_ctor_set(v___x_1181_, 2, v___x_1178_);
return v___x_1181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(lean_object* v_type_1184_, lean_object* v_stx_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_, lean_object* v_a_1189_, lean_object* v_a_1190_, lean_object* v_a_1191_){
_start:
{
lean_object* v_fileName_1193_; lean_object* v_fileMap_1194_; lean_object* v_options_1195_; lean_object* v_currRecDepth_1196_; lean_object* v_maxRecDepth_1197_; lean_object* v_ref_1198_; lean_object* v_currNamespace_1199_; lean_object* v_openDecls_1200_; lean_object* v_initHeartbeats_1201_; lean_object* v_maxHeartbeats_1202_; lean_object* v_quotContext_1203_; lean_object* v_currMacroScope_1204_; uint8_t v_diag_1205_; lean_object* v_cancelTk_x3f_1206_; uint8_t v_suppressElabErrors_1207_; lean_object* v_inheritedTraceOptions_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; uint8_t v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___f_1217_; lean_object* v___x_1218_; uint8_t v___x_1219_; lean_object* v___x_1220_; lean_object* v___f_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v_ref_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; 
v_fileName_1193_ = lean_ctor_get(v_a_1190_, 0);
v_fileMap_1194_ = lean_ctor_get(v_a_1190_, 1);
v_options_1195_ = lean_ctor_get(v_a_1190_, 2);
v_currRecDepth_1196_ = lean_ctor_get(v_a_1190_, 3);
v_maxRecDepth_1197_ = lean_ctor_get(v_a_1190_, 4);
v_ref_1198_ = lean_ctor_get(v_a_1190_, 5);
v_currNamespace_1199_ = lean_ctor_get(v_a_1190_, 6);
v_openDecls_1200_ = lean_ctor_get(v_a_1190_, 7);
v_initHeartbeats_1201_ = lean_ctor_get(v_a_1190_, 8);
v_maxHeartbeats_1202_ = lean_ctor_get(v_a_1190_, 9);
v_quotContext_1203_ = lean_ctor_get(v_a_1190_, 10);
v_currMacroScope_1204_ = lean_ctor_get(v_a_1190_, 11);
v_diag_1205_ = lean_ctor_get_uint8(v_a_1190_, sizeof(void*)*14);
v_cancelTk_x3f_1206_ = lean_ctor_get(v_a_1190_, 12);
v_suppressElabErrors_1207_ = lean_ctor_get_uint8(v_a_1190_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1208_ = lean_ctor_get(v_a_1190_, 13);
v___x_1209_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__4, &lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__4_once, _init_lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__4);
v___x_1210_ = ((lean_object*)(lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___closed__5));
v___x_1211_ = lean_box(0);
lean_inc(v_type_1184_);
v___x_1212_ = l_Lean_mkConst(v_type_1184_, v___x_1211_);
v___x_1213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1213_, 0, v___x_1212_);
v___x_1214_ = 1;
v___x_1215_ = lean_box(0);
v___x_1216_ = lean_box(v___x_1214_);
lean_inc(v_stx_1185_);
v___f_1217_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__0___boxed), 11, 4);
lean_closure_set(v___f_1217_, 0, v_stx_1185_);
lean_closure_set(v___f_1217_, 1, v___x_1213_);
lean_closure_set(v___f_1217_, 2, v___x_1216_);
lean_closure_set(v___f_1217_, 3, v___x_1215_);
v___x_1218_ = lean_alloc_closure((void*)(lp_aesop_Lean_Elab_Term_withoutErrToSorry___at___00Aesop_Frontend_elabConfigUnsafe_spec__1___boxed), 9, 2);
lean_closure_set(v___x_1218_, 0, lean_box(0));
lean_closure_set(v___x_1218_, 1, v___f_1217_);
v___x_1219_ = 1;
v___x_1220_ = lean_box(v___x_1219_);
v___f_1221_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1221_, 0, v___x_1218_);
lean_closure_set(v___f_1221_, 1, v___x_1220_);
v___x_1222_ = lean_alloc_closure((void*)(lp_aesop_Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2___boxed), 9, 2);
lean_closure_set(v___x_1222_, 0, lean_box(0));
lean_closure_set(v___x_1222_, 1, v___f_1221_);
v___x_1223_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___boxed), 11, 4);
lean_closure_set(v___x_1223_, 0, lean_box(0));
lean_closure_set(v___x_1223_, 1, v___x_1209_);
lean_closure_set(v___x_1223_, 2, v___x_1210_);
lean_closure_set(v___x_1223_, 3, v___x_1222_);
v_ref_1224_ = l_Lean_replaceRef(v_stx_1185_, v_ref_1198_);
lean_dec(v_stx_1185_);
lean_inc_ref(v_inheritedTraceOptions_1208_);
lean_inc(v_cancelTk_x3f_1206_);
lean_inc(v_currMacroScope_1204_);
lean_inc(v_quotContext_1203_);
lean_inc(v_maxHeartbeats_1202_);
lean_inc(v_initHeartbeats_1201_);
lean_inc(v_openDecls_1200_);
lean_inc(v_currNamespace_1199_);
lean_inc(v_maxRecDepth_1197_);
lean_inc(v_currRecDepth_1196_);
lean_inc_ref(v_options_1195_);
lean_inc_ref(v_fileMap_1194_);
lean_inc_ref(v_fileName_1193_);
v___x_1225_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1225_, 0, v_fileName_1193_);
lean_ctor_set(v___x_1225_, 1, v_fileMap_1194_);
lean_ctor_set(v___x_1225_, 2, v_options_1195_);
lean_ctor_set(v___x_1225_, 3, v_currRecDepth_1196_);
lean_ctor_set(v___x_1225_, 4, v_maxRecDepth_1197_);
lean_ctor_set(v___x_1225_, 5, v_ref_1224_);
lean_ctor_set(v___x_1225_, 6, v_currNamespace_1199_);
lean_ctor_set(v___x_1225_, 7, v_openDecls_1200_);
lean_ctor_set(v___x_1225_, 8, v_initHeartbeats_1201_);
lean_ctor_set(v___x_1225_, 9, v_maxHeartbeats_1202_);
lean_ctor_set(v___x_1225_, 10, v_quotContext_1203_);
lean_ctor_set(v___x_1225_, 11, v_currMacroScope_1204_);
lean_ctor_set(v___x_1225_, 12, v_cancelTk_x3f_1206_);
lean_ctor_set(v___x_1225_, 13, v_inheritedTraceOptions_1208_);
lean_ctor_set_uint8(v___x_1225_, sizeof(void*)*14, v_diag_1205_);
lean_ctor_set_uint8(v___x_1225_, sizeof(void*)*14 + 1, v_suppressElabErrors_1207_);
v___x_1226_ = lp_aesop_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00Aesop_Frontend_elabConfigUnsafe_spec__4___redArg(v___x_1223_, v_a_1186_, v_a_1187_, v_a_1188_, v_a_1189_, v___x_1225_, v_a_1191_);
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_object* v_a_1227_; uint8_t v___x_1228_; lean_object* v___x_1229_; 
v_a_1227_ = lean_ctor_get(v___x_1226_, 0);
lean_inc(v_a_1227_);
lean_dec_ref_known(v___x_1226_, 1);
v___x_1228_ = 1;
v___x_1229_ = l_Lean_Meta_evalExpr_x27___redArg(v_type_1184_, v_a_1227_, v___x_1228_, v___x_1214_, v_a_1188_, v_a_1189_, v___x_1225_, v_a_1191_);
lean_dec_ref_known(v___x_1225_, 14);
return v___x_1229_;
}
else
{
lean_object* v_a_1230_; lean_object* v___x_1232_; uint8_t v_isShared_1233_; uint8_t v_isSharedCheck_1237_; 
lean_dec_ref_known(v___x_1225_, 14);
lean_dec(v_type_1184_);
v_a_1230_ = lean_ctor_get(v___x_1226_, 0);
v_isSharedCheck_1237_ = !lean_is_exclusive(v___x_1226_);
if (v_isSharedCheck_1237_ == 0)
{
v___x_1232_ = v___x_1226_;
v_isShared_1233_ = v_isSharedCheck_1237_;
goto v_resetjp_1231_;
}
else
{
lean_inc(v_a_1230_);
lean_dec(v___x_1226_);
v___x_1232_ = lean_box(0);
v_isShared_1233_ = v_isSharedCheck_1237_;
goto v_resetjp_1231_;
}
v_resetjp_1231_:
{
lean_object* v___x_1235_; 
if (v_isShared_1233_ == 0)
{
v___x_1235_ = v___x_1232_;
goto v_reusejp_1234_;
}
else
{
lean_object* v_reuseFailAlloc_1236_; 
v_reuseFailAlloc_1236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1236_, 0, v_a_1230_);
v___x_1235_ = v_reuseFailAlloc_1236_;
goto v_reusejp_1234_;
}
v_reusejp_1234_:
{
return v___x_1235_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg___boxed(lean_object* v_type_1238_, lean_object* v_stx_1239_, lean_object* v_a_1240_, lean_object* v_a_1241_, lean_object* v_a_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_){
_start:
{
lean_object* v_res_1247_; 
v_res_1247_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v_type_1238_, v_stx_1239_, v_a_1240_, v_a_1241_, v_a_1242_, v_a_1243_, v_a_1244_, v_a_1245_);
lean_dec(v_a_1245_);
lean_dec_ref(v_a_1244_);
lean_dec(v_a_1243_);
lean_dec_ref(v_a_1242_);
lean_dec(v_a_1241_);
lean_dec_ref(v_a_1240_);
return v_res_1247_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe(lean_object* v_00_u03b1_1248_, lean_object* v_type_1249_, lean_object* v_stx_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_, lean_object* v_a_1255_, lean_object* v_a_1256_){
_start:
{
lean_object* v___x_1258_; 
v___x_1258_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v_type_1249_, v_stx_1250_, v_a_1251_, v_a_1252_, v_a_1253_, v_a_1254_, v_a_1255_, v_a_1256_);
return v___x_1258_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabConfigUnsafe___boxed(lean_object* v_00_u03b1_1259_, lean_object* v_type_1260_, lean_object* v_stx_1261_, lean_object* v_a_1262_, lean_object* v_a_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_, lean_object* v_a_1266_, lean_object* v_a_1267_, lean_object* v_a_1268_){
_start:
{
lean_object* v_res_1269_; 
v_res_1269_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe(v_00_u03b1_1259_, v_type_1260_, v_stx_1261_, v_a_1262_, v_a_1263_, v_a_1264_, v_a_1265_, v_a_1266_, v_a_1267_);
lean_dec(v_a_1267_);
lean_dec_ref(v_a_1266_);
lean_dec(v_a_1265_);
lean_dec_ref(v_a_1264_);
lean_dec(v_a_1263_);
lean_dec_ref(v_a_1262_);
return v_res_1269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5(lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_){
_start:
{
lean_object* v___x_1277_; 
v___x_1277_ = lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___redArg(v___y_1273_, v___y_1274_, v___y_1275_);
return v___x_1277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5___boxed(lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v_res_1285_; 
v_res_1285_ = lp_aesop_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__2_spec__5(v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v___y_1281_);
lean_dec_ref(v___y_1280_);
lean_dec(v___y_1279_);
lean_dec_ref(v___y_1278_);
return v_res_1285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7(lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_){
_start:
{
lean_object* v___x_1293_; 
v___x_1293_ = lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___redArg(v___y_1291_);
return v___x_1293_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7___boxed(lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v_res_1301_; 
v_res_1301_ = lp_aesop_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3_spec__7(v___y_1294_, v___y_1295_, v___y_1296_, v___y_1297_, v___y_1298_, v___y_1299_);
lean_dec(v___y_1299_);
lean_dec_ref(v___y_1298_);
lean_dec(v___y_1297_);
lean_dec_ref(v___y_1296_);
lean_dec(v___y_1295_);
lean_dec_ref(v___y_1294_);
return v_res_1301_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3(lean_object* v_00_u03b1_1302_, lean_object* v_x_1303_, lean_object* v_ctx_x3f_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_){
_start:
{
lean_object* v___x_1312_; 
v___x_1312_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___redArg(v_x_1303_, v_ctx_x3f_1304_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_, v___y_1310_);
return v___x_1312_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1313_, lean_object* v_x_1314_, lean_object* v_ctx_x3f_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_aesop___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Aesop_Frontend_elabConfigUnsafe_spec__2_spec__3(v_00_u03b1_1313_, v_x_1314_, v_ctx_x3f_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_);
lean_dec(v___y_1321_);
lean_dec_ref(v___y_1320_);
lean_dec(v___y_1319_);
lean_dec_ref(v___y_1318_);
lean_dec(v___y_1317_);
lean_dec_ref(v___y_1316_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1(lean_object* v_stx_1328_, lean_object* v_a_1329_, lean_object* v_a_1330_, lean_object* v_a_1331_, lean_object* v_a_1332_, lean_object* v_a_1333_, lean_object* v_a_1334_){
_start:
{
lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1336_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1));
v___x_1337_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v___x_1336_, v_stx_1328_, v_a_1329_, v_a_1330_, v_a_1331_, v_a_1332_, v_a_1333_, v_a_1334_);
return v___x_1337_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___boxed(lean_object* v_stx_1338_, lean_object* v_a_1339_, lean_object* v_a_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_){
_start:
{
lean_object* v_res_1346_; 
v_res_1346_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1(v_stx_1338_, v_a_1339_, v_a_1340_, v_a_1341_, v_a_1342_, v_a_1343_, v_a_1344_);
lean_dec(v_a_1344_);
lean_dec_ref(v_a_1343_);
lean_dec(v_a_1342_);
lean_dec_ref(v_a_1341_);
lean_dec(v_a_1340_);
lean_dec_ref(v_a_1339_);
return v_res_1346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabOptions(lean_object* v_stx_1347_, lean_object* v_a_1348_, lean_object* v_a_1349_, lean_object* v_a_1350_, lean_object* v_a_1351_, lean_object* v_a_1352_, lean_object* v_a_1353_){
_start:
{
lean_object* v___x_1355_; lean_object* v___x_1356_; 
v___x_1355_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabOptions_unsafe__1___closed__1));
v___x_1356_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v___x_1355_, v_stx_1347_, v_a_1348_, v_a_1349_, v_a_1350_, v_a_1351_, v_a_1352_, v_a_1353_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabOptions___boxed(lean_object* v_stx_1357_, lean_object* v_a_1358_, lean_object* v_a_1359_, lean_object* v_a_1360_, lean_object* v_a_1361_, lean_object* v_a_1362_, lean_object* v_a_1363_, lean_object* v_a_1364_){
_start:
{
lean_object* v_res_1365_; 
v_res_1365_ = lp_aesop_Aesop_Frontend_elabOptions(v_stx_1357_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_, v_a_1362_, v_a_1363_);
lean_dec(v_a_1363_);
lean_dec_ref(v_a_1362_);
lean_dec(v_a_1361_);
lean_dec_ref(v_a_1360_);
lean_dec(v_a_1359_);
lean_dec_ref(v_a_1358_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1(lean_object* v_stx_1374_, lean_object* v_a_1375_, lean_object* v_a_1376_, lean_object* v_a_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_, lean_object* v_a_1380_){
_start:
{
lean_object* v___x_1382_; lean_object* v___x_1383_; 
v___x_1382_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3));
v___x_1383_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v___x_1382_, v_stx_1374_, v_a_1375_, v_a_1376_, v_a_1377_, v_a_1378_, v_a_1379_, v_a_1380_);
return v___x_1383_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___boxed(lean_object* v_stx_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_, lean_object* v_a_1388_, lean_object* v_a_1389_, lean_object* v_a_1390_, lean_object* v_a_1391_){
_start:
{
lean_object* v_res_1392_; 
v_res_1392_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1(v_stx_1384_, v_a_1385_, v_a_1386_, v_a_1387_, v_a_1388_, v_a_1389_, v_a_1390_);
lean_dec(v_a_1390_);
lean_dec_ref(v_a_1389_);
lean_dec(v_a_1388_);
lean_dec_ref(v_a_1387_);
lean_dec(v_a_1386_);
lean_dec_ref(v_a_1385_);
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfig(lean_object* v_stx_1393_, lean_object* v_a_1394_, lean_object* v_a_1395_, lean_object* v_a_1396_, lean_object* v_a_1397_, lean_object* v_a_1398_, lean_object* v_a_1399_){
_start:
{
lean_object* v___x_1401_; lean_object* v___x_1402_; 
v___x_1401_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfig_unsafe__1___closed__3));
v___x_1402_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v___x_1401_, v_stx_1393_, v_a_1394_, v_a_1395_, v_a_1396_, v_a_1397_, v_a_1398_, v_a_1399_);
return v___x_1402_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfig___boxed(lean_object* v_stx_1403_, lean_object* v_a_1404_, lean_object* v_a_1405_, lean_object* v_a_1406_, lean_object* v_a_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_, lean_object* v_a_1410_){
_start:
{
lean_object* v_res_1411_; 
v_res_1411_ = lp_aesop_Aesop_Frontend_elabSimpConfig(v_stx_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, v_a_1408_, v_a_1409_);
lean_dec(v_a_1409_);
lean_dec_ref(v_a_1408_);
lean_dec(v_a_1407_);
lean_dec_ref(v_a_1406_);
lean_dec(v_a_1405_);
lean_dec_ref(v_a_1404_);
return v_res_1411_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1(lean_object* v_stx_1418_, lean_object* v_a_1419_, lean_object* v_a_1420_, lean_object* v_a_1421_, lean_object* v_a_1422_, lean_object* v_a_1423_, lean_object* v_a_1424_){
_start:
{
lean_object* v___x_1426_; lean_object* v___x_1427_; 
v___x_1426_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1));
v___x_1427_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v___x_1426_, v_stx_1418_, v_a_1419_, v_a_1420_, v_a_1421_, v_a_1422_, v_a_1423_, v_a_1424_);
return v___x_1427_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___boxed(lean_object* v_stx_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_, lean_object* v_a_1431_, lean_object* v_a_1432_, lean_object* v_a_1433_, lean_object* v_a_1434_, lean_object* v_a_1435_){
_start:
{
lean_object* v_res_1436_; 
v_res_1436_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1(v_stx_1428_, v_a_1429_, v_a_1430_, v_a_1431_, v_a_1432_, v_a_1433_, v_a_1434_);
lean_dec(v_a_1434_);
lean_dec_ref(v_a_1433_);
lean_dec(v_a_1432_);
lean_dec_ref(v_a_1431_);
lean_dec(v_a_1430_);
lean_dec_ref(v_a_1429_);
return v_res_1436_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfigCtx(lean_object* v_stx_1437_, lean_object* v_a_1438_, lean_object* v_a_1439_, lean_object* v_a_1440_, lean_object* v_a_1441_, lean_object* v_a_1442_, lean_object* v_a_1443_){
_start:
{
lean_object* v___x_1445_; lean_object* v___x_1446_; 
v___x_1445_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_elabSimpConfigCtx_unsafe__1___closed__1));
v___x_1446_ = lp_aesop_Aesop_Frontend_elabConfigUnsafe___redArg(v___x_1445_, v_stx_1437_, v_a_1438_, v_a_1439_, v_a_1440_, v_a_1441_, v_a_1442_, v_a_1443_);
return v___x_1446_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabSimpConfigCtx___boxed(lean_object* v_stx_1447_, lean_object* v_a_1448_, lean_object* v_a_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v_res_1455_; 
v_res_1455_ = lp_aesop_Aesop_Frontend_elabSimpConfigCtx(v_stx_1447_, v_a_1448_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_);
lean_dec(v_a_1453_);
lean_dec_ref(v_a_1452_);
lean_dec(v_a_1451_);
lean_dec_ref(v_a_1450_);
lean_dec(v_a_1449_);
lean_dec_ref(v_a_1448_);
return v_res_1455_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; 
v___x_1456_ = lean_box(0);
v___x_1457_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1458_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1458_, 0, v___x_1457_);
lean_ctor_set(v___x_1458_, 1, v___x_1456_);
return v___x_1458_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg(){
_start:
{
lean_object* v___x_1460_; lean_object* v___x_1461_; 
v___x_1460_ = lean_obj_once(&lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0, &lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0);
v___x_1461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1461_, 0, v___x_1460_);
return v___x_1461_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___boxed(lean_object* v___y_1462_){
_start:
{
lean_object* v_res_1463_; 
v_res_1463_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
return v_res_1463_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0(lean_object* v_00_u03b1_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_){
_start:
{
lean_object* v___x_1473_; 
v___x_1473_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
return v___x_1473_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___boxed(lean_object* v_00_u03b1_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_){
_start:
{
lean_object* v_res_1483_; 
v_res_1483_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0(v_00_u03b1_1474_, v___y_1475_, v___y_1476_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v___y_1477_);
lean_dec_ref(v___y_1476_);
lean_dec(v___y_1475_);
return v_res_1483_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(lean_object* v_a_1484_, lean_object* v_x_1485_){
_start:
{
if (lean_obj_tag(v_x_1485_) == 0)
{
uint8_t v___x_1486_; 
v___x_1486_ = 0;
return v___x_1486_;
}
else
{
lean_object* v_key_1487_; lean_object* v_tail_1488_; uint8_t v___x_1489_; 
v_key_1487_ = lean_ctor_get(v_x_1485_, 0);
v_tail_1488_ = lean_ctor_get(v_x_1485_, 2);
v___x_1489_ = lean_name_eq(v_key_1487_, v_a_1484_);
if (v___x_1489_ == 0)
{
v_x_1485_ = v_tail_1488_;
goto _start;
}
else
{
return v___x_1489_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg___boxed(lean_object* v_a_1491_, lean_object* v_x_1492_){
_start:
{
uint8_t v_res_1493_; lean_object* v_r_1494_; 
v_res_1493_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(v_a_1491_, v_x_1492_);
lean_dec(v_x_1492_);
lean_dec(v_a_1491_);
v_r_1494_ = lean_box(v_res_1493_);
return v_r_1494_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg(lean_object* v_a_1495_, lean_object* v_x_1496_){
_start:
{
if (lean_obj_tag(v_x_1496_) == 0)
{
return v_x_1496_;
}
else
{
lean_object* v_key_1497_; lean_object* v_value_1498_; lean_object* v_tail_1499_; lean_object* v___x_1501_; uint8_t v_isShared_1502_; uint8_t v_isSharedCheck_1508_; 
v_key_1497_ = lean_ctor_get(v_x_1496_, 0);
v_value_1498_ = lean_ctor_get(v_x_1496_, 1);
v_tail_1499_ = lean_ctor_get(v_x_1496_, 2);
v_isSharedCheck_1508_ = !lean_is_exclusive(v_x_1496_);
if (v_isSharedCheck_1508_ == 0)
{
v___x_1501_ = v_x_1496_;
v_isShared_1502_ = v_isSharedCheck_1508_;
goto v_resetjp_1500_;
}
else
{
lean_inc(v_tail_1499_);
lean_inc(v_value_1498_);
lean_inc(v_key_1497_);
lean_dec(v_x_1496_);
v___x_1501_ = lean_box(0);
v_isShared_1502_ = v_isSharedCheck_1508_;
goto v_resetjp_1500_;
}
v_resetjp_1500_:
{
uint8_t v___x_1503_; 
v___x_1503_ = lean_name_eq(v_key_1497_, v_a_1495_);
if (v___x_1503_ == 0)
{
lean_object* v___x_1504_; lean_object* v___x_1506_; 
v___x_1504_ = lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg(v_a_1495_, v_tail_1499_);
if (v_isShared_1502_ == 0)
{
lean_ctor_set(v___x_1501_, 2, v___x_1504_);
v___x_1506_ = v___x_1501_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1507_; 
v_reuseFailAlloc_1507_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1507_, 0, v_key_1497_);
lean_ctor_set(v_reuseFailAlloc_1507_, 1, v_value_1498_);
lean_ctor_set(v_reuseFailAlloc_1507_, 2, v___x_1504_);
v___x_1506_ = v_reuseFailAlloc_1507_;
goto v_reusejp_1505_;
}
v_reusejp_1505_:
{
return v___x_1506_;
}
}
else
{
lean_del_object(v___x_1501_);
lean_dec(v_value_1498_);
lean_dec(v_key_1497_);
return v_tail_1499_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg___boxed(lean_object* v_a_1509_, lean_object* v_x_1510_){
_start:
{
lean_object* v_res_1511_; 
v_res_1511_ = lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg(v_a_1509_, v_x_1510_);
lean_dec(v_a_1509_);
return v_res_1511_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg(lean_object* v_m_1512_, lean_object* v_a_1513_){
_start:
{
lean_object* v_size_1514_; lean_object* v_buckets_1515_; lean_object* v___x_1516_; uint64_t v___y_1518_; 
v_size_1514_ = lean_ctor_get(v_m_1512_, 0);
v_buckets_1515_ = lean_ctor_get(v_m_1512_, 1);
v___x_1516_ = lean_array_get_size(v_buckets_1515_);
if (lean_obj_tag(v_a_1513_) == 0)
{
uint64_t v___x_1547_; 
v___x_1547_ = 1723ULL;
v___y_1518_ = v___x_1547_;
goto v___jp_1517_;
}
else
{
uint64_t v_hash_1548_; 
v_hash_1548_ = lean_ctor_get_uint64(v_a_1513_, sizeof(void*)*2);
v___y_1518_ = v_hash_1548_;
goto v___jp_1517_;
}
v___jp_1517_:
{
uint64_t v___x_1519_; uint64_t v___x_1520_; uint64_t v_fold_1521_; uint64_t v___x_1522_; uint64_t v___x_1523_; uint64_t v___x_1524_; size_t v___x_1525_; size_t v___x_1526_; size_t v___x_1527_; size_t v___x_1528_; size_t v___x_1529_; lean_object* v_bkt_1530_; uint8_t v___x_1531_; 
v___x_1519_ = 32ULL;
v___x_1520_ = lean_uint64_shift_right(v___y_1518_, v___x_1519_);
v_fold_1521_ = lean_uint64_xor(v___y_1518_, v___x_1520_);
v___x_1522_ = 16ULL;
v___x_1523_ = lean_uint64_shift_right(v_fold_1521_, v___x_1522_);
v___x_1524_ = lean_uint64_xor(v_fold_1521_, v___x_1523_);
v___x_1525_ = lean_uint64_to_usize(v___x_1524_);
v___x_1526_ = lean_usize_of_nat(v___x_1516_);
v___x_1527_ = ((size_t)1ULL);
v___x_1528_ = lean_usize_sub(v___x_1526_, v___x_1527_);
v___x_1529_ = lean_usize_land(v___x_1525_, v___x_1528_);
v_bkt_1530_ = lean_array_uget_borrowed(v_buckets_1515_, v___x_1529_);
v___x_1531_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(v_a_1513_, v_bkt_1530_);
if (v___x_1531_ == 0)
{
return v_m_1512_;
}
else
{
lean_object* v___x_1533_; uint8_t v_isShared_1534_; uint8_t v_isSharedCheck_1544_; 
lean_inc(v_bkt_1530_);
lean_inc_ref(v_buckets_1515_);
lean_inc(v_size_1514_);
v_isSharedCheck_1544_ = !lean_is_exclusive(v_m_1512_);
if (v_isSharedCheck_1544_ == 0)
{
lean_object* v_unused_1545_; lean_object* v_unused_1546_; 
v_unused_1545_ = lean_ctor_get(v_m_1512_, 1);
lean_dec(v_unused_1545_);
v_unused_1546_ = lean_ctor_get(v_m_1512_, 0);
lean_dec(v_unused_1546_);
v___x_1533_ = v_m_1512_;
v_isShared_1534_ = v_isSharedCheck_1544_;
goto v_resetjp_1532_;
}
else
{
lean_dec(v_m_1512_);
v___x_1533_ = lean_box(0);
v_isShared_1534_ = v_isSharedCheck_1544_;
goto v_resetjp_1532_;
}
v_resetjp_1532_:
{
lean_object* v___x_1535_; lean_object* v_buckets_x27_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1542_; 
v___x_1535_ = lean_box(0);
v_buckets_x27_1536_ = lean_array_uset(v_buckets_1515_, v___x_1529_, v___x_1535_);
v___x_1537_ = lean_unsigned_to_nat(1u);
v___x_1538_ = lean_nat_sub(v_size_1514_, v___x_1537_);
lean_dec(v_size_1514_);
v___x_1539_ = lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg(v_a_1513_, v_bkt_1530_);
v___x_1540_ = lean_array_uset(v_buckets_x27_1536_, v___x_1529_, v___x_1539_);
if (v_isShared_1534_ == 0)
{
lean_ctor_set(v___x_1533_, 1, v___x_1540_);
lean_ctor_set(v___x_1533_, 0, v___x_1538_);
v___x_1542_ = v___x_1533_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v___x_1538_);
lean_ctor_set(v_reuseFailAlloc_1543_, 1, v___x_1540_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg___boxed(lean_object* v_m_1549_, lean_object* v_a_1550_){
_start:
{
lean_object* v_res_1551_; 
v_res_1551_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg(v_m_1549_, v_a_1550_);
lean_dec(v_a_1550_);
return v_res_1551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5(lean_object* v_msgData_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_){
_start:
{
lean_object* v___x_1558_; lean_object* v_env_1559_; lean_object* v___x_1560_; lean_object* v_mctx_1561_; lean_object* v_lctx_1562_; lean_object* v_options_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; 
v___x_1558_ = lean_st_ref_get(v___y_1556_);
v_env_1559_ = lean_ctor_get(v___x_1558_, 0);
lean_inc_ref(v_env_1559_);
lean_dec(v___x_1558_);
v___x_1560_ = lean_st_ref_get(v___y_1554_);
v_mctx_1561_ = lean_ctor_get(v___x_1560_, 0);
lean_inc_ref(v_mctx_1561_);
lean_dec(v___x_1560_);
v_lctx_1562_ = lean_ctor_get(v___y_1553_, 2);
v_options_1563_ = lean_ctor_get(v___y_1555_, 2);
lean_inc_ref(v_options_1563_);
lean_inc_ref(v_lctx_1562_);
v___x_1564_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1564_, 0, v_env_1559_);
lean_ctor_set(v___x_1564_, 1, v_mctx_1561_);
lean_ctor_set(v___x_1564_, 2, v_lctx_1562_);
lean_ctor_set(v___x_1564_, 3, v_options_1563_);
v___x_1565_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1565_, 0, v___x_1564_);
lean_ctor_set(v___x_1565_, 1, v_msgData_1552_);
v___x_1566_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1566_, 0, v___x_1565_);
return v___x_1566_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5___boxed(lean_object* v_msgData_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_, lean_object* v___y_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_){
_start:
{
lean_object* v_res_1573_; 
v_res_1573_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5(v_msgData_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_);
lean_dec(v___y_1571_);
lean_dec_ref(v___y_1570_);
lean_dec(v___y_1569_);
lean_dec_ref(v___y_1568_);
return v_res_1573_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg(lean_object* v_msg_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_){
_start:
{
lean_object* v_ref_1580_; lean_object* v___x_1581_; lean_object* v_a_1582_; lean_object* v___x_1584_; uint8_t v_isShared_1585_; uint8_t v_isSharedCheck_1590_; 
v_ref_1580_ = lean_ctor_get(v___y_1577_, 5);
v___x_1581_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5(v_msg_1574_, v___y_1575_, v___y_1576_, v___y_1577_, v___y_1578_);
v_a_1582_ = lean_ctor_get(v___x_1581_, 0);
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1581_);
if (v_isSharedCheck_1590_ == 0)
{
v___x_1584_ = v___x_1581_;
v_isShared_1585_ = v_isSharedCheck_1590_;
goto v_resetjp_1583_;
}
else
{
lean_inc(v_a_1582_);
lean_dec(v___x_1581_);
v___x_1584_ = lean_box(0);
v_isShared_1585_ = v_isSharedCheck_1590_;
goto v_resetjp_1583_;
}
v_resetjp_1583_:
{
lean_object* v___x_1586_; lean_object* v___x_1588_; 
lean_inc(v_ref_1580_);
v___x_1586_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1586_, 0, v_ref_1580_);
lean_ctor_set(v___x_1586_, 1, v_a_1582_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set_tag(v___x_1584_, 1);
lean_ctor_set(v___x_1584_, 0, v___x_1586_);
v___x_1588_ = v___x_1584_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v___x_1586_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg___boxed(lean_object* v_msg_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_){
_start:
{
lean_object* v_res_1597_; 
v_res_1597_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg(v_msg_1591_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
lean_dec(v___y_1595_);
lean_dec_ref(v___y_1594_);
lean_dec(v___y_1593_);
lean_dec_ref(v___y_1592_);
return v_res_1597_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3_spec__11___redArg(lean_object* v_x_1598_, lean_object* v_x_1599_){
_start:
{
if (lean_obj_tag(v_x_1599_) == 0)
{
return v_x_1598_;
}
else
{
lean_object* v_key_1600_; lean_object* v_value_1601_; lean_object* v_tail_1602_; lean_object* v___x_1604_; uint8_t v_isShared_1605_; uint8_t v_isSharedCheck_1628_; 
v_key_1600_ = lean_ctor_get(v_x_1599_, 0);
v_value_1601_ = lean_ctor_get(v_x_1599_, 1);
v_tail_1602_ = lean_ctor_get(v_x_1599_, 2);
v_isSharedCheck_1628_ = !lean_is_exclusive(v_x_1599_);
if (v_isSharedCheck_1628_ == 0)
{
v___x_1604_ = v_x_1599_;
v_isShared_1605_ = v_isSharedCheck_1628_;
goto v_resetjp_1603_;
}
else
{
lean_inc(v_tail_1602_);
lean_inc(v_value_1601_);
lean_inc(v_key_1600_);
lean_dec(v_x_1599_);
v___x_1604_ = lean_box(0);
v_isShared_1605_ = v_isSharedCheck_1628_;
goto v_resetjp_1603_;
}
v_resetjp_1603_:
{
lean_object* v___x_1606_; uint64_t v___y_1608_; 
v___x_1606_ = lean_array_get_size(v_x_1598_);
if (lean_obj_tag(v_key_1600_) == 0)
{
uint64_t v___x_1626_; 
v___x_1626_ = 1723ULL;
v___y_1608_ = v___x_1626_;
goto v___jp_1607_;
}
else
{
uint64_t v_hash_1627_; 
v_hash_1627_ = lean_ctor_get_uint64(v_key_1600_, sizeof(void*)*2);
v___y_1608_ = v_hash_1627_;
goto v___jp_1607_;
}
v___jp_1607_:
{
uint64_t v___x_1609_; uint64_t v___x_1610_; uint64_t v_fold_1611_; uint64_t v___x_1612_; uint64_t v___x_1613_; uint64_t v___x_1614_; size_t v___x_1615_; size_t v___x_1616_; size_t v___x_1617_; size_t v___x_1618_; size_t v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1622_; 
v___x_1609_ = 32ULL;
v___x_1610_ = lean_uint64_shift_right(v___y_1608_, v___x_1609_);
v_fold_1611_ = lean_uint64_xor(v___y_1608_, v___x_1610_);
v___x_1612_ = 16ULL;
v___x_1613_ = lean_uint64_shift_right(v_fold_1611_, v___x_1612_);
v___x_1614_ = lean_uint64_xor(v_fold_1611_, v___x_1613_);
v___x_1615_ = lean_uint64_to_usize(v___x_1614_);
v___x_1616_ = lean_usize_of_nat(v___x_1606_);
v___x_1617_ = ((size_t)1ULL);
v___x_1618_ = lean_usize_sub(v___x_1616_, v___x_1617_);
v___x_1619_ = lean_usize_land(v___x_1615_, v___x_1618_);
v___x_1620_ = lean_array_uget_borrowed(v_x_1598_, v___x_1619_);
lean_inc(v___x_1620_);
if (v_isShared_1605_ == 0)
{
lean_ctor_set(v___x_1604_, 2, v___x_1620_);
v___x_1622_ = v___x_1604_;
goto v_reusejp_1621_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_key_1600_);
lean_ctor_set(v_reuseFailAlloc_1625_, 1, v_value_1601_);
lean_ctor_set(v_reuseFailAlloc_1625_, 2, v___x_1620_);
v___x_1622_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1621_;
}
v_reusejp_1621_:
{
lean_object* v___x_1623_; 
v___x_1623_ = lean_array_uset(v_x_1598_, v___x_1619_, v___x_1622_);
v_x_1598_ = v___x_1623_;
v_x_1599_ = v_tail_1602_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3___redArg(lean_object* v_i_1629_, lean_object* v_source_1630_, lean_object* v_target_1631_){
_start:
{
lean_object* v___x_1632_; uint8_t v___x_1633_; 
v___x_1632_ = lean_array_get_size(v_source_1630_);
v___x_1633_ = lean_nat_dec_lt(v_i_1629_, v___x_1632_);
if (v___x_1633_ == 0)
{
lean_dec_ref(v_source_1630_);
lean_dec(v_i_1629_);
return v_target_1631_;
}
else
{
lean_object* v_es_1634_; lean_object* v___x_1635_; lean_object* v_source_1636_; lean_object* v_target_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; 
v_es_1634_ = lean_array_fget(v_source_1630_, v_i_1629_);
v___x_1635_ = lean_box(0);
v_source_1636_ = lean_array_fset(v_source_1630_, v_i_1629_, v___x_1635_);
v_target_1637_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3_spec__11___redArg(v_target_1631_, v_es_1634_);
v___x_1638_ = lean_unsigned_to_nat(1u);
v___x_1639_ = lean_nat_add(v_i_1629_, v___x_1638_);
lean_dec(v_i_1629_);
v_i_1629_ = v___x_1639_;
v_source_1630_ = v_source_1636_;
v_target_1631_ = v_target_1637_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2___redArg(lean_object* v_data_1641_){
_start:
{
lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v_nbuckets_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; 
v___x_1642_ = lean_array_get_size(v_data_1641_);
v___x_1643_ = lean_unsigned_to_nat(2u);
v_nbuckets_1644_ = lean_nat_mul(v___x_1642_, v___x_1643_);
v___x_1645_ = lean_unsigned_to_nat(0u);
v___x_1646_ = lean_box(0);
v___x_1647_ = lean_mk_array(v_nbuckets_1644_, v___x_1646_);
v___x_1648_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3___redArg(v___x_1645_, v_data_1641_, v___x_1647_);
return v___x_1648_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1___redArg(lean_object* v_m_1649_, lean_object* v_a_1650_, lean_object* v_b_1651_){
_start:
{
lean_object* v_size_1652_; lean_object* v_buckets_1653_; lean_object* v___x_1654_; uint64_t v___y_1656_; 
v_size_1652_ = lean_ctor_get(v_m_1649_, 0);
v_buckets_1653_ = lean_ctor_get(v_m_1649_, 1);
v___x_1654_ = lean_array_get_size(v_buckets_1653_);
if (lean_obj_tag(v_a_1650_) == 0)
{
uint64_t v___x_1693_; 
v___x_1693_ = 1723ULL;
v___y_1656_ = v___x_1693_;
goto v___jp_1655_;
}
else
{
uint64_t v_hash_1694_; 
v_hash_1694_ = lean_ctor_get_uint64(v_a_1650_, sizeof(void*)*2);
v___y_1656_ = v_hash_1694_;
goto v___jp_1655_;
}
v___jp_1655_:
{
uint64_t v___x_1657_; uint64_t v___x_1658_; uint64_t v_fold_1659_; uint64_t v___x_1660_; uint64_t v___x_1661_; uint64_t v___x_1662_; size_t v___x_1663_; size_t v___x_1664_; size_t v___x_1665_; size_t v___x_1666_; size_t v___x_1667_; lean_object* v_bkt_1668_; uint8_t v___x_1669_; 
v___x_1657_ = 32ULL;
v___x_1658_ = lean_uint64_shift_right(v___y_1656_, v___x_1657_);
v_fold_1659_ = lean_uint64_xor(v___y_1656_, v___x_1658_);
v___x_1660_ = 16ULL;
v___x_1661_ = lean_uint64_shift_right(v_fold_1659_, v___x_1660_);
v___x_1662_ = lean_uint64_xor(v_fold_1659_, v___x_1661_);
v___x_1663_ = lean_uint64_to_usize(v___x_1662_);
v___x_1664_ = lean_usize_of_nat(v___x_1654_);
v___x_1665_ = ((size_t)1ULL);
v___x_1666_ = lean_usize_sub(v___x_1664_, v___x_1665_);
v___x_1667_ = lean_usize_land(v___x_1663_, v___x_1666_);
v_bkt_1668_ = lean_array_uget_borrowed(v_buckets_1653_, v___x_1667_);
v___x_1669_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(v_a_1650_, v_bkt_1668_);
if (v___x_1669_ == 0)
{
lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1690_; 
lean_inc_ref(v_buckets_1653_);
lean_inc(v_size_1652_);
v_isSharedCheck_1690_ = !lean_is_exclusive(v_m_1649_);
if (v_isSharedCheck_1690_ == 0)
{
lean_object* v_unused_1691_; lean_object* v_unused_1692_; 
v_unused_1691_ = lean_ctor_get(v_m_1649_, 1);
lean_dec(v_unused_1691_);
v_unused_1692_ = lean_ctor_get(v_m_1649_, 0);
lean_dec(v_unused_1692_);
v___x_1671_ = v_m_1649_;
v_isShared_1672_ = v_isSharedCheck_1690_;
goto v_resetjp_1670_;
}
else
{
lean_dec(v_m_1649_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1690_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v___x_1673_; lean_object* v_size_x27_1674_; lean_object* v___x_1675_; lean_object* v_buckets_x27_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; uint8_t v___x_1682_; 
v___x_1673_ = lean_unsigned_to_nat(1u);
v_size_x27_1674_ = lean_nat_add(v_size_1652_, v___x_1673_);
lean_dec(v_size_1652_);
lean_inc(v_bkt_1668_);
v___x_1675_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1675_, 0, v_a_1650_);
lean_ctor_set(v___x_1675_, 1, v_b_1651_);
lean_ctor_set(v___x_1675_, 2, v_bkt_1668_);
v_buckets_x27_1676_ = lean_array_uset(v_buckets_1653_, v___x_1667_, v___x_1675_);
v___x_1677_ = lean_unsigned_to_nat(4u);
v___x_1678_ = lean_nat_mul(v_size_x27_1674_, v___x_1677_);
v___x_1679_ = lean_unsigned_to_nat(3u);
v___x_1680_ = lean_nat_div(v___x_1678_, v___x_1679_);
lean_dec(v___x_1678_);
v___x_1681_ = lean_array_get_size(v_buckets_x27_1676_);
v___x_1682_ = lean_nat_dec_le(v___x_1680_, v___x_1681_);
lean_dec(v___x_1680_);
if (v___x_1682_ == 0)
{
lean_object* v_val_1683_; lean_object* v___x_1685_; 
v_val_1683_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2___redArg(v_buckets_x27_1676_);
if (v_isShared_1672_ == 0)
{
lean_ctor_set(v___x_1671_, 1, v_val_1683_);
lean_ctor_set(v___x_1671_, 0, v_size_x27_1674_);
v___x_1685_ = v___x_1671_;
goto v_reusejp_1684_;
}
else
{
lean_object* v_reuseFailAlloc_1686_; 
v_reuseFailAlloc_1686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1686_, 0, v_size_x27_1674_);
lean_ctor_set(v_reuseFailAlloc_1686_, 1, v_val_1683_);
v___x_1685_ = v_reuseFailAlloc_1686_;
goto v_reusejp_1684_;
}
v_reusejp_1684_:
{
return v___x_1685_;
}
}
else
{
lean_object* v___x_1688_; 
if (v_isShared_1672_ == 0)
{
lean_ctor_set(v___x_1671_, 1, v_buckets_x27_1676_);
lean_ctor_set(v___x_1671_, 0, v_size_x27_1674_);
v___x_1688_ = v___x_1671_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v_size_x27_1674_);
lean_ctor_set(v_reuseFailAlloc_1689_, 1, v_buckets_x27_1676_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
else
{
lean_dec(v_b_1651_);
lean_dec(v_a_1650_);
return v_m_1649_;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg(lean_object* v_m_1695_, lean_object* v_a_1696_){
_start:
{
lean_object* v_buckets_1697_; lean_object* v___x_1698_; uint64_t v___y_1700_; 
v_buckets_1697_ = lean_ctor_get(v_m_1695_, 1);
v___x_1698_ = lean_array_get_size(v_buckets_1697_);
if (lean_obj_tag(v_a_1696_) == 0)
{
uint64_t v___x_1714_; 
v___x_1714_ = 1723ULL;
v___y_1700_ = v___x_1714_;
goto v___jp_1699_;
}
else
{
uint64_t v_hash_1715_; 
v_hash_1715_ = lean_ctor_get_uint64(v_a_1696_, sizeof(void*)*2);
v___y_1700_ = v_hash_1715_;
goto v___jp_1699_;
}
v___jp_1699_:
{
uint64_t v___x_1701_; uint64_t v___x_1702_; uint64_t v_fold_1703_; uint64_t v___x_1704_; uint64_t v___x_1705_; uint64_t v___x_1706_; size_t v___x_1707_; size_t v___x_1708_; size_t v___x_1709_; size_t v___x_1710_; size_t v___x_1711_; lean_object* v___x_1712_; uint8_t v___x_1713_; 
v___x_1701_ = 32ULL;
v___x_1702_ = lean_uint64_shift_right(v___y_1700_, v___x_1701_);
v_fold_1703_ = lean_uint64_xor(v___y_1700_, v___x_1702_);
v___x_1704_ = 16ULL;
v___x_1705_ = lean_uint64_shift_right(v_fold_1703_, v___x_1704_);
v___x_1706_ = lean_uint64_xor(v_fold_1703_, v___x_1705_);
v___x_1707_ = lean_uint64_to_usize(v___x_1706_);
v___x_1708_ = lean_usize_of_nat(v___x_1698_);
v___x_1709_ = ((size_t)1ULL);
v___x_1710_ = lean_usize_sub(v___x_1708_, v___x_1709_);
v___x_1711_ = lean_usize_land(v___x_1707_, v___x_1710_);
v___x_1712_ = lean_array_uget_borrowed(v_buckets_1697_, v___x_1711_);
v___x_1713_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(v_a_1696_, v___x_1712_);
return v___x_1713_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg___boxed(lean_object* v_m_1716_, lean_object* v_a_1717_){
_start:
{
uint8_t v_res_1718_; lean_object* v_r_1719_; 
v_res_1718_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg(v_m_1716_, v_a_1717_);
lean_dec(v_a_1717_);
lean_dec_ref(v_m_1716_);
v_r_1719_ = lean_box(v_res_1718_);
return v_r_1719_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__1(void){
_start:
{
lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1721_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__0));
v___x_1722_ = l_Lean_stringToMessageData(v___x_1721_);
return v___x_1722_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__3(void){
_start:
{
lean_object* v___x_1724_; lean_object* v___x_1725_; 
v___x_1724_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__2));
v___x_1725_ = l_Lean_stringToMessageData(v___x_1724_);
return v___x_1725_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__5(void){
_start:
{
lean_object* v___x_1727_; lean_object* v___x_1728_; 
v___x_1727_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__4));
v___x_1728_ = l_Lean_stringToMessageData(v___x_1727_);
return v___x_1728_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__7(void){
_start:
{
lean_object* v___x_1730_; lean_object* v___x_1731_; 
v___x_1730_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__6));
v___x_1731_ = l_Lean_stringToMessageData(v___x_1730_);
return v___x_1731_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5(lean_object* v_as_1732_, size_t v_sz_1733_, size_t v_i_1734_, lean_object* v_b_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_){
_start:
{
lean_object* v_a_1745_; uint8_t v___x_1749_; 
v___x_1749_ = lean_usize_dec_lt(v_i_1734_, v_sz_1733_);
if (v___x_1749_ == 0)
{
lean_object* v___x_1750_; 
v___x_1750_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1750_, 0, v_b_1735_);
return v___x_1750_;
}
else
{
lean_object* v___x_1751_; lean_object* v_a_1752_; uint8_t v___x_1753_; 
v___x_1751_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__2));
v_a_1752_ = lean_array_uget_borrowed(v_as_1732_, v_i_1734_);
lean_inc(v_a_1752_);
v___x_1753_ = l_Lean_Syntax_isOfKind(v_a_1752_, v___x_1751_);
if (v___x_1753_ == 0)
{
lean_object* v___x_1754_; 
v___x_1754_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
if (lean_obj_tag(v___x_1754_) == 0)
{
lean_dec_ref_known(v___x_1754_, 1);
v_a_1745_ = v_b_1735_;
goto v___jp_1744_;
}
else
{
lean_object* v_a_1755_; lean_object* v___x_1757_; uint8_t v_isShared_1758_; uint8_t v_isSharedCheck_1762_; 
lean_dec_ref(v_b_1735_);
v_a_1755_ = lean_ctor_get(v___x_1754_, 0);
v_isSharedCheck_1762_ = !lean_is_exclusive(v___x_1754_);
if (v_isSharedCheck_1762_ == 0)
{
v___x_1757_ = v___x_1754_;
v_isShared_1758_ = v_isSharedCheck_1762_;
goto v_resetjp_1756_;
}
else
{
lean_inc(v_a_1755_);
lean_dec(v___x_1754_);
v___x_1757_ = lean_box(0);
v_isShared_1758_ = v_isSharedCheck_1762_;
goto v_resetjp_1756_;
}
v_resetjp_1756_:
{
lean_object* v___x_1760_; 
if (v_isShared_1758_ == 0)
{
v___x_1760_ = v___x_1757_;
goto v_reusejp_1759_;
}
else
{
lean_object* v_reuseFailAlloc_1761_; 
v_reuseFailAlloc_1761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1761_, 0, v_a_1755_);
v___x_1760_ = v_reuseFailAlloc_1761_;
goto v_reusejp_1759_;
}
v_reusejp_1759_:
{
return v___x_1760_;
}
}
}
}
else
{
lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; uint8_t v___x_1766_; 
v___x_1763_ = lean_unsigned_to_nat(0u);
v___x_1764_ = lean_unsigned_to_nat(1u);
v___x_1765_ = l_Lean_Syntax_getArg(v_a_1752_, v___x_1763_);
lean_inc(v___x_1765_);
v___x_1766_ = l_Lean_Syntax_matchesNull(v___x_1765_, v___x_1764_);
if (v___x_1766_ == 0)
{
uint8_t v___x_1767_; 
v___x_1767_ = l_Lean_Syntax_matchesNull(v___x_1765_, v___x_1763_);
if (v___x_1767_ == 0)
{
lean_object* v___x_1768_; 
v___x_1768_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
if (lean_obj_tag(v___x_1768_) == 0)
{
lean_dec_ref_known(v___x_1768_, 1);
v_a_1745_ = v_b_1735_;
goto v___jp_1744_;
}
else
{
lean_object* v_a_1769_; lean_object* v___x_1771_; uint8_t v_isShared_1772_; uint8_t v_isSharedCheck_1776_; 
lean_dec_ref(v_b_1735_);
v_a_1769_ = lean_ctor_get(v___x_1768_, 0);
v_isSharedCheck_1776_ = !lean_is_exclusive(v___x_1768_);
if (v_isSharedCheck_1776_ == 0)
{
v___x_1771_ = v___x_1768_;
v_isShared_1772_ = v_isSharedCheck_1776_;
goto v_resetjp_1770_;
}
else
{
lean_inc(v_a_1769_);
lean_dec(v___x_1768_);
v___x_1771_ = lean_box(0);
v_isShared_1772_ = v_isSharedCheck_1776_;
goto v_resetjp_1770_;
}
v_resetjp_1770_:
{
lean_object* v___x_1774_; 
if (v_isShared_1772_ == 0)
{
v___x_1774_ = v___x_1771_;
goto v_reusejp_1773_;
}
else
{
lean_object* v_reuseFailAlloc_1775_; 
v_reuseFailAlloc_1775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1775_, 0, v_a_1769_);
v___x_1774_ = v_reuseFailAlloc_1775_;
goto v_reusejp_1773_;
}
v_reusejp_1773_:
{
return v___x_1774_;
}
}
}
}
else
{
lean_object* v___x_1777_; lean_object* v___x_1778_; uint8_t v___x_1779_; 
v___x_1777_ = l_Lean_Syntax_getArg(v_a_1752_, v___x_1764_);
v___x_1778_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__9));
lean_inc(v___x_1777_);
v___x_1779_ = l_Lean_Syntax_isOfKind(v___x_1777_, v___x_1778_);
if (v___x_1779_ == 0)
{
lean_object* v___x_1780_; 
lean_dec(v___x_1777_);
v___x_1780_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
if (lean_obj_tag(v___x_1780_) == 0)
{
lean_dec_ref_known(v___x_1780_, 1);
v_a_1745_ = v_b_1735_;
goto v___jp_1744_;
}
else
{
lean_object* v_a_1781_; lean_object* v___x_1783_; uint8_t v_isShared_1784_; uint8_t v_isSharedCheck_1788_; 
lean_dec_ref(v_b_1735_);
v_a_1781_ = lean_ctor_get(v___x_1780_, 0);
v_isSharedCheck_1788_ = !lean_is_exclusive(v___x_1780_);
if (v_isSharedCheck_1788_ == 0)
{
v___x_1783_ = v___x_1780_;
v_isShared_1784_ = v_isSharedCheck_1788_;
goto v_resetjp_1782_;
}
else
{
lean_inc(v_a_1781_);
lean_dec(v___x_1780_);
v___x_1783_ = lean_box(0);
v_isShared_1784_ = v_isSharedCheck_1788_;
goto v_resetjp_1782_;
}
v_resetjp_1782_:
{
lean_object* v___x_1786_; 
if (v_isShared_1784_ == 0)
{
v___x_1786_ = v___x_1783_;
goto v_reusejp_1785_;
}
else
{
lean_object* v_reuseFailAlloc_1787_; 
v_reuseFailAlloc_1787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1787_, 0, v_a_1781_);
v___x_1786_ = v_reuseFailAlloc_1787_;
goto v_reusejp_1785_;
}
v_reusejp_1785_:
{
return v___x_1786_;
}
}
}
}
else
{
lean_object* v___x_1789_; uint8_t v___x_1793_; 
v___x_1789_ = lp_aesop_Aesop_Frontend_RuleSetName_elab(v___x_1777_);
lean_dec(v___x_1777_);
v___x_1793_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg(v_b_1735_, v___x_1789_);
if (v___x_1793_ == 0)
{
goto v___jp_1790_;
}
else
{
lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; 
v___x_1794_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__1);
lean_inc(v___x_1789_);
v___x_1795_ = l_Lean_MessageData_ofName(v___x_1789_);
v___x_1796_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1796_, 0, v___x_1794_);
lean_ctor_set(v___x_1796_, 1, v___x_1795_);
v___x_1797_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__3);
v___x_1798_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1798_, 0, v___x_1796_);
lean_ctor_set(v___x_1798_, 1, v___x_1797_);
v___x_1799_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg(v___x_1798_, v___y_1739_, v___y_1740_, v___y_1741_, v___y_1742_);
if (lean_obj_tag(v___x_1799_) == 0)
{
lean_dec_ref_known(v___x_1799_, 1);
goto v___jp_1790_;
}
else
{
lean_object* v_a_1800_; lean_object* v___x_1802_; uint8_t v_isShared_1803_; uint8_t v_isSharedCheck_1807_; 
lean_dec(v___x_1789_);
lean_dec_ref(v_b_1735_);
v_a_1800_ = lean_ctor_get(v___x_1799_, 0);
v_isSharedCheck_1807_ = !lean_is_exclusive(v___x_1799_);
if (v_isSharedCheck_1807_ == 0)
{
v___x_1802_ = v___x_1799_;
v_isShared_1803_ = v_isSharedCheck_1807_;
goto v_resetjp_1801_;
}
else
{
lean_inc(v_a_1800_);
lean_dec(v___x_1799_);
v___x_1802_ = lean_box(0);
v_isShared_1803_ = v_isSharedCheck_1807_;
goto v_resetjp_1801_;
}
v_resetjp_1801_:
{
lean_object* v___x_1805_; 
if (v_isShared_1803_ == 0)
{
v___x_1805_ = v___x_1802_;
goto v_reusejp_1804_;
}
else
{
lean_object* v_reuseFailAlloc_1806_; 
v_reuseFailAlloc_1806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1806_, 0, v_a_1800_);
v___x_1805_ = v_reuseFailAlloc_1806_;
goto v_reusejp_1804_;
}
v_reusejp_1804_:
{
return v___x_1805_;
}
}
}
}
v___jp_1790_:
{
lean_object* v___x_1791_; lean_object* v___x_1792_; 
v___x_1791_ = lean_box(0);
v___x_1792_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1___redArg(v_b_1735_, v___x_1789_, v___x_1791_);
v_a_1745_ = v___x_1792_;
goto v___jp_1744_;
}
}
}
}
else
{
lean_object* v___x_1808_; lean_object* v___x_1809_; uint8_t v___x_1810_; 
lean_dec(v___x_1765_);
v___x_1808_ = l_Lean_Syntax_getArg(v_a_1752_, v___x_1764_);
v___x_1809_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_ruleSetSpec___closed__9));
lean_inc(v___x_1808_);
v___x_1810_ = l_Lean_Syntax_isOfKind(v___x_1808_, v___x_1809_);
if (v___x_1810_ == 0)
{
lean_object* v___x_1811_; 
lean_dec(v___x_1808_);
v___x_1811_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
if (lean_obj_tag(v___x_1811_) == 0)
{
lean_dec_ref_known(v___x_1811_, 1);
v_a_1745_ = v_b_1735_;
goto v___jp_1744_;
}
else
{
lean_object* v_a_1812_; lean_object* v___x_1814_; uint8_t v_isShared_1815_; uint8_t v_isSharedCheck_1819_; 
lean_dec_ref(v_b_1735_);
v_a_1812_ = lean_ctor_get(v___x_1811_, 0);
v_isSharedCheck_1819_ = !lean_is_exclusive(v___x_1811_);
if (v_isSharedCheck_1819_ == 0)
{
v___x_1814_ = v___x_1811_;
v_isShared_1815_ = v_isSharedCheck_1819_;
goto v_resetjp_1813_;
}
else
{
lean_inc(v_a_1812_);
lean_dec(v___x_1811_);
v___x_1814_ = lean_box(0);
v_isShared_1815_ = v_isSharedCheck_1819_;
goto v_resetjp_1813_;
}
v_resetjp_1813_:
{
lean_object* v___x_1817_; 
if (v_isShared_1815_ == 0)
{
v___x_1817_ = v___x_1814_;
goto v_reusejp_1816_;
}
else
{
lean_object* v_reuseFailAlloc_1818_; 
v_reuseFailAlloc_1818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1818_, 0, v_a_1812_);
v___x_1817_ = v_reuseFailAlloc_1818_;
goto v_reusejp_1816_;
}
v_reusejp_1816_:
{
return v___x_1817_;
}
}
}
}
else
{
lean_object* v___x_1820_; uint8_t v___x_1823_; 
v___x_1820_ = lp_aesop_Aesop_Frontend_RuleSetName_elab(v___x_1808_);
lean_dec(v___x_1808_);
v___x_1823_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg(v_b_1735_, v___x_1820_);
if (v___x_1823_ == 0)
{
lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; 
v___x_1824_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__5, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__5_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__5);
lean_inc(v___x_1820_);
v___x_1825_ = l_Lean_MessageData_ofName(v___x_1820_);
v___x_1826_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1826_, 0, v___x_1824_);
lean_ctor_set(v___x_1826_, 1, v___x_1825_);
v___x_1827_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__7, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__7_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___closed__7);
v___x_1828_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1828_, 0, v___x_1826_);
lean_ctor_set(v___x_1828_, 1, v___x_1827_);
v___x_1829_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg(v___x_1828_, v___y_1739_, v___y_1740_, v___y_1741_, v___y_1742_);
if (lean_obj_tag(v___x_1829_) == 0)
{
lean_dec_ref_known(v___x_1829_, 1);
goto v___jp_1821_;
}
else
{
lean_object* v_a_1830_; lean_object* v___x_1832_; uint8_t v_isShared_1833_; uint8_t v_isSharedCheck_1837_; 
lean_dec(v___x_1820_);
lean_dec_ref(v_b_1735_);
v_a_1830_ = lean_ctor_get(v___x_1829_, 0);
v_isSharedCheck_1837_ = !lean_is_exclusive(v___x_1829_);
if (v_isSharedCheck_1837_ == 0)
{
v___x_1832_ = v___x_1829_;
v_isShared_1833_ = v_isSharedCheck_1837_;
goto v_resetjp_1831_;
}
else
{
lean_inc(v_a_1830_);
lean_dec(v___x_1829_);
v___x_1832_ = lean_box(0);
v_isShared_1833_ = v_isSharedCheck_1837_;
goto v_resetjp_1831_;
}
v_resetjp_1831_:
{
lean_object* v___x_1835_; 
if (v_isShared_1833_ == 0)
{
v___x_1835_ = v___x_1832_;
goto v_reusejp_1834_;
}
else
{
lean_object* v_reuseFailAlloc_1836_; 
v_reuseFailAlloc_1836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1836_, 0, v_a_1830_);
v___x_1835_ = v_reuseFailAlloc_1836_;
goto v_reusejp_1834_;
}
v_reusejp_1834_:
{
return v___x_1835_;
}
}
}
}
else
{
goto v___jp_1821_;
}
v___jp_1821_:
{
lean_object* v___x_1822_; 
v___x_1822_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg(v_b_1735_, v___x_1820_);
lean_dec(v___x_1820_);
v_a_1745_ = v___x_1822_;
goto v___jp_1744_;
}
}
}
}
}
v___jp_1744_:
{
size_t v___x_1746_; size_t v___x_1747_; 
v___x_1746_ = ((size_t)1ULL);
v___x_1747_ = lean_usize_add(v_i_1734_, v___x_1746_);
v_i_1734_ = v___x_1747_;
v_b_1735_ = v_a_1745_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5___boxed(lean_object* v_as_1838_, lean_object* v_sz_1839_, lean_object* v_i_1840_, lean_object* v_b_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_){
_start:
{
size_t v_sz_boxed_1850_; size_t v_i_boxed_1851_; lean_object* v_res_1852_; 
v_sz_boxed_1850_ = lean_unbox_usize(v_sz_1839_);
lean_dec(v_sz_1839_);
v_i_boxed_1851_ = lean_unbox_usize(v_i_1840_);
lean_dec(v_i_1840_);
v_res_1852_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5(v_as_1838_, v_sz_boxed_1850_, v_i_boxed_1851_, v_b_1841_, v___y_1842_, v___y_1843_, v___y_1844_, v___y_1845_, v___y_1846_, v___y_1847_, v___y_1848_);
lean_dec(v___y_1848_);
lean_dec_ref(v___y_1847_);
lean_dec(v___y_1846_);
lean_dec_ref(v___y_1845_);
lean_dec(v___y_1844_);
lean_dec_ref(v___y_1843_);
lean_dec(v___y_1842_);
lean_dec_ref(v_as_1838_);
return v_res_1852_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg(lean_object* v_goal_1853_, size_t v_sz_1854_, size_t v_i_1855_, lean_object* v_bs_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_){
_start:
{
uint8_t v___x_1864_; 
v___x_1864_ = lean_usize_dec_lt(v_i_1855_, v_sz_1854_);
if (v___x_1864_ == 0)
{
lean_object* v___x_1865_; 
lean_dec(v_goal_1853_);
v___x_1865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1865_, 0, v_bs_1856_);
return v___x_1865_;
}
else
{
lean_object* v_v_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; 
v_v_1866_ = lean_array_uget_borrowed(v_bs_1856_, v_i_1855_);
lean_inc(v_goal_1853_);
v___x_1867_ = lp_aesop_Aesop_ElabM_Context_forAdditionalRules(v_goal_1853_);
lean_inc(v_v_1866_);
v___x_1868_ = lp_aesop_Aesop_Frontend_RuleExpr_elab(v_v_1866_, v___x_1867_, v___y_1857_, v___y_1858_, v___y_1859_, v___y_1860_, v___y_1861_, v___y_1862_);
lean_dec_ref(v___x_1867_);
if (lean_obj_tag(v___x_1868_) == 0)
{
lean_object* v_a_1869_; lean_object* v___x_1870_; lean_object* v_bs_x27_1871_; size_t v___x_1872_; size_t v___x_1873_; lean_object* v___x_1874_; 
v_a_1869_ = lean_ctor_get(v___x_1868_, 0);
lean_inc(v_a_1869_);
lean_dec_ref_known(v___x_1868_, 1);
v___x_1870_ = lean_unsigned_to_nat(0u);
v_bs_x27_1871_ = lean_array_uset(v_bs_1856_, v_i_1855_, v___x_1870_);
v___x_1872_ = ((size_t)1ULL);
v___x_1873_ = lean_usize_add(v_i_1855_, v___x_1872_);
v___x_1874_ = lean_array_uset(v_bs_x27_1871_, v_i_1855_, v_a_1869_);
v_i_1855_ = v___x_1873_;
v_bs_1856_ = v___x_1874_;
goto _start;
}
else
{
lean_object* v_a_1876_; lean_object* v___x_1878_; uint8_t v_isShared_1879_; uint8_t v_isSharedCheck_1883_; 
lean_dec_ref(v_bs_1856_);
lean_dec(v_goal_1853_);
v_a_1876_ = lean_ctor_get(v___x_1868_, 0);
v_isSharedCheck_1883_ = !lean_is_exclusive(v___x_1868_);
if (v_isSharedCheck_1883_ == 0)
{
v___x_1878_ = v___x_1868_;
v_isShared_1879_ = v_isSharedCheck_1883_;
goto v_resetjp_1877_;
}
else
{
lean_inc(v_a_1876_);
lean_dec(v___x_1868_);
v___x_1878_ = lean_box(0);
v_isShared_1879_ = v_isSharedCheck_1883_;
goto v_resetjp_1877_;
}
v_resetjp_1877_:
{
lean_object* v___x_1881_; 
if (v_isShared_1879_ == 0)
{
v___x_1881_ = v___x_1878_;
goto v_reusejp_1880_;
}
else
{
lean_object* v_reuseFailAlloc_1882_; 
v_reuseFailAlloc_1882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1882_, 0, v_a_1876_);
v___x_1881_ = v_reuseFailAlloc_1882_;
goto v_reusejp_1880_;
}
v_reusejp_1880_:
{
return v___x_1881_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg___boxed(lean_object* v_goal_1884_, lean_object* v_sz_1885_, lean_object* v_i_1886_, lean_object* v_bs_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_){
_start:
{
size_t v_sz_boxed_1895_; size_t v_i_boxed_1896_; lean_object* v_res_1897_; 
v_sz_boxed_1895_ = lean_unbox_usize(v_sz_1885_);
lean_dec(v_sz_1885_);
v_i_boxed_1896_ = lean_unbox_usize(v_i_1886_);
lean_dec(v_i_1886_);
v_res_1897_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg(v_goal_1884_, v_sz_boxed_1895_, v_i_boxed_1896_, v_bs_1887_, v___y_1888_, v___y_1889_, v___y_1890_, v___y_1891_, v___y_1892_, v___y_1893_);
lean_dec(v___y_1893_);
lean_dec_ref(v___y_1892_);
lean_dec(v___y_1891_);
lean_dec_ref(v___y_1890_);
lean_dec(v___y_1889_);
lean_dec_ref(v___y_1888_);
return v_res_1897_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg(lean_object* v_goal_1898_, size_t v_sz_1899_, size_t v_i_1900_, lean_object* v_bs_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_){
_start:
{
uint8_t v___x_1909_; 
v___x_1909_ = lean_usize_dec_lt(v_i_1900_, v_sz_1899_);
if (v___x_1909_ == 0)
{
lean_object* v___x_1910_; 
lean_dec(v_goal_1898_);
v___x_1910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1910_, 0, v_bs_1901_);
return v___x_1910_;
}
else
{
lean_object* v_v_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; 
v_v_1911_ = lean_array_uget_borrowed(v_bs_1901_, v_i_1900_);
lean_inc(v_goal_1898_);
v___x_1912_ = lp_aesop_Aesop_ElabM_Context_forErasing(v_goal_1898_);
lean_inc(v_v_1911_);
v___x_1913_ = lp_aesop_Aesop_Frontend_RuleExpr_elab(v_v_1911_, v___x_1912_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_);
lean_dec_ref(v___x_1912_);
if (lean_obj_tag(v___x_1913_) == 0)
{
lean_object* v_a_1914_; lean_object* v___x_1915_; lean_object* v_bs_x27_1916_; size_t v___x_1917_; size_t v___x_1918_; lean_object* v___x_1919_; 
v_a_1914_ = lean_ctor_get(v___x_1913_, 0);
lean_inc(v_a_1914_);
lean_dec_ref_known(v___x_1913_, 1);
v___x_1915_ = lean_unsigned_to_nat(0u);
v_bs_x27_1916_ = lean_array_uset(v_bs_1901_, v_i_1900_, v___x_1915_);
v___x_1917_ = ((size_t)1ULL);
v___x_1918_ = lean_usize_add(v_i_1900_, v___x_1917_);
v___x_1919_ = lean_array_uset(v_bs_x27_1916_, v_i_1900_, v_a_1914_);
v_i_1900_ = v___x_1918_;
v_bs_1901_ = v___x_1919_;
goto _start;
}
else
{
lean_object* v_a_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_1928_; 
lean_dec_ref(v_bs_1901_);
lean_dec(v_goal_1898_);
v_a_1921_ = lean_ctor_get(v___x_1913_, 0);
v_isSharedCheck_1928_ = !lean_is_exclusive(v___x_1913_);
if (v_isSharedCheck_1928_ == 0)
{
v___x_1923_ = v___x_1913_;
v_isShared_1924_ = v_isSharedCheck_1928_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_a_1921_);
lean_dec(v___x_1913_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_1928_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
lean_object* v___x_1926_; 
if (v_isShared_1924_ == 0)
{
v___x_1926_ = v___x_1923_;
goto v_reusejp_1925_;
}
else
{
lean_object* v_reuseFailAlloc_1927_; 
v_reuseFailAlloc_1927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1927_, 0, v_a_1921_);
v___x_1926_ = v_reuseFailAlloc_1927_;
goto v_reusejp_1925_;
}
v_reusejp_1925_:
{
return v___x_1926_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg___boxed(lean_object* v_goal_1929_, lean_object* v_sz_1930_, lean_object* v_i_1931_, lean_object* v_bs_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_){
_start:
{
size_t v_sz_boxed_1940_; size_t v_i_boxed_1941_; lean_object* v_res_1942_; 
v_sz_boxed_1940_ = lean_unbox_usize(v_sz_1930_);
lean_dec(v_sz_1930_);
v_i_boxed_1941_ = lean_unbox_usize(v_i_1931_);
lean_dec(v_i_1931_);
v_res_1942_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg(v_goal_1929_, v_sz_boxed_1940_, v_i_boxed_1941_, v_bs_1932_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_, v___y_1937_, v___y_1938_);
lean_dec(v___y_1938_);
lean_dec_ref(v___y_1937_);
lean_dec(v___y_1936_);
lean_dec_ref(v___y_1935_);
lean_dec(v___y_1934_);
lean_dec_ref(v___y_1933_);
return v_res_1942_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause(lean_object* v_goal_1943_, uint8_t v_traceScript_1944_, lean_object* v_stx_1945_, lean_object* v_a_1946_, lean_object* v_a_1947_, lean_object* v_a_1948_, lean_object* v_a_1949_, lean_object* v_a_1950_, lean_object* v_a_1951_, lean_object* v_a_1952_){
_start:
{
lean_object* v_fileName_1954_; lean_object* v_fileMap_1955_; lean_object* v_options_1956_; lean_object* v_currRecDepth_1957_; lean_object* v_maxRecDepth_1958_; lean_object* v_ref_1959_; lean_object* v_currNamespace_1960_; lean_object* v_openDecls_1961_; lean_object* v_initHeartbeats_1962_; lean_object* v_maxHeartbeats_1963_; lean_object* v_quotContext_1964_; lean_object* v_currMacroScope_1965_; uint8_t v_diag_1966_; lean_object* v_cancelTk_x3f_1967_; uint8_t v_suppressElabErrors_1968_; lean_object* v_inheritedTraceOptions_1969_; lean_object* v___x_1970_; uint8_t v___x_1971_; lean_object* v_ref_1972_; lean_object* v___x_1973_; 
v_fileName_1954_ = lean_ctor_get(v_a_1951_, 0);
v_fileMap_1955_ = lean_ctor_get(v_a_1951_, 1);
v_options_1956_ = lean_ctor_get(v_a_1951_, 2);
v_currRecDepth_1957_ = lean_ctor_get(v_a_1951_, 3);
v_maxRecDepth_1958_ = lean_ctor_get(v_a_1951_, 4);
v_ref_1959_ = lean_ctor_get(v_a_1951_, 5);
v_currNamespace_1960_ = lean_ctor_get(v_a_1951_, 6);
v_openDecls_1961_ = lean_ctor_get(v_a_1951_, 7);
v_initHeartbeats_1962_ = lean_ctor_get(v_a_1951_, 8);
v_maxHeartbeats_1963_ = lean_ctor_get(v_a_1951_, 9);
v_quotContext_1964_ = lean_ctor_get(v_a_1951_, 10);
v_currMacroScope_1965_ = lean_ctor_get(v_a_1951_, 11);
v_diag_1966_ = lean_ctor_get_uint8(v_a_1951_, sizeof(void*)*14);
v_cancelTk_x3f_1967_ = lean_ctor_get(v_a_1951_, 12);
v_suppressElabErrors_1968_ = lean_ctor_get_uint8(v_a_1951_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1969_ = lean_ctor_get(v_a_1951_, 13);
v___x_1970_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Add___x29___closed__1));
lean_inc(v_stx_1945_);
v___x_1971_ = l_Lean_Syntax_isOfKind(v_stx_1945_, v___x_1970_);
v_ref_1972_ = l_Lean_replaceRef(v_stx_1945_, v_ref_1959_);
lean_inc_ref(v_inheritedTraceOptions_1969_);
lean_inc(v_cancelTk_x3f_1967_);
lean_inc(v_currMacroScope_1965_);
lean_inc(v_quotContext_1964_);
lean_inc(v_maxHeartbeats_1963_);
lean_inc(v_initHeartbeats_1962_);
lean_inc(v_openDecls_1961_);
lean_inc(v_currNamespace_1960_);
lean_inc(v_maxRecDepth_1958_);
lean_inc(v_currRecDepth_1957_);
lean_inc_ref(v_options_1956_);
lean_inc_ref(v_fileMap_1955_);
lean_inc_ref(v_fileName_1954_);
v___x_1973_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1973_, 0, v_fileName_1954_);
lean_ctor_set(v___x_1973_, 1, v_fileMap_1955_);
lean_ctor_set(v___x_1973_, 2, v_options_1956_);
lean_ctor_set(v___x_1973_, 3, v_currRecDepth_1957_);
lean_ctor_set(v___x_1973_, 4, v_maxRecDepth_1958_);
lean_ctor_set(v___x_1973_, 5, v_ref_1972_);
lean_ctor_set(v___x_1973_, 6, v_currNamespace_1960_);
lean_ctor_set(v___x_1973_, 7, v_openDecls_1961_);
lean_ctor_set(v___x_1973_, 8, v_initHeartbeats_1962_);
lean_ctor_set(v___x_1973_, 9, v_maxHeartbeats_1963_);
lean_ctor_set(v___x_1973_, 10, v_quotContext_1964_);
lean_ctor_set(v___x_1973_, 11, v_currMacroScope_1965_);
lean_ctor_set(v___x_1973_, 12, v_cancelTk_x3f_1967_);
lean_ctor_set(v___x_1973_, 13, v_inheritedTraceOptions_1969_);
lean_ctor_set_uint8(v___x_1973_, sizeof(void*)*14, v_diag_1966_);
lean_ctor_set_uint8(v___x_1973_, sizeof(void*)*14 + 1, v_suppressElabErrors_1968_);
if (v___x_1971_ == 0)
{
lean_object* v___x_1974_; uint8_t v___x_1975_; 
v___x_1974_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Erase___x29___closed__1));
lean_inc(v_stx_1945_);
v___x_1975_ = l_Lean_Syntax_isOfKind(v_stx_1945_, v___x_1974_);
if (v___x_1975_ == 0)
{
lean_object* v___x_1976_; uint8_t v___x_1977_; 
lean_dec(v_goal_1943_);
v___x_1976_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Rule__sets_x3a_x3d_x5b___x5d_x29___closed__1));
lean_inc(v_stx_1945_);
v___x_1977_ = l_Lean_Syntax_isOfKind(v_stx_1945_, v___x_1976_);
if (v___x_1977_ == 0)
{
lean_object* v___x_1978_; uint8_t v___x_1979_; 
v___x_1978_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Config_x3a_x3d___x29___closed__1));
lean_inc(v_stx_1945_);
v___x_1979_ = l_Lean_Syntax_isOfKind(v_stx_1945_, v___x_1978_);
if (v___x_1979_ == 0)
{
lean_object* v___x_1980_; uint8_t v___x_1981_; 
lean_dec_ref_known(v___x_1973_, 14);
v___x_1980_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_tactic__clause_x28Simp__config_x3a_x3d___x29___closed__1));
lean_inc(v_stx_1945_);
v___x_1981_ = l_Lean_Syntax_isOfKind(v_stx_1945_, v___x_1980_);
if (v___x_1981_ == 0)
{
lean_object* v___x_1982_; 
lean_dec(v_stx_1945_);
v___x_1982_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg();
return v___x_1982_;
}
else
{
lean_object* v___x_1983_; lean_object* v_additionalRules_1984_; lean_object* v_erasedRules_1985_; lean_object* v_enabledRuleSets_1986_; lean_object* v_options_1987_; lean_object* v_simpConfig_1988_; lean_object* v___x_1990_; uint8_t v_isShared_1991_; uint8_t v_isSharedCheck_2001_; 
v___x_1983_ = lean_st_ref_take(v_a_1946_);
v_additionalRules_1984_ = lean_ctor_get(v___x_1983_, 0);
v_erasedRules_1985_ = lean_ctor_get(v___x_1983_, 1);
v_enabledRuleSets_1986_ = lean_ctor_get(v___x_1983_, 2);
v_options_1987_ = lean_ctor_get(v___x_1983_, 3);
v_simpConfig_1988_ = lean_ctor_get(v___x_1983_, 4);
v_isSharedCheck_2001_ = !lean_is_exclusive(v___x_1983_);
if (v_isSharedCheck_2001_ == 0)
{
lean_object* v_unused_2002_; 
v_unused_2002_ = lean_ctor_get(v___x_1983_, 5);
lean_dec(v_unused_2002_);
v___x_1990_ = v___x_1983_;
v_isShared_1991_ = v_isSharedCheck_2001_;
goto v_resetjp_1989_;
}
else
{
lean_inc(v_simpConfig_1988_);
lean_inc(v_options_1987_);
lean_inc(v_enabledRuleSets_1986_);
lean_inc(v_erasedRules_1985_);
lean_inc(v_additionalRules_1984_);
lean_dec(v___x_1983_);
v___x_1990_ = lean_box(0);
v_isShared_1991_ = v_isSharedCheck_2001_;
goto v_resetjp_1989_;
}
v_resetjp_1989_:
{
lean_object* v___x_1992_; lean_object* v_t_1993_; lean_object* v___x_1994_; lean_object* v___x_1996_; 
v___x_1992_ = lean_unsigned_to_nat(3u);
v_t_1993_ = l_Lean_Syntax_getArg(v_stx_1945_, v___x_1992_);
lean_dec(v_stx_1945_);
v___x_1994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1994_, 0, v_t_1993_);
if (v_isShared_1991_ == 0)
{
lean_ctor_set(v___x_1990_, 5, v___x_1994_);
v___x_1996_ = v___x_1990_;
goto v_reusejp_1995_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v_additionalRules_1984_);
lean_ctor_set(v_reuseFailAlloc_2000_, 1, v_erasedRules_1985_);
lean_ctor_set(v_reuseFailAlloc_2000_, 2, v_enabledRuleSets_1986_);
lean_ctor_set(v_reuseFailAlloc_2000_, 3, v_options_1987_);
lean_ctor_set(v_reuseFailAlloc_2000_, 4, v_simpConfig_1988_);
lean_ctor_set(v_reuseFailAlloc_2000_, 5, v___x_1994_);
v___x_1996_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1995_;
}
v_reusejp_1995_:
{
lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; 
v___x_1997_ = lean_st_ref_set(v_a_1946_, v___x_1996_);
v___x_1998_ = lean_box(0);
v___x_1999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1999_, 0, v___x_1998_);
return v___x_1999_;
}
}
}
}
else
{
lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; 
v___x_2003_ = lean_unsigned_to_nat(3u);
v___x_2004_ = l_Lean_Syntax_getArg(v_stx_1945_, v___x_2003_);
lean_dec(v_stx_1945_);
v___x_2005_ = lp_aesop_Aesop_Frontend_elabOptions(v___x_2004_, v_a_1947_, v_a_1948_, v_a_1949_, v_a_1950_, v___x_1973_, v_a_1952_);
lean_dec_ref_known(v___x_1973_, 14);
if (lean_obj_tag(v___x_2005_) == 0)
{
lean_object* v_a_2006_; lean_object* v___x_2008_; uint8_t v_isShared_2009_; uint8_t v_isSharedCheck_2055_; 
v_a_2006_ = lean_ctor_get(v___x_2005_, 0);
v_isSharedCheck_2055_ = !lean_is_exclusive(v___x_2005_);
if (v_isSharedCheck_2055_ == 0)
{
v___x_2008_ = v___x_2005_;
v_isShared_2009_ = v_isSharedCheck_2055_;
goto v_resetjp_2007_;
}
else
{
lean_inc(v_a_2006_);
lean_dec(v___x_2005_);
v___x_2008_ = lean_box(0);
v_isShared_2009_ = v_isSharedCheck_2055_;
goto v_resetjp_2007_;
}
v_resetjp_2007_:
{
uint8_t v_strategy_2010_; lean_object* v_maxRuleApplicationDepth_2011_; lean_object* v_maxRuleApplications_2012_; lean_object* v_maxGoals_2013_; lean_object* v_maxNormIterations_2014_; lean_object* v_maxSafePrefixRuleApplications_2015_; uint8_t v_applyHypsTransparency_2016_; uint8_t v_assumptionTransparency_2017_; uint8_t v_destructProductsTransparency_2018_; lean_object* v_introsTransparency_x3f_2019_; uint8_t v_terminal_2020_; uint8_t v_warnOnNonterminal_2021_; uint8_t v_traceScript_2022_; uint8_t v_enableSimp_2023_; uint8_t v_useSimpAll_2024_; uint8_t v_useDefaultSimpSet_2025_; uint8_t v_enableUnfold_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2054_; 
v_strategy_2010_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6);
v_maxRuleApplicationDepth_2011_ = lean_ctor_get(v_a_2006_, 0);
v_maxRuleApplications_2012_ = lean_ctor_get(v_a_2006_, 1);
v_maxGoals_2013_ = lean_ctor_get(v_a_2006_, 2);
v_maxNormIterations_2014_ = lean_ctor_get(v_a_2006_, 3);
v_maxSafePrefixRuleApplications_2015_ = lean_ctor_get(v_a_2006_, 4);
v_applyHypsTransparency_2016_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 1);
v_assumptionTransparency_2017_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 2);
v_destructProductsTransparency_2018_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 3);
v_introsTransparency_x3f_2019_ = lean_ctor_get(v_a_2006_, 5);
v_terminal_2020_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 4);
v_warnOnNonterminal_2021_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 5);
v_traceScript_2022_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 6);
v_enableSimp_2023_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 7);
v_useSimpAll_2024_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 8);
v_useDefaultSimpSet_2025_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 9);
v_enableUnfold_2026_ = lean_ctor_get_uint8(v_a_2006_, sizeof(void*)*6 + 10);
v_isSharedCheck_2054_ = !lean_is_exclusive(v_a_2006_);
if (v_isSharedCheck_2054_ == 0)
{
v___x_2028_ = v_a_2006_;
v_isShared_2029_ = v_isSharedCheck_2054_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_introsTransparency_x3f_2019_);
lean_inc(v_maxSafePrefixRuleApplications_2015_);
lean_inc(v_maxNormIterations_2014_);
lean_inc(v_maxGoals_2013_);
lean_inc(v_maxRuleApplications_2012_);
lean_inc(v_maxRuleApplicationDepth_2011_);
lean_dec(v_a_2006_);
v___x_2028_ = lean_box(0);
v_isShared_2029_ = v_isSharedCheck_2054_;
goto v_resetjp_2027_;
}
v_resetjp_2027_:
{
uint8_t v___y_2031_; 
if (v_traceScript_2022_ == 0)
{
v___y_2031_ = v_traceScript_1944_;
goto v___jp_2030_;
}
else
{
v___y_2031_ = v_traceScript_2022_;
goto v___jp_2030_;
}
v___jp_2030_:
{
lean_object* v___x_2032_; lean_object* v_additionalRules_2033_; lean_object* v_erasedRules_2034_; lean_object* v_enabledRuleSets_2035_; lean_object* v_simpConfig_2036_; lean_object* v_simpConfigSyntax_x3f_2037_; lean_object* v___x_2039_; uint8_t v_isShared_2040_; uint8_t v_isSharedCheck_2052_; 
v___x_2032_ = lean_st_ref_take(v_a_1946_);
v_additionalRules_2033_ = lean_ctor_get(v___x_2032_, 0);
v_erasedRules_2034_ = lean_ctor_get(v___x_2032_, 1);
v_enabledRuleSets_2035_ = lean_ctor_get(v___x_2032_, 2);
v_simpConfig_2036_ = lean_ctor_get(v___x_2032_, 4);
v_simpConfigSyntax_x3f_2037_ = lean_ctor_get(v___x_2032_, 5);
v_isSharedCheck_2052_ = !lean_is_exclusive(v___x_2032_);
if (v_isSharedCheck_2052_ == 0)
{
lean_object* v_unused_2053_; 
v_unused_2053_ = lean_ctor_get(v___x_2032_, 3);
lean_dec(v_unused_2053_);
v___x_2039_ = v___x_2032_;
v_isShared_2040_ = v_isSharedCheck_2052_;
goto v_resetjp_2038_;
}
else
{
lean_inc(v_simpConfigSyntax_x3f_2037_);
lean_inc(v_simpConfig_2036_);
lean_inc(v_enabledRuleSets_2035_);
lean_inc(v_erasedRules_2034_);
lean_inc(v_additionalRules_2033_);
lean_dec(v___x_2032_);
v___x_2039_ = lean_box(0);
v_isShared_2040_ = v_isSharedCheck_2052_;
goto v_resetjp_2038_;
}
v_resetjp_2038_:
{
lean_object* v___x_2042_; 
if (v_isShared_2029_ == 0)
{
v___x_2042_ = v___x_2028_;
goto v_reusejp_2041_;
}
else
{
lean_object* v_reuseFailAlloc_2051_; 
v_reuseFailAlloc_2051_ = lean_alloc_ctor(0, 6, 11);
lean_ctor_set(v_reuseFailAlloc_2051_, 0, v_maxRuleApplicationDepth_2011_);
lean_ctor_set(v_reuseFailAlloc_2051_, 1, v_maxRuleApplications_2012_);
lean_ctor_set(v_reuseFailAlloc_2051_, 2, v_maxGoals_2013_);
lean_ctor_set(v_reuseFailAlloc_2051_, 3, v_maxNormIterations_2014_);
lean_ctor_set(v_reuseFailAlloc_2051_, 4, v_maxSafePrefixRuleApplications_2015_);
lean_ctor_set(v_reuseFailAlloc_2051_, 5, v_introsTransparency_x3f_2019_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6, v_strategy_2010_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 1, v_applyHypsTransparency_2016_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 2, v_assumptionTransparency_2017_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 3, v_destructProductsTransparency_2018_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 4, v_terminal_2020_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 5, v_warnOnNonterminal_2021_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 7, v_enableSimp_2023_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 8, v_useSimpAll_2024_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 9, v_useDefaultSimpSet_2025_);
lean_ctor_set_uint8(v_reuseFailAlloc_2051_, sizeof(void*)*6 + 10, v_enableUnfold_2026_);
v___x_2042_ = v_reuseFailAlloc_2051_;
goto v_reusejp_2041_;
}
v_reusejp_2041_:
{
lean_object* v___x_2044_; 
lean_ctor_set_uint8(v___x_2042_, sizeof(void*)*6 + 6, v___y_2031_);
if (v_isShared_2040_ == 0)
{
lean_ctor_set(v___x_2039_, 3, v___x_2042_);
v___x_2044_ = v___x_2039_;
goto v_reusejp_2043_;
}
else
{
lean_object* v_reuseFailAlloc_2050_; 
v_reuseFailAlloc_2050_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_2050_, 0, v_additionalRules_2033_);
lean_ctor_set(v_reuseFailAlloc_2050_, 1, v_erasedRules_2034_);
lean_ctor_set(v_reuseFailAlloc_2050_, 2, v_enabledRuleSets_2035_);
lean_ctor_set(v_reuseFailAlloc_2050_, 3, v___x_2042_);
lean_ctor_set(v_reuseFailAlloc_2050_, 4, v_simpConfig_2036_);
lean_ctor_set(v_reuseFailAlloc_2050_, 5, v_simpConfigSyntax_x3f_2037_);
v___x_2044_ = v_reuseFailAlloc_2050_;
goto v_reusejp_2043_;
}
v_reusejp_2043_:
{
lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2048_; 
v___x_2045_ = lean_st_ref_set(v_a_1946_, v___x_2044_);
v___x_2046_ = lean_box(0);
if (v_isShared_2009_ == 0)
{
lean_ctor_set(v___x_2008_, 0, v___x_2046_);
v___x_2048_ = v___x_2008_;
goto v_reusejp_2047_;
}
else
{
lean_object* v_reuseFailAlloc_2049_; 
v_reuseFailAlloc_2049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2049_, 0, v___x_2046_);
v___x_2048_ = v_reuseFailAlloc_2049_;
goto v_reusejp_2047_;
}
v_reusejp_2047_:
{
return v___x_2048_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_2056_; lean_object* v___x_2058_; uint8_t v_isShared_2059_; uint8_t v_isSharedCheck_2063_; 
v_a_2056_ = lean_ctor_get(v___x_2005_, 0);
v_isSharedCheck_2063_ = !lean_is_exclusive(v___x_2005_);
if (v_isSharedCheck_2063_ == 0)
{
v___x_2058_ = v___x_2005_;
v_isShared_2059_ = v_isSharedCheck_2063_;
goto v_resetjp_2057_;
}
else
{
lean_inc(v_a_2056_);
lean_dec(v___x_2005_);
v___x_2058_ = lean_box(0);
v_isShared_2059_ = v_isSharedCheck_2063_;
goto v_resetjp_2057_;
}
v_resetjp_2057_:
{
lean_object* v___x_2061_; 
if (v_isShared_2059_ == 0)
{
v___x_2061_ = v___x_2058_;
goto v_reusejp_2060_;
}
else
{
lean_object* v_reuseFailAlloc_2062_; 
v_reuseFailAlloc_2062_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2062_, 0, v_a_2056_);
v___x_2061_ = v_reuseFailAlloc_2062_;
goto v_reusejp_2060_;
}
v_reusejp_2060_:
{
return v___x_2061_;
}
}
}
}
}
else
{
lean_object* v___x_2064_; lean_object* v_enabledRuleSets_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v_specs_2068_; lean_object* v___x_2069_; size_t v_sz_2070_; size_t v___x_2071_; lean_object* v___x_2072_; 
v___x_2064_ = lean_st_ref_get(v_a_1946_);
v_enabledRuleSets_2065_ = lean_ctor_get(v___x_2064_, 2);
lean_inc_ref(v_enabledRuleSets_2065_);
lean_dec(v___x_2064_);
v___x_2066_ = lean_unsigned_to_nat(4u);
v___x_2067_ = l_Lean_Syntax_getArg(v_stx_1945_, v___x_2066_);
lean_dec(v_stx_1945_);
v_specs_2068_ = l_Lean_Syntax_getArgs(v___x_2067_);
lean_dec(v___x_2067_);
v___x_2069_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_specs_2068_);
lean_dec_ref(v_specs_2068_);
v_sz_2070_ = lean_array_size(v___x_2069_);
v___x_2071_ = ((size_t)0ULL);
v___x_2072_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__5(v___x_2069_, v_sz_2070_, v___x_2071_, v_enabledRuleSets_2065_, v_a_1946_, v_a_1947_, v_a_1948_, v_a_1949_, v_a_1950_, v___x_1973_, v_a_1952_);
lean_dec_ref_known(v___x_1973_, 14);
lean_dec_ref(v___x_2069_);
if (lean_obj_tag(v___x_2072_) == 0)
{
lean_object* v_a_2073_; lean_object* v___x_2075_; uint8_t v_isShared_2076_; uint8_t v_isSharedCheck_2096_; 
v_a_2073_ = lean_ctor_get(v___x_2072_, 0);
v_isSharedCheck_2096_ = !lean_is_exclusive(v___x_2072_);
if (v_isSharedCheck_2096_ == 0)
{
v___x_2075_ = v___x_2072_;
v_isShared_2076_ = v_isSharedCheck_2096_;
goto v_resetjp_2074_;
}
else
{
lean_inc(v_a_2073_);
lean_dec(v___x_2072_);
v___x_2075_ = lean_box(0);
v_isShared_2076_ = v_isSharedCheck_2096_;
goto v_resetjp_2074_;
}
v_resetjp_2074_:
{
lean_object* v___x_2077_; lean_object* v_additionalRules_2078_; lean_object* v_erasedRules_2079_; lean_object* v_options_2080_; lean_object* v_simpConfig_2081_; lean_object* v_simpConfigSyntax_x3f_2082_; lean_object* v___x_2084_; uint8_t v_isShared_2085_; uint8_t v_isSharedCheck_2094_; 
v___x_2077_ = lean_st_ref_take(v_a_1946_);
v_additionalRules_2078_ = lean_ctor_get(v___x_2077_, 0);
v_erasedRules_2079_ = lean_ctor_get(v___x_2077_, 1);
v_options_2080_ = lean_ctor_get(v___x_2077_, 3);
v_simpConfig_2081_ = lean_ctor_get(v___x_2077_, 4);
v_simpConfigSyntax_x3f_2082_ = lean_ctor_get(v___x_2077_, 5);
v_isSharedCheck_2094_ = !lean_is_exclusive(v___x_2077_);
if (v_isSharedCheck_2094_ == 0)
{
lean_object* v_unused_2095_; 
v_unused_2095_ = lean_ctor_get(v___x_2077_, 2);
lean_dec(v_unused_2095_);
v___x_2084_ = v___x_2077_;
v_isShared_2085_ = v_isSharedCheck_2094_;
goto v_resetjp_2083_;
}
else
{
lean_inc(v_simpConfigSyntax_x3f_2082_);
lean_inc(v_simpConfig_2081_);
lean_inc(v_options_2080_);
lean_inc(v_erasedRules_2079_);
lean_inc(v_additionalRules_2078_);
lean_dec(v___x_2077_);
v___x_2084_ = lean_box(0);
v_isShared_2085_ = v_isSharedCheck_2094_;
goto v_resetjp_2083_;
}
v_resetjp_2083_:
{
lean_object* v___x_2087_; 
if (v_isShared_2085_ == 0)
{
lean_ctor_set(v___x_2084_, 2, v_a_2073_);
v___x_2087_ = v___x_2084_;
goto v_reusejp_2086_;
}
else
{
lean_object* v_reuseFailAlloc_2093_; 
v_reuseFailAlloc_2093_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_2093_, 0, v_additionalRules_2078_);
lean_ctor_set(v_reuseFailAlloc_2093_, 1, v_erasedRules_2079_);
lean_ctor_set(v_reuseFailAlloc_2093_, 2, v_a_2073_);
lean_ctor_set(v_reuseFailAlloc_2093_, 3, v_options_2080_);
lean_ctor_set(v_reuseFailAlloc_2093_, 4, v_simpConfig_2081_);
lean_ctor_set(v_reuseFailAlloc_2093_, 5, v_simpConfigSyntax_x3f_2082_);
v___x_2087_ = v_reuseFailAlloc_2093_;
goto v_reusejp_2086_;
}
v_reusejp_2086_:
{
lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2091_; 
v___x_2088_ = lean_st_ref_set(v_a_1946_, v___x_2087_);
v___x_2089_ = lean_box(0);
if (v_isShared_2076_ == 0)
{
lean_ctor_set(v___x_2075_, 0, v___x_2089_);
v___x_2091_ = v___x_2075_;
goto v_reusejp_2090_;
}
else
{
lean_object* v_reuseFailAlloc_2092_; 
v_reuseFailAlloc_2092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2092_, 0, v___x_2089_);
v___x_2091_ = v_reuseFailAlloc_2092_;
goto v_reusejp_2090_;
}
v_reusejp_2090_:
{
return v___x_2091_;
}
}
}
}
}
else
{
lean_object* v_a_2097_; lean_object* v___x_2099_; uint8_t v_isShared_2100_; uint8_t v_isSharedCheck_2104_; 
v_a_2097_ = lean_ctor_get(v___x_2072_, 0);
v_isSharedCheck_2104_ = !lean_is_exclusive(v___x_2072_);
if (v_isSharedCheck_2104_ == 0)
{
v___x_2099_ = v___x_2072_;
v_isShared_2100_ = v_isSharedCheck_2104_;
goto v_resetjp_2098_;
}
else
{
lean_inc(v_a_2097_);
lean_dec(v___x_2072_);
v___x_2099_ = lean_box(0);
v_isShared_2100_ = v_isSharedCheck_2104_;
goto v_resetjp_2098_;
}
v_resetjp_2098_:
{
lean_object* v___x_2102_; 
if (v_isShared_2100_ == 0)
{
v___x_2102_ = v___x_2099_;
goto v_reusejp_2101_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v_a_2097_);
v___x_2102_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2101_;
}
v_reusejp_2101_:
{
return v___x_2102_;
}
}
}
}
}
else
{
lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v_es_2107_; lean_object* v___x_2108_; size_t v_sz_2109_; size_t v___x_2110_; lean_object* v___x_2111_; 
v___x_2105_ = lean_unsigned_to_nat(2u);
v___x_2106_ = l_Lean_Syntax_getArg(v_stx_1945_, v___x_2105_);
lean_dec(v_stx_1945_);
v_es_2107_ = l_Lean_Syntax_getArgs(v___x_2106_);
lean_dec(v___x_2106_);
v___x_2108_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_es_2107_);
lean_dec_ref(v_es_2107_);
v_sz_2109_ = lean_array_size(v___x_2108_);
v___x_2110_ = ((size_t)0ULL);
v___x_2111_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg(v_goal_1943_, v_sz_2109_, v___x_2110_, v___x_2108_, v_a_1947_, v_a_1948_, v_a_1949_, v_a_1950_, v___x_1973_, v_a_1952_);
lean_dec_ref_known(v___x_1973_, 14);
if (lean_obj_tag(v___x_2111_) == 0)
{
lean_object* v_a_2112_; lean_object* v___x_2114_; uint8_t v_isShared_2115_; uint8_t v_isSharedCheck_2136_; 
v_a_2112_ = lean_ctor_get(v___x_2111_, 0);
v_isSharedCheck_2136_ = !lean_is_exclusive(v___x_2111_);
if (v_isSharedCheck_2136_ == 0)
{
v___x_2114_ = v___x_2111_;
v_isShared_2115_ = v_isSharedCheck_2136_;
goto v_resetjp_2113_;
}
else
{
lean_inc(v_a_2112_);
lean_dec(v___x_2111_);
v___x_2114_ = lean_box(0);
v_isShared_2115_ = v_isSharedCheck_2136_;
goto v_resetjp_2113_;
}
v_resetjp_2113_:
{
lean_object* v___x_2116_; lean_object* v_additionalRules_2117_; lean_object* v_erasedRules_2118_; lean_object* v_enabledRuleSets_2119_; lean_object* v_options_2120_; lean_object* v_simpConfig_2121_; lean_object* v_simpConfigSyntax_x3f_2122_; lean_object* v___x_2124_; uint8_t v_isShared_2125_; uint8_t v_isSharedCheck_2135_; 
v___x_2116_ = lean_st_ref_take(v_a_1946_);
v_additionalRules_2117_ = lean_ctor_get(v___x_2116_, 0);
v_erasedRules_2118_ = lean_ctor_get(v___x_2116_, 1);
v_enabledRuleSets_2119_ = lean_ctor_get(v___x_2116_, 2);
v_options_2120_ = lean_ctor_get(v___x_2116_, 3);
v_simpConfig_2121_ = lean_ctor_get(v___x_2116_, 4);
v_simpConfigSyntax_x3f_2122_ = lean_ctor_get(v___x_2116_, 5);
v_isSharedCheck_2135_ = !lean_is_exclusive(v___x_2116_);
if (v_isSharedCheck_2135_ == 0)
{
v___x_2124_ = v___x_2116_;
v_isShared_2125_ = v_isSharedCheck_2135_;
goto v_resetjp_2123_;
}
else
{
lean_inc(v_simpConfigSyntax_x3f_2122_);
lean_inc(v_simpConfig_2121_);
lean_inc(v_options_2120_);
lean_inc(v_enabledRuleSets_2119_);
lean_inc(v_erasedRules_2118_);
lean_inc(v_additionalRules_2117_);
lean_dec(v___x_2116_);
v___x_2124_ = lean_box(0);
v_isShared_2125_ = v_isSharedCheck_2135_;
goto v_resetjp_2123_;
}
v_resetjp_2123_:
{
lean_object* v___x_2126_; lean_object* v___x_2128_; 
v___x_2126_ = l_Array_append___redArg(v_erasedRules_2118_, v_a_2112_);
lean_dec(v_a_2112_);
if (v_isShared_2125_ == 0)
{
lean_ctor_set(v___x_2124_, 1, v___x_2126_);
v___x_2128_ = v___x_2124_;
goto v_reusejp_2127_;
}
else
{
lean_object* v_reuseFailAlloc_2134_; 
v_reuseFailAlloc_2134_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_2134_, 0, v_additionalRules_2117_);
lean_ctor_set(v_reuseFailAlloc_2134_, 1, v___x_2126_);
lean_ctor_set(v_reuseFailAlloc_2134_, 2, v_enabledRuleSets_2119_);
lean_ctor_set(v_reuseFailAlloc_2134_, 3, v_options_2120_);
lean_ctor_set(v_reuseFailAlloc_2134_, 4, v_simpConfig_2121_);
lean_ctor_set(v_reuseFailAlloc_2134_, 5, v_simpConfigSyntax_x3f_2122_);
v___x_2128_ = v_reuseFailAlloc_2134_;
goto v_reusejp_2127_;
}
v_reusejp_2127_:
{
lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2132_; 
v___x_2129_ = lean_st_ref_set(v_a_1946_, v___x_2128_);
v___x_2130_ = lean_box(0);
if (v_isShared_2115_ == 0)
{
lean_ctor_set(v___x_2114_, 0, v___x_2130_);
v___x_2132_ = v___x_2114_;
goto v_reusejp_2131_;
}
else
{
lean_object* v_reuseFailAlloc_2133_; 
v_reuseFailAlloc_2133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2133_, 0, v___x_2130_);
v___x_2132_ = v_reuseFailAlloc_2133_;
goto v_reusejp_2131_;
}
v_reusejp_2131_:
{
return v___x_2132_;
}
}
}
}
}
else
{
lean_object* v_a_2137_; lean_object* v___x_2139_; uint8_t v_isShared_2140_; uint8_t v_isSharedCheck_2144_; 
v_a_2137_ = lean_ctor_get(v___x_2111_, 0);
v_isSharedCheck_2144_ = !lean_is_exclusive(v___x_2111_);
if (v_isSharedCheck_2144_ == 0)
{
v___x_2139_ = v___x_2111_;
v_isShared_2140_ = v_isSharedCheck_2144_;
goto v_resetjp_2138_;
}
else
{
lean_inc(v_a_2137_);
lean_dec(v___x_2111_);
v___x_2139_ = lean_box(0);
v_isShared_2140_ = v_isSharedCheck_2144_;
goto v_resetjp_2138_;
}
v_resetjp_2138_:
{
lean_object* v___x_2142_; 
if (v_isShared_2140_ == 0)
{
v___x_2142_ = v___x_2139_;
goto v_reusejp_2141_;
}
else
{
lean_object* v_reuseFailAlloc_2143_; 
v_reuseFailAlloc_2143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2143_, 0, v_a_2137_);
v___x_2142_ = v_reuseFailAlloc_2143_;
goto v_reusejp_2141_;
}
v_reusejp_2141_:
{
return v___x_2142_;
}
}
}
}
}
else
{
lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v_es_2147_; lean_object* v___x_2148_; size_t v_sz_2149_; size_t v___x_2150_; lean_object* v___x_2151_; 
v___x_2145_ = lean_unsigned_to_nat(2u);
v___x_2146_ = l_Lean_Syntax_getArg(v_stx_1945_, v___x_2145_);
lean_dec(v_stx_1945_);
v_es_2147_ = l_Lean_Syntax_getArgs(v___x_2146_);
lean_dec(v___x_2146_);
v___x_2148_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_es_2147_);
lean_dec_ref(v_es_2147_);
v_sz_2149_ = lean_array_size(v___x_2148_);
v___x_2150_ = ((size_t)0ULL);
v___x_2151_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg(v_goal_1943_, v_sz_2149_, v___x_2150_, v___x_2148_, v_a_1947_, v_a_1948_, v_a_1949_, v_a_1950_, v___x_1973_, v_a_1952_);
lean_dec_ref_known(v___x_1973_, 14);
if (lean_obj_tag(v___x_2151_) == 0)
{
lean_object* v_a_2152_; lean_object* v___x_2154_; uint8_t v_isShared_2155_; uint8_t v_isSharedCheck_2176_; 
v_a_2152_ = lean_ctor_get(v___x_2151_, 0);
v_isSharedCheck_2176_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2176_ == 0)
{
v___x_2154_ = v___x_2151_;
v_isShared_2155_ = v_isSharedCheck_2176_;
goto v_resetjp_2153_;
}
else
{
lean_inc(v_a_2152_);
lean_dec(v___x_2151_);
v___x_2154_ = lean_box(0);
v_isShared_2155_ = v_isSharedCheck_2176_;
goto v_resetjp_2153_;
}
v_resetjp_2153_:
{
lean_object* v___x_2156_; lean_object* v_additionalRules_2157_; lean_object* v_erasedRules_2158_; lean_object* v_enabledRuleSets_2159_; lean_object* v_options_2160_; lean_object* v_simpConfig_2161_; lean_object* v_simpConfigSyntax_x3f_2162_; lean_object* v___x_2164_; uint8_t v_isShared_2165_; uint8_t v_isSharedCheck_2175_; 
v___x_2156_ = lean_st_ref_take(v_a_1946_);
v_additionalRules_2157_ = lean_ctor_get(v___x_2156_, 0);
v_erasedRules_2158_ = lean_ctor_get(v___x_2156_, 1);
v_enabledRuleSets_2159_ = lean_ctor_get(v___x_2156_, 2);
v_options_2160_ = lean_ctor_get(v___x_2156_, 3);
v_simpConfig_2161_ = lean_ctor_get(v___x_2156_, 4);
v_simpConfigSyntax_x3f_2162_ = lean_ctor_get(v___x_2156_, 5);
v_isSharedCheck_2175_ = !lean_is_exclusive(v___x_2156_);
if (v_isSharedCheck_2175_ == 0)
{
v___x_2164_ = v___x_2156_;
v_isShared_2165_ = v_isSharedCheck_2175_;
goto v_resetjp_2163_;
}
else
{
lean_inc(v_simpConfigSyntax_x3f_2162_);
lean_inc(v_simpConfig_2161_);
lean_inc(v_options_2160_);
lean_inc(v_enabledRuleSets_2159_);
lean_inc(v_erasedRules_2158_);
lean_inc(v_additionalRules_2157_);
lean_dec(v___x_2156_);
v___x_2164_ = lean_box(0);
v_isShared_2165_ = v_isSharedCheck_2175_;
goto v_resetjp_2163_;
}
v_resetjp_2163_:
{
lean_object* v___x_2166_; lean_object* v___x_2168_; 
v___x_2166_ = l_Array_append___redArg(v_additionalRules_2157_, v_a_2152_);
lean_dec(v_a_2152_);
if (v_isShared_2165_ == 0)
{
lean_ctor_set(v___x_2164_, 0, v___x_2166_);
v___x_2168_ = v___x_2164_;
goto v_reusejp_2167_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v___x_2166_);
lean_ctor_set(v_reuseFailAlloc_2174_, 1, v_erasedRules_2158_);
lean_ctor_set(v_reuseFailAlloc_2174_, 2, v_enabledRuleSets_2159_);
lean_ctor_set(v_reuseFailAlloc_2174_, 3, v_options_2160_);
lean_ctor_set(v_reuseFailAlloc_2174_, 4, v_simpConfig_2161_);
lean_ctor_set(v_reuseFailAlloc_2174_, 5, v_simpConfigSyntax_x3f_2162_);
v___x_2168_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2167_;
}
v_reusejp_2167_:
{
lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2172_; 
v___x_2169_ = lean_st_ref_set(v_a_1946_, v___x_2168_);
v___x_2170_ = lean_box(0);
if (v_isShared_2155_ == 0)
{
lean_ctor_set(v___x_2154_, 0, v___x_2170_);
v___x_2172_ = v___x_2154_;
goto v_reusejp_2171_;
}
else
{
lean_object* v_reuseFailAlloc_2173_; 
v_reuseFailAlloc_2173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2173_, 0, v___x_2170_);
v___x_2172_ = v_reuseFailAlloc_2173_;
goto v_reusejp_2171_;
}
v_reusejp_2171_:
{
return v___x_2172_;
}
}
}
}
}
else
{
lean_object* v_a_2177_; lean_object* v___x_2179_; uint8_t v_isShared_2180_; uint8_t v_isSharedCheck_2184_; 
v_a_2177_ = lean_ctor_get(v___x_2151_, 0);
v_isSharedCheck_2184_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2184_ == 0)
{
v___x_2179_ = v___x_2151_;
v_isShared_2180_ = v_isSharedCheck_2184_;
goto v_resetjp_2178_;
}
else
{
lean_inc(v_a_2177_);
lean_dec(v___x_2151_);
v___x_2179_ = lean_box(0);
v_isShared_2180_ = v_isSharedCheck_2184_;
goto v_resetjp_2178_;
}
v_resetjp_2178_:
{
lean_object* v___x_2182_; 
if (v_isShared_2180_ == 0)
{
v___x_2182_ = v___x_2179_;
goto v_reusejp_2181_;
}
else
{
lean_object* v_reuseFailAlloc_2183_; 
v_reuseFailAlloc_2183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2183_, 0, v_a_2177_);
v___x_2182_ = v_reuseFailAlloc_2183_;
goto v_reusejp_2181_;
}
v_reusejp_2181_:
{
return v___x_2182_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause___boxed(lean_object* v_goal_2185_, lean_object* v_traceScript_2186_, lean_object* v_stx_2187_, lean_object* v_a_2188_, lean_object* v_a_2189_, lean_object* v_a_2190_, lean_object* v_a_2191_, lean_object* v_a_2192_, lean_object* v_a_2193_, lean_object* v_a_2194_, lean_object* v_a_2195_){
_start:
{
uint8_t v_traceScript_boxed_2196_; lean_object* v_res_2197_; 
v_traceScript_boxed_2196_ = lean_unbox(v_traceScript_2186_);
v_res_2197_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause(v_goal_2185_, v_traceScript_boxed_2196_, v_stx_2187_, v_a_2188_, v_a_2189_, v_a_2190_, v_a_2191_, v_a_2192_, v_a_2193_, v_a_2194_);
lean_dec(v_a_2194_);
lean_dec_ref(v_a_2193_);
lean_dec(v_a_2192_);
lean_dec_ref(v_a_2191_);
lean_dec(v_a_2190_);
lean_dec_ref(v_a_2189_);
lean_dec(v_a_2188_);
return v_res_2197_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1(lean_object* v_00_u03b2_2198_, lean_object* v_m_2199_, lean_object* v_a_2200_, lean_object* v_b_2201_){
_start:
{
lean_object* v___x_2202_; 
v___x_2202_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1___redArg(v_m_2199_, v_a_2200_, v_b_2201_);
return v___x_2202_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2(lean_object* v_00_u03b2_2203_, lean_object* v_m_2204_, lean_object* v_a_2205_){
_start:
{
uint8_t v___x_2206_; 
v___x_2206_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___redArg(v_m_2204_, v_a_2205_);
return v___x_2206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2___boxed(lean_object* v_00_u03b2_2207_, lean_object* v_m_2208_, lean_object* v_a_2209_){
_start:
{
uint8_t v_res_2210_; lean_object* v_r_2211_; 
v_res_2210_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__2(v_00_u03b2_2207_, v_m_2208_, v_a_2209_);
lean_dec(v_a_2209_);
lean_dec_ref(v_m_2208_);
v_r_2211_ = lean_box(v_res_2210_);
return v_r_2211_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3(lean_object* v_00_u03b1_2212_, lean_object* v_msg_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_, lean_object* v___y_2216_, lean_object* v___y_2217_, lean_object* v___y_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_){
_start:
{
lean_object* v___x_2222_; 
v___x_2222_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___redArg(v_msg_2213_, v___y_2217_, v___y_2218_, v___y_2219_, v___y_2220_);
return v___x_2222_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3___boxed(lean_object* v_00_u03b1_2223_, lean_object* v_msg_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_, lean_object* v___y_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_){
_start:
{
lean_object* v_res_2233_; 
v_res_2233_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3(v_00_u03b1_2223_, v_msg_2224_, v___y_2225_, v___y_2226_, v___y_2227_, v___y_2228_, v___y_2229_, v___y_2230_, v___y_2231_);
lean_dec(v___y_2231_);
lean_dec_ref(v___y_2230_);
lean_dec(v___y_2229_);
lean_dec_ref(v___y_2228_);
lean_dec(v___y_2227_);
lean_dec_ref(v___y_2226_);
lean_dec(v___y_2225_);
return v_res_2233_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4(lean_object* v_00_u03b2_2234_, lean_object* v_m_2235_, lean_object* v_a_2236_){
_start:
{
lean_object* v___x_2237_; 
v___x_2237_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___redArg(v_m_2235_, v_a_2236_);
return v___x_2237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4___boxed(lean_object* v_00_u03b2_2238_, lean_object* v_m_2239_, lean_object* v_a_2240_){
_start:
{
lean_object* v_res_2241_; 
v_res_2241_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4(v_00_u03b2_2238_, v_m_2239_, v_a_2240_);
lean_dec(v_a_2240_);
return v_res_2241_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6(lean_object* v_goal_2242_, size_t v_sz_2243_, size_t v_i_2244_, lean_object* v_bs_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_, lean_object* v___y_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_){
_start:
{
lean_object* v___x_2254_; 
v___x_2254_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___redArg(v_goal_2242_, v_sz_2243_, v_i_2244_, v_bs_2245_, v___y_2247_, v___y_2248_, v___y_2249_, v___y_2250_, v___y_2251_, v___y_2252_);
return v___x_2254_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6___boxed(lean_object* v_goal_2255_, lean_object* v_sz_2256_, lean_object* v_i_2257_, lean_object* v_bs_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_){
_start:
{
size_t v_sz_boxed_2267_; size_t v_i_boxed_2268_; lean_object* v_res_2269_; 
v_sz_boxed_2267_ = lean_unbox_usize(v_sz_2256_);
lean_dec(v_sz_2256_);
v_i_boxed_2268_ = lean_unbox_usize(v_i_2257_);
lean_dec(v_i_2257_);
v_res_2269_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__6(v_goal_2255_, v_sz_boxed_2267_, v_i_boxed_2268_, v_bs_2258_, v___y_2259_, v___y_2260_, v___y_2261_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_);
lean_dec(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v___y_2263_);
lean_dec_ref(v___y_2262_);
lean_dec(v___y_2261_);
lean_dec_ref(v___y_2260_);
lean_dec(v___y_2259_);
return v_res_2269_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7(lean_object* v_goal_2270_, size_t v_sz_2271_, size_t v_i_2272_, lean_object* v_bs_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_){
_start:
{
lean_object* v___x_2282_; 
v___x_2282_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___redArg(v_goal_2270_, v_sz_2271_, v_i_2272_, v_bs_2273_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_, v___y_2280_);
return v___x_2282_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7___boxed(lean_object* v_goal_2283_, lean_object* v_sz_2284_, lean_object* v_i_2285_, lean_object* v_bs_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_, lean_object* v___y_2294_){
_start:
{
size_t v_sz_boxed_2295_; size_t v_i_boxed_2296_; lean_object* v_res_2297_; 
v_sz_boxed_2295_ = lean_unbox_usize(v_sz_2284_);
lean_dec(v_sz_2284_);
v_i_boxed_2296_ = lean_unbox_usize(v_i_2285_);
lean_dec(v_i_2285_);
v_res_2297_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__7(v_goal_2283_, v_sz_boxed_2295_, v_i_boxed_2296_, v_bs_2286_, v___y_2287_, v___y_2288_, v___y_2289_, v___y_2290_, v___y_2291_, v___y_2292_, v___y_2293_);
lean_dec(v___y_2293_);
lean_dec_ref(v___y_2292_);
lean_dec(v___y_2291_);
lean_dec_ref(v___y_2290_);
lean_dec(v___y_2289_);
lean_dec_ref(v___y_2288_);
lean_dec(v___y_2287_);
return v_res_2297_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1(lean_object* v_00_u03b2_2298_, lean_object* v_a_2299_, lean_object* v_x_2300_){
_start:
{
uint8_t v___x_2301_; 
v___x_2301_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___redArg(v_a_2299_, v_x_2300_);
return v___x_2301_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1___boxed(lean_object* v_00_u03b2_2302_, lean_object* v_a_2303_, lean_object* v_x_2304_){
_start:
{
uint8_t v_res_2305_; lean_object* v_r_2306_; 
v_res_2305_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__1(v_00_u03b2_2302_, v_a_2303_, v_x_2304_);
lean_dec(v_x_2304_);
lean_dec(v_a_2303_);
v_r_2306_ = lean_box(v_res_2305_);
return v_r_2306_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2(lean_object* v_00_u03b2_2307_, lean_object* v_data_2308_){
_start:
{
lean_object* v___x_2309_; 
v___x_2309_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2___redArg(v_data_2308_);
return v___x_2309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7(lean_object* v_00_u03b2_2310_, lean_object* v_a_2311_, lean_object* v_x_2312_){
_start:
{
lean_object* v___x_2313_; 
v___x_2313_ = lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___redArg(v_a_2311_, v_x_2312_);
return v___x_2313_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7___boxed(lean_object* v_00_u03b2_2314_, lean_object* v_a_2315_, lean_object* v_x_2316_){
_start:
{
lean_object* v_res_2317_; 
v_res_2317_ = lp_aesop_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__4_spec__7(v_00_u03b2_2314_, v_a_2315_, v_x_2316_);
lean_dec(v_a_2315_);
return v_res_2317_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_2318_, lean_object* v_i_2319_, lean_object* v_source_2320_, lean_object* v_target_2321_){
_start:
{
lean_object* v___x_2322_; 
v___x_2322_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3___redArg(v_i_2319_, v_source_2320_, v_target_2321_);
return v___x_2322_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3_spec__11(lean_object* v_00_u03b2_2323_, lean_object* v_x_2324_, lean_object* v_x_2325_){
_start:
{
lean_object* v___x_2326_; 
v___x_2326_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__1_spec__2_spec__3_spec__11___redArg(v_x_2324_, v_x_2325_);
return v___x_2326_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0(lean_object* v_goal_2327_, uint8_t v_traceScript_2328_, lean_object* v_as_2329_, size_t v_i_2330_, size_t v_stop_2331_, lean_object* v_b_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_){
_start:
{
uint8_t v___x_2341_; 
v___x_2341_ = lean_usize_dec_eq(v_i_2330_, v_stop_2331_);
if (v___x_2341_ == 0)
{
lean_object* v___x_2342_; lean_object* v___x_2343_; 
v___x_2342_ = lean_array_uget_borrowed(v_as_2329_, v_i_2330_);
lean_inc(v___x_2342_);
lean_inc(v_goal_2327_);
v___x_2343_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause(v_goal_2327_, v_traceScript_2328_, v___x_2342_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_, v___y_2337_, v___y_2338_, v___y_2339_);
if (lean_obj_tag(v___x_2343_) == 0)
{
lean_object* v_a_2344_; size_t v___x_2345_; size_t v___x_2346_; 
v_a_2344_ = lean_ctor_get(v___x_2343_, 0);
lean_inc(v_a_2344_);
lean_dec_ref_known(v___x_2343_, 1);
v___x_2345_ = ((size_t)1ULL);
v___x_2346_ = lean_usize_add(v_i_2330_, v___x_2345_);
v_i_2330_ = v___x_2346_;
v_b_2332_ = v_a_2344_;
goto _start;
}
else
{
lean_dec(v_goal_2327_);
return v___x_2343_;
}
}
else
{
lean_object* v___x_2348_; 
lean_dec(v_goal_2327_);
v___x_2348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2348_, 0, v_b_2332_);
return v___x_2348_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0___boxed(lean_object* v_goal_2349_, lean_object* v_traceScript_2350_, lean_object* v_as_2351_, lean_object* v_i_2352_, lean_object* v_stop_2353_, lean_object* v_b_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_, lean_object* v___y_2360_, lean_object* v___y_2361_, lean_object* v___y_2362_){
_start:
{
uint8_t v_traceScript_boxed_2363_; size_t v_i_boxed_2364_; size_t v_stop_boxed_2365_; lean_object* v_res_2366_; 
v_traceScript_boxed_2363_ = lean_unbox(v_traceScript_2350_);
v_i_boxed_2364_ = lean_unbox_usize(v_i_2352_);
lean_dec(v_i_2352_);
v_stop_boxed_2365_ = lean_unbox_usize(v_stop_2353_);
lean_dec(v_stop_2353_);
v_res_2366_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0(v_goal_2349_, v_traceScript_boxed_2363_, v_as_2351_, v_i_boxed_2364_, v_stop_boxed_2365_, v_b_2354_, v___y_2355_, v___y_2356_, v___y_2357_, v___y_2358_, v___y_2359_, v___y_2360_, v___y_2361_);
lean_dec(v___y_2361_);
lean_dec_ref(v___y_2360_);
lean_dec(v___y_2359_);
lean_dec_ref(v___y_2358_);
lean_dec(v___y_2357_);
lean_dec_ref(v___y_2356_);
lean_dec(v___y_2355_);
lean_dec_ref(v_as_2351_);
return v_res_2366_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go(lean_object* v_goal_2383_, uint8_t v_traceScript_2384_, lean_object* v_clauses_2385_, lean_object* v_a_2386_, lean_object* v_a_2387_, lean_object* v_a_2388_, lean_object* v_a_2389_, lean_object* v_a_2390_, lean_object* v_a_2391_){
_start:
{
lean_object* v_additionalRules_2394_; lean_object* v_erasedRules_2395_; lean_object* v_enabledRuleSets_2396_; lean_object* v_options_2397_; lean_object* v_simpConfigSyntax_x3f_2398_; lean_object* v_simpConfig_2399_; lean_object* v___x_2402_; 
v___x_2402_ = lp_aesop_Aesop_getDefaultRuleSetNames();
if (lean_obj_tag(v___x_2402_) == 0)
{
lean_object* v_a_2403_; lean_object* v___x_2404_; lean_object* v___x_2405_; uint8_t v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v___x_2409_; lean_object* v___x_2410_; uint8_t v___x_2411_; uint8_t v___x_2412_; lean_object* v___x_2413_; uint8_t v___x_2414_; uint8_t v___x_2415_; lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___y_2463_; lean_object* v___x_2472_; uint8_t v___x_2473_; 
v_a_2403_ = lean_ctor_get(v___x_2402_, 0);
lean_inc(v_a_2403_);
lean_dec_ref_known(v___x_2402_, 1);
v___x_2404_ = lean_unsigned_to_nat(0u);
v___x_2405_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__0));
v___x_2406_ = 0;
v___x_2407_ = lean_unsigned_to_nat(30u);
v___x_2408_ = lean_unsigned_to_nat(200u);
v___x_2409_ = lean_unsigned_to_nat(100u);
v___x_2410_ = lean_unsigned_to_nat(50u);
v___x_2411_ = 1;
v___x_2412_ = 2;
v___x_2413_ = lean_box(0);
v___x_2414_ = 0;
v___x_2415_ = 1;
v___x_2416_ = lean_alloc_ctor(0, 6, 11);
lean_ctor_set(v___x_2416_, 0, v___x_2407_);
lean_ctor_set(v___x_2416_, 1, v___x_2408_);
lean_ctor_set(v___x_2416_, 2, v___x_2404_);
lean_ctor_set(v___x_2416_, 3, v___x_2409_);
lean_ctor_set(v___x_2416_, 4, v___x_2410_);
lean_ctor_set(v___x_2416_, 5, v___x_2413_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6, v___x_2406_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 1, v___x_2411_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 2, v___x_2411_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 3, v___x_2412_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 4, v___x_2414_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 5, v___x_2415_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 6, v_traceScript_2384_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 7, v___x_2415_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 8, v___x_2415_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 9, v___x_2415_);
lean_ctor_set_uint8(v___x_2416_, sizeof(void*)*6 + 10, v___x_2415_);
v___x_2417_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__1));
v___x_2418_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2418_, 0, v___x_2405_);
lean_ctor_set(v___x_2418_, 1, v___x_2405_);
lean_ctor_set(v___x_2418_, 2, v_a_2403_);
lean_ctor_set(v___x_2418_, 3, v___x_2416_);
lean_ctor_set(v___x_2418_, 4, v___x_2417_);
lean_ctor_set(v___x_2418_, 5, v___x_2413_);
v___x_2419_ = lean_st_mk_ref(v___x_2418_);
v___x_2472_ = lean_array_get_size(v_clauses_2385_);
v___x_2473_ = lean_nat_dec_lt(v___x_2404_, v___x_2472_);
if (v___x_2473_ == 0)
{
lean_dec(v_goal_2383_);
goto v___jp_2420_;
}
else
{
lean_object* v___x_2474_; uint8_t v___x_2475_; 
v___x_2474_ = lean_box(0);
v___x_2475_ = lean_nat_dec_le(v___x_2472_, v___x_2472_);
if (v___x_2475_ == 0)
{
if (v___x_2473_ == 0)
{
lean_dec(v_goal_2383_);
goto v___jp_2420_;
}
else
{
size_t v___x_2476_; size_t v___x_2477_; lean_object* v___x_2478_; 
v___x_2476_ = ((size_t)0ULL);
v___x_2477_ = lean_usize_of_nat(v___x_2472_);
v___x_2478_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0(v_goal_2383_, v_traceScript_2384_, v_clauses_2385_, v___x_2476_, v___x_2477_, v___x_2474_, v___x_2419_, v_a_2386_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_, v_a_2391_);
v___y_2463_ = v___x_2478_;
goto v___jp_2462_;
}
}
else
{
size_t v___x_2479_; size_t v___x_2480_; lean_object* v___x_2481_; 
v___x_2479_ = ((size_t)0ULL);
v___x_2480_ = lean_usize_of_nat(v___x_2472_);
v___x_2481_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go_spec__0(v_goal_2383_, v_traceScript_2384_, v_clauses_2385_, v___x_2479_, v___x_2480_, v___x_2474_, v___x_2419_, v_a_2386_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_, v_a_2391_);
v___y_2463_ = v___x_2481_;
goto v___jp_2462_;
}
}
v___jp_2420_:
{
lean_object* v___x_2421_; lean_object* v_simpConfigSyntax_x3f_2422_; 
v___x_2421_ = lean_st_ref_get(v___x_2419_);
lean_dec(v___x_2419_);
v_simpConfigSyntax_x3f_2422_ = lean_ctor_get(v___x_2421_, 5);
lean_inc(v_simpConfigSyntax_x3f_2422_);
if (lean_obj_tag(v_simpConfigSyntax_x3f_2422_) == 1)
{
lean_object* v_options_2423_; uint8_t v_useSimpAll_2424_; 
v_options_2423_ = lean_ctor_get(v___x_2421_, 3);
lean_inc_ref(v_options_2423_);
v_useSimpAll_2424_ = lean_ctor_get_uint8(v_options_2423_, sizeof(void*)*6 + 8);
if (v_useSimpAll_2424_ == 0)
{
lean_object* v_additionalRules_2425_; lean_object* v_erasedRules_2426_; lean_object* v_enabledRuleSets_2427_; lean_object* v_val_2428_; lean_object* v___x_2429_; 
v_additionalRules_2425_ = lean_ctor_get(v___x_2421_, 0);
lean_inc_ref(v_additionalRules_2425_);
v_erasedRules_2426_ = lean_ctor_get(v___x_2421_, 1);
lean_inc_ref(v_erasedRules_2426_);
v_enabledRuleSets_2427_ = lean_ctor_get(v___x_2421_, 2);
lean_inc_ref(v_enabledRuleSets_2427_);
lean_dec(v___x_2421_);
v_val_2428_ = lean_ctor_get(v_simpConfigSyntax_x3f_2422_, 0);
lean_inc(v_val_2428_);
v___x_2429_ = lp_aesop_Aesop_Frontend_elabSimpConfig(v_val_2428_, v_a_2386_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_, v_a_2391_);
if (lean_obj_tag(v___x_2429_) == 0)
{
lean_object* v_a_2430_; 
v_a_2430_ = lean_ctor_get(v___x_2429_, 0);
lean_inc(v_a_2430_);
lean_dec_ref_known(v___x_2429_, 1);
v_additionalRules_2394_ = v_additionalRules_2425_;
v_erasedRules_2395_ = v_erasedRules_2426_;
v_enabledRuleSets_2396_ = v_enabledRuleSets_2427_;
v_options_2397_ = v_options_2423_;
v_simpConfigSyntax_x3f_2398_ = v_simpConfigSyntax_x3f_2422_;
v_simpConfig_2399_ = v_a_2430_;
goto v___jp_2393_;
}
else
{
lean_object* v_a_2431_; lean_object* v___x_2433_; uint8_t v_isShared_2434_; uint8_t v_isSharedCheck_2438_; 
lean_dec_ref(v_enabledRuleSets_2427_);
lean_dec_ref(v_erasedRules_2426_);
lean_dec_ref(v_additionalRules_2425_);
lean_dec_ref(v_options_2423_);
lean_dec_ref_known(v_simpConfigSyntax_x3f_2422_, 1);
v_a_2431_ = lean_ctor_get(v___x_2429_, 0);
v_isSharedCheck_2438_ = !lean_is_exclusive(v___x_2429_);
if (v_isSharedCheck_2438_ == 0)
{
v___x_2433_ = v___x_2429_;
v_isShared_2434_ = v_isSharedCheck_2438_;
goto v_resetjp_2432_;
}
else
{
lean_inc(v_a_2431_);
lean_dec(v___x_2429_);
v___x_2433_ = lean_box(0);
v_isShared_2434_ = v_isSharedCheck_2438_;
goto v_resetjp_2432_;
}
v_resetjp_2432_:
{
lean_object* v___x_2436_; 
if (v_isShared_2434_ == 0)
{
v___x_2436_ = v___x_2433_;
goto v_reusejp_2435_;
}
else
{
lean_object* v_reuseFailAlloc_2437_; 
v_reuseFailAlloc_2437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2437_, 0, v_a_2431_);
v___x_2436_ = v_reuseFailAlloc_2437_;
goto v_reusejp_2435_;
}
v_reusejp_2435_:
{
return v___x_2436_;
}
}
}
}
else
{
lean_object* v_additionalRules_2439_; lean_object* v_erasedRules_2440_; lean_object* v_enabledRuleSets_2441_; lean_object* v_val_2442_; lean_object* v___x_2443_; 
v_additionalRules_2439_ = lean_ctor_get(v___x_2421_, 0);
lean_inc_ref(v_additionalRules_2439_);
v_erasedRules_2440_ = lean_ctor_get(v___x_2421_, 1);
lean_inc_ref(v_erasedRules_2440_);
v_enabledRuleSets_2441_ = lean_ctor_get(v___x_2421_, 2);
lean_inc_ref(v_enabledRuleSets_2441_);
lean_dec(v___x_2421_);
v_val_2442_ = lean_ctor_get(v_simpConfigSyntax_x3f_2422_, 0);
lean_inc(v_val_2442_);
v___x_2443_ = lp_aesop_Aesop_Frontend_elabSimpConfigCtx(v_val_2442_, v_a_2386_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_, v_a_2391_);
if (lean_obj_tag(v___x_2443_) == 0)
{
lean_object* v_a_2444_; 
v_a_2444_ = lean_ctor_get(v___x_2443_, 0);
lean_inc(v_a_2444_);
lean_dec_ref_known(v___x_2443_, 1);
v_additionalRules_2394_ = v_additionalRules_2439_;
v_erasedRules_2395_ = v_erasedRules_2440_;
v_enabledRuleSets_2396_ = v_enabledRuleSets_2441_;
v_options_2397_ = v_options_2423_;
v_simpConfigSyntax_x3f_2398_ = v_simpConfigSyntax_x3f_2422_;
v_simpConfig_2399_ = v_a_2444_;
goto v___jp_2393_;
}
else
{
lean_object* v_a_2445_; lean_object* v___x_2447_; uint8_t v_isShared_2448_; uint8_t v_isSharedCheck_2452_; 
lean_dec_ref(v_enabledRuleSets_2441_);
lean_dec_ref(v_erasedRules_2440_);
lean_dec_ref(v_additionalRules_2439_);
lean_dec_ref(v_options_2423_);
lean_dec_ref_known(v_simpConfigSyntax_x3f_2422_, 1);
v_a_2445_ = lean_ctor_get(v___x_2443_, 0);
v_isSharedCheck_2452_ = !lean_is_exclusive(v___x_2443_);
if (v_isSharedCheck_2452_ == 0)
{
v___x_2447_ = v___x_2443_;
v_isShared_2448_ = v_isSharedCheck_2452_;
goto v_resetjp_2446_;
}
else
{
lean_inc(v_a_2445_);
lean_dec(v___x_2443_);
v___x_2447_ = lean_box(0);
v_isShared_2448_ = v_isSharedCheck_2452_;
goto v_resetjp_2446_;
}
v_resetjp_2446_:
{
lean_object* v___x_2450_; 
if (v_isShared_2448_ == 0)
{
v___x_2450_ = v___x_2447_;
goto v_reusejp_2449_;
}
else
{
lean_object* v_reuseFailAlloc_2451_; 
v_reuseFailAlloc_2451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2451_, 0, v_a_2445_);
v___x_2450_ = v_reuseFailAlloc_2451_;
goto v_reusejp_2449_;
}
v_reusejp_2449_:
{
return v___x_2450_;
}
}
}
}
}
else
{
lean_object* v_options_2453_; uint8_t v_useSimpAll_2454_; 
v_options_2453_ = lean_ctor_get(v___x_2421_, 3);
lean_inc_ref(v_options_2453_);
v_useSimpAll_2454_ = lean_ctor_get_uint8(v_options_2453_, sizeof(void*)*6 + 8);
if (v_useSimpAll_2454_ == 0)
{
lean_object* v_additionalRules_2455_; lean_object* v_erasedRules_2456_; lean_object* v_enabledRuleSets_2457_; 
v_additionalRules_2455_ = lean_ctor_get(v___x_2421_, 0);
lean_inc_ref(v_additionalRules_2455_);
v_erasedRules_2456_ = lean_ctor_get(v___x_2421_, 1);
lean_inc_ref(v_erasedRules_2456_);
v_enabledRuleSets_2457_ = lean_ctor_get(v___x_2421_, 2);
lean_inc_ref(v_enabledRuleSets_2457_);
lean_dec(v___x_2421_);
v_additionalRules_2394_ = v_additionalRules_2455_;
v_erasedRules_2395_ = v_erasedRules_2456_;
v_enabledRuleSets_2396_ = v_enabledRuleSets_2457_;
v_options_2397_ = v_options_2453_;
v_simpConfigSyntax_x3f_2398_ = v_simpConfigSyntax_x3f_2422_;
v_simpConfig_2399_ = v___x_2417_;
goto v___jp_2393_;
}
else
{
lean_object* v_additionalRules_2458_; lean_object* v_erasedRules_2459_; lean_object* v_enabledRuleSets_2460_; lean_object* v___x_2461_; 
v_additionalRules_2458_ = lean_ctor_get(v___x_2421_, 0);
lean_inc_ref(v_additionalRules_2458_);
v_erasedRules_2459_ = lean_ctor_get(v___x_2421_, 1);
lean_inc_ref(v_erasedRules_2459_);
v_enabledRuleSets_2460_ = lean_ctor_get(v___x_2421_, 2);
lean_inc_ref(v_enabledRuleSets_2460_);
lean_dec(v___x_2421_);
v___x_2461_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___closed__2));
v_additionalRules_2394_ = v_additionalRules_2458_;
v_erasedRules_2395_ = v_erasedRules_2459_;
v_enabledRuleSets_2396_ = v_enabledRuleSets_2460_;
v_options_2397_ = v_options_2453_;
v_simpConfigSyntax_x3f_2398_ = v_simpConfigSyntax_x3f_2422_;
v_simpConfig_2399_ = v___x_2461_;
goto v___jp_2393_;
}
}
}
v___jp_2462_:
{
if (lean_obj_tag(v___y_2463_) == 0)
{
lean_dec_ref_known(v___y_2463_, 1);
goto v___jp_2420_;
}
else
{
lean_object* v_a_2464_; lean_object* v___x_2466_; uint8_t v_isShared_2467_; uint8_t v_isSharedCheck_2471_; 
lean_dec(v___x_2419_);
v_a_2464_ = lean_ctor_get(v___y_2463_, 0);
v_isSharedCheck_2471_ = !lean_is_exclusive(v___y_2463_);
if (v_isSharedCheck_2471_ == 0)
{
v___x_2466_ = v___y_2463_;
v_isShared_2467_ = v_isSharedCheck_2471_;
goto v_resetjp_2465_;
}
else
{
lean_inc(v_a_2464_);
lean_dec(v___y_2463_);
v___x_2466_ = lean_box(0);
v_isShared_2467_ = v_isSharedCheck_2471_;
goto v_resetjp_2465_;
}
v_resetjp_2465_:
{
lean_object* v___x_2469_; 
if (v_isShared_2467_ == 0)
{
v___x_2469_ = v___x_2466_;
goto v_reusejp_2468_;
}
else
{
lean_object* v_reuseFailAlloc_2470_; 
v_reuseFailAlloc_2470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2470_, 0, v_a_2464_);
v___x_2469_ = v_reuseFailAlloc_2470_;
goto v_reusejp_2468_;
}
v_reusejp_2468_:
{
return v___x_2469_;
}
}
}
}
}
else
{
lean_object* v_a_2482_; lean_object* v___x_2484_; uint8_t v_isShared_2485_; uint8_t v_isSharedCheck_2494_; 
lean_dec(v_goal_2383_);
v_a_2482_ = lean_ctor_get(v___x_2402_, 0);
v_isSharedCheck_2494_ = !lean_is_exclusive(v___x_2402_);
if (v_isSharedCheck_2494_ == 0)
{
v___x_2484_ = v___x_2402_;
v_isShared_2485_ = v_isSharedCheck_2494_;
goto v_resetjp_2483_;
}
else
{
lean_inc(v_a_2482_);
lean_dec(v___x_2402_);
v___x_2484_ = lean_box(0);
v_isShared_2485_ = v_isSharedCheck_2494_;
goto v_resetjp_2483_;
}
v_resetjp_2483_:
{
lean_object* v_ref_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2492_; 
v_ref_2486_ = lean_ctor_get(v_a_2390_, 5);
v___x_2487_ = lean_io_error_to_string(v_a_2482_);
v___x_2488_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2488_, 0, v___x_2487_);
v___x_2489_ = l_Lean_MessageData_ofFormat(v___x_2488_);
lean_inc(v_ref_2486_);
v___x_2490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2490_, 0, v_ref_2486_);
lean_ctor_set(v___x_2490_, 1, v___x_2489_);
if (v_isShared_2485_ == 0)
{
lean_ctor_set(v___x_2484_, 0, v___x_2490_);
v___x_2492_ = v___x_2484_;
goto v_reusejp_2491_;
}
else
{
lean_object* v_reuseFailAlloc_2493_; 
v_reuseFailAlloc_2493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2493_, 0, v___x_2490_);
v___x_2492_ = v_reuseFailAlloc_2493_;
goto v_reusejp_2491_;
}
v_reusejp_2491_:
{
return v___x_2492_;
}
}
}
v___jp_2393_:
{
lean_object* v___x_2400_; lean_object* v___x_2401_; 
v___x_2400_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2400_, 0, v_additionalRules_2394_);
lean_ctor_set(v___x_2400_, 1, v_erasedRules_2395_);
lean_ctor_set(v___x_2400_, 2, v_enabledRuleSets_2396_);
lean_ctor_set(v___x_2400_, 3, v_options_2397_);
lean_ctor_set(v___x_2400_, 4, v_simpConfig_2399_);
lean_ctor_set(v___x_2400_, 5, v_simpConfigSyntax_x3f_2398_);
v___x_2401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2401_, 0, v___x_2400_);
return v___x_2401_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go___boxed(lean_object* v_goal_2495_, lean_object* v_traceScript_2496_, lean_object* v_clauses_2497_, lean_object* v_a_2498_, lean_object* v_a_2499_, lean_object* v_a_2500_, lean_object* v_a_2501_, lean_object* v_a_2502_, lean_object* v_a_2503_, lean_object* v_a_2504_){
_start:
{
uint8_t v_traceScript_boxed_2505_; lean_object* v_res_2506_; 
v_traceScript_boxed_2505_ = lean_unbox(v_traceScript_2496_);
v_res_2506_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go(v_goal_2495_, v_traceScript_boxed_2505_, v_clauses_2497_, v_a_2498_, v_a_2499_, v_a_2500_, v_a_2501_, v_a_2502_, v_a_2503_);
lean_dec(v_a_2503_);
lean_dec_ref(v_a_2502_);
lean_dec(v_a_2501_);
lean_dec_ref(v_a_2500_);
lean_dec(v_a_2499_);
lean_dec_ref(v_a_2498_);
lean_dec_ref(v_clauses_2497_);
return v_res_2506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg(){
_start:
{
lean_object* v___x_2508_; lean_object* v___x_2509_; 
v___x_2508_ = lean_obj_once(&lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0, &lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__0___redArg___closed__0);
v___x_2509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2509_, 0, v___x_2508_);
return v___x_2509_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg___boxed(lean_object* v___y_2510_){
_start:
{
lean_object* v_res_2511_; 
v_res_2511_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg();
return v_res_2511_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0(lean_object* v_00_u03b1_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_){
_start:
{
lean_object* v___x_2520_; 
v___x_2520_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg();
return v___x_2520_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___boxed(lean_object* v_00_u03b1_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_, lean_object* v___y_2526_, lean_object* v___y_2527_, lean_object* v___y_2528_){
_start:
{
lean_object* v_res_2529_; 
v_res_2529_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0(v_00_u03b1_2521_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_, v___y_2526_, v___y_2527_);
lean_dec(v___y_2527_);
lean_dec_ref(v___y_2526_);
lean_dec(v___y_2525_);
lean_dec_ref(v___y_2524_);
lean_dec(v___y_2523_);
lean_dec_ref(v___y_2522_);
return v_res_2529_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_parse(lean_object* v_stx_2530_, lean_object* v_goal_2531_, lean_object* v_a_2532_, lean_object* v_a_2533_, lean_object* v_a_2534_, lean_object* v_a_2535_, lean_object* v_a_2536_, lean_object* v_a_2537_){
_start:
{
lean_object* v_fileName_2539_; lean_object* v_fileMap_2540_; lean_object* v_options_2541_; lean_object* v_currRecDepth_2542_; lean_object* v_maxRecDepth_2543_; lean_object* v_ref_2544_; lean_object* v_currNamespace_2545_; lean_object* v_openDecls_2546_; lean_object* v_initHeartbeats_2547_; lean_object* v_maxHeartbeats_2548_; lean_object* v_quotContext_2549_; lean_object* v_currMacroScope_2550_; uint8_t v_diag_2551_; lean_object* v_cancelTk_x3f_2552_; uint8_t v_suppressElabErrors_2553_; lean_object* v_inheritedTraceOptions_2554_; lean_object* v___x_2555_; uint8_t v___x_2556_; lean_object* v_ref_2557_; lean_object* v___x_2558_; 
v_fileName_2539_ = lean_ctor_get(v_a_2536_, 0);
v_fileMap_2540_ = lean_ctor_get(v_a_2536_, 1);
v_options_2541_ = lean_ctor_get(v_a_2536_, 2);
v_currRecDepth_2542_ = lean_ctor_get(v_a_2536_, 3);
v_maxRecDepth_2543_ = lean_ctor_get(v_a_2536_, 4);
v_ref_2544_ = lean_ctor_get(v_a_2536_, 5);
v_currNamespace_2545_ = lean_ctor_get(v_a_2536_, 6);
v_openDecls_2546_ = lean_ctor_get(v_a_2536_, 7);
v_initHeartbeats_2547_ = lean_ctor_get(v_a_2536_, 8);
v_maxHeartbeats_2548_ = lean_ctor_get(v_a_2536_, 9);
v_quotContext_2549_ = lean_ctor_get(v_a_2536_, 10);
v_currMacroScope_2550_ = lean_ctor_get(v_a_2536_, 11);
v_diag_2551_ = lean_ctor_get_uint8(v_a_2536_, sizeof(void*)*14);
v_cancelTk_x3f_2552_ = lean_ctor_get(v_a_2536_, 12);
v_suppressElabErrors_2553_ = lean_ctor_get_uint8(v_a_2536_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2554_ = lean_ctor_get(v_a_2536_, 13);
v___x_2555_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_aesopTactic___closed__1));
lean_inc(v_stx_2530_);
v___x_2556_ = l_Lean_Syntax_isOfKind(v_stx_2530_, v___x_2555_);
v_ref_2557_ = l_Lean_replaceRef(v_stx_2530_, v_ref_2544_);
lean_inc_ref(v_inheritedTraceOptions_2554_);
lean_inc(v_cancelTk_x3f_2552_);
lean_inc(v_currMacroScope_2550_);
lean_inc(v_quotContext_2549_);
lean_inc(v_maxHeartbeats_2548_);
lean_inc(v_initHeartbeats_2547_);
lean_inc(v_openDecls_2546_);
lean_inc(v_currNamespace_2545_);
lean_inc(v_maxRecDepth_2543_);
lean_inc(v_currRecDepth_2542_);
lean_inc_ref(v_options_2541_);
lean_inc_ref(v_fileMap_2540_);
lean_inc_ref(v_fileName_2539_);
v___x_2558_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2558_, 0, v_fileName_2539_);
lean_ctor_set(v___x_2558_, 1, v_fileMap_2540_);
lean_ctor_set(v___x_2558_, 2, v_options_2541_);
lean_ctor_set(v___x_2558_, 3, v_currRecDepth_2542_);
lean_ctor_set(v___x_2558_, 4, v_maxRecDepth_2543_);
lean_ctor_set(v___x_2558_, 5, v_ref_2557_);
lean_ctor_set(v___x_2558_, 6, v_currNamespace_2545_);
lean_ctor_set(v___x_2558_, 7, v_openDecls_2546_);
lean_ctor_set(v___x_2558_, 8, v_initHeartbeats_2547_);
lean_ctor_set(v___x_2558_, 9, v_maxHeartbeats_2548_);
lean_ctor_set(v___x_2558_, 10, v_quotContext_2549_);
lean_ctor_set(v___x_2558_, 11, v_currMacroScope_2550_);
lean_ctor_set(v___x_2558_, 12, v_cancelTk_x3f_2552_);
lean_ctor_set(v___x_2558_, 13, v_inheritedTraceOptions_2554_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*14, v_diag_2551_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*14 + 1, v_suppressElabErrors_2553_);
if (v___x_2556_ == 0)
{
lean_object* v___x_2559_; uint8_t v___x_2560_; 
v___x_2559_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_aesopTactic_x3f___closed__1));
lean_inc(v_stx_2530_);
v___x_2560_ = l_Lean_Syntax_isOfKind(v_stx_2530_, v___x_2559_);
if (v___x_2560_ == 0)
{
lean_object* v___x_2561_; 
lean_dec_ref_known(v___x_2558_, 14);
lean_dec(v_goal_2531_);
lean_dec(v_stx_2530_);
v___x_2561_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_TacticConfig_parse_spec__0___redArg();
return v___x_2561_;
}
else
{
lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v_clauses_2564_; lean_object* v___x_2565_; 
v___x_2562_ = lean_unsigned_to_nat(1u);
v___x_2563_ = l_Lean_Syntax_getArg(v_stx_2530_, v___x_2562_);
lean_dec(v_stx_2530_);
v_clauses_2564_ = l_Lean_Syntax_getArgs(v___x_2563_);
lean_dec(v___x_2563_);
v___x_2565_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go(v_goal_2531_, v___x_2560_, v_clauses_2564_, v_a_2532_, v_a_2533_, v_a_2534_, v_a_2535_, v___x_2558_, v_a_2537_);
lean_dec_ref_known(v___x_2558_, 14);
lean_dec_ref(v_clauses_2564_);
return v___x_2565_;
}
}
else
{
lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v_clauses_2568_; uint8_t v___x_2569_; lean_object* v___x_2570_; 
v___x_2566_ = lean_unsigned_to_nat(1u);
v___x_2567_ = l_Lean_Syntax_getArg(v_stx_2530_, v___x_2566_);
lean_dec(v_stx_2530_);
v_clauses_2568_ = l_Lean_Syntax_getArgs(v___x_2567_);
lean_dec(v___x_2567_);
v___x_2569_ = 0;
v___x_2570_ = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_go(v_goal_2531_, v___x_2569_, v_clauses_2568_, v_a_2532_, v_a_2533_, v_a_2534_, v_a_2535_, v___x_2558_, v_a_2537_);
lean_dec_ref_known(v___x_2558_, 14);
lean_dec_ref(v_clauses_2568_);
return v___x_2570_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_parse___boxed(lean_object* v_stx_2571_, lean_object* v_goal_2572_, lean_object* v_a_2573_, lean_object* v_a_2574_, lean_object* v_a_2575_, lean_object* v_a_2576_, lean_object* v_a_2577_, lean_object* v_a_2578_, lean_object* v_a_2579_){
_start:
{
lean_object* v_res_2580_; 
v_res_2580_ = lp_aesop_Aesop_Frontend_TacticConfig_parse(v_stx_2571_, v_goal_2572_, v_a_2573_, v_a_2574_, v_a_2575_, v_a_2576_, v_a_2577_, v_a_2578_);
lean_dec(v_a_2578_);
lean_dec_ref(v_a_2577_);
lean_dec(v_a_2576_);
lean_dec_ref(v_a_2575_);
lean_dec(v_a_2574_);
lean_dec_ref(v_a_2573_);
return v_res_2580_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1(lean_object* v_opts_2581_, lean_object* v_opt_2582_){
_start:
{
lean_object* v_name_2583_; lean_object* v_defValue_2584_; lean_object* v_map_2585_; lean_object* v___x_2586_; 
v_name_2583_ = lean_ctor_get(v_opt_2582_, 0);
v_defValue_2584_ = lean_ctor_get(v_opt_2582_, 1);
v_map_2585_ = lean_ctor_get(v_opts_2581_, 0);
v___x_2586_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2585_, v_name_2583_);
if (lean_obj_tag(v___x_2586_) == 0)
{
uint8_t v___x_2587_; 
v___x_2587_ = lean_unbox(v_defValue_2584_);
return v___x_2587_;
}
else
{
lean_object* v_val_2588_; 
v_val_2588_ = lean_ctor_get(v___x_2586_, 0);
lean_inc(v_val_2588_);
lean_dec_ref_known(v___x_2586_, 1);
if (lean_obj_tag(v_val_2588_) == 1)
{
uint8_t v_v_2589_; 
v_v_2589_ = lean_ctor_get_uint8(v_val_2588_, 0);
lean_dec_ref_known(v_val_2588_, 0);
return v_v_2589_;
}
else
{
uint8_t v___x_2590_; 
lean_dec(v_val_2588_);
v___x_2590_ = lean_unbox(v_defValue_2584_);
return v___x_2590_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1___boxed(lean_object* v_opts_2591_, lean_object* v_opt_2592_){
_start:
{
uint8_t v_res_2593_; lean_object* v_r_2594_; 
v_res_2593_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1(v_opts_2591_, v_opt_2592_);
lean_dec_ref(v_opt_2592_);
lean_dec_ref(v_opts_2591_);
v_r_2594_ = lean_box(v_res_2593_);
return v_r_2594_;
}
}
static lean_object* _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0(void){
_start:
{
lean_object* v___x_2595_; lean_object* v___x_2596_; 
v___x_2595_ = lean_box(1);
v___x_2596_ = l_Lean_MessageData_ofFormat(v___x_2595_);
return v___x_2596_;
}
}
static lean_object* _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__3(void){
_start:
{
lean_object* v___x_2600_; lean_object* v___x_2601_; 
v___x_2600_ = ((lean_object*)(lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__2));
v___x_2601_ = l_Lean_MessageData_ofFormat(v___x_2600_);
return v___x_2601_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2(lean_object* v_x_2602_, lean_object* v_x_2603_){
_start:
{
if (lean_obj_tag(v_x_2603_) == 0)
{
return v_x_2602_;
}
else
{
lean_object* v_head_2604_; lean_object* v_tail_2605_; lean_object* v___x_2607_; uint8_t v_isShared_2608_; uint8_t v_isSharedCheck_2627_; 
v_head_2604_ = lean_ctor_get(v_x_2603_, 0);
v_tail_2605_ = lean_ctor_get(v_x_2603_, 1);
v_isSharedCheck_2627_ = !lean_is_exclusive(v_x_2603_);
if (v_isSharedCheck_2627_ == 0)
{
v___x_2607_ = v_x_2603_;
v_isShared_2608_ = v_isSharedCheck_2627_;
goto v_resetjp_2606_;
}
else
{
lean_inc(v_tail_2605_);
lean_inc(v_head_2604_);
lean_dec(v_x_2603_);
v___x_2607_ = lean_box(0);
v_isShared_2608_ = v_isSharedCheck_2627_;
goto v_resetjp_2606_;
}
v_resetjp_2606_:
{
lean_object* v_before_2609_; lean_object* v___x_2611_; uint8_t v_isShared_2612_; uint8_t v_isSharedCheck_2625_; 
v_before_2609_ = lean_ctor_get(v_head_2604_, 0);
v_isSharedCheck_2625_ = !lean_is_exclusive(v_head_2604_);
if (v_isSharedCheck_2625_ == 0)
{
lean_object* v_unused_2626_; 
v_unused_2626_ = lean_ctor_get(v_head_2604_, 1);
lean_dec(v_unused_2626_);
v___x_2611_ = v_head_2604_;
v_isShared_2612_ = v_isSharedCheck_2625_;
goto v_resetjp_2610_;
}
else
{
lean_inc(v_before_2609_);
lean_dec(v_head_2604_);
v___x_2611_ = lean_box(0);
v_isShared_2612_ = v_isSharedCheck_2625_;
goto v_resetjp_2610_;
}
v_resetjp_2610_:
{
lean_object* v___x_2613_; lean_object* v___x_2615_; 
v___x_2613_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0);
if (v_isShared_2612_ == 0)
{
lean_ctor_set_tag(v___x_2611_, 7);
lean_ctor_set(v___x_2611_, 1, v___x_2613_);
lean_ctor_set(v___x_2611_, 0, v_x_2602_);
v___x_2615_ = v___x_2611_;
goto v_reusejp_2614_;
}
else
{
lean_object* v_reuseFailAlloc_2624_; 
v_reuseFailAlloc_2624_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2624_, 0, v_x_2602_);
lean_ctor_set(v_reuseFailAlloc_2624_, 1, v___x_2613_);
v___x_2615_ = v_reuseFailAlloc_2624_;
goto v_reusejp_2614_;
}
v_reusejp_2614_:
{
lean_object* v___x_2616_; lean_object* v___x_2618_; 
v___x_2616_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__3, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__3_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__3);
if (v_isShared_2608_ == 0)
{
lean_ctor_set_tag(v___x_2607_, 7);
lean_ctor_set(v___x_2607_, 1, v___x_2616_);
lean_ctor_set(v___x_2607_, 0, v___x_2615_);
v___x_2618_ = v___x_2607_;
goto v_reusejp_2617_;
}
else
{
lean_object* v_reuseFailAlloc_2623_; 
v_reuseFailAlloc_2623_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2623_, 0, v___x_2615_);
lean_ctor_set(v_reuseFailAlloc_2623_, 1, v___x_2616_);
v___x_2618_ = v_reuseFailAlloc_2623_;
goto v_reusejp_2617_;
}
v_reusejp_2617_:
{
lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; 
v___x_2619_ = l_Lean_MessageData_ofSyntax(v_before_2609_);
v___x_2620_ = l_Lean_indentD(v___x_2619_);
v___x_2621_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2621_, 0, v___x_2618_);
lean_ctor_set(v___x_2621_, 1, v___x_2620_);
v_x_2602_ = v___x_2621_;
v_x_2603_ = v_tail_2605_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_2631_; lean_object* v___x_2632_; 
v___x_2631_ = ((lean_object*)(lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__1));
v___x_2632_ = l_Lean_MessageData_ofFormat(v___x_2631_);
return v___x_2632_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg(lean_object* v_msgData_2633_, lean_object* v_macroStack_2634_, lean_object* v___y_2635_){
_start:
{
lean_object* v_options_2637_; lean_object* v___x_2638_; uint8_t v___x_2639_; 
v_options_2637_ = lean_ctor_get(v___y_2635_, 2);
v___x_2638_ = l_Lean_Elab_pp_macroStack;
v___x_2639_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1(v_options_2637_, v___x_2638_);
if (v___x_2639_ == 0)
{
lean_object* v___x_2640_; 
lean_dec(v_macroStack_2634_);
v___x_2640_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2640_, 0, v_msgData_2633_);
return v___x_2640_;
}
else
{
if (lean_obj_tag(v_macroStack_2634_) == 0)
{
lean_object* v___x_2641_; 
v___x_2641_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2641_, 0, v_msgData_2633_);
return v___x_2641_;
}
else
{
lean_object* v_head_2642_; lean_object* v_after_2643_; lean_object* v___x_2645_; uint8_t v_isShared_2646_; uint8_t v_isSharedCheck_2658_; 
v_head_2642_ = lean_ctor_get(v_macroStack_2634_, 0);
lean_inc(v_head_2642_);
v_after_2643_ = lean_ctor_get(v_head_2642_, 1);
v_isSharedCheck_2658_ = !lean_is_exclusive(v_head_2642_);
if (v_isSharedCheck_2658_ == 0)
{
lean_object* v_unused_2659_; 
v_unused_2659_ = lean_ctor_get(v_head_2642_, 0);
lean_dec(v_unused_2659_);
v___x_2645_ = v_head_2642_;
v_isShared_2646_ = v_isSharedCheck_2658_;
goto v_resetjp_2644_;
}
else
{
lean_inc(v_after_2643_);
lean_dec(v_head_2642_);
v___x_2645_ = lean_box(0);
v_isShared_2646_ = v_isSharedCheck_2658_;
goto v_resetjp_2644_;
}
v_resetjp_2644_:
{
lean_object* v___x_2647_; lean_object* v___x_2649_; 
v___x_2647_ = lean_obj_once(&lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0, &lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0_once, _init_lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2___closed__0);
if (v_isShared_2646_ == 0)
{
lean_ctor_set_tag(v___x_2645_, 7);
lean_ctor_set(v___x_2645_, 1, v___x_2647_);
lean_ctor_set(v___x_2645_, 0, v_msgData_2633_);
v___x_2649_ = v___x_2645_;
goto v_reusejp_2648_;
}
else
{
lean_object* v_reuseFailAlloc_2657_; 
v_reuseFailAlloc_2657_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2657_, 0, v_msgData_2633_);
lean_ctor_set(v_reuseFailAlloc_2657_, 1, v___x_2647_);
v___x_2649_ = v_reuseFailAlloc_2657_;
goto v_reusejp_2648_;
}
v_reusejp_2648_:
{
lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v_msgData_2654_; lean_object* v___x_2655_; lean_object* v___x_2656_; 
v___x_2650_ = lean_obj_once(&lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__2, &lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__2_once, _init_lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___closed__2);
v___x_2651_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2651_, 0, v___x_2649_);
lean_ctor_set(v___x_2651_, 1, v___x_2650_);
v___x_2652_ = l_Lean_MessageData_ofSyntax(v_after_2643_);
v___x_2653_ = l_Lean_indentD(v___x_2652_);
v_msgData_2654_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2654_, 0, v___x_2651_);
lean_ctor_set(v_msgData_2654_, 1, v___x_2653_);
v___x_2655_ = lp_aesop_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__2(v_msgData_2654_, v_macroStack_2634_);
v___x_2656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2656_, 0, v___x_2655_);
return v___x_2656_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg___boxed(lean_object* v_msgData_2660_, lean_object* v_macroStack_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_){
_start:
{
lean_object* v_res_2664_; 
v_res_2664_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg(v_msgData_2660_, v_macroStack_2661_, v___y_2662_);
lean_dec_ref(v___y_2662_);
return v_res_2664_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg(lean_object* v_msg_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_, lean_object* v___y_2670_, lean_object* v___y_2671_){
_start:
{
lean_object* v_ref_2673_; lean_object* v___x_2674_; lean_object* v_a_2675_; lean_object* v_macroStack_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; lean_object* v_a_2679_; lean_object* v___x_2681_; uint8_t v_isShared_2682_; uint8_t v_isSharedCheck_2687_; 
v_ref_2673_ = lean_ctor_get(v___y_2670_, 5);
v___x_2674_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Frontend_Tactic_0__Aesop_Frontend_TacticConfig_parse_addClause_spec__3_spec__5(v_msg_2665_, v___y_2668_, v___y_2669_, v___y_2670_, v___y_2671_);
v_a_2675_ = lean_ctor_get(v___x_2674_, 0);
lean_inc(v_a_2675_);
lean_dec_ref(v___x_2674_);
v_macroStack_2676_ = lean_ctor_get(v___y_2666_, 1);
v___x_2677_ = l_Lean_Elab_getBetterRef(v_ref_2673_, v_macroStack_2676_);
lean_inc(v_macroStack_2676_);
v___x_2678_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg(v_a_2675_, v_macroStack_2676_, v___y_2670_);
v_a_2679_ = lean_ctor_get(v___x_2678_, 0);
v_isSharedCheck_2687_ = !lean_is_exclusive(v___x_2678_);
if (v_isSharedCheck_2687_ == 0)
{
v___x_2681_ = v___x_2678_;
v_isShared_2682_ = v_isSharedCheck_2687_;
goto v_resetjp_2680_;
}
else
{
lean_inc(v_a_2679_);
lean_dec(v___x_2678_);
v___x_2681_ = lean_box(0);
v_isShared_2682_ = v_isSharedCheck_2687_;
goto v_resetjp_2680_;
}
v_resetjp_2680_:
{
lean_object* v___x_2683_; lean_object* v___x_2685_; 
v___x_2683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2683_, 0, v___x_2677_);
lean_ctor_set(v___x_2683_, 1, v_a_2679_);
if (v_isShared_2682_ == 0)
{
lean_ctor_set_tag(v___x_2681_, 1);
lean_ctor_set(v___x_2681_, 0, v___x_2683_);
v___x_2685_ = v___x_2681_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2686_; 
v_reuseFailAlloc_2686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2686_, 0, v___x_2683_);
v___x_2685_ = v_reuseFailAlloc_2686_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
return v___x_2685_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg___boxed(lean_object* v_msg_2688_, lean_object* v___y_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_, lean_object* v___y_2692_, lean_object* v___y_2693_, lean_object* v___y_2694_, lean_object* v___y_2695_){
_start:
{
lean_object* v_res_2696_; 
v_res_2696_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg(v_msg_2688_, v___y_2689_, v___y_2690_, v___y_2691_, v___y_2692_, v___y_2693_, v___y_2694_);
lean_dec(v___y_2694_);
lean_dec_ref(v___y_2693_);
lean_dec(v___y_2692_);
lean_dec_ref(v___y_2691_);
lean_dec(v___y_2690_);
lean_dec_ref(v___y_2689_);
return v_res_2696_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__1(void){
_start:
{
lean_object* v___x_2698_; lean_object* v___x_2699_; 
v___x_2698_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__0));
v___x_2699_ = l_Lean_stringToMessageData(v___x_2698_);
return v___x_2699_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__3(void){
_start:
{
lean_object* v___x_2701_; lean_object* v___x_2702_; 
v___x_2701_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__2));
v___x_2702_ = l_Lean_stringToMessageData(v___x_2701_);
return v___x_2702_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1(lean_object* v_as_2703_, size_t v_sz_2704_, size_t v_i_2705_, lean_object* v_b_2706_, lean_object* v___y_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_){
_start:
{
uint8_t v___x_2714_; 
v___x_2714_ = lean_usize_dec_lt(v_i_2705_, v_sz_2704_);
if (v___x_2714_ == 0)
{
lean_object* v___x_2715_; 
v___x_2715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2715_, 0, v_b_2706_);
return v___x_2715_;
}
else
{
lean_object* v_a_2716_; lean_object* v___x_2717_; lean_object* v_fst_2718_; lean_object* v_snd_2719_; lean_object* v___x_2721_; uint8_t v_isShared_2722_; uint8_t v_isSharedCheck_2745_; 
v_a_2716_ = lean_array_uget_borrowed(v_as_2703_, v_i_2705_);
lean_inc(v_a_2716_);
v___x_2717_ = lp_aesop_Aesop_LocalRuleSet_erase(v_b_2706_, v_a_2716_);
v_fst_2718_ = lean_ctor_get(v___x_2717_, 0);
v_snd_2719_ = lean_ctor_get(v___x_2717_, 1);
v_isSharedCheck_2745_ = !lean_is_exclusive(v___x_2717_);
if (v_isSharedCheck_2745_ == 0)
{
v___x_2721_ = v___x_2717_;
v_isShared_2722_ = v_isSharedCheck_2745_;
goto v_resetjp_2720_;
}
else
{
lean_inc(v_snd_2719_);
lean_inc(v_fst_2718_);
lean_dec(v___x_2717_);
v___x_2721_ = lean_box(0);
v_isShared_2722_ = v_isSharedCheck_2745_;
goto v_resetjp_2720_;
}
v_resetjp_2720_:
{
uint8_t v___x_2727_; 
v___x_2727_ = lean_unbox(v_snd_2719_);
lean_dec(v_snd_2719_);
if (v___x_2727_ == 0)
{
lean_object* v_name_2728_; lean_object* v___x_2729_; lean_object* v___x_2730_; lean_object* v___x_2732_; 
v_name_2728_ = lean_ctor_get(v_a_2716_, 0);
v___x_2729_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__1);
lean_inc(v_name_2728_);
v___x_2730_ = l_Lean_MessageData_ofName(v_name_2728_);
if (v_isShared_2722_ == 0)
{
lean_ctor_set_tag(v___x_2721_, 7);
lean_ctor_set(v___x_2721_, 1, v___x_2730_);
lean_ctor_set(v___x_2721_, 0, v___x_2729_);
v___x_2732_ = v___x_2721_;
goto v_reusejp_2731_;
}
else
{
lean_object* v_reuseFailAlloc_2744_; 
v_reuseFailAlloc_2744_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2744_, 0, v___x_2729_);
lean_ctor_set(v_reuseFailAlloc_2744_, 1, v___x_2730_);
v___x_2732_ = v_reuseFailAlloc_2744_;
goto v_reusejp_2731_;
}
v_reusejp_2731_:
{
lean_object* v___x_2733_; lean_object* v___x_2734_; lean_object* v___x_2735_; 
v___x_2733_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___closed__3);
v___x_2734_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2734_, 0, v___x_2732_);
lean_ctor_set(v___x_2734_, 1, v___x_2733_);
v___x_2735_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg(v___x_2734_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_);
if (lean_obj_tag(v___x_2735_) == 0)
{
lean_dec_ref_known(v___x_2735_, 1);
goto v___jp_2723_;
}
else
{
lean_object* v_a_2736_; lean_object* v___x_2738_; uint8_t v_isShared_2739_; uint8_t v_isSharedCheck_2743_; 
lean_dec(v_fst_2718_);
v_a_2736_ = lean_ctor_get(v___x_2735_, 0);
v_isSharedCheck_2743_ = !lean_is_exclusive(v___x_2735_);
if (v_isSharedCheck_2743_ == 0)
{
v___x_2738_ = v___x_2735_;
v_isShared_2739_ = v_isSharedCheck_2743_;
goto v_resetjp_2737_;
}
else
{
lean_inc(v_a_2736_);
lean_dec(v___x_2735_);
v___x_2738_ = lean_box(0);
v_isShared_2739_ = v_isSharedCheck_2743_;
goto v_resetjp_2737_;
}
v_resetjp_2737_:
{
lean_object* v___x_2741_; 
if (v_isShared_2739_ == 0)
{
v___x_2741_ = v___x_2738_;
goto v_reusejp_2740_;
}
else
{
lean_object* v_reuseFailAlloc_2742_; 
v_reuseFailAlloc_2742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2742_, 0, v_a_2736_);
v___x_2741_ = v_reuseFailAlloc_2742_;
goto v_reusejp_2740_;
}
v_reusejp_2740_:
{
return v___x_2741_;
}
}
}
}
}
else
{
lean_del_object(v___x_2721_);
goto v___jp_2723_;
}
v___jp_2723_:
{
size_t v___x_2724_; size_t v___x_2725_; 
v___x_2724_ = ((size_t)1ULL);
v___x_2725_ = lean_usize_add(v_i_2705_, v___x_2724_);
v_i_2705_ = v___x_2725_;
v_b_2706_ = v_fst_2718_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1___boxed(lean_object* v_as_2746_, lean_object* v_sz_2747_, lean_object* v_i_2748_, lean_object* v_b_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_, lean_object* v___y_2755_, lean_object* v___y_2756_){
_start:
{
size_t v_sz_boxed_2757_; size_t v_i_boxed_2758_; lean_object* v_res_2759_; 
v_sz_boxed_2757_ = lean_unbox_usize(v_sz_2747_);
lean_dec(v_sz_2747_);
v_i_boxed_2758_ = lean_unbox_usize(v_i_2748_);
lean_dec(v_i_2748_);
v_res_2759_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1(v_as_2746_, v_sz_boxed_2757_, v_i_boxed_2758_, v_b_2749_, v___y_2750_, v___y_2751_, v___y_2752_, v___y_2753_, v___y_2754_, v___y_2755_);
lean_dec(v___y_2755_);
lean_dec_ref(v___y_2754_);
lean_dec(v___y_2753_);
lean_dec_ref(v___y_2752_);
lean_dec(v___y_2751_);
lean_dec_ref(v___y_2750_);
lean_dec_ref(v_as_2746_);
return v_res_2759_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__4(lean_object* v_goal_2760_, lean_object* v_as_2761_, size_t v_sz_2762_, size_t v_i_2763_, lean_object* v_b_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_, lean_object* v___y_2768_, lean_object* v___y_2769_, lean_object* v___y_2770_){
_start:
{
uint8_t v___x_2772_; 
v___x_2772_ = lean_usize_dec_lt(v_i_2763_, v_sz_2762_);
if (v___x_2772_ == 0)
{
lean_object* v___x_2773_; 
lean_dec(v_goal_2760_);
v___x_2773_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2773_, 0, v_b_2764_);
return v___x_2773_;
}
else
{
lean_object* v_a_2774_; lean_object* v___x_2775_; lean_object* v___x_2776_; 
v_a_2774_ = lean_array_uget_borrowed(v_as_2761_, v_i_2763_);
lean_inc(v_goal_2760_);
v___x_2775_ = lp_aesop_Aesop_ElabM_Context_forErasing(v_goal_2760_);
lean_inc(v_a_2774_);
v___x_2776_ = lp_aesop_Aesop_Frontend_RuleExpr_toLocalRuleFilters(v_a_2774_, v___x_2775_, v___y_2765_, v___y_2766_, v___y_2767_, v___y_2768_, v___y_2769_, v___y_2770_);
lean_dec_ref(v___x_2775_);
if (lean_obj_tag(v___x_2776_) == 0)
{
lean_object* v_a_2777_; size_t v_sz_2778_; size_t v___x_2779_; lean_object* v___x_2780_; 
v_a_2777_ = lean_ctor_get(v___x_2776_, 0);
lean_inc(v_a_2777_);
lean_dec_ref_known(v___x_2776_, 1);
v_sz_2778_ = lean_array_size(v_a_2777_);
v___x_2779_ = ((size_t)0ULL);
v___x_2780_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__1(v_a_2777_, v_sz_2778_, v___x_2779_, v_b_2764_, v___y_2765_, v___y_2766_, v___y_2767_, v___y_2768_, v___y_2769_, v___y_2770_);
lean_dec(v_a_2777_);
if (lean_obj_tag(v___x_2780_) == 0)
{
lean_object* v_a_2781_; size_t v___x_2782_; size_t v___x_2783_; 
v_a_2781_ = lean_ctor_get(v___x_2780_, 0);
lean_inc(v_a_2781_);
lean_dec_ref_known(v___x_2780_, 1);
v___x_2782_ = ((size_t)1ULL);
v___x_2783_ = lean_usize_add(v_i_2763_, v___x_2782_);
v_i_2763_ = v___x_2783_;
v_b_2764_ = v_a_2781_;
goto _start;
}
else
{
lean_dec(v_goal_2760_);
return v___x_2780_;
}
}
else
{
lean_object* v_a_2785_; lean_object* v___x_2787_; uint8_t v_isShared_2788_; uint8_t v_isSharedCheck_2792_; 
lean_dec_ref(v_b_2764_);
lean_dec(v_goal_2760_);
v_a_2785_ = lean_ctor_get(v___x_2776_, 0);
v_isSharedCheck_2792_ = !lean_is_exclusive(v___x_2776_);
if (v_isSharedCheck_2792_ == 0)
{
v___x_2787_ = v___x_2776_;
v_isShared_2788_ = v_isSharedCheck_2792_;
goto v_resetjp_2786_;
}
else
{
lean_inc(v_a_2785_);
lean_dec(v___x_2776_);
v___x_2787_ = lean_box(0);
v_isShared_2788_ = v_isSharedCheck_2792_;
goto v_resetjp_2786_;
}
v_resetjp_2786_:
{
lean_object* v___x_2790_; 
if (v_isShared_2788_ == 0)
{
v___x_2790_ = v___x_2787_;
goto v_reusejp_2789_;
}
else
{
lean_object* v_reuseFailAlloc_2791_; 
v_reuseFailAlloc_2791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2791_, 0, v_a_2785_);
v___x_2790_ = v_reuseFailAlloc_2791_;
goto v_reusejp_2789_;
}
v_reusejp_2789_:
{
return v___x_2790_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__4___boxed(lean_object* v_goal_2793_, lean_object* v_as_2794_, lean_object* v_sz_2795_, lean_object* v_i_2796_, lean_object* v_b_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_, lean_object* v___y_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_){
_start:
{
size_t v_sz_boxed_2805_; size_t v_i_boxed_2806_; lean_object* v_res_2807_; 
v_sz_boxed_2805_ = lean_unbox_usize(v_sz_2795_);
lean_dec(v_sz_2795_);
v_i_boxed_2806_ = lean_unbox_usize(v_i_2796_);
lean_dec(v_i_2796_);
v_res_2807_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__4(v_goal_2793_, v_as_2794_, v_sz_boxed_2805_, v_i_boxed_2806_, v_b_2797_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_, v___y_2802_, v___y_2803_);
lean_dec(v___y_2803_);
lean_dec_ref(v___y_2802_);
lean_dec(v___y_2801_);
lean_dec_ref(v___y_2800_);
lean_dec(v___y_2799_);
lean_dec_ref(v___y_2798_);
lean_dec_ref(v_as_2794_);
return v_res_2807_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg(lean_object* v_as_2808_, size_t v_sz_2809_, size_t v_i_2810_, lean_object* v_b_2811_){
_start:
{
uint8_t v___x_2813_; 
v___x_2813_ = lean_usize_dec_lt(v_i_2810_, v_sz_2809_);
if (v___x_2813_ == 0)
{
lean_object* v___x_2814_; 
v___x_2814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2814_, 0, v_b_2811_);
return v___x_2814_;
}
else
{
lean_object* v_a_2815_; lean_object* v___x_2816_; size_t v___x_2817_; size_t v___x_2818_; 
v_a_2815_ = lean_array_uget_borrowed(v_as_2808_, v_i_2810_);
lean_inc(v_a_2815_);
v___x_2816_ = lp_aesop_Aesop_LocalRuleSet_add(v_b_2811_, v_a_2815_);
v___x_2817_ = ((size_t)1ULL);
v___x_2818_ = lean_usize_add(v_i_2810_, v___x_2817_);
v_i_2810_ = v___x_2818_;
v_b_2811_ = v___x_2816_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg___boxed(lean_object* v_as_2820_, lean_object* v_sz_2821_, lean_object* v_i_2822_, lean_object* v_b_2823_, lean_object* v___y_2824_){
_start:
{
size_t v_sz_boxed_2825_; size_t v_i_boxed_2826_; lean_object* v_res_2827_; 
v_sz_boxed_2825_ = lean_unbox_usize(v_sz_2821_);
lean_dec(v_sz_2821_);
v_i_boxed_2826_ = lean_unbox_usize(v_i_2822_);
lean_dec(v_i_2822_);
v_res_2827_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg(v_as_2820_, v_sz_boxed_2825_, v_i_boxed_2826_, v_b_2823_);
lean_dec_ref(v_as_2820_);
return v_res_2827_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__3(lean_object* v_goal_2828_, lean_object* v_as_2829_, size_t v_sz_2830_, size_t v_i_2831_, lean_object* v_b_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_){
_start:
{
uint8_t v___x_2840_; 
v___x_2840_ = lean_usize_dec_lt(v_i_2831_, v_sz_2830_);
if (v___x_2840_ == 0)
{
lean_object* v___x_2841_; 
lean_dec(v_goal_2828_);
v___x_2841_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2841_, 0, v_b_2832_);
return v___x_2841_;
}
else
{
lean_object* v_a_2842_; lean_object* v___x_2843_; 
v_a_2842_ = lean_array_uget_borrowed(v_as_2829_, v_i_2831_);
lean_inc(v_a_2842_);
lean_inc(v_goal_2828_);
v___x_2843_ = lp_aesop_Aesop_Frontend_RuleExpr_buildAdditionalLocalRules(v_goal_2828_, v_a_2842_, v___y_2833_, v___y_2834_, v___y_2835_, v___y_2836_, v___y_2837_, v___y_2838_);
if (lean_obj_tag(v___x_2843_) == 0)
{
lean_object* v_a_2844_; size_t v_sz_2845_; size_t v___x_2846_; lean_object* v___x_2847_; 
v_a_2844_ = lean_ctor_get(v___x_2843_, 0);
lean_inc(v_a_2844_);
lean_dec_ref_known(v___x_2843_, 1);
v_sz_2845_ = lean_array_size(v_a_2844_);
v___x_2846_ = ((size_t)0ULL);
v___x_2847_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg(v_a_2844_, v_sz_2845_, v___x_2846_, v_b_2832_);
lean_dec(v_a_2844_);
if (lean_obj_tag(v___x_2847_) == 0)
{
lean_object* v_a_2848_; size_t v___x_2849_; size_t v___x_2850_; 
v_a_2848_ = lean_ctor_get(v___x_2847_, 0);
lean_inc(v_a_2848_);
lean_dec_ref_known(v___x_2847_, 1);
v___x_2849_ = ((size_t)1ULL);
v___x_2850_ = lean_usize_add(v_i_2831_, v___x_2849_);
v_i_2831_ = v___x_2850_;
v_b_2832_ = v_a_2848_;
goto _start;
}
else
{
lean_dec(v_goal_2828_);
return v___x_2847_;
}
}
else
{
lean_object* v_a_2852_; lean_object* v___x_2854_; uint8_t v_isShared_2855_; uint8_t v_isSharedCheck_2859_; 
lean_dec_ref(v_b_2832_);
lean_dec(v_goal_2828_);
v_a_2852_ = lean_ctor_get(v___x_2843_, 0);
v_isSharedCheck_2859_ = !lean_is_exclusive(v___x_2843_);
if (v_isSharedCheck_2859_ == 0)
{
v___x_2854_ = v___x_2843_;
v_isShared_2855_ = v_isSharedCheck_2859_;
goto v_resetjp_2853_;
}
else
{
lean_inc(v_a_2852_);
lean_dec(v___x_2843_);
v___x_2854_ = lean_box(0);
v_isShared_2855_ = v_isSharedCheck_2859_;
goto v_resetjp_2853_;
}
v_resetjp_2853_:
{
lean_object* v___x_2857_; 
if (v_isShared_2855_ == 0)
{
v___x_2857_ = v___x_2854_;
goto v_reusejp_2856_;
}
else
{
lean_object* v_reuseFailAlloc_2858_; 
v_reuseFailAlloc_2858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2858_, 0, v_a_2852_);
v___x_2857_ = v_reuseFailAlloc_2858_;
goto v_reusejp_2856_;
}
v_reusejp_2856_:
{
return v___x_2857_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__3___boxed(lean_object* v_goal_2860_, lean_object* v_as_2861_, lean_object* v_sz_2862_, lean_object* v_i_2863_, lean_object* v_b_2864_, lean_object* v___y_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_, lean_object* v___y_2870_, lean_object* v___y_2871_){
_start:
{
size_t v_sz_boxed_2872_; size_t v_i_boxed_2873_; lean_object* v_res_2874_; 
v_sz_boxed_2872_ = lean_unbox_usize(v_sz_2862_);
lean_dec(v_sz_2862_);
v_i_boxed_2873_ = lean_unbox_usize(v_i_2863_);
lean_dec(v_i_2863_);
v_res_2874_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__3(v_goal_2860_, v_as_2861_, v_sz_boxed_2872_, v_i_boxed_2873_, v_b_2864_, v___y_2865_, v___y_2866_, v___y_2867_, v___y_2868_, v___y_2869_, v___y_2870_);
lean_dec(v___y_2870_);
lean_dec_ref(v___y_2869_);
lean_dec(v___y_2868_);
lean_dec_ref(v___y_2867_);
lean_dec(v___y_2866_);
lean_dec_ref(v___y_2865_);
lean_dec_ref(v_as_2861_);
return v_res_2874_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_updateRuleSet(lean_object* v_rs_2875_, lean_object* v_c_2876_, lean_object* v_goal_2877_, lean_object* v_a_2878_, lean_object* v_a_2879_, lean_object* v_a_2880_, lean_object* v_a_2881_, lean_object* v_a_2882_, lean_object* v_a_2883_){
_start:
{
lean_object* v_additionalRules_2885_; lean_object* v_erasedRules_2886_; size_t v_sz_2887_; size_t v___x_2888_; lean_object* v___x_2889_; 
v_additionalRules_2885_ = lean_ctor_get(v_c_2876_, 0);
v_erasedRules_2886_ = lean_ctor_get(v_c_2876_, 1);
v_sz_2887_ = lean_array_size(v_additionalRules_2885_);
v___x_2888_ = ((size_t)0ULL);
lean_inc(v_goal_2877_);
v___x_2889_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__3(v_goal_2877_, v_additionalRules_2885_, v_sz_2887_, v___x_2888_, v_rs_2875_, v_a_2878_, v_a_2879_, v_a_2880_, v_a_2881_, v_a_2882_, v_a_2883_);
if (lean_obj_tag(v___x_2889_) == 0)
{
lean_object* v_a_2890_; size_t v_sz_2891_; lean_object* v___x_2892_; 
v_a_2890_ = lean_ctor_get(v___x_2889_, 0);
lean_inc(v_a_2890_);
lean_dec_ref_known(v___x_2889_, 1);
v_sz_2891_ = lean_array_size(v_erasedRules_2886_);
v___x_2892_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__4(v_goal_2877_, v_erasedRules_2886_, v_sz_2891_, v___x_2888_, v_a_2890_, v_a_2878_, v_a_2879_, v_a_2880_, v_a_2881_, v_a_2882_, v_a_2883_);
return v___x_2892_;
}
else
{
lean_dec(v_goal_2877_);
return v___x_2889_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_updateRuleSet___boxed(lean_object* v_rs_2893_, lean_object* v_c_2894_, lean_object* v_goal_2895_, lean_object* v_a_2896_, lean_object* v_a_2897_, lean_object* v_a_2898_, lean_object* v_a_2899_, lean_object* v_a_2900_, lean_object* v_a_2901_, lean_object* v_a_2902_){
_start:
{
lean_object* v_res_2903_; 
v_res_2903_ = lp_aesop_Aesop_Frontend_TacticConfig_updateRuleSet(v_rs_2893_, v_c_2894_, v_goal_2895_, v_a_2896_, v_a_2897_, v_a_2898_, v_a_2899_, v_a_2900_, v_a_2901_);
lean_dec(v_a_2901_);
lean_dec_ref(v_a_2900_);
lean_dec(v_a_2899_);
lean_dec_ref(v_a_2898_);
lean_dec(v_a_2897_);
lean_dec_ref(v_a_2896_);
lean_dec_ref(v_c_2894_);
return v_res_2903_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0(lean_object* v_00_u03b1_2904_, lean_object* v_msg_2905_, lean_object* v___y_2906_, lean_object* v___y_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_){
_start:
{
lean_object* v___x_2913_; 
v___x_2913_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___redArg(v_msg_2905_, v___y_2906_, v___y_2907_, v___y_2908_, v___y_2909_, v___y_2910_, v___y_2911_);
return v___x_2913_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0___boxed(lean_object* v_00_u03b1_2914_, lean_object* v_msg_2915_, lean_object* v___y_2916_, lean_object* v___y_2917_, lean_object* v___y_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_){
_start:
{
lean_object* v_res_2923_; 
v_res_2923_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0(v_00_u03b1_2914_, v_msg_2915_, v___y_2916_, v___y_2917_, v___y_2918_, v___y_2919_, v___y_2920_, v___y_2921_);
lean_dec(v___y_2921_);
lean_dec_ref(v___y_2920_);
lean_dec(v___y_2919_);
lean_dec_ref(v___y_2918_);
lean_dec(v___y_2917_);
lean_dec_ref(v___y_2916_);
return v_res_2923_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2(lean_object* v_as_2924_, size_t v_sz_2925_, size_t v_i_2926_, lean_object* v_b_2927_, lean_object* v___y_2928_, lean_object* v___y_2929_, lean_object* v___y_2930_, lean_object* v___y_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_){
_start:
{
lean_object* v___x_2935_; 
v___x_2935_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___redArg(v_as_2924_, v_sz_2925_, v_i_2926_, v_b_2927_);
return v___x_2935_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2___boxed(lean_object* v_as_2936_, lean_object* v_sz_2937_, lean_object* v_i_2938_, lean_object* v_b_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_, lean_object* v___y_2943_, lean_object* v___y_2944_, lean_object* v___y_2945_, lean_object* v___y_2946_){
_start:
{
size_t v_sz_boxed_2947_; size_t v_i_boxed_2948_; lean_object* v_res_2949_; 
v_sz_boxed_2947_ = lean_unbox_usize(v_sz_2937_);
lean_dec(v_sz_2937_);
v_i_boxed_2948_ = lean_unbox_usize(v_i_2938_);
lean_dec(v_i_2938_);
v_res_2949_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__2(v_as_2936_, v_sz_boxed_2947_, v_i_boxed_2948_, v_b_2939_, v___y_2940_, v___y_2941_, v___y_2942_, v___y_2943_, v___y_2944_, v___y_2945_);
lean_dec(v___y_2945_);
lean_dec_ref(v___y_2944_);
lean_dec(v___y_2943_);
lean_dec_ref(v___y_2942_);
lean_dec(v___y_2941_);
lean_dec_ref(v___y_2940_);
lean_dec_ref(v_as_2936_);
return v_res_2949_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0(lean_object* v_msgData_2950_, lean_object* v_macroStack_2951_, lean_object* v___y_2952_, lean_object* v___y_2953_, lean_object* v___y_2954_, lean_object* v___y_2955_, lean_object* v___y_2956_, lean_object* v___y_2957_){
_start:
{
lean_object* v___x_2959_; 
v___x_2959_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___redArg(v_msgData_2950_, v_macroStack_2951_, v___y_2956_);
return v___x_2959_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0___boxed(lean_object* v_msgData_2960_, lean_object* v_macroStack_2961_, lean_object* v___y_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_, lean_object* v___y_2966_, lean_object* v___y_2967_, lean_object* v___y_2968_){
_start:
{
lean_object* v_res_2969_; 
v_res_2969_ = lp_aesop_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0(v_msgData_2960_, v_macroStack_2961_, v___y_2962_, v___y_2963_, v___y_2964_, v___y_2965_, v___y_2966_, v___y_2967_);
lean_dec(v___y_2967_);
lean_dec_ref(v___y_2966_);
lean_dec(v___y_2965_);
lean_dec_ref(v___y_2964_);
lean_dec(v___y_2963_);
lean_dec_ref(v___y_2962_);
return v_res_2969_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg(lean_object* v_mvarId_2970_, lean_object* v_x_2971_, lean_object* v___y_2972_, lean_object* v___y_2973_, lean_object* v___y_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_){
_start:
{
lean_object* v___f_2979_; lean_object* v___x_2980_; 
lean_inc(v___y_2973_);
lean_inc_ref(v___y_2972_);
v___f_2979_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_withLCtx___at___00Aesop_Frontend_elabConfigUnsafe_spec__3___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2979_, 0, v_x_2971_);
lean_closure_set(v___f_2979_, 1, v___y_2972_);
lean_closure_set(v___f_2979_, 2, v___y_2973_);
v___x_2980_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2970_, v___f_2979_, v___y_2974_, v___y_2975_, v___y_2976_, v___y_2977_);
if (lean_obj_tag(v___x_2980_) == 0)
{
return v___x_2980_;
}
else
{
lean_object* v_a_2981_; lean_object* v___x_2983_; uint8_t v_isShared_2984_; uint8_t v_isSharedCheck_2988_; 
v_a_2981_ = lean_ctor_get(v___x_2980_, 0);
v_isSharedCheck_2988_ = !lean_is_exclusive(v___x_2980_);
if (v_isSharedCheck_2988_ == 0)
{
v___x_2983_ = v___x_2980_;
v_isShared_2984_ = v_isSharedCheck_2988_;
goto v_resetjp_2982_;
}
else
{
lean_inc(v_a_2981_);
lean_dec(v___x_2980_);
v___x_2983_ = lean_box(0);
v_isShared_2984_ = v_isSharedCheck_2988_;
goto v_resetjp_2982_;
}
v_resetjp_2982_:
{
lean_object* v___x_2986_; 
if (v_isShared_2984_ == 0)
{
v___x_2986_ = v___x_2983_;
goto v_reusejp_2985_;
}
else
{
lean_object* v_reuseFailAlloc_2987_; 
v_reuseFailAlloc_2987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2987_, 0, v_a_2981_);
v___x_2986_ = v_reuseFailAlloc_2987_;
goto v_reusejp_2985_;
}
v_reusejp_2985_:
{
return v___x_2986_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg___boxed(lean_object* v_mvarId_2989_, lean_object* v_x_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_){
_start:
{
lean_object* v_res_2998_; 
v_res_2998_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg(v_mvarId_2989_, v_x_2990_, v___y_2991_, v___y_2992_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_);
lean_dec(v___y_2996_);
lean_dec_ref(v___y_2995_);
lean_dec(v___y_2994_);
lean_dec_ref(v___y_2993_);
lean_dec(v___y_2992_);
lean_dec_ref(v___y_2991_);
return v_res_2998_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1(lean_object* v_00_u03b1_2999_, lean_object* v_mvarId_3000_, lean_object* v_x_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_){
_start:
{
lean_object* v___x_3009_; 
v___x_3009_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg(v_mvarId_3000_, v_x_3001_, v___y_3002_, v___y_3003_, v___y_3004_, v___y_3005_, v___y_3006_, v___y_3007_);
return v___x_3009_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___boxed(lean_object* v_00_u03b1_3010_, lean_object* v_mvarId_3011_, lean_object* v_x_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_, lean_object* v___y_3019_){
_start:
{
lean_object* v_res_3020_; 
v_res_3020_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1(v_00_u03b1_3010_, v_mvarId_3011_, v_x_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_, v___y_3017_, v___y_3018_);
lean_dec(v___y_3018_);
lean_dec_ref(v___y_3017_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
lean_dec(v___y_3014_);
lean_dec_ref(v___y_3013_);
return v_res_3020_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg(lean_object* v_opt_3021_, lean_object* v___y_3022_){
_start:
{
lean_object* v_options_3024_; uint8_t v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3027_; 
v_options_3024_ = lean_ctor_get(v___y_3022_, 2);
v___x_3025_ = lp_aesop_Aesop_Check_get(v_options_3024_, v_opt_3021_);
v___x_3026_ = lean_box(v___x_3025_);
v___x_3027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3027_, 0, v___x_3026_);
return v___x_3027_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg___boxed(lean_object* v_opt_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_){
_start:
{
lean_object* v_res_3031_; 
v_res_3031_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg(v_opt_3028_, v___y_3029_);
lean_dec_ref(v___y_3029_);
lean_dec_ref(v_opt_3028_);
return v_res_3031_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0(lean_object* v_opts_3032_, lean_object* v_forwardMaxDepth_x3f_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_, lean_object* v___y_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_){
_start:
{
uint8_t v_a_3042_; lean_object* v___y_3046_; lean_object* v_options_3049_; lean_object* v___x_3050_; uint8_t v___x_3051_; 
v_options_3049_ = lean_ctor_get(v___y_3038_, 2);
v___x_3050_ = lp_aesop_Aesop_aesop_dev_generateScript;
v___x_3051_ = lp_aesop_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Aesop_Frontend_TacticConfig_updateRuleSet_spec__0_spec__0_spec__1(v_options_3049_, v___x_3050_);
if (v___x_3051_ == 0)
{
uint8_t v_traceScript_3052_; 
v_traceScript_3052_ = lean_ctor_get_uint8(v_opts_3032_, sizeof(void*)*6 + 6);
if (v_traceScript_3052_ == 0)
{
lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v_a_3055_; uint8_t v___x_3056_; 
v___x_3053_ = lp_aesop_Aesop_Check_script;
v___x_3054_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg(v___x_3053_, v___y_3038_);
v_a_3055_ = lean_ctor_get(v___x_3054_, 0);
lean_inc(v_a_3055_);
v___x_3056_ = lean_unbox(v_a_3055_);
lean_dec(v_a_3055_);
if (v___x_3056_ == 0)
{
lean_object* v___x_3057_; lean_object* v___x_3058_; 
lean_dec_ref(v___x_3054_);
v___x_3057_ = lp_aesop_Aesop_Check_script_steps;
v___x_3058_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg(v___x_3057_, v___y_3038_);
v___y_3046_ = v___x_3058_;
goto v___jp_3045_;
}
else
{
v___y_3046_ = v___x_3054_;
goto v___jp_3045_;
}
}
else
{
v_a_3042_ = v_traceScript_3052_;
goto v___jp_3041_;
}
}
else
{
v_a_3042_ = v___x_3051_;
goto v___jp_3041_;
}
v___jp_3041_:
{
lean_object* v___x_3043_; lean_object* v___x_3044_; 
v___x_3043_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_3043_, 0, v_opts_3032_);
lean_ctor_set(v___x_3043_, 1, v_forwardMaxDepth_x3f_3033_);
lean_ctor_set_uint8(v___x_3043_, sizeof(void*)*2, v_a_3042_);
v___x_3044_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3044_, 0, v___x_3043_);
return v___x_3044_;
}
v___jp_3045_:
{
lean_object* v_a_3047_; uint8_t v___x_3048_; 
v_a_3047_ = lean_ctor_get(v___y_3046_, 0);
lean_inc(v_a_3047_);
lean_dec_ref(v___y_3046_);
v___x_3048_ = lean_unbox(v_a_3047_);
lean_dec(v_a_3047_);
v_a_3042_ = v___x_3048_;
goto v___jp_3041_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0___boxed(lean_object* v_opts_3059_, lean_object* v_forwardMaxDepth_x3f_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_){
_start:
{
lean_object* v_res_3068_; 
v_res_3068_ = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0(v_opts_3059_, v_forwardMaxDepth_x3f_3060_, v___y_3061_, v___y_3062_, v___y_3063_, v___y_3064_, v___y_3065_, v___y_3066_);
lean_dec(v___y_3066_);
lean_dec_ref(v___y_3065_);
lean_dec(v___y_3064_);
lean_dec_ref(v___y_3063_);
lean_dec(v___y_3062_);
lean_dec_ref(v___y_3061_);
return v_res_3068_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___lam__0(lean_object* v___y_3069_, lean_object* v_options_3070_, lean_object* v_c_3071_, lean_object* v_goal_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_, lean_object* v___y_3077_, lean_object* v___y_3078_){
_start:
{
lean_object* v___x_3080_; 
v___x_3080_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___y_3069_, v___y_3077_, v___y_3078_);
if (lean_obj_tag(v___x_3080_) == 0)
{
lean_object* v_a_3081_; lean_object* v___x_3082_; lean_object* v___x_3083_; 
v_a_3081_ = lean_ctor_get(v___x_3080_, 0);
lean_inc(v_a_3081_);
lean_dec_ref_known(v___x_3080_, 1);
v___x_3082_ = lean_box(0);
v___x_3083_ = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0(v_options_3070_, v___x_3082_, v___y_3073_, v___y_3074_, v___y_3075_, v___y_3076_, v___y_3077_, v___y_3078_);
if (lean_obj_tag(v___x_3083_) == 0)
{
lean_object* v_a_3084_; lean_object* v___x_3085_; 
v_a_3084_ = lean_ctor_get(v___x_3083_, 0);
lean_inc(v_a_3084_);
lean_dec_ref_known(v___x_3083_, 1);
v___x_3085_ = lp_aesop_Aesop_mkLocalRuleSet(v_a_3081_, v_a_3084_, v___y_3077_, v___y_3078_);
lean_dec(v_a_3084_);
lean_dec(v_a_3081_);
if (lean_obj_tag(v___x_3085_) == 0)
{
lean_object* v_a_3086_; lean_object* v___x_3087_; 
v_a_3086_ = lean_ctor_get(v___x_3085_, 0);
lean_inc(v_a_3086_);
lean_dec_ref_known(v___x_3085_, 1);
v___x_3087_ = lp_aesop_Aesop_Frontend_TacticConfig_updateRuleSet(v_a_3086_, v_c_3071_, v_goal_3072_, v___y_3073_, v___y_3074_, v___y_3075_, v___y_3076_, v___y_3077_, v___y_3078_);
return v___x_3087_;
}
else
{
lean_dec(v_goal_3072_);
return v___x_3085_;
}
}
else
{
lean_object* v_a_3088_; lean_object* v___x_3090_; uint8_t v_isShared_3091_; uint8_t v_isSharedCheck_3095_; 
lean_dec(v_a_3081_);
lean_dec(v_goal_3072_);
v_a_3088_ = lean_ctor_get(v___x_3083_, 0);
v_isSharedCheck_3095_ = !lean_is_exclusive(v___x_3083_);
if (v_isSharedCheck_3095_ == 0)
{
v___x_3090_ = v___x_3083_;
v_isShared_3091_ = v_isSharedCheck_3095_;
goto v_resetjp_3089_;
}
else
{
lean_inc(v_a_3088_);
lean_dec(v___x_3083_);
v___x_3090_ = lean_box(0);
v_isShared_3091_ = v_isSharedCheck_3095_;
goto v_resetjp_3089_;
}
v_resetjp_3089_:
{
lean_object* v___x_3093_; 
if (v_isShared_3091_ == 0)
{
v___x_3093_ = v___x_3090_;
goto v_reusejp_3092_;
}
else
{
lean_object* v_reuseFailAlloc_3094_; 
v_reuseFailAlloc_3094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3094_, 0, v_a_3088_);
v___x_3093_ = v_reuseFailAlloc_3094_;
goto v_reusejp_3092_;
}
v_reusejp_3092_:
{
return v___x_3093_;
}
}
}
}
else
{
lean_object* v_a_3096_; lean_object* v___x_3098_; uint8_t v_isShared_3099_; uint8_t v_isSharedCheck_3103_; 
lean_dec(v_goal_3072_);
lean_dec_ref(v_options_3070_);
v_a_3096_ = lean_ctor_get(v___x_3080_, 0);
v_isSharedCheck_3103_ = !lean_is_exclusive(v___x_3080_);
if (v_isSharedCheck_3103_ == 0)
{
v___x_3098_ = v___x_3080_;
v_isShared_3099_ = v_isSharedCheck_3103_;
goto v_resetjp_3097_;
}
else
{
lean_inc(v_a_3096_);
lean_dec(v___x_3080_);
v___x_3098_ = lean_box(0);
v_isShared_3099_ = v_isSharedCheck_3103_;
goto v_resetjp_3097_;
}
v_resetjp_3097_:
{
lean_object* v___x_3101_; 
if (v_isShared_3099_ == 0)
{
v___x_3101_ = v___x_3098_;
goto v_reusejp_3100_;
}
else
{
lean_object* v_reuseFailAlloc_3102_; 
v_reuseFailAlloc_3102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3102_, 0, v_a_3096_);
v___x_3101_ = v_reuseFailAlloc_3102_;
goto v_reusejp_3100_;
}
v_reusejp_3100_:
{
return v___x_3101_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___lam__0___boxed(lean_object* v___y_3104_, lean_object* v_options_3105_, lean_object* v_c_3106_, lean_object* v_goal_3107_, lean_object* v___y_3108_, lean_object* v___y_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_){
_start:
{
lean_object* v_res_3115_; 
v_res_3115_ = lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___lam__0(v___y_3104_, v_options_3105_, v_c_3106_, v_goal_3107_, v___y_3108_, v___y_3109_, v___y_3110_, v___y_3111_, v___y_3112_, v___y_3113_);
lean_dec(v___y_3113_);
lean_dec_ref(v___y_3112_);
lean_dec(v___y_3111_);
lean_dec_ref(v___y_3110_);
lean_dec(v___y_3109_);
lean_dec_ref(v___y_3108_);
lean_dec_ref(v_c_3106_);
return v_res_3115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__2(lean_object* v_x_3116_, lean_object* v_x_3117_){
_start:
{
if (lean_obj_tag(v_x_3117_) == 0)
{
return v_x_3116_;
}
else
{
lean_object* v_key_3118_; lean_object* v_tail_3119_; lean_object* v___x_3120_; 
v_key_3118_ = lean_ctor_get(v_x_3117_, 0);
lean_inc(v_key_3118_);
v_tail_3119_ = lean_ctor_get(v_x_3117_, 2);
lean_inc(v_tail_3119_);
lean_dec_ref_known(v_x_3117_, 3);
v___x_3120_ = lean_array_push(v_x_3116_, v_key_3118_);
v_x_3116_ = v___x_3120_;
v_x_3117_ = v_tail_3119_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3(lean_object* v_as_3122_, size_t v_i_3123_, size_t v_stop_3124_, lean_object* v_b_3125_){
_start:
{
uint8_t v___x_3126_; 
v___x_3126_ = lean_usize_dec_eq(v_i_3123_, v_stop_3124_);
if (v___x_3126_ == 0)
{
lean_object* v___x_3127_; lean_object* v___x_3128_; size_t v___x_3129_; size_t v___x_3130_; 
v___x_3127_ = lean_array_uget_borrowed(v_as_3122_, v_i_3123_);
lean_inc(v___x_3127_);
v___x_3128_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__2(v_b_3125_, v___x_3127_);
v___x_3129_ = ((size_t)1ULL);
v___x_3130_ = lean_usize_add(v_i_3123_, v___x_3129_);
v_i_3123_ = v___x_3130_;
v_b_3125_ = v___x_3128_;
goto _start;
}
else
{
return v_b_3125_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3___boxed(lean_object* v_as_3132_, lean_object* v_i_3133_, lean_object* v_stop_3134_, lean_object* v_b_3135_){
_start:
{
size_t v_i_boxed_3136_; size_t v_stop_boxed_3137_; lean_object* v_res_3138_; 
v_i_boxed_3136_ = lean_unbox_usize(v_i_3133_);
lean_dec(v_i_3133_);
v_stop_boxed_3137_ = lean_unbox_usize(v_stop_3134_);
lean_dec(v_stop_3134_);
v_res_3138_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3(v_as_3132_, v_i_boxed_3136_, v_stop_boxed_3137_, v_b_3135_);
lean_dec_ref(v_as_3132_);
return v_res_3138_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(lean_object* v_goal_3139_, lean_object* v_c_3140_, lean_object* v_a_3141_, lean_object* v_a_3142_, lean_object* v_a_3143_, lean_object* v_a_3144_, lean_object* v_a_3145_, lean_object* v_a_3146_){
_start:
{
lean_object* v_enabledRuleSets_3148_; lean_object* v_options_3149_; lean_object* v___y_3151_; lean_object* v_size_3154_; lean_object* v_buckets_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; lean_object* v___x_3158_; uint8_t v___x_3159_; 
v_enabledRuleSets_3148_ = lean_ctor_get(v_c_3140_, 2);
v_options_3149_ = lean_ctor_get(v_c_3140_, 3);
lean_inc_ref(v_options_3149_);
v_size_3154_ = lean_ctor_get(v_enabledRuleSets_3148_, 0);
v_buckets_3155_ = lean_ctor_get(v_enabledRuleSets_3148_, 1);
v___x_3156_ = lean_mk_empty_array_with_capacity(v_size_3154_);
v___x_3157_ = lean_unsigned_to_nat(0u);
v___x_3158_ = lean_array_get_size(v_buckets_3155_);
v___x_3159_ = lean_nat_dec_lt(v___x_3157_, v___x_3158_);
if (v___x_3159_ == 0)
{
v___y_3151_ = v___x_3156_;
goto v___jp_3150_;
}
else
{
uint8_t v___x_3160_; 
v___x_3160_ = lean_nat_dec_le(v___x_3158_, v___x_3158_);
if (v___x_3160_ == 0)
{
if (v___x_3159_ == 0)
{
v___y_3151_ = v___x_3156_;
goto v___jp_3150_;
}
else
{
size_t v___x_3161_; size_t v___x_3162_; lean_object* v___x_3163_; 
v___x_3161_ = ((size_t)0ULL);
v___x_3162_ = lean_usize_of_nat(v___x_3158_);
v___x_3163_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3(v_buckets_3155_, v___x_3161_, v___x_3162_, v___x_3156_);
v___y_3151_ = v___x_3163_;
goto v___jp_3150_;
}
}
else
{
size_t v___x_3164_; size_t v___x_3165_; lean_object* v___x_3166_; 
v___x_3164_ = ((size_t)0ULL);
v___x_3165_ = lean_usize_of_nat(v___x_3158_);
v___x_3166_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__3(v_buckets_3155_, v___x_3164_, v___x_3165_, v___x_3156_);
v___y_3151_ = v___x_3166_;
goto v___jp_3150_;
}
}
v___jp_3150_:
{
lean_object* v___f_3152_; lean_object* v___x_3153_; 
lean_inc(v_goal_3139_);
v___f_3152_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___lam__0___boxed), 11, 4);
lean_closure_set(v___f_3152_, 0, v___y_3151_);
lean_closure_set(v___f_3152_, 1, v_options_3149_);
lean_closure_set(v___f_3152_, 2, v_c_3140_);
lean_closure_set(v___f_3152_, 3, v_goal_3139_);
v___x_3153_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__1___redArg(v_goal_3139_, v___f_3152_, v_a_3141_, v_a_3142_, v_a_3143_, v_a_3144_, v_a_3145_, v_a_3146_);
return v___x_3153_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet___boxed(lean_object* v_goal_3167_, lean_object* v_c_3168_, lean_object* v_a_3169_, lean_object* v_a_3170_, lean_object* v_a_3171_, lean_object* v_a_3172_, lean_object* v_a_3173_, lean_object* v_a_3174_, lean_object* v_a_3175_){
_start:
{
lean_object* v_res_3176_; 
v_res_3176_ = lp_aesop_Aesop_Frontend_TacticConfig_getRuleSet(v_goal_3167_, v_c_3168_, v_a_3169_, v_a_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_);
lean_dec(v_a_3174_);
lean_dec_ref(v_a_3173_);
lean_dec(v_a_3172_);
lean_dec_ref(v_a_3171_);
lean_dec(v_a_3170_);
lean_dec_ref(v_a_3169_);
return v_res_3176_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0(lean_object* v_opt_3177_, lean_object* v___y_3178_, lean_object* v___y_3179_, lean_object* v___y_3180_, lean_object* v___y_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_){
_start:
{
lean_object* v___x_3185_; 
v___x_3185_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___redArg(v_opt_3177_, v___y_3182_);
return v___x_3185_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0___boxed(lean_object* v_opt_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_){
_start:
{
lean_object* v_res_3194_; 
v_res_3194_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_TacticConfig_getRuleSet_spec__0_spec__0(v_opt_3186_, v___y_3187_, v___y_3188_, v___y_3189_, v___y_3190_, v___y_3191_, v___y_3192_);
lean_dec(v___y_3192_);
lean_dec_ref(v___y_3191_);
lean_dec(v___y_3190_);
lean_dec_ref(v___y_3189_);
lean_dec(v___y_3188_);
lean_dec_ref(v___y_3187_);
lean_dec_ref(v_opt_3186_);
return v_res_3194_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_RuleExpr(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_SyntheticMVars(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Eval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_RuleExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_SyntheticMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Lean_Parser_Category_Aesop_tactic__clause = _init_lp_aesop_Lean_Parser_Category_Aesop_tactic__clause();
lean_mark_persistent(lp_aesop_Lean_Parser_Category_Aesop_tactic__clause);
res = lp_aesop___private_Aesop_Frontend_Tactic_0__Aesop_Frontend_Parser_initFn_00___x40_Aesop_Frontend_Tactic_1696840279____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_RuleExpr(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin);
lean_object* initialize_Lean_Elab_SyntheticMVars(uint8_t builtin);
lean_object* initialize_Lean_Meta_Eval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_RuleExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_SyntheticMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Frontend_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Frontend_Tactic(builtin);
}
#ifdef __cplusplus
}
#endif
