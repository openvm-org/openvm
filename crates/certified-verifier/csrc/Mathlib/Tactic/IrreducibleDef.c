// Lean compiler output
// Module: Mathlib.Tactic.IrreducibleDef
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Eqns public import Mathlib.Util.TermReduce
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_addProtected(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_mkArray3___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Syntax_formatStx(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_extractMacroScopes(lean_object*);
lean_object* lean_name_append_after(lean_object*, lean_object*);
lean_object* l_Lean_MacroScopesView_review(lean_object*);
lean_object* l_Lean_mkIdentFrom(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkProj(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "IrreducibleDef"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(39, 86, 5, 247, 167, 179, 63, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(154, 148, 158, 0, 1, 159, 86, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(59, 30, 62, 234, 60, 136, 55, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(141, 225, 195, 158, 170, 125, 145, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(164, 65, 127, 33, 158, 213, 167, 190)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "termEta_helper_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(205, 247, 107, 117, 52, 245, 159, 75)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "eta_helper "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__21_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__20_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__23_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__16_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__24_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__25_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper__ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__25_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "not an equation: "};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "termVal_proj_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 55, 123, 130, 106, 89, 31, 79)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "val_proj "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__23_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj__ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__11_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__16_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__15_value),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__18_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__12_value),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__20 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__20_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__21_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__22 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__22_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__23 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__23_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__24 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__24_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__24_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subtype"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__26 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__26_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__26_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__29 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__29_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__30 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__30_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__30_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__31 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__31_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__29_value),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__31_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__32 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__32_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__33 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__33_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__33_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__35 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__35_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__36 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__36_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "commandStop_at_first_error__"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(4, 99, 6, 224, 110, 216, 0, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "stop_at_first_error"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ppLine"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(117, 61, 38, 245, 158, 59, 171, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__11_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__12_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error____ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "irredDefLemma"};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 203, 227, 68, 81, 158, 73, 234)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__4_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lemma"};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__5_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__7_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__9_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__3_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__11_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__12_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__15_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__36_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__16_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__17_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__0_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__18_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Command_irredDefLemma = (const lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__19_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "command_Irreducible_def____"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(27, 94, 157, 71, 100, 93, 43, 49)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(113, 135, 0, 93, 130, 217, 220, 132)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__3_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "irreducible_def"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__5_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__4_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__6_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__7_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(210, 155, 24, 168, 139, 44, 164, 47)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__9_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__11_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__13_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__11_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__15_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ppIndent"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(240, 142, 232, 190, 100, 212, 29, 41)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__17_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optDeclSig"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(75, 210, 209, 245, 33, 1, 43, 130)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__20 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__20_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__17_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__20_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__15_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__21_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__22 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__22_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declVal"};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__23 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__23_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__23_value),LEAN_SCALAR_PTR_LITERAL(19, 167, 222, 34, 119, 174, 4, 130)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__24 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__24_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__24_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__25 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__25_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__22_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__25_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__26 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__26_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__26_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__27 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__27_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Command_command__Irreducible__def________ = (const lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__27_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "val_proj"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "theorem"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(238, 116, 137, 74, 194, 103, 58, 54)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eta_helper"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "TermReduce"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "deltaStx"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "delta%"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "by"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__9_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "delta"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "rwSeq"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "rw"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__15_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__16_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rwRuleSeq"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__17_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__18_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rwRule"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__19_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "show"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__20 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__20_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_=_"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(167, 251, 107, 62, 223, 239, 203, 78)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__22 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__22_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__23 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__23_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fromTerm"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__24 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__24_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "from"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__25 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__25_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Subtype.ext"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__26 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__26_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__27;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ext"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__28 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__28_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__29 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__29_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__30 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__30_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fieldIdx"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__31 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__31_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(243, 141, 165, 29, 238, 211, 61, 163)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__32 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__32_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "2"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__33 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__33_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "symm"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__34 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__34_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__35;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__34_value),LEAN_SCALAR_PTR_LITERAL(56, 55, 151, 246, 89, 173, 177, 197)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__36 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__36_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__37 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__37_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__38 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__38_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "attribute"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__39 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__39_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(79, 30, 18, 84, 71, 173, 185, 159)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "attrInstance"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__41 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__41_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "attrKind"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__42 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__42_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__43 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__43_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__44 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__44_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__43_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__44_value),LEAN_SCALAR_PTR_LITERAL(107, 67, 254, 234, 65, 174, 209, 53)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "irreducible"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__46 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__46_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__47;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__46_value),LEAN_SCALAR_PTR_LITERAL(61, 207, 43, 193, 214, 202, 115, 95)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__48 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__48_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "eqns"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__49 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__49_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__49_value),LEAN_SCALAR_PTR_LITERAL(189, 205, 217, 20, 6, 134, 86, 247)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__50 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__50_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "private"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__51 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__51_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__51_value),LEAN_SCALAR_PTR_LITERAL(213, 248, 16, 228, 25, 227, 72, 143)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52_value;
static const lean_array_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "opaque"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__54 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__54_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__54_value),LEAN_SCALAR_PTR_LITERAL(111, 217, 152, 21, 13, 97, 204, 102)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "wrapped"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__56 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__56_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__57;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__56_value),LEAN_SCALAR_PTR_LITERAL(247, 186, 68, 24, 215, 140, 232, 224)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__58 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__58_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__59 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__59_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__59_value),LEAN_SCALAR_PTR_LITERAL(22, 101, 130, 251, 183, 19, 113, 82)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__61 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__61_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__61_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__63 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__63_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__64 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__64_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__64_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__66;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__67 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__67_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__68 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__68_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__69 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__69_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__69_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__71 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__71_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "explicitUniv"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__72 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__72_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__72_value),LEAN_SCALAR_PTR_LITERAL(206, 217, 218, 63, 82, 102, 26, 62)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ".{"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__74 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__74_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__75 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__75_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__76 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__76_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__77 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__77_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__77_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__79 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__79_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__80 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__80_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__80_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__82 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__82_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__83 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__83_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__84_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__84;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__83_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__85 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__85_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__85_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__86 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__86_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__87 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__87_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Termination"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__88 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__88_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "suffix"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__89 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__89_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__88_value),LEAN_SCALAR_PTR_LITERAL(128, 225, 226, 49, 186, 161, 212, 105)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__89_value),LEAN_SCALAR_PTR_LITERAL(245, 187, 99, 45, 217, 244, 244, 120)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "definition"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__91 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__91_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__91_value),LEAN_SCALAR_PTR_LITERAL(248, 187, 217, 228, 39, 184, 218, 135)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__93 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__93_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__94_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__94;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__91_value),LEAN_SCALAR_PTR_LITERAL(25, 9, 118, 99, 178, 247, 49, 165)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__95 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__95_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__96 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__96_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__96_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__98_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__98;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "unsupported modifiers "};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__99 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__99_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__101 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__101_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__101_value),LEAN_SCALAR_PTR_LITERAL(79, 160, 60, 55, 136, 115, 80, 144)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "noncomputable"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__103 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__103_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__103_value),LEAN_SCALAR_PTR_LITERAL(103, 51, 14, 127, 141, 25, 244, 148)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "protected"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__105 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__105_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__105_value),LEAN_SCALAR_PTR_LITERAL(33, 80, 123, 180, 50, 194, 119, 199)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "attributes"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__107 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__107_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__107_value),LEAN_SCALAR_PTR_LITERAL(66, 184, 196, 169, 25, 125, 40, 35)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__109 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__109_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__109_value),LEAN_SCALAR_PTR_LITERAL(44, 76, 179, 33, 27, 4, 201, 125)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__111_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_def"};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__111 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__111_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__23_value),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__112 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__112_value;
static const lean_array_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__113 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__113_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(26, 9, 103, 232, 183, 57, 246, 75)}};
static const lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116 = (const lean_object*)&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = lean_box(0);
v___x_58_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v___x_57_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg(){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_61_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0);
v___x_62_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___boxed(lean_object* v___y_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg();
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0(lean_object* v_00_u03b1_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg();
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___boxed(lean_object* v_00_u03b1_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0(v_00_u03b1_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg(lean_object* v_e_83_, lean_object* v___y_84_){
_start:
{
uint8_t v___x_86_; 
v___x_86_ = l_Lean_Expr_hasMVar(v_e_83_);
if (v___x_86_ == 0)
{
lean_object* v___x_87_; 
v___x_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_87_, 0, v_e_83_);
return v___x_87_;
}
else
{
lean_object* v___x_88_; lean_object* v_mctx_89_; lean_object* v___x_90_; lean_object* v_fst_91_; lean_object* v_snd_92_; lean_object* v___x_93_; lean_object* v_cache_94_; lean_object* v_zetaDeltaFVarIds_95_; lean_object* v_postponed_96_; lean_object* v_diag_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_106_; 
v___x_88_ = lean_st_ref_get(v___y_84_);
v_mctx_89_ = lean_ctor_get(v___x_88_, 0);
lean_inc_ref(v_mctx_89_);
lean_dec(v___x_88_);
v___x_90_ = l_Lean_instantiateMVarsCore(v_mctx_89_, v_e_83_);
v_fst_91_ = lean_ctor_get(v___x_90_, 0);
lean_inc(v_fst_91_);
v_snd_92_ = lean_ctor_get(v___x_90_, 1);
lean_inc(v_snd_92_);
lean_dec_ref(v___x_90_);
v___x_93_ = lean_st_ref_take(v___y_84_);
v_cache_94_ = lean_ctor_get(v___x_93_, 1);
v_zetaDeltaFVarIds_95_ = lean_ctor_get(v___x_93_, 2);
v_postponed_96_ = lean_ctor_get(v___x_93_, 3);
v_diag_97_ = lean_ctor_get(v___x_93_, 4);
v_isSharedCheck_106_ = !lean_is_exclusive(v___x_93_);
if (v_isSharedCheck_106_ == 0)
{
lean_object* v_unused_107_; 
v_unused_107_ = lean_ctor_get(v___x_93_, 0);
lean_dec(v_unused_107_);
v___x_99_ = v___x_93_;
v_isShared_100_ = v_isSharedCheck_106_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_diag_97_);
lean_inc(v_postponed_96_);
lean_inc(v_zetaDeltaFVarIds_95_);
lean_inc(v_cache_94_);
lean_dec(v___x_93_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_106_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_102_; 
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 0, v_snd_92_);
v___x_102_ = v___x_99_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v_snd_92_);
lean_ctor_set(v_reuseFailAlloc_105_, 1, v_cache_94_);
lean_ctor_set(v_reuseFailAlloc_105_, 2, v_zetaDeltaFVarIds_95_);
lean_ctor_set(v_reuseFailAlloc_105_, 3, v_postponed_96_);
lean_ctor_set(v_reuseFailAlloc_105_, 4, v_diag_97_);
v___x_102_ = v_reuseFailAlloc_105_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_103_ = lean_st_ref_set(v___y_84_, v___x_102_);
v___x_104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_104_, 0, v_fst_91_);
return v___x_104_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg___boxed(lean_object* v_e_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg(v_e_108_, v___y_109_);
lean_dec(v___y_109_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2(lean_object* v_e_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg(v_e_112_, v___y_116_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___boxed(lean_object* v_e_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2(v_e_121_, v___y_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
lean_dec(v___y_123_);
lean_dec_ref(v___y_122_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___lam__0(lean_object* v_k_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v_b_133_, lean_object* v_c_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v___x_140_; 
lean_inc(v___y_138_);
lean_inc_ref(v___y_137_);
lean_inc(v___y_136_);
lean_inc_ref(v___y_135_);
lean_inc(v___y_132_);
lean_inc_ref(v___y_131_);
v___x_140_ = lean_apply_9(v_k_130_, v_b_133_, v_c_134_, v___y_131_, v___y_132_, v___y_135_, v___y_136_, v___y_137_, v___y_138_, lean_box(0));
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___lam__0___boxed(lean_object* v_k_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v_b_144_, lean_object* v_c_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___lam__0(v_k_141_, v___y_142_, v___y_143_, v_b_144_, v_c_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec(v___y_147_);
lean_dec_ref(v___y_146_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg(lean_object* v_e_152_, lean_object* v_k_153_, uint8_t v_cleanupAnnotations_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_){
_start:
{
lean_object* v___f_162_; uint8_t v___x_163_; uint8_t v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
lean_inc(v___y_156_);
lean_inc_ref(v___y_155_);
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_162_, 0, v_k_153_);
lean_closure_set(v___f_162_, 1, v___y_155_);
lean_closure_set(v___f_162_, 2, v___y_156_);
v___x_163_ = 1;
v___x_164_ = 0;
v___x_165_ = lean_box(0);
v___x_166_ = l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_box(0), v_e_152_, v___x_163_, v___x_164_, v___x_163_, v___x_164_, v___x_165_, v___f_162_, v_cleanupAnnotations_154_, v___y_157_, v___y_158_, v___y_159_, v___y_160_);
if (lean_obj_tag(v___x_166_) == 0)
{
return v___x_166_;
}
else
{
lean_object* v_a_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_174_; 
v_a_167_ = lean_ctor_get(v___x_166_, 0);
v_isSharedCheck_174_ = !lean_is_exclusive(v___x_166_);
if (v_isSharedCheck_174_ == 0)
{
v___x_169_ = v___x_166_;
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_a_167_);
lean_dec(v___x_166_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_172_; 
if (v_isShared_170_ == 0)
{
v___x_172_ = v___x_169_;
goto v_reusejp_171_;
}
else
{
lean_object* v_reuseFailAlloc_173_; 
v_reuseFailAlloc_173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_173_, 0, v_a_167_);
v___x_172_ = v_reuseFailAlloc_173_;
goto v_reusejp_171_;
}
v_reusejp_171_:
{
return v___x_172_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg___boxed(lean_object* v_e_175_, lean_object* v_k_176_, lean_object* v_cleanupAnnotations_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_185_; lean_object* v_res_186_; 
v_cleanupAnnotations_boxed_185_ = lean_unbox(v_cleanupAnnotations_177_);
v_res_186_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg(v_e_175_, v_k_176_, v_cleanupAnnotations_boxed_185_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_);
lean_dec(v___y_183_);
lean_dec_ref(v___y_182_);
lean_dec(v___y_181_);
lean_dec_ref(v___y_180_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3(lean_object* v_00_u03b1_187_, lean_object* v_e_188_, lean_object* v_k_189_, uint8_t v_cleanupAnnotations_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg(v_e_188_, v_k_189_, v_cleanupAnnotations_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___boxed(lean_object* v_00_u03b1_199_, lean_object* v_e_200_, lean_object* v_k_201_, lean_object* v_cleanupAnnotations_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_210_; lean_object* v_res_211_; 
v_cleanupAnnotations_boxed_210_ = lean_unbox(v_cleanupAnnotations_202_);
v_res_211_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3(v_00_u03b1_199_, v_e_200_, v_k_201_, v_cleanupAnnotations_boxed_210_, v___y_203_, v___y_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___lam__0(lean_object* v___x_212_, uint8_t v___x_213_, uint8_t v___x_214_, lean_object* v_xs_215_, lean_object* v_rhs_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_224_ = l_Lean_mkAppN(v___x_212_, v_xs_215_);
v___x_225_ = l_Lean_Expr_headBeta(v___x_224_);
v___x_226_ = l_Lean_Meta_mkEq(v___x_225_, v_rhs_216_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
if (lean_obj_tag(v___x_226_) == 0)
{
lean_object* v_a_227_; uint8_t v___x_228_; lean_object* v___x_229_; 
v_a_227_ = lean_ctor_get(v___x_226_, 0);
lean_inc(v_a_227_);
lean_dec_ref_known(v___x_226_, 1);
v___x_228_ = 1;
v___x_229_ = l_Lean_Meta_mkForallFVars(v_xs_215_, v_a_227_, v___x_213_, v___x_214_, v___x_214_, v___x_228_, v___y_219_, v___y_220_, v___y_221_, v___y_222_);
return v___x_229_;
}
else
{
return v___x_226_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___lam__0___boxed(lean_object* v___x_230_, lean_object* v___x_231_, lean_object* v___x_232_, lean_object* v_xs_233_, lean_object* v_rhs_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
uint8_t v___x_5049__boxed_242_; uint8_t v___x_5050__boxed_243_; lean_object* v_res_244_; 
v___x_5049__boxed_242_ = lean_unbox(v___x_231_);
v___x_5050__boxed_243_ = lean_unbox(v___x_232_);
v_res_244_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___lam__0(v___x_230_, v___x_5049__boxed_242_, v___x_5050__boxed_243_, v_xs_233_, v_rhs_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
lean_dec_ref(v_xs_233_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__1(lean_object* v_msgData_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
lean_object* v___x_251_; lean_object* v_env_252_; lean_object* v___x_253_; lean_object* v_mctx_254_; lean_object* v_lctx_255_; lean_object* v_options_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_251_ = lean_st_ref_get(v___y_249_);
v_env_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc_ref(v_env_252_);
lean_dec(v___x_251_);
v___x_253_ = lean_st_ref_get(v___y_247_);
v_mctx_254_ = lean_ctor_get(v___x_253_, 0);
lean_inc_ref(v_mctx_254_);
lean_dec(v___x_253_);
v_lctx_255_ = lean_ctor_get(v___y_246_, 2);
v_options_256_ = lean_ctor_get(v___y_248_, 2);
lean_inc_ref(v_options_256_);
lean_inc_ref(v_lctx_255_);
v___x_257_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_257_, 0, v_env_252_);
lean_ctor_set(v___x_257_, 1, v_mctx_254_);
lean_ctor_set(v___x_257_, 2, v_lctx_255_);
lean_ctor_set(v___x_257_, 3, v_options_256_);
v___x_258_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_msgData_245_);
v___x_259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_259_, 0, v___x_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__1___boxed(lean_object* v_msgData_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__1(v_msgData_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_266_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0(void){
_start:
{
lean_object* v___x_267_; lean_object* v___x_268_; 
v___x_267_ = lean_box(1);
v___x_268_ = l_Lean_MessageData_ofFormat(v___x_267_);
return v___x_268_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__3(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_272_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__2));
v___x_273_ = l_Lean_MessageData_ofFormat(v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6(lean_object* v_x_274_, lean_object* v_x_275_){
_start:
{
if (lean_obj_tag(v_x_275_) == 0)
{
return v_x_274_;
}
else
{
lean_object* v_head_276_; lean_object* v_tail_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_299_; 
v_head_276_ = lean_ctor_get(v_x_275_, 0);
v_tail_277_ = lean_ctor_get(v_x_275_, 1);
v_isSharedCheck_299_ = !lean_is_exclusive(v_x_275_);
if (v_isSharedCheck_299_ == 0)
{
v___x_279_ = v_x_275_;
v_isShared_280_ = v_isSharedCheck_299_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_tail_277_);
lean_inc(v_head_276_);
lean_dec(v_x_275_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_299_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v_before_281_; lean_object* v___x_283_; uint8_t v_isShared_284_; uint8_t v_isSharedCheck_297_; 
v_before_281_ = lean_ctor_get(v_head_276_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v_head_276_);
if (v_isSharedCheck_297_ == 0)
{
lean_object* v_unused_298_; 
v_unused_298_ = lean_ctor_get(v_head_276_, 1);
lean_dec(v_unused_298_);
v___x_283_ = v_head_276_;
v_isShared_284_ = v_isSharedCheck_297_;
goto v_resetjp_282_;
}
else
{
lean_inc(v_before_281_);
lean_dec(v_head_276_);
v___x_283_ = lean_box(0);
v_isShared_284_ = v_isSharedCheck_297_;
goto v_resetjp_282_;
}
v_resetjp_282_:
{
lean_object* v___x_285_; lean_object* v___x_287_; 
v___x_285_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0);
if (v_isShared_284_ == 0)
{
lean_ctor_set_tag(v___x_283_, 7);
lean_ctor_set(v___x_283_, 1, v___x_285_);
lean_ctor_set(v___x_283_, 0, v_x_274_);
v___x_287_ = v___x_283_;
goto v_reusejp_286_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_x_274_);
lean_ctor_set(v_reuseFailAlloc_296_, 1, v___x_285_);
v___x_287_ = v_reuseFailAlloc_296_;
goto v_reusejp_286_;
}
v_reusejp_286_:
{
lean_object* v___x_288_; lean_object* v___x_290_; 
v___x_288_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__3);
if (v_isShared_280_ == 0)
{
lean_ctor_set_tag(v___x_279_, 7);
lean_ctor_set(v___x_279_, 1, v___x_288_);
lean_ctor_set(v___x_279_, 0, v___x_287_);
v___x_290_ = v___x_279_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_295_; 
v_reuseFailAlloc_295_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_295_, 0, v___x_287_);
lean_ctor_set(v_reuseFailAlloc_295_, 1, v___x_288_);
v___x_290_ = v_reuseFailAlloc_295_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_291_ = l_Lean_MessageData_ofSyntax(v_before_281_);
v___x_292_ = l_Lean_indentD(v___x_291_);
v___x_293_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_293_, 0, v___x_290_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
v_x_274_ = v___x_293_;
v_x_275_ = v_tail_277_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5(lean_object* v_opts_300_, lean_object* v_opt_301_){
_start:
{
lean_object* v_name_302_; lean_object* v_defValue_303_; lean_object* v_map_304_; lean_object* v___x_305_; 
v_name_302_ = lean_ctor_get(v_opt_301_, 0);
v_defValue_303_ = lean_ctor_get(v_opt_301_, 1);
v_map_304_ = lean_ctor_get(v_opts_300_, 0);
v___x_305_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_304_, v_name_302_);
if (lean_obj_tag(v___x_305_) == 0)
{
uint8_t v___x_306_; 
v___x_306_ = lean_unbox(v_defValue_303_);
return v___x_306_;
}
else
{
lean_object* v_val_307_; 
v_val_307_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_val_307_);
lean_dec_ref_known(v___x_305_, 1);
if (lean_obj_tag(v_val_307_) == 1)
{
uint8_t v_v_308_; 
v_v_308_ = lean_ctor_get_uint8(v_val_307_, 0);
lean_dec_ref_known(v_val_307_, 0);
return v_v_308_;
}
else
{
uint8_t v___x_309_; 
lean_dec(v_val_307_);
v___x_309_ = lean_unbox(v_defValue_303_);
return v___x_309_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5___boxed(lean_object* v_opts_310_, lean_object* v_opt_311_){
_start:
{
uint8_t v_res_312_; lean_object* v_r_313_; 
v_res_312_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5(v_opts_310_, v_opt_311_);
lean_dec_ref(v_opt_311_);
lean_dec_ref(v_opts_310_);
v_r_313_ = lean_box(v_res_312_);
return v_r_313_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_317_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__1));
v___x_318_ = l_Lean_MessageData_ofFormat(v___x_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg(lean_object* v_msgData_319_, lean_object* v_macroStack_320_, lean_object* v___y_321_){
_start:
{
lean_object* v_options_323_; lean_object* v___x_324_; uint8_t v___x_325_; 
v_options_323_ = lean_ctor_get(v___y_321_, 2);
v___x_324_ = l_Lean_Elab_pp_macroStack;
v___x_325_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5(v_options_323_, v___x_324_);
if (v___x_325_ == 0)
{
lean_object* v___x_326_; 
lean_dec(v_macroStack_320_);
v___x_326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_326_, 0, v_msgData_319_);
return v___x_326_;
}
else
{
if (lean_obj_tag(v_macroStack_320_) == 0)
{
lean_object* v___x_327_; 
v___x_327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_327_, 0, v_msgData_319_);
return v___x_327_;
}
else
{
lean_object* v_head_328_; lean_object* v_after_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_344_; 
v_head_328_ = lean_ctor_get(v_macroStack_320_, 0);
lean_inc(v_head_328_);
v_after_329_ = lean_ctor_get(v_head_328_, 1);
v_isSharedCheck_344_ = !lean_is_exclusive(v_head_328_);
if (v_isSharedCheck_344_ == 0)
{
lean_object* v_unused_345_; 
v_unused_345_ = lean_ctor_get(v_head_328_, 0);
lean_dec(v_unused_345_);
v___x_331_ = v_head_328_;
v_isShared_332_ = v_isSharedCheck_344_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_after_329_);
lean_dec(v_head_328_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_344_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_333_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0);
if (v_isShared_332_ == 0)
{
lean_ctor_set_tag(v___x_331_, 7);
lean_ctor_set(v___x_331_, 1, v___x_333_);
lean_ctor_set(v___x_331_, 0, v_msgData_319_);
v___x_335_ = v___x_331_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v_msgData_319_);
lean_ctor_set(v_reuseFailAlloc_343_, 1, v___x_333_);
v___x_335_ = v_reuseFailAlloc_343_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_msgData_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_336_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2);
v___x_337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_335_);
lean_ctor_set(v___x_337_, 1, v___x_336_);
v___x_338_ = l_Lean_MessageData_ofSyntax(v_after_329_);
v___x_339_ = l_Lean_indentD(v___x_338_);
v_msgData_340_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_340_, 0, v___x_337_);
lean_ctor_set(v_msgData_340_, 1, v___x_339_);
v___x_341_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6(v_msgData_340_, v_macroStack_320_);
v___x_342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
return v___x_342_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_346_, lean_object* v_macroStack_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg(v_msgData_346_, v_macroStack_347_, v___y_348_);
lean_dec_ref(v___y_348_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg(lean_object* v_msg_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v_ref_359_; lean_object* v___x_360_; lean_object* v_a_361_; lean_object* v_macroStack_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_373_; 
v_ref_359_ = lean_ctor_get(v___y_356_, 5);
v___x_360_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__1(v_msg_351_, v___y_354_, v___y_355_, v___y_356_, v___y_357_);
v_a_361_ = lean_ctor_get(v___x_360_, 0);
lean_inc(v_a_361_);
lean_dec_ref(v___x_360_);
v_macroStack_362_ = lean_ctor_get(v___y_352_, 1);
v___x_363_ = l_Lean_Elab_getBetterRef(v_ref_359_, v_macroStack_362_);
lean_inc(v_macroStack_362_);
v___x_364_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg(v_a_361_, v_macroStack_362_, v___y_356_);
v_a_365_ = lean_ctor_get(v___x_364_, 0);
v_isSharedCheck_373_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_373_ == 0)
{
v___x_367_ = v___x_364_;
v_isShared_368_ = v_isSharedCheck_373_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_364_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_373_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
lean_object* v___x_369_; lean_object* v___x_371_; 
v___x_369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_363_);
lean_ctor_set(v___x_369_, 1, v_a_365_);
if (v_isShared_368_ == 0)
{
lean_ctor_set_tag(v___x_367_, 1);
lean_ctor_set(v___x_367_, 0, v___x_369_);
v___x_371_ = v___x_367_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_369_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg___boxed(lean_object* v_msg_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg(v_msg_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
lean_dec(v___y_376_);
lean_dec_ref(v___y_375_);
return v_res_382_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__3(void){
_start:
{
lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_387_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__2));
v___x_388_ = l_Lean_stringToMessageData(v___x_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg(lean_object* v_stx_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_){
_start:
{
lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_397_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__16));
lean_inc(v_stx_389_);
v___x_398_ = l_Lean_Syntax_isOfKind(v_stx_389_, v___x_397_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; 
lean_dec(v_stx_389_);
v___x_399_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg();
return v___x_399_;
}
else
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_400_ = lean_unsigned_to_nat(1u);
v___x_401_ = l_Lean_Syntax_getArg(v_stx_389_, v___x_400_);
lean_dec(v_stx_389_);
v___x_402_ = lean_box(0);
v___x_403_ = l_Lean_Elab_Term_elabTerm(v___x_401_, v___x_402_, v___x_398_, v___x_398_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
if (lean_obj_tag(v___x_403_) == 0)
{
lean_object* v_a_404_; lean_object* v___x_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v_a_404_ = lean_ctor_get(v___x_403_, 0);
lean_inc(v_a_404_);
lean_dec_ref_known(v___x_403_, 1);
v___x_405_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1));
v___x_406_ = lean_unsigned_to_nat(3u);
v___x_407_ = l_Lean_Expr_isAppOfArity(v_a_404_, v___x_405_, v___x_406_);
if (v___x_407_ == 0)
{
lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_408_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__3, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__3);
v___x_409_ = l_Lean_MessageData_ofExpr(v_a_404_);
v___x_410_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_408_);
lean_ctor_set(v___x_410_, 1, v___x_409_);
v___x_411_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg(v___x_410_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
return v___x_411_;
}
else
{
uint8_t v___x_412_; uint8_t v___x_413_; lean_object* v___x_414_; 
v___x_412_ = 0;
v___x_413_ = 0;
v___x_414_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_412_, v___x_413_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
if (lean_obj_tag(v___x_414_) == 0)
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v_a_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___f_422_; lean_object* v___x_423_; 
lean_dec_ref_known(v___x_414_, 1);
v___x_415_ = l_Lean_Expr_appArg_x21(v_a_404_);
v___x_416_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__2___redArg(v___x_415_, v_a_393_);
v_a_417_ = lean_ctor_get(v___x_416_, 0);
lean_inc(v_a_417_);
lean_dec_ref(v___x_416_);
v___x_418_ = l_Lean_Expr_appFn_x21(v_a_404_);
lean_dec(v_a_404_);
v___x_419_ = l_Lean_Expr_appArg_x21(v___x_418_);
lean_dec_ref(v___x_418_);
v___x_420_ = lean_box(v___x_413_);
v___x_421_ = lean_box(v___x_398_);
v___f_422_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___lam__0___boxed), 12, 3);
lean_closure_set(v___f_422_, 0, v___x_419_);
lean_closure_set(v___f_422_, 1, v___x_420_);
lean_closure_set(v___f_422_, 2, v___x_421_);
v___x_423_ = lp_mathlib_Lean_Meta_lambdaTelescope___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__3___redArg(v_a_417_, v___f_422_, v___x_413_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
return v___x_423_;
}
else
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
lean_dec(v_a_404_);
v_a_424_ = lean_ctor_get(v___x_414_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_414_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_414_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_429_; 
if (v_isShared_427_ == 0)
{
v___x_429_ = v___x_426_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_a_424_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
}
else
{
return v___x_403_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___boxed(lean_object* v_stx_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg(v_stx_432_, v_a_433_, v_a_434_, v_a_435_, v_a_436_, v_a_437_, v_a_438_);
lean_dec(v_a_438_);
lean_dec_ref(v_a_437_);
lean_dec(v_a_436_);
lean_dec_ref(v_a_435_);
lean_dec(v_a_434_);
lean_dec_ref(v_a_433_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1(lean_object* v_stx_441_, lean_object* v_x_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg(v_stx_441_, v_a_443_, v_a_444_, v_a_445_, v_a_446_, v_a_447_, v_a_448_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___boxed(lean_object* v_stx_451_, lean_object* v_x_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1(v_stx_451_, v_x_452_, v_a_453_, v_a_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_);
lean_dec(v_a_458_);
lean_dec_ref(v_a_457_);
lean_dec(v_a_456_);
lean_dec_ref(v_a_455_);
lean_dec(v_a_454_);
lean_dec_ref(v_a_453_);
lean_dec(v_x_452_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1(lean_object* v_00_u03b1_461_, lean_object* v_msg_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___redArg(v_msg_462_, v___y_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1___boxed(lean_object* v_00_u03b1_471_, lean_object* v_msg_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1(v_00_u03b1_471_, v_msg_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_475_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2(lean_object* v_msgData_481_, lean_object* v_macroStack_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg(v_msgData_481_, v_macroStack_482_, v___y_487_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___boxed(lean_object* v_msgData_491_, lean_object* v_macroStack_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2(v_msgData_491_, v_macroStack_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
lean_dec(v___y_494_);
lean_dec_ref(v___y_493_);
return v_res_500_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10(void){
_start:
{
lean_object* v___x_536_; lean_object* v___x_537_; 
v___x_536_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__9));
v___x_537_ = l_String_toRawSubstring_x27(v___x_536_);
return v___x_537_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27(void){
_start:
{
lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_576_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__26));
v___x_577_ = l_String_toRawSubstring_x27(v___x_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg(lean_object* v_stx_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; uint8_t v___x_610_; 
v___x_607_ = lean_box(0);
v___x_608_ = lean_unsigned_to_nat(0u);
v___x_609_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__1));
lean_inc(v_stx_599_);
v___x_610_ = l_Lean_Syntax_isOfKind(v_stx_599_, v___x_609_);
if (v___x_610_ == 0)
{
lean_object* v___x_611_; 
lean_dec(v_stx_599_);
v___x_611_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg();
return v___x_611_;
}
else
{
lean_object* v_ref_612_; lean_object* v_quotContext_613_; lean_object* v_currMacroScope_614_; lean_object* v___x_615_; lean_object* v___x_616_; uint8_t v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; 
v_ref_612_ = lean_ctor_get(v_a_604_, 5);
v_quotContext_613_ = lean_ctor_get(v_a_604_, 10);
v_currMacroScope_614_ = lean_ctor_get(v_a_604_, 11);
v___x_615_ = lean_unsigned_to_nat(1u);
v___x_616_ = l_Lean_Syntax_getArg(v_stx_599_, v___x_615_);
lean_dec(v_stx_599_);
v___x_617_ = 0;
v___x_618_ = l_Lean_SourceInfo_fromRef(v_ref_612_, v___x_617_);
v___x_619_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__3));
v___x_620_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5));
v___x_621_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__6));
lean_inc_n(v___x_618_, 12);
v___x_622_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_618_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__8));
v___x_624_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10);
lean_inc_n(v_currMacroScope_614_, 2);
lean_inc_n(v_quotContext_613_, 2);
v___x_625_ = l_Lean_addMacroScope(v_quotContext_613_, v___x_607_, v_currMacroScope_614_);
v___x_626_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__20));
v___x_627_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_627_, 0, v___x_618_);
lean_ctor_set(v___x_627_, 1, v___x_624_);
lean_ctor_set(v___x_627_, 2, v___x_625_);
lean_ctor_set(v___x_627_, 3, v___x_626_);
v___x_628_ = l_Lean_Syntax_node1(v___x_618_, v___x_623_, v___x_627_);
v___x_629_ = l_Lean_Syntax_node2(v___x_618_, v___x_620_, v___x_622_, v___x_628_);
v___x_630_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__21));
v___x_631_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_631_, 0, v___x_618_);
lean_ctor_set(v___x_631_, 1, v___x_630_);
v___x_632_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__23));
v___x_633_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25));
v___x_634_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27);
v___x_635_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28));
v___x_636_ = l_Lean_addMacroScope(v_quotContext_613_, v___x_635_, v_currMacroScope_614_);
v___x_637_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__32));
v___x_638_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_638_, 0, v___x_618_);
lean_ctor_set(v___x_638_, 1, v___x_634_);
lean_ctor_set(v___x_638_, 2, v___x_636_);
lean_ctor_set(v___x_638_, 3, v___x_637_);
v___x_639_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34));
v___x_640_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__35));
v___x_641_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_641_, 0, v___x_618_);
lean_ctor_set(v___x_641_, 1, v___x_640_);
v___x_642_ = l_Lean_Syntax_node1(v___x_618_, v___x_639_, v___x_641_);
v___x_643_ = l_Lean_Syntax_node1(v___x_618_, v___x_632_, v___x_642_);
v___x_644_ = l_Lean_Syntax_node2(v___x_618_, v___x_633_, v___x_638_, v___x_643_);
v___x_645_ = l_Lean_Syntax_node1(v___x_618_, v___x_632_, v___x_644_);
v___x_646_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__36));
v___x_647_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_647_, 0, v___x_618_);
lean_ctor_set(v___x_647_, 1, v___x_646_);
v___x_648_ = l_Lean_Syntax_node5(v___x_618_, v___x_619_, v___x_629_, v___x_616_, v___x_631_, v___x_645_, v___x_647_);
v___x_649_ = lean_box(0);
v___x_650_ = l_Lean_Elab_Term_elabTerm(v___x_648_, v___x_649_, v___x_610_, v___x_610_, v_a_600_, v_a_601_, v_a_602_, v_a_603_, v_a_604_, v_a_605_);
if (lean_obj_tag(v___x_650_) == 0)
{
lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_659_; 
v_a_651_ = lean_ctor_get(v___x_650_, 0);
v_isSharedCheck_659_ = !lean_is_exclusive(v___x_650_);
if (v_isSharedCheck_659_ == 0)
{
v___x_653_ = v___x_650_;
v_isShared_654_ = v_isSharedCheck_659_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_650_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_659_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
lean_object* v___x_655_; lean_object* v___x_657_; 
v___x_655_ = l_Lean_mkProj(v___x_635_, v___x_608_, v_a_651_);
if (v_isShared_654_ == 0)
{
lean_ctor_set(v___x_653_, 0, v___x_655_);
v___x_657_ = v___x_653_;
goto v_reusejp_656_;
}
else
{
lean_object* v_reuseFailAlloc_658_; 
v_reuseFailAlloc_658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_658_, 0, v___x_655_);
v___x_657_ = v_reuseFailAlloc_658_;
goto v_reusejp_656_;
}
v_reusejp_656_:
{
return v___x_657_;
}
}
}
else
{
return v___x_650_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___boxed(lean_object* v_stx_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_){
_start:
{
lean_object* v_res_668_; 
v_res_668_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg(v_stx_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_);
lean_dec(v_a_666_);
lean_dec_ref(v_a_665_);
lean_dec(v_a_664_);
lean_dec_ref(v_a_663_);
lean_dec(v_a_662_);
lean_dec_ref(v_a_661_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1(lean_object* v_stx_669_, lean_object* v_x_670_, lean_object* v_a_671_, lean_object* v_a_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_){
_start:
{
lean_object* v___x_678_; 
v___x_678_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg(v_stx_669_, v_a_671_, v_a_672_, v_a_673_, v_a_674_, v_a_675_, v_a_676_);
return v___x_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___boxed(lean_object* v_stx_679_, lean_object* v_x_680_, lean_object* v_a_681_, lean_object* v_a_682_, lean_object* v_a_683_, lean_object* v_a_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1(v_stx_679_, v_x_680_, v_a_681_, v_a_682_, v_a_683_, v_a_684_, v_a_685_, v_a_686_);
lean_dec(v_a_686_);
lean_dec_ref(v_a_685_);
lean_dec(v_a_684_);
lean_dec_ref(v_a_683_);
lean_dec(v_a_682_);
lean_dec_ref(v_a_681_);
lean_dec(v_x_680_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg(){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_727_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__0___redArg___closed__0);
v___x_728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_728_, 0, v___x_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg___boxed(lean_object* v___y_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0(lean_object* v_00_u03b1_731_, lean_object* v___y_732_, lean_object* v___y_733_){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___boxed(lean_object* v_00_u03b1_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_){
_start:
{
lean_object* v_res_740_; 
v_res_740_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0(v_00_u03b1_736_, v___y_737_, v___y_738_);
lean_dec(v___y_738_);
lean_dec_ref(v___y_737_);
return v_res_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__1(size_t v_sz_741_, size_t v_i_742_, lean_object* v_bs_743_){
_start:
{
uint8_t v___x_744_; 
v___x_744_ = lean_usize_dec_lt(v_i_742_, v_sz_741_);
if (v___x_744_ == 0)
{
lean_object* v___x_745_; 
v___x_745_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_745_, 0, v_bs_743_);
return v___x_745_;
}
else
{
lean_object* v_v_746_; lean_object* v___x_747_; lean_object* v_bs_x27_748_; size_t v___x_749_; size_t v___x_750_; lean_object* v___x_751_; 
v_v_746_ = lean_array_uget(v_bs_743_, v_i_742_);
v___x_747_ = lean_unsigned_to_nat(0u);
v_bs_x27_748_ = lean_array_uset(v_bs_743_, v_i_742_, v___x_747_);
v___x_749_ = ((size_t)1ULL);
v___x_750_ = lean_usize_add(v_i_742_, v___x_749_);
v___x_751_ = lean_array_uset(v_bs_x27_748_, v_i_742_, v_v_746_);
v_i_742_ = v___x_750_;
v_bs_743_ = v___x_751_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__1___boxed(lean_object* v_sz_753_, lean_object* v_i_754_, lean_object* v_bs_755_){
_start:
{
size_t v_sz_boxed_756_; size_t v_i_boxed_757_; lean_object* v_res_758_; 
v_sz_boxed_756_ = lean_unbox_usize(v_sz_753_);
lean_dec(v_sz_753_);
v_i_boxed_757_ = lean_unbox_usize(v_i_754_);
lean_dec(v_i_754_);
v_res_758_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__1(v_sz_boxed_756_, v_i_boxed_757_, v_bs_755_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__2(lean_object* v_as_759_, size_t v_sz_760_, size_t v_i_761_, lean_object* v_b_762_, lean_object* v___y_763_, lean_object* v___y_764_){
_start:
{
uint8_t v___x_766_; 
v___x_766_ = lean_usize_dec_lt(v_i_761_, v_sz_760_);
if (v___x_766_ == 0)
{
lean_object* v___x_767_; 
v___x_767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_767_, 0, v_b_762_);
return v___x_767_;
}
else
{
lean_object* v_a_768_; lean_object* v___x_769_; 
v_a_768_ = lean_array_uget_borrowed(v_as_759_, v_i_761_);
lean_inc(v_a_768_);
v___x_769_ = l_Lean_Elab_Command_elabCommand(v_a_768_, v___y_763_, v___y_764_);
if (lean_obj_tag(v___x_769_) == 0)
{
lean_object* v___x_771_; uint8_t v_isShared_772_; uint8_t v_isSharedCheck_783_; 
v_isSharedCheck_783_ = !lean_is_exclusive(v___x_769_);
if (v_isSharedCheck_783_ == 0)
{
lean_object* v_unused_784_; 
v_unused_784_ = lean_ctor_get(v___x_769_, 0);
lean_dec(v_unused_784_);
v___x_771_ = v___x_769_;
v_isShared_772_ = v_isSharedCheck_783_;
goto v_resetjp_770_;
}
else
{
lean_dec(v___x_769_);
v___x_771_ = lean_box(0);
v_isShared_772_ = v_isSharedCheck_783_;
goto v_resetjp_770_;
}
v_resetjp_770_:
{
lean_object* v___x_773_; lean_object* v_messages_774_; lean_object* v___x_775_; uint8_t v___x_776_; 
v___x_773_ = lean_st_ref_get(v___y_764_);
v_messages_774_ = lean_ctor_get(v___x_773_, 1);
lean_inc_ref(v_messages_774_);
lean_dec(v___x_773_);
v___x_775_ = lean_box(0);
v___x_776_ = l_Lean_MessageLog_hasErrors(v_messages_774_);
lean_dec_ref(v_messages_774_);
if (v___x_776_ == 0)
{
size_t v___x_777_; size_t v___x_778_; 
lean_del_object(v___x_771_);
v___x_777_ = ((size_t)1ULL);
v___x_778_ = lean_usize_add(v_i_761_, v___x_777_);
v_i_761_ = v___x_778_;
v_b_762_ = v___x_775_;
goto _start;
}
else
{
lean_object* v___x_781_; 
if (v_isShared_772_ == 0)
{
lean_ctor_set(v___x_771_, 0, v___x_775_);
v___x_781_ = v___x_771_;
goto v_reusejp_780_;
}
else
{
lean_object* v_reuseFailAlloc_782_; 
v_reuseFailAlloc_782_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_782_, 0, v___x_775_);
v___x_781_ = v_reuseFailAlloc_782_;
goto v_reusejp_780_;
}
v_reusejp_780_:
{
return v___x_781_;
}
}
}
}
else
{
return v___x_769_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__2___boxed(lean_object* v_as_785_, lean_object* v_sz_786_, lean_object* v_i_787_, lean_object* v_b_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_){
_start:
{
size_t v_sz_boxed_792_; size_t v_i_boxed_793_; lean_object* v_res_794_; 
v_sz_boxed_792_ = lean_unbox_usize(v_sz_786_);
lean_dec(v_sz_786_);
v_i_boxed_793_ = lean_unbox_usize(v_i_787_);
lean_dec(v_i_787_);
v_res_794_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__2(v_as_785_, v_sz_boxed_792_, v_i_boxed_793_, v_b_788_, v___y_789_, v___y_790_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
lean_dec_ref(v_as_785_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1(lean_object* v_x_795_, lean_object* v_a_796_, lean_object* v_a_797_){
_start:
{
lean_object* v___x_799_; uint8_t v___x_800_; 
v___x_799_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__1));
lean_inc(v_x_795_);
v___x_800_ = l_Lean_Syntax_isOfKind(v_x_795_, v___x_799_);
if (v___x_800_ == 0)
{
lean_object* v___x_801_; 
lean_dec(v_x_795_);
v___x_801_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
return v___x_801_;
}
else
{
lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; size_t v_sz_805_; size_t v___x_806_; lean_object* v___x_807_; 
v___x_802_ = lean_unsigned_to_nat(1u);
v___x_803_ = l_Lean_Syntax_getArg(v_x_795_, v___x_802_);
lean_dec(v_x_795_);
v___x_804_ = l_Lean_Syntax_getArgs(v___x_803_);
lean_dec(v___x_803_);
v_sz_805_ = lean_array_size(v___x_804_);
v___x_806_ = ((size_t)0ULL);
v___x_807_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__1(v_sz_805_, v___x_806_, v___x_804_);
if (lean_obj_tag(v___x_807_) == 0)
{
lean_object* v___x_808_; 
v___x_808_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
return v___x_808_;
}
else
{
lean_object* v_val_809_; lean_object* v___x_810_; size_t v_sz_811_; lean_object* v___x_812_; 
v_val_809_ = lean_ctor_get(v___x_807_, 0);
lean_inc(v_val_809_);
lean_dec_ref_known(v___x_807_, 1);
v___x_810_ = lean_box(0);
v_sz_811_ = lean_array_size(v_val_809_);
v___x_812_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__2(v_val_809_, v_sz_811_, v___x_806_, v___x_810_, v_a_796_, v_a_797_);
lean_dec(v_val_809_);
if (lean_obj_tag(v___x_812_) == 0)
{
lean_object* v___x_814_; uint8_t v_isShared_815_; uint8_t v_isSharedCheck_819_; 
v_isSharedCheck_819_ = !lean_is_exclusive(v___x_812_);
if (v_isSharedCheck_819_ == 0)
{
lean_object* v_unused_820_; 
v_unused_820_ = lean_ctor_get(v___x_812_, 0);
lean_dec(v_unused_820_);
v___x_814_ = v___x_812_;
v_isShared_815_ = v_isSharedCheck_819_;
goto v_resetjp_813_;
}
else
{
lean_dec(v___x_812_);
v___x_814_ = lean_box(0);
v_isShared_815_ = v_isSharedCheck_819_;
goto v_resetjp_813_;
}
v_resetjp_813_:
{
lean_object* v___x_817_; 
if (v_isShared_815_ == 0)
{
lean_ctor_set(v___x_814_, 0, v___x_810_);
v___x_817_ = v___x_814_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v___x_810_);
v___x_817_ = v_reuseFailAlloc_818_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
return v___x_817_;
}
}
}
else
{
return v___x_812_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1___boxed(lean_object* v_x_821_, lean_object* v_a_822_, lean_object* v_a_823_, lean_object* v_a_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1(v_x_821_, v_a_822_, v_a_823_);
lean_dec(v_a_823_);
lean_dec_ref(v_a_822_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg(lean_object* v___y_942_){
_start:
{
lean_object* v___x_944_; lean_object* v_env_945_; lean_object* v___x_946_; lean_object* v_mainModule_947_; lean_object* v___x_948_; 
v___x_944_ = lean_st_ref_get(v___y_942_);
v_env_945_ = lean_ctor_get(v___x_944_, 0);
lean_inc_ref(v_env_945_);
lean_dec(v___x_944_);
v___x_946_ = l_Lean_Environment_header(v_env_945_);
lean_dec_ref(v_env_945_);
v_mainModule_947_ = lean_ctor_get(v___x_946_, 0);
lean_inc(v_mainModule_947_);
lean_dec_ref(v___x_946_);
v___x_948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_948_, 0, v_mainModule_947_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg___boxed(lean_object* v___y_949_, lean_object* v___y_950_){
_start:
{
lean_object* v_res_951_; 
v_res_951_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg(v___y_949_);
lean_dec(v___y_949_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1(lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg(v___y_953_);
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___boxed(lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_){
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1(v___y_956_, v___y_957_);
lean_dec(v___y_957_);
lean_dec_ref(v___y_956_);
return v_res_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__0(size_t v_sz_960_, size_t v_i_961_, lean_object* v_bs_962_){
_start:
{
uint8_t v___x_963_; 
v___x_963_ = lean_usize_dec_lt(v_i_961_, v_sz_960_);
if (v___x_963_ == 0)
{
return v_bs_962_;
}
else
{
lean_object* v_v_964_; lean_object* v___x_965_; lean_object* v_bs_x27_966_; size_t v___x_967_; size_t v___x_968_; lean_object* v___x_969_; 
v_v_964_ = lean_array_uget(v_bs_962_, v_i_961_);
v___x_965_ = lean_unsigned_to_nat(0u);
v_bs_x27_966_ = lean_array_uset(v_bs_962_, v_i_961_, v___x_965_);
v___x_967_ = ((size_t)1ULL);
v___x_968_ = lean_usize_add(v_i_961_, v___x_967_);
v___x_969_ = lean_array_uset(v_bs_x27_966_, v_i_961_, v_v_964_);
v_i_961_ = v___x_968_;
v_bs_962_ = v___x_969_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__0___boxed(lean_object* v_sz_971_, lean_object* v_i_972_, lean_object* v_bs_973_){
_start:
{
size_t v_sz_boxed_974_; size_t v_i_boxed_975_; lean_object* v_res_976_; 
v_sz_boxed_974_ = lean_unbox_usize(v_sz_971_);
lean_dec(v_sz_971_);
v_i_boxed_975_ = lean_unbox_usize(v_i_972_);
lean_dec(v_i_972_);
v_res_976_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__0(v_sz_boxed_974_, v_i_boxed_975_, v_bs_973_);
return v_res_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg(lean_object* v_msgData_977_, lean_object* v_macroStack_978_, lean_object* v___y_979_){
_start:
{
lean_object* v___x_981_; lean_object* v_scopes_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v_opts_985_; lean_object* v___x_986_; uint8_t v___x_987_; 
v___x_981_ = lean_st_ref_get(v___y_979_);
v_scopes_982_ = lean_ctor_get(v___x_981_, 2);
lean_inc(v_scopes_982_);
lean_dec(v___x_981_);
v___x_983_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_984_ = l_List_head_x21___redArg(v___x_983_, v_scopes_982_);
lean_dec(v_scopes_982_);
v_opts_985_ = lean_ctor_get(v___x_984_, 1);
lean_inc_ref(v_opts_985_);
lean_dec(v___x_984_);
v___x_986_ = l_Lean_Elab_pp_macroStack;
v___x_987_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__5(v_opts_985_, v___x_986_);
lean_dec_ref(v_opts_985_);
if (v___x_987_ == 0)
{
lean_object* v___x_988_; 
lean_dec(v_macroStack_978_);
v___x_988_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_988_, 0, v_msgData_977_);
return v___x_988_;
}
else
{
if (lean_obj_tag(v_macroStack_978_) == 0)
{
lean_object* v___x_989_; 
v___x_989_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_989_, 0, v_msgData_977_);
return v___x_989_;
}
else
{
lean_object* v_head_990_; lean_object* v_after_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_1006_; 
v_head_990_ = lean_ctor_get(v_macroStack_978_, 0);
lean_inc(v_head_990_);
v_after_991_ = lean_ctor_get(v_head_990_, 1);
v_isSharedCheck_1006_ = !lean_is_exclusive(v_head_990_);
if (v_isSharedCheck_1006_ == 0)
{
lean_object* v_unused_1007_; 
v_unused_1007_ = lean_ctor_get(v_head_990_, 0);
lean_dec(v_unused_1007_);
v___x_993_ = v_head_990_;
v_isShared_994_ = v_isSharedCheck_1006_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_after_991_);
lean_dec(v_head_990_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_1006_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_995_; lean_object* v___x_997_; 
v___x_995_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6___closed__0);
if (v_isShared_994_ == 0)
{
lean_ctor_set_tag(v___x_993_, 7);
lean_ctor_set(v___x_993_, 1, v___x_995_);
lean_ctor_set(v___x_993_, 0, v_msgData_977_);
v___x_997_ = v___x_993_;
goto v_reusejp_996_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v_msgData_977_);
lean_ctor_set(v_reuseFailAlloc_1005_, 1, v___x_995_);
v___x_997_ = v_reuseFailAlloc_1005_;
goto v_reusejp_996_;
}
v_reusejp_996_:
{
lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v_msgData_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; 
v___x_998_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2___redArg___closed__2);
v___x_999_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_999_, 0, v___x_997_);
lean_ctor_set(v___x_999_, 1, v___x_998_);
v___x_1000_ = l_Lean_MessageData_ofSyntax(v_after_991_);
v___x_1001_ = l_Lean_indentD(v___x_1000_);
v_msgData_1002_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1002_, 0, v___x_999_);
lean_ctor_set(v_msgData_1002_, 1, v___x_1001_);
v___x_1003_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1_spec__1_spec__2_spec__6(v_msgData_1002_, v_macroStack_978_);
v___x_1004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1004_, 0, v___x_1003_);
return v___x_1004_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_1008_, lean_object* v_macroStack_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_){
_start:
{
lean_object* v_res_1012_; 
v_res_1012_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg(v_msgData_1008_, v_macroStack_1009_, v___y_1010_);
lean_dec(v___y_1010_);
return v_res_1012_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_1013_; 
v___x_1013_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1013_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; 
v___x_1014_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__0);
v___x_1015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1014_);
return v___x_1015_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; 
v___x_1016_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1);
v___x_1017_ = lean_unsigned_to_nat(0u);
v___x_1018_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1018_, 0, v___x_1017_);
lean_ctor_set(v___x_1018_, 1, v___x_1017_);
lean_ctor_set(v___x_1018_, 2, v___x_1017_);
lean_ctor_set(v___x_1018_, 3, v___x_1017_);
lean_ctor_set(v___x_1018_, 4, v___x_1016_);
lean_ctor_set(v___x_1018_, 5, v___x_1016_);
lean_ctor_set(v___x_1018_, 6, v___x_1016_);
lean_ctor_set(v___x_1018_, 7, v___x_1016_);
lean_ctor_set(v___x_1018_, 8, v___x_1016_);
lean_ctor_set(v___x_1018_, 9, v___x_1016_);
return v___x_1018_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; 
v___x_1019_ = lean_unsigned_to_nat(32u);
v___x_1020_ = lean_mk_empty_array_with_capacity(v___x_1019_);
v___x_1021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1021_, 0, v___x_1020_);
return v___x_1021_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__4(void){
_start:
{
size_t v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; 
v___x_1022_ = ((size_t)5ULL);
v___x_1023_ = lean_unsigned_to_nat(0u);
v___x_1024_ = lean_unsigned_to_nat(32u);
v___x_1025_ = lean_mk_empty_array_with_capacity(v___x_1024_);
v___x_1026_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__3);
v___x_1027_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1027_, 0, v___x_1026_);
lean_ctor_set(v___x_1027_, 1, v___x_1025_);
lean_ctor_set(v___x_1027_, 2, v___x_1023_);
lean_ctor_set(v___x_1027_, 3, v___x_1023_);
lean_ctor_set_usize(v___x_1027_, 4, v___x_1022_);
return v___x_1027_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_1028_ = lean_box(1);
v___x_1029_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__4);
v___x_1030_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__1);
v___x_1031_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1031_, 0, v___x_1030_);
lean_ctor_set(v___x_1031_, 1, v___x_1029_);
lean_ctor_set(v___x_1031_, 2, v___x_1028_);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg(lean_object* v_msgData_1032_, lean_object* v___y_1033_){
_start:
{
lean_object* v___x_1035_; lean_object* v_env_1036_; lean_object* v___x_1037_; lean_object* v_scopes_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v_opts_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; 
v___x_1035_ = lean_st_ref_get(v___y_1033_);
v_env_1036_ = lean_ctor_get(v___x_1035_, 0);
lean_inc_ref(v_env_1036_);
lean_dec(v___x_1035_);
v___x_1037_ = lean_st_ref_get(v___y_1033_);
v_scopes_1038_ = lean_ctor_get(v___x_1037_, 2);
lean_inc(v_scopes_1038_);
lean_dec(v___x_1037_);
v___x_1039_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1040_ = l_List_head_x21___redArg(v___x_1039_, v_scopes_1038_);
lean_dec(v_scopes_1038_);
v_opts_1041_ = lean_ctor_get(v___x_1040_, 1);
lean_inc_ref(v_opts_1041_);
lean_dec(v___x_1040_);
v___x_1042_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__2);
v___x_1043_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___closed__5);
v___x_1044_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1044_, 0, v_env_1036_);
lean_ctor_set(v___x_1044_, 1, v___x_1042_);
lean_ctor_set(v___x_1044_, 2, v___x_1043_);
lean_ctor_set(v___x_1044_, 3, v_opts_1041_);
v___x_1045_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1045_, 0, v___x_1044_);
lean_ctor_set(v___x_1045_, 1, v_msgData_1032_);
v___x_1046_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1045_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg___boxed(lean_object* v_msgData_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_){
_start:
{
lean_object* v_res_1050_; 
v_res_1050_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg(v_msgData_1047_, v___y_1048_);
lean_dec(v___y_1048_);
return v_res_1050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(lean_object* v_msg_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_){
_start:
{
lean_object* v___x_1055_; 
v___x_1055_ = l_Lean_Elab_Command_getRef___redArg(v___y_1052_);
if (lean_obj_tag(v___x_1055_) == 0)
{
lean_object* v_a_1056_; lean_object* v_macroStack_1057_; lean_object* v___x_1058_; lean_object* v_a_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v_a_1062_; lean_object* v___x_1064_; uint8_t v_isShared_1065_; uint8_t v_isSharedCheck_1070_; 
v_a_1056_ = lean_ctor_get(v___x_1055_, 0);
lean_inc(v_a_1056_);
lean_dec_ref_known(v___x_1055_, 1);
v_macroStack_1057_ = lean_ctor_get(v___y_1052_, 4);
v___x_1058_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg(v_msg_1051_, v___y_1053_);
v_a_1059_ = lean_ctor_get(v___x_1058_, 0);
lean_inc(v_a_1059_);
lean_dec_ref(v___x_1058_);
v___x_1060_ = l_Lean_Elab_getBetterRef(v_a_1056_, v_macroStack_1057_);
lean_dec(v_a_1056_);
lean_inc(v_macroStack_1057_);
v___x_1061_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg(v_a_1059_, v_macroStack_1057_, v___y_1053_);
v_a_1062_ = lean_ctor_get(v___x_1061_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1061_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1064_ = v___x_1061_;
v_isShared_1065_ = v_isSharedCheck_1070_;
goto v_resetjp_1063_;
}
else
{
lean_inc(v_a_1062_);
lean_dec(v___x_1061_);
v___x_1064_ = lean_box(0);
v_isShared_1065_ = v_isSharedCheck_1070_;
goto v_resetjp_1063_;
}
v_resetjp_1063_:
{
lean_object* v___x_1066_; lean_object* v___x_1068_; 
v___x_1066_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1060_);
lean_ctor_set(v___x_1066_, 1, v_a_1062_);
if (v_isShared_1065_ == 0)
{
lean_ctor_set_tag(v___x_1064_, 1);
lean_ctor_set(v___x_1064_, 0, v___x_1066_);
v___x_1068_ = v___x_1064_;
goto v_reusejp_1067_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v___x_1066_);
v___x_1068_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1067_;
}
v_reusejp_1067_:
{
return v___x_1068_;
}
}
}
else
{
lean_object* v_a_1071_; lean_object* v___x_1073_; uint8_t v_isShared_1074_; uint8_t v_isSharedCheck_1078_; 
lean_dec_ref(v_msg_1051_);
v_a_1071_ = lean_ctor_get(v___x_1055_, 0);
v_isSharedCheck_1078_ = !lean_is_exclusive(v___x_1055_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1073_ = v___x_1055_;
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
else
{
lean_inc(v_a_1071_);
lean_dec(v___x_1055_);
v___x_1073_ = lean_box(0);
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
v_resetjp_1072_:
{
lean_object* v___x_1076_; 
if (v_isShared_1074_ == 0)
{
v___x_1076_ = v___x_1073_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_a_1071_);
v___x_1076_ = v_reuseFailAlloc_1077_;
goto v_reusejp_1075_;
}
v_reusejp_1075_:
{
return v___x_1076_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg___boxed(lean_object* v_msg_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v_msg_1079_, v___y_1080_, v___y_1081_);
lean_dec(v___y_1081_);
lean_dec_ref(v___y_1080_);
return v_res_1083_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__27(void){
_start:
{
lean_object* v___x_1116_; lean_object* v___x_1117_; 
v___x_1116_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__26));
v___x_1117_ = l_String_toRawSubstring_x27(v___x_1116_);
return v___x_1117_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__35(void){
_start:
{
lean_object* v___x_1126_; lean_object* v___x_1127_; 
v___x_1126_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__34));
v___x_1127_ = l_String_toRawSubstring_x27(v___x_1126_);
return v___x_1127_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__47(void){
_start:
{
lean_object* v___x_1148_; lean_object* v___x_1149_; 
v___x_1148_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__46));
v___x_1149_ = l_String_toRawSubstring_x27(v___x_1148_);
return v___x_1149_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__57(void){
_start:
{
lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1170_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__56));
v___x_1171_ = l_String_toRawSubstring_x27(v___x_1170_);
return v___x_1171_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__66(void){
_start:
{
lean_object* v___x_1195_; lean_object* v___x_1196_; 
v___x_1195_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__0));
v___x_1196_ = l_String_toRawSubstring_x27(v___x_1195_);
return v___x_1196_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__84(void){
_start:
{
lean_object* v___x_1233_; lean_object* v___x_1234_; 
v___x_1233_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__83));
v___x_1234_ = l_String_toRawSubstring_x27(v___x_1233_);
return v___x_1234_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__94(void){
_start:
{
lean_object* v___x_1255_; lean_object* v___x_1256_; 
v___x_1255_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__91));
v___x_1256_ = l_String_toRawSubstring_x27(v___x_1255_);
return v___x_1256_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__98(void){
_start:
{
lean_object* v___x_1265_; 
v___x_1265_ = l_Array_mkArray0(lean_box(0));
return v___x_1265_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100(void){
_start:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; 
v___x_1267_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__99));
v___x_1268_ = l_Lean_stringToMessageData(v___x_1267_);
return v___x_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1(lean_object* v_x_1321_, lean_object* v_a_1322_, lean_object* v_a_1323_){
_start:
{
lean_object* v___x_1328_; lean_object* v___x_1329_; uint8_t v___x_1330_; 
v___x_1328_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__9));
v___x_1329_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command_command__Irreducible__def_________00__closed__1));
lean_inc(v_x_1321_);
v___x_1330_ = l_Lean_Syntax_isOfKind(v_x_1321_, v___x_1329_);
if (v___x_1330_ == 0)
{
lean_object* v___x_1331_; 
lean_dec(v_x_1321_);
v___x_1331_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
return v___x_1331_;
}
else
{
lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1347_; lean_object* v___y_1348_; lean_object* v___y_1349_; lean_object* v___y_1350_; lean_object* v___y_1351_; lean_object* v___y_1352_; lean_object* v___y_1353_; lean_object* v___y_1354_; lean_object* v___y_1355_; lean_object* v___y_1356_; lean_object* v___y_1357_; lean_object* v___y_1358_; lean_object* v___y_1359_; lean_object* v___y_1360_; lean_object* v___y_1361_; lean_object* v___y_1362_; lean_object* v___y_1363_; lean_object* v___y_1364_; lean_object* v___y_1365_; lean_object* v___y_1366_; lean_object* v___y_1367_; lean_object* v___y_1368_; lean_object* v___y_1369_; lean_object* v___y_1370_; lean_object* v___y_1371_; lean_object* v___y_1372_; lean_object* v___y_1373_; lean_object* v___y_1374_; lean_object* v___y_1375_; lean_object* v___y_1376_; lean_object* v___y_1377_; lean_object* v___y_1378_; lean_object* v___y_1379_; lean_object* v___y_1380_; lean_object* v___y_1381_; lean_object* v___y_1382_; lean_object* v___y_1383_; lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v___y_1386_; lean_object* v___y_1387_; lean_object* v___y_1388_; lean_object* v___y_1389_; lean_object* v___y_1390_; lean_object* v___y_1391_; lean_object* v___y_1392_; lean_object* v___y_1393_; lean_object* v___y_1394_; lean_object* v___y_1395_; uint8_t v___y_1396_; lean_object* v___y_1397_; lean_object* v___y_1398_; lean_object* v___y_1399_; lean_object* v___y_1400_; lean_object* v___y_1401_; lean_object* v___y_1402_; lean_object* v___y_1403_; lean_object* v___y_1404_; lean_object* v___y_1405_; lean_object* v___y_1608_; lean_object* v___y_1609_; lean_object* v___y_1610_; lean_object* v___y_1611_; lean_object* v___y_1612_; lean_object* v___y_1613_; lean_object* v___y_1614_; lean_object* v___y_1615_; lean_object* v___y_1616_; lean_object* v___y_1617_; lean_object* v___y_1618_; lean_object* v___y_1619_; lean_object* v___y_1620_; lean_object* v___y_1621_; lean_object* v___y_1622_; lean_object* v___y_1623_; lean_object* v___y_1624_; lean_object* v___y_1625_; lean_object* v___y_1626_; lean_object* v___y_1627_; lean_object* v___y_1628_; lean_object* v___y_1629_; lean_object* v___y_1630_; lean_object* v___y_1631_; lean_object* v___y_1632_; lean_object* v___y_1633_; lean_object* v___y_1634_; lean_object* v___y_1635_; lean_object* v___y_1636_; lean_object* v___y_1637_; lean_object* v___y_1638_; lean_object* v___y_1639_; lean_object* v___y_1640_; lean_object* v___y_1641_; lean_object* v___y_1642_; lean_object* v___y_1643_; lean_object* v___y_1644_; lean_object* v___y_1645_; lean_object* v___y_1646_; lean_object* v___y_1647_; lean_object* v___y_1648_; lean_object* v___y_1649_; lean_object* v___y_1650_; lean_object* v___y_1651_; lean_object* v___y_1652_; lean_object* v___y_1653_; lean_object* v___y_1654_; lean_object* v___y_1655_; lean_object* v___y_1656_; lean_object* v___y_1657_; lean_object* v___y_1658_; uint8_t v___y_1659_; lean_object* v___y_1660_; lean_object* v___y_1661_; lean_object* v___y_1662_; lean_object* v___y_1663_; lean_object* v___y_1664_; lean_object* v___y_1665_; lean_object* v___y_1666_; lean_object* v___y_1667_; lean_object* v___y_1668_; lean_object* v___x_1679_; lean_object* v___y_1681_; lean_object* v___y_1682_; lean_object* v___y_1683_; lean_object* v___y_1684_; lean_object* v___y_1685_; lean_object* v___y_1686_; lean_object* v___y_1687_; lean_object* v___y_1688_; lean_object* v___y_1689_; lean_object* v___y_1690_; lean_object* v___y_1691_; lean_object* v___y_1692_; lean_object* v___y_1693_; lean_object* v___y_1694_; lean_object* v___y_1695_; lean_object* v___y_1696_; lean_object* v___y_1697_; lean_object* v___y_1698_; lean_object* v___y_1699_; lean_object* v___y_1700_; lean_object* v___y_1701_; lean_object* v___y_1702_; lean_object* v___y_1703_; lean_object* v___y_1704_; lean_object* v___y_1705_; uint8_t v___y_1706_; lean_object* v___y_1707_; lean_object* v___y_1708_; lean_object* v___y_1709_; lean_object* v___y_1710_; lean_object* v___y_1711_; lean_object* v___y_1712_; lean_object* v___y_1713_; lean_object* v___y_1826_; lean_object* v___y_1827_; lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1830_; lean_object* v___y_1831_; lean_object* v___y_1832_; lean_object* v___y_1833_; lean_object* v___y_1834_; lean_object* v___y_1835_; lean_object* v___y_1836_; lean_object* v___y_1837_; lean_object* v___y_1838_; lean_object* v___y_1839_; lean_object* v___y_1840_; lean_object* v___y_1841_; lean_object* v___y_1842_; lean_object* v___y_1843_; lean_object* v___y_1844_; lean_object* v___y_1845_; uint8_t v___y_1846_; lean_object* v___y_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v___y_1850_; lean_object* v___y_1851_; lean_object* v___y_1852_; lean_object* v___y_1853_; lean_object* v___y_1875_; lean_object* v___y_1876_; lean_object* v___y_1877_; lean_object* v___y_1878_; lean_object* v___y_1879_; lean_object* v___y_1880_; lean_object* v___y_1881_; lean_object* v___y_1882_; lean_object* v___y_1883_; lean_object* v___y_1884_; lean_object* v___y_1885_; lean_object* v___y_1886_; lean_object* v___y_1887_; lean_object* v___y_1888_; lean_object* v___y_1889_; lean_object* v___y_1890_; lean_object* v___y_1891_; lean_object* v___y_1892_; lean_object* v___y_1893_; lean_object* v___y_1894_; uint8_t v___y_1895_; lean_object* v___y_1896_; lean_object* v___y_1897_; lean_object* v___y_1898_; lean_object* v___y_1899_; lean_object* v___y_1900_; lean_object* v___y_1901_; lean_object* v___y_1902_; lean_object* v___y_1909_; lean_object* v___y_1910_; lean_object* v___y_1911_; lean_object* v___y_1912_; lean_object* v___y_1913_; lean_object* v___y_1914_; lean_object* v___y_1915_; lean_object* v___y_1916_; uint8_t v___y_1917_; lean_object* v___y_1918_; lean_object* v___y_1919_; lean_object* v___y_1920_; lean_object* v___y_1921_; lean_object* v___y_1922_; lean_object* v___y_1923_; lean_object* v___y_1924_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v_a_1927_; lean_object* v___y_1942_; lean_object* v___y_1943_; lean_object* v___y_1944_; lean_object* v___y_1945_; lean_object* v___y_1946_; lean_object* v___y_1947_; lean_object* v___y_1948_; uint8_t v___y_1949_; lean_object* v___y_1950_; lean_object* v___y_1951_; lean_object* v___y_1952_; lean_object* v___y_1953_; lean_object* v___y_1954_; lean_object* v___y_1955_; lean_object* v___y_1956_; lean_object* v___y_1957_; lean_object* v___y_1985_; lean_object* v___y_1986_; lean_object* v___y_1987_; lean_object* v___y_1988_; lean_object* v___y_1989_; lean_object* v___y_1990_; lean_object* v___y_1991_; uint8_t v___y_1992_; lean_object* v___y_1993_; lean_object* v___y_1994_; lean_object* v___y_1995_; lean_object* v___y_1996_; lean_object* v___y_1997_; lean_object* v___y_1998_; lean_object* v___y_1999_; lean_object* v___y_2000_; lean_object* v___y_2014_; lean_object* v___y_2015_; lean_object* v___y_2016_; lean_object* v___y_2017_; lean_object* v___y_2018_; lean_object* v___y_2019_; lean_object* v___y_2020_; lean_object* v___y_2021_; lean_object* v___y_2022_; lean_object* v___y_2023_; lean_object* v___y_2024_; lean_object* v___y_2025_; lean_object* v_uns_2026_; lean_object* v___y_2027_; lean_object* v___y_2028_; lean_object* v___y_2041_; lean_object* v___y_2042_; lean_object* v___y_2043_; lean_object* v___y_2044_; lean_object* v___y_2045_; lean_object* v___y_2046_; lean_object* v___y_2047_; lean_object* v___y_2048_; lean_object* v___y_2049_; lean_object* v___y_2050_; lean_object* v___y_2051_; lean_object* v_nc_2052_; lean_object* v___y_2053_; lean_object* v___y_2054_; lean_object* v___y_2076_; lean_object* v___y_2077_; lean_object* v___y_2078_; lean_object* v___y_2079_; lean_object* v___y_2080_; lean_object* v___y_2081_; lean_object* v___y_2082_; lean_object* v___y_2083_; lean_object* v___y_2084_; lean_object* v___y_2085_; lean_object* v_prot_2086_; lean_object* v___y_2087_; lean_object* v___y_2088_; lean_object* v___y_2110_; lean_object* v___y_2111_; lean_object* v___y_2112_; lean_object* v___y_2113_; lean_object* v___y_2114_; lean_object* v___y_2115_; lean_object* v___y_2116_; lean_object* v___y_2117_; lean_object* v___y_2118_; lean_object* v_vis_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2143_; lean_object* v___y_2144_; lean_object* v___y_2145_; lean_object* v___y_2146_; lean_object* v___y_2147_; lean_object* v___y_2148_; lean_object* v___y_2149_; lean_object* v___y_2150_; lean_object* v_attrs_2151_; lean_object* v___y_2152_; lean_object* v___y_2153_; lean_object* v___y_2167_; lean_object* v___y_2168_; lean_object* v___y_2169_; lean_object* v___y_2170_; lean_object* v___y_2171_; lean_object* v___y_2172_; lean_object* v___y_2173_; lean_object* v_doc_2174_; lean_object* v___y_2175_; lean_object* v___y_2176_; lean_object* v___y_2200_; lean_object* v___y_2201_; lean_object* v___y_2202_; lean_object* v___y_2203_; lean_object* v___y_2204_; lean_object* v___y_2205_; lean_object* v_n__def_2206_; lean_object* v___y_2207_; lean_object* v___y_2208_; lean_object* v___y_2237_; lean_object* v___y_2238_; lean_object* v___y_2239_; lean_object* v___y_2240_; lean_object* v___y_2241_; lean_object* v___y_2242_; lean_object* v___y_2243_; lean_object* v___y_2244_; lean_object* v___y_2245_; lean_object* v___y_2267_; lean_object* v___y_2268_; lean_object* v___y_2269_; lean_object* v___y_2270_; lean_object* v___y_2271_; lean_object* v___y_2272_; lean_object* v___y_2273_; lean_object* v___y_2274_; lean_object* v___y_2275_; lean_object* v___y_2279_; lean_object* v___y_2280_; lean_object* v___y_2281_; lean_object* v___y_2282_; lean_object* v_fst_2283_; lean_object* v_snd_2284_; lean_object* v___y_2285_; lean_object* v___y_2286_; lean_object* v___y_2289_; lean_object* v___x_2331_; 
v___x_1332_ = lean_unsigned_to_nat(0u);
v___x_1333_ = l_Lean_Syntax_getArg(v_x_1321_, v___x_1332_);
v___x_1334_ = lean_unsigned_to_nat(1u);
v___x_1335_ = lean_unsigned_to_nat(2u);
v___x_1336_ = l_Lean_Syntax_getArg(v_x_1321_, v___x_1335_);
v___x_1337_ = lean_unsigned_to_nat(3u);
v___x_1338_ = l_Lean_Syntax_getArg(v_x_1321_, v___x_1337_);
v___x_1339_ = lean_unsigned_to_nat(4u);
v___x_1340_ = l_Lean_Syntax_getArg(v_x_1321_, v___x_1339_);
v___x_1341_ = lean_unsigned_to_nat(5u);
v___x_1342_ = l_Lean_Syntax_getArg(v_x_1321_, v___x_1341_);
lean_dec(v_x_1321_);
v___x_1343_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__0));
v___x_1679_ = lean_box(0);
v___x_2331_ = l_Lean_Syntax_getOptional_x3f(v___x_1338_);
lean_dec(v___x_1338_);
if (lean_obj_tag(v___x_2331_) == 0)
{
lean_object* v___x_2332_; 
v___x_2332_ = lean_box(0);
v___y_2289_ = v___x_2332_;
goto v___jp_2288_;
}
else
{
lean_object* v_val_2333_; lean_object* v___x_2335_; uint8_t v_isShared_2336_; uint8_t v_isSharedCheck_2340_; 
v_val_2333_ = lean_ctor_get(v___x_2331_, 0);
v_isSharedCheck_2340_ = !lean_is_exclusive(v___x_2331_);
if (v_isSharedCheck_2340_ == 0)
{
v___x_2335_ = v___x_2331_;
v_isShared_2336_ = v_isSharedCheck_2340_;
goto v_resetjp_2334_;
}
else
{
lean_inc(v_val_2333_);
lean_dec(v___x_2331_);
v___x_2335_ = lean_box(0);
v_isShared_2336_ = v_isSharedCheck_2340_;
goto v_resetjp_2334_;
}
v_resetjp_2334_:
{
lean_object* v___x_2338_; 
if (v_isShared_2336_ == 0)
{
v___x_2338_ = v___x_2335_;
goto v_reusejp_2337_;
}
else
{
lean_object* v_reuseFailAlloc_2339_; 
v_reuseFailAlloc_2339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2339_, 0, v_val_2333_);
v___x_2338_ = v_reuseFailAlloc_2339_;
goto v_reusejp_2337_;
}
v_reusejp_2337_:
{
v___y_2289_ = v___x_2338_;
goto v___jp_2288_;
}
}
}
v___jp_1344_:
{
lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; 
lean_inc_ref_n(v___y_1381_, 2);
v___x_1406_ = l_Array_append___redArg(v___y_1381_, v___y_1405_);
lean_dec_ref(v___y_1405_);
lean_inc_n(v___y_1369_, 13);
lean_inc_n(v___y_1393_, 84);
v___x_1407_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1407_, 0, v___y_1393_);
lean_ctor_set(v___x_1407_, 1, v___y_1369_);
lean_ctor_set(v___x_1407_, 2, v___x_1406_);
lean_inc(v___y_1383_);
lean_inc_ref(v___x_1407_);
lean_inc_n(v___y_1401_, 23);
lean_inc_n(v___y_1351_, 2);
v___x_1408_ = l_Lean_Syntax_node7(v___y_1393_, v___y_1351_, v___y_1360_, v___y_1401_, v___x_1407_, v___y_1401_, v___y_1359_, v___y_1383_, v___y_1401_);
lean_inc(v___y_1372_);
lean_inc_n(v___y_1368_, 4);
lean_inc_n(v___y_1361_, 2);
v___x_1409_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1361_, v___y_1368_, v___y_1372_);
lean_inc(v___y_1377_);
v___x_1410_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1377_, v___y_1401_, v___y_1401_);
v___x_1411_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termVal__proj___00__closed__0));
lean_inc_n(v___y_1378_, 2);
v___x_1412_ = l_Lean_Name_str___override(v___y_1378_, v___x_1411_);
v___x_1413_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__0));
v___x_1414_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1414_, 0, v___y_1393_);
lean_ctor_set(v___x_1414_, 1, v___x_1413_);
lean_inc(v___y_1357_);
lean_inc(v___y_1364_);
lean_inc(v___y_1366_);
lean_inc_n(v___y_1403_, 2);
lean_inc(v___y_1345_);
v___x_1415_ = l_Lean_Syntax_node4(v___y_1393_, v___y_1345_, v___y_1403_, v___y_1366_, v___y_1364_, v___y_1357_);
lean_inc_n(v___y_1349_, 3);
lean_inc_n(v___y_1385_, 3);
v___x_1416_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1385_, v___y_1349_, v___x_1415_);
v___x_1417_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1412_, v___x_1414_, v___x_1416_);
lean_inc(v___y_1365_);
lean_inc(v___y_1367_);
lean_inc(v___y_1370_);
v___x_1418_ = l_Lean_Syntax_node4(v___y_1393_, v___y_1370_, v___y_1367_, v___x_1417_, v___y_1365_, v___y_1401_);
lean_inc(v___y_1400_);
v___x_1419_ = l_Lean_Syntax_node5(v___y_1393_, v___y_1400_, v___y_1394_, v___x_1409_, v___x_1410_, v___x_1418_, v___y_1401_);
lean_inc_n(v___y_1371_, 2);
v___x_1420_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1371_, v___x_1408_, v___x_1419_);
v___x_1421_ = l_Lean_Syntax_node7(v___y_1393_, v___y_1351_, v___y_1401_, v___y_1401_, v___x_1407_, v___y_1401_, v___y_1401_, v___y_1383_, v___y_1401_);
v___x_1422_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__1));
v___x_1423_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__2));
v___x_1424_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1424_, 0, v___y_1393_);
lean_ctor_set(v___x_1424_, 1, v___x_1422_);
lean_inc(v___y_1392_);
v___x_1425_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1361_, v___y_1392_, v___y_1372_);
v___x_1426_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__15));
v___x_1427_ = l_Lean_Name_str___override(v___y_1378_, v___x_1426_);
v___x_1428_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__3));
v___x_1429_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1429_, 0, v___y_1393_);
lean_ctor_set(v___x_1429_, 1, v___x_1428_);
v___x_1430_ = l_Lean_Syntax_node4(v___y_1393_, v___y_1345_, v___y_1368_, v___y_1366_, v___y_1364_, v___y_1357_);
v___x_1431_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1385_, v___y_1349_, v___x_1430_);
v___x_1432_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__4));
v___x_1433_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__5));
v___x_1434_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__6));
lean_inc_ref(v___y_1362_);
v___x_1435_ = l_Lean_Name_mkStr4(v___y_1362_, v___x_1432_, v___x_1433_, v___x_1434_);
v___x_1436_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__7));
v___x_1437_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1437_, 0, v___y_1393_);
lean_ctor_set(v___x_1437_, 1, v___x_1436_);
lean_inc(v___y_1353_);
v___x_1438_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1385_, v___y_1349_, v___y_1353_);
v___x_1439_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1435_, v___x_1437_, v___x_1438_);
v___x_1440_ = l_Lean_Syntax_node3(v___y_1393_, v___y_1355_, v___y_1382_, v___x_1439_, v___y_1402_);
v___x_1441_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1385_, v___y_1349_, v___x_1440_);
v___x_1442_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1369_, v___x_1431_, v___x_1441_);
lean_inc(v___y_1373_);
v___x_1443_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1373_, v___y_1389_, v___x_1442_);
v___x_1444_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1427_, v___x_1429_, v___x_1443_);
v___x_1445_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1404_, v___y_1397_, v___x_1444_);
v___x_1446_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1350_, v___y_1401_, v___x_1445_);
v___x_1447_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__8));
lean_inc_ref_n(v___y_1384_, 6);
v___x_1448_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1384_, v___x_1447_);
v___x_1449_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__9));
v___x_1450_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1450_, 0, v___y_1393_);
lean_ctor_set(v___x_1450_, 1, v___x_1449_);
v___x_1451_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__10));
lean_inc_ref_n(v___y_1356_, 9);
v___x_1452_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1451_);
v___x_1453_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__11));
v___x_1454_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1453_);
v___x_1455_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__12));
v___x_1456_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1455_);
v___x_1457_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1457_, 0, v___y_1393_);
lean_ctor_set(v___x_1457_, 1, v___x_1455_);
v___x_1458_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1456_, v___x_1457_, v___y_1401_);
v___x_1459_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__13));
v___x_1460_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1459_);
v___x_1461_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1461_, 0, v___y_1393_);
lean_ctor_set(v___x_1461_, 1, v___x_1459_);
v___x_1462_ = l_Lean_Syntax_node1(v___y_1393_, v___y_1369_, v___y_1368_);
lean_inc_n(v___x_1462_, 2);
v___x_1463_ = l_Lean_Syntax_node3(v___y_1393_, v___x_1460_, v___x_1461_, v___x_1462_, v___y_1401_);
v___x_1464_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__14));
v___x_1465_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1464_);
v___x_1466_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__15));
v___x_1467_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1467_, 0, v___y_1393_);
lean_ctor_set(v___x_1467_, 1, v___x_1466_);
v___x_1468_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__16));
v___x_1469_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1468_);
v___x_1470_ = l_Lean_Syntax_node1(v___y_1393_, v___x_1469_, v___y_1401_);
v___x_1471_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__17));
v___x_1472_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1471_);
v___x_1473_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__18));
v___x_1474_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1474_, 0, v___y_1393_);
lean_ctor_set(v___x_1474_, 1, v___x_1473_);
v___x_1475_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__19));
v___x_1476_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1475_);
v___x_1477_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__20));
v___x_1478_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1384_, v___x_1477_);
v___x_1479_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1479_, 0, v___y_1393_);
lean_ctor_set(v___x_1479_, 1, v___x_1477_);
v___x_1480_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__22));
v___x_1481_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__23));
v___x_1482_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1482_, 0, v___y_1393_);
lean_ctor_set(v___x_1482_, 1, v___x_1481_);
v___x_1483_ = l_Lean_Syntax_node3(v___y_1393_, v___y_1369_, v___y_1379_, v___y_1390_, v___y_1375_);
v___x_1484_ = l_Lean_Syntax_node3(v___y_1393_, v___y_1388_, v___y_1398_, v___x_1483_, v___y_1395_);
v___x_1485_ = l_Lean_Syntax_node3(v___y_1393_, v___x_1480_, v___y_1403_, v___x_1482_, v___x_1484_);
v___x_1486_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__24));
v___x_1487_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1384_, v___x_1486_);
v___x_1488_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__25));
v___x_1489_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1489_, 0, v___y_1393_);
lean_ctor_set(v___x_1489_, 1, v___x_1488_);
v___x_1490_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__27, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__27_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__27);
v___x_1491_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__28));
lean_inc_ref(v___y_1374_);
v___x_1492_ = l_Lean_Name_mkStr2(v___y_1374_, v___x_1491_);
lean_inc_n(v___y_1358_, 2);
lean_inc(v___x_1492_);
lean_inc_n(v___y_1354_, 2);
v___x_1493_ = l_Lean_addMacroScope(v___y_1354_, v___x_1492_, v___y_1358_);
lean_inc(v___y_1347_);
v___x_1494_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1494_, 0, v___x_1492_);
lean_ctor_set(v___x_1494_, 1, v___y_1347_);
lean_inc_n(v___y_1380_, 3);
v___x_1495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1495_, 0, v___x_1494_);
lean_ctor_set(v___x_1495_, 1, v___y_1380_);
v___x_1496_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1496_, 0, v___y_1393_);
lean_ctor_set(v___x_1496_, 1, v___x_1490_);
lean_ctor_set(v___x_1496_, 2, v___x_1493_);
lean_ctor_set(v___x_1496_, 3, v___x_1495_);
v___x_1497_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__29));
v___x_1498_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1384_, v___x_1497_);
v___x_1499_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__30));
v___x_1500_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1500_, 0, v___y_1393_);
lean_ctor_set(v___x_1500_, 1, v___x_1499_);
v___x_1501_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__32));
v___x_1502_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__33));
v___x_1503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1503_, 0, v___y_1393_);
lean_ctor_set(v___x_1503_, 1, v___x_1502_);
v___x_1504_ = l_Lean_Syntax_node1(v___y_1393_, v___x_1501_, v___x_1503_);
lean_inc_ref(v___x_1500_);
lean_inc(v___x_1498_);
v___x_1505_ = l_Lean_Syntax_node3(v___y_1393_, v___x_1498_, v___y_1403_, v___x_1500_, v___x_1504_);
v___x_1506_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__35, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__35_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__35);
v___x_1507_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__36));
v___x_1508_ = l_Lean_addMacroScope(v___y_1354_, v___x_1507_, v___y_1358_);
v___x_1509_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1509_, 0, v___y_1393_);
lean_ctor_set(v___x_1509_, 1, v___x_1506_);
lean_ctor_set(v___x_1509_, 2, v___x_1508_);
lean_ctor_set(v___x_1509_, 3, v___y_1380_);
v___x_1510_ = l_Lean_Syntax_node3(v___y_1393_, v___x_1498_, v___x_1505_, v___x_1500_, v___x_1509_);
v___x_1511_ = l_Lean_Syntax_node1(v___y_1393_, v___y_1369_, v___x_1510_);
v___x_1512_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1373_, v___x_1496_, v___x_1511_);
v___x_1513_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1487_, v___x_1489_, v___x_1512_);
v___x_1514_ = l_Lean_Syntax_node3(v___y_1393_, v___x_1478_, v___x_1479_, v___x_1485_, v___x_1513_);
v___x_1515_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1476_, v___y_1401_, v___x_1514_);
v___x_1516_ = l_Lean_Syntax_node1(v___y_1393_, v___y_1369_, v___x_1515_);
v___x_1517_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__37));
v___x_1518_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1518_, 0, v___y_1393_);
lean_ctor_set(v___x_1518_, 1, v___x_1517_);
lean_inc_ref_n(v___x_1518_, 3);
lean_inc_ref_n(v___x_1474_, 3);
v___x_1519_ = l_Lean_Syntax_node3(v___y_1393_, v___x_1472_, v___x_1474_, v___x_1516_, v___x_1518_);
v___x_1520_ = l_Lean_Syntax_node4(v___y_1393_, v___x_1465_, v___x_1467_, v___x_1470_, v___x_1519_, v___y_1401_);
v___x_1521_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__38));
v___x_1522_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1356_, v___x_1521_);
lean_inc_ref(v___y_1399_);
v___x_1523_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1523_, 0, v___y_1393_);
lean_ctor_set(v___x_1523_, 1, v___y_1399_);
v___x_1524_ = l_Lean_Syntax_node1(v___y_1393_, v___x_1522_, v___x_1523_);
v___x_1525_ = l_Lean_Syntax_node7(v___y_1393_, v___y_1369_, v___x_1458_, v___y_1401_, v___x_1463_, v___y_1401_, v___x_1520_, v___y_1401_, v___x_1524_);
v___x_1526_ = l_Lean_Syntax_node1(v___y_1393_, v___x_1454_, v___x_1525_);
v___x_1527_ = l_Lean_Syntax_node1(v___y_1393_, v___x_1452_, v___x_1526_);
v___x_1528_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1448_, v___x_1450_, v___x_1527_);
v___x_1529_ = l_Lean_Syntax_node4(v___y_1393_, v___y_1370_, v___y_1367_, v___x_1528_, v___y_1365_, v___y_1401_);
v___x_1530_ = l_Lean_Syntax_node4(v___y_1393_, v___x_1423_, v___x_1424_, v___x_1425_, v___x_1446_, v___x_1529_);
v___x_1531_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1371_, v___x_1421_, v___x_1530_);
v___x_1532_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__39));
v___x_1533_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__40));
v___x_1534_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1534_, 0, v___y_1393_);
lean_ctor_set(v___x_1534_, 1, v___x_1532_);
v___x_1535_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__41));
v___x_1536_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1384_, v___x_1535_);
v___x_1537_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__42));
v___x_1538_ = l_Lean_Name_mkStr4(v___x_1328_, v___x_1343_, v___y_1384_, v___x_1537_);
v___x_1539_ = l_Lean_Syntax_node1(v___y_1393_, v___x_1538_, v___y_1401_);
v___x_1540_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__45));
v___x_1541_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__47, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__47_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__47);
v___x_1542_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__48));
v___x_1543_ = l_Lean_addMacroScope(v___y_1354_, v___x_1542_, v___y_1358_);
v___x_1544_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1544_, 0, v___y_1393_);
lean_ctor_set(v___x_1544_, 1, v___x_1541_);
lean_ctor_set(v___x_1544_, 2, v___x_1543_);
lean_ctor_set(v___x_1544_, 3, v___y_1380_);
v___x_1545_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1540_, v___x_1544_, v___y_1401_);
lean_inc(v___x_1539_);
lean_inc(v___x_1536_);
v___x_1546_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1536_, v___x_1539_, v___x_1545_);
v___x_1547_ = l_Lean_Syntax_node1(v___y_1393_, v___y_1369_, v___x_1546_);
v___x_1548_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1369_, v___y_1368_, v___y_1353_);
lean_inc_ref_n(v___x_1534_, 2);
v___x_1549_ = l_Lean_Syntax_node5(v___y_1393_, v___x_1533_, v___x_1534_, v___x_1474_, v___x_1547_, v___x_1518_, v___x_1548_);
v___x_1550_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__49));
v___x_1551_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__50));
v___x_1552_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1552_, 0, v___y_1393_);
lean_ctor_set(v___x_1552_, 1, v___x_1550_);
v___x_1553_ = l_Lean_Syntax_node1(v___y_1393_, v___y_1369_, v___y_1392_);
v___x_1554_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1551_, v___x_1552_, v___x_1553_);
v___x_1555_ = l_Lean_Syntax_node2(v___y_1393_, v___x_1536_, v___x_1539_, v___x_1554_);
v___x_1556_ = l_Lean_Syntax_node1(v___y_1393_, v___y_1369_, v___x_1555_);
v___x_1557_ = l_Lean_Syntax_node5(v___y_1393_, v___x_1533_, v___x_1534_, v___x_1474_, v___x_1556_, v___x_1518_, v___x_1462_);
v___x_1558_ = l_Array_append___redArg(v___y_1381_, v___y_1376_);
lean_dec_ref(v___y_1376_);
v___x_1559_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1559_, 0, v___y_1393_);
lean_ctor_set(v___x_1559_, 1, v___y_1369_);
lean_ctor_set(v___x_1559_, 2, v___x_1558_);
v___x_1560_ = l_Lean_Syntax_node5(v___y_1393_, v___x_1533_, v___x_1534_, v___x_1474_, v___x_1559_, v___x_1518_, v___x_1462_);
v___x_1561_ = l_Lean_Syntax_node7(v___y_1393_, v___y_1369_, v___y_1387_, v___y_1348_, v___x_1420_, v___x_1531_, v___x_1549_, v___x_1557_, v___x_1560_);
lean_inc(v___y_1363_);
v___x_1562_ = l_Lean_Syntax_node2(v___y_1393_, v___y_1363_, v___y_1391_, v___x_1561_);
v___x_1563_ = l_Lean_Elab_Command_elabCommand(v___x_1562_, v___y_1386_, v___y_1346_);
if (lean_obj_tag(v___x_1563_) == 0)
{
lean_dec_ref_known(v___x_1563_, 1);
if (lean_obj_tag(v___y_1352_) == 0)
{
lean_dec(v___y_1368_);
goto v___jp_1325_;
}
else
{
lean_dec_ref_known(v___y_1352_, 1);
if (v___y_1396_ == 0)
{
lean_dec(v___y_1368_);
goto v___jp_1325_;
}
else
{
lean_object* v___x_1564_; 
v___x_1564_ = l_Lean_Elab_Command_getScope___redArg(v___y_1346_);
if (lean_obj_tag(v___x_1564_) == 0)
{
lean_object* v_a_1565_; lean_object* v___x_1567_; uint8_t v_isShared_1568_; uint8_t v_isSharedCheck_1598_; 
v_a_1565_ = lean_ctor_get(v___x_1564_, 0);
v_isSharedCheck_1598_ = !lean_is_exclusive(v___x_1564_);
if (v_isSharedCheck_1598_ == 0)
{
v___x_1567_ = v___x_1564_;
v_isShared_1568_ = v_isSharedCheck_1598_;
goto v_resetjp_1566_;
}
else
{
lean_inc(v_a_1565_);
lean_dec(v___x_1564_);
v___x_1567_ = lean_box(0);
v_isShared_1568_ = v_isSharedCheck_1598_;
goto v_resetjp_1566_;
}
v_resetjp_1566_:
{
lean_object* v___x_1569_; lean_object* v_currNamespace_1570_; lean_object* v_env_1571_; lean_object* v_messages_1572_; lean_object* v_scopes_1573_; lean_object* v_usedQuotCtxts_1574_; lean_object* v_nextMacroScope_1575_; lean_object* v_maxRecDepth_1576_; lean_object* v_ngen_1577_; lean_object* v_auxDeclNGen_1578_; lean_object* v_infoState_1579_; lean_object* v_traceState_1580_; lean_object* v_snapshotTasks_1581_; lean_object* v_prevLinterStates_1582_; lean_object* v___x_1584_; uint8_t v_isShared_1585_; uint8_t v_isSharedCheck_1597_; 
v___x_1569_ = lean_st_ref_take(v___y_1346_);
v_currNamespace_1570_ = lean_ctor_get(v_a_1565_, 2);
lean_inc(v_currNamespace_1570_);
lean_dec(v_a_1565_);
v_env_1571_ = lean_ctor_get(v___x_1569_, 0);
v_messages_1572_ = lean_ctor_get(v___x_1569_, 1);
v_scopes_1573_ = lean_ctor_get(v___x_1569_, 2);
v_usedQuotCtxts_1574_ = lean_ctor_get(v___x_1569_, 3);
v_nextMacroScope_1575_ = lean_ctor_get(v___x_1569_, 4);
v_maxRecDepth_1576_ = lean_ctor_get(v___x_1569_, 5);
v_ngen_1577_ = lean_ctor_get(v___x_1569_, 6);
v_auxDeclNGen_1578_ = lean_ctor_get(v___x_1569_, 7);
v_infoState_1579_ = lean_ctor_get(v___x_1569_, 8);
v_traceState_1580_ = lean_ctor_get(v___x_1569_, 9);
v_snapshotTasks_1581_ = lean_ctor_get(v___x_1569_, 10);
v_prevLinterStates_1582_ = lean_ctor_get(v___x_1569_, 11);
v_isSharedCheck_1597_ = !lean_is_exclusive(v___x_1569_);
if (v_isSharedCheck_1597_ == 0)
{
v___x_1584_ = v___x_1569_;
v_isShared_1585_ = v_isSharedCheck_1597_;
goto v_resetjp_1583_;
}
else
{
lean_inc(v_prevLinterStates_1582_);
lean_inc(v_snapshotTasks_1581_);
lean_inc(v_traceState_1580_);
lean_inc(v_infoState_1579_);
lean_inc(v_auxDeclNGen_1578_);
lean_inc(v_ngen_1577_);
lean_inc(v_maxRecDepth_1576_);
lean_inc(v_nextMacroScope_1575_);
lean_inc(v_usedQuotCtxts_1574_);
lean_inc(v_scopes_1573_);
lean_inc(v_messages_1572_);
lean_inc(v_env_1571_);
lean_dec(v___x_1569_);
v___x_1584_ = lean_box(0);
v_isShared_1585_ = v_isSharedCheck_1597_;
goto v_resetjp_1583_;
}
v_resetjp_1583_:
{
lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1590_; 
v___x_1586_ = l_Lean_TSyntax_getId(v___y_1368_);
lean_dec(v___y_1368_);
v___x_1587_ = l_Lean_Name_append(v_currNamespace_1570_, v___x_1586_);
v___x_1588_ = l_Lean_addProtected(v_env_1571_, v___x_1587_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v___x_1588_);
v___x_1590_ = v___x_1584_;
goto v_reusejp_1589_;
}
else
{
lean_object* v_reuseFailAlloc_1596_; 
v_reuseFailAlloc_1596_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1596_, 0, v___x_1588_);
lean_ctor_set(v_reuseFailAlloc_1596_, 1, v_messages_1572_);
lean_ctor_set(v_reuseFailAlloc_1596_, 2, v_scopes_1573_);
lean_ctor_set(v_reuseFailAlloc_1596_, 3, v_usedQuotCtxts_1574_);
lean_ctor_set(v_reuseFailAlloc_1596_, 4, v_nextMacroScope_1575_);
lean_ctor_set(v_reuseFailAlloc_1596_, 5, v_maxRecDepth_1576_);
lean_ctor_set(v_reuseFailAlloc_1596_, 6, v_ngen_1577_);
lean_ctor_set(v_reuseFailAlloc_1596_, 7, v_auxDeclNGen_1578_);
lean_ctor_set(v_reuseFailAlloc_1596_, 8, v_infoState_1579_);
lean_ctor_set(v_reuseFailAlloc_1596_, 9, v_traceState_1580_);
lean_ctor_set(v_reuseFailAlloc_1596_, 10, v_snapshotTasks_1581_);
lean_ctor_set(v_reuseFailAlloc_1596_, 11, v_prevLinterStates_1582_);
v___x_1590_ = v_reuseFailAlloc_1596_;
goto v_reusejp_1589_;
}
v_reusejp_1589_:
{
lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1594_; 
v___x_1591_ = lean_st_ref_set(v___y_1346_, v___x_1590_);
v___x_1592_ = lean_box(0);
if (v_isShared_1568_ == 0)
{
lean_ctor_set(v___x_1567_, 0, v___x_1592_);
v___x_1594_ = v___x_1567_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v___x_1592_);
v___x_1594_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1593_;
}
v_reusejp_1593_:
{
return v___x_1594_;
}
}
}
}
}
else
{
lean_object* v_a_1599_; lean_object* v___x_1601_; uint8_t v_isShared_1602_; uint8_t v_isSharedCheck_1606_; 
lean_dec(v___y_1368_);
v_a_1599_ = lean_ctor_get(v___x_1564_, 0);
v_isSharedCheck_1606_ = !lean_is_exclusive(v___x_1564_);
if (v_isSharedCheck_1606_ == 0)
{
v___x_1601_ = v___x_1564_;
v_isShared_1602_ = v_isSharedCheck_1606_;
goto v_resetjp_1600_;
}
else
{
lean_inc(v_a_1599_);
lean_dec(v___x_1564_);
v___x_1601_ = lean_box(0);
v_isShared_1602_ = v_isSharedCheck_1606_;
goto v_resetjp_1600_;
}
v_resetjp_1600_:
{
lean_object* v___x_1604_; 
if (v_isShared_1602_ == 0)
{
v___x_1604_ = v___x_1601_;
goto v_reusejp_1603_;
}
else
{
lean_object* v_reuseFailAlloc_1605_; 
v_reuseFailAlloc_1605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1605_, 0, v_a_1599_);
v___x_1604_ = v_reuseFailAlloc_1605_;
goto v_reusejp_1603_;
}
v_reusejp_1603_:
{
return v___x_1604_;
}
}
}
}
}
}
else
{
lean_dec(v___y_1368_);
lean_dec(v___y_1352_);
return v___x_1563_;
}
}
v___jp_1607_:
{
lean_object* v___x_1669_; lean_object* v___x_1670_; 
lean_inc_ref(v___y_1644_);
v___x_1669_ = l_Array_append___redArg(v___y_1644_, v___y_1668_);
lean_dec_ref(v___y_1668_);
lean_inc(v___y_1632_);
lean_inc(v___y_1656_);
v___x_1670_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1670_, 0, v___y_1656_);
lean_ctor_set(v___x_1670_, 1, v___y_1632_);
lean_ctor_set(v___x_1670_, 2, v___x_1669_);
if (lean_obj_tag(v___y_1630_) == 1)
{
lean_object* v_val_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; 
v_val_1671_ = lean_ctor_get(v___y_1630_, 0);
lean_inc(v_val_1671_);
lean_dec_ref_known(v___y_1630_, 1);
v___x_1672_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__51));
v___x_1673_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52));
v___x_1674_ = l_Lean_SourceInfo_fromRef(v_val_1671_, v___x_1330_);
lean_dec(v_val_1671_);
v___x_1675_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1675_, 0, v___x_1674_);
lean_ctor_set(v___x_1675_, 1, v___x_1672_);
lean_inc(v___y_1656_);
v___x_1676_ = l_Lean_Syntax_node1(v___y_1656_, v___x_1673_, v___x_1675_);
v___x_1677_ = l_Array_mkArray1___redArg(v___x_1676_);
v___y_1345_ = v___y_1608_;
v___y_1346_ = v___y_1609_;
v___y_1347_ = v___y_1610_;
v___y_1348_ = v___y_1611_;
v___y_1349_ = v___y_1612_;
v___y_1350_ = v___y_1613_;
v___y_1351_ = v___y_1614_;
v___y_1352_ = v___y_1615_;
v___y_1353_ = v___y_1616_;
v___y_1354_ = v___y_1617_;
v___y_1355_ = v___y_1618_;
v___y_1356_ = v___y_1619_;
v___y_1357_ = v___y_1621_;
v___y_1358_ = v___y_1620_;
v___y_1359_ = v___y_1622_;
v___y_1360_ = v___x_1670_;
v___y_1361_ = v___y_1624_;
v___y_1362_ = v___y_1623_;
v___y_1363_ = v___y_1625_;
v___y_1364_ = v___y_1626_;
v___y_1365_ = v___y_1627_;
v___y_1366_ = v___y_1628_;
v___y_1367_ = v___y_1629_;
v___y_1368_ = v___y_1631_;
v___y_1369_ = v___y_1632_;
v___y_1370_ = v___y_1633_;
v___y_1371_ = v___y_1634_;
v___y_1372_ = v___y_1635_;
v___y_1373_ = v___y_1636_;
v___y_1374_ = v___y_1637_;
v___y_1375_ = v___y_1638_;
v___y_1376_ = v___y_1639_;
v___y_1377_ = v___y_1640_;
v___y_1378_ = v___y_1641_;
v___y_1379_ = v___y_1642_;
v___y_1380_ = v___y_1643_;
v___y_1381_ = v___y_1644_;
v___y_1382_ = v___y_1645_;
v___y_1383_ = v___y_1646_;
v___y_1384_ = v___y_1647_;
v___y_1385_ = v___y_1648_;
v___y_1386_ = v___y_1649_;
v___y_1387_ = v___y_1650_;
v___y_1388_ = v___y_1651_;
v___y_1389_ = v___y_1652_;
v___y_1390_ = v___y_1653_;
v___y_1391_ = v___y_1654_;
v___y_1392_ = v___y_1655_;
v___y_1393_ = v___y_1656_;
v___y_1394_ = v___y_1657_;
v___y_1395_ = v___y_1658_;
v___y_1396_ = v___y_1659_;
v___y_1397_ = v___y_1660_;
v___y_1398_ = v___y_1661_;
v___y_1399_ = v___y_1662_;
v___y_1400_ = v___y_1665_;
v___y_1401_ = v___y_1664_;
v___y_1402_ = v___y_1663_;
v___y_1403_ = v___y_1666_;
v___y_1404_ = v___y_1667_;
v___y_1405_ = v___x_1677_;
goto v___jp_1344_;
}
else
{
lean_object* v___x_1678_; 
lean_dec(v___y_1630_);
v___x_1678_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53));
v___y_1345_ = v___y_1608_;
v___y_1346_ = v___y_1609_;
v___y_1347_ = v___y_1610_;
v___y_1348_ = v___y_1611_;
v___y_1349_ = v___y_1612_;
v___y_1350_ = v___y_1613_;
v___y_1351_ = v___y_1614_;
v___y_1352_ = v___y_1615_;
v___y_1353_ = v___y_1616_;
v___y_1354_ = v___y_1617_;
v___y_1355_ = v___y_1618_;
v___y_1356_ = v___y_1619_;
v___y_1357_ = v___y_1621_;
v___y_1358_ = v___y_1620_;
v___y_1359_ = v___y_1622_;
v___y_1360_ = v___x_1670_;
v___y_1361_ = v___y_1624_;
v___y_1362_ = v___y_1623_;
v___y_1363_ = v___y_1625_;
v___y_1364_ = v___y_1626_;
v___y_1365_ = v___y_1627_;
v___y_1366_ = v___y_1628_;
v___y_1367_ = v___y_1629_;
v___y_1368_ = v___y_1631_;
v___y_1369_ = v___y_1632_;
v___y_1370_ = v___y_1633_;
v___y_1371_ = v___y_1634_;
v___y_1372_ = v___y_1635_;
v___y_1373_ = v___y_1636_;
v___y_1374_ = v___y_1637_;
v___y_1375_ = v___y_1638_;
v___y_1376_ = v___y_1639_;
v___y_1377_ = v___y_1640_;
v___y_1378_ = v___y_1641_;
v___y_1379_ = v___y_1642_;
v___y_1380_ = v___y_1643_;
v___y_1381_ = v___y_1644_;
v___y_1382_ = v___y_1645_;
v___y_1383_ = v___y_1646_;
v___y_1384_ = v___y_1647_;
v___y_1385_ = v___y_1648_;
v___y_1386_ = v___y_1649_;
v___y_1387_ = v___y_1650_;
v___y_1388_ = v___y_1651_;
v___y_1389_ = v___y_1652_;
v___y_1390_ = v___y_1653_;
v___y_1391_ = v___y_1654_;
v___y_1392_ = v___y_1655_;
v___y_1393_ = v___y_1656_;
v___y_1394_ = v___y_1657_;
v___y_1395_ = v___y_1658_;
v___y_1396_ = v___y_1659_;
v___y_1397_ = v___y_1660_;
v___y_1398_ = v___y_1661_;
v___y_1399_ = v___y_1662_;
v___y_1400_ = v___y_1665_;
v___y_1401_ = v___y_1664_;
v___y_1402_ = v___y_1663_;
v___y_1403_ = v___y_1666_;
v___y_1404_ = v___y_1667_;
v___y_1405_ = v___x_1678_;
goto v___jp_1344_;
}
}
v___jp_1680_:
{
lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; size_t v_sz_1775_; size_t v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; 
lean_inc_ref_n(v___y_1687_, 2);
v___x_1714_ = l_Array_append___redArg(v___y_1687_, v___y_1713_);
lean_dec_ref(v___y_1713_);
lean_inc_n(v___y_1709_, 6);
lean_inc_n(v___y_1704_, 42);
v___x_1715_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1715_, 0, v___y_1704_);
lean_ctor_set(v___x_1715_, 1, v___y_1709_);
lean_ctor_set(v___x_1715_, 2, v___x_1714_);
lean_inc_ref_n(v___x_1715_, 2);
lean_inc_n(v___y_1691_, 2);
lean_inc_n(v___y_1698_, 2);
v___x_1716_ = l_Lean_Syntax_node2(v___y_1704_, v___y_1698_, v___y_1691_, v___x_1715_);
lean_inc_n(v___y_1711_, 5);
lean_inc(v___y_1705_);
lean_inc(v___y_1712_);
v___x_1717_ = l_Lean_Syntax_node5(v___y_1704_, v___y_1712_, v___y_1705_, v___x_1716_, v___x_1340_, v___x_1342_, v___y_1711_);
lean_inc(v___y_1695_);
lean_inc_n(v___y_1710_, 2);
v___x_1718_ = l_Lean_Syntax_node2(v___y_1704_, v___y_1710_, v___y_1695_, v___x_1717_);
v___x_1719_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__54));
v___x_1720_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__55));
v___x_1721_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1721_, 0, v___y_1704_);
lean_ctor_set(v___x_1721_, 1, v___x_1719_);
v___x_1722_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__57, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__57_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__57);
v___x_1723_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__58));
lean_inc_n(v___y_1694_, 5);
lean_inc_n(v___y_1690_, 5);
v___x_1724_ = l_Lean_addMacroScope(v___y_1690_, v___x_1723_, v___y_1694_);
lean_inc_n(v___y_1685_, 5);
v___x_1725_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1725_, 0, v___y_1704_);
lean_ctor_set(v___x_1725_, 1, v___x_1722_);
lean_ctor_set(v___x_1725_, 2, v___x_1724_);
lean_ctor_set(v___x_1725_, 3, v___y_1685_);
lean_inc_ref(v___x_1725_);
v___x_1726_ = l_Lean_Syntax_node2(v___y_1704_, v___y_1698_, v___x_1725_, v___x_1715_);
v___x_1727_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__60));
v___x_1728_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__1));
v___x_1729_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__62));
v___x_1730_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__21));
v___x_1731_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1731_, 0, v___y_1704_);
lean_ctor_set(v___x_1731_, 1, v___x_1730_);
v___x_1732_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__25));
v___x_1733_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__26));
v___x_1734_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__27);
v___x_1735_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__28));
v___x_1736_ = l_Lean_addMacroScope(v___y_1690_, v___x_1735_, v___y_1694_);
v___x_1737_ = lean_box(0);
v___x_1738_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__63));
v___x_1739_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__30));
v___x_1740_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1740_, 0, v___x_1739_);
lean_ctor_set(v___x_1740_, 1, v___y_1685_);
v___x_1741_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1741_, 0, v___x_1738_);
lean_ctor_set(v___x_1741_, 1, v___x_1740_);
v___x_1742_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1742_, 0, v___y_1704_);
lean_ctor_set(v___x_1742_, 1, v___x_1734_);
lean_ctor_set(v___x_1742_, 2, v___x_1736_);
lean_ctor_set(v___x_1742_, 3, v___x_1741_);
v___x_1743_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__65));
v___x_1744_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__5));
v___x_1745_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__6));
v___x_1746_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1746_, 0, v___y_1704_);
lean_ctor_set(v___x_1746_, 1, v___x_1745_);
v___x_1747_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__8));
v___x_1748_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__10);
v___x_1749_ = l_Lean_addMacroScope(v___y_1690_, v___x_1679_, v___y_1694_);
v___x_1750_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__12));
v___x_1751_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__15));
v___x_1752_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__17));
v___x_1753_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1753_, 0, v___x_1752_);
lean_ctor_set(v___x_1753_, 1, v___y_1685_);
v___x_1754_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1754_, 0, v___x_1751_);
lean_ctor_set(v___x_1754_, 1, v___x_1753_);
v___x_1755_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1755_, 0, v___x_1750_);
lean_ctor_set(v___x_1755_, 1, v___x_1754_);
v___x_1756_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1756_, 0, v___y_1704_);
lean_ctor_set(v___x_1756_, 1, v___x_1748_);
lean_ctor_set(v___x_1756_, 2, v___x_1749_);
lean_ctor_set(v___x_1756_, 3, v___x_1755_);
v___x_1757_ = l_Lean_Syntax_node1(v___y_1704_, v___x_1747_, v___x_1756_);
v___x_1758_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1744_, v___x_1746_, v___x_1757_);
v___x_1759_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__66, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__66_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__66);
v___x_1760_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termEta__helper____1___redArg___closed__1));
v___x_1761_ = l_Lean_addMacroScope(v___y_1690_, v___x_1760_, v___y_1694_);
v___x_1762_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__67));
v___x_1763_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__68));
v___x_1764_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1764_, 0, v___x_1763_);
lean_ctor_set(v___x_1764_, 1, v___y_1685_);
v___x_1765_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1765_, 0, v___x_1762_);
lean_ctor_set(v___x_1765_, 1, v___x_1764_);
v___x_1766_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1766_, 0, v___y_1704_);
lean_ctor_set(v___x_1766_, 1, v___x_1759_);
lean_ctor_set(v___x_1766_, 2, v___x_1761_);
lean_ctor_set(v___x_1766_, 3, v___x_1765_);
v___x_1767_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__70));
v___x_1768_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__71));
v___x_1769_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1769_, 0, v___y_1704_);
lean_ctor_set(v___x_1769_, 1, v___x_1768_);
v___x_1770_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__73));
v___x_1771_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__74));
v___x_1772_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1772_, 0, v___y_1704_);
lean_ctor_set(v___x_1772_, 1, v___x_1771_);
v___x_1773_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__75));
v___x_1774_ = l_Lean_Syntax_TSepArray_getElems___redArg(v___y_1700_);
lean_dec_ref(v___y_1700_);
v_sz_1775_ = lean_array_size(v___x_1774_);
v___x_1776_ = ((size_t)0ULL);
v___x_1777_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__0(v_sz_1775_, v___x_1776_, v___x_1774_);
v___x_1778_ = l_Lean_Syntax_SepArray_ofElems(v___x_1773_, v___x_1777_);
lean_dec_ref(v___x_1777_);
v___x_1779_ = l_Array_append___redArg(v___y_1687_, v___x_1778_);
lean_dec_ref(v___x_1778_);
v___x_1780_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1780_, 0, v___y_1704_);
lean_ctor_set(v___x_1780_, 1, v___y_1709_);
lean_ctor_set(v___x_1780_, 2, v___x_1779_);
v___x_1781_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__76));
v___x_1782_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1782_, 0, v___y_1704_);
lean_ctor_set(v___x_1782_, 1, v___x_1781_);
lean_inc_ref(v___x_1782_);
lean_inc_ref(v___x_1780_);
lean_inc_ref(v___x_1772_);
v___x_1783_ = l_Lean_Syntax_node4(v___y_1704_, v___x_1770_, v___y_1691_, v___x_1772_, v___x_1780_, v___x_1782_);
lean_inc_ref(v___x_1769_);
v___x_1784_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1767_, v___x_1769_, v___x_1783_);
lean_inc(v___x_1784_);
v___x_1785_ = l_Lean_Syntax_node1(v___y_1704_, v___y_1709_, v___x_1784_);
lean_inc_ref(v___x_1766_);
v___x_1786_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1732_, v___x_1766_, v___x_1785_);
v___x_1787_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__36));
v___x_1788_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1788_, 0, v___y_1704_);
lean_ctor_set(v___x_1788_, 1, v___x_1787_);
lean_inc_ref(v___x_1788_);
lean_inc(v___x_1758_);
v___x_1789_ = l_Lean_Syntax_node3(v___y_1704_, v___x_1743_, v___x_1758_, v___x_1786_, v___x_1788_);
v___x_1790_ = l_Lean_Syntax_node1(v___y_1704_, v___y_1709_, v___x_1789_);
v___x_1791_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1732_, v___x_1742_, v___x_1790_);
lean_inc_ref(v___x_1731_);
v___x_1792_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1729_, v___x_1731_, v___x_1791_);
v___x_1793_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1727_, v___y_1711_, v___x_1792_);
v___x_1794_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__78));
v___x_1795_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__79));
v___x_1796_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1796_, 0, v___y_1704_);
lean_ctor_set(v___x_1796_, 1, v___x_1795_);
v___x_1797_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__81));
v___x_1798_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__82));
v___x_1799_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1799_, 0, v___y_1704_);
lean_ctor_set(v___x_1799_, 1, v___x_1798_);
v___x_1800_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__34));
v___x_1801_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__35));
v___x_1802_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1802_, 0, v___y_1704_);
lean_ctor_set(v___x_1802_, 1, v___x_1801_);
v___x_1803_ = l_Lean_Syntax_node1(v___y_1704_, v___x_1800_, v___x_1802_);
v___x_1804_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1804_, 0, v___y_1704_);
lean_ctor_set(v___x_1804_, 1, v___x_1773_);
v___x_1805_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__83));
v___x_1806_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__84, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__84_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__84);
v___x_1807_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__85));
v___x_1808_ = l_Lean_addMacroScope(v___y_1690_, v___x_1807_, v___y_1694_);
v___x_1809_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__86));
v___x_1810_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1810_, 0, v___x_1809_);
lean_ctor_set(v___x_1810_, 1, v___y_1685_);
v___x_1811_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1811_, 0, v___y_1704_);
lean_ctor_set(v___x_1811_, 1, v___x_1806_);
lean_ctor_set(v___x_1811_, 2, v___x_1808_);
lean_ctor_set(v___x_1811_, 3, v___x_1810_);
lean_inc_ref(v___x_1811_);
lean_inc_ref(v___x_1804_);
v___x_1812_ = l_Lean_Syntax_node3(v___y_1704_, v___y_1709_, v___x_1803_, v___x_1804_, v___x_1811_);
v___x_1813_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__87));
v___x_1814_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1814_, 0, v___y_1704_);
lean_ctor_set(v___x_1814_, 1, v___x_1813_);
lean_inc_ref(v___x_1814_);
lean_inc_ref(v___x_1799_);
v___x_1815_ = l_Lean_Syntax_node3(v___y_1704_, v___x_1797_, v___x_1799_, v___x_1812_, v___x_1814_);
v___x_1816_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__90));
v___x_1817_ = l_Lean_Syntax_node2(v___y_1704_, v___x_1816_, v___y_1711_, v___y_1711_);
lean_inc(v___x_1817_);
lean_inc_ref(v___x_1796_);
v___x_1818_ = l_Lean_Syntax_node4(v___y_1704_, v___x_1794_, v___x_1796_, v___x_1815_, v___x_1817_, v___y_1711_);
v___x_1819_ = l_Lean_Syntax_node1(v___y_1704_, v___y_1709_, v___x_1818_);
v___x_1820_ = l_Lean_Syntax_node4(v___y_1704_, v___x_1720_, v___x_1721_, v___x_1726_, v___x_1793_, v___x_1819_);
v___x_1821_ = l_Lean_Syntax_node2(v___y_1704_, v___y_1710_, v___y_1695_, v___x_1820_);
if (lean_obj_tag(v___y_1696_) == 1)
{
lean_object* v_val_1822_; lean_object* v___x_1823_; 
v_val_1822_ = lean_ctor_get(v___y_1696_, 0);
lean_inc(v_val_1822_);
lean_dec_ref_known(v___y_1696_, 1);
v___x_1823_ = l_Array_mkArray1___redArg(v_val_1822_);
v___y_1608_ = v___x_1770_;
v___y_1609_ = v___y_1683_;
v___y_1610_ = v___x_1737_;
v___y_1611_ = v___x_1821_;
v___y_1612_ = v___x_1769_;
v___y_1613_ = v___x_1727_;
v___y_1614_ = v___y_1686_;
v___y_1615_ = v___y_1689_;
v___y_1616_ = v___y_1691_;
v___y_1617_ = v___y_1690_;
v___y_1618_ = v___x_1743_;
v___y_1619_ = v___y_1692_;
v___y_1620_ = v___y_1694_;
v___y_1621_ = v___x_1782_;
v___y_1622_ = v___y_1697_;
v___y_1623_ = v___y_1699_;
v___y_1624_ = v___y_1698_;
v___y_1625_ = v___y_1701_;
v___y_1626_ = v___x_1780_;
v___y_1627_ = v___x_1817_;
v___y_1628_ = v___x_1772_;
v___y_1629_ = v___x_1796_;
v___y_1630_ = v___y_1707_;
v___y_1631_ = v___y_1708_;
v___y_1632_ = v___y_1709_;
v___y_1633_ = v___x_1794_;
v___y_1634_ = v___y_1710_;
v___y_1635_ = v___x_1715_;
v___y_1636_ = v___x_1732_;
v___y_1637_ = v___x_1733_;
v___y_1638_ = v___x_1811_;
v___y_1639_ = v___y_1681_;
v___y_1640_ = v___y_1682_;
v___y_1641_ = v___y_1684_;
v___y_1642_ = v___x_1784_;
v___y_1643_ = v___y_1685_;
v___y_1644_ = v___y_1687_;
v___y_1645_ = v___x_1758_;
v___y_1646_ = v___y_1688_;
v___y_1647_ = v___x_1728_;
v___y_1648_ = v___x_1767_;
v___y_1649_ = v___y_1693_;
v___y_1650_ = v___x_1718_;
v___y_1651_ = v___x_1797_;
v___y_1652_ = v___x_1766_;
v___y_1653_ = v___x_1804_;
v___y_1654_ = v___y_1703_;
v___y_1655_ = v___y_1702_;
v___y_1656_ = v___y_1704_;
v___y_1657_ = v___y_1705_;
v___y_1658_ = v___x_1814_;
v___y_1659_ = v___y_1706_;
v___y_1660_ = v___x_1731_;
v___y_1661_ = v___x_1799_;
v___y_1662_ = v___x_1805_;
v___y_1663_ = v___x_1788_;
v___y_1664_ = v___y_1711_;
v___y_1665_ = v___y_1712_;
v___y_1666_ = v___x_1725_;
v___y_1667_ = v___x_1729_;
v___y_1668_ = v___x_1823_;
goto v___jp_1607_;
}
else
{
lean_object* v___x_1824_; 
lean_dec(v___y_1696_);
v___x_1824_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53));
v___y_1608_ = v___x_1770_;
v___y_1609_ = v___y_1683_;
v___y_1610_ = v___x_1737_;
v___y_1611_ = v___x_1821_;
v___y_1612_ = v___x_1769_;
v___y_1613_ = v___x_1727_;
v___y_1614_ = v___y_1686_;
v___y_1615_ = v___y_1689_;
v___y_1616_ = v___y_1691_;
v___y_1617_ = v___y_1690_;
v___y_1618_ = v___x_1743_;
v___y_1619_ = v___y_1692_;
v___y_1620_ = v___y_1694_;
v___y_1621_ = v___x_1782_;
v___y_1622_ = v___y_1697_;
v___y_1623_ = v___y_1699_;
v___y_1624_ = v___y_1698_;
v___y_1625_ = v___y_1701_;
v___y_1626_ = v___x_1780_;
v___y_1627_ = v___x_1817_;
v___y_1628_ = v___x_1772_;
v___y_1629_ = v___x_1796_;
v___y_1630_ = v___y_1707_;
v___y_1631_ = v___y_1708_;
v___y_1632_ = v___y_1709_;
v___y_1633_ = v___x_1794_;
v___y_1634_ = v___y_1710_;
v___y_1635_ = v___x_1715_;
v___y_1636_ = v___x_1732_;
v___y_1637_ = v___x_1733_;
v___y_1638_ = v___x_1811_;
v___y_1639_ = v___y_1681_;
v___y_1640_ = v___y_1682_;
v___y_1641_ = v___y_1684_;
v___y_1642_ = v___x_1784_;
v___y_1643_ = v___y_1685_;
v___y_1644_ = v___y_1687_;
v___y_1645_ = v___x_1758_;
v___y_1646_ = v___y_1688_;
v___y_1647_ = v___x_1728_;
v___y_1648_ = v___x_1767_;
v___y_1649_ = v___y_1693_;
v___y_1650_ = v___x_1718_;
v___y_1651_ = v___x_1797_;
v___y_1652_ = v___x_1766_;
v___y_1653_ = v___x_1804_;
v___y_1654_ = v___y_1703_;
v___y_1655_ = v___y_1702_;
v___y_1656_ = v___y_1704_;
v___y_1657_ = v___y_1705_;
v___y_1658_ = v___x_1814_;
v___y_1659_ = v___y_1706_;
v___y_1660_ = v___x_1731_;
v___y_1661_ = v___x_1799_;
v___y_1662_ = v___x_1805_;
v___y_1663_ = v___x_1788_;
v___y_1664_ = v___y_1711_;
v___y_1665_ = v___y_1712_;
v___y_1666_ = v___x_1725_;
v___y_1667_ = v___x_1729_;
v___y_1668_ = v___x_1824_;
goto v___jp_1607_;
}
}
v___jp_1825_:
{
lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; 
lean_inc_ref(v___y_1831_);
v___x_1854_ = l_Array_append___redArg(v___y_1831_, v___y_1853_);
lean_dec_ref(v___y_1853_);
lean_inc(v___y_1850_);
lean_inc_n(v___y_1845_, 4);
v___x_1855_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1855_, 0, v___y_1845_);
lean_ctor_set(v___x_1855_, 1, v___y_1850_);
lean_ctor_set(v___x_1855_, 2, v___x_1854_);
lean_inc_ref(v___x_1855_);
lean_inc(v___y_1837_);
lean_inc_n(v___y_1852_, 5);
lean_inc(v___y_1830_);
v___x_1856_ = l_Lean_Syntax_node7(v___y_1845_, v___y_1830_, v___y_1852_, v___y_1852_, v___y_1852_, v___y_1852_, v___y_1837_, v___x_1855_, v___y_1852_);
v___x_1857_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__92));
v___x_1858_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__93));
v___x_1859_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1859_, 0, v___y_1845_);
lean_ctor_set(v___x_1859_, 1, v___x_1858_);
v___x_1860_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__94, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__94_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__94);
v___x_1861_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__95));
lean_inc(v___y_1836_);
lean_inc(v___y_1833_);
v___x_1862_ = l_Lean_addMacroScope(v___y_1833_, v___x_1861_, v___y_1836_);
v___x_1863_ = lean_box(0);
v___x_1864_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1864_, 0, v___y_1845_);
lean_ctor_set(v___x_1864_, 1, v___x_1860_);
lean_ctor_set(v___x_1864_, 2, v___x_1862_);
lean_ctor_set(v___x_1864_, 3, v___x_1863_);
if (lean_obj_tag(v___y_1847_) == 1)
{
lean_object* v_val_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; 
v_val_1865_ = lean_ctor_get(v___y_1847_, 0);
lean_inc(v_val_1865_);
lean_dec_ref_known(v___y_1847_, 1);
v___x_1866_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__74));
lean_inc_n(v___y_1845_, 3);
v___x_1867_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1867_, 0, v___y_1845_);
lean_ctor_set(v___x_1867_, 1, v___x_1866_);
lean_inc_ref(v___y_1831_);
v___x_1868_ = l_Array_append___redArg(v___y_1831_, v_val_1865_);
lean_dec(v_val_1865_);
lean_inc(v___y_1850_);
v___x_1869_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1869_, 0, v___y_1845_);
lean_ctor_set(v___x_1869_, 1, v___y_1850_);
lean_ctor_set(v___x_1869_, 2, v___x_1868_);
v___x_1870_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__76));
v___x_1871_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1871_, 0, v___y_1845_);
lean_ctor_set(v___x_1871_, 1, v___x_1870_);
v___x_1872_ = l_Array_mkArray3___redArg(v___x_1867_, v___x_1869_, v___x_1871_);
v___y_1681_ = v___y_1827_;
v___y_1682_ = v___y_1826_;
v___y_1683_ = v___y_1828_;
v___y_1684_ = v___y_1829_;
v___y_1685_ = v___x_1863_;
v___y_1686_ = v___y_1830_;
v___y_1687_ = v___y_1831_;
v___y_1688_ = v___x_1855_;
v___y_1689_ = v___y_1832_;
v___y_1690_ = v___y_1833_;
v___y_1691_ = v___x_1864_;
v___y_1692_ = v___y_1834_;
v___y_1693_ = v___y_1835_;
v___y_1694_ = v___y_1836_;
v___y_1695_ = v___x_1856_;
v___y_1696_ = v___y_1838_;
v___y_1697_ = v___y_1837_;
v___y_1698_ = v___y_1839_;
v___y_1699_ = v___y_1840_;
v___y_1700_ = v___y_1841_;
v___y_1701_ = v___y_1842_;
v___y_1702_ = v___y_1843_;
v___y_1703_ = v___y_1844_;
v___y_1704_ = v___y_1845_;
v___y_1705_ = v___x_1859_;
v___y_1706_ = v___y_1846_;
v___y_1707_ = v___y_1849_;
v___y_1708_ = v___y_1848_;
v___y_1709_ = v___y_1850_;
v___y_1710_ = v___y_1851_;
v___y_1711_ = v___y_1852_;
v___y_1712_ = v___x_1857_;
v___y_1713_ = v___x_1872_;
goto v___jp_1680_;
}
else
{
lean_object* v___x_1873_; 
lean_dec(v___y_1847_);
v___x_1873_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53));
v___y_1681_ = v___y_1827_;
v___y_1682_ = v___y_1826_;
v___y_1683_ = v___y_1828_;
v___y_1684_ = v___y_1829_;
v___y_1685_ = v___x_1863_;
v___y_1686_ = v___y_1830_;
v___y_1687_ = v___y_1831_;
v___y_1688_ = v___x_1855_;
v___y_1689_ = v___y_1832_;
v___y_1690_ = v___y_1833_;
v___y_1691_ = v___x_1864_;
v___y_1692_ = v___y_1834_;
v___y_1693_ = v___y_1835_;
v___y_1694_ = v___y_1836_;
v___y_1695_ = v___x_1856_;
v___y_1696_ = v___y_1838_;
v___y_1697_ = v___y_1837_;
v___y_1698_ = v___y_1839_;
v___y_1699_ = v___y_1840_;
v___y_1700_ = v___y_1841_;
v___y_1701_ = v___y_1842_;
v___y_1702_ = v___y_1843_;
v___y_1703_ = v___y_1844_;
v___y_1704_ = v___y_1845_;
v___y_1705_ = v___x_1859_;
v___y_1706_ = v___y_1846_;
v___y_1707_ = v___y_1849_;
v___y_1708_ = v___y_1848_;
v___y_1709_ = v___y_1850_;
v___y_1710_ = v___y_1851_;
v___y_1711_ = v___y_1852_;
v___y_1712_ = v___x_1857_;
v___y_1713_ = v___x_1873_;
goto v___jp_1680_;
}
}
v___jp_1874_:
{
lean_object* v___x_1903_; lean_object* v___x_1904_; 
lean_inc_ref(v___y_1880_);
v___x_1903_ = l_Array_append___redArg(v___y_1880_, v___y_1902_);
lean_dec_ref(v___y_1902_);
lean_inc(v___y_1899_);
lean_inc(v___y_1894_);
v___x_1904_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1904_, 0, v___y_1894_);
lean_ctor_set(v___x_1904_, 1, v___y_1899_);
lean_ctor_set(v___x_1904_, 2, v___x_1903_);
if (lean_obj_tag(v___y_1881_) == 1)
{
lean_object* v_val_1905_; lean_object* v___x_1906_; 
v_val_1905_ = lean_ctor_get(v___y_1881_, 0);
lean_inc(v_val_1905_);
lean_dec_ref_known(v___y_1881_, 1);
v___x_1906_ = l_Array_mkArray1___redArg(v_val_1905_);
v___y_1826_ = v___y_1876_;
v___y_1827_ = v___y_1875_;
v___y_1828_ = v___y_1877_;
v___y_1829_ = v___y_1878_;
v___y_1830_ = v___y_1879_;
v___y_1831_ = v___y_1880_;
v___y_1832_ = v___y_1882_;
v___y_1833_ = v___y_1883_;
v___y_1834_ = v___y_1884_;
v___y_1835_ = v___y_1885_;
v___y_1836_ = v___y_1886_;
v___y_1837_ = v___x_1904_;
v___y_1838_ = v___y_1887_;
v___y_1839_ = v___y_1888_;
v___y_1840_ = v___y_1889_;
v___y_1841_ = v___y_1890_;
v___y_1842_ = v___y_1891_;
v___y_1843_ = v___y_1892_;
v___y_1844_ = v___y_1893_;
v___y_1845_ = v___y_1894_;
v___y_1846_ = v___y_1895_;
v___y_1847_ = v___y_1896_;
v___y_1848_ = v___y_1897_;
v___y_1849_ = v___y_1898_;
v___y_1850_ = v___y_1899_;
v___y_1851_ = v___y_1900_;
v___y_1852_ = v___y_1901_;
v___y_1853_ = v___x_1906_;
goto v___jp_1825_;
}
else
{
lean_object* v___x_1907_; 
lean_dec(v___y_1881_);
v___x_1907_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53));
v___y_1826_ = v___y_1876_;
v___y_1827_ = v___y_1875_;
v___y_1828_ = v___y_1877_;
v___y_1829_ = v___y_1878_;
v___y_1830_ = v___y_1879_;
v___y_1831_ = v___y_1880_;
v___y_1832_ = v___y_1882_;
v___y_1833_ = v___y_1883_;
v___y_1834_ = v___y_1884_;
v___y_1835_ = v___y_1885_;
v___y_1836_ = v___y_1886_;
v___y_1837_ = v___x_1904_;
v___y_1838_ = v___y_1887_;
v___y_1839_ = v___y_1888_;
v___y_1840_ = v___y_1889_;
v___y_1841_ = v___y_1890_;
v___y_1842_ = v___y_1891_;
v___y_1843_ = v___y_1892_;
v___y_1844_ = v___y_1893_;
v___y_1845_ = v___y_1894_;
v___y_1846_ = v___y_1895_;
v___y_1847_ = v___y_1896_;
v___y_1848_ = v___y_1897_;
v___y_1849_ = v___y_1898_;
v___y_1850_ = v___y_1899_;
v___y_1851_ = v___y_1900_;
v___y_1852_ = v___y_1901_;
v___y_1853_ = v___x_1907_;
goto v___jp_1825_;
}
}
v___jp_1908_:
{
lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; 
v___x_1928_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__2));
v___x_1929_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__4));
v___x_1930_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_termEta__helper___00__closed__14));
v___x_1931_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__1));
v___x_1932_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_IrreducibleDef_0__Lean_Elab_Command_commandStop__at__first__error_____00__closed__2));
lean_inc_n(v___y_1915_, 2);
v___x_1933_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1933_, 0, v___y_1915_);
lean_ctor_set(v___x_1933_, 1, v___x_1932_);
v___x_1934_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__termVal__proj____1___redArg___closed__23));
v___x_1935_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__97));
v___x_1936_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__98, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__98_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__98);
v___x_1937_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1937_, 0, v___y_1915_);
lean_ctor_set(v___x_1937_, 1, v___x_1934_);
lean_ctor_set(v___x_1937_, 2, v___x_1936_);
if (lean_obj_tag(v___y_1916_) == 1)
{
lean_object* v_val_1938_; lean_object* v___x_1939_; 
v_val_1938_ = lean_ctor_get(v___y_1916_, 0);
lean_inc(v_val_1938_);
lean_dec_ref_known(v___y_1916_, 1);
v___x_1939_ = l_Array_mkArray1___redArg(v_val_1938_);
v___y_1875_ = v___y_1910_;
v___y_1876_ = v___y_1911_;
v___y_1877_ = v___y_1913_;
v___y_1878_ = v___x_1930_;
v___y_1879_ = v___y_1918_;
v___y_1880_ = v___x_1936_;
v___y_1881_ = v___y_1920_;
v___y_1882_ = v___y_1921_;
v___y_1883_ = v_a_1927_;
v___y_1884_ = v___x_1929_;
v___y_1885_ = v___y_1924_;
v___y_1886_ = v___y_1925_;
v___y_1887_ = v___y_1926_;
v___y_1888_ = v___y_1909_;
v___y_1889_ = v___x_1928_;
v___y_1890_ = v___y_1912_;
v___y_1891_ = v___x_1931_;
v___y_1892_ = v___y_1914_;
v___y_1893_ = v___x_1933_;
v___y_1894_ = v___y_1915_;
v___y_1895_ = v___y_1917_;
v___y_1896_ = v___y_1919_;
v___y_1897_ = v___y_1923_;
v___y_1898_ = v___y_1922_;
v___y_1899_ = v___x_1934_;
v___y_1900_ = v___x_1935_;
v___y_1901_ = v___x_1937_;
v___y_1902_ = v___x_1939_;
goto v___jp_1874_;
}
else
{
lean_object* v___x_1940_; 
lean_dec(v___y_1916_);
v___x_1940_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53));
v___y_1875_ = v___y_1910_;
v___y_1876_ = v___y_1911_;
v___y_1877_ = v___y_1913_;
v___y_1878_ = v___x_1930_;
v___y_1879_ = v___y_1918_;
v___y_1880_ = v___x_1936_;
v___y_1881_ = v___y_1920_;
v___y_1882_ = v___y_1921_;
v___y_1883_ = v_a_1927_;
v___y_1884_ = v___x_1929_;
v___y_1885_ = v___y_1924_;
v___y_1886_ = v___y_1925_;
v___y_1887_ = v___y_1926_;
v___y_1888_ = v___y_1909_;
v___y_1889_ = v___x_1928_;
v___y_1890_ = v___y_1912_;
v___y_1891_ = v___x_1931_;
v___y_1892_ = v___y_1914_;
v___y_1893_ = v___x_1933_;
v___y_1894_ = v___y_1915_;
v___y_1895_ = v___y_1917_;
v___y_1896_ = v___y_1919_;
v___y_1897_ = v___y_1923_;
v___y_1898_ = v___y_1922_;
v___y_1899_ = v___x_1934_;
v___y_1900_ = v___x_1935_;
v___y_1901_ = v___x_1937_;
v___y_1902_ = v___x_1940_;
goto v___jp_1874_;
}
}
v___jp_1941_:
{
lean_object* v___x_1958_; 
v___x_1958_ = l_Lean_Elab_Command_getRef___redArg(v___y_1955_);
if (lean_obj_tag(v___x_1958_) == 0)
{
lean_object* v_a_1959_; lean_object* v___x_1960_; 
v_a_1959_ = lean_ctor_get(v___x_1958_, 0);
lean_inc(v_a_1959_);
lean_dec_ref_known(v___x_1958_, 1);
v___x_1960_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_1955_);
if (lean_obj_tag(v___x_1960_) == 0)
{
lean_object* v_a_1961_; lean_object* v_quotContext_x3f_1962_; uint8_t v___x_1963_; lean_object* v___x_1964_; 
v_a_1961_ = lean_ctor_get(v___x_1960_, 0);
lean_inc(v_a_1961_);
lean_dec_ref_known(v___x_1960_, 1);
v_quotContext_x3f_1962_ = lean_ctor_get(v___y_1955_, 5);
v___x_1963_ = 0;
v___x_1964_ = l_Lean_SourceInfo_fromRef(v_a_1959_, v___x_1963_);
lean_dec(v_a_1959_);
if (lean_obj_tag(v_quotContext_x3f_1962_) == 0)
{
lean_object* v___x_1965_; lean_object* v_a_1966_; 
v___x_1965_ = lp_mathlib_Lean_getMainModule___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__1___redArg(v___y_1946_);
v_a_1966_ = lean_ctor_get(v___x_1965_, 0);
lean_inc(v_a_1966_);
lean_dec_ref(v___x_1965_);
v___y_1909_ = v___y_1942_;
v___y_1910_ = v___y_1943_;
v___y_1911_ = v___y_1944_;
v___y_1912_ = v___y_1945_;
v___y_1913_ = v___y_1946_;
v___y_1914_ = v___y_1947_;
v___y_1915_ = v___x_1964_;
v___y_1916_ = v___y_1948_;
v___y_1917_ = v___y_1949_;
v___y_1918_ = v___y_1950_;
v___y_1919_ = v___y_1951_;
v___y_1920_ = v___y_1952_;
v___y_1921_ = v___y_1953_;
v___y_1922_ = v___y_1957_;
v___y_1923_ = v___y_1954_;
v___y_1924_ = v___y_1955_;
v___y_1925_ = v_a_1961_;
v___y_1926_ = v___y_1956_;
v_a_1927_ = v_a_1966_;
goto v___jp_1908_;
}
else
{
lean_object* v_val_1967_; 
v_val_1967_ = lean_ctor_get(v_quotContext_x3f_1962_, 0);
lean_inc(v_val_1967_);
v___y_1909_ = v___y_1942_;
v___y_1910_ = v___y_1943_;
v___y_1911_ = v___y_1944_;
v___y_1912_ = v___y_1945_;
v___y_1913_ = v___y_1946_;
v___y_1914_ = v___y_1947_;
v___y_1915_ = v___x_1964_;
v___y_1916_ = v___y_1948_;
v___y_1917_ = v___y_1949_;
v___y_1918_ = v___y_1950_;
v___y_1919_ = v___y_1951_;
v___y_1920_ = v___y_1952_;
v___y_1921_ = v___y_1953_;
v___y_1922_ = v___y_1957_;
v___y_1923_ = v___y_1954_;
v___y_1924_ = v___y_1955_;
v___y_1925_ = v_a_1961_;
v___y_1926_ = v___y_1956_;
v_a_1927_ = v_val_1967_;
goto v___jp_1908_;
}
}
else
{
lean_object* v_a_1968_; lean_object* v___x_1970_; uint8_t v_isShared_1971_; uint8_t v_isSharedCheck_1975_; 
lean_dec(v_a_1959_);
lean_dec(v___y_1957_);
lean_dec(v___y_1956_);
lean_dec(v___y_1954_);
lean_dec(v___y_1953_);
lean_dec(v___y_1952_);
lean_dec(v___y_1951_);
lean_dec(v___y_1948_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1945_);
lean_dec_ref(v___y_1943_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v_a_1968_ = lean_ctor_get(v___x_1960_, 0);
v_isSharedCheck_1975_ = !lean_is_exclusive(v___x_1960_);
if (v_isSharedCheck_1975_ == 0)
{
v___x_1970_ = v___x_1960_;
v_isShared_1971_ = v_isSharedCheck_1975_;
goto v_resetjp_1969_;
}
else
{
lean_inc(v_a_1968_);
lean_dec(v___x_1960_);
v___x_1970_ = lean_box(0);
v_isShared_1971_ = v_isSharedCheck_1975_;
goto v_resetjp_1969_;
}
v_resetjp_1969_:
{
lean_object* v___x_1973_; 
if (v_isShared_1971_ == 0)
{
v___x_1973_ = v___x_1970_;
goto v_reusejp_1972_;
}
else
{
lean_object* v_reuseFailAlloc_1974_; 
v_reuseFailAlloc_1974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1974_, 0, v_a_1968_);
v___x_1973_ = v_reuseFailAlloc_1974_;
goto v_reusejp_1972_;
}
v_reusejp_1972_:
{
return v___x_1973_;
}
}
}
}
else
{
lean_object* v_a_1976_; lean_object* v___x_1978_; uint8_t v_isShared_1979_; uint8_t v_isSharedCheck_1983_; 
lean_dec(v___y_1957_);
lean_dec(v___y_1956_);
lean_dec(v___y_1954_);
lean_dec(v___y_1953_);
lean_dec(v___y_1952_);
lean_dec(v___y_1951_);
lean_dec(v___y_1948_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1945_);
lean_dec_ref(v___y_1943_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v_a_1976_ = lean_ctor_get(v___x_1958_, 0);
v_isSharedCheck_1983_ = !lean_is_exclusive(v___x_1958_);
if (v_isSharedCheck_1983_ == 0)
{
v___x_1978_ = v___x_1958_;
v_isShared_1979_ = v_isSharedCheck_1983_;
goto v_resetjp_1977_;
}
else
{
lean_inc(v_a_1976_);
lean_dec(v___x_1958_);
v___x_1978_ = lean_box(0);
v_isShared_1979_ = v_isSharedCheck_1983_;
goto v_resetjp_1977_;
}
v_resetjp_1977_:
{
lean_object* v___x_1981_; 
if (v_isShared_1979_ == 0)
{
v___x_1981_ = v___x_1978_;
goto v_reusejp_1980_;
}
else
{
lean_object* v_reuseFailAlloc_1982_; 
v_reuseFailAlloc_1982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1982_, 0, v_a_1976_);
v___x_1981_ = v_reuseFailAlloc_1982_;
goto v_reusejp_1980_;
}
v_reusejp_1980_:
{
return v___x_1981_;
}
}
}
}
v___jp_1984_:
{
if (lean_obj_tag(v___y_1985_) == 0)
{
lean_object* v___x_2001_; 
v___x_2001_ = lean_box(0);
v___y_1942_ = v___y_1986_;
v___y_1943_ = v___y_2000_;
v___y_1944_ = v___y_1987_;
v___y_1945_ = v___y_1988_;
v___y_1946_ = v___y_1989_;
v___y_1947_ = v___y_1990_;
v___y_1948_ = v___y_1991_;
v___y_1949_ = v___y_1992_;
v___y_1950_ = v___y_1993_;
v___y_1951_ = v___y_1994_;
v___y_1952_ = v___y_1995_;
v___y_1953_ = v___y_1996_;
v___y_1954_ = v___y_1997_;
v___y_1955_ = v___y_1998_;
v___y_1956_ = v___y_1999_;
v___y_1957_ = v___x_2001_;
goto v___jp_1941_;
}
else
{
lean_object* v_val_2002_; lean_object* v___x_2004_; uint8_t v_isShared_2005_; uint8_t v_isSharedCheck_2012_; 
v_val_2002_ = lean_ctor_get(v___y_1985_, 0);
v_isSharedCheck_2012_ = !lean_is_exclusive(v___y_1985_);
if (v_isSharedCheck_2012_ == 0)
{
v___x_2004_ = v___y_1985_;
v_isShared_2005_ = v_isSharedCheck_2012_;
goto v_resetjp_2003_;
}
else
{
lean_inc(v_val_2002_);
lean_dec(v___y_1985_);
v___x_2004_ = lean_box(0);
v_isShared_2005_ = v_isSharedCheck_2012_;
goto v_resetjp_2003_;
}
v_resetjp_2003_:
{
lean_object* v___x_2007_; 
lean_inc(v_val_2002_);
if (v_isShared_2005_ == 0)
{
v___x_2007_ = v___x_2004_;
goto v_reusejp_2006_;
}
else
{
lean_object* v_reuseFailAlloc_2011_; 
v_reuseFailAlloc_2011_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2011_, 0, v_val_2002_);
v___x_2007_ = v_reuseFailAlloc_2011_;
goto v_reusejp_2006_;
}
v_reusejp_2006_:
{
lean_object* v___x_2008_; uint8_t v___x_2009_; 
v___x_2008_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__52));
v___x_2009_ = l_Lean_Syntax_isOfKind(v_val_2002_, v___x_2008_);
if (v___x_2009_ == 0)
{
if (v___x_2009_ == 0)
{
lean_object* v___x_2010_; 
lean_dec_ref(v___x_2007_);
v___x_2010_ = lean_box(0);
v___y_1942_ = v___y_1986_;
v___y_1943_ = v___y_2000_;
v___y_1944_ = v___y_1987_;
v___y_1945_ = v___y_1988_;
v___y_1946_ = v___y_1989_;
v___y_1947_ = v___y_1990_;
v___y_1948_ = v___y_1991_;
v___y_1949_ = v___y_1992_;
v___y_1950_ = v___y_1993_;
v___y_1951_ = v___y_1994_;
v___y_1952_ = v___y_1995_;
v___y_1953_ = v___y_1996_;
v___y_1954_ = v___y_1997_;
v___y_1955_ = v___y_1998_;
v___y_1956_ = v___y_1999_;
v___y_1957_ = v___x_2010_;
goto v___jp_1941_;
}
else
{
v___y_1942_ = v___y_1986_;
v___y_1943_ = v___y_2000_;
v___y_1944_ = v___y_1987_;
v___y_1945_ = v___y_1988_;
v___y_1946_ = v___y_1989_;
v___y_1947_ = v___y_1990_;
v___y_1948_ = v___y_1991_;
v___y_1949_ = v___y_1992_;
v___y_1950_ = v___y_1993_;
v___y_1951_ = v___y_1994_;
v___y_1952_ = v___y_1995_;
v___y_1953_ = v___y_1996_;
v___y_1954_ = v___y_1997_;
v___y_1955_ = v___y_1998_;
v___y_1956_ = v___y_1999_;
v___y_1957_ = v___x_2007_;
goto v___jp_1941_;
}
}
else
{
v___y_1942_ = v___y_1986_;
v___y_1943_ = v___y_2000_;
v___y_1944_ = v___y_1987_;
v___y_1945_ = v___y_1988_;
v___y_1946_ = v___y_1989_;
v___y_1947_ = v___y_1990_;
v___y_1948_ = v___y_1991_;
v___y_1949_ = v___y_1992_;
v___y_1950_ = v___y_1993_;
v___y_1951_ = v___y_1994_;
v___y_1952_ = v___y_1995_;
v___y_1953_ = v___y_1996_;
v___y_1954_ = v___y_1997_;
v___y_1955_ = v___y_1998_;
v___y_1956_ = v___y_1999_;
v___y_1957_ = v___x_2007_;
goto v___jp_1941_;
}
}
}
}
}
v___jp_2013_:
{
lean_object* v___x_2029_; lean_object* v___x_2030_; uint8_t v___x_2031_; 
v___x_2029_ = lean_unsigned_to_nat(6u);
v___x_2030_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_2029_);
v___x_2031_ = l_Lean_Syntax_matchesNull(v___x_2030_, v___x_1332_);
if (v___x_2031_ == 0)
{
lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; 
lean_dec(v_uns_2026_);
lean_dec(v___y_2025_);
lean_dec(v___y_2024_);
lean_dec(v___y_2022_);
lean_dec(v___y_2021_);
lean_dec(v___y_2020_);
lean_dec(v___y_2019_);
lean_dec_ref(v___y_2017_);
lean_dec(v___y_2015_);
lean_dec(v___y_2014_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2032_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2033_ = lean_box(0);
v___x_2034_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2033_, v___x_2031_);
v___x_2035_ = l_Lean_MessageData_ofFormat(v___x_2034_);
v___x_2036_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2036_, 0, v___x_2032_);
lean_ctor_set(v___x_2036_, 1, v___x_2035_);
v___x_2037_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2036_, v___y_2027_, v___y_2028_);
return v___x_2037_;
}
else
{
lean_dec(v___x_1333_);
if (lean_obj_tag(v___y_2021_) == 0)
{
lean_object* v___x_2038_; 
v___x_2038_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__53));
v___y_1985_ = v___y_2014_;
v___y_1986_ = v___y_2016_;
v___y_1987_ = v___y_2018_;
v___y_1988_ = v___y_2017_;
v___y_1989_ = v___y_2028_;
v___y_1990_ = v___y_2019_;
v___y_1991_ = v___y_2022_;
v___y_1992_ = v___x_2031_;
v___y_1993_ = v___y_2023_;
v___y_1994_ = v___y_2024_;
v___y_1995_ = v_uns_2026_;
v___y_1996_ = v___y_2015_;
v___y_1997_ = v___y_2020_;
v___y_1998_ = v___y_2027_;
v___y_1999_ = v___y_2025_;
v___y_2000_ = v___x_2038_;
goto v___jp_1984_;
}
else
{
lean_object* v_val_2039_; 
v_val_2039_ = lean_ctor_get(v___y_2021_, 0);
lean_inc(v_val_2039_);
lean_dec_ref_known(v___y_2021_, 1);
v___y_1985_ = v___y_2014_;
v___y_1986_ = v___y_2016_;
v___y_1987_ = v___y_2018_;
v___y_1988_ = v___y_2017_;
v___y_1989_ = v___y_2028_;
v___y_1990_ = v___y_2019_;
v___y_1991_ = v___y_2022_;
v___y_1992_ = v___x_2031_;
v___y_1993_ = v___y_2023_;
v___y_1994_ = v___y_2024_;
v___y_1995_ = v_uns_2026_;
v___y_1996_ = v___y_2015_;
v___y_1997_ = v___y_2020_;
v___y_1998_ = v___y_2027_;
v___y_1999_ = v___y_2025_;
v___y_2000_ = v_val_2039_;
goto v___jp_1984_;
}
}
}
v___jp_2040_:
{
lean_object* v___x_2055_; uint8_t v___x_2056_; 
v___x_2055_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_1341_);
v___x_2056_ = l_Lean_Syntax_isNone(v___x_2055_);
if (v___x_2056_ == 0)
{
uint8_t v___x_2057_; 
lean_inc(v___x_2055_);
v___x_2057_ = l_Lean_Syntax_matchesNull(v___x_2055_, v___x_1334_);
if (v___x_2057_ == 0)
{
lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; 
lean_dec(v___x_2055_);
lean_dec(v_nc_2052_);
lean_dec(v___y_2051_);
lean_dec(v___y_2050_);
lean_dec(v___y_2048_);
lean_dec(v___y_2047_);
lean_dec(v___y_2046_);
lean_dec_ref(v___y_2045_);
lean_dec(v___y_2043_);
lean_dec(v___y_2041_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2058_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2059_ = lean_box(0);
v___x_2060_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2059_, v___x_2057_);
v___x_2061_ = l_Lean_MessageData_ofFormat(v___x_2060_);
v___x_2062_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2058_);
lean_ctor_set(v___x_2062_, 1, v___x_2061_);
v___x_2063_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2062_, v___y_2053_, v___y_2054_);
return v___x_2063_;
}
else
{
lean_object* v_uns_2064_; lean_object* v___x_2065_; uint8_t v___x_2066_; 
v_uns_2064_ = l_Lean_Syntax_getArg(v___x_2055_, v___x_1332_);
lean_dec(v___x_2055_);
v___x_2065_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__102));
lean_inc(v_uns_2064_);
v___x_2066_ = l_Lean_Syntax_isOfKind(v_uns_2064_, v___x_2065_);
if (v___x_2066_ == 0)
{
lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; 
lean_dec(v_uns_2064_);
lean_dec(v_nc_2052_);
lean_dec(v___y_2051_);
lean_dec(v___y_2050_);
lean_dec(v___y_2048_);
lean_dec(v___y_2047_);
lean_dec(v___y_2046_);
lean_dec_ref(v___y_2045_);
lean_dec(v___y_2043_);
lean_dec(v___y_2041_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2067_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2068_ = lean_box(0);
v___x_2069_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2068_, v___x_2066_);
v___x_2070_ = l_Lean_MessageData_ofFormat(v___x_2069_);
v___x_2071_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2071_, 0, v___x_2067_);
lean_ctor_set(v___x_2071_, 1, v___x_2070_);
v___x_2072_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2071_, v___y_2053_, v___y_2054_);
return v___x_2072_;
}
else
{
lean_object* v___x_2073_; 
v___x_2073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2073_, 0, v_uns_2064_);
v___y_2014_ = v___y_2041_;
v___y_2015_ = v___y_2043_;
v___y_2016_ = v___y_2042_;
v___y_2017_ = v___y_2045_;
v___y_2018_ = v___y_2044_;
v___y_2019_ = v___y_2046_;
v___y_2020_ = v___y_2047_;
v___y_2021_ = v___y_2048_;
v___y_2022_ = v_nc_2052_;
v___y_2023_ = v___y_2049_;
v___y_2024_ = v___y_2050_;
v___y_2025_ = v___y_2051_;
v_uns_2026_ = v___x_2073_;
v___y_2027_ = v___y_2053_;
v___y_2028_ = v___y_2054_;
goto v___jp_2013_;
}
}
}
else
{
lean_object* v___x_2074_; 
lean_dec(v___x_2055_);
v___x_2074_ = lean_box(0);
v___y_2014_ = v___y_2041_;
v___y_2015_ = v___y_2043_;
v___y_2016_ = v___y_2042_;
v___y_2017_ = v___y_2045_;
v___y_2018_ = v___y_2044_;
v___y_2019_ = v___y_2046_;
v___y_2020_ = v___y_2047_;
v___y_2021_ = v___y_2048_;
v___y_2022_ = v_nc_2052_;
v___y_2023_ = v___y_2049_;
v___y_2024_ = v___y_2050_;
v___y_2025_ = v___y_2051_;
v_uns_2026_ = v___x_2074_;
v___y_2027_ = v___y_2053_;
v___y_2028_ = v___y_2054_;
goto v___jp_2013_;
}
}
v___jp_2075_:
{
lean_object* v___x_2089_; uint8_t v___x_2090_; 
v___x_2089_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_1339_);
v___x_2090_ = l_Lean_Syntax_isNone(v___x_2089_);
if (v___x_2090_ == 0)
{
uint8_t v___x_2091_; 
lean_inc(v___x_2089_);
v___x_2091_ = l_Lean_Syntax_matchesNull(v___x_2089_, v___x_1334_);
if (v___x_2091_ == 0)
{
lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; 
lean_dec(v___x_2089_);
lean_dec(v_prot_2086_);
lean_dec(v___y_2085_);
lean_dec(v___y_2084_);
lean_dec(v___y_2082_);
lean_dec(v___y_2081_);
lean_dec(v___y_2080_);
lean_dec_ref(v___y_2078_);
lean_dec(v___y_2076_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2092_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2093_ = lean_box(0);
v___x_2094_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2093_, v___x_2091_);
v___x_2095_ = l_Lean_MessageData_ofFormat(v___x_2094_);
v___x_2096_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2096_, 0, v___x_2092_);
lean_ctor_set(v___x_2096_, 1, v___x_2095_);
v___x_2097_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2096_, v___y_2087_, v___y_2088_);
return v___x_2097_;
}
else
{
lean_object* v_nc_2098_; lean_object* v___x_2099_; uint8_t v___x_2100_; 
v_nc_2098_ = l_Lean_Syntax_getArg(v___x_2089_, v___x_1332_);
lean_dec(v___x_2089_);
v___x_2099_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__104));
lean_inc(v_nc_2098_);
v___x_2100_ = l_Lean_Syntax_isOfKind(v_nc_2098_, v___x_2099_);
if (v___x_2100_ == 0)
{
lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; 
lean_dec(v_nc_2098_);
lean_dec(v_prot_2086_);
lean_dec(v___y_2085_);
lean_dec(v___y_2084_);
lean_dec(v___y_2082_);
lean_dec(v___y_2081_);
lean_dec(v___y_2080_);
lean_dec_ref(v___y_2078_);
lean_dec(v___y_2076_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2101_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2102_ = lean_box(0);
v___x_2103_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2102_, v___x_2100_);
v___x_2104_ = l_Lean_MessageData_ofFormat(v___x_2103_);
v___x_2105_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2105_, 0, v___x_2101_);
lean_ctor_set(v___x_2105_, 1, v___x_2104_);
v___x_2106_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2105_, v___y_2087_, v___y_2088_);
return v___x_2106_;
}
else
{
lean_object* v___x_2107_; 
v___x_2107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2107_, 0, v_nc_2098_);
v___y_2041_ = v___y_2076_;
v___y_2042_ = v___y_2077_;
v___y_2043_ = v_prot_2086_;
v___y_2044_ = v___y_2079_;
v___y_2045_ = v___y_2078_;
v___y_2046_ = v___y_2080_;
v___y_2047_ = v___y_2081_;
v___y_2048_ = v___y_2082_;
v___y_2049_ = v___y_2083_;
v___y_2050_ = v___y_2084_;
v___y_2051_ = v___y_2085_;
v_nc_2052_ = v___x_2107_;
v___y_2053_ = v___y_2087_;
v___y_2054_ = v___y_2088_;
goto v___jp_2040_;
}
}
}
else
{
lean_object* v___x_2108_; 
lean_dec(v___x_2089_);
v___x_2108_ = lean_box(0);
v___y_2041_ = v___y_2076_;
v___y_2042_ = v___y_2077_;
v___y_2043_ = v_prot_2086_;
v___y_2044_ = v___y_2079_;
v___y_2045_ = v___y_2078_;
v___y_2046_ = v___y_2080_;
v___y_2047_ = v___y_2081_;
v___y_2048_ = v___y_2082_;
v___y_2049_ = v___y_2083_;
v___y_2050_ = v___y_2084_;
v___y_2051_ = v___y_2085_;
v_nc_2052_ = v___x_2108_;
v___y_2053_ = v___y_2087_;
v___y_2054_ = v___y_2088_;
goto v___jp_2040_;
}
}
v___jp_2109_:
{
lean_object* v___x_2122_; uint8_t v___x_2123_; 
v___x_2122_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_1337_);
v___x_2123_ = l_Lean_Syntax_isNone(v___x_2122_);
if (v___x_2123_ == 0)
{
uint8_t v___x_2124_; 
lean_inc(v___x_2122_);
v___x_2124_ = l_Lean_Syntax_matchesNull(v___x_2122_, v___x_1334_);
if (v___x_2124_ == 0)
{
lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; 
lean_dec(v___x_2122_);
lean_dec(v_vis_2119_);
lean_dec(v___y_2118_);
lean_dec(v___y_2117_);
lean_dec(v___y_2115_);
lean_dec(v___y_2114_);
lean_dec(v___y_2113_);
lean_dec_ref(v___y_2112_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2125_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2126_ = lean_box(0);
v___x_2127_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2126_, v___x_2124_);
v___x_2128_ = l_Lean_MessageData_ofFormat(v___x_2127_);
v___x_2129_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2129_, 0, v___x_2125_);
lean_ctor_set(v___x_2129_, 1, v___x_2128_);
v___x_2130_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2129_, v___y_2120_, v___y_2121_);
return v___x_2130_;
}
else
{
lean_object* v_prot_2131_; lean_object* v___x_2132_; uint8_t v___x_2133_; 
v_prot_2131_ = l_Lean_Syntax_getArg(v___x_2122_, v___x_1332_);
lean_dec(v___x_2122_);
v___x_2132_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__106));
lean_inc(v_prot_2131_);
v___x_2133_ = l_Lean_Syntax_isOfKind(v_prot_2131_, v___x_2132_);
if (v___x_2133_ == 0)
{
lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; 
lean_dec(v_prot_2131_);
lean_dec(v_vis_2119_);
lean_dec(v___y_2118_);
lean_dec(v___y_2117_);
lean_dec(v___y_2115_);
lean_dec(v___y_2114_);
lean_dec(v___y_2113_);
lean_dec_ref(v___y_2112_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2134_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2135_ = lean_box(0);
v___x_2136_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2135_, v___x_2133_);
v___x_2137_ = l_Lean_MessageData_ofFormat(v___x_2136_);
v___x_2138_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2138_, 0, v___x_2134_);
lean_ctor_set(v___x_2138_, 1, v___x_2137_);
v___x_2139_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2138_, v___y_2120_, v___y_2121_);
return v___x_2139_;
}
else
{
lean_object* v___x_2140_; 
v___x_2140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2140_, 0, v_prot_2131_);
v___y_2076_ = v_vis_2119_;
v___y_2077_ = v___y_2110_;
v___y_2078_ = v___y_2112_;
v___y_2079_ = v___y_2111_;
v___y_2080_ = v___y_2113_;
v___y_2081_ = v___y_2114_;
v___y_2082_ = v___y_2115_;
v___y_2083_ = v___y_2116_;
v___y_2084_ = v___y_2117_;
v___y_2085_ = v___y_2118_;
v_prot_2086_ = v___x_2140_;
v___y_2087_ = v___y_2120_;
v___y_2088_ = v___y_2121_;
goto v___jp_2075_;
}
}
}
else
{
lean_object* v___x_2141_; 
lean_dec(v___x_2122_);
v___x_2141_ = lean_box(0);
v___y_2076_ = v_vis_2119_;
v___y_2077_ = v___y_2110_;
v___y_2078_ = v___y_2112_;
v___y_2079_ = v___y_2111_;
v___y_2080_ = v___y_2113_;
v___y_2081_ = v___y_2114_;
v___y_2082_ = v___y_2115_;
v___y_2083_ = v___y_2116_;
v___y_2084_ = v___y_2117_;
v___y_2085_ = v___y_2118_;
v_prot_2086_ = v___x_2141_;
v___y_2087_ = v___y_2120_;
v___y_2088_ = v___y_2121_;
goto v___jp_2075_;
}
}
v___jp_2142_:
{
lean_object* v___x_2154_; uint8_t v___x_2155_; 
v___x_2154_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_1335_);
v___x_2155_ = l_Lean_Syntax_isNone(v___x_2154_);
if (v___x_2155_ == 0)
{
uint8_t v___x_2156_; 
lean_inc(v___x_2154_);
v___x_2156_ = l_Lean_Syntax_matchesNull(v___x_2154_, v___x_1334_);
if (v___x_2156_ == 0)
{
lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; 
lean_dec(v___x_2154_);
lean_dec(v_attrs_2151_);
lean_dec(v___y_2150_);
lean_dec(v___y_2149_);
lean_dec(v___y_2147_);
lean_dec(v___y_2146_);
lean_dec_ref(v___y_2144_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2157_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2158_ = lean_box(0);
v___x_2159_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2158_, v___x_2156_);
v___x_2160_ = l_Lean_MessageData_ofFormat(v___x_2159_);
v___x_2161_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2161_, 0, v___x_2157_);
lean_ctor_set(v___x_2161_, 1, v___x_2160_);
v___x_2162_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2161_, v___y_2152_, v___y_2153_);
return v___x_2162_;
}
else
{
lean_object* v_vis_2163_; lean_object* v___x_2164_; 
v_vis_2163_ = l_Lean_Syntax_getArg(v___x_2154_, v___x_1332_);
lean_dec(v___x_2154_);
v___x_2164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2164_, 0, v_vis_2163_);
v___y_2110_ = v___y_2143_;
v___y_2111_ = v___y_2145_;
v___y_2112_ = v___y_2144_;
v___y_2113_ = v___y_2146_;
v___y_2114_ = v___y_2147_;
v___y_2115_ = v_attrs_2151_;
v___y_2116_ = v___y_2148_;
v___y_2117_ = v___y_2149_;
v___y_2118_ = v___y_2150_;
v_vis_2119_ = v___x_2164_;
v___y_2120_ = v___y_2152_;
v___y_2121_ = v___y_2153_;
goto v___jp_2109_;
}
}
else
{
lean_object* v___x_2165_; 
lean_dec(v___x_2154_);
v___x_2165_ = lean_box(0);
v___y_2110_ = v___y_2143_;
v___y_2111_ = v___y_2145_;
v___y_2112_ = v___y_2144_;
v___y_2113_ = v___y_2146_;
v___y_2114_ = v___y_2147_;
v___y_2115_ = v_attrs_2151_;
v___y_2116_ = v___y_2148_;
v___y_2117_ = v___y_2149_;
v___y_2118_ = v___y_2150_;
v_vis_2119_ = v___x_2165_;
v___y_2120_ = v___y_2152_;
v___y_2121_ = v___y_2153_;
goto v___jp_2109_;
}
}
v___jp_2166_:
{
lean_object* v___x_2177_; uint8_t v___x_2178_; 
v___x_2177_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_1334_);
v___x_2178_ = l_Lean_Syntax_isNone(v___x_2177_);
if (v___x_2178_ == 0)
{
uint8_t v___x_2179_; 
lean_inc(v___x_2177_);
v___x_2179_ = l_Lean_Syntax_matchesNull(v___x_2177_, v___x_1334_);
if (v___x_2179_ == 0)
{
lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; 
lean_dec(v___x_2177_);
lean_dec(v_doc_2174_);
lean_dec(v___y_2173_);
lean_dec(v___y_2171_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2180_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2181_ = lean_box(0);
v___x_2182_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2181_, v___x_2179_);
v___x_2183_ = l_Lean_MessageData_ofFormat(v___x_2182_);
v___x_2184_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2184_, 0, v___x_2180_);
lean_ctor_set(v___x_2184_, 1, v___x_2183_);
v___x_2185_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2184_, v___y_2175_, v___y_2176_);
return v___x_2185_;
}
else
{
lean_object* v___x_2186_; lean_object* v___x_2187_; uint8_t v___x_2188_; 
v___x_2186_ = l_Lean_Syntax_getArg(v___x_2177_, v___x_1332_);
lean_dec(v___x_2177_);
v___x_2187_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__108));
lean_inc(v___x_2186_);
v___x_2188_ = l_Lean_Syntax_isOfKind(v___x_2186_, v___x_2187_);
if (v___x_2188_ == 0)
{
lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; 
lean_dec(v___x_2186_);
lean_dec(v_doc_2174_);
lean_dec(v___y_2173_);
lean_dec(v___y_2171_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2189_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2190_ = lean_box(0);
v___x_2191_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2190_, v___x_2188_);
v___x_2192_ = l_Lean_MessageData_ofFormat(v___x_2191_);
v___x_2193_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2193_, 0, v___x_2189_);
lean_ctor_set(v___x_2193_, 1, v___x_2192_);
v___x_2194_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2193_, v___y_2175_, v___y_2176_);
return v___x_2194_;
}
else
{
lean_object* v___x_2195_; lean_object* v_attrs_2196_; lean_object* v___x_2197_; 
v___x_2195_ = l_Lean_Syntax_getArg(v___x_2186_, v___x_1334_);
lean_dec(v___x_2186_);
v_attrs_2196_ = l_Lean_Syntax_getArgs(v___x_2195_);
lean_dec(v___x_2195_);
v___x_2197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2197_, 0, v_attrs_2196_);
v___y_2143_ = v___y_2167_;
v___y_2144_ = v___y_2169_;
v___y_2145_ = v___y_2168_;
v___y_2146_ = v___y_2170_;
v___y_2147_ = v___y_2171_;
v___y_2148_ = v___y_2172_;
v___y_2149_ = v___y_2173_;
v___y_2150_ = v_doc_2174_;
v_attrs_2151_ = v___x_2197_;
v___y_2152_ = v___y_2175_;
v___y_2153_ = v___y_2176_;
goto v___jp_2142_;
}
}
}
else
{
lean_object* v___x_2198_; 
lean_dec(v___x_2177_);
v___x_2198_ = lean_box(0);
v___y_2143_ = v___y_2167_;
v___y_2144_ = v___y_2169_;
v___y_2145_ = v___y_2168_;
v___y_2146_ = v___y_2170_;
v___y_2147_ = v___y_2171_;
v___y_2148_ = v___y_2172_;
v___y_2149_ = v___y_2173_;
v___y_2150_ = v_doc_2174_;
v_attrs_2151_ = v___x_2198_;
v___y_2152_ = v___y_2175_;
v___y_2153_ = v___y_2176_;
goto v___jp_2142_;
}
}
v___jp_2199_:
{
uint8_t v___x_2209_; 
lean_inc(v___x_1333_);
v___x_2209_ = l_Lean_Syntax_isOfKind(v___x_1333_, v___y_2204_);
if (v___x_2209_ == 0)
{
lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; 
lean_dec(v_n__def_2206_);
lean_dec(v___y_2205_);
lean_dec(v___y_2203_);
lean_dec_ref(v___y_2201_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2210_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2211_ = lean_box(0);
v___x_2212_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2211_, v___x_2209_);
v___x_2213_ = l_Lean_MessageData_ofFormat(v___x_2212_);
v___x_2214_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2214_, 0, v___x_2210_);
lean_ctor_set(v___x_2214_, 1, v___x_2213_);
v___x_2215_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2214_, v___y_2207_, v___y_2208_);
return v___x_2215_;
}
else
{
lean_object* v___x_2216_; uint8_t v___x_2217_; 
v___x_2216_ = l_Lean_Syntax_getArg(v___x_1333_, v___x_1332_);
v___x_2217_ = l_Lean_Syntax_isNone(v___x_2216_);
if (v___x_2217_ == 0)
{
uint8_t v___x_2218_; 
lean_inc(v___x_2216_);
v___x_2218_ = l_Lean_Syntax_matchesNull(v___x_2216_, v___x_1334_);
if (v___x_2218_ == 0)
{
lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; 
lean_dec(v___x_2216_);
lean_dec(v_n__def_2206_);
lean_dec(v___y_2205_);
lean_dec(v___y_2203_);
lean_dec_ref(v___y_2201_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2219_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2220_ = lean_box(0);
v___x_2221_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2220_, v___x_2218_);
v___x_2222_ = l_Lean_MessageData_ofFormat(v___x_2221_);
v___x_2223_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2223_, 0, v___x_2219_);
lean_ctor_set(v___x_2223_, 1, v___x_2222_);
v___x_2224_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2223_, v___y_2207_, v___y_2208_);
return v___x_2224_;
}
else
{
lean_object* v_doc_2225_; lean_object* v___x_2226_; uint8_t v___x_2227_; 
v_doc_2225_ = l_Lean_Syntax_getArg(v___x_2216_, v___x_1332_);
lean_dec(v___x_2216_);
v___x_2226_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__110));
lean_inc(v_doc_2225_);
v___x_2227_ = l_Lean_Syntax_isOfKind(v_doc_2225_, v___x_2226_);
if (v___x_2227_ == 0)
{
lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; 
lean_dec(v_doc_2225_);
lean_dec(v_n__def_2206_);
lean_dec(v___y_2205_);
lean_dec(v___y_2203_);
lean_dec_ref(v___y_2201_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
v___x_2228_ = lean_obj_once(&lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100, &lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100_once, _init_lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__100);
v___x_2229_ = lean_box(0);
v___x_2230_ = l_Lean_Syntax_formatStx(v___x_1333_, v___x_2229_, v___x_2227_);
v___x_2231_ = l_Lean_MessageData_ofFormat(v___x_2230_);
v___x_2232_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2232_, 0, v___x_2228_);
lean_ctor_set(v___x_2232_, 1, v___x_2231_);
v___x_2233_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v___x_2232_, v___y_2207_, v___y_2208_);
return v___x_2233_;
}
else
{
lean_object* v___x_2234_; 
v___x_2234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2234_, 0, v_doc_2225_);
v___y_2167_ = v___y_2200_;
v___y_2168_ = v___y_2202_;
v___y_2169_ = v___y_2201_;
v___y_2170_ = v_n__def_2206_;
v___y_2171_ = v___y_2203_;
v___y_2172_ = v___y_2204_;
v___y_2173_ = v___y_2205_;
v_doc_2174_ = v___x_2234_;
v___y_2175_ = v___y_2207_;
v___y_2176_ = v___y_2208_;
goto v___jp_2166_;
}
}
}
else
{
lean_object* v___x_2235_; 
lean_dec(v___x_2216_);
v___x_2235_ = lean_box(0);
v___y_2167_ = v___y_2200_;
v___y_2168_ = v___y_2202_;
v___y_2169_ = v___y_2201_;
v___y_2170_ = v_n__def_2206_;
v___y_2171_ = v___y_2203_;
v___y_2172_ = v___y_2204_;
v___y_2173_ = v___y_2205_;
v_doc_2174_ = v___x_2235_;
v___y_2175_ = v___y_2207_;
v___y_2176_ = v___y_2208_;
goto v___jp_2166_;
}
}
}
v___jp_2236_:
{
lean_object* v___x_2246_; uint8_t v___x_2247_; 
v___x_2246_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__1));
lean_inc(v___y_2245_);
v___x_2247_ = l_Lean_Syntax_isOfKind(v___y_2245_, v___x_2246_);
if (v___x_2247_ == 0)
{
lean_object* v___x_2248_; lean_object* v_scopes_2249_; lean_object* v_name_2250_; lean_object* v_imported_2251_; lean_object* v_ctx_2252_; lean_object* v_scopes_2253_; lean_object* v___x_2255_; uint8_t v_isShared_2256_; uint8_t v_isSharedCheck_2264_; 
lean_dec(v___y_2245_);
v___x_2248_ = l_Lean_TSyntax_getId(v___y_2242_);
v_scopes_2249_ = l_Lean_extractMacroScopes(v___x_2248_);
v_name_2250_ = lean_ctor_get(v_scopes_2249_, 0);
v_imported_2251_ = lean_ctor_get(v_scopes_2249_, 1);
v_ctx_2252_ = lean_ctor_get(v_scopes_2249_, 2);
v_scopes_2253_ = lean_ctor_get(v_scopes_2249_, 3);
v_isSharedCheck_2264_ = !lean_is_exclusive(v_scopes_2249_);
if (v_isSharedCheck_2264_ == 0)
{
v___x_2255_ = v_scopes_2249_;
v_isShared_2256_ = v_isSharedCheck_2264_;
goto v_resetjp_2254_;
}
else
{
lean_inc(v_scopes_2253_);
lean_inc(v_ctx_2252_);
lean_inc(v_imported_2251_);
lean_inc(v_name_2250_);
lean_dec(v_scopes_2249_);
v___x_2255_ = lean_box(0);
v_isShared_2256_ = v_isSharedCheck_2264_;
goto v_resetjp_2254_;
}
v_resetjp_2254_:
{
lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2260_; 
v___x_2257_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__111));
v___x_2258_ = lean_name_append_after(v_name_2250_, v___x_2257_);
if (v_isShared_2256_ == 0)
{
lean_ctor_set(v___x_2255_, 0, v___x_2258_);
v___x_2260_ = v___x_2255_;
goto v_reusejp_2259_;
}
else
{
lean_object* v_reuseFailAlloc_2263_; 
v_reuseFailAlloc_2263_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2263_, 0, v___x_2258_);
lean_ctor_set(v_reuseFailAlloc_2263_, 1, v_imported_2251_);
lean_ctor_set(v_reuseFailAlloc_2263_, 2, v_ctx_2252_);
lean_ctor_set(v_reuseFailAlloc_2263_, 3, v_scopes_2253_);
v___x_2260_ = v_reuseFailAlloc_2263_;
goto v_reusejp_2259_;
}
v_reusejp_2259_:
{
lean_object* v___x_2261_; lean_object* v___x_2262_; 
v___x_2261_ = l_Lean_MacroScopesView_review(v___x_2260_);
v___x_2262_ = l_Lean_mkIdentFrom(v___y_2242_, v___x_2261_, v___x_2247_);
v___y_2200_ = v___y_2238_;
v___y_2201_ = v___y_2240_;
v___y_2202_ = v___y_2239_;
v___y_2203_ = v___y_2242_;
v___y_2204_ = v___y_2243_;
v___y_2205_ = v___y_2244_;
v_n__def_2206_ = v___x_2262_;
v___y_2207_ = v___y_2241_;
v___y_2208_ = v___y_2237_;
goto v___jp_2199_;
}
}
}
else
{
lean_object* v_id_2265_; 
v_id_2265_ = l_Lean_Syntax_getArg(v___y_2245_, v___x_1337_);
lean_dec(v___y_2245_);
v___y_2200_ = v___y_2238_;
v___y_2201_ = v___y_2240_;
v___y_2202_ = v___y_2239_;
v___y_2203_ = v___y_2242_;
v___y_2204_ = v___y_2243_;
v___y_2205_ = v___y_2244_;
v_n__def_2206_ = v_id_2265_;
v___y_2207_ = v___y_2241_;
v___y_2208_ = v___y_2237_;
goto v___jp_2199_;
}
}
v___jp_2266_:
{
if (lean_obj_tag(v___y_2272_) == 0)
{
lean_object* v___x_2276_; 
v___x_2276_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__112));
v___y_2237_ = v___y_2267_;
v___y_2238_ = v___y_2268_;
v___y_2239_ = v___y_2269_;
v___y_2240_ = v___y_2275_;
v___y_2241_ = v___y_2270_;
v___y_2242_ = v___y_2271_;
v___y_2243_ = v___y_2273_;
v___y_2244_ = v___y_2274_;
v___y_2245_ = v___x_2276_;
goto v___jp_2236_;
}
else
{
lean_object* v_val_2277_; 
v_val_2277_ = lean_ctor_get(v___y_2272_, 0);
lean_inc(v_val_2277_);
lean_dec_ref_known(v___y_2272_, 1);
v___y_2237_ = v___y_2267_;
v___y_2238_ = v___y_2268_;
v___y_2239_ = v___y_2269_;
v___y_2240_ = v___y_2275_;
v___y_2241_ = v___y_2270_;
v___y_2242_ = v___y_2271_;
v___y_2243_ = v___y_2273_;
v___y_2244_ = v___y_2274_;
v___y_2245_ = v_val_2277_;
goto v___jp_2236_;
}
}
v___jp_2278_:
{
lean_object* v___x_2287_; 
v___x_2287_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__113));
lean_inc(v_snd_2284_);
v___y_2267_ = v___y_2286_;
v___y_2268_ = v___y_2279_;
v___y_2269_ = v___y_2280_;
v___y_2270_ = v___y_2285_;
v___y_2271_ = v_fst_2283_;
v___y_2272_ = v___y_2281_;
v___y_2273_ = v___y_2282_;
v___y_2274_ = v_snd_2284_;
v___y_2275_ = v___x_2287_;
goto v___jp_2266_;
}
v___jp_2288_:
{
lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; uint8_t v___x_2293_; 
v___x_2290_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__114));
v___x_2291_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__115));
v___x_2292_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___closed__116));
lean_inc(v___x_1336_);
v___x_2293_ = l_Lean_Syntax_isOfKind(v___x_1336_, v___x_2290_);
if (v___x_2293_ == 0)
{
lean_object* v___x_2294_; lean_object* v_a_2295_; lean_object* v___x_2297_; uint8_t v_isShared_2298_; uint8_t v_isSharedCheck_2302_; 
lean_dec(v___y_2289_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
lean_dec(v___x_1336_);
lean_dec(v___x_1333_);
v___x_2294_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
v_a_2295_ = lean_ctor_get(v___x_2294_, 0);
v_isSharedCheck_2302_ = !lean_is_exclusive(v___x_2294_);
if (v_isSharedCheck_2302_ == 0)
{
v___x_2297_ = v___x_2294_;
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
else
{
lean_inc(v_a_2295_);
lean_dec(v___x_2294_);
v___x_2297_ = lean_box(0);
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
v_resetjp_2296_:
{
lean_object* v___x_2300_; 
if (v_isShared_2298_ == 0)
{
v___x_2300_ = v___x_2297_;
goto v_reusejp_2299_;
}
else
{
lean_object* v_reuseFailAlloc_2301_; 
v_reuseFailAlloc_2301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2301_, 0, v_a_2295_);
v___x_2300_ = v_reuseFailAlloc_2301_;
goto v_reusejp_2299_;
}
v_reusejp_2299_:
{
return v___x_2300_;
}
}
}
else
{
lean_object* v_n_2303_; lean_object* v___x_2304_; uint8_t v___x_2305_; 
v_n_2303_ = l_Lean_Syntax_getArg(v___x_1336_, v___x_1332_);
v___x_2304_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command_irredDefLemma___closed__14));
lean_inc(v_n_2303_);
v___x_2305_ = l_Lean_Syntax_isOfKind(v_n_2303_, v___x_2304_);
if (v___x_2305_ == 0)
{
lean_object* v___x_2306_; lean_object* v_a_2307_; lean_object* v___x_2309_; uint8_t v_isShared_2310_; uint8_t v_isSharedCheck_2314_; 
lean_dec(v_n_2303_);
lean_dec(v___y_2289_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
lean_dec(v___x_1336_);
lean_dec(v___x_1333_);
v___x_2306_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
v_a_2307_ = lean_ctor_get(v___x_2306_, 0);
v_isSharedCheck_2314_ = !lean_is_exclusive(v___x_2306_);
if (v_isSharedCheck_2314_ == 0)
{
v___x_2309_ = v___x_2306_;
v_isShared_2310_ = v_isSharedCheck_2314_;
goto v_resetjp_2308_;
}
else
{
lean_inc(v_a_2307_);
lean_dec(v___x_2306_);
v___x_2309_ = lean_box(0);
v_isShared_2310_ = v_isSharedCheck_2314_;
goto v_resetjp_2308_;
}
v_resetjp_2308_:
{
lean_object* v___x_2312_; 
if (v_isShared_2310_ == 0)
{
v___x_2312_ = v___x_2309_;
goto v_reusejp_2311_;
}
else
{
lean_object* v_reuseFailAlloc_2313_; 
v_reuseFailAlloc_2313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2313_, 0, v_a_2307_);
v___x_2312_ = v_reuseFailAlloc_2313_;
goto v_reusejp_2311_;
}
v_reusejp_2311_:
{
return v___x_2312_;
}
}
}
else
{
lean_object* v___x_2315_; uint8_t v___x_2316_; 
v___x_2315_ = l_Lean_Syntax_getArg(v___x_1336_, v___x_1334_);
lean_dec(v___x_1336_);
v___x_2316_ = l_Lean_Syntax_isNone(v___x_2315_);
if (v___x_2316_ == 0)
{
uint8_t v___x_2317_; 
lean_inc(v___x_2315_);
v___x_2317_ = l_Lean_Syntax_matchesNull(v___x_2315_, v___x_1337_);
if (v___x_2317_ == 0)
{
lean_object* v___x_2318_; lean_object* v_a_2319_; lean_object* v___x_2321_; uint8_t v_isShared_2322_; uint8_t v_isSharedCheck_2326_; 
lean_dec(v___x_2315_);
lean_dec(v_n_2303_);
lean_dec(v___y_2289_);
lean_dec(v___x_1342_);
lean_dec(v___x_1340_);
lean_dec(v___x_1333_);
v___x_2318_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules____private__Mathlib__Tactic__IrreducibleDef__0__Lean__Elab__Command__commandStop__at__first__error______1_spec__0___redArg();
v_a_2319_ = lean_ctor_get(v___x_2318_, 0);
v_isSharedCheck_2326_ = !lean_is_exclusive(v___x_2318_);
if (v_isSharedCheck_2326_ == 0)
{
v___x_2321_ = v___x_2318_;
v_isShared_2322_ = v_isSharedCheck_2326_;
goto v_resetjp_2320_;
}
else
{
lean_inc(v_a_2319_);
lean_dec(v___x_2318_);
v___x_2321_ = lean_box(0);
v_isShared_2322_ = v_isSharedCheck_2326_;
goto v_resetjp_2320_;
}
v_resetjp_2320_:
{
lean_object* v___x_2324_; 
if (v_isShared_2322_ == 0)
{
v___x_2324_ = v___x_2321_;
goto v_reusejp_2323_;
}
else
{
lean_object* v_reuseFailAlloc_2325_; 
v_reuseFailAlloc_2325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2325_, 0, v_a_2319_);
v___x_2324_ = v_reuseFailAlloc_2325_;
goto v_reusejp_2323_;
}
v_reusejp_2323_:
{
return v___x_2324_;
}
}
}
else
{
lean_object* v___x_2327_; lean_object* v_us_2328_; lean_object* v___x_2329_; 
v___x_2327_ = l_Lean_Syntax_getArg(v___x_2315_, v___x_1334_);
lean_dec(v___x_2315_);
v_us_2328_ = l_Lean_Syntax_getArgs(v___x_2327_);
lean_dec(v___x_2327_);
lean_inc_ref(v_us_2328_);
v___x_2329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2329_, 0, v_us_2328_);
v___y_2267_ = v_a_1323_;
v___y_2268_ = v___x_2290_;
v___y_2269_ = v___x_2292_;
v___y_2270_ = v_a_1322_;
v___y_2271_ = v_n_2303_;
v___y_2272_ = v___y_2289_;
v___y_2273_ = v___x_2291_;
v___y_2274_ = v___x_2329_;
v___y_2275_ = v_us_2328_;
goto v___jp_2266_;
}
}
else
{
lean_object* v___x_2330_; 
lean_dec(v___x_2315_);
v___x_2330_ = lean_box(0);
v___y_2279_ = v___x_2290_;
v___y_2280_ = v___x_2292_;
v___y_2281_ = v___y_2289_;
v___y_2282_ = v___x_2291_;
v_fst_2283_ = v_n_2303_;
v_snd_2284_ = v___x_2330_;
v___y_2285_ = v_a_1322_;
v___y_2286_ = v_a_1323_;
goto v___jp_2278_;
}
}
}
}
}
v___jp_1325_:
{
lean_object* v___x_1326_; lean_object* v___x_1327_; 
v___x_1326_ = lean_box(0);
v___x_1327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
return v___x_1327_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1___boxed(lean_object* v_x_2341_, lean_object* v_a_2342_, lean_object* v_a_2343_, lean_object* v_a_2344_){
_start:
{
lean_object* v_res_2345_; 
v_res_2345_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1(v_x_2341_, v_a_2342_, v_a_2343_);
lean_dec(v_a_2343_);
lean_dec_ref(v_a_2342_);
return v_res_2345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2(lean_object* v_msgData_2346_, lean_object* v___y_2347_, lean_object* v___y_2348_){
_start:
{
lean_object* v___x_2350_; 
v___x_2350_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___redArg(v_msgData_2346_, v___y_2348_);
return v___x_2350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2___boxed(lean_object* v_msgData_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_){
_start:
{
lean_object* v_res_2355_; 
v_res_2355_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__2(v_msgData_2351_, v___y_2352_, v___y_2353_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
return v_res_2355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2(lean_object* v_00_u03b1_2356_, lean_object* v_msg_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_){
_start:
{
lean_object* v___x_2361_; 
v___x_2361_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___redArg(v_msg_2357_, v___y_2358_, v___y_2359_);
return v___x_2361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2___boxed(lean_object* v_00_u03b1_2362_, lean_object* v_msg_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_){
_start:
{
lean_object* v_res_2367_; 
v_res_2367_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2(v_00_u03b1_2362_, v_msg_2363_, v___y_2364_, v___y_2365_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
return v_res_2367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3(lean_object* v_msgData_2368_, lean_object* v_macroStack_2369_, lean_object* v___y_2370_, lean_object* v___y_2371_){
_start:
{
lean_object* v___x_2373_; 
v___x_2373_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___redArg(v_msgData_2368_, v_macroStack_2369_, v___y_2371_);
return v___x_2373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3___boxed(lean_object* v_msgData_2374_, lean_object* v_macroStack_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_){
_start:
{
lean_object* v_res_2379_; 
v_res_2379_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Command___aux__Mathlib__Tactic__IrreducibleDef______elabRules__Lean__Elab__Command__command__Irreducible__def__________1_spec__2_spec__3(v_msgData_2374_, v_macroStack_2375_, v___y_2376_, v___y_2377_);
lean_dec(v___y_2377_);
lean_dec_ref(v___y_2376_);
return v_res_2379_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_TermReduce(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_IrreducibleDef(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_TermReduce(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_IrreducibleDef(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_TermReduce(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_IrreducibleDef(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_TermReduce(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_IrreducibleDef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_IrreducibleDef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_IrreducibleDef(builtin);
}
#ifdef __cplusplus
}
#endif
