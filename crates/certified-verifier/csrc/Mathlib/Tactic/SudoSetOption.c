// Lean compiler output
// Module: Mathlib.Tactic.SudoSetOption
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_isNatLit_x3f(lean_object*);
lean_object* l_Lean_Syntax_isStrLit_x3f(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_modifyScope___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "unsupported option value "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "commandSudoSet_option___"};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__0 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__0_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(64, 2, 25, 238, 109, 229, 152, 31)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__1 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__1_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__2 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__2_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__3 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "sudo "};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__4 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__4_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__4_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__5 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__5_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "set_option "};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__6 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__6_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__6_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__7 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__7_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__5_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__7_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__8 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__8_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__9 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__9_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__10 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__10_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__10_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__11 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__11_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__8_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__11_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__12 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__12_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__13 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__13_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__14 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__14_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__15 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__15_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__16 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__16_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__16_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__17 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__17_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__14_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__17_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__18 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__18_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__12_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__18_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__19 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__19_value;
static const lean_string_object lp_mathlib_commandSudoSet__option_______00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__20 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__20_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__20_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__21 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__21_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__22 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__22_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__19_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__22_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__23 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__23_value;
static const lean_ctor_object lp_mathlib_commandSudoSet__option_______00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__23_value)}};
static const lean_object* lp_mathlib_commandSudoSet__option_______00__closed__24 = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib_commandSudoSet__option______ = (const lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__24_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_termSudoSet__option______In___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "termSudoSet_option___In_"};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__0 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__0_value;
static const lean_ctor_object lp_mathlib_termSudoSet__option______In___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(187, 222, 232, 20, 156, 126, 217, 219)}};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__1 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__1_value;
static const lean_string_object lp_mathlib_termSudoSet__option______In___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__2 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__2_value;
static const lean_ctor_object lp_mathlib_termSudoSet__option______In___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__2_value)}};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__3 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__3_value;
static const lean_ctor_object lp_mathlib_termSudoSet__option______In___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__23_value),((lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__3_value)}};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__4 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__4_value;
static const lean_ctor_object lp_mathlib_termSudoSet__option______In___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__3_value),((lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__4_value),((lean_object*)&lp_mathlib_commandSudoSet__option_______00__closed__22_value)}};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__5 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__5_value;
static const lean_ctor_object lp_mathlib_termSudoSet__option______In___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__5_value)}};
static const lean_object* lp_mathlib_termSudoSet__option______In___00__closed__6 = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_termSudoSet__option______In__ = (const lean_object*)&lp_mathlib_termSudoSet__option______In___00__closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0(lean_object* v_opts_4_, lean_object* v_name_5_, lean_object* v_toPure_6_, lean_object* v_val_7_){
_start:
{
lean_object* v_map_8_; uint8_t v_hasTrace_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_25_; 
v_map_8_ = lean_ctor_get(v_opts_4_, 0);
v_hasTrace_9_ = lean_ctor_get_uint8(v_opts_4_, sizeof(void*)*1);
v_isSharedCheck_25_ = !lean_is_exclusive(v_opts_4_);
if (v_isSharedCheck_25_ == 0)
{
v___x_11_ = v_opts_4_;
v_isShared_12_ = v_isSharedCheck_25_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_map_8_);
lean_dec(v_opts_4_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_25_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = l_Lean_Syntax_getId(v_name_5_);
lean_inc(v___x_13_);
v___x_14_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_13_, v_val_7_, v_map_8_);
if (v_hasTrace_9_ == 0)
{
lean_object* v___x_15_; uint8_t v___x_16_; lean_object* v___x_18_; 
v___x_15_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__1));
v___x_16_ = l_Lean_Name_isPrefixOf(v___x_15_, v___x_13_);
lean_dec(v___x_13_);
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 0, v___x_14_);
v___x_18_ = v___x_11_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v___x_14_);
v___x_18_ = v_reuseFailAlloc_20_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
lean_object* v___x_19_; 
lean_ctor_set_uint8(v___x_18_, sizeof(void*)*1, v___x_16_);
v___x_19_ = lean_apply_2(v_toPure_6_, lean_box(0), v___x_18_);
return v___x_19_;
}
}
else
{
lean_object* v___x_22_; 
lean_dec(v___x_13_);
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 0, v___x_14_);
v___x_22_ = v___x_11_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v___x_14_);
lean_ctor_set_uint8(v_reuseFailAlloc_24_, sizeof(void*)*1, v_hasTrace_9_);
v___x_22_ = v_reuseFailAlloc_24_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
lean_object* v___x_23_; 
v___x_23_ = lean_apply_2(v_toPure_6_, lean_box(0), v___x_22_);
return v___x_23_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___boxed(lean_object* v_opts_26_, lean_object* v_name_27_, lean_object* v_toPure_28_, lean_object* v_val_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0(v_opts_26_, v_name_27_, v_toPure_28_, v_val_29_);
lean_dec(v_name_27_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__1(lean_object* v___f_31_, lean_object* v_val_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_apply_1(v___f_31_, v_val_32_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__0));
v___x_36_ = l_Lean_stringToMessageData(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_name_41_, lean_object* v_val_42_, lean_object* v_opts_43_){
_start:
{
lean_object* v_toApplicative_44_; lean_object* v_toBind_45_; lean_object* v_toPure_46_; lean_object* v___f_47_; lean_object* v___f_48_; 
v_toApplicative_44_ = lean_ctor_get(v_inst_39_, 0);
v_toBind_45_ = lean_ctor_get(v_inst_39_, 1);
lean_inc(v_toBind_45_);
v_toPure_46_ = lean_ctor_get(v_toApplicative_44_, 1);
lean_inc(v_toPure_46_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_47_, 0, v_opts_43_);
lean_closure_set(v___f_47_, 1, v_name_41_);
lean_closure_set(v___f_47_, 2, v_toPure_46_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__1), 2, 1);
lean_closure_set(v___f_48_, 0, v___f_47_);
if (lean_obj_tag(v_val_42_) == 3)
{
lean_object* v_val_77_; 
v_val_77_ = lean_ctor_get(v_val_42_, 2);
if (lean_obj_tag(v_val_77_) == 1)
{
lean_object* v_pre_78_; 
v_pre_78_ = lean_ctor_get(v_val_77_, 0);
if (lean_obj_tag(v_pre_78_) == 0)
{
lean_object* v_str_79_; lean_object* v___x_80_; uint8_t v___x_81_; 
v_str_79_ = lean_ctor_get(v_val_77_, 1);
v___x_80_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__2));
v___x_81_ = lean_string_dec_eq(v_str_79_, v___x_80_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; uint8_t v___x_83_; 
v___x_82_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__3));
v___x_83_ = lean_string_dec_eq(v_str_79_, v___x_82_);
if (v___x_83_ == 0)
{
goto v___jp_49_;
}
else
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
lean_inc(v_toPure_46_);
lean_dec_ref_known(v_val_42_, 4);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
v___x_84_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_84_, 0, v___x_81_);
v___x_85_ = lean_apply_2(v_toPure_46_, lean_box(0), v___x_84_);
v___x_86_ = lean_apply_4(v_toBind_45_, lean_box(0), lean_box(0), v___x_85_, v___f_48_);
return v___x_86_;
}
}
else
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
lean_inc(v_toPure_46_);
lean_dec_ref_known(v_val_42_, 4);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
v___x_87_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_87_, 0, v___x_81_);
v___x_88_ = lean_apply_2(v_toPure_46_, lean_box(0), v___x_87_);
v___x_89_ = lean_apply_4(v_toBind_45_, lean_box(0), lean_box(0), v___x_88_, v___f_48_);
return v___x_89_;
}
}
else
{
goto v___jp_49_;
}
}
else
{
goto v___jp_49_;
}
}
else
{
goto v___jp_49_;
}
v___jp_49_:
{
lean_object* v___x_50_; 
v___x_50_ = l_Lean_Syntax_isNatLit_x3f(v_val_42_);
if (lean_obj_tag(v___x_50_) == 0)
{
lean_object* v___x_51_; 
v___x_51_ = l_Lean_Syntax_isStrLit_x3f(v_val_42_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_52_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1);
v___x_53_ = l_Lean_MessageData_ofSyntax(v_val_42_);
v___x_54_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_52_);
lean_ctor_set(v___x_54_, 1, v___x_53_);
v___x_55_ = l_Lean_throwError___redArg(v_inst_39_, v_inst_40_, v___x_54_);
v___x_56_ = lean_apply_4(v_toBind_45_, lean_box(0), lean_box(0), v___x_55_, v___f_48_);
return v___x_56_;
}
else
{
lean_object* v_val_57_; lean_object* v___x_59_; uint8_t v_isShared_60_; uint8_t v_isSharedCheck_66_; 
lean_inc(v_toPure_46_);
lean_dec(v_val_42_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
v_val_57_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_66_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_66_ == 0)
{
v___x_59_ = v___x_51_;
v_isShared_60_ = v_isSharedCheck_66_;
goto v_resetjp_58_;
}
else
{
lean_inc(v_val_57_);
lean_dec(v___x_51_);
v___x_59_ = lean_box(0);
v_isShared_60_ = v_isSharedCheck_66_;
goto v_resetjp_58_;
}
v_resetjp_58_:
{
lean_object* v___x_62_; 
if (v_isShared_60_ == 0)
{
lean_ctor_set_tag(v___x_59_, 0);
v___x_62_ = v___x_59_;
goto v_reusejp_61_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v_val_57_);
v___x_62_ = v_reuseFailAlloc_65_;
goto v_reusejp_61_;
}
v_reusejp_61_:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = lean_apply_2(v_toPure_46_, lean_box(0), v___x_62_);
v___x_64_ = lean_apply_4(v_toBind_45_, lean_box(0), lean_box(0), v___x_63_, v___f_48_);
return v___x_64_;
}
}
}
}
else
{
lean_object* v_val_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_76_; 
lean_inc(v_toPure_46_);
lean_dec(v_val_42_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
v_val_67_ = lean_ctor_get(v___x_50_, 0);
v_isSharedCheck_76_ = !lean_is_exclusive(v___x_50_);
if (v_isSharedCheck_76_ == 0)
{
v___x_69_ = v___x_50_;
v_isShared_70_ = v_isSharedCheck_76_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_val_67_);
lean_dec(v___x_50_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_76_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v___x_72_; 
if (v_isShared_70_ == 0)
{
lean_ctor_set_tag(v___x_69_, 3);
v___x_72_ = v___x_69_;
goto v_reusejp_71_;
}
else
{
lean_object* v_reuseFailAlloc_75_; 
v_reuseFailAlloc_75_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_75_, 0, v_val_67_);
v___x_72_ = v_reuseFailAlloc_75_;
goto v_reusejp_71_;
}
v_reusejp_71_:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = lean_apply_2(v_toPure_46_, lean_box(0), v___x_72_);
v___x_74_ = lean_apply_4(v_toBind_45_, lean_box(0), lean_box(0), v___x_73_, v___f_48_);
return v___x_74_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption(lean_object* v_m_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_name_93_, lean_object* v_val_94_, lean_object* v_opts_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg(v_inst_91_, v_inst_92_, v_name_93_, v_val_94_, v_opts_95_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_152_ = lean_box(0);
v___x_153_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v___x_152_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg(){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0);
v___x_157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___boxed(lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg();
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0(lean_object* v_00_u03b1_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg();
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___boxed(lean_object* v_00_u03b1_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0(v_00_u03b1_165_, v___y_166_, v___y_167_);
lean_dec(v___y_167_);
lean_dec_ref(v___y_166_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2(lean_object* v_opts_170_, lean_object* v_opt_171_){
_start:
{
lean_object* v_name_172_; lean_object* v_defValue_173_; lean_object* v_map_174_; lean_object* v___x_175_; 
v_name_172_ = lean_ctor_get(v_opt_171_, 0);
v_defValue_173_ = lean_ctor_get(v_opt_171_, 1);
v_map_174_ = lean_ctor_get(v_opts_170_, 0);
v___x_175_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_174_, v_name_172_);
if (lean_obj_tag(v___x_175_) == 0)
{
lean_inc(v_defValue_173_);
return v_defValue_173_;
}
else
{
lean_object* v_val_176_; 
v_val_176_ = lean_ctor_get(v___x_175_, 0);
lean_inc(v_val_176_);
lean_dec_ref_known(v___x_175_, 1);
if (lean_obj_tag(v_val_176_) == 3)
{
lean_object* v_v_177_; 
v_v_177_ = lean_ctor_get(v_val_176_, 0);
lean_inc(v_v_177_);
lean_dec_ref_known(v_val_176_, 1);
return v_v_177_;
}
else
{
lean_dec(v_val_176_);
lean_inc(v_defValue_173_);
return v_defValue_173_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2___boxed(lean_object* v_opts_178_, lean_object* v_opt_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2(v_opts_178_, v_opt_179_);
lean_dec_ref(v_opt_179_);
lean_dec_ref(v_opts_178_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1___lam__0(lean_object* v_a_181_, lean_object* v_scope_182_){
_start:
{
lean_object* v_header_183_; lean_object* v_currNamespace_184_; lean_object* v_openDecls_185_; lean_object* v_levelNames_186_; lean_object* v_varDecls_187_; lean_object* v_varUIds_188_; lean_object* v_includedVars_189_; lean_object* v_omittedVars_190_; uint8_t v_isNoncomputable_191_; uint8_t v_isPublic_192_; uint8_t v_isMeta_193_; lean_object* v_attrs_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_201_; 
v_header_183_ = lean_ctor_get(v_scope_182_, 0);
v_currNamespace_184_ = lean_ctor_get(v_scope_182_, 2);
v_openDecls_185_ = lean_ctor_get(v_scope_182_, 3);
v_levelNames_186_ = lean_ctor_get(v_scope_182_, 4);
v_varDecls_187_ = lean_ctor_get(v_scope_182_, 5);
v_varUIds_188_ = lean_ctor_get(v_scope_182_, 6);
v_includedVars_189_ = lean_ctor_get(v_scope_182_, 7);
v_omittedVars_190_ = lean_ctor_get(v_scope_182_, 8);
v_isNoncomputable_191_ = lean_ctor_get_uint8(v_scope_182_, sizeof(void*)*10);
v_isPublic_192_ = lean_ctor_get_uint8(v_scope_182_, sizeof(void*)*10 + 1);
v_isMeta_193_ = lean_ctor_get_uint8(v_scope_182_, sizeof(void*)*10 + 2);
v_attrs_194_ = lean_ctor_get(v_scope_182_, 9);
v_isSharedCheck_201_ = !lean_is_exclusive(v_scope_182_);
if (v_isSharedCheck_201_ == 0)
{
lean_object* v_unused_202_; 
v_unused_202_ = lean_ctor_get(v_scope_182_, 1);
lean_dec(v_unused_202_);
v___x_196_ = v_scope_182_;
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_attrs_194_);
lean_inc(v_omittedVars_190_);
lean_inc(v_includedVars_189_);
lean_inc(v_varUIds_188_);
lean_inc(v_varDecls_187_);
lean_inc(v_levelNames_186_);
lean_inc(v_openDecls_185_);
lean_inc(v_currNamespace_184_);
lean_inc(v_header_183_);
lean_dec(v_scope_182_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_199_; 
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 1, v_a_181_);
v___x_199_ = v___x_196_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(0, 10, 3);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v_header_183_);
lean_ctor_set(v_reuseFailAlloc_200_, 1, v_a_181_);
lean_ctor_set(v_reuseFailAlloc_200_, 2, v_currNamespace_184_);
lean_ctor_set(v_reuseFailAlloc_200_, 3, v_openDecls_185_);
lean_ctor_set(v_reuseFailAlloc_200_, 4, v_levelNames_186_);
lean_ctor_set(v_reuseFailAlloc_200_, 5, v_varDecls_187_);
lean_ctor_set(v_reuseFailAlloc_200_, 6, v_varUIds_188_);
lean_ctor_set(v_reuseFailAlloc_200_, 7, v_includedVars_189_);
lean_ctor_set(v_reuseFailAlloc_200_, 8, v_omittedVars_190_);
lean_ctor_set(v_reuseFailAlloc_200_, 9, v_attrs_194_);
lean_ctor_set_uint8(v_reuseFailAlloc_200_, sizeof(void*)*10, v_isNoncomputable_191_);
lean_ctor_set_uint8(v_reuseFailAlloc_200_, sizeof(void*)*10 + 1, v_isPublic_192_);
lean_ctor_set_uint8(v_reuseFailAlloc_200_, sizeof(void*)*10 + 2, v_isMeta_193_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_203_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_204_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
return v___x_205_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_206_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1);
v___x_207_ = lean_unsigned_to_nat(0u);
v___x_208_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v___x_207_);
lean_ctor_set(v___x_208_, 2, v___x_207_);
lean_ctor_set(v___x_208_, 3, v___x_207_);
lean_ctor_set(v___x_208_, 4, v___x_206_);
lean_ctor_set(v___x_208_, 5, v___x_206_);
lean_ctor_set(v___x_208_, 6, v___x_206_);
lean_ctor_set(v___x_208_, 7, v___x_206_);
lean_ctor_set(v___x_208_, 8, v___x_206_);
lean_ctor_set(v___x_208_, 9, v___x_206_);
return v___x_208_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = lean_unsigned_to_nat(32u);
v___x_210_ = lean_mk_empty_array_with_capacity(v___x_209_);
v___x_211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_212_ = ((size_t)5ULL);
v___x_213_ = lean_unsigned_to_nat(0u);
v___x_214_ = lean_unsigned_to_nat(32u);
v___x_215_ = lean_mk_empty_array_with_capacity(v___x_214_);
v___x_216_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__3);
v___x_217_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_217_, 0, v___x_216_);
lean_ctor_set(v___x_217_, 1, v___x_215_);
lean_ctor_set(v___x_217_, 2, v___x_213_);
lean_ctor_set(v___x_217_, 3, v___x_213_);
lean_ctor_set_usize(v___x_217_, 4, v___x_212_);
return v___x_217_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_218_ = lean_box(1);
v___x_219_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__4);
v___x_220_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__1);
v___x_221_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set(v___x_221_, 1, v___x_219_);
lean_ctor_set(v___x_221_, 2, v___x_218_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg(lean_object* v_msgData_222_, lean_object* v___y_223_){
_start:
{
lean_object* v___x_225_; lean_object* v_env_226_; lean_object* v___x_227_; lean_object* v_scopes_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v_opts_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_225_ = lean_st_ref_get(v___y_223_);
v_env_226_ = lean_ctor_get(v___x_225_, 0);
lean_inc_ref(v_env_226_);
lean_dec(v___x_225_);
v___x_227_ = lean_st_ref_get(v___y_223_);
v_scopes_228_ = lean_ctor_get(v___x_227_, 2);
lean_inc(v_scopes_228_);
lean_dec(v___x_227_);
v___x_229_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_230_ = l_List_head_x21___redArg(v___x_229_, v_scopes_228_);
lean_dec(v_scopes_228_);
v_opts_231_ = lean_ctor_get(v___x_230_, 1);
lean_inc_ref(v_opts_231_);
lean_dec(v___x_230_);
v___x_232_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__2);
v___x_233_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___closed__5);
v___x_234_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_234_, 0, v_env_226_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
lean_ctor_set(v___x_234_, 2, v___x_233_);
lean_ctor_set(v___x_234_, 3, v_opts_231_);
v___x_235_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_234_);
lean_ctor_set(v___x_235_, 1, v_msgData_222_);
v___x_236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg(v_msgData_237_, v___y_238_);
lean_dec(v___y_238_);
return v_res_240_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5(lean_object* v_opts_241_, lean_object* v_opt_242_){
_start:
{
lean_object* v_name_243_; lean_object* v_defValue_244_; lean_object* v_map_245_; lean_object* v___x_246_; 
v_name_243_ = lean_ctor_get(v_opt_242_, 0);
v_defValue_244_ = lean_ctor_get(v_opt_242_, 1);
v_map_245_ = lean_ctor_get(v_opts_241_, 0);
v___x_246_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_245_, v_name_243_);
if (lean_obj_tag(v___x_246_) == 0)
{
uint8_t v___x_247_; 
v___x_247_ = lean_unbox(v_defValue_244_);
return v___x_247_;
}
else
{
lean_object* v_val_248_; 
v_val_248_ = lean_ctor_get(v___x_246_, 0);
lean_inc(v_val_248_);
lean_dec_ref_known(v___x_246_, 1);
if (lean_obj_tag(v_val_248_) == 1)
{
uint8_t v_v_249_; 
v_v_249_ = lean_ctor_get_uint8(v_val_248_, 0);
lean_dec_ref_known(v_val_248_, 0);
return v_v_249_;
}
else
{
uint8_t v___x_250_; 
lean_dec(v_val_248_);
v___x_250_ = lean_unbox(v_defValue_244_);
return v___x_250_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5___boxed(lean_object* v_opts_251_, lean_object* v_opt_252_){
_start:
{
uint8_t v_res_253_; lean_object* v_r_254_; 
v_res_253_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5(v_opts_251_, v_opt_252_);
lean_dec_ref(v_opt_252_);
lean_dec_ref(v_opts_251_);
v_r_254_ = lean_box(v_res_253_);
return v_r_254_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0(void){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_255_ = lean_box(1);
v___x_256_ = l_Lean_MessageData_ofFormat(v___x_255_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__3(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_260_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__2));
v___x_261_ = l_Lean_MessageData_ofFormat(v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6(lean_object* v_x_262_, lean_object* v_x_263_){
_start:
{
if (lean_obj_tag(v_x_263_) == 0)
{
return v_x_262_;
}
else
{
lean_object* v_head_264_; lean_object* v_tail_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_287_; 
v_head_264_ = lean_ctor_get(v_x_263_, 0);
v_tail_265_ = lean_ctor_get(v_x_263_, 1);
v_isSharedCheck_287_ = !lean_is_exclusive(v_x_263_);
if (v_isSharedCheck_287_ == 0)
{
v___x_267_ = v_x_263_;
v_isShared_268_ = v_isSharedCheck_287_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_tail_265_);
lean_inc(v_head_264_);
lean_dec(v_x_263_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_287_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v_before_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_285_; 
v_before_269_ = lean_ctor_get(v_head_264_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v_head_264_);
if (v_isSharedCheck_285_ == 0)
{
lean_object* v_unused_286_; 
v_unused_286_ = lean_ctor_get(v_head_264_, 1);
lean_dec(v_unused_286_);
v___x_271_ = v_head_264_;
v_isShared_272_ = v_isSharedCheck_285_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_before_269_);
lean_dec(v_head_264_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_285_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
lean_object* v___x_273_; lean_object* v___x_275_; 
v___x_273_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0);
if (v_isShared_272_ == 0)
{
lean_ctor_set_tag(v___x_271_, 7);
lean_ctor_set(v___x_271_, 1, v___x_273_);
lean_ctor_set(v___x_271_, 0, v_x_262_);
v___x_275_ = v___x_271_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_x_262_);
lean_ctor_set(v_reuseFailAlloc_284_, 1, v___x_273_);
v___x_275_ = v_reuseFailAlloc_284_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
lean_object* v___x_276_; lean_object* v___x_278_; 
v___x_276_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__3);
if (v_isShared_268_ == 0)
{
lean_ctor_set_tag(v___x_267_, 7);
lean_ctor_set(v___x_267_, 1, v___x_276_);
lean_ctor_set(v___x_267_, 0, v___x_275_);
v___x_278_ = v___x_267_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v___x_275_);
lean_ctor_set(v_reuseFailAlloc_283_, 1, v___x_276_);
v___x_278_ = v_reuseFailAlloc_283_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_279_ = l_Lean_MessageData_ofSyntax(v_before_269_);
v___x_280_ = l_Lean_indentD(v___x_279_);
v___x_281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_278_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
v_x_262_ = v___x_281_;
v_x_263_ = v_tail_265_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__1));
v___x_292_ = l_Lean_MessageData_ofFormat(v___x_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg(lean_object* v_msgData_293_, lean_object* v_macroStack_294_, lean_object* v___y_295_){
_start:
{
lean_object* v___x_297_; lean_object* v_scopes_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v_opts_301_; lean_object* v___x_302_; uint8_t v___x_303_; 
v___x_297_ = lean_st_ref_get(v___y_295_);
v_scopes_298_ = lean_ctor_get(v___x_297_, 2);
lean_inc(v_scopes_298_);
lean_dec(v___x_297_);
v___x_299_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_300_ = l_List_head_x21___redArg(v___x_299_, v_scopes_298_);
lean_dec(v_scopes_298_);
v_opts_301_ = lean_ctor_get(v___x_300_, 1);
lean_inc_ref(v_opts_301_);
lean_dec(v___x_300_);
v___x_302_ = l_Lean_Elab_pp_macroStack;
v___x_303_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5(v_opts_301_, v___x_302_);
lean_dec_ref(v_opts_301_);
if (v___x_303_ == 0)
{
lean_object* v___x_304_; 
lean_dec(v_macroStack_294_);
v___x_304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_304_, 0, v_msgData_293_);
return v___x_304_;
}
else
{
if (lean_obj_tag(v_macroStack_294_) == 0)
{
lean_object* v___x_305_; 
v___x_305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_305_, 0, v_msgData_293_);
return v___x_305_;
}
else
{
lean_object* v_head_306_; lean_object* v_after_307_; lean_object* v___x_309_; uint8_t v_isShared_310_; uint8_t v_isSharedCheck_322_; 
v_head_306_ = lean_ctor_get(v_macroStack_294_, 0);
lean_inc(v_head_306_);
v_after_307_ = lean_ctor_get(v_head_306_, 1);
v_isSharedCheck_322_ = !lean_is_exclusive(v_head_306_);
if (v_isSharedCheck_322_ == 0)
{
lean_object* v_unused_323_; 
v_unused_323_ = lean_ctor_get(v_head_306_, 0);
lean_dec(v_unused_323_);
v___x_309_ = v_head_306_;
v_isShared_310_ = v_isSharedCheck_322_;
goto v_resetjp_308_;
}
else
{
lean_inc(v_after_307_);
lean_dec(v_head_306_);
v___x_309_ = lean_box(0);
v_isShared_310_ = v_isSharedCheck_322_;
goto v_resetjp_308_;
}
v_resetjp_308_:
{
lean_object* v___x_311_; lean_object* v___x_313_; 
v___x_311_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0);
if (v_isShared_310_ == 0)
{
lean_ctor_set_tag(v___x_309_, 7);
lean_ctor_set(v___x_309_, 1, v___x_311_);
lean_ctor_set(v___x_309_, 0, v_msgData_293_);
v___x_313_ = v___x_309_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v_msgData_293_);
lean_ctor_set(v_reuseFailAlloc_321_, 1, v___x_311_);
v___x_313_ = v_reuseFailAlloc_321_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v_msgData_318_; lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_314_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2);
v___x_315_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_313_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
v___x_316_ = l_Lean_MessageData_ofSyntax(v_after_307_);
v___x_317_ = l_Lean_indentD(v___x_316_);
v_msgData_318_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_318_, 0, v___x_315_);
lean_ctor_set(v_msgData_318_, 1, v___x_317_);
v___x_319_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6(v_msgData_318_, v_macroStack_294_);
v___x_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
return v___x_320_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_msgData_324_, lean_object* v_macroStack_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg(v_msgData_324_, v_macroStack_325_, v___y_326_);
lean_dec(v___y_326_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg(lean_object* v_msg_329_, lean_object* v___y_330_, lean_object* v___y_331_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = l_Lean_Elab_Command_getRef___redArg(v___y_330_);
if (lean_obj_tag(v___x_333_) == 0)
{
lean_object* v_a_334_; lean_object* v_macroStack_335_; lean_object* v___x_336_; lean_object* v_a_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_a_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_348_; 
v_a_334_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_a_334_);
lean_dec_ref_known(v___x_333_, 1);
v_macroStack_335_ = lean_ctor_get(v___y_330_, 4);
v___x_336_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg(v_msg_329_, v___y_331_);
v_a_337_ = lean_ctor_get(v___x_336_, 0);
lean_inc(v_a_337_);
lean_dec_ref(v___x_336_);
v___x_338_ = l_Lean_Elab_getBetterRef(v_a_334_, v_macroStack_335_);
lean_dec(v_a_334_);
lean_inc(v_macroStack_335_);
v___x_339_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg(v_a_337_, v_macroStack_335_, v___y_331_);
v_a_340_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_348_ == 0)
{
v___x_342_ = v___x_339_;
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_a_340_);
lean_dec(v___x_339_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
lean_object* v___x_344_; lean_object* v___x_346_; 
v___x_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_338_);
lean_ctor_set(v___x_344_, 1, v_a_340_);
if (v_isShared_343_ == 0)
{
lean_ctor_set_tag(v___x_342_, 1);
lean_ctor_set(v___x_342_, 0, v___x_344_);
v___x_346_ = v___x_342_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v___x_344_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
}
else
{
lean_object* v_a_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_356_; 
lean_dec_ref(v_msg_329_);
v_a_349_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_356_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_356_ == 0)
{
v___x_351_ = v___x_333_;
v_isShared_352_ = v_isSharedCheck_356_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_a_349_);
lean_dec(v___x_333_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_356_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
lean_object* v___x_354_; 
if (v_isShared_352_ == 0)
{
v___x_354_ = v___x_351_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v_a_349_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg___boxed(lean_object* v_msg_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg(v_msg_357_, v___y_358_, v___y_359_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1(lean_object* v_name_362_, lean_object* v_val_363_, lean_object* v_opts_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v_val_369_; 
if (lean_obj_tag(v_val_363_) == 3)
{
lean_object* v_val_419_; 
v_val_419_ = lean_ctor_get(v_val_363_, 2);
if (lean_obj_tag(v_val_419_) == 1)
{
lean_object* v_pre_420_; 
v_pre_420_ = lean_ctor_get(v_val_419_, 0);
if (lean_obj_tag(v_pre_420_) == 0)
{
lean_object* v_str_421_; lean_object* v___x_422_; uint8_t v___x_423_; 
v_str_421_ = lean_ctor_get(v_val_419_, 1);
v___x_422_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__2));
v___x_423_ = lean_string_dec_eq(v_str_421_, v___x_422_);
if (v___x_423_ == 0)
{
lean_object* v___x_424_; uint8_t v___x_425_; 
v___x_424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__3));
v___x_425_ = lean_string_dec_eq(v_str_421_, v___x_424_);
if (v___x_425_ == 0)
{
goto v___jp_388_;
}
else
{
lean_object* v___x_426_; 
lean_dec_ref_known(v_val_363_, 4);
v___x_426_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_426_, 0, v___x_423_);
v_val_369_ = v___x_426_;
goto v___jp_368_;
}
}
else
{
lean_object* v___x_427_; 
lean_dec_ref_known(v_val_363_, 4);
v___x_427_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_427_, 0, v___x_423_);
v_val_369_ = v___x_427_;
goto v___jp_368_;
}
}
else
{
goto v___jp_388_;
}
}
else
{
goto v___jp_388_;
}
}
else
{
goto v___jp_388_;
}
v___jp_368_:
{
lean_object* v_map_370_; uint8_t v_hasTrace_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_387_; 
v_map_370_ = lean_ctor_get(v_opts_364_, 0);
v_hasTrace_371_ = lean_ctor_get_uint8(v_opts_364_, sizeof(void*)*1);
v_isSharedCheck_387_ = !lean_is_exclusive(v_opts_364_);
if (v_isSharedCheck_387_ == 0)
{
v___x_373_ = v_opts_364_;
v_isShared_374_ = v_isSharedCheck_387_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_map_370_);
lean_dec(v_opts_364_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_387_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_375_ = l_Lean_Syntax_getId(v_name_362_);
lean_inc(v___x_375_);
v___x_376_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_375_, v_val_369_, v_map_370_);
if (v_hasTrace_371_ == 0)
{
lean_object* v___x_377_; uint8_t v___x_378_; lean_object* v___x_380_; 
v___x_377_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__1));
v___x_378_ = l_Lean_Name_isPrefixOf(v___x_377_, v___x_375_);
lean_dec(v___x_375_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_376_);
v___x_380_ = v___x_373_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v___x_376_);
v___x_380_ = v_reuseFailAlloc_382_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
lean_object* v___x_381_; 
lean_ctor_set_uint8(v___x_380_, sizeof(void*)*1, v___x_378_);
v___x_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
return v___x_381_;
}
}
else
{
lean_object* v___x_384_; 
lean_dec(v___x_375_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_376_);
v___x_384_ = v___x_373_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v___x_376_);
lean_ctor_set_uint8(v_reuseFailAlloc_386_, sizeof(void*)*1, v_hasTrace_371_);
v___x_384_ = v_reuseFailAlloc_386_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
lean_object* v___x_385_; 
v___x_385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_385_, 0, v___x_384_);
return v___x_385_;
}
}
}
}
v___jp_388_:
{
lean_object* v___x_389_; 
v___x_389_ = l_Lean_Syntax_isNatLit_x3f(v_val_363_);
if (lean_obj_tag(v___x_389_) == 0)
{
lean_object* v___x_390_; 
v___x_390_ = l_Lean_Syntax_isStrLit_x3f(v_val_363_);
if (lean_obj_tag(v___x_390_) == 0)
{
lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v_a_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_402_; 
lean_dec_ref(v_opts_364_);
v___x_391_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1);
v___x_392_ = l_Lean_MessageData_ofSyntax(v_val_363_);
v___x_393_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_391_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
v___x_394_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg(v___x_393_, v___y_365_, v___y_366_);
v_a_395_ = lean_ctor_get(v___x_394_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_394_);
if (v_isSharedCheck_402_ == 0)
{
v___x_397_ = v___x_394_;
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_a_395_);
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
v___x_400_ = v___x_397_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_a_395_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
else
{
lean_object* v_val_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_410_; 
lean_dec(v_val_363_);
v_val_403_ = lean_ctor_get(v___x_390_, 0);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_390_);
if (v_isSharedCheck_410_ == 0)
{
v___x_405_ = v___x_390_;
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_val_403_);
lean_dec(v___x_390_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_408_; 
if (v_isShared_406_ == 0)
{
lean_ctor_set_tag(v___x_405_, 0);
v___x_408_ = v___x_405_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v_val_403_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
v_val_369_ = v___x_408_;
goto v___jp_368_;
}
}
}
}
else
{
lean_object* v_val_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_418_; 
lean_dec(v_val_363_);
v_val_411_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_418_ == 0)
{
v___x_413_ = v___x_389_;
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_val_411_);
lean_dec(v___x_389_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
lean_object* v___x_416_; 
if (v_isShared_414_ == 0)
{
lean_ctor_set_tag(v___x_413_, 3);
v___x_416_ = v___x_413_;
goto v_reusejp_415_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v_val_411_);
v___x_416_ = v_reuseFailAlloc_417_;
goto v_reusejp_415_;
}
v_reusejp_415_:
{
v_val_369_ = v___x_416_;
goto v___jp_368_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1___boxed(lean_object* v_name_428_, lean_object* v_val_429_, lean_object* v_opts_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1(v_name_428_, v_val_429_, v_opts_430_, v___y_431_, v___y_432_);
lean_dec(v___y_432_);
lean_dec_ref(v___y_431_);
lean_dec(v_name_428_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1(lean_object* v_x_435_, lean_object* v_a_436_, lean_object* v_a_437_){
_start:
{
lean_object* v___x_439_; uint8_t v___x_440_; 
v___x_439_ = ((lean_object*)(lp_mathlib_commandSudoSet__option_______00__closed__1));
lean_inc(v_x_435_);
v___x_440_ = l_Lean_Syntax_isOfKind(v_x_435_, v___x_439_);
if (v___x_440_ == 0)
{
lean_object* v___x_441_; 
lean_dec(v_x_435_);
v___x_441_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg();
return v___x_441_;
}
else
{
lean_object* v___x_442_; lean_object* v_scopes_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v_opts_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_442_ = lean_st_ref_get(v_a_437_);
v_scopes_443_ = lean_ctor_get(v___x_442_, 2);
lean_inc(v_scopes_443_);
lean_dec(v___x_442_);
v___x_444_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_445_ = l_List_head_x21___redArg(v___x_444_, v_scopes_443_);
lean_dec(v_scopes_443_);
v_opts_446_ = lean_ctor_get(v___x_445_, 1);
lean_inc_ref(v_opts_446_);
lean_dec(v___x_445_);
v___x_447_ = lean_unsigned_to_nat(2u);
v___x_448_ = l_Lean_Syntax_getArg(v_x_435_, v___x_447_);
v___x_449_ = lean_unsigned_to_nat(4u);
v___x_450_ = l_Lean_Syntax_getArg(v_x_435_, v___x_449_);
lean_dec(v_x_435_);
v___x_451_ = lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1(v___x_448_, v___x_450_, v_opts_446_, v_a_436_, v_a_437_);
lean_dec(v___x_448_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v_a_452_; lean_object* v___x_453_; lean_object* v_env_454_; lean_object* v_messages_455_; lean_object* v_scopes_456_; lean_object* v_usedQuotCtxts_457_; lean_object* v_nextMacroScope_458_; lean_object* v_ngen_459_; lean_object* v_auxDeclNGen_460_; lean_object* v_infoState_461_; lean_object* v_traceState_462_; lean_object* v_snapshotTasks_463_; lean_object* v_prevLinterStates_464_; lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_476_; 
v_a_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_a_452_);
lean_dec_ref_known(v___x_451_, 1);
v___x_453_ = lean_st_ref_take(v_a_437_);
v_env_454_ = lean_ctor_get(v___x_453_, 0);
v_messages_455_ = lean_ctor_get(v___x_453_, 1);
v_scopes_456_ = lean_ctor_get(v___x_453_, 2);
v_usedQuotCtxts_457_ = lean_ctor_get(v___x_453_, 3);
v_nextMacroScope_458_ = lean_ctor_get(v___x_453_, 4);
v_ngen_459_ = lean_ctor_get(v___x_453_, 6);
v_auxDeclNGen_460_ = lean_ctor_get(v___x_453_, 7);
v_infoState_461_ = lean_ctor_get(v___x_453_, 8);
v_traceState_462_ = lean_ctor_get(v___x_453_, 9);
v_snapshotTasks_463_ = lean_ctor_get(v___x_453_, 10);
v_prevLinterStates_464_ = lean_ctor_get(v___x_453_, 11);
v_isSharedCheck_476_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_476_ == 0)
{
lean_object* v_unused_477_; 
v_unused_477_ = lean_ctor_get(v___x_453_, 5);
lean_dec(v_unused_477_);
v___x_466_ = v___x_453_;
v_isShared_467_ = v_isSharedCheck_476_;
goto v_resetjp_465_;
}
else
{
lean_inc(v_prevLinterStates_464_);
lean_inc(v_snapshotTasks_463_);
lean_inc(v_traceState_462_);
lean_inc(v_infoState_461_);
lean_inc(v_auxDeclNGen_460_);
lean_inc(v_ngen_459_);
lean_inc(v_nextMacroScope_458_);
lean_inc(v_usedQuotCtxts_457_);
lean_inc(v_scopes_456_);
lean_inc(v_messages_455_);
lean_inc(v_env_454_);
lean_dec(v___x_453_);
v___x_466_ = lean_box(0);
v_isShared_467_ = v_isSharedCheck_476_;
goto v_resetjp_465_;
}
v_resetjp_465_:
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_471_; 
v___x_468_ = l_Lean_maxRecDepth;
v___x_469_ = lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2(v_a_452_, v___x_468_);
if (v_isShared_467_ == 0)
{
lean_ctor_set(v___x_466_, 5, v___x_469_);
v___x_471_ = v___x_466_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_475_; 
v_reuseFailAlloc_475_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_475_, 0, v_env_454_);
lean_ctor_set(v_reuseFailAlloc_475_, 1, v_messages_455_);
lean_ctor_set(v_reuseFailAlloc_475_, 2, v_scopes_456_);
lean_ctor_set(v_reuseFailAlloc_475_, 3, v_usedQuotCtxts_457_);
lean_ctor_set(v_reuseFailAlloc_475_, 4, v_nextMacroScope_458_);
lean_ctor_set(v_reuseFailAlloc_475_, 5, v___x_469_);
lean_ctor_set(v_reuseFailAlloc_475_, 6, v_ngen_459_);
lean_ctor_set(v_reuseFailAlloc_475_, 7, v_auxDeclNGen_460_);
lean_ctor_set(v_reuseFailAlloc_475_, 8, v_infoState_461_);
lean_ctor_set(v_reuseFailAlloc_475_, 9, v_traceState_462_);
lean_ctor_set(v_reuseFailAlloc_475_, 10, v_snapshotTasks_463_);
lean_ctor_set(v_reuseFailAlloc_475_, 11, v_prevLinterStates_464_);
v___x_471_ = v_reuseFailAlloc_475_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
lean_object* v___x_472_; lean_object* v___f_473_; lean_object* v___x_474_; 
v___x_472_ = lean_st_ref_set(v_a_437_, v___x_471_);
v___f_473_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1___lam__0), 2, 1);
lean_closure_set(v___f_473_, 0, v_a_452_);
v___x_474_ = l_Lean_Elab_Command_modifyScope___redArg(v___f_473_, v_a_437_);
return v___x_474_;
}
}
}
else
{
lean_object* v_a_478_; lean_object* v___x_480_; uint8_t v_isShared_481_; uint8_t v_isSharedCheck_485_; 
v_a_478_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_485_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_485_ == 0)
{
v___x_480_ = v___x_451_;
v_isShared_481_ = v_isSharedCheck_485_;
goto v_resetjp_479_;
}
else
{
lean_inc(v_a_478_);
lean_dec(v___x_451_);
v___x_480_ = lean_box(0);
v_isShared_481_ = v_isSharedCheck_485_;
goto v_resetjp_479_;
}
v_resetjp_479_:
{
lean_object* v___x_483_; 
if (v_isShared_481_ == 0)
{
v___x_483_ = v___x_480_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_a_478_);
v___x_483_ = v_reuseFailAlloc_484_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
return v___x_483_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1___boxed(lean_object* v_x_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1(v_x_486_, v_a_487_, v_a_488_);
lean_dec(v_a_488_);
lean_dec_ref(v_a_487_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3(lean_object* v_msgData_491_, lean_object* v___y_492_, lean_object* v___y_493_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___redArg(v_msgData_491_, v___y_493_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__3(v_msgData_496_, v___y_497_, v___y_498_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
return v_res_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1(lean_object* v_00_u03b1_501_, lean_object* v_msg_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___redArg(v_msg_502_, v___y_503_, v___y_504_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1___boxed(lean_object* v_00_u03b1_507_, lean_object* v_msg_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1(v_00_u03b1_507_, v_msg_508_, v___y_509_, v___y_510_);
lean_dec(v___y_510_);
lean_dec_ref(v___y_509_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4(lean_object* v_msgData_513_, lean_object* v_macroStack_514_, lean_object* v___y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg(v_msgData_513_, v_macroStack_514_, v___y_516_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___boxed(lean_object* v_msgData_519_, lean_object* v_macroStack_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4(v_msgData_519_, v_macroStack_520_, v___y_521_, v___y_522_);
lean_dec(v___y_522_);
lean_dec_ref(v___y_521_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg(){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; 
v___x_545_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__0___redArg___closed__0);
v___x_546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg___boxed(lean_object* v___y_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg();
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0(lean_object* v_00_u03b1_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg();
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___boxed(lean_object* v_00_u03b1_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0(v_00_u03b1_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_, v___y_564_);
lean_dec(v___y_564_);
lean_dec_ref(v___y_563_);
lean_dec(v___y_562_);
lean_dec_ref(v___y_561_);
lean_dec(v___y_560_);
lean_dec_ref(v___y_559_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__2(lean_object* v_msgData_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_){
_start:
{
lean_object* v___x_573_; lean_object* v_env_574_; lean_object* v___x_575_; lean_object* v_mctx_576_; lean_object* v_lctx_577_; lean_object* v_options_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_573_ = lean_st_ref_get(v___y_571_);
v_env_574_ = lean_ctor_get(v___x_573_, 0);
lean_inc_ref(v_env_574_);
lean_dec(v___x_573_);
v___x_575_ = lean_st_ref_get(v___y_569_);
v_mctx_576_ = lean_ctor_get(v___x_575_, 0);
lean_inc_ref(v_mctx_576_);
lean_dec(v___x_575_);
v_lctx_577_ = lean_ctor_get(v___y_568_, 2);
v_options_578_ = lean_ctor_get(v___y_570_, 2);
lean_inc_ref(v_options_578_);
lean_inc_ref(v_lctx_577_);
v___x_579_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_579_, 0, v_env_574_);
lean_ctor_set(v___x_579_, 1, v_mctx_576_);
lean_ctor_set(v___x_579_, 2, v_lctx_577_);
lean_ctor_set(v___x_579_, 3, v_options_578_);
v___x_580_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_580_, 0, v___x_579_);
lean_ctor_set(v___x_580_, 1, v_msgData_567_);
v___x_581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_581_, 0, v___x_580_);
return v___x_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__2(v_msgData_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg(lean_object* v_msgData_589_, lean_object* v_macroStack_590_, lean_object* v___y_591_){
_start:
{
lean_object* v_options_593_; lean_object* v___x_594_; uint8_t v___x_595_; 
v_options_593_ = lean_ctor_get(v___y_591_, 2);
v___x_594_ = l_Lean_Elab_pp_macroStack;
v___x_595_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__5(v_options_593_, v___x_594_);
if (v___x_595_ == 0)
{
lean_object* v___x_596_; 
lean_dec(v_macroStack_590_);
v___x_596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_596_, 0, v_msgData_589_);
return v___x_596_;
}
else
{
if (lean_obj_tag(v_macroStack_590_) == 0)
{
lean_object* v___x_597_; 
v___x_597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_597_, 0, v_msgData_589_);
return v___x_597_;
}
else
{
lean_object* v_head_598_; lean_object* v_after_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_614_; 
v_head_598_ = lean_ctor_get(v_macroStack_590_, 0);
lean_inc(v_head_598_);
v_after_599_ = lean_ctor_get(v_head_598_, 1);
v_isSharedCheck_614_ = !lean_is_exclusive(v_head_598_);
if (v_isSharedCheck_614_ == 0)
{
lean_object* v_unused_615_; 
v_unused_615_ = lean_ctor_get(v_head_598_, 0);
lean_dec(v_unused_615_);
v___x_601_ = v_head_598_;
v_isShared_602_ = v_isSharedCheck_614_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_after_599_);
lean_dec(v_head_598_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_614_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_603_; lean_object* v___x_605_; 
v___x_603_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6___closed__0);
if (v_isShared_602_ == 0)
{
lean_ctor_set_tag(v___x_601_, 7);
lean_ctor_set(v___x_601_, 1, v___x_603_);
lean_ctor_set(v___x_601_, 0, v_msgData_589_);
v___x_605_ = v___x_601_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_msgData_589_);
lean_ctor_set(v_reuseFailAlloc_613_, 1, v___x_603_);
v___x_605_ = v_reuseFailAlloc_613_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v_msgData_610_; lean_object* v___x_611_; lean_object* v___x_612_; 
v___x_606_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4___redArg___closed__2);
v___x_607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_605_);
lean_ctor_set(v___x_607_, 1, v___x_606_);
v___x_608_ = l_Lean_MessageData_ofSyntax(v_after_599_);
v___x_609_ = l_Lean_indentD(v___x_608_);
v_msgData_610_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_610_, 0, v___x_607_);
lean_ctor_set(v_msgData_610_, 1, v___x_609_);
v___x_611_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__1_spec__1_spec__4_spec__6(v_msgData_610_, v_macroStack_590_);
v___x_612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_612_, 0, v___x_611_);
return v___x_612_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_616_, lean_object* v_macroStack_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg(v_msgData_616_, v_macroStack_617_, v___y_618_);
lean_dec_ref(v___y_618_);
return v_res_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg(lean_object* v_msg_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_){
_start:
{
lean_object* v_ref_629_; lean_object* v___x_630_; lean_object* v_a_631_; lean_object* v_macroStack_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_643_; 
v_ref_629_ = lean_ctor_get(v___y_626_, 5);
v___x_630_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__2(v_msg_621_, v___y_624_, v___y_625_, v___y_626_, v___y_627_);
v_a_631_ = lean_ctor_get(v___x_630_, 0);
lean_inc(v_a_631_);
lean_dec_ref(v___x_630_);
v_macroStack_632_ = lean_ctor_get(v___y_622_, 1);
v___x_633_ = l_Lean_Elab_getBetterRef(v_ref_629_, v_macroStack_632_);
lean_inc(v_macroStack_632_);
v___x_634_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg(v_a_631_, v_macroStack_632_, v___y_626_);
v_a_635_ = lean_ctor_get(v___x_634_, 0);
v_isSharedCheck_643_ = !lean_is_exclusive(v___x_634_);
if (v_isSharedCheck_643_ == 0)
{
v___x_637_ = v___x_634_;
v_isShared_638_ = v_isSharedCheck_643_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_634_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_643_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_639_; lean_object* v___x_641_; 
v___x_639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_639_, 0, v___x_633_);
lean_ctor_set(v___x_639_, 1, v_a_635_);
if (v_isShared_638_ == 0)
{
lean_ctor_set_tag(v___x_637_, 1);
lean_ctor_set(v___x_637_, 0, v___x_639_);
v___x_641_ = v___x_637_;
goto v_reusejp_640_;
}
else
{
lean_object* v_reuseFailAlloc_642_; 
v_reuseFailAlloc_642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_642_, 0, v___x_639_);
v___x_641_ = v_reuseFailAlloc_642_;
goto v_reusejp_640_;
}
v_reusejp_640_:
{
return v___x_641_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg___boxed(lean_object* v_msg_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_){
_start:
{
lean_object* v_res_652_; 
v_res_652_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg(v_msg_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_);
lean_dec(v___y_650_);
lean_dec_ref(v___y_649_);
lean_dec(v___y_648_);
lean_dec_ref(v___y_647_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
return v_res_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1(lean_object* v_name_653_, lean_object* v_val_654_, lean_object* v_opts_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v_val_664_; 
if (lean_obj_tag(v_val_654_) == 3)
{
lean_object* v_val_714_; 
v_val_714_ = lean_ctor_get(v_val_654_, 2);
if (lean_obj_tag(v_val_714_) == 1)
{
lean_object* v_pre_715_; 
v_pre_715_ = lean_ctor_get(v_val_714_, 0);
if (lean_obj_tag(v_pre_715_) == 0)
{
lean_object* v_str_716_; lean_object* v___x_717_; uint8_t v___x_718_; 
v_str_716_ = lean_ctor_get(v_val_714_, 1);
v___x_717_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__2));
v___x_718_ = lean_string_dec_eq(v_str_716_, v___x_717_);
if (v___x_718_ == 0)
{
lean_object* v___x_719_; uint8_t v___x_720_; 
v___x_719_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__3));
v___x_720_ = lean_string_dec_eq(v_str_716_, v___x_719_);
if (v___x_720_ == 0)
{
goto v___jp_683_;
}
else
{
lean_object* v___x_721_; 
lean_dec_ref_known(v_val_654_, 4);
v___x_721_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_721_, 0, v___x_718_);
v_val_664_ = v___x_721_;
goto v___jp_663_;
}
}
else
{
lean_object* v___x_722_; 
lean_dec_ref_known(v_val_654_, 4);
v___x_722_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_722_, 0, v___x_718_);
v_val_664_ = v___x_722_;
goto v___jp_663_;
}
}
else
{
goto v___jp_683_;
}
}
else
{
goto v___jp_683_;
}
}
else
{
goto v___jp_683_;
}
v___jp_663_:
{
lean_object* v_map_665_; uint8_t v_hasTrace_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_682_; 
v_map_665_ = lean_ctor_get(v_opts_655_, 0);
v_hasTrace_666_ = lean_ctor_get_uint8(v_opts_655_, sizeof(void*)*1);
v_isSharedCheck_682_ = !lean_is_exclusive(v_opts_655_);
if (v_isSharedCheck_682_ == 0)
{
v___x_668_ = v_opts_655_;
v_isShared_669_ = v_isSharedCheck_682_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_map_665_);
lean_dec(v_opts_655_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_682_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_670_ = l_Lean_Syntax_getId(v_name_653_);
lean_inc(v___x_670_);
v___x_671_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_670_, v_val_664_, v_map_665_);
if (v_hasTrace_666_ == 0)
{
lean_object* v___x_672_; uint8_t v___x_673_; lean_object* v___x_675_; 
v___x_672_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___lam__0___closed__1));
v___x_673_ = l_Lean_Name_isPrefixOf(v___x_672_, v___x_670_);
lean_dec(v___x_670_);
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 0, v___x_671_);
v___x_675_ = v___x_668_;
goto v_reusejp_674_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v___x_671_);
v___x_675_ = v_reuseFailAlloc_677_;
goto v_reusejp_674_;
}
v_reusejp_674_:
{
lean_object* v___x_676_; 
lean_ctor_set_uint8(v___x_675_, sizeof(void*)*1, v___x_673_);
v___x_676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_676_, 0, v___x_675_);
return v___x_676_;
}
}
else
{
lean_object* v___x_679_; 
lean_dec(v___x_670_);
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 0, v___x_671_);
v___x_679_ = v___x_668_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_681_; 
v_reuseFailAlloc_681_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_681_, 0, v___x_671_);
lean_ctor_set_uint8(v_reuseFailAlloc_681_, sizeof(void*)*1, v_hasTrace_666_);
v___x_679_ = v_reuseFailAlloc_681_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
lean_object* v___x_680_; 
v___x_680_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_680_, 0, v___x_679_);
return v___x_680_;
}
}
}
}
v___jp_683_:
{
lean_object* v___x_684_; 
v___x_684_ = l_Lean_Syntax_isNatLit_x3f(v_val_654_);
if (lean_obj_tag(v___x_684_) == 0)
{
lean_object* v___x_685_; 
v___x_685_ = l_Lean_Syntax_isStrLit_x3f(v_val_654_);
if (lean_obj_tag(v___x_685_) == 0)
{
lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
lean_dec_ref(v_opts_655_);
v___x_686_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___redArg___closed__1);
v___x_687_ = l_Lean_MessageData_ofSyntax(v_val_654_);
v___x_688_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_688_, 0, v___x_686_);
lean_ctor_set(v___x_688_, 1, v___x_687_);
v___x_689_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg(v___x_688_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_);
v_a_690_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_689_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_689_);
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
else
{
lean_object* v_val_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_705_; 
lean_dec(v_val_654_);
v_val_698_ = lean_ctor_get(v___x_685_, 0);
v_isSharedCheck_705_ = !lean_is_exclusive(v___x_685_);
if (v_isSharedCheck_705_ == 0)
{
v___x_700_ = v___x_685_;
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_val_698_);
lean_dec(v___x_685_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_703_; 
if (v_isShared_701_ == 0)
{
lean_ctor_set_tag(v___x_700_, 0);
v___x_703_ = v___x_700_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v_val_698_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
v_val_664_ = v___x_703_;
goto v___jp_663_;
}
}
}
}
else
{
lean_object* v_val_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_713_; 
lean_dec(v_val_654_);
v_val_706_ = lean_ctor_get(v___x_684_, 0);
v_isSharedCheck_713_ = !lean_is_exclusive(v___x_684_);
if (v_isSharedCheck_713_ == 0)
{
v___x_708_ = v___x_684_;
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_val_706_);
lean_dec(v___x_684_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
lean_object* v___x_711_; 
if (v_isShared_709_ == 0)
{
lean_ctor_set_tag(v___x_708_, 3);
v___x_711_ = v___x_708_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v_val_706_);
v___x_711_ = v_reuseFailAlloc_712_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
v_val_664_ = v___x_711_;
goto v___jp_663_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1___boxed(lean_object* v_name_723_, lean_object* v_val_724_, lean_object* v_opts_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1(v_name_723_, v_val_724_, v_opts_725_, v___y_726_, v___y_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
lean_dec(v___y_727_);
lean_dec_ref(v___y_726_);
lean_dec(v_name_723_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___lam__0(lean_object* v_stx_734_, lean_object* v_expectedType_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_){
_start:
{
lean_object* v___x_743_; uint8_t v___x_744_; 
v___x_743_ = ((lean_object*)(lp_mathlib_termSudoSet__option______In___00__closed__1));
lean_inc(v_stx_734_);
v___x_744_ = l_Lean_Syntax_isOfKind(v_stx_734_, v___x_743_);
if (v___x_744_ == 0)
{
lean_object* v___x_745_; 
lean_dec_ref(v_expectedType_735_);
lean_dec(v_stx_734_);
v___x_745_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__0___redArg();
return v___x_745_;
}
else
{
lean_object* v_fileName_746_; lean_object* v_fileMap_747_; lean_object* v_options_748_; lean_object* v_currRecDepth_749_; lean_object* v_ref_750_; lean_object* v_currNamespace_751_; lean_object* v_openDecls_752_; lean_object* v_initHeartbeats_753_; lean_object* v_maxHeartbeats_754_; lean_object* v_quotContext_755_; lean_object* v_currMacroScope_756_; uint8_t v_diag_757_; lean_object* v_cancelTk_x3f_758_; uint8_t v_suppressElabErrors_759_; lean_object* v_inheritedTraceOptions_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; 
v_fileName_746_ = lean_ctor_get(v___y_740_, 0);
v_fileMap_747_ = lean_ctor_get(v___y_740_, 1);
v_options_748_ = lean_ctor_get(v___y_740_, 2);
v_currRecDepth_749_ = lean_ctor_get(v___y_740_, 3);
v_ref_750_ = lean_ctor_get(v___y_740_, 5);
v_currNamespace_751_ = lean_ctor_get(v___y_740_, 6);
v_openDecls_752_ = lean_ctor_get(v___y_740_, 7);
v_initHeartbeats_753_ = lean_ctor_get(v___y_740_, 8);
v_maxHeartbeats_754_ = lean_ctor_get(v___y_740_, 9);
v_quotContext_755_ = lean_ctor_get(v___y_740_, 10);
v_currMacroScope_756_ = lean_ctor_get(v___y_740_, 11);
v_diag_757_ = lean_ctor_get_uint8(v___y_740_, sizeof(void*)*14);
v_cancelTk_x3f_758_ = lean_ctor_get(v___y_740_, 12);
v_suppressElabErrors_759_ = lean_ctor_get_uint8(v___y_740_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_760_ = lean_ctor_get(v___y_740_, 13);
v___x_761_ = lean_unsigned_to_nat(2u);
v___x_762_ = l_Lean_Syntax_getArg(v_stx_734_, v___x_761_);
v___x_763_ = lean_unsigned_to_nat(4u);
v___x_764_ = l_Lean_Syntax_getArg(v_stx_734_, v___x_763_);
lean_inc_ref(v_options_748_);
v___x_765_ = lp_mathlib___private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1(v___x_762_, v___x_764_, v_options_748_, v___y_736_, v___y_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_);
lean_dec(v___x_762_);
if (lean_obj_tag(v___x_765_) == 0)
{
lean_object* v_a_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; 
v_a_766_ = lean_ctor_get(v___x_765_, 0);
lean_inc(v_a_766_);
lean_dec_ref_known(v___x_765_, 1);
v___x_767_ = lean_unsigned_to_nat(6u);
v___x_768_ = l_Lean_Syntax_getArg(v_stx_734_, v___x_767_);
lean_dec(v_stx_734_);
v___x_769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_769_, 0, v_expectedType_735_);
v___x_770_ = l_Lean_maxRecDepth;
v___x_771_ = lp_mathlib_Lean_Option_get___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__commandSudoSet__option________1_spec__2(v_a_766_, v___x_770_);
lean_inc_ref(v_inheritedTraceOptions_760_);
lean_inc(v_cancelTk_x3f_758_);
lean_inc(v_currMacroScope_756_);
lean_inc(v_quotContext_755_);
lean_inc(v_maxHeartbeats_754_);
lean_inc(v_initHeartbeats_753_);
lean_inc(v_openDecls_752_);
lean_inc(v_currNamespace_751_);
lean_inc(v_ref_750_);
lean_inc(v_currRecDepth_749_);
lean_inc_ref(v_fileMap_747_);
lean_inc_ref(v_fileName_746_);
v___x_772_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_772_, 0, v_fileName_746_);
lean_ctor_set(v___x_772_, 1, v_fileMap_747_);
lean_ctor_set(v___x_772_, 2, v_a_766_);
lean_ctor_set(v___x_772_, 3, v_currRecDepth_749_);
lean_ctor_set(v___x_772_, 4, v___x_771_);
lean_ctor_set(v___x_772_, 5, v_ref_750_);
lean_ctor_set(v___x_772_, 6, v_currNamespace_751_);
lean_ctor_set(v___x_772_, 7, v_openDecls_752_);
lean_ctor_set(v___x_772_, 8, v_initHeartbeats_753_);
lean_ctor_set(v___x_772_, 9, v_maxHeartbeats_754_);
lean_ctor_set(v___x_772_, 10, v_quotContext_755_);
lean_ctor_set(v___x_772_, 11, v_currMacroScope_756_);
lean_ctor_set(v___x_772_, 12, v_cancelTk_x3f_758_);
lean_ctor_set(v___x_772_, 13, v_inheritedTraceOptions_760_);
lean_ctor_set_uint8(v___x_772_, sizeof(void*)*14, v_diag_757_);
lean_ctor_set_uint8(v___x_772_, sizeof(void*)*14 + 1, v_suppressElabErrors_759_);
v___x_773_ = l_Lean_Elab_Term_elabTerm(v___x_768_, v___x_769_, v___x_744_, v___x_744_, v___y_736_, v___y_737_, v___y_738_, v___y_739_, v___x_772_, v___y_741_);
lean_dec_ref_known(v___x_772_, 14);
return v___x_773_;
}
else
{
lean_object* v_a_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_781_; 
lean_dec_ref(v_expectedType_735_);
lean_dec(v_stx_734_);
v_a_774_ = lean_ctor_get(v___x_765_, 0);
v_isSharedCheck_781_ = !lean_is_exclusive(v___x_765_);
if (v_isSharedCheck_781_ == 0)
{
v___x_776_ = v___x_765_;
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_a_774_);
lean_dec(v___x_765_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_779_; 
if (v_isShared_777_ == 0)
{
v___x_779_ = v___x_776_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v_a_774_);
v___x_779_ = v_reuseFailAlloc_780_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
return v___x_779_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___lam__0___boxed(lean_object* v_stx_782_, lean_object* v_expectedType_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_){
_start:
{
lean_object* v_res_791_; 
v_res_791_ = lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___lam__0(v_stx_782_, v_expectedType_783_, v___y_784_, v___y_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_);
lean_dec(v___y_789_);
lean_dec_ref(v___y_788_);
lean_dec(v___y_787_);
lean_dec_ref(v___y_786_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
return v_res_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1(lean_object* v_stx_792_, lean_object* v_expectedType_x3f_793_, lean_object* v_a_794_, lean_object* v_a_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_){
_start:
{
lean_object* v___f_801_; lean_object* v___x_802_; 
v___f_801_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___lam__0___boxed), 9, 1);
lean_closure_set(v___f_801_, 0, v_stx_792_);
v___x_802_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_793_, v___f_801_, v_a_794_, v_a_795_, v_a_796_, v_a_797_, v_a_798_, v_a_799_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1___boxed(lean_object* v_stx_803_, lean_object* v_expectedType_x3f_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_){
_start:
{
lean_object* v_res_812_; 
v_res_812_ = lp_mathlib___aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1(v_stx_803_, v_expectedType_x3f_804_, v_a_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_, v_a_810_);
lean_dec(v_a_810_);
lean_dec_ref(v_a_809_);
lean_dec(v_a_808_);
lean_dec_ref(v_a_807_);
lean_dec(v_a_806_);
lean_dec_ref(v_a_805_);
return v_res_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1(lean_object* v_00_u03b1_813_, lean_object* v_msg_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
lean_object* v___x_822_; 
v___x_822_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___redArg(v_msg_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1___boxed(lean_object* v_00_u03b1_823_, lean_object* v_msg_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1(v_00_u03b1_823_, v_msg_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_, v___y_830_);
lean_dec(v___y_830_);
lean_dec_ref(v___y_829_);
lean_dec(v___y_828_);
lean_dec_ref(v___y_827_);
lean_dec(v___y_826_);
lean_dec_ref(v___y_825_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3(lean_object* v_msgData_833_, lean_object* v_macroStack_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_){
_start:
{
lean_object* v___x_842_; 
v___x_842_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___redArg(v_msgData_833_, v_macroStack_834_, v___y_839_);
return v___x_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_843_, lean_object* v_macroStack_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_SudoSetOption_0__setOption___at___00__aux__Mathlib__Tactic__SudoSetOption______elabRules__termSudoSet__option______In____1_spec__1_spec__1_spec__3(v_msgData_843_, v_macroStack_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
return v_res_852_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SudoSetOption(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_SudoSetOption(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_SudoSetOption(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SudoSetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_SudoSetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_SudoSetOption(builtin);
}
#ifdef __cplusplus
}
#endif
