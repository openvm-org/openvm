// Lean compiler output
// Module: Mathlib.Tactic.Observe
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.TryThis public meta import Lean.Elab.Tactic.ElabTerm public meta import Lean.Meta.Tactic.LibrarySearch
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Meta_LibrarySearch_solveByElim(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTag___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermWithHoles(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_LibrarySearch_librarySearch(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_reportOutOfHeartbeats(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkMVar(lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_note(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Tactic_TryThis_addHaveSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "LibrarySearch"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "observe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__2_value),LEAN_SCALAR_PTR_LITERAL(11, 232, 222, 97, 221, 61, 193, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3_value),LEAN_SCALAR_PTR_LITERAL(220, 141, 141, 211, 33, 7, 255, 9)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__14_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__17_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__26_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__25_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__32_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__33_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__34_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__35_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__36_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__38_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__31_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__39_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__29_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__42_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__43_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_observe = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__43_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "library_search"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(38, 188, 18, 2, 33, 114, 181, 46)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "observe did not find a solution"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "this"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(38, 116, 214, 236, 212, 160, 188, 150)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticObserve\?__:_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__2_value),LEAN_SCALAR_PTR_LITERAL(11, 232, 222, 97, 221, 61, 193, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(117, 32, 198, 172, 85, 49, 88, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "observe\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "tacticObserve\?__:_Using__,,"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__2_value),LEAN_SCALAR_PTR_LITERAL(11, 232, 222, 97, 221, 61, 193, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(102, 6, 131, 230, 91, 32, 42, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__39_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "using"};
static const lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_104_ = lean_box(0);
v___x_105_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_106_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v___x_104_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg(){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___closed__0);
v___x_109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg___boxed(lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0(lean_object* v_00_u03b1_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___boxed(lean_object* v_00_u03b1_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0(v_00_u03b1_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_, v___y_131_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg(lean_object* v_e_134_, lean_object* v___y_135_){
_start:
{
uint8_t v___x_137_; 
v___x_137_ = l_Lean_Expr_hasMVar(v_e_134_);
if (v___x_137_ == 0)
{
lean_object* v___x_138_; 
v___x_138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_138_, 0, v_e_134_);
return v___x_138_;
}
else
{
lean_object* v___x_139_; lean_object* v_mctx_140_; lean_object* v___x_141_; lean_object* v_fst_142_; lean_object* v_snd_143_; lean_object* v___x_144_; lean_object* v_cache_145_; lean_object* v_zetaDeltaFVarIds_146_; lean_object* v_postponed_147_; lean_object* v_diag_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_157_; 
v___x_139_ = lean_st_ref_get(v___y_135_);
v_mctx_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc_ref(v_mctx_140_);
lean_dec(v___x_139_);
v___x_141_ = l_Lean_instantiateMVarsCore(v_mctx_140_, v_e_134_);
v_fst_142_ = lean_ctor_get(v___x_141_, 0);
lean_inc(v_fst_142_);
v_snd_143_ = lean_ctor_get(v___x_141_, 1);
lean_inc(v_snd_143_);
lean_dec_ref(v___x_141_);
v___x_144_ = lean_st_ref_take(v___y_135_);
v_cache_145_ = lean_ctor_get(v___x_144_, 1);
v_zetaDeltaFVarIds_146_ = lean_ctor_get(v___x_144_, 2);
v_postponed_147_ = lean_ctor_get(v___x_144_, 3);
v_diag_148_ = lean_ctor_get(v___x_144_, 4);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_157_ == 0)
{
lean_object* v_unused_158_; 
v_unused_158_ = lean_ctor_get(v___x_144_, 0);
lean_dec(v_unused_158_);
v___x_150_ = v___x_144_;
v_isShared_151_ = v_isSharedCheck_157_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_diag_148_);
lean_inc(v_postponed_147_);
lean_inc(v_zetaDeltaFVarIds_146_);
lean_inc(v_cache_145_);
lean_dec(v___x_144_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_157_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
if (v_isShared_151_ == 0)
{
lean_ctor_set(v___x_150_, 0, v_snd_143_);
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v_snd_143_);
lean_ctor_set(v_reuseFailAlloc_156_, 1, v_cache_145_);
lean_ctor_set(v_reuseFailAlloc_156_, 2, v_zetaDeltaFVarIds_146_);
lean_ctor_set(v_reuseFailAlloc_156_, 3, v_postponed_147_);
lean_ctor_set(v_reuseFailAlloc_156_, 4, v_diag_148_);
v___x_153_ = v_reuseFailAlloc_156_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_154_ = lean_st_ref_set(v___y_135_, v___x_153_);
v___x_155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_155_, 0, v_fst_142_);
return v___x_155_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg___boxed(lean_object* v_e_159_, lean_object* v___y_160_, lean_object* v___y_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg(v_e_159_, v___y_160_);
lean_dec(v___y_160_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2(lean_object* v_e_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg(v_e_163_, v___y_169_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___boxed(lean_object* v_e_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2(v_e_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__0(lean_object* v_g_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_){
_start:
{
lean_object* v___x_191_; uint8_t v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_191_ = lean_box(0);
v___x_192_ = 0;
v___x_193_ = lean_unsigned_to_nat(6u);
v___x_194_ = l_Lean_Meta_LibrarySearch_solveByElim(v___x_191_, v___x_192_, v_g_185_, v___x_193_, v___x_192_, v___x_192_, v___y_186_, v___y_187_, v___y_188_, v___y_189_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__0___boxed(lean_object* v_g_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__0(v_g_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__1(uint8_t v___x_202_, lean_object* v_x_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = lean_box(v___x_202_);
v___x_210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_210_, 0, v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__1___boxed(lean_object* v___x_211_, lean_object* v_x_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_){
_start:
{
uint8_t v___x_9026__boxed_218_; lean_object* v_res_219_; 
v___x_9026__boxed_218_ = lean_unbox(v___x_211_);
v_res_219_ = lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__1(v___x_9026__boxed_218_, v_x_212_, v___y_213_, v___y_214_, v___y_215_, v___y_216_);
lean_dec(v___y_216_);
lean_dec_ref(v___y_215_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v_x_212_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1_spec__1(lean_object* v_msgData_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
lean_object* v___x_226_; lean_object* v_env_227_; lean_object* v___x_228_; lean_object* v_mctx_229_; lean_object* v_lctx_230_; lean_object* v_options_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_226_ = lean_st_ref_get(v___y_224_);
v_env_227_ = lean_ctor_get(v___x_226_, 0);
lean_inc_ref(v_env_227_);
lean_dec(v___x_226_);
v___x_228_ = lean_st_ref_get(v___y_222_);
v_mctx_229_ = lean_ctor_get(v___x_228_, 0);
lean_inc_ref(v_mctx_229_);
lean_dec(v___x_228_);
v_lctx_230_ = lean_ctor_get(v___y_221_, 2);
v_options_231_ = lean_ctor_get(v___y_223_, 2);
lean_inc_ref(v_options_231_);
lean_inc_ref(v_lctx_230_);
v___x_232_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_232_, 0, v_env_227_);
lean_ctor_set(v___x_232_, 1, v_mctx_229_);
lean_ctor_set(v___x_232_, 2, v_lctx_230_);
lean_ctor_set(v___x_232_, 3, v_options_231_);
v___x_233_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
lean_ctor_set(v___x_233_, 1, v_msgData_220_);
v___x_234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1_spec__1___boxed(lean_object* v_msgData_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1_spec__1(v_msgData_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
lean_dec(v___y_239_);
lean_dec_ref(v___y_238_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg(lean_object* v_msg_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v_ref_248_; lean_object* v___x_249_; lean_object* v_a_250_; lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_258_; 
v_ref_248_ = lean_ctor_get(v___y_245_, 5);
v___x_249_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1_spec__1(v_msg_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
v_a_250_ = lean_ctor_get(v___x_249_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_249_);
if (v_isSharedCheck_258_ == 0)
{
v___x_252_ = v___x_249_;
v_isShared_253_ = v_isSharedCheck_258_;
goto v_resetjp_251_;
}
else
{
lean_inc(v_a_250_);
lean_dec(v___x_249_);
v___x_252_ = lean_box(0);
v_isShared_253_ = v_isSharedCheck_258_;
goto v_resetjp_251_;
}
v_resetjp_251_:
{
lean_object* v___x_254_; lean_object* v___x_256_; 
lean_inc(v_ref_248_);
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, v_ref_248_);
lean_ctor_set(v___x_254_, 1, v_a_250_);
if (v_isShared_253_ == 0)
{
lean_ctor_set_tag(v___x_252_, 1);
lean_ctor_set(v___x_252_, 0, v___x_254_);
v___x_256_ = v___x_252_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v___x_254_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg___boxed(lean_object* v_msg_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg(v_msg_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
return v_res_265_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__3(void){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__2));
v___x_271_ = l_Lean_stringToMessageData(v___x_270_);
return v___x_271_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__5(void){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__4));
v___x_274_ = l_Lean_stringToMessageData(v___x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2(lean_object* v___x_275_, uint8_t v___x_276_, lean_object* v___f_277_, lean_object* v___f_278_, lean_object* v_tk_279_, lean_object* v___y_280_, lean_object* v_trace_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = l_Lean_Elab_Tactic_getMainTag___redArg(v___y_283_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
if (lean_obj_tag(v___x_291_) == 0)
{
lean_object* v_a_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
v_a_292_ = lean_ctor_get(v___x_291_, 0);
lean_inc(v_a_292_);
lean_dec_ref_known(v___x_291_, 1);
v___x_293_ = lean_box(0);
v___x_294_ = l_Lean_Elab_Tactic_elabTermWithHoles(v___x_275_, v___x_293_, v_a_292_, v___x_276_, v___x_293_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
if (lean_obj_tag(v___x_294_) == 0)
{
lean_object* v_a_295_; lean_object* v_fst_296_; lean_object* v___x_297_; uint8_t v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v_a_295_ = lean_ctor_get(v___x_294_, 0);
lean_inc(v_a_295_);
lean_dec_ref_known(v___x_294_, 1);
v_fst_296_ = lean_ctor_get(v_a_295_, 0);
lean_inc(v_fst_296_);
lean_dec(v_a_295_);
v___x_297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_297_, 0, v_fst_296_);
v___x_298_ = 0;
v___x_299_ = lean_box(0);
lean_inc_ref(v___x_297_);
v___x_300_ = l_Lean_Meta_mkFreshExprMVar(v___x_297_, v___x_298_, v___x_299_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
if (lean_obj_tag(v___x_300_) == 0)
{
lean_object* v_a_301_; 
v_a_301_ = lean_ctor_get(v___x_300_, 0);
lean_inc(v_a_301_);
lean_dec_ref_known(v___x_300_, 1);
if (lean_obj_tag(v_a_301_) == 2)
{
lean_object* v_mvarId_302_; lean_object* v___x_303_; uint8_t v___x_304_; lean_object* v___x_305_; 
v_mvarId_302_ = lean_ctor_get(v_a_301_, 0);
lean_inc_n(v_mvarId_302_, 2);
lean_dec_ref_known(v_a_301_, 1);
v___x_303_ = lean_unsigned_to_nat(10u);
v___x_304_ = 0;
v___x_305_ = l_Lean_Meta_LibrarySearch_librarySearch(v_mvarId_302_, v___f_277_, v___f_278_, v___x_303_, v___x_276_, v___x_304_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v_a_306_; 
v_a_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_a_306_);
lean_dec_ref_known(v___x_305_, 1);
if (lean_obj_tag(v_a_306_) == 1)
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
lean_dec_ref_known(v_a_306_, 1);
lean_dec(v_mvarId_302_);
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v_trace_281_);
lean_dec(v___y_280_);
v___x_307_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__1));
v___x_308_ = lean_unsigned_to_nat(90u);
v___x_309_ = l_Lean_reportOutOfHeartbeats(v___x_307_, v_tk_279_, v___x_308_, v___y_288_, v___y_289_);
lean_dec(v_tk_279_);
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v___x_310_; lean_object* v___x_311_; 
lean_dec_ref_known(v___x_309_, 1);
v___x_310_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__3, &lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__3);
v___x_311_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg(v___x_310_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
return v___x_311_;
}
else
{
return v___x_309_;
}
}
else
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v_a_314_; lean_object* v___x_316_; uint8_t v_isShared_317_; uint8_t v_isSharedCheck_378_; 
lean_dec(v_a_306_);
v___x_312_ = l_Lean_mkMVar(v_mvarId_302_);
v___x_313_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__2___redArg(v___x_312_, v___y_287_);
v_a_314_ = lean_ctor_get(v___x_313_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_313_);
if (v_isSharedCheck_378_ == 0)
{
v___x_316_ = v___x_313_;
v_isShared_317_ = v_isSharedCheck_378_;
goto v_resetjp_315_;
}
else
{
lean_inc(v_a_314_);
lean_dec(v___x_313_);
v___x_316_ = lean_box(0);
v_isShared_317_ = v_isSharedCheck_378_;
goto v_resetjp_315_;
}
v_resetjp_315_:
{
lean_object* v___x_318_; lean_object* v___y_320_; lean_object* v___y_321_; lean_object* v___y_322_; lean_object* v___y_323_; lean_object* v___y_324_; 
v___x_318_ = l_Lean_Expr_headBeta(v_a_314_);
if (lean_obj_tag(v_trace_281_) == 0)
{
lean_del_object(v___x_316_);
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v_tk_279_);
v___y_320_ = v___y_283_;
v___y_321_ = v___y_286_;
v___y_322_ = v___y_287_;
v___y_323_ = v___y_288_;
v___y_324_ = v___y_289_;
goto v___jp_319_;
}
else
{
lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_376_; 
v_isSharedCheck_376_ = !lean_is_exclusive(v_trace_281_);
if (v_isSharedCheck_376_ == 0)
{
lean_object* v_unused_377_; 
v_unused_377_ = lean_ctor_get(v_trace_281_, 0);
lean_dec(v_unused_377_);
v___x_357_ = v_trace_281_;
v_isShared_358_ = v_isSharedCheck_376_;
goto v_resetjp_356_;
}
else
{
lean_dec(v_trace_281_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_376_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
if (v___x_276_ == 0)
{
lean_del_object(v___x_357_);
lean_del_object(v___x_316_);
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v_tk_279_);
v___y_320_ = v___y_283_;
v___y_321_ = v___y_286_;
v___y_322_ = v___y_287_;
v___y_323_ = v___y_288_;
v___y_324_ = v___y_289_;
goto v___jp_319_;
}
else
{
lean_object* v___x_359_; 
v___x_359_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_283_, v___y_285_, v___y_287_, v___y_289_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v_a_360_; lean_object* v___x_362_; 
v_a_360_ = lean_ctor_get(v___x_359_, 0);
lean_inc(v_a_360_);
lean_dec_ref_known(v___x_359_, 1);
lean_inc(v___y_280_);
if (v_isShared_358_ == 0)
{
lean_ctor_set(v___x_357_, 0, v___y_280_);
v___x_362_ = v___x_357_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v___y_280_);
v___x_362_ = v_reuseFailAlloc_367_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_364_; 
if (v_isShared_317_ == 0)
{
lean_ctor_set_tag(v___x_316_, 1);
lean_ctor_set(v___x_316_, 0, v_a_360_);
v___x_364_ = v___x_316_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_a_360_);
v___x_364_ = v_reuseFailAlloc_366_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
lean_object* v___x_365_; 
lean_inc_ref(v___x_318_);
v___x_365_ = l_Lean_Meta_Tactic_TryThis_addHaveSuggestion(v_tk_279_, v___x_362_, v___x_297_, v___x_318_, v___x_293_, v___x_364_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_dec_ref_known(v___x_365_, 1);
v___y_320_ = v___y_283_;
v___y_321_ = v___y_286_;
v___y_322_ = v___y_287_;
v___y_323_ = v___y_288_;
v___y_324_ = v___y_289_;
goto v___jp_319_;
}
else
{
lean_dec_ref(v___x_318_);
lean_dec(v___y_280_);
return v___x_365_;
}
}
}
}
else
{
lean_object* v_a_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_375_; 
lean_del_object(v___x_357_);
lean_dec_ref(v___x_318_);
lean_del_object(v___x_316_);
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v___y_280_);
lean_dec(v_tk_279_);
v_a_368_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_375_ == 0)
{
v___x_370_ = v___x_359_;
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_a_368_);
lean_dec(v___x_359_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
lean_object* v___x_373_; 
if (v_isShared_371_ == 0)
{
v___x_373_ = v___x_370_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_a_368_);
v___x_373_ = v_reuseFailAlloc_374_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
return v___x_373_;
}
}
}
}
}
}
v___jp_319_:
{
lean_object* v___x_325_; 
v___x_325_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
if (lean_obj_tag(v___x_325_) == 0)
{
lean_object* v_a_326_; lean_object* v___x_327_; 
v_a_326_ = lean_ctor_get(v___x_325_, 0);
lean_inc(v_a_326_);
lean_dec_ref_known(v___x_325_, 1);
v___x_327_ = l_Lean_MVarId_note(v_a_326_, v___y_280_, v___x_318_, v___x_293_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
if (lean_obj_tag(v___x_327_) == 0)
{
lean_object* v_a_328_; lean_object* v_snd_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_338_; 
v_a_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_a_328_);
lean_dec_ref_known(v___x_327_, 1);
v_snd_329_ = lean_ctor_get(v_a_328_, 1);
v_isSharedCheck_338_ = !lean_is_exclusive(v_a_328_);
if (v_isSharedCheck_338_ == 0)
{
lean_object* v_unused_339_; 
v_unused_339_ = lean_ctor_get(v_a_328_, 0);
lean_dec(v_unused_339_);
v___x_331_ = v_a_328_;
v_isShared_332_ = v_isSharedCheck_338_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_snd_329_);
lean_dec(v_a_328_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_338_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_333_ = lean_box(0);
if (v_isShared_332_ == 0)
{
lean_ctor_set_tag(v___x_331_, 1);
lean_ctor_set(v___x_331_, 1, v___x_333_);
lean_ctor_set(v___x_331_, 0, v_snd_329_);
v___x_335_ = v___x_331_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v_snd_329_);
lean_ctor_set(v_reuseFailAlloc_337_, 1, v___x_333_);
v___x_335_ = v_reuseFailAlloc_337_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_336_; 
v___x_336_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_335_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
return v___x_336_;
}
}
}
else
{
lean_object* v_a_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_347_; 
v_a_340_ = lean_ctor_get(v___x_327_, 0);
v_isSharedCheck_347_ = !lean_is_exclusive(v___x_327_);
if (v_isSharedCheck_347_ == 0)
{
v___x_342_ = v___x_327_;
v_isShared_343_ = v_isSharedCheck_347_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_a_340_);
lean_dec(v___x_327_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_347_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
lean_object* v___x_345_; 
if (v_isShared_343_ == 0)
{
v___x_345_ = v___x_342_;
goto v_reusejp_344_;
}
else
{
lean_object* v_reuseFailAlloc_346_; 
v_reuseFailAlloc_346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_346_, 0, v_a_340_);
v___x_345_ = v_reuseFailAlloc_346_;
goto v_reusejp_344_;
}
v_reusejp_344_:
{
return v___x_345_;
}
}
}
}
else
{
lean_object* v_a_348_; lean_object* v___x_350_; uint8_t v_isShared_351_; uint8_t v_isSharedCheck_355_; 
lean_dec_ref(v___x_318_);
lean_dec(v___y_280_);
v_a_348_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_355_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_355_ == 0)
{
v___x_350_ = v___x_325_;
v_isShared_351_ = v_isSharedCheck_355_;
goto v_resetjp_349_;
}
else
{
lean_inc(v_a_348_);
lean_dec(v___x_325_);
v___x_350_ = lean_box(0);
v_isShared_351_ = v_isSharedCheck_355_;
goto v_resetjp_349_;
}
v_resetjp_349_:
{
lean_object* v___x_353_; 
if (v_isShared_351_ == 0)
{
v___x_353_ = v___x_350_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v_a_348_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_386_; 
lean_dec(v_mvarId_302_);
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v_trace_281_);
lean_dec(v___y_280_);
lean_dec(v_tk_279_);
v_a_379_ = lean_ctor_get(v___x_305_, 0);
v_isSharedCheck_386_ = !lean_is_exclusive(v___x_305_);
if (v_isSharedCheck_386_ == 0)
{
v___x_381_ = v___x_305_;
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_a_379_);
lean_dec(v___x_305_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_384_; 
if (v_isShared_382_ == 0)
{
v___x_384_ = v___x_381_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v_a_379_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
}
}
else
{
lean_object* v___x_387_; lean_object* v___x_388_; 
lean_dec(v_a_301_);
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v_trace_281_);
lean_dec(v___y_280_);
lean_dec(v_tk_279_);
lean_dec_ref(v___f_278_);
lean_dec_ref(v___f_277_);
v___x_387_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__5, &lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___closed__5);
v___x_388_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg(v___x_387_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
return v___x_388_;
}
}
else
{
lean_object* v_a_389_; lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_396_; 
lean_dec_ref_known(v___x_297_, 1);
lean_dec(v_trace_281_);
lean_dec(v___y_280_);
lean_dec(v_tk_279_);
lean_dec_ref(v___f_278_);
lean_dec_ref(v___f_277_);
v_a_389_ = lean_ctor_get(v___x_300_, 0);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_300_);
if (v_isSharedCheck_396_ == 0)
{
v___x_391_ = v___x_300_;
v_isShared_392_ = v_isSharedCheck_396_;
goto v_resetjp_390_;
}
else
{
lean_inc(v_a_389_);
lean_dec(v___x_300_);
v___x_391_ = lean_box(0);
v_isShared_392_ = v_isSharedCheck_396_;
goto v_resetjp_390_;
}
v_resetjp_390_:
{
lean_object* v___x_394_; 
if (v_isShared_392_ == 0)
{
v___x_394_ = v___x_391_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v_a_389_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
}
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
lean_dec(v_trace_281_);
lean_dec(v___y_280_);
lean_dec(v_tk_279_);
lean_dec_ref(v___f_278_);
lean_dec_ref(v___f_277_);
v_a_397_ = lean_ctor_get(v___x_294_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_294_);
if (v_isSharedCheck_404_ == 0)
{
v___x_399_ = v___x_294_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_294_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
if (v_isShared_400_ == 0)
{
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_a_397_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
else
{
lean_object* v_a_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
lean_dec(v_trace_281_);
lean_dec(v___y_280_);
lean_dec(v_tk_279_);
lean_dec_ref(v___f_278_);
lean_dec_ref(v___f_277_);
lean_dec(v___x_275_);
v_a_405_ = lean_ctor_get(v___x_291_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_291_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_291_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_291_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_405_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___boxed(lean_object* v___x_413_, lean_object* v___x_414_, lean_object* v___f_415_, lean_object* v___f_416_, lean_object* v_tk_417_, lean_object* v___y_418_, lean_object* v_trace_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
uint8_t v___x_9131__boxed_429_; lean_object* v_res_430_; 
v___x_9131__boxed_429_ = lean_unbox(v___x_414_);
v_res_430_ = lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2(v___x_413_, v___x_9131__boxed_429_, v___f_415_, v___f_416_, v_tk_417_, v___y_418_, v_trace_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
lean_dec(v___y_423_);
lean_dec_ref(v___y_422_);
lean_dec(v___y_421_);
lean_dec_ref(v___y_420_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__3(size_t v_sz_431_, size_t v_i_432_, lean_object* v_bs_433_){
_start:
{
uint8_t v___x_434_; 
v___x_434_ = lean_usize_dec_lt(v_i_432_, v_sz_431_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; 
v___x_435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_435_, 0, v_bs_433_);
return v___x_435_;
}
else
{
lean_object* v_v_436_; lean_object* v___x_437_; lean_object* v_bs_x27_438_; size_t v___x_439_; size_t v___x_440_; lean_object* v___x_441_; 
v_v_436_ = lean_array_uget(v_bs_433_, v_i_432_);
v___x_437_ = lean_unsigned_to_nat(0u);
v_bs_x27_438_ = lean_array_uset(v_bs_433_, v_i_432_, v___x_437_);
v___x_439_ = ((size_t)1ULL);
v___x_440_ = lean_usize_add(v_i_432_, v___x_439_);
v___x_441_ = lean_array_uset(v_bs_x27_438_, v_i_432_, v_v_436_);
v_i_432_ = v___x_440_;
v_bs_433_ = v___x_441_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__3___boxed(lean_object* v_sz_443_, lean_object* v_i_444_, lean_object* v_bs_445_){
_start:
{
size_t v_sz_boxed_446_; size_t v_i_boxed_447_; lean_object* v_res_448_; 
v_sz_boxed_446_ = lean_unbox_usize(v_sz_443_);
lean_dec(v_sz_443_);
v_i_boxed_447_ = lean_unbox_usize(v_i_444_);
lean_dec(v_i_444_);
v_res_448_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__3(v_sz_boxed_446_, v_i_boxed_447_, v_bs_445_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4(uint8_t v___x_449_, uint8_t v___x_450_, lean_object* v_as_451_, size_t v_i_452_, size_t v_stop_453_, lean_object* v_b_454_){
_start:
{
lean_object* v___y_456_; uint8_t v___x_460_; 
v___x_460_ = lean_usize_dec_eq(v_i_452_, v_stop_453_);
if (v___x_460_ == 0)
{
lean_object* v_fst_461_; uint8_t v___x_462_; 
v_fst_461_ = lean_ctor_get(v_b_454_, 0);
v___x_462_ = lean_unbox(v_fst_461_);
if (v___x_462_ == 0)
{
lean_object* v_snd_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_471_; 
v_snd_463_ = lean_ctor_get(v_b_454_, 1);
v_isSharedCheck_471_ = !lean_is_exclusive(v_b_454_);
if (v_isSharedCheck_471_ == 0)
{
lean_object* v_unused_472_; 
v_unused_472_ = lean_ctor_get(v_b_454_, 0);
lean_dec(v_unused_472_);
v___x_465_ = v_b_454_;
v_isShared_466_ = v_isSharedCheck_471_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_snd_463_);
lean_dec(v_b_454_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_471_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
lean_object* v___x_467_; lean_object* v___x_469_; 
v___x_467_ = lean_box(v___x_449_);
if (v_isShared_466_ == 0)
{
lean_ctor_set(v___x_465_, 0, v___x_467_);
v___x_469_ = v___x_465_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_467_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v_snd_463_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
v___y_456_ = v___x_469_;
goto v___jp_455_;
}
}
}
else
{
lean_object* v_snd_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_483_; 
v_snd_473_ = lean_ctor_get(v_b_454_, 1);
v_isSharedCheck_483_ = !lean_is_exclusive(v_b_454_);
if (v_isSharedCheck_483_ == 0)
{
lean_object* v_unused_484_; 
v_unused_484_ = lean_ctor_get(v_b_454_, 0);
lean_dec(v_unused_484_);
v___x_475_ = v_b_454_;
v_isShared_476_ = v_isSharedCheck_483_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_snd_473_);
lean_dec(v_b_454_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_483_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_481_; 
v___x_477_ = lean_array_uget_borrowed(v_as_451_, v_i_452_);
lean_inc(v___x_477_);
v___x_478_ = lean_array_push(v_snd_473_, v___x_477_);
v___x_479_ = lean_box(v___x_450_);
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 1, v___x_478_);
lean_ctor_set(v___x_475_, 0, v___x_479_);
v___x_481_ = v___x_475_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v___x_479_);
lean_ctor_set(v_reuseFailAlloc_482_, 1, v___x_478_);
v___x_481_ = v_reuseFailAlloc_482_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
v___y_456_ = v___x_481_;
goto v___jp_455_;
}
}
}
}
else
{
return v_b_454_;
}
v___jp_455_:
{
size_t v___x_457_; size_t v___x_458_; 
v___x_457_ = ((size_t)1ULL);
v___x_458_ = lean_usize_add(v_i_452_, v___x_457_);
v_i_452_ = v___x_458_;
v_b_454_ = v___y_456_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4___boxed(lean_object* v___x_485_, lean_object* v___x_486_, lean_object* v_as_487_, lean_object* v_i_488_, lean_object* v_stop_489_, lean_object* v_b_490_){
_start:
{
uint8_t v___x_9447__boxed_491_; uint8_t v___x_9448__boxed_492_; size_t v_i_boxed_493_; size_t v_stop_boxed_494_; lean_object* v_res_495_; 
v___x_9447__boxed_491_ = lean_unbox(v___x_485_);
v___x_9448__boxed_492_ = lean_unbox(v___x_486_);
v_i_boxed_493_ = lean_unbox_usize(v_i_488_);
lean_dec(v_i_488_);
v_stop_boxed_494_ = lean_unbox_usize(v_stop_489_);
lean_dec(v_stop_489_);
v_res_495_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4(v___x_9447__boxed_491_, v___x_9448__boxed_492_, v_as_487_, v_i_boxed_493_, v_stop_boxed_494_, v_b_490_);
lean_dec_ref(v_as_487_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1(lean_object* v_x_502_, lean_object* v_a_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_, lean_object* v_a_509_, lean_object* v_a_510_){
_start:
{
lean_object* v___x_512_; uint8_t v___x_513_; 
v___x_512_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4));
lean_inc(v_x_502_);
v___x_513_ = l_Lean_Syntax_isOfKind(v_x_502_, v___x_512_);
if (v___x_513_ == 0)
{
lean_object* v___x_514_; 
lean_dec(v_x_502_);
v___x_514_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_514_;
}
else
{
lean_object* v___f_515_; lean_object* v___x_516_; lean_object* v___f_517_; lean_object* v___x_518_; lean_object* v_tk_519_; lean_object* v___y_521_; lean_object* v___y_522_; lean_object* v___y_523_; lean_object* v___y_524_; lean_object* v___y_525_; lean_object* v___y_526_; lean_object* v___y_527_; lean_object* v___y_528_; lean_object* v___y_529_; lean_object* v___y_530_; lean_object* v___y_531_; lean_object* v___y_536_; lean_object* v___y_537_; lean_object* v___y_538_; lean_object* v___y_539_; lean_object* v___y_540_; lean_object* v___y_541_; lean_object* v___y_542_; lean_object* v___y_543_; lean_object* v___y_544_; lean_object* v___y_545_; lean_object* v___y_546_; lean_object* v___y_551_; lean_object* v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___y_556_; lean_object* v___y_557_; lean_object* v___y_558_; lean_object* v___y_559_; lean_object* v___y_560_; lean_object* v___y_561_; lean_object* v___y_562_; lean_object* v___x_567_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_573_; lean_object* v___y_574_; lean_object* v___y_575_; lean_object* v___y_576_; lean_object* v___y_577_; lean_object* v___y_578_; lean_object* v_n_x3f_579_; lean_object* v_trace_604_; lean_object* v___y_605_; lean_object* v___y_606_; lean_object* v___y_607_; lean_object* v___y_608_; lean_object* v___y_609_; lean_object* v___y_610_; lean_object* v___y_611_; lean_object* v___y_612_; lean_object* v___x_624_; uint8_t v___x_625_; 
v___f_515_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__0));
v___x_516_ = lean_box(v___x_513_);
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__1___boxed), 7, 1);
lean_closure_set(v___f_517_, 0, v___x_516_);
v___x_518_ = lean_unsigned_to_nat(0u);
v_tk_519_ = l_Lean_Syntax_getArg(v_x_502_, v___x_518_);
v___x_567_ = lean_unsigned_to_nat(1u);
v___x_624_ = l_Lean_Syntax_getArg(v_x_502_, v___x_567_);
v___x_625_ = l_Lean_Syntax_isNone(v___x_624_);
if (v___x_625_ == 0)
{
uint8_t v___x_626_; 
lean_inc(v___x_624_);
v___x_626_ = l_Lean_Syntax_matchesNull(v___x_624_, v___x_567_);
if (v___x_626_ == 0)
{
lean_object* v___x_627_; 
lean_dec(v___x_624_);
lean_dec(v_tk_519_);
lean_dec_ref(v___f_517_);
lean_dec(v_x_502_);
v___x_627_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_627_;
}
else
{
lean_object* v_trace_628_; lean_object* v___x_629_; 
v_trace_628_ = l_Lean_Syntax_getArg(v___x_624_, v___x_518_);
lean_dec(v___x_624_);
v___x_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_629_, 0, v_trace_628_);
v_trace_604_ = v___x_629_;
v___y_605_ = v_a_503_;
v___y_606_ = v_a_504_;
v___y_607_ = v_a_505_;
v___y_608_ = v_a_506_;
v___y_609_ = v_a_507_;
v___y_610_ = v_a_508_;
v___y_611_ = v_a_509_;
v___y_612_ = v_a_510_;
goto v___jp_603_;
}
}
else
{
lean_object* v___x_630_; 
lean_dec(v___x_624_);
v___x_630_ = lean_box(0);
v_trace_604_ = v___x_630_;
v___y_605_ = v_a_503_;
v___y_606_ = v_a_504_;
v___y_607_ = v_a_505_;
v___y_608_ = v_a_506_;
v___y_609_ = v_a_507_;
v___y_610_ = v_a_508_;
v___y_611_ = v_a_509_;
v___y_612_ = v_a_510_;
goto v___jp_603_;
}
v___jp_520_:
{
lean_object* v___x_532_; lean_object* v___f_533_; lean_object* v___x_534_; 
v___x_532_ = lean_box(v___x_513_);
v___f_533_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___lam__2___boxed), 16, 7);
lean_closure_set(v___f_533_, 0, v___y_521_);
lean_closure_set(v___f_533_, 1, v___x_532_);
lean_closure_set(v___f_533_, 2, v___f_515_);
lean_closure_set(v___f_533_, 3, v___f_517_);
lean_closure_set(v___f_533_, 4, v_tk_519_);
lean_closure_set(v___f_533_, 5, v___y_531_);
lean_closure_set(v___f_533_, 6, v___y_522_);
v___x_534_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_533_, v___y_530_, v___y_526_, v___y_527_, v___y_525_, v___y_523_, v___y_529_, v___y_524_, v___y_528_);
return v___x_534_;
}
v___jp_535_:
{
if (lean_obj_tag(v___y_538_) == 0)
{
lean_object* v___x_547_; 
v___x_547_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__2));
v___y_521_ = v___y_536_;
v___y_522_ = v___y_537_;
v___y_523_ = v___y_543_;
v___y_524_ = v___y_545_;
v___y_525_ = v___y_542_;
v___y_526_ = v___y_540_;
v___y_527_ = v___y_541_;
v___y_528_ = v___y_546_;
v___y_529_ = v___y_544_;
v___y_530_ = v___y_539_;
v___y_531_ = v___x_547_;
goto v___jp_520_;
}
else
{
lean_object* v_val_548_; lean_object* v___x_549_; 
v_val_548_ = lean_ctor_get(v___y_538_, 0);
lean_inc(v_val_548_);
lean_dec_ref_known(v___y_538_, 1);
v___x_549_ = l_Lean_TSyntax_getId(v_val_548_);
lean_dec(v_val_548_);
v___y_521_ = v___y_536_;
v___y_522_ = v___y_537_;
v___y_523_ = v___y_543_;
v___y_524_ = v___y_545_;
v___y_525_ = v___y_542_;
v___y_526_ = v___y_540_;
v___y_527_ = v___y_541_;
v___y_528_ = v___y_546_;
v___y_529_ = v___y_544_;
v___y_530_ = v___y_539_;
v___y_531_ = v___x_549_;
goto v___jp_520_;
}
}
v___jp_550_:
{
size_t v_sz_563_; size_t v___x_564_; lean_object* v___x_565_; 
v_sz_563_ = lean_array_size(v___y_562_);
v___x_564_ = ((size_t)0ULL);
v___x_565_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__3(v_sz_563_, v___x_564_, v___y_562_);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v___x_566_; 
lean_dec(v___y_554_);
lean_dec(v___y_552_);
lean_dec(v___y_551_);
lean_dec(v_tk_519_);
lean_dec_ref(v___f_517_);
v___x_566_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_566_;
}
else
{
lean_dec_ref_known(v___x_565_, 1);
v___y_536_ = v___y_551_;
v___y_537_ = v___y_552_;
v___y_538_ = v___y_554_;
v___y_539_ = v___y_560_;
v___y_540_ = v___y_559_;
v___y_541_ = v___y_558_;
v___y_542_ = v___y_553_;
v___y_543_ = v___y_561_;
v___y_544_ = v___y_555_;
v___y_545_ = v___y_556_;
v___y_546_ = v___y_557_;
goto v___jp_535_;
}
}
v___jp_568_:
{
lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; uint8_t v___x_584_; 
v___x_580_ = lean_unsigned_to_nat(4u);
v___x_581_ = l_Lean_Syntax_getArg(v_x_502_, v___x_580_);
v___x_582_ = lean_unsigned_to_nat(5u);
v___x_583_ = l_Lean_Syntax_getArg(v_x_502_, v___x_582_);
lean_dec(v_x_502_);
v___x_584_ = l_Lean_Syntax_isNone(v___x_583_);
if (v___x_584_ == 0)
{
uint8_t v___x_585_; 
lean_inc(v___x_583_);
v___x_585_ = l_Lean_Syntax_matchesNull(v___x_583_, v___y_578_);
if (v___x_585_ == 0)
{
lean_object* v___x_586_; 
lean_dec(v___x_583_);
lean_dec(v___x_581_);
lean_dec(v_n_x3f_579_);
lean_dec(v___y_576_);
lean_dec(v_tk_519_);
lean_dec_ref(v___f_517_);
v___x_586_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_586_;
}
else
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; uint8_t v___x_591_; 
v___x_587_ = l_Lean_Syntax_getArg(v___x_583_, v___x_567_);
lean_dec(v___x_583_);
v___x_588_ = l_Lean_Syntax_getArgs(v___x_587_);
lean_dec(v___x_587_);
v___x_589_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__3));
v___x_590_ = lean_array_get_size(v___x_588_);
v___x_591_ = lean_nat_dec_lt(v___x_518_, v___x_590_);
if (v___x_591_ == 0)
{
lean_dec_ref(v___x_588_);
v___y_551_ = v___x_581_;
v___y_552_ = v___y_576_;
v___y_553_ = v___y_572_;
v___y_554_ = v_n_x3f_579_;
v___y_555_ = v___y_575_;
v___y_556_ = v___y_571_;
v___y_557_ = v___y_569_;
v___y_558_ = v___y_573_;
v___y_559_ = v___y_570_;
v___y_560_ = v___y_574_;
v___y_561_ = v___y_577_;
v___y_562_ = v___x_589_;
goto v___jp_550_;
}
else
{
lean_object* v___x_592_; lean_object* v___x_593_; uint8_t v___x_594_; 
v___x_592_ = lean_box(v___x_585_);
v___x_593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
lean_ctor_set(v___x_593_, 1, v___x_589_);
v___x_594_ = lean_nat_dec_le(v___x_590_, v___x_590_);
if (v___x_594_ == 0)
{
if (v___x_591_ == 0)
{
lean_dec_ref_known(v___x_593_, 2);
lean_dec_ref(v___x_588_);
v___y_551_ = v___x_581_;
v___y_552_ = v___y_576_;
v___y_553_ = v___y_572_;
v___y_554_ = v_n_x3f_579_;
v___y_555_ = v___y_575_;
v___y_556_ = v___y_571_;
v___y_557_ = v___y_569_;
v___y_558_ = v___y_573_;
v___y_559_ = v___y_570_;
v___y_560_ = v___y_574_;
v___y_561_ = v___y_577_;
v___y_562_ = v___x_589_;
goto v___jp_550_;
}
else
{
size_t v___x_595_; size_t v___x_596_; lean_object* v___x_597_; lean_object* v_snd_598_; 
v___x_595_ = ((size_t)0ULL);
v___x_596_ = lean_usize_of_nat(v___x_590_);
v___x_597_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4(v___x_585_, v___x_584_, v___x_588_, v___x_595_, v___x_596_, v___x_593_);
lean_dec_ref(v___x_588_);
v_snd_598_ = lean_ctor_get(v___x_597_, 1);
lean_inc(v_snd_598_);
lean_dec_ref(v___x_597_);
v___y_551_ = v___x_581_;
v___y_552_ = v___y_576_;
v___y_553_ = v___y_572_;
v___y_554_ = v_n_x3f_579_;
v___y_555_ = v___y_575_;
v___y_556_ = v___y_571_;
v___y_557_ = v___y_569_;
v___y_558_ = v___y_573_;
v___y_559_ = v___y_570_;
v___y_560_ = v___y_574_;
v___y_561_ = v___y_577_;
v___y_562_ = v_snd_598_;
goto v___jp_550_;
}
}
else
{
size_t v___x_599_; size_t v___x_600_; lean_object* v___x_601_; lean_object* v_snd_602_; 
v___x_599_ = ((size_t)0ULL);
v___x_600_ = lean_usize_of_nat(v___x_590_);
v___x_601_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__4(v___x_585_, v___x_584_, v___x_588_, v___x_599_, v___x_600_, v___x_593_);
lean_dec_ref(v___x_588_);
v_snd_602_ = lean_ctor_get(v___x_601_, 1);
lean_inc(v_snd_602_);
lean_dec_ref(v___x_601_);
v___y_551_ = v___x_581_;
v___y_552_ = v___y_576_;
v___y_553_ = v___y_572_;
v___y_554_ = v_n_x3f_579_;
v___y_555_ = v___y_575_;
v___y_556_ = v___y_571_;
v___y_557_ = v___y_569_;
v___y_558_ = v___y_573_;
v___y_559_ = v___y_570_;
v___y_560_ = v___y_574_;
v___y_561_ = v___y_577_;
v___y_562_ = v_snd_602_;
goto v___jp_550_;
}
}
}
}
else
{
lean_dec(v___x_583_);
v___y_536_ = v___x_581_;
v___y_537_ = v___y_576_;
v___y_538_ = v_n_x3f_579_;
v___y_539_ = v___y_574_;
v___y_540_ = v___y_570_;
v___y_541_ = v___y_573_;
v___y_542_ = v___y_572_;
v___y_543_ = v___y_577_;
v___y_544_ = v___y_575_;
v___y_545_ = v___y_571_;
v___y_546_ = v___y_569_;
goto v___jp_535_;
}
}
v___jp_603_:
{
lean_object* v___x_613_; lean_object* v___x_614_; uint8_t v___x_615_; 
v___x_613_ = lean_unsigned_to_nat(2u);
v___x_614_ = l_Lean_Syntax_getArg(v_x_502_, v___x_613_);
v___x_615_ = l_Lean_Syntax_isNone(v___x_614_);
if (v___x_615_ == 0)
{
uint8_t v___x_616_; 
lean_inc(v___x_614_);
v___x_616_ = l_Lean_Syntax_matchesNull(v___x_614_, v___x_567_);
if (v___x_616_ == 0)
{
lean_object* v___x_617_; 
lean_dec(v___x_614_);
lean_dec(v_trace_604_);
lean_dec(v_tk_519_);
lean_dec_ref(v___f_517_);
lean_dec(v_x_502_);
v___x_617_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_617_;
}
else
{
lean_object* v_n_x3f_618_; lean_object* v___x_619_; uint8_t v___x_620_; 
v_n_x3f_618_ = l_Lean_Syntax_getArg(v___x_614_, v___x_518_);
lean_dec(v___x_614_);
v___x_619_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__18));
lean_inc(v_n_x3f_618_);
v___x_620_ = l_Lean_Syntax_isOfKind(v_n_x3f_618_, v___x_619_);
if (v___x_620_ == 0)
{
lean_object* v___x_621_; 
lean_dec(v_n_x3f_618_);
lean_dec(v_trace_604_);
lean_dec(v_tk_519_);
lean_dec_ref(v___f_517_);
lean_dec(v_x_502_);
v___x_621_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__0___redArg();
return v___x_621_;
}
else
{
lean_object* v___x_622_; 
v___x_622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_622_, 0, v_n_x3f_618_);
v___y_569_ = v___y_612_;
v___y_570_ = v___y_606_;
v___y_571_ = v___y_611_;
v___y_572_ = v___y_608_;
v___y_573_ = v___y_607_;
v___y_574_ = v___y_605_;
v___y_575_ = v___y_610_;
v___y_576_ = v_trace_604_;
v___y_577_ = v___y_609_;
v___y_578_ = v___x_613_;
v_n_x3f_579_ = v___x_622_;
goto v___jp_568_;
}
}
}
else
{
lean_object* v___x_623_; 
lean_dec(v___x_614_);
v___x_623_ = lean_box(0);
v___y_569_ = v___y_612_;
v___y_570_ = v___y_606_;
v___y_571_ = v___y_611_;
v___y_572_ = v___y_608_;
v___y_573_ = v___y_607_;
v___y_574_ = v___y_605_;
v___y_575_ = v___y_610_;
v___y_576_ = v_trace_604_;
v___y_577_ = v___y_609_;
v___y_578_ = v___x_613_;
v_n_x3f_579_ = v___x_623_;
goto v___jp_568_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___boxed(lean_object* v_x_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_){
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1(v_x_631_, v_a_632_, v_a_633_, v_a_634_, v_a_635_, v_a_636_, v_a_637_, v_a_638_, v_a_639_);
lean_dec(v_a_639_);
lean_dec_ref(v_a_638_);
lean_dec(v_a_637_);
lean_dec_ref(v_a_636_);
lean_dec(v_a_635_);
lean_dec_ref(v_a_634_);
lean_dec(v_a_633_);
lean_dec_ref(v_a_632_);
return v_res_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1(lean_object* v_00_u03b1_642_, lean_object* v_msg_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_){
_start:
{
lean_object* v___x_653_; 
v___x_653_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___redArg(v_msg_643_, v___y_648_, v___y_649_, v___y_650_, v___y_651_);
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1___boxed(lean_object* v_00_u03b1_654_, lean_object* v_msg_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1_spec__1(v_00_u03b1_654_, v_msg_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, v___y_662_, v___y_663_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
return v_res_665_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3(void){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = l_Array_mkArray0(lean_box(0));
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1(lean_object* v_x_698_, lean_object* v_a_699_, lean_object* v_a_700_){
_start:
{
lean_object* v___x_701_; uint8_t v___x_702_; 
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a___00__closed__1));
lean_inc(v_x_698_);
v___x_702_ = l_Lean_Syntax_isOfKind(v_x_698_, v___x_701_);
if (v___x_702_ == 0)
{
lean_object* v___x_703_; lean_object* v___x_704_; 
lean_dec(v_x_698_);
v___x_703_ = lean_box(1);
v___x_704_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_704_, 0, v___x_703_);
lean_ctor_set(v___x_704_, 1, v_a_700_);
return v___x_704_;
}
else
{
lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___y_710_; lean_object* v___y_711_; lean_object* v___y_712_; lean_object* v___y_713_; lean_object* v___y_714_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_725_; lean_object* v___x_740_; 
v___x_705_ = lean_unsigned_to_nat(1u);
v___x_706_ = l_Lean_Syntax_getArg(v_x_698_, v___x_705_);
v___x_707_ = lean_unsigned_to_nat(3u);
v___x_708_ = l_Lean_Syntax_getArg(v_x_698_, v___x_707_);
lean_dec(v_x_698_);
v___x_740_ = l_Lean_Syntax_getOptional_x3f(v___x_706_);
lean_dec(v___x_706_);
if (lean_obj_tag(v___x_740_) == 0)
{
lean_object* v___x_741_; 
v___x_741_ = lean_box(0);
v___y_725_ = v___x_741_;
goto v___jp_724_;
}
else
{
lean_object* v_val_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
v_val_742_ = lean_ctor_get(v___x_740_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_740_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_740_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_val_742_);
lean_dec(v___x_740_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_val_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
v___y_725_ = v___x_747_;
goto v___jp_724_;
}
}
}
v___jp_709_:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; 
lean_inc_ref_n(v___y_710_, 2);
v___x_717_ = l_Array_append___redArg(v___y_710_, v___y_716_);
lean_dec_ref(v___y_716_);
lean_inc(v___y_714_);
lean_inc_n(v___y_711_, 3);
v___x_718_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_718_, 0, v___y_711_);
lean_ctor_set(v___x_718_, 1, v___y_714_);
lean_ctor_set(v___x_718_, 2, v___x_717_);
v___x_719_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__0));
v___x_720_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_720_, 0, v___y_711_);
lean_ctor_set(v___x_720_, 1, v___x_719_);
v___x_721_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_721_, 0, v___y_711_);
lean_ctor_set(v___x_721_, 1, v___y_714_);
lean_ctor_set(v___x_721_, 2, v___y_710_);
lean_inc(v___y_713_);
v___x_722_ = l_Lean_Syntax_node6(v___y_711_, v___y_713_, v___y_715_, v___y_712_, v___x_718_, v___x_720_, v___x_708_, v___x_721_);
v___x_723_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_722_);
lean_ctor_set(v___x_723_, 1, v_a_700_);
return v___x_723_;
}
v___jp_724_:
{
lean_object* v_ref_726_; uint8_t v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; 
v_ref_726_ = lean_ctor_get(v_a_699_, 5);
v___x_727_ = 0;
v___x_728_ = l_Lean_SourceInfo_fromRef(v_ref_726_, v___x_727_);
v___x_729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3));
v___x_730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4));
lean_inc_n(v___x_728_, 3);
v___x_731_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_731_, 0, v___x_728_);
lean_ctor_set(v___x_731_, 1, v___x_729_);
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__2));
v___x_733_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__10));
v___x_734_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_728_);
lean_ctor_set(v___x_734_, 1, v___x_733_);
v___x_735_ = l_Lean_Syntax_node1(v___x_728_, v___x_732_, v___x_734_);
v___x_736_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3, &lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3);
if (lean_obj_tag(v___y_725_) == 1)
{
lean_object* v_val_737_; lean_object* v___x_738_; 
v_val_737_ = lean_ctor_get(v___y_725_, 0);
lean_inc(v_val_737_);
lean_dec_ref_known(v___y_725_, 1);
v___x_738_ = l_Array_mkArray1___redArg(v_val_737_);
v___y_710_ = v___x_736_;
v___y_711_ = v___x_728_;
v___y_712_ = v___x_735_;
v___y_713_ = v___x_730_;
v___y_714_ = v___x_732_;
v___y_715_ = v___x_731_;
v___y_716_ = v___x_738_;
goto v___jp_709_;
}
else
{
lean_object* v___x_739_; 
lean_dec(v___y_725_);
v___x_739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__3));
v___y_710_ = v___x_736_;
v___y_711_ = v___x_728_;
v___y_712_ = v___x_735_;
v___y_713_ = v___x_730_;
v___y_714_ = v___x_732_;
v___y_715_ = v___x_731_;
v___y_716_ = v___x_739_;
goto v___jp_709_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___boxed(lean_object* v_x_750_, lean_object* v_a_751_, lean_object* v_a_752_){
_start:
{
lean_object* v_res_753_; 
v_res_753_ = lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1(v_x_750_, v_a_751_, v_a_752_);
lean_dec_ref(v_a_751_);
return v_res_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1_spec__0(size_t v_sz_773_, size_t v_i_774_, lean_object* v_bs_775_){
_start:
{
uint8_t v___x_776_; 
v___x_776_ = lean_usize_dec_lt(v_i_774_, v_sz_773_);
if (v___x_776_ == 0)
{
return v_bs_775_;
}
else
{
lean_object* v_v_777_; lean_object* v___x_778_; lean_object* v_bs_x27_779_; size_t v___x_780_; size_t v___x_781_; lean_object* v___x_782_; 
v_v_777_ = lean_array_uget(v_bs_775_, v_i_774_);
v___x_778_ = lean_unsigned_to_nat(0u);
v_bs_x27_779_ = lean_array_uset(v_bs_775_, v_i_774_, v___x_778_);
v___x_780_ = ((size_t)1ULL);
v___x_781_ = lean_usize_add(v_i_774_, v___x_780_);
v___x_782_ = lean_array_uset(v_bs_x27_779_, v_i_774_, v_v_777_);
v_i_774_ = v___x_781_;
v_bs_775_ = v___x_782_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1_spec__0___boxed(lean_object* v_sz_784_, lean_object* v_i_785_, lean_object* v_bs_786_){
_start:
{
size_t v_sz_boxed_787_; size_t v_i_boxed_788_; lean_object* v_res_789_; 
v_sz_boxed_787_ = lean_unbox_usize(v_sz_784_);
lean_dec(v_sz_784_);
v_i_boxed_788_ = lean_unbox_usize(v_i_785_);
lean_dec(v_i_785_);
v_res_789_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1_spec__0(v_sz_boxed_787_, v_i_boxed_788_, v_bs_786_);
return v_res_789_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__1(void){
_start:
{
lean_object* v___x_791_; lean_object* v___x_792_; 
v___x_791_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__36));
v___x_792_ = l_Lean_mkAtom(v___x_791_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1(lean_object* v_x_793_, lean_object* v_a_794_, lean_object* v_a_795_){
_start:
{
lean_object* v___x_796_; uint8_t v___x_797_; 
v___x_796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_tacticObserve_x3f_____x3a__Using_____x2c_x2c___closed__1));
lean_inc(v_x_793_);
v___x_797_ = l_Lean_Syntax_isOfKind(v_x_793_, v___x_796_);
if (v___x_797_ == 0)
{
lean_object* v___x_798_; lean_object* v___x_799_; 
lean_dec(v_x_793_);
v___x_798_ = lean_box(1);
v___x_799_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_799_, 0, v___x_798_);
lean_ctor_set(v___x_799_, 1, v_a_795_);
return v___x_799_;
}
else
{
lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v_terms_806_; lean_object* v___y_808_; lean_object* v___y_809_; lean_object* v___y_810_; lean_object* v___y_811_; lean_object* v___y_812_; lean_object* v___y_813_; lean_object* v___y_814_; lean_object* v___y_833_; lean_object* v___x_848_; 
v___x_800_ = lean_unsigned_to_nat(1u);
v___x_801_ = l_Lean_Syntax_getArg(v_x_793_, v___x_800_);
v___x_802_ = lean_unsigned_to_nat(3u);
v___x_803_ = l_Lean_Syntax_getArg(v_x_793_, v___x_802_);
v___x_804_ = lean_unsigned_to_nat(5u);
v___x_805_ = l_Lean_Syntax_getArg(v_x_793_, v___x_804_);
lean_dec(v_x_793_);
v_terms_806_ = l_Lean_Syntax_getArgs(v___x_805_);
lean_dec(v___x_805_);
v___x_848_ = l_Lean_Syntax_getOptional_x3f(v___x_801_);
lean_dec(v___x_801_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v___x_849_; 
v___x_849_ = lean_box(0);
v___y_833_ = v___x_849_;
goto v___jp_832_;
}
else
{
lean_object* v_val_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_857_; 
v_val_850_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_857_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_857_ == 0)
{
v___x_852_ = v___x_848_;
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_val_850_);
lean_dec(v___x_848_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v___x_855_; 
if (v_isShared_853_ == 0)
{
v___x_855_ = v___x_852_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v_val_850_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
v___y_833_ = v___x_855_;
goto v___jp_832_;
}
}
}
v___jp_807_:
{
lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; size_t v_sz_822_; size_t v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
lean_inc_ref_n(v___y_809_, 2);
v___x_815_ = l_Array_append___redArg(v___y_809_, v___y_814_);
lean_dec_ref(v___y_814_);
lean_inc_n(v___y_811_, 2);
lean_inc_n(v___y_812_, 5);
v___x_816_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_816_, 0, v___y_812_);
lean_ctor_set(v___x_816_, 1, v___y_811_);
lean_ctor_set(v___x_816_, 2, v___x_815_);
v___x_817_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__0));
v___x_818_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_818_, 0, v___y_812_);
lean_ctor_set(v___x_818_, 1, v___x_817_);
v___x_819_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__0));
v___x_820_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_820_, 0, v___y_812_);
lean_ctor_set(v___x_820_, 1, v___x_819_);
v___x_821_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_terms_806_);
lean_dec_ref(v_terms_806_);
v_sz_822_ = lean_array_size(v___x_821_);
v___x_823_ = ((size_t)0ULL);
v___x_824_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1_spec__0(v_sz_822_, v___x_823_, v___x_821_);
v___x_825_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__1, &lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___closed__1);
v___x_826_ = l_Lean_mkSepArray(v___x_824_, v___x_825_);
lean_dec_ref(v___x_824_);
v___x_827_ = l_Array_append___redArg(v___y_809_, v___x_826_);
lean_dec_ref(v___x_826_);
v___x_828_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_828_, 0, v___y_812_);
lean_ctor_set(v___x_828_, 1, v___y_811_);
lean_ctor_set(v___x_828_, 2, v___x_827_);
v___x_829_ = l_Lean_Syntax_node2(v___y_812_, v___y_811_, v___x_820_, v___x_828_);
lean_inc(v___y_810_);
v___x_830_ = l_Lean_Syntax_node6(v___y_812_, v___y_810_, v___y_813_, v___y_808_, v___x_816_, v___x_818_, v___x_803_, v___x_829_);
v___x_831_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_831_, 0, v___x_830_);
lean_ctor_set(v___x_831_, 1, v_a_795_);
return v___x_831_;
}
v___jp_832_:
{
lean_object* v_ref_834_; uint8_t v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v_ref_834_ = lean_ctor_get(v_a_794_, 5);
v___x_835_ = 0;
v___x_836_ = l_Lean_SourceInfo_fromRef(v_ref_834_, v___x_835_);
v___x_837_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__3));
v___x_838_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__4));
lean_inc_n(v___x_836_, 3);
v___x_839_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_839_, 0, v___x_836_);
lean_ctor_set(v___x_839_, 1, v___x_837_);
v___x_840_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__2));
v___x_841_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch_observe___closed__10));
v___x_842_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_842_, 0, v___x_836_);
lean_ctor_set(v___x_842_, 1, v___x_841_);
v___x_843_ = l_Lean_Syntax_node1(v___x_836_, v___x_840_, v___x_842_);
v___x_844_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3, &lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a____1___closed__3);
if (lean_obj_tag(v___y_833_) == 1)
{
lean_object* v_val_845_; lean_object* v___x_846_; 
v_val_845_ = lean_ctor_get(v___y_833_, 0);
lean_inc(v_val_845_);
lean_dec_ref_known(v___y_833_, 1);
v___x_846_ = l_Array_mkArray1___redArg(v_val_845_);
v___y_808_ = v___x_843_;
v___y_809_ = v___x_844_;
v___y_810_ = v___x_838_;
v___y_811_ = v___x_840_;
v___y_812_ = v___x_836_;
v___y_813_ = v___x_839_;
v___y_814_ = v___x_846_;
goto v___jp_807_;
}
else
{
lean_object* v___x_847_; 
lean_dec(v___y_833_);
v___x_847_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______elabRules__Mathlib__Tactic__LibrarySearch__observe__1___closed__3));
v___y_808_ = v___x_843_;
v___y_809_ = v___x_844_;
v___y_810_ = v___x_838_;
v___y_811_ = v___x_840_;
v___y_812_ = v___x_836_;
v___y_813_ = v___x_839_;
v___y_814_ = v___x_847_;
goto v___jp_807_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1___boxed(lean_object* v_x_858_, lean_object* v_a_859_, lean_object* v_a_860_){
_start:
{
lean_object* v_res_861_; 
v_res_861_ = lp_mathlib_Mathlib_Tactic_LibrarySearch___aux__Mathlib__Tactic__Observe______macroRules__Mathlib__Tactic__LibrarySearch__tacticObserve_x3f_____x3a__Using_____x2c_x2c__1(v_x_858_, v_a_859_, v_a_860_);
lean_dec_ref(v_a_859_);
return v_res_861_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Observe(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_LibrarySearch(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Observe(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_LibrarySearch(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_LibrarySearch(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Observe(uint8_t builtin) {
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
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_LibrarySearch(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Observe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Observe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Observe(builtin);
}
#ifdef __cplusplus
}
#endif
