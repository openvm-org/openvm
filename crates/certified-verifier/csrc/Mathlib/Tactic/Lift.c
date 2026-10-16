// Lean compiler output
// Module: Mathlib.Tactic.Lift
// Imports: public import Init public meta import Init public meta import Batteries.Lean.Expr public meta import Batteries.Lean.Meta.UnusedNames public meta import Lean.Elab.Tactic.RCases public import Mathlib.Tactic.TypeStar
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
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lp_batteries_Lean_LocalContext_getUnusedUserName(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isFVar(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_RCases_rcases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_toSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Elab_Term_elabType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Elab_Tactic_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isProp(lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__2_value),LEAN_SCALAR_PTR_LITERAL(107, 145, 218, 225, 209, 114, 73, 219)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lift "};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " to "};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__16_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__25_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__29_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_lift___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__32_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__33_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__31_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__35_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__28_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__38_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__39_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_lift___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_lift___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__42_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_lift = (const lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__42_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "CanLift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__3_value),LEAN_SCALAR_PTR_LITERAL(32, 145, 244, 197, 69, 137, 43, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Mathlib_Tactic_Lift_main_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Mathlib_Tactic_Lift_main_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__4_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__8_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "negConfigItem"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__10_value),LEAN_SCALAR_PTR_LITERAL(196, 29, 29, 161, 247, 206, 181, 221)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "failIfUnchanged"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__13_value),LEAN_SCALAR_PTR_LITERAL(6, 104, 167, 161, 191, 186, 8, 81)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__19 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__19_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__23 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__23_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "locationHyp"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__26 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__26_value),LEAN_SCALAR_PTR_LITERAL(229, 146, 67, 234, 45, 36, 143, 176)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8_spec__9(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Lift_main_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "clear"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_lift___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(190, 197, 160, 206, 26, 199, 189, 206)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "lift tactic failed: unreachable code was reached"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "prf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__3_value),LEAN_SCALAR_PTR_LITERAL(32, 145, 244, 197, 69, 137, 43, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__11_value),LEAN_SCALAR_PTR_LITERAL(199, 111, 155, 253, 142, 69, 204, 253)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tmpVar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(151, 183, 188, 109, 228, 65, 208, 93)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__14_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__14_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 82, .m_capacity = 82, .m_length = 81, .m_data = "lift tactic failed. When lifting an expression, a new variable name must be given"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "lift tactic failed. Tactic is only applicable when the target is a proposition."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Lift_main___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Lift_main___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Lift_main___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(lean_object* v_e_106_, lean_object* v___y_107_){
_start:
{
uint8_t v___x_109_; 
v___x_109_ = l_Lean_Expr_hasMVar(v_e_106_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; 
v___x_110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_110_, 0, v_e_106_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v_mctx_112_; lean_object* v___x_113_; lean_object* v_fst_114_; lean_object* v_snd_115_; lean_object* v___x_116_; lean_object* v_cache_117_; lean_object* v_zetaDeltaFVarIds_118_; lean_object* v_postponed_119_; lean_object* v_diag_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_129_; 
v___x_111_ = lean_st_ref_get(v___y_107_);
v_mctx_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc_ref(v_mctx_112_);
lean_dec(v___x_111_);
v___x_113_ = l_Lean_instantiateMVarsCore(v_mctx_112_, v_e_106_);
v_fst_114_ = lean_ctor_get(v___x_113_, 0);
lean_inc(v_fst_114_);
v_snd_115_ = lean_ctor_get(v___x_113_, 1);
lean_inc(v_snd_115_);
lean_dec_ref(v___x_113_);
v___x_116_ = lean_st_ref_take(v___y_107_);
v_cache_117_ = lean_ctor_get(v___x_116_, 1);
v_zetaDeltaFVarIds_118_ = lean_ctor_get(v___x_116_, 2);
v_postponed_119_ = lean_ctor_get(v___x_116_, 3);
v_diag_120_ = lean_ctor_get(v___x_116_, 4);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_129_ == 0)
{
lean_object* v_unused_130_; 
v_unused_130_ = lean_ctor_get(v___x_116_, 0);
lean_dec(v_unused_130_);
v___x_122_ = v___x_116_;
v_isShared_123_ = v_isSharedCheck_129_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_diag_120_);
lean_inc(v_postponed_119_);
lean_inc(v_zetaDeltaFVarIds_118_);
lean_inc(v_cache_117_);
lean_dec(v___x_116_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_129_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_125_; 
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 0, v_snd_115_);
v___x_125_ = v___x_122_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v_snd_115_);
lean_ctor_set(v_reuseFailAlloc_128_, 1, v_cache_117_);
lean_ctor_set(v_reuseFailAlloc_128_, 2, v_zetaDeltaFVarIds_118_);
lean_ctor_set(v_reuseFailAlloc_128_, 3, v_postponed_119_);
lean_ctor_set(v_reuseFailAlloc_128_, 4, v_diag_120_);
v___x_125_ = v_reuseFailAlloc_128_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lean_st_ref_set(v___y_107_, v___x_125_);
v___x_127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_127_, 0, v_fst_114_);
return v___x_127_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg___boxed(lean_object* v_e_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(v_e_131_, v___y_132_);
lean_dec(v___y_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0(lean_object* v_e_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(v_e_135_, v___y_137_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___boxed(lean_object* v_e_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0(v_e_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
return v_res_148_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__2(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = lean_box(0);
v___x_153_ = l_Lean_Expr_sort___override(v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst(lean_object* v_old__tp_157_, lean_object* v_new__tp_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_){
_start:
{
lean_object* v___x_164_; uint8_t v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; uint8_t v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__1));
v___x_165_ = 0;
lean_inc_ref(v_old__tp_157_);
lean_inc_ref(v_new__tp_158_);
v___x_166_ = l_Lean_Expr_forallE___override(v___x_164_, v_new__tp_158_, v_old__tp_157_, v___x_165_);
v___x_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_167_, 0, v___x_166_);
v___x_168_ = 0;
v___x_169_ = lean_box(0);
v___x_170_ = l_Lean_Meta_mkFreshExprMVar(v___x_167_, v___x_168_, v___x_169_, v_a_159_, v_a_160_, v_a_161_, v_a_162_);
if (lean_obj_tag(v___x_170_) == 0)
{
lean_object* v_a_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v_a_171_ = lean_ctor_get(v___x_170_, 0);
lean_inc(v_a_171_);
lean_dec_ref_known(v___x_170_, 1);
v___x_172_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__2, &lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__2);
lean_inc_ref(v_old__tp_157_);
v___x_173_ = l_Lean_Expr_forallE___override(v___x_164_, v_old__tp_157_, v___x_172_, v___x_165_);
v___x_174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
v___x_175_ = l_Lean_Meta_mkFreshExprMVar(v___x_174_, v___x_168_, v___x_169_, v_a_159_, v_a_160_, v_a_161_, v_a_162_);
if (lean_obj_tag(v___x_175_) == 0)
{
lean_object* v_a_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v_a_176_ = lean_ctor_get(v___x_175_, 0);
lean_inc_n(v_a_176_, 2);
lean_dec_ref_known(v___x_175_, 1);
v___x_177_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_getInst___closed__4));
v___x_178_ = lean_unsigned_to_nat(4u);
v___x_179_ = lean_mk_empty_array_with_capacity(v___x_178_);
v___x_180_ = lean_array_push(v___x_179_, v_old__tp_157_);
v___x_181_ = lean_array_push(v___x_180_, v_new__tp_158_);
lean_inc(v_a_171_);
v___x_182_ = lean_array_push(v___x_181_, v_a_171_);
v___x_183_ = lean_array_push(v___x_182_, v_a_176_);
v___x_184_ = l_Lean_Meta_mkAppM(v___x_177_, v___x_183_, v_a_159_, v_a_160_, v_a_161_, v_a_162_);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_a_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v_a_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_185_);
lean_dec_ref_known(v___x_184_, 1);
v___x_186_ = lean_box(0);
v___x_187_ = l_Lean_Meta_synthInstance(v_a_185_, v___x_186_, v_a_159_, v_a_160_, v_a_161_, v_a_162_);
if (lean_obj_tag(v___x_187_) == 0)
{
lean_object* v_a_188_; lean_object* v___x_189_; lean_object* v_a_190_; lean_object* v___x_191_; lean_object* v_a_192_; lean_object* v___x_193_; lean_object* v_a_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_203_; 
v_a_188_ = lean_ctor_get(v___x_187_, 0);
lean_inc(v_a_188_);
lean_dec_ref_known(v___x_187_, 1);
v___x_189_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(v_a_176_, v_a_160_);
v_a_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_a_190_);
lean_dec_ref(v___x_189_);
v___x_191_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(v_a_171_, v_a_160_);
v_a_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc(v_a_192_);
lean_dec_ref(v___x_191_);
v___x_193_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_getInst_spec__0___redArg(v_a_188_, v_a_160_);
v_a_194_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_203_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_203_ == 0)
{
v___x_196_ = v___x_193_;
v_isShared_197_ = v_isSharedCheck_203_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_a_194_);
lean_dec(v___x_193_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_203_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_201_; 
v___x_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_198_, 0, v_a_192_);
lean_ctor_set(v___x_198_, 1, v_a_194_);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v_a_190_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 0, v___x_199_);
v___x_201_ = v___x_196_;
goto v_reusejp_200_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v___x_199_);
v___x_201_ = v_reuseFailAlloc_202_;
goto v_reusejp_200_;
}
v_reusejp_200_:
{
return v___x_201_;
}
}
}
else
{
lean_object* v_a_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_211_; 
lean_dec(v_a_176_);
lean_dec(v_a_171_);
v_a_204_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_211_ == 0)
{
v___x_206_ = v___x_187_;
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_a_204_);
lean_dec(v___x_187_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_209_; 
if (v_isShared_207_ == 0)
{
v___x_209_ = v___x_206_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v_a_204_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
}
else
{
lean_object* v_a_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
lean_dec(v_a_176_);
lean_dec(v_a_171_);
v_a_212_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_219_ == 0)
{
v___x_214_ = v___x_184_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_a_212_);
lean_dec(v___x_184_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_217_; 
if (v_isShared_215_ == 0)
{
v___x_217_ = v___x_214_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_a_212_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
}
else
{
lean_object* v_a_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_227_; 
lean_dec(v_a_171_);
lean_dec_ref(v_new__tp_158_);
lean_dec_ref(v_old__tp_157_);
v_a_220_ = lean_ctor_get(v___x_175_, 0);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_227_ == 0)
{
v___x_222_ = v___x_175_;
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_a_220_);
lean_dec(v___x_175_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_225_; 
if (v_isShared_223_ == 0)
{
v___x_225_ = v___x_222_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v_a_220_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
else
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
lean_dec_ref(v_new__tp_158_);
lean_dec_ref(v_old__tp_157_);
v_a_228_ = lean_ctor_get(v___x_170_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_235_ == 0)
{
v___x_230_ = v___x_170_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_170_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_a_228_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_getInst___boxed(lean_object* v_old__tp_236_, lean_object* v_new__tp_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_Mathlib_Tactic_Lift_getInst(v_old__tp_236_, v_new__tp_237_, v_a_238_, v_a_239_, v_a_240_, v_a_241_);
lean_dec(v_a_241_);
lean_dec_ref(v_a_240_);
lean_dec(v_a_239_);
lean_dec_ref(v_a_238_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg(lean_object* v_e_244_, lean_object* v___y_245_){
_start:
{
uint8_t v___x_247_; 
v___x_247_ = l_Lean_Expr_hasMVar(v_e_244_);
if (v___x_247_ == 0)
{
lean_object* v___x_248_; 
v___x_248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_248_, 0, v_e_244_);
return v___x_248_;
}
else
{
lean_object* v___x_249_; lean_object* v_mctx_250_; lean_object* v___x_251_; lean_object* v_fst_252_; lean_object* v_snd_253_; lean_object* v___x_254_; lean_object* v_cache_255_; lean_object* v_zetaDeltaFVarIds_256_; lean_object* v_postponed_257_; lean_object* v_diag_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_267_; 
v___x_249_ = lean_st_ref_get(v___y_245_);
v_mctx_250_ = lean_ctor_get(v___x_249_, 0);
lean_inc_ref(v_mctx_250_);
lean_dec(v___x_249_);
v___x_251_ = l_Lean_instantiateMVarsCore(v_mctx_250_, v_e_244_);
v_fst_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_fst_252_);
v_snd_253_ = lean_ctor_get(v___x_251_, 1);
lean_inc(v_snd_253_);
lean_dec_ref(v___x_251_);
v___x_254_ = lean_st_ref_take(v___y_245_);
v_cache_255_ = lean_ctor_get(v___x_254_, 1);
v_zetaDeltaFVarIds_256_ = lean_ctor_get(v___x_254_, 2);
v_postponed_257_ = lean_ctor_get(v___x_254_, 3);
v_diag_258_ = lean_ctor_get(v___x_254_, 4);
v_isSharedCheck_267_ = !lean_is_exclusive(v___x_254_);
if (v_isSharedCheck_267_ == 0)
{
lean_object* v_unused_268_; 
v_unused_268_ = lean_ctor_get(v___x_254_, 0);
lean_dec(v_unused_268_);
v___x_260_ = v___x_254_;
v_isShared_261_ = v_isSharedCheck_267_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_diag_258_);
lean_inc(v_postponed_257_);
lean_inc(v_zetaDeltaFVarIds_256_);
lean_inc(v_cache_255_);
lean_dec(v___x_254_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_267_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_263_; 
if (v_isShared_261_ == 0)
{
lean_ctor_set(v___x_260_, 0, v_snd_253_);
v___x_263_ = v___x_260_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_266_; 
v_reuseFailAlloc_266_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_266_, 0, v_snd_253_);
lean_ctor_set(v_reuseFailAlloc_266_, 1, v_cache_255_);
lean_ctor_set(v_reuseFailAlloc_266_, 2, v_zetaDeltaFVarIds_256_);
lean_ctor_set(v_reuseFailAlloc_266_, 3, v_postponed_257_);
lean_ctor_set(v_reuseFailAlloc_266_, 4, v_diag_258_);
v___x_263_ = v_reuseFailAlloc_266_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_264_ = lean_st_ref_set(v___y_245_, v___x_263_);
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v_fst_252_);
return v___x_265_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg___boxed(lean_object* v_e_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg(v_e_269_, v___y_270_);
lean_dec(v___y_270_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1(lean_object* v_e_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg(v_e_273_, v___y_279_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___boxed(lean_object* v_e_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1(v_e_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_);
lean_dec(v___y_292_);
lean_dec_ref(v___y_291_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___redArg(lean_object* v_suggestion_295_, lean_object* v___y_296_){
_start:
{
lean_object* v_lctx_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v_lctx_298_ = lean_ctor_get(v___y_296_, 2);
v___x_299_ = lp_batteries_Lean_LocalContext_getUnusedUserName(v_lctx_298_, v_suggestion_295_);
v___x_300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___redArg___boxed(lean_object* v_suggestion_301_, lean_object* v___y_302_, lean_object* v___y_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___redArg(v_suggestion_301_, v___y_302_);
lean_dec_ref(v___y_302_);
lean_dec(v_suggestion_301_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4(lean_object* v_suggestion_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___redArg(v_suggestion_305_, v___y_310_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4___boxed(lean_object* v_suggestion_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_Lean_Meta_getUnusedUserName___at___00Mathlib_Tactic_Lift_main_spec__4(v_suggestion_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec(v___y_320_);
lean_dec_ref(v___y_319_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
lean_dec(v_suggestion_316_);
return v_res_326_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Mathlib_Tactic_Lift_main_spec__5(lean_object* v_x_327_, lean_object* v_x_328_){
_start:
{
if (lean_obj_tag(v_x_327_) == 0)
{
if (lean_obj_tag(v_x_328_) == 0)
{
uint8_t v___x_329_; 
v___x_329_ = 1;
return v___x_329_;
}
else
{
uint8_t v___x_330_; 
v___x_330_ = 0;
return v___x_330_;
}
}
else
{
if (lean_obj_tag(v_x_328_) == 0)
{
uint8_t v___x_331_; 
v___x_331_ = 0;
return v___x_331_;
}
else
{
lean_object* v_val_332_; lean_object* v_val_333_; uint8_t v___x_334_; 
v_val_332_ = lean_ctor_get(v_x_327_, 0);
v_val_333_ = lean_ctor_get(v_x_328_, 0);
v___x_334_ = l_Lean_Syntax_structEq(v_val_332_, v_val_333_);
return v___x_334_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Mathlib_Tactic_Lift_main_spec__5___boxed(lean_object* v_x_335_, lean_object* v_x_336_){
_start:
{
uint8_t v_res_337_; lean_object* v_r_338_; 
v_res_337_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_Tactic_Lift_main_spec__5(v_x_335_, v_x_336_);
lean_dec(v_x_336_);
lean_dec(v_x_335_);
v_r_338_ = lean_box(v_res_337_);
return v_r_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0_spec__0(lean_object* v_msgData_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_){
_start:
{
lean_object* v___x_345_; lean_object* v_env_346_; lean_object* v___x_347_; lean_object* v_mctx_348_; lean_object* v_lctx_349_; lean_object* v_options_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_345_ = lean_st_ref_get(v___y_343_);
v_env_346_ = lean_ctor_get(v___x_345_, 0);
lean_inc_ref(v_env_346_);
lean_dec(v___x_345_);
v___x_347_ = lean_st_ref_get(v___y_341_);
v_mctx_348_ = lean_ctor_get(v___x_347_, 0);
lean_inc_ref(v_mctx_348_);
lean_dec(v___x_347_);
v_lctx_349_ = lean_ctor_get(v___y_340_, 2);
v_options_350_ = lean_ctor_get(v___y_342_, 2);
lean_inc_ref(v_options_350_);
lean_inc_ref(v_lctx_349_);
v___x_351_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_351_, 0, v_env_346_);
lean_ctor_set(v___x_351_, 1, v_mctx_348_);
lean_ctor_set(v___x_351_, 2, v_lctx_349_);
lean_ctor_set(v___x_351_, 3, v_options_350_);
v___x_352_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v_msgData_339_);
v___x_353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_353_, 0, v___x_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0_spec__0___boxed(lean_object* v_msgData_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0_spec__0(v_msgData_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_);
lean_dec(v___y_358_);
lean_dec_ref(v___y_357_);
lean_dec(v___y_356_);
lean_dec_ref(v___y_355_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(lean_object* v_msg_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_){
_start:
{
lean_object* v_ref_367_; lean_object* v___x_368_; lean_object* v_a_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_377_; 
v_ref_367_ = lean_ctor_get(v___y_364_, 5);
v___x_368_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0_spec__0(v_msg_361_, v___y_362_, v___y_363_, v___y_364_, v___y_365_);
v_a_369_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_377_ == 0)
{
v___x_371_ = v___x_368_;
v_isShared_372_ = v_isSharedCheck_377_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_a_369_);
lean_dec(v___x_368_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_377_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_373_; lean_object* v___x_375_; 
lean_inc(v_ref_367_);
v___x_373_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_373_, 0, v_ref_367_);
lean_ctor_set(v___x_373_, 1, v_a_369_);
if (v_isShared_372_ == 0)
{
lean_ctor_set_tag(v___x_371_, 1);
lean_ctor_set(v___x_371_, 0, v___x_373_);
v___x_375_ = v___x_371_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v___x_373_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg___boxed(lean_object* v_msg_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(v_msg_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_);
lean_dec(v___y_382_);
lean_dec_ref(v___y_381_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
return v_res_384_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14(void){
_start:
{
lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_416_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__13));
v___x_417_ = l_String_toRawSubstring_x27(v___x_416_);
return v___x_417_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16(void){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = l_Array_mkArray0(lean_box(0));
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10(lean_object* v_newEqName_444_, lean_object* v___x_445_, lean_object* v_as_446_, size_t v_sz_447_, size_t v_i_448_, lean_object* v_b_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_){
_start:
{
uint8_t v___x_459_; 
v___x_459_ = lean_usize_dec_lt(v_i_448_, v_sz_447_);
if (v___x_459_ == 0)
{
lean_object* v___x_460_; 
lean_dec(v___x_445_);
v___x_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_460_, 0, v_b_449_);
return v___x_460_;
}
else
{
lean_object* v_snd_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_536_; 
v_snd_461_ = lean_ctor_get(v_b_449_, 1);
v_isSharedCheck_536_ = !lean_is_exclusive(v_b_449_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; 
v_unused_537_ = lean_ctor_get(v_b_449_, 0);
lean_dec(v_unused_537_);
v___x_463_ = v_b_449_;
v_isShared_464_ = v_isSharedCheck_536_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_snd_461_);
lean_dec(v_b_449_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_536_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v___x_465_; lean_object* v_a_467_; lean_object* v_a_474_; 
v___x_465_ = lean_box(0);
v_a_474_ = lean_array_uget_borrowed(v_as_446_, v_i_448_);
if (lean_obj_tag(v_a_474_) == 0)
{
v_a_467_ = v_snd_461_;
goto v___jp_466_;
}
else
{
lean_object* v_val_475_; lean_object* v___x_476_; lean_object* v___x_477_; uint8_t v___x_478_; 
lean_dec(v_snd_461_);
v_val_475_ = lean_ctor_get(v_a_474_, 0);
v___x_476_ = lean_box(0);
v___x_477_ = l_Lean_LocalDecl_userName(v_val_475_);
v___x_478_ = lean_name_eq(v___x_477_, v_newEqName_444_);
if (v___x_478_ == 0)
{
lean_object* v_ref_479_; lean_object* v_quotContext_480_; lean_object* v_currMacroScope_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; 
v_ref_479_ = lean_ctor_get(v___y_456_, 5);
v_quotContext_480_ = lean_ctor_get(v___y_456_, 10);
v_currMacroScope_481_ = lean_ctor_get(v___y_456_, 11);
v___x_482_ = l_Lean_mkIdent(v___x_477_);
v___x_483_ = l_Lean_SourceInfo_fromRef(v_ref_479_, v___x_478_);
v___x_484_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2));
v___x_485_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3));
lean_inc_n(v___x_483_, 22);
v___x_486_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_483_);
lean_ctor_set(v___x_486_, 1, v___x_484_);
v___x_487_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5));
v___x_488_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_489_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9));
v___x_490_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11));
v___x_491_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12));
v___x_492_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_483_);
lean_ctor_set(v___x_492_, 1, v___x_491_);
v___x_493_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14);
v___x_494_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15));
lean_inc(v_currMacroScope_481_);
lean_inc(v_quotContext_480_);
v___x_495_ = l_Lean_addMacroScope(v_quotContext_480_, v___x_494_, v_currMacroScope_481_);
v___x_496_ = lean_box(0);
v___x_497_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_497_, 0, v___x_483_);
lean_ctor_set(v___x_497_, 1, v___x_493_);
lean_ctor_set(v___x_497_, 2, v___x_495_);
lean_ctor_set(v___x_497_, 3, v___x_496_);
v___x_498_ = l_Lean_Syntax_node2(v___x_483_, v___x_490_, v___x_492_, v___x_497_);
v___x_499_ = l_Lean_Syntax_node1(v___x_483_, v___x_489_, v___x_498_);
v___x_500_ = l_Lean_Syntax_node1(v___x_483_, v___x_488_, v___x_499_);
v___x_501_ = l_Lean_Syntax_node1(v___x_483_, v___x_487_, v___x_500_);
v___x_502_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16);
v___x_503_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_503_, 0, v___x_483_);
lean_ctor_set(v___x_503_, 1, v___x_488_);
lean_ctor_set(v___x_503_, 2, v___x_502_);
v___x_504_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17));
v___x_505_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_483_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = l_Lean_Syntax_node1(v___x_483_, v___x_488_, v___x_505_);
v___x_507_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18));
v___x_508_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_483_);
lean_ctor_set(v___x_508_, 1, v___x_507_);
v___x_509_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20));
v___x_510_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21));
v___x_511_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_511_, 0, v___x_483_);
lean_ctor_set(v___x_511_, 1, v___x_510_);
v___x_512_ = l_Lean_Syntax_node1(v___x_483_, v___x_488_, v___x_511_);
lean_inc(v___x_445_);
lean_inc_ref(v___x_503_);
v___x_513_ = l_Lean_Syntax_node3(v___x_483_, v___x_509_, v___x_503_, v___x_512_, v___x_445_);
v___x_514_ = l_Lean_Syntax_node1(v___x_483_, v___x_488_, v___x_513_);
v___x_515_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22));
v___x_516_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_516_, 0, v___x_483_);
lean_ctor_set(v___x_516_, 1, v___x_515_);
v___x_517_ = l_Lean_Syntax_node3(v___x_483_, v___x_488_, v___x_508_, v___x_514_, v___x_516_);
v___x_518_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24));
v___x_519_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25));
v___x_520_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_483_);
lean_ctor_set(v___x_520_, 1, v___x_519_);
v___x_521_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27));
v___x_522_ = l_Lean_Syntax_node1(v___x_483_, v___x_488_, v___x_482_);
v___x_523_ = l_Lean_Syntax_node1(v___x_483_, v___x_521_, v___x_522_);
v___x_524_ = l_Lean_Syntax_node2(v___x_483_, v___x_518_, v___x_520_, v___x_523_);
v___x_525_ = l_Lean_Syntax_node1(v___x_483_, v___x_488_, v___x_524_);
v___x_526_ = l_Lean_Syntax_node6(v___x_483_, v___x_485_, v___x_486_, v___x_501_, v___x_503_, v___x_506_, v___x_517_, v___x_525_);
v___x_527_ = l_Lean_Elab_Tactic_evalTactic(v___x_526_, v___y_450_, v___y_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_);
if (lean_obj_tag(v___x_527_) == 0)
{
lean_dec_ref_known(v___x_527_, 1);
v_a_467_ = v___x_476_;
goto v___jp_466_;
}
else
{
lean_object* v_a_528_; lean_object* v___x_530_; uint8_t v_isShared_531_; uint8_t v_isSharedCheck_535_; 
lean_del_object(v___x_463_);
lean_dec(v___x_445_);
v_a_528_ = lean_ctor_get(v___x_527_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_527_);
if (v_isSharedCheck_535_ == 0)
{
v___x_530_ = v___x_527_;
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
else
{
lean_inc(v_a_528_);
lean_dec(v___x_527_);
v___x_530_ = lean_box(0);
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
v_resetjp_529_:
{
lean_object* v___x_533_; 
if (v_isShared_531_ == 0)
{
v___x_533_ = v___x_530_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v_a_528_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
}
}
else
{
lean_dec(v___x_477_);
v_a_467_ = v___x_476_;
goto v___jp_466_;
}
}
v___jp_466_:
{
lean_object* v___x_469_; 
if (v_isShared_464_ == 0)
{
lean_ctor_set(v___x_463_, 1, v_a_467_);
lean_ctor_set(v___x_463_, 0, v___x_465_);
v___x_469_ = v___x_463_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v___x_465_);
lean_ctor_set(v_reuseFailAlloc_473_, 1, v_a_467_);
v___x_469_ = v_reuseFailAlloc_473_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
size_t v___x_470_; size_t v___x_471_; 
v___x_470_ = ((size_t)1ULL);
v___x_471_ = lean_usize_add(v_i_448_, v___x_470_);
v_i_448_ = v___x_471_;
v_b_449_ = v___x_469_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___boxed(lean_object* v_newEqName_538_, lean_object* v___x_539_, lean_object* v_as_540_, lean_object* v_sz_541_, lean_object* v_i_542_, lean_object* v_b_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_){
_start:
{
size_t v_sz_boxed_553_; size_t v_i_boxed_554_; lean_object* v_res_555_; 
v_sz_boxed_553_ = lean_unbox_usize(v_sz_541_);
lean_dec(v_sz_541_);
v_i_boxed_554_ = lean_unbox_usize(v_i_542_);
lean_dec(v_i_542_);
v_res_555_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10(v_newEqName_538_, v___x_539_, v_as_540_, v_sz_boxed_553_, v_i_boxed_554_, v_b_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
lean_dec(v___y_545_);
lean_dec_ref(v___y_544_);
lean_dec_ref(v_as_540_);
lean_dec(v_newEqName_538_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5(lean_object* v_newEqName_556_, lean_object* v___x_557_, lean_object* v_as_558_, size_t v_sz_559_, size_t v_i_560_, lean_object* v_b_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
uint8_t v___x_571_; 
v___x_571_ = lean_usize_dec_lt(v_i_560_, v_sz_559_);
if (v___x_571_ == 0)
{
lean_object* v___x_572_; 
lean_dec(v___x_557_);
v___x_572_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_572_, 0, v_b_561_);
return v___x_572_;
}
else
{
lean_object* v_snd_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_648_; 
v_snd_573_ = lean_ctor_get(v_b_561_, 1);
v_isSharedCheck_648_ = !lean_is_exclusive(v_b_561_);
if (v_isSharedCheck_648_ == 0)
{
lean_object* v_unused_649_; 
v_unused_649_ = lean_ctor_get(v_b_561_, 0);
lean_dec(v_unused_649_);
v___x_575_ = v_b_561_;
v_isShared_576_ = v_isSharedCheck_648_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_snd_573_);
lean_dec(v_b_561_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_648_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_577_; lean_object* v_a_579_; lean_object* v_a_586_; 
v___x_577_ = lean_box(0);
v_a_586_ = lean_array_uget_borrowed(v_as_558_, v_i_560_);
if (lean_obj_tag(v_a_586_) == 0)
{
v_a_579_ = v_snd_573_;
goto v___jp_578_;
}
else
{
lean_object* v_val_587_; lean_object* v___x_588_; lean_object* v___x_589_; uint8_t v___x_590_; 
lean_dec(v_snd_573_);
v_val_587_ = lean_ctor_get(v_a_586_, 0);
v___x_588_ = lean_box(0);
v___x_589_ = l_Lean_LocalDecl_userName(v_val_587_);
v___x_590_ = lean_name_eq(v___x_589_, v_newEqName_556_);
if (v___x_590_ == 0)
{
lean_object* v_ref_591_; lean_object* v_quotContext_592_; lean_object* v_currMacroScope_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; 
v_ref_591_ = lean_ctor_get(v___y_568_, 5);
v_quotContext_592_ = lean_ctor_get(v___y_568_, 10);
v_currMacroScope_593_ = lean_ctor_get(v___y_568_, 11);
v___x_594_ = l_Lean_mkIdent(v___x_589_);
v___x_595_ = l_Lean_SourceInfo_fromRef(v_ref_591_, v___x_590_);
v___x_596_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2));
v___x_597_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3));
lean_inc_n(v___x_595_, 22);
v___x_598_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_595_);
lean_ctor_set(v___x_598_, 1, v___x_596_);
v___x_599_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5));
v___x_600_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_601_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9));
v___x_602_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11));
v___x_603_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12));
v___x_604_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_604_, 0, v___x_595_);
lean_ctor_set(v___x_604_, 1, v___x_603_);
v___x_605_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14);
v___x_606_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15));
lean_inc(v_currMacroScope_593_);
lean_inc(v_quotContext_592_);
v___x_607_ = l_Lean_addMacroScope(v_quotContext_592_, v___x_606_, v_currMacroScope_593_);
v___x_608_ = lean_box(0);
v___x_609_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_609_, 0, v___x_595_);
lean_ctor_set(v___x_609_, 1, v___x_605_);
lean_ctor_set(v___x_609_, 2, v___x_607_);
lean_ctor_set(v___x_609_, 3, v___x_608_);
v___x_610_ = l_Lean_Syntax_node2(v___x_595_, v___x_602_, v___x_604_, v___x_609_);
v___x_611_ = l_Lean_Syntax_node1(v___x_595_, v___x_601_, v___x_610_);
v___x_612_ = l_Lean_Syntax_node1(v___x_595_, v___x_600_, v___x_611_);
v___x_613_ = l_Lean_Syntax_node1(v___x_595_, v___x_599_, v___x_612_);
v___x_614_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16);
v___x_615_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_615_, 0, v___x_595_);
lean_ctor_set(v___x_615_, 1, v___x_600_);
lean_ctor_set(v___x_615_, 2, v___x_614_);
v___x_616_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17));
v___x_617_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_617_, 0, v___x_595_);
lean_ctor_set(v___x_617_, 1, v___x_616_);
v___x_618_ = l_Lean_Syntax_node1(v___x_595_, v___x_600_, v___x_617_);
v___x_619_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18));
v___x_620_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_595_);
lean_ctor_set(v___x_620_, 1, v___x_619_);
v___x_621_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20));
v___x_622_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21));
v___x_623_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_623_, 0, v___x_595_);
lean_ctor_set(v___x_623_, 1, v___x_622_);
v___x_624_ = l_Lean_Syntax_node1(v___x_595_, v___x_600_, v___x_623_);
lean_inc(v___x_557_);
lean_inc_ref(v___x_615_);
v___x_625_ = l_Lean_Syntax_node3(v___x_595_, v___x_621_, v___x_615_, v___x_624_, v___x_557_);
v___x_626_ = l_Lean_Syntax_node1(v___x_595_, v___x_600_, v___x_625_);
v___x_627_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22));
v___x_628_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_595_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
v___x_629_ = l_Lean_Syntax_node3(v___x_595_, v___x_600_, v___x_620_, v___x_626_, v___x_628_);
v___x_630_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24));
v___x_631_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25));
v___x_632_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_632_, 0, v___x_595_);
lean_ctor_set(v___x_632_, 1, v___x_631_);
v___x_633_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27));
v___x_634_ = l_Lean_Syntax_node1(v___x_595_, v___x_600_, v___x_594_);
v___x_635_ = l_Lean_Syntax_node1(v___x_595_, v___x_633_, v___x_634_);
v___x_636_ = l_Lean_Syntax_node2(v___x_595_, v___x_630_, v___x_632_, v___x_635_);
v___x_637_ = l_Lean_Syntax_node1(v___x_595_, v___x_600_, v___x_636_);
v___x_638_ = l_Lean_Syntax_node6(v___x_595_, v___x_597_, v___x_598_, v___x_613_, v___x_615_, v___x_618_, v___x_629_, v___x_637_);
v___x_639_ = l_Lean_Elab_Tactic_evalTactic(v___x_638_, v___y_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
if (lean_obj_tag(v___x_639_) == 0)
{
lean_dec_ref_known(v___x_639_, 1);
v_a_579_ = v___x_588_;
goto v___jp_578_;
}
else
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_del_object(v___x_575_);
lean_dec(v___x_557_);
v_a_640_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_639_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_639_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
}
else
{
lean_dec(v___x_589_);
v_a_579_ = v___x_588_;
goto v___jp_578_;
}
}
v___jp_578_:
{
lean_object* v___x_581_; 
if (v_isShared_576_ == 0)
{
lean_ctor_set(v___x_575_, 1, v_a_579_);
lean_ctor_set(v___x_575_, 0, v___x_577_);
v___x_581_ = v___x_575_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_585_; 
v_reuseFailAlloc_585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_585_, 0, v___x_577_);
lean_ctor_set(v_reuseFailAlloc_585_, 1, v_a_579_);
v___x_581_ = v_reuseFailAlloc_585_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
size_t v___x_582_; size_t v___x_583_; lean_object* v___x_584_; 
v___x_582_ = ((size_t)1ULL);
v___x_583_ = lean_usize_add(v_i_560_, v___x_582_);
v___x_584_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10(v_newEqName_556_, v___x_557_, v_as_558_, v_sz_559_, v___x_583_, v___x_581_, v___y_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
return v___x_584_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5___boxed(lean_object* v_newEqName_650_, lean_object* v___x_651_, lean_object* v_as_652_, lean_object* v_sz_653_, lean_object* v_i_654_, lean_object* v_b_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
size_t v_sz_boxed_665_; size_t v_i_boxed_666_; lean_object* v_res_667_; 
v_sz_boxed_665_ = lean_unbox_usize(v_sz_653_);
lean_dec(v_sz_653_);
v_i_boxed_666_ = lean_unbox_usize(v_i_654_);
lean_dec(v_i_654_);
v_res_667_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5(v_newEqName_650_, v___x_651_, v_as_652_, v_sz_boxed_665_, v_i_boxed_666_, v_b_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, v___y_662_, v___y_663_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec_ref(v_as_652_);
lean_dec(v_newEqName_650_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8_spec__9(lean_object* v_newEqName_668_, lean_object* v___x_669_, lean_object* v_as_670_, size_t v_sz_671_, size_t v_i_672_, lean_object* v_b_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_){
_start:
{
uint8_t v___x_683_; 
v___x_683_ = lean_usize_dec_lt(v_i_672_, v_sz_671_);
if (v___x_683_ == 0)
{
lean_object* v___x_684_; 
lean_dec(v___x_669_);
v___x_684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_684_, 0, v_b_673_);
return v___x_684_;
}
else
{
lean_object* v_snd_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_760_; 
v_snd_685_ = lean_ctor_get(v_b_673_, 1);
v_isSharedCheck_760_ = !lean_is_exclusive(v_b_673_);
if (v_isSharedCheck_760_ == 0)
{
lean_object* v_unused_761_; 
v_unused_761_ = lean_ctor_get(v_b_673_, 0);
lean_dec(v_unused_761_);
v___x_687_ = v_b_673_;
v_isShared_688_ = v_isSharedCheck_760_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_snd_685_);
lean_dec(v_b_673_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_760_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_689_; lean_object* v_a_691_; lean_object* v_a_698_; 
v___x_689_ = lean_box(0);
v_a_698_ = lean_array_uget_borrowed(v_as_670_, v_i_672_);
if (lean_obj_tag(v_a_698_) == 0)
{
v_a_691_ = v_snd_685_;
goto v___jp_690_;
}
else
{
lean_object* v_val_699_; lean_object* v___x_700_; lean_object* v___x_701_; uint8_t v___x_702_; 
lean_dec(v_snd_685_);
v_val_699_ = lean_ctor_get(v_a_698_, 0);
v___x_700_ = lean_box(0);
v___x_701_ = l_Lean_LocalDecl_userName(v_val_699_);
v___x_702_ = lean_name_eq(v___x_701_, v_newEqName_668_);
if (v___x_702_ == 0)
{
lean_object* v_ref_703_; lean_object* v_quotContext_704_; lean_object* v_currMacroScope_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; 
v_ref_703_ = lean_ctor_get(v___y_680_, 5);
v_quotContext_704_ = lean_ctor_get(v___y_680_, 10);
v_currMacroScope_705_ = lean_ctor_get(v___y_680_, 11);
v___x_706_ = l_Lean_mkIdent(v___x_701_);
v___x_707_ = l_Lean_SourceInfo_fromRef(v_ref_703_, v___x_702_);
v___x_708_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2));
v___x_709_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3));
lean_inc_n(v___x_707_, 22);
v___x_710_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_710_, 0, v___x_707_);
lean_ctor_set(v___x_710_, 1, v___x_708_);
v___x_711_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5));
v___x_712_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_713_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9));
v___x_714_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11));
v___x_715_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12));
v___x_716_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_716_, 0, v___x_707_);
lean_ctor_set(v___x_716_, 1, v___x_715_);
v___x_717_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14);
v___x_718_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15));
lean_inc(v_currMacroScope_705_);
lean_inc(v_quotContext_704_);
v___x_719_ = l_Lean_addMacroScope(v_quotContext_704_, v___x_718_, v_currMacroScope_705_);
v___x_720_ = lean_box(0);
v___x_721_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_721_, 0, v___x_707_);
lean_ctor_set(v___x_721_, 1, v___x_717_);
lean_ctor_set(v___x_721_, 2, v___x_719_);
lean_ctor_set(v___x_721_, 3, v___x_720_);
v___x_722_ = l_Lean_Syntax_node2(v___x_707_, v___x_714_, v___x_716_, v___x_721_);
v___x_723_ = l_Lean_Syntax_node1(v___x_707_, v___x_713_, v___x_722_);
v___x_724_ = l_Lean_Syntax_node1(v___x_707_, v___x_712_, v___x_723_);
v___x_725_ = l_Lean_Syntax_node1(v___x_707_, v___x_711_, v___x_724_);
v___x_726_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16);
v___x_727_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_727_, 0, v___x_707_);
lean_ctor_set(v___x_727_, 1, v___x_712_);
lean_ctor_set(v___x_727_, 2, v___x_726_);
v___x_728_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17));
v___x_729_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_729_, 0, v___x_707_);
lean_ctor_set(v___x_729_, 1, v___x_728_);
v___x_730_ = l_Lean_Syntax_node1(v___x_707_, v___x_712_, v___x_729_);
v___x_731_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18));
v___x_732_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_732_, 0, v___x_707_);
lean_ctor_set(v___x_732_, 1, v___x_731_);
v___x_733_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20));
v___x_734_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21));
v___x_735_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_735_, 0, v___x_707_);
lean_ctor_set(v___x_735_, 1, v___x_734_);
v___x_736_ = l_Lean_Syntax_node1(v___x_707_, v___x_712_, v___x_735_);
lean_inc(v___x_669_);
lean_inc_ref(v___x_727_);
v___x_737_ = l_Lean_Syntax_node3(v___x_707_, v___x_733_, v___x_727_, v___x_736_, v___x_669_);
v___x_738_ = l_Lean_Syntax_node1(v___x_707_, v___x_712_, v___x_737_);
v___x_739_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22));
v___x_740_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_707_);
lean_ctor_set(v___x_740_, 1, v___x_739_);
v___x_741_ = l_Lean_Syntax_node3(v___x_707_, v___x_712_, v___x_732_, v___x_738_, v___x_740_);
v___x_742_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24));
v___x_743_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25));
v___x_744_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_744_, 0, v___x_707_);
lean_ctor_set(v___x_744_, 1, v___x_743_);
v___x_745_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27));
v___x_746_ = l_Lean_Syntax_node1(v___x_707_, v___x_712_, v___x_706_);
v___x_747_ = l_Lean_Syntax_node1(v___x_707_, v___x_745_, v___x_746_);
v___x_748_ = l_Lean_Syntax_node2(v___x_707_, v___x_742_, v___x_744_, v___x_747_);
v___x_749_ = l_Lean_Syntax_node1(v___x_707_, v___x_712_, v___x_748_);
v___x_750_ = l_Lean_Syntax_node6(v___x_707_, v___x_709_, v___x_710_, v___x_725_, v___x_727_, v___x_730_, v___x_741_, v___x_749_);
v___x_751_ = l_Lean_Elab_Tactic_evalTactic(v___x_750_, v___y_674_, v___y_675_, v___y_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_);
if (lean_obj_tag(v___x_751_) == 0)
{
lean_dec_ref_known(v___x_751_, 1);
v_a_691_ = v___x_700_;
goto v___jp_690_;
}
else
{
lean_object* v_a_752_; lean_object* v___x_754_; uint8_t v_isShared_755_; uint8_t v_isSharedCheck_759_; 
lean_del_object(v___x_687_);
lean_dec(v___x_669_);
v_a_752_ = lean_ctor_get(v___x_751_, 0);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_751_);
if (v_isSharedCheck_759_ == 0)
{
v___x_754_ = v___x_751_;
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
else
{
lean_inc(v_a_752_);
lean_dec(v___x_751_);
v___x_754_ = lean_box(0);
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
v_resetjp_753_:
{
lean_object* v___x_757_; 
if (v_isShared_755_ == 0)
{
v___x_757_ = v___x_754_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_758_; 
v_reuseFailAlloc_758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_758_, 0, v_a_752_);
v___x_757_ = v_reuseFailAlloc_758_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
return v___x_757_;
}
}
}
}
else
{
lean_dec(v___x_701_);
v_a_691_ = v___x_700_;
goto v___jp_690_;
}
}
v___jp_690_:
{
lean_object* v___x_693_; 
if (v_isShared_688_ == 0)
{
lean_ctor_set(v___x_687_, 1, v_a_691_);
lean_ctor_set(v___x_687_, 0, v___x_689_);
v___x_693_ = v___x_687_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v___x_689_);
lean_ctor_set(v_reuseFailAlloc_697_, 1, v_a_691_);
v___x_693_ = v_reuseFailAlloc_697_;
goto v_reusejp_692_;
}
v_reusejp_692_:
{
size_t v___x_694_; size_t v___x_695_; 
v___x_694_ = ((size_t)1ULL);
v___x_695_ = lean_usize_add(v_i_672_, v___x_694_);
v_i_672_ = v___x_695_;
v_b_673_ = v___x_693_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8_spec__9___boxed(lean_object* v_newEqName_762_, lean_object* v___x_763_, lean_object* v_as_764_, lean_object* v_sz_765_, lean_object* v_i_766_, lean_object* v_b_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_){
_start:
{
size_t v_sz_boxed_777_; size_t v_i_boxed_778_; lean_object* v_res_779_; 
v_sz_boxed_777_ = lean_unbox_usize(v_sz_765_);
lean_dec(v_sz_765_);
v_i_boxed_778_ = lean_unbox_usize(v_i_766_);
lean_dec(v_i_766_);
v_res_779_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8_spec__9(v_newEqName_762_, v___x_763_, v_as_764_, v_sz_boxed_777_, v_i_boxed_778_, v_b_767_, v___y_768_, v___y_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_);
lean_dec(v___y_775_);
lean_dec_ref(v___y_774_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
lean_dec(v___y_769_);
lean_dec_ref(v___y_768_);
lean_dec_ref(v_as_764_);
lean_dec(v_newEqName_762_);
return v_res_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8(lean_object* v_newEqName_780_, lean_object* v___x_781_, lean_object* v_as_782_, size_t v_sz_783_, size_t v_i_784_, lean_object* v_b_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
uint8_t v___x_795_; 
v___x_795_ = lean_usize_dec_lt(v_i_784_, v_sz_783_);
if (v___x_795_ == 0)
{
lean_object* v___x_796_; 
lean_dec(v___x_781_);
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v_b_785_);
return v___x_796_;
}
else
{
lean_object* v_snd_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_872_; 
v_snd_797_ = lean_ctor_get(v_b_785_, 1);
v_isSharedCheck_872_ = !lean_is_exclusive(v_b_785_);
if (v_isSharedCheck_872_ == 0)
{
lean_object* v_unused_873_; 
v_unused_873_ = lean_ctor_get(v_b_785_, 0);
lean_dec(v_unused_873_);
v___x_799_ = v_b_785_;
v_isShared_800_ = v_isSharedCheck_872_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_snd_797_);
lean_dec(v_b_785_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_872_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_801_; lean_object* v_a_803_; lean_object* v_a_810_; 
v___x_801_ = lean_box(0);
v_a_810_ = lean_array_uget_borrowed(v_as_782_, v_i_784_);
if (lean_obj_tag(v_a_810_) == 0)
{
v_a_803_ = v_snd_797_;
goto v___jp_802_;
}
else
{
lean_object* v_val_811_; lean_object* v___x_812_; lean_object* v___x_813_; uint8_t v___x_814_; 
lean_dec(v_snd_797_);
v_val_811_ = lean_ctor_get(v_a_810_, 0);
v___x_812_ = lean_box(0);
v___x_813_ = l_Lean_LocalDecl_userName(v_val_811_);
v___x_814_ = lean_name_eq(v___x_813_, v_newEqName_780_);
if (v___x_814_ == 0)
{
lean_object* v_ref_815_; lean_object* v_quotContext_816_; lean_object* v_currMacroScope_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; 
v_ref_815_ = lean_ctor_get(v___y_792_, 5);
v_quotContext_816_ = lean_ctor_get(v___y_792_, 10);
v_currMacroScope_817_ = lean_ctor_get(v___y_792_, 11);
v___x_818_ = l_Lean_mkIdent(v___x_813_);
v___x_819_ = l_Lean_SourceInfo_fromRef(v_ref_815_, v___x_814_);
v___x_820_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2));
v___x_821_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3));
lean_inc_n(v___x_819_, 22);
v___x_822_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_822_, 0, v___x_819_);
lean_ctor_set(v___x_822_, 1, v___x_820_);
v___x_823_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5));
v___x_824_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_825_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9));
v___x_826_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11));
v___x_827_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12));
v___x_828_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_828_, 0, v___x_819_);
lean_ctor_set(v___x_828_, 1, v___x_827_);
v___x_829_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14);
v___x_830_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15));
lean_inc(v_currMacroScope_817_);
lean_inc(v_quotContext_816_);
v___x_831_ = l_Lean_addMacroScope(v_quotContext_816_, v___x_830_, v_currMacroScope_817_);
v___x_832_ = lean_box(0);
v___x_833_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_833_, 0, v___x_819_);
lean_ctor_set(v___x_833_, 1, v___x_829_);
lean_ctor_set(v___x_833_, 2, v___x_831_);
lean_ctor_set(v___x_833_, 3, v___x_832_);
v___x_834_ = l_Lean_Syntax_node2(v___x_819_, v___x_826_, v___x_828_, v___x_833_);
v___x_835_ = l_Lean_Syntax_node1(v___x_819_, v___x_825_, v___x_834_);
v___x_836_ = l_Lean_Syntax_node1(v___x_819_, v___x_824_, v___x_835_);
v___x_837_ = l_Lean_Syntax_node1(v___x_819_, v___x_823_, v___x_836_);
v___x_838_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16);
v___x_839_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_839_, 0, v___x_819_);
lean_ctor_set(v___x_839_, 1, v___x_824_);
lean_ctor_set(v___x_839_, 2, v___x_838_);
v___x_840_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17));
v___x_841_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_841_, 0, v___x_819_);
lean_ctor_set(v___x_841_, 1, v___x_840_);
v___x_842_ = l_Lean_Syntax_node1(v___x_819_, v___x_824_, v___x_841_);
v___x_843_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18));
v___x_844_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_844_, 0, v___x_819_);
lean_ctor_set(v___x_844_, 1, v___x_843_);
v___x_845_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20));
v___x_846_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21));
v___x_847_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_819_);
lean_ctor_set(v___x_847_, 1, v___x_846_);
v___x_848_ = l_Lean_Syntax_node1(v___x_819_, v___x_824_, v___x_847_);
lean_inc(v___x_781_);
lean_inc_ref(v___x_839_);
v___x_849_ = l_Lean_Syntax_node3(v___x_819_, v___x_845_, v___x_839_, v___x_848_, v___x_781_);
v___x_850_ = l_Lean_Syntax_node1(v___x_819_, v___x_824_, v___x_849_);
v___x_851_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22));
v___x_852_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_852_, 0, v___x_819_);
lean_ctor_set(v___x_852_, 1, v___x_851_);
v___x_853_ = l_Lean_Syntax_node3(v___x_819_, v___x_824_, v___x_844_, v___x_850_, v___x_852_);
v___x_854_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__24));
v___x_855_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__25));
v___x_856_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_856_, 0, v___x_819_);
lean_ctor_set(v___x_856_, 1, v___x_855_);
v___x_857_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__27));
v___x_858_ = l_Lean_Syntax_node1(v___x_819_, v___x_824_, v___x_818_);
v___x_859_ = l_Lean_Syntax_node1(v___x_819_, v___x_857_, v___x_858_);
v___x_860_ = l_Lean_Syntax_node2(v___x_819_, v___x_854_, v___x_856_, v___x_859_);
v___x_861_ = l_Lean_Syntax_node1(v___x_819_, v___x_824_, v___x_860_);
v___x_862_ = l_Lean_Syntax_node6(v___x_819_, v___x_821_, v___x_822_, v___x_837_, v___x_839_, v___x_842_, v___x_853_, v___x_861_);
v___x_863_ = l_Lean_Elab_Tactic_evalTactic(v___x_862_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_);
if (lean_obj_tag(v___x_863_) == 0)
{
lean_dec_ref_known(v___x_863_, 1);
v_a_803_ = v___x_812_;
goto v___jp_802_;
}
else
{
lean_object* v_a_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_871_; 
lean_del_object(v___x_799_);
lean_dec(v___x_781_);
v_a_864_ = lean_ctor_get(v___x_863_, 0);
v_isSharedCheck_871_ = !lean_is_exclusive(v___x_863_);
if (v_isSharedCheck_871_ == 0)
{
v___x_866_ = v___x_863_;
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_a_864_);
lean_dec(v___x_863_);
v___x_866_ = lean_box(0);
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
v_resetjp_865_:
{
lean_object* v___x_869_; 
if (v_isShared_867_ == 0)
{
v___x_869_ = v___x_866_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v_a_864_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
}
}
else
{
lean_dec(v___x_813_);
v_a_803_ = v___x_812_;
goto v___jp_802_;
}
}
v___jp_802_:
{
lean_object* v___x_805_; 
if (v_isShared_800_ == 0)
{
lean_ctor_set(v___x_799_, 1, v_a_803_);
lean_ctor_set(v___x_799_, 0, v___x_801_);
v___x_805_ = v___x_799_;
goto v_reusejp_804_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v___x_801_);
lean_ctor_set(v_reuseFailAlloc_809_, 1, v_a_803_);
v___x_805_ = v_reuseFailAlloc_809_;
goto v_reusejp_804_;
}
v_reusejp_804_:
{
size_t v___x_806_; size_t v___x_807_; lean_object* v___x_808_; 
v___x_806_ = ((size_t)1ULL);
v___x_807_ = lean_usize_add(v_i_784_, v___x_806_);
v___x_808_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8_spec__9(v_newEqName_780_, v___x_781_, v_as_782_, v_sz_783_, v___x_807_, v___x_805_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_);
return v___x_808_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8___boxed(lean_object* v_newEqName_874_, lean_object* v___x_875_, lean_object* v_as_876_, lean_object* v_sz_877_, lean_object* v_i_878_, lean_object* v_b_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_){
_start:
{
size_t v_sz_boxed_889_; size_t v_i_boxed_890_; lean_object* v_res_891_; 
v_sz_boxed_889_ = lean_unbox_usize(v_sz_877_);
lean_dec(v_sz_877_);
v_i_boxed_890_ = lean_unbox_usize(v_i_878_);
lean_dec(v_i_878_);
v_res_891_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8(v_newEqName_874_, v___x_875_, v_as_876_, v_sz_boxed_889_, v_i_boxed_890_, v_b_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_);
lean_dec(v___y_887_);
lean_dec_ref(v___y_886_);
lean_dec(v___y_885_);
lean_dec_ref(v___y_884_);
lean_dec(v___y_883_);
lean_dec_ref(v___y_882_);
lean_dec(v___y_881_);
lean_dec_ref(v___y_880_);
lean_dec_ref(v_as_876_);
lean_dec(v_newEqName_874_);
return v_res_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4(lean_object* v_init_892_, lean_object* v_newEqName_893_, lean_object* v___x_894_, lean_object* v_n_895_, lean_object* v_b_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_){
_start:
{
if (lean_obj_tag(v_n_895_) == 0)
{
lean_object* v_cs_906_; lean_object* v___x_907_; lean_object* v___x_908_; size_t v_sz_909_; size_t v___x_910_; lean_object* v___x_911_; 
v_cs_906_ = lean_ctor_get(v_n_895_, 0);
v___x_907_ = lean_box(0);
v___x_908_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v_b_896_);
v_sz_909_ = lean_array_size(v_cs_906_);
v___x_910_ = ((size_t)0ULL);
v___x_911_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__7(v_init_892_, v_newEqName_893_, v___x_894_, v_cs_906_, v_sz_909_, v___x_910_, v___x_908_, v___y_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_, v___y_904_);
if (lean_obj_tag(v___x_911_) == 0)
{
lean_object* v_a_912_; lean_object* v___x_914_; uint8_t v_isShared_915_; uint8_t v_isSharedCheck_926_; 
v_a_912_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_926_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_926_ == 0)
{
v___x_914_ = v___x_911_;
v_isShared_915_ = v_isSharedCheck_926_;
goto v_resetjp_913_;
}
else
{
lean_inc(v_a_912_);
lean_dec(v___x_911_);
v___x_914_ = lean_box(0);
v_isShared_915_ = v_isSharedCheck_926_;
goto v_resetjp_913_;
}
v_resetjp_913_:
{
lean_object* v_fst_916_; 
v_fst_916_ = lean_ctor_get(v_a_912_, 0);
if (lean_obj_tag(v_fst_916_) == 0)
{
lean_object* v_snd_917_; lean_object* v___x_918_; lean_object* v___x_920_; 
v_snd_917_ = lean_ctor_get(v_a_912_, 1);
lean_inc(v_snd_917_);
lean_dec(v_a_912_);
v___x_918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_918_, 0, v_snd_917_);
if (v_isShared_915_ == 0)
{
lean_ctor_set(v___x_914_, 0, v___x_918_);
v___x_920_ = v___x_914_;
goto v_reusejp_919_;
}
else
{
lean_object* v_reuseFailAlloc_921_; 
v_reuseFailAlloc_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_921_, 0, v___x_918_);
v___x_920_ = v_reuseFailAlloc_921_;
goto v_reusejp_919_;
}
v_reusejp_919_:
{
return v___x_920_;
}
}
else
{
lean_object* v_val_922_; lean_object* v___x_924_; 
lean_inc_ref(v_fst_916_);
lean_dec(v_a_912_);
v_val_922_ = lean_ctor_get(v_fst_916_, 0);
lean_inc(v_val_922_);
lean_dec_ref_known(v_fst_916_, 1);
if (v_isShared_915_ == 0)
{
lean_ctor_set(v___x_914_, 0, v_val_922_);
v___x_924_ = v___x_914_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, v_val_922_);
v___x_924_ = v_reuseFailAlloc_925_;
goto v_reusejp_923_;
}
v_reusejp_923_:
{
return v___x_924_;
}
}
}
}
else
{
lean_object* v_a_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_934_; 
v_a_927_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_934_ == 0)
{
v___x_929_ = v___x_911_;
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_a_927_);
lean_dec(v___x_911_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_932_; 
if (v_isShared_930_ == 0)
{
v___x_932_ = v___x_929_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_a_927_);
v___x_932_ = v_reuseFailAlloc_933_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
return v___x_932_;
}
}
}
}
else
{
lean_object* v_vs_935_; lean_object* v___x_936_; lean_object* v___x_937_; size_t v_sz_938_; size_t v___x_939_; lean_object* v___x_940_; 
v_vs_935_ = lean_ctor_get(v_n_895_, 0);
v___x_936_ = lean_box(0);
v___x_937_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_937_, 0, v___x_936_);
lean_ctor_set(v___x_937_, 1, v_b_896_);
v_sz_938_ = lean_array_size(v_vs_935_);
v___x_939_ = ((size_t)0ULL);
v___x_940_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__8(v_newEqName_893_, v___x_894_, v_vs_935_, v_sz_938_, v___x_939_, v___x_937_, v___y_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_, v___y_904_);
if (lean_obj_tag(v___x_940_) == 0)
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_955_; 
v_a_941_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_955_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_955_ == 0)
{
v___x_943_ = v___x_940_;
v_isShared_944_ = v_isSharedCheck_955_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_940_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_955_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v_fst_945_; 
v_fst_945_ = lean_ctor_get(v_a_941_, 0);
if (lean_obj_tag(v_fst_945_) == 0)
{
lean_object* v_snd_946_; lean_object* v___x_947_; lean_object* v___x_949_; 
v_snd_946_ = lean_ctor_get(v_a_941_, 1);
lean_inc(v_snd_946_);
lean_dec(v_a_941_);
v___x_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_947_, 0, v_snd_946_);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 0, v___x_947_);
v___x_949_ = v___x_943_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_950_; 
v_reuseFailAlloc_950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_950_, 0, v___x_947_);
v___x_949_ = v_reuseFailAlloc_950_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
return v___x_949_;
}
}
else
{
lean_object* v_val_951_; lean_object* v___x_953_; 
lean_inc_ref(v_fst_945_);
lean_dec(v_a_941_);
v_val_951_ = lean_ctor_get(v_fst_945_, 0);
lean_inc(v_val_951_);
lean_dec_ref_known(v_fst_945_, 1);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 0, v_val_951_);
v___x_953_ = v___x_943_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v_val_951_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
return v___x_953_;
}
}
}
}
else
{
lean_object* v_a_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_963_; 
v_a_956_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_963_ == 0)
{
v___x_958_ = v___x_940_;
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_a_956_);
lean_dec(v___x_940_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
if (v_isShared_959_ == 0)
{
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v_a_956_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
return v___x_961_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__7(lean_object* v_init_964_, lean_object* v_newEqName_965_, lean_object* v___x_966_, lean_object* v_as_967_, size_t v_sz_968_, size_t v_i_969_, lean_object* v_b_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_){
_start:
{
uint8_t v___x_980_; 
v___x_980_ = lean_usize_dec_lt(v_i_969_, v_sz_968_);
if (v___x_980_ == 0)
{
lean_object* v___x_981_; 
lean_dec(v___x_966_);
v___x_981_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_981_, 0, v_b_970_);
return v___x_981_;
}
else
{
lean_object* v_snd_982_; lean_object* v___x_984_; uint8_t v_isShared_985_; uint8_t v_isSharedCheck_1016_; 
v_snd_982_ = lean_ctor_get(v_b_970_, 1);
v_isSharedCheck_1016_ = !lean_is_exclusive(v_b_970_);
if (v_isSharedCheck_1016_ == 0)
{
lean_object* v_unused_1017_; 
v_unused_1017_ = lean_ctor_get(v_b_970_, 0);
lean_dec(v_unused_1017_);
v___x_984_ = v_b_970_;
v_isShared_985_ = v_isSharedCheck_1016_;
goto v_resetjp_983_;
}
else
{
lean_inc(v_snd_982_);
lean_dec(v_b_970_);
v___x_984_ = lean_box(0);
v_isShared_985_ = v_isSharedCheck_1016_;
goto v_resetjp_983_;
}
v_resetjp_983_:
{
lean_object* v_a_986_; lean_object* v___x_987_; 
v_a_986_ = lean_array_uget_borrowed(v_as_967_, v_i_969_);
lean_inc(v_snd_982_);
lean_inc(v___x_966_);
v___x_987_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4(v_init_964_, v_newEqName_965_, v___x_966_, v_a_986_, v_snd_982_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_, v___y_977_, v___y_978_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v_a_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_1007_; 
v_a_988_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_990_ = v___x_987_;
v_isShared_991_ = v_isSharedCheck_1007_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_a_988_);
lean_dec(v___x_987_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_1007_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
if (lean_obj_tag(v_a_988_) == 0)
{
lean_object* v___x_992_; lean_object* v___x_994_; 
lean_dec(v___x_966_);
v___x_992_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_992_, 0, v_a_988_);
if (v_isShared_985_ == 0)
{
lean_ctor_set(v___x_984_, 0, v___x_992_);
v___x_994_ = v___x_984_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_998_; 
v_reuseFailAlloc_998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_998_, 0, v___x_992_);
lean_ctor_set(v_reuseFailAlloc_998_, 1, v_snd_982_);
v___x_994_ = v_reuseFailAlloc_998_;
goto v_reusejp_993_;
}
v_reusejp_993_:
{
lean_object* v___x_996_; 
if (v_isShared_991_ == 0)
{
lean_ctor_set(v___x_990_, 0, v___x_994_);
v___x_996_ = v___x_990_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v___x_994_);
v___x_996_ = v_reuseFailAlloc_997_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
return v___x_996_;
}
}
}
else
{
lean_object* v_a_999_; lean_object* v___x_1000_; lean_object* v___x_1002_; 
lean_del_object(v___x_990_);
lean_dec(v_snd_982_);
v_a_999_ = lean_ctor_get(v_a_988_, 0);
lean_inc(v_a_999_);
lean_dec_ref_known(v_a_988_, 1);
v___x_1000_ = lean_box(0);
if (v_isShared_985_ == 0)
{
lean_ctor_set(v___x_984_, 1, v_a_999_);
lean_ctor_set(v___x_984_, 0, v___x_1000_);
v___x_1002_ = v___x_984_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v___x_1000_);
lean_ctor_set(v_reuseFailAlloc_1006_, 1, v_a_999_);
v___x_1002_ = v_reuseFailAlloc_1006_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
size_t v___x_1003_; size_t v___x_1004_; 
v___x_1003_ = ((size_t)1ULL);
v___x_1004_ = lean_usize_add(v_i_969_, v___x_1003_);
v_i_969_ = v___x_1004_;
v_b_970_ = v___x_1002_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
lean_del_object(v___x_984_);
lean_dec(v_snd_982_);
lean_dec(v___x_966_);
v_a_1008_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1015_ == 0)
{
v___x_1010_ = v___x_987_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_inc(v_a_1008_);
lean_dec(v___x_987_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
v___x_1013_ = v___x_1010_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v_a_1008_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__7___boxed(lean_object* v_init_1018_, lean_object* v_newEqName_1019_, lean_object* v___x_1020_, lean_object* v_as_1021_, lean_object* v_sz_1022_, lean_object* v_i_1023_, lean_object* v_b_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_){
_start:
{
size_t v_sz_boxed_1034_; size_t v_i_boxed_1035_; lean_object* v_res_1036_; 
v_sz_boxed_1034_ = lean_unbox_usize(v_sz_1022_);
lean_dec(v_sz_1022_);
v_i_boxed_1035_ = lean_unbox_usize(v_i_1023_);
lean_dec(v_i_1023_);
v_res_1036_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4_spec__7(v_init_1018_, v_newEqName_1019_, v___x_1020_, v_as_1021_, v_sz_boxed_1034_, v_i_boxed_1035_, v_b_1024_, v___y_1025_, v___y_1026_, v___y_1027_, v___y_1028_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
lean_dec(v___y_1032_);
lean_dec_ref(v___y_1031_);
lean_dec(v___y_1030_);
lean_dec_ref(v___y_1029_);
lean_dec(v___y_1028_);
lean_dec_ref(v___y_1027_);
lean_dec(v___y_1026_);
lean_dec_ref(v___y_1025_);
lean_dec_ref(v_as_1021_);
lean_dec(v_newEqName_1019_);
return v_res_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4___boxed(lean_object* v_init_1037_, lean_object* v_newEqName_1038_, lean_object* v___x_1039_, lean_object* v_n_1040_, lean_object* v_b_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4(v_init_1037_, v_newEqName_1038_, v___x_1039_, v_n_1040_, v_b_1041_, v___y_1042_, v___y_1043_, v___y_1044_, v___y_1045_, v___y_1046_, v___y_1047_, v___y_1048_, v___y_1049_);
lean_dec(v___y_1049_);
lean_dec_ref(v___y_1048_);
lean_dec(v___y_1047_);
lean_dec_ref(v___y_1046_);
lean_dec(v___y_1045_);
lean_dec_ref(v___y_1044_);
lean_dec(v___y_1043_);
lean_dec_ref(v___y_1042_);
lean_dec_ref(v_n_1040_);
lean_dec(v_newEqName_1038_);
return v_res_1051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3(lean_object* v_newEqName_1052_, lean_object* v___x_1053_, lean_object* v_t_1054_, lean_object* v_init_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_){
_start:
{
lean_object* v_root_1065_; lean_object* v_tail_1066_; lean_object* v___x_1067_; 
v_root_1065_ = lean_ctor_get(v_t_1054_, 0);
v_tail_1066_ = lean_ctor_get(v_t_1054_, 1);
lean_inc(v___x_1053_);
v___x_1067_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__4(v_init_1055_, v_newEqName_1052_, v___x_1053_, v_root_1065_, v_init_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_, v___y_1062_, v___y_1063_);
if (lean_obj_tag(v___x_1067_) == 0)
{
lean_object* v_a_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1104_; 
v_a_1068_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1070_ = v___x_1067_;
v_isShared_1071_ = v_isSharedCheck_1104_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_a_1068_);
lean_dec(v___x_1067_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1104_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
if (lean_obj_tag(v_a_1068_) == 0)
{
lean_object* v_a_1072_; lean_object* v___x_1074_; 
lean_dec(v___x_1053_);
v_a_1072_ = lean_ctor_get(v_a_1068_, 0);
lean_inc(v_a_1072_);
lean_dec_ref_known(v_a_1068_, 1);
if (v_isShared_1071_ == 0)
{
lean_ctor_set(v___x_1070_, 0, v_a_1072_);
v___x_1074_ = v___x_1070_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v_a_1072_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
else
{
lean_object* v_a_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; size_t v_sz_1079_; size_t v___x_1080_; lean_object* v___x_1081_; 
lean_del_object(v___x_1070_);
v_a_1076_ = lean_ctor_get(v_a_1068_, 0);
lean_inc(v_a_1076_);
lean_dec_ref_known(v_a_1068_, 1);
v___x_1077_ = lean_box(0);
v___x_1078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1078_, 0, v___x_1077_);
lean_ctor_set(v___x_1078_, 1, v_a_1076_);
v_sz_1079_ = lean_array_size(v_tail_1066_);
v___x_1080_ = ((size_t)0ULL);
v___x_1081_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5(v_newEqName_1052_, v___x_1053_, v_tail_1066_, v_sz_1079_, v___x_1080_, v___x_1078_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_, v___y_1062_, v___y_1063_);
if (lean_obj_tag(v___x_1081_) == 0)
{
lean_object* v_a_1082_; lean_object* v___x_1084_; uint8_t v_isShared_1085_; uint8_t v_isSharedCheck_1095_; 
v_a_1082_ = lean_ctor_get(v___x_1081_, 0);
v_isSharedCheck_1095_ = !lean_is_exclusive(v___x_1081_);
if (v_isSharedCheck_1095_ == 0)
{
v___x_1084_ = v___x_1081_;
v_isShared_1085_ = v_isSharedCheck_1095_;
goto v_resetjp_1083_;
}
else
{
lean_inc(v_a_1082_);
lean_dec(v___x_1081_);
v___x_1084_ = lean_box(0);
v_isShared_1085_ = v_isSharedCheck_1095_;
goto v_resetjp_1083_;
}
v_resetjp_1083_:
{
lean_object* v_fst_1086_; 
v_fst_1086_ = lean_ctor_get(v_a_1082_, 0);
if (lean_obj_tag(v_fst_1086_) == 0)
{
lean_object* v_snd_1087_; lean_object* v___x_1089_; 
v_snd_1087_ = lean_ctor_get(v_a_1082_, 1);
lean_inc(v_snd_1087_);
lean_dec(v_a_1082_);
if (v_isShared_1085_ == 0)
{
lean_ctor_set(v___x_1084_, 0, v_snd_1087_);
v___x_1089_ = v___x_1084_;
goto v_reusejp_1088_;
}
else
{
lean_object* v_reuseFailAlloc_1090_; 
v_reuseFailAlloc_1090_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1090_, 0, v_snd_1087_);
v___x_1089_ = v_reuseFailAlloc_1090_;
goto v_reusejp_1088_;
}
v_reusejp_1088_:
{
return v___x_1089_;
}
}
else
{
lean_object* v_val_1091_; lean_object* v___x_1093_; 
lean_inc_ref(v_fst_1086_);
lean_dec(v_a_1082_);
v_val_1091_ = lean_ctor_get(v_fst_1086_, 0);
lean_inc(v_val_1091_);
lean_dec_ref_known(v_fst_1086_, 1);
if (v_isShared_1085_ == 0)
{
lean_ctor_set(v___x_1084_, 0, v_val_1091_);
v___x_1093_ = v___x_1084_;
goto v_reusejp_1092_;
}
else
{
lean_object* v_reuseFailAlloc_1094_; 
v_reuseFailAlloc_1094_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1094_, 0, v_val_1091_);
v___x_1093_ = v_reuseFailAlloc_1094_;
goto v_reusejp_1092_;
}
v_reusejp_1092_:
{
return v___x_1093_;
}
}
}
}
else
{
lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1103_; 
v_a_1096_ = lean_ctor_get(v___x_1081_, 0);
v_isSharedCheck_1103_ = !lean_is_exclusive(v___x_1081_);
if (v_isSharedCheck_1103_ == 0)
{
v___x_1098_ = v___x_1081_;
v_isShared_1099_ = v_isSharedCheck_1103_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v___x_1081_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1103_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1101_; 
if (v_isShared_1099_ == 0)
{
v___x_1101_ = v___x_1098_;
goto v_reusejp_1100_;
}
else
{
lean_object* v_reuseFailAlloc_1102_; 
v_reuseFailAlloc_1102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1102_, 0, v_a_1096_);
v___x_1101_ = v_reuseFailAlloc_1102_;
goto v_reusejp_1100_;
}
v_reusejp_1100_:
{
return v___x_1101_;
}
}
}
}
}
}
else
{
lean_object* v_a_1105_; lean_object* v___x_1107_; uint8_t v_isShared_1108_; uint8_t v_isSharedCheck_1112_; 
lean_dec(v___x_1053_);
v_a_1105_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1112_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1112_ == 0)
{
v___x_1107_ = v___x_1067_;
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
else
{
lean_inc(v_a_1105_);
lean_dec(v___x_1067_);
v___x_1107_ = lean_box(0);
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
v_resetjp_1106_:
{
lean_object* v___x_1110_; 
if (v_isShared_1108_ == 0)
{
v___x_1110_ = v___x_1107_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v_a_1105_);
v___x_1110_ = v_reuseFailAlloc_1111_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
return v___x_1110_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3___boxed(lean_object* v_newEqName_1113_, lean_object* v___x_1114_, lean_object* v_t_1115_, lean_object* v_init_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_){
_start:
{
lean_object* v_res_1126_; 
v_res_1126_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3(v_newEqName_1113_, v___x_1114_, v_t_1115_, v_init_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_);
lean_dec(v___y_1124_);
lean_dec_ref(v___y_1123_);
lean_dec(v___y_1122_);
lean_dec_ref(v___y_1121_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
lean_dec(v___y_1118_);
lean_dec_ref(v___y_1117_);
lean_dec_ref(v_t_1115_);
lean_dec(v_newEqName_1113_);
return v_res_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Lift_main_spec__2(lean_object* v_a_1127_, lean_object* v_a_1128_){
_start:
{
if (lean_obj_tag(v_a_1127_) == 0)
{
lean_object* v___x_1129_; 
v___x_1129_ = l_List_reverse___redArg(v_a_1128_);
return v___x_1129_;
}
else
{
lean_object* v_head_1130_; lean_object* v_tail_1131_; lean_object* v___x_1133_; uint8_t v_isShared_1134_; uint8_t v_isSharedCheck_1141_; 
v_head_1130_ = lean_ctor_get(v_a_1127_, 0);
v_tail_1131_ = lean_ctor_get(v_a_1127_, 1);
v_isSharedCheck_1141_ = !lean_is_exclusive(v_a_1127_);
if (v_isSharedCheck_1141_ == 0)
{
v___x_1133_ = v_a_1127_;
v_isShared_1134_ = v_isSharedCheck_1141_;
goto v_resetjp_1132_;
}
else
{
lean_inc(v_tail_1131_);
lean_inc(v_head_1130_);
lean_dec(v_a_1127_);
v___x_1133_ = lean_box(0);
v_isShared_1134_ = v_isSharedCheck_1141_;
goto v_resetjp_1132_;
}
v_resetjp_1132_:
{
lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1138_; 
v___x_1135_ = lean_box(0);
v___x_1136_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1136_, 0, v___x_1135_);
lean_ctor_set(v___x_1136_, 1, v_head_1130_);
if (v_isShared_1134_ == 0)
{
lean_ctor_set(v___x_1133_, 1, v_a_1128_);
lean_ctor_set(v___x_1133_, 0, v___x_1136_);
v___x_1138_ = v___x_1133_;
goto v_reusejp_1137_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v___x_1136_);
lean_ctor_set(v_reuseFailAlloc_1140_, 1, v_a_1128_);
v___x_1138_ = v_reuseFailAlloc_1140_;
goto v_reusejp_1137_;
}
v_reusejp_1137_:
{
v_a_1127_ = v_tail_1131_;
v_a_1128_ = v___x_1138_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__10(void){
_start:
{
lean_object* v___x_1168_; lean_object* v___x_1169_; 
v___x_1168_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__9));
v___x_1169_ = l_Lean_stringToMessageData(v___x_1168_);
return v___x_1169_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__17(void){
_start:
{
lean_object* v___x_1180_; lean_object* v___x_1181_; 
v___x_1180_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__16));
v___x_1181_ = l_Lean_stringToMessageData(v___x_1180_);
return v___x_1181_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__19(void){
_start:
{
lean_object* v___x_1183_; lean_object* v___x_1184_; 
v___x_1183_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__18));
v___x_1184_ = l_Lean_stringToMessageData(v___x_1183_);
return v___x_1184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0(lean_object* v_e_1185_, lean_object* v___x_1186_, uint8_t v___x_1187_, lean_object* v_hUsing_1188_, uint8_t v_keepUsing_1189_, uint8_t v___y_1190_, uint8_t v___y_1191_, lean_object* v___y_1192_, lean_object* v_newVarName_1193_, lean_object* v_t_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_){
_start:
{
lean_object* v___y_1205_; lean_object* v___y_1206_; lean_object* v___y_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v___y_1212_; lean_object* v___y_1213_; lean_object* v___y_1238_; lean_object* v___y_1239_; lean_object* v___y_1240_; lean_object* v___y_1241_; lean_object* v___y_1242_; lean_object* v___y_1243_; lean_object* v___y_1244_; lean_object* v___y_1245_; lean_object* v___y_1246_; lean_object* v___y_1270_; lean_object* v___y_1271_; lean_object* v___y_1272_; lean_object* v___y_1273_; lean_object* v___y_1274_; lean_object* v___y_1275_; lean_object* v___y_1276_; lean_object* v___y_1277_; lean_object* v___y_1278_; lean_object* v___y_1279_; lean_object* v___x_1289_; 
lean_inc(v___x_1186_);
v___x_1289_ = l_Lean_Elab_Tactic_elabTerm(v_e_1185_, v___x_1186_, v___x_1187_, v___y_1195_, v___y_1196_, v___y_1197_, v___y_1198_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_object* v_a_1290_; lean_object* v___x_1291_; 
v_a_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1290_);
lean_dec_ref_known(v___x_1289_, 1);
v___x_1291_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1196_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_);
if (lean_obj_tag(v___x_1291_) == 0)
{
lean_object* v_a_1292_; lean_object* v___y_1294_; lean_object* v___y_1295_; lean_object* v___y_1296_; lean_object* v_newEqName_1297_; lean_object* v___y_1298_; lean_object* v___y_1299_; lean_object* v___y_1300_; lean_object* v___y_1301_; lean_object* v___y_1302_; lean_object* v___y_1303_; lean_object* v___y_1304_; lean_object* v___y_1305_; lean_object* v___y_1381_; lean_object* v___y_1382_; lean_object* v___y_1383_; lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v_newVarName_1386_; lean_object* v___y_1387_; lean_object* v___y_1388_; lean_object* v___y_1389_; lean_object* v___y_1390_; lean_object* v___y_1391_; lean_object* v___y_1392_; lean_object* v___y_1393_; lean_object* v___y_1394_; lean_object* v___y_1447_; lean_object* v___y_1448_; lean_object* v___y_1449_; lean_object* v___y_1450_; lean_object* v_prf_1451_; lean_object* v___y_1452_; lean_object* v___y_1453_; lean_object* v___y_1454_; lean_object* v___y_1455_; lean_object* v___y_1456_; lean_object* v___y_1457_; lean_object* v___y_1458_; lean_object* v___y_1459_; lean_object* v___y_1474_; lean_object* v___y_1475_; lean_object* v___y_1476_; lean_object* v___y_1477_; lean_object* v___y_1478_; lean_object* v___y_1479_; lean_object* v___y_1480_; lean_object* v___y_1481_; lean_object* v___y_1553_; lean_object* v___y_1554_; lean_object* v___y_1555_; lean_object* v___y_1556_; lean_object* v___y_1557_; lean_object* v___y_1558_; lean_object* v___y_1559_; lean_object* v___y_1560_; lean_object* v___x_1566_; 
v_a_1292_ = lean_ctor_get(v___x_1291_, 0);
lean_inc_n(v_a_1292_, 2);
lean_dec_ref_known(v___x_1291_, 1);
v___x_1566_ = l_Lean_MVarId_getType(v_a_1292_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_);
if (lean_obj_tag(v___x_1566_) == 0)
{
lean_object* v_a_1567_; lean_object* v___x_1568_; lean_object* v_a_1569_; lean_object* v___x_1570_; 
v_a_1567_ = lean_ctor_get(v___x_1566_, 0);
lean_inc(v_a_1567_);
lean_dec_ref_known(v___x_1566_, 1);
v___x_1568_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg(v_a_1567_, v___y_1200_);
v_a_1569_ = lean_ctor_get(v___x_1568_, 0);
lean_inc(v_a_1569_);
lean_dec_ref(v___x_1568_);
lean_inc(v___y_1202_);
lean_inc_ref(v___y_1201_);
lean_inc(v___y_1200_);
lean_inc_ref(v___y_1199_);
v___x_1570_ = lean_infer_type(v_a_1569_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_);
if (lean_obj_tag(v___x_1570_) == 0)
{
lean_object* v_a_1571_; uint8_t v___x_1572_; 
v_a_1571_ = lean_ctor_get(v___x_1570_, 0);
lean_inc(v_a_1571_);
lean_dec_ref_known(v___x_1570_, 1);
v___x_1572_ = l_Lean_Expr_isProp(v_a_1571_);
lean_dec(v_a_1571_);
if (v___x_1572_ == 0)
{
lean_object* v___x_1573_; lean_object* v___x_1574_; 
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v_t_1194_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v___x_1573_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__19, &lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__19);
v___x_1574_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(v___x_1573_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
return v___x_1574_;
}
else
{
v___y_1553_ = v___y_1195_;
v___y_1554_ = v___y_1196_;
v___y_1555_ = v___y_1197_;
v___y_1556_ = v___y_1198_;
v___y_1557_ = v___y_1199_;
v___y_1558_ = v___y_1200_;
v___y_1559_ = v___y_1201_;
v___y_1560_ = v___y_1202_;
goto v___jp_1552_;
}
}
else
{
lean_object* v_a_1575_; lean_object* v___x_1577_; uint8_t v_isShared_1578_; uint8_t v_isSharedCheck_1582_; 
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v_t_1194_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1575_ = lean_ctor_get(v___x_1570_, 0);
v_isSharedCheck_1582_ = !lean_is_exclusive(v___x_1570_);
if (v_isSharedCheck_1582_ == 0)
{
v___x_1577_ = v___x_1570_;
v_isShared_1578_ = v_isSharedCheck_1582_;
goto v_resetjp_1576_;
}
else
{
lean_inc(v_a_1575_);
lean_dec(v___x_1570_);
v___x_1577_ = lean_box(0);
v_isShared_1578_ = v_isSharedCheck_1582_;
goto v_resetjp_1576_;
}
v_resetjp_1576_:
{
lean_object* v___x_1580_; 
if (v_isShared_1578_ == 0)
{
v___x_1580_ = v___x_1577_;
goto v_reusejp_1579_;
}
else
{
lean_object* v_reuseFailAlloc_1581_; 
v_reuseFailAlloc_1581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1581_, 0, v_a_1575_);
v___x_1580_ = v_reuseFailAlloc_1581_;
goto v_reusejp_1579_;
}
v_reusejp_1579_:
{
return v___x_1580_;
}
}
}
}
else
{
lean_object* v_a_1583_; lean_object* v___x_1585_; uint8_t v_isShared_1586_; uint8_t v_isSharedCheck_1590_; 
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v_t_1194_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1583_ = lean_ctor_get(v___x_1566_, 0);
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1566_);
if (v_isSharedCheck_1590_ == 0)
{
v___x_1585_ = v___x_1566_;
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
else
{
lean_inc(v_a_1583_);
lean_dec(v___x_1566_);
v___x_1585_ = lean_box(0);
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
v_resetjp_1584_:
{
lean_object* v___x_1588_; 
if (v_isShared_1586_ == 0)
{
v___x_1588_ = v___x_1585_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v_a_1583_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
v___jp_1293_:
{
lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
lean_inc_n(v_newEqName_1297_, 2);
v___x_1306_ = l_Lean_mkIdent(v_newEqName_1297_);
v___x_1307_ = lean_box(0);
v___x_1308_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1308_, 0, v___x_1307_);
lean_ctor_set(v___x_1308_, 1, v___y_1295_);
v___x_1309_ = lean_unsigned_to_nat(1u);
v___x_1310_ = lean_mk_empty_array_with_capacity(v___x_1309_);
v___x_1311_ = lean_array_push(v___x_1310_, v___x_1308_);
v___x_1312_ = lean_box(0);
v___x_1313_ = lean_box(0);
v___x_1314_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1314_, 0, v_newEqName_1297_);
lean_ctor_set(v___x_1314_, 1, v___x_1313_);
v___x_1315_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1315_, 0, v___y_1294_);
lean_ctor_set(v___x_1315_, 1, v___x_1314_);
v___x_1316_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Lift_main_spec__2(v___x_1315_, v___x_1313_);
v___x_1317_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1317_, 0, v___x_1312_);
lean_ctor_set(v___x_1317_, 1, v___x_1316_);
v___x_1318_ = l_Lean_Elab_Tactic_RCases_rcases(v___x_1311_, v___x_1317_, v_a_1292_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_);
if (lean_obj_tag(v___x_1318_) == 0)
{
lean_object* v_a_1319_; lean_object* v___x_1320_; 
v_a_1319_ = lean_ctor_get(v___x_1318_, 0);
lean_inc(v_a_1319_);
lean_dec_ref_known(v___x_1318_, 1);
v___x_1320_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_1319_, v___y_1299_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_);
if (lean_obj_tag(v___x_1320_) == 0)
{
lean_dec_ref_known(v___x_1320_, 1);
if (v___y_1190_ == 0)
{
lean_dec(v_newEqName_1297_);
v___y_1270_ = v___x_1306_;
v___y_1271_ = v___y_1296_;
v___y_1272_ = v___y_1298_;
v___y_1273_ = v___y_1299_;
v___y_1274_ = v___y_1300_;
v___y_1275_ = v___y_1301_;
v___y_1276_ = v___y_1302_;
v___y_1277_ = v___y_1303_;
v___y_1278_ = v___y_1304_;
v___y_1279_ = v___y_1305_;
goto v___jp_1269_;
}
else
{
lean_object* v_lctx_1321_; lean_object* v_decls_1322_; lean_object* v___x_1324_; uint8_t v_isShared_1325_; uint8_t v_isSharedCheck_1369_; 
v_lctx_1321_ = lean_ctor_get(v___y_1302_, 2);
lean_inc_ref(v_lctx_1321_);
v_decls_1322_ = lean_ctor_get(v_lctx_1321_, 1);
v_isSharedCheck_1369_ = !lean_is_exclusive(v_lctx_1321_);
if (v_isSharedCheck_1369_ == 0)
{
lean_object* v_unused_1370_; lean_object* v_unused_1371_; 
v_unused_1370_ = lean_ctor_get(v_lctx_1321_, 2);
lean_dec(v_unused_1370_);
v_unused_1371_ = lean_ctor_get(v_lctx_1321_, 0);
lean_dec(v_unused_1371_);
v___x_1324_ = v_lctx_1321_;
v_isShared_1325_ = v_isSharedCheck_1369_;
goto v_resetjp_1323_;
}
else
{
lean_inc(v_decls_1322_);
lean_dec(v_lctx_1321_);
v___x_1324_ = lean_box(0);
v_isShared_1325_ = v_isSharedCheck_1369_;
goto v_resetjp_1323_;
}
v_resetjp_1323_:
{
lean_object* v___x_1326_; lean_object* v___x_1327_; 
v___x_1326_ = lean_box(0);
lean_inc(v___x_1306_);
v___x_1327_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3(v_newEqName_1297_, v___x_1306_, v_decls_1322_, v___x_1326_, v___y_1298_, v___y_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_);
lean_dec_ref(v_decls_1322_);
lean_dec(v_newEqName_1297_);
if (lean_obj_tag(v___x_1327_) == 0)
{
lean_object* v_ref_1328_; lean_object* v_quotContext_1329_; lean_object* v_currMacroScope_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1351_; 
lean_dec_ref_known(v___x_1327_, 1);
v_ref_1328_ = lean_ctor_get(v___y_1304_, 5);
v_quotContext_1329_ = lean_ctor_get(v___y_1304_, 10);
v_currMacroScope_1330_ = lean_ctor_get(v___y_1304_, 11);
v___x_1331_ = l_Lean_SourceInfo_fromRef(v_ref_1328_, v___x_1187_);
v___x_1332_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__2));
v___x_1333_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__3));
lean_inc_n(v___x_1331_, 8);
v___x_1334_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1334_, 0, v___x_1331_);
lean_ctor_set(v___x_1334_, 1, v___x_1332_);
v___x_1335_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__5));
v___x_1336_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_1337_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__9));
v___x_1338_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__11));
v___x_1339_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__12));
v___x_1340_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1340_, 0, v___x_1331_);
lean_ctor_set(v___x_1340_, 1, v___x_1339_);
v___x_1341_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__14);
v___x_1342_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__15));
lean_inc(v_currMacroScope_1330_);
lean_inc(v_quotContext_1329_);
v___x_1343_ = l_Lean_addMacroScope(v_quotContext_1329_, v___x_1342_, v_currMacroScope_1330_);
v___x_1344_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1344_, 0, v___x_1331_);
lean_ctor_set(v___x_1344_, 1, v___x_1341_);
lean_ctor_set(v___x_1344_, 2, v___x_1343_);
lean_ctor_set(v___x_1344_, 3, v___x_1313_);
v___x_1345_ = l_Lean_Syntax_node2(v___x_1331_, v___x_1338_, v___x_1340_, v___x_1344_);
v___x_1346_ = l_Lean_Syntax_node1(v___x_1331_, v___x_1337_, v___x_1345_);
v___x_1347_ = l_Lean_Syntax_node1(v___x_1331_, v___x_1336_, v___x_1346_);
v___x_1348_ = l_Lean_Syntax_node1(v___x_1331_, v___x_1335_, v___x_1347_);
v___x_1349_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__16);
if (v_isShared_1325_ == 0)
{
lean_ctor_set_tag(v___x_1324_, 1);
lean_ctor_set(v___x_1324_, 2, v___x_1349_);
lean_ctor_set(v___x_1324_, 1, v___x_1336_);
lean_ctor_set(v___x_1324_, 0, v___x_1331_);
v___x_1351_ = v___x_1324_;
goto v_reusejp_1350_;
}
else
{
lean_object* v_reuseFailAlloc_1368_; 
v_reuseFailAlloc_1368_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1368_, 0, v___x_1331_);
lean_ctor_set(v_reuseFailAlloc_1368_, 1, v___x_1336_);
lean_ctor_set(v_reuseFailAlloc_1368_, 2, v___x_1349_);
v___x_1351_ = v_reuseFailAlloc_1368_;
goto v_reusejp_1350_;
}
v_reusejp_1350_:
{
lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1352_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__17));
lean_inc_n(v___x_1331_, 9);
v___x_1353_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1331_);
lean_ctor_set(v___x_1353_, 1, v___x_1352_);
v___x_1354_ = l_Lean_Syntax_node1(v___x_1331_, v___x_1336_, v___x_1353_);
v___x_1355_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__18));
v___x_1356_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1356_, 0, v___x_1331_);
lean_ctor_set(v___x_1356_, 1, v___x_1355_);
v___x_1357_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__20));
v___x_1358_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__21));
v___x_1359_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1359_, 0, v___x_1331_);
lean_ctor_set(v___x_1359_, 1, v___x_1358_);
v___x_1360_ = l_Lean_Syntax_node1(v___x_1331_, v___x_1336_, v___x_1359_);
lean_inc(v___x_1306_);
lean_inc_ref_n(v___x_1351_, 2);
v___x_1361_ = l_Lean_Syntax_node3(v___x_1331_, v___x_1357_, v___x_1351_, v___x_1360_, v___x_1306_);
v___x_1362_ = l_Lean_Syntax_node1(v___x_1331_, v___x_1336_, v___x_1361_);
v___x_1363_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__22));
v___x_1364_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1364_, 0, v___x_1331_);
lean_ctor_set(v___x_1364_, 1, v___x_1363_);
v___x_1365_ = l_Lean_Syntax_node3(v___x_1331_, v___x_1336_, v___x_1356_, v___x_1362_, v___x_1364_);
v___x_1366_ = l_Lean_Syntax_node6(v___x_1331_, v___x_1333_, v___x_1334_, v___x_1348_, v___x_1351_, v___x_1354_, v___x_1365_, v___x_1351_);
v___x_1367_ = l_Lean_Elab_Tactic_evalTactic(v___x_1366_, v___y_1298_, v___y_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_);
if (lean_obj_tag(v___x_1367_) == 0)
{
lean_dec_ref_known(v___x_1367_, 1);
v___y_1270_ = v___x_1306_;
v___y_1271_ = v___y_1296_;
v___y_1272_ = v___y_1298_;
v___y_1273_ = v___y_1299_;
v___y_1274_ = v___y_1300_;
v___y_1275_ = v___y_1301_;
v___y_1276_ = v___y_1302_;
v___y_1277_ = v___y_1303_;
v___y_1278_ = v___y_1304_;
v___y_1279_ = v___y_1305_;
goto v___jp_1269_;
}
else
{
lean_dec(v___x_1306_);
lean_dec(v___y_1305_);
lean_dec_ref(v___y_1304_);
lean_dec(v___y_1303_);
lean_dec_ref(v___y_1302_);
lean_dec_ref(v___y_1296_);
lean_dec(v_hUsing_1188_);
return v___x_1367_;
}
}
}
else
{
lean_del_object(v___x_1324_);
lean_dec(v___x_1306_);
lean_dec(v___y_1305_);
lean_dec_ref(v___y_1304_);
lean_dec(v___y_1303_);
lean_dec_ref(v___y_1302_);
lean_dec_ref(v___y_1296_);
lean_dec(v_hUsing_1188_);
return v___x_1327_;
}
}
}
}
else
{
lean_dec(v___x_1306_);
lean_dec(v___y_1305_);
lean_dec_ref(v___y_1304_);
lean_dec(v___y_1303_);
lean_dec_ref(v___y_1302_);
lean_dec(v_newEqName_1297_);
lean_dec_ref(v___y_1296_);
lean_dec(v_hUsing_1188_);
return v___x_1320_;
}
}
else
{
lean_object* v_a_1372_; lean_object* v___x_1374_; uint8_t v_isShared_1375_; uint8_t v_isSharedCheck_1379_; 
lean_dec(v___x_1306_);
lean_dec(v___y_1305_);
lean_dec_ref(v___y_1304_);
lean_dec(v___y_1303_);
lean_dec_ref(v___y_1302_);
lean_dec(v_newEqName_1297_);
lean_dec_ref(v___y_1296_);
lean_dec(v_hUsing_1188_);
v_a_1372_ = lean_ctor_get(v___x_1318_, 0);
v_isSharedCheck_1379_ = !lean_is_exclusive(v___x_1318_);
if (v_isSharedCheck_1379_ == 0)
{
v___x_1374_ = v___x_1318_;
v_isShared_1375_ = v_isSharedCheck_1379_;
goto v_resetjp_1373_;
}
else
{
lean_inc(v_a_1372_);
lean_dec(v___x_1318_);
v___x_1374_ = lean_box(0);
v_isShared_1375_ = v_isSharedCheck_1379_;
goto v_resetjp_1373_;
}
v_resetjp_1373_:
{
lean_object* v___x_1377_; 
if (v_isShared_1375_ == 0)
{
v___x_1377_ = v___x_1374_;
goto v_reusejp_1376_;
}
else
{
lean_object* v_reuseFailAlloc_1378_; 
v_reuseFailAlloc_1378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1378_, 0, v_a_1372_);
v___x_1377_ = v_reuseFailAlloc_1378_;
goto v_reusejp_1376_;
}
v_reusejp_1376_:
{
return v___x_1377_;
}
}
}
}
v___jp_1380_:
{
lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; 
v___x_1395_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__12));
v___x_1396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1396_, 0, v___y_1385_);
v___x_1397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1397_, 0, v___y_1382_);
v___x_1398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1398_, 0, v___y_1384_);
v___x_1399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1399_, 0, v___y_1381_);
v___x_1400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1400_, 0, v_a_1290_);
lean_inc_ref(v___y_1383_);
v___x_1401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1401_, 0, v___y_1383_);
v___x_1402_ = lean_unsigned_to_nat(7u);
v___x_1403_ = lean_mk_empty_array_with_capacity(v___x_1402_);
v___x_1404_ = lean_array_push(v___x_1403_, v___x_1186_);
v___x_1405_ = lean_array_push(v___x_1404_, v___x_1396_);
v___x_1406_ = lean_array_push(v___x_1405_, v___x_1397_);
v___x_1407_ = lean_array_push(v___x_1406_, v___x_1398_);
v___x_1408_ = lean_array_push(v___x_1407_, v___x_1399_);
v___x_1409_ = lean_array_push(v___x_1408_, v___x_1400_);
v___x_1410_ = lean_array_push(v___x_1409_, v___x_1401_);
v___x_1411_ = l_Lean_Meta_mkAppOptM(v___x_1395_, v___x_1410_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1411_) == 0)
{
lean_object* v_a_1412_; lean_object* v___x_1413_; lean_object* v_a_1414_; lean_object* v___x_1415_; 
v_a_1412_ = lean_ctor_get(v___x_1411_, 0);
lean_inc(v_a_1412_);
lean_dec_ref_known(v___x_1411_, 1);
v___x_1413_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Lift_main_spec__1___redArg(v_a_1412_, v___y_1392_);
v_a_1414_ = lean_ctor_get(v___x_1413_, 0);
lean_inc(v_a_1414_);
lean_dec_ref(v___x_1413_);
v___x_1415_ = lp_batteries_Lean_Expr_toSyntax(v_a_1414_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1415_) == 0)
{
if (v___y_1190_ == 0)
{
lean_object* v_a_1416_; 
v_a_1416_ = lean_ctor_get(v___x_1415_, 0);
lean_inc(v_a_1416_);
lean_dec_ref_known(v___x_1415_, 1);
v___y_1294_ = v_newVarName_1386_;
v___y_1295_ = v_a_1416_;
v___y_1296_ = v___y_1383_;
v_newEqName_1297_ = v___y_1192_;
v___y_1298_ = v___y_1387_;
v___y_1299_ = v___y_1388_;
v___y_1300_ = v___y_1389_;
v___y_1301_ = v___y_1390_;
v___y_1302_ = v___y_1391_;
v___y_1303_ = v___y_1392_;
v___y_1304_ = v___y_1393_;
v___y_1305_ = v___y_1394_;
goto v___jp_1293_;
}
else
{
if (v___y_1191_ == 0)
{
lean_object* v_a_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; 
lean_dec(v___y_1192_);
v_a_1417_ = lean_ctor_get(v___x_1415_, 0);
lean_inc(v_a_1417_);
lean_dec_ref_known(v___x_1415_, 1);
v___x_1418_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__15));
v___x_1419_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_1418_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1419_) == 0)
{
lean_object* v_a_1420_; 
v_a_1420_ = lean_ctor_get(v___x_1419_, 0);
lean_inc(v_a_1420_);
lean_dec_ref_known(v___x_1419_, 1);
v___y_1294_ = v_newVarName_1386_;
v___y_1295_ = v_a_1417_;
v___y_1296_ = v___y_1383_;
v_newEqName_1297_ = v_a_1420_;
v___y_1298_ = v___y_1387_;
v___y_1299_ = v___y_1388_;
v___y_1300_ = v___y_1389_;
v___y_1301_ = v___y_1390_;
v___y_1302_ = v___y_1391_;
v___y_1303_ = v___y_1392_;
v___y_1304_ = v___y_1393_;
v___y_1305_ = v___y_1394_;
goto v___jp_1293_;
}
else
{
lean_object* v_a_1421_; lean_object* v___x_1423_; uint8_t v_isShared_1424_; uint8_t v_isSharedCheck_1428_; 
lean_dec(v_a_1417_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v_newVarName_1386_);
lean_dec_ref(v___y_1383_);
lean_dec(v_a_1292_);
lean_dec(v_hUsing_1188_);
v_a_1421_ = lean_ctor_get(v___x_1419_, 0);
v_isSharedCheck_1428_ = !lean_is_exclusive(v___x_1419_);
if (v_isSharedCheck_1428_ == 0)
{
v___x_1423_ = v___x_1419_;
v_isShared_1424_ = v_isSharedCheck_1428_;
goto v_resetjp_1422_;
}
else
{
lean_inc(v_a_1421_);
lean_dec(v___x_1419_);
v___x_1423_ = lean_box(0);
v_isShared_1424_ = v_isSharedCheck_1428_;
goto v_resetjp_1422_;
}
v_resetjp_1422_:
{
lean_object* v___x_1426_; 
if (v_isShared_1424_ == 0)
{
v___x_1426_ = v___x_1423_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1427_; 
v_reuseFailAlloc_1427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1427_, 0, v_a_1421_);
v___x_1426_ = v_reuseFailAlloc_1427_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
return v___x_1426_;
}
}
}
}
else
{
lean_object* v_a_1429_; 
v_a_1429_ = lean_ctor_get(v___x_1415_, 0);
lean_inc(v_a_1429_);
lean_dec_ref_known(v___x_1415_, 1);
v___y_1294_ = v_newVarName_1386_;
v___y_1295_ = v_a_1429_;
v___y_1296_ = v___y_1383_;
v_newEqName_1297_ = v___y_1192_;
v___y_1298_ = v___y_1387_;
v___y_1299_ = v___y_1388_;
v___y_1300_ = v___y_1389_;
v___y_1301_ = v___y_1390_;
v___y_1302_ = v___y_1391_;
v___y_1303_ = v___y_1392_;
v___y_1304_ = v___y_1393_;
v___y_1305_ = v___y_1394_;
goto v___jp_1293_;
}
}
}
else
{
lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1437_; 
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v_newVarName_1386_);
lean_dec_ref(v___y_1383_);
lean_dec(v_a_1292_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
v_a_1430_ = lean_ctor_get(v___x_1415_, 0);
v_isSharedCheck_1437_ = !lean_is_exclusive(v___x_1415_);
if (v_isSharedCheck_1437_ == 0)
{
v___x_1432_ = v___x_1415_;
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1415_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1435_; 
if (v_isShared_1433_ == 0)
{
v___x_1435_ = v___x_1432_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1436_; 
v_reuseFailAlloc_1436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1436_, 0, v_a_1430_);
v___x_1435_ = v_reuseFailAlloc_1436_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
return v___x_1435_;
}
}
}
}
else
{
lean_object* v_a_1438_; lean_object* v___x_1440_; uint8_t v_isShared_1441_; uint8_t v_isSharedCheck_1445_; 
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v_newVarName_1386_);
lean_dec_ref(v___y_1383_);
lean_dec(v_a_1292_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
v_a_1438_ = lean_ctor_get(v___x_1411_, 0);
v_isSharedCheck_1445_ = !lean_is_exclusive(v___x_1411_);
if (v_isSharedCheck_1445_ == 0)
{
v___x_1440_ = v___x_1411_;
v_isShared_1441_ = v_isSharedCheck_1445_;
goto v_resetjp_1439_;
}
else
{
lean_inc(v_a_1438_);
lean_dec(v___x_1411_);
v___x_1440_ = lean_box(0);
v_isShared_1441_ = v_isSharedCheck_1445_;
goto v_resetjp_1439_;
}
v_resetjp_1439_:
{
lean_object* v___x_1443_; 
if (v_isShared_1441_ == 0)
{
v___x_1443_ = v___x_1440_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1444_; 
v_reuseFailAlloc_1444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1444_, 0, v_a_1438_);
v___x_1443_ = v_reuseFailAlloc_1444_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
return v___x_1443_;
}
}
}
}
v___jp_1446_:
{
if (lean_obj_tag(v_newVarName_1193_) == 0)
{
lean_object* v___x_1460_; lean_object* v___x_1461_; 
v___x_1460_ = l_Lean_Expr_fvarId_x21(v_a_1290_);
v___x_1461_ = l_Lean_FVarId_getUserName___redArg(v___x_1460_, v___y_1456_, v___y_1458_, v___y_1459_);
if (lean_obj_tag(v___x_1461_) == 0)
{
lean_object* v_a_1462_; 
v_a_1462_ = lean_ctor_get(v___x_1461_, 0);
lean_inc(v_a_1462_);
lean_dec_ref_known(v___x_1461_, 1);
v___y_1381_ = v___y_1447_;
v___y_1382_ = v___y_1448_;
v___y_1383_ = v_prf_1451_;
v___y_1384_ = v___y_1449_;
v___y_1385_ = v___y_1450_;
v_newVarName_1386_ = v_a_1462_;
v___y_1387_ = v___y_1452_;
v___y_1388_ = v___y_1453_;
v___y_1389_ = v___y_1454_;
v___y_1390_ = v___y_1455_;
v___y_1391_ = v___y_1456_;
v___y_1392_ = v___y_1457_;
v___y_1393_ = v___y_1458_;
v___y_1394_ = v___y_1459_;
goto v___jp_1380_;
}
else
{
lean_object* v_a_1463_; lean_object* v___x_1465_; uint8_t v_isShared_1466_; uint8_t v_isSharedCheck_1470_; 
lean_dec(v___y_1459_);
lean_dec_ref(v___y_1458_);
lean_dec(v___y_1457_);
lean_dec_ref(v___y_1456_);
lean_dec_ref(v_prf_1451_);
lean_dec_ref(v___y_1450_);
lean_dec_ref(v___y_1449_);
lean_dec_ref(v___y_1448_);
lean_dec_ref(v___y_1447_);
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1463_ = lean_ctor_get(v___x_1461_, 0);
v_isSharedCheck_1470_ = !lean_is_exclusive(v___x_1461_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1465_ = v___x_1461_;
v_isShared_1466_ = v_isSharedCheck_1470_;
goto v_resetjp_1464_;
}
else
{
lean_inc(v_a_1463_);
lean_dec(v___x_1461_);
v___x_1465_ = lean_box(0);
v_isShared_1466_ = v_isSharedCheck_1470_;
goto v_resetjp_1464_;
}
v_resetjp_1464_:
{
lean_object* v___x_1468_; 
if (v_isShared_1466_ == 0)
{
v___x_1468_ = v___x_1465_;
goto v_reusejp_1467_;
}
else
{
lean_object* v_reuseFailAlloc_1469_; 
v_reuseFailAlloc_1469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1469_, 0, v_a_1463_);
v___x_1468_ = v_reuseFailAlloc_1469_;
goto v_reusejp_1467_;
}
v_reusejp_1467_:
{
return v___x_1468_;
}
}
}
}
else
{
lean_object* v_val_1471_; lean_object* v___x_1472_; 
v_val_1471_ = lean_ctor_get(v_newVarName_1193_, 0);
v___x_1472_ = l_Lean_TSyntax_getId(v_val_1471_);
v___y_1381_ = v___y_1447_;
v___y_1382_ = v___y_1448_;
v___y_1383_ = v_prf_1451_;
v___y_1384_ = v___y_1449_;
v___y_1385_ = v___y_1450_;
v_newVarName_1386_ = v___x_1472_;
v___y_1387_ = v___y_1452_;
v___y_1388_ = v___y_1453_;
v___y_1389_ = v___y_1454_;
v___y_1390_ = v___y_1455_;
v___y_1391_ = v___y_1456_;
v___y_1392_ = v___y_1457_;
v___y_1393_ = v___y_1458_;
v___y_1394_ = v___y_1459_;
goto v___jp_1380_;
}
}
v___jp_1473_:
{
lean_object* v___x_1482_; 
v___x_1482_ = l_Lean_Elab_Term_elabType(v_t_1194_, v___y_1476_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
if (lean_obj_tag(v___x_1482_) == 0)
{
lean_object* v_a_1483_; lean_object* v___x_1484_; 
v_a_1483_ = lean_ctor_get(v___x_1482_, 0);
lean_inc(v_a_1483_);
lean_dec_ref_known(v___x_1482_, 1);
lean_inc(v___y_1481_);
lean_inc_ref(v___y_1480_);
lean_inc(v___y_1479_);
lean_inc_ref(v___y_1478_);
lean_inc(v_a_1290_);
v___x_1484_ = lean_infer_type(v_a_1290_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
if (lean_obj_tag(v___x_1484_) == 0)
{
lean_object* v_a_1485_; lean_object* v___x_1486_; 
v_a_1485_ = lean_ctor_get(v___x_1484_, 0);
lean_inc(v_a_1485_);
lean_dec_ref_known(v___x_1484_, 1);
lean_inc(v_a_1483_);
v___x_1486_ = lp_mathlib_Mathlib_Tactic_Lift_getInst(v_a_1485_, v_a_1483_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
if (lean_obj_tag(v___x_1486_) == 0)
{
lean_object* v_a_1487_; lean_object* v_snd_1488_; 
v_a_1487_ = lean_ctor_get(v___x_1486_, 0);
lean_inc(v_a_1487_);
lean_dec_ref_known(v___x_1486_, 1);
v_snd_1488_ = lean_ctor_get(v_a_1487_, 1);
lean_inc(v_snd_1488_);
if (lean_obj_tag(v_hUsing_1188_) == 0)
{
lean_object* v_fst_1489_; lean_object* v_fst_1490_; lean_object* v_snd_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; uint8_t v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; 
v_fst_1489_ = lean_ctor_get(v_a_1487_, 0);
lean_inc_n(v_fst_1489_, 2);
lean_dec(v_a_1487_);
v_fst_1490_ = lean_ctor_get(v_snd_1488_, 0);
lean_inc(v_fst_1490_);
v_snd_1491_ = lean_ctor_get(v_snd_1488_, 1);
lean_inc(v_snd_1491_);
lean_dec(v_snd_1488_);
v___x_1492_ = lean_unsigned_to_nat(1u);
v___x_1493_ = lean_mk_empty_array_with_capacity(v___x_1492_);
lean_inc(v_a_1290_);
v___x_1494_ = lean_array_push(v___x_1493_, v_a_1290_);
v___x_1495_ = l_Lean_Expr_betaRev(v_fst_1489_, v___x_1494_, v___x_1187_, v___x_1187_);
lean_dec_ref(v___x_1494_);
v___x_1496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1496_, 0, v___x_1495_);
v___x_1497_ = 0;
v___x_1498_ = lean_box(0);
v___x_1499_ = l_Lean_Meta_mkFreshExprMVar(v___x_1496_, v___x_1497_, v___x_1498_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
if (lean_obj_tag(v___x_1499_) == 0)
{
lean_object* v_a_1500_; 
v_a_1500_ = lean_ctor_get(v___x_1499_, 0);
lean_inc(v_a_1500_);
lean_dec_ref_known(v___x_1499_, 1);
v___y_1447_ = v_snd_1491_;
v___y_1448_ = v_fst_1490_;
v___y_1449_ = v_fst_1489_;
v___y_1450_ = v_a_1483_;
v_prf_1451_ = v_a_1500_;
v___y_1452_ = v___y_1474_;
v___y_1453_ = v___y_1475_;
v___y_1454_ = v___y_1476_;
v___y_1455_ = v___y_1477_;
v___y_1456_ = v___y_1478_;
v___y_1457_ = v___y_1479_;
v___y_1458_ = v___y_1480_;
v___y_1459_ = v___y_1481_;
goto v___jp_1446_;
}
else
{
lean_object* v_a_1501_; lean_object* v___x_1503_; uint8_t v_isShared_1504_; uint8_t v_isSharedCheck_1508_; 
lean_dec(v_snd_1491_);
lean_dec(v_fst_1490_);
lean_dec(v_fst_1489_);
lean_dec(v_a_1483_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1192_);
lean_dec(v___x_1186_);
v_a_1501_ = lean_ctor_get(v___x_1499_, 0);
v_isSharedCheck_1508_ = !lean_is_exclusive(v___x_1499_);
if (v_isSharedCheck_1508_ == 0)
{
v___x_1503_ = v___x_1499_;
v_isShared_1504_ = v_isSharedCheck_1508_;
goto v_resetjp_1502_;
}
else
{
lean_inc(v_a_1501_);
lean_dec(v___x_1499_);
v___x_1503_ = lean_box(0);
v_isShared_1504_ = v_isSharedCheck_1508_;
goto v_resetjp_1502_;
}
v_resetjp_1502_:
{
lean_object* v___x_1506_; 
if (v_isShared_1504_ == 0)
{
v___x_1506_ = v___x_1503_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1507_; 
v_reuseFailAlloc_1507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1507_, 0, v_a_1501_);
v___x_1506_ = v_reuseFailAlloc_1507_;
goto v_reusejp_1505_;
}
v_reusejp_1505_:
{
return v___x_1506_;
}
}
}
}
else
{
lean_object* v_fst_1509_; lean_object* v_fst_1510_; lean_object* v_snd_1511_; lean_object* v_val_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; 
v_fst_1509_ = lean_ctor_get(v_a_1487_, 0);
lean_inc_n(v_fst_1509_, 2);
lean_dec(v_a_1487_);
v_fst_1510_ = lean_ctor_get(v_snd_1488_, 0);
lean_inc(v_fst_1510_);
v_snd_1511_ = lean_ctor_get(v_snd_1488_, 1);
lean_inc(v_snd_1511_);
lean_dec(v_snd_1488_);
v_val_1512_ = lean_ctor_get(v_hUsing_1188_, 0);
v___x_1513_ = lean_unsigned_to_nat(1u);
v___x_1514_ = lean_mk_empty_array_with_capacity(v___x_1513_);
lean_inc(v_a_1290_);
v___x_1515_ = lean_array_push(v___x_1514_, v_a_1290_);
v___x_1516_ = l_Lean_Expr_betaRev(v_fst_1509_, v___x_1515_, v___x_1187_, v___x_1187_);
lean_dec_ref(v___x_1515_);
v___x_1517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1517_, 0, v___x_1516_);
lean_inc(v_val_1512_);
v___x_1518_ = l_Lean_Elab_Tactic_elabTermEnsuringType(v_val_1512_, v___x_1517_, v___x_1187_, v___y_1474_, v___y_1475_, v___y_1476_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
if (lean_obj_tag(v___x_1518_) == 0)
{
lean_object* v_a_1519_; 
v_a_1519_ = lean_ctor_get(v___x_1518_, 0);
lean_inc(v_a_1519_);
lean_dec_ref_known(v___x_1518_, 1);
v___y_1447_ = v_snd_1511_;
v___y_1448_ = v_fst_1510_;
v___y_1449_ = v_fst_1509_;
v___y_1450_ = v_a_1483_;
v_prf_1451_ = v_a_1519_;
v___y_1452_ = v___y_1474_;
v___y_1453_ = v___y_1475_;
v___y_1454_ = v___y_1476_;
v___y_1455_ = v___y_1477_;
v___y_1456_ = v___y_1478_;
v___y_1457_ = v___y_1479_;
v___y_1458_ = v___y_1480_;
v___y_1459_ = v___y_1481_;
goto v___jp_1446_;
}
else
{
lean_object* v_a_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1527_; 
lean_dec(v_snd_1511_);
lean_dec(v_fst_1510_);
lean_dec_ref_known(v_hUsing_1188_, 1);
lean_dec(v_fst_1509_);
lean_dec(v_a_1483_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1192_);
lean_dec(v___x_1186_);
v_a_1520_ = lean_ctor_get(v___x_1518_, 0);
v_isSharedCheck_1527_ = !lean_is_exclusive(v___x_1518_);
if (v_isSharedCheck_1527_ == 0)
{
v___x_1522_ = v___x_1518_;
v_isShared_1523_ = v_isSharedCheck_1527_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_a_1520_);
lean_dec(v___x_1518_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1527_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v___x_1525_; 
if (v_isShared_1523_ == 0)
{
v___x_1525_ = v___x_1522_;
goto v_reusejp_1524_;
}
else
{
lean_object* v_reuseFailAlloc_1526_; 
v_reuseFailAlloc_1526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1526_, 0, v_a_1520_);
v___x_1525_ = v_reuseFailAlloc_1526_;
goto v_reusejp_1524_;
}
v_reusejp_1524_:
{
return v___x_1525_;
}
}
}
}
}
else
{
lean_object* v_a_1528_; lean_object* v___x_1530_; uint8_t v_isShared_1531_; uint8_t v_isSharedCheck_1535_; 
lean_dec(v_a_1483_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1528_ = lean_ctor_get(v___x_1486_, 0);
v_isSharedCheck_1535_ = !lean_is_exclusive(v___x_1486_);
if (v_isSharedCheck_1535_ == 0)
{
v___x_1530_ = v___x_1486_;
v_isShared_1531_ = v_isSharedCheck_1535_;
goto v_resetjp_1529_;
}
else
{
lean_inc(v_a_1528_);
lean_dec(v___x_1486_);
v___x_1530_ = lean_box(0);
v_isShared_1531_ = v_isSharedCheck_1535_;
goto v_resetjp_1529_;
}
v_resetjp_1529_:
{
lean_object* v___x_1533_; 
if (v_isShared_1531_ == 0)
{
v___x_1533_ = v___x_1530_;
goto v_reusejp_1532_;
}
else
{
lean_object* v_reuseFailAlloc_1534_; 
v_reuseFailAlloc_1534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1534_, 0, v_a_1528_);
v___x_1533_ = v_reuseFailAlloc_1534_;
goto v_reusejp_1532_;
}
v_reusejp_1532_:
{
return v___x_1533_;
}
}
}
}
else
{
lean_object* v_a_1536_; lean_object* v___x_1538_; uint8_t v_isShared_1539_; uint8_t v_isSharedCheck_1543_; 
lean_dec(v_a_1483_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1536_ = lean_ctor_get(v___x_1484_, 0);
v_isSharedCheck_1543_ = !lean_is_exclusive(v___x_1484_);
if (v_isSharedCheck_1543_ == 0)
{
v___x_1538_ = v___x_1484_;
v_isShared_1539_ = v_isSharedCheck_1543_;
goto v_resetjp_1537_;
}
else
{
lean_inc(v_a_1536_);
lean_dec(v___x_1484_);
v___x_1538_ = lean_box(0);
v_isShared_1539_ = v_isSharedCheck_1543_;
goto v_resetjp_1537_;
}
v_resetjp_1537_:
{
lean_object* v___x_1541_; 
if (v_isShared_1539_ == 0)
{
v___x_1541_ = v___x_1538_;
goto v_reusejp_1540_;
}
else
{
lean_object* v_reuseFailAlloc_1542_; 
v_reuseFailAlloc_1542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1542_, 0, v_a_1536_);
v___x_1541_ = v_reuseFailAlloc_1542_;
goto v_reusejp_1540_;
}
v_reusejp_1540_:
{
return v___x_1541_;
}
}
}
}
else
{
lean_object* v_a_1544_; lean_object* v___x_1546_; uint8_t v_isShared_1547_; uint8_t v_isSharedCheck_1551_; 
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1544_ = lean_ctor_get(v___x_1482_, 0);
v_isSharedCheck_1551_ = !lean_is_exclusive(v___x_1482_);
if (v_isSharedCheck_1551_ == 0)
{
v___x_1546_ = v___x_1482_;
v_isShared_1547_ = v_isSharedCheck_1551_;
goto v_resetjp_1545_;
}
else
{
lean_inc(v_a_1544_);
lean_dec(v___x_1482_);
v___x_1546_ = lean_box(0);
v_isShared_1547_ = v_isSharedCheck_1551_;
goto v_resetjp_1545_;
}
v_resetjp_1545_:
{
lean_object* v___x_1549_; 
if (v_isShared_1547_ == 0)
{
v___x_1549_ = v___x_1546_;
goto v_reusejp_1548_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v_a_1544_);
v___x_1549_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1548_;
}
v_reusejp_1548_:
{
return v___x_1549_;
}
}
}
}
v___jp_1552_:
{
lean_object* v___x_1561_; uint8_t v___x_1562_; 
v___x_1561_ = lean_box(0);
v___x_1562_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_Tactic_Lift_main_spec__5(v_newVarName_1193_, v___x_1561_);
if (v___x_1562_ == 0)
{
v___y_1474_ = v___y_1553_;
v___y_1475_ = v___y_1554_;
v___y_1476_ = v___y_1555_;
v___y_1477_ = v___y_1556_;
v___y_1478_ = v___y_1557_;
v___y_1479_ = v___y_1558_;
v___y_1480_ = v___y_1559_;
v___y_1481_ = v___y_1560_;
goto v___jp_1473_;
}
else
{
uint8_t v___x_1563_; 
v___x_1563_ = l_Lean_Expr_isFVar(v_a_1290_);
if (v___x_1563_ == 0)
{
lean_object* v___x_1564_; lean_object* v___x_1565_; 
lean_dec(v_a_1292_);
lean_dec(v_a_1290_);
lean_dec(v_t_1194_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v___x_1564_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__17, &lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__17);
v___x_1565_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(v___x_1564_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_);
lean_dec(v___y_1560_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1558_);
lean_dec_ref(v___y_1557_);
return v___x_1565_;
}
else
{
v___y_1474_ = v___y_1553_;
v___y_1475_ = v___y_1554_;
v___y_1476_ = v___y_1555_;
v___y_1477_ = v___y_1556_;
v___y_1478_ = v___y_1557_;
v___y_1479_ = v___y_1558_;
v___y_1480_ = v___y_1559_;
v___y_1481_ = v___y_1560_;
goto v___jp_1473_;
}
}
}
}
else
{
lean_object* v_a_1591_; lean_object* v___x_1593_; uint8_t v_isShared_1594_; uint8_t v_isSharedCheck_1598_; 
lean_dec(v_a_1290_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v_t_1194_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1591_ = lean_ctor_get(v___x_1291_, 0);
v_isSharedCheck_1598_ = !lean_is_exclusive(v___x_1291_);
if (v_isSharedCheck_1598_ == 0)
{
v___x_1593_ = v___x_1291_;
v_isShared_1594_ = v_isSharedCheck_1598_;
goto v_resetjp_1592_;
}
else
{
lean_inc(v_a_1591_);
lean_dec(v___x_1291_);
v___x_1593_ = lean_box(0);
v_isShared_1594_ = v_isSharedCheck_1598_;
goto v_resetjp_1592_;
}
v_resetjp_1592_:
{
lean_object* v___x_1596_; 
if (v_isShared_1594_ == 0)
{
v___x_1596_ = v___x_1593_;
goto v_reusejp_1595_;
}
else
{
lean_object* v_reuseFailAlloc_1597_; 
v_reuseFailAlloc_1597_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1597_, 0, v_a_1591_);
v___x_1596_ = v_reuseFailAlloc_1597_;
goto v_reusejp_1595_;
}
v_reusejp_1595_:
{
return v___x_1596_;
}
}
}
}
else
{
lean_object* v_a_1599_; lean_object* v___x_1601_; uint8_t v_isShared_1602_; uint8_t v_isSharedCheck_1606_; 
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v_t_1194_);
lean_dec(v___y_1192_);
lean_dec(v_hUsing_1188_);
lean_dec(v___x_1186_);
v_a_1599_ = lean_ctor_get(v___x_1289_, 0);
v_isSharedCheck_1606_ = !lean_is_exclusive(v___x_1289_);
if (v_isSharedCheck_1606_ == 0)
{
v___x_1601_ = v___x_1289_;
v_isShared_1602_ = v_isSharedCheck_1606_;
goto v_resetjp_1600_;
}
else
{
lean_inc(v_a_1599_);
lean_dec(v___x_1289_);
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
v___jp_1204_:
{
if (lean_obj_tag(v_hUsing_1188_) == 0)
{
lean_object* v___x_1214_; 
v___x_1214_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_1207_);
if (lean_obj_tag(v___x_1214_) == 0)
{
lean_object* v_a_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; 
v_a_1215_ = lean_ctor_get(v___x_1214_, 0);
lean_inc(v_a_1215_);
lean_dec_ref_known(v___x_1214_, 1);
v___x_1216_ = l_Lean_Expr_mvarId_x21(v___y_1205_);
lean_dec_ref(v___y_1205_);
v___x_1217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1217_, 0, v___x_1216_);
lean_ctor_set(v___x_1217_, 1, v_a_1215_);
v___x_1218_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_setGoals___boxed), 10, 1);
lean_closure_set(v___x_1218_, 0, v___x_1217_);
v___x_1219_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_1218_, v___y_1206_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_);
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
return v___x_1219_;
}
else
{
lean_object* v_a_1220_; lean_object* v___x_1222_; uint8_t v_isShared_1223_; uint8_t v_isSharedCheck_1227_; 
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec_ref(v___y_1205_);
v_a_1220_ = lean_ctor_get(v___x_1214_, 0);
v_isSharedCheck_1227_ = !lean_is_exclusive(v___x_1214_);
if (v_isSharedCheck_1227_ == 0)
{
v___x_1222_ = v___x_1214_;
v_isShared_1223_ = v_isSharedCheck_1227_;
goto v_resetjp_1221_;
}
else
{
lean_inc(v_a_1220_);
lean_dec(v___x_1214_);
v___x_1222_ = lean_box(0);
v_isShared_1223_ = v_isSharedCheck_1227_;
goto v_resetjp_1221_;
}
v_resetjp_1221_:
{
lean_object* v___x_1225_; 
if (v_isShared_1223_ == 0)
{
v___x_1225_ = v___x_1222_;
goto v_reusejp_1224_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v_a_1220_);
v___x_1225_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1224_;
}
v_reusejp_1224_:
{
return v___x_1225_;
}
}
}
}
else
{
lean_object* v___x_1229_; uint8_t v_isShared_1230_; uint8_t v_isSharedCheck_1235_; 
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec_ref(v___y_1205_);
v_isSharedCheck_1235_ = !lean_is_exclusive(v_hUsing_1188_);
if (v_isSharedCheck_1235_ == 0)
{
lean_object* v_unused_1236_; 
v_unused_1236_ = lean_ctor_get(v_hUsing_1188_, 0);
lean_dec(v_unused_1236_);
v___x_1229_ = v_hUsing_1188_;
v_isShared_1230_ = v_isSharedCheck_1235_;
goto v_resetjp_1228_;
}
else
{
lean_dec(v_hUsing_1188_);
v___x_1229_ = lean_box(0);
v_isShared_1230_ = v_isSharedCheck_1235_;
goto v_resetjp_1228_;
}
v_resetjp_1228_:
{
lean_object* v___x_1231_; lean_object* v___x_1233_; 
v___x_1231_ = lean_box(0);
if (v_isShared_1230_ == 0)
{
lean_ctor_set_tag(v___x_1229_, 0);
lean_ctor_set(v___x_1229_, 0, v___x_1231_);
v___x_1233_ = v___x_1229_;
goto v_reusejp_1232_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v___x_1231_);
v___x_1233_ = v_reuseFailAlloc_1234_;
goto v_reusejp_1232_;
}
v_reusejp_1232_:
{
return v___x_1233_;
}
}
}
}
v___jp_1237_:
{
uint8_t v___x_1247_; 
v___x_1247_ = l_Lean_Expr_isFVar(v___y_1238_);
if (v___x_1247_ == 0)
{
v___y_1205_ = v___y_1238_;
v___y_1206_ = v___y_1239_;
v___y_1207_ = v___y_1240_;
v___y_1208_ = v___y_1241_;
v___y_1209_ = v___y_1242_;
v___y_1210_ = v___y_1243_;
v___y_1211_ = v___y_1244_;
v___y_1212_ = v___y_1245_;
v___y_1213_ = v___y_1246_;
goto v___jp_1204_;
}
else
{
if (v_keepUsing_1189_ == 0)
{
if (v___x_1247_ == 0)
{
v___y_1205_ = v___y_1238_;
v___y_1206_ = v___y_1239_;
v___y_1207_ = v___y_1240_;
v___y_1208_ = v___y_1241_;
v___y_1209_ = v___y_1242_;
v___y_1210_ = v___y_1243_;
v___y_1211_ = v___y_1244_;
v___y_1212_ = v___y_1245_;
v___y_1213_ = v___y_1246_;
goto v___jp_1204_;
}
else
{
if (lean_obj_tag(v_hUsing_1188_) == 1)
{
lean_object* v_val_1248_; lean_object* v_ref_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; 
v_val_1248_ = lean_ctor_get(v_hUsing_1188_, 0);
v_ref_1249_ = lean_ctor_get(v___y_1245_, 5);
v___x_1250_ = l_Lean_SourceInfo_fromRef(v_ref_1249_, v_keepUsing_1189_);
v___x_1251_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__1));
v___x_1252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__2));
lean_inc_n(v___x_1250_, 7);
v___x_1253_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1253_, 0, v___x_1250_);
lean_ctor_set(v___x_1253_, 1, v___x_1252_);
v___x_1254_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__4));
v___x_1255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__6));
v___x_1256_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_1257_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__7));
v___x_1258_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8));
v___x_1259_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1259_, 0, v___x_1250_);
lean_ctor_set(v___x_1259_, 1, v___x_1257_);
lean_inc(v_val_1248_);
v___x_1260_ = l_Lean_Syntax_node1(v___x_1250_, v___x_1256_, v_val_1248_);
v___x_1261_ = l_Lean_Syntax_node2(v___x_1250_, v___x_1258_, v___x_1259_, v___x_1260_);
v___x_1262_ = l_Lean_Syntax_node1(v___x_1250_, v___x_1256_, v___x_1261_);
v___x_1263_ = l_Lean_Syntax_node1(v___x_1250_, v___x_1255_, v___x_1262_);
v___x_1264_ = l_Lean_Syntax_node1(v___x_1250_, v___x_1254_, v___x_1263_);
v___x_1265_ = l_Lean_Syntax_node2(v___x_1250_, v___x_1251_, v___x_1253_, v___x_1264_);
v___x_1266_ = l_Lean_Elab_Tactic_evalTactic(v___x_1265_, v___y_1239_, v___y_1240_, v___y_1241_, v___y_1242_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_);
if (lean_obj_tag(v___x_1266_) == 0)
{
lean_dec_ref_known(v___x_1266_, 1);
v___y_1205_ = v___y_1238_;
v___y_1206_ = v___y_1239_;
v___y_1207_ = v___y_1240_;
v___y_1208_ = v___y_1241_;
v___y_1209_ = v___y_1242_;
v___y_1210_ = v___y_1243_;
v___y_1211_ = v___y_1244_;
v___y_1212_ = v___y_1245_;
v___y_1213_ = v___y_1246_;
goto v___jp_1204_;
}
else
{
lean_dec_ref_known(v_hUsing_1188_, 1);
lean_dec(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec_ref(v___y_1238_);
return v___x_1266_;
}
}
else
{
lean_object* v___x_1267_; lean_object* v___x_1268_; 
lean_dec_ref(v___y_1238_);
lean_dec(v_hUsing_1188_);
v___x_1267_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__10, &lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__10);
v___x_1268_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(v___x_1267_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_);
lean_dec(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
return v___x_1268_;
}
}
}
else
{
v___y_1205_ = v___y_1238_;
v___y_1206_ = v___y_1239_;
v___y_1207_ = v___y_1240_;
v___y_1208_ = v___y_1241_;
v___y_1209_ = v___y_1242_;
v___y_1210_ = v___y_1243_;
v___y_1211_ = v___y_1244_;
v___y_1212_ = v___y_1245_;
v___y_1213_ = v___y_1246_;
goto v___jp_1204_;
}
}
}
v___jp_1269_:
{
if (v___y_1190_ == 0)
{
lean_dec(v___y_1270_);
v___y_1238_ = v___y_1271_;
v___y_1239_ = v___y_1272_;
v___y_1240_ = v___y_1273_;
v___y_1241_ = v___y_1274_;
v___y_1242_ = v___y_1275_;
v___y_1243_ = v___y_1276_;
v___y_1244_ = v___y_1277_;
v___y_1245_ = v___y_1278_;
v___y_1246_ = v___y_1279_;
goto v___jp_1237_;
}
else
{
if (v___y_1191_ == 0)
{
lean_object* v_ref_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
v_ref_1280_ = lean_ctor_get(v___y_1278_, 5);
v___x_1281_ = l_Lean_SourceInfo_fromRef(v_ref_1280_, v___y_1191_);
v___x_1282_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__7));
v___x_1283_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___closed__8));
lean_inc_n(v___x_1281_, 2);
v___x_1284_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1284_, 0, v___x_1281_);
lean_ctor_set(v___x_1284_, 1, v___x_1282_);
v___x_1285_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Lift_main_spec__3_spec__5_spec__10___closed__7));
v___x_1286_ = l_Lean_Syntax_node1(v___x_1281_, v___x_1285_, v___y_1270_);
v___x_1287_ = l_Lean_Syntax_node2(v___x_1281_, v___x_1283_, v___x_1284_, v___x_1286_);
v___x_1288_ = l_Lean_Elab_Tactic_evalTactic(v___x_1287_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_, v___y_1279_);
if (lean_obj_tag(v___x_1288_) == 0)
{
lean_dec_ref_known(v___x_1288_, 1);
v___y_1238_ = v___y_1271_;
v___y_1239_ = v___y_1272_;
v___y_1240_ = v___y_1273_;
v___y_1241_ = v___y_1274_;
v___y_1242_ = v___y_1275_;
v___y_1243_ = v___y_1276_;
v___y_1244_ = v___y_1277_;
v___y_1245_ = v___y_1278_;
v___y_1246_ = v___y_1279_;
goto v___jp_1237_;
}
else
{
lean_dec(v___y_1279_);
lean_dec_ref(v___y_1278_);
lean_dec(v___y_1277_);
lean_dec_ref(v___y_1276_);
lean_dec_ref(v___y_1271_);
lean_dec(v_hUsing_1188_);
return v___x_1288_;
}
}
else
{
lean_dec(v___y_1270_);
v___y_1238_ = v___y_1271_;
v___y_1239_ = v___y_1272_;
v___y_1240_ = v___y_1273_;
v___y_1241_ = v___y_1274_;
v___y_1242_ = v___y_1275_;
v___y_1243_ = v___y_1276_;
v___y_1244_ = v___y_1277_;
v___y_1245_ = v___y_1278_;
v___y_1246_ = v___y_1279_;
goto v___jp_1237_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___boxed(lean_object** _args){
lean_object* v_e_1607_ = _args[0];
lean_object* v___x_1608_ = _args[1];
lean_object* v___x_1609_ = _args[2];
lean_object* v_hUsing_1610_ = _args[3];
lean_object* v_keepUsing_1611_ = _args[4];
lean_object* v___y_1612_ = _args[5];
lean_object* v___y_1613_ = _args[6];
lean_object* v___y_1614_ = _args[7];
lean_object* v_newVarName_1615_ = _args[8];
lean_object* v_t_1616_ = _args[9];
lean_object* v___y_1617_ = _args[10];
lean_object* v___y_1618_ = _args[11];
lean_object* v___y_1619_ = _args[12];
lean_object* v___y_1620_ = _args[13];
lean_object* v___y_1621_ = _args[14];
lean_object* v___y_1622_ = _args[15];
lean_object* v___y_1623_ = _args[16];
lean_object* v___y_1624_ = _args[17];
lean_object* v___y_1625_ = _args[18];
_start:
{
uint8_t v___x_38808__boxed_1626_; uint8_t v_keepUsing_boxed_1627_; uint8_t v___y_38809__boxed_1628_; uint8_t v___y_38810__boxed_1629_; lean_object* v_res_1630_; 
v___x_38808__boxed_1626_ = lean_unbox(v___x_1609_);
v_keepUsing_boxed_1627_ = lean_unbox(v_keepUsing_1611_);
v___y_38809__boxed_1628_ = lean_unbox(v___y_1612_);
v___y_38810__boxed_1629_ = lean_unbox(v___y_1613_);
v_res_1630_ = lp_mathlib_Mathlib_Tactic_Lift_main___lam__0(v_e_1607_, v___x_1608_, v___x_38808__boxed_1626_, v_hUsing_1610_, v_keepUsing_boxed_1627_, v___y_38809__boxed_1628_, v___y_38810__boxed_1629_, v___y_1614_, v_newVarName_1615_, v_t_1616_, v___y_1617_, v___y_1618_, v___y_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_);
lean_dec(v___y_1620_);
lean_dec_ref(v___y_1619_);
lean_dec(v___y_1618_);
lean_dec_ref(v___y_1617_);
lean_dec(v_newVarName_1615_);
return v_res_1630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main(lean_object* v_e_1634_, lean_object* v_t_1635_, lean_object* v_hUsing_1636_, lean_object* v_newVarName_1637_, lean_object* v_newEqName_1638_, uint8_t v_keepUsing_1639_, lean_object* v_a_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_, lean_object* v_a_1643_, lean_object* v_a_1644_, lean_object* v_a_1645_, lean_object* v_a_1646_, lean_object* v_a_1647_){
_start:
{
lean_object* v___y_1650_; uint8_t v___y_1651_; uint8_t v___y_1652_; uint8_t v___y_1662_; lean_object* v___y_1663_; uint8_t v___y_1669_; 
if (lean_obj_tag(v_newVarName_1637_) == 0)
{
uint8_t v___x_1673_; 
v___x_1673_ = 0;
v___y_1669_ = v___x_1673_;
goto v___jp_1668_;
}
else
{
uint8_t v___x_1674_; 
v___x_1674_ = 1;
v___y_1669_ = v___x_1674_;
goto v___jp_1668_;
}
v___jp_1649_:
{
lean_object* v___x_1653_; uint8_t v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___f_1659_; lean_object* v___x_1660_; 
v___x_1653_ = lean_box(0);
v___x_1654_ = 0;
v___x_1655_ = lean_box(v___x_1654_);
v___x_1656_ = lean_box(v_keepUsing_1639_);
v___x_1657_ = lean_box(v___y_1651_);
v___x_1658_ = lean_box(v___y_1652_);
v___f_1659_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Lift_main___lam__0___boxed), 19, 10);
lean_closure_set(v___f_1659_, 0, v_e_1634_);
lean_closure_set(v___f_1659_, 1, v___x_1653_);
lean_closure_set(v___f_1659_, 2, v___x_1655_);
lean_closure_set(v___f_1659_, 3, v_hUsing_1636_);
lean_closure_set(v___f_1659_, 4, v___x_1656_);
lean_closure_set(v___f_1659_, 5, v___x_1657_);
lean_closure_set(v___f_1659_, 6, v___x_1658_);
lean_closure_set(v___f_1659_, 7, v___y_1650_);
lean_closure_set(v___f_1659_, 8, v_newVarName_1637_);
lean_closure_set(v___f_1659_, 9, v_t_1635_);
v___x_1660_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1659_, v_a_1640_, v_a_1641_, v_a_1642_, v_a_1643_, v_a_1644_, v_a_1645_, v_a_1646_, v_a_1647_);
return v___x_1660_;
}
v___jp_1661_:
{
lean_object* v___x_1664_; uint8_t v___x_1665_; 
v___x_1664_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___closed__1));
v___x_1665_ = lean_name_eq(v___y_1663_, v___x_1664_);
if (v___x_1665_ == 0)
{
uint8_t v___x_1666_; 
v___x_1666_ = 1;
v___y_1650_ = v___y_1663_;
v___y_1651_ = v___y_1662_;
v___y_1652_ = v___x_1666_;
goto v___jp_1649_;
}
else
{
uint8_t v___x_1667_; 
v___x_1667_ = 0;
v___y_1650_ = v___y_1663_;
v___y_1651_ = v___y_1662_;
v___y_1652_ = v___x_1667_;
goto v___jp_1649_;
}
}
v___jp_1668_:
{
if (lean_obj_tag(v_newEqName_1638_) == 0)
{
lean_object* v___x_1670_; 
v___x_1670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Lift_main___closed__1));
v___y_1662_ = v___y_1669_;
v___y_1663_ = v___x_1670_;
goto v___jp_1661_;
}
else
{
lean_object* v_val_1671_; lean_object* v___x_1672_; 
v_val_1671_ = lean_ctor_get(v_newEqName_1638_, 0);
v___x_1672_ = l_Lean_Syntax_getId(v_val_1671_);
v___y_1662_ = v___y_1669_;
v___y_1663_ = v___x_1672_;
goto v___jp_1661_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Lift_main___boxed(lean_object* v_e_1675_, lean_object* v_t_1676_, lean_object* v_hUsing_1677_, lean_object* v_newVarName_1678_, lean_object* v_newEqName_1679_, lean_object* v_keepUsing_1680_, lean_object* v_a_1681_, lean_object* v_a_1682_, lean_object* v_a_1683_, lean_object* v_a_1684_, lean_object* v_a_1685_, lean_object* v_a_1686_, lean_object* v_a_1687_, lean_object* v_a_1688_, lean_object* v_a_1689_){
_start:
{
uint8_t v_keepUsing_boxed_1690_; lean_object* v_res_1691_; 
v_keepUsing_boxed_1690_ = lean_unbox(v_keepUsing_1680_);
v_res_1691_ = lp_mathlib_Mathlib_Tactic_Lift_main(v_e_1675_, v_t_1676_, v_hUsing_1677_, v_newVarName_1678_, v_newEqName_1679_, v_keepUsing_boxed_1690_, v_a_1681_, v_a_1682_, v_a_1683_, v_a_1684_, v_a_1685_, v_a_1686_, v_a_1687_, v_a_1688_);
lean_dec(v_a_1688_);
lean_dec_ref(v_a_1687_);
lean_dec(v_a_1686_);
lean_dec_ref(v_a_1685_);
lean_dec(v_a_1684_);
lean_dec_ref(v_a_1683_);
lean_dec(v_a_1682_);
lean_dec_ref(v_a_1681_);
lean_dec(v_newEqName_1679_);
return v_res_1691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0(lean_object* v_00_u03b1_1692_, lean_object* v_msg_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_){
_start:
{
lean_object* v___x_1703_; 
v___x_1703_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___redArg(v_msg_1693_, v___y_1698_, v___y_1699_, v___y_1700_, v___y_1701_);
return v___x_1703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0___boxed(lean_object* v_00_u03b1_1704_, lean_object* v_msg_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_){
_start:
{
lean_object* v_res_1715_; 
v_res_1715_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Lift_main_spec__0(v_00_u03b1_1704_, v_msg_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_, v___y_1713_);
lean_dec(v___y_1713_);
lean_dec_ref(v___y_1712_);
lean_dec(v___y_1711_);
lean_dec_ref(v___y_1710_);
lean_dec(v___y_1709_);
lean_dec_ref(v___y_1708_);
lean_dec(v___y_1707_);
lean_dec_ref(v___y_1706_);
return v_res_1715_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; 
v___x_1716_ = lean_box(0);
v___x_1717_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1718_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1718_, 0, v___x_1717_);
lean_ctor_set(v___x_1718_, 1, v___x_1716_);
return v___x_1718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1720_; lean_object* v___x_1721_; 
v___x_1720_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___closed__0);
v___x_1721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1721_, 0, v___x_1720_);
return v___x_1721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg___boxed(lean_object* v___y_1722_){
_start:
{
lean_object* v_res_1723_; 
v_res_1723_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v_res_1723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0(lean_object* v_00_u03b1_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_){
_start:
{
lean_object* v___x_1734_; 
v___x_1734_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v___x_1734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___boxed(lean_object* v_00_u03b1_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_){
_start:
{
lean_object* v_res_1745_; 
v_res_1745_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0(v_00_u03b1_1735_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_, v___y_1740_, v___y_1741_, v___y_1742_, v___y_1743_);
lean_dec(v___y_1743_);
lean_dec_ref(v___y_1742_);
lean_dec(v___y_1741_);
lean_dec_ref(v___y_1740_);
lean_dec(v___y_1739_);
lean_dec_ref(v___y_1738_);
lean_dec(v___y_1737_);
lean_dec_ref(v___y_1736_);
return v_res_1745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1(lean_object* v_x_1746_, lean_object* v_a_1747_, lean_object* v_a_1748_, lean_object* v_a_1749_, lean_object* v_a_1750_, lean_object* v_a_1751_, lean_object* v_a_1752_, lean_object* v_a_1753_, lean_object* v_a_1754_){
_start:
{
lean_object* v___x_1756_; uint8_t v___x_1757_; 
v___x_1756_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_lift___closed__3));
lean_inc(v_x_1746_);
v___x_1757_ = l_Lean_Syntax_isOfKind(v_x_1746_, v___x_1756_);
if (v___x_1757_ == 0)
{
lean_object* v___x_1758_; 
lean_dec(v_x_1746_);
v___x_1758_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v___x_1758_;
}
else
{
lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v_e_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v_t_1764_; lean_object* v___y_1766_; lean_object* v___y_1767_; lean_object* v___y_1768_; lean_object* v___y_1769_; lean_object* v___y_1770_; uint8_t v___y_1771_; lean_object* v___y_1772_; lean_object* v___y_1773_; lean_object* v___y_1774_; lean_object* v___y_1775_; lean_object* v___y_1776_; lean_object* v___y_1777_; lean_object* v___y_1782_; lean_object* v___y_1783_; lean_object* v___y_1784_; lean_object* v___y_1785_; lean_object* v___y_1786_; lean_object* v___y_1787_; lean_object* v___y_1788_; lean_object* v___y_1789_; lean_object* v___y_1790_; lean_object* v___y_1791_; lean_object* v___y_1792_; uint8_t v___y_1793_; lean_object* v___y_1797_; lean_object* v___y_1798_; lean_object* v___y_1799_; lean_object* v___y_1800_; lean_object* v___y_1801_; lean_object* v___y_1802_; lean_object* v___y_1803_; lean_object* v___y_1804_; lean_object* v___y_1805_; lean_object* v___y_1806_; lean_object* v___y_1807_; lean_object* v___y_1810_; lean_object* v_newVarName_1811_; lean_object* v_newEqName_1812_; lean_object* v_newPrfName_1813_; lean_object* v___y_1814_; lean_object* v___y_1815_; lean_object* v___y_1816_; lean_object* v___y_1817_; lean_object* v___y_1818_; lean_object* v___y_1819_; lean_object* v___y_1820_; lean_object* v___y_1821_; lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1830_; lean_object* v_newPrfName_1831_; lean_object* v___y_1832_; lean_object* v___y_1833_; lean_object* v___y_1834_; lean_object* v___y_1835_; lean_object* v___y_1836_; lean_object* v___y_1837_; lean_object* v___y_1838_; lean_object* v___y_1839_; lean_object* v___y_1844_; lean_object* v___y_1845_; lean_object* v___y_1846_; lean_object* v_newEqName_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v___y_1850_; lean_object* v___y_1851_; lean_object* v___y_1852_; lean_object* v___y_1853_; lean_object* v___y_1854_; lean_object* v___y_1855_; lean_object* v___x_1863_; lean_object* v_h_1865_; lean_object* v___y_1866_; lean_object* v___y_1867_; lean_object* v___y_1868_; lean_object* v___y_1869_; lean_object* v___y_1870_; lean_object* v___y_1871_; lean_object* v___y_1872_; lean_object* v___y_1873_; lean_object* v___x_1888_; uint8_t v___x_1889_; 
v___x_1759_ = lean_unsigned_to_nat(0u);
v___x_1760_ = lean_unsigned_to_nat(1u);
v_e_1761_ = l_Lean_Syntax_getArg(v_x_1746_, v___x_1760_);
v___x_1762_ = lean_unsigned_to_nat(2u);
v___x_1763_ = lean_unsigned_to_nat(3u);
v_t_1764_ = l_Lean_Syntax_getArg(v_x_1746_, v___x_1763_);
v___x_1863_ = lean_unsigned_to_nat(4u);
v___x_1888_ = l_Lean_Syntax_getArg(v_x_1746_, v___x_1863_);
v___x_1889_ = l_Lean_Syntax_isNone(v___x_1888_);
if (v___x_1889_ == 0)
{
uint8_t v___x_1890_; 
lean_inc(v___x_1888_);
v___x_1890_ = l_Lean_Syntax_matchesNull(v___x_1888_, v___x_1762_);
if (v___x_1890_ == 0)
{
lean_object* v___x_1891_; 
lean_dec(v___x_1888_);
lean_dec(v_t_1764_);
lean_dec(v_e_1761_);
lean_dec(v_x_1746_);
v___x_1891_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v___x_1891_;
}
else
{
lean_object* v_h_1892_; lean_object* v___x_1893_; 
v_h_1892_ = l_Lean_Syntax_getArg(v___x_1888_, v___x_1760_);
lean_dec(v___x_1888_);
v___x_1893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1893_, 0, v_h_1892_);
v_h_1865_ = v___x_1893_;
v___y_1866_ = v_a_1747_;
v___y_1867_ = v_a_1748_;
v___y_1868_ = v_a_1749_;
v___y_1869_ = v_a_1750_;
v___y_1870_ = v_a_1751_;
v___y_1871_ = v_a_1752_;
v___y_1872_ = v_a_1753_;
v___y_1873_ = v_a_1754_;
goto v___jp_1864_;
}
}
else
{
lean_object* v___x_1894_; 
lean_dec(v___x_1888_);
v___x_1894_ = lean_box(0);
v_h_1865_ = v___x_1894_;
v___y_1866_ = v_a_1747_;
v___y_1867_ = v_a_1748_;
v___y_1868_ = v_a_1749_;
v___y_1869_ = v_a_1750_;
v___y_1870_ = v_a_1751_;
v___y_1871_ = v_a_1752_;
v___y_1872_ = v_a_1753_;
v___y_1873_ = v_a_1754_;
goto v___jp_1864_;
}
v___jp_1765_:
{
lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; 
v___x_1778_ = lean_box(v___y_1771_);
v___x_1779_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Lift_main___boxed), 15, 6);
lean_closure_set(v___x_1779_, 0, v_e_1761_);
lean_closure_set(v___x_1779_, 1, v_t_1764_);
lean_closure_set(v___x_1779_, 2, v___y_1775_);
lean_closure_set(v___x_1779_, 3, v___y_1769_);
lean_closure_set(v___x_1779_, 4, v___y_1777_);
lean_closure_set(v___x_1779_, 5, v___x_1778_);
v___x_1780_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_1779_, v___y_1767_, v___y_1776_, v___y_1768_, v___y_1766_, v___y_1773_, v___y_1772_, v___y_1770_, v___y_1774_);
return v___x_1780_;
}
v___jp_1781_:
{
if (lean_obj_tag(v___y_1787_) == 0)
{
lean_object* v___x_1794_; 
v___x_1794_ = lean_box(0);
v___y_1766_ = v___y_1783_;
v___y_1767_ = v___y_1782_;
v___y_1768_ = v___y_1784_;
v___y_1769_ = v___y_1785_;
v___y_1770_ = v___y_1786_;
v___y_1771_ = v___y_1793_;
v___y_1772_ = v___y_1789_;
v___y_1773_ = v___y_1788_;
v___y_1774_ = v___y_1791_;
v___y_1775_ = v___y_1790_;
v___y_1776_ = v___y_1792_;
v___y_1777_ = v___x_1794_;
goto v___jp_1765_;
}
else
{
lean_object* v_val_1795_; 
v_val_1795_ = lean_ctor_get(v___y_1787_, 0);
lean_inc(v_val_1795_);
lean_dec_ref_known(v___y_1787_, 1);
v___y_1766_ = v___y_1783_;
v___y_1767_ = v___y_1782_;
v___y_1768_ = v___y_1784_;
v___y_1769_ = v___y_1785_;
v___y_1770_ = v___y_1786_;
v___y_1771_ = v___y_1793_;
v___y_1772_ = v___y_1789_;
v___y_1773_ = v___y_1788_;
v___y_1774_ = v___y_1791_;
v___y_1775_ = v___y_1790_;
v___y_1776_ = v___y_1792_;
v___y_1777_ = v_val_1795_;
goto v___jp_1765_;
}
}
v___jp_1796_:
{
uint8_t v___x_1808_; 
v___x_1808_ = 0;
v___y_1782_ = v___y_1798_;
v___y_1783_ = v___y_1797_;
v___y_1784_ = v___y_1799_;
v___y_1785_ = v___y_1800_;
v___y_1786_ = v___y_1801_;
v___y_1787_ = v___y_1804_;
v___y_1788_ = v___y_1803_;
v___y_1789_ = v___y_1802_;
v___y_1790_ = v___y_1806_;
v___y_1791_ = v___y_1805_;
v___y_1792_ = v___y_1807_;
v___y_1793_ = v___x_1808_;
goto v___jp_1781_;
}
v___jp_1809_:
{
if (lean_obj_tag(v___y_1810_) == 1)
{
if (lean_obj_tag(v_newPrfName_1813_) == 0)
{
v___y_1797_ = v___y_1817_;
v___y_1798_ = v___y_1814_;
v___y_1799_ = v___y_1816_;
v___y_1800_ = v_newVarName_1811_;
v___y_1801_ = v___y_1820_;
v___y_1802_ = v___y_1819_;
v___y_1803_ = v___y_1818_;
v___y_1804_ = v_newEqName_1812_;
v___y_1805_ = v___y_1821_;
v___y_1806_ = v___y_1810_;
v___y_1807_ = v___y_1815_;
goto v___jp_1796_;
}
else
{
lean_object* v_val_1822_; 
v_val_1822_ = lean_ctor_get(v_newPrfName_1813_, 0);
lean_inc(v_val_1822_);
lean_dec_ref_known(v_newPrfName_1813_, 1);
if (lean_obj_tag(v_val_1822_) == 1)
{
lean_object* v_val_1823_; lean_object* v_val_1824_; uint8_t v___x_1825_; 
v_val_1823_ = lean_ctor_get(v___y_1810_, 0);
v_val_1824_ = lean_ctor_get(v_val_1822_, 0);
lean_inc(v_val_1824_);
lean_dec_ref_known(v_val_1822_, 1);
v___x_1825_ = l_Lean_Syntax_structEq(v_val_1823_, v_val_1824_);
lean_dec(v_val_1824_);
v___y_1782_ = v___y_1814_;
v___y_1783_ = v___y_1817_;
v___y_1784_ = v___y_1816_;
v___y_1785_ = v_newVarName_1811_;
v___y_1786_ = v___y_1820_;
v___y_1787_ = v_newEqName_1812_;
v___y_1788_ = v___y_1818_;
v___y_1789_ = v___y_1819_;
v___y_1790_ = v___y_1810_;
v___y_1791_ = v___y_1821_;
v___y_1792_ = v___y_1815_;
v___y_1793_ = v___x_1825_;
goto v___jp_1781_;
}
else
{
lean_dec(v_val_1822_);
v___y_1797_ = v___y_1817_;
v___y_1798_ = v___y_1814_;
v___y_1799_ = v___y_1816_;
v___y_1800_ = v_newVarName_1811_;
v___y_1801_ = v___y_1820_;
v___y_1802_ = v___y_1819_;
v___y_1803_ = v___y_1818_;
v___y_1804_ = v_newEqName_1812_;
v___y_1805_ = v___y_1821_;
v___y_1806_ = v___y_1810_;
v___y_1807_ = v___y_1815_;
goto v___jp_1796_;
}
}
}
else
{
uint8_t v___x_1826_; 
lean_dec(v_newPrfName_1813_);
v___x_1826_ = 0;
v___y_1782_ = v___y_1814_;
v___y_1783_ = v___y_1817_;
v___y_1784_ = v___y_1816_;
v___y_1785_ = v_newVarName_1811_;
v___y_1786_ = v___y_1820_;
v___y_1787_ = v_newEqName_1812_;
v___y_1788_ = v___y_1818_;
v___y_1789_ = v___y_1819_;
v___y_1790_ = v___y_1810_;
v___y_1791_ = v___y_1821_;
v___y_1792_ = v___y_1815_;
v___y_1793_ = v___x_1826_;
goto v___jp_1781_;
}
}
v___jp_1827_:
{
lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; 
v___x_1840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1840_, 0, v___y_1829_);
v___x_1841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1841_, 0, v___y_1828_);
v___x_1842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1842_, 0, v_newPrfName_1831_);
v___y_1810_ = v___y_1830_;
v_newVarName_1811_ = v___x_1840_;
v_newEqName_1812_ = v___x_1841_;
v_newPrfName_1813_ = v___x_1842_;
v___y_1814_ = v___y_1832_;
v___y_1815_ = v___y_1833_;
v___y_1816_ = v___y_1834_;
v___y_1817_ = v___y_1835_;
v___y_1818_ = v___y_1836_;
v___y_1819_ = v___y_1837_;
v___y_1820_ = v___y_1838_;
v___y_1821_ = v___y_1839_;
goto v___jp_1809_;
}
v___jp_1843_:
{
lean_object* v___x_1856_; uint8_t v___x_1857_; 
v___x_1856_ = l_Lean_Syntax_getArg(v___y_1844_, v___x_1763_);
lean_dec(v___y_1844_);
v___x_1857_ = l_Lean_Syntax_isNone(v___x_1856_);
if (v___x_1857_ == 0)
{
uint8_t v___x_1858_; 
lean_inc(v___x_1856_);
v___x_1858_ = l_Lean_Syntax_matchesNull(v___x_1856_, v___x_1760_);
if (v___x_1858_ == 0)
{
lean_object* v___x_1859_; 
lean_dec(v___x_1856_);
lean_dec(v_newEqName_1847_);
lean_dec(v___y_1846_);
lean_dec(v___y_1845_);
lean_dec(v_t_1764_);
lean_dec(v_e_1761_);
v___x_1859_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v___x_1859_;
}
else
{
lean_object* v_newPrfName_1860_; lean_object* v___x_1861_; 
v_newPrfName_1860_ = l_Lean_Syntax_getArg(v___x_1856_, v___x_1759_);
lean_dec(v___x_1856_);
v___x_1861_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1861_, 0, v_newPrfName_1860_);
v___y_1828_ = v_newEqName_1847_;
v___y_1829_ = v___y_1845_;
v___y_1830_ = v___y_1846_;
v_newPrfName_1831_ = v___x_1861_;
v___y_1832_ = v___y_1848_;
v___y_1833_ = v___y_1849_;
v___y_1834_ = v___y_1850_;
v___y_1835_ = v___y_1851_;
v___y_1836_ = v___y_1852_;
v___y_1837_ = v___y_1853_;
v___y_1838_ = v___y_1854_;
v___y_1839_ = v___y_1855_;
goto v___jp_1827_;
}
}
else
{
lean_object* v___x_1862_; 
lean_dec(v___x_1856_);
v___x_1862_ = lean_box(0);
v___y_1828_ = v_newEqName_1847_;
v___y_1829_ = v___y_1845_;
v___y_1830_ = v___y_1846_;
v_newPrfName_1831_ = v___x_1862_;
v___y_1832_ = v___y_1848_;
v___y_1833_ = v___y_1849_;
v___y_1834_ = v___y_1850_;
v___y_1835_ = v___y_1851_;
v___y_1836_ = v___y_1852_;
v___y_1837_ = v___y_1853_;
v___y_1838_ = v___y_1854_;
v___y_1839_ = v___y_1855_;
goto v___jp_1827_;
}
}
v___jp_1864_:
{
lean_object* v___x_1874_; lean_object* v___x_1875_; uint8_t v___x_1876_; 
v___x_1874_ = lean_unsigned_to_nat(5u);
v___x_1875_ = l_Lean_Syntax_getArg(v_x_1746_, v___x_1874_);
lean_dec(v_x_1746_);
v___x_1876_ = l_Lean_Syntax_isNone(v___x_1875_);
if (v___x_1876_ == 0)
{
uint8_t v___x_1877_; 
lean_inc(v___x_1875_);
v___x_1877_ = l_Lean_Syntax_matchesNull(v___x_1875_, v___x_1863_);
if (v___x_1877_ == 0)
{
lean_object* v___x_1878_; 
lean_dec(v___x_1875_);
lean_dec(v_h_1865_);
lean_dec(v_t_1764_);
lean_dec(v_e_1761_);
v___x_1878_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v___x_1878_;
}
else
{
lean_object* v_newVarName_1879_; lean_object* v___x_1880_; uint8_t v___x_1881_; 
v_newVarName_1879_ = l_Lean_Syntax_getArg(v___x_1875_, v___x_1760_);
v___x_1880_ = l_Lean_Syntax_getArg(v___x_1875_, v___x_1762_);
v___x_1881_ = l_Lean_Syntax_isNone(v___x_1880_);
if (v___x_1881_ == 0)
{
uint8_t v___x_1882_; 
lean_inc(v___x_1880_);
v___x_1882_ = l_Lean_Syntax_matchesNull(v___x_1880_, v___x_1760_);
if (v___x_1882_ == 0)
{
lean_object* v___x_1883_; 
lean_dec(v___x_1880_);
lean_dec(v_newVarName_1879_);
lean_dec(v___x_1875_);
lean_dec(v_h_1865_);
lean_dec(v_t_1764_);
lean_dec(v_e_1761_);
v___x_1883_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1_spec__0___redArg();
return v___x_1883_;
}
else
{
lean_object* v_newEqName_1884_; lean_object* v___x_1885_; 
v_newEqName_1884_ = l_Lean_Syntax_getArg(v___x_1880_, v___x_1759_);
lean_dec(v___x_1880_);
v___x_1885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1885_, 0, v_newEqName_1884_);
v___y_1844_ = v___x_1875_;
v___y_1845_ = v_newVarName_1879_;
v___y_1846_ = v_h_1865_;
v_newEqName_1847_ = v___x_1885_;
v___y_1848_ = v___y_1866_;
v___y_1849_ = v___y_1867_;
v___y_1850_ = v___y_1868_;
v___y_1851_ = v___y_1869_;
v___y_1852_ = v___y_1870_;
v___y_1853_ = v___y_1871_;
v___y_1854_ = v___y_1872_;
v___y_1855_ = v___y_1873_;
goto v___jp_1843_;
}
}
else
{
lean_object* v___x_1886_; 
lean_dec(v___x_1880_);
v___x_1886_ = lean_box(0);
v___y_1844_ = v___x_1875_;
v___y_1845_ = v_newVarName_1879_;
v___y_1846_ = v_h_1865_;
v_newEqName_1847_ = v___x_1886_;
v___y_1848_ = v___y_1866_;
v___y_1849_ = v___y_1867_;
v___y_1850_ = v___y_1868_;
v___y_1851_ = v___y_1869_;
v___y_1852_ = v___y_1870_;
v___y_1853_ = v___y_1871_;
v___y_1854_ = v___y_1872_;
v___y_1855_ = v___y_1873_;
goto v___jp_1843_;
}
}
}
else
{
lean_object* v___x_1887_; 
lean_dec(v___x_1875_);
v___x_1887_ = lean_box(0);
v___y_1810_ = v_h_1865_;
v_newVarName_1811_ = v___x_1887_;
v_newEqName_1812_ = v___x_1887_;
v_newPrfName_1813_ = v___x_1887_;
v___y_1814_ = v___y_1866_;
v___y_1815_ = v___y_1867_;
v___y_1816_ = v___y_1868_;
v___y_1817_ = v___y_1869_;
v___y_1818_ = v___y_1870_;
v___y_1819_ = v___y_1871_;
v___y_1820_ = v___y_1872_;
v___y_1821_ = v___y_1873_;
goto v___jp_1809_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1___boxed(lean_object* v_x_1895_, lean_object* v_a_1896_, lean_object* v_a_1897_, lean_object* v_a_1898_, lean_object* v_a_1899_, lean_object* v_a_1900_, lean_object* v_a_1901_, lean_object* v_a_1902_, lean_object* v_a_1903_, lean_object* v_a_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Lift______elabRules__Mathlib__Tactic__lift__1(v_x_1895_, v_a_1896_, v_a_1897_, v_a_1898_, v_a_1899_, v_a_1900_, v_a_1901_, v_a_1902_, v_a_1903_);
lean_dec(v_a_1903_);
lean_dec_ref(v_a_1902_);
lean_dec(v_a_1901_);
lean_dec_ref(v_a_1900_);
lean_dec(v_a_1899_);
lean_dec_ref(v_a_1898_);
lean_dec(v_a_1897_);
lean_dec_ref(v_a_1896_);
return v_res_1905_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Expr(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_UnusedNames(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_RCases(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_UnusedNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_RCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Expr(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_UnusedNames(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_RCases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_UnusedNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_RCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Lift(builtin);
}
#ifdef __cplusplus
}
#endif
