// Lean compiler output
// Module: Mathlib.SetTheory.Ordinal.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.SuccPred public import Mathlib.Data.Sum.Order public import Mathlib.Order.IsNormal public import Mathlib.Order.Shrink public import Mathlib.SetTheory.Cardinal.Basic public import Mathlib.Tactic.PPWithUniv
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchLambda(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchLambda___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellOrder_inhabited;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_isEquivalent;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_type(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal_termTypeLT___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Ordinal"};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__0 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__0_value;
static const lean_string_object lp_mathlib_Ordinal_termTypeLT___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "termTypeLT_"};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__1 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(108, 0, 58, 178, 166, 80, 226, 110)}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__2 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__2_value;
static const lean_string_object lp_mathlib_Ordinal_termTypeLT___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__3 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__4 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__4_value;
static const lean_string_object lp_mathlib_Ordinal_termTypeLT___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "typeLT "};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__5 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__5_value)}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__6 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__6_value;
static const lean_string_object lp_mathlib_Ordinal_termTypeLT___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__7 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__8 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__8_value),((lean_object*)(((size_t)(70) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__9 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__4_value),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__6_value),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__9_value)}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__10 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Ordinal_termTypeLT___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__10_value)}};
static const lean_object* lp_mathlib_Ordinal_termTypeLT___00__closed__11 = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_termTypeLT__ = (const lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__11_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__3 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__5 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__5_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__7 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__7_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Ordinal.type"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__8 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__8_value;
static lean_once_cell_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__9;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "type"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__10 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(59, 91, 242, 99, 57, 182, 110, 76)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__12 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__13 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__13_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__14 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__14_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__15 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__15_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__16 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__16_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__18 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__18_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value_aux_1),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__20 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__20_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__21 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__21_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__22 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__22_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__23 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__23_value;
static lean_once_cell_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__24;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__25 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__25_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__25_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__26 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__26_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__27 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__27_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(102, 52, 160, 208, 138, 190, 17, 238)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__28 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__28_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__28_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__29 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__29_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Equiv"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__30 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__30_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(0, 253, 123, 237, 128, 91, 245, 83)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__31 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__31_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__31_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__32 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__32_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__33 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__33_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__34 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__34_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__34_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__35 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__35_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Cardinal"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__36 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__36_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(247, 176, 10, 15, 146, 217, 130, 31)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__37 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__37_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__37_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__38 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__38_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__39 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__39_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__40 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__40_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__40_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__41 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__41_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__41_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__42 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__42_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__38_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__42_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__43 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__43_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__35_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__43_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__44 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__44_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__32_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__44_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__45 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__45_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__29_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__45_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__46 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__46_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__26_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__46_value)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__47 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__47_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_<_"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__48 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__48_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__48_value),LEAN_SCALAR_PTR_LITERAL(192, 242, 106, 74, 199, 131, 133, 95)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__49 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__49_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__50 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__50_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value_aux_1),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value_aux_2),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(215, 94, 65, 66, 49, 100, 151, 85)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__52 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__52_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__53 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__53_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__54 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__54_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferInstance"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__55 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__55_value;
static lean_once_cell_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__56;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__55_value),LEAN_SCALAR_PTR_LITERAL(17, 162, 120, 176, 98, 85, 114, 76)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__57 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__57_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__57_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__58 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__58_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__58_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__59 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__59_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__1_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__1_value)} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__0_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__1_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__2___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__2_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___boxed, .m_arity = 11, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__1_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__0_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__2_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__2_value)} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__3 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__3_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__4 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__4_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__5 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__5_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__3_value)} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__6 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__6_value;
static const lean_closure_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__5_value),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__6_value)} };
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__7 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_zero;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_inhabited;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_one;
static const lean_ctor_object lp_mathlib_Ordinal_partialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal_partialOrder___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_partialOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_partialOrder = (const lean_object*)&lp_mathlib_Ordinal_partialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_instOrderBot;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_typein___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_typein___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Ordinal_typein___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal_typein___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal_typein___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_typein___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal_typein___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_typein___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal_typein___closed__1 = (const lean_object*)&lp_mathlib_Ordinal_typein___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_typein(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_wellFoundedRelation;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_card(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_card___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_lift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_lift___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_liftInitialSeg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_liftInitialSeg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Ordinal_liftInitialSeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal_liftInitialSeg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal_liftInitialSeg___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_liftInitialSeg___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_liftInitialSeg = (const lean_object*)&lp_mathlib_Ordinal_liftInitialSeg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_omega0;
static const lean_string_object lp_mathlib_Ordinal_term_u03c9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termω"};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(16, 238, 207, 100, 28, 52, 123, 75)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___closed__1 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__1_value;
static const lean_string_object lp_mathlib_Ordinal_term_u03c9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "ω"};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___closed__2 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__2_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__2_value)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___closed__3 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal_term_u03c9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__3_value)}};
static const lean_object* lp_mathlib_Ordinal_term_u03c9___closed__4 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_term_u03c9 = (const lean_object*)&lp_mathlib_Ordinal_term_u03c9___closed__4_value;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Ordinal.omega0"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__1;
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "omega0"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__2 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_termTypeLT___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 160, 33, 233, 30, 237, 55, 251)}};
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(218, 134, 197, 182, 224, 208, 57, 29)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__4 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__5 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__0 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__1 = (const lean_object*)&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_add___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_add___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Ordinal_add___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal_add___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal_add___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_add___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_add = (const lean_object*)&lp_mathlib_Ordinal_add___closed__0_value;
static const lean_closure_object lp_mathlib_Ordinal_addMonoidWithOne___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_unaryCast___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_add___closed__0_value)} };
static const lean_object* lp_mathlib_Ordinal_addMonoidWithOne___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__0_value;
static const lean_closure_object lp_mathlib_Ordinal_addMonoidWithOne___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_nsmulRec___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_add___closed__0_value)} };
static const lean_object* lp_mathlib_Ordinal_addMonoidWithOne___closed__1 = (const lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__1_value;
static const lean_ctor_object lp_mathlib_Ordinal_addMonoidWithOne___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordinal_add___closed__0_value),((lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__1_value)}};
static const lean_object* lp_mathlib_Ordinal_addMonoidWithOne___closed__2 = (const lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__2_value;
static const lean_ctor_object lp_mathlib_Ordinal_addMonoidWithOne___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__0_value),((lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal_addMonoidWithOne___closed__3 = (const lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_addMonoidWithOne = (const lean_object*)&lp_mathlib_Ordinal_addMonoidWithOne___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_instSuccOrder___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_instSuccOrder___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Ordinal_instSuccOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal_instSuccOrder___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal_instSuccOrder___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_instSuccOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_instSuccOrder = (const lean_object*)&lp_mathlib_Ordinal_instSuccOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_instSuccAddOrder = (const lean_object*)&lp_mathlib_Ordinal_instSuccOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_uniqueIioOne;
static lean_object* _init_lp_mathlib_WellOrder_inhabited(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_isEquivalent(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_type(lean_object* v_00_u03b1_3_, lean_object* v_r_4_, lean_object* v_wo_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__9(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__8));
v___x_51_ = l_String_toRawSubstring_x27(v___x_50_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__24(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_82_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__23));
v___x_83_ = l_String_toRawSubstring_x27(v___x_82_);
return v___x_83_;
}
}
static lean_object* _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__56(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__55));
v___x_145_ = l_String_toRawSubstring_x27(v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1(lean_object* v_x_154_, lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_157_ = ((lean_object*)(lp_mathlib_Ordinal_termTypeLT___00__closed__2));
lean_inc(v_x_154_);
v___x_158_ = l_Lean_Syntax_isOfKind(v_x_154_, v___x_157_);
if (v___x_158_ == 0)
{
lean_object* v___x_159_; lean_object* v___x_160_; 
lean_dec(v_x_154_);
v___x_159_ = lean_box(1);
v___x_160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v_a_156_);
return v___x_160_;
}
else
{
lean_object* v_quotContext_161_; lean_object* v_currMacroScope_162_; lean_object* v_ref_163_; lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v_quotContext_161_ = lean_ctor_get(v_a_155_, 1);
v_currMacroScope_162_ = lean_ctor_get(v_a_155_, 2);
v_ref_163_ = lean_ctor_get(v_a_155_, 5);
v___x_164_ = lean_unsigned_to_nat(1u);
v___x_165_ = l_Lean_Syntax_getArg(v_x_154_, v___x_164_);
lean_dec(v_x_154_);
v___x_166_ = 0;
v___x_167_ = l_Lean_SourceInfo_fromRef(v_ref_163_, v___x_166_);
v___x_168_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__4));
v___x_169_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__6));
v___x_170_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__7));
lean_inc_n(v___x_167_, 15);
v___x_171_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_167_);
lean_ctor_set(v___x_171_, 1, v___x_170_);
v___x_172_ = lean_obj_once(&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__9, &lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__9_once, _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__9);
v___x_173_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11));
lean_inc_n(v_currMacroScope_162_, 3);
lean_inc_n(v_quotContext_161_, 3);
v___x_174_ = l_Lean_addMacroScope(v_quotContext_161_, v___x_173_, v_currMacroScope_162_);
v___x_175_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__13));
v___x_176_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_176_, 0, v___x_167_);
lean_ctor_set(v___x_176_, 1, v___x_172_);
lean_ctor_set(v___x_176_, 2, v___x_174_);
lean_ctor_set(v___x_176_, 3, v___x_175_);
v___x_177_ = l_Lean_Syntax_node2(v___x_167_, v___x_169_, v___x_171_, v___x_176_);
v___x_178_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__15));
v___x_179_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__17));
v___x_180_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__19));
v___x_181_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__20));
v___x_182_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_167_);
lean_ctor_set(v___x_182_, 1, v___x_181_);
v___x_183_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__22));
v___x_184_ = lean_obj_once(&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__24, &lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__24_once, _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__24);
v___x_185_ = lean_box(0);
v___x_186_ = l_Lean_addMacroScope(v_quotContext_161_, v___x_185_, v_currMacroScope_162_);
v___x_187_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__47));
v___x_188_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_188_, 0, v___x_167_);
lean_ctor_set(v___x_188_, 1, v___x_184_);
lean_ctor_set(v___x_188_, 2, v___x_186_);
lean_ctor_set(v___x_188_, 3, v___x_187_);
v___x_189_ = l_Lean_Syntax_node1(v___x_167_, v___x_183_, v___x_188_);
lean_inc(v___x_189_);
v___x_190_ = l_Lean_Syntax_node2(v___x_167_, v___x_180_, v___x_182_, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__49));
v___x_192_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__51));
v___x_193_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__52));
v___x_194_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_167_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v___x_195_ = l_Lean_Syntax_node2(v___x_167_, v___x_192_, v___x_194_, v___x_189_);
v___x_196_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__53));
v___x_197_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_167_);
lean_ctor_set(v___x_197_, 1, v___x_196_);
lean_inc(v___x_195_);
v___x_198_ = l_Lean_Syntax_node3(v___x_167_, v___x_191_, v___x_195_, v___x_197_, v___x_195_);
v___x_199_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__54));
v___x_200_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_167_);
lean_ctor_set(v___x_200_, 1, v___x_199_);
v___x_201_ = l_Lean_Syntax_node3(v___x_167_, v___x_179_, v___x_190_, v___x_198_, v___x_200_);
v___x_202_ = lean_obj_once(&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__56, &lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__56_once, _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__56);
v___x_203_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__57));
v___x_204_ = l_Lean_addMacroScope(v_quotContext_161_, v___x_203_, v_currMacroScope_162_);
v___x_205_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__59));
v___x_206_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_206_, 0, v___x_167_);
lean_ctor_set(v___x_206_, 1, v___x_202_);
lean_ctor_set(v___x_206_, 2, v___x_204_);
lean_ctor_set(v___x_206_, 3, v___x_205_);
v___x_207_ = l_Lean_Syntax_node3(v___x_167_, v___x_178_, v___x_165_, v___x_201_, v___x_206_);
v___x_208_ = l_Lean_Syntax_node2(v___x_167_, v___x_168_, v___x_177_, v___x_207_);
v___x_209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v_a_156_);
return v___x_209_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___boxed(lean_object* v_x_210_, lean_object* v_a_211_, lean_object* v_a_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1(v_x_210_, v_a_211_, v_a_212_);
lean_dec_ref(v_a_211_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg(lean_object* v___y_214_){
_start:
{
lean_object* v_subExpr_216_; lean_object* v_expr_217_; lean_object* v___x_218_; 
v_subExpr_216_ = lean_ctor_get(v___y_214_, 3);
v_expr_217_ = lean_ctor_get(v_subExpr_216_, 0);
lean_inc_ref(v_expr_217_);
v___x_218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_218_, 0, v_expr_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg___boxed(lean_object* v___y_219_, lean_object* v___y_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg(v___y_219_);
lean_dec_ref(v___y_219_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0(lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg(v___y_222_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___boxed(lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0(v___y_230_, v___y_231_, v___y_232_, v___y_233_, v___y_234_, v___y_235_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
lean_dec(v___y_233_);
lean_dec_ref(v___y_232_);
lean_dec(v___y_231_);
lean_dec_ref(v___y_230_);
return v_res_237_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0(lean_object* v_x_243_){
_start:
{
lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_244_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___closed__2));
v___x_245_ = l_Lean_Expr_isConstOf(v_x_243_, v___x_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0___boxed(lean_object* v_x_246_){
_start:
{
uint8_t v_res_247_; lean_object* v_r_248_; 
v_res_247_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__0(v_x_246_);
lean_dec_ref(v_x_246_);
v_r_248_ = lean_box(v_res_247_);
return v_r_248_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__1(lean_object* v_x_249_){
_start:
{
lean_object* v___x_250_; uint8_t v___x_251_; 
v___x_250_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__termTypeLT____1___closed__11));
v___x_251_ = l_Lean_Expr_isConstOf(v_x_249_, v___x_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__1___boxed(lean_object* v_x_252_){
_start:
{
uint8_t v_res_253_; lean_object* v_r_254_; 
v_res_253_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__1(v_x_252_);
lean_dec_ref(v_x_252_);
v_r_254_ = lean_box(v_res_253_);
return v_r_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__2(lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_263_, 0, v___y_255_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__2___boxed(lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__2(v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_);
lean_dec(v___y_270_);
lean_dec_ref(v___y_269_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
return v_res_272_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4(lean_object* v_n_273_, lean_object* v_x_274_){
_start:
{
uint8_t v___x_275_; 
v___x_275_ = lean_expr_eqv(v_x_274_, v_n_273_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4___boxed(lean_object* v_n_276_, lean_object* v_x_277_){
_start:
{
uint8_t v_res_278_; lean_object* v_r_279_; 
v_res_278_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4(v_n_276_, v_x_277_);
lean_dec_ref(v_x_277_);
lean_dec_ref(v_n_276_);
v_r_279_ = lean_box(v_res_278_);
return v_r_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__5(lean_object* v___f_280_, lean_object* v___x_281_, lean_object* v___f_282_, lean_object* v___f_283_, lean_object* v_n_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v___f_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4___boxed), 2, 1);
lean_closure_set(v___f_293_, 0, v_n_284_);
v___x_294_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_294_, 0, v___f_280_);
v___x_295_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_295_, 0, v___x_294_);
lean_closure_set(v___x_295_, 1, v___x_281_);
v___x_296_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_296_, 0, v___x_295_);
lean_closure_set(v___x_296_, 1, v___f_282_);
v___x_297_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_297_, 0, v___f_283_);
v___x_298_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_298_, 0, v___x_296_);
lean_closure_set(v___x_298_, 1, v___x_297_);
v___x_299_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_299_, 0, v___f_293_);
v___x_300_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_298_, v___x_299_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__5___boxed(lean_object* v___f_301_, lean_object* v___x_302_, lean_object* v___f_303_, lean_object* v___f_304_, lean_object* v_n_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__5(v___f_301_, v___x_302_, v___f_303_, v___f_304_, v_n_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__3(lean_object* v___f_315_, lean_object* v___x_316_, lean_object* v___f_317_, lean_object* v_n_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
lean_object* v___f_327_; lean_object* v___f_328_; lean_object* v___x_329_; 
v___f_327_ = lean_alloc_closure((void*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__4___boxed), 2, 1);
lean_closure_set(v___f_327_, 0, v_n_318_);
lean_inc_ref(v___x_316_);
v___f_328_ = lean_alloc_closure((void*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__5___boxed), 13, 4);
lean_closure_set(v___f_328_, 0, v___f_315_);
lean_closure_set(v___f_328_, 1, v___x_316_);
lean_closure_set(v___f_328_, 2, v___f_317_);
lean_closure_set(v___f_328_, 3, v___f_327_);
v___x_329_ = lp_mathlib_Mathlib_Notation3_matchLambda(v___x_316_, v___f_328_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__3___boxed(lean_object* v___f_330_, lean_object* v___x_331_, lean_object* v___f_332_, lean_object* v_n_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__3(v___f_330_, v___x_331_, v___f_332_, v_n_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
lean_dec(v___y_340_);
lean_dec_ref(v___y_339_);
lean_dec(v___y_338_);
lean_dec_ref(v___y_337_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
return v_res_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__6(lean_object* v_a_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v_ref_351_; uint8_t v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v_ref_351_ = lean_ctor_get(v___y_348_, 5);
v___x_352_ = 0;
v___x_353_ = l_Lean_SourceInfo_fromRef(v_ref_351_, v___x_352_);
v___x_354_ = ((lean_object*)(lp_mathlib_Ordinal_termTypeLT___00__closed__2));
v___x_355_ = ((lean_object*)(lp_mathlib_Ordinal_termTypeLT___00__closed__5));
lean_inc(v___x_353_);
v___x_356_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_353_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
v___x_357_ = l_Lean_Syntax_node2(v___x_353_, v___x_354_, v___x_356_, v_a_343_);
v___x_358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__6___boxed(lean_object* v_a_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__6(v_a_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_, v___y_364_, v___y_365_);
lean_dec(v___y_365_);
lean_dec_ref(v___y_364_);
lean_dec(v___y_363_);
lean_dec_ref(v___y_362_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7(lean_object* v___f_373_, lean_object* v___f_374_, lean_object* v___f_375_, lean_object* v___f_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_){
_start:
{
lean_object* v___x_384_; lean_object* v_a_385_; lean_object* v___x_387_; uint8_t v_isShared_388_; uint8_t v_isSharedCheck_414_; 
v___x_384_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1_spec__0___redArg(v___y_377_);
v_a_385_ = lean_ctor_get(v___x_384_, 0);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_384_);
if (v_isSharedCheck_414_ == 0)
{
v___x_387_ = v___x_384_;
v_isShared_388_ = v_isSharedCheck_414_;
goto v_resetjp_386_;
}
else
{
lean_inc(v_a_385_);
lean_dec(v___x_384_);
v___x_387_ = lean_box(0);
v_isShared_388_ = v_isSharedCheck_414_;
goto v_resetjp_386_;
}
v_resetjp_386_:
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___f_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_389_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_389_, 0, v___f_373_);
v___x_390_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__1));
v___x_391_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___closed__2));
v___f_392_ = lean_alloc_closure((void*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__3___boxed), 12, 3);
lean_closure_set(v___f_392_, 0, v___f_374_);
lean_closure_set(v___f_392_, 1, v___x_391_);
lean_closure_set(v___f_392_, 2, v___f_375_);
v___x_393_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_393_, 0, v___x_389_);
lean_closure_set(v___x_393_, 1, v___x_391_);
v___x_394_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchLambda___boxed), 10, 2);
lean_closure_set(v___x_394_, 0, v___x_391_);
lean_closure_set(v___x_394_, 1, v___f_392_);
v___x_395_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_395_, 0, v___x_393_);
lean_closure_set(v___x_395_, 1, v___x_394_);
v___x_396_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_397_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_395_, v___f_376_, v___x_396_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_);
if (lean_obj_tag(v___x_397_) == 0)
{
lean_object* v_a_398_; lean_object* v___x_400_; 
v_a_398_ = lean_ctor_get(v___x_397_, 0);
lean_inc(v_a_398_);
lean_dec_ref_known(v___x_397_, 1);
if (v_isShared_388_ == 0)
{
lean_ctor_set_tag(v___x_387_, 1);
v___x_400_ = v___x_387_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_385_);
v___x_400_ = v_reuseFailAlloc_405_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
lean_object* v___x_401_; 
v___x_401_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_398_, v___x_390_, v___x_400_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_);
lean_dec(v_a_398_);
if (lean_obj_tag(v___x_401_) == 0)
{
lean_object* v_a_402_; lean_object* v___f_403_; lean_object* v___x_404_; 
v_a_402_ = lean_ctor_get(v___x_401_, 0);
lean_inc(v_a_402_);
lean_dec_ref_known(v___x_401_, 1);
v___f_403_ = lean_alloc_closure((void*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__6___boxed), 8, 1);
lean_closure_set(v___f_403_, 0, v_a_402_);
v___x_404_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_403_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_);
return v___x_404_;
}
else
{
return v___x_401_;
}
}
}
else
{
lean_object* v_a_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_413_; 
lean_del_object(v___x_387_);
lean_dec(v_a_385_);
v_a_406_ = lean_ctor_get(v___x_397_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_397_);
if (v_isSharedCheck_413_ == 0)
{
v___x_408_ = v___x_397_;
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_a_406_);
lean_dec(v___x_397_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_411_; 
if (v_isShared_409_ == 0)
{
v___x_411_ = v___x_408_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_a_406_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7___boxed(lean_object* v___f_415_, lean_object* v___f_416_, lean_object* v___f_417_, lean_object* v___f_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___lam__7(v___f_415_, v___f_416_, v___f_417_, v___f_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1(lean_object* v_a_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_){
_start:
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_449_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__4));
v___x_450_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___closed__7));
v___x_451_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_449_, v___x_450_, v_a_442_, v_a_443_, v_a_444_, v_a_445_, v_a_446_, v_a_447_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1___boxed(lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______delab__app__Ordinal__termTypeLT____1(v_a_452_, v_a_453_, v_a_454_, v_a_455_, v_a_456_, v_a_457_);
lean_dec(v_a_457_);
lean_dec_ref(v_a_456_);
lean_dec(v_a_455_);
lean_dec_ref(v_a_454_);
lean_dec(v_a_453_);
lean_dec_ref(v_a_452_);
return v_res_459_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_zero(void){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lean_box(0);
return v___x_460_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_inhabited(void){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lean_box(0);
return v___x_461_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_one(void){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lean_box(0);
return v___x_462_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_instOrderBot(void){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lean_box(0);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_typein___lam__0(lean_object* v_a_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lean_box(0);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_typein___lam__0___boxed(lean_object* v_a_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_Ordinal_typein___lam__0(v_a_470_);
lean_dec(v_a_470_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_typein(lean_object* v_00_u03b1_476_, lean_object* v_r_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Ordinal_typein___closed__1));
return v___x_479_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_wellFoundedRelation(void){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lean_box(0);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_card(lean_object* v_a_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lean_box(0);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_card___boxed(lean_object* v_a_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_mathlib_Ordinal_card(v_a_483_);
lean_dec(v_a_483_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_lift(lean_object* v_o_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lean_box(0);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_lift___boxed(lean_object* v_o_487_){
_start:
{
lean_object* v_res_488_; 
v_res_488_ = lp_mathlib_Ordinal_lift(v_o_487_);
lean_dec(v_o_487_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_liftInitialSeg___lam__0(lean_object* v___y_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lean_box(0);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_liftInitialSeg___lam__0___boxed(lean_object* v___y_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_Ordinal_liftInitialSeg___lam__0(v___y_491_);
lean_dec(v___y_491_);
return v_res_492_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_omega0(void){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lean_box(0);
return v___x_495_;
}
}
static lean_object* _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__1(void){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_509_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__0));
v___x_510_ = l_String_toRawSubstring_x27(v___x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1(lean_object* v_x_521_, lean_object* v_a_522_, lean_object* v_a_523_){
_start:
{
lean_object* v___x_524_; uint8_t v___x_525_; 
v___x_524_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___closed__1));
v___x_525_ = l_Lean_Syntax_isOfKind(v_x_521_, v___x_524_);
if (v___x_525_ == 0)
{
lean_object* v___x_526_; lean_object* v___x_527_; 
v___x_526_ = lean_box(1);
v___x_527_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_527_, 0, v___x_526_);
lean_ctor_set(v___x_527_, 1, v_a_523_);
return v___x_527_;
}
else
{
lean_object* v_quotContext_528_; lean_object* v_currMacroScope_529_; lean_object* v_ref_530_; uint8_t v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; 
v_quotContext_528_ = lean_ctor_get(v_a_522_, 1);
v_currMacroScope_529_ = lean_ctor_get(v_a_522_, 2);
v_ref_530_ = lean_ctor_get(v_a_522_, 5);
v___x_531_ = 0;
v___x_532_ = l_Lean_SourceInfo_fromRef(v_ref_530_, v___x_531_);
v___x_533_ = lean_obj_once(&lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__1, &lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__1_once, _init_lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__1);
v___x_534_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__3));
lean_inc(v_currMacroScope_529_);
lean_inc(v_quotContext_528_);
v___x_535_ = l_Lean_addMacroScope(v_quotContext_528_, v___x_534_, v_currMacroScope_529_);
v___x_536_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___closed__5));
v___x_537_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_537_, 0, v___x_532_);
lean_ctor_set(v___x_537_, 1, v___x_533_);
lean_ctor_set(v___x_537_, 2, v___x_535_);
lean_ctor_set(v___x_537_, 3, v___x_536_);
v___x_538_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_538_, 0, v___x_537_);
lean_ctor_set(v___x_538_, 1, v_a_523_);
return v___x_538_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1___boxed(lean_object* v_x_539_, lean_object* v_a_540_, lean_object* v_a_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______macroRules__Ordinal__term_u03c9__1(v_x_539_, v_a_540_, v_a_541_);
lean_dec_ref(v_a_540_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1(lean_object* v_x_546_, lean_object* v_a_547_, lean_object* v_a_548_){
_start:
{
lean_object* v___x_549_; uint8_t v___x_550_; 
v___x_549_ = ((lean_object*)(lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___closed__1));
lean_inc(v_x_546_);
v___x_550_ = l_Lean_Syntax_isOfKind(v_x_546_, v___x_549_);
if (v___x_550_ == 0)
{
lean_object* v___x_551_; lean_object* v___x_552_; 
lean_dec(v_x_546_);
v___x_551_ = lean_box(0);
v___x_552_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_552_, 0, v___x_551_);
lean_ctor_set(v___x_552_, 1, v_a_548_);
return v___x_552_;
}
else
{
lean_object* v_ref_553_; uint8_t v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v_ref_553_ = l_Lean_replaceRef(v_x_546_, v_a_547_);
lean_dec(v_x_546_);
v___x_554_ = 0;
v___x_555_ = l_Lean_SourceInfo_fromRef(v_ref_553_, v___x_554_);
lean_dec(v_ref_553_);
v___x_556_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___closed__1));
v___x_557_ = ((lean_object*)(lp_mathlib_Ordinal_term_u03c9___closed__2));
lean_inc(v___x_555_);
v___x_558_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_555_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
v___x_559_ = l_Lean_Syntax_node1(v___x_555_, v___x_556_, v___x_558_);
v___x_560_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_560_, 0, v___x_559_);
lean_ctor_set(v___x_560_, 1, v_a_548_);
return v___x_560_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1___boxed(lean_object* v_x_561_, lean_object* v_a_562_, lean_object* v_a_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_Ordinal___aux__Mathlib__SetTheory__Ordinal__Basic______unexpand__Ordinal__omega0__1(v_x_561_, v_a_562_, v_a_563_);
lean_dec(v_a_562_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_add___lam__0(lean_object* v_o_u2081_565_, lean_object* v_o_u2082_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lean_box(0);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_add___lam__0___boxed(lean_object* v_o_u2081_568_, lean_object* v_o_u2082_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Ordinal_add___lam__0(v_o_u2081_568_, v_o_u2082_569_);
lean_dec(v_o_u2082_569_);
lean_dec(v_o_u2081_568_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_instSuccOrder___lam__0(lean_object* v_o_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lean_box(0);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_instSuccOrder___lam__0___boxed(lean_object* v_o_590_){
_start:
{
lean_object* v_res_591_; 
v_res_591_ = lp_mathlib_Ordinal_instSuccOrder___lam__0(v_o_590_);
lean_dec(v_o_590_);
return v_res_591_;
}
}
static lean_object* _init_lp_mathlib_Ordinal_uniqueIioOne(void){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lean_box(0);
return v___x_595_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_IsNormal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Shrink(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_IsNormal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_WellOrder_inhabited = _init_lp_mathlib_WellOrder_inhabited();
lean_mark_persistent(lp_mathlib_WellOrder_inhabited);
lp_mathlib_Ordinal_isEquivalent = _init_lp_mathlib_Ordinal_isEquivalent();
lean_mark_persistent(lp_mathlib_Ordinal_isEquivalent);
lp_mathlib_Ordinal_zero = _init_lp_mathlib_Ordinal_zero();
lean_mark_persistent(lp_mathlib_Ordinal_zero);
lp_mathlib_Ordinal_inhabited = _init_lp_mathlib_Ordinal_inhabited();
lean_mark_persistent(lp_mathlib_Ordinal_inhabited);
lp_mathlib_Ordinal_one = _init_lp_mathlib_Ordinal_one();
lean_mark_persistent(lp_mathlib_Ordinal_one);
lp_mathlib_Ordinal_instOrderBot = _init_lp_mathlib_Ordinal_instOrderBot();
lean_mark_persistent(lp_mathlib_Ordinal_instOrderBot);
lp_mathlib_Ordinal_wellFoundedRelation = _init_lp_mathlib_Ordinal_wellFoundedRelation();
lean_mark_persistent(lp_mathlib_Ordinal_wellFoundedRelation);
lp_mathlib_Ordinal_omega0 = _init_lp_mathlib_Ordinal_omega0();
lean_mark_persistent(lp_mathlib_Ordinal_omega0);
lp_mathlib_Ordinal_uniqueIioOne = _init_lp_mathlib_Ordinal_uniqueIioOne();
lean_mark_persistent(lp_mathlib_Ordinal_uniqueIioOne);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sum_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_IsNormal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Shrink(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sum_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_IsNormal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
