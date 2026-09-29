// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Finprod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Pi public import Mathlib.Algebra.FiniteSupport.Defs public import Mathlib.Algebra.Module.Torsion.Free public import Mathlib.Algebra.Notation.FiniteSupport public import Mathlib.Algebra.Order.Ring.Defs import Mathlib.Algebra.FiniteSupport.Basic import Mathlib.Algebra.Module.End import Mathlib.Algebra.Order.BigOperators.Ring.Finset
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
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_getBinders(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchScoped(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term∑ᶠ_,_"};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(3, 211, 33, 183, 166, 238, 73, 138)}};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__3_value;
static const lean_string_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∑ᶠ"};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__4_value)}};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__5_value;
static lean_once_cell_t lp_mathlib_term_u2211_u1da0___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__6;
static const lean_string_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__7 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__7_value)}};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__8_value;
static lean_once_cell_t lp_mathlib_term_u2211_u1da0___x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__9;
static const lean_string_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__10 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_term_u2211_u1da0___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__11_value),((lean_object*)(((size_t)(67) << 1) | 1))}};
static const lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_term_u2211_u1da0___x2c___00__closed__12_value;
static lean_once_cell_t lp_mathlib_term_u2211_u1da0___x2c___00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__13;
static lean_once_cell_t lp_mathlib_term_u2211_u1da0___x2c___00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2211_u1da0___x2c___00__closed__14;
LEAN_EXPORT lean_object* lp_mathlib_term_u2211_u1da0___x2c__;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Notation3"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termExpand_binders%(_=>_)_,_"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 176, 22, 214, 10, 13, 147, 22)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(120, 7, 237, 26, 3, 243, 131, 214)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "expand_binders%"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__12_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "finsum"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__15_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__16;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(28, 49, 217, 77, 255, 134, 225, 48)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__17_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__18_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__19_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__22_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "∑ᶠ "};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extBinders"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(142, 202, 111, 171, 129, 134, 17, 161)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "extBinderCollection"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(144, 58, 22, 199, 215, 82, 42, 232)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__3_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__4_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__5_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__4_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__5_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u220f_u1da0___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term∏ᶠ_,_"};
static const lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_term_u220f_u1da0___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_u220f_u1da0___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u220f_u1da0___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 150, 169, 31, 114, 8, 64, 196)}};
static const lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_term_u220f_u1da0___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_term_u220f_u1da0___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∏ᶠ"};
static const lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_term_u220f_u1da0___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_u220f_u1da0___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u220f_u1da0___x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_term_u220f_u1da0___x2c___00__closed__3_value;
static lean_once_cell_t lp_mathlib_term_u220f_u1da0___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__4;
static lean_once_cell_t lp_mathlib_term_u220f_u1da0___x2c___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__5;
static lean_once_cell_t lp_mathlib_term_u220f_u1da0___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__6;
static lean_once_cell_t lp_mathlib_term_u220f_u1da0___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u220f_u1da0___x2c___00__closed__7;
LEAN_EXPORT lean_object* lp_mathlib_term_u220f_u1da0___x2c__;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "finprod"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(156, 209, 45, 173, 51, 119, 48, 135)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "∏ᶠ "};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__4_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_10_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_11_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__5));
v___x_12_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__3));
v___x_13_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v___x_11_);
lean_ctor_set(v___x_13_, 2, v___x_10_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__9(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__8));
v___x_18_ = lean_obj_once(&lp_mathlib_term_u2211_u1da0___x2c___00__closed__6, &lp_mathlib_term_u2211_u1da0___x2c___00__closed__6_once, _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__6);
v___x_19_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__3));
v___x_20_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_18_);
lean_ctor_set(v___x_20_, 2, v___x_17_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__13(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_27_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__12));
v___x_28_ = lean_obj_once(&lp_mathlib_term_u2211_u1da0___x2c___00__closed__9, &lp_mathlib_term_u2211_u1da0___x2c___00__closed__9_once, _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__9);
v___x_29_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__3));
v___x_30_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___x_28_);
lean_ctor_set(v___x_30_, 2, v___x_27_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__14(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_31_ = lean_obj_once(&lp_mathlib_term_u2211_u1da0___x2c___00__closed__13, &lp_mathlib_term_u2211_u1da0___x2c___00__closed__13_once, _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__13);
v___x_32_ = lean_unsigned_to_nat(1022u);
v___x_33_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__1));
v___x_34_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v___x_32_);
lean_ctor_set(v___x_34_, 2, v___x_31_);
return v___x_34_;
}
}
static lean_object* _init_lp_mathlib_term_u2211_u1da0___x2c__(void){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_obj_once(&lp_mathlib_term_u2211_u1da0___x2c___00__closed__14, &lp_mathlib_term_u2211_u1da0___x2c___00__closed__14_once, _init_lp_mathlib_term_u2211_u1da0___x2c___00__closed__14);
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__6));
v___x_47_ = l_String_toRawSubstring_x27(v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__16(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_61_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__15));
v___x_62_ = l_String_toRawSubstring_x27(v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1(lean_object* v_x_76_, lean_object* v_a_77_, lean_object* v_a_78_){
_start:
{
lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_79_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__1));
lean_inc(v_x_76_);
v___x_80_ = l_Lean_Syntax_isOfKind(v_x_76_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; lean_object* v___x_82_; 
lean_dec(v_x_76_);
v___x_81_ = lean_box(1);
v___x_82_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v_a_78_);
return v___x_82_;
}
else
{
lean_object* v_quotContext_83_; lean_object* v_currMacroScope_84_; lean_object* v_ref_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; uint8_t v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v_quotContext_83_ = lean_ctor_get(v_a_77_, 1);
v_currMacroScope_84_ = lean_ctor_get(v_a_77_, 2);
v_ref_85_ = lean_ctor_get(v_a_77_, 5);
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = l_Lean_Syntax_getArg(v_x_76_, v___x_86_);
v___x_88_ = lean_unsigned_to_nat(3u);
v___x_89_ = l_Lean_Syntax_getArg(v_x_76_, v___x_88_);
lean_dec(v_x_76_);
v___x_90_ = 0;
v___x_91_ = l_Lean_SourceInfo_fromRef(v_ref_85_, v___x_90_);
v___x_92_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3));
v___x_93_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__4));
lean_inc_n(v___x_91_, 9);
v___x_94_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_91_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__5));
v___x_96_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_91_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v___x_97_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7);
v___x_98_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_84_, 2);
lean_inc_n(v_quotContext_83_, 2);
v___x_99_ = l_Lean_addMacroScope(v_quotContext_83_, v___x_98_, v_currMacroScope_84_);
v___x_100_ = lean_box(0);
v___x_101_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_101_, 0, v___x_91_);
lean_ctor_set(v___x_101_, 1, v___x_97_);
lean_ctor_set(v___x_101_, 2, v___x_99_);
lean_ctor_set(v___x_101_, 3, v___x_100_);
v___x_102_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__9));
v___x_103_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_91_);
lean_ctor_set(v___x_103_, 1, v___x_102_);
v___x_104_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14));
v___x_105_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__16, &lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__16);
v___x_106_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__17));
v___x_107_ = l_Lean_addMacroScope(v_quotContext_83_, v___x_106_, v_currMacroScope_84_);
v___x_108_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__19));
v___x_109_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_109_, 0, v___x_91_);
lean_ctor_set(v___x_109_, 1, v___x_105_);
lean_ctor_set(v___x_109_, 2, v___x_107_);
lean_ctor_set(v___x_109_, 3, v___x_108_);
v___x_110_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21));
lean_inc_ref(v___x_101_);
v___x_111_ = l_Lean_Syntax_node1(v___x_91_, v___x_110_, v___x_101_);
v___x_112_ = l_Lean_Syntax_node2(v___x_91_, v___x_104_, v___x_109_, v___x_111_);
v___x_113_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__22));
v___x_114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_91_);
lean_ctor_set(v___x_114_, 1, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__23));
v___x_116_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_91_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
v___x_117_ = lean_unsigned_to_nat(9u);
v___x_118_ = lean_mk_empty_array_with_capacity(v___x_117_);
v___x_119_ = lean_array_push(v___x_118_, v___x_94_);
v___x_120_ = lean_array_push(v___x_119_, v___x_96_);
v___x_121_ = lean_array_push(v___x_120_, v___x_101_);
v___x_122_ = lean_array_push(v___x_121_, v___x_103_);
v___x_123_ = lean_array_push(v___x_122_, v___x_112_);
v___x_124_ = lean_array_push(v___x_123_, v___x_114_);
v___x_125_ = lean_array_push(v___x_124_, v___x_87_);
v___x_126_ = lean_array_push(v___x_125_, v___x_116_);
v___x_127_ = lean_array_push(v___x_126_, v___x_89_);
v___x_128_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_128_, 0, v___x_91_);
lean_ctor_set(v___x_128_, 1, v___x_92_);
lean_ctor_set(v___x_128_, 2, v___x_127_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v_a_78_);
return v___x_129_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___boxed(lean_object* v_x_130_, lean_object* v_a_131_, lean_object* v_a_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1(v_x_130_, v_a_131_, v_a_132_);
lean_dec_ref(v_a_131_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg(lean_object* v___y_134_){
_start:
{
lean_object* v_subExpr_136_; lean_object* v_expr_137_; lean_object* v___x_138_; 
v_subExpr_136_ = lean_ctor_get(v___y_134_, 3);
v_expr_137_ = lean_ctor_get(v_subExpr_136_, 0);
lean_inc_ref(v_expr_137_);
v___x_138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_138_, 0, v_expr_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg___boxed(lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg(v___y_139_);
lean_dec_ref(v___y_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0(lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg(v___y_142_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___boxed(lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0(v___y_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
lean_dec(v___y_153_);
lean_dec_ref(v___y_152_);
lean_dec(v___y_151_);
lean_dec_ref(v___y_150_);
return v_res_157_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__0(lean_object* v_x_158_){
_start:
{
lean_object* v___x_159_; uint8_t v___x_160_; 
v___x_159_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__17));
v___x_160_ = l_Lean_Expr_isConstOf(v_x_158_, v___x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__0___boxed(lean_object* v_x_161_){
_start:
{
uint8_t v_res_162_; lean_object* v_r_163_; 
v_res_162_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__0(v_x_161_);
lean_dec_ref(v_x_161_);
v_r_163_ = lean_box(v_res_162_);
return v_r_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__1(lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_172_, 0, v___y_164_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__1___boxed(lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__1(v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2(uint8_t v___x_183_, lean_object* v___x_184_, lean_object* v_a_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_ref_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
v_ref_193_ = lean_ctor_get(v___y_190_, 5);
v___x_194_ = l_Lean_SourceInfo_fromRef(v_ref_193_, v___x_183_);
v___x_195_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__1));
v___x_196_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_194_, 2);
v___x_197_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_194_);
lean_ctor_set(v___x_197_, 1, v___x_196_);
v___x_198_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__7));
v___x_199_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_194_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v___x_200_ = l_Lean_Syntax_node4(v___x_194_, v___x_195_, v___x_197_, v___x_184_, v___x_199_, v_a_185_);
v___x_201_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___boxed(lean_object* v___x_202_, lean_object* v___x_203_, lean_object* v_a_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
uint8_t v___x_6998__boxed_212_; lean_object* v_res_213_; 
v___x_6998__boxed_212_ = lean_unbox(v___x_202_);
v_res_213_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2(v___x_6998__boxed_212_, v___x_203_, v_a_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_);
lean_dec(v___y_210_);
lean_dec_ref(v___y_209_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
return v_res_213_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6(void){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = l_Array_mkArray0(lean_box(0));
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3(lean_object* v___f_232_, lean_object* v___f_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v___x_241_; lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_289_; 
v___x_241_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg(v___y_234_);
v_a_242_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_289_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_289_ == 0)
{
v___x_244_ = v___x_241_;
v_isShared_245_ = v_isSharedCheck_289_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_241_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_289_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_246_; lean_object* v___y_248_; lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_246_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__1));
v___x_278_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_279_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_246_, v___x_278_, v___y_234_, v___y_236_);
if (lean_obj_tag(v___x_279_) == 0)
{
lean_object* v_a_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v_a_280_ = lean_ctor_get(v___x_279_, 0);
lean_inc(v_a_280_);
lean_dec_ref_known(v___x_279_, 1);
v___x_281_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_281_, 0, v___f_232_);
lean_inc_ref_n(v___f_233_, 2);
v___x_282_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_282_, 0, v___x_281_);
lean_closure_set(v___x_282_, 1, v___f_233_);
v___x_283_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_283_, 0, v___x_282_);
lean_closure_set(v___x_283_, 1, v___f_233_);
v___x_284_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8));
v___x_285_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_285_, 0, v___x_283_);
lean_closure_set(v___x_285_, 1, v___f_233_);
v___x_286_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__9));
v___x_287_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_287_, 0, v___x_285_);
lean_closure_set(v___x_287_, 1, v___x_286_);
v___x_288_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_246_, v___x_284_, v___x_287_, v_a_280_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
v___y_248_ = v___x_288_;
goto v___jp_247_;
}
else
{
lean_dec_ref(v___f_233_);
lean_dec_ref(v___f_232_);
v___y_248_ = v___x_279_;
goto v___jp_247_;
}
v___jp_247_:
{
if (lean_obj_tag(v___y_248_) == 0)
{
lean_object* v_a_249_; lean_object* v_ref_250_; lean_object* v___x_252_; 
v_a_249_ = lean_ctor_get(v___y_248_, 0);
lean_inc(v_a_249_);
lean_dec_ref_known(v___y_248_, 1);
v_ref_250_ = lean_ctor_get(v___y_238_, 5);
if (v_isShared_245_ == 0)
{
lean_ctor_set_tag(v___x_244_, 1);
v___x_252_ = v___x_244_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_a_242_);
v___x_252_ = v_reuseFailAlloc_269_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_249_, v___x_246_, v___x_252_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
if (lean_obj_tag(v___x_253_) == 0)
{
lean_object* v_a_254_; uint8_t v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___f_267_; lean_object* v___x_268_; 
v_a_254_ = lean_ctor_get(v___x_253_, 0);
lean_inc(v_a_254_);
lean_dec_ref_known(v___x_253_, 1);
v___x_255_ = 0;
v___x_256_ = l_Lean_SourceInfo_fromRef(v_ref_250_, v___x_255_);
v___x_257_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5));
v___x_258_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21));
v___x_259_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6);
v___x_260_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_249_);
lean_dec(v_a_249_);
v___x_261_ = l_Array_append___redArg(v___x_259_, v___x_260_);
lean_dec_ref(v___x_260_);
lean_inc_n(v___x_256_, 2);
v___x_262_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_262_, 0, v___x_256_);
lean_ctor_set(v___x_262_, 1, v___x_258_);
lean_ctor_set(v___x_262_, 2, v___x_261_);
v___x_263_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8));
v___x_264_ = l_Lean_Syntax_node1(v___x_256_, v___x_263_, v___x_262_);
v___x_265_ = l_Lean_Syntax_node1(v___x_256_, v___x_257_, v___x_264_);
v___x_266_ = lean_box(v___x_255_);
v___f_267_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_267_, 0, v___x_266_);
lean_closure_set(v___f_267_, 1, v___x_265_);
lean_closure_set(v___f_267_, 2, v_a_254_);
v___x_268_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_267_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
return v___x_268_;
}
else
{
lean_dec(v_a_249_);
return v___x_253_;
}
}
}
else
{
lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_277_; 
lean_del_object(v___x_244_);
lean_dec(v_a_242_);
v_a_270_ = lean_ctor_get(v___y_248_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v___y_248_);
if (v_isSharedCheck_277_ == 0)
{
v___x_272_ = v___y_248_;
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___y_248_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_275_; 
if (v_isShared_273_ == 0)
{
v___x_275_ = v___x_272_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v_a_270_);
v___x_275_ = v_reuseFailAlloc_276_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
return v___x_275_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___boxed(lean_object* v___f_290_, lean_object* v___f_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3(v___f_290_, v___f_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_);
lean_dec(v___y_297_);
lean_dec_ref(v___y_296_);
lean_dec(v___y_295_);
lean_dec_ref(v___y_294_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1(lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_, lean_object* v_a_318_){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_320_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__3));
v___x_321_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__6));
v___x_322_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_320_, v___x_321_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_, v_a_318_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___boxed(lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_, lean_object* v_a_327_, lean_object* v_a_328_, lean_object* v_a_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1(v_a_323_, v_a_324_, v_a_325_, v_a_326_, v_a_327_, v_a_328_);
lean_dec(v_a_328_);
lean_dec_ref(v_a_327_);
lean_dec(v_a_326_);
lean_dec_ref(v_a_325_);
lean_dec(v_a_324_);
lean_dec_ref(v_a_323_);
return v_res_330_;
}
}
static lean_object* _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_337_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_338_ = ((lean_object*)(lp_mathlib_term_u220f_u1da0___x2c___00__closed__3));
v___x_339_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__3));
v___x_340_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
lean_ctor_set(v___x_340_, 1, v___x_338_);
lean_ctor_set(v___x_340_, 2, v___x_337_);
return v___x_340_;
}
}
static lean_object* _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__5(void){
_start:
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_341_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__8));
v___x_342_ = lean_obj_once(&lp_mathlib_term_u220f_u1da0___x2c___00__closed__4, &lp_mathlib_term_u220f_u1da0___x2c___00__closed__4_once, _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__4);
v___x_343_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__3));
v___x_344_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_344_, 0, v___x_343_);
lean_ctor_set(v___x_344_, 1, v___x_342_);
lean_ctor_set(v___x_344_, 2, v___x_341_);
return v___x_344_;
}
}
static lean_object* _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_345_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__12));
v___x_346_ = lean_obj_once(&lp_mathlib_term_u220f_u1da0___x2c___00__closed__5, &lp_mathlib_term_u220f_u1da0___x2c___00__closed__5_once, _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__5);
v___x_347_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__3));
v___x_348_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v___x_346_);
lean_ctor_set(v___x_348_, 2, v___x_345_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_349_ = lean_obj_once(&lp_mathlib_term_u220f_u1da0___x2c___00__closed__6, &lp_mathlib_term_u220f_u1da0___x2c___00__closed__6_once, _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__6);
v___x_350_ = lean_unsigned_to_nat(1022u);
v___x_351_ = ((lean_object*)(lp_mathlib_term_u220f_u1da0___x2c___00__closed__1));
v___x_352_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v___x_350_);
lean_ctor_set(v___x_352_, 2, v___x_349_);
return v___x_352_;
}
}
static lean_object* _init_lp_mathlib_term_u220f_u1da0___x2c__(void){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lean_obj_once(&lp_mathlib_term_u220f_u1da0___x2c___00__closed__7, &lp_mathlib_term_u220f_u1da0___x2c___00__closed__7_once, _init_lp_mathlib_term_u220f_u1da0___x2c___00__closed__7);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__1(void){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_355_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__0));
v___x_356_ = l_String_toRawSubstring_x27(v___x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1(lean_object* v_x_365_, lean_object* v_a_366_, lean_object* v_a_367_){
_start:
{
lean_object* v___x_368_; uint8_t v___x_369_; 
v___x_368_ = ((lean_object*)(lp_mathlib_term_u220f_u1da0___x2c___00__closed__1));
lean_inc(v_x_365_);
v___x_369_ = l_Lean_Syntax_isOfKind(v_x_365_, v___x_368_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; lean_object* v___x_371_; 
lean_dec(v_x_365_);
v___x_370_ = lean_box(1);
v___x_371_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v_a_367_);
return v___x_371_;
}
else
{
lean_object* v_quotContext_372_; lean_object* v_currMacroScope_373_; lean_object* v_ref_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; uint8_t v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v_quotContext_372_ = lean_ctor_get(v_a_366_, 1);
v_currMacroScope_373_ = lean_ctor_get(v_a_366_, 2);
v_ref_374_ = lean_ctor_get(v_a_366_, 5);
v___x_375_ = lean_unsigned_to_nat(1u);
v___x_376_ = l_Lean_Syntax_getArg(v_x_365_, v___x_375_);
v___x_377_ = lean_unsigned_to_nat(3u);
v___x_378_ = l_Lean_Syntax_getArg(v_x_365_, v___x_377_);
lean_dec(v_x_365_);
v___x_379_ = 0;
v___x_380_ = l_Lean_SourceInfo_fromRef(v_ref_374_, v___x_379_);
v___x_381_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__3));
v___x_382_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__4));
lean_inc_n(v___x_380_, 9);
v___x_383_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_383_, 0, v___x_380_);
lean_ctor_set(v___x_383_, 1, v___x_382_);
v___x_384_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__5));
v___x_385_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_385_, 0, v___x_380_);
lean_ctor_set(v___x_385_, 1, v___x_384_);
v___x_386_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__7);
v___x_387_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_373_, 2);
lean_inc_n(v_quotContext_372_, 2);
v___x_388_ = l_Lean_addMacroScope(v_quotContext_372_, v___x_387_, v_currMacroScope_373_);
v___x_389_ = lean_box(0);
v___x_390_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_390_, 0, v___x_380_);
lean_ctor_set(v___x_390_, 1, v___x_386_);
lean_ctor_set(v___x_390_, 2, v___x_388_);
lean_ctor_set(v___x_390_, 3, v___x_389_);
v___x_391_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__9));
v___x_392_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_380_);
lean_ctor_set(v___x_392_, 1, v___x_391_);
v___x_393_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__14));
v___x_394_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__1);
v___x_395_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__2));
v___x_396_ = l_Lean_addMacroScope(v_quotContext_372_, v___x_395_, v_currMacroScope_373_);
v___x_397_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__4));
v___x_398_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_398_, 0, v___x_380_);
lean_ctor_set(v___x_398_, 1, v___x_394_);
lean_ctor_set(v___x_398_, 2, v___x_396_);
lean_ctor_set(v___x_398_, 3, v___x_397_);
v___x_399_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21));
lean_inc_ref(v___x_390_);
v___x_400_ = l_Lean_Syntax_node1(v___x_380_, v___x_399_, v___x_390_);
v___x_401_ = l_Lean_Syntax_node2(v___x_380_, v___x_393_, v___x_398_, v___x_400_);
v___x_402_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__22));
v___x_403_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_380_);
lean_ctor_set(v___x_403_, 1, v___x_402_);
v___x_404_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__23));
v___x_405_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_380_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
v___x_406_ = lean_unsigned_to_nat(9u);
v___x_407_ = lean_mk_empty_array_with_capacity(v___x_406_);
v___x_408_ = lean_array_push(v___x_407_, v___x_383_);
v___x_409_ = lean_array_push(v___x_408_, v___x_385_);
v___x_410_ = lean_array_push(v___x_409_, v___x_390_);
v___x_411_ = lean_array_push(v___x_410_, v___x_392_);
v___x_412_ = lean_array_push(v___x_411_, v___x_401_);
v___x_413_ = lean_array_push(v___x_412_, v___x_403_);
v___x_414_ = lean_array_push(v___x_413_, v___x_376_);
v___x_415_ = lean_array_push(v___x_414_, v___x_405_);
v___x_416_ = lean_array_push(v___x_415_, v___x_378_);
v___x_417_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_417_, 0, v___x_380_);
lean_ctor_set(v___x_417_, 1, v___x_381_);
lean_ctor_set(v___x_417_, 2, v___x_416_);
v___x_418_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_418_, 0, v___x_417_);
lean_ctor_set(v___x_418_, 1, v_a_367_);
return v___x_418_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___boxed(lean_object* v_x_419_, lean_object* v_a_420_, lean_object* v_a_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1(v_x_419_, v_a_420_, v_a_421_);
lean_dec_ref(v_a_420_);
return v_res_422_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__0(lean_object* v_x_423_){
_start:
{
lean_object* v___x_424_; uint8_t v___x_425_; 
v___x_424_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u220f_u1da0___x2c____1___closed__2));
v___x_425_ = l_Lean_Expr_isConstOf(v_x_423_, v___x_424_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__0___boxed(lean_object* v_x_426_){
_start:
{
uint8_t v_res_427_; lean_object* v_r_428_; 
v_res_427_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__0(v_x_426_);
lean_dec_ref(v_x_426_);
v_r_428_ = lean_box(v_res_427_);
return v_r_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2(uint8_t v___x_430_, lean_object* v___x_431_, lean_object* v_a_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_){
_start:
{
lean_object* v_ref_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v_ref_440_ = lean_ctor_get(v___y_437_, 5);
v___x_441_ = l_Lean_SourceInfo_fromRef(v_ref_440_, v___x_430_);
v___x_442_ = ((lean_object*)(lp_mathlib_term_u220f_u1da0___x2c___00__closed__1));
v___x_443_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_441_, 2);
v___x_444_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_441_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
v___x_445_ = ((lean_object*)(lp_mathlib_term_u2211_u1da0___x2c___00__closed__7));
v___x_446_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_446_, 0, v___x_441_);
lean_ctor_set(v___x_446_, 1, v___x_445_);
v___x_447_ = l_Lean_Syntax_node4(v___x_441_, v___x_442_, v___x_444_, v___x_431_, v___x_446_, v_a_432_);
v___x_448_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_448_, 0, v___x_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___boxed(lean_object* v___x_449_, lean_object* v___x_450_, lean_object* v_a_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
uint8_t v___x_6595__boxed_459_; lean_object* v_res_460_; 
v___x_6595__boxed_459_ = lean_unbox(v___x_449_);
v_res_460_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2(v___x_6595__boxed_459_, v___x_450_, v_a_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_);
lean_dec(v___y_457_);
lean_dec_ref(v___y_456_);
lean_dec(v___y_455_);
lean_dec_ref(v___y_454_);
lean_dec(v___y_453_);
lean_dec_ref(v___y_452_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__1(lean_object* v___f_461_, lean_object* v___f_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_470_; lean_object* v_a_471_; lean_object* v___x_473_; uint8_t v_isShared_474_; uint8_t v_isSharedCheck_518_; 
v___x_470_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1_spec__0___redArg(v___y_463_);
v_a_471_ = lean_ctor_get(v___x_470_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_470_);
if (v_isSharedCheck_518_ == 0)
{
v___x_473_ = v___x_470_;
v_isShared_474_ = v_isSharedCheck_518_;
goto v_resetjp_472_;
}
else
{
lean_inc(v_a_471_);
lean_dec(v___x_470_);
v___x_473_ = lean_box(0);
v_isShared_474_ = v_isSharedCheck_518_;
goto v_resetjp_472_;
}
v_resetjp_472_:
{
lean_object* v___x_475_; lean_object* v___y_477_; lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_475_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__1));
v___x_507_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_508_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_475_, v___x_507_, v___y_463_, v___y_465_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v_a_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v_a_509_ = lean_ctor_get(v___x_508_, 0);
lean_inc(v_a_509_);
lean_dec_ref_known(v___x_508_, 1);
v___x_510_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_510_, 0, v___f_461_);
lean_inc_ref_n(v___f_462_, 2);
v___x_511_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_511_, 0, v___x_510_);
lean_closure_set(v___x_511_, 1, v___f_462_);
v___x_512_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_512_, 0, v___x_511_);
lean_closure_set(v___x_512_, 1, v___f_462_);
v___x_513_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__8));
v___x_514_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_514_, 0, v___x_512_);
lean_closure_set(v___x_514_, 1, v___f_462_);
v___x_515_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__9));
v___x_516_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_516_, 0, v___x_514_);
lean_closure_set(v___x_516_, 1, v___x_515_);
v___x_517_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_475_, v___x_513_, v___x_516_, v_a_509_, v___y_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
v___y_477_ = v___x_517_;
goto v___jp_476_;
}
else
{
lean_dec_ref(v___f_462_);
lean_dec_ref(v___f_461_);
v___y_477_ = v___x_508_;
goto v___jp_476_;
}
v___jp_476_:
{
if (lean_obj_tag(v___y_477_) == 0)
{
lean_object* v_a_478_; lean_object* v_ref_479_; lean_object* v___x_481_; 
v_a_478_ = lean_ctor_get(v___y_477_, 0);
lean_inc(v_a_478_);
lean_dec_ref_known(v___y_477_, 1);
v_ref_479_ = lean_ctor_get(v___y_467_, 5);
if (v_isShared_474_ == 0)
{
lean_ctor_set_tag(v___x_473_, 1);
v___x_481_ = v___x_473_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v_a_471_);
v___x_481_ = v_reuseFailAlloc_498_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_478_, v___x_475_, v___x_481_, v___y_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
if (lean_obj_tag(v___x_482_) == 0)
{
lean_object* v_a_483_; uint8_t v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___f_496_; lean_object* v___x_497_; 
v_a_483_ = lean_ctor_get(v___x_482_, 0);
lean_inc(v_a_483_);
lean_dec_ref_known(v___x_482_, 1);
v___x_484_ = 0;
v___x_485_ = l_Lean_SourceInfo_fromRef(v_ref_479_, v___x_484_);
v___x_486_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__5));
v___x_487_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______macroRules__term_u2211_u1da0___x2c____1___closed__21));
v___x_488_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__6);
v___x_489_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_478_);
lean_dec(v_a_478_);
v___x_490_ = l_Array_append___redArg(v___x_488_, v___x_489_);
lean_dec_ref(v___x_489_);
lean_inc_n(v___x_485_, 2);
v___x_491_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_491_, 0, v___x_485_);
lean_ctor_set(v___x_491_, 1, v___x_487_);
lean_ctor_set(v___x_491_, 2, v___x_490_);
v___x_492_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___lam__3___closed__8));
v___x_493_ = l_Lean_Syntax_node1(v___x_485_, v___x_492_, v___x_491_);
v___x_494_ = l_Lean_Syntax_node1(v___x_485_, v___x_486_, v___x_493_);
v___x_495_ = lean_box(v___x_484_);
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_496_, 0, v___x_495_);
lean_closure_set(v___f_496_, 1, v___x_494_);
lean_closure_set(v___f_496_, 2, v_a_483_);
v___x_497_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_496_, v___y_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
return v___x_497_;
}
else
{
lean_dec(v_a_478_);
return v___x_482_;
}
}
}
else
{
lean_object* v_a_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_506_; 
lean_del_object(v___x_473_);
lean_dec(v_a_471_);
v_a_499_ = lean_ctor_get(v___y_477_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___y_477_);
if (v_isSharedCheck_506_ == 0)
{
v___x_501_ = v___y_477_;
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_a_499_);
lean_dec(v___y_477_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
lean_object* v___x_504_; 
if (v_isShared_502_ == 0)
{
v___x_504_ = v___x_501_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_a_499_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
return v___x_504_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__1___boxed(lean_object* v___f_519_, lean_object* v___f_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___lam__1(v___f_519_, v___f_520_, v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
lean_dec(v___y_524_);
lean_dec_ref(v___y_523_);
lean_dec(v___y_522_);
lean_dec_ref(v___y_521_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1(lean_object* v_a_539_, lean_object* v_a_540_, lean_object* v_a_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_){
_start:
{
lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_546_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u2211_u1da0___x2c____1___closed__3));
v___x_547_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___closed__3));
v___x_548_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_546_, v___x_547_, v_a_539_, v_a_540_, v_a_541_, v_a_542_, v_a_543_, v_a_544_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1___boxed(lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_, lean_object* v_a_555_){
_start:
{
lean_object* v_res_556_; 
v_res_556_ = lp_mathlib___aux__Mathlib__Algebra__BigOperators__Finprod______delab__app__term_u220f_u1da0___x2c____1(v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_, v_a_554_);
lean_dec(v_a_554_);
lean_dec_ref(v_a_553_);
lean_dec(v_a_552_);
lean_dec_ref(v_a_551_);
lean_dec(v_a_550_);
lean_dec_ref(v_a_549_);
return v_res_556_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_FiniteSupport(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_FiniteSupport(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_term_u2211_u1da0___x2c__ = _init_lp_mathlib_term_u2211_u1da0___x2c__();
lean_mark_persistent(lp_mathlib_term_u2211_u1da0___x2c__);
lp_mathlib_term_u220f_u1da0___x2c__ = _init_lp_mathlib_term_u220f_u1da0___x2c__();
lean_mark_persistent(lp_mathlib_term_u220f_u1da0___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_FiniteSupport(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_FiniteSupport_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_FiniteSupport(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FiniteSupport_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(builtin);
}
#ifdef __cplusplus
}
#endif
