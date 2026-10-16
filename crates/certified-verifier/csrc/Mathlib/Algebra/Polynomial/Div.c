// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Div
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.IsField public import Mathlib.Algebra.Polynomial.Inductions public import Mathlib.Algebra.Polynomial.Monic public import Mathlib.Order.Lattice.Nat public import Mathlib.RingTheory.Multiplicity
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Polynomial"};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0_value;
static const lean_string_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_/ₘ_"};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__1 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(63, 221, 152, 109, 229, 51, 169, 47)}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2_value;
static const lean_string_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__3 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__4 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__4_value;
static const lean_string_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " /ₘ "};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__5 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__5_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__6 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__6_value;
static const lean_string_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__7 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__8 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__8_value),((lean_object*)(((size_t)(71) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__9 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__4_value),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__6_value),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__9_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__10 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x2f_u2098___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__10_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x2f_u2098___00__closed__11 = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Polynomial_term___x2f_u2098__ = (const lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__11_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__0 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__0_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__1 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__1_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__2 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__2_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__3 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "divByMonic"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__5 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__6;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(215, 78, 111, 218, 26, 20, 43, 1)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__7 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(37, 29, 106, 208, 74, 35, 112, 102)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__8 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__9 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__10 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__10_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__11 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__12 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__0 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__1 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_%ₘ_"};
static const lean_object* lp_mathlib_Polynomial_term___x25_u2098___00__closed__0 = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 57, 97, 191, 26, 207, 97, 217)}};
static const lean_object* lp_mathlib_Polynomial_term___x25_u2098___00__closed__1 = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__1_value;
static const lean_string_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " %ₘ "};
static const lean_object* lp_mathlib_Polynomial_term___x25_u2098___00__closed__2 = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__2_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x25_u2098___00__closed__3 = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__4_value),((lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__3_value),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__9_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x25_u2098___00__closed__4 = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Polynomial_term___x25_u2098___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__1_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__4_value)}};
static const lean_object* lp_mathlib_Polynomial_term___x25_u2098___00__closed__5 = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Polynomial_term___x25_u2098__ = (const lean_object*)&lp_mathlib_Polynomial_term___x25_u2098___00__closed__5_value;
static const lean_string_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "modByMonic"};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__0 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__1;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 72, 255, 184, 62, 107, 23, 15)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__2 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Polynomial_term___x2f_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 51, 58, 39, 110, 26, 18, 160)}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__3 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__4 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__5 = (const lean_object*)&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__modByMonic__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__modByMonic__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter___redArg(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_h__1_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_3(v_h__1_3_, v_x_1_, v_x_2_, lean_box(0));
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter(lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_motive_7_, lean_object* v_x_8_, lean_object* v_x_9_, lean_object* v_x_10_, lean_object* v_h__1_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_3(v_h__1_11_, v_x_8_, v_x_9_, lean_box(0));
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter___boxed(lean_object* v_R_13_, lean_object* v_inst_14_, lean_object* v_motive_15_, lean_object* v_x_16_, lean_object* v_x_17_, lean_object* v_x_18_, lean_object* v_h__1_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib___private_Mathlib_Algebra_Polynomial_Div_0__Polynomial_divModByMonicAux_match__1_splitter(v_R_13_, v_inst_14_, v_motive_15_, v_x_16_, v_x_17_, v_x_18_, v_h__1_19_);
lean_dec_ref(v_inst_14_);
return v_res_20_;
}
}
static lean_object* _init_lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__6(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__5));
v___x_58_ = l_String_toRawSubstring_x27(v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1(lean_object* v_x_73_, lean_object* v_a_74_, lean_object* v_a_75_){
_start:
{
lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_76_ = ((lean_object*)(lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2));
lean_inc(v_x_73_);
v___x_77_ = l_Lean_Syntax_isOfKind(v_x_73_, v___x_76_);
if (v___x_77_ == 0)
{
lean_object* v___x_78_; lean_object* v___x_79_; 
lean_dec(v_x_73_);
v___x_78_ = lean_box(1);
v___x_79_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_75_);
return v___x_79_;
}
else
{
lean_object* v_quotContext_80_; lean_object* v_currMacroScope_81_; lean_object* v_ref_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; uint8_t v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v_quotContext_80_ = lean_ctor_get(v_a_74_, 1);
v_currMacroScope_81_ = lean_ctor_get(v_a_74_, 2);
v_ref_82_ = lean_ctor_get(v_a_74_, 5);
v___x_83_ = lean_unsigned_to_nat(0u);
v___x_84_ = l_Lean_Syntax_getArg(v_x_73_, v___x_83_);
v___x_85_ = lean_unsigned_to_nat(2u);
v___x_86_ = l_Lean_Syntax_getArg(v_x_73_, v___x_85_);
lean_dec(v_x_73_);
v___x_87_ = 0;
v___x_88_ = l_Lean_SourceInfo_fromRef(v_ref_82_, v___x_87_);
v___x_89_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4));
v___x_90_ = lean_obj_once(&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__6, &lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__6_once, _init_lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__6);
v___x_91_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__7));
lean_inc(v_currMacroScope_81_);
lean_inc(v_quotContext_80_);
v___x_92_ = l_Lean_addMacroScope(v_quotContext_80_, v___x_91_, v_currMacroScope_81_);
v___x_93_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__10));
lean_inc_n(v___x_88_, 2);
v___x_94_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_94_, 0, v___x_88_);
lean_ctor_set(v___x_94_, 1, v___x_90_);
lean_ctor_set(v___x_94_, 2, v___x_92_);
lean_ctor_set(v___x_94_, 3, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__12));
v___x_96_ = l_Lean_Syntax_node2(v___x_88_, v___x_95_, v___x_84_, v___x_86_);
v___x_97_ = l_Lean_Syntax_node2(v___x_88_, v___x_89_, v___x_94_, v___x_96_);
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_75_);
return v___x_98_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___boxed(lean_object* v_x_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1(v_x_99_, v_a_100_, v_a_101_);
lean_dec_ref(v_a_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1(lean_object* v_x_106_, lean_object* v_a_107_, lean_object* v_a_108_){
_start:
{
lean_object* v___x_109_; uint8_t v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4));
lean_inc(v_x_106_);
v___x_110_ = l_Lean_Syntax_isOfKind(v_x_106_, v___x_109_);
if (v___x_110_ == 0)
{
lean_object* v___x_111_; lean_object* v___x_112_; 
lean_dec(v_x_106_);
v___x_111_ = lean_box(0);
v___x_112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_108_);
return v___x_112_;
}
else
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_113_ = lean_unsigned_to_nat(0u);
v___x_114_ = l_Lean_Syntax_getArg(v_x_106_, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__1));
lean_inc(v___x_114_);
v___x_116_ = l_Lean_Syntax_isOfKind(v___x_114_, v___x_115_);
if (v___x_116_ == 0)
{
lean_object* v___x_117_; lean_object* v___x_118_; 
lean_dec(v___x_114_);
lean_dec(v_x_106_);
v___x_117_ = lean_box(0);
v___x_118_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v_a_108_);
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_119_ = lean_unsigned_to_nat(1u);
v___x_120_ = l_Lean_Syntax_getArg(v_x_106_, v___x_119_);
lean_dec(v_x_106_);
v___x_121_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_120_);
v___x_122_ = l_Lean_Syntax_matchesNull(v___x_120_, v___x_121_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; lean_object* v___x_124_; 
lean_dec(v___x_120_);
lean_dec(v___x_114_);
v___x_123_ = lean_box(0);
v___x_124_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v_a_108_);
return v___x_124_;
}
else
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v_ref_127_; uint8_t v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_125_ = l_Lean_Syntax_getArg(v___x_120_, v___x_113_);
v___x_126_ = l_Lean_Syntax_getArg(v___x_120_, v___x_119_);
lean_dec(v___x_120_);
v_ref_127_ = l_Lean_replaceRef(v___x_114_, v_a_107_);
lean_dec(v___x_114_);
v___x_128_ = 0;
v___x_129_ = l_Lean_SourceInfo_fromRef(v_ref_127_, v___x_128_);
lean_dec(v_ref_127_);
v___x_130_ = ((lean_object*)(lp_mathlib_Polynomial_term___x2f_u2098___00__closed__2));
v___x_131_ = ((lean_object*)(lp_mathlib_Polynomial_term___x2f_u2098___00__closed__5));
lean_inc(v___x_129_);
v___x_132_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_129_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
v___x_133_ = l_Lean_Syntax_node3(v___x_129_, v___x_130_, v___x_125_, v___x_132_, v___x_126_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_108_);
return v___x_134_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___boxed(lean_object* v_x_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1(v_x_135_, v_a_136_, v_a_137_);
lean_dec(v_a_136_);
return v_res_138_;
}
}
static lean_object* _init_lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__1(void){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__0));
v___x_157_ = l_String_toRawSubstring_x27(v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1(lean_object* v_x_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
lean_object* v___x_172_; uint8_t v___x_173_; 
v___x_172_ = ((lean_object*)(lp_mathlib_Polynomial_term___x25_u2098___00__closed__1));
lean_inc(v_x_169_);
v___x_173_ = l_Lean_Syntax_isOfKind(v_x_169_, v___x_172_);
if (v___x_173_ == 0)
{
lean_object* v___x_174_; lean_object* v___x_175_; 
lean_dec(v_x_169_);
v___x_174_ = lean_box(1);
v___x_175_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set(v___x_175_, 1, v_a_171_);
return v___x_175_;
}
else
{
lean_object* v_quotContext_176_; lean_object* v_currMacroScope_177_; lean_object* v_ref_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; uint8_t v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v_quotContext_176_ = lean_ctor_get(v_a_170_, 1);
v_currMacroScope_177_ = lean_ctor_get(v_a_170_, 2);
v_ref_178_ = lean_ctor_get(v_a_170_, 5);
v___x_179_ = lean_unsigned_to_nat(0u);
v___x_180_ = l_Lean_Syntax_getArg(v_x_169_, v___x_179_);
v___x_181_ = lean_unsigned_to_nat(2u);
v___x_182_ = l_Lean_Syntax_getArg(v_x_169_, v___x_181_);
lean_dec(v_x_169_);
v___x_183_ = 0;
v___x_184_ = l_Lean_SourceInfo_fromRef(v_ref_178_, v___x_183_);
v___x_185_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4));
v___x_186_ = lean_obj_once(&lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__1, &lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__1_once, _init_lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__1);
v___x_187_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__2));
lean_inc(v_currMacroScope_177_);
lean_inc(v_quotContext_176_);
v___x_188_ = l_Lean_addMacroScope(v_quotContext_176_, v___x_187_, v_currMacroScope_177_);
v___x_189_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___closed__5));
lean_inc_n(v___x_184_, 2);
v___x_190_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_190_, 0, v___x_184_);
lean_ctor_set(v___x_190_, 1, v___x_186_);
lean_ctor_set(v___x_190_, 2, v___x_188_);
lean_ctor_set(v___x_190_, 3, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__12));
v___x_192_ = l_Lean_Syntax_node2(v___x_184_, v___x_191_, v___x_180_, v___x_182_);
v___x_193_ = l_Lean_Syntax_node2(v___x_184_, v___x_185_, v___x_190_, v___x_192_);
v___x_194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_193_);
lean_ctor_set(v___x_194_, 1, v_a_171_);
return v___x_194_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1___boxed(lean_object* v_x_195_, lean_object* v_a_196_, lean_object* v_a_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x25_u2098____1(v_x_195_, v_a_196_, v_a_197_);
lean_dec_ref(v_a_196_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__modByMonic__1(lean_object* v_x_199_, lean_object* v_a_200_, lean_object* v_a_201_){
_start:
{
lean_object* v___x_202_; uint8_t v___x_203_; 
v___x_202_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______macroRules__Polynomial__term___x2f_u2098____1___closed__4));
lean_inc(v_x_199_);
v___x_203_ = l_Lean_Syntax_isOfKind(v_x_199_, v___x_202_);
if (v___x_203_ == 0)
{
lean_object* v___x_204_; lean_object* v___x_205_; 
lean_dec(v_x_199_);
v___x_204_ = lean_box(0);
v___x_205_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
lean_ctor_set(v___x_205_, 1, v_a_201_);
return v___x_205_;
}
else
{
lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; uint8_t v___x_209_; 
v___x_206_ = lean_unsigned_to_nat(0u);
v___x_207_ = l_Lean_Syntax_getArg(v_x_199_, v___x_206_);
v___x_208_ = ((lean_object*)(lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__divByMonic__1___closed__1));
lean_inc(v___x_207_);
v___x_209_ = l_Lean_Syntax_isOfKind(v___x_207_, v___x_208_);
if (v___x_209_ == 0)
{
lean_object* v___x_210_; lean_object* v___x_211_; 
lean_dec(v___x_207_);
lean_dec(v_x_199_);
v___x_210_ = lean_box(0);
v___x_211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v_a_201_);
return v___x_211_;
}
else
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; 
v___x_212_ = lean_unsigned_to_nat(1u);
v___x_213_ = l_Lean_Syntax_getArg(v_x_199_, v___x_212_);
lean_dec(v_x_199_);
v___x_214_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_213_);
v___x_215_ = l_Lean_Syntax_matchesNull(v___x_213_, v___x_214_);
if (v___x_215_ == 0)
{
lean_object* v___x_216_; lean_object* v___x_217_; 
lean_dec(v___x_213_);
lean_dec(v___x_207_);
v___x_216_ = lean_box(0);
v___x_217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_216_);
lean_ctor_set(v___x_217_, 1, v_a_201_);
return v___x_217_;
}
else
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v_ref_220_; uint8_t v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_218_ = l_Lean_Syntax_getArg(v___x_213_, v___x_206_);
v___x_219_ = l_Lean_Syntax_getArg(v___x_213_, v___x_212_);
lean_dec(v___x_213_);
v_ref_220_ = l_Lean_replaceRef(v___x_207_, v_a_200_);
lean_dec(v___x_207_);
v___x_221_ = 0;
v___x_222_ = l_Lean_SourceInfo_fromRef(v_ref_220_, v___x_221_);
lean_dec(v_ref_220_);
v___x_223_ = ((lean_object*)(lp_mathlib_Polynomial_term___x25_u2098___00__closed__1));
v___x_224_ = ((lean_object*)(lp_mathlib_Polynomial_term___x25_u2098___00__closed__2));
lean_inc(v___x_222_);
v___x_225_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_222_);
lean_ctor_set(v___x_225_, 1, v___x_224_);
v___x_226_ = l_Lean_Syntax_node3(v___x_222_, v___x_223_, v___x_218_, v___x_225_, v___x_219_);
v___x_227_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_226_);
lean_ctor_set(v___x_227_, 1, v_a_201_);
return v___x_227_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__modByMonic__1___boxed(lean_object* v_x_228_, lean_object* v_a_229_, lean_object* v_a_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_Polynomial___aux__Mathlib__Algebra__Polynomial__Div______unexpand__Polynomial__modByMonic__1(v_x_228_, v_a_229_, v_a_230_);
lean_dec(v_a_229_);
return v_res_231_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_IsField(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Inductions(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Monic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lattice_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Multiplicity(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Div(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_IsField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Inductions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Monic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lattice_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Multiplicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Div(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_IsField(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Inductions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Monic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lattice_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Multiplicity(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Div(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_IsField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Inductions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Monic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lattice_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Multiplicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Div(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Div(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Div(builtin);
}
#ifdef __cplusplus
}
#endif
