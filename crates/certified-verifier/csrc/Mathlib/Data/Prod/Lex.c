// Lean compiler output
// Module: Mathlib.Data.Prod.Lex
// Imports: public import Init public meta import Init public import Mathlib.Data.Prod.Basic public import Mathlib.Order.BoundedOrder.Basic public import Mathlib.Order.Lattice public import Mathlib.Order.Lex public import Mathlib.Tactic.Tauto public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_instDecidableEqProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
uint8_t lp_mathlib_Prod_Lex_decidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_Lex_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instDecidableEqLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0_value;
static const lean_string_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Lex"};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1_value;
static const lean_string_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 8, .m_data = "term_×ₗ_"};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__2 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(197, 185, 120, 51, 217, 37, 16, 88)}};
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(43, 177, 108, 122, 204, 8, 47, 62)}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value;
static const lean_string_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__4 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__5 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__5_value;
static const lean_string_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 4, .m_data = " ×ₗ "};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__6 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__6_value)}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__7 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__7_value;
static const lean_string_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__8 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__9 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__9_value),((lean_object*)(((size_t)(34) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__10 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__5_value),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__7_value),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__10_value)}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__11 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3_value),((lean_object*)(((size_t)(35) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__11_value)}};
static const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__12 = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Prod_Lex_term___xd7_u2097__ = (const lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__12_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__0 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__0_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__1 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__1_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__2 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__2_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__3 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4_value;
static lean_once_cell_t lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__5;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(47, 205, 122, 164, 96, 181, 7, 42)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__6 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__6_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__7 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(197, 185, 120, 51, 217, 37, 16, 88)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__8 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__8_value)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__9 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__10 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__7_value),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__10_value)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__11 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__11_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__12 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__12_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__13 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__13_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__14 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__14_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__16 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__16_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__18 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__18_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__19 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__19_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__20 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__20_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__21 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__21_value;
static lean_once_cell_t lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__22;
static lean_once_cell_t lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__23;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__25 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__25_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24_value)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__26 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__26_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__27 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__27_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__25_value),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__27_value)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__28 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__28_value;
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__29 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__29_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__0 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__1 = (const lean_object*)&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instWellFoundedRelationLexOfWellFoundedLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Prod_Lex_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prod_Lex_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_Prod_Lex_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instOrdLexProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instOrdLexProd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Prod_Lex_orderBot___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prod_Lex_orderBot___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_boundedOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_boundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_boundedOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__5(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_39_ = ((lean_object*)(lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__1));
v___x_40_ = l_String_toRawSubstring_x27(v___x_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__22(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_77_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__21));
v___x_78_ = l_String_toRawSubstring_x27(v___x_77_);
return v___x_78_;
}
}
static lean_object* _init_lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__23(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = ((lean_object*)(lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__0));
v___x_80_ = l_String_toRawSubstring_x27(v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1(lean_object* v_x_95_, lean_object* v_a_96_, lean_object* v_a_97_){
_start:
{
lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_98_ = ((lean_object*)(lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3));
lean_inc(v_x_95_);
v___x_99_ = l_Lean_Syntax_isOfKind(v_x_95_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v_x_95_);
v___x_100_ = lean_box(1);
v___x_101_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v_a_97_);
return v___x_101_;
}
else
{
lean_object* v_quotContext_102_; lean_object* v_currMacroScope_103_; lean_object* v_ref_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_quotContext_102_ = lean_ctor_get(v_a_96_, 1);
v_currMacroScope_103_ = lean_ctor_get(v_a_96_, 2);
v_ref_104_ = lean_ctor_get(v_a_96_, 5);
v___x_105_ = lean_unsigned_to_nat(0u);
v___x_106_ = l_Lean_Syntax_getArg(v_x_95_, v___x_105_);
v___x_107_ = lean_unsigned_to_nat(2u);
v___x_108_ = l_Lean_Syntax_getArg(v_x_95_, v___x_107_);
lean_dec(v_x_95_);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_104_, v___x_109_);
v___x_111_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4));
v___x_112_ = lean_obj_once(&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__5, &lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__5_once, _init_lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__5);
v___x_113_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__6));
lean_inc_n(v_currMacroScope_103_, 3);
lean_inc_n(v_quotContext_102_, 3);
v___x_114_ = l_Lean_addMacroScope(v_quotContext_102_, v___x_113_, v_currMacroScope_103_);
v___x_115_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__10));
v___x_116_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__11));
lean_inc_n(v___x_110_, 11);
v___x_117_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_117_, 0, v___x_110_);
lean_ctor_set(v___x_117_, 1, v___x_112_);
lean_ctor_set(v___x_117_, 2, v___x_114_);
lean_ctor_set(v___x_117_, 3, v___x_116_);
v___x_118_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__13));
v___x_119_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__15));
v___x_120_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__17));
v___x_121_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__18));
v___x_122_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_110_);
lean_ctor_set(v___x_122_, 1, v___x_121_);
v___x_123_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__20));
v___x_124_ = lean_obj_once(&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__22, &lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__22_once, _init_lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__22);
v___x_125_ = lean_box(0);
v___x_126_ = l_Lean_addMacroScope(v_quotContext_102_, v___x_125_, v_currMacroScope_103_);
v___x_127_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_127_, 0, v___x_110_);
lean_ctor_set(v___x_127_, 1, v___x_124_);
lean_ctor_set(v___x_127_, 2, v___x_126_);
lean_ctor_set(v___x_127_, 3, v___x_115_);
v___x_128_ = l_Lean_Syntax_node1(v___x_110_, v___x_123_, v___x_127_);
v___x_129_ = l_Lean_Syntax_node2(v___x_110_, v___x_120_, v___x_122_, v___x_128_);
v___x_130_ = lean_obj_once(&lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__23, &lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__23_once, _init_lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__23);
v___x_131_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24));
v___x_132_ = l_Lean_addMacroScope(v_quotContext_102_, v___x_131_, v_currMacroScope_103_);
v___x_133_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__28));
v___x_134_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_134_, 0, v___x_110_);
lean_ctor_set(v___x_134_, 1, v___x_130_);
lean_ctor_set(v___x_134_, 2, v___x_132_);
lean_ctor_set(v___x_134_, 3, v___x_133_);
v___x_135_ = l_Lean_Syntax_node2(v___x_110_, v___x_118_, v___x_106_, v___x_108_);
v___x_136_ = l_Lean_Syntax_node2(v___x_110_, v___x_111_, v___x_134_, v___x_135_);
v___x_137_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__29));
v___x_138_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_110_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = l_Lean_Syntax_node3(v___x_110_, v___x_119_, v___x_129_, v___x_136_, v___x_138_);
v___x_140_ = l_Lean_Syntax_node1(v___x_110_, v___x_118_, v___x_139_);
v___x_141_ = l_Lean_Syntax_node2(v___x_110_, v___x_111_, v___x_117_, v___x_140_);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v_a_97_);
return v___x_142_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___boxed(lean_object* v_x_143_, lean_object* v_a_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1(v_x_143_, v_a_144_, v_a_145_);
lean_dec_ref(v_a_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1(lean_object* v_x_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v___x_153_; uint8_t v___x_154_; 
v___x_153_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__4));
lean_inc(v_x_150_);
v___x_154_ = l_Lean_Syntax_isOfKind(v_x_150_, v___x_153_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; lean_object* v___x_156_; 
lean_dec(v_x_150_);
v___x_155_ = lean_box(0);
v___x_156_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
lean_ctor_set(v___x_156_, 1, v_a_152_);
return v___x_156_;
}
else
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; uint8_t v___x_160_; 
v___x_157_ = lean_unsigned_to_nat(0u);
v___x_158_ = l_Lean_Syntax_getArg(v_x_150_, v___x_157_);
v___x_159_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___closed__1));
lean_inc(v___x_158_);
v___x_160_ = l_Lean_Syntax_isOfKind(v___x_158_, v___x_159_);
if (v___x_160_ == 0)
{
lean_object* v___x_161_; lean_object* v___x_162_; 
lean_dec(v___x_158_);
lean_dec(v_x_150_);
v___x_161_ = lean_box(0);
v___x_162_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_161_);
lean_ctor_set(v___x_162_, 1, v_a_152_);
return v___x_162_;
}
else
{
lean_object* v___x_163_; lean_object* v___x_164_; uint8_t v___x_165_; 
v___x_163_ = lean_unsigned_to_nat(1u);
v___x_164_ = l_Lean_Syntax_getArg(v_x_150_, v___x_163_);
lean_dec(v_x_150_);
lean_inc(v___x_164_);
v___x_165_ = l_Lean_Syntax_matchesNull(v___x_164_, v___x_163_);
if (v___x_165_ == 0)
{
lean_object* v___x_166_; lean_object* v___x_167_; 
lean_dec(v___x_164_);
lean_dec(v___x_158_);
v___x_166_ = lean_box(0);
v___x_167_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_166_);
lean_ctor_set(v___x_167_, 1, v_a_152_);
return v___x_167_;
}
else
{
lean_object* v___x_168_; uint8_t v___x_169_; 
v___x_168_ = l_Lean_Syntax_getArg(v___x_164_, v___x_157_);
lean_dec(v___x_164_);
lean_inc(v___x_168_);
v___x_169_ = l_Lean_Syntax_isOfKind(v___x_168_, v___x_153_);
if (v___x_169_ == 0)
{
lean_object* v___x_170_; lean_object* v___x_171_; 
lean_dec(v___x_168_);
lean_dec(v___x_158_);
v___x_170_ = lean_box(0);
v___x_171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v_a_152_);
return v___x_171_;
}
else
{
lean_object* v___x_172_; lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_172_ = l_Lean_Syntax_getArg(v___x_168_, v___x_157_);
v___x_173_ = ((lean_object*)(lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______macroRules__Prod__Lex__term___xd7_u2097____1___closed__24));
v___x_174_ = l_Lean_Syntax_matchesIdent(v___x_172_, v___x_173_);
lean_dec(v___x_172_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v___x_176_; 
lean_dec(v___x_168_);
lean_dec(v___x_158_);
v___x_175_ = lean_box(0);
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v_a_152_);
return v___x_176_;
}
else
{
lean_object* v___x_177_; lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_177_ = l_Lean_Syntax_getArg(v___x_168_, v___x_163_);
lean_dec(v___x_168_);
v___x_178_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_177_);
v___x_179_ = l_Lean_Syntax_matchesNull(v___x_177_, v___x_178_);
if (v___x_179_ == 0)
{
lean_object* v___x_180_; lean_object* v___x_181_; 
lean_dec(v___x_177_);
lean_dec(v___x_158_);
v___x_180_ = lean_box(0);
v___x_181_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
lean_ctor_set(v___x_181_, 1, v_a_152_);
return v___x_181_;
}
else
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v_ref_184_; uint8_t v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_182_ = l_Lean_Syntax_getArg(v___x_177_, v___x_157_);
v___x_183_ = l_Lean_Syntax_getArg(v___x_177_, v___x_163_);
lean_dec(v___x_177_);
v_ref_184_ = l_Lean_replaceRef(v___x_158_, v_a_151_);
lean_dec(v___x_158_);
v___x_185_ = 0;
v___x_186_ = l_Lean_SourceInfo_fromRef(v_ref_184_, v___x_185_);
lean_dec(v_ref_184_);
v___x_187_ = ((lean_object*)(lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__3));
v___x_188_ = ((lean_object*)(lp_mathlib_Prod_Lex_term___xd7_u2097___00__closed__6));
lean_inc(v___x_186_);
v___x_189_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_186_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = l_Lean_Syntax_node3(v___x_186_, v___x_187_, v___x_182_, v___x_189_, v___x_183_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v_a_152_);
return v___x_191_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1___boxed(lean_object* v_x_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_Prod_Lex___aux__Mathlib__Data__Prod__Lex______unexpand__Lex__1(v_x_192_, v_a_193_, v_a_194_);
lean_dec(v_a_193_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLE(lean_object* v_00_u03b1_196_, lean_object* v_00_u03b2_197_, lean_object* v_inst_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lean_box(0);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLT(lean_object* v_00_u03b1_201_, lean_object* v_00_u03b2_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lean_box(0);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instWellFoundedRelationLexOfWellFoundedLT(lean_object* v_00_u03b1_206_, lean_object* v_00_u03b2_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lean_box(0);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPreorder(lean_object* v_00_u03b1_216_, lean_object* v_00_u03b2_217_, lean_object* v_inst_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = ((lean_object*)(lp_mathlib_Prod_Lex_instPreorder___closed__0));
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPreorder___boxed(lean_object* v_00_u03b1_221_, lean_object* v_00_u03b2_222_, lean_object* v_inst_223_, lean_object* v_inst_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Prod_Lex_instPreorder(v_00_u03b1_221_, v_00_u03b2_222_, v_inst_223_, v_inst_224_);
lean_dec_ref(v_inst_224_);
lean_dec_ref(v_inst_223_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder___redArg(lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_Prod_Lex_instPreorder(lean_box(0), lean_box(0), v_inst_226_, v_inst_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder___redArg___boxed(lean_object* v_inst_229_, lean_object* v_inst_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_Prod_Lex_instPartialOrder___redArg(v_inst_229_, v_inst_230_);
lean_dec_ref(v_inst_230_);
lean_dec_ref(v_inst_229_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder(lean_object* v_00_u03b1_232_, lean_object* v_00_u03b2_233_, lean_object* v_inst_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib_Prod_Lex_instPreorder(lean_box(0), lean_box(0), v_inst_234_, v_inst_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instPartialOrder___boxed(lean_object* v_00_u03b1_237_, lean_object* v_00_u03b2_238_, lean_object* v_inst_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Prod_Lex_instPartialOrder(v_00_u03b1_237_, v_00_u03b2_238_, v_inst_239_, v_inst_240_);
lean_dec_ref(v_inst_240_);
lean_dec_ref(v_inst_239_);
return v_res_241_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0(lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_a_244_, lean_object* v_a_245_){
_start:
{
lean_object* v_fst_246_; lean_object* v_snd_247_; lean_object* v_fst_248_; lean_object* v_snd_249_; lean_object* v___x_250_; uint8_t v___x_251_; 
v_fst_246_ = lean_ctor_get(v_a_244_, 0);
lean_inc(v_fst_246_);
v_snd_247_ = lean_ctor_get(v_a_244_, 1);
lean_inc(v_snd_247_);
lean_dec_ref(v_a_244_);
v_fst_248_ = lean_ctor_get(v_a_245_, 0);
lean_inc(v_fst_248_);
v_snd_249_ = lean_ctor_get(v_a_245_, 1);
lean_inc(v_snd_249_);
lean_dec_ref(v_a_245_);
v___x_250_ = lean_apply_2(v_inst_242_, v_fst_246_, v_fst_248_);
v___x_251_ = lean_unbox(v___x_250_);
if (v___x_251_ == 1)
{
lean_object* v___x_252_; uint8_t v___x_253_; 
v___x_252_ = lean_apply_2(v_inst_243_, v_snd_247_, v_snd_249_);
v___x_253_ = lean_unbox(v___x_252_);
return v___x_253_;
}
else
{
uint8_t v___x_254_; 
lean_dec(v_snd_249_);
lean_dec(v_snd_247_);
lean_dec_ref(v_inst_243_);
v___x_254_ = lean_unbox(v___x_250_);
return v___x_254_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0___boxed(lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_a_257_, lean_object* v_a_258_){
_start:
{
uint8_t v_res_259_; lean_object* v_r_260_; 
v_res_259_ = lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0(v_inst_255_, v_inst_256_, v_a_257_, v_a_258_);
v_r_260_ = lean_box(v_res_259_);
return v_r_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instOrdLexProd___redArg(lean_object* v_inst_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v___f_263_; 
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_263_, 0, v_inst_261_);
lean_closure_set(v___f_263_, 1, v_inst_262_);
return v___f_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instOrdLexProd(lean_object* v_00_u03b1_264_, lean_object* v_00_u03b2_265_, lean_object* v_inst_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___f_268_; 
v___f_268_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_268_, 0, v_inst_266_);
lean_closure_set(v___f_268_, 1, v_inst_267_);
return v___f_268_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0(lean_object* v_toDecidableEq_269_, lean_object* v_a_270_, lean_object* v_b_271_){
_start:
{
lean_object* v___x_272_; uint8_t v___x_273_; 
v___x_272_ = lean_apply_2(v_toDecidableEq_269_, v_a_270_, v_b_271_);
v___x_273_ = lean_unbox(v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0___boxed(lean_object* v_toDecidableEq_274_, lean_object* v_a_275_, lean_object* v_b_276_){
_start:
{
uint8_t v_res_277_; lean_object* v_r_278_; 
v_res_277_ = lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0(v_toDecidableEq_274_, v_a_275_, v_b_276_);
v_r_278_ = lean_box(v_res_277_);
return v_r_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__1(lean_object* v___f_279_, lean_object* v_toDecidableLT_280_, lean_object* v_toDecidableLE_281_, lean_object* v_a_282_, lean_object* v_b_283_){
_start:
{
uint8_t v___x_284_; 
lean_inc_ref(v_b_283_);
lean_inc_ref(v_a_282_);
v___x_284_ = lp_mathlib_Prod_Lex_decidable___redArg(v___f_279_, v_toDecidableLT_280_, v_toDecidableLE_281_, v_a_282_, v_b_283_);
if (v___x_284_ == 0)
{
lean_dec_ref(v_a_282_);
return v_b_283_;
}
else
{
lean_dec_ref(v_b_283_);
return v_a_282_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__3(lean_object* v___f_285_, lean_object* v_toDecidableLT_286_, lean_object* v_toDecidableLE_287_, lean_object* v_a_288_, lean_object* v_b_289_){
_start:
{
uint8_t v___x_290_; 
lean_inc_ref(v_b_289_);
lean_inc_ref(v_a_288_);
v___x_290_ = lp_mathlib_Prod_Lex_decidable___redArg(v___f_285_, v_toDecidableLT_286_, v_toDecidableLE_287_, v_a_288_, v_b_289_);
if (v___x_290_ == 0)
{
lean_dec_ref(v_b_289_);
return v_a_288_;
}
else
{
lean_dec_ref(v_a_288_);
return v_b_289_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__5(lean_object* v___f_291_, lean_object* v___f_292_, lean_object* v_a_293_, lean_object* v_b_294_){
_start:
{
uint8_t v___x_295_; 
v___x_295_ = l_instDecidableEqProd___redArg(v___f_291_, v___f_292_, v_a_293_, v_b_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__5___boxed(lean_object* v___f_296_, lean_object* v___f_297_, lean_object* v_a_298_, lean_object* v_b_299_){
_start:
{
uint8_t v_res_300_; lean_object* v_r_301_; 
v_res_300_ = lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__5(v___f_296_, v___f_297_, v_a_298_, v_b_299_);
v_r_301_ = lean_box(v_res_300_);
return v_r_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder___redArg(lean_object* v_inst_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v_toPartialOrder_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v_toPartialOrder_309_; lean_object* v___x_310_; lean_object* v_toOrd_311_; lean_object* v_toDecidableEq_312_; lean_object* v_toDecidableLT_313_; lean_object* v_toOrd_314_; lean_object* v_toDecidableLE_315_; lean_object* v_toDecidableEq_316_; lean_object* v_toDecidableLT_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_333_; 
v___x_304_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_302_);
v___x_305_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_304_);
v_toPartialOrder_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc_ref(v_toPartialOrder_306_);
lean_dec_ref(v___x_305_);
v___x_307_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_303_);
v___x_308_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_307_);
v_toPartialOrder_309_ = lean_ctor_get(v___x_308_, 0);
lean_inc_ref(v_toPartialOrder_309_);
lean_dec_ref(v___x_308_);
v___x_310_ = lp_mathlib_Prod_Lex_instPreorder(lean_box(0), lean_box(0), v_toPartialOrder_306_, v_toPartialOrder_309_);
lean_dec_ref(v_toPartialOrder_309_);
lean_dec_ref(v_toPartialOrder_306_);
v_toOrd_311_ = lean_ctor_get(v_inst_302_, 3);
lean_inc_ref(v_toOrd_311_);
v_toDecidableEq_312_ = lean_ctor_get(v_inst_302_, 5);
lean_inc_ref(v_toDecidableEq_312_);
v_toDecidableLT_313_ = lean_ctor_get(v_inst_302_, 6);
lean_inc_ref(v_toDecidableLT_313_);
lean_dec_ref(v_inst_302_);
v_toOrd_314_ = lean_ctor_get(v_inst_303_, 3);
v_toDecidableLE_315_ = lean_ctor_get(v_inst_303_, 4);
v_toDecidableEq_316_ = lean_ctor_get(v_inst_303_, 5);
v_toDecidableLT_317_ = lean_ctor_get(v_inst_303_, 6);
v_isSharedCheck_333_ = !lean_is_exclusive(v_inst_303_);
if (v_isSharedCheck_333_ == 0)
{
lean_object* v_unused_334_; lean_object* v_unused_335_; lean_object* v_unused_336_; 
v_unused_334_ = lean_ctor_get(v_inst_303_, 2);
lean_dec(v_unused_334_);
v_unused_335_ = lean_ctor_get(v_inst_303_, 1);
lean_dec(v_unused_335_);
v_unused_336_ = lean_ctor_get(v_inst_303_, 0);
lean_dec(v_unused_336_);
v___x_319_ = v_inst_303_;
v_isShared_320_ = v_isSharedCheck_333_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_toDecidableLT_317_);
lean_inc(v_toDecidableEq_316_);
lean_inc(v_toDecidableLE_315_);
lean_inc(v_toOrd_314_);
lean_dec(v_inst_303_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_333_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___f_321_; lean_object* v___f_322_; lean_object* v___f_323_; lean_object* v___f_324_; lean_object* v___f_325_; lean_object* v___f_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_331_; 
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_321_, 0, v_toDecidableEq_312_);
lean_inc_ref_n(v_toDecidableLE_315_, 2);
lean_inc_ref_n(v_toDecidableLT_313_, 3);
lean_inc_ref_n(v___f_321_, 4);
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__1), 5, 3);
lean_closure_set(v___f_322_, 0, v___f_321_);
lean_closure_set(v___f_322_, 1, v_toDecidableLT_313_);
lean_closure_set(v___f_322_, 2, v_toDecidableLE_315_);
v___f_323_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__3), 5, 3);
lean_closure_set(v___f_323_, 0, v___f_321_);
lean_closure_set(v___f_323_, 1, v_toDecidableLT_313_);
lean_closure_set(v___f_323_, 2, v_toDecidableLE_315_);
v___f_324_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_324_, 0, v_toDecidableEq_316_);
v___f_325_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instLinearOrder___redArg___lam__5___boxed), 4, 2);
lean_closure_set(v___f_325_, 0, v___f_321_);
lean_closure_set(v___f_325_, 1, v___f_324_);
v___f_326_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_instOrdLexProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_326_, 0, v_toOrd_311_);
lean_closure_set(v___f_326_, 1, v_toOrd_314_);
v___x_327_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_decidable___boxed), 9, 7);
lean_closure_set(v___x_327_, 0, lean_box(0));
lean_closure_set(v___x_327_, 1, lean_box(0));
lean_closure_set(v___x_327_, 2, v___f_321_);
lean_closure_set(v___x_327_, 3, lean_box(0));
lean_closure_set(v___x_327_, 4, lean_box(0));
lean_closure_set(v___x_327_, 5, v_toDecidableLT_313_);
lean_closure_set(v___x_327_, 6, v_toDecidableLE_315_);
v___x_328_ = lean_alloc_closure((void*)(lp_mathlib_instDecidableEqLex___boxed), 4, 2);
lean_closure_set(v___x_328_, 0, lean_box(0));
lean_closure_set(v___x_328_, 1, v___f_325_);
v___x_329_ = lean_alloc_closure((void*)(lp_mathlib_Prod_Lex_decidable___boxed), 9, 7);
lean_closure_set(v___x_329_, 0, lean_box(0));
lean_closure_set(v___x_329_, 1, lean_box(0));
lean_closure_set(v___x_329_, 2, v___f_321_);
lean_closure_set(v___x_329_, 3, lean_box(0));
lean_closure_set(v___x_329_, 4, lean_box(0));
lean_closure_set(v___x_329_, 5, v_toDecidableLT_313_);
lean_closure_set(v___x_329_, 6, v_toDecidableLT_317_);
if (v_isShared_320_ == 0)
{
lean_ctor_set(v___x_319_, 6, v___x_329_);
lean_ctor_set(v___x_319_, 5, v___x_328_);
lean_ctor_set(v___x_319_, 4, v___x_327_);
lean_ctor_set(v___x_319_, 3, v___f_326_);
lean_ctor_set(v___x_319_, 2, v___f_323_);
lean_ctor_set(v___x_319_, 1, v___f_322_);
lean_ctor_set(v___x_319_, 0, v___x_310_);
v___x_331_ = v___x_319_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v___x_310_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v___f_322_);
lean_ctor_set(v_reuseFailAlloc_332_, 2, v___f_323_);
lean_ctor_set(v_reuseFailAlloc_332_, 3, v___f_326_);
lean_ctor_set(v_reuseFailAlloc_332_, 4, v___x_327_);
lean_ctor_set(v_reuseFailAlloc_332_, 5, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_332_, 6, v___x_329_);
v___x_331_ = v_reuseFailAlloc_332_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
return v___x_331_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_instLinearOrder(lean_object* v_00_u03b1_337_, lean_object* v_00_u03b2_338_, lean_object* v_inst_339_, lean_object* v_inst_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_Prod_Lex_instLinearOrder___redArg(v_inst_339_, v_inst_340_);
return v___x_341_;
}
}
static lean_object* _init_lp_mathlib_Prod_Lex_orderBot___redArg___closed__0(void){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderBot___redArg(lean_object* v_inst_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___x_345_; lean_object* v_toFun_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_345_ = lean_obj_once(&lp_mathlib_Prod_Lex_orderBot___redArg___closed__0, &lp_mathlib_Prod_Lex_orderBot___redArg___closed__0_once, _init_lp_mathlib_Prod_Lex_orderBot___redArg___closed__0);
v_toFun_346_ = lean_ctor_get(v___x_345_, 0);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v_inst_343_);
lean_ctor_set(v___x_347_, 1, v_inst_344_);
lean_inc(v_toFun_346_);
v___x_348_ = lean_apply_1(v_toFun_346_, v___x_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderBot(lean_object* v_00_u03b1_349_, lean_object* v_00_u03b2_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_){
_start:
{
lean_object* v___x_355_; 
v___x_355_ = lp_mathlib_Prod_Lex_orderBot___redArg(v_inst_353_, v_inst_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderBot___boxed(lean_object* v_00_u03b1_356_, lean_object* v_00_u03b2_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_Prod_Lex_orderBot(v_00_u03b1_356_, v_00_u03b2_357_, v_inst_358_, v_inst_359_, v_inst_360_, v_inst_361_);
lean_dec_ref(v_inst_359_);
lean_dec_ref(v_inst_358_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderTop___redArg(lean_object* v_inst_363_, lean_object* v_inst_364_){
_start:
{
lean_object* v___x_365_; lean_object* v_toFun_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_365_ = lean_obj_once(&lp_mathlib_Prod_Lex_orderBot___redArg___closed__0, &lp_mathlib_Prod_Lex_orderBot___redArg___closed__0_once, _init_lp_mathlib_Prod_Lex_orderBot___redArg___closed__0);
v_toFun_366_ = lean_ctor_get(v___x_365_, 0);
v___x_367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_367_, 0, v_inst_363_);
lean_ctor_set(v___x_367_, 1, v_inst_364_);
lean_inc(v_toFun_366_);
v___x_368_ = lean_apply_1(v_toFun_366_, v___x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderTop(lean_object* v_00_u03b1_369_, lean_object* v_00_u03b2_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_mathlib_Prod_Lex_orderTop___redArg(v_inst_373_, v_inst_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_orderTop___boxed(lean_object* v_00_u03b1_376_, lean_object* v_00_u03b2_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_Prod_Lex_orderTop(v_00_u03b1_376_, v_00_u03b2_377_, v_inst_378_, v_inst_379_, v_inst_380_, v_inst_381_);
lean_dec_ref(v_inst_379_);
lean_dec_ref(v_inst_378_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_boundedOrder___redArg(lean_object* v_inst_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v_toOrderTop_385_; lean_object* v_toOrderBot_386_; lean_object* v_toOrderTop_387_; lean_object* v_toOrderBot_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_397_; 
v_toOrderTop_385_ = lean_ctor_get(v_inst_383_, 0);
lean_inc(v_toOrderTop_385_);
v_toOrderBot_386_ = lean_ctor_get(v_inst_383_, 1);
lean_inc(v_toOrderBot_386_);
lean_dec_ref(v_inst_383_);
v_toOrderTop_387_ = lean_ctor_get(v_inst_384_, 0);
v_toOrderBot_388_ = lean_ctor_get(v_inst_384_, 1);
v_isSharedCheck_397_ = !lean_is_exclusive(v_inst_384_);
if (v_isSharedCheck_397_ == 0)
{
v___x_390_ = v_inst_384_;
v_isShared_391_ = v_isSharedCheck_397_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_toOrderBot_388_);
lean_inc(v_toOrderTop_387_);
lean_dec(v_inst_384_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_397_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_395_; 
v___x_392_ = lp_mathlib_Prod_Lex_orderTop___redArg(v_toOrderTop_385_, v_toOrderTop_387_);
v___x_393_ = lp_mathlib_Prod_Lex_orderBot___redArg(v_toOrderBot_386_, v_toOrderBot_388_);
if (v_isShared_391_ == 0)
{
lean_ctor_set(v___x_390_, 1, v___x_393_);
lean_ctor_set(v___x_390_, 0, v___x_392_);
v___x_395_ = v___x_390_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_396_; 
v_reuseFailAlloc_396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_396_, 0, v___x_392_);
lean_ctor_set(v_reuseFailAlloc_396_, 1, v___x_393_);
v___x_395_ = v_reuseFailAlloc_396_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
return v___x_395_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_boundedOrder(lean_object* v_00_u03b1_398_, lean_object* v_00_u03b2_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_mathlib_Prod_Lex_boundedOrder___redArg(v_inst_402_, v_inst_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_boundedOrder___boxed(lean_object* v_00_u03b1_405_, lean_object* v_00_u03b2_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_Prod_Lex_boundedOrder(v_00_u03b1_405_, v_00_u03b2_406_, v_inst_407_, v_inst_408_, v_inst_409_, v_inst_410_);
lean_dec_ref(v_inst_408_);
lean_dec_ref(v_inst_407_);
return v_res_411_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_Lex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Prod_Lex(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Prod_Lex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Prod_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Prod_Lex(builtin);
}
#ifdef __cplusplus
}
#endif
