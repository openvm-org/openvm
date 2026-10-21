// Lean compiler output
// Module: Mathlib.Data.Finsupp.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.FiniteSupport.Defs public import Mathlib.Data.Multiset.Find
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2080___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_→₀_"};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(182, 80, 115, 120, 96, 192, 196, 236)}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2080___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_u2080___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " →₀ "};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u2080___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2080___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_u2080___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2080__ = (const lean_object*)&lp_mathlib_term___u2192_u2080___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Finsupp"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(226, 26, 254, 66, 110, 152, 140, 234)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_instDecidableEq___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finsupp_instDecidableEq___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finsupp_instDecidableEq___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finsupp_instDecidableEq___redArg___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finsupp_Defs_0__Finsupp_embDomain_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finsupp_Defs_0__Finsupp_embDomain_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__6(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__5));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_term___u2192_u2080___00__closed__1));
lean_inc(v_x_54_);
v___x_58_ = l_Lean_Syntax_isOfKind(v_x_54_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec(v_x_54_);
v___x_59_ = lean_box(1);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v_a_56_);
return v___x_60_;
}
else
{
lean_object* v_quotContext_61_; lean_object* v_currMacroScope_62_; lean_object* v_ref_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_quotContext_61_ = lean_ctor_get(v_a_55_, 1);
v_currMacroScope_62_ = lean_ctor_get(v_a_55_, 2);
v_ref_63_ = lean_ctor_get(v_a_55_, 5);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = l_Lean_Syntax_getArg(v_x_54_, v___x_64_);
v___x_66_ = lean_unsigned_to_nat(2u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_54_, v___x_66_);
lean_dec(v_x_54_);
v___x_68_ = 0;
v___x_69_ = l_Lean_SourceInfo_fromRef(v_ref_63_, v___x_68_);
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__6, &lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__6);
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__7));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__11));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__13));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______macroRules__term___u2192_u2080____1___closed__4));
lean_inc(v_x_87_);
v___x_91_ = l_Lean_Syntax_isOfKind(v_x_87_, v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_93_; 
lean_dec(v_x_87_);
v___x_92_ = lean_box(0);
v___x_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_89_);
return v___x_93_;
}
else
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = l_Lean_Syntax_getArg(v_x_87_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___closed__1));
lean_inc(v___x_95_);
v___x_97_ = l_Lean_Syntax_isOfKind(v___x_95_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v___x_95_);
lean_dec(v_x_87_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_89_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(1u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_87_, v___x_100_);
lean_dec(v_x_87_);
v___x_102_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_101_);
v___x_103_ = l_Lean_Syntax_matchesNull(v___x_101_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v___x_101_);
lean_dec(v___x_95_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_89_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v_ref_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_106_ = l_Lean_Syntax_getArg(v___x_101_, v___x_94_);
v___x_107_ = l_Lean_Syntax_getArg(v___x_101_, v___x_100_);
lean_dec(v___x_101_);
v_ref_108_ = l_Lean_replaceRef(v___x_95_, v_a_88_);
lean_dec(v___x_95_);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_109_);
lean_dec(v_ref_108_);
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192_u2080___00__closed__1));
v___x_112_ = ((lean_object*)(lp_mathlib_term___u2192_u2080___00__closed__4));
lean_inc(v___x_110_);
v___x_113_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v___x_114_ = l_Lean_Syntax_node3(v___x_110_, v___x_111_, v___x_106_, v___x_113_, v___x_107_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_89_);
return v___x_115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib___aux__Mathlib__Data__Finsupp__Defs______unexpand__Finsupp__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero___redArg___lam__0(lean_object* v_inst_120_, lean_object* v_x_121_){
_start:
{
lean_inc(v_inst_120_);
return v_inst_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero___redArg___lam__0___boxed(lean_object* v_inst_122_, lean_object* v_x_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Finsupp_instZero___redArg___lam__0(v_inst_122_, v_x_123_);
lean_dec(v_x_123_);
lean_dec(v_inst_122_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero___redArg(lean_object* v_inst_125_){
_start:
{
lean_object* v___f_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_126_, 0, v_inst_125_);
v___x_127_ = lean_box(0);
v___x_128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v___f_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instZero(lean_object* v_00_u03b1_129_, lean_object* v_M_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_Finsupp_instZero___redArg(v_inst_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instInhabited___redArg(lean_object* v_inst_133_){
_start:
{
lean_object* v___f_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_134_, 0, v_inst_133_);
v___x_135_ = lean_box(0);
v___x_136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
lean_ctor_set(v___x_136_, 1, v___f_134_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instInhabited(lean_object* v_00_u03b1_137_, lean_object* v_M_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_Finsupp_instInhabited___redArg(v_inst_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___lam__0(lean_object* v_self_141_, lean_object* v___y_142_){
_start:
{
lean_object* v_toFun_143_; lean_object* v___x_144_; 
v_toFun_143_ = lean_ctor_get(v_self_141_, 1);
lean_inc(v_toFun_143_);
lean_dec_ref(v_self_141_);
v___x_144_ = lean_apply_1(v_toFun_143_, v___y_142_);
return v___x_144_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_instDecidableEq___redArg___lam__1(lean_object* v___f_145_, lean_object* v_f_146_, lean_object* v_g_147_, lean_object* v_inst_148_, lean_object* v_a_149_, lean_object* v_h_150_){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; uint8_t v___x_154_; 
lean_inc(v___f_145_);
lean_inc(v_a_149_);
v___x_151_ = lean_apply_2(v___f_145_, v_f_146_, v_a_149_);
v___x_152_ = lean_apply_2(v___f_145_, v_g_147_, v_a_149_);
v___x_153_ = lean_apply_2(v_inst_148_, v___x_151_, v___x_152_);
v___x_154_ = lean_unbox(v___x_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___lam__1___boxed(lean_object* v___f_155_, lean_object* v_f_156_, lean_object* v_g_157_, lean_object* v_inst_158_, lean_object* v_a_159_, lean_object* v_h_160_){
_start:
{
uint8_t v_res_161_; lean_object* v_r_162_; 
v_res_161_ = lp_mathlib_Finsupp_instDecidableEq___redArg___lam__1(v___f_155_, v_f_156_, v_g_157_, v_inst_158_, v_a_159_, v_h_160_);
v_r_162_ = lean_box(v_res_161_);
return v_r_162_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_instDecidableEq___redArg(lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_f_166_, lean_object* v_g_167_){
_start:
{
lean_object* v_support_168_; lean_object* v_support_169_; uint8_t v___x_170_; 
v_support_168_ = lean_ctor_get(v_f_166_, 0);
lean_inc_n(v_support_168_, 2);
v_support_169_ = lean_ctor_get(v_g_167_, 0);
lean_inc(v_support_169_);
v___x_170_ = l_List_decidablePerm___redArg(v_inst_164_, v_support_168_, v_support_169_);
if (v___x_170_ == 0)
{
lean_dec(v_support_168_);
lean_dec_ref(v_g_167_);
lean_dec_ref(v_f_166_);
lean_dec_ref(v_inst_165_);
return v___x_170_;
}
else
{
lean_object* v___f_171_; lean_object* v___f_172_; uint8_t v___x_173_; 
v___f_171_ = ((lean_object*)(lp_mathlib_Finsupp_instDecidableEq___redArg___closed__0));
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_instDecidableEq___redArg___lam__1___boxed), 6, 4);
lean_closure_set(v___f_172_, 0, v___f_171_);
lean_closure_set(v___f_172_, 1, v_f_166_);
lean_closure_set(v___f_172_, 2, v_g_167_);
lean_closure_set(v___f_172_, 3, v_inst_165_);
v___x_173_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_support_168_, v___f_172_);
return v___x_173_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___redArg___boxed(lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_f_176_, lean_object* v_g_177_){
_start:
{
uint8_t v_res_178_; lean_object* v_r_179_; 
v_res_178_ = lp_mathlib_Finsupp_instDecidableEq___redArg(v_inst_174_, v_inst_175_, v_f_176_, v_g_177_);
v_r_179_ = lean_box(v_res_178_);
return v_r_179_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_instDecidableEq(lean_object* v_00_u03b1_180_, lean_object* v_M_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_f_185_, lean_object* v_g_186_){
_start:
{
uint8_t v___x_187_; 
v___x_187_ = lp_mathlib_Finsupp_instDecidableEq___redArg(v_inst_183_, v_inst_184_, v_f_185_, v_g_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instDecidableEq___boxed(lean_object* v_00_u03b1_188_, lean_object* v_M_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_f_193_, lean_object* v_g_194_){
_start:
{
uint8_t v_res_195_; lean_object* v_r_196_; 
v_res_195_ = lp_mathlib_Finsupp_instDecidableEq(v_00_u03b1_188_, v_M_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_f_193_, v_g_194_);
lean_dec(v_inst_190_);
v_r_196_ = lean_box(v_res_195_);
return v_r_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finsupp_Defs_0__Finsupp_embDomain_match__1_splitter___redArg(lean_object* v_x_197_, lean_object* v_h__1_198_, lean_object* v_h__2_199_){
_start:
{
if (lean_obj_tag(v_x_197_) == 0)
{
lean_object* v___x_200_; lean_object* v___x_201_; 
lean_dec(v_h__1_198_);
v___x_200_ = lean_box(0);
v___x_201_ = lean_apply_1(v_h__2_199_, v___x_200_);
return v___x_201_;
}
else
{
lean_object* v_val_202_; lean_object* v___x_203_; 
lean_dec(v_h__2_199_);
v_val_202_ = lean_ctor_get(v_x_197_, 0);
lean_inc(v_val_202_);
lean_dec_ref_known(v_x_197_, 1);
v___x_203_ = lean_apply_1(v_h__1_198_, v_val_202_);
return v___x_203_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Finsupp_Defs_0__Finsupp_embDomain_match__1_splitter(lean_object* v_00_u03b1_204_, lean_object* v_motive_205_, lean_object* v_x_206_, lean_object* v_h__1_207_, lean_object* v_h__2_208_){
_start:
{
if (lean_obj_tag(v_x_206_) == 0)
{
lean_object* v___x_209_; lean_object* v___x_210_; 
lean_dec(v_h__1_207_);
v___x_209_ = lean_box(0);
v___x_210_ = lean_apply_1(v_h__2_208_, v___x_209_);
return v___x_210_;
}
else
{
lean_object* v_val_211_; lean_object* v___x_212_; 
lean_dec(v_h__2_208_);
v_val_211_ = lean_ctor_get(v_x_206_, 0);
lean_inc(v_val_211_);
lean_dec_ref_known(v_x_206_, 1);
v___x_212_ = lean_apply_1(v_h__1_207_, v_val_211_);
return v___x_212_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Find(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finsupp_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Find(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finsupp_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
