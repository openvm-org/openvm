// Lean compiler output
// Module: Mathlib.Order.SymmDiff
// Imports: public import Init public meta import Init public import Mathlib.Order.BooleanAlgebra.Basic public import Mathlib.Logic.Equiv.Basic
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_bihimp___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_bihimp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_symmDiff_term___u2206___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "symmDiff"};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__0 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__0_value;
static const lean_string_object lp_mathlib_symmDiff_term___u2206___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∆_"};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__1 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__1_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 73, 144, 185, 177, 40, 230, 237)}};
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(105, 91, 244, 95, 124, 67, 148, 57)}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__2 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__2_value;
static const lean_string_object lp_mathlib_symmDiff_term___u2206___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__3 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__3_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__4 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__4_value;
static const lean_string_object lp_mathlib_symmDiff_term___u2206___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ∆ "};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__5 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__5_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__5_value)}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__6 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__6_value;
static const lean_string_object lp_mathlib_symmDiff_term___u2206___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__7 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__7_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__8 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__8_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__8_value),((lean_object*)(((size_t)(101) << 1) | 1))}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__9 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__9_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__4_value),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__6_value),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__9_value)}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__10 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__10_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u2206___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__2_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__10_value)}};
static const lean_object* lp_mathlib_symmDiff_term___u2206___00__closed__11 = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_symmDiff_term___u2206__ = (const lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__11_value;
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__0 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__0_value;
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__1 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__1_value;
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__2 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__2_value;
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__3 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__3_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4_value;
static lean_once_cell_t lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__5;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 73, 144, 185, 177, 40, 230, 237)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__6 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__6_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__7 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__7_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__6_value)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__8 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__8_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__9 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__9_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__7_value),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__9_value)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__10 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__10_value;
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__11 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__11_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__12 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__0 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__0_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__1 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_symmDiff_term___u21d4___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⇔_"};
static const lean_object* lp_mathlib_symmDiff_term___u21d4___00__closed__0 = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__0_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u21d4___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 73, 144, 185, 177, 40, 230, 237)}};
static const lean_ctor_object lp_mathlib_symmDiff_term___u21d4___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(219, 64, 65, 14, 43, 143, 77, 198)}};
static const lean_object* lp_mathlib_symmDiff_term___u21d4___00__closed__1 = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__1_value;
static const lean_string_object lp_mathlib_symmDiff_term___u21d4___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⇔ "};
static const lean_object* lp_mathlib_symmDiff_term___u21d4___00__closed__2 = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__2_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u21d4___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__2_value)}};
static const lean_object* lp_mathlib_symmDiff_term___u21d4___00__closed__3 = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__3_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u21d4___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__4_value),((lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__3_value),((lean_object*)&lp_mathlib_symmDiff_term___u2206___00__closed__9_value)}};
static const lean_object* lp_mathlib_symmDiff_term___u21d4___00__closed__4 = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__4_value;
static const lean_ctor_object lp_mathlib_symmDiff_term___u21d4___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__1_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__4_value)}};
static const lean_object* lp_mathlib_symmDiff_term___u21d4___00__closed__5 = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_symmDiff_term___u21d4__ = (const lean_object*)&lp_mathlib_symmDiff_term___u21d4___00__closed__5_value;
static const lean_string_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "bihimp"};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__0 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__0_value;
static lean_once_cell_t lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__1;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(53, 5, 160, 128, 224, 128, 230, 115)}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__2 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__2_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__3 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__3_value;
static const lean_ctor_object lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__4 = (const lean_object*)&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__bihimp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__bihimp__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_a_3_, lean_object* v_b_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
lean_inc(v_inst_2_);
lean_inc(v_b_4_);
lean_inc(v_a_3_);
v___x_5_ = lean_apply_2(v_inst_2_, v_a_3_, v_b_4_);
v___x_6_ = lean_apply_2(v_inst_2_, v_b_4_, v_a_3_);
v___x_7_ = lean_apply_2(v_inst_1_, v___x_5_, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_a_11_, lean_object* v_b_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_symmDiff___redArg(v_inst_9_, v_inst_10_, v_a_11_, v_b_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_bihimp___redArg(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_a_16_, lean_object* v_b_17_){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
lean_inc(v_inst_15_);
lean_inc(v_a_16_);
lean_inc(v_b_17_);
v___x_18_ = lean_apply_2(v_inst_15_, v_b_17_, v_a_16_);
v___x_19_ = lean_apply_2(v_inst_15_, v_a_16_, v_b_17_);
v___x_20_ = lean_apply_2(v_inst_14_, v___x_18_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_bihimp(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_bihimp___redArg(v_inst_22_, v_inst_23_, v_a_24_, v_b_25_);
return v___x_26_;
}
}
static lean_object* _init_lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__5(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib_symmDiff_term___u2206___00__closed__0));
v___x_63_ = l_String_toRawSubstring_x27(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v___x_83_; uint8_t v___x_84_; 
v___x_83_ = ((lean_object*)(lp_mathlib_symmDiff_term___u2206___00__closed__2));
lean_inc(v_x_80_);
v___x_84_ = l_Lean_Syntax_isOfKind(v_x_80_, v___x_83_);
if (v___x_84_ == 0)
{
lean_object* v___x_85_; lean_object* v___x_86_; 
lean_dec(v_x_80_);
v___x_85_ = lean_box(1);
v___x_86_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
lean_ctor_set(v___x_86_, 1, v_a_82_);
return v___x_86_;
}
else
{
lean_object* v_quotContext_87_; lean_object* v_currMacroScope_88_; lean_object* v_ref_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; uint8_t v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v_quotContext_87_ = lean_ctor_get(v_a_81_, 1);
v_currMacroScope_88_ = lean_ctor_get(v_a_81_, 2);
v_ref_89_ = lean_ctor_get(v_a_81_, 5);
v___x_90_ = lean_unsigned_to_nat(0u);
v___x_91_ = l_Lean_Syntax_getArg(v_x_80_, v___x_90_);
v___x_92_ = lean_unsigned_to_nat(2u);
v___x_93_ = l_Lean_Syntax_getArg(v_x_80_, v___x_92_);
lean_dec(v_x_80_);
v___x_94_ = 0;
v___x_95_ = l_Lean_SourceInfo_fromRef(v_ref_89_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4));
v___x_97_ = lean_obj_once(&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__5, &lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__5_once, _init_lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__5);
v___x_98_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__6));
lean_inc(v_currMacroScope_88_);
lean_inc(v_quotContext_87_);
v___x_99_ = l_Lean_addMacroScope(v_quotContext_87_, v___x_98_, v_currMacroScope_88_);
v___x_100_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__10));
lean_inc_n(v___x_95_, 2);
v___x_101_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_101_, 0, v___x_95_);
lean_ctor_set(v___x_101_, 1, v___x_97_);
lean_ctor_set(v___x_101_, 2, v___x_99_);
lean_ctor_set(v___x_101_, 3, v___x_100_);
v___x_102_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__12));
v___x_103_ = l_Lean_Syntax_node2(v___x_95_, v___x_102_, v___x_91_, v___x_93_);
v___x_104_ = l_Lean_Syntax_node2(v___x_95_, v___x_96_, v___x_101_, v___x_103_);
v___x_105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_82_);
return v___x_105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___boxed(lean_object* v_x_106_, lean_object* v_a_107_, lean_object* v_a_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1(v_x_106_, v_a_107_, v_a_108_);
lean_dec_ref(v_a_107_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1(lean_object* v_x_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4));
lean_inc(v_x_113_);
v___x_117_ = l_Lean_Syntax_isOfKind(v_x_113_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v_x_113_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v_a_115_);
return v___x_119_;
}
else
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_120_ = lean_unsigned_to_nat(0u);
v___x_121_ = l_Lean_Syntax_getArg(v_x_113_, v___x_120_);
v___x_122_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__1));
lean_inc(v___x_121_);
v___x_123_ = l_Lean_Syntax_isOfKind(v___x_121_, v___x_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_dec(v___x_121_);
lean_dec(v_x_113_);
v___x_124_ = lean_box(0);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_115_);
return v___x_125_;
}
else
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; uint8_t v___x_129_; 
v___x_126_ = lean_unsigned_to_nat(1u);
v___x_127_ = l_Lean_Syntax_getArg(v_x_113_, v___x_126_);
lean_dec(v_x_113_);
v___x_128_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_127_);
v___x_129_ = l_Lean_Syntax_matchesNull(v___x_127_, v___x_128_);
if (v___x_129_ == 0)
{
lean_object* v___x_130_; lean_object* v___x_131_; 
lean_dec(v___x_127_);
lean_dec(v___x_121_);
v___x_130_ = lean_box(0);
v___x_131_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
lean_ctor_set(v___x_131_, 1, v_a_115_);
return v___x_131_;
}
else
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v_ref_134_; uint8_t v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_132_ = l_Lean_Syntax_getArg(v___x_127_, v___x_120_);
v___x_133_ = l_Lean_Syntax_getArg(v___x_127_, v___x_126_);
lean_dec(v___x_127_);
v_ref_134_ = l_Lean_replaceRef(v___x_121_, v_a_114_);
lean_dec(v___x_121_);
v___x_135_ = 0;
v___x_136_ = l_Lean_SourceInfo_fromRef(v_ref_134_, v___x_135_);
lean_dec(v_ref_134_);
v___x_137_ = ((lean_object*)(lp_mathlib_symmDiff_term___u2206___00__closed__2));
v___x_138_ = ((lean_object*)(lp_mathlib_symmDiff_term___u2206___00__closed__5));
lean_inc(v___x_136_);
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_136_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = l_Lean_Syntax_node3(v___x_136_, v___x_137_, v___x_132_, v___x_139_, v___x_133_);
v___x_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
lean_ctor_set(v___x_141_, 1, v_a_115_);
return v___x_141_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___boxed(lean_object* v_x_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1(v_x_142_, v_a_143_, v_a_144_);
lean_dec(v_a_143_);
return v_res_145_;
}
}
static lean_object* _init_lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__1(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__0));
v___x_164_ = l_String_toRawSubstring_x27(v___x_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1(lean_object* v_x_173_, lean_object* v_a_174_, lean_object* v_a_175_){
_start:
{
lean_object* v___x_176_; uint8_t v___x_177_; 
v___x_176_ = ((lean_object*)(lp_mathlib_symmDiff_term___u21d4___00__closed__1));
lean_inc(v_x_173_);
v___x_177_ = l_Lean_Syntax_isOfKind(v_x_173_, v___x_176_);
if (v___x_177_ == 0)
{
lean_object* v___x_178_; lean_object* v___x_179_; 
lean_dec(v_x_173_);
v___x_178_ = lean_box(1);
v___x_179_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
lean_ctor_set(v___x_179_, 1, v_a_175_);
return v___x_179_;
}
else
{
lean_object* v_quotContext_180_; lean_object* v_currMacroScope_181_; lean_object* v_ref_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; uint8_t v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v_quotContext_180_ = lean_ctor_get(v_a_174_, 1);
v_currMacroScope_181_ = lean_ctor_get(v_a_174_, 2);
v_ref_182_ = lean_ctor_get(v_a_174_, 5);
v___x_183_ = lean_unsigned_to_nat(0u);
v___x_184_ = l_Lean_Syntax_getArg(v_x_173_, v___x_183_);
v___x_185_ = lean_unsigned_to_nat(2u);
v___x_186_ = l_Lean_Syntax_getArg(v_x_173_, v___x_185_);
lean_dec(v_x_173_);
v___x_187_ = 0;
v___x_188_ = l_Lean_SourceInfo_fromRef(v_ref_182_, v___x_187_);
v___x_189_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4));
v___x_190_ = lean_obj_once(&lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__1, &lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__1_once, _init_lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__1);
v___x_191_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__2));
lean_inc(v_currMacroScope_181_);
lean_inc(v_quotContext_180_);
v___x_192_ = l_Lean_addMacroScope(v_quotContext_180_, v___x_191_, v_currMacroScope_181_);
v___x_193_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___closed__4));
lean_inc_n(v___x_188_, 2);
v___x_194_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_194_, 0, v___x_188_);
lean_ctor_set(v___x_194_, 1, v___x_190_);
lean_ctor_set(v___x_194_, 2, v___x_192_);
lean_ctor_set(v___x_194_, 3, v___x_193_);
v___x_195_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__12));
v___x_196_ = l_Lean_Syntax_node2(v___x_188_, v___x_195_, v___x_184_, v___x_186_);
v___x_197_ = l_Lean_Syntax_node2(v___x_188_, v___x_189_, v___x_194_, v___x_196_);
v___x_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set(v___x_198_, 1, v_a_175_);
return v___x_198_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1___boxed(lean_object* v_x_199_, lean_object* v_a_200_, lean_object* v_a_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u21d4____1(v_x_199_, v_a_200_, v_a_201_);
lean_dec_ref(v_a_200_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__bihimp__1(lean_object* v_x_203_, lean_object* v_a_204_, lean_object* v_a_205_){
_start:
{
lean_object* v___x_206_; uint8_t v___x_207_; 
v___x_206_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______macroRules__symmDiff__term___u2206____1___closed__4));
lean_inc(v_x_203_);
v___x_207_ = l_Lean_Syntax_isOfKind(v_x_203_, v___x_206_);
if (v___x_207_ == 0)
{
lean_object* v___x_208_; lean_object* v___x_209_; 
lean_dec(v_x_203_);
v___x_208_ = lean_box(0);
v___x_209_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v_a_205_);
return v___x_209_;
}
else
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; uint8_t v___x_213_; 
v___x_210_ = lean_unsigned_to_nat(0u);
v___x_211_ = l_Lean_Syntax_getArg(v_x_203_, v___x_210_);
v___x_212_ = ((lean_object*)(lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__symmDiff__1___closed__1));
lean_inc(v___x_211_);
v___x_213_ = l_Lean_Syntax_isOfKind(v___x_211_, v___x_212_);
if (v___x_213_ == 0)
{
lean_object* v___x_214_; lean_object* v___x_215_; 
lean_dec(v___x_211_);
lean_dec(v_x_203_);
v___x_214_ = lean_box(0);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_a_205_);
return v___x_215_;
}
else
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; uint8_t v___x_219_; 
v___x_216_ = lean_unsigned_to_nat(1u);
v___x_217_ = l_Lean_Syntax_getArg(v_x_203_, v___x_216_);
lean_dec(v_x_203_);
v___x_218_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_217_);
v___x_219_ = l_Lean_Syntax_matchesNull(v___x_217_, v___x_218_);
if (v___x_219_ == 0)
{
lean_object* v___x_220_; lean_object* v___x_221_; 
lean_dec(v___x_217_);
lean_dec(v___x_211_);
v___x_220_ = lean_box(0);
v___x_221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set(v___x_221_, 1, v_a_205_);
return v___x_221_;
}
else
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v_ref_224_; uint8_t v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_222_ = l_Lean_Syntax_getArg(v___x_217_, v___x_210_);
v___x_223_ = l_Lean_Syntax_getArg(v___x_217_, v___x_216_);
lean_dec(v___x_217_);
v_ref_224_ = l_Lean_replaceRef(v___x_211_, v_a_204_);
lean_dec(v___x_211_);
v___x_225_ = 0;
v___x_226_ = l_Lean_SourceInfo_fromRef(v_ref_224_, v___x_225_);
lean_dec(v_ref_224_);
v___x_227_ = ((lean_object*)(lp_mathlib_symmDiff_term___u21d4___00__closed__1));
v___x_228_ = ((lean_object*)(lp_mathlib_symmDiff_term___u21d4___00__closed__2));
lean_inc(v___x_226_);
v___x_229_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_226_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v___x_230_ = l_Lean_Syntax_node3(v___x_226_, v___x_227_, v___x_222_, v___x_229_, v___x_223_);
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
lean_ctor_set(v___x_231_, 1, v_a_205_);
return v___x_231_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__bihimp__1___boxed(lean_object* v_x_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_symmDiff___aux__Mathlib__Order__SymmDiff______unexpand__bihimp__1(v_x_232_, v_a_233_, v_a_234_);
lean_dec(v_a_233_);
return v_res_235_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
}
#ifdef __cplusplus
}
#endif
