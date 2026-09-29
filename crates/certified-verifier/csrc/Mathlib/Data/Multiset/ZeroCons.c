// Lean compiler output
// Module: Mathlib.Data.Multiset.ZeroCons
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.Defs public import Mathlib.Order.BoundedOrder.Basic
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_zero(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instZero(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instEmptyCollection(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inhabitedMultiset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cons(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Multiset"};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__0 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__0_value;
static const lean_string_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_::ₘ_"};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__1 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 131, 115, 119, 79, 192, 198, 77)}};
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(135, 107, 62, 250, 10, 130, 78, 193)}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2_value;
static const lean_string_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__3 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__4 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__4_value;
static const lean_string_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ::ₘ "};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__5 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__5_value)}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__6 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__6_value;
static const lean_string_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__7 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__8 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__8_value),((lean_object*)(((size_t)(67) << 1) | 1))}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__9 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__4_value),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__6_value),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__9_value)}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__10 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2_value),((lean_object*)(((size_t)(67) << 1) | 1)),((lean_object*)(((size_t)(68) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__10_value)}};
static const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__11 = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Multiset_term___x3a_x3a_u2098__ = (const lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__11_value;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__0 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__0_value;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__1 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__1_value;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__2 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__2_value;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__3 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4_value;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Multiset.cons"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__5 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__6;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__7 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 131, 115, 119, 79, 192, 198, 77)}};
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 130, 88, 150, 183, 113, 109, 83)}};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__9 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__10 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__10_value;
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__11 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__12 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__0 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__1 = (const lean_object*)&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_instInsert___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_cons, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_instInsert___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instInsert___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instInsert(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSingleton___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Multiset_instSingleton___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_instSingleton___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_instSingleton___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instSingleton___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSingleton(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instOrderBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_zero(lean_object* v_00_u03b1_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instZero(lean_object* v_00_u03b1_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instEmptyCollection(lean_object* v_00_u03b1_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_inhabitedMultiset(lean_object* v_00_u03b1_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instUniqueOfIsEmpty(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_box(0);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cons___redArg(lean_object* v_a_12_, lean_object* v_s_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_14_, 0, v_a_12_);
lean_ctor_set(v___x_14_, 1, v_s_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_cons(lean_object* v_00_u03b1_15_, lean_object* v_a_16_, lean_object* v_s_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_18_, 0, v_a_16_);
lean_ctor_set(v___x_18_, 1, v_s_17_);
return v___x_18_;
}
}
static lean_object* _init_lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__6(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__5));
v___x_57_ = l_String_toRawSubstring_x27(v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1(lean_object* v_x_71_, lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; 
v___x_74_ = ((lean_object*)(lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2));
lean_inc(v_x_71_);
v___x_75_ = l_Lean_Syntax_isOfKind(v_x_71_, v___x_74_);
if (v___x_75_ == 0)
{
lean_object* v___x_76_; lean_object* v___x_77_; 
lean_dec(v_x_71_);
v___x_76_ = lean_box(1);
v___x_77_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v_a_73_);
return v___x_77_;
}
else
{
lean_object* v_quotContext_78_; lean_object* v_currMacroScope_79_; lean_object* v_ref_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; uint8_t v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v_quotContext_78_ = lean_ctor_get(v_a_72_, 1);
v_currMacroScope_79_ = lean_ctor_get(v_a_72_, 2);
v_ref_80_ = lean_ctor_get(v_a_72_, 5);
v___x_81_ = lean_unsigned_to_nat(0u);
v___x_82_ = l_Lean_Syntax_getArg(v_x_71_, v___x_81_);
v___x_83_ = lean_unsigned_to_nat(2u);
v___x_84_ = l_Lean_Syntax_getArg(v_x_71_, v___x_83_);
lean_dec(v_x_71_);
v___x_85_ = 0;
v___x_86_ = l_Lean_SourceInfo_fromRef(v_ref_80_, v___x_85_);
v___x_87_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4));
v___x_88_ = lean_obj_once(&lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__6, &lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__6_once, _init_lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__6);
v___x_89_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__8));
lean_inc(v_currMacroScope_79_);
lean_inc(v_quotContext_78_);
v___x_90_ = l_Lean_addMacroScope(v_quotContext_78_, v___x_89_, v_currMacroScope_79_);
v___x_91_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__10));
lean_inc_n(v___x_86_, 2);
v___x_92_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_92_, 0, v___x_86_);
lean_ctor_set(v___x_92_, 1, v___x_88_);
lean_ctor_set(v___x_92_, 2, v___x_90_);
lean_ctor_set(v___x_92_, 3, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__12));
v___x_94_ = l_Lean_Syntax_node2(v___x_86_, v___x_93_, v___x_82_, v___x_84_);
v___x_95_ = l_Lean_Syntax_node2(v___x_86_, v___x_87_, v___x_92_, v___x_94_);
v___x_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_73_);
return v___x_96_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___boxed(lean_object* v_x_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1(v_x_97_, v_a_98_, v_a_99_);
lean_dec_ref(v_a_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1(lean_object* v_x_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v___x_107_; uint8_t v___x_108_; 
v___x_107_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______macroRules__Multiset__term___x3a_x3a_u2098____1___closed__4));
lean_inc(v_x_104_);
v___x_108_ = l_Lean_Syntax_isOfKind(v_x_104_, v___x_107_);
if (v___x_108_ == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v_x_104_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_a_106_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = l_Lean_Syntax_getArg(v_x_104_, v___x_111_);
v___x_113_ = ((lean_object*)(lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___closed__1));
lean_inc(v___x_112_);
v___x_114_ = l_Lean_Syntax_isOfKind(v___x_112_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v___x_112_);
lean_dec(v_x_104_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_106_);
return v___x_116_;
}
else
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_117_ = lean_unsigned_to_nat(1u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_104_, v___x_117_);
lean_dec(v_x_104_);
v___x_119_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_118_);
v___x_120_ = l_Lean_Syntax_matchesNull(v___x_118_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v___x_118_);
lean_dec(v___x_112_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_106_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v_ref_125_; uint8_t v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_123_ = l_Lean_Syntax_getArg(v___x_118_, v___x_111_);
v___x_124_ = l_Lean_Syntax_getArg(v___x_118_, v___x_117_);
lean_dec(v___x_118_);
v_ref_125_ = l_Lean_replaceRef(v___x_112_, v_a_105_);
lean_dec(v___x_112_);
v___x_126_ = 0;
v___x_127_ = l_Lean_SourceInfo_fromRef(v_ref_125_, v___x_126_);
lean_dec(v_ref_125_);
v___x_128_ = ((lean_object*)(lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__2));
v___x_129_ = ((lean_object*)(lp_mathlib_Multiset_term___x3a_x3a_u2098___00__closed__5));
lean_inc(v___x_127_);
v___x_130_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_127_);
lean_ctor_set(v___x_130_, 1, v___x_129_);
v___x_131_ = l_Lean_Syntax_node3(v___x_127_, v___x_128_, v___x_123_, v___x_130_, v___x_124_);
v___x_132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v_a_106_);
return v___x_132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1___boxed(lean_object* v_x_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_Multiset___aux__Mathlib__Data__Multiset__ZeroCons______unexpand__Multiset__cons__1(v_x_133_, v_a_134_, v_a_135_);
lean_dec(v_a_134_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instInsert(lean_object* v_00_u03b1_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = ((lean_object*)(lp_mathlib_Multiset_instInsert___closed__0));
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___redArg___lam__0(lean_object* v_C__cons_140_, lean_object* v_a_141_, lean_object* v_l_142_, lean_object* v_b_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lean_apply_3(v_C__cons_140_, v_a_141_, v_l_142_, v_b_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___redArg(lean_object* v_C__0_145_, lean_object* v_C__cons_146_, lean_object* v_m_147_){
_start:
{
lean_object* v___f_148_; lean_object* v___x_149_; 
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_rec___redArg___lam__0), 4, 1);
lean_closure_set(v___f_148_, 0, v_C__cons_146_);
v___x_149_ = lp_mathlib_List_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_3_(v_C__0_145_, v___f_148_, v_m_147_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___redArg___boxed(lean_object* v_C__0_150_, lean_object* v_C__cons_151_, lean_object* v_m_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_Multiset_rec___redArg(v_C__0_150_, v_C__cons_151_, v_m_152_);
lean_dec(v_C__0_150_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec(lean_object* v_00_u03b1_154_, lean_object* v_C_155_, lean_object* v_C__0_156_, lean_object* v_C__cons_157_, lean_object* v_C__cons__heq_158_, lean_object* v_m_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_Multiset_rec___redArg(v_C__0_156_, v_C__cons_157_, v_m_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_rec___boxed(lean_object* v_00_u03b1_161_, lean_object* v_C_162_, lean_object* v_C__0_163_, lean_object* v_C__cons_164_, lean_object* v_C__cons__heq_165_, lean_object* v_m_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Multiset_rec(v_00_u03b1_161_, v_C_162_, v_C__0_163_, v_C__cons_164_, v_C__cons__heq_165_, v_m_166_);
lean_dec(v_C__0_163_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn___redArg(lean_object* v_m_168_, lean_object* v_C__0_169_, lean_object* v_C__cons_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_Multiset_rec___redArg(v_C__0_169_, v_C__cons_170_, v_m_168_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn___redArg___boxed(lean_object* v_m_172_, lean_object* v_C__0_173_, lean_object* v_C__cons_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_Multiset_recOn___redArg(v_m_172_, v_C__0_173_, v_C__cons_174_);
lean_dec(v_C__0_173_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn(lean_object* v_00_u03b1_176_, lean_object* v_C_177_, lean_object* v_m_178_, lean_object* v_C__0_179_, lean_object* v_C__cons_180_, lean_object* v_C__cons__heq_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_Multiset_rec___redArg(v_C__0_179_, v_C__cons_180_, v_m_178_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_recOn___boxed(lean_object* v_00_u03b1_183_, lean_object* v_C_184_, lean_object* v_m_185_, lean_object* v_C__0_186_, lean_object* v_C__cons_187_, lean_object* v_C__cons__heq_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Multiset_recOn(v_00_u03b1_183_, v_C_184_, v_m_185_, v_C__0_186_, v_C__cons_187_, v_C__cons__heq_188_);
lean_dec(v_C__0_186_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSingleton___lam__0(lean_object* v_a_190_){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_191_ = lean_box(0);
v___x_192_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_192_, 0, v_a_190_);
lean_ctor_set(v___x_192_, 1, v___x_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSingleton(lean_object* v_00_u03b1_194_){
_start:
{
lean_object* v___f_195_; 
v___f_195_ = ((lean_object*)(lp_mathlib_Multiset_instSingleton___closed__0));
return v___f_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instOrderBot(lean_object* v_00_u03b1_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lean_box(0);
return v___x_197_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
}
#ifdef __cplusplus
}
#endif
