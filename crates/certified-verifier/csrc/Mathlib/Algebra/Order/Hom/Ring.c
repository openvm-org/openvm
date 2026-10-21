// Lean compiler output
// Module: Mathlib.Algebra.Order.Hom.Ring
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Hom.MonoidWithZero public import Mathlib.Algebra.Ring.Equiv
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
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 10, .m_data = "term_→+*o_"};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(229, 58, 107, 71, 122, 205, 7, 118)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = " →+*o "};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__7_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192_x2b_x2ao___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192_x2b_x2ao___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_x2b_x2ao__ = (const lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "OrderRingHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(143, 167, 160, 189, 155, 52, 22, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_x2b_x2ao___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 10, .m_data = "term_≃+*o_"};
static const lean_object* lp_mathlib_term___u2243_x2b_x2ao___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2ao___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 44, 57, 97, 130, 134, 248, 3)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2ao___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_x2b_x2ao___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = " ≃+*o "};
static const lean_object* lp_mathlib_term___u2243_x2b_x2ao___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2ao___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2ao___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2ao___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_x2b_x2ao___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2ao___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2ao___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2ao___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_x2b_x2ao__ = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2ao___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "OrderRingIso"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(137, 119, 0, 32, 160, 3, 191, 68)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingIso__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingIso__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHomClass_toOrderRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHomClass_toOrderRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHomClass_toOrderRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingHomOfOrderHomClassOfRingHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingHomOfOrderHomClassOfRingHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIsoClass_toOrderRingIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIsoClass_toOrderRingIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIsoClass_toOrderRingIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingIsoOfOrderIsoClassOfRingEquivClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingIsoOfOrderIsoClassOfRingEquivClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderRingHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderRingHom_id___closed__0 = (const lean_object*)&lp_mathlib_OrderRingHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_OrderRingHom_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OrderRingHom_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_OrderRingHom_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_OrderRingIso_instCoeOutRingEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderRingIso_instCoeOutRingEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___closed__0 = (const lean_object*)&lp_mathlib_OrderRingIso_instCoeOutRingEquiv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderRingIso_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderRingIso_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_refl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_refl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__6(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__5));
v___x_36_ = l_String_toRawSubstring_x27(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1(lean_object* v_x_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_term___u2192_x2b_x2ao___00__closed__1));
lean_inc(v_x_53_);
v___x_57_ = l_Lean_Syntax_isOfKind(v_x_53_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_x_53_);
v___x_58_ = lean_box(1);
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_55_);
return v___x_59_;
}
else
{
lean_object* v_quotContext_60_; lean_object* v_currMacroScope_61_; lean_object* v_ref_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v_quotContext_60_ = lean_ctor_get(v_a_54_, 1);
v_currMacroScope_61_ = lean_ctor_get(v_a_54_, 2);
v_ref_62_ = lean_ctor_get(v_a_54_, 5);
v___x_63_ = lean_unsigned_to_nat(0u);
v___x_64_ = l_Lean_Syntax_getArg(v_x_53_, v___x_63_);
v___x_65_ = lean_unsigned_to_nat(2u);
v___x_66_ = l_Lean_Syntax_getArg(v_x_53_, v___x_65_);
lean_dec(v_x_53_);
v___x_67_ = 0;
v___x_68_ = l_Lean_SourceInfo_fromRef(v_ref_62_, v___x_67_);
v___x_69_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4));
v___x_70_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__6);
v___x_71_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__7));
lean_inc(v_currMacroScope_61_);
lean_inc(v_quotContext_60_);
v___x_72_ = l_Lean_addMacroScope(v_quotContext_60_, v___x_71_, v_currMacroScope_61_);
v___x_73_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__11));
lean_inc_n(v___x_68_, 2);
v___x_74_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_74_, 0, v___x_68_);
lean_ctor_set(v___x_74_, 1, v___x_70_);
lean_ctor_set(v___x_74_, 2, v___x_72_);
lean_ctor_set(v___x_74_, 3, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__13));
v___x_76_ = l_Lean_Syntax_node2(v___x_68_, v___x_75_, v___x_64_, v___x_66_);
v___x_77_ = l_Lean_Syntax_node2(v___x_68_, v___x_69_, v___x_74_, v___x_76_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_55_);
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___boxed(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1(v_x_79_, v_a_80_, v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1(lean_object* v_x_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4));
lean_inc(v_x_86_);
v___x_90_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_89_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; lean_object* v___x_92_; 
lean_dec(v_x_86_);
v___x_91_ = lean_box(0);
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v_a_88_);
return v___x_92_;
}
else
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_93_ = lean_unsigned_to_nat(0u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_86_, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__1));
lean_inc(v___x_94_);
v___x_96_ = l_Lean_Syntax_isOfKind(v___x_94_, v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; 
lean_dec(v___x_94_);
lean_dec(v_x_86_);
v___x_97_ = lean_box(0);
v___x_98_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_88_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_99_ = lean_unsigned_to_nat(1u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_86_, v___x_99_);
lean_dec(v_x_86_);
v___x_101_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_100_);
v___x_102_ = l_Lean_Syntax_matchesNull(v___x_100_, v___x_101_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec(v___x_100_);
lean_dec(v___x_94_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v_a_88_);
return v___x_104_;
}
else
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v_ref_107_; uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_105_ = l_Lean_Syntax_getArg(v___x_100_, v___x_93_);
v___x_106_ = l_Lean_Syntax_getArg(v___x_100_, v___x_99_);
lean_dec(v___x_100_);
v_ref_107_ = l_Lean_replaceRef(v___x_94_, v_a_87_);
lean_dec(v___x_94_);
v___x_108_ = 0;
v___x_109_ = l_Lean_SourceInfo_fromRef(v_ref_107_, v___x_108_);
lean_dec(v_ref_107_);
v___x_110_ = ((lean_object*)(lp_mathlib_term___u2192_x2b_x2ao___00__closed__1));
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192_x2b_x2ao___00__closed__4));
lean_inc(v___x_109_);
v___x_112_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_109_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
v___x_113_ = l_Lean_Syntax_node3(v___x_109_, v___x_110_, v___x_105_, v___x_112_, v___x_106_);
v___x_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_a_88_);
return v___x_114_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___boxed(lean_object* v_x_115_, lean_object* v_a_116_, lean_object* v_a_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1(v_x_115_, v_a_116_, v_a_117_);
lean_dec(v_a_116_);
return v_res_118_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__1(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__0));
v___x_136_ = l_String_toRawSubstring_x27(v___x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1(lean_object* v_x_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v___x_153_; uint8_t v___x_154_; 
v___x_153_ = ((lean_object*)(lp_mathlib_term___u2243_x2b_x2ao___00__closed__1));
lean_inc(v_x_150_);
v___x_154_ = l_Lean_Syntax_isOfKind(v_x_150_, v___x_153_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; lean_object* v___x_156_; 
lean_dec(v_x_150_);
v___x_155_ = lean_box(1);
v___x_156_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
lean_ctor_set(v___x_156_, 1, v_a_152_);
return v___x_156_;
}
else
{
lean_object* v_quotContext_157_; lean_object* v_currMacroScope_158_; lean_object* v_ref_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; uint8_t v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v_quotContext_157_ = lean_ctor_get(v_a_151_, 1);
v_currMacroScope_158_ = lean_ctor_get(v_a_151_, 2);
v_ref_159_ = lean_ctor_get(v_a_151_, 5);
v___x_160_ = lean_unsigned_to_nat(0u);
v___x_161_ = l_Lean_Syntax_getArg(v_x_150_, v___x_160_);
v___x_162_ = lean_unsigned_to_nat(2u);
v___x_163_ = l_Lean_Syntax_getArg(v_x_150_, v___x_162_);
lean_dec(v_x_150_);
v___x_164_ = 0;
v___x_165_ = l_Lean_SourceInfo_fromRef(v_ref_159_, v___x_164_);
v___x_166_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4));
v___x_167_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__1);
v___x_168_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__2));
lean_inc(v_currMacroScope_158_);
lean_inc(v_quotContext_157_);
v___x_169_ = l_Lean_addMacroScope(v_quotContext_157_, v___x_168_, v_currMacroScope_158_);
v___x_170_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___closed__6));
lean_inc_n(v___x_165_, 2);
v___x_171_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_171_, 0, v___x_165_);
lean_ctor_set(v___x_171_, 1, v___x_167_);
lean_ctor_set(v___x_171_, 2, v___x_169_);
lean_ctor_set(v___x_171_, 3, v___x_170_);
v___x_172_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__13));
v___x_173_ = l_Lean_Syntax_node2(v___x_165_, v___x_172_, v___x_161_, v___x_163_);
v___x_174_ = l_Lean_Syntax_node2(v___x_165_, v___x_166_, v___x_171_, v___x_173_);
v___x_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set(v___x_175_, 1, v_a_152_);
return v___x_175_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1___boxed(lean_object* v_x_176_, lean_object* v_a_177_, lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2243_x2b_x2ao____1(v_x_176_, v_a_177_, v_a_178_);
lean_dec_ref(v_a_177_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingIso__1(lean_object* v_x_180_, lean_object* v_a_181_, lean_object* v_a_182_){
_start:
{
lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_183_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______macroRules__term___u2192_x2b_x2ao____1___closed__4));
lean_inc(v_x_180_);
v___x_184_ = l_Lean_Syntax_isOfKind(v_x_180_, v___x_183_);
if (v___x_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; 
lean_dec(v_x_180_);
v___x_185_ = lean_box(0);
v___x_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v_a_182_);
return v___x_186_;
}
else
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; uint8_t v___x_190_; 
v___x_187_ = lean_unsigned_to_nat(0u);
v___x_188_ = l_Lean_Syntax_getArg(v_x_180_, v___x_187_);
v___x_189_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingHom__1___closed__1));
lean_inc(v___x_188_);
v___x_190_ = l_Lean_Syntax_isOfKind(v___x_188_, v___x_189_);
if (v___x_190_ == 0)
{
lean_object* v___x_191_; lean_object* v___x_192_; 
lean_dec(v___x_188_);
lean_dec(v_x_180_);
v___x_191_ = lean_box(0);
v___x_192_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_192_, 0, v___x_191_);
lean_ctor_set(v___x_192_, 1, v_a_182_);
return v___x_192_;
}
else
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; uint8_t v___x_196_; 
v___x_193_ = lean_unsigned_to_nat(1u);
v___x_194_ = l_Lean_Syntax_getArg(v_x_180_, v___x_193_);
lean_dec(v_x_180_);
v___x_195_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_194_);
v___x_196_ = l_Lean_Syntax_matchesNull(v___x_194_, v___x_195_);
if (v___x_196_ == 0)
{
lean_object* v___x_197_; lean_object* v___x_198_; 
lean_dec(v___x_194_);
lean_dec(v___x_188_);
v___x_197_ = lean_box(0);
v___x_198_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set(v___x_198_, 1, v_a_182_);
return v___x_198_;
}
else
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v_ref_201_; uint8_t v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_199_ = l_Lean_Syntax_getArg(v___x_194_, v___x_187_);
v___x_200_ = l_Lean_Syntax_getArg(v___x_194_, v___x_193_);
lean_dec(v___x_194_);
v_ref_201_ = l_Lean_replaceRef(v___x_188_, v_a_181_);
lean_dec(v___x_188_);
v___x_202_ = 0;
v___x_203_ = l_Lean_SourceInfo_fromRef(v_ref_201_, v___x_202_);
lean_dec(v_ref_201_);
v___x_204_ = ((lean_object*)(lp_mathlib_term___u2243_x2b_x2ao___00__closed__1));
v___x_205_ = ((lean_object*)(lp_mathlib_term___u2243_x2b_x2ao___00__closed__2));
lean_inc(v___x_203_);
v___x_206_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_203_);
lean_ctor_set(v___x_206_, 1, v___x_205_);
v___x_207_ = l_Lean_Syntax_node3(v___x_203_, v___x_204_, v___x_199_, v___x_206_, v___x_200_);
v___x_208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_182_);
return v___x_208_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingIso__1___boxed(lean_object* v_x_209_, lean_object* v_a_210_, lean_object* v_a_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib___aux__Mathlib__Algebra__Order__Hom__Ring______unexpand__OrderRingIso__1(v_x_209_, v_a_210_, v_a_211_);
lean_dec(v_a_210_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHomClass_toOrderRingHom___redArg(lean_object* v_inst_213_, lean_object* v_f_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lean_apply_1(v_inst_213_, v_f_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHomClass_toOrderRingHom(lean_object* v_F_216_, lean_object* v_00_u03b1_217_, lean_object* v_00_u03b2_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_f_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lean_apply_1(v_inst_219_, v_f_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHomClass_toOrderRingHom___boxed(lean_object* v_F_228_, lean_object* v_00_u03b1_229_, lean_object* v_00_u03b2_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_f_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_OrderRingHomClass_toOrderRingHom(v_F_228_, v_00_u03b1_229_, v_00_u03b2_230_, v_inst_231_, v_inst_232_, v_inst_233_, v_inst_234_, v_inst_235_, v_inst_236_, v_inst_237_, v_f_238_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v_inst_234_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingHomOfOrderHomClassOfRingHomClass___redArg(lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_alloc_closure((void*)(lp_mathlib_OrderRingHomClass_toOrderRingHom___boxed), 11, 10);
lean_closure_set(v___x_245_, 0, lean_box(0));
lean_closure_set(v___x_245_, 1, lean_box(0));
lean_closure_set(v___x_245_, 2, lean_box(0));
lean_closure_set(v___x_245_, 3, v_inst_240_);
lean_closure_set(v___x_245_, 4, v_inst_241_);
lean_closure_set(v___x_245_, 5, v_inst_242_);
lean_closure_set(v___x_245_, 6, v_inst_243_);
lean_closure_set(v___x_245_, 7, v_inst_244_);
lean_closure_set(v___x_245_, 8, lean_box(0));
lean_closure_set(v___x_245_, 9, lean_box(0));
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingHomOfOrderHomClassOfRingHomClass(lean_object* v_F_246_, lean_object* v_00_u03b1_247_, lean_object* v_00_u03b2_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lean_alloc_closure((void*)(lp_mathlib_OrderRingHomClass_toOrderRingHom___boxed), 11, 10);
lean_closure_set(v___x_256_, 0, lean_box(0));
lean_closure_set(v___x_256_, 1, lean_box(0));
lean_closure_set(v___x_256_, 2, lean_box(0));
lean_closure_set(v___x_256_, 3, v_inst_249_);
lean_closure_set(v___x_256_, 4, v_inst_250_);
lean_closure_set(v___x_256_, 5, v_inst_251_);
lean_closure_set(v___x_256_, 6, v_inst_252_);
lean_closure_set(v___x_256_, 7, v_inst_253_);
lean_closure_set(v___x_256_, 8, lean_box(0));
lean_closure_set(v___x_256_, 9, lean_box(0));
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIsoClass_toOrderRingIso___redArg(lean_object* v_inst_257_, lean_object* v_f_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_257_, v_f_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIsoClass_toOrderRingIso(lean_object* v_F_260_, lean_object* v_00_u03b1_261_, lean_object* v_00_u03b2_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_f_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_263_, v_f_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIsoClass_toOrderRingIso___boxed(lean_object* v_F_274_, lean_object* v_00_u03b1_275_, lean_object* v_00_u03b2_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_f_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_OrderRingIsoClass_toOrderRingIso(v_F_274_, v_00_u03b1_275_, v_00_u03b2_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_inst_281_, v_inst_282_, v_inst_283_, v_inst_284_, v_inst_285_, v_f_286_);
lean_dec(v_inst_282_);
lean_dec(v_inst_281_);
lean_dec(v_inst_279_);
lean_dec(v_inst_278_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingIsoOfOrderIsoClassOfRingEquivClass___redArg(lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_inst_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lean_alloc_closure((void*)(lp_mathlib_OrderRingIsoClass_toOrderRingIso___boxed), 13, 12);
lean_closure_set(v___x_295_, 0, lean_box(0));
lean_closure_set(v___x_295_, 1, lean_box(0));
lean_closure_set(v___x_295_, 2, lean_box(0));
lean_closure_set(v___x_295_, 3, v_inst_288_);
lean_closure_set(v___x_295_, 4, v_inst_289_);
lean_closure_set(v___x_295_, 5, v_inst_290_);
lean_closure_set(v___x_295_, 6, v_inst_291_);
lean_closure_set(v___x_295_, 7, v_inst_292_);
lean_closure_set(v___x_295_, 8, v_inst_293_);
lean_closure_set(v___x_295_, 9, v_inst_294_);
lean_closure_set(v___x_295_, 10, lean_box(0));
lean_closure_set(v___x_295_, 11, lean_box(0));
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderRingIsoOfOrderIsoClassOfRingEquivClass(lean_object* v_F_296_, lean_object* v_00_u03b1_297_, lean_object* v_00_u03b2_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lean_alloc_closure((void*)(lp_mathlib_OrderRingIsoClass_toOrderRingIso___boxed), 13, 12);
lean_closure_set(v___x_308_, 0, lean_box(0));
lean_closure_set(v___x_308_, 1, lean_box(0));
lean_closure_set(v___x_308_, 2, lean_box(0));
lean_closure_set(v___x_308_, 3, v_inst_299_);
lean_closure_set(v___x_308_, 4, v_inst_300_);
lean_closure_set(v___x_308_, 5, v_inst_301_);
lean_closure_set(v___x_308_, 6, v_inst_302_);
lean_closure_set(v___x_308_, 7, v_inst_303_);
lean_closure_set(v___x_308_, 8, v_inst_304_);
lean_closure_set(v___x_308_, 9, v_inst_305_);
lean_closure_set(v___x_308_, 10, lean_box(0));
lean_closure_set(v___x_308_, 11, lean_box(0));
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom___redArg(lean_object* v_f_309_){
_start:
{
lean_inc(v_f_309_);
return v_f_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom___redArg___boxed(lean_object* v_f_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_OrderRingHom_toOrderAddMonoidHom___redArg(v_f_310_);
lean_dec(v_f_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom(lean_object* v_00_u03b1_312_, lean_object* v_00_u03b2_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_f_318_){
_start:
{
lean_inc(v_f_318_);
return v_f_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderAddMonoidHom___boxed(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_f_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_OrderRingHom_toOrderAddMonoidHom(v_00_u03b1_319_, v_00_u03b2_320_, v_inst_321_, v_inst_322_, v_inst_323_, v_inst_324_, v_f_325_);
lean_dec(v_f_325_);
lean_dec_ref(v_inst_324_);
lean_dec_ref(v_inst_323_);
lean_dec_ref(v_inst_322_);
lean_dec_ref(v_inst_321_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___redArg(lean_object* v_f_327_){
_start:
{
lean_inc(v_f_327_);
return v_f_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___redArg___boxed(lean_object* v_f_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___redArg(v_f_328_);
lean_dec(v_f_328_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom(lean_object* v_00_u03b1_330_, lean_object* v_00_u03b2_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_f_336_){
_start:
{
lean_inc(v_f_336_);
return v_f_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom___boxed(lean_object* v_00_u03b1_337_, lean_object* v_00_u03b2_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_f_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_OrderRingHom_toOrderMonoidWithZeroHom(v_00_u03b1_337_, v_00_u03b2_338_, v_inst_339_, v_inst_340_, v_inst_341_, v_inst_342_, v_f_343_);
lean_dec(v_f_343_);
lean_dec_ref(v_inst_342_);
lean_dec_ref(v_inst_341_);
lean_dec_ref(v_inst_340_);
lean_dec_ref(v_inst_339_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy___redArg(lean_object* v_f_x27_345_){
_start:
{
lean_inc(v_f_x27_345_);
return v_f_x27_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy___redArg___boxed(lean_object* v_f_x27_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_OrderRingHom_copy___redArg(v_f_x27_346_);
lean_dec(v_f_x27_346_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy(lean_object* v_00_u03b1_348_, lean_object* v_00_u03b2_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_f_354_, lean_object* v_f_x27_355_, lean_object* v_h_356_){
_start:
{
lean_inc(v_f_x27_355_);
return v_f_x27_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_copy___boxed(lean_object* v_00_u03b1_357_, lean_object* v_00_u03b2_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_f_363_, lean_object* v_f_x27_364_, lean_object* v_h_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_OrderRingHom_copy(v_00_u03b1_357_, v_00_u03b2_358_, v_inst_359_, v_inst_360_, v_inst_361_, v_inst_362_, v_f_363_, v_f_x27_364_, v_h_365_);
lean_dec(v_f_x27_364_);
lean_dec(v_f_363_);
lean_dec_ref(v_inst_362_);
lean_dec_ref(v_inst_361_);
lean_dec_ref(v_inst_360_);
lean_dec_ref(v_inst_359_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_id(lean_object* v_00_u03b1_368_, lean_object* v_inst_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = ((lean_object*)(lp_mathlib_OrderRingHom_id___closed__0));
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_id___boxed(lean_object* v_00_u03b1_372_, lean_object* v_inst_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib_OrderRingHom_id(v_00_u03b1_372_, v_inst_373_, v_inst_374_);
lean_dec_ref(v_inst_374_);
lean_dec_ref(v_inst_373_);
return v_res_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instInhabited(lean_object* v_00_u03b1_376_, lean_object* v_inst_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___f_379_; 
v___f_379_ = ((lean_object*)(lp_mathlib_OrderRingHom_id___closed__0));
return v___f_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instInhabited___boxed(lean_object* v_00_u03b1_380_, lean_object* v_inst_381_, lean_object* v_inst_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_OrderRingHom_instInhabited(v_00_u03b1_380_, v_inst_381_, v_inst_382_);
lean_dec_ref(v_inst_382_);
lean_dec_ref(v_inst_381_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_comp___redArg(lean_object* v_f_384_, lean_object* v_g_385_){
_start:
{
lean_object* v___f_386_; 
v___f_386_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_386_, 0, v_g_385_);
lean_closure_set(v___f_386_, 1, v_f_384_);
return v___f_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_comp(lean_object* v_00_u03b1_387_, lean_object* v_00_u03b2_388_, lean_object* v_00_u03b3_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_f_396_, lean_object* v_g_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_398_, 0, v_g_397_);
lean_closure_set(v___f_398_, 1, v_f_396_);
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_comp___boxed(lean_object* v_00_u03b1_399_, lean_object* v_00_u03b2_400_, lean_object* v_00_u03b3_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_f_408_, lean_object* v_g_409_){
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_OrderRingHom_comp(v_00_u03b1_399_, v_00_u03b2_400_, v_00_u03b3_401_, v_inst_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_inst_407_, v_f_408_, v_g_409_);
lean_dec_ref(v_inst_407_);
lean_dec_ref(v_inst_406_);
lean_dec_ref(v_inst_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_inst_402_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPreorder(lean_object* v_00_u03b1_414_, lean_object* v_00_u03b2_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = ((lean_object*)(lp_mathlib_OrderRingHom_instPreorder___closed__0));
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPreorder___boxed(lean_object* v_00_u03b1_421_, lean_object* v_00_u03b2_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_OrderRingHom_instPreorder(v_00_u03b1_421_, v_00_u03b2_422_, v_inst_423_, v_inst_424_, v_inst_425_, v_inst_426_);
lean_dec_ref(v_inst_426_);
lean_dec_ref(v_inst_425_);
lean_dec_ref(v_inst_424_);
lean_dec_ref(v_inst_423_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPartialOrder(lean_object* v_00_u03b1_428_, lean_object* v_00_u03b2_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = ((lean_object*)(lp_mathlib_OrderRingHom_instPreorder___closed__0));
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingHom_instPartialOrder___boxed(lean_object* v_00_u03b1_435_, lean_object* v_00_u03b2_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_OrderRingHom_instPartialOrder(v_00_u03b1_435_, v_00_u03b2_436_, v_inst_437_, v_inst_438_, v_inst_439_, v_inst_440_);
lean_dec_ref(v_inst_440_);
lean_dec_ref(v_inst_439_);
lean_dec_ref(v_inst_438_);
lean_dec_ref(v_inst_437_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso___redArg(lean_object* v_f_442_){
_start:
{
lean_inc_ref(v_f_442_);
return v_f_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso___redArg___boxed(lean_object* v_f_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib_OrderRingIso_toOrderIso___redArg(v_f_443_);
lean_dec_ref(v_f_443_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso(lean_object* v_00_u03b1_445_, lean_object* v_00_u03b2_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_f_453_){
_start:
{
lean_inc_ref(v_f_453_);
return v_f_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderIso___boxed(lean_object* v_00_u03b1_454_, lean_object* v_00_u03b2_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_f_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_mathlib_OrderRingIso_toOrderIso(v_00_u03b1_454_, v_00_u03b2_455_, v_inst_456_, v_inst_457_, v_inst_458_, v_inst_459_, v_inst_460_, v_inst_461_, v_f_462_);
lean_dec_ref(v_f_462_);
lean_dec(v_inst_460_);
lean_dec(v_inst_459_);
lean_dec(v_inst_457_);
lean_dec(v_inst_456_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___lam__0(lean_object* v_self_464_){
_start:
{
lean_inc_ref(v_self_464_);
return v_self_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___lam__0___boxed(lean_object* v_self_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib_OrderRingIso_instCoeOutRingEquiv___lam__0(v_self_465_);
lean_dec_ref(v_self_465_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv(lean_object* v_00_u03b1_468_, lean_object* v_00_u03b2_469_, lean_object* v_inst_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___f_476_; 
v___f_476_ = ((lean_object*)(lp_mathlib_OrderRingIso_instCoeOutRingEquiv___closed__0));
return v___f_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instCoeOutRingEquiv___boxed(lean_object* v_00_u03b1_477_, lean_object* v_00_u03b2_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_mathlib_OrderRingIso_instCoeOutRingEquiv(v_00_u03b1_477_, v_00_u03b2_478_, v_inst_479_, v_inst_480_, v_inst_481_, v_inst_482_, v_inst_483_, v_inst_484_);
lean_dec(v_inst_483_);
lean_dec(v_inst_482_);
lean_dec(v_inst_480_);
lean_dec(v_inst_479_);
return v_res_485_;
}
}
static lean_object* _init_lp_mathlib_OrderRingIso_refl___closed__0(void){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_refl(lean_object* v_00_u03b1_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lean_obj_once(&lp_mathlib_OrderRingIso_refl___closed__0, &lp_mathlib_OrderRingIso_refl___closed__0_once, _init_lp_mathlib_OrderRingIso_refl___closed__0);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_refl___boxed(lean_object* v_00_u03b1_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_inst_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_OrderRingIso_refl(v_00_u03b1_492_, v_inst_493_, v_inst_494_, v_inst_495_);
lean_dec(v_inst_494_);
lean_dec(v_inst_493_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instInhabited(lean_object* v_00_u03b1_497_, lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_inst_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lean_obj_once(&lp_mathlib_OrderRingIso_refl___closed__0, &lp_mathlib_OrderRingIso_refl___closed__0_once, _init_lp_mathlib_OrderRingIso_refl___closed__0);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_instInhabited___boxed(lean_object* v_00_u03b1_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_){
_start:
{
lean_object* v_res_506_; 
v_res_506_ = lp_mathlib_OrderRingIso_instInhabited(v_00_u03b1_502_, v_inst_503_, v_inst_504_, v_inst_505_);
lean_dec(v_inst_504_);
lean_dec(v_inst_503_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_symm___redArg(lean_object* v_e_507_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_mathlib_Equiv_symm___redArg(v_e_507_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_symm(lean_object* v_00_u03b1_509_, lean_object* v_00_u03b2_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_e_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_Equiv_symm___redArg(v_e_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_symm___boxed(lean_object* v_00_u03b1_519_, lean_object* v_00_u03b2_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_e_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib_OrderRingIso_symm(v_00_u03b1_519_, v_00_u03b2_520_, v_inst_521_, v_inst_522_, v_inst_523_, v_inst_524_, v_inst_525_, v_inst_526_, v_e_527_);
lean_dec(v_inst_525_);
lean_dec(v_inst_524_);
lean_dec(v_inst_522_);
lean_dec(v_inst_521_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_Simps_symm__apply___redArg(lean_object* v_e_529_, lean_object* v_a_530_){
_start:
{
lean_object* v___x_531_; lean_object* v_toFun_532_; lean_object* v___x_533_; 
v___x_531_ = lp_mathlib_Equiv_symm___redArg(v_e_529_);
v_toFun_532_ = lean_ctor_get(v___x_531_, 0);
lean_inc(v_toFun_532_);
lean_dec_ref(v___x_531_);
v___x_533_ = lean_apply_1(v_toFun_532_, v_a_530_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_Simps_symm__apply(lean_object* v_00_u03b1_534_, lean_object* v_00_u03b2_535_, lean_object* v_inst_536_, lean_object* v_inst_537_, lean_object* v_inst_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_e_542_, lean_object* v_a_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_OrderRingIso_Simps_symm__apply___redArg(v_e_542_, v_a_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_Simps_symm__apply___boxed(lean_object* v_00_u03b1_545_, lean_object* v_00_u03b2_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_e_553_, lean_object* v_a_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_OrderRingIso_Simps_symm__apply(v_00_u03b1_545_, v_00_u03b2_546_, v_inst_547_, v_inst_548_, v_inst_549_, v_inst_550_, v_inst_551_, v_inst_552_, v_e_553_, v_a_554_);
lean_dec(v_inst_551_);
lean_dec(v_inst_550_);
lean_dec(v_inst_548_);
lean_dec(v_inst_547_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_trans___redArg(lean_object* v_f_556_, lean_object* v_g_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lp_mathlib_Equiv_trans___redArg(v_f_556_, v_g_557_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_trans(lean_object* v_00_u03b1_559_, lean_object* v_00_u03b2_560_, lean_object* v_00_u03b3_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_inst_569_, lean_object* v_inst_570_, lean_object* v_f_571_, lean_object* v_g_572_){
_start:
{
lean_object* v___x_573_; 
v___x_573_ = lp_mathlib_Equiv_trans___redArg(v_f_571_, v_g_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_trans___boxed(lean_object* v_00_u03b1_574_, lean_object* v_00_u03b2_575_, lean_object* v_00_u03b3_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_f_586_, lean_object* v_g_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib_OrderRingIso_trans(v_00_u03b1_574_, v_00_u03b2_575_, v_00_u03b3_576_, v_inst_577_, v_inst_578_, v_inst_579_, v_inst_580_, v_inst_581_, v_inst_582_, v_inst_583_, v_inst_584_, v_inst_585_, v_f_586_, v_g_587_);
lean_dec(v_inst_584_);
lean_dec(v_inst_583_);
lean_dec(v_inst_581_);
lean_dec(v_inst_580_);
lean_dec(v_inst_578_);
lean_dec(v_inst_577_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom___redArg(lean_object* v_f_589_){
_start:
{
lean_object* v_toFun_590_; 
v_toFun_590_ = lean_ctor_get(v_f_589_, 0);
lean_inc(v_toFun_590_);
return v_toFun_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom___redArg___boxed(lean_object* v_f_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_OrderRingIso_toOrderRingHom___redArg(v_f_591_);
lean_dec_ref(v_f_591_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom(lean_object* v_00_u03b1_593_, lean_object* v_00_u03b2_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_inst_598_, lean_object* v_f_599_){
_start:
{
lean_object* v_toFun_600_; 
v_toFun_600_ = lean_ctor_get(v_f_599_, 0);
lean_inc(v_toFun_600_);
return v_toFun_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderRingIso_toOrderRingHom___boxed(lean_object* v_00_u03b1_601_, lean_object* v_00_u03b2_602_, lean_object* v_inst_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_f_607_){
_start:
{
lean_object* v_res_608_; 
v_res_608_ = lp_mathlib_OrderRingIso_toOrderRingHom(v_00_u03b1_601_, v_00_u03b2_602_, v_inst_603_, v_inst_604_, v_inst_605_, v_inst_606_, v_f_607_);
lean_dec_ref(v_f_607_);
lean_dec_ref(v_inst_606_);
lean_dec_ref(v_inst_605_);
lean_dec_ref(v_inst_604_);
lean_dec_ref(v_inst_603_);
return v_res_608_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
}
#ifdef __cplusplus
}
#endif
