// Lean compiler output
// Module: Mathlib.Algebra.Order.Group.Unbundled.Abs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Even public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Algebra.Order.Group.Lattice public meta import Mathlib.Tactic.ToDual
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mabs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mabs___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mabs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mabs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_abs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_abs___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_abs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_abs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 10, .m_data = "term|___|ₘ"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__0 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__0_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__0_value),LEAN_SCALAR_PTR_LITERAL(17, 12, 219, 254, 154, 46, 3, 210)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__1 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__1_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__2 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__2_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__3 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__3_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__4 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__4_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__4_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__5 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__5_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__6 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__6_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__6_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__7 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__7_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__8 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__8_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__8_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__9 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__9_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "noWs"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__10 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__10_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__10_value),LEAN_SCALAR_PTR_LITERAL(92, 29, 204, 148, 167, 109, 242, 21)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__11 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__11_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__11_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__12 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__12_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__3_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__9_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__12_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__13 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__13_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__7_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__13_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__14 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__14_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__5_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__14_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__15 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__15_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__16 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__16_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__16_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__17 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__17_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__18 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__18_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__3_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__15_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__18_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__19 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__19_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__5_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__12_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__20 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__20_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__3_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__19_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__20_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__21 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__21_value;
static const lean_string_object lp_mathlib_term_x7c_______x7c_u2098___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "|ₘ"};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__22 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__22_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__22_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__23 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__23_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__3_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__21_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__23_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__24 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__24_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c_u2098___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__24_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c_u2098___closed__25 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__25_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_x7c_______x7c_u2098 = (const lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__25_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "mabs"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 214, 30, 58, 171, 1, 251, 249)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_x7c_______x7c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term|___|"};
static const lean_object* lp_mathlib_term_x7c_______x7c___closed__0 = (const lean_object*)&lp_mathlib_term_x7c_______x7c___closed__0_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(196, 95, 93, 142, 197, 135, 20, 28)}};
static const lean_object* lp_mathlib_term_x7c_______x7c___closed__1 = (const lean_object*)&lp_mathlib_term_x7c_______x7c___closed__1_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__3_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__21_value),((lean_object*)&lp_mathlib_term_x7c_______x7c_u2098___closed__9_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c___closed__2 = (const lean_object*)&lp_mathlib_term_x7c_______x7c___closed__2_value;
static const lean_ctor_object lp_mathlib_term_x7c_______x7c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_x7c_______x7c___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_x7c_______x7c___closed__2_value)}};
static const lean_object* lp_mathlib_term_x7c_______x7c___closed__3 = (const lean_object*)&lp_mathlib_term_x7c_______x7c___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_x7c_______x7c = (const lean_object*)&lp_mathlib_term_x7c_______x7c___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "abs"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(11, 180, 28, 55, 197, 20, 206, 35)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_mabs_unexpander___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term-_"};
static const lean_object* lp_mathlib_mabs_unexpander___closed__0 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__0_value;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_mabs_unexpander___closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 127, 37, 42, 155, 196, 209, 131)}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__1 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__1_value;
static lean_once_cell_t lp_mathlib_mabs_unexpander___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mabs_unexpander___closed__2;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_fakeMod"};
static const lean_object* lp_mathlib_mabs_unexpander___closed__3 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__3_value;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_mabs_unexpander___closed__3_value),LEAN_SCALAR_PTR_LITERAL(168, 44, 241, 255, 153, 255, 67, 53)}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__4 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__4_value;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_mabs_unexpander___closed__5 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__5_value;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__6_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__6_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__6_value_aux_2),((lean_object*)&lp_mathlib_mabs_unexpander___closed__5_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__6 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__6_value;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_mabs_unexpander___closed__7 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__7_value;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__8_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__8_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__8_value_aux_2),((lean_object*)&lp_mathlib_mabs_unexpander___closed__7_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__8 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__8_value;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_mabs_unexpander___closed__9 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__9_value;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_mabs_unexpander___closed__10 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__10_value;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_mabs_unexpander___closed__10_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__11 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__11_value;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_mabs_unexpander___closed__12 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__12_value;
static lean_once_cell_t lp_mathlib_mabs_unexpander___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mabs_unexpander___closed__13;
static lean_once_cell_t lp_mathlib_mabs_unexpander___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mabs_unexpander___closed__14;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__7_value)}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__15 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__15_value;
static const lean_ctor_object lp_mathlib_mabs_unexpander___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mabs_unexpander___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_mabs_unexpander___closed__16 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__16_value;
static const lean_string_object lp_mathlib_mabs_unexpander___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_mabs_unexpander___closed__17 = (const lean_object*)&lp_mathlib_mabs_unexpander___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_mabs_unexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mabs_unexpander___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_abs_unexpander___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__2_value)}};
static const lean_object* lp_mathlib_abs_unexpander___closed__0 = (const lean_object*)&lp_mathlib_abs_unexpander___closed__0_value;
static const lean_ctor_object lp_mathlib_abs_unexpander___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_abs_unexpander___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_abs_unexpander___closed__1 = (const lean_object*)&lp_mathlib_abs_unexpander___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_abs_unexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_abs_unexpander___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mabs___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_a_3_){
_start:
{
lean_object* v_toSemilatticeSup_4_; lean_object* v___x_5_; lean_object* v_toInv_6_; lean_object* v_sup_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v_toSemilatticeSup_4_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_toSemilatticeSup_4_);
lean_dec_ref(v_inst_1_);
v___x_5_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_2_);
v_toInv_6_ = lean_ctor_get(v___x_5_, 1);
lean_inc(v_toInv_6_);
lean_dec_ref(v___x_5_);
v_sup_7_ = lean_ctor_get(v_toSemilatticeSup_4_, 1);
lean_inc(v_sup_7_);
lean_dec_ref(v_toSemilatticeSup_4_);
lean_inc(v_a_3_);
v___x_8_ = lean_apply_1(v_toInv_6_, v_a_3_);
v___x_9_ = lean_apply_2(v_sup_7_, v_a_3_, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mabs___redArg___boxed(lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_a_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_mabs___redArg(v_inst_10_, v_inst_11_, v_a_12_);
lean_dec_ref(v_inst_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mabs(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_a_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_mabs___redArg(v_inst_15_, v_inst_16_, v_a_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mabs___boxed(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_a_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_mabs(v_00_u03b1_19_, v_inst_20_, v_inst_21_, v_a_22_);
lean_dec_ref(v_inst_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs___redArg(lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_a_26_){
_start:
{
lean_object* v_toSemilatticeSup_27_; lean_object* v___x_28_; lean_object* v_toNeg_29_; lean_object* v_sup_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v_toSemilatticeSup_27_ = lean_ctor_get(v_inst_24_, 0);
lean_inc_ref(v_toSemilatticeSup_27_);
lean_dec_ref(v_inst_24_);
v___x_28_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_25_);
v_toNeg_29_ = lean_ctor_get(v___x_28_, 1);
lean_inc(v_toNeg_29_);
lean_dec_ref(v___x_28_);
v_sup_30_ = lean_ctor_get(v_toSemilatticeSup_27_, 1);
lean_inc(v_sup_30_);
lean_dec_ref(v_toSemilatticeSup_27_);
lean_inc(v_a_26_);
v___x_31_ = lean_apply_1(v_toNeg_29_, v_a_26_);
v___x_32_ = lean_apply_2(v_sup_30_, v_a_26_, v___x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs___redArg___boxed(lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_a_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_abs___redArg(v_inst_33_, v_inst_34_, v_a_35_);
lean_dec_ref(v_inst_34_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_a_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_abs___redArg(v_inst_38_, v_inst_39_, v_a_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs___boxed(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_a_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_abs(v_00_u03b1_42_, v_inst_43_, v_inst_44_, v_a_45_);
lean_dec_ref(v_inst_44_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__6(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__5));
v___x_117_ = l_String_toRawSubstring_x27(v___x_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1(lean_object* v_x_129_, lean_object* v_a_130_, lean_object* v_a_131_){
_start:
{
lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_132_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__1));
lean_inc(v_x_129_);
v___x_133_ = l_Lean_Syntax_isOfKind(v_x_129_, v___x_132_);
if (v___x_133_ == 0)
{
lean_object* v___x_134_; lean_object* v___x_135_; 
lean_dec(v_x_129_);
v___x_134_ = lean_box(1);
v___x_135_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_131_);
return v___x_135_;
}
else
{
lean_object* v_quotContext_136_; lean_object* v_currMacroScope_137_; lean_object* v_ref_138_; lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v_quotContext_136_ = lean_ctor_get(v_a_130_, 1);
v_currMacroScope_137_ = lean_ctor_get(v_a_130_, 2);
v_ref_138_ = lean_ctor_get(v_a_130_, 5);
v___x_139_ = lean_unsigned_to_nat(1u);
v___x_140_ = l_Lean_Syntax_getArg(v_x_129_, v___x_139_);
lean_dec(v_x_129_);
v___x_141_ = 0;
v___x_142_ = l_Lean_SourceInfo_fromRef(v_ref_138_, v___x_141_);
v___x_143_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4));
v___x_144_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__6);
v___x_145_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__7));
lean_inc(v_currMacroScope_137_);
lean_inc(v_quotContext_136_);
v___x_146_ = l_Lean_addMacroScope(v_quotContext_136_, v___x_145_, v_currMacroScope_137_);
v___x_147_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__9));
lean_inc_n(v___x_142_, 2);
v___x_148_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_148_, 0, v___x_142_);
lean_ctor_set(v___x_148_, 1, v___x_144_);
lean_ctor_set(v___x_148_, 2, v___x_146_);
lean_ctor_set(v___x_148_, 3, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__11));
v___x_150_ = l_Lean_Syntax_node1(v___x_142_, v___x_149_, v___x_140_);
v___x_151_ = l_Lean_Syntax_node2(v___x_142_, v___x_143_, v___x_148_, v___x_150_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_a_131_);
return v___x_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___boxed(lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1(v_x_153_, v_a_154_, v_a_155_);
lean_dec_ref(v_a_154_);
return v_res_156_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__1(void){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__0));
v___x_171_ = l_String_toRawSubstring_x27(v___x_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1(lean_object* v_x_180_, lean_object* v_a_181_, lean_object* v_a_182_){
_start:
{
lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_183_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c___closed__1));
lean_inc(v_x_180_);
v___x_184_ = l_Lean_Syntax_isOfKind(v_x_180_, v___x_183_);
if (v___x_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; 
lean_dec(v_x_180_);
v___x_185_ = lean_box(1);
v___x_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v_a_182_);
return v___x_186_;
}
else
{
lean_object* v_quotContext_187_; lean_object* v_currMacroScope_188_; lean_object* v_ref_189_; lean_object* v___x_190_; lean_object* v___x_191_; uint8_t v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v_quotContext_187_ = lean_ctor_get(v_a_181_, 1);
v_currMacroScope_188_ = lean_ctor_get(v_a_181_, 2);
v_ref_189_ = lean_ctor_get(v_a_181_, 5);
v___x_190_ = lean_unsigned_to_nat(1u);
v___x_191_ = l_Lean_Syntax_getArg(v_x_180_, v___x_190_);
lean_dec(v_x_180_);
v___x_192_ = 0;
v___x_193_ = l_Lean_SourceInfo_fromRef(v_ref_189_, v___x_192_);
v___x_194_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4));
v___x_195_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__1);
v___x_196_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__2));
lean_inc(v_currMacroScope_188_);
lean_inc(v_quotContext_187_);
v___x_197_ = l_Lean_addMacroScope(v_quotContext_187_, v___x_196_, v_currMacroScope_188_);
v___x_198_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___closed__4));
lean_inc_n(v___x_193_, 2);
v___x_199_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_199_, 0, v___x_193_);
lean_ctor_set(v___x_199_, 1, v___x_195_);
lean_ctor_set(v___x_199_, 2, v___x_197_);
lean_ctor_set(v___x_199_, 3, v___x_198_);
v___x_200_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__11));
v___x_201_ = l_Lean_Syntax_node1(v___x_193_, v___x_200_, v___x_191_);
v___x_202_ = l_Lean_Syntax_node2(v___x_193_, v___x_194_, v___x_199_, v___x_201_);
v___x_203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_203_, 0, v___x_202_);
lean_ctor_set(v___x_203_, 1, v_a_182_);
return v___x_203_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1___boxed(lean_object* v_x_204_, lean_object* v_a_205_, lean_object* v_a_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c__1(v_x_204_, v_a_205_, v_a_206_);
lean_dec_ref(v_a_205_);
return v_res_207_;
}
}
static lean_object* _init_lp_mathlib_mabs_unexpander___closed__2(void){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = l_Array_mkArray0(lean_box(0));
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_mabs_unexpander___closed__13(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__12));
v___x_233_ = l_String_toRawSubstring_x27(v___x_232_);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib_mabs_unexpander___closed__14(void){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_234_ = lean_unsigned_to_nat(0u);
v___x_235_ = lean_box(0);
v___x_236_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__4));
v___x_237_ = l_Lean_addMacroScope(v___x_236_, v___x_235_, v___x_234_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mabs_unexpander(lean_object* v_x_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_247_; uint8_t v___x_248_; 
v___x_247_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4));
lean_inc(v_x_244_);
v___x_248_ = l_Lean_Syntax_isOfKind(v_x_244_, v___x_247_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; 
lean_dec(v_x_244_);
v___x_249_ = lean_box(0);
v___x_250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v_a_246_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; lean_object* v___x_252_; uint8_t v___x_253_; 
v___x_251_ = lean_unsigned_to_nat(1u);
v___x_252_ = l_Lean_Syntax_getArg(v_x_244_, v___x_251_);
lean_dec(v_x_244_);
lean_inc(v___x_252_);
v___x_253_ = l_Lean_Syntax_matchesNull(v___x_252_, v___x_251_);
if (v___x_253_ == 0)
{
lean_object* v___x_254_; lean_object* v___x_255_; 
lean_dec(v___x_252_);
v___x_254_ = lean_box(0);
v___x_255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v_a_246_);
return v___x_255_;
}
else
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; uint8_t v___x_259_; 
v___x_256_ = lean_unsigned_to_nat(0u);
v___x_257_ = l_Lean_Syntax_getArg(v___x_252_, v___x_256_);
lean_dec(v___x_252_);
v___x_258_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c___closed__1));
lean_inc(v___x_257_);
v___x_259_ = l_Lean_Syntax_isOfKind(v___x_257_, v___x_258_);
if (v___x_259_ == 0)
{
lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_260_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__1));
lean_inc(v___x_257_);
v___x_261_ = l_Lean_Syntax_isOfKind(v___x_257_, v___x_260_);
if (v___x_261_ == 0)
{
lean_object* v___x_262_; uint8_t v___x_263_; 
v___x_262_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__1));
lean_inc(v___x_257_);
v___x_263_ = l_Lean_Syntax_isOfKind(v___x_257_, v___x_262_);
if (v___x_263_ == 0)
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_264_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_263_);
v___x_265_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_266_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_264_, 4);
v___x_267_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_264_);
lean_ctor_set(v___x_267_, 1, v___x_266_);
v___x_268_ = l_Lean_Syntax_node1(v___x_264_, v___x_265_, v___x_267_);
v___x_269_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_270_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_270_, 0, v___x_264_);
lean_ctor_set(v___x_270_, 1, v___x_265_);
lean_ctor_set(v___x_270_, 2, v___x_269_);
v___x_271_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_272_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_264_);
lean_ctor_set(v___x_272_, 1, v___x_271_);
v___x_273_ = l_Lean_Syntax_node4(v___x_264_, v___x_260_, v___x_268_, v___x_257_, v___x_270_, v___x_272_);
v___x_274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_274_, 0, v___x_273_);
lean_ctor_set(v___x_274_, 1, v_a_246_);
return v___x_274_;
}
else
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_275_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_261_);
v___x_276_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_277_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_275_, 10);
v___x_278_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_278_, 0, v___x_275_);
lean_ctor_set(v___x_278_, 1, v___x_277_);
v___x_279_ = l_Lean_Syntax_node1(v___x_275_, v___x_276_, v___x_278_);
v___x_280_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__6));
v___x_281_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__8));
v___x_282_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__9));
v___x_283_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_275_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
v___x_284_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__11));
v___x_285_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__13, &lp_mathlib_mabs_unexpander___closed__13_once, _init_lp_mathlib_mabs_unexpander___closed__13);
v___x_286_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__14, &lp_mathlib_mabs_unexpander___closed__14_once, _init_lp_mathlib_mabs_unexpander___closed__14);
v___x_287_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__16));
v___x_288_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_288_, 0, v___x_275_);
lean_ctor_set(v___x_288_, 1, v___x_285_);
lean_ctor_set(v___x_288_, 2, v___x_286_);
lean_ctor_set(v___x_288_, 3, v___x_287_);
v___x_289_ = l_Lean_Syntax_node1(v___x_275_, v___x_284_, v___x_288_);
v___x_290_ = l_Lean_Syntax_node2(v___x_275_, v___x_281_, v___x_283_, v___x_289_);
v___x_291_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__17));
v___x_292_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_275_);
lean_ctor_set(v___x_292_, 1, v___x_291_);
v___x_293_ = l_Lean_Syntax_node3(v___x_275_, v___x_280_, v___x_290_, v___x_257_, v___x_292_);
v___x_294_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_295_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_295_, 0, v___x_275_);
lean_ctor_set(v___x_295_, 1, v___x_276_);
lean_ctor_set(v___x_295_, 2, v___x_294_);
v___x_296_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_297_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_275_);
lean_ctor_set(v___x_297_, 1, v___x_296_);
v___x_298_ = l_Lean_Syntax_node4(v___x_275_, v___x_260_, v___x_279_, v___x_293_, v___x_295_, v___x_297_);
v___x_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v_a_246_);
return v___x_299_;
}
}
else
{
lean_object* v___x_300_; lean_object* v___x_301_; uint8_t v___x_302_; 
v___x_300_ = l_Lean_Syntax_getArg(v___x_257_, v___x_256_);
v___x_301_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_302_ = l_Lean_Syntax_isOfKind(v___x_300_, v___x_301_);
if (v___x_302_ == 0)
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_303_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_302_);
v___x_304_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_303_, 4);
v___x_305_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_303_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = l_Lean_Syntax_node1(v___x_303_, v___x_301_, v___x_305_);
v___x_307_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_308_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_308_, 0, v___x_303_);
lean_ctor_set(v___x_308_, 1, v___x_301_);
lean_ctor_set(v___x_308_, 2, v___x_307_);
v___x_309_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_310_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_303_);
lean_ctor_set(v___x_310_, 1, v___x_309_);
v___x_311_ = l_Lean_Syntax_node4(v___x_303_, v___x_260_, v___x_306_, v___x_257_, v___x_308_, v___x_310_);
v___x_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
lean_ctor_set(v___x_312_, 1, v_a_246_);
return v___x_312_;
}
else
{
lean_object* v___x_313_; lean_object* v___x_314_; uint8_t v___x_315_; 
v___x_313_ = lean_unsigned_to_nat(2u);
v___x_314_ = l_Lean_Syntax_getArg(v___x_257_, v___x_313_);
v___x_315_ = l_Lean_Syntax_isOfKind(v___x_314_, v___x_301_);
if (v___x_315_ == 0)
{
lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_316_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_315_);
v___x_317_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_316_, 4);
v___x_318_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_316_);
lean_ctor_set(v___x_318_, 1, v___x_317_);
v___x_319_ = l_Lean_Syntax_node1(v___x_316_, v___x_301_, v___x_318_);
v___x_320_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_321_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_321_, 0, v___x_316_);
lean_ctor_set(v___x_321_, 1, v___x_301_);
lean_ctor_set(v___x_321_, 2, v___x_320_);
v___x_322_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_323_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_323_, 0, v___x_316_);
lean_ctor_set(v___x_323_, 1, v___x_322_);
v___x_324_ = l_Lean_Syntax_node4(v___x_316_, v___x_260_, v___x_319_, v___x_257_, v___x_321_, v___x_323_);
v___x_325_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_324_);
lean_ctor_set(v___x_325_, 1, v_a_246_);
return v___x_325_;
}
else
{
lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_326_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_259_);
v___x_327_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_326_, 10);
v___x_328_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_326_);
lean_ctor_set(v___x_328_, 1, v___x_327_);
v___x_329_ = l_Lean_Syntax_node1(v___x_326_, v___x_301_, v___x_328_);
v___x_330_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__6));
v___x_331_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__8));
v___x_332_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__9));
v___x_333_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_326_);
lean_ctor_set(v___x_333_, 1, v___x_332_);
v___x_334_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__11));
v___x_335_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__13, &lp_mathlib_mabs_unexpander___closed__13_once, _init_lp_mathlib_mabs_unexpander___closed__13);
v___x_336_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__14, &lp_mathlib_mabs_unexpander___closed__14_once, _init_lp_mathlib_mabs_unexpander___closed__14);
v___x_337_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__16));
v___x_338_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_338_, 0, v___x_326_);
lean_ctor_set(v___x_338_, 1, v___x_335_);
lean_ctor_set(v___x_338_, 2, v___x_336_);
lean_ctor_set(v___x_338_, 3, v___x_337_);
v___x_339_ = l_Lean_Syntax_node1(v___x_326_, v___x_334_, v___x_338_);
v___x_340_ = l_Lean_Syntax_node2(v___x_326_, v___x_331_, v___x_333_, v___x_339_);
v___x_341_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__17));
v___x_342_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_326_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = l_Lean_Syntax_node3(v___x_326_, v___x_330_, v___x_340_, v___x_257_, v___x_342_);
v___x_344_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_345_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_345_, 0, v___x_326_);
lean_ctor_set(v___x_345_, 1, v___x_301_);
lean_ctor_set(v___x_345_, 2, v___x_344_);
v___x_346_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_347_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_326_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = l_Lean_Syntax_node4(v___x_326_, v___x_260_, v___x_329_, v___x_343_, v___x_345_, v___x_347_);
v___x_349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_349_, 0, v___x_348_);
lean_ctor_set(v___x_349_, 1, v_a_246_);
return v___x_349_;
}
}
}
}
else
{
lean_object* v___x_350_; lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_350_ = l_Lean_Syntax_getArg(v___x_257_, v___x_256_);
v___x_351_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_352_ = l_Lean_Syntax_isOfKind(v___x_350_, v___x_351_);
if (v___x_352_ == 0)
{
lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_353_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_352_);
v___x_354_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__1));
v___x_355_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_353_, 4);
v___x_356_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_353_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
v___x_357_ = l_Lean_Syntax_node1(v___x_353_, v___x_351_, v___x_356_);
v___x_358_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_359_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_359_, 0, v___x_353_);
lean_ctor_set(v___x_359_, 1, v___x_351_);
lean_ctor_set(v___x_359_, 2, v___x_358_);
v___x_360_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_361_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_353_);
lean_ctor_set(v___x_361_, 1, v___x_360_);
v___x_362_ = l_Lean_Syntax_node4(v___x_353_, v___x_354_, v___x_357_, v___x_257_, v___x_359_, v___x_361_);
v___x_363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v_a_246_);
return v___x_363_;
}
else
{
lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; 
v___x_364_ = lean_unsigned_to_nat(2u);
v___x_365_ = l_Lean_Syntax_getArg(v___x_257_, v___x_364_);
v___x_366_ = l_Lean_Syntax_isOfKind(v___x_365_, v___x_351_);
if (v___x_366_ == 0)
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_367_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_366_);
v___x_368_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__1));
v___x_369_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_367_, 4);
v___x_370_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_370_, 0, v___x_367_);
lean_ctor_set(v___x_370_, 1, v___x_369_);
v___x_371_ = l_Lean_Syntax_node1(v___x_367_, v___x_351_, v___x_370_);
v___x_372_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_373_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_373_, 0, v___x_367_);
lean_ctor_set(v___x_373_, 1, v___x_351_);
lean_ctor_set(v___x_373_, 2, v___x_372_);
v___x_374_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_375_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_367_);
lean_ctor_set(v___x_375_, 1, v___x_374_);
v___x_376_ = l_Lean_Syntax_node4(v___x_367_, v___x_368_, v___x_371_, v___x_257_, v___x_373_, v___x_375_);
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set(v___x_377_, 1, v_a_246_);
return v___x_377_;
}
else
{
uint8_t v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_378_ = 0;
v___x_379_ = l_Lean_SourceInfo_fromRef(v_a_245_, v___x_378_);
v___x_380_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__1));
v___x_381_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_379_, 10);
v___x_382_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_382_, 0, v___x_379_);
lean_ctor_set(v___x_382_, 1, v___x_381_);
v___x_383_ = l_Lean_Syntax_node1(v___x_379_, v___x_351_, v___x_382_);
v___x_384_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__6));
v___x_385_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__8));
v___x_386_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__9));
v___x_387_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_387_, 0, v___x_379_);
lean_ctor_set(v___x_387_, 1, v___x_386_);
v___x_388_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__11));
v___x_389_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__13, &lp_mathlib_mabs_unexpander___closed__13_once, _init_lp_mathlib_mabs_unexpander___closed__13);
v___x_390_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__14, &lp_mathlib_mabs_unexpander___closed__14_once, _init_lp_mathlib_mabs_unexpander___closed__14);
v___x_391_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__16));
v___x_392_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_392_, 0, v___x_379_);
lean_ctor_set(v___x_392_, 1, v___x_389_);
lean_ctor_set(v___x_392_, 2, v___x_390_);
lean_ctor_set(v___x_392_, 3, v___x_391_);
v___x_393_ = l_Lean_Syntax_node1(v___x_379_, v___x_388_, v___x_392_);
v___x_394_ = l_Lean_Syntax_node2(v___x_379_, v___x_385_, v___x_387_, v___x_393_);
v___x_395_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__17));
v___x_396_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_396_, 0, v___x_379_);
lean_ctor_set(v___x_396_, 1, v___x_395_);
v___x_397_ = l_Lean_Syntax_node3(v___x_379_, v___x_384_, v___x_394_, v___x_257_, v___x_396_);
v___x_398_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_399_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_399_, 0, v___x_379_);
lean_ctor_set(v___x_399_, 1, v___x_351_);
lean_ctor_set(v___x_399_, 2, v___x_398_);
v___x_400_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__22));
v___x_401_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_401_, 0, v___x_379_);
lean_ctor_set(v___x_401_, 1, v___x_400_);
v___x_402_ = l_Lean_Syntax_node4(v___x_379_, v___x_380_, v___x_383_, v___x_397_, v___x_399_, v___x_401_);
v___x_403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
lean_ctor_set(v___x_403_, 1, v_a_246_);
return v___x_403_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mabs_unexpander___boxed(lean_object* v_x_404_, lean_object* v_a_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_mabs_unexpander(v_x_404_, v_a_405_, v_a_406_);
lean_dec(v_a_405_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs_unexpander(lean_object* v_x_413_, lean_object* v_a_414_, lean_object* v_a_415_){
_start:
{
lean_object* v___x_416_; uint8_t v___x_417_; 
v___x_416_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Order__Group__Unbundled__Abs______macroRules__term_x7c_______x7c_u2098__1___closed__4));
lean_inc(v_x_413_);
v___x_417_ = l_Lean_Syntax_isOfKind(v_x_413_, v___x_416_);
if (v___x_417_ == 0)
{
lean_object* v___x_418_; lean_object* v___x_419_; 
lean_dec(v_x_413_);
v___x_418_ = lean_box(0);
v___x_419_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
lean_ctor_set(v___x_419_, 1, v_a_415_);
return v___x_419_;
}
else
{
lean_object* v___x_420_; lean_object* v___x_421_; uint8_t v___x_422_; 
v___x_420_ = lean_unsigned_to_nat(1u);
v___x_421_ = l_Lean_Syntax_getArg(v_x_413_, v___x_420_);
lean_dec(v_x_413_);
lean_inc(v___x_421_);
v___x_422_ = l_Lean_Syntax_matchesNull(v___x_421_, v___x_420_);
if (v___x_422_ == 0)
{
lean_object* v___x_423_; lean_object* v___x_424_; 
lean_dec(v___x_421_);
v___x_423_ = lean_box(0);
v___x_424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
lean_ctor_set(v___x_424_, 1, v_a_415_);
return v___x_424_;
}
else
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_425_ = lean_unsigned_to_nat(0u);
v___x_426_ = l_Lean_Syntax_getArg(v___x_421_, v___x_425_);
lean_dec(v___x_421_);
v___x_427_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c___closed__1));
lean_inc(v___x_426_);
v___x_428_ = l_Lean_Syntax_isOfKind(v___x_426_, v___x_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; uint8_t v___x_430_; 
v___x_429_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__1));
lean_inc(v___x_426_);
v___x_430_ = l_Lean_Syntax_isOfKind(v___x_426_, v___x_429_);
if (v___x_430_ == 0)
{
lean_object* v___x_431_; uint8_t v___x_432_; 
v___x_431_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__1));
lean_inc(v___x_426_);
v___x_432_ = l_Lean_Syntax_isOfKind(v___x_426_, v___x_431_);
if (v___x_432_ == 0)
{
lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_433_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_432_);
v___x_434_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_435_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_433_, 3);
v___x_436_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_433_);
lean_ctor_set(v___x_436_, 1, v___x_435_);
lean_inc_ref(v___x_436_);
v___x_437_ = l_Lean_Syntax_node1(v___x_433_, v___x_434_, v___x_436_);
v___x_438_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_439_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_439_, 0, v___x_433_);
lean_ctor_set(v___x_439_, 1, v___x_434_);
lean_ctor_set(v___x_439_, 2, v___x_438_);
v___x_440_ = l_Lean_Syntax_node4(v___x_433_, v___x_427_, v___x_437_, v___x_426_, v___x_439_, v___x_436_);
v___x_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_440_);
lean_ctor_set(v___x_441_, 1, v_a_415_);
return v___x_441_;
}
else
{
lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_442_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_430_);
v___x_443_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_444_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_442_, 9);
v___x_445_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_445_, 0, v___x_442_);
lean_ctor_set(v___x_445_, 1, v___x_444_);
lean_inc_ref(v___x_445_);
v___x_446_ = l_Lean_Syntax_node1(v___x_442_, v___x_443_, v___x_445_);
v___x_447_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__6));
v___x_448_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__8));
v___x_449_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__9));
v___x_450_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_450_, 0, v___x_442_);
lean_ctor_set(v___x_450_, 1, v___x_449_);
v___x_451_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__11));
v___x_452_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__13, &lp_mathlib_mabs_unexpander___closed__13_once, _init_lp_mathlib_mabs_unexpander___closed__13);
v___x_453_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__14, &lp_mathlib_mabs_unexpander___closed__14_once, _init_lp_mathlib_mabs_unexpander___closed__14);
v___x_454_ = ((lean_object*)(lp_mathlib_abs_unexpander___closed__1));
v___x_455_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_455_, 0, v___x_442_);
lean_ctor_set(v___x_455_, 1, v___x_452_);
lean_ctor_set(v___x_455_, 2, v___x_453_);
lean_ctor_set(v___x_455_, 3, v___x_454_);
v___x_456_ = l_Lean_Syntax_node1(v___x_442_, v___x_451_, v___x_455_);
v___x_457_ = l_Lean_Syntax_node2(v___x_442_, v___x_448_, v___x_450_, v___x_456_);
v___x_458_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__17));
v___x_459_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_459_, 0, v___x_442_);
lean_ctor_set(v___x_459_, 1, v___x_458_);
v___x_460_ = l_Lean_Syntax_node3(v___x_442_, v___x_447_, v___x_457_, v___x_426_, v___x_459_);
v___x_461_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_462_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_462_, 0, v___x_442_);
lean_ctor_set(v___x_462_, 1, v___x_443_);
lean_ctor_set(v___x_462_, 2, v___x_461_);
v___x_463_ = l_Lean_Syntax_node4(v___x_442_, v___x_427_, v___x_446_, v___x_460_, v___x_462_, v___x_445_);
v___x_464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_464_, 0, v___x_463_);
lean_ctor_set(v___x_464_, 1, v_a_415_);
return v___x_464_;
}
}
else
{
lean_object* v___x_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v___x_465_ = l_Lean_Syntax_getArg(v___x_426_, v___x_425_);
v___x_466_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_467_ = l_Lean_Syntax_isOfKind(v___x_465_, v___x_466_);
if (v___x_467_ == 0)
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_468_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_467_);
v___x_469_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_468_, 3);
v___x_470_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_468_);
lean_ctor_set(v___x_470_, 1, v___x_469_);
lean_inc_ref(v___x_470_);
v___x_471_ = l_Lean_Syntax_node1(v___x_468_, v___x_466_, v___x_470_);
v___x_472_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_473_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_473_, 0, v___x_468_);
lean_ctor_set(v___x_473_, 1, v___x_466_);
lean_ctor_set(v___x_473_, 2, v___x_472_);
v___x_474_ = l_Lean_Syntax_node4(v___x_468_, v___x_427_, v___x_471_, v___x_426_, v___x_473_, v___x_470_);
v___x_475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
lean_ctor_set(v___x_475_, 1, v_a_415_);
return v___x_475_;
}
else
{
lean_object* v___x_476_; lean_object* v___x_477_; uint8_t v___x_478_; 
v___x_476_ = lean_unsigned_to_nat(2u);
v___x_477_ = l_Lean_Syntax_getArg(v___x_426_, v___x_476_);
v___x_478_ = l_Lean_Syntax_isOfKind(v___x_477_, v___x_466_);
if (v___x_478_ == 0)
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_479_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_478_);
v___x_480_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_479_, 3);
v___x_481_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_479_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
lean_inc_ref(v___x_481_);
v___x_482_ = l_Lean_Syntax_node1(v___x_479_, v___x_466_, v___x_481_);
v___x_483_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_484_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_484_, 0, v___x_479_);
lean_ctor_set(v___x_484_, 1, v___x_466_);
lean_ctor_set(v___x_484_, 2, v___x_483_);
v___x_485_ = l_Lean_Syntax_node4(v___x_479_, v___x_427_, v___x_482_, v___x_426_, v___x_484_, v___x_481_);
v___x_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_485_);
lean_ctor_set(v___x_486_, 1, v_a_415_);
return v___x_486_;
}
else
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_487_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_428_);
v___x_488_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_487_, 9);
v___x_489_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_487_);
lean_ctor_set(v___x_489_, 1, v___x_488_);
lean_inc_ref(v___x_489_);
v___x_490_ = l_Lean_Syntax_node1(v___x_487_, v___x_466_, v___x_489_);
v___x_491_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__6));
v___x_492_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__8));
v___x_493_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__9));
v___x_494_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_487_);
lean_ctor_set(v___x_494_, 1, v___x_493_);
v___x_495_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__11));
v___x_496_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__13, &lp_mathlib_mabs_unexpander___closed__13_once, _init_lp_mathlib_mabs_unexpander___closed__13);
v___x_497_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__14, &lp_mathlib_mabs_unexpander___closed__14_once, _init_lp_mathlib_mabs_unexpander___closed__14);
v___x_498_ = ((lean_object*)(lp_mathlib_abs_unexpander___closed__1));
v___x_499_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_499_, 0, v___x_487_);
lean_ctor_set(v___x_499_, 1, v___x_496_);
lean_ctor_set(v___x_499_, 2, v___x_497_);
lean_ctor_set(v___x_499_, 3, v___x_498_);
v___x_500_ = l_Lean_Syntax_node1(v___x_487_, v___x_495_, v___x_499_);
v___x_501_ = l_Lean_Syntax_node2(v___x_487_, v___x_492_, v___x_494_, v___x_500_);
v___x_502_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__17));
v___x_503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_487_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
v___x_504_ = l_Lean_Syntax_node3(v___x_487_, v___x_491_, v___x_501_, v___x_426_, v___x_503_);
v___x_505_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_506_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_506_, 0, v___x_487_);
lean_ctor_set(v___x_506_, 1, v___x_466_);
lean_ctor_set(v___x_506_, 2, v___x_505_);
v___x_507_ = l_Lean_Syntax_node4(v___x_487_, v___x_427_, v___x_490_, v___x_504_, v___x_506_, v___x_489_);
v___x_508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_507_);
lean_ctor_set(v___x_508_, 1, v_a_415_);
return v___x_508_;
}
}
}
}
else
{
lean_object* v___x_509_; lean_object* v___x_510_; uint8_t v___x_511_; 
v___x_509_ = l_Lean_Syntax_getArg(v___x_426_, v___x_425_);
v___x_510_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__5));
v___x_511_ = l_Lean_Syntax_isOfKind(v___x_509_, v___x_510_);
if (v___x_511_ == 0)
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; 
v___x_512_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_511_);
v___x_513_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_512_, 3);
v___x_514_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_512_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
lean_inc_ref(v___x_514_);
v___x_515_ = l_Lean_Syntax_node1(v___x_512_, v___x_510_, v___x_514_);
v___x_516_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_517_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_517_, 0, v___x_512_);
lean_ctor_set(v___x_517_, 1, v___x_510_);
lean_ctor_set(v___x_517_, 2, v___x_516_);
v___x_518_ = l_Lean_Syntax_node4(v___x_512_, v___x_427_, v___x_515_, v___x_426_, v___x_517_, v___x_514_);
v___x_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_519_, 0, v___x_518_);
lean_ctor_set(v___x_519_, 1, v_a_415_);
return v___x_519_;
}
else
{
lean_object* v___x_520_; lean_object* v___x_521_; uint8_t v___x_522_; 
v___x_520_ = lean_unsigned_to_nat(2u);
v___x_521_ = l_Lean_Syntax_getArg(v___x_426_, v___x_520_);
v___x_522_ = l_Lean_Syntax_isOfKind(v___x_521_, v___x_510_);
if (v___x_522_ == 0)
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_523_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_522_);
v___x_524_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_523_, 3);
v___x_525_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_523_);
lean_ctor_set(v___x_525_, 1, v___x_524_);
lean_inc_ref(v___x_525_);
v___x_526_ = l_Lean_Syntax_node1(v___x_523_, v___x_510_, v___x_525_);
v___x_527_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_528_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_528_, 0, v___x_523_);
lean_ctor_set(v___x_528_, 1, v___x_510_);
lean_ctor_set(v___x_528_, 2, v___x_527_);
v___x_529_ = l_Lean_Syntax_node4(v___x_523_, v___x_427_, v___x_526_, v___x_426_, v___x_528_, v___x_525_);
v___x_530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_530_, 0, v___x_529_);
lean_ctor_set(v___x_530_, 1, v_a_415_);
return v___x_530_;
}
else
{
uint8_t v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_531_ = 0;
v___x_532_ = l_Lean_SourceInfo_fromRef(v_a_414_, v___x_531_);
v___x_533_ = ((lean_object*)(lp_mathlib_term_x7c_______x7c_u2098___closed__8));
lean_inc_n(v___x_532_, 9);
v___x_534_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_532_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
lean_inc_ref(v___x_534_);
v___x_535_ = l_Lean_Syntax_node1(v___x_532_, v___x_510_, v___x_534_);
v___x_536_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__6));
v___x_537_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__8));
v___x_538_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__9));
v___x_539_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_532_);
lean_ctor_set(v___x_539_, 1, v___x_538_);
v___x_540_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__11));
v___x_541_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__13, &lp_mathlib_mabs_unexpander___closed__13_once, _init_lp_mathlib_mabs_unexpander___closed__13);
v___x_542_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__14, &lp_mathlib_mabs_unexpander___closed__14_once, _init_lp_mathlib_mabs_unexpander___closed__14);
v___x_543_ = ((lean_object*)(lp_mathlib_abs_unexpander___closed__1));
v___x_544_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_544_, 0, v___x_532_);
lean_ctor_set(v___x_544_, 1, v___x_541_);
lean_ctor_set(v___x_544_, 2, v___x_542_);
lean_ctor_set(v___x_544_, 3, v___x_543_);
v___x_545_ = l_Lean_Syntax_node1(v___x_532_, v___x_540_, v___x_544_);
v___x_546_ = l_Lean_Syntax_node2(v___x_532_, v___x_537_, v___x_539_, v___x_545_);
v___x_547_ = ((lean_object*)(lp_mathlib_mabs_unexpander___closed__17));
v___x_548_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_548_, 0, v___x_532_);
lean_ctor_set(v___x_548_, 1, v___x_547_);
v___x_549_ = l_Lean_Syntax_node3(v___x_532_, v___x_536_, v___x_546_, v___x_426_, v___x_548_);
v___x_550_ = lean_obj_once(&lp_mathlib_mabs_unexpander___closed__2, &lp_mathlib_mabs_unexpander___closed__2_once, _init_lp_mathlib_mabs_unexpander___closed__2);
v___x_551_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_551_, 0, v___x_532_);
lean_ctor_set(v___x_551_, 1, v___x_510_);
lean_ctor_set(v___x_551_, 2, v___x_550_);
v___x_552_ = l_Lean_Syntax_node4(v___x_532_, v___x_427_, v___x_535_, v___x_549_, v___x_551_, v___x_534_);
v___x_553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_553_, 0, v___x_552_);
lean_ctor_set(v___x_553_, 1, v_a_415_);
return v___x_553_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs_unexpander___boxed(lean_object* v_x_554_, lean_object* v_a_555_, lean_object* v_a_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_abs_unexpander(v_x_554_, v_a_555_, v_a_556_);
lean_dec(v_a_555_);
return v_res_557_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Even(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Lattice(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Even(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Abs(builtin);
}
#ifdef __cplusplus
}
#endif
