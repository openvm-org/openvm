// Lean compiler output
// Module: Mathlib.RingTheory.UniqueFactorizationDomain.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Multiset.Basic public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.Algebra.GroupWithZero.Associated public import Mathlib.Algebra.GroupWithZero.Submonoid.Primal public import Mathlib.Order.WellFounded
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingTheory"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(81, 182, 200, 127, 246, 185, 232, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "UniqueFactorizationDomain"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 255, 82, 117, 15, 178, 105, 205)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Defs"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(127, 38, 164, 253, 5, 181, 238, 95)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(50, 253, 215, 8, 165, 191, 97, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_~ᵤ_"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(236, 49, 158, 200, 91, 239, 162, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ~ᵤ "};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__18_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__12_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__21_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64__ = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Associated"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(129, 148, 232, 81, 178, 191, 175, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__6(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__5));
v___x_60_ = l_String_toRawSubstring_x27(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1(lean_object* v_x_77_, lean_object* v_a_78_, lean_object* v_a_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__12));
lean_inc(v_x_77_);
v___x_82_ = l_Lean_Syntax_isOfKind(v_x_77_, v___x_81_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v_x_77_);
v___x_83_ = lean_box(1);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_79_);
return v___x_84_;
}
else
{
lean_object* v_quotContext_85_; lean_object* v_currMacroScope_86_; lean_object* v_ref_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; uint8_t v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v_quotContext_85_ = lean_ctor_get(v_a_78_, 1);
v_currMacroScope_86_ = lean_ctor_get(v_a_78_, 2);
v_ref_87_ = lean_ctor_get(v_a_78_, 5);
v___x_88_ = l_Lean_Syntax_getArg(v_x_77_, v___x_80_);
v___x_89_ = lean_unsigned_to_nat(2u);
v___x_90_ = l_Lean_Syntax_getArg(v_x_77_, v___x_89_);
lean_dec(v_x_77_);
v___x_91_ = 0;
v___x_92_ = l_Lean_SourceInfo_fromRef(v_ref_87_, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4));
v___x_94_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__6, &lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__6_once, _init_lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__6);
v___x_95_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__7));
lean_inc(v_currMacroScope_86_);
lean_inc(v_quotContext_85_);
v___x_96_ = l_Lean_addMacroScope(v_quotContext_85_, v___x_95_, v_currMacroScope_86_);
v___x_97_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__11));
lean_inc_n(v___x_92_, 2);
v___x_98_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_98_, 0, v___x_92_);
lean_ctor_set(v___x_98_, 1, v___x_94_);
lean_ctor_set(v___x_98_, 2, v___x_96_);
lean_ctor_set(v___x_98_, 3, v___x_97_);
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__13));
v___x_100_ = l_Lean_Syntax_node2(v___x_92_, v___x_99_, v___x_88_, v___x_90_);
v___x_101_ = l_Lean_Syntax_node2(v___x_92_, v___x_93_, v___x_98_, v___x_100_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v_a_79_);
return v___x_102_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___boxed(lean_object* v_x_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1(v_x_103_, v_a_104_, v_a_105_);
lean_dec_ref(v_a_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1(lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______macroRules____private__Mathlib__RingTheory__UniqueFactorizationDomain__Defs__0__term___x7e_u1d64____1___closed__4));
lean_inc(v_x_110_);
v___x_114_ = l_Lean_Syntax_isOfKind(v_x_110_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v_x_110_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_112_);
return v___x_116_;
}
else
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_117_ = lean_unsigned_to_nat(0u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_110_, v___x_117_);
v___x_119_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___closed__1));
lean_inc(v___x_118_);
v___x_120_ = l_Lean_Syntax_isOfKind(v___x_118_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v___x_118_);
lean_dec(v_x_110_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_112_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_123_ = lean_unsigned_to_nat(1u);
v___x_124_ = l_Lean_Syntax_getArg(v_x_110_, v___x_123_);
lean_dec(v_x_110_);
v___x_125_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_124_);
v___x_126_ = l_Lean_Syntax_matchesNull(v___x_124_, v___x_125_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec(v___x_124_);
lean_dec(v___x_118_);
v___x_127_ = lean_box(0);
v___x_128_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v_a_112_);
return v___x_128_;
}
else
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v_ref_131_; uint8_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_129_ = l_Lean_Syntax_getArg(v___x_124_, v___x_117_);
v___x_130_ = l_Lean_Syntax_getArg(v___x_124_, v___x_123_);
lean_dec(v___x_124_);
v_ref_131_ = l_Lean_replaceRef(v___x_118_, v_a_111_);
lean_dec(v___x_118_);
v___x_132_ = 0;
v___x_133_ = l_Lean_SourceInfo_fromRef(v_ref_131_, v___x_132_);
lean_dec(v_ref_131_);
v___x_134_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__12));
v___x_135_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0__term___x7e_u1d64___00__closed__15));
lean_inc(v___x_133_);
v___x_136_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_133_);
lean_ctor_set(v___x_136_, 1, v___x_135_);
v___x_137_ = l_Lean_Syntax_node3(v___x_133_, v___x_134_, v___x_129_, v___x_136_, v___x_130_);
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
lean_ctor_set(v___x_138_, 1, v_a_112_);
return v___x_138_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1___boxed(lean_object* v_x_139_, lean_object* v_a_140_, lean_object* v_a_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib___private_Mathlib_RingTheory_UniqueFactorizationDomain_Defs_0____aux__Mathlib__RingTheory__UniqueFactorizationDomain__Defs______unexpand__Associated__1(v_x_139_, v_a_140_, v_a_141_);
lean_dec(v_a_140_);
return v_res_142_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Submonoid_Primal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Submonoid_Primal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Submonoid_Primal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WellFounded(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Submonoid_Primal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WellFounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
