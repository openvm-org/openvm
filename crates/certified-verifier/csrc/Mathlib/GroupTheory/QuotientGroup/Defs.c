// Lean compiler output
// Module: Mathlib.GroupTheory.QuotientGroup.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Ker public import Mathlib.GroupTheory.Congruence.Hom public import Mathlib.GroupTheory.Coset.Defs
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
lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Quotient_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_map_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_con(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_con___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_con(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_con___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_commGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_commGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addCommGroup(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "GroupTheory"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(21, 126, 254, 74, 51, 201, 216, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "QuotientGroup"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(42, 76, 215, 158, 185, 28, 159, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Defs"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(186, 70, 120, 124, 151, 246, 42, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(91, 189, 147, 64, 79, 227, 226, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(60, 166, 154, 122, 190, 188, 123, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "termQ"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__12_value),LEAN_SCALAR_PTR_LITERAL(56, 168, 59, 146, 187, 42, 89, 64)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " Q"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__13_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⧸_"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 111, 223, 60, 234, 196, 87, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "G"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__3;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(216, 211, 51, 65, 145, 150, 96, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(193, 52, 230, 80, 73, 178, 33, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 242, 112, 36, 24, 65, 120, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(73, 63, 16, 228, 41, 21, 136, 155)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(133, 47, 83, 147, 189, 244, 14, 29)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__11_value),((lean_object*)(((size_t)(2046108037) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(33, 33, 218, 69, 33, 125, 49, 32)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(162, 126, 201, 207, 123, 142, 236, 11)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(46, 224, 233, 235, 148, 248, 10, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__16_value),((lean_object*)(((size_t)(18) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(222, 45, 153, 191, 128, 166, 241, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⧸"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "N"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__21_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__22;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(144, 2, 116, 232, 240, 236, 195, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(144, 2, 116, 232, 240, 236, 195, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__24_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(17, 78, 60, 226, 45, 52, 159, 77)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__25_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(196, 74, 253, 98, 91, 226, 17, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__26_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(219, 44, 246, 180, 28, 105, 98, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__27_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(188, 53, 246, 59, 197, 214, 80, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__28_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(20, 206, 139, 169, 42, 102, 14, 182)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__29_value),((lean_object*)(((size_t)(2046108037) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(188, 29, 104, 9, 16, 169, 46, 77)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__30_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(19, 82, 127, 178, 150, 243, 115, 57)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__31_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(83, 151, 243, 22, 235, 111, 218, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__32_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__32_value),((lean_object*)(((size_t)(25) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(159, 21, 92, 30, 129, 252, 135, 195)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__35_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_subgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_subgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_orderIsoCon___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_orderIsoCon___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_orderIsoCon___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_orderIsoCon___closed__0_value;
static const lean_closure_object lp_mathlib_Subgroup_orderIsoCon___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_orderIsoCon___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_orderIsoCon___closed__1 = (const lean_object*)&lp_mathlib_Subgroup_orderIsoCon___closed__1_value;
static const lean_ctor_object lp_mathlib_Subgroup_orderIsoCon___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subgroup_orderIsoCon___closed__0_value),((lean_object*)&lp_mathlib_Subgroup_orderIsoCon___closed__1_value)}};
static const lean_object* lp_mathlib_Subgroup_orderIsoCon___closed__2 = (const lean_object*)&lp_mathlib_Subgroup_orderIsoCon___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AddSubgroup_orderIsoAddCon___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_orderIsoAddCon___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___closed__0 = (const lean_object*)&lp_mathlib_AddSubgroup_orderIsoAddCon___closed__0_value;
static const lean_closure_object lp_mathlib_AddSubgroup_orderIsoAddCon___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_orderIsoAddCon___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___closed__1 = (const lean_object*)&lp_mathlib_AddSubgroup_orderIsoAddCon___closed__1_value;
static const lean_ctor_object lp_mathlib_AddSubgroup_orderIsoAddCon___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddSubgroup_orderIsoAddCon___closed__0_value),((lean_object*)&lp_mathlib_AddSubgroup_orderIsoAddCon___closed__1_value)}};
static const lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___closed__2 = (const lean_object*)&lp_mathlib_AddSubgroup_orderIsoAddCon___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_con(lean_object* v_G_1_, lean_object* v_inst_2_, lean_object* v_N_3_, lean_object* v_nN_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_con___boxed(lean_object* v_G_6_, lean_object* v_inst_7_, lean_object* v_N_8_, lean_object* v_nN_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_QuotientGroup_con(v_G_6_, v_inst_7_, v_N_8_, v_nN_9_);
lean_dec_ref(v_inst_7_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_con(lean_object* v_G_11_, lean_object* v_inst_12_, lean_object* v_N_13_, lean_object* v_nN_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_con___boxed(lean_object* v_G_16_, lean_object* v_inst_17_, lean_object* v_N_18_, lean_object* v_nN_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_QuotientAddGroup_con(v_G_16_, v_inst_17_, v_N_18_, v_nN_19_);
lean_dec_ref(v_inst_17_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__4___redArg(lean_object* v_inst_21_, lean_object* v_n_22_, lean_object* v_x_23_){
_start:
{
lean_object* v_toMonoid_24_; lean_object* v_toNPow_25_; lean_object* v___x_26_; 
v_toMonoid_24_ = lean_ctor_get(v_inst_21_, 0);
lean_inc_ref(v_toMonoid_24_);
lean_dec_ref(v_inst_21_);
v_toNPow_25_ = lean_ctor_get(v_toMonoid_24_, 2);
lean_inc(v_toNPow_25_);
lean_dec_ref(v_toMonoid_24_);
v___x_26_ = lean_apply_2(v_toNPow_25_, v_n_22_, v_x_23_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__4(lean_object* v_G_27_, lean_object* v_inst_28_, lean_object* v_N_29_, lean_object* v_nN_30_, lean_object* v_n_31_, lean_object* v_x_32_){
_start:
{
lean_object* v_toMonoid_33_; lean_object* v_toNPow_34_; lean_object* v___x_35_; 
v_toMonoid_33_ = lean_ctor_get(v_inst_28_, 0);
lean_inc_ref(v_toMonoid_33_);
lean_dec_ref(v_inst_28_);
v_toNPow_34_ = lean_ctor_get(v_toMonoid_33_, 2);
lean_inc(v_toNPow_34_);
lean_dec_ref(v_toMonoid_33_);
v___x_35_ = lean_apply_2(v_toNPow_34_, v_n_31_, v_x_32_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__8___redArg(lean_object* v_inst_36_, lean_object* v_n_37_, lean_object* v_x_38_){
_start:
{
lean_object* v_toZPow_39_; lean_object* v___x_40_; 
v_toZPow_39_ = lean_ctor_get(v_inst_36_, 3);
lean_inc(v_toZPow_39_);
lean_dec_ref(v_inst_36_);
v___x_40_ = lean_apply_2(v_toZPow_39_, v_n_37_, v_x_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___aux__8(lean_object* v_G_41_, lean_object* v_inst_42_, lean_object* v_N_43_, lean_object* v_nN_44_, lean_object* v_n_45_, lean_object* v_x_46_){
_start:
{
lean_object* v_toZPow_47_; lean_object* v___x_48_; 
v_toZPow_47_ = lean_ctor_get(v_inst_42_, 3);
lean_inc(v_toZPow_47_);
lean_dec_ref(v_inst_42_);
v___x_48_ = lean_apply_2(v_toZPow_47_, v_n_45_, v_x_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg___lam__0(lean_object* v_toDiv_49_, lean_object* v_x1_50_, lean_object* v_x2_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_apply_2(v_toDiv_49_, v_x1_50_, v_x2_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg___lam__1(lean_object* v_toMul_53_, lean_object* v_x1_54_, lean_object* v_x2_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_apply_2(v_toMul_53_, v_x1_54_, v_x2_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg(lean_object* v_inst_57_, lean_object* v_N_58_){
_start:
{
lean_object* v___x_59_; lean_object* v_toMonoid_60_; lean_object* v_toInv_61_; lean_object* v_toDiv_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v_toOne_65_; lean_object* v_toMul_66_; lean_object* v___f_67_; lean_object* v___f_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_59_ = lean_box(0);
v_toMonoid_60_ = lean_ctor_get(v_inst_57_, 0);
v_toInv_61_ = lean_ctor_get(v_inst_57_, 1);
v_toDiv_62_ = lean_ctor_get(v_inst_57_, 2);
v___x_63_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_60_);
v___x_64_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_63_);
v_toOne_65_ = lean_ctor_get(v___x_64_, 0);
lean_inc(v_toOne_65_);
v_toMul_66_ = lean_ctor_get(v___x_64_, 1);
lean_inc(v_toMul_66_);
lean_dec_ref(v___x_64_);
lean_inc(v_toDiv_62_);
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_Quotient_group___redArg___lam__0), 3, 1);
lean_closure_set(v___f_67_, 0, v_toDiv_62_);
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_Quotient_group___redArg___lam__1), 3, 1);
lean_closure_set(v___f_68_, 0, v_toMul_66_);
v___x_69_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_69_, 0, lean_box(0));
lean_closure_set(v___x_69_, 1, lean_box(0));
lean_closure_set(v___x_69_, 2, v___x_59_);
lean_closure_set(v___x_69_, 3, v___x_59_);
lean_closure_set(v___x_69_, 4, lean_box(0));
lean_closure_set(v___x_69_, 5, v___x_59_);
lean_closure_set(v___x_69_, 6, v___f_68_);
lean_closure_set(v___x_69_, 7, lean_box(0));
lean_inc_ref(v_inst_57_);
v___x_70_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_Quotient_group___aux__4), 6, 4);
lean_closure_set(v___x_70_, 0, lean_box(0));
lean_closure_set(v___x_70_, 1, v_inst_57_);
lean_closure_set(v___x_70_, 2, v_N_58_);
lean_closure_set(v___x_70_, 3, lean_box(0));
v___x_71_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_71_, 0, v_toOne_65_);
lean_ctor_set(v___x_71_, 1, v___x_69_);
lean_ctor_set(v___x_71_, 2, v___x_70_);
lean_inc(v_toInv_61_);
v___x_72_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_x27), 7, 6);
lean_closure_set(v___x_72_, 0, lean_box(0));
lean_closure_set(v___x_72_, 1, lean_box(0));
lean_closure_set(v___x_72_, 2, v___x_59_);
lean_closure_set(v___x_72_, 3, v___x_59_);
lean_closure_set(v___x_72_, 4, v_toInv_61_);
lean_closure_set(v___x_72_, 5, lean_box(0));
v___x_73_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_73_, 0, lean_box(0));
lean_closure_set(v___x_73_, 1, lean_box(0));
lean_closure_set(v___x_73_, 2, v___x_59_);
lean_closure_set(v___x_73_, 3, v___x_59_);
lean_closure_set(v___x_73_, 4, lean_box(0));
lean_closure_set(v___x_73_, 5, v___x_59_);
lean_closure_set(v___x_73_, 6, v___f_67_);
lean_closure_set(v___x_73_, 7, lean_box(0));
v___x_74_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_Quotient_group___aux__8), 6, 4);
lean_closure_set(v___x_74_, 0, lean_box(0));
lean_closure_set(v___x_74_, 1, v_inst_57_);
lean_closure_set(v___x_74_, 2, v_N_58_);
lean_closure_set(v___x_74_, 3, lean_box(0));
v___x_75_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_71_);
lean_ctor_set(v___x_75_, 1, v___x_72_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_group(lean_object* v_G_76_, lean_object* v_inst_77_, lean_object* v_N_78_, lean_object* v_nN_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_mathlib_QuotientGroup_Quotient_group___redArg(v_inst_77_, v_N_78_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__4___redArg(lean_object* v_inst_81_, lean_object* v_n_82_, lean_object* v_x_83_){
_start:
{
lean_object* v_toAddMonoid_84_; lean_object* v_toNSMul_85_; lean_object* v___x_86_; 
v_toAddMonoid_84_ = lean_ctor_get(v_inst_81_, 0);
lean_inc_ref(v_toAddMonoid_84_);
lean_dec_ref(v_inst_81_);
v_toNSMul_85_ = lean_ctor_get(v_toAddMonoid_84_, 2);
lean_inc(v_toNSMul_85_);
lean_dec_ref(v_toAddMonoid_84_);
v___x_86_ = lean_apply_2(v_toNSMul_85_, v_n_82_, v_x_83_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__4(lean_object* v_G_87_, lean_object* v_inst_88_, lean_object* v_N_89_, lean_object* v_nN_90_, lean_object* v_n_91_, lean_object* v_x_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__4___redArg(v_inst_88_, v_n_91_, v_x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__8___redArg(lean_object* v_inst_94_, lean_object* v_n_95_, lean_object* v_x_96_){
_start:
{
lean_object* v_toZSMul_97_; lean_object* v___x_98_; 
v_toZSMul_97_ = lean_ctor_get(v_inst_94_, 3);
lean_inc(v_toZSMul_97_);
lean_dec_ref(v_inst_94_);
v___x_98_ = lean_apply_2(v_toZSMul_97_, v_n_95_, v_x_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__8(lean_object* v_G_99_, lean_object* v_inst_100_, lean_object* v_N_101_, lean_object* v_nN_102_, lean_object* v_n_103_, lean_object* v_x_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__8___redArg(v_inst_100_, v_n_103_, v_x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg___lam__0(lean_object* v_toSub_106_, lean_object* v_x1_107_, lean_object* v_x2_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lean_apply_2(v_toSub_106_, v_x1_107_, v_x2_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg___lam__1(lean_object* v_toAdd_110_, lean_object* v_x1_111_, lean_object* v_x2_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lean_apply_2(v_toAdd_110_, v_x1_111_, v_x2_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object* v_inst_114_, lean_object* v_N_115_){
_start:
{
lean_object* v___x_116_; lean_object* v_toAddMonoid_117_; lean_object* v_toNeg_118_; lean_object* v_toSub_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v_toZero_122_; lean_object* v_toAdd_123_; lean_object* v___f_124_; lean_object* v___f_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_116_ = lean_box(0);
v_toAddMonoid_117_ = lean_ctor_get(v_inst_114_, 0);
v_toNeg_118_ = lean_ctor_get(v_inst_114_, 1);
v_toSub_119_ = lean_ctor_get(v_inst_114_, 2);
v___x_120_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_117_);
v___x_121_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_120_);
v_toZero_122_ = lean_ctor_get(v___x_121_, 0);
lean_inc(v_toZero_122_);
v_toAdd_123_ = lean_ctor_get(v___x_121_, 1);
lean_inc(v_toAdd_123_);
lean_dec_ref(v___x_121_);
lean_inc(v_toSub_119_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_124_, 0, v_toSub_119_);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg___lam__1), 3, 1);
lean_closure_set(v___f_125_, 0, v_toAdd_123_);
v___x_126_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_126_, 0, lean_box(0));
lean_closure_set(v___x_126_, 1, lean_box(0));
lean_closure_set(v___x_126_, 2, v___x_116_);
lean_closure_set(v___x_126_, 3, v___x_116_);
lean_closure_set(v___x_126_, 4, lean_box(0));
lean_closure_set(v___x_126_, 5, v___x_116_);
lean_closure_set(v___x_126_, 6, v___f_125_);
lean_closure_set(v___x_126_, 7, lean_box(0));
lean_inc_ref(v_inst_114_);
v___x_127_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__4), 6, 4);
lean_closure_set(v___x_127_, 0, lean_box(0));
lean_closure_set(v___x_127_, 1, v_inst_114_);
lean_closure_set(v___x_127_, 2, v_N_115_);
lean_closure_set(v___x_127_, 3, lean_box(0));
v___x_128_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_128_, 0, v_toZero_122_);
lean_ctor_set(v___x_128_, 1, v___x_126_);
lean_ctor_set(v___x_128_, 2, v___x_127_);
lean_inc(v_toNeg_118_);
v___x_129_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_x27), 7, 6);
lean_closure_set(v___x_129_, 0, lean_box(0));
lean_closure_set(v___x_129_, 1, lean_box(0));
lean_closure_set(v___x_129_, 2, v___x_116_);
lean_closure_set(v___x_129_, 3, v___x_116_);
lean_closure_set(v___x_129_, 4, v_toNeg_118_);
lean_closure_set(v___x_129_, 5, lean_box(0));
v___x_130_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_130_, 0, lean_box(0));
lean_closure_set(v___x_130_, 1, lean_box(0));
lean_closure_set(v___x_130_, 2, v___x_116_);
lean_closure_set(v___x_130_, 3, v___x_116_);
lean_closure_set(v___x_130_, 4, lean_box(0));
lean_closure_set(v___x_130_, 5, v___x_116_);
lean_closure_set(v___x_130_, 6, v___f_124_);
lean_closure_set(v___x_130_, 7, lean_box(0));
v___x_131_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_Quotient_addGroup___aux__8), 6, 4);
lean_closure_set(v___x_131_, 0, lean_box(0));
lean_closure_set(v___x_131_, 1, v_inst_114_);
lean_closure_set(v___x_131_, 2, v_N_115_);
lean_closure_set(v___x_131_, 3, lean_box(0));
v___x_132_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_132_, 0, v___x_128_);
lean_ctor_set(v___x_132_, 1, v___x_129_);
lean_ctor_set(v___x_132_, 2, v___x_130_);
lean_ctor_set(v___x_132_, 3, v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup(lean_object* v_G_133_, lean_object* v_inst_134_, lean_object* v_N_135_, lean_object* v_nN_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(v_inst_134_, v_N_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk_x27___redArg(lean_object* v_inst_138_, lean_object* v_N_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_140_, 0, lean_box(0));
lean_closure_set(v___x_140_, 1, v_inst_138_);
lean_closure_set(v___x_140_, 2, v_N_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk_x27(lean_object* v_G_141_, lean_object* v_inst_142_, lean_object* v_N_143_, lean_object* v_nN_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_145_, 0, lean_box(0));
lean_closure_set(v___x_145_, 1, v_inst_142_);
lean_closure_set(v___x_145_, 2, v_N_143_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk_x27___redArg(lean_object* v_inst_146_, lean_object* v_N_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_148_, 0, lean_box(0));
lean_closure_set(v___x_148_, 1, v_inst_146_);
lean_closure_set(v___x_148_, 2, v_N_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk_x27(lean_object* v_G_149_, lean_object* v_inst_150_, lean_object* v_N_151_, lean_object* v_nN_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_153_, 0, lean_box(0));
lean_closure_set(v___x_153_, 1, v_inst_150_);
lean_closure_set(v___x_153_, 2, v_N_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_commGroup___redArg(lean_object* v_inst_154_, lean_object* v_N_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_QuotientGroup_Quotient_group___redArg(v_inst_154_, v_N_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_Quotient_commGroup(lean_object* v_G_157_, lean_object* v_inst_158_, lean_object* v_N_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_QuotientGroup_Quotient_group___redArg(v_inst_158_, v_N_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addCommGroup___redArg(lean_object* v_inst_161_, lean_object* v_N_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(v_inst_161_, v_N_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_Quotient_addCommGroup(lean_object* v_G_164_, lean_object* v_inst_165_, lean_object* v_N_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(v_inst_165_, v_N_166_);
return v___x_167_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__3(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_210_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__2));
v___x_211_ = l_String_toRawSubstring_x27(v___x_210_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__22(void){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_255_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__21));
v___x_256_ = l_String_toRawSubstring_x27(v___x_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1(lean_object* v_x_295_, lean_object* v_a_296_, lean_object* v_a_297_){
_start:
{
lean_object* v___x_298_; uint8_t v___x_299_; 
v___x_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup_termQ___closed__13));
v___x_299_ = l_Lean_Syntax_isOfKind(v_x_295_, v___x_298_);
if (v___x_299_ == 0)
{
lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_300_ = lean_box(1);
v___x_301_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_300_);
lean_ctor_set(v___x_301_, 1, v_a_297_);
return v___x_301_;
}
else
{
lean_object* v_quotContext_302_; lean_object* v_currMacroScope_303_; lean_object* v_ref_304_; uint8_t v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v_quotContext_302_ = lean_ctor_get(v_a_296_, 1);
v_currMacroScope_303_ = lean_ctor_get(v_a_296_, 2);
v_ref_304_ = lean_ctor_get(v_a_296_, 5);
v___x_305_ = 0;
v___x_306_ = l_Lean_SourceInfo_fromRef(v_ref_304_, v___x_305_);
v___x_307_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__1));
v___x_308_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__3, &lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__3_once, _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__3);
v___x_309_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__4));
lean_inc_n(v_currMacroScope_303_, 2);
lean_inc_n(v_quotContext_302_, 2);
v___x_310_ = l_Lean_addMacroScope(v_quotContext_302_, v___x_309_, v_currMacroScope_303_);
v___x_311_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__19));
lean_inc_n(v___x_306_, 3);
v___x_312_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_312_, 0, v___x_306_);
lean_ctor_set(v___x_312_, 1, v___x_308_);
lean_ctor_set(v___x_312_, 2, v___x_310_);
lean_ctor_set(v___x_312_, 3, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__20));
v___x_314_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_314_, 0, v___x_306_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
v___x_315_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__22, &lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__22_once, _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__22);
v___x_316_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__23));
v___x_317_ = l_Lean_addMacroScope(v_quotContext_302_, v___x_316_, v_currMacroScope_303_);
v___x_318_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___closed__35));
v___x_319_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_319_, 0, v___x_306_);
lean_ctor_set(v___x_319_, 1, v___x_315_);
lean_ctor_set(v___x_319_, 2, v___x_317_);
lean_ctor_set(v___x_319_, 3, v___x_318_);
v___x_320_ = l_Lean_Syntax_node3(v___x_306_, v___x_307_, v___x_312_, v___x_314_, v___x_319_);
v___x_321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_320_);
lean_ctor_set(v___x_321_, 1, v_a_297_);
return v___x_321_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1___boxed(lean_object* v_x_322_, lean_object* v_a_323_, lean_object* v_a_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Defs_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Defs______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Defs__0__QuotientGroup__termQ__1(v_x_322_, v_a_323_, v_a_324_);
lean_dec_ref(v_a_323_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_subgroup(lean_object* v_G_326_, lean_object* v_inst_327_, lean_object* v_c_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lean_box(0);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_subgroup___boxed(lean_object* v_G_330_, lean_object* v_inst_331_, lean_object* v_c_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_Con_subgroup(v_G_330_, v_inst_331_, v_c_332_);
lean_dec_ref(v_inst_331_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubgroup(lean_object* v_G_334_, lean_object* v_inst_335_, lean_object* v_c_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lean_box(0);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubgroup___boxed(lean_object* v_G_338_, lean_object* v_inst_339_, lean_object* v_c_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_AddCon_addSubgroup(v_G_338_, v_inst_339_, v_c_340_);
lean_dec_ref(v_inst_339_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon___lam__0(lean_object* v_N_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lean_box(0);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon___lam__1(lean_object* v_c_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lean_box(0);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon(lean_object* v_G_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = ((lean_object*)(lp_mathlib_Subgroup_orderIsoCon___closed__2));
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_orderIsoCon___boxed(lean_object* v_G_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_Subgroup_orderIsoCon(v_G_354_, v_inst_355_);
lean_dec_ref(v_inst_355_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___lam__0(lean_object* v_N_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lean_box(0);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___lam__1(lean_object* v_c_359_){
_start:
{
lean_object* v___x_360_; 
v___x_360_ = lean_box(0);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon(lean_object* v_G_366_, lean_object* v_inst_367_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = ((lean_object*)(lp_mathlib_AddSubgroup_orderIsoAddCon___closed__2));
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_orderIsoAddCon___boxed(lean_object* v_G_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_AddSubgroup_orderIsoAddCon(v_G_369_, v_inst_370_);
lean_dec_ref(v_inst_370_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___redArg(lean_object* v_00_u03c6_372_){
_start:
{
lean_object* v___f_373_; 
v___f_373_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_373_, 0, v_00_u03c6_372_);
return v___f_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift(lean_object* v_G_374_, lean_object* v_M_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_N_378_, lean_object* v_nN_379_, lean_object* v_00_u03c6_380_, lean_object* v_HN_381_){
_start:
{
lean_object* v___f_382_; 
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_382_, 0, v_00_u03c6_380_);
return v___f_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___boxed(lean_object* v_G_383_, lean_object* v_M_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_N_387_, lean_object* v_nN_388_, lean_object* v_00_u03c6_389_, lean_object* v_HN_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_QuotientGroup_lift(v_G_383_, v_M_384_, v_inst_385_, v_inst_386_, v_N_387_, v_nN_388_, v_00_u03c6_389_, v_HN_390_);
lean_dec_ref(v_inst_386_);
lean_dec_ref(v_inst_385_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_lift___redArg(lean_object* v_00_u03c6_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_393_, 0, v_00_u03c6_392_);
return v___f_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_lift(lean_object* v_G_394_, lean_object* v_M_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_N_398_, lean_object* v_nN_399_, lean_object* v_00_u03c6_400_, lean_object* v_HN_401_){
_start:
{
lean_object* v___f_402_; 
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_402_, 0, v_00_u03c6_400_);
return v___f_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_lift___boxed(lean_object* v_G_403_, lean_object* v_M_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_N_407_, lean_object* v_nN_408_, lean_object* v_00_u03c6_409_, lean_object* v_HN_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_QuotientAddGroup_lift(v_G_403_, v_M_404_, v_inst_405_, v_inst_406_, v_N_407_, v_nN_408_, v_00_u03c6_409_, v_HN_410_);
lean_dec_ref(v_inst_406_);
lean_dec_ref(v_inst_405_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_map___redArg(lean_object* v_inst_412_, lean_object* v_M_413_, lean_object* v_f_414_){
_start:
{
lean_object* v___x_415_; lean_object* v___f_416_; lean_object* v___f_417_; 
v___x_415_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_415_, 0, lean_box(0));
lean_closure_set(v___x_415_, 1, v_inst_412_);
lean_closure_set(v___x_415_, 2, v_M_413_);
v___f_416_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_416_, 0, v_f_414_);
lean_closure_set(v___f_416_, 1, v___x_415_);
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_417_, 0, v___f_416_);
return v___f_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_map(lean_object* v_G_418_, lean_object* v_H_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_N_422_, lean_object* v_nN_423_, lean_object* v_M_424_, lean_object* v_inst_425_, lean_object* v_f_426_, lean_object* v_h_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lp_mathlib_QuotientGroup_map___redArg(v_inst_421_, v_M_424_, v_f_426_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_map___boxed(lean_object* v_G_429_, lean_object* v_H_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_N_433_, lean_object* v_nN_434_, lean_object* v_M_435_, lean_object* v_inst_436_, lean_object* v_f_437_, lean_object* v_h_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_QuotientGroup_map(v_G_429_, v_H_430_, v_inst_431_, v_inst_432_, v_N_433_, v_nN_434_, v_M_435_, v_inst_436_, v_f_437_, v_h_438_);
lean_dec_ref(v_inst_431_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_map___redArg(lean_object* v_inst_440_, lean_object* v_M_441_, lean_object* v_f_442_){
_start:
{
lean_object* v___x_443_; lean_object* v___f_444_; lean_object* v___f_445_; 
v___x_443_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_443_, 0, lean_box(0));
lean_closure_set(v___x_443_, 1, v_inst_440_);
lean_closure_set(v___x_443_, 2, v_M_441_);
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_444_, 0, v_f_442_);
lean_closure_set(v___f_444_, 1, v___x_443_);
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_445_, 0, v___f_444_);
return v___f_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_map(lean_object* v_G_446_, lean_object* v_H_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_N_450_, lean_object* v_nN_451_, lean_object* v_M_452_, lean_object* v_inst_453_, lean_object* v_f_454_, lean_object* v_h_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_QuotientAddGroup_map___redArg(v_inst_449_, v_M_452_, v_f_454_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_map___boxed(lean_object* v_G_457_, lean_object* v_H_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_N_461_, lean_object* v_nN_462_, lean_object* v_M_463_, lean_object* v_inst_464_, lean_object* v_f_465_, lean_object* v_h_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_QuotientAddGroup_map(v_G_457_, v_H_458_, v_inst_459_, v_inst_460_, v_N_461_, v_nN_462_, v_M_463_, v_inst_464_, v_f_465_, v_h_466_);
lean_dec_ref(v_inst_459_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__0(lean_object* v_e_468_, lean_object* v___y_469_){
_start:
{
lean_object* v_toFun_470_; lean_object* v___x_471_; 
v_toFun_470_ = lean_ctor_get(v_e_468_, 0);
lean_inc(v_toFun_470_);
lean_dec_ref(v_e_468_);
v___x_471_ = lean_apply_1(v_toFun_470_, v___y_469_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__1(lean_object* v_inst_472_, lean_object* v_H_x27_473_, lean_object* v___f_474_, lean_object* v___y_475_){
_start:
{
lean_object* v___x_110__overap_476_; lean_object* v___x_477_; 
v___x_110__overap_476_ = lp_mathlib_QuotientGroup_map___redArg(v_inst_472_, v_H_x27_473_, v___f_474_);
v___x_477_ = lean_apply_1(v___x_110__overap_476_, v___y_475_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__2(lean_object* v___x_478_, lean_object* v___y_479_){
_start:
{
lean_object* v_toFun_480_; lean_object* v___x_481_; 
v_toFun_480_ = lean_ctor_get(v___x_478_, 0);
lean_inc(v_toFun_480_);
lean_dec_ref(v___x_478_);
v___x_481_ = lean_apply_1(v_toFun_480_, v___y_479_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg___lam__3(lean_object* v_inst_482_, lean_object* v_G_x27_483_, lean_object* v___f_484_, lean_object* v___y_485_){
_start:
{
lean_object* v___x_117__overap_486_; lean_object* v___x_487_; 
v___x_117__overap_486_ = lp_mathlib_QuotientGroup_map___redArg(v_inst_482_, v_G_x27_483_, v___f_484_);
v___x_487_ = lean_apply_1(v___x_117__overap_486_, v___y_485_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr___redArg(lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_G_x27_490_, lean_object* v_H_x27_491_, lean_object* v_e_492_){
_start:
{
lean_object* v___f_493_; lean_object* v___f_494_; lean_object* v___x_495_; lean_object* v___f_496_; lean_object* v___f_497_; lean_object* v___x_498_; 
lean_inc_ref(v_e_492_);
v___f_493_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_congr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_493_, 0, v_e_492_);
v___f_494_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_congr___redArg___lam__1), 4, 3);
lean_closure_set(v___f_494_, 0, v_inst_489_);
lean_closure_set(v___f_494_, 1, v_H_x27_491_);
lean_closure_set(v___f_494_, 2, v___f_493_);
v___x_495_ = lp_mathlib_Equiv_symm___redArg(v_e_492_);
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_congr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_496_, 0, v___x_495_);
v___f_497_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_congr___redArg___lam__3), 4, 3);
lean_closure_set(v___f_497_, 0, v_inst_488_);
lean_closure_set(v___f_497_, 1, v_G_x27_490_);
lean_closure_set(v___f_497_, 2, v___f_496_);
v___x_498_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_498_, 0, v___f_494_);
lean_ctor_set(v___x_498_, 1, v___f_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_congr(lean_object* v_G_499_, lean_object* v_H_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_G_x27_503_, lean_object* v_H_x27_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_e_507_, lean_object* v_he_508_){
_start:
{
lean_object* v___x_509_; 
v___x_509_ = lp_mathlib_QuotientGroup_congr___redArg(v_inst_501_, v_inst_502_, v_G_x27_503_, v_H_x27_504_, v_e_507_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr___redArg___lam__1(lean_object* v_inst_510_, lean_object* v_H_x27_511_, lean_object* v___f_512_, lean_object* v___y_513_){
_start:
{
lean_object* v___x_110__overap_514_; lean_object* v___x_515_; 
v___x_110__overap_514_ = lp_mathlib_QuotientAddGroup_map___redArg(v_inst_510_, v_H_x27_511_, v___f_512_);
v___x_515_ = lean_apply_1(v___x_110__overap_514_, v___y_513_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr___redArg___lam__2(lean_object* v_inst_516_, lean_object* v_G_x27_517_, lean_object* v___f_518_, lean_object* v___y_519_){
_start:
{
lean_object* v___x_117__overap_520_; lean_object* v___x_521_; 
v___x_117__overap_520_ = lp_mathlib_QuotientAddGroup_map___redArg(v_inst_516_, v_G_x27_517_, v___f_518_);
v___x_521_ = lean_apply_1(v___x_117__overap_520_, v___y_519_);
return v___x_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr___redArg(lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_G_x27_524_, lean_object* v_H_x27_525_, lean_object* v_e_526_){
_start:
{
lean_object* v___f_527_; lean_object* v___f_528_; lean_object* v___x_529_; lean_object* v___f_530_; lean_object* v___f_531_; lean_object* v___x_532_; 
lean_inc_ref(v_e_526_);
v___f_527_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_congr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_527_, 0, v_e_526_);
v___f_528_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_congr___redArg___lam__1), 4, 3);
lean_closure_set(v___f_528_, 0, v_inst_523_);
lean_closure_set(v___f_528_, 1, v_H_x27_525_);
lean_closure_set(v___f_528_, 2, v___f_527_);
v___x_529_ = lp_mathlib_Equiv_symm___redArg(v_e_526_);
v___f_530_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_congr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_530_, 0, v___x_529_);
v___f_531_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_congr___redArg___lam__2), 4, 3);
lean_closure_set(v___f_531_, 0, v_inst_522_);
lean_closure_set(v___f_531_, 1, v_G_x27_524_);
lean_closure_set(v___f_531_, 2, v___f_530_);
v___x_532_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_532_, 0, v___f_528_);
lean_ctor_set(v___x_532_, 1, v___f_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_congr(lean_object* v_G_533_, lean_object* v_H_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_G_x27_537_, lean_object* v_H_x27_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_e_541_, lean_object* v_he_542_){
_start:
{
lean_object* v___x_543_; 
v___x_543_ = lp_mathlib_QuotientAddGroup_congr___redArg(v_inst_535_, v_inst_536_, v_G_x27_537_, v_H_x27_538_, v_e_541_);
return v___x_543_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
