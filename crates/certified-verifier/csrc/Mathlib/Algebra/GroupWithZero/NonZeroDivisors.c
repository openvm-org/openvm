// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.NonZeroDivisors
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Membership public import Mathlib.Algebra.GroupWithZero.Action.Defs public import Mathlib.Algebra.GroupWithZero.Associated public import Mathlib.Algebra.GroupWithZero.Regular public import Mathlib.Algebra.Regular.SMul public import Mathlib.Algebra.BigOperators.Group.Finset.Defs import Mathlib.Algebra.GroupWithZero.Action.Regular
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
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsLeft(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsLeft___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsRight(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsRight___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_nonZeroDivisors_term___u2070___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "nonZeroDivisors"};
static const lean_object* lp_mathlib_nonZeroDivisors_term___u2070___closed__0 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__0_value;
static const lean_string_object lp_mathlib_nonZeroDivisors_term___u2070___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_⁰"};
static const lean_object* lp_mathlib_nonZeroDivisors_term___u2070___closed__1 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__1_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors_term___u2070___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__0_value),LEAN_SCALAR_PTR_LITERAL(15, 158, 93, 131, 113, 208, 91, 70)}};
static const lean_ctor_object lp_mathlib_nonZeroDivisors_term___u2070___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__2_value_aux_0),((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 124, 200, 6, 234, 196, 157, 29)}};
static const lean_object* lp_mathlib_nonZeroDivisors_term___u2070___closed__2 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__2_value;
static const lean_string_object lp_mathlib_nonZeroDivisors_term___u2070___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⁰"};
static const lean_object* lp_mathlib_nonZeroDivisors_term___u2070___closed__3 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__3_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors_term___u2070___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__3_value)}};
static const lean_object* lp_mathlib_nonZeroDivisors_term___u2070___closed__4 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__4_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors_term___u2070___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__2_value),((lean_object*)(((size_t)(9000) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__4_value)}};
static const lean_object* lp_mathlib_nonZeroDivisors_term___u2070___closed__5 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_nonZeroDivisors_term___u2070 = (const lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__5_value;
static const lean_string_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__0 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__0_value;
static const lean_string_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__1 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__1_value;
static const lean_string_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__2 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__2_value;
static const lean_string_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__3 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__3_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4_value;
static lean_once_cell_t lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__5;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroDivisors_term___u2070___closed__0_value),LEAN_SCALAR_PTR_LITERAL(15, 158, 93, 131, 113, 208, 91, 70)}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__6 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__6_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__7 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__7_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__6_value)}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__8 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__8_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__9 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__9_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__7_value),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__9_value)}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__10 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__10_value;
static const lean_string_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__11 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__11_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__12 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__0 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__0_value;
static const lean_ctor_object lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__1 = (const lean_object*)&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "nonZeroSMulDivisors"};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__0 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__0_value;
static const lean_string_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_⁰[_]"};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__1 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 69, 0, 254, 97, 197, 163, 1)}};
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(16, 119, 196, 158, 204, 81, 54, 197)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2_value;
static const lean_string_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__3 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__4 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__4_value;
static const lean_string_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⁰["};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__5 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__5_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__5_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__6 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__6_value;
static const lean_string_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__7 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__8 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__9 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__9_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__4_value),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__6_value),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__9_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__10 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__10_value;
static const lean_string_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__11 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__11_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__12 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__4_value),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__10_value),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__12_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__13 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__13_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2_value),((lean_object*)(((size_t)(9000) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__13_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__14 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__14_value;
static lean_once_cell_t lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__0;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 69, 0, 254, 97, 197, 163, 1)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__1 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__1_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__2 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__2_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__1_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__3 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__4 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__4_value;
static const lean_ctor_object lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__2_value),((lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__4_value)}};
static const lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__5 = (const lean_object*)&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroSMulDivisors__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroSMulDivisors__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLeftCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsLeftCancelMulZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLeftCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsLeftCancelMulZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRightCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsRightCancelMulZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRightCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsRightCancelMulZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_unitsNonZeroDivisorsEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_unitsNonZeroDivisorsEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___closed__0 = (const lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_unitsNonZeroDivisorsEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubmonoidClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___closed__1 = (const lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_unitsNonZeroDivisorsEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__1_value)} };
static const lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___closed__2 = (const lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__2_value;
static const lean_ctor_object lp_mathlib_unitsNonZeroDivisorsEquiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__2_value),((lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___closed__3 = (const lean_object*)&lp_mathlib_unitsNonZeroDivisorsEquiv___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_associatesNonZeroDivisorsEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_associatesNonZeroDivisorsEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_associatesNonZeroDivisorsEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesNonZeroDivisorsEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsLeft(lean_object* v_M_u2080_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsLeft___boxed(lean_object* v_M_u2080_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_nonZeroDivisorsLeft(v_M_u2080_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsRight(lean_object* v_M_u2080_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisorsRight___boxed(lean_object* v_M_u2080_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_nonZeroDivisorsRight(v_M_u2080_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors(lean_object* v_M_u2080_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___boxed(lean_object* v_M_u2080_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_nonZeroDivisors(v_M_u2080_16_, v_inst_17_);
lean_dec_ref(v_inst_17_);
return v_res_18_;
}
}
static lean_object* _init_lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__5(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = ((lean_object*)(lp_mathlib_nonZeroDivisors_term___u2070___closed__0));
v___x_43_ = l_String_toRawSubstring_x27(v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1(lean_object* v_x_60_, lean_object* v_a_61_, lean_object* v_a_62_){
_start:
{
lean_object* v___x_63_; uint8_t v___x_64_; 
v___x_63_ = ((lean_object*)(lp_mathlib_nonZeroDivisors_term___u2070___closed__2));
lean_inc(v_x_60_);
v___x_64_ = l_Lean_Syntax_isOfKind(v_x_60_, v___x_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; lean_object* v___x_66_; 
lean_dec(v_x_60_);
v___x_65_ = lean_box(1);
v___x_66_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v_a_62_);
return v___x_66_;
}
else
{
lean_object* v_quotContext_67_; lean_object* v_currMacroScope_68_; lean_object* v_ref_69_; lean_object* v___x_70_; lean_object* v___x_71_; uint8_t v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v_quotContext_67_ = lean_ctor_get(v_a_61_, 1);
v_currMacroScope_68_ = lean_ctor_get(v_a_61_, 2);
v_ref_69_ = lean_ctor_get(v_a_61_, 5);
v___x_70_ = lean_unsigned_to_nat(0u);
v___x_71_ = l_Lean_Syntax_getArg(v_x_60_, v___x_70_);
lean_dec(v_x_60_);
v___x_72_ = 0;
v___x_73_ = l_Lean_SourceInfo_fromRef(v_ref_69_, v___x_72_);
v___x_74_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4));
v___x_75_ = lean_obj_once(&lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__5, &lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__5_once, _init_lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__5);
v___x_76_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__6));
lean_inc(v_currMacroScope_68_);
lean_inc(v_quotContext_67_);
v___x_77_ = l_Lean_addMacroScope(v_quotContext_67_, v___x_76_, v_currMacroScope_68_);
v___x_78_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__10));
lean_inc_n(v___x_73_, 2);
v___x_79_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_79_, 0, v___x_73_);
lean_ctor_set(v___x_79_, 1, v___x_75_);
lean_ctor_set(v___x_79_, 2, v___x_77_);
lean_ctor_set(v___x_79_, 3, v___x_78_);
v___x_80_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__12));
v___x_81_ = l_Lean_Syntax_node1(v___x_73_, v___x_80_, v___x_71_);
v___x_82_ = l_Lean_Syntax_node2(v___x_73_, v___x_74_, v___x_79_, v___x_81_);
v___x_83_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v_a_62_);
return v___x_83_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___boxed(lean_object* v_x_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1(v_x_84_, v_a_85_, v_a_86_);
lean_dec_ref(v_a_85_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1(lean_object* v_x_91_, lean_object* v_a_92_, lean_object* v_a_93_){
_start:
{
lean_object* v___x_94_; uint8_t v___x_95_; 
v___x_94_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4));
lean_inc(v_x_91_);
v___x_95_ = l_Lean_Syntax_isOfKind(v_x_91_, v___x_94_);
if (v___x_95_ == 0)
{
lean_object* v___x_96_; lean_object* v___x_97_; 
lean_dec(v_x_91_);
v___x_96_ = lean_box(0);
v___x_97_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v_a_93_);
return v___x_97_;
}
else
{
lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; uint8_t v___x_101_; 
v___x_98_ = lean_unsigned_to_nat(0u);
v___x_99_ = l_Lean_Syntax_getArg(v_x_91_, v___x_98_);
v___x_100_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__1));
lean_inc(v___x_99_);
v___x_101_ = l_Lean_Syntax_isOfKind(v___x_99_, v___x_100_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_103_; 
lean_dec(v___x_99_);
lean_dec(v_x_91_);
v___x_102_ = lean_box(0);
v___x_103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_a_93_);
return v___x_103_;
}
else
{
lean_object* v___x_104_; lean_object* v___x_105_; uint8_t v___x_106_; 
v___x_104_ = lean_unsigned_to_nat(1u);
v___x_105_ = l_Lean_Syntax_getArg(v_x_91_, v___x_104_);
lean_dec(v_x_91_);
lean_inc(v___x_105_);
v___x_106_ = l_Lean_Syntax_matchesNull(v___x_105_, v___x_104_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; lean_object* v___x_108_; 
lean_dec(v___x_105_);
lean_dec(v___x_99_);
v___x_107_ = lean_box(0);
v___x_108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v_a_93_);
return v___x_108_;
}
else
{
lean_object* v___x_109_; lean_object* v_ref_110_; uint8_t v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_109_ = l_Lean_Syntax_getArg(v___x_105_, v___x_98_);
lean_dec(v___x_105_);
v_ref_110_ = l_Lean_replaceRef(v___x_99_, v_a_92_);
lean_dec(v___x_99_);
v___x_111_ = 0;
v___x_112_ = l_Lean_SourceInfo_fromRef(v_ref_110_, v___x_111_);
lean_dec(v_ref_110_);
v___x_113_ = ((lean_object*)(lp_mathlib_nonZeroDivisors_term___u2070___closed__2));
v___x_114_ = ((lean_object*)(lp_mathlib_nonZeroDivisors_term___u2070___closed__3));
lean_inc(v___x_112_);
v___x_115_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_112_);
lean_ctor_set(v___x_115_, 1, v___x_114_);
v___x_116_ = l_Lean_Syntax_node2(v___x_112_, v___x_113_, v___x_109_, v___x_115_);
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_a_93_);
return v___x_117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___boxed(lean_object* v_x_118_, lean_object* v_a_119_, lean_object* v_a_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1(v_x_118_, v_a_119_, v_a_120_);
lean_dec(v_a_119_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors(lean_object* v_M_u2080_122_, lean_object* v_inst_123_, lean_object* v_M_124_, lean_object* v_inst_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_box(0);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___boxed(lean_object* v_M_u2080_128_, lean_object* v_inst_129_, lean_object* v_M_130_, lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_nonZeroSMulDivisors(v_M_u2080_128_, v_inst_129_, v_M_130_, v_inst_131_, v_inst_132_);
lean_dec(v_inst_132_);
lean_dec(v_inst_131_);
lean_dec_ref(v_inst_129_);
return v_res_133_;
}
}
static lean_object* _init_lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__0(void){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__0));
v___x_169_ = l_String_toRawSubstring_x27(v___x_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1(lean_object* v_x_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_186_; uint8_t v___x_187_; 
v___x_186_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2));
lean_inc(v_x_183_);
v___x_187_ = l_Lean_Syntax_isOfKind(v_x_183_, v___x_186_);
if (v___x_187_ == 0)
{
lean_object* v___x_188_; lean_object* v___x_189_; 
lean_dec(v_x_183_);
v___x_188_ = lean_box(1);
v___x_189_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
lean_ctor_set(v___x_189_, 1, v_a_185_);
return v___x_189_;
}
else
{
lean_object* v_quotContext_190_; lean_object* v_currMacroScope_191_; lean_object* v_ref_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; uint8_t v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v_quotContext_190_ = lean_ctor_get(v_a_184_, 1);
v_currMacroScope_191_ = lean_ctor_get(v_a_184_, 2);
v_ref_192_ = lean_ctor_get(v_a_184_, 5);
v___x_193_ = lean_unsigned_to_nat(0u);
v___x_194_ = l_Lean_Syntax_getArg(v_x_183_, v___x_193_);
v___x_195_ = lean_unsigned_to_nat(2u);
v___x_196_ = l_Lean_Syntax_getArg(v_x_183_, v___x_195_);
lean_dec(v_x_183_);
v___x_197_ = 0;
v___x_198_ = l_Lean_SourceInfo_fromRef(v_ref_192_, v___x_197_);
v___x_199_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4));
v___x_200_ = lean_obj_once(&lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__0, &lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__0_once, _init_lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__0);
v___x_201_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__1));
lean_inc(v_currMacroScope_191_);
lean_inc(v_quotContext_190_);
v___x_202_ = l_Lean_addMacroScope(v_quotContext_190_, v___x_201_, v_currMacroScope_191_);
v___x_203_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___closed__5));
lean_inc_n(v___x_198_, 2);
v___x_204_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_204_, 0, v___x_198_);
lean_ctor_set(v___x_204_, 1, v___x_200_);
lean_ctor_set(v___x_204_, 2, v___x_202_);
lean_ctor_set(v___x_204_, 3, v___x_203_);
v___x_205_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__12));
v___x_206_ = l_Lean_Syntax_node2(v___x_198_, v___x_205_, v___x_194_, v___x_196_);
v___x_207_ = l_Lean_Syntax_node2(v___x_198_, v___x_199_, v___x_204_, v___x_206_);
v___x_208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_185_);
return v___x_208_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1___boxed(lean_object* v_x_209_, lean_object* v_a_210_, lean_object* v_a_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroSMulDivisors__term___u2070_x5b___x5d__1(v_x_209_, v_a_210_, v_a_211_);
lean_dec_ref(v_a_210_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroSMulDivisors__1(lean_object* v_x_213_, lean_object* v_a_214_, lean_object* v_a_215_){
_start:
{
lean_object* v___x_216_; uint8_t v___x_217_; 
v___x_216_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______macroRules__nonZeroDivisors__term___u2070__1___closed__4));
lean_inc(v_x_213_);
v___x_217_ = l_Lean_Syntax_isOfKind(v_x_213_, v___x_216_);
if (v___x_217_ == 0)
{
lean_object* v___x_218_; lean_object* v___x_219_; 
lean_dec(v_x_213_);
v___x_218_ = lean_box(0);
v___x_219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
lean_ctor_set(v___x_219_, 1, v_a_215_);
return v___x_219_;
}
else
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_220_ = lean_unsigned_to_nat(0u);
v___x_221_ = l_Lean_Syntax_getArg(v_x_213_, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib_nonZeroDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroDivisors__1___closed__1));
lean_inc(v___x_221_);
v___x_223_ = l_Lean_Syntax_isOfKind(v___x_221_, v___x_222_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; lean_object* v___x_225_; 
lean_dec(v___x_221_);
lean_dec(v_x_213_);
v___x_224_ = lean_box(0);
v___x_225_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v_a_215_);
return v___x_225_;
}
else
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; uint8_t v___x_229_; 
v___x_226_ = lean_unsigned_to_nat(1u);
v___x_227_ = l_Lean_Syntax_getArg(v_x_213_, v___x_226_);
lean_dec(v_x_213_);
v___x_228_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_227_);
v___x_229_ = l_Lean_Syntax_matchesNull(v___x_227_, v___x_228_);
if (v___x_229_ == 0)
{
lean_object* v___x_230_; lean_object* v___x_231_; 
lean_dec(v___x_227_);
lean_dec(v___x_221_);
v___x_230_ = lean_box(0);
v___x_231_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
lean_ctor_set(v___x_231_, 1, v_a_215_);
return v___x_231_;
}
else
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v_ref_234_; uint8_t v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_232_ = l_Lean_Syntax_getArg(v___x_227_, v___x_220_);
v___x_233_ = l_Lean_Syntax_getArg(v___x_227_, v___x_226_);
lean_dec(v___x_227_);
v_ref_234_ = l_Lean_replaceRef(v___x_221_, v_a_214_);
lean_dec(v___x_221_);
v___x_235_ = 0;
v___x_236_ = l_Lean_SourceInfo_fromRef(v_ref_234_, v___x_235_);
lean_dec(v_ref_234_);
v___x_237_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__2));
v___x_238_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__5));
lean_inc_n(v___x_236_, 2);
v___x_239_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_236_);
lean_ctor_set(v___x_239_, 1, v___x_238_);
v___x_240_ = ((lean_object*)(lp_mathlib_nonZeroSMulDivisors_term___u2070_x5b___x5d___closed__11));
v___x_241_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_236_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = l_Lean_Syntax_node4(v___x_236_, v___x_237_, v___x_232_, v___x_239_, v___x_233_, v___x_241_);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set(v___x_243_, 1, v_a_215_);
return v___x_243_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroSMulDivisors__1___boxed(lean_object* v_x_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_nonZeroSMulDivisors___aux__Mathlib__Algebra__GroupWithZero__NonZeroDivisors______unexpand__nonZeroSMulDivisors__1(v_x_244_, v_a_245_, v_a_246_);
lean_dec(v_a_245_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLeftCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsLeftCancelMulZero___redArg(lean_object* v_inst_248_){
_start:
{
lean_object* v_toMonoid_249_; lean_object* v___x_250_; 
v_toMonoid_249_ = lean_ctor_get(v_inst_248_, 0);
lean_inc_ref(v_toMonoid_249_);
lean_dec_ref(v_inst_248_);
v___x_250_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_toMonoid_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLeftCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsLeftCancelMulZero(lean_object* v_M_u2080_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_instLeftCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsLeftCancelMulZero___redArg(v_inst_252_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRightCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsRightCancelMulZero___redArg(lean_object* v_inst_256_){
_start:
{
lean_object* v_toMonoid_257_; lean_object* v___x_258_; 
v_toMonoid_257_ = lean_ctor_get(v_inst_256_, 0);
lean_inc_ref(v_toMonoid_257_);
lean_dec_ref(v_inst_256_);
v___x_258_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_toMonoid_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRightCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsRightCancelMulZero(lean_object* v_M_u2080_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_mathlib_instRightCancelMonoidSubtypeMemSubmonoidNonZeroDivisorsOfIsRightCancelMulZero___redArg(v_inst_260_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___lam__0(lean_object* v_u_264_){
_start:
{
lean_object* v_val_265_; lean_object* v_inv_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_273_; 
v_val_265_ = lean_ctor_get(v_u_264_, 0);
v_inv_266_ = lean_ctor_get(v_u_264_, 1);
v_isSharedCheck_273_ = !lean_is_exclusive(v_u_264_);
if (v_isSharedCheck_273_ == 0)
{
v___x_268_ = v_u_264_;
v_isShared_269_ = v_isSharedCheck_273_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_inv_266_);
lean_inc(v_val_265_);
lean_dec(v_u_264_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_273_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_271_; 
if (v_isShared_269_ == 0)
{
v___x_271_ = v___x_268_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v_val_265_);
lean_ctor_set(v_reuseFailAlloc_272_, 1, v_inv_266_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
return v___x_271_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv(lean_object* v_M_u2080_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = ((lean_object*)(lp_mathlib_unitsNonZeroDivisorsEquiv___closed__3));
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsNonZeroDivisorsEquiv___boxed(lean_object* v_M_u2080_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_unitsNonZeroDivisorsEquiv(v_M_u2080_284_, v_inst_285_);
lean_dec_ref(v_inst_285_);
return v_res_286_;
}
}
static lean_object* _init_lp_mathlib_associatesNonZeroDivisorsEquiv___closed__0(void){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_287_ = lean_box(0);
v___x_288_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_box(0), lean_box(0), v___x_287_, v___x_287_, lean_box(0), lean_box(0), lean_box(0));
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesNonZeroDivisorsEquiv(lean_object* v_M_u2080_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lean_obj_once(&lp_mathlib_associatesNonZeroDivisorsEquiv___closed__0, &lp_mathlib_associatesNonZeroDivisorsEquiv___closed__0_once, _init_lp_mathlib_associatesNonZeroDivisorsEquiv___closed__0);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesNonZeroDivisorsEquiv___boxed(lean_object* v_M_u2080_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_associatesNonZeroDivisorsEquiv(v_M_u2080_292_, v_inst_293_);
lean_dec_ref(v_inst_293_);
return v_res_294_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_SMul(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Regular(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_SMul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Regular(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
}
#ifdef __cplusplus
}
#endif
