// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.Quotient
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Actions public import Mathlib.Data.Fintype.BigOperators public import Mathlib.Dynamics.PeriodicPts.Defs public import Mathlib.GroupTheory.Commutator.Basic public import Mathlib.GroupTheory.Coset.Basic public import Mathlib.GroupTheory.GroupAction.Basic public import Mathlib.GroupTheory.GroupAction.ConjAct public import Mathlib.GroupTheory.GroupAction.Hom public import Mathlib.GroupTheory.Subgroup.Centralizer
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_quotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_quotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_quotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_toQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_toQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofQuotientStabilizer___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofQuotientStabilizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofQuotientStabilizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofQuotientStabilizer___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofQuotientStabilizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofQuotientStabilizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "GroupTheory"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__4_value),LEAN_SCALAR_PTR_LITERAL(21, 126, 254, 74, 51, 201, 216, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "GroupAction"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__6_value),LEAN_SCALAR_PTR_LITERAL(176, 146, 31, 162, 21, 62, 246, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Quotient"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(164, 142, 243, 126, 203, 109, 21, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(133, 168, 242, 150, 1, 195, 68, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MulAction"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__11_value),LEAN_SCALAR_PTR_LITERAL(250, 123, 147, 125, 14, 146, 33, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termΩ"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__13_value),LEAN_SCALAR_PTR_LITERAL(92, 214, 225, 222, 248, 30, 102, 141)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "Ω"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__14_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_<|_"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 38, 96, 140, 215, 46, 31, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__2;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(11, 28, 55, 226, 140, 171, 240, 146)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__11_value),LEAN_SCALAR_PTR_LITERAL(28, 59, 20, 151, 135, 131, 123, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 101, 144, 155, 57, 242, 225, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__5_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "QuotientGroup"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(28, 33, 115, 233, 209, 60, 6, 50)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 67, 160, 135, 15, 59, 147, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__8_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__11_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "<|"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "orbitRel"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__19_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__20;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(98, 123, 177, 95, 99, 141, 29, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__11_value),LEAN_SCALAR_PTR_LITERAL(28, 59, 20, 151, 135, 131, 123, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(65, 252, 211, 246, 104, 52, 3, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__22_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__23_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__25_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__26_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__28_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "G"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__29_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__30;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__32_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__32_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(216, 211, 51, 65, 145, 150, 96, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__34_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__2_value),LEAN_SCALAR_PTR_LITERAL(193, 52, 230, 80, 73, 178, 33, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__35_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__35_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 242, 112, 36, 24, 65, 120, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__36_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__36_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__6_value),LEAN_SCALAR_PTR_LITERAL(99, 191, 166, 110, 130, 162, 202, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__37_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__37_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(211, 232, 82, 60, 252, 215, 14, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__38_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__38_value),((lean_object*)(((size_t)(570478017) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(141, 61, 171, 96, 98, 165, 64, 172)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__39_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__39_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__40_value),LEAN_SCALAR_PTR_LITERAL(110, 148, 85, 219, 86, 72, 248, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__41_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__42_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__41_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(194, 36, 188, 87, 245, 220, 249, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__43_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__43_value),((lean_object*)(((size_t)(5) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(16, 73, 113, 35, 36, 83, 111, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__44_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__45_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__45_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__46_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "X"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__47_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__48;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__47_value),LEAN_SCALAR_PTR_LITERAL(65, 0, 14, 115, 192, 130, 41, 147)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__49 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__49_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__47_value),LEAN_SCALAR_PTR_LITERAL(65, 0, 14, 115, 192, 130, 41, 147)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__50 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__50_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__50_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(140, 110, 221, 82, 66, 46, 48, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__51 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__51_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__51_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__2_value),LEAN_SCALAR_PTR_LITERAL(221, 22, 65, 169, 232, 237, 103, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__52 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__52_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__52_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__4_value),LEAN_SCALAR_PTR_LITERAL(238, 27, 225, 19, 39, 207, 123, 229)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__53 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__53_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__53_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 127, 149, 80, 53, 145, 144, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__54 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__54_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__54_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(127, 130, 193, 228, 16, 32, 83, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__55 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__55_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__55_value),((lean_object*)(((size_t)(570478017) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(89, 178, 124, 170, 50, 97, 23, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__56 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__56_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__56_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__40_value),LEAN_SCALAR_PTR_LITERAL(138, 65, 220, 107, 226, 220, 149, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__57 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__57_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__57_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(54, 79, 249, 92, 163, 26, 238, 96)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__58 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__58_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__58_value),((lean_object*)(((size_t)(6) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(178, 135, 56, 80, 109, 134, 23, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__59 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__59_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__59_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__60 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__60_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__60_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__61 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__61_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_b_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_b_2_, v___y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_quotient___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient(lean_object* v_G_7_, lean_object* v_X_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_H_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_quotient___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_inst_11_);
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_quotient___boxed(lean_object* v_G_15_, lean_object* v_X_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_H_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_MulAction_quotient(v_G_15_, v_X_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_H_20_, v_inst_21_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_quotient___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_quotient___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_quotient(lean_object* v_G_25_, lean_object* v_X_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_H_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___f_32_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_quotient___redArg___lam__0), 3, 1);
lean_closure_set(v___f_32_, 0, v_inst_29_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_quotient___boxed(lean_object* v_G_33_, lean_object* v_X_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_H_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_AddAction_quotient(v_G_33_, v_X_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_H_38_, v_inst_39_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_35_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_toQuotient___redArg(lean_object* v_inst_41_, lean_object* v_H_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_43_, 0, lean_box(0));
lean_closure_set(v___x_43_, 1, v_inst_41_);
lean_closure_set(v___x_43_, 2, v_H_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_toQuotient(lean_object* v_G_44_, lean_object* v_inst_45_, lean_object* v_H_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_47_, 0, lean_box(0));
lean_closure_set(v___x_47_, 1, v_inst_45_);
lean_closure_set(v___x_47_, 2, v_H_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofQuotientStabilizer___redArg(lean_object* v_inst_48_, lean_object* v_x_49_, lean_object* v_g_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lean_apply_2(v_inst_48_, v_g_50_, v_x_49_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofQuotientStabilizer(lean_object* v_G_52_, lean_object* v_X_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_x_56_, lean_object* v_g_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_apply_2(v_inst_55_, v_g_57_, v_x_56_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_ofQuotientStabilizer___boxed(lean_object* v_G_59_, lean_object* v_X_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_x_63_, lean_object* v_g_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_MulAction_ofQuotientStabilizer(v_G_59_, v_X_60_, v_inst_61_, v_inst_62_, v_x_63_, v_g_64_);
lean_dec_ref(v_inst_61_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofQuotientStabilizer___redArg(lean_object* v_inst_66_, lean_object* v_x_67_, lean_object* v_g_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lean_apply_2(v_inst_66_, v_g_68_, v_x_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofQuotientStabilizer(lean_object* v_G_70_, lean_object* v_X_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_x_74_, lean_object* v_g_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lean_apply_2(v_inst_73_, v_g_75_, v_x_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_ofQuotientStabilizer___boxed(lean_object* v_G_77_, lean_object* v_X_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_x_81_, lean_object* v_g_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_AddAction_ofQuotientStabilizer(v_G_77_, v_X_78_, v_inst_79_, v_inst_80_, v_x_81_, v_g_82_);
lean_dec_ref(v_inst_79_);
return v_res_83_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__2(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__8));
v___x_127_ = l_String_toRawSubstring_x27(v___x_126_);
return v___x_127_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__20(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__19));
v___x_165_ = l_String_toRawSubstring_x27(v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__30(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_186_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__29));
v___x_187_ = l_String_toRawSubstring_x27(v___x_186_);
return v___x_187_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__48(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__47));
v___x_231_ = l_String_toRawSubstring_x27(v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1(lean_object* v_x_270_, lean_object* v_a_271_, lean_object* v_a_272_){
_start:
{
lean_object* v___x_273_; uint8_t v___x_274_; 
v___x_273_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction_term_u03a9___closed__14));
v___x_274_ = l_Lean_Syntax_isOfKind(v_x_270_, v___x_273_);
if (v___x_274_ == 0)
{
lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_275_ = lean_box(1);
v___x_276_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
lean_ctor_set(v___x_276_, 1, v_a_272_);
return v___x_276_;
}
else
{
lean_object* v_quotContext_277_; lean_object* v_currMacroScope_278_; lean_object* v_ref_279_; uint8_t v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
v_quotContext_277_ = lean_ctor_get(v_a_271_, 1);
v_currMacroScope_278_ = lean_ctor_get(v_a_271_, 2);
v_ref_279_ = lean_ctor_get(v_a_271_, 5);
v___x_280_ = 0;
v___x_281_ = l_Lean_SourceInfo_fromRef(v_ref_279_, v___x_280_);
v___x_282_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__1));
v___x_283_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__2, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__2_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__2);
v___x_284_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__3));
lean_inc_n(v_currMacroScope_278_, 4);
lean_inc_n(v_quotContext_277_, 4);
v___x_285_ = l_Lean_addMacroScope(v_quotContext_277_, v___x_284_, v_currMacroScope_278_);
v___x_286_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__12));
lean_inc_n(v___x_281_, 7);
v___x_287_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_287_, 0, v___x_281_);
lean_ctor_set(v___x_287_, 1, v___x_283_);
lean_ctor_set(v___x_287_, 2, v___x_285_);
lean_ctor_set(v___x_287_, 3, v___x_286_);
v___x_288_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__13));
v___x_289_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_281_);
lean_ctor_set(v___x_289_, 1, v___x_288_);
v___x_290_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__18));
v___x_291_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__20, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__20_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__20);
v___x_292_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__21));
v___x_293_ = l_Lean_addMacroScope(v_quotContext_277_, v___x_292_, v_currMacroScope_278_);
v___x_294_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__26));
v___x_295_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_295_, 0, v___x_281_);
lean_ctor_set(v___x_295_, 1, v___x_291_);
lean_ctor_set(v___x_295_, 2, v___x_293_);
lean_ctor_set(v___x_295_, 3, v___x_294_);
v___x_296_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__28));
v___x_297_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__30, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__30_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__30);
v___x_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__31));
v___x_299_ = l_Lean_addMacroScope(v_quotContext_277_, v___x_298_, v_currMacroScope_278_);
v___x_300_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__46));
v___x_301_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_301_, 0, v___x_281_);
lean_ctor_set(v___x_301_, 1, v___x_297_);
lean_ctor_set(v___x_301_, 2, v___x_299_);
lean_ctor_set(v___x_301_, 3, v___x_300_);
v___x_302_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__48, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__48_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__48);
v___x_303_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__49));
v___x_304_ = l_Lean_addMacroScope(v_quotContext_277_, v___x_303_, v_currMacroScope_278_);
v___x_305_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___closed__61));
v___x_306_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_306_, 0, v___x_281_);
lean_ctor_set(v___x_306_, 1, v___x_302_);
lean_ctor_set(v___x_306_, 2, v___x_304_);
lean_ctor_set(v___x_306_, 3, v___x_305_);
v___x_307_ = l_Lean_Syntax_node2(v___x_281_, v___x_296_, v___x_301_, v___x_306_);
v___x_308_ = l_Lean_Syntax_node2(v___x_281_, v___x_290_, v___x_295_, v___x_307_);
v___x_309_ = l_Lean_Syntax_node3(v___x_281_, v___x_282_, v___x_287_, v___x_289_, v___x_308_);
v___x_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v_a_272_);
return v___x_310_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1___boxed(lean_object* v_x_311_, lean_object* v_a_312_, lean_object* v_a_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Quotient_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Quotient______macroRules____private__Mathlib__GroupTheory__GroupAction__Quotient__0__MulAction__term_u03a9__1(v_x_311_, v_a_312_, v_a_313_);
lean_dec_ref(v_a_312_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1___redArg(lean_object* v_x_315_, lean_object* v_h__1_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lean_apply_2(v_h__1_316_, v_x_315_, lean_box(0));
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1(lean_object* v_G_318_, lean_object* v_X_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_x_322_, lean_object* v_motive_323_, lean_object* v_x_324_, lean_object* v_h__1_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lean_apply_2(v_h__1_325_, v_x_324_, lean_box(0));
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1___boxed(lean_object* v_G_327_, lean_object* v_X_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_x_331_, lean_object* v_motive_332_, lean_object* v_x_333_, lean_object* v_h__1_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_AddAction_sigmaFixedByEquivOrbitsProdAddGroup_match__1(v_G_327_, v_X_328_, v_inst_329_, v_inst_330_, v_x_331_, v_motive_332_, v_x_333_, v_h__1_334_);
lean_dec(v_x_331_);
lean_dec(v_inst_330_);
lean_dec_ref(v_inst_329_);
return v_res_335_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Quotient(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Quotient(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Quotient(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Actions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_GroupAction_Quotient(builtin);
}
#ifdef __cplusplus
}
#endif
