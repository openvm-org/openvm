// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.CompTypeclasses public import Mathlib.Algebra.Module.Defs public import Mathlib.Algebra.Notation.Prod public import Mathlib.Algebra.Regular.SMul public import Mathlib.Algebra.Ring.Action.Basic
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MulActionHomLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 18, .m_data = "MulActionHomLocal≺"};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(36, 53, 18, 162, 29, 115, 158, 90)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_MulActionHomLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value;
static const lean_string_object lp_mathlib_MulActionHomLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " →ₑ["};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__4_value)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__5_value;
static const lean_string_object lp_mathlib_MulActionHomLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__7_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__8 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__9 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__9_value;
static const lean_string_object lp_mathlib_MulActionHomLocal_u227a___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__10 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__10_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__10_value)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__11 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__9_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__12 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__12_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__13 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__12_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__14 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__14_value;
static const lean_ctor_object lp_mathlib_MulActionHomLocal_u227a___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__14_value)}};
static const lean_object* lp_mathlib_MulActionHomLocal_u227a___closed__15 = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_MulActionHomLocal_u227a = (const lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__15_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulActionHom"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(183, 69, 93, 9, 61, 31, 28, 199)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MulActionHomIdLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 20, .m_data = "MulActionHomIdLocal≺"};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_MulActionHomIdLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 189, 26, 250, 199, 54, 119, 0)}};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_MulActionHomIdLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " →["};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_MulActionHomIdLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_MulActionHomIdLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_MulActionHomIdLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_MulActionHomIdLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_MulActionHomIdLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_MulActionHomIdLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_MulActionHomIdLocal_u227a = (const lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__7_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__11_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__14_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(223, 78, 141, 85, 50, 255, 216, 83)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__17_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__18_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_AddActionHomLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 18, .m_data = "AddActionHomLocal≺"};
static const lean_object* lp_mathlib_AddActionHomLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_AddActionHomLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_AddActionHomLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddActionHomLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 30, 209, 148, 130, 239, 227, 149)}};
static const lean_object* lp_mathlib_AddActionHomLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_AddActionHomLocal_u227a___closed__1_value;
static const lean_ctor_object lp_mathlib_AddActionHomLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_AddActionHomLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__14_value)}};
static const lean_object* lp_mathlib_AddActionHomLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_AddActionHomLocal_u227a___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_AddActionHomLocal_u227a = (const lean_object*)&lp_mathlib_AddActionHomLocal_u227a___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddActionHom"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(21, 238, 63, 118, 120, 67, 32, 84)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_AddActionHomIdLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 20, .m_data = "AddActionHomIdLocal≺"};
static const lean_object* lp_mathlib_AddActionHomIdLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_AddActionHomIdLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_AddActionHomIdLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddActionHomIdLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 184, 76, 115, 206, 13, 215, 29)}};
static const lean_object* lp_mathlib_AddActionHomIdLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_AddActionHomIdLocal_u227a___closed__1_value;
static const lean_ctor_object lp_mathlib_AddActionHomIdLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_AddActionHomIdLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHomIdLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_AddActionHomIdLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_AddActionHomIdLocal_u227a___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_AddActionHomIdLocal_u227a = (const lean_object*)&lp_mathlib_AddActionHomIdLocal_u227a___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomIdLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomIdLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeMulActionHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instFunLikeMulActionHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instFunLikeMulActionHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instFunLikeMulActionHom___closed__0 = (const lean_object*)&lp_mathlib_instFunLikeMulActionHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeAddActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeAddActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionSemiHomClass_toMulActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionSemiHomClass_toMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionSemiHomClass_toMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionSemiHomClass_toAddActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionSemiHomClass_toAddActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionSemiHomClass_toAddActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCoeTCOfMulActionSemiHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCoeTCOfMulActionSemiHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instCoeTCOfAddActionSemiHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instCoeTCOfAddActionSemiHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulActionHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulActionHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulActionHom_id___closed__0 = (const lean_object*)&lp_mathlib_MulActionHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAddCommClass_toAddActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAddCommClass_toAddActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAddCommClass_toAddActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulActionHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddActionHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulActionHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulActionHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulActionHom_fst___closed__0 = (const lean_object*)&lp_mathlib_MulActionHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulActionHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulActionHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulActionHom_snd___closed__0 = (const lean_object*)&lp_mathlib_MulActionHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instVAddOfVAddCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instVAddOfVAddCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instVAddOfVAddCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMulActionOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMulActionOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMulActionOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instAddActionOfVAddCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instAddActionOfVAddCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instAddActionOfVAddCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instModuleOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instModuleOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instModuleOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_instMonoidId___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulActionHom_End_instMonoidId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulActionHom_End_instMonoidId___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulActionHom_End_instMonoidId___closed__0 = (const lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__0_value;
static const lean_closure_object lp_mathlib_MulActionHom_End_instMonoidId___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowBinRecAuto___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__0_value),((lean_object*)&lp_mathlib_MulActionHom_id___closed__0_value)} };
static const lean_object* lp_mathlib_MulActionHom_End_instMonoidId___closed__1 = (const lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__1_value;
static const lean_ctor_object lp_mathlib_MulActionHom_End_instMonoidId___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHom_id___closed__0_value),((lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__0_value),((lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__1_value)}};
static const lean_object* lp_mathlib_MulActionHom_End_instMonoidId___closed__2 = (const lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_instMonoidId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_instMonoidId___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddActionHom_End_instAddMonoidId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_nsmulBinRecAuto___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__0_value),((lean_object*)&lp_mathlib_MulActionHom_id___closed__0_value)} };
static const lean_object* lp_mathlib_AddActionHom_End_instAddMonoidId___closed__0 = (const lean_object*)&lp_mathlib_AddActionHom_End_instAddMonoidId___closed__0_value;
static const lean_ctor_object lp_mathlib_AddActionHom_End_instAddMonoidId___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHom_id___closed__0_value),((lean_object*)&lp_mathlib_MulActionHom_End_instMonoidId___closed__0_value),((lean_object*)&lp_mathlib_AddActionHom_End_instAddMonoidId___closed__0_value)}};
static const lean_object* lp_mathlib_AddActionHom_End_instAddMonoidId___closed__1 = (const lean_object*)&lp_mathlib_AddActionHom_End_instAddMonoidId___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_instAddMonoidId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_instAddMonoidId___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 25, .m_data = "DistribMulActionHomLocal≺"};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 96, 112, 150, 102, 116, 99, 189)}};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " →ₑ+["};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_DistribMulActionHomLocal_u227a = (const lean_object*)&lp_mathlib_DistribMulActionHomLocal_u227a___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "DistribMulActionHom"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 113, 200, 142, 57, 94, 207, 36)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 27, .m_data = "DistribMulActionHomIdLocal≺"};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 59, 213, 160, 88, 32, 39, 164)}};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " →+["};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_DistribMulActionHomIdLocal_u227a = (const lean_object*)&lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MonoidHom.id"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MonoidHom"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(42, 146, 241, 17, 119, 0, 235, 30)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(26, 88, 52, 83, 162, 234, 201, 148)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 28, .m_data = "MulDistribMulActionHomLocal≺"};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(219, 160, 65, 137, 240, 110, 221, 104)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " →ₑ*["};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_MulDistribMulActionHomLocal_u227a = (const lean_object*)&lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "MulDistribMulActionHom"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 246, 94, 236, 151, 131, 174, 238)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 30, .m_data = "MulDistribMulActionHomIdLocal≺"};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 75, 96, 89, 159, 84, 107, 104)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " →*["};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_MulDistribMulActionHomIdLocal_u227a = (const lean_object*)&lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomIdLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomIdLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulDistribMulActionHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulDistribMulActionHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_MulDistribMulActionHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instCoeTCOfMulDistribMulActionSemiHomClassCoeMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instCoeTCOfMulDistribMulActionSemiHomClassCoeMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instCoeTCOfAddDistribAddActionSemiHomClassCoeAddMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instCoeTCOfAddDistribAddActionSemiHomClassCoeAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toDistribMulActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toDistribMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toDistribMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_id(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_id___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_id(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_id___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instOneId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instOneId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instZeroId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instZeroId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_comp___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_comp___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 26, .m_data = "MulSemiringActionHomLocal≺"};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(91, 72, 191, 246, 44, 145, 56, 109)}};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = " →ₑ+*["};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_MulSemiringActionHomLocal_u227a = (const lean_object*)&lp_mathlib_MulSemiringActionHomLocal_u227a___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "MulSemiringActionHom"};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(41, 165, 77, 93, 173, 67, 132, 108)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 28, .m_data = "MulSemiringActionHomIdLocal≺"};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__0 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__0_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(131, 140, 42, 154, 4, 207, 162, 134)}};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__1 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__1_value;
static const lean_string_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " →+*["};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__2 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__2_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__2_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__3 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__3_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__8_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__4 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__4_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__4_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__11_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__5 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__5_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__3_value),((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__5_value),((lean_object*)&lp_mathlib_MulActionHomLocal_u227a___closed__13_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__6 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__6_value;
static const lean_ctor_object lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__6_value)}};
static const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__7 = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_MulSemiringActionHomIdLocal_u227a = (const lean_object*)&lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomIdLocal_u227a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomIdLocal_u227a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_instCoeTCOfMulSemiringActionSemiHomClassCoeMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_instCoeTCOfMulSemiringActionSemiHomClassCoeMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_id(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_id___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_comp___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__5));
v___x_51_ = l_String_toRawSubstring_x27(v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; uint8_t v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__1));
lean_inc(v_x_68_);
v___x_72_ = l_Lean_Syntax_isOfKind(v_x_68_, v___x_71_);
if (v___x_72_ == 0)
{
lean_object* v___x_73_; lean_object* v___x_74_; 
lean_dec(v_x_68_);
v___x_73_ = lean_box(1);
v___x_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_a_70_);
return v___x_74_;
}
else
{
lean_object* v_quotContext_75_; lean_object* v_currMacroScope_76_; lean_object* v_ref_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; uint8_t v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v_quotContext_75_ = lean_ctor_get(v_a_69_, 1);
v_currMacroScope_76_ = lean_ctor_get(v_a_69_, 2);
v_ref_77_ = lean_ctor_get(v_a_69_, 5);
v___x_78_ = lean_unsigned_to_nat(0u);
v___x_79_ = l_Lean_Syntax_getArg(v_x_68_, v___x_78_);
v___x_80_ = lean_unsigned_to_nat(2u);
v___x_81_ = l_Lean_Syntax_getArg(v_x_68_, v___x_80_);
v___x_82_ = lean_unsigned_to_nat(4u);
v___x_83_ = l_Lean_Syntax_getArg(v_x_68_, v___x_82_);
lean_dec(v_x_68_);
v___x_84_ = 0;
v___x_85_ = l_Lean_SourceInfo_fromRef(v_ref_77_, v___x_84_);
v___x_86_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_87_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6);
v___x_88_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7));
lean_inc(v_currMacroScope_76_);
lean_inc(v_quotContext_75_);
v___x_89_ = l_Lean_addMacroScope(v_quotContext_75_, v___x_88_, v_currMacroScope_76_);
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__11));
lean_inc_n(v___x_85_, 2);
v___x_91_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_91_, 0, v___x_85_);
lean_ctor_set(v___x_91_, 1, v___x_87_);
lean_ctor_set(v___x_91_, 2, v___x_89_);
lean_ctor_set(v___x_91_, 3, v___x_90_);
v___x_92_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_93_ = l_Lean_Syntax_node3(v___x_85_, v___x_92_, v___x_81_, v___x_79_, v___x_83_);
v___x_94_ = l_Lean_Syntax_node2(v___x_85_, v___x_86_, v___x_91_, v___x_93_);
v___x_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_a_70_);
return v___x_95_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___boxed(lean_object* v_x_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1(v_x_96_, v_a_97_, v_a_98_);
lean_dec_ref(v_a_97_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1(lean_object* v_x_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
lean_object* v___x_106_; uint8_t v___x_107_; 
v___x_106_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_103_);
v___x_107_ = l_Lean_Syntax_isOfKind(v_x_103_, v___x_106_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; lean_object* v___x_109_; 
lean_dec(v_x_103_);
v___x_108_ = lean_box(0);
v___x_109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v_a_105_);
return v___x_109_;
}
else
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v___x_110_ = lean_unsigned_to_nat(0u);
v___x_111_ = l_Lean_Syntax_getArg(v_x_103_, v___x_110_);
v___x_112_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_111_);
v___x_113_ = l_Lean_Syntax_isOfKind(v___x_111_, v___x_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; lean_object* v___x_115_; 
lean_dec(v___x_111_);
lean_dec(v_x_103_);
v___x_114_ = lean_box(0);
v___x_115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_105_);
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; 
v___x_116_ = lean_unsigned_to_nat(1u);
v___x_117_ = l_Lean_Syntax_getArg(v_x_103_, v___x_116_);
lean_dec(v_x_103_);
v___x_118_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_117_);
v___x_119_ = l_Lean_Syntax_matchesNull(v___x_117_, v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; 
lean_dec(v___x_117_);
lean_dec(v___x_111_);
v___x_120_ = lean_box(0);
v___x_121_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v_a_105_);
return v___x_121_;
}
else
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v_ref_126_; uint8_t v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_122_ = l_Lean_Syntax_getArg(v___x_117_, v___x_110_);
v___x_123_ = l_Lean_Syntax_getArg(v___x_117_, v___x_116_);
v___x_124_ = lean_unsigned_to_nat(2u);
v___x_125_ = l_Lean_Syntax_getArg(v___x_117_, v___x_124_);
lean_dec(v___x_117_);
v_ref_126_ = l_Lean_replaceRef(v___x_111_, v_a_104_);
lean_dec(v___x_111_);
v___x_127_ = 0;
v___x_128_ = l_Lean_SourceInfo_fromRef(v_ref_126_, v___x_127_);
lean_dec(v_ref_126_);
v___x_129_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__1));
v___x_130_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__4));
lean_inc_n(v___x_128_, 2);
v___x_131_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_128_);
lean_ctor_set(v___x_131_, 1, v___x_130_);
v___x_132_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_133_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_128_);
lean_ctor_set(v___x_133_, 1, v___x_132_);
v___x_134_ = l_Lean_Syntax_node5(v___x_128_, v___x_129_, v___x_123_, v___x_131_, v___x_122_, v___x_133_, v___x_125_);
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_105_);
return v___x_135_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___boxed(lean_object* v_x_136_, lean_object* v_a_137_, lean_object* v_a_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1(v_x_136_, v_a_137_, v_a_138_);
lean_dec(v_a_137_);
return v_res_139_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_181_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__7));
v___x_182_ = l_String_toRawSubstring_x27(v___x_181_);
return v___x_182_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_196_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__14));
v___x_197_ = l_String_toRawSubstring_x27(v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1(lean_object* v_x_207_, lean_object* v_a_208_, lean_object* v_a_209_){
_start:
{
lean_object* v___x_210_; uint8_t v___x_211_; 
v___x_210_ = ((lean_object*)(lp_mathlib_MulActionHomIdLocal_u227a___closed__1));
lean_inc(v_x_207_);
v___x_211_ = l_Lean_Syntax_isOfKind(v_x_207_, v___x_210_);
if (v___x_211_ == 0)
{
lean_object* v___x_212_; lean_object* v___x_213_; 
lean_dec(v_x_207_);
v___x_212_ = lean_box(1);
v___x_213_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_212_);
lean_ctor_set(v___x_213_, 1, v_a_209_);
return v___x_213_;
}
else
{
lean_object* v_quotContext_214_; lean_object* v_currMacroScope_215_; lean_object* v_ref_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; uint8_t v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
v_quotContext_214_ = lean_ctor_get(v_a_208_, 1);
v_currMacroScope_215_ = lean_ctor_get(v_a_208_, 2);
v_ref_216_ = lean_ctor_get(v_a_208_, 5);
v___x_217_ = lean_unsigned_to_nat(0u);
v___x_218_ = l_Lean_Syntax_getArg(v_x_207_, v___x_217_);
v___x_219_ = lean_unsigned_to_nat(2u);
v___x_220_ = l_Lean_Syntax_getArg(v_x_207_, v___x_219_);
v___x_221_ = lean_unsigned_to_nat(4u);
v___x_222_ = l_Lean_Syntax_getArg(v_x_207_, v___x_221_);
lean_dec(v_x_207_);
v___x_223_ = 0;
v___x_224_ = l_Lean_SourceInfo_fromRef(v_ref_216_, v___x_223_);
v___x_225_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_226_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__6);
v___x_227_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__7));
lean_inc_n(v_currMacroScope_215_, 3);
lean_inc_n(v_quotContext_214_, 3);
v___x_228_ = l_Lean_addMacroScope(v_quotContext_214_, v___x_227_, v_currMacroScope_215_);
v___x_229_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__11));
lean_inc_n(v___x_224_, 13);
v___x_230_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_230_, 0, v___x_224_);
lean_ctor_set(v___x_230_, 1, v___x_226_);
lean_ctor_set(v___x_230_, 2, v___x_228_);
lean_ctor_set(v___x_230_, 3, v___x_229_);
v___x_231_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_232_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1));
v___x_233_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3));
v___x_234_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4));
v___x_235_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_224_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
v___x_236_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6));
v___x_237_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8);
v___x_238_ = lean_box(0);
v___x_239_ = l_Lean_addMacroScope(v_quotContext_214_, v___x_238_, v_currMacroScope_215_);
v___x_240_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10));
v___x_241_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_241_, 0, v___x_224_);
lean_ctor_set(v___x_241_, 1, v___x_237_);
lean_ctor_set(v___x_241_, 2, v___x_239_);
lean_ctor_set(v___x_241_, 3, v___x_240_);
v___x_242_ = l_Lean_Syntax_node1(v___x_224_, v___x_236_, v___x_241_);
v___x_243_ = l_Lean_Syntax_node2(v___x_224_, v___x_233_, v___x_235_, v___x_242_);
v___x_244_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12));
v___x_245_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__13));
v___x_246_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_224_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v___x_247_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15);
v___x_248_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16));
v___x_249_ = l_Lean_addMacroScope(v_quotContext_214_, v___x_248_, v_currMacroScope_215_);
v___x_250_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__18));
v___x_251_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_251_, 0, v___x_224_);
lean_ctor_set(v___x_251_, 1, v___x_247_);
lean_ctor_set(v___x_251_, 2, v___x_249_);
lean_ctor_set(v___x_251_, 3, v___x_250_);
v___x_252_ = l_Lean_Syntax_node2(v___x_224_, v___x_244_, v___x_246_, v___x_251_);
v___x_253_ = l_Lean_Syntax_node1(v___x_224_, v___x_231_, v___x_220_);
v___x_254_ = l_Lean_Syntax_node2(v___x_224_, v___x_225_, v___x_252_, v___x_253_);
v___x_255_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19));
v___x_256_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_224_);
lean_ctor_set(v___x_256_, 1, v___x_255_);
v___x_257_ = l_Lean_Syntax_node3(v___x_224_, v___x_232_, v___x_243_, v___x_254_, v___x_256_);
v___x_258_ = l_Lean_Syntax_node3(v___x_224_, v___x_231_, v___x_257_, v___x_218_, v___x_222_);
v___x_259_ = l_Lean_Syntax_node2(v___x_224_, v___x_225_, v___x_230_, v___x_258_);
v___x_260_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_259_);
lean_ctor_set(v___x_260_, 1, v_a_209_);
return v___x_260_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___boxed(lean_object* v_x_261_, lean_object* v_a_262_, lean_object* v_a_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1(v_x_261_, v_a_262_, v_a_263_);
lean_dec_ref(v_a_262_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__2(lean_object* v_x_265_, lean_object* v_a_266_, lean_object* v_a_267_){
_start:
{
lean_object* v___x_268_; uint8_t v___x_269_; 
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_265_);
v___x_269_ = l_Lean_Syntax_isOfKind(v_x_265_, v___x_268_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; lean_object* v___x_271_; 
lean_dec(v_x_265_);
v___x_270_ = lean_box(0);
v___x_271_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
lean_ctor_set(v___x_271_, 1, v_a_267_);
return v___x_271_;
}
else
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; uint8_t v___x_275_; 
v___x_272_ = lean_unsigned_to_nat(0u);
v___x_273_ = l_Lean_Syntax_getArg(v_x_265_, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_273_);
v___x_275_ = l_Lean_Syntax_isOfKind(v___x_273_, v___x_274_);
if (v___x_275_ == 0)
{
lean_object* v___x_276_; lean_object* v___x_277_; 
lean_dec(v___x_273_);
lean_dec(v_x_265_);
v___x_276_ = lean_box(0);
v___x_277_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set(v___x_277_, 1, v_a_267_);
return v___x_277_;
}
else
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_278_ = lean_unsigned_to_nat(1u);
v___x_279_ = l_Lean_Syntax_getArg(v_x_265_, v___x_278_);
lean_dec(v_x_265_);
v___x_280_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_279_);
v___x_281_ = l_Lean_Syntax_matchesNull(v___x_279_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_283_; 
lean_dec(v___x_279_);
lean_dec(v___x_273_);
v___x_282_ = lean_box(0);
v___x_283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_282_);
lean_ctor_set(v___x_283_, 1, v_a_267_);
return v___x_283_;
}
else
{
lean_object* v___x_284_; uint8_t v___x_285_; 
v___x_284_ = l_Lean_Syntax_getArg(v___x_279_, v___x_272_);
lean_inc(v___x_284_);
v___x_285_ = l_Lean_Syntax_isOfKind(v___x_284_, v___x_268_);
if (v___x_285_ == 0)
{
lean_object* v___x_286_; lean_object* v___x_287_; 
lean_dec(v___x_284_);
lean_dec(v___x_279_);
lean_dec(v___x_273_);
v___x_286_ = lean_box(0);
v___x_287_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
lean_ctor_set(v___x_287_, 1, v_a_267_);
return v___x_287_;
}
else
{
lean_object* v___x_288_; lean_object* v___x_289_; uint8_t v___x_290_; 
v___x_288_ = l_Lean_Syntax_getArg(v___x_284_, v___x_272_);
v___x_289_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12));
lean_inc(v___x_288_);
v___x_290_ = l_Lean_Syntax_isOfKind(v___x_288_, v___x_289_);
if (v___x_290_ == 0)
{
lean_object* v___x_291_; lean_object* v___x_292_; 
lean_dec(v___x_288_);
lean_dec(v___x_284_);
lean_dec(v___x_279_);
lean_dec(v___x_273_);
v___x_291_ = lean_box(0);
v___x_292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_291_);
lean_ctor_set(v___x_292_, 1, v_a_267_);
return v___x_292_;
}
else
{
lean_object* v___x_293_; lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_293_ = l_Lean_Syntax_getArg(v___x_288_, v___x_278_);
lean_dec(v___x_288_);
v___x_294_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16));
v___x_295_ = l_Lean_Syntax_matchesIdent(v___x_293_, v___x_294_);
lean_dec(v___x_293_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; 
lean_dec(v___x_284_);
lean_dec(v___x_279_);
lean_dec(v___x_273_);
v___x_296_ = lean_box(0);
v___x_297_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v_a_267_);
return v___x_297_;
}
else
{
lean_object* v___x_298_; uint8_t v___x_299_; 
v___x_298_ = l_Lean_Syntax_getArg(v___x_284_, v___x_278_);
lean_dec(v___x_284_);
lean_inc(v___x_298_);
v___x_299_ = l_Lean_Syntax_matchesNull(v___x_298_, v___x_278_);
if (v___x_299_ == 0)
{
lean_object* v___x_300_; lean_object* v___x_301_; 
lean_dec(v___x_298_);
lean_dec(v___x_279_);
lean_dec(v___x_273_);
v___x_300_ = lean_box(0);
v___x_301_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_300_);
lean_ctor_set(v___x_301_, 1, v_a_267_);
return v___x_301_;
}
else
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v_ref_306_; uint8_t v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_302_ = l_Lean_Syntax_getArg(v___x_298_, v___x_272_);
lean_dec(v___x_298_);
v___x_303_ = l_Lean_Syntax_getArg(v___x_279_, v___x_278_);
v___x_304_ = lean_unsigned_to_nat(2u);
v___x_305_ = l_Lean_Syntax_getArg(v___x_279_, v___x_304_);
lean_dec(v___x_279_);
v_ref_306_ = l_Lean_replaceRef(v___x_273_, v_a_266_);
lean_dec(v___x_273_);
v___x_307_ = 0;
v___x_308_ = l_Lean_SourceInfo_fromRef(v_ref_306_, v___x_307_);
lean_dec(v_ref_306_);
v___x_309_ = ((lean_object*)(lp_mathlib_MulActionHomIdLocal_u227a___closed__1));
v___x_310_ = ((lean_object*)(lp_mathlib_MulActionHomIdLocal_u227a___closed__2));
lean_inc_n(v___x_308_, 2);
v___x_311_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_308_);
lean_ctor_set(v___x_311_, 1, v___x_310_);
v___x_312_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_313_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_308_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
v___x_314_ = l_Lean_Syntax_node5(v___x_308_, v___x_309_, v___x_303_, v___x_311_, v___x_302_, v___x_313_, v___x_305_);
v___x_315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_314_);
lean_ctor_set(v___x_315_, 1, v_a_267_);
return v___x_315_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__2___boxed(lean_object* v_x_316_, lean_object* v_a_317_, lean_object* v_a_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__2(v_x_316_, v_a_317_, v_a_318_);
lean_dec(v_a_317_);
return v_res_319_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_330_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__0));
v___x_331_ = l_String_toRawSubstring_x27(v___x_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1(lean_object* v_x_345_, lean_object* v_a_346_, lean_object* v_a_347_){
_start:
{
lean_object* v___x_348_; uint8_t v___x_349_; 
v___x_348_ = ((lean_object*)(lp_mathlib_AddActionHomLocal_u227a___closed__1));
lean_inc(v_x_345_);
v___x_349_ = l_Lean_Syntax_isOfKind(v_x_345_, v___x_348_);
if (v___x_349_ == 0)
{
lean_object* v___x_350_; lean_object* v___x_351_; 
lean_dec(v_x_345_);
v___x_350_ = lean_box(1);
v___x_351_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
lean_ctor_set(v___x_351_, 1, v_a_347_);
return v___x_351_;
}
else
{
lean_object* v_quotContext_352_; lean_object* v_currMacroScope_353_; lean_object* v_ref_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; uint8_t v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v_quotContext_352_ = lean_ctor_get(v_a_346_, 1);
v_currMacroScope_353_ = lean_ctor_get(v_a_346_, 2);
v_ref_354_ = lean_ctor_get(v_a_346_, 5);
v___x_355_ = lean_unsigned_to_nat(0u);
v___x_356_ = l_Lean_Syntax_getArg(v_x_345_, v___x_355_);
v___x_357_ = lean_unsigned_to_nat(2u);
v___x_358_ = l_Lean_Syntax_getArg(v_x_345_, v___x_357_);
v___x_359_ = lean_unsigned_to_nat(4u);
v___x_360_ = l_Lean_Syntax_getArg(v_x_345_, v___x_359_);
lean_dec(v_x_345_);
v___x_361_ = 0;
v___x_362_ = l_Lean_SourceInfo_fromRef(v_ref_354_, v___x_361_);
v___x_363_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_364_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1);
v___x_365_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2));
lean_inc(v_currMacroScope_353_);
lean_inc(v_quotContext_352_);
v___x_366_ = l_Lean_addMacroScope(v_quotContext_352_, v___x_365_, v_currMacroScope_353_);
v___x_367_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_362_, 2);
v___x_368_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_368_, 0, v___x_362_);
lean_ctor_set(v___x_368_, 1, v___x_364_);
lean_ctor_set(v___x_368_, 2, v___x_366_);
lean_ctor_set(v___x_368_, 3, v___x_367_);
v___x_369_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_370_ = l_Lean_Syntax_node3(v___x_362_, v___x_369_, v___x_358_, v___x_356_, v___x_360_);
v___x_371_ = l_Lean_Syntax_node2(v___x_362_, v___x_363_, v___x_368_, v___x_370_);
v___x_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v_a_347_);
return v___x_372_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___boxed(lean_object* v_x_373_, lean_object* v_a_374_, lean_object* v_a_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1(v_x_373_, v_a_374_, v_a_375_);
lean_dec_ref(v_a_374_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__1(lean_object* v_x_377_, lean_object* v_a_378_, lean_object* v_a_379_){
_start:
{
lean_object* v___x_380_; uint8_t v___x_381_; 
v___x_380_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_377_);
v___x_381_ = l_Lean_Syntax_isOfKind(v_x_377_, v___x_380_);
if (v___x_381_ == 0)
{
lean_object* v___x_382_; lean_object* v___x_383_; 
lean_dec(v_x_377_);
v___x_382_ = lean_box(0);
v___x_383_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_383_, 0, v___x_382_);
lean_ctor_set(v___x_383_, 1, v_a_379_);
return v___x_383_;
}
else
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; uint8_t v___x_387_; 
v___x_384_ = lean_unsigned_to_nat(0u);
v___x_385_ = l_Lean_Syntax_getArg(v_x_377_, v___x_384_);
v___x_386_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_385_);
v___x_387_ = l_Lean_Syntax_isOfKind(v___x_385_, v___x_386_);
if (v___x_387_ == 0)
{
lean_object* v___x_388_; lean_object* v___x_389_; 
lean_dec(v___x_385_);
lean_dec(v_x_377_);
v___x_388_ = lean_box(0);
v___x_389_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_388_);
lean_ctor_set(v___x_389_, 1, v_a_379_);
return v___x_389_;
}
else
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; uint8_t v___x_393_; 
v___x_390_ = lean_unsigned_to_nat(1u);
v___x_391_ = l_Lean_Syntax_getArg(v_x_377_, v___x_390_);
lean_dec(v_x_377_);
v___x_392_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_391_);
v___x_393_ = l_Lean_Syntax_matchesNull(v___x_391_, v___x_392_);
if (v___x_393_ == 0)
{
lean_object* v___x_394_; lean_object* v___x_395_; 
lean_dec(v___x_391_);
lean_dec(v___x_385_);
v___x_394_ = lean_box(0);
v___x_395_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
lean_ctor_set(v___x_395_, 1, v_a_379_);
return v___x_395_;
}
else
{
lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v_ref_400_; uint8_t v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_396_ = l_Lean_Syntax_getArg(v___x_391_, v___x_384_);
v___x_397_ = l_Lean_Syntax_getArg(v___x_391_, v___x_390_);
v___x_398_ = lean_unsigned_to_nat(2u);
v___x_399_ = l_Lean_Syntax_getArg(v___x_391_, v___x_398_);
lean_dec(v___x_391_);
v_ref_400_ = l_Lean_replaceRef(v___x_385_, v_a_378_);
lean_dec(v___x_385_);
v___x_401_ = 0;
v___x_402_ = l_Lean_SourceInfo_fromRef(v_ref_400_, v___x_401_);
lean_dec(v_ref_400_);
v___x_403_ = ((lean_object*)(lp_mathlib_AddActionHomLocal_u227a___closed__1));
v___x_404_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__4));
lean_inc_n(v___x_402_, 2);
v___x_405_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_402_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
v___x_406_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_407_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_407_, 0, v___x_402_);
lean_ctor_set(v___x_407_, 1, v___x_406_);
v___x_408_ = l_Lean_Syntax_node5(v___x_402_, v___x_403_, v___x_397_, v___x_405_, v___x_396_, v___x_407_, v___x_399_);
v___x_409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_408_);
lean_ctor_set(v___x_409_, 1, v_a_379_);
return v___x_409_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__1___boxed(lean_object* v_x_410_, lean_object* v_a_411_, lean_object* v_a_412_){
_start:
{
lean_object* v_res_413_; 
v_res_413_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__1(v_x_410_, v_a_411_, v_a_412_);
lean_dec(v_a_411_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomIdLocal_u227a__1(lean_object* v_x_423_, lean_object* v_a_424_, lean_object* v_a_425_){
_start:
{
lean_object* v___x_426_; uint8_t v___x_427_; 
v___x_426_ = ((lean_object*)(lp_mathlib_AddActionHomIdLocal_u227a___closed__1));
lean_inc(v_x_423_);
v___x_427_ = l_Lean_Syntax_isOfKind(v_x_423_, v___x_426_);
if (v___x_427_ == 0)
{
lean_object* v___x_428_; lean_object* v___x_429_; 
lean_dec(v_x_423_);
v___x_428_ = lean_box(1);
v___x_429_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
lean_ctor_set(v___x_429_, 1, v_a_425_);
return v___x_429_;
}
else
{
lean_object* v_quotContext_430_; lean_object* v_currMacroScope_431_; lean_object* v_ref_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; uint8_t v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; 
v_quotContext_430_ = lean_ctor_get(v_a_424_, 1);
v_currMacroScope_431_ = lean_ctor_get(v_a_424_, 2);
v_ref_432_ = lean_ctor_get(v_a_424_, 5);
v___x_433_ = lean_unsigned_to_nat(0u);
v___x_434_ = l_Lean_Syntax_getArg(v_x_423_, v___x_433_);
v___x_435_ = lean_unsigned_to_nat(2u);
v___x_436_ = l_Lean_Syntax_getArg(v_x_423_, v___x_435_);
v___x_437_ = lean_unsigned_to_nat(4u);
v___x_438_ = l_Lean_Syntax_getArg(v_x_423_, v___x_437_);
lean_dec(v_x_423_);
v___x_439_ = 0;
v___x_440_ = l_Lean_SourceInfo_fromRef(v_ref_432_, v___x_439_);
v___x_441_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_442_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__1);
v___x_443_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__2));
lean_inc_n(v_currMacroScope_431_, 3);
lean_inc_n(v_quotContext_430_, 3);
v___x_444_ = l_Lean_addMacroScope(v_quotContext_430_, v___x_443_, v_currMacroScope_431_);
v___x_445_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_440_, 13);
v___x_446_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_446_, 0, v___x_440_);
lean_ctor_set(v___x_446_, 1, v___x_442_);
lean_ctor_set(v___x_446_, 2, v___x_444_);
lean_ctor_set(v___x_446_, 3, v___x_445_);
v___x_447_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_448_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1));
v___x_449_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3));
v___x_450_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4));
v___x_451_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_451_, 0, v___x_440_);
lean_ctor_set(v___x_451_, 1, v___x_450_);
v___x_452_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6));
v___x_453_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8);
v___x_454_ = lean_box(0);
v___x_455_ = l_Lean_addMacroScope(v_quotContext_430_, v___x_454_, v_currMacroScope_431_);
v___x_456_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10));
v___x_457_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_457_, 0, v___x_440_);
lean_ctor_set(v___x_457_, 1, v___x_453_);
lean_ctor_set(v___x_457_, 2, v___x_455_);
lean_ctor_set(v___x_457_, 3, v___x_456_);
v___x_458_ = l_Lean_Syntax_node1(v___x_440_, v___x_452_, v___x_457_);
v___x_459_ = l_Lean_Syntax_node2(v___x_440_, v___x_449_, v___x_451_, v___x_458_);
v___x_460_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12));
v___x_461_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__13));
v___x_462_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_440_);
lean_ctor_set(v___x_462_, 1, v___x_461_);
v___x_463_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__15);
v___x_464_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16));
v___x_465_ = l_Lean_addMacroScope(v_quotContext_430_, v___x_464_, v_currMacroScope_431_);
v___x_466_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__18));
v___x_467_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_467_, 0, v___x_440_);
lean_ctor_set(v___x_467_, 1, v___x_463_);
lean_ctor_set(v___x_467_, 2, v___x_465_);
lean_ctor_set(v___x_467_, 3, v___x_466_);
v___x_468_ = l_Lean_Syntax_node2(v___x_440_, v___x_460_, v___x_462_, v___x_467_);
v___x_469_ = l_Lean_Syntax_node1(v___x_440_, v___x_447_, v___x_436_);
v___x_470_ = l_Lean_Syntax_node2(v___x_440_, v___x_441_, v___x_468_, v___x_469_);
v___x_471_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19));
v___x_472_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_472_, 0, v___x_440_);
lean_ctor_set(v___x_472_, 1, v___x_471_);
v___x_473_ = l_Lean_Syntax_node3(v___x_440_, v___x_448_, v___x_459_, v___x_470_, v___x_472_);
v___x_474_ = l_Lean_Syntax_node3(v___x_440_, v___x_447_, v___x_473_, v___x_434_, v___x_438_);
v___x_475_ = l_Lean_Syntax_node2(v___x_440_, v___x_441_, v___x_446_, v___x_474_);
v___x_476_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_476_, 0, v___x_475_);
lean_ctor_set(v___x_476_, 1, v_a_425_);
return v___x_476_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomIdLocal_u227a__1___boxed(lean_object* v_x_477_, lean_object* v_a_478_, lean_object* v_a_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__AddActionHomIdLocal_u227a__1(v_x_477_, v_a_478_, v_a_479_);
lean_dec_ref(v_a_478_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__2(lean_object* v_x_481_, lean_object* v_a_482_, lean_object* v_a_483_){
_start:
{
lean_object* v___x_484_; uint8_t v___x_485_; 
v___x_484_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_481_);
v___x_485_ = l_Lean_Syntax_isOfKind(v_x_481_, v___x_484_);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; lean_object* v___x_487_; 
lean_dec(v_x_481_);
v___x_486_ = lean_box(0);
v___x_487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set(v___x_487_, 1, v_a_483_);
return v___x_487_;
}
else
{
lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; uint8_t v___x_491_; 
v___x_488_ = lean_unsigned_to_nat(0u);
v___x_489_ = l_Lean_Syntax_getArg(v_x_481_, v___x_488_);
v___x_490_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_489_);
v___x_491_ = l_Lean_Syntax_isOfKind(v___x_489_, v___x_490_);
if (v___x_491_ == 0)
{
lean_object* v___x_492_; lean_object* v___x_493_; 
lean_dec(v___x_489_);
lean_dec(v_x_481_);
v___x_492_ = lean_box(0);
v___x_493_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
lean_ctor_set(v___x_493_, 1, v_a_483_);
return v___x_493_;
}
else
{
lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; uint8_t v___x_497_; 
v___x_494_ = lean_unsigned_to_nat(1u);
v___x_495_ = l_Lean_Syntax_getArg(v_x_481_, v___x_494_);
lean_dec(v_x_481_);
v___x_496_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_495_);
v___x_497_ = l_Lean_Syntax_matchesNull(v___x_495_, v___x_496_);
if (v___x_497_ == 0)
{
lean_object* v___x_498_; lean_object* v___x_499_; 
lean_dec(v___x_495_);
lean_dec(v___x_489_);
v___x_498_ = lean_box(0);
v___x_499_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_499_, 0, v___x_498_);
lean_ctor_set(v___x_499_, 1, v_a_483_);
return v___x_499_;
}
else
{
lean_object* v___x_500_; uint8_t v___x_501_; 
v___x_500_ = l_Lean_Syntax_getArg(v___x_495_, v___x_488_);
lean_inc(v___x_500_);
v___x_501_ = l_Lean_Syntax_isOfKind(v___x_500_, v___x_484_);
if (v___x_501_ == 0)
{
lean_object* v___x_502_; lean_object* v___x_503_; 
lean_dec(v___x_500_);
lean_dec(v___x_495_);
lean_dec(v___x_489_);
v___x_502_ = lean_box(0);
v___x_503_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_502_);
lean_ctor_set(v___x_503_, 1, v_a_483_);
return v___x_503_;
}
else
{
lean_object* v___x_504_; lean_object* v___x_505_; uint8_t v___x_506_; 
v___x_504_ = l_Lean_Syntax_getArg(v___x_500_, v___x_488_);
v___x_505_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__12));
lean_inc(v___x_504_);
v___x_506_ = l_Lean_Syntax_isOfKind(v___x_504_, v___x_505_);
if (v___x_506_ == 0)
{
lean_object* v___x_507_; lean_object* v___x_508_; 
lean_dec(v___x_504_);
lean_dec(v___x_500_);
lean_dec(v___x_495_);
lean_dec(v___x_489_);
v___x_507_ = lean_box(0);
v___x_508_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_507_);
lean_ctor_set(v___x_508_, 1, v_a_483_);
return v___x_508_;
}
else
{
lean_object* v___x_509_; lean_object* v___x_510_; uint8_t v___x_511_; 
v___x_509_ = l_Lean_Syntax_getArg(v___x_504_, v___x_494_);
lean_dec(v___x_504_);
v___x_510_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__16));
v___x_511_ = l_Lean_Syntax_matchesIdent(v___x_509_, v___x_510_);
lean_dec(v___x_509_);
if (v___x_511_ == 0)
{
lean_object* v___x_512_; lean_object* v___x_513_; 
lean_dec(v___x_500_);
lean_dec(v___x_495_);
lean_dec(v___x_489_);
v___x_512_ = lean_box(0);
v___x_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v_a_483_);
return v___x_513_;
}
else
{
lean_object* v___x_514_; uint8_t v___x_515_; 
v___x_514_ = l_Lean_Syntax_getArg(v___x_500_, v___x_494_);
lean_dec(v___x_500_);
lean_inc(v___x_514_);
v___x_515_ = l_Lean_Syntax_matchesNull(v___x_514_, v___x_494_);
if (v___x_515_ == 0)
{
lean_object* v___x_516_; lean_object* v___x_517_; 
lean_dec(v___x_514_);
lean_dec(v___x_495_);
lean_dec(v___x_489_);
v___x_516_ = lean_box(0);
v___x_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_517_, 0, v___x_516_);
lean_ctor_set(v___x_517_, 1, v_a_483_);
return v___x_517_;
}
else
{
lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v_ref_522_; uint8_t v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; 
v___x_518_ = l_Lean_Syntax_getArg(v___x_514_, v___x_488_);
lean_dec(v___x_514_);
v___x_519_ = l_Lean_Syntax_getArg(v___x_495_, v___x_494_);
v___x_520_ = lean_unsigned_to_nat(2u);
v___x_521_ = l_Lean_Syntax_getArg(v___x_495_, v___x_520_);
lean_dec(v___x_495_);
v_ref_522_ = l_Lean_replaceRef(v___x_489_, v_a_482_);
lean_dec(v___x_489_);
v___x_523_ = 0;
v___x_524_ = l_Lean_SourceInfo_fromRef(v_ref_522_, v___x_523_);
lean_dec(v_ref_522_);
v___x_525_ = ((lean_object*)(lp_mathlib_AddActionHomIdLocal_u227a___closed__1));
v___x_526_ = ((lean_object*)(lp_mathlib_MulActionHomIdLocal_u227a___closed__2));
lean_inc_n(v___x_524_, 2);
v___x_527_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_527_, 0, v___x_524_);
lean_ctor_set(v___x_527_, 1, v___x_526_);
v___x_528_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_529_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_529_, 0, v___x_524_);
lean_ctor_set(v___x_529_, 1, v___x_528_);
v___x_530_ = l_Lean_Syntax_node5(v___x_524_, v___x_525_, v___x_519_, v___x_527_, v___x_518_, v___x_529_, v___x_521_);
v___x_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_530_);
lean_ctor_set(v___x_531_, 1, v_a_483_);
return v___x_531_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__2___boxed(lean_object* v_x_532_, lean_object* v_a_533_, lean_object* v_a_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__AddActionHom__2(v_x_532_, v_a_533_, v_a_534_);
lean_dec(v_a_533_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeMulActionHom___lam__0(lean_object* v_self_536_, lean_object* v___y_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lean_apply_1(v_self_536_, v___y_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeMulActionHom(lean_object* v_M_540_, lean_object* v_N_541_, lean_object* v_00_u03c6_542_, lean_object* v_X_543_, lean_object* v_inst_544_, lean_object* v_Y_545_, lean_object* v_inst_546_){
_start:
{
lean_object* v___f_547_; 
v___f_547_ = ((lean_object*)(lp_mathlib_instFunLikeMulActionHom___closed__0));
return v___f_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeMulActionHom___boxed(lean_object* v_M_548_, lean_object* v_N_549_, lean_object* v_00_u03c6_550_, lean_object* v_X_551_, lean_object* v_inst_552_, lean_object* v_Y_553_, lean_object* v_inst_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_instFunLikeMulActionHom(v_M_548_, v_N_549_, v_00_u03c6_550_, v_X_551_, v_inst_552_, v_Y_553_, v_inst_554_);
lean_dec(v_inst_554_);
lean_dec(v_inst_552_);
lean_dec(v_00_u03c6_550_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeAddActionHom(lean_object* v_M_556_, lean_object* v_N_557_, lean_object* v_00_u03c6_558_, lean_object* v_X_559_, lean_object* v_inst_560_, lean_object* v_Y_561_, lean_object* v_inst_562_){
_start:
{
lean_object* v___f_563_; 
v___f_563_ = ((lean_object*)(lp_mathlib_instFunLikeMulActionHom___closed__0));
return v___f_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFunLikeAddActionHom___boxed(lean_object* v_M_564_, lean_object* v_N_565_, lean_object* v_00_u03c6_566_, lean_object* v_X_567_, lean_object* v_inst_568_, lean_object* v_Y_569_, lean_object* v_inst_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib_instFunLikeAddActionHom(v_M_564_, v_N_565_, v_00_u03c6_566_, v_X_567_, v_inst_568_, v_Y_569_, v_inst_570_);
lean_dec(v_inst_570_);
lean_dec(v_inst_568_);
lean_dec(v_00_u03c6_566_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionSemiHomClass_toMulActionHom___redArg(lean_object* v_inst_572_, lean_object* v_f_573_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lean_apply_1(v_inst_572_, v_f_573_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionSemiHomClass_toMulActionHom(lean_object* v_M_575_, lean_object* v_N_576_, lean_object* v_00_u03c6_577_, lean_object* v_X_578_, lean_object* v_inst_579_, lean_object* v_Y_580_, lean_object* v_inst_581_, lean_object* v_F_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_f_585_){
_start:
{
lean_object* v___x_586_; 
v___x_586_ = lean_apply_1(v_inst_583_, v_f_585_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionSemiHomClass_toMulActionHom___boxed(lean_object* v_M_587_, lean_object* v_N_588_, lean_object* v_00_u03c6_589_, lean_object* v_X_590_, lean_object* v_inst_591_, lean_object* v_Y_592_, lean_object* v_inst_593_, lean_object* v_F_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_f_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_MulActionSemiHomClass_toMulActionHom(v_M_587_, v_N_588_, v_00_u03c6_589_, v_X_590_, v_inst_591_, v_Y_592_, v_inst_593_, v_F_594_, v_inst_595_, v_inst_596_, v_f_597_);
lean_dec(v_inst_593_);
lean_dec(v_inst_591_);
lean_dec(v_00_u03c6_589_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionSemiHomClass_toAddActionHom___redArg(lean_object* v_inst_599_, lean_object* v_f_600_){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lean_apply_1(v_inst_599_, v_f_600_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionSemiHomClass_toAddActionHom(lean_object* v_M_602_, lean_object* v_N_603_, lean_object* v_00_u03c6_604_, lean_object* v_X_605_, lean_object* v_inst_606_, lean_object* v_Y_607_, lean_object* v_inst_608_, lean_object* v_F_609_, lean_object* v_inst_610_, lean_object* v_inst_611_, lean_object* v_f_612_){
_start:
{
lean_object* v___x_613_; 
v___x_613_ = lean_apply_1(v_inst_610_, v_f_612_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionSemiHomClass_toAddActionHom___boxed(lean_object* v_M_614_, lean_object* v_N_615_, lean_object* v_00_u03c6_616_, lean_object* v_X_617_, lean_object* v_inst_618_, lean_object* v_Y_619_, lean_object* v_inst_620_, lean_object* v_F_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_f_624_){
_start:
{
lean_object* v_res_625_; 
v_res_625_ = lp_mathlib_AddActionSemiHomClass_toAddActionHom(v_M_614_, v_N_615_, v_00_u03c6_616_, v_X_617_, v_inst_618_, v_Y_619_, v_inst_620_, v_F_621_, v_inst_622_, v_inst_623_, v_f_624_);
lean_dec(v_inst_620_);
lean_dec(v_inst_618_);
lean_dec(v_00_u03c6_616_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCoeTCOfMulActionSemiHomClass___redArg(lean_object* v_00_u03c6_626_, lean_object* v_inst_627_, lean_object* v_inst_628_, lean_object* v_inst_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lean_alloc_closure((void*)(lp_mathlib_MulActionSemiHomClass_toMulActionHom___boxed), 11, 10);
lean_closure_set(v___x_630_, 0, lean_box(0));
lean_closure_set(v___x_630_, 1, lean_box(0));
lean_closure_set(v___x_630_, 2, v_00_u03c6_626_);
lean_closure_set(v___x_630_, 3, lean_box(0));
lean_closure_set(v___x_630_, 4, v_inst_627_);
lean_closure_set(v___x_630_, 5, lean_box(0));
lean_closure_set(v___x_630_, 6, v_inst_628_);
lean_closure_set(v___x_630_, 7, lean_box(0));
lean_closure_set(v___x_630_, 8, v_inst_629_);
lean_closure_set(v___x_630_, 9, lean_box(0));
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCoeTCOfMulActionSemiHomClass(lean_object* v_M_631_, lean_object* v_N_632_, lean_object* v_00_u03c6_633_, lean_object* v_X_634_, lean_object* v_inst_635_, lean_object* v_Y_636_, lean_object* v_inst_637_, lean_object* v_F_638_, lean_object* v_inst_639_, lean_object* v_inst_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lean_alloc_closure((void*)(lp_mathlib_MulActionSemiHomClass_toMulActionHom___boxed), 11, 10);
lean_closure_set(v___x_641_, 0, lean_box(0));
lean_closure_set(v___x_641_, 1, lean_box(0));
lean_closure_set(v___x_641_, 2, v_00_u03c6_633_);
lean_closure_set(v___x_641_, 3, lean_box(0));
lean_closure_set(v___x_641_, 4, v_inst_635_);
lean_closure_set(v___x_641_, 5, lean_box(0));
lean_closure_set(v___x_641_, 6, v_inst_637_);
lean_closure_set(v___x_641_, 7, lean_box(0));
lean_closure_set(v___x_641_, 8, v_inst_639_);
lean_closure_set(v___x_641_, 9, lean_box(0));
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instCoeTCOfAddActionSemiHomClass___redArg(lean_object* v_00_u03c6_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_inst_645_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lean_alloc_closure((void*)(lp_mathlib_AddActionSemiHomClass_toAddActionHom___boxed), 11, 10);
lean_closure_set(v___x_646_, 0, lean_box(0));
lean_closure_set(v___x_646_, 1, lean_box(0));
lean_closure_set(v___x_646_, 2, v_00_u03c6_642_);
lean_closure_set(v___x_646_, 3, lean_box(0));
lean_closure_set(v___x_646_, 4, v_inst_643_);
lean_closure_set(v___x_646_, 5, lean_box(0));
lean_closure_set(v___x_646_, 6, v_inst_644_);
lean_closure_set(v___x_646_, 7, lean_box(0));
lean_closure_set(v___x_646_, 8, v_inst_645_);
lean_closure_set(v___x_646_, 9, lean_box(0));
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instCoeTCOfAddActionSemiHomClass(lean_object* v_M_647_, lean_object* v_N_648_, lean_object* v_00_u03c6_649_, lean_object* v_X_650_, lean_object* v_inst_651_, lean_object* v_Y_652_, lean_object* v_inst_653_, lean_object* v_F_654_, lean_object* v_inst_655_, lean_object* v_inst_656_){
_start:
{
lean_object* v___x_657_; 
v___x_657_ = lean_alloc_closure((void*)(lp_mathlib_AddActionSemiHomClass_toAddActionHom___boxed), 11, 10);
lean_closure_set(v___x_657_, 0, lean_box(0));
lean_closure_set(v___x_657_, 1, lean_box(0));
lean_closure_set(v___x_657_, 2, v_00_u03c6_649_);
lean_closure_set(v___x_657_, 3, lean_box(0));
lean_closure_set(v___x_657_, 4, v_inst_651_);
lean_closure_set(v___x_657_, 5, lean_box(0));
lean_closure_set(v___x_657_, 6, v_inst_653_);
lean_closure_set(v___x_657_, 7, lean_box(0));
lean_closure_set(v___x_657_, 8, v_inst_655_);
lean_closure_set(v___x_657_, 9, lean_box(0));
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq___redArg(lean_object* v_f_658_){
_start:
{
lean_inc(v_f_658_);
return v_f_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq___redArg___boxed(lean_object* v_f_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_MulActionHom_ofEq___redArg(v_f_659_);
lean_dec(v_f_659_);
return v_res_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq(lean_object* v_M_661_, lean_object* v_N_662_, lean_object* v_00_u03c6_663_, lean_object* v_X_664_, lean_object* v_inst_665_, lean_object* v_Y_666_, lean_object* v_inst_667_, lean_object* v_00_u03c6_x27_668_, lean_object* v_h_669_, lean_object* v_f_670_){
_start:
{
lean_inc(v_f_670_);
return v_f_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_ofEq___boxed(lean_object* v_M_671_, lean_object* v_N_672_, lean_object* v_00_u03c6_673_, lean_object* v_X_674_, lean_object* v_inst_675_, lean_object* v_Y_676_, lean_object* v_inst_677_, lean_object* v_00_u03c6_x27_678_, lean_object* v_h_679_, lean_object* v_f_680_){
_start:
{
lean_object* v_res_681_; 
v_res_681_ = lp_mathlib_MulActionHom_ofEq(v_M_671_, v_N_672_, v_00_u03c6_673_, v_X_674_, v_inst_675_, v_Y_676_, v_inst_677_, v_00_u03c6_x27_678_, v_h_679_, v_f_680_);
lean_dec(v_f_680_);
lean_dec(v_00_u03c6_x27_678_);
lean_dec(v_inst_677_);
lean_dec(v_inst_675_);
lean_dec(v_00_u03c6_673_);
return v_res_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq___redArg(lean_object* v_f_682_){
_start:
{
lean_inc(v_f_682_);
return v_f_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq___redArg___boxed(lean_object* v_f_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_mathlib_AddActionHom_ofEq___redArg(v_f_683_);
lean_dec(v_f_683_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq(lean_object* v_M_685_, lean_object* v_N_686_, lean_object* v_00_u03c6_687_, lean_object* v_X_688_, lean_object* v_inst_689_, lean_object* v_Y_690_, lean_object* v_inst_691_, lean_object* v_00_u03c6_x27_692_, lean_object* v_h_693_, lean_object* v_f_694_){
_start:
{
lean_inc(v_f_694_);
return v_f_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_ofEq___boxed(lean_object* v_M_695_, lean_object* v_N_696_, lean_object* v_00_u03c6_697_, lean_object* v_X_698_, lean_object* v_inst_699_, lean_object* v_Y_700_, lean_object* v_inst_701_, lean_object* v_00_u03c6_x27_702_, lean_object* v_h_703_, lean_object* v_f_704_){
_start:
{
lean_object* v_res_705_; 
v_res_705_ = lp_mathlib_AddActionHom_ofEq(v_M_695_, v_N_696_, v_00_u03c6_697_, v_X_698_, v_inst_699_, v_Y_700_, v_inst_701_, v_00_u03c6_x27_702_, v_h_703_, v_f_704_);
lean_dec(v_f_704_);
lean_dec(v_00_u03c6_x27_702_);
lean_dec(v_inst_701_);
lean_dec(v_inst_699_);
lean_dec(v_00_u03c6_697_);
return v_res_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id___lam__0(lean_object* v_x_706_){
_start:
{
lean_inc(v_x_706_);
return v_x_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id___lam__0___boxed(lean_object* v_x_707_){
_start:
{
lean_object* v_res_708_; 
v_res_708_ = lp_mathlib_MulActionHom_id___lam__0(v_x_707_);
lean_dec(v_x_707_);
return v_res_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id(lean_object* v_M_710_, lean_object* v_X_711_, lean_object* v_inst_712_){
_start:
{
lean_object* v___f_713_; 
v___f_713_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_id___boxed(lean_object* v_M_714_, lean_object* v_X_715_, lean_object* v_inst_716_){
_start:
{
lean_object* v_res_717_; 
v_res_717_ = lp_mathlib_MulActionHom_id(v_M_714_, v_X_715_, v_inst_716_);
lean_dec(v_inst_716_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_id(lean_object* v_M_718_, lean_object* v_X_719_, lean_object* v_inst_720_){
_start:
{
lean_object* v___f_721_; 
v___f_721_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_id___boxed(lean_object* v_M_722_, lean_object* v_X_723_, lean_object* v_inst_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_mathlib_AddActionHom_id(v_M_722_, v_X_723_, v_inst_724_);
lean_dec(v_inst_724_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp___redArg___lam__0(lean_object* v_f_726_, lean_object* v_g_727_, lean_object* v_x_728_){
_start:
{
lean_object* v___x_729_; lean_object* v___x_730_; 
v___x_729_ = lean_apply_1(v_f_726_, v_x_728_);
v___x_730_ = lean_apply_1(v_g_727_, v___x_729_);
return v___x_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp___redArg(lean_object* v_g_731_, lean_object* v_f_732_){
_start:
{
lean_object* v___f_733_; 
v___f_733_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_733_, 0, v_f_732_);
lean_closure_set(v___f_733_, 1, v_g_731_);
return v___f_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp(lean_object* v_M_734_, lean_object* v_N_735_, lean_object* v_P_736_, lean_object* v_00_u03c6_737_, lean_object* v_00_u03c8_738_, lean_object* v_00_u03c7_739_, lean_object* v_X_740_, lean_object* v_inst_741_, lean_object* v_Y_742_, lean_object* v_inst_743_, lean_object* v_Z_744_, lean_object* v_inst_745_, lean_object* v_g_746_, lean_object* v_f_747_, lean_object* v_00_u03ba_748_){
_start:
{
lean_object* v___f_749_; 
v___f_749_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_749_, 0, v_f_747_);
lean_closure_set(v___f_749_, 1, v_g_746_);
return v___f_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_comp___boxed(lean_object* v_M_750_, lean_object* v_N_751_, lean_object* v_P_752_, lean_object* v_00_u03c6_753_, lean_object* v_00_u03c8_754_, lean_object* v_00_u03c7_755_, lean_object* v_X_756_, lean_object* v_inst_757_, lean_object* v_Y_758_, lean_object* v_inst_759_, lean_object* v_Z_760_, lean_object* v_inst_761_, lean_object* v_g_762_, lean_object* v_f_763_, lean_object* v_00_u03ba_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_mathlib_MulActionHom_comp(v_M_750_, v_N_751_, v_P_752_, v_00_u03c6_753_, v_00_u03c8_754_, v_00_u03c7_755_, v_X_756_, v_inst_757_, v_Y_758_, v_inst_759_, v_Z_760_, v_inst_761_, v_g_762_, v_f_763_, v_00_u03ba_764_);
lean_dec(v_inst_761_);
lean_dec(v_inst_759_);
lean_dec(v_inst_757_);
lean_dec(v_00_u03c7_755_);
lean_dec(v_00_u03c8_754_);
lean_dec(v_00_u03c6_753_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_comp___redArg(lean_object* v_g_766_, lean_object* v_f_767_){
_start:
{
lean_object* v___f_768_; 
v___f_768_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_768_, 0, v_f_767_);
lean_closure_set(v___f_768_, 1, v_g_766_);
return v___f_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_comp(lean_object* v_M_769_, lean_object* v_N_770_, lean_object* v_P_771_, lean_object* v_00_u03c6_772_, lean_object* v_00_u03c8_773_, lean_object* v_00_u03c7_774_, lean_object* v_X_775_, lean_object* v_inst_776_, lean_object* v_Y_777_, lean_object* v_inst_778_, lean_object* v_Z_779_, lean_object* v_inst_780_, lean_object* v_g_781_, lean_object* v_f_782_, lean_object* v_00_u03ba_783_){
_start:
{
lean_object* v___f_784_; 
v___f_784_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_784_, 0, v_f_782_);
lean_closure_set(v___f_784_, 1, v_g_781_);
return v___f_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_comp___boxed(lean_object* v_M_785_, lean_object* v_N_786_, lean_object* v_P_787_, lean_object* v_00_u03c6_788_, lean_object* v_00_u03c8_789_, lean_object* v_00_u03c7_790_, lean_object* v_X_791_, lean_object* v_inst_792_, lean_object* v_Y_793_, lean_object* v_inst_794_, lean_object* v_Z_795_, lean_object* v_inst_796_, lean_object* v_g_797_, lean_object* v_f_798_, lean_object* v_00_u03ba_799_){
_start:
{
lean_object* v_res_800_; 
v_res_800_ = lp_mathlib_AddActionHom_comp(v_M_785_, v_N_786_, v_P_787_, v_00_u03c6_788_, v_00_u03c8_789_, v_00_u03c7_790_, v_X_791_, v_inst_792_, v_Y_793_, v_inst_794_, v_Z_795_, v_inst_796_, v_g_797_, v_f_798_, v_00_u03ba_799_);
lean_dec(v_inst_796_);
lean_dec(v_inst_794_);
lean_dec(v_inst_792_);
lean_dec(v_00_u03c7_790_);
lean_dec(v_00_u03c8_789_);
lean_dec(v_00_u03c6_788_);
return v_res_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse___redArg(lean_object* v_g_801_){
_start:
{
lean_inc(v_g_801_);
return v_g_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse___redArg___boxed(lean_object* v_g_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_mathlib_MulActionHom_inverse___redArg(v_g_802_);
lean_dec(v_g_802_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse(lean_object* v_M_804_, lean_object* v_X_805_, lean_object* v_inst_806_, lean_object* v_Y_u2081_807_, lean_object* v_inst_808_, lean_object* v_f_809_, lean_object* v_g_810_, lean_object* v_h_u2081_811_, lean_object* v_h_u2082_812_){
_start:
{
lean_inc(v_g_810_);
return v_g_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse___boxed(lean_object* v_M_813_, lean_object* v_X_814_, lean_object* v_inst_815_, lean_object* v_Y_u2081_816_, lean_object* v_inst_817_, lean_object* v_f_818_, lean_object* v_g_819_, lean_object* v_h_u2081_820_, lean_object* v_h_u2082_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_MulActionHom_inverse(v_M_813_, v_X_814_, v_inst_815_, v_Y_u2081_816_, v_inst_817_, v_f_818_, v_g_819_, v_h_u2081_820_, v_h_u2082_821_);
lean_dec(v_g_819_);
lean_dec(v_f_818_);
lean_dec(v_inst_817_);
lean_dec(v_inst_815_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse___redArg(lean_object* v_g_823_){
_start:
{
lean_inc(v_g_823_);
return v_g_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse___redArg___boxed(lean_object* v_g_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_mathlib_AddActionHom_inverse___redArg(v_g_824_);
lean_dec(v_g_824_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse(lean_object* v_M_826_, lean_object* v_X_827_, lean_object* v_inst_828_, lean_object* v_Y_u2081_829_, lean_object* v_inst_830_, lean_object* v_f_831_, lean_object* v_g_832_, lean_object* v_h_u2081_833_, lean_object* v_h_u2082_834_){
_start:
{
lean_inc(v_g_832_);
return v_g_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse___boxed(lean_object* v_M_835_, lean_object* v_X_836_, lean_object* v_inst_837_, lean_object* v_Y_u2081_838_, lean_object* v_inst_839_, lean_object* v_f_840_, lean_object* v_g_841_, lean_object* v_h_u2081_842_, lean_object* v_h_u2082_843_){
_start:
{
lean_object* v_res_844_; 
v_res_844_ = lp_mathlib_AddActionHom_inverse(v_M_835_, v_X_836_, v_inst_837_, v_Y_u2081_838_, v_inst_839_, v_f_840_, v_g_841_, v_h_u2081_842_, v_h_u2082_843_);
lean_dec(v_g_841_);
lean_dec(v_f_840_);
lean_dec(v_inst_839_);
lean_dec(v_inst_837_);
return v_res_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27___redArg(lean_object* v_g_845_){
_start:
{
lean_inc(v_g_845_);
return v_g_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27___redArg___boxed(lean_object* v_g_846_){
_start:
{
lean_object* v_res_847_; 
v_res_847_ = lp_mathlib_MulActionHom_inverse_x27___redArg(v_g_846_);
lean_dec(v_g_846_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27(lean_object* v_M_848_, lean_object* v_N_849_, lean_object* v_00_u03c6_850_, lean_object* v_X_851_, lean_object* v_inst_852_, lean_object* v_Y_853_, lean_object* v_inst_854_, lean_object* v_00_u03c6_x27_855_, lean_object* v_f_856_, lean_object* v_g_857_, lean_object* v_k_858_, lean_object* v_h_u2081_859_, lean_object* v_h_u2082_860_){
_start:
{
lean_inc(v_g_857_);
return v_g_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_inverse_x27___boxed(lean_object* v_M_861_, lean_object* v_N_862_, lean_object* v_00_u03c6_863_, lean_object* v_X_864_, lean_object* v_inst_865_, lean_object* v_Y_866_, lean_object* v_inst_867_, lean_object* v_00_u03c6_x27_868_, lean_object* v_f_869_, lean_object* v_g_870_, lean_object* v_k_871_, lean_object* v_h_u2081_872_, lean_object* v_h_u2082_873_){
_start:
{
lean_object* v_res_874_; 
v_res_874_ = lp_mathlib_MulActionHom_inverse_x27(v_M_861_, v_N_862_, v_00_u03c6_863_, v_X_864_, v_inst_865_, v_Y_866_, v_inst_867_, v_00_u03c6_x27_868_, v_f_869_, v_g_870_, v_k_871_, v_h_u2081_872_, v_h_u2082_873_);
lean_dec(v_g_870_);
lean_dec(v_f_869_);
lean_dec(v_00_u03c6_x27_868_);
lean_dec(v_inst_867_);
lean_dec(v_inst_865_);
lean_dec(v_00_u03c6_863_);
return v_res_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27___redArg(lean_object* v_g_875_){
_start:
{
lean_inc(v_g_875_);
return v_g_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27___redArg___boxed(lean_object* v_g_876_){
_start:
{
lean_object* v_res_877_; 
v_res_877_ = lp_mathlib_AddActionHom_inverse_x27___redArg(v_g_876_);
lean_dec(v_g_876_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27(lean_object* v_M_878_, lean_object* v_N_879_, lean_object* v_00_u03c6_880_, lean_object* v_X_881_, lean_object* v_inst_882_, lean_object* v_Y_883_, lean_object* v_inst_884_, lean_object* v_00_u03c6_x27_885_, lean_object* v_f_886_, lean_object* v_g_887_, lean_object* v_k_888_, lean_object* v_h_u2081_889_, lean_object* v_h_u2082_890_){
_start:
{
lean_inc(v_g_887_);
return v_g_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_inverse_x27___boxed(lean_object* v_M_891_, lean_object* v_N_892_, lean_object* v_00_u03c6_893_, lean_object* v_X_894_, lean_object* v_inst_895_, lean_object* v_Y_896_, lean_object* v_inst_897_, lean_object* v_00_u03c6_x27_898_, lean_object* v_f_899_, lean_object* v_g_900_, lean_object* v_k_901_, lean_object* v_h_u2081_902_, lean_object* v_h_u2082_903_){
_start:
{
lean_object* v_res_904_; 
v_res_904_ = lp_mathlib_AddActionHom_inverse_x27(v_M_891_, v_N_892_, v_00_u03c6_893_, v_X_894_, v_inst_895_, v_Y_896_, v_inst_897_, v_00_u03c6_x27_898_, v_f_899_, v_g_900_, v_k_901_, v_h_u2081_902_, v_h_u2082_903_);
lean_dec(v_g_900_);
lean_dec(v_f_899_);
lean_dec(v_00_u03c6_x27_898_);
lean_dec(v_inst_897_);
lean_dec(v_inst_895_);
lean_dec(v_00_u03c6_893_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0(lean_object* v_inst_905_, lean_object* v_c_906_, lean_object* v_x_907_){
_start:
{
lean_object* v___x_908_; 
v___x_908_ = lean_apply_2(v_inst_905_, v_c_906_, v_x_907_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom___redArg(lean_object* v_inst_909_, lean_object* v_c_910_){
_start:
{
lean_object* v___f_911_; 
v___f_911_ = lean_alloc_closure((void*)(lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_911_, 0, v_inst_909_);
lean_closure_set(v___f_911_, 1, v_c_910_);
return v___f_911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom(lean_object* v_M_912_, lean_object* v_N_913_, lean_object* v_00_u03b1_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_c_918_){
_start:
{
lean_object* v___f_919_; 
v___f_919_ = lean_alloc_closure((void*)(lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_919_, 0, v_inst_915_);
lean_closure_set(v___f_919_, 1, v_c_918_);
return v___f_919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toMulActionHom___boxed(lean_object* v_M_920_, lean_object* v_N_921_, lean_object* v_00_u03b1_922_, lean_object* v_inst_923_, lean_object* v_inst_924_, lean_object* v_inst_925_, lean_object* v_c_926_){
_start:
{
lean_object* v_res_927_; 
v_res_927_ = lp_mathlib_SMulCommClass_toMulActionHom(v_M_920_, v_N_921_, v_00_u03b1_922_, v_inst_923_, v_inst_924_, v_inst_925_, v_c_926_);
lean_dec(v_inst_924_);
return v_res_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAddCommClass_toAddActionHom___redArg(lean_object* v_inst_928_, lean_object* v_c_929_){
_start:
{
lean_object* v___f_930_; 
v___f_930_ = lean_alloc_closure((void*)(lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_930_, 0, v_inst_928_);
lean_closure_set(v___f_930_, 1, v_c_929_);
return v___f_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAddCommClass_toAddActionHom(lean_object* v_M_931_, lean_object* v_N_932_, lean_object* v_00_u03b1_933_, lean_object* v_inst_934_, lean_object* v_inst_935_, lean_object* v_inst_936_, lean_object* v_c_937_){
_start:
{
lean_object* v___f_938_; 
v___f_938_ = lean_alloc_closure((void*)(lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_938_, 0, v_inst_934_);
lean_closure_set(v___f_938_, 1, v_c_937_);
return v___f_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAddCommClass_toAddActionHom___boxed(lean_object* v_M_939_, lean_object* v_N_940_, lean_object* v_00_u03b1_941_, lean_object* v_inst_942_, lean_object* v_inst_943_, lean_object* v_inst_944_, lean_object* v_c_945_){
_start:
{
lean_object* v_res_946_; 
v_res_946_ = lp_mathlib_VAddCommClass_toAddActionHom(v_M_939_, v_N_940_, v_00_u03b1_941_, v_inst_942_, v_inst_943_, v_inst_944_, v_c_945_);
lean_dec(v_inst_943_);
return v_res_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulActionHom___redArg(lean_object* v_i_947_){
_start:
{
lean_object* v___x_948_; 
v___x_948_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_948_, 0, lean_box(0));
lean_closure_set(v___x_948_, 1, lean_box(0));
lean_closure_set(v___x_948_, 2, v_i_947_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulActionHom(lean_object* v_00_u03b9_949_, lean_object* v_M_950_, lean_object* v_X_951_, lean_object* v_inst_952_, lean_object* v_i_953_){
_start:
{
lean_object* v___x_954_; 
v___x_954_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_954_, 0, lean_box(0));
lean_closure_set(v___x_954_, 1, lean_box(0));
lean_closure_set(v___x_954_, 2, v_i_953_);
return v___x_954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulActionHom___boxed(lean_object* v_00_u03b9_955_, lean_object* v_M_956_, lean_object* v_X_957_, lean_object* v_inst_958_, lean_object* v_i_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_mathlib_Pi_evalMulActionHom(v_00_u03b9_955_, v_M_956_, v_X_957_, v_inst_958_, v_i_959_);
lean_dec(v_inst_958_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddActionHom___redArg(lean_object* v_i_961_){
_start:
{
lean_object* v___x_962_; 
v___x_962_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_962_, 0, lean_box(0));
lean_closure_set(v___x_962_, 1, lean_box(0));
lean_closure_set(v___x_962_, 2, v_i_961_);
return v___x_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddActionHom(lean_object* v_00_u03b9_963_, lean_object* v_M_964_, lean_object* v_X_965_, lean_object* v_inst_966_, lean_object* v_i_967_){
_start:
{
lean_object* v___x_968_; 
v___x_968_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_968_, 0, lean_box(0));
lean_closure_set(v___x_968_, 1, lean_box(0));
lean_closure_set(v___x_968_, 2, v_i_967_);
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddActionHom___boxed(lean_object* v_00_u03b9_969_, lean_object* v_M_970_, lean_object* v_X_971_, lean_object* v_inst_972_, lean_object* v_i_973_){
_start:
{
lean_object* v_res_974_; 
v_res_974_ = lp_mathlib_Pi_evalAddActionHom(v_00_u03b9_969_, v_M_970_, v_X_971_, v_inst_972_, v_i_973_);
lean_dec(v_inst_972_);
return v_res_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst___lam__0(lean_object* v_self_975_){
_start:
{
lean_object* v_fst_976_; 
v_fst_976_ = lean_ctor_get(v_self_975_, 0);
lean_inc(v_fst_976_);
return v_fst_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst___lam__0___boxed(lean_object* v_self_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_MulActionHom_fst___lam__0(v_self_977_);
lean_dec_ref(v_self_977_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst(lean_object* v_M_980_, lean_object* v_00_u03b1_981_, lean_object* v_00_u03b2_982_, lean_object* v_inst_983_, lean_object* v_inst_984_){
_start:
{
lean_object* v___f_985_; 
v___f_985_ = ((lean_object*)(lp_mathlib_MulActionHom_fst___closed__0));
return v___f_985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_fst___boxed(lean_object* v_M_986_, lean_object* v_00_u03b1_987_, lean_object* v_00_u03b2_988_, lean_object* v_inst_989_, lean_object* v_inst_990_){
_start:
{
lean_object* v_res_991_; 
v_res_991_ = lp_mathlib_MulActionHom_fst(v_M_986_, v_00_u03b1_987_, v_00_u03b2_988_, v_inst_989_, v_inst_990_);
lean_dec(v_inst_990_);
lean_dec(v_inst_989_);
return v_res_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_fst(lean_object* v_M_992_, lean_object* v_00_u03b1_993_, lean_object* v_00_u03b2_994_, lean_object* v_inst_995_, lean_object* v_inst_996_){
_start:
{
lean_object* v___f_997_; 
v___f_997_ = ((lean_object*)(lp_mathlib_MulActionHom_fst___closed__0));
return v___f_997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_fst___boxed(lean_object* v_M_998_, lean_object* v_00_u03b1_999_, lean_object* v_00_u03b2_1000_, lean_object* v_inst_1001_, lean_object* v_inst_1002_){
_start:
{
lean_object* v_res_1003_; 
v_res_1003_ = lp_mathlib_AddActionHom_fst(v_M_998_, v_00_u03b1_999_, v_00_u03b2_1000_, v_inst_1001_, v_inst_1002_);
lean_dec(v_inst_1002_);
lean_dec(v_inst_1001_);
return v_res_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd___lam__0(lean_object* v_self_1004_){
_start:
{
lean_object* v_snd_1005_; 
v_snd_1005_ = lean_ctor_get(v_self_1004_, 1);
lean_inc(v_snd_1005_);
return v_snd_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd___lam__0___boxed(lean_object* v_self_1006_){
_start:
{
lean_object* v_res_1007_; 
v_res_1007_ = lp_mathlib_MulActionHom_snd___lam__0(v_self_1006_);
lean_dec_ref(v_self_1006_);
return v_res_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd(lean_object* v_M_1009_, lean_object* v_00_u03b1_1010_, lean_object* v_00_u03b2_1011_, lean_object* v_inst_1012_, lean_object* v_inst_1013_){
_start:
{
lean_object* v___f_1014_; 
v___f_1014_ = ((lean_object*)(lp_mathlib_MulActionHom_snd___closed__0));
return v___f_1014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_snd___boxed(lean_object* v_M_1015_, lean_object* v_00_u03b1_1016_, lean_object* v_00_u03b2_1017_, lean_object* v_inst_1018_, lean_object* v_inst_1019_){
_start:
{
lean_object* v_res_1020_; 
v_res_1020_ = lp_mathlib_MulActionHom_snd(v_M_1015_, v_00_u03b1_1016_, v_00_u03b2_1017_, v_inst_1018_, v_inst_1019_);
lean_dec(v_inst_1019_);
lean_dec(v_inst_1018_);
return v_res_1020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_snd(lean_object* v_M_1021_, lean_object* v_00_u03b1_1022_, lean_object* v_00_u03b2_1023_, lean_object* v_inst_1024_, lean_object* v_inst_1025_){
_start:
{
lean_object* v___f_1026_; 
v___f_1026_ = ((lean_object*)(lp_mathlib_MulActionHom_snd___closed__0));
return v___f_1026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_snd___boxed(lean_object* v_M_1027_, lean_object* v_00_u03b1_1028_, lean_object* v_00_u03b2_1029_, lean_object* v_inst_1030_, lean_object* v_inst_1031_){
_start:
{
lean_object* v_res_1032_; 
v_res_1032_ = lp_mathlib_AddActionHom_snd(v_M_1027_, v_00_u03b1_1028_, v_00_u03b2_1029_, v_inst_1030_, v_inst_1031_);
lean_dec(v_inst_1031_);
lean_dec(v_inst_1030_);
return v_res_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod___redArg___lam__0(lean_object* v_f_1033_, lean_object* v_g_1034_, lean_object* v_x_1035_){
_start:
{
lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; 
lean_inc(v_x_1035_);
v___x_1036_ = lean_apply_1(v_f_1033_, v_x_1035_);
v___x_1037_ = lean_apply_1(v_g_1034_, v_x_1035_);
v___x_1038_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1036_);
lean_ctor_set(v___x_1038_, 1, v___x_1037_);
return v___x_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod___redArg(lean_object* v_f_1039_, lean_object* v_g_1040_){
_start:
{
lean_object* v___f_1041_; 
v___f_1041_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1041_, 0, v_f_1039_);
lean_closure_set(v___f_1041_, 1, v_g_1040_);
return v___f_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod(lean_object* v_M_1042_, lean_object* v_N_1043_, lean_object* v_00_u03b1_1044_, lean_object* v_00_u03b3_1045_, lean_object* v_00_u03b4_1046_, lean_object* v_inst_1047_, lean_object* v_inst_1048_, lean_object* v_inst_1049_, lean_object* v_00_u03c3_1050_, lean_object* v_f_1051_, lean_object* v_g_1052_){
_start:
{
lean_object* v___f_1053_; 
v___f_1053_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1053_, 0, v_f_1051_);
lean_closure_set(v___f_1053_, 1, v_g_1052_);
return v___f_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prod___boxed(lean_object* v_M_1054_, lean_object* v_N_1055_, lean_object* v_00_u03b1_1056_, lean_object* v_00_u03b3_1057_, lean_object* v_00_u03b4_1058_, lean_object* v_inst_1059_, lean_object* v_inst_1060_, lean_object* v_inst_1061_, lean_object* v_00_u03c3_1062_, lean_object* v_f_1063_, lean_object* v_g_1064_){
_start:
{
lean_object* v_res_1065_; 
v_res_1065_ = lp_mathlib_MulActionHom_prod(v_M_1054_, v_N_1055_, v_00_u03b1_1056_, v_00_u03b3_1057_, v_00_u03b4_1058_, v_inst_1059_, v_inst_1060_, v_inst_1061_, v_00_u03c3_1062_, v_f_1063_, v_g_1064_);
lean_dec(v_00_u03c3_1062_);
lean_dec(v_inst_1061_);
lean_dec(v_inst_1060_);
lean_dec(v_inst_1059_);
return v_res_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prod___redArg(lean_object* v_f_1066_, lean_object* v_g_1067_){
_start:
{
lean_object* v___f_1068_; 
v___f_1068_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1068_, 0, v_f_1066_);
lean_closure_set(v___f_1068_, 1, v_g_1067_);
return v___f_1068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prod(lean_object* v_M_1069_, lean_object* v_N_1070_, lean_object* v_00_u03b1_1071_, lean_object* v_00_u03b3_1072_, lean_object* v_00_u03b4_1073_, lean_object* v_inst_1074_, lean_object* v_inst_1075_, lean_object* v_inst_1076_, lean_object* v_00_u03c3_1077_, lean_object* v_f_1078_, lean_object* v_g_1079_){
_start:
{
lean_object* v___f_1080_; 
v___f_1080_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1080_, 0, v_f_1078_);
lean_closure_set(v___f_1080_, 1, v_g_1079_);
return v___f_1080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prod___boxed(lean_object* v_M_1081_, lean_object* v_N_1082_, lean_object* v_00_u03b1_1083_, lean_object* v_00_u03b3_1084_, lean_object* v_00_u03b4_1085_, lean_object* v_inst_1086_, lean_object* v_inst_1087_, lean_object* v_inst_1088_, lean_object* v_00_u03c3_1089_, lean_object* v_f_1090_, lean_object* v_g_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib_AddActionHom_prod(v_M_1081_, v_N_1082_, v_00_u03b1_1083_, v_00_u03b3_1084_, v_00_u03b4_1085_, v_inst_1086_, v_inst_1087_, v_inst_1088_, v_00_u03c3_1089_, v_f_1090_, v_g_1091_);
lean_dec(v_00_u03c3_1089_);
lean_dec(v_inst_1088_);
lean_dec(v_inst_1087_);
lean_dec(v_inst_1086_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___redArg___lam__0(lean_object* v_f_1093_, lean_object* v___y_1094_){
_start:
{
lean_object* v___x_1095_; 
v___x_1095_ = lean_apply_1(v_f_1093_, v___y_1094_);
return v___x_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___redArg___lam__1(lean_object* v_g_1096_, lean_object* v___y_1097_){
_start:
{
lean_object* v___x_1098_; 
v___x_1098_ = lean_apply_1(v_g_1096_, v___y_1097_);
return v___x_1098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___redArg(lean_object* v_f_1099_, lean_object* v_g_1100_){
_start:
{
lean_object* v___f_1101_; lean_object* v___f_1102_; lean_object* v___x_1103_; 
v___f_1101_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1101_, 0, v_f_1099_);
v___f_1102_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1102_, 0, v_g_1100_);
v___x_1103_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_1103_, 0, lean_box(0));
lean_closure_set(v___x_1103_, 1, lean_box(0));
lean_closure_set(v___x_1103_, 2, lean_box(0));
lean_closure_set(v___x_1103_, 3, lean_box(0));
lean_closure_set(v___x_1103_, 4, v___f_1101_);
lean_closure_set(v___x_1103_, 5, v___f_1102_);
return v___x_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap(lean_object* v_M_1104_, lean_object* v_N_1105_, lean_object* v_00_u03b1_1106_, lean_object* v_00_u03b2_1107_, lean_object* v_00_u03b3_1108_, lean_object* v_00_u03b4_1109_, lean_object* v_inst_1110_, lean_object* v_inst_1111_, lean_object* v_inst_1112_, lean_object* v_inst_1113_, lean_object* v_00_u03c3_1114_, lean_object* v_f_1115_, lean_object* v_g_1116_){
_start:
{
lean_object* v___x_1117_; 
v___x_1117_ = lp_mathlib_MulActionHom_prodMap___redArg(v_f_1115_, v_g_1116_);
return v___x_1117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_prodMap___boxed(lean_object* v_M_1118_, lean_object* v_N_1119_, lean_object* v_00_u03b1_1120_, lean_object* v_00_u03b2_1121_, lean_object* v_00_u03b3_1122_, lean_object* v_00_u03b4_1123_, lean_object* v_inst_1124_, lean_object* v_inst_1125_, lean_object* v_inst_1126_, lean_object* v_inst_1127_, lean_object* v_00_u03c3_1128_, lean_object* v_f_1129_, lean_object* v_g_1130_){
_start:
{
lean_object* v_res_1131_; 
v_res_1131_ = lp_mathlib_MulActionHom_prodMap(v_M_1118_, v_N_1119_, v_00_u03b1_1120_, v_00_u03b2_1121_, v_00_u03b3_1122_, v_00_u03b4_1123_, v_inst_1124_, v_inst_1125_, v_inst_1126_, v_inst_1127_, v_00_u03c3_1128_, v_f_1129_, v_g_1130_);
lean_dec(v_00_u03c3_1128_);
lean_dec(v_inst_1127_);
lean_dec(v_inst_1126_);
lean_dec(v_inst_1125_);
lean_dec(v_inst_1124_);
return v_res_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prodMap___redArg(lean_object* v_f_1132_, lean_object* v_g_1133_){
_start:
{
lean_object* v___f_1134_; lean_object* v___f_1135_; lean_object* v___x_1136_; 
v___f_1134_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1134_, 0, v_f_1132_);
v___f_1135_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1135_, 0, v_g_1133_);
v___x_1136_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_1136_, 0, lean_box(0));
lean_closure_set(v___x_1136_, 1, lean_box(0));
lean_closure_set(v___x_1136_, 2, lean_box(0));
lean_closure_set(v___x_1136_, 3, lean_box(0));
lean_closure_set(v___x_1136_, 4, v___f_1134_);
lean_closure_set(v___x_1136_, 5, v___f_1135_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prodMap(lean_object* v_M_1137_, lean_object* v_N_1138_, lean_object* v_00_u03b1_1139_, lean_object* v_00_u03b2_1140_, lean_object* v_00_u03b3_1141_, lean_object* v_00_u03b4_1142_, lean_object* v_inst_1143_, lean_object* v_inst_1144_, lean_object* v_inst_1145_, lean_object* v_inst_1146_, lean_object* v_00_u03c3_1147_, lean_object* v_f_1148_, lean_object* v_g_1149_){
_start:
{
lean_object* v___x_1150_; 
v___x_1150_ = lp_mathlib_AddActionHom_prodMap___redArg(v_f_1148_, v_g_1149_);
return v___x_1150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_prodMap___boxed(lean_object* v_M_1151_, lean_object* v_N_1152_, lean_object* v_00_u03b1_1153_, lean_object* v_00_u03b2_1154_, lean_object* v_00_u03b3_1155_, lean_object* v_00_u03b4_1156_, lean_object* v_inst_1157_, lean_object* v_inst_1158_, lean_object* v_inst_1159_, lean_object* v_inst_1160_, lean_object* v_00_u03c3_1161_, lean_object* v_f_1162_, lean_object* v_g_1163_){
_start:
{
lean_object* v_res_1164_; 
v_res_1164_ = lp_mathlib_AddActionHom_prodMap(v_M_1151_, v_N_1152_, v_00_u03b1_1153_, v_00_u03b2_1154_, v_00_u03b3_1155_, v_00_u03b4_1156_, v_inst_1157_, v_inst_1158_, v_inst_1159_, v_inst_1160_, v_00_u03c3_1161_, v_f_1162_, v_g_1163_);
lean_dec(v_00_u03c3_1161_);
lean_dec(v_inst_1160_);
lean_dec(v_inst_1159_);
lean_dec(v_inst_1158_);
lean_dec(v_inst_1157_);
return v_res_1164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0(lean_object* v_inst_1165_, lean_object* v_h_1166_, lean_object* v_f_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v___x_1169_; lean_object* v___x_1170_; 
v___x_1169_ = lean_apply_1(v_f_1167_, v___y_1168_);
v___x_1170_ = lean_apply_2(v_inst_1165_, v_h_1166_, v___x_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg(lean_object* v_inst_1171_){
_start:
{
lean_object* v___f_1172_; 
v___f_1172_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1172_, 0, v_inst_1171_);
return v___f_1172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass(lean_object* v_R_1173_, lean_object* v_M_1174_, lean_object* v_N_1175_, lean_object* v_X_1176_, lean_object* v_Y_1177_, lean_object* v_00_u03c3_1178_, lean_object* v_inst_1179_, lean_object* v_inst_1180_, lean_object* v_inst_1181_, lean_object* v_inst_1182_){
_start:
{
lean_object* v___f_1183_; 
v___f_1183_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1183_, 0, v_inst_1181_);
return v___f_1183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSMulOfSMulCommClass___boxed(lean_object* v_R_1184_, lean_object* v_M_1185_, lean_object* v_N_1186_, lean_object* v_X_1187_, lean_object* v_Y_1188_, lean_object* v_00_u03c3_1189_, lean_object* v_inst_1190_, lean_object* v_inst_1191_, lean_object* v_inst_1192_, lean_object* v_inst_1193_){
_start:
{
lean_object* v_res_1194_; 
v_res_1194_ = lp_mathlib_MulActionHom_instSMulOfSMulCommClass(v_R_1184_, v_M_1185_, v_N_1186_, v_X_1187_, v_Y_1188_, v_00_u03c3_1189_, v_inst_1190_, v_inst_1191_, v_inst_1192_, v_inst_1193_);
lean_dec(v_inst_1191_);
lean_dec(v_inst_1190_);
lean_dec(v_00_u03c3_1189_);
return v_res_1194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instVAddOfVAddCommClass___redArg(lean_object* v_inst_1195_){
_start:
{
lean_object* v___f_1196_; 
v___f_1196_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1196_, 0, v_inst_1195_);
return v___f_1196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instVAddOfVAddCommClass(lean_object* v_R_1197_, lean_object* v_M_1198_, lean_object* v_N_1199_, lean_object* v_X_1200_, lean_object* v_Y_1201_, lean_object* v_00_u03c3_1202_, lean_object* v_inst_1203_, lean_object* v_inst_1204_, lean_object* v_inst_1205_, lean_object* v_inst_1206_){
_start:
{
lean_object* v___f_1207_; 
v___f_1207_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1207_, 0, v_inst_1205_);
return v___f_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instVAddOfVAddCommClass___boxed(lean_object* v_R_1208_, lean_object* v_M_1209_, lean_object* v_N_1210_, lean_object* v_X_1211_, lean_object* v_Y_1212_, lean_object* v_00_u03c3_1213_, lean_object* v_inst_1214_, lean_object* v_inst_1215_, lean_object* v_inst_1216_, lean_object* v_inst_1217_){
_start:
{
lean_object* v_res_1218_; 
v_res_1218_ = lp_mathlib_AddActionHom_instVAddOfVAddCommClass(v_R_1208_, v_M_1209_, v_N_1210_, v_X_1211_, v_Y_1212_, v_00_u03c3_1213_, v_inst_1214_, v_inst_1215_, v_inst_1216_, v_inst_1217_);
lean_dec(v_inst_1215_);
lean_dec(v_inst_1214_);
lean_dec(v_00_u03c3_1213_);
return v_res_1218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___redArg___lam__0(lean_object* v_inst_1219_, lean_object* v_x_1220_){
_start:
{
lean_inc(v_inst_1219_);
return v_inst_1219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___redArg___lam__0___boxed(lean_object* v_inst_1221_, lean_object* v_x_1222_){
_start:
{
lean_object* v_res_1223_; 
v_res_1223_ = lp_mathlib_MulActionHom_instZero___redArg___lam__0(v_inst_1221_, v_x_1222_);
lean_dec(v_x_1222_);
lean_dec(v_inst_1221_);
return v_res_1223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___redArg(lean_object* v_inst_1224_){
_start:
{
lean_object* v___f_1225_; 
v___f_1225_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1225_, 0, v_inst_1224_);
return v___f_1225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero(lean_object* v_M_1226_, lean_object* v_N_1227_, lean_object* v_X_1228_, lean_object* v_Y_1229_, lean_object* v_00_u03c3_1230_, lean_object* v_inst_1231_, lean_object* v_inst_1232_, lean_object* v_inst_1233_){
_start:
{
lean_object* v___f_1234_; 
v___f_1234_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1234_, 0, v_inst_1232_);
return v___f_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instZero___boxed(lean_object* v_M_1235_, lean_object* v_N_1236_, lean_object* v_X_1237_, lean_object* v_Y_1238_, lean_object* v_00_u03c3_1239_, lean_object* v_inst_1240_, lean_object* v_inst_1241_, lean_object* v_inst_1242_){
_start:
{
lean_object* v_res_1243_; 
v_res_1243_ = lp_mathlib_MulActionHom_instZero(v_M_1235_, v_N_1236_, v_X_1237_, v_Y_1238_, v_00_u03c3_1239_, v_inst_1240_, v_inst_1241_, v_inst_1242_);
lean_dec(v_inst_1242_);
lean_dec(v_inst_1240_);
lean_dec(v_00_u03c3_1239_);
return v_res_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass___redArg___lam__0(lean_object* v_toAdd_1244_, lean_object* v_f_1245_, lean_object* v_g_1246_, lean_object* v___y_1247_){
_start:
{
lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; 
lean_inc(v___y_1247_);
v___x_1248_ = lean_apply_1(v_f_1245_, v___y_1247_);
v___x_1249_ = lean_apply_1(v_g_1246_, v___y_1247_);
v___x_1250_ = lean_apply_2(v_toAdd_1244_, v___x_1248_, v___x_1249_);
return v___x_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass___redArg(lean_object* v_inst_1251_){
_start:
{
lean_object* v___x_1252_; lean_object* v_toZero_1253_; lean_object* v_toAdd_1254_; lean_object* v___x_1256_; uint8_t v_isShared_1257_; uint8_t v_isSharedCheck_1263_; 
v___x_1252_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_1251_);
v_toZero_1253_ = lean_ctor_get(v___x_1252_, 0);
v_toAdd_1254_ = lean_ctor_get(v___x_1252_, 1);
v_isSharedCheck_1263_ = !lean_is_exclusive(v___x_1252_);
if (v_isSharedCheck_1263_ == 0)
{
v___x_1256_ = v___x_1252_;
v_isShared_1257_ = v_isSharedCheck_1263_;
goto v_resetjp_1255_;
}
else
{
lean_inc(v_toAdd_1254_);
lean_inc(v_toZero_1253_);
lean_dec(v___x_1252_);
v___x_1256_ = lean_box(0);
v_isShared_1257_ = v_isSharedCheck_1263_;
goto v_resetjp_1255_;
}
v_resetjp_1255_:
{
lean_object* v___f_1258_; lean_object* v___f_1259_; lean_object* v___x_1261_; 
v___f_1258_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instAddZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1258_, 0, v_toAdd_1254_);
v___f_1259_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1259_, 0, v_toZero_1253_);
if (v_isShared_1257_ == 0)
{
lean_ctor_set(v___x_1256_, 1, v___f_1258_);
lean_ctor_set(v___x_1256_, 0, v___f_1259_);
v___x_1261_ = v___x_1256_;
goto v_reusejp_1260_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v___f_1259_);
lean_ctor_set(v_reuseFailAlloc_1262_, 1, v___f_1258_);
v___x_1261_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1260_;
}
v_reusejp_1260_:
{
return v___x_1261_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass(lean_object* v_M_1264_, lean_object* v_N_1265_, lean_object* v_X_1266_, lean_object* v_Y_1267_, lean_object* v_00_u03c3_1268_, lean_object* v_inst_1269_, lean_object* v_inst_1270_, lean_object* v_inst_1271_){
_start:
{
lean_object* v___x_1272_; 
v___x_1272_ = lp_mathlib_MulActionHom_instAddZeroClass___redArg(v_inst_1270_);
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddZeroClass___boxed(lean_object* v_M_1273_, lean_object* v_N_1274_, lean_object* v_X_1275_, lean_object* v_Y_1276_, lean_object* v_00_u03c3_1277_, lean_object* v_inst_1278_, lean_object* v_inst_1279_, lean_object* v_inst_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_mathlib_MulActionHom_instAddZeroClass(v_M_1273_, v_N_1274_, v_X_1275_, v_Y_1276_, v_00_u03c3_1277_, v_inst_1278_, v_inst_1279_, v_inst_1280_);
lean_dec(v_inst_1280_);
lean_dec(v_inst_1278_);
lean_dec(v_00_u03c3_1277_);
return v_res_1281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddMonoid___redArg(lean_object* v_inst_1282_){
_start:
{
lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v_toZero_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v_toAdd_1288_; lean_object* v_toNSMul_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1300_; 
v___x_1283_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1282_);
lean_inc_ref(v___x_1283_);
v___x_1284_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1283_);
v_toZero_1285_ = lean_ctor_get(v___x_1284_, 0);
lean_inc(v_toZero_1285_);
lean_dec_ref(v___x_1284_);
v___x_1286_ = lp_mathlib_MulActionHom_instAddZeroClass___redArg(v___x_1283_);
v___x_1287_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1286_);
v_toAdd_1288_ = lean_ctor_get(v___x_1287_, 1);
lean_inc(v_toAdd_1288_);
lean_dec_ref(v___x_1287_);
v_toNSMul_1289_ = lean_ctor_get(v_inst_1282_, 2);
v_isSharedCheck_1300_ = !lean_is_exclusive(v_inst_1282_);
if (v_isSharedCheck_1300_ == 0)
{
lean_object* v_unused_1301_; lean_object* v_unused_1302_; 
v_unused_1301_ = lean_ctor_get(v_inst_1282_, 1);
lean_dec(v_unused_1301_);
v_unused_1302_ = lean_ctor_get(v_inst_1282_, 0);
lean_dec(v_unused_1302_);
v___x_1291_ = v_inst_1282_;
v_isShared_1292_ = v_isSharedCheck_1300_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_toNSMul_1289_);
lean_dec(v_inst_1282_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1300_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
lean_object* v___f_1293_; lean_object* v___f_1294_; lean_object* v___f_1295_; lean_object* v___f_1296_; lean_object* v___x_1298_; 
v___f_1293_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1293_, 0, v_toZero_1285_);
v___f_1294_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1294_, 0, v_toNSMul_1289_);
v___f_1295_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1295_, 0, v___f_1294_);
v___f_1296_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1296_, 0, v___f_1295_);
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 2, v___f_1296_);
lean_ctor_set(v___x_1291_, 1, v_toAdd_1288_);
lean_ctor_set(v___x_1291_, 0, v___f_1293_);
v___x_1298_ = v___x_1291_;
goto v_reusejp_1297_;
}
else
{
lean_object* v_reuseFailAlloc_1299_; 
v_reuseFailAlloc_1299_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1299_, 0, v___f_1293_);
lean_ctor_set(v_reuseFailAlloc_1299_, 1, v_toAdd_1288_);
lean_ctor_set(v_reuseFailAlloc_1299_, 2, v___f_1296_);
v___x_1298_ = v_reuseFailAlloc_1299_;
goto v_reusejp_1297_;
}
v_reusejp_1297_:
{
return v___x_1298_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddMonoid(lean_object* v_M_1303_, lean_object* v_N_1304_, lean_object* v_X_1305_, lean_object* v_Y_1306_, lean_object* v_00_u03c3_1307_, lean_object* v_inst_1308_, lean_object* v_inst_1309_, lean_object* v_inst_1310_){
_start:
{
lean_object* v___x_1311_; 
v___x_1311_ = lp_mathlib_MulActionHom_instAddMonoid___redArg(v_inst_1309_);
return v___x_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddMonoid___boxed(lean_object* v_M_1312_, lean_object* v_N_1313_, lean_object* v_X_1314_, lean_object* v_Y_1315_, lean_object* v_00_u03c3_1316_, lean_object* v_inst_1317_, lean_object* v_inst_1318_, lean_object* v_inst_1319_){
_start:
{
lean_object* v_res_1320_; 
v_res_1320_ = lp_mathlib_MulActionHom_instAddMonoid(v_M_1312_, v_N_1313_, v_X_1314_, v_Y_1315_, v_00_u03c3_1316_, v_inst_1317_, v_inst_1318_, v_inst_1319_);
lean_dec(v_inst_1319_);
lean_dec(v_inst_1317_);
lean_dec(v_00_u03c3_1316_);
return v_res_1320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommMonoid___redArg(lean_object* v_inst_1321_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = lp_mathlib_MulActionHom_instAddMonoid___redArg(v_inst_1321_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommMonoid(lean_object* v_M_1323_, lean_object* v_N_1324_, lean_object* v_X_1325_, lean_object* v_Y_1326_, lean_object* v_00_u03c3_1327_, lean_object* v_inst_1328_, lean_object* v_inst_1329_, lean_object* v_inst_1330_){
_start:
{
lean_object* v___x_1331_; 
v___x_1331_ = lp_mathlib_MulActionHom_instAddMonoid___redArg(v_inst_1329_);
return v___x_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommMonoid___boxed(lean_object* v_M_1332_, lean_object* v_N_1333_, lean_object* v_X_1334_, lean_object* v_Y_1335_, lean_object* v_00_u03c3_1336_, lean_object* v_inst_1337_, lean_object* v_inst_1338_, lean_object* v_inst_1339_){
_start:
{
lean_object* v_res_1340_; 
v_res_1340_ = lp_mathlib_MulActionHom_instAddCommMonoid(v_M_1332_, v_N_1333_, v_X_1334_, v_Y_1335_, v_00_u03c3_1336_, v_inst_1337_, v_inst_1338_, v_inst_1339_);
lean_dec(v_inst_1339_);
lean_dec(v_inst_1337_);
lean_dec(v_00_u03c3_1336_);
return v_res_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMulActionOfSMulCommClass___redArg(lean_object* v_inst_1341_){
_start:
{
lean_object* v___f_1342_; 
v___f_1342_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1342_, 0, v_inst_1341_);
return v___f_1342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMulActionOfSMulCommClass(lean_object* v_R_1343_, lean_object* v_M_1344_, lean_object* v_N_1345_, lean_object* v_X_1346_, lean_object* v_Y_1347_, lean_object* v_00_u03c3_1348_, lean_object* v_inst_1349_, lean_object* v_inst_1350_, lean_object* v_inst_1351_, lean_object* v_inst_1352_, lean_object* v_inst_1353_){
_start:
{
lean_object* v___f_1354_; 
v___f_1354_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1354_, 0, v_inst_1352_);
return v___f_1354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMulActionOfSMulCommClass___boxed(lean_object* v_R_1355_, lean_object* v_M_1356_, lean_object* v_N_1357_, lean_object* v_X_1358_, lean_object* v_Y_1359_, lean_object* v_00_u03c3_1360_, lean_object* v_inst_1361_, lean_object* v_inst_1362_, lean_object* v_inst_1363_, lean_object* v_inst_1364_, lean_object* v_inst_1365_){
_start:
{
lean_object* v_res_1366_; 
v_res_1366_ = lp_mathlib_MulActionHom_instMulActionOfSMulCommClass(v_R_1355_, v_M_1356_, v_N_1357_, v_X_1358_, v_Y_1359_, v_00_u03c3_1360_, v_inst_1361_, v_inst_1362_, v_inst_1363_, v_inst_1364_, v_inst_1365_);
lean_dec_ref(v_inst_1363_);
lean_dec(v_inst_1362_);
lean_dec(v_inst_1361_);
lean_dec(v_00_u03c3_1360_);
return v_res_1366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instAddActionOfVAddCommClass___redArg(lean_object* v_inst_1367_){
_start:
{
lean_object* v___f_1368_; 
v___f_1368_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1368_, 0, v_inst_1367_);
return v___f_1368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instAddActionOfVAddCommClass(lean_object* v_R_1369_, lean_object* v_M_1370_, lean_object* v_N_1371_, lean_object* v_X_1372_, lean_object* v_Y_1373_, lean_object* v_00_u03c3_1374_, lean_object* v_inst_1375_, lean_object* v_inst_1376_, lean_object* v_inst_1377_, lean_object* v_inst_1378_, lean_object* v_inst_1379_){
_start:
{
lean_object* v___f_1380_; 
v___f_1380_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1380_, 0, v_inst_1378_);
return v___f_1380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_instAddActionOfVAddCommClass___boxed(lean_object* v_R_1381_, lean_object* v_M_1382_, lean_object* v_N_1383_, lean_object* v_X_1384_, lean_object* v_Y_1385_, lean_object* v_00_u03c3_1386_, lean_object* v_inst_1387_, lean_object* v_inst_1388_, lean_object* v_inst_1389_, lean_object* v_inst_1390_, lean_object* v_inst_1391_){
_start:
{
lean_object* v_res_1392_; 
v_res_1392_ = lp_mathlib_AddActionHom_instAddActionOfVAddCommClass(v_R_1381_, v_M_1382_, v_N_1383_, v_X_1384_, v_Y_1385_, v_00_u03c3_1386_, v_inst_1387_, v_inst_1388_, v_inst_1389_, v_inst_1390_, v_inst_1391_);
lean_dec_ref(v_inst_1389_);
lean_dec(v_inst_1388_);
lean_dec(v_inst_1387_);
lean_dec(v_00_u03c3_1386_);
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass___redArg(lean_object* v_inst_1393_){
_start:
{
lean_object* v___f_1394_; 
v___f_1394_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1394_, 0, v_inst_1393_);
return v___f_1394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass(lean_object* v_R_1395_, lean_object* v_M_1396_, lean_object* v_N_1397_, lean_object* v_X_1398_, lean_object* v_Y_1399_, lean_object* v_00_u03c3_1400_, lean_object* v_inst_1401_, lean_object* v_inst_1402_, lean_object* v_inst_1403_, lean_object* v_inst_1404_, lean_object* v_inst_1405_){
_start:
{
lean_object* v___f_1406_; 
v___f_1406_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1406_, 0, v_inst_1404_);
return v___f_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass___boxed(lean_object* v_R_1407_, lean_object* v_M_1408_, lean_object* v_N_1409_, lean_object* v_X_1410_, lean_object* v_Y_1411_, lean_object* v_00_u03c3_1412_, lean_object* v_inst_1413_, lean_object* v_inst_1414_, lean_object* v_inst_1415_, lean_object* v_inst_1416_, lean_object* v_inst_1417_){
_start:
{
lean_object* v_res_1418_; 
v_res_1418_ = lp_mathlib_MulActionHom_instDistribSMulOfSMulCommClass(v_R_1407_, v_M_1408_, v_N_1409_, v_X_1410_, v_Y_1411_, v_00_u03c3_1412_, v_inst_1413_, v_inst_1414_, v_inst_1415_, v_inst_1416_, v_inst_1417_);
lean_dec(v_inst_1415_);
lean_dec(v_inst_1414_);
lean_dec_ref(v_inst_1413_);
lean_dec(v_00_u03c3_1412_);
return v_res_1418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass___redArg(lean_object* v_inst_1419_){
_start:
{
lean_object* v___f_1420_; 
v___f_1420_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1420_, 0, v_inst_1419_);
return v___f_1420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass(lean_object* v_R_1421_, lean_object* v_M_1422_, lean_object* v_N_1423_, lean_object* v_X_1424_, lean_object* v_Y_1425_, lean_object* v_00_u03c3_1426_, lean_object* v_inst_1427_, lean_object* v_inst_1428_, lean_object* v_inst_1429_, lean_object* v_inst_1430_, lean_object* v_inst_1431_, lean_object* v_inst_1432_){
_start:
{
lean_object* v___f_1433_; 
v___f_1433_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1433_, 0, v_inst_1431_);
return v___f_1433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass___boxed(lean_object* v_R_1434_, lean_object* v_M_1435_, lean_object* v_N_1436_, lean_object* v_X_1437_, lean_object* v_Y_1438_, lean_object* v_00_u03c3_1439_, lean_object* v_inst_1440_, lean_object* v_inst_1441_, lean_object* v_inst_1442_, lean_object* v_inst_1443_, lean_object* v_inst_1444_, lean_object* v_inst_1445_){
_start:
{
lean_object* v_res_1446_; 
v_res_1446_ = lp_mathlib_MulActionHom_instDistribMulActionOfSMulCommClass(v_R_1434_, v_M_1435_, v_N_1436_, v_X_1437_, v_Y_1438_, v_00_u03c3_1439_, v_inst_1440_, v_inst_1441_, v_inst_1442_, v_inst_1443_, v_inst_1444_, v_inst_1445_);
lean_dec(v_inst_1443_);
lean_dec(v_inst_1442_);
lean_dec_ref(v_inst_1441_);
lean_dec_ref(v_inst_1440_);
lean_dec(v_00_u03c3_1439_);
return v_res_1446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instModuleOfSMulCommClass___redArg(lean_object* v_inst_1447_){
_start:
{
lean_object* v___f_1448_; 
v___f_1448_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1448_, 0, v_inst_1447_);
return v___f_1448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instModuleOfSMulCommClass(lean_object* v_R_1449_, lean_object* v_M_1450_, lean_object* v_N_1451_, lean_object* v_X_1452_, lean_object* v_Y_1453_, lean_object* v_00_u03c3_1454_, lean_object* v_inst_1455_, lean_object* v_inst_1456_, lean_object* v_inst_1457_, lean_object* v_inst_1458_, lean_object* v_inst_1459_, lean_object* v_inst_1460_){
_start:
{
lean_object* v___f_1461_; 
v___f_1461_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1461_, 0, v_inst_1459_);
return v___f_1461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instModuleOfSMulCommClass___boxed(lean_object* v_R_1462_, lean_object* v_M_1463_, lean_object* v_N_1464_, lean_object* v_X_1465_, lean_object* v_Y_1466_, lean_object* v_00_u03c3_1467_, lean_object* v_inst_1468_, lean_object* v_inst_1469_, lean_object* v_inst_1470_, lean_object* v_inst_1471_, lean_object* v_inst_1472_, lean_object* v_inst_1473_){
_start:
{
lean_object* v_res_1474_; 
v_res_1474_ = lp_mathlib_MulActionHom_instModuleOfSMulCommClass(v_R_1462_, v_M_1463_, v_N_1464_, v_X_1465_, v_Y_1466_, v_00_u03c3_1467_, v_inst_1468_, v_inst_1469_, v_inst_1470_, v_inst_1471_, v_inst_1472_, v_inst_1473_);
lean_dec(v_inst_1471_);
lean_dec(v_inst_1470_);
lean_dec_ref(v_inst_1469_);
lean_dec_ref(v_inst_1468_);
lean_dec(v_00_u03c3_1467_);
return v_res_1474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___redArg___lam__0(lean_object* v_toSub_1475_, lean_object* v_f_1476_, lean_object* v_g_1477_, lean_object* v___y_1478_){
_start:
{
lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; 
lean_inc(v___y_1478_);
v___x_1479_ = lean_apply_1(v_f_1476_, v___y_1478_);
v___x_1480_ = lean_apply_1(v_g_1477_, v___y_1478_);
v___x_1481_ = lean_apply_2(v_toSub_1475_, v___x_1479_, v___x_1480_);
return v___x_1481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___redArg___lam__1(lean_object* v_toNeg_1482_, lean_object* v_f_1483_, lean_object* v___y_1484_){
_start:
{
lean_object* v___x_1485_; lean_object* v___x_1486_; 
v___x_1485_ = lean_apply_1(v_f_1483_, v___y_1484_);
v___x_1486_ = lean_apply_1(v_toNeg_1482_, v___x_1485_);
return v___x_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___redArg(lean_object* v_inst_1487_){
_start:
{
lean_object* v_toAddMonoid_1488_; lean_object* v_toSub_1489_; lean_object* v_toZSMul_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1494_; uint8_t v_isShared_1495_; uint8_t v_isSharedCheck_1505_; 
v_toAddMonoid_1488_ = lean_ctor_get(v_inst_1487_, 0);
v_toSub_1489_ = lean_ctor_get(v_inst_1487_, 2);
lean_inc(v_toSub_1489_);
v_toZSMul_1490_ = lean_ctor_get(v_inst_1487_, 3);
lean_inc(v_toZSMul_1490_);
lean_inc_ref(v_toAddMonoid_1488_);
v___x_1491_ = lp_mathlib_MulActionHom_instAddMonoid___redArg(v_toAddMonoid_1488_);
v___x_1492_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_1487_);
v_isSharedCheck_1505_ = !lean_is_exclusive(v_inst_1487_);
if (v_isSharedCheck_1505_ == 0)
{
lean_object* v_unused_1506_; lean_object* v_unused_1507_; lean_object* v_unused_1508_; lean_object* v_unused_1509_; 
v_unused_1506_ = lean_ctor_get(v_inst_1487_, 3);
lean_dec(v_unused_1506_);
v_unused_1507_ = lean_ctor_get(v_inst_1487_, 2);
lean_dec(v_unused_1507_);
v_unused_1508_ = lean_ctor_get(v_inst_1487_, 1);
lean_dec(v_unused_1508_);
v_unused_1509_ = lean_ctor_get(v_inst_1487_, 0);
lean_dec(v_unused_1509_);
v___x_1494_ = v_inst_1487_;
v_isShared_1495_ = v_isSharedCheck_1505_;
goto v_resetjp_1493_;
}
else
{
lean_dec(v_inst_1487_);
v___x_1494_ = lean_box(0);
v_isShared_1495_ = v_isSharedCheck_1505_;
goto v_resetjp_1493_;
}
v_resetjp_1493_:
{
lean_object* v_toNeg_1496_; lean_object* v___f_1497_; lean_object* v___f_1498_; lean_object* v___f_1499_; lean_object* v___f_1500_; lean_object* v___f_1501_; lean_object* v___x_1503_; 
v_toNeg_1496_ = lean_ctor_get(v___x_1492_, 1);
lean_inc(v_toNeg_1496_);
lean_dec_ref(v___x_1492_);
v___f_1497_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instAddGroup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1497_, 0, v_toSub_1489_);
v___f_1498_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instAddGroup___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1498_, 0, v_toNeg_1496_);
v___f_1499_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1499_, 0, v_toZSMul_1490_);
v___f_1500_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instSMulOfSMulCommClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1500_, 0, v___f_1499_);
v___f_1501_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1501_, 0, v___f_1500_);
if (v_isShared_1495_ == 0)
{
lean_ctor_set(v___x_1494_, 3, v___f_1501_);
lean_ctor_set(v___x_1494_, 2, v___f_1497_);
lean_ctor_set(v___x_1494_, 1, v___f_1498_);
lean_ctor_set(v___x_1494_, 0, v___x_1491_);
v___x_1503_ = v___x_1494_;
goto v_reusejp_1502_;
}
else
{
lean_object* v_reuseFailAlloc_1504_; 
v_reuseFailAlloc_1504_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1504_, 0, v___x_1491_);
lean_ctor_set(v_reuseFailAlloc_1504_, 1, v___f_1498_);
lean_ctor_set(v_reuseFailAlloc_1504_, 2, v___f_1497_);
lean_ctor_set(v_reuseFailAlloc_1504_, 3, v___f_1501_);
v___x_1503_ = v_reuseFailAlloc_1504_;
goto v_reusejp_1502_;
}
v_reusejp_1502_:
{
return v___x_1503_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup(lean_object* v_M_1510_, lean_object* v_N_1511_, lean_object* v_X_1512_, lean_object* v_Y_1513_, lean_object* v_00_u03c3_1514_, lean_object* v_inst_1515_, lean_object* v_inst_1516_, lean_object* v_inst_1517_){
_start:
{
lean_object* v___x_1518_; 
v___x_1518_ = lp_mathlib_MulActionHom_instAddGroup___redArg(v_inst_1516_);
return v___x_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddGroup___boxed(lean_object* v_M_1519_, lean_object* v_N_1520_, lean_object* v_X_1521_, lean_object* v_Y_1522_, lean_object* v_00_u03c3_1523_, lean_object* v_inst_1524_, lean_object* v_inst_1525_, lean_object* v_inst_1526_){
_start:
{
lean_object* v_res_1527_; 
v_res_1527_ = lp_mathlib_MulActionHom_instAddGroup(v_M_1519_, v_N_1520_, v_X_1521_, v_Y_1522_, v_00_u03c3_1523_, v_inst_1524_, v_inst_1525_, v_inst_1526_);
lean_dec(v_inst_1526_);
lean_dec(v_inst_1524_);
lean_dec(v_00_u03c3_1523_);
return v_res_1527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommGroup___redArg(lean_object* v_inst_1528_){
_start:
{
lean_object* v___x_1529_; 
v___x_1529_ = lp_mathlib_MulActionHom_instAddGroup___redArg(v_inst_1528_);
return v___x_1529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommGroup(lean_object* v_M_1530_, lean_object* v_N_1531_, lean_object* v_X_1532_, lean_object* v_Y_1533_, lean_object* v_00_u03c3_1534_, lean_object* v_inst_1535_, lean_object* v_inst_1536_, lean_object* v_inst_1537_){
_start:
{
lean_object* v___x_1538_; 
v___x_1538_ = lp_mathlib_MulActionHom_instAddGroup___redArg(v_inst_1536_);
return v___x_1538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instAddCommGroup___boxed(lean_object* v_M_1539_, lean_object* v_N_1540_, lean_object* v_X_1541_, lean_object* v_Y_1542_, lean_object* v_00_u03c3_1543_, lean_object* v_inst_1544_, lean_object* v_inst_1545_, lean_object* v_inst_1546_){
_start:
{
lean_object* v_res_1547_; 
v_res_1547_ = lp_mathlib_MulActionHom_instAddCommGroup(v_M_1539_, v_N_1540_, v_X_1541_, v_Y_1542_, v_00_u03c3_1543_, v_inst_1544_, v_inst_1545_, v_inst_1546_);
lean_dec(v_inst_1546_);
lean_dec(v_inst_1544_);
lean_dec(v_00_u03c3_1543_);
return v_res_1547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___lam__0(lean_object* v_toMul_1548_, lean_object* v_f_1549_, lean_object* v_g_1550_, lean_object* v___y_1551_){
_start:
{
lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; 
lean_inc(v___y_1551_);
v___x_1552_ = lean_apply_1(v_f_1549_, v___y_1551_);
v___x_1553_ = lean_apply_1(v_g_1550_, v___y_1551_);
v___x_1554_ = lean_apply_2(v_toMul_1548_, v___x_1552_, v___x_1553_);
return v___x_1554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___lam__1(lean_object* v_toOne_1555_, lean_object* v_x_1556_){
_start:
{
lean_inc(v_toOne_1555_);
return v_toOne_1555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___lam__1___boxed(lean_object* v_toOne_1557_, lean_object* v_x_1558_){
_start:
{
lean_object* v_res_1559_; 
v_res_1559_ = lp_mathlib_MulActionHom_instMonoid___redArg___lam__1(v_toOne_1557_, v_x_1558_);
lean_dec(v_x_1558_);
lean_dec(v_toOne_1557_);
return v_res_1559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg(lean_object* v_inst_1560_){
_start:
{
lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v_toOne_1563_; lean_object* v_toMul_1564_; lean_object* v___f_1565_; lean_object* v___f_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; 
v___x_1561_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1560_);
v___x_1562_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1561_);
v_toOne_1563_ = lean_ctor_get(v___x_1562_, 0);
lean_inc(v_toOne_1563_);
v_toMul_1564_ = lean_ctor_get(v___x_1562_, 1);
lean_inc(v_toMul_1564_);
lean_dec_ref(v___x_1562_);
v___f_1565_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1565_, 0, v_toMul_1564_);
v___f_1566_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_instMonoid___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_1566_, 0, v_toOne_1563_);
lean_inc_ref(v___f_1566_);
lean_inc_ref(v___f_1565_);
v___x_1567_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_1567_, 0, lean_box(0));
lean_closure_set(v___x_1567_, 1, v___f_1565_);
lean_closure_set(v___x_1567_, 2, v___f_1566_);
v___x_1568_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1568_, 0, v___f_1566_);
lean_ctor_set(v___x_1568_, 1, v___f_1565_);
lean_ctor_set(v___x_1568_, 2, v___x_1567_);
return v___x_1568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___redArg___boxed(lean_object* v_inst_1569_){
_start:
{
lean_object* v_res_1570_; 
v_res_1570_ = lp_mathlib_MulActionHom_instMonoid___redArg(v_inst_1569_);
lean_dec_ref(v_inst_1569_);
return v_res_1570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid(lean_object* v_M_1571_, lean_object* v_N_1572_, lean_object* v_X_1573_, lean_object* v_Y_1574_, lean_object* v_00_u03c3_1575_, lean_object* v_inst_1576_, lean_object* v_inst_1577_, lean_object* v_inst_1578_, lean_object* v_inst_1579_){
_start:
{
lean_object* v___x_1580_; 
v___x_1580_ = lp_mathlib_MulActionHom_instMonoid___redArg(v_inst_1578_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instMonoid___boxed(lean_object* v_M_1581_, lean_object* v_N_1582_, lean_object* v_X_1583_, lean_object* v_Y_1584_, lean_object* v_00_u03c3_1585_, lean_object* v_inst_1586_, lean_object* v_inst_1587_, lean_object* v_inst_1588_, lean_object* v_inst_1589_){
_start:
{
lean_object* v_res_1590_; 
v_res_1590_ = lp_mathlib_MulActionHom_instMonoid(v_M_1581_, v_N_1582_, v_X_1583_, v_Y_1584_, v_00_u03c3_1585_, v_inst_1586_, v_inst_1587_, v_inst_1588_, v_inst_1589_);
lean_dec(v_inst_1589_);
lean_dec_ref(v_inst_1588_);
lean_dec_ref(v_inst_1587_);
lean_dec(v_inst_1586_);
lean_dec(v_00_u03c3_1585_);
return v_res_1590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid___redArg(lean_object* v_inst_1591_){
_start:
{
lean_object* v___x_1592_; 
v___x_1592_ = lp_mathlib_MulActionHom_instMonoid___redArg(v_inst_1591_);
return v___x_1592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid___redArg___boxed(lean_object* v_inst_1593_){
_start:
{
lean_object* v_res_1594_; 
v_res_1594_ = lp_mathlib_MulActionHom_instCommMonoid___redArg(v_inst_1593_);
lean_dec_ref(v_inst_1593_);
return v_res_1594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid(lean_object* v_M_1595_, lean_object* v_N_1596_, lean_object* v_X_1597_, lean_object* v_Y_1598_, lean_object* v_00_u03c3_1599_, lean_object* v_inst_1600_, lean_object* v_inst_1601_, lean_object* v_inst_1602_, lean_object* v_inst_1603_){
_start:
{
lean_object* v___x_1604_; 
v___x_1604_ = lp_mathlib_MulActionHom_instMonoid___redArg(v_inst_1602_);
return v___x_1604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommMonoid___boxed(lean_object* v_M_1605_, lean_object* v_N_1606_, lean_object* v_X_1607_, lean_object* v_Y_1608_, lean_object* v_00_u03c3_1609_, lean_object* v_inst_1610_, lean_object* v_inst_1611_, lean_object* v_inst_1612_, lean_object* v_inst_1613_){
_start:
{
lean_object* v_res_1614_; 
v_res_1614_ = lp_mathlib_MulActionHom_instCommMonoid(v_M_1605_, v_N_1606_, v_X_1607_, v_Y_1608_, v_00_u03c3_1609_, v_inst_1610_, v_inst_1611_, v_inst_1612_, v_inst_1613_);
lean_dec(v_inst_1613_);
lean_dec_ref(v_inst_1612_);
lean_dec_ref(v_inst_1611_);
lean_dec(v_inst_1610_);
lean_dec(v_00_u03c3_1609_);
return v_res_1614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSemiring___redArg(lean_object* v_inst_1615_){
_start:
{
lean_object* v_toAddCommMonoid_1616_; lean_object* v_toMonoid_1617_; lean_object* v___x_1619_; uint8_t v_isShared_1620_; uint8_t v_isSharedCheck_1630_; 
v_toAddCommMonoid_1616_ = lean_ctor_get(v_inst_1615_, 0);
v_toMonoid_1617_ = lean_ctor_get(v_inst_1615_, 1);
v_isSharedCheck_1630_ = !lean_is_exclusive(v_inst_1615_);
if (v_isSharedCheck_1630_ == 0)
{
lean_object* v_unused_1631_; 
v_unused_1631_ = lean_ctor_get(v_inst_1615_, 2);
lean_dec(v_unused_1631_);
v___x_1619_ = v_inst_1615_;
v_isShared_1620_ = v_isSharedCheck_1630_;
goto v_resetjp_1618_;
}
else
{
lean_inc(v_toMonoid_1617_);
lean_inc(v_toAddCommMonoid_1616_);
lean_dec(v_inst_1615_);
v___x_1619_ = lean_box(0);
v_isShared_1620_ = v_isSharedCheck_1630_;
goto v_resetjp_1618_;
}
v_resetjp_1618_:
{
lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v_toOne_1623_; lean_object* v_toZero_1624_; lean_object* v_toAdd_1625_; lean_object* v___x_1626_; lean_object* v___x_1628_; 
v___x_1621_ = lp_mathlib_MulActionHom_instMonoid___redArg(v_toMonoid_1617_);
lean_dec_ref(v_toMonoid_1617_);
v___x_1622_ = lp_mathlib_MulActionHom_instAddMonoid___redArg(v_toAddCommMonoid_1616_);
v_toOne_1623_ = lean_ctor_get(v___x_1621_, 0);
lean_inc(v_toOne_1623_);
v_toZero_1624_ = lean_ctor_get(v___x_1622_, 0);
lean_inc(v_toZero_1624_);
v_toAdd_1625_ = lean_ctor_get(v___x_1622_, 1);
lean_inc(v_toAdd_1625_);
v___x_1626_ = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(v___x_1626_, 0, lean_box(0));
lean_closure_set(v___x_1626_, 1, v_toOne_1623_);
lean_closure_set(v___x_1626_, 2, v_toZero_1624_);
lean_closure_set(v___x_1626_, 3, v_toAdd_1625_);
if (v_isShared_1620_ == 0)
{
lean_ctor_set(v___x_1619_, 2, v___x_1626_);
lean_ctor_set(v___x_1619_, 1, v___x_1621_);
lean_ctor_set(v___x_1619_, 0, v___x_1622_);
v___x_1628_ = v___x_1619_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v___x_1622_);
lean_ctor_set(v_reuseFailAlloc_1629_, 1, v___x_1621_);
lean_ctor_set(v_reuseFailAlloc_1629_, 2, v___x_1626_);
v___x_1628_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
return v___x_1628_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSemiring(lean_object* v_M_1632_, lean_object* v_N_1633_, lean_object* v_X_1634_, lean_object* v_Y_1635_, lean_object* v_00_u03c3_1636_, lean_object* v_inst_1637_, lean_object* v_inst_1638_, lean_object* v_inst_1639_, lean_object* v_inst_1640_){
_start:
{
lean_object* v___x_1641_; 
v___x_1641_ = lp_mathlib_MulActionHom_instSemiring___redArg(v_inst_1639_);
return v___x_1641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instSemiring___boxed(lean_object* v_M_1642_, lean_object* v_N_1643_, lean_object* v_X_1644_, lean_object* v_Y_1645_, lean_object* v_00_u03c3_1646_, lean_object* v_inst_1647_, lean_object* v_inst_1648_, lean_object* v_inst_1649_, lean_object* v_inst_1650_){
_start:
{
lean_object* v_res_1651_; 
v_res_1651_ = lp_mathlib_MulActionHom_instSemiring(v_M_1642_, v_N_1643_, v_X_1644_, v_Y_1645_, v_00_u03c3_1646_, v_inst_1647_, v_inst_1648_, v_inst_1649_, v_inst_1650_);
lean_dec(v_inst_1650_);
lean_dec_ref(v_inst_1648_);
lean_dec(v_inst_1647_);
lean_dec(v_00_u03c3_1646_);
return v_res_1651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommSemiring___redArg(lean_object* v_inst_1652_){
_start:
{
lean_object* v___x_1653_; 
v___x_1653_ = lp_mathlib_MulActionHom_instSemiring___redArg(v_inst_1652_);
return v___x_1653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommSemiring(lean_object* v_M_1654_, lean_object* v_N_1655_, lean_object* v_X_1656_, lean_object* v_Y_1657_, lean_object* v_00_u03c3_1658_, lean_object* v_inst_1659_, lean_object* v_inst_1660_, lean_object* v_inst_1661_, lean_object* v_inst_1662_){
_start:
{
lean_object* v___x_1663_; 
v___x_1663_ = lp_mathlib_MulActionHom_instSemiring___redArg(v_inst_1661_);
return v___x_1663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommSemiring___boxed(lean_object* v_M_1664_, lean_object* v_N_1665_, lean_object* v_X_1666_, lean_object* v_Y_1667_, lean_object* v_00_u03c3_1668_, lean_object* v_inst_1669_, lean_object* v_inst_1670_, lean_object* v_inst_1671_, lean_object* v_inst_1672_){
_start:
{
lean_object* v_res_1673_; 
v_res_1673_ = lp_mathlib_MulActionHom_instCommSemiring(v_M_1664_, v_N_1665_, v_X_1666_, v_Y_1667_, v_00_u03c3_1668_, v_inst_1669_, v_inst_1670_, v_inst_1671_, v_inst_1672_);
lean_dec(v_inst_1672_);
lean_dec_ref(v_inst_1670_);
lean_dec(v_inst_1669_);
lean_dec(v_00_u03c3_1668_);
return v_res_1673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instRing___redArg(lean_object* v_inst_1674_){
_start:
{
lean_object* v_toSemiring_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1679_; uint8_t v_isShared_1680_; uint8_t v_isSharedCheck_1690_; 
v_toSemiring_1675_ = lean_ctor_get(v_inst_1674_, 0);
lean_inc_ref(v_toSemiring_1675_);
v___x_1676_ = lp_mathlib_MulActionHom_instSemiring___redArg(v_toSemiring_1675_);
v___x_1677_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_1674_);
v_isSharedCheck_1690_ = !lean_is_exclusive(v_inst_1674_);
if (v_isSharedCheck_1690_ == 0)
{
lean_object* v_unused_1691_; lean_object* v_unused_1692_; lean_object* v_unused_1693_; lean_object* v_unused_1694_; lean_object* v_unused_1695_; 
v_unused_1691_ = lean_ctor_get(v_inst_1674_, 4);
lean_dec(v_unused_1691_);
v_unused_1692_ = lean_ctor_get(v_inst_1674_, 3);
lean_dec(v_unused_1692_);
v_unused_1693_ = lean_ctor_get(v_inst_1674_, 2);
lean_dec(v_unused_1693_);
v_unused_1694_ = lean_ctor_get(v_inst_1674_, 1);
lean_dec(v_unused_1694_);
v_unused_1695_ = lean_ctor_get(v_inst_1674_, 0);
lean_dec(v_unused_1695_);
v___x_1679_ = v_inst_1674_;
v_isShared_1680_ = v_isSharedCheck_1690_;
goto v_resetjp_1678_;
}
else
{
lean_dec(v_inst_1674_);
v___x_1679_ = lean_box(0);
v_isShared_1680_ = v_isSharedCheck_1690_;
goto v_resetjp_1678_;
}
v_resetjp_1678_:
{
lean_object* v___x_1681_; lean_object* v_toNeg_1682_; lean_object* v_toSub_1683_; lean_object* v_toZSMul_1684_; lean_object* v_toNatCast_1685_; lean_object* v___x_1686_; lean_object* v___x_1688_; 
v___x_1681_ = lp_mathlib_MulActionHom_instAddGroup___redArg(v___x_1677_);
v_toNeg_1682_ = lean_ctor_get(v___x_1681_, 1);
lean_inc_n(v_toNeg_1682_, 2);
v_toSub_1683_ = lean_ctor_get(v___x_1681_, 2);
lean_inc(v_toSub_1683_);
v_toZSMul_1684_ = lean_ctor_get(v___x_1681_, 3);
lean_inc(v_toZSMul_1684_);
lean_dec_ref(v___x_1681_);
v_toNatCast_1685_ = lean_ctor_get(v___x_1676_, 2);
lean_inc(v_toNatCast_1685_);
v___x_1686_ = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(v___x_1686_, 0, lean_box(0));
lean_closure_set(v___x_1686_, 1, v_toNatCast_1685_);
lean_closure_set(v___x_1686_, 2, v_toNeg_1682_);
if (v_isShared_1680_ == 0)
{
lean_ctor_set(v___x_1679_, 4, v___x_1686_);
lean_ctor_set(v___x_1679_, 3, v_toZSMul_1684_);
lean_ctor_set(v___x_1679_, 2, v_toSub_1683_);
lean_ctor_set(v___x_1679_, 1, v_toNeg_1682_);
lean_ctor_set(v___x_1679_, 0, v___x_1676_);
v___x_1688_ = v___x_1679_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v___x_1676_);
lean_ctor_set(v_reuseFailAlloc_1689_, 1, v_toNeg_1682_);
lean_ctor_set(v_reuseFailAlloc_1689_, 2, v_toSub_1683_);
lean_ctor_set(v_reuseFailAlloc_1689_, 3, v_toZSMul_1684_);
lean_ctor_set(v_reuseFailAlloc_1689_, 4, v___x_1686_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instRing(lean_object* v_M_1696_, lean_object* v_N_1697_, lean_object* v_X_1698_, lean_object* v_Y_1699_, lean_object* v_00_u03c3_1700_, lean_object* v_inst_1701_, lean_object* v_inst_1702_, lean_object* v_inst_1703_, lean_object* v_inst_1704_){
_start:
{
lean_object* v___x_1705_; 
v___x_1705_ = lp_mathlib_MulActionHom_instRing___redArg(v_inst_1703_);
return v___x_1705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instRing___boxed(lean_object* v_M_1706_, lean_object* v_N_1707_, lean_object* v_X_1708_, lean_object* v_Y_1709_, lean_object* v_00_u03c3_1710_, lean_object* v_inst_1711_, lean_object* v_inst_1712_, lean_object* v_inst_1713_, lean_object* v_inst_1714_){
_start:
{
lean_object* v_res_1715_; 
v_res_1715_ = lp_mathlib_MulActionHom_instRing(v_M_1706_, v_N_1707_, v_X_1708_, v_Y_1709_, v_00_u03c3_1710_, v_inst_1711_, v_inst_1712_, v_inst_1713_, v_inst_1714_);
lean_dec(v_inst_1714_);
lean_dec_ref(v_inst_1712_);
lean_dec(v_inst_1711_);
lean_dec(v_00_u03c3_1710_);
return v_res_1715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommRing___redArg(lean_object* v_inst_1716_){
_start:
{
lean_object* v___x_1717_; 
v___x_1717_ = lp_mathlib_MulActionHom_instRing___redArg(v_inst_1716_);
return v___x_1717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommRing(lean_object* v_M_1718_, lean_object* v_N_1719_, lean_object* v_X_1720_, lean_object* v_Y_1721_, lean_object* v_00_u03c3_1722_, lean_object* v_inst_1723_, lean_object* v_inst_1724_, lean_object* v_inst_1725_, lean_object* v_inst_1726_){
_start:
{
lean_object* v___x_1727_; 
v___x_1727_ = lp_mathlib_MulActionHom_instRing___redArg(v_inst_1725_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_instCommRing___boxed(lean_object* v_M_1728_, lean_object* v_N_1729_, lean_object* v_X_1730_, lean_object* v_Y_1731_, lean_object* v_00_u03c3_1732_, lean_object* v_inst_1733_, lean_object* v_inst_1734_, lean_object* v_inst_1735_, lean_object* v_inst_1736_){
_start:
{
lean_object* v_res_1737_; 
v_res_1737_ = lp_mathlib_MulActionHom_instCommRing(v_M_1728_, v_N_1729_, v_X_1730_, v_Y_1731_, v_00_u03c3_1732_, v_inst_1733_, v_inst_1734_, v_inst_1735_, v_inst_1736_);
lean_dec(v_inst_1736_);
lean_dec_ref(v_inst_1734_);
lean_dec(v_inst_1733_);
lean_dec(v_00_u03c3_1732_);
return v_res_1737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_instMonoidId___lam__0(lean_object* v_f_1738_, lean_object* v_g_1739_, lean_object* v___y_1740_){
_start:
{
lean_object* v___x_1741_; 
v___x_1741_ = lp_mathlib_MulActionHom_comp___redArg___lam__0(v_g_1739_, v_f_1738_, v___y_1740_);
return v___x_1741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_instMonoidId(lean_object* v_M_1750_, lean_object* v_X_1751_, lean_object* v_inst_1752_){
_start:
{
lean_object* v___x_1753_; 
v___x_1753_ = ((lean_object*)(lp_mathlib_MulActionHom_End_instMonoidId___closed__2));
return v___x_1753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_instMonoidId___boxed(lean_object* v_M_1754_, lean_object* v_X_1755_, lean_object* v_inst_1756_){
_start:
{
lean_object* v_res_1757_; 
v_res_1757_ = lp_mathlib_MulActionHom_End_instMonoidId(v_M_1754_, v_X_1755_, v_inst_1756_);
lean_dec(v_inst_1756_);
return v_res_1757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_instAddMonoidId(lean_object* v_M_1765_, lean_object* v_X_1766_, lean_object* v_inst_1767_){
_start:
{
lean_object* v___x_1768_; 
v___x_1768_ = ((lean_object*)(lp_mathlib_AddActionHom_End_instAddMonoidId___closed__1));
return v___x_1768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_instAddMonoidId___boxed(lean_object* v_M_1769_, lean_object* v_X_1770_, lean_object* v_inst_1771_){
_start:
{
lean_object* v_res_1772_; 
v_res_1772_ = lp_mathlib_AddActionHom_End_instAddMonoidId(v_M_1769_, v_X_1770_, v_inst_1771_);
lean_dec(v_inst_1771_);
return v_res_1772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__0(lean_object* v_toMul_1773_, lean_object* v_m_1774_, lean_object* v___y_1775_){
_start:
{
lean_object* v___x_1776_; 
v___x_1776_ = lean_apply_2(v_toMul_1773_, v___y_1775_, v_m_1774_);
return v___x_1776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__1(lean_object* v_toOne_1777_, lean_object* v_f_1778_){
_start:
{
lean_object* v___x_1779_; 
v___x_1779_ = lean_apply_1(v_f_1778_, v_toOne_1777_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg(lean_object* v_inst_1780_){
_start:
{
lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v_toOne_1783_; lean_object* v_toMul_1784_; lean_object* v___x_1786_; uint8_t v_isShared_1787_; uint8_t v_isSharedCheck_1793_; 
v___x_1781_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1780_);
v___x_1782_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1781_);
v_toOne_1783_ = lean_ctor_get(v___x_1782_, 0);
v_toMul_1784_ = lean_ctor_get(v___x_1782_, 1);
v_isSharedCheck_1793_ = !lean_is_exclusive(v___x_1782_);
if (v_isSharedCheck_1793_ == 0)
{
v___x_1786_ = v___x_1782_;
v_isShared_1787_ = v_isSharedCheck_1793_;
goto v_resetjp_1785_;
}
else
{
lean_inc(v_toMul_1784_);
lean_inc(v_toOne_1783_);
lean_dec(v___x_1782_);
v___x_1786_ = lean_box(0);
v_isShared_1787_ = v_isSharedCheck_1793_;
goto v_resetjp_1785_;
}
v_resetjp_1785_:
{
lean_object* v___f_1788_; lean_object* v___f_1789_; lean_object* v___x_1791_; 
v___f_1788_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1788_, 0, v_toMul_1784_);
v___f_1789_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1789_, 0, v_toOne_1783_);
if (v_isShared_1787_ == 0)
{
lean_ctor_set(v___x_1786_, 1, v___f_1788_);
lean_ctor_set(v___x_1786_, 0, v___f_1789_);
v___x_1791_ = v___x_1786_;
goto v_reusejp_1790_;
}
else
{
lean_object* v_reuseFailAlloc_1792_; 
v_reuseFailAlloc_1792_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1792_, 0, v___f_1789_);
lean_ctor_set(v_reuseFailAlloc_1792_, 1, v___f_1788_);
v___x_1791_ = v_reuseFailAlloc_1792_;
goto v_reusejp_1790_;
}
v_reusejp_1790_:
{
return v___x_1791_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___boxed(lean_object* v_inst_1794_){
_start:
{
lean_object* v_res_1795_; 
v_res_1795_ = lp_mathlib_MulActionHom_End_equivMulOpposite___redArg(v_inst_1794_);
lean_dec_ref(v_inst_1794_);
return v_res_1795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite(lean_object* v_M_1796_, lean_object* v_inst_1797_){
_start:
{
lean_object* v___x_1798_; 
v___x_1798_ = lp_mathlib_MulActionHom_End_equivMulOpposite___redArg(v_inst_1797_);
return v___x_1798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_equivMulOpposite___boxed(lean_object* v_M_1799_, lean_object* v_inst_1800_){
_start:
{
lean_object* v_res_1801_; 
v_res_1801_ = lp_mathlib_MulActionHom_End_equivMulOpposite(v_M_1799_, v_inst_1800_);
lean_dec_ref(v_inst_1800_);
return v_res_1801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__0(lean_object* v_toAdd_1802_, lean_object* v_m_1803_, lean_object* v___y_1804_){
_start:
{
lean_object* v___x_1805_; 
v___x_1805_ = lean_apply_2(v_toAdd_1802_, v___y_1804_, v_m_1803_);
return v___x_1805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__1(lean_object* v_toZero_1806_, lean_object* v_f_1807_){
_start:
{
lean_object* v___x_1808_; 
v___x_1808_ = lean_apply_1(v_f_1807_, v_toZero_1806_);
return v___x_1808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg(lean_object* v_inst_1809_){
_start:
{
lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v_toZero_1812_; lean_object* v_toAdd_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1822_; 
v___x_1810_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1809_);
v___x_1811_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1810_);
v_toZero_1812_ = lean_ctor_get(v___x_1811_, 0);
v_toAdd_1813_ = lean_ctor_get(v___x_1811_, 1);
v_isSharedCheck_1822_ = !lean_is_exclusive(v___x_1811_);
if (v_isSharedCheck_1822_ == 0)
{
v___x_1815_ = v___x_1811_;
v_isShared_1816_ = v_isSharedCheck_1822_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_toAdd_1813_);
lean_inc(v_toZero_1812_);
lean_dec(v___x_1811_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1822_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___f_1817_; lean_object* v___f_1818_; lean_object* v___x_1820_; 
v___f_1817_ = lean_alloc_closure((void*)(lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1817_, 0, v_toAdd_1813_);
v___f_1818_ = lean_alloc_closure((void*)(lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1818_, 0, v_toZero_1812_);
if (v_isShared_1816_ == 0)
{
lean_ctor_set(v___x_1815_, 1, v___f_1817_);
lean_ctor_set(v___x_1815_, 0, v___f_1818_);
v___x_1820_ = v___x_1815_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v___f_1818_);
lean_ctor_set(v_reuseFailAlloc_1821_, 1, v___f_1817_);
v___x_1820_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
return v___x_1820_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___boxed(lean_object* v_inst_1823_){
_start:
{
lean_object* v_res_1824_; 
v_res_1824_ = lp_mathlib_AddActionHom_End_equivAddOpposite___redArg(v_inst_1823_);
lean_dec_ref(v_inst_1823_);
return v_res_1824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite(lean_object* v_M_1825_, lean_object* v_inst_1826_){
_start:
{
lean_object* v___x_1827_; 
v___x_1827_ = lp_mathlib_AddActionHom_End_equivAddOpposite___redArg(v_inst_1826_);
return v___x_1827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_equivAddOpposite___boxed(lean_object* v_M_1828_, lean_object* v_inst_1829_){
_start:
{
lean_object* v_res_1830_; 
v_res_1830_ = lp_mathlib_AddActionHom_End_equivAddOpposite(v_M_1828_, v_inst_1829_);
lean_dec_ref(v_inst_1829_);
return v_res_1830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg___lam__0(lean_object* v_toMul_1831_, lean_object* v_m_1832_, lean_object* v___y_1833_){
_start:
{
lean_object* v___x_1834_; 
v___x_1834_ = lean_apply_2(v_toMul_1831_, v_m_1832_, v___y_1833_);
return v___x_1834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg(lean_object* v_inst_1835_){
_start:
{
lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v_toOne_1838_; lean_object* v_toMul_1839_; lean_object* v___x_1841_; uint8_t v_isShared_1842_; uint8_t v_isSharedCheck_1848_; 
v___x_1836_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1835_);
v___x_1837_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1836_);
v_toOne_1838_ = lean_ctor_get(v___x_1837_, 0);
v_toMul_1839_ = lean_ctor_get(v___x_1837_, 1);
v_isSharedCheck_1848_ = !lean_is_exclusive(v___x_1837_);
if (v_isSharedCheck_1848_ == 0)
{
v___x_1841_ = v___x_1837_;
v_isShared_1842_ = v_isSharedCheck_1848_;
goto v_resetjp_1840_;
}
else
{
lean_inc(v_toMul_1839_);
lean_inc(v_toOne_1838_);
lean_dec(v___x_1837_);
v___x_1841_ = lean_box(0);
v_isShared_1842_ = v_isSharedCheck_1848_;
goto v_resetjp_1840_;
}
v_resetjp_1840_:
{
lean_object* v___f_1843_; lean_object* v___f_1844_; lean_object* v___x_1846_; 
v___f_1843_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1843_, 0, v_toMul_1839_);
v___f_1844_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_End_equivMulOpposite___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1844_, 0, v_toOne_1838_);
if (v_isShared_1842_ == 0)
{
lean_ctor_set(v___x_1841_, 1, v___f_1843_);
lean_ctor_set(v___x_1841_, 0, v___f_1844_);
v___x_1846_ = v___x_1841_;
goto v_reusejp_1845_;
}
else
{
lean_object* v_reuseFailAlloc_1847_; 
v_reuseFailAlloc_1847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1847_, 0, v___f_1844_);
lean_ctor_set(v_reuseFailAlloc_1847_, 1, v___f_1843_);
v___x_1846_ = v_reuseFailAlloc_1847_;
goto v_reusejp_1845_;
}
v_reusejp_1845_:
{
return v___x_1846_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg___boxed(lean_object* v_inst_1849_){
_start:
{
lean_object* v_res_1850_; 
v_res_1850_ = lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg(v_inst_1849_);
lean_dec_ref(v_inst_1849_);
return v_res_1850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv(lean_object* v_M_1851_, lean_object* v_inst_1852_){
_start:
{
lean_object* v___x_1853_; 
v___x_1853_ = lp_mathlib_MulActionHom_End_mulOppositeEquiv___redArg(v_inst_1852_);
return v___x_1853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulActionHom_End_mulOppositeEquiv___boxed(lean_object* v_M_1854_, lean_object* v_inst_1855_){
_start:
{
lean_object* v_res_1856_; 
v_res_1856_ = lp_mathlib_MulActionHom_End_mulOppositeEquiv(v_M_1854_, v_inst_1855_);
lean_dec_ref(v_inst_1855_);
return v_res_1856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg___lam__0(lean_object* v_toAdd_1857_, lean_object* v_m_1858_, lean_object* v___y_1859_){
_start:
{
lean_object* v___x_1860_; 
v___x_1860_ = lean_apply_2(v_toAdd_1857_, v_m_1858_, v___y_1859_);
return v___x_1860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg(lean_object* v_inst_1861_){
_start:
{
lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v_toZero_1864_; lean_object* v_toAdd_1865_; lean_object* v___x_1867_; uint8_t v_isShared_1868_; uint8_t v_isSharedCheck_1874_; 
v___x_1862_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1861_);
v___x_1863_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1862_);
v_toZero_1864_ = lean_ctor_get(v___x_1863_, 0);
v_toAdd_1865_ = lean_ctor_get(v___x_1863_, 1);
v_isSharedCheck_1874_ = !lean_is_exclusive(v___x_1863_);
if (v_isSharedCheck_1874_ == 0)
{
v___x_1867_ = v___x_1863_;
v_isShared_1868_ = v_isSharedCheck_1874_;
goto v_resetjp_1866_;
}
else
{
lean_inc(v_toAdd_1865_);
lean_inc(v_toZero_1864_);
lean_dec(v___x_1863_);
v___x_1867_ = lean_box(0);
v_isShared_1868_ = v_isSharedCheck_1874_;
goto v_resetjp_1866_;
}
v_resetjp_1866_:
{
lean_object* v___f_1869_; lean_object* v___f_1870_; lean_object* v___x_1872_; 
v___f_1869_ = lean_alloc_closure((void*)(lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1869_, 0, v_toAdd_1865_);
v___f_1870_ = lean_alloc_closure((void*)(lp_mathlib_AddActionHom_End_equivAddOpposite___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1870_, 0, v_toZero_1864_);
if (v_isShared_1868_ == 0)
{
lean_ctor_set(v___x_1867_, 1, v___f_1869_);
lean_ctor_set(v___x_1867_, 0, v___f_1870_);
v___x_1872_ = v___x_1867_;
goto v_reusejp_1871_;
}
else
{
lean_object* v_reuseFailAlloc_1873_; 
v_reuseFailAlloc_1873_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1873_, 0, v___f_1870_);
lean_ctor_set(v_reuseFailAlloc_1873_, 1, v___f_1869_);
v___x_1872_ = v_reuseFailAlloc_1873_;
goto v_reusejp_1871_;
}
v_reusejp_1871_:
{
return v___x_1872_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg___boxed(lean_object* v_inst_1875_){
_start:
{
lean_object* v_res_1876_; 
v_res_1876_ = lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg(v_inst_1875_);
lean_dec_ref(v_inst_1875_);
return v_res_1876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv(lean_object* v_M_1877_, lean_object* v_inst_1878_){
_start:
{
lean_object* v___x_1879_; 
v___x_1879_ = lp_mathlib_AddActionHom_End_addOppositeEquiv___redArg(v_inst_1878_);
return v___x_1879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddActionHom_End_addOppositeEquiv___boxed(lean_object* v_M_1880_, lean_object* v_inst_1881_){
_start:
{
lean_object* v_res_1882_; 
v_res_1882_ = lp_mathlib_AddActionHom_End_addOppositeEquiv(v_M_1880_, v_inst_1881_);
lean_dec_ref(v_inst_1881_);
return v_res_1882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom___redArg(lean_object* v_self_1883_){
_start:
{
lean_inc(v_self_1883_);
return v_self_1883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom___redArg___boxed(lean_object* v_self_1884_){
_start:
{
lean_object* v_res_1885_; 
v_res_1885_ = lp_mathlib_DistribMulActionHom_toAddMonoidHom___redArg(v_self_1884_);
lean_dec(v_self_1884_);
return v_res_1885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom(lean_object* v_M_1886_, lean_object* v_inst_1887_, lean_object* v_N_1888_, lean_object* v_inst_1889_, lean_object* v_00_u03c6_1890_, lean_object* v_A_1891_, lean_object* v_inst_1892_, lean_object* v_inst_1893_, lean_object* v_B_1894_, lean_object* v_inst_1895_, lean_object* v_inst_1896_, lean_object* v_self_1897_){
_start:
{
lean_inc(v_self_1897_);
return v_self_1897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_toAddMonoidHom___boxed(lean_object* v_M_1898_, lean_object* v_inst_1899_, lean_object* v_N_1900_, lean_object* v_inst_1901_, lean_object* v_00_u03c6_1902_, lean_object* v_A_1903_, lean_object* v_inst_1904_, lean_object* v_inst_1905_, lean_object* v_B_1906_, lean_object* v_inst_1907_, lean_object* v_inst_1908_, lean_object* v_self_1909_){
_start:
{
lean_object* v_res_1910_; 
v_res_1910_ = lp_mathlib_DistribMulActionHom_toAddMonoidHom(v_M_1898_, v_inst_1899_, v_N_1900_, v_inst_1901_, v_00_u03c6_1902_, v_A_1903_, v_inst_1904_, v_inst_1905_, v_B_1906_, v_inst_1907_, v_inst_1908_, v_self_1909_);
lean_dec(v_self_1909_);
lean_dec(v_inst_1908_);
lean_dec_ref(v_inst_1907_);
lean_dec(v_inst_1905_);
lean_dec_ref(v_inst_1904_);
lean_dec(v_00_u03c6_1902_);
lean_dec_ref(v_inst_1901_);
lean_dec_ref(v_inst_1899_);
return v_res_1910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom___redArg(lean_object* v_self_1911_){
_start:
{
lean_inc(v_self_1911_);
return v_self_1911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom___redArg___boxed(lean_object* v_self_1912_){
_start:
{
lean_object* v_res_1913_; 
v_res_1913_ = lp_mathlib_MulDistribMulActionHom_toMonoidHom___redArg(v_self_1912_);
lean_dec(v_self_1912_);
return v_res_1913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom(lean_object* v_M_1914_, lean_object* v_inst_1915_, lean_object* v_N_1916_, lean_object* v_inst_1917_, lean_object* v_00_u03c6_1918_, lean_object* v_A_1919_, lean_object* v_inst_1920_, lean_object* v_inst_1921_, lean_object* v_B_1922_, lean_object* v_inst_1923_, lean_object* v_inst_1924_, lean_object* v_self_1925_){
_start:
{
lean_inc(v_self_1925_);
return v_self_1925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_toMonoidHom___boxed(lean_object* v_M_1926_, lean_object* v_inst_1927_, lean_object* v_N_1928_, lean_object* v_inst_1929_, lean_object* v_00_u03c6_1930_, lean_object* v_A_1931_, lean_object* v_inst_1932_, lean_object* v_inst_1933_, lean_object* v_B_1934_, lean_object* v_inst_1935_, lean_object* v_inst_1936_, lean_object* v_self_1937_){
_start:
{
lean_object* v_res_1938_; 
v_res_1938_ = lp_mathlib_MulDistribMulActionHom_toMonoidHom(v_M_1926_, v_inst_1927_, v_N_1928_, v_inst_1929_, v_00_u03c6_1930_, v_A_1931_, v_inst_1932_, v_inst_1933_, v_B_1934_, v_inst_1935_, v_inst_1936_, v_self_1937_);
lean_dec(v_self_1937_);
lean_dec(v_inst_1936_);
lean_dec_ref(v_inst_1935_);
lean_dec(v_inst_1933_);
lean_dec_ref(v_inst_1932_);
lean_dec(v_00_u03c6_1930_);
lean_dec_ref(v_inst_1929_);
lean_dec_ref(v_inst_1927_);
return v_res_1938_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1(void){
_start:
{
lean_object* v___x_1964_; lean_object* v___x_1965_; 
v___x_1964_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__0));
v___x_1965_ = l_String_toRawSubstring_x27(v___x_1964_);
return v___x_1965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1(lean_object* v_x_1979_, lean_object* v_a_1980_, lean_object* v_a_1981_){
_start:
{
lean_object* v___x_1982_; uint8_t v___x_1983_; 
v___x_1982_ = ((lean_object*)(lp_mathlib_DistribMulActionHomLocal_u227a___closed__1));
lean_inc(v_x_1979_);
v___x_1983_ = l_Lean_Syntax_isOfKind(v_x_1979_, v___x_1982_);
if (v___x_1983_ == 0)
{
lean_object* v___x_1984_; lean_object* v___x_1985_; 
lean_dec(v_x_1979_);
v___x_1984_ = lean_box(1);
v___x_1985_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1985_, 0, v___x_1984_);
lean_ctor_set(v___x_1985_, 1, v_a_1981_);
return v___x_1985_;
}
else
{
lean_object* v_quotContext_1986_; lean_object* v_currMacroScope_1987_; lean_object* v_ref_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; uint8_t v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; 
v_quotContext_1986_ = lean_ctor_get(v_a_1980_, 1);
v_currMacroScope_1987_ = lean_ctor_get(v_a_1980_, 2);
v_ref_1988_ = lean_ctor_get(v_a_1980_, 5);
v___x_1989_ = lean_unsigned_to_nat(0u);
v___x_1990_ = l_Lean_Syntax_getArg(v_x_1979_, v___x_1989_);
v___x_1991_ = lean_unsigned_to_nat(2u);
v___x_1992_ = l_Lean_Syntax_getArg(v_x_1979_, v___x_1991_);
v___x_1993_ = lean_unsigned_to_nat(4u);
v___x_1994_ = l_Lean_Syntax_getArg(v_x_1979_, v___x_1993_);
lean_dec(v_x_1979_);
v___x_1995_ = 0;
v___x_1996_ = l_Lean_SourceInfo_fromRef(v_ref_1988_, v___x_1995_);
v___x_1997_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_1998_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1);
v___x_1999_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2));
lean_inc(v_currMacroScope_1987_);
lean_inc(v_quotContext_1986_);
v___x_2000_ = l_Lean_addMacroScope(v_quotContext_1986_, v___x_1999_, v_currMacroScope_1987_);
v___x_2001_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_1996_, 2);
v___x_2002_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2002_, 0, v___x_1996_);
lean_ctor_set(v___x_2002_, 1, v___x_1998_);
lean_ctor_set(v___x_2002_, 2, v___x_2000_);
lean_ctor_set(v___x_2002_, 3, v___x_2001_);
v___x_2003_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_2004_ = l_Lean_Syntax_node3(v___x_1996_, v___x_2003_, v___x_1992_, v___x_1990_, v___x_1994_);
v___x_2005_ = l_Lean_Syntax_node2(v___x_1996_, v___x_1997_, v___x_2002_, v___x_2004_);
v___x_2006_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2006_, 0, v___x_2005_);
lean_ctor_set(v___x_2006_, 1, v_a_1981_);
return v___x_2006_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___boxed(lean_object* v_x_2007_, lean_object* v_a_2008_, lean_object* v_a_2009_){
_start:
{
lean_object* v_res_2010_; 
v_res_2010_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1(v_x_2007_, v_a_2008_, v_a_2009_);
lean_dec_ref(v_a_2008_);
return v_res_2010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__1(lean_object* v_x_2011_, lean_object* v_a_2012_, lean_object* v_a_2013_){
_start:
{
lean_object* v___x_2014_; uint8_t v___x_2015_; 
v___x_2014_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_2011_);
v___x_2015_ = l_Lean_Syntax_isOfKind(v_x_2011_, v___x_2014_);
if (v___x_2015_ == 0)
{
lean_object* v___x_2016_; lean_object* v___x_2017_; 
lean_dec(v_x_2011_);
v___x_2016_ = lean_box(0);
v___x_2017_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2017_, 0, v___x_2016_);
lean_ctor_set(v___x_2017_, 1, v_a_2013_);
return v___x_2017_;
}
else
{
lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; uint8_t v___x_2021_; 
v___x_2018_ = lean_unsigned_to_nat(0u);
v___x_2019_ = l_Lean_Syntax_getArg(v_x_2011_, v___x_2018_);
v___x_2020_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_2019_);
v___x_2021_ = l_Lean_Syntax_isOfKind(v___x_2019_, v___x_2020_);
if (v___x_2021_ == 0)
{
lean_object* v___x_2022_; lean_object* v___x_2023_; 
lean_dec(v___x_2019_);
lean_dec(v_x_2011_);
v___x_2022_ = lean_box(0);
v___x_2023_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2023_, 0, v___x_2022_);
lean_ctor_set(v___x_2023_, 1, v_a_2013_);
return v___x_2023_;
}
else
{
lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; uint8_t v___x_2027_; 
v___x_2024_ = lean_unsigned_to_nat(1u);
v___x_2025_ = l_Lean_Syntax_getArg(v_x_2011_, v___x_2024_);
lean_dec(v_x_2011_);
v___x_2026_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_2025_);
v___x_2027_ = l_Lean_Syntax_matchesNull(v___x_2025_, v___x_2026_);
if (v___x_2027_ == 0)
{
lean_object* v___x_2028_; lean_object* v___x_2029_; 
lean_dec(v___x_2025_);
lean_dec(v___x_2019_);
v___x_2028_ = lean_box(0);
v___x_2029_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2029_, 0, v___x_2028_);
lean_ctor_set(v___x_2029_, 1, v_a_2013_);
return v___x_2029_;
}
else
{
lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v_ref_2034_; uint8_t v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; 
v___x_2030_ = l_Lean_Syntax_getArg(v___x_2025_, v___x_2018_);
v___x_2031_ = l_Lean_Syntax_getArg(v___x_2025_, v___x_2024_);
v___x_2032_ = lean_unsigned_to_nat(2u);
v___x_2033_ = l_Lean_Syntax_getArg(v___x_2025_, v___x_2032_);
lean_dec(v___x_2025_);
v_ref_2034_ = l_Lean_replaceRef(v___x_2019_, v_a_2012_);
lean_dec(v___x_2019_);
v___x_2035_ = 0;
v___x_2036_ = l_Lean_SourceInfo_fromRef(v_ref_2034_, v___x_2035_);
lean_dec(v_ref_2034_);
v___x_2037_ = ((lean_object*)(lp_mathlib_DistribMulActionHomLocal_u227a___closed__1));
v___x_2038_ = ((lean_object*)(lp_mathlib_DistribMulActionHomLocal_u227a___closed__2));
lean_inc_n(v___x_2036_, 2);
v___x_2039_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2039_, 0, v___x_2036_);
lean_ctor_set(v___x_2039_, 1, v___x_2038_);
v___x_2040_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_2041_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2041_, 0, v___x_2036_);
lean_ctor_set(v___x_2041_, 1, v___x_2040_);
v___x_2042_ = l_Lean_Syntax_node5(v___x_2036_, v___x_2037_, v___x_2031_, v___x_2039_, v___x_2030_, v___x_2041_, v___x_2033_);
v___x_2043_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2043_, 0, v___x_2042_);
lean_ctor_set(v___x_2043_, 1, v_a_2013_);
return v___x_2043_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__1___boxed(lean_object* v_x_2044_, lean_object* v_a_2045_, lean_object* v_a_2046_){
_start:
{
lean_object* v_res_2047_; 
v_res_2047_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__1(v_x_2044_, v_a_2045_, v_a_2046_);
lean_dec(v_a_2045_);
return v_res_2047_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1(void){
_start:
{
lean_object* v___x_2073_; lean_object* v___x_2074_; 
v___x_2073_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__0));
v___x_2074_ = l_String_toRawSubstring_x27(v___x_2073_);
return v___x_2074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1(lean_object* v_x_2085_, lean_object* v_a_2086_, lean_object* v_a_2087_){
_start:
{
lean_object* v___x_2088_; uint8_t v___x_2089_; 
v___x_2088_ = ((lean_object*)(lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__1));
lean_inc(v_x_2085_);
v___x_2089_ = l_Lean_Syntax_isOfKind(v_x_2085_, v___x_2088_);
if (v___x_2089_ == 0)
{
lean_object* v___x_2090_; lean_object* v___x_2091_; 
lean_dec(v_x_2085_);
v___x_2090_ = lean_box(1);
v___x_2091_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2091_, 0, v___x_2090_);
lean_ctor_set(v___x_2091_, 1, v_a_2087_);
return v___x_2091_;
}
else
{
lean_object* v_quotContext_2092_; lean_object* v_currMacroScope_2093_; lean_object* v_ref_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; uint8_t v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; 
v_quotContext_2092_ = lean_ctor_get(v_a_2086_, 1);
v_currMacroScope_2093_ = lean_ctor_get(v_a_2086_, 2);
v_ref_2094_ = lean_ctor_get(v_a_2086_, 5);
v___x_2095_ = lean_unsigned_to_nat(0u);
v___x_2096_ = l_Lean_Syntax_getArg(v_x_2085_, v___x_2095_);
v___x_2097_ = lean_unsigned_to_nat(2u);
v___x_2098_ = l_Lean_Syntax_getArg(v_x_2085_, v___x_2097_);
v___x_2099_ = lean_unsigned_to_nat(4u);
v___x_2100_ = l_Lean_Syntax_getArg(v_x_2085_, v___x_2099_);
lean_dec(v_x_2085_);
v___x_2101_ = 0;
v___x_2102_ = l_Lean_SourceInfo_fromRef(v_ref_2094_, v___x_2101_);
v___x_2103_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_2104_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__1);
v___x_2105_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__2));
lean_inc_n(v_currMacroScope_2093_, 3);
lean_inc_n(v_quotContext_2092_, 3);
v___x_2106_ = l_Lean_addMacroScope(v_quotContext_2092_, v___x_2105_, v_currMacroScope_2093_);
v___x_2107_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_2102_, 11);
v___x_2108_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2108_, 0, v___x_2102_);
lean_ctor_set(v___x_2108_, 1, v___x_2104_);
lean_ctor_set(v___x_2108_, 2, v___x_2106_);
lean_ctor_set(v___x_2108_, 3, v___x_2107_);
v___x_2109_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_2110_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1));
v___x_2111_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3));
v___x_2112_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4));
v___x_2113_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2113_, 0, v___x_2102_);
lean_ctor_set(v___x_2113_, 1, v___x_2112_);
v___x_2114_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6));
v___x_2115_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8);
v___x_2116_ = lean_box(0);
v___x_2117_ = l_Lean_addMacroScope(v_quotContext_2092_, v___x_2116_, v_currMacroScope_2093_);
v___x_2118_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10));
v___x_2119_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2119_, 0, v___x_2102_);
lean_ctor_set(v___x_2119_, 1, v___x_2115_);
lean_ctor_set(v___x_2119_, 2, v___x_2117_);
lean_ctor_set(v___x_2119_, 3, v___x_2118_);
v___x_2120_ = l_Lean_Syntax_node1(v___x_2102_, v___x_2114_, v___x_2119_);
v___x_2121_ = l_Lean_Syntax_node2(v___x_2102_, v___x_2111_, v___x_2113_, v___x_2120_);
v___x_2122_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1);
v___x_2123_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3));
v___x_2124_ = l_Lean_addMacroScope(v_quotContext_2092_, v___x_2123_, v_currMacroScope_2093_);
v___x_2125_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__5));
v___x_2126_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2126_, 0, v___x_2102_);
lean_ctor_set(v___x_2126_, 1, v___x_2122_);
lean_ctor_set(v___x_2126_, 2, v___x_2124_);
lean_ctor_set(v___x_2126_, 3, v___x_2125_);
v___x_2127_ = l_Lean_Syntax_node1(v___x_2102_, v___x_2109_, v___x_2098_);
v___x_2128_ = l_Lean_Syntax_node2(v___x_2102_, v___x_2103_, v___x_2126_, v___x_2127_);
v___x_2129_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19));
v___x_2130_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2130_, 0, v___x_2102_);
lean_ctor_set(v___x_2130_, 1, v___x_2129_);
v___x_2131_ = l_Lean_Syntax_node3(v___x_2102_, v___x_2110_, v___x_2121_, v___x_2128_, v___x_2130_);
v___x_2132_ = l_Lean_Syntax_node3(v___x_2102_, v___x_2109_, v___x_2131_, v___x_2096_, v___x_2100_);
v___x_2133_ = l_Lean_Syntax_node2(v___x_2102_, v___x_2103_, v___x_2108_, v___x_2132_);
v___x_2134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2134_, 0, v___x_2133_);
lean_ctor_set(v___x_2134_, 1, v_a_2087_);
return v___x_2134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___boxed(lean_object* v_x_2135_, lean_object* v_a_2136_, lean_object* v_a_2137_){
_start:
{
lean_object* v_res_2138_; 
v_res_2138_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1(v_x_2135_, v_a_2136_, v_a_2137_);
lean_dec_ref(v_a_2136_);
return v_res_2138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__2(lean_object* v_x_2139_, lean_object* v_a_2140_, lean_object* v_a_2141_){
_start:
{
lean_object* v___x_2142_; uint8_t v___x_2143_; 
v___x_2142_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_2139_);
v___x_2143_ = l_Lean_Syntax_isOfKind(v_x_2139_, v___x_2142_);
if (v___x_2143_ == 0)
{
lean_object* v___x_2144_; lean_object* v___x_2145_; 
lean_dec(v_x_2139_);
v___x_2144_ = lean_box(0);
v___x_2145_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2145_, 0, v___x_2144_);
lean_ctor_set(v___x_2145_, 1, v_a_2141_);
return v___x_2145_;
}
else
{
lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; uint8_t v___x_2149_; 
v___x_2146_ = lean_unsigned_to_nat(0u);
v___x_2147_ = l_Lean_Syntax_getArg(v_x_2139_, v___x_2146_);
v___x_2148_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_2147_);
v___x_2149_ = l_Lean_Syntax_isOfKind(v___x_2147_, v___x_2148_);
if (v___x_2149_ == 0)
{
lean_object* v___x_2150_; lean_object* v___x_2151_; 
lean_dec(v___x_2147_);
lean_dec(v_x_2139_);
v___x_2150_ = lean_box(0);
v___x_2151_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2151_, 0, v___x_2150_);
lean_ctor_set(v___x_2151_, 1, v_a_2141_);
return v___x_2151_;
}
else
{
lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; uint8_t v___x_2155_; 
v___x_2152_ = lean_unsigned_to_nat(1u);
v___x_2153_ = l_Lean_Syntax_getArg(v_x_2139_, v___x_2152_);
lean_dec(v_x_2139_);
v___x_2154_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_2153_);
v___x_2155_ = l_Lean_Syntax_matchesNull(v___x_2153_, v___x_2154_);
if (v___x_2155_ == 0)
{
lean_object* v___x_2156_; lean_object* v___x_2157_; 
lean_dec(v___x_2153_);
lean_dec(v___x_2147_);
v___x_2156_ = lean_box(0);
v___x_2157_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2157_, 0, v___x_2156_);
lean_ctor_set(v___x_2157_, 1, v_a_2141_);
return v___x_2157_;
}
else
{
lean_object* v___x_2158_; uint8_t v___x_2159_; 
v___x_2158_ = l_Lean_Syntax_getArg(v___x_2153_, v___x_2146_);
lean_inc(v___x_2158_);
v___x_2159_ = l_Lean_Syntax_isOfKind(v___x_2158_, v___x_2142_);
if (v___x_2159_ == 0)
{
lean_object* v___x_2160_; lean_object* v___x_2161_; 
lean_dec(v___x_2158_);
lean_dec(v___x_2153_);
lean_dec(v___x_2147_);
v___x_2160_ = lean_box(0);
v___x_2161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2161_, 0, v___x_2160_);
lean_ctor_set(v___x_2161_, 1, v_a_2141_);
return v___x_2161_;
}
else
{
lean_object* v___x_2162_; lean_object* v___x_2163_; uint8_t v___x_2164_; 
v___x_2162_ = l_Lean_Syntax_getArg(v___x_2158_, v___x_2146_);
v___x_2163_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3));
v___x_2164_ = l_Lean_Syntax_matchesIdent(v___x_2162_, v___x_2163_);
lean_dec(v___x_2162_);
if (v___x_2164_ == 0)
{
lean_object* v___x_2165_; lean_object* v___x_2166_; 
lean_dec(v___x_2158_);
lean_dec(v___x_2153_);
lean_dec(v___x_2147_);
v___x_2165_ = lean_box(0);
v___x_2166_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2166_, 0, v___x_2165_);
lean_ctor_set(v___x_2166_, 1, v_a_2141_);
return v___x_2166_;
}
else
{
lean_object* v___x_2167_; uint8_t v___x_2168_; 
v___x_2167_ = l_Lean_Syntax_getArg(v___x_2158_, v___x_2152_);
lean_dec(v___x_2158_);
lean_inc(v___x_2167_);
v___x_2168_ = l_Lean_Syntax_matchesNull(v___x_2167_, v___x_2152_);
if (v___x_2168_ == 0)
{
lean_object* v___x_2169_; lean_object* v___x_2170_; 
lean_dec(v___x_2167_);
lean_dec(v___x_2153_);
lean_dec(v___x_2147_);
v___x_2169_ = lean_box(0);
v___x_2170_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2170_, 0, v___x_2169_);
lean_ctor_set(v___x_2170_, 1, v_a_2141_);
return v___x_2170_;
}
else
{
lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v_ref_2175_; uint8_t v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; 
v___x_2171_ = l_Lean_Syntax_getArg(v___x_2167_, v___x_2146_);
lean_dec(v___x_2167_);
v___x_2172_ = l_Lean_Syntax_getArg(v___x_2153_, v___x_2152_);
v___x_2173_ = lean_unsigned_to_nat(2u);
v___x_2174_ = l_Lean_Syntax_getArg(v___x_2153_, v___x_2173_);
lean_dec(v___x_2153_);
v_ref_2175_ = l_Lean_replaceRef(v___x_2147_, v_a_2140_);
lean_dec(v___x_2147_);
v___x_2176_ = 0;
v___x_2177_ = l_Lean_SourceInfo_fromRef(v_ref_2175_, v___x_2176_);
lean_dec(v_ref_2175_);
v___x_2178_ = ((lean_object*)(lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__1));
v___x_2179_ = ((lean_object*)(lp_mathlib_DistribMulActionHomIdLocal_u227a___closed__2));
lean_inc_n(v___x_2177_, 2);
v___x_2180_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2180_, 0, v___x_2177_);
lean_ctor_set(v___x_2180_, 1, v___x_2179_);
v___x_2181_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_2182_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2182_, 0, v___x_2177_);
lean_ctor_set(v___x_2182_, 1, v___x_2181_);
v___x_2183_ = l_Lean_Syntax_node5(v___x_2177_, v___x_2178_, v___x_2172_, v___x_2180_, v___x_2171_, v___x_2182_, v___x_2174_);
v___x_2184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2184_, 0, v___x_2183_);
lean_ctor_set(v___x_2184_, 1, v_a_2141_);
return v___x_2184_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__2___boxed(lean_object* v_x_2185_, lean_object* v_a_2186_, lean_object* v_a_2187_){
_start:
{
lean_object* v_res_2188_; 
v_res_2188_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__DistribMulActionHom__2(v_x_2185_, v_a_2186_, v_a_2187_);
lean_dec(v_a_2186_);
return v_res_2188_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1(void){
_start:
{
lean_object* v___x_2214_; lean_object* v___x_2215_; 
v___x_2214_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__0));
v___x_2215_ = l_String_toRawSubstring_x27(v___x_2214_);
return v___x_2215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1(lean_object* v_x_2229_, lean_object* v_a_2230_, lean_object* v_a_2231_){
_start:
{
lean_object* v___x_2232_; uint8_t v___x_2233_; 
v___x_2232_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__1));
lean_inc(v_x_2229_);
v___x_2233_ = l_Lean_Syntax_isOfKind(v_x_2229_, v___x_2232_);
if (v___x_2233_ == 0)
{
lean_object* v___x_2234_; lean_object* v___x_2235_; 
lean_dec(v_x_2229_);
v___x_2234_ = lean_box(1);
v___x_2235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2235_, 0, v___x_2234_);
lean_ctor_set(v___x_2235_, 1, v_a_2231_);
return v___x_2235_;
}
else
{
lean_object* v_quotContext_2236_; lean_object* v_currMacroScope_2237_; lean_object* v_ref_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; uint8_t v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; 
v_quotContext_2236_ = lean_ctor_get(v_a_2230_, 1);
v_currMacroScope_2237_ = lean_ctor_get(v_a_2230_, 2);
v_ref_2238_ = lean_ctor_get(v_a_2230_, 5);
v___x_2239_ = lean_unsigned_to_nat(0u);
v___x_2240_ = l_Lean_Syntax_getArg(v_x_2229_, v___x_2239_);
v___x_2241_ = lean_unsigned_to_nat(2u);
v___x_2242_ = l_Lean_Syntax_getArg(v_x_2229_, v___x_2241_);
v___x_2243_ = lean_unsigned_to_nat(4u);
v___x_2244_ = l_Lean_Syntax_getArg(v_x_2229_, v___x_2243_);
lean_dec(v_x_2229_);
v___x_2245_ = 0;
v___x_2246_ = l_Lean_SourceInfo_fromRef(v_ref_2238_, v___x_2245_);
v___x_2247_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_2248_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1);
v___x_2249_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2));
lean_inc(v_currMacroScope_2237_);
lean_inc(v_quotContext_2236_);
v___x_2250_ = l_Lean_addMacroScope(v_quotContext_2236_, v___x_2249_, v_currMacroScope_2237_);
v___x_2251_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_2246_, 2);
v___x_2252_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2252_, 0, v___x_2246_);
lean_ctor_set(v___x_2252_, 1, v___x_2248_);
lean_ctor_set(v___x_2252_, 2, v___x_2250_);
lean_ctor_set(v___x_2252_, 3, v___x_2251_);
v___x_2253_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_2254_ = l_Lean_Syntax_node3(v___x_2246_, v___x_2253_, v___x_2242_, v___x_2240_, v___x_2244_);
v___x_2255_ = l_Lean_Syntax_node2(v___x_2246_, v___x_2247_, v___x_2252_, v___x_2254_);
v___x_2256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2256_, 0, v___x_2255_);
lean_ctor_set(v___x_2256_, 1, v_a_2231_);
return v___x_2256_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___boxed(lean_object* v_x_2257_, lean_object* v_a_2258_, lean_object* v_a_2259_){
_start:
{
lean_object* v_res_2260_; 
v_res_2260_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1(v_x_2257_, v_a_2258_, v_a_2259_);
lean_dec_ref(v_a_2258_);
return v_res_2260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__1(lean_object* v_x_2261_, lean_object* v_a_2262_, lean_object* v_a_2263_){
_start:
{
lean_object* v___x_2264_; uint8_t v___x_2265_; 
v___x_2264_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_2261_);
v___x_2265_ = l_Lean_Syntax_isOfKind(v_x_2261_, v___x_2264_);
if (v___x_2265_ == 0)
{
lean_object* v___x_2266_; lean_object* v___x_2267_; 
lean_dec(v_x_2261_);
v___x_2266_ = lean_box(0);
v___x_2267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2267_, 0, v___x_2266_);
lean_ctor_set(v___x_2267_, 1, v_a_2263_);
return v___x_2267_;
}
else
{
lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; uint8_t v___x_2271_; 
v___x_2268_ = lean_unsigned_to_nat(0u);
v___x_2269_ = l_Lean_Syntax_getArg(v_x_2261_, v___x_2268_);
v___x_2270_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_2269_);
v___x_2271_ = l_Lean_Syntax_isOfKind(v___x_2269_, v___x_2270_);
if (v___x_2271_ == 0)
{
lean_object* v___x_2272_; lean_object* v___x_2273_; 
lean_dec(v___x_2269_);
lean_dec(v_x_2261_);
v___x_2272_ = lean_box(0);
v___x_2273_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2273_, 0, v___x_2272_);
lean_ctor_set(v___x_2273_, 1, v_a_2263_);
return v___x_2273_;
}
else
{
lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; uint8_t v___x_2277_; 
v___x_2274_ = lean_unsigned_to_nat(1u);
v___x_2275_ = l_Lean_Syntax_getArg(v_x_2261_, v___x_2274_);
lean_dec(v_x_2261_);
v___x_2276_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_2275_);
v___x_2277_ = l_Lean_Syntax_matchesNull(v___x_2275_, v___x_2276_);
if (v___x_2277_ == 0)
{
lean_object* v___x_2278_; lean_object* v___x_2279_; 
lean_dec(v___x_2275_);
lean_dec(v___x_2269_);
v___x_2278_ = lean_box(0);
v___x_2279_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2279_, 0, v___x_2278_);
lean_ctor_set(v___x_2279_, 1, v_a_2263_);
return v___x_2279_;
}
else
{
lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v_ref_2284_; uint8_t v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; 
v___x_2280_ = l_Lean_Syntax_getArg(v___x_2275_, v___x_2268_);
v___x_2281_ = l_Lean_Syntax_getArg(v___x_2275_, v___x_2274_);
v___x_2282_ = lean_unsigned_to_nat(2u);
v___x_2283_ = l_Lean_Syntax_getArg(v___x_2275_, v___x_2282_);
lean_dec(v___x_2275_);
v_ref_2284_ = l_Lean_replaceRef(v___x_2269_, v_a_2262_);
lean_dec(v___x_2269_);
v___x_2285_ = 0;
v___x_2286_ = l_Lean_SourceInfo_fromRef(v_ref_2284_, v___x_2285_);
lean_dec(v_ref_2284_);
v___x_2287_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__1));
v___x_2288_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHomLocal_u227a___closed__2));
lean_inc_n(v___x_2286_, 2);
v___x_2289_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2289_, 0, v___x_2286_);
lean_ctor_set(v___x_2289_, 1, v___x_2288_);
v___x_2290_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_2291_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2291_, 0, v___x_2286_);
lean_ctor_set(v___x_2291_, 1, v___x_2290_);
v___x_2292_ = l_Lean_Syntax_node5(v___x_2286_, v___x_2287_, v___x_2281_, v___x_2289_, v___x_2280_, v___x_2291_, v___x_2283_);
v___x_2293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2293_, 0, v___x_2292_);
lean_ctor_set(v___x_2293_, 1, v_a_2263_);
return v___x_2293_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__1___boxed(lean_object* v_x_2294_, lean_object* v_a_2295_, lean_object* v_a_2296_){
_start:
{
lean_object* v_res_2297_; 
v_res_2297_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__1(v_x_2294_, v_a_2295_, v_a_2296_);
lean_dec(v_a_2295_);
return v_res_2297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomIdLocal_u227a__1(lean_object* v_x_2322_, lean_object* v_a_2323_, lean_object* v_a_2324_){
_start:
{
lean_object* v___x_2325_; uint8_t v___x_2326_; 
v___x_2325_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__1));
lean_inc(v_x_2322_);
v___x_2326_ = l_Lean_Syntax_isOfKind(v_x_2322_, v___x_2325_);
if (v___x_2326_ == 0)
{
lean_object* v___x_2327_; lean_object* v___x_2328_; 
lean_dec(v_x_2322_);
v___x_2327_ = lean_box(1);
v___x_2328_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2328_, 0, v___x_2327_);
lean_ctor_set(v___x_2328_, 1, v_a_2324_);
return v___x_2328_;
}
else
{
lean_object* v_quotContext_2329_; lean_object* v_currMacroScope_2330_; lean_object* v_ref_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; uint8_t v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; 
v_quotContext_2329_ = lean_ctor_get(v_a_2323_, 1);
v_currMacroScope_2330_ = lean_ctor_get(v_a_2323_, 2);
v_ref_2331_ = lean_ctor_get(v_a_2323_, 5);
v___x_2332_ = lean_unsigned_to_nat(0u);
v___x_2333_ = l_Lean_Syntax_getArg(v_x_2322_, v___x_2332_);
v___x_2334_ = lean_unsigned_to_nat(2u);
v___x_2335_ = l_Lean_Syntax_getArg(v_x_2322_, v___x_2334_);
v___x_2336_ = lean_unsigned_to_nat(4u);
v___x_2337_ = l_Lean_Syntax_getArg(v_x_2322_, v___x_2336_);
lean_dec(v_x_2322_);
v___x_2338_ = 0;
v___x_2339_ = l_Lean_SourceInfo_fromRef(v_ref_2331_, v___x_2338_);
v___x_2340_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_2341_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__1);
v___x_2342_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__2));
lean_inc_n(v_currMacroScope_2330_, 3);
lean_inc_n(v_quotContext_2329_, 3);
v___x_2343_ = l_Lean_addMacroScope(v_quotContext_2329_, v___x_2342_, v_currMacroScope_2330_);
v___x_2344_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_2339_, 11);
v___x_2345_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2345_, 0, v___x_2339_);
lean_ctor_set(v___x_2345_, 1, v___x_2341_);
lean_ctor_set(v___x_2345_, 2, v___x_2343_);
lean_ctor_set(v___x_2345_, 3, v___x_2344_);
v___x_2346_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_2347_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1));
v___x_2348_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3));
v___x_2349_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4));
v___x_2350_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2350_, 0, v___x_2339_);
lean_ctor_set(v___x_2350_, 1, v___x_2349_);
v___x_2351_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6));
v___x_2352_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8);
v___x_2353_ = lean_box(0);
v___x_2354_ = l_Lean_addMacroScope(v_quotContext_2329_, v___x_2353_, v_currMacroScope_2330_);
v___x_2355_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10));
v___x_2356_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2356_, 0, v___x_2339_);
lean_ctor_set(v___x_2356_, 1, v___x_2352_);
lean_ctor_set(v___x_2356_, 2, v___x_2354_);
lean_ctor_set(v___x_2356_, 3, v___x_2355_);
v___x_2357_ = l_Lean_Syntax_node1(v___x_2339_, v___x_2351_, v___x_2356_);
v___x_2358_ = l_Lean_Syntax_node2(v___x_2339_, v___x_2348_, v___x_2350_, v___x_2357_);
v___x_2359_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1);
v___x_2360_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3));
v___x_2361_ = l_Lean_addMacroScope(v_quotContext_2329_, v___x_2360_, v_currMacroScope_2330_);
v___x_2362_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__5));
v___x_2363_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2363_, 0, v___x_2339_);
lean_ctor_set(v___x_2363_, 1, v___x_2359_);
lean_ctor_set(v___x_2363_, 2, v___x_2361_);
lean_ctor_set(v___x_2363_, 3, v___x_2362_);
v___x_2364_ = l_Lean_Syntax_node1(v___x_2339_, v___x_2346_, v___x_2335_);
v___x_2365_ = l_Lean_Syntax_node2(v___x_2339_, v___x_2340_, v___x_2363_, v___x_2364_);
v___x_2366_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19));
v___x_2367_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2367_, 0, v___x_2339_);
lean_ctor_set(v___x_2367_, 1, v___x_2366_);
v___x_2368_ = l_Lean_Syntax_node3(v___x_2339_, v___x_2347_, v___x_2358_, v___x_2365_, v___x_2367_);
v___x_2369_ = l_Lean_Syntax_node3(v___x_2339_, v___x_2346_, v___x_2368_, v___x_2333_, v___x_2337_);
v___x_2370_ = l_Lean_Syntax_node2(v___x_2339_, v___x_2340_, v___x_2345_, v___x_2369_);
v___x_2371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2371_, 0, v___x_2370_);
lean_ctor_set(v___x_2371_, 1, v_a_2324_);
return v___x_2371_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomIdLocal_u227a__1___boxed(lean_object* v_x_2372_, lean_object* v_a_2373_, lean_object* v_a_2374_){
_start:
{
lean_object* v_res_2375_; 
v_res_2375_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulDistribMulActionHomIdLocal_u227a__1(v_x_2372_, v_a_2373_, v_a_2374_);
lean_dec_ref(v_a_2373_);
return v_res_2375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__2(lean_object* v_x_2376_, lean_object* v_a_2377_, lean_object* v_a_2378_){
_start:
{
lean_object* v___x_2379_; uint8_t v___x_2380_; 
v___x_2379_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_2376_);
v___x_2380_ = l_Lean_Syntax_isOfKind(v_x_2376_, v___x_2379_);
if (v___x_2380_ == 0)
{
lean_object* v___x_2381_; lean_object* v___x_2382_; 
lean_dec(v_x_2376_);
v___x_2381_ = lean_box(0);
v___x_2382_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2382_, 0, v___x_2381_);
lean_ctor_set(v___x_2382_, 1, v_a_2378_);
return v___x_2382_;
}
else
{
lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; uint8_t v___x_2386_; 
v___x_2383_ = lean_unsigned_to_nat(0u);
v___x_2384_ = l_Lean_Syntax_getArg(v_x_2376_, v___x_2383_);
v___x_2385_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_2384_);
v___x_2386_ = l_Lean_Syntax_isOfKind(v___x_2384_, v___x_2385_);
if (v___x_2386_ == 0)
{
lean_object* v___x_2387_; lean_object* v___x_2388_; 
lean_dec(v___x_2384_);
lean_dec(v_x_2376_);
v___x_2387_ = lean_box(0);
v___x_2388_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2388_, 0, v___x_2387_);
lean_ctor_set(v___x_2388_, 1, v_a_2378_);
return v___x_2388_;
}
else
{
lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; uint8_t v___x_2392_; 
v___x_2389_ = lean_unsigned_to_nat(1u);
v___x_2390_ = l_Lean_Syntax_getArg(v_x_2376_, v___x_2389_);
lean_dec(v_x_2376_);
v___x_2391_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_2390_);
v___x_2392_ = l_Lean_Syntax_matchesNull(v___x_2390_, v___x_2391_);
if (v___x_2392_ == 0)
{
lean_object* v___x_2393_; lean_object* v___x_2394_; 
lean_dec(v___x_2390_);
lean_dec(v___x_2384_);
v___x_2393_ = lean_box(0);
v___x_2394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2394_, 0, v___x_2393_);
lean_ctor_set(v___x_2394_, 1, v_a_2378_);
return v___x_2394_;
}
else
{
lean_object* v___x_2395_; uint8_t v___x_2396_; 
v___x_2395_ = l_Lean_Syntax_getArg(v___x_2390_, v___x_2383_);
lean_inc(v___x_2395_);
v___x_2396_ = l_Lean_Syntax_isOfKind(v___x_2395_, v___x_2379_);
if (v___x_2396_ == 0)
{
lean_object* v___x_2397_; lean_object* v___x_2398_; 
lean_dec(v___x_2395_);
lean_dec(v___x_2390_);
lean_dec(v___x_2384_);
v___x_2397_ = lean_box(0);
v___x_2398_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2398_, 0, v___x_2397_);
lean_ctor_set(v___x_2398_, 1, v_a_2378_);
return v___x_2398_;
}
else
{
lean_object* v___x_2399_; lean_object* v___x_2400_; uint8_t v___x_2401_; 
v___x_2399_ = l_Lean_Syntax_getArg(v___x_2395_, v___x_2383_);
v___x_2400_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3));
v___x_2401_ = l_Lean_Syntax_matchesIdent(v___x_2399_, v___x_2400_);
lean_dec(v___x_2399_);
if (v___x_2401_ == 0)
{
lean_object* v___x_2402_; lean_object* v___x_2403_; 
lean_dec(v___x_2395_);
lean_dec(v___x_2390_);
lean_dec(v___x_2384_);
v___x_2402_ = lean_box(0);
v___x_2403_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2403_, 0, v___x_2402_);
lean_ctor_set(v___x_2403_, 1, v_a_2378_);
return v___x_2403_;
}
else
{
lean_object* v___x_2404_; uint8_t v___x_2405_; 
v___x_2404_ = l_Lean_Syntax_getArg(v___x_2395_, v___x_2389_);
lean_dec(v___x_2395_);
lean_inc(v___x_2404_);
v___x_2405_ = l_Lean_Syntax_matchesNull(v___x_2404_, v___x_2389_);
if (v___x_2405_ == 0)
{
lean_object* v___x_2406_; lean_object* v___x_2407_; 
lean_dec(v___x_2404_);
lean_dec(v___x_2390_);
lean_dec(v___x_2384_);
v___x_2406_ = lean_box(0);
v___x_2407_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2407_, 0, v___x_2406_);
lean_ctor_set(v___x_2407_, 1, v_a_2378_);
return v___x_2407_;
}
else
{
lean_object* v___x_2408_; lean_object* v___x_2409_; lean_object* v___x_2410_; lean_object* v___x_2411_; lean_object* v_ref_2412_; uint8_t v___x_2413_; lean_object* v___x_2414_; lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; 
v___x_2408_ = l_Lean_Syntax_getArg(v___x_2404_, v___x_2383_);
lean_dec(v___x_2404_);
v___x_2409_ = l_Lean_Syntax_getArg(v___x_2390_, v___x_2389_);
v___x_2410_ = lean_unsigned_to_nat(2u);
v___x_2411_ = l_Lean_Syntax_getArg(v___x_2390_, v___x_2410_);
lean_dec(v___x_2390_);
v_ref_2412_ = l_Lean_replaceRef(v___x_2384_, v_a_2377_);
lean_dec(v___x_2384_);
v___x_2413_ = 0;
v___x_2414_ = l_Lean_SourceInfo_fromRef(v_ref_2412_, v___x_2413_);
lean_dec(v_ref_2412_);
v___x_2415_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__1));
v___x_2416_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHomIdLocal_u227a___closed__2));
lean_inc_n(v___x_2414_, 2);
v___x_2417_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2417_, 0, v___x_2414_);
lean_ctor_set(v___x_2417_, 1, v___x_2416_);
v___x_2418_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_2419_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2419_, 0, v___x_2414_);
lean_ctor_set(v___x_2419_, 1, v___x_2418_);
v___x_2420_ = l_Lean_Syntax_node5(v___x_2414_, v___x_2415_, v___x_2409_, v___x_2417_, v___x_2408_, v___x_2419_, v___x_2411_);
v___x_2421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2421_, 0, v___x_2420_);
lean_ctor_set(v___x_2421_, 1, v_a_2378_);
return v___x_2421_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__2___boxed(lean_object* v_x_2422_, lean_object* v_a_2423_, lean_object* v_a_2424_){
_start:
{
lean_object* v_res_2425_; 
v_res_2425_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulDistribMulActionHom__2(v_x_2422_, v_a_2423_, v_a_2424_);
lean_dec(v_a_2423_);
return v_res_2425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike___lam__0(lean_object* v_m_2426_, lean_object* v___y_2427_){
_start:
{
lean_object* v___x_2428_; 
v___x_2428_ = lean_apply_1(v_m_2426_, v___y_2427_);
return v___x_2428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike(lean_object* v_M_2430_, lean_object* v_inst_2431_, lean_object* v_N_2432_, lean_object* v_inst_2433_, lean_object* v_00_u03c6_2434_, lean_object* v_A_2435_, lean_object* v_inst_2436_, lean_object* v_inst_2437_, lean_object* v_B_2438_, lean_object* v_inst_2439_, lean_object* v_inst_2440_){
_start:
{
lean_object* v___f_2441_; 
v___f_2441_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHom_instFunLike___closed__0));
return v___f_2441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instFunLike___boxed(lean_object* v_M_2442_, lean_object* v_inst_2443_, lean_object* v_N_2444_, lean_object* v_inst_2445_, lean_object* v_00_u03c6_2446_, lean_object* v_A_2447_, lean_object* v_inst_2448_, lean_object* v_inst_2449_, lean_object* v_B_2450_, lean_object* v_inst_2451_, lean_object* v_inst_2452_){
_start:
{
lean_object* v_res_2453_; 
v_res_2453_ = lp_mathlib_MulDistribMulActionHom_instFunLike(v_M_2442_, v_inst_2443_, v_N_2444_, v_inst_2445_, v_00_u03c6_2446_, v_A_2447_, v_inst_2448_, v_inst_2449_, v_B_2450_, v_inst_2451_, v_inst_2452_);
lean_dec(v_inst_2452_);
lean_dec_ref(v_inst_2451_);
lean_dec(v_inst_2449_);
lean_dec_ref(v_inst_2448_);
lean_dec(v_00_u03c6_2446_);
lean_dec_ref(v_inst_2445_);
lean_dec_ref(v_inst_2443_);
return v_res_2453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instFunLike(lean_object* v_M_2454_, lean_object* v_inst_2455_, lean_object* v_N_2456_, lean_object* v_inst_2457_, lean_object* v_00_u03c6_2458_, lean_object* v_A_2459_, lean_object* v_inst_2460_, lean_object* v_inst_2461_, lean_object* v_B_2462_, lean_object* v_inst_2463_, lean_object* v_inst_2464_){
_start:
{
lean_object* v___f_2465_; 
v___f_2465_ = ((lean_object*)(lp_mathlib_MulDistribMulActionHom_instFunLike___closed__0));
return v___f_2465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instFunLike___boxed(lean_object* v_M_2466_, lean_object* v_inst_2467_, lean_object* v_N_2468_, lean_object* v_inst_2469_, lean_object* v_00_u03c6_2470_, lean_object* v_A_2471_, lean_object* v_inst_2472_, lean_object* v_inst_2473_, lean_object* v_B_2474_, lean_object* v_inst_2475_, lean_object* v_inst_2476_){
_start:
{
lean_object* v_res_2477_; 
v_res_2477_ = lp_mathlib_DistribMulActionHom_instFunLike(v_M_2466_, v_inst_2467_, v_N_2468_, v_inst_2469_, v_00_u03c6_2470_, v_A_2471_, v_inst_2472_, v_inst_2473_, v_B_2474_, v_inst_2475_, v_inst_2476_);
lean_dec(v_inst_2476_);
lean_dec_ref(v_inst_2475_);
lean_dec(v_inst_2473_);
lean_dec_ref(v_inst_2472_);
lean_dec(v_00_u03c6_2470_);
lean_dec_ref(v_inst_2469_);
lean_dec_ref(v_inst_2467_);
return v_res_2477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom___redArg(lean_object* v_inst_2478_, lean_object* v_f_2479_){
_start:
{
lean_object* v___x_2480_; 
v___x_2480_ = lean_apply_1(v_inst_2478_, v_f_2479_);
return v___x_2480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom(lean_object* v_M_2481_, lean_object* v_inst_2482_, lean_object* v_N_2483_, lean_object* v_inst_2484_, lean_object* v_00_u03c6_2485_, lean_object* v_A_2486_, lean_object* v_inst_2487_, lean_object* v_inst_2488_, lean_object* v_B_2489_, lean_object* v_inst_2490_, lean_object* v_inst_2491_, lean_object* v_F_2492_, lean_object* v_inst_2493_, lean_object* v_inst_2494_, lean_object* v_f_2495_){
_start:
{
lean_object* v___x_2496_; 
v___x_2496_ = lean_apply_1(v_inst_2493_, v_f_2495_);
return v___x_2496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom___boxed(lean_object* v_M_2497_, lean_object* v_inst_2498_, lean_object* v_N_2499_, lean_object* v_inst_2500_, lean_object* v_00_u03c6_2501_, lean_object* v_A_2502_, lean_object* v_inst_2503_, lean_object* v_inst_2504_, lean_object* v_B_2505_, lean_object* v_inst_2506_, lean_object* v_inst_2507_, lean_object* v_F_2508_, lean_object* v_inst_2509_, lean_object* v_inst_2510_, lean_object* v_f_2511_){
_start:
{
lean_object* v_res_2512_; 
v_res_2512_ = lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom(v_M_2497_, v_inst_2498_, v_N_2499_, v_inst_2500_, v_00_u03c6_2501_, v_A_2502_, v_inst_2503_, v_inst_2504_, v_B_2505_, v_inst_2506_, v_inst_2507_, v_F_2508_, v_inst_2509_, v_inst_2510_, v_f_2511_);
lean_dec(v_inst_2507_);
lean_dec_ref(v_inst_2506_);
lean_dec(v_inst_2504_);
lean_dec_ref(v_inst_2503_);
lean_dec(v_00_u03c6_2501_);
lean_dec_ref(v_inst_2500_);
lean_dec_ref(v_inst_2498_);
return v_res_2512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom___redArg(lean_object* v_inst_2513_, lean_object* v_f_2514_){
_start:
{
lean_object* v___x_2515_; 
v___x_2515_ = lean_apply_1(v_inst_2513_, v_f_2514_);
return v___x_2515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom(lean_object* v_M_2516_, lean_object* v_inst_2517_, lean_object* v_N_2518_, lean_object* v_inst_2519_, lean_object* v_00_u03c6_2520_, lean_object* v_A_2521_, lean_object* v_inst_2522_, lean_object* v_inst_2523_, lean_object* v_B_2524_, lean_object* v_inst_2525_, lean_object* v_inst_2526_, lean_object* v_F_2527_, lean_object* v_inst_2528_, lean_object* v_inst_2529_, lean_object* v_f_2530_){
_start:
{
lean_object* v___x_2531_; 
v___x_2531_ = lean_apply_1(v_inst_2528_, v_f_2530_);
return v___x_2531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom___boxed(lean_object* v_M_2532_, lean_object* v_inst_2533_, lean_object* v_N_2534_, lean_object* v_inst_2535_, lean_object* v_00_u03c6_2536_, lean_object* v_A_2537_, lean_object* v_inst_2538_, lean_object* v_inst_2539_, lean_object* v_B_2540_, lean_object* v_inst_2541_, lean_object* v_inst_2542_, lean_object* v_F_2543_, lean_object* v_inst_2544_, lean_object* v_inst_2545_, lean_object* v_f_2546_){
_start:
{
lean_object* v_res_2547_; 
v_res_2547_ = lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom(v_M_2532_, v_inst_2533_, v_N_2534_, v_inst_2535_, v_00_u03c6_2536_, v_A_2537_, v_inst_2538_, v_inst_2539_, v_B_2540_, v_inst_2541_, v_inst_2542_, v_F_2543_, v_inst_2544_, v_inst_2545_, v_f_2546_);
lean_dec(v_inst_2542_);
lean_dec_ref(v_inst_2541_);
lean_dec(v_inst_2539_);
lean_dec_ref(v_inst_2538_);
lean_dec(v_00_u03c6_2536_);
lean_dec_ref(v_inst_2535_);
lean_dec_ref(v_inst_2533_);
return v_res_2547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instCoeTCOfMulDistribMulActionSemiHomClassCoeMonoidHom___redArg(lean_object* v_inst_2548_, lean_object* v_inst_2549_, lean_object* v_00_u03c6_2550_, lean_object* v_inst_2551_, lean_object* v_inst_2552_, lean_object* v_inst_2553_, lean_object* v_inst_2554_, lean_object* v_inst_2555_){
_start:
{
lean_object* v___x_2556_; 
v___x_2556_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom___boxed), 15, 14);
lean_closure_set(v___x_2556_, 0, lean_box(0));
lean_closure_set(v___x_2556_, 1, v_inst_2548_);
lean_closure_set(v___x_2556_, 2, lean_box(0));
lean_closure_set(v___x_2556_, 3, v_inst_2549_);
lean_closure_set(v___x_2556_, 4, v_00_u03c6_2550_);
lean_closure_set(v___x_2556_, 5, lean_box(0));
lean_closure_set(v___x_2556_, 6, v_inst_2551_);
lean_closure_set(v___x_2556_, 7, v_inst_2552_);
lean_closure_set(v___x_2556_, 8, lean_box(0));
lean_closure_set(v___x_2556_, 9, v_inst_2553_);
lean_closure_set(v___x_2556_, 10, v_inst_2554_);
lean_closure_set(v___x_2556_, 11, lean_box(0));
lean_closure_set(v___x_2556_, 12, v_inst_2555_);
lean_closure_set(v___x_2556_, 13, lean_box(0));
return v___x_2556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instCoeTCOfMulDistribMulActionSemiHomClassCoeMonoidHom(lean_object* v_M_2557_, lean_object* v_inst_2558_, lean_object* v_N_2559_, lean_object* v_inst_2560_, lean_object* v_00_u03c6_2561_, lean_object* v_A_2562_, lean_object* v_inst_2563_, lean_object* v_inst_2564_, lean_object* v_B_2565_, lean_object* v_inst_2566_, lean_object* v_inst_2567_, lean_object* v_F_2568_, lean_object* v_inst_2569_, lean_object* v_inst_2570_){
_start:
{
lean_object* v___x_2571_; 
v___x_2571_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulActionSemiHomClass_toMulDistribMulActionHom___boxed), 15, 14);
lean_closure_set(v___x_2571_, 0, lean_box(0));
lean_closure_set(v___x_2571_, 1, v_inst_2558_);
lean_closure_set(v___x_2571_, 2, lean_box(0));
lean_closure_set(v___x_2571_, 3, v_inst_2560_);
lean_closure_set(v___x_2571_, 4, v_00_u03c6_2561_);
lean_closure_set(v___x_2571_, 5, lean_box(0));
lean_closure_set(v___x_2571_, 6, v_inst_2563_);
lean_closure_set(v___x_2571_, 7, v_inst_2564_);
lean_closure_set(v___x_2571_, 8, lean_box(0));
lean_closure_set(v___x_2571_, 9, v_inst_2566_);
lean_closure_set(v___x_2571_, 10, v_inst_2567_);
lean_closure_set(v___x_2571_, 11, lean_box(0));
lean_closure_set(v___x_2571_, 12, v_inst_2569_);
lean_closure_set(v___x_2571_, 13, lean_box(0));
return v___x_2571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instCoeTCOfAddDistribAddActionSemiHomClassCoeAddMonoidHom___redArg(lean_object* v_inst_2572_, lean_object* v_inst_2573_, lean_object* v_00_u03c6_2574_, lean_object* v_inst_2575_, lean_object* v_inst_2576_, lean_object* v_inst_2577_, lean_object* v_inst_2578_, lean_object* v_inst_2579_){
_start:
{
lean_object* v___x_2580_; 
v___x_2580_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom___boxed), 15, 14);
lean_closure_set(v___x_2580_, 0, lean_box(0));
lean_closure_set(v___x_2580_, 1, v_inst_2572_);
lean_closure_set(v___x_2580_, 2, lean_box(0));
lean_closure_set(v___x_2580_, 3, v_inst_2573_);
lean_closure_set(v___x_2580_, 4, v_00_u03c6_2574_);
lean_closure_set(v___x_2580_, 5, lean_box(0));
lean_closure_set(v___x_2580_, 6, v_inst_2575_);
lean_closure_set(v___x_2580_, 7, v_inst_2576_);
lean_closure_set(v___x_2580_, 8, lean_box(0));
lean_closure_set(v___x_2580_, 9, v_inst_2577_);
lean_closure_set(v___x_2580_, 10, v_inst_2578_);
lean_closure_set(v___x_2580_, 11, lean_box(0));
lean_closure_set(v___x_2580_, 12, v_inst_2579_);
lean_closure_set(v___x_2580_, 13, lean_box(0));
return v___x_2580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instCoeTCOfAddDistribAddActionSemiHomClassCoeAddMonoidHom(lean_object* v_M_2581_, lean_object* v_inst_2582_, lean_object* v_N_2583_, lean_object* v_inst_2584_, lean_object* v_00_u03c6_2585_, lean_object* v_A_2586_, lean_object* v_inst_2587_, lean_object* v_inst_2588_, lean_object* v_B_2589_, lean_object* v_inst_2590_, lean_object* v_inst_2591_, lean_object* v_F_2592_, lean_object* v_inst_2593_, lean_object* v_inst_2594_){
_start:
{
lean_object* v___x_2595_; 
v___x_2595_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulActionSemiHomClass_toDistribMulActionHom___boxed), 15, 14);
lean_closure_set(v___x_2595_, 0, lean_box(0));
lean_closure_set(v___x_2595_, 1, v_inst_2582_);
lean_closure_set(v___x_2595_, 2, lean_box(0));
lean_closure_set(v___x_2595_, 3, v_inst_2584_);
lean_closure_set(v___x_2595_, 4, v_00_u03c6_2585_);
lean_closure_set(v___x_2595_, 5, lean_box(0));
lean_closure_set(v___x_2595_, 6, v_inst_2587_);
lean_closure_set(v___x_2595_, 7, v_inst_2588_);
lean_closure_set(v___x_2595_, 8, lean_box(0));
lean_closure_set(v___x_2595_, 9, v_inst_2590_);
lean_closure_set(v___x_2595_, 10, v_inst_2591_);
lean_closure_set(v___x_2595_, 11, lean_box(0));
lean_closure_set(v___x_2595_, 12, v_inst_2593_);
lean_closure_set(v___x_2595_, 13, lean_box(0));
return v___x_2595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toDistribMulActionHom___redArg(lean_object* v_inst_2596_, lean_object* v_c_2597_){
_start:
{
lean_object* v___f_2598_; 
v___f_2598_ = lean_alloc_closure((void*)(lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2598_, 0, v_inst_2596_);
lean_closure_set(v___f_2598_, 1, v_c_2597_);
return v___f_2598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toDistribMulActionHom(lean_object* v_M_2599_, lean_object* v_N_2600_, lean_object* v_A_2601_, lean_object* v_inst_2602_, lean_object* v_inst_2603_, lean_object* v_inst_2604_, lean_object* v_inst_2605_, lean_object* v_inst_2606_, lean_object* v_c_2607_){
_start:
{
lean_object* v___f_2608_; 
v___f_2608_ = lean_alloc_closure((void*)(lp_mathlib_SMulCommClass_toMulActionHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2608_, 0, v_inst_2604_);
lean_closure_set(v___f_2608_, 1, v_c_2607_);
return v___f_2608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMulCommClass_toDistribMulActionHom___boxed(lean_object* v_M_2609_, lean_object* v_N_2610_, lean_object* v_A_2611_, lean_object* v_inst_2612_, lean_object* v_inst_2613_, lean_object* v_inst_2614_, lean_object* v_inst_2615_, lean_object* v_inst_2616_, lean_object* v_c_2617_){
_start:
{
lean_object* v_res_2618_; 
v_res_2618_ = lp_mathlib_SMulCommClass_toDistribMulActionHom(v_M_2609_, v_N_2610_, v_A_2611_, v_inst_2612_, v_inst_2613_, v_inst_2614_, v_inst_2615_, v_inst_2616_, v_c_2617_);
lean_dec(v_inst_2615_);
lean_dec_ref(v_inst_2613_);
lean_dec_ref(v_inst_2612_);
return v_res_2618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_id(lean_object* v_M_2619_, lean_object* v_inst_2620_, lean_object* v_A_2621_, lean_object* v_inst_2622_, lean_object* v_inst_2623_){
_start:
{
lean_object* v___f_2624_; 
v___f_2624_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_2624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_id___boxed(lean_object* v_M_2625_, lean_object* v_inst_2626_, lean_object* v_A_2627_, lean_object* v_inst_2628_, lean_object* v_inst_2629_){
_start:
{
lean_object* v_res_2630_; 
v_res_2630_ = lp_mathlib_MulDistribMulActionHom_id(v_M_2625_, v_inst_2626_, v_A_2627_, v_inst_2628_, v_inst_2629_);
lean_dec(v_inst_2629_);
lean_dec_ref(v_inst_2628_);
lean_dec_ref(v_inst_2626_);
return v_res_2630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_id(lean_object* v_M_2631_, lean_object* v_inst_2632_, lean_object* v_A_2633_, lean_object* v_inst_2634_, lean_object* v_inst_2635_){
_start:
{
lean_object* v___f_2636_; 
v___f_2636_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_2636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_id___boxed(lean_object* v_M_2637_, lean_object* v_inst_2638_, lean_object* v_A_2639_, lean_object* v_inst_2640_, lean_object* v_inst_2641_){
_start:
{
lean_object* v_res_2642_; 
v_res_2642_ = lp_mathlib_DistribMulActionHom_id(v_M_2637_, v_inst_2638_, v_A_2639_, v_inst_2640_, v_inst_2641_);
lean_dec(v_inst_2641_);
lean_dec_ref(v_inst_2640_);
lean_dec_ref(v_inst_2638_);
return v_res_2642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0(lean_object* v_toZero_2643_, lean_object* v_x_2644_){
_start:
{
lean_inc(v_toZero_2643_);
return v_toZero_2643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0___boxed(lean_object* v_toZero_2645_, lean_object* v_x_2646_){
_start:
{
lean_object* v_res_2647_; 
v_res_2647_ = lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0(v_toZero_2645_, v_x_2646_);
lean_dec(v_x_2646_);
lean_dec(v_toZero_2645_);
return v_res_2647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg(lean_object* v_inst_2648_){
_start:
{
lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v_toZero_2651_; lean_object* v___f_2652_; 
v___x_2649_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_2648_);
v___x_2650_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_2649_);
v_toZero_2651_ = lean_ctor_get(v___x_2650_, 0);
lean_inc(v_toZero_2651_);
lean_dec_ref(v___x_2650_);
v___f_2652_ = lean_alloc_closure((void*)(lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_2652_, 0, v_toZero_2651_);
return v___f_2652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___redArg___boxed(lean_object* v_inst_2653_){
_start:
{
lean_object* v_res_2654_; 
v_res_2654_ = lp_mathlib_DistriMulActionHom_instZero___redArg(v_inst_2653_);
lean_dec_ref(v_inst_2653_);
return v_res_2654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero(lean_object* v_M_2655_, lean_object* v_inst_2656_, lean_object* v_N_2657_, lean_object* v_inst_2658_, lean_object* v_00_u03c6_2659_, lean_object* v_A_2660_, lean_object* v_inst_2661_, lean_object* v_inst_2662_, lean_object* v_B_2663_, lean_object* v_inst_2664_, lean_object* v_inst_2665_){
_start:
{
lean_object* v___x_2666_; 
v___x_2666_ = lp_mathlib_DistriMulActionHom_instZero___redArg(v_inst_2664_);
return v___x_2666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistriMulActionHom_instZero___boxed(lean_object* v_M_2667_, lean_object* v_inst_2668_, lean_object* v_N_2669_, lean_object* v_inst_2670_, lean_object* v_00_u03c6_2671_, lean_object* v_A_2672_, lean_object* v_inst_2673_, lean_object* v_inst_2674_, lean_object* v_B_2675_, lean_object* v_inst_2676_, lean_object* v_inst_2677_){
_start:
{
lean_object* v_res_2678_; 
v_res_2678_ = lp_mathlib_DistriMulActionHom_instZero(v_M_2667_, v_inst_2668_, v_N_2669_, v_inst_2670_, v_00_u03c6_2671_, v_A_2672_, v_inst_2673_, v_inst_2674_, v_B_2675_, v_inst_2676_, v_inst_2677_);
lean_dec(v_inst_2677_);
lean_dec_ref(v_inst_2676_);
lean_dec(v_inst_2674_);
lean_dec_ref(v_inst_2673_);
lean_dec(v_00_u03c6_2671_);
lean_dec_ref(v_inst_2670_);
lean_dec_ref(v_inst_2668_);
return v_res_2678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instOneId(lean_object* v_M_2679_, lean_object* v_inst_2680_, lean_object* v_A_2681_, lean_object* v_inst_2682_, lean_object* v_inst_2683_){
_start:
{
lean_object* v___f_2684_; 
v___f_2684_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_2684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instOneId___boxed(lean_object* v_M_2685_, lean_object* v_inst_2686_, lean_object* v_A_2687_, lean_object* v_inst_2688_, lean_object* v_inst_2689_){
_start:
{
lean_object* v_res_2690_; 
v_res_2690_ = lp_mathlib_MulDistribMulActionHom_instOneId(v_M_2685_, v_inst_2686_, v_A_2687_, v_inst_2688_, v_inst_2689_);
lean_dec(v_inst_2689_);
lean_dec_ref(v_inst_2688_);
lean_dec_ref(v_inst_2686_);
return v_res_2690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instZeroId(lean_object* v_M_2691_, lean_object* v_inst_2692_, lean_object* v_A_2693_, lean_object* v_inst_2694_, lean_object* v_inst_2695_){
_start:
{
lean_object* v___f_2696_; 
v___f_2696_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_2696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_instZeroId___boxed(lean_object* v_M_2697_, lean_object* v_inst_2698_, lean_object* v_A_2699_, lean_object* v_inst_2700_, lean_object* v_inst_2701_){
_start:
{
lean_object* v_res_2702_; 
v_res_2702_ = lp_mathlib_DistribMulActionHom_instZeroId(v_M_2697_, v_inst_2698_, v_A_2699_, v_inst_2700_, v_inst_2701_);
lean_dec(v_inst_2701_);
lean_dec_ref(v_inst_2700_);
lean_dec_ref(v_inst_2698_);
return v_res_2702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___redArg(lean_object* v_inst_2703_){
_start:
{
lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v_toZero_2706_; lean_object* v___f_2707_; 
v___x_2704_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_2703_);
v___x_2705_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_2704_);
v_toZero_2706_ = lean_ctor_get(v___x_2705_, 0);
lean_inc(v_toZero_2706_);
lean_dec_ref(v___x_2705_);
v___f_2707_ = lean_alloc_closure((void*)(lp_mathlib_DistriMulActionHom_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_2707_, 0, v_toZero_2706_);
return v___f_2707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___redArg___boxed(lean_object* v_inst_2708_){
_start:
{
lean_object* v_res_2709_; 
v_res_2709_ = lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___redArg(v_inst_2708_);
lean_dec_ref(v_inst_2708_);
return v_res_2709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom(lean_object* v_M_2710_, lean_object* v_inst_2711_, lean_object* v_N_2712_, lean_object* v_inst_2713_, lean_object* v_00_u03c6_2714_, lean_object* v_A_2715_, lean_object* v_inst_2716_, lean_object* v_inst_2717_, lean_object* v_B_2718_, lean_object* v_inst_2719_, lean_object* v_inst_2720_){
_start:
{
lean_object* v___x_2721_; 
v___x_2721_ = lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___redArg(v_inst_2719_);
return v___x_2721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom___boxed(lean_object* v_M_2722_, lean_object* v_inst_2723_, lean_object* v_N_2724_, lean_object* v_inst_2725_, lean_object* v_00_u03c6_2726_, lean_object* v_A_2727_, lean_object* v_inst_2728_, lean_object* v_inst_2729_, lean_object* v_B_2730_, lean_object* v_inst_2731_, lean_object* v_inst_2732_){
_start:
{
lean_object* v_res_2733_; 
v_res_2733_ = lp_mathlib_MulDistribMulActionHom_instInhabitedDistribMulActionHom(v_M_2722_, v_inst_2723_, v_N_2724_, v_inst_2725_, v_00_u03c6_2726_, v_A_2727_, v_inst_2728_, v_inst_2729_, v_B_2730_, v_inst_2731_, v_inst_2732_);
lean_dec(v_inst_2732_);
lean_dec_ref(v_inst_2731_);
lean_dec(v_inst_2729_);
lean_dec_ref(v_inst_2728_);
lean_dec(v_00_u03c6_2726_);
lean_dec_ref(v_inst_2725_);
lean_dec_ref(v_inst_2723_);
return v_res_2733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_comp___redArg(lean_object* v_g_2734_, lean_object* v_f_2735_){
_start:
{
lean_object* v___f_2736_; lean_object* v___f_2737_; lean_object* v___f_2738_; 
v___f_2736_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2736_, 0, v_g_2734_);
v___f_2737_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2737_, 0, v_f_2735_);
v___f_2738_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2738_, 0, v___f_2737_);
lean_closure_set(v___f_2738_, 1, v___f_2736_);
return v___f_2738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_comp(lean_object* v_M_2739_, lean_object* v_inst_2740_, lean_object* v_N_2741_, lean_object* v_inst_2742_, lean_object* v_P_2743_, lean_object* v_inst_2744_, lean_object* v_00_u03c6_2745_, lean_object* v_00_u03c8_2746_, lean_object* v_00_u03c7_2747_, lean_object* v_A_2748_, lean_object* v_inst_2749_, lean_object* v_inst_2750_, lean_object* v_B_2751_, lean_object* v_inst_2752_, lean_object* v_inst_2753_, lean_object* v_C_2754_, lean_object* v_inst_2755_, lean_object* v_inst_2756_, lean_object* v_00_u03ba_2757_, lean_object* v_g_2758_, lean_object* v_f_2759_){
_start:
{
lean_object* v___x_2760_; 
v___x_2760_ = lp_mathlib_MulDistribMulActionHom_comp___redArg(v_g_2758_, v_f_2759_);
return v___x_2760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_comp___boxed(lean_object** _args){
lean_object* v_M_2761_ = _args[0];
lean_object* v_inst_2762_ = _args[1];
lean_object* v_N_2763_ = _args[2];
lean_object* v_inst_2764_ = _args[3];
lean_object* v_P_2765_ = _args[4];
lean_object* v_inst_2766_ = _args[5];
lean_object* v_00_u03c6_2767_ = _args[6];
lean_object* v_00_u03c8_2768_ = _args[7];
lean_object* v_00_u03c7_2769_ = _args[8];
lean_object* v_A_2770_ = _args[9];
lean_object* v_inst_2771_ = _args[10];
lean_object* v_inst_2772_ = _args[11];
lean_object* v_B_2773_ = _args[12];
lean_object* v_inst_2774_ = _args[13];
lean_object* v_inst_2775_ = _args[14];
lean_object* v_C_2776_ = _args[15];
lean_object* v_inst_2777_ = _args[16];
lean_object* v_inst_2778_ = _args[17];
lean_object* v_00_u03ba_2779_ = _args[18];
lean_object* v_g_2780_ = _args[19];
lean_object* v_f_2781_ = _args[20];
_start:
{
lean_object* v_res_2782_; 
v_res_2782_ = lp_mathlib_MulDistribMulActionHom_comp(v_M_2761_, v_inst_2762_, v_N_2763_, v_inst_2764_, v_P_2765_, v_inst_2766_, v_00_u03c6_2767_, v_00_u03c8_2768_, v_00_u03c7_2769_, v_A_2770_, v_inst_2771_, v_inst_2772_, v_B_2773_, v_inst_2774_, v_inst_2775_, v_C_2776_, v_inst_2777_, v_inst_2778_, v_00_u03ba_2779_, v_g_2780_, v_f_2781_);
lean_dec(v_inst_2778_);
lean_dec_ref(v_inst_2777_);
lean_dec(v_inst_2775_);
lean_dec_ref(v_inst_2774_);
lean_dec(v_inst_2772_);
lean_dec_ref(v_inst_2771_);
lean_dec(v_00_u03c7_2769_);
lean_dec(v_00_u03c8_2768_);
lean_dec(v_00_u03c6_2767_);
lean_dec_ref(v_inst_2766_);
lean_dec_ref(v_inst_2764_);
lean_dec_ref(v_inst_2762_);
return v_res_2782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_comp___redArg(lean_object* v_g_2783_, lean_object* v_f_2784_){
_start:
{
lean_object* v___f_2785_; lean_object* v___f_2786_; lean_object* v___f_2787_; 
v___f_2785_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2785_, 0, v_g_2783_);
v___f_2786_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2786_, 0, v_f_2784_);
v___f_2787_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2787_, 0, v___f_2786_);
lean_closure_set(v___f_2787_, 1, v___f_2785_);
return v___f_2787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_comp(lean_object* v_M_2788_, lean_object* v_inst_2789_, lean_object* v_N_2790_, lean_object* v_inst_2791_, lean_object* v_P_2792_, lean_object* v_inst_2793_, lean_object* v_00_u03c6_2794_, lean_object* v_00_u03c8_2795_, lean_object* v_00_u03c7_2796_, lean_object* v_A_2797_, lean_object* v_inst_2798_, lean_object* v_inst_2799_, lean_object* v_B_2800_, lean_object* v_inst_2801_, lean_object* v_inst_2802_, lean_object* v_C_2803_, lean_object* v_inst_2804_, lean_object* v_inst_2805_, lean_object* v_00_u03ba_2806_, lean_object* v_g_2807_, lean_object* v_f_2808_){
_start:
{
lean_object* v___x_2809_; 
v___x_2809_ = lp_mathlib_DistribMulActionHom_comp___redArg(v_g_2807_, v_f_2808_);
return v___x_2809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_comp___boxed(lean_object** _args){
lean_object* v_M_2810_ = _args[0];
lean_object* v_inst_2811_ = _args[1];
lean_object* v_N_2812_ = _args[2];
lean_object* v_inst_2813_ = _args[3];
lean_object* v_P_2814_ = _args[4];
lean_object* v_inst_2815_ = _args[5];
lean_object* v_00_u03c6_2816_ = _args[6];
lean_object* v_00_u03c8_2817_ = _args[7];
lean_object* v_00_u03c7_2818_ = _args[8];
lean_object* v_A_2819_ = _args[9];
lean_object* v_inst_2820_ = _args[10];
lean_object* v_inst_2821_ = _args[11];
lean_object* v_B_2822_ = _args[12];
lean_object* v_inst_2823_ = _args[13];
lean_object* v_inst_2824_ = _args[14];
lean_object* v_C_2825_ = _args[15];
lean_object* v_inst_2826_ = _args[16];
lean_object* v_inst_2827_ = _args[17];
lean_object* v_00_u03ba_2828_ = _args[18];
lean_object* v_g_2829_ = _args[19];
lean_object* v_f_2830_ = _args[20];
_start:
{
lean_object* v_res_2831_; 
v_res_2831_ = lp_mathlib_DistribMulActionHom_comp(v_M_2810_, v_inst_2811_, v_N_2812_, v_inst_2813_, v_P_2814_, v_inst_2815_, v_00_u03c6_2816_, v_00_u03c8_2817_, v_00_u03c7_2818_, v_A_2819_, v_inst_2820_, v_inst_2821_, v_B_2822_, v_inst_2823_, v_inst_2824_, v_C_2825_, v_inst_2826_, v_inst_2827_, v_00_u03ba_2828_, v_g_2829_, v_f_2830_);
lean_dec(v_inst_2827_);
lean_dec_ref(v_inst_2826_);
lean_dec(v_inst_2824_);
lean_dec_ref(v_inst_2823_);
lean_dec(v_inst_2821_);
lean_dec_ref(v_inst_2820_);
lean_dec(v_00_u03c7_2818_);
lean_dec(v_00_u03c8_2817_);
lean_dec(v_00_u03c6_2816_);
lean_dec_ref(v_inst_2815_);
lean_dec_ref(v_inst_2813_);
lean_dec_ref(v_inst_2811_);
return v_res_2831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse___redArg(lean_object* v_g_2832_){
_start:
{
lean_inc(v_g_2832_);
return v_g_2832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse___redArg___boxed(lean_object* v_g_2833_){
_start:
{
lean_object* v_res_2834_; 
v_res_2834_ = lp_mathlib_MulDistribMulActionHom_inverse___redArg(v_g_2833_);
lean_dec(v_g_2833_);
return v_res_2834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse(lean_object* v_M_2835_, lean_object* v_inst_2836_, lean_object* v_A_2837_, lean_object* v_inst_2838_, lean_object* v_inst_2839_, lean_object* v_B_u2081_2840_, lean_object* v_inst_2841_, lean_object* v_inst_2842_, lean_object* v_f_2843_, lean_object* v_g_2844_, lean_object* v_h_u2081_2845_, lean_object* v_h_u2082_2846_){
_start:
{
lean_inc(v_g_2844_);
return v_g_2844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulActionHom_inverse___boxed(lean_object* v_M_2847_, lean_object* v_inst_2848_, lean_object* v_A_2849_, lean_object* v_inst_2850_, lean_object* v_inst_2851_, lean_object* v_B_u2081_2852_, lean_object* v_inst_2853_, lean_object* v_inst_2854_, lean_object* v_f_2855_, lean_object* v_g_2856_, lean_object* v_h_u2081_2857_, lean_object* v_h_u2082_2858_){
_start:
{
lean_object* v_res_2859_; 
v_res_2859_ = lp_mathlib_MulDistribMulActionHom_inverse(v_M_2847_, v_inst_2848_, v_A_2849_, v_inst_2850_, v_inst_2851_, v_B_u2081_2852_, v_inst_2853_, v_inst_2854_, v_f_2855_, v_g_2856_, v_h_u2081_2857_, v_h_u2082_2858_);
lean_dec(v_g_2856_);
lean_dec(v_f_2855_);
lean_dec(v_inst_2854_);
lean_dec_ref(v_inst_2853_);
lean_dec(v_inst_2851_);
lean_dec_ref(v_inst_2850_);
lean_dec_ref(v_inst_2848_);
return v_res_2859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse___redArg(lean_object* v_g_2860_){
_start:
{
lean_inc(v_g_2860_);
return v_g_2860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse___redArg___boxed(lean_object* v_g_2861_){
_start:
{
lean_object* v_res_2862_; 
v_res_2862_ = lp_mathlib_DistribMulActionHom_inverse___redArg(v_g_2861_);
lean_dec(v_g_2861_);
return v_res_2862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse(lean_object* v_M_2863_, lean_object* v_inst_2864_, lean_object* v_A_2865_, lean_object* v_inst_2866_, lean_object* v_inst_2867_, lean_object* v_B_u2081_2868_, lean_object* v_inst_2869_, lean_object* v_inst_2870_, lean_object* v_f_2871_, lean_object* v_g_2872_, lean_object* v_h_u2081_2873_, lean_object* v_h_u2082_2874_){
_start:
{
lean_inc(v_g_2872_);
return v_g_2872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulActionHom_inverse___boxed(lean_object* v_M_2875_, lean_object* v_inst_2876_, lean_object* v_A_2877_, lean_object* v_inst_2878_, lean_object* v_inst_2879_, lean_object* v_B_u2081_2880_, lean_object* v_inst_2881_, lean_object* v_inst_2882_, lean_object* v_f_2883_, lean_object* v_g_2884_, lean_object* v_h_u2081_2885_, lean_object* v_h_u2082_2886_){
_start:
{
lean_object* v_res_2887_; 
v_res_2887_ = lp_mathlib_DistribMulActionHom_inverse(v_M_2875_, v_inst_2876_, v_A_2877_, v_inst_2878_, v_inst_2879_, v_B_u2081_2880_, v_inst_2881_, v_inst_2882_, v_f_2883_, v_g_2884_, v_h_u2081_2885_, v_h_u2082_2886_);
lean_dec(v_g_2884_);
lean_dec(v_f_2883_);
lean_dec(v_inst_2882_);
lean_dec_ref(v_inst_2881_);
lean_dec(v_inst_2879_);
lean_dec_ref(v_inst_2878_);
lean_dec_ref(v_inst_2876_);
return v_res_2887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom___redArg(lean_object* v_self_2888_){
_start:
{
lean_inc(v_self_2888_);
return v_self_2888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom___redArg___boxed(lean_object* v_self_2889_){
_start:
{
lean_object* v_res_2890_; 
v_res_2890_ = lp_mathlib_MulSemiringActionHom_toRingHom___redArg(v_self_2889_);
lean_dec(v_self_2889_);
return v_res_2890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom(lean_object* v_M_2891_, lean_object* v_inst_2892_, lean_object* v_N_2893_, lean_object* v_inst_2894_, lean_object* v_00_u03c6_2895_, lean_object* v_R_2896_, lean_object* v_inst_2897_, lean_object* v_inst_2898_, lean_object* v_S_2899_, lean_object* v_inst_2900_, lean_object* v_inst_2901_, lean_object* v_self_2902_){
_start:
{
lean_inc(v_self_2902_);
return v_self_2902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_toRingHom___boxed(lean_object* v_M_2903_, lean_object* v_inst_2904_, lean_object* v_N_2905_, lean_object* v_inst_2906_, lean_object* v_00_u03c6_2907_, lean_object* v_R_2908_, lean_object* v_inst_2909_, lean_object* v_inst_2910_, lean_object* v_S_2911_, lean_object* v_inst_2912_, lean_object* v_inst_2913_, lean_object* v_self_2914_){
_start:
{
lean_object* v_res_2915_; 
v_res_2915_ = lp_mathlib_MulSemiringActionHom_toRingHom(v_M_2903_, v_inst_2904_, v_N_2905_, v_inst_2906_, v_00_u03c6_2907_, v_R_2908_, v_inst_2909_, v_inst_2910_, v_S_2911_, v_inst_2912_, v_inst_2913_, v_self_2914_);
lean_dec(v_self_2914_);
lean_dec(v_inst_2913_);
lean_dec_ref(v_inst_2912_);
lean_dec(v_inst_2910_);
lean_dec_ref(v_inst_2909_);
lean_dec(v_00_u03c6_2907_);
lean_dec_ref(v_inst_2906_);
lean_dec_ref(v_inst_2904_);
return v_res_2915_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1(void){
_start:
{
lean_object* v___x_2941_; lean_object* v___x_2942_; 
v___x_2941_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__0));
v___x_2942_ = l_String_toRawSubstring_x27(v___x_2941_);
return v___x_2942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1(lean_object* v_x_2956_, lean_object* v_a_2957_, lean_object* v_a_2958_){
_start:
{
lean_object* v___x_2959_; uint8_t v___x_2960_; 
v___x_2959_ = ((lean_object*)(lp_mathlib_MulSemiringActionHomLocal_u227a___closed__1));
lean_inc(v_x_2956_);
v___x_2960_ = l_Lean_Syntax_isOfKind(v_x_2956_, v___x_2959_);
if (v___x_2960_ == 0)
{
lean_object* v___x_2961_; lean_object* v___x_2962_; 
lean_dec(v_x_2956_);
v___x_2961_ = lean_box(1);
v___x_2962_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2962_, 0, v___x_2961_);
lean_ctor_set(v___x_2962_, 1, v_a_2958_);
return v___x_2962_;
}
else
{
lean_object* v_quotContext_2963_; lean_object* v_currMacroScope_2964_; lean_object* v_ref_2965_; lean_object* v___x_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; uint8_t v___x_2972_; lean_object* v___x_2973_; lean_object* v___x_2974_; lean_object* v___x_2975_; lean_object* v___x_2976_; lean_object* v___x_2977_; lean_object* v___x_2978_; lean_object* v___x_2979_; lean_object* v___x_2980_; lean_object* v___x_2981_; lean_object* v___x_2982_; lean_object* v___x_2983_; 
v_quotContext_2963_ = lean_ctor_get(v_a_2957_, 1);
v_currMacroScope_2964_ = lean_ctor_get(v_a_2957_, 2);
v_ref_2965_ = lean_ctor_get(v_a_2957_, 5);
v___x_2966_ = lean_unsigned_to_nat(0u);
v___x_2967_ = l_Lean_Syntax_getArg(v_x_2956_, v___x_2966_);
v___x_2968_ = lean_unsigned_to_nat(2u);
v___x_2969_ = l_Lean_Syntax_getArg(v_x_2956_, v___x_2968_);
v___x_2970_ = lean_unsigned_to_nat(4u);
v___x_2971_ = l_Lean_Syntax_getArg(v_x_2956_, v___x_2970_);
lean_dec(v_x_2956_);
v___x_2972_ = 0;
v___x_2973_ = l_Lean_SourceInfo_fromRef(v_ref_2965_, v___x_2972_);
v___x_2974_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_2975_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1);
v___x_2976_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2));
lean_inc(v_currMacroScope_2964_);
lean_inc(v_quotContext_2963_);
v___x_2977_ = l_Lean_addMacroScope(v_quotContext_2963_, v___x_2976_, v_currMacroScope_2964_);
v___x_2978_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_2973_, 2);
v___x_2979_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2979_, 0, v___x_2973_);
lean_ctor_set(v___x_2979_, 1, v___x_2975_);
lean_ctor_set(v___x_2979_, 2, v___x_2977_);
lean_ctor_set(v___x_2979_, 3, v___x_2978_);
v___x_2980_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_2981_ = l_Lean_Syntax_node3(v___x_2973_, v___x_2980_, v___x_2969_, v___x_2967_, v___x_2971_);
v___x_2982_ = l_Lean_Syntax_node2(v___x_2973_, v___x_2974_, v___x_2979_, v___x_2981_);
v___x_2983_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2983_, 0, v___x_2982_);
lean_ctor_set(v___x_2983_, 1, v_a_2958_);
return v___x_2983_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___boxed(lean_object* v_x_2984_, lean_object* v_a_2985_, lean_object* v_a_2986_){
_start:
{
lean_object* v_res_2987_; 
v_res_2987_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1(v_x_2984_, v_a_2985_, v_a_2986_);
lean_dec_ref(v_a_2985_);
return v_res_2987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__1(lean_object* v_x_2988_, lean_object* v_a_2989_, lean_object* v_a_2990_){
_start:
{
lean_object* v___x_2991_; uint8_t v___x_2992_; 
v___x_2991_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_2988_);
v___x_2992_ = l_Lean_Syntax_isOfKind(v_x_2988_, v___x_2991_);
if (v___x_2992_ == 0)
{
lean_object* v___x_2993_; lean_object* v___x_2994_; 
lean_dec(v_x_2988_);
v___x_2993_ = lean_box(0);
v___x_2994_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2994_, 0, v___x_2993_);
lean_ctor_set(v___x_2994_, 1, v_a_2990_);
return v___x_2994_;
}
else
{
lean_object* v___x_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; uint8_t v___x_2998_; 
v___x_2995_ = lean_unsigned_to_nat(0u);
v___x_2996_ = l_Lean_Syntax_getArg(v_x_2988_, v___x_2995_);
v___x_2997_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_2996_);
v___x_2998_ = l_Lean_Syntax_isOfKind(v___x_2996_, v___x_2997_);
if (v___x_2998_ == 0)
{
lean_object* v___x_2999_; lean_object* v___x_3000_; 
lean_dec(v___x_2996_);
lean_dec(v_x_2988_);
v___x_2999_ = lean_box(0);
v___x_3000_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3000_, 0, v___x_2999_);
lean_ctor_set(v___x_3000_, 1, v_a_2990_);
return v___x_3000_;
}
else
{
lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; uint8_t v___x_3004_; 
v___x_3001_ = lean_unsigned_to_nat(1u);
v___x_3002_ = l_Lean_Syntax_getArg(v_x_2988_, v___x_3001_);
lean_dec(v_x_2988_);
v___x_3003_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_3002_);
v___x_3004_ = l_Lean_Syntax_matchesNull(v___x_3002_, v___x_3003_);
if (v___x_3004_ == 0)
{
lean_object* v___x_3005_; lean_object* v___x_3006_; 
lean_dec(v___x_3002_);
lean_dec(v___x_2996_);
v___x_3005_ = lean_box(0);
v___x_3006_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3006_, 0, v___x_3005_);
lean_ctor_set(v___x_3006_, 1, v_a_2990_);
return v___x_3006_;
}
else
{
lean_object* v___x_3007_; lean_object* v___x_3008_; lean_object* v___x_3009_; lean_object* v___x_3010_; lean_object* v_ref_3011_; uint8_t v___x_3012_; lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; 
v___x_3007_ = l_Lean_Syntax_getArg(v___x_3002_, v___x_2995_);
v___x_3008_ = l_Lean_Syntax_getArg(v___x_3002_, v___x_3001_);
v___x_3009_ = lean_unsigned_to_nat(2u);
v___x_3010_ = l_Lean_Syntax_getArg(v___x_3002_, v___x_3009_);
lean_dec(v___x_3002_);
v_ref_3011_ = l_Lean_replaceRef(v___x_2996_, v_a_2989_);
lean_dec(v___x_2996_);
v___x_3012_ = 0;
v___x_3013_ = l_Lean_SourceInfo_fromRef(v_ref_3011_, v___x_3012_);
lean_dec(v_ref_3011_);
v___x_3014_ = ((lean_object*)(lp_mathlib_MulSemiringActionHomLocal_u227a___closed__1));
v___x_3015_ = ((lean_object*)(lp_mathlib_MulSemiringActionHomLocal_u227a___closed__2));
lean_inc_n(v___x_3013_, 2);
v___x_3016_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3016_, 0, v___x_3013_);
lean_ctor_set(v___x_3016_, 1, v___x_3015_);
v___x_3017_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_3018_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3018_, 0, v___x_3013_);
lean_ctor_set(v___x_3018_, 1, v___x_3017_);
v___x_3019_ = l_Lean_Syntax_node5(v___x_3013_, v___x_3014_, v___x_3008_, v___x_3016_, v___x_3007_, v___x_3018_, v___x_3010_);
v___x_3020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3020_, 0, v___x_3019_);
lean_ctor_set(v___x_3020_, 1, v_a_2990_);
return v___x_3020_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__1___boxed(lean_object* v_x_3021_, lean_object* v_a_3022_, lean_object* v_a_3023_){
_start:
{
lean_object* v_res_3024_; 
v_res_3024_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__1(v_x_3021_, v_a_3022_, v_a_3023_);
lean_dec(v_a_3022_);
return v_res_3024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomIdLocal_u227a__1(lean_object* v_x_3049_, lean_object* v_a_3050_, lean_object* v_a_3051_){
_start:
{
lean_object* v___x_3052_; uint8_t v___x_3053_; 
v___x_3052_ = ((lean_object*)(lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__1));
lean_inc(v_x_3049_);
v___x_3053_ = l_Lean_Syntax_isOfKind(v_x_3049_, v___x_3052_);
if (v___x_3053_ == 0)
{
lean_object* v___x_3054_; lean_object* v___x_3055_; 
lean_dec(v_x_3049_);
v___x_3054_ = lean_box(1);
v___x_3055_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3055_, 0, v___x_3054_);
lean_ctor_set(v___x_3055_, 1, v_a_3051_);
return v___x_3055_;
}
else
{
lean_object* v_quotContext_3056_; lean_object* v_currMacroScope_3057_; lean_object* v_ref_3058_; lean_object* v___x_3059_; lean_object* v___x_3060_; lean_object* v___x_3061_; lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; uint8_t v___x_3065_; lean_object* v___x_3066_; lean_object* v___x_3067_; lean_object* v___x_3068_; lean_object* v___x_3069_; lean_object* v___x_3070_; lean_object* v___x_3071_; lean_object* v___x_3072_; lean_object* v___x_3073_; lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; lean_object* v___x_3079_; lean_object* v___x_3080_; lean_object* v___x_3081_; lean_object* v___x_3082_; lean_object* v___x_3083_; lean_object* v___x_3084_; lean_object* v___x_3085_; lean_object* v___x_3086_; lean_object* v___x_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; lean_object* v___x_3092_; lean_object* v___x_3093_; lean_object* v___x_3094_; lean_object* v___x_3095_; lean_object* v___x_3096_; lean_object* v___x_3097_; lean_object* v___x_3098_; 
v_quotContext_3056_ = lean_ctor_get(v_a_3050_, 1);
v_currMacroScope_3057_ = lean_ctor_get(v_a_3050_, 2);
v_ref_3058_ = lean_ctor_get(v_a_3050_, 5);
v___x_3059_ = lean_unsigned_to_nat(0u);
v___x_3060_ = l_Lean_Syntax_getArg(v_x_3049_, v___x_3059_);
v___x_3061_ = lean_unsigned_to_nat(2u);
v___x_3062_ = l_Lean_Syntax_getArg(v_x_3049_, v___x_3061_);
v___x_3063_ = lean_unsigned_to_nat(4u);
v___x_3064_ = l_Lean_Syntax_getArg(v_x_3049_, v___x_3063_);
lean_dec(v_x_3049_);
v___x_3065_ = 0;
v___x_3066_ = l_Lean_SourceInfo_fromRef(v_ref_3058_, v___x_3065_);
v___x_3067_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
v___x_3068_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__1);
v___x_3069_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__2));
lean_inc_n(v_currMacroScope_3057_, 3);
lean_inc_n(v_quotContext_3056_, 3);
v___x_3070_ = l_Lean_addMacroScope(v_quotContext_3056_, v___x_3069_, v_currMacroScope_3057_);
v___x_3071_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomLocal_u227a__1___closed__6));
lean_inc_n(v___x_3066_, 11);
v___x_3072_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3072_, 0, v___x_3066_);
lean_ctor_set(v___x_3072_, 1, v___x_3068_);
lean_ctor_set(v___x_3072_, 2, v___x_3070_);
lean_ctor_set(v___x_3072_, 3, v___x_3071_);
v___x_3073_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__13));
v___x_3074_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__1));
v___x_3075_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__3));
v___x_3076_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__4));
v___x_3077_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3077_, 0, v___x_3066_);
lean_ctor_set(v___x_3077_, 1, v___x_3076_);
v___x_3078_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__6));
v___x_3079_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__8);
v___x_3080_ = lean_box(0);
v___x_3081_ = l_Lean_addMacroScope(v_quotContext_3056_, v___x_3080_, v_currMacroScope_3057_);
v___x_3082_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__10));
v___x_3083_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3083_, 0, v___x_3066_);
lean_ctor_set(v___x_3083_, 1, v___x_3079_);
lean_ctor_set(v___x_3083_, 2, v___x_3081_);
lean_ctor_set(v___x_3083_, 3, v___x_3082_);
v___x_3084_ = l_Lean_Syntax_node1(v___x_3066_, v___x_3078_, v___x_3083_);
v___x_3085_ = l_Lean_Syntax_node2(v___x_3066_, v___x_3075_, v___x_3077_, v___x_3084_);
v___x_3086_ = lean_obj_once(&lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1, &lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__1);
v___x_3087_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3));
v___x_3088_ = l_Lean_addMacroScope(v_quotContext_3056_, v___x_3087_, v_currMacroScope_3057_);
v___x_3089_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__5));
v___x_3090_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3090_, 0, v___x_3066_);
lean_ctor_set(v___x_3090_, 1, v___x_3086_);
lean_ctor_set(v___x_3090_, 2, v___x_3088_);
lean_ctor_set(v___x_3090_, 3, v___x_3089_);
v___x_3091_ = l_Lean_Syntax_node1(v___x_3066_, v___x_3073_, v___x_3062_);
v___x_3092_ = l_Lean_Syntax_node2(v___x_3066_, v___x_3067_, v___x_3090_, v___x_3091_);
v___x_3093_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomIdLocal_u227a__1___closed__19));
v___x_3094_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3094_, 0, v___x_3066_);
lean_ctor_set(v___x_3094_, 1, v___x_3093_);
v___x_3095_ = l_Lean_Syntax_node3(v___x_3066_, v___x_3074_, v___x_3085_, v___x_3092_, v___x_3094_);
v___x_3096_ = l_Lean_Syntax_node3(v___x_3066_, v___x_3073_, v___x_3095_, v___x_3060_, v___x_3064_);
v___x_3097_ = l_Lean_Syntax_node2(v___x_3066_, v___x_3067_, v___x_3072_, v___x_3096_);
v___x_3098_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3098_, 0, v___x_3097_);
lean_ctor_set(v___x_3098_, 1, v_a_3051_);
return v___x_3098_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomIdLocal_u227a__1___boxed(lean_object* v_x_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_){
_start:
{
lean_object* v_res_3102_; 
v_res_3102_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulSemiringActionHomIdLocal_u227a__1(v_x_3099_, v_a_3100_, v_a_3101_);
lean_dec_ref(v_a_3100_);
return v_res_3102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__2(lean_object* v_x_3103_, lean_object* v_a_3104_, lean_object* v_a_3105_){
_start:
{
lean_object* v___x_3106_; uint8_t v___x_3107_; 
v___x_3106_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__MulActionHomLocal_u227a__1___closed__4));
lean_inc(v_x_3103_);
v___x_3107_ = l_Lean_Syntax_isOfKind(v_x_3103_, v___x_3106_);
if (v___x_3107_ == 0)
{
lean_object* v___x_3108_; lean_object* v___x_3109_; 
lean_dec(v_x_3103_);
v___x_3108_ = lean_box(0);
v___x_3109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3109_, 0, v___x_3108_);
lean_ctor_set(v___x_3109_, 1, v_a_3105_);
return v___x_3109_;
}
else
{
lean_object* v___x_3110_; lean_object* v___x_3111_; lean_object* v___x_3112_; uint8_t v___x_3113_; 
v___x_3110_ = lean_unsigned_to_nat(0u);
v___x_3111_ = l_Lean_Syntax_getArg(v_x_3103_, v___x_3110_);
v___x_3112_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulActionHom__1___closed__1));
lean_inc(v___x_3111_);
v___x_3113_ = l_Lean_Syntax_isOfKind(v___x_3111_, v___x_3112_);
if (v___x_3113_ == 0)
{
lean_object* v___x_3114_; lean_object* v___x_3115_; 
lean_dec(v___x_3111_);
lean_dec(v_x_3103_);
v___x_3114_ = lean_box(0);
v___x_3115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3115_, 0, v___x_3114_);
lean_ctor_set(v___x_3115_, 1, v_a_3105_);
return v___x_3115_;
}
else
{
lean_object* v___x_3116_; lean_object* v___x_3117_; lean_object* v___x_3118_; uint8_t v___x_3119_; 
v___x_3116_ = lean_unsigned_to_nat(1u);
v___x_3117_ = l_Lean_Syntax_getArg(v_x_3103_, v___x_3116_);
lean_dec(v_x_3103_);
v___x_3118_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_3117_);
v___x_3119_ = l_Lean_Syntax_matchesNull(v___x_3117_, v___x_3118_);
if (v___x_3119_ == 0)
{
lean_object* v___x_3120_; lean_object* v___x_3121_; 
lean_dec(v___x_3117_);
lean_dec(v___x_3111_);
v___x_3120_ = lean_box(0);
v___x_3121_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3121_, 0, v___x_3120_);
lean_ctor_set(v___x_3121_, 1, v_a_3105_);
return v___x_3121_;
}
else
{
lean_object* v___x_3122_; uint8_t v___x_3123_; 
v___x_3122_ = l_Lean_Syntax_getArg(v___x_3117_, v___x_3110_);
lean_inc(v___x_3122_);
v___x_3123_ = l_Lean_Syntax_isOfKind(v___x_3122_, v___x_3106_);
if (v___x_3123_ == 0)
{
lean_object* v___x_3124_; lean_object* v___x_3125_; 
lean_dec(v___x_3122_);
lean_dec(v___x_3117_);
lean_dec(v___x_3111_);
v___x_3124_ = lean_box(0);
v___x_3125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3125_, 0, v___x_3124_);
lean_ctor_set(v___x_3125_, 1, v_a_3105_);
return v___x_3125_;
}
else
{
lean_object* v___x_3126_; lean_object* v___x_3127_; uint8_t v___x_3128_; 
v___x_3126_ = l_Lean_Syntax_getArg(v___x_3122_, v___x_3110_);
v___x_3127_ = ((lean_object*)(lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______macroRules__DistribMulActionHomIdLocal_u227a__1___closed__3));
v___x_3128_ = l_Lean_Syntax_matchesIdent(v___x_3126_, v___x_3127_);
lean_dec(v___x_3126_);
if (v___x_3128_ == 0)
{
lean_object* v___x_3129_; lean_object* v___x_3130_; 
lean_dec(v___x_3122_);
lean_dec(v___x_3117_);
lean_dec(v___x_3111_);
v___x_3129_ = lean_box(0);
v___x_3130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3130_, 0, v___x_3129_);
lean_ctor_set(v___x_3130_, 1, v_a_3105_);
return v___x_3130_;
}
else
{
lean_object* v___x_3131_; uint8_t v___x_3132_; 
v___x_3131_ = l_Lean_Syntax_getArg(v___x_3122_, v___x_3116_);
lean_dec(v___x_3122_);
lean_inc(v___x_3131_);
v___x_3132_ = l_Lean_Syntax_matchesNull(v___x_3131_, v___x_3116_);
if (v___x_3132_ == 0)
{
lean_object* v___x_3133_; lean_object* v___x_3134_; 
lean_dec(v___x_3131_);
lean_dec(v___x_3117_);
lean_dec(v___x_3111_);
v___x_3133_ = lean_box(0);
v___x_3134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3134_, 0, v___x_3133_);
lean_ctor_set(v___x_3134_, 1, v_a_3105_);
return v___x_3134_;
}
else
{
lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; lean_object* v___x_3138_; lean_object* v_ref_3139_; uint8_t v___x_3140_; lean_object* v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; lean_object* v___x_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; 
v___x_3135_ = l_Lean_Syntax_getArg(v___x_3131_, v___x_3110_);
lean_dec(v___x_3131_);
v___x_3136_ = l_Lean_Syntax_getArg(v___x_3117_, v___x_3116_);
v___x_3137_ = lean_unsigned_to_nat(2u);
v___x_3138_ = l_Lean_Syntax_getArg(v___x_3117_, v___x_3137_);
lean_dec(v___x_3117_);
v_ref_3139_ = l_Lean_replaceRef(v___x_3111_, v_a_3104_);
lean_dec(v___x_3111_);
v___x_3140_ = 0;
v___x_3141_ = l_Lean_SourceInfo_fromRef(v_ref_3139_, v___x_3140_);
lean_dec(v_ref_3139_);
v___x_3142_ = ((lean_object*)(lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__1));
v___x_3143_ = ((lean_object*)(lp_mathlib_MulSemiringActionHomIdLocal_u227a___closed__2));
lean_inc_n(v___x_3141_, 2);
v___x_3144_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3144_, 0, v___x_3141_);
lean_ctor_set(v___x_3144_, 1, v___x_3143_);
v___x_3145_ = ((lean_object*)(lp_mathlib_MulActionHomLocal_u227a___closed__10));
v___x_3146_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3146_, 0, v___x_3141_);
lean_ctor_set(v___x_3146_, 1, v___x_3145_);
v___x_3147_ = l_Lean_Syntax_node5(v___x_3141_, v___x_3142_, v___x_3136_, v___x_3144_, v___x_3135_, v___x_3146_, v___x_3138_);
v___x_3148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3148_, 0, v___x_3147_);
lean_ctor_set(v___x_3148_, 1, v_a_3105_);
return v___x_3148_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__2___boxed(lean_object* v_x_3149_, lean_object* v_a_3150_, lean_object* v_a_3151_){
_start:
{
lean_object* v_res_3152_; 
v_res_3152_ = lp_mathlib___aux__Mathlib__GroupTheory__GroupAction__Hom______unexpand__MulSemiringActionHom__2(v_x_3149_, v_a_3150_, v_a_3151_);
lean_dec(v_a_3150_);
return v_res_3152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom___redArg(lean_object* v_inst_3153_, lean_object* v_f_3154_){
_start:
{
lean_object* v___x_3155_; 
v___x_3155_ = lean_apply_1(v_inst_3153_, v_f_3154_);
return v___x_3155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom(lean_object* v_M_3156_, lean_object* v_inst_3157_, lean_object* v_N_3158_, lean_object* v_inst_3159_, lean_object* v_00_u03c6_3160_, lean_object* v_R_3161_, lean_object* v_inst_3162_, lean_object* v_inst_3163_, lean_object* v_S_3164_, lean_object* v_inst_3165_, lean_object* v_inst_3166_, lean_object* v_F_3167_, lean_object* v_inst_3168_, lean_object* v_inst_3169_, lean_object* v_f_3170_){
_start:
{
lean_object* v___x_3171_; 
v___x_3171_ = lean_apply_1(v_inst_3168_, v_f_3170_);
return v___x_3171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom___boxed(lean_object* v_M_3172_, lean_object* v_inst_3173_, lean_object* v_N_3174_, lean_object* v_inst_3175_, lean_object* v_00_u03c6_3176_, lean_object* v_R_3177_, lean_object* v_inst_3178_, lean_object* v_inst_3179_, lean_object* v_S_3180_, lean_object* v_inst_3181_, lean_object* v_inst_3182_, lean_object* v_F_3183_, lean_object* v_inst_3184_, lean_object* v_inst_3185_, lean_object* v_f_3186_){
_start:
{
lean_object* v_res_3187_; 
v_res_3187_ = lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom(v_M_3172_, v_inst_3173_, v_N_3174_, v_inst_3175_, v_00_u03c6_3176_, v_R_3177_, v_inst_3178_, v_inst_3179_, v_S_3180_, v_inst_3181_, v_inst_3182_, v_F_3183_, v_inst_3184_, v_inst_3185_, v_f_3186_);
lean_dec(v_inst_3182_);
lean_dec_ref(v_inst_3181_);
lean_dec(v_inst_3179_);
lean_dec_ref(v_inst_3178_);
lean_dec(v_00_u03c6_3176_);
lean_dec_ref(v_inst_3175_);
lean_dec_ref(v_inst_3173_);
return v_res_3187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_instCoeTCOfMulSemiringActionSemiHomClassCoeMonoidHom___redArg(lean_object* v_inst_3188_, lean_object* v_inst_3189_, lean_object* v_00_u03c6_3190_, lean_object* v_inst_3191_, lean_object* v_inst_3192_, lean_object* v_inst_3193_, lean_object* v_inst_3194_, lean_object* v_inst_3195_){
_start:
{
lean_object* v___x_3196_; 
v___x_3196_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom___boxed), 15, 14);
lean_closure_set(v___x_3196_, 0, lean_box(0));
lean_closure_set(v___x_3196_, 1, v_inst_3188_);
lean_closure_set(v___x_3196_, 2, lean_box(0));
lean_closure_set(v___x_3196_, 3, v_inst_3189_);
lean_closure_set(v___x_3196_, 4, v_00_u03c6_3190_);
lean_closure_set(v___x_3196_, 5, lean_box(0));
lean_closure_set(v___x_3196_, 6, v_inst_3191_);
lean_closure_set(v___x_3196_, 7, v_inst_3192_);
lean_closure_set(v___x_3196_, 8, lean_box(0));
lean_closure_set(v___x_3196_, 9, v_inst_3193_);
lean_closure_set(v___x_3196_, 10, v_inst_3194_);
lean_closure_set(v___x_3196_, 11, lean_box(0));
lean_closure_set(v___x_3196_, 12, v_inst_3195_);
lean_closure_set(v___x_3196_, 13, lean_box(0));
return v___x_3196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_instCoeTCOfMulSemiringActionSemiHomClassCoeMonoidHom(lean_object* v_M_3197_, lean_object* v_inst_3198_, lean_object* v_N_3199_, lean_object* v_inst_3200_, lean_object* v_00_u03c6_3201_, lean_object* v_R_3202_, lean_object* v_inst_3203_, lean_object* v_inst_3204_, lean_object* v_S_3205_, lean_object* v_inst_3206_, lean_object* v_inst_3207_, lean_object* v_F_3208_, lean_object* v_inst_3209_, lean_object* v_inst_3210_){
_start:
{
lean_object* v___x_3211_; 
v___x_3211_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringActionHomClass_toMulSemiringActionHom___boxed), 15, 14);
lean_closure_set(v___x_3211_, 0, lean_box(0));
lean_closure_set(v___x_3211_, 1, v_inst_3198_);
lean_closure_set(v___x_3211_, 2, lean_box(0));
lean_closure_set(v___x_3211_, 3, v_inst_3200_);
lean_closure_set(v___x_3211_, 4, v_00_u03c6_3201_);
lean_closure_set(v___x_3211_, 5, lean_box(0));
lean_closure_set(v___x_3211_, 6, v_inst_3203_);
lean_closure_set(v___x_3211_, 7, v_inst_3204_);
lean_closure_set(v___x_3211_, 8, lean_box(0));
lean_closure_set(v___x_3211_, 9, v_inst_3206_);
lean_closure_set(v___x_3211_, 10, v_inst_3207_);
lean_closure_set(v___x_3211_, 11, lean_box(0));
lean_closure_set(v___x_3211_, 12, v_inst_3209_);
lean_closure_set(v___x_3211_, 13, lean_box(0));
return v___x_3211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_id(lean_object* v_M_3212_, lean_object* v_inst_3213_, lean_object* v_R_3214_, lean_object* v_inst_3215_, lean_object* v_inst_3216_){
_start:
{
lean_object* v___f_3217_; 
v___f_3217_ = ((lean_object*)(lp_mathlib_MulActionHom_id___closed__0));
return v___f_3217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_id___boxed(lean_object* v_M_3218_, lean_object* v_inst_3219_, lean_object* v_R_3220_, lean_object* v_inst_3221_, lean_object* v_inst_3222_){
_start:
{
lean_object* v_res_3223_; 
v_res_3223_ = lp_mathlib_MulSemiringActionHom_id(v_M_3218_, v_inst_3219_, v_R_3220_, v_inst_3221_, v_inst_3222_);
lean_dec(v_inst_3222_);
lean_dec_ref(v_inst_3221_);
lean_dec_ref(v_inst_3219_);
return v_res_3223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_comp___redArg(lean_object* v_g_3224_, lean_object* v_f_3225_){
_start:
{
lean_object* v___f_3226_; lean_object* v___f_3227_; lean_object* v___x_3228_; 
v___f_3226_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_3226_, 0, v_g_3224_);
v___f_3227_ = lean_alloc_closure((void*)(lp_mathlib_MulActionHom_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_3227_, 0, v_f_3225_);
v___x_3228_ = lp_mathlib_DistribMulActionHom_comp___redArg(v___f_3226_, v___f_3227_);
return v___x_3228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_comp(lean_object* v_M_3229_, lean_object* v_inst_3230_, lean_object* v_N_3231_, lean_object* v_inst_3232_, lean_object* v_P_3233_, lean_object* v_inst_3234_, lean_object* v_00_u03c6_3235_, lean_object* v_00_u03c8_3236_, lean_object* v_00_u03c7_3237_, lean_object* v_R_3238_, lean_object* v_inst_3239_, lean_object* v_inst_3240_, lean_object* v_S_3241_, lean_object* v_inst_3242_, lean_object* v_inst_3243_, lean_object* v_T_3244_, lean_object* v_inst_3245_, lean_object* v_inst_3246_, lean_object* v_g_3247_, lean_object* v_f_3248_, lean_object* v_00_u03ba_3249_){
_start:
{
lean_object* v___x_3250_; 
v___x_3250_ = lp_mathlib_MulSemiringActionHom_comp___redArg(v_g_3247_, v_f_3248_);
return v___x_3250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_comp___boxed(lean_object** _args){
lean_object* v_M_3251_ = _args[0];
lean_object* v_inst_3252_ = _args[1];
lean_object* v_N_3253_ = _args[2];
lean_object* v_inst_3254_ = _args[3];
lean_object* v_P_3255_ = _args[4];
lean_object* v_inst_3256_ = _args[5];
lean_object* v_00_u03c6_3257_ = _args[6];
lean_object* v_00_u03c8_3258_ = _args[7];
lean_object* v_00_u03c7_3259_ = _args[8];
lean_object* v_R_3260_ = _args[9];
lean_object* v_inst_3261_ = _args[10];
lean_object* v_inst_3262_ = _args[11];
lean_object* v_S_3263_ = _args[12];
lean_object* v_inst_3264_ = _args[13];
lean_object* v_inst_3265_ = _args[14];
lean_object* v_T_3266_ = _args[15];
lean_object* v_inst_3267_ = _args[16];
lean_object* v_inst_3268_ = _args[17];
lean_object* v_g_3269_ = _args[18];
lean_object* v_f_3270_ = _args[19];
lean_object* v_00_u03ba_3271_ = _args[20];
_start:
{
lean_object* v_res_3272_; 
v_res_3272_ = lp_mathlib_MulSemiringActionHom_comp(v_M_3251_, v_inst_3252_, v_N_3253_, v_inst_3254_, v_P_3255_, v_inst_3256_, v_00_u03c6_3257_, v_00_u03c8_3258_, v_00_u03c7_3259_, v_R_3260_, v_inst_3261_, v_inst_3262_, v_S_3263_, v_inst_3264_, v_inst_3265_, v_T_3266_, v_inst_3267_, v_inst_3268_, v_g_3269_, v_f_3270_, v_00_u03ba_3271_);
lean_dec(v_inst_3268_);
lean_dec_ref(v_inst_3267_);
lean_dec(v_inst_3265_);
lean_dec_ref(v_inst_3264_);
lean_dec(v_inst_3262_);
lean_dec_ref(v_inst_3261_);
lean_dec(v_00_u03c7_3259_);
lean_dec(v_00_u03c8_3258_);
lean_dec(v_00_u03c6_3257_);
lean_dec_ref(v_inst_3256_);
lean_dec_ref(v_inst_3254_);
lean_dec_ref(v_inst_3252_);
return v_res_3272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27___redArg(lean_object* v_g_3273_){
_start:
{
lean_inc(v_g_3273_);
return v_g_3273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27___redArg___boxed(lean_object* v_g_3274_){
_start:
{
lean_object* v_res_3275_; 
v_res_3275_ = lp_mathlib_MulSemiringActionHom_inverse_x27___redArg(v_g_3274_);
lean_dec(v_g_3274_);
return v_res_3275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27(lean_object* v_M_3276_, lean_object* v_inst_3277_, lean_object* v_N_3278_, lean_object* v_inst_3279_, lean_object* v_00_u03c6_3280_, lean_object* v_00_u03c6_x27_3281_, lean_object* v_R_3282_, lean_object* v_inst_3283_, lean_object* v_inst_3284_, lean_object* v_S_3285_, lean_object* v_inst_3286_, lean_object* v_inst_3287_, lean_object* v_f_3288_, lean_object* v_g_3289_, lean_object* v_k_3290_, lean_object* v_h_u2081_3291_, lean_object* v_h_u2082_3292_){
_start:
{
lean_inc(v_g_3289_);
return v_g_3289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse_x27___boxed(lean_object** _args){
lean_object* v_M_3293_ = _args[0];
lean_object* v_inst_3294_ = _args[1];
lean_object* v_N_3295_ = _args[2];
lean_object* v_inst_3296_ = _args[3];
lean_object* v_00_u03c6_3297_ = _args[4];
lean_object* v_00_u03c6_x27_3298_ = _args[5];
lean_object* v_R_3299_ = _args[6];
lean_object* v_inst_3300_ = _args[7];
lean_object* v_inst_3301_ = _args[8];
lean_object* v_S_3302_ = _args[9];
lean_object* v_inst_3303_ = _args[10];
lean_object* v_inst_3304_ = _args[11];
lean_object* v_f_3305_ = _args[12];
lean_object* v_g_3306_ = _args[13];
lean_object* v_k_3307_ = _args[14];
lean_object* v_h_u2081_3308_ = _args[15];
lean_object* v_h_u2082_3309_ = _args[16];
_start:
{
lean_object* v_res_3310_; 
v_res_3310_ = lp_mathlib_MulSemiringActionHom_inverse_x27(v_M_3293_, v_inst_3294_, v_N_3295_, v_inst_3296_, v_00_u03c6_3297_, v_00_u03c6_x27_3298_, v_R_3299_, v_inst_3300_, v_inst_3301_, v_S_3302_, v_inst_3303_, v_inst_3304_, v_f_3305_, v_g_3306_, v_k_3307_, v_h_u2081_3308_, v_h_u2082_3309_);
lean_dec(v_g_3306_);
lean_dec(v_f_3305_);
lean_dec(v_inst_3304_);
lean_dec_ref(v_inst_3303_);
lean_dec(v_inst_3301_);
lean_dec_ref(v_inst_3300_);
lean_dec(v_00_u03c6_x27_3298_);
lean_dec(v_00_u03c6_3297_);
lean_dec_ref(v_inst_3296_);
lean_dec_ref(v_inst_3294_);
return v_res_3310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse___redArg(lean_object* v_g_3311_){
_start:
{
lean_inc(v_g_3311_);
return v_g_3311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse___redArg___boxed(lean_object* v_g_3312_){
_start:
{
lean_object* v_res_3313_; 
v_res_3313_ = lp_mathlib_MulSemiringActionHom_inverse___redArg(v_g_3312_);
lean_dec(v_g_3312_);
return v_res_3313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse(lean_object* v_M_3314_, lean_object* v_inst_3315_, lean_object* v_R_3316_, lean_object* v_inst_3317_, lean_object* v_inst_3318_, lean_object* v_S_u2081_3319_, lean_object* v_inst_3320_, lean_object* v_inst_3321_, lean_object* v_f_3322_, lean_object* v_g_3323_, lean_object* v_h_u2081_3324_, lean_object* v_h_u2082_3325_){
_start:
{
lean_inc(v_g_3323_);
return v_g_3323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringActionHom_inverse___boxed(lean_object* v_M_3326_, lean_object* v_inst_3327_, lean_object* v_R_3328_, lean_object* v_inst_3329_, lean_object* v_inst_3330_, lean_object* v_S_u2081_3331_, lean_object* v_inst_3332_, lean_object* v_inst_3333_, lean_object* v_f_3334_, lean_object* v_g_3335_, lean_object* v_h_u2081_3336_, lean_object* v_h_u2082_3337_){
_start:
{
lean_object* v_res_3338_; 
v_res_3338_ = lp_mathlib_MulSemiringActionHom_inverse(v_M_3326_, v_inst_3327_, v_R_3328_, v_inst_3329_, v_inst_3330_, v_S_u2081_3331_, v_inst_3332_, v_inst_3333_, v_f_3334_, v_g_3335_, v_h_u2081_3336_, v_h_u2082_3337_);
lean_dec(v_g_3335_);
lean_dec(v_f_3334_);
lean_dec(v_inst_3333_);
lean_dec_ref(v_inst_3332_);
lean_dec(v_inst_3330_);
lean_dec_ref(v_inst_3329_);
lean_dec_ref(v_inst_3327_);
return v_res_3338_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_CompTypeclasses(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_SMul(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_CompTypeclasses(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_SMul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
