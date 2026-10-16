// Lean compiler output
// Module: Mathlib.Algebra.Module.Equiv.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.LinearMap.Defs
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toAddEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 12, .m_data = "term_≃ₛₗ[_]_"};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 229, 67, 69, 118, 47, 161, 4)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 5, .m_data = " ≃ₛₗ["};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__9_value;
static const lean_string_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__10_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__10_value)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__11 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__11_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__9_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__12 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__12_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__12_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__13 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__13_value;
static const lean_ctor_object lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__13_value)}};
static const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__14 = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_u209b_u2097_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearEquiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(232, 132, 244, 142, 203, 20, 27, 167)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 11, .m_data = "term_≃ₗ[_]_"};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 118, 126, 160, 179, 186, 235, 176)}};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ≃ₗ["};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_u2097_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__7_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__11_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__11_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__13_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__9_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__13_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingHom.id"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__15_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "RingHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__17_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__18_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(77, 63, 71, 191, 78, 103, 81, 221)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearEquivClass_semilinearEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearEquivClass_semilinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearEquivClass_semilinearEquiv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv_instCoeLinearMap___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_instCoeLinearMap___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_instCoeLinearMap___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_refl___closed__0;
static const lean_closure_object lp_mathlib_LinearEquiv_refl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_refl___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv_refl___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_refl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_refl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_trans___boxed(lean_object**);
static const lean_string_object lp_mathlib_LinearEquiv_transNotation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "transNotation"};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__0_value;
static const lean_ctor_object lp_mathlib_LinearEquiv_transNotation___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(232, 132, 244, 142, 203, 20, 27, 167)}};
static const lean_ctor_object lp_mathlib_LinearEquiv_transNotation___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__1_value_aux_0),((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(58, 119, 52, 128, 52, 21, 93, 204)}};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__1_value;
static const lean_string_object lp_mathlib_LinearEquiv_transNotation___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 5, .m_data = " ≪≫ₗ "};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__2 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__2_value;
static const lean_ctor_object lp_mathlib_LinearEquiv_transNotation___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__2_value)}};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__3 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__3_value;
static const lean_ctor_object lp_mathlib_LinearEquiv_transNotation___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(81) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__4 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__4_value;
static const lean_ctor_object lp_mathlib_LinearEquiv_transNotation___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__3_value),((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__4_value)}};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__5 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__5_value;
static const lean_ctor_object lp_mathlib_LinearEquiv_transNotation___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__1_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__5_value)}};
static const lean_object* lp_mathlib_LinearEquiv_transNotation___closed__6 = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_LinearEquiv_transNotation = (const lean_object*)&lp_mathlib_LinearEquiv_transNotation___closed__6_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__0_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__2 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__2_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "LinearEquiv.trans"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__3 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__3_value;
static lean_once_cell_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__4;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__5 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__5_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(232, 132, 244, 142, 203, 20, 27, 167)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 213, 211, 55, 32, 203, 36, 223)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__7 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__7_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__8 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__8_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__9 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__9_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__11 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__11_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__9_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__13_value)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__12 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__12_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "RingHomCompTriple.ids"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__13 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__13_value;
static lean_once_cell_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__14;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "RingHomCompTriple"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__15 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__15_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ids"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__16 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__16_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(214, 89, 23, 247, 178, 105, 253, 75)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(201, 105, 57, 30, 169, 30, 220, 222)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__18 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__18_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__19 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__19_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "RingHomInvPair.ids"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__20 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__20_value;
static lean_once_cell_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__21;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "RingHomInvPair"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__22 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__22_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(80, 176, 233, 176, 55, 199, 114, 117)}};
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(31, 110, 1, 46, 204, 240, 234, 159)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__24 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__24_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__25 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__25_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "e₁"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(31, 65, 22, 12, 232, 25, 215, 199)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__1_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__1_value)} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__2 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__2_value;
static const lean_string_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "e₂"};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__3 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__3_value;
static const lean_ctor_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(18, 249, 199, 200, 73, 162, 180, 193)}};
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__4 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__4_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__4_value)} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__5 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__0_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__1_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__2 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__2_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__3___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__3 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__3_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___boxed, .m_arity = 11, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__0_value),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__3_value),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__1_value),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__2_value)} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__4 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__4_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__5 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__5_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__6 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__6_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(31) << 1) | 1)),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__4_value)} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__7 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__7_value;
static const lean_closure_object lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__6_value),((lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__7_value)} };
static const lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__8 = (const lean_object*)&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symmEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symmEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_cast___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_cast___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toAddEquiv___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toLinearMap_2_; lean_object* v_invFun_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_toLinearMap_2_ = lean_ctor_get(v_self_1_, 0);
v_invFun_3_ = lean_ctor_get(v_self_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_self_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_invFun_3_);
lean_inc(v_toLinearMap_2_);
lean_dec(v_self_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_8_; 
if (v_isShared_6_ == 0)
{
v___x_8_ = v___x_5_;
goto v_reusejp_7_;
}
else
{
lean_object* v_reuseFailAlloc_9_; 
v_reuseFailAlloc_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_toLinearMap_2_);
lean_ctor_set(v_reuseFailAlloc_9_, 1, v_invFun_3_);
v___x_8_ = v_reuseFailAlloc_9_;
goto v_reusejp_7_;
}
v_reusejp_7_:
{
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toAddEquiv(lean_object* v_R_11_, lean_object* v_S_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_00_u03c3_15_, lean_object* v_00_u03c3_x27_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_M_19_, lean_object* v_M_u2082_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_self_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_self_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toAddEquiv___boxed(lean_object* v_R_27_, lean_object* v_S_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_00_u03c3_31_, lean_object* v_00_u03c3_x27_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_M_35_, lean_object* v_M_u2082_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_self_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_LinearEquiv_toAddEquiv(v_R_27_, v_S_28_, v_inst_29_, v_inst_30_, v_00_u03c3_31_, v_00_u03c3_x27_32_, v_inst_33_, v_inst_34_, v_M_35_, v_M_u2082_36_, v_inst_37_, v_inst_38_, v_inst_39_, v_inst_40_, v_self_41_);
lean_dec(v_inst_40_);
lean_dec(v_inst_39_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
lean_dec(v_00_u03c3_x27_32_);
lean_dec(v_00_u03c3_31_);
lean_dec_ref(v_inst_30_);
lean_dec_ref(v_inst_29_);
return v_res_42_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__5));
v___x_90_ = l_String_toRawSubstring_x27(v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1(lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_){
_start:
{
lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__1));
lean_inc(v_x_107_);
v___x_111_ = l_Lean_Syntax_isOfKind(v_x_107_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v_x_107_);
v___x_112_ = lean_box(1);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_109_);
return v___x_113_;
}
else
{
lean_object* v_quotContext_114_; lean_object* v_currMacroScope_115_; lean_object* v_ref_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v_quotContext_114_ = lean_ctor_get(v_a_108_, 1);
v_currMacroScope_115_ = lean_ctor_get(v_a_108_, 2);
v_ref_116_ = lean_ctor_get(v_a_108_, 5);
v___x_117_ = lean_unsigned_to_nat(0u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_107_, v___x_117_);
v___x_119_ = lean_unsigned_to_nat(2u);
v___x_120_ = l_Lean_Syntax_getArg(v_x_107_, v___x_119_);
v___x_121_ = lean_unsigned_to_nat(4u);
v___x_122_ = l_Lean_Syntax_getArg(v_x_107_, v___x_121_);
lean_dec(v_x_107_);
v___x_123_ = 0;
v___x_124_ = l_Lean_SourceInfo_fromRef(v_ref_116_, v___x_123_);
v___x_125_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4));
v___x_126_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6);
v___x_127_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7));
lean_inc(v_currMacroScope_115_);
lean_inc(v_quotContext_114_);
v___x_128_ = l_Lean_addMacroScope(v_quotContext_114_, v___x_127_, v_currMacroScope_115_);
v___x_129_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__11));
lean_inc_n(v___x_124_, 2);
v___x_130_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_130_, 0, v___x_124_);
lean_ctor_set(v___x_130_, 1, v___x_126_);
lean_ctor_set(v___x_130_, 2, v___x_128_);
lean_ctor_set(v___x_130_, 3, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__13));
v___x_132_ = l_Lean_Syntax_node3(v___x_124_, v___x_131_, v___x_120_, v___x_118_, v___x_122_);
v___x_133_ = l_Lean_Syntax_node2(v___x_124_, v___x_125_, v___x_130_, v___x_132_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_109_);
return v___x_134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___boxed(lean_object* v_x_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1(v_x_135_, v_a_136_, v_a_137_);
lean_dec_ref(v_a_136_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1(lean_object* v_x_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v___x_145_; uint8_t v___x_146_; 
v___x_145_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4));
lean_inc(v_x_142_);
v___x_146_ = l_Lean_Syntax_isOfKind(v_x_142_, v___x_145_);
if (v___x_146_ == 0)
{
lean_object* v___x_147_; lean_object* v___x_148_; 
lean_dec(v_x_142_);
v___x_147_ = lean_box(0);
v___x_148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v_a_144_);
return v___x_148_;
}
else
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_149_ = lean_unsigned_to_nat(0u);
v___x_150_ = l_Lean_Syntax_getArg(v_x_142_, v___x_149_);
v___x_151_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__1));
lean_inc(v___x_150_);
v___x_152_ = l_Lean_Syntax_isOfKind(v___x_150_, v___x_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v___x_154_; 
lean_dec(v___x_150_);
lean_dec(v_x_142_);
v___x_153_ = lean_box(0);
v___x_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_144_);
return v___x_154_;
}
else
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_155_ = lean_unsigned_to_nat(1u);
v___x_156_ = l_Lean_Syntax_getArg(v_x_142_, v___x_155_);
lean_dec(v_x_142_);
v___x_157_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_156_);
v___x_158_ = l_Lean_Syntax_matchesNull(v___x_156_, v___x_157_);
if (v___x_158_ == 0)
{
lean_object* v___x_159_; lean_object* v___x_160_; 
lean_dec(v___x_156_);
lean_dec(v___x_150_);
v___x_159_ = lean_box(0);
v___x_160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v_a_144_);
return v___x_160_;
}
else
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v_ref_165_; uint8_t v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_161_ = l_Lean_Syntax_getArg(v___x_156_, v___x_149_);
v___x_162_ = l_Lean_Syntax_getArg(v___x_156_, v___x_155_);
v___x_163_ = lean_unsigned_to_nat(2u);
v___x_164_ = l_Lean_Syntax_getArg(v___x_156_, v___x_163_);
lean_dec(v___x_156_);
v_ref_165_ = l_Lean_replaceRef(v___x_150_, v_a_143_);
lean_dec(v___x_150_);
v___x_166_ = 0;
v___x_167_ = l_Lean_SourceInfo_fromRef(v_ref_165_, v___x_166_);
lean_dec(v_ref_165_);
v___x_168_ = ((lean_object*)(lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__1));
v___x_169_ = ((lean_object*)(lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__4));
lean_inc_n(v___x_167_, 2);
v___x_170_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_167_);
lean_ctor_set(v___x_170_, 1, v___x_169_);
v___x_171_ = ((lean_object*)(lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__10));
v___x_172_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_167_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = l_Lean_Syntax_node5(v___x_167_, v___x_168_, v___x_162_, v___x_170_, v___x_161_, v___x_172_, v___x_164_);
v___x_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_a_144_);
return v___x_174_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___boxed(lean_object* v_x_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1(v_x_175_, v_a_176_, v_a_177_);
lean_dec(v_a_176_);
return v_res_178_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__7));
v___x_221_ = l_String_toRawSubstring_x27(v___x_220_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__15));
v___x_237_ = l_String_toRawSubstring_x27(v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1(lean_object* v_x_250_, lean_object* v_a_251_, lean_object* v_a_252_){
_start:
{
lean_object* v___x_253_; uint8_t v___x_254_; 
v___x_253_ = ((lean_object*)(lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__1));
lean_inc(v_x_250_);
v___x_254_ = l_Lean_Syntax_isOfKind(v_x_250_, v___x_253_);
if (v___x_254_ == 0)
{
lean_object* v___x_255_; lean_object* v___x_256_; 
lean_dec(v_x_250_);
v___x_255_ = lean_box(1);
v___x_256_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v_a_252_);
return v___x_256_;
}
else
{
lean_object* v_quotContext_257_; lean_object* v_currMacroScope_258_; lean_object* v_ref_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; uint8_t v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v_quotContext_257_ = lean_ctor_get(v_a_251_, 1);
v_currMacroScope_258_ = lean_ctor_get(v_a_251_, 2);
v_ref_259_ = lean_ctor_get(v_a_251_, 5);
v___x_260_ = lean_unsigned_to_nat(0u);
v___x_261_ = l_Lean_Syntax_getArg(v_x_250_, v___x_260_);
v___x_262_ = lean_unsigned_to_nat(2u);
v___x_263_ = l_Lean_Syntax_getArg(v_x_250_, v___x_262_);
v___x_264_ = lean_unsigned_to_nat(4u);
v___x_265_ = l_Lean_Syntax_getArg(v_x_250_, v___x_264_);
lean_dec(v_x_250_);
v___x_266_ = 0;
v___x_267_ = l_Lean_SourceInfo_fromRef(v_ref_259_, v___x_266_);
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4));
v___x_269_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__6);
v___x_270_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__7));
lean_inc_n(v_currMacroScope_258_, 3);
lean_inc_n(v_quotContext_257_, 3);
v___x_271_ = l_Lean_addMacroScope(v_quotContext_257_, v___x_270_, v_currMacroScope_258_);
v___x_272_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__11));
lean_inc_n(v___x_267_, 11);
v___x_273_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_273_, 0, v___x_267_);
lean_ctor_set(v___x_273_, 1, v___x_269_);
lean_ctor_set(v___x_273_, 2, v___x_271_);
lean_ctor_set(v___x_273_, 3, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__13));
v___x_275_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1));
v___x_276_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3));
v___x_277_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__4));
v___x_278_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_278_, 0, v___x_267_);
lean_ctor_set(v___x_278_, 1, v___x_277_);
v___x_279_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__6));
v___x_280_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8, &lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8);
v___x_281_ = lean_box(0);
v___x_282_ = l_Lean_addMacroScope(v_quotContext_257_, v___x_281_, v_currMacroScope_258_);
v___x_283_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__14));
v___x_284_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_284_, 0, v___x_267_);
lean_ctor_set(v___x_284_, 1, v___x_280_);
lean_ctor_set(v___x_284_, 2, v___x_282_);
lean_ctor_set(v___x_284_, 3, v___x_283_);
v___x_285_ = l_Lean_Syntax_node1(v___x_267_, v___x_279_, v___x_284_);
v___x_286_ = l_Lean_Syntax_node2(v___x_267_, v___x_276_, v___x_278_, v___x_285_);
v___x_287_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16, &lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16);
v___x_288_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19));
v___x_289_ = l_Lean_addMacroScope(v_quotContext_257_, v___x_288_, v_currMacroScope_258_);
v___x_290_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__21));
v___x_291_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_291_, 0, v___x_267_);
lean_ctor_set(v___x_291_, 1, v___x_287_);
lean_ctor_set(v___x_291_, 2, v___x_289_);
lean_ctor_set(v___x_291_, 3, v___x_290_);
v___x_292_ = l_Lean_Syntax_node1(v___x_267_, v___x_274_, v___x_263_);
v___x_293_ = l_Lean_Syntax_node2(v___x_267_, v___x_268_, v___x_291_, v___x_292_);
v___x_294_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__22));
v___x_295_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_295_, 0, v___x_267_);
lean_ctor_set(v___x_295_, 1, v___x_294_);
v___x_296_ = l_Lean_Syntax_node3(v___x_267_, v___x_275_, v___x_286_, v___x_293_, v___x_295_);
v___x_297_ = l_Lean_Syntax_node3(v___x_267_, v___x_274_, v___x_296_, v___x_261_, v___x_265_);
v___x_298_ = l_Lean_Syntax_node2(v___x_267_, v___x_268_, v___x_273_, v___x_297_);
v___x_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v_a_252_);
return v___x_299_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___boxed(lean_object* v_x_300_, lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1(v_x_300_, v_a_301_, v_a_302_);
lean_dec_ref(v_a_301_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__2(lean_object* v_x_304_, lean_object* v_a_305_, lean_object* v_a_306_){
_start:
{
lean_object* v___x_307_; uint8_t v___x_308_; 
v___x_307_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4));
lean_inc(v_x_304_);
v___x_308_ = l_Lean_Syntax_isOfKind(v_x_304_, v___x_307_);
if (v___x_308_ == 0)
{
lean_object* v___x_309_; lean_object* v___x_310_; 
lean_dec(v_x_304_);
v___x_309_ = lean_box(0);
v___x_310_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v_a_306_);
return v___x_310_;
}
else
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v___x_311_ = lean_unsigned_to_nat(0u);
v___x_312_ = l_Lean_Syntax_getArg(v_x_304_, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__1___closed__1));
lean_inc(v___x_312_);
v___x_314_ = l_Lean_Syntax_isOfKind(v___x_312_, v___x_313_);
if (v___x_314_ == 0)
{
lean_object* v___x_315_; lean_object* v___x_316_; 
lean_dec(v___x_312_);
lean_dec(v_x_304_);
v___x_315_ = lean_box(0);
v___x_316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v_a_306_);
return v___x_316_;
}
else
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; uint8_t v___x_320_; 
v___x_317_ = lean_unsigned_to_nat(1u);
v___x_318_ = l_Lean_Syntax_getArg(v_x_304_, v___x_317_);
lean_dec(v_x_304_);
v___x_319_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_318_);
v___x_320_ = l_Lean_Syntax_matchesNull(v___x_318_, v___x_319_);
if (v___x_320_ == 0)
{
lean_object* v___x_321_; lean_object* v___x_322_; 
lean_dec(v___x_318_);
lean_dec(v___x_312_);
v___x_321_ = lean_box(0);
v___x_322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_321_);
lean_ctor_set(v___x_322_, 1, v_a_306_);
return v___x_322_;
}
else
{
lean_object* v___x_323_; uint8_t v___x_324_; 
v___x_323_ = l_Lean_Syntax_getArg(v___x_318_, v___x_311_);
lean_inc(v___x_323_);
v___x_324_ = l_Lean_Syntax_isOfKind(v___x_323_, v___x_307_);
if (v___x_324_ == 0)
{
lean_object* v___x_325_; lean_object* v___x_326_; 
lean_dec(v___x_323_);
lean_dec(v___x_318_);
lean_dec(v___x_312_);
v___x_325_ = lean_box(0);
v___x_326_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
lean_ctor_set(v___x_326_, 1, v_a_306_);
return v___x_326_;
}
else
{
lean_object* v___x_327_; lean_object* v___x_328_; uint8_t v___x_329_; 
v___x_327_ = l_Lean_Syntax_getArg(v___x_323_, v___x_311_);
v___x_328_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19));
v___x_329_ = l_Lean_Syntax_matchesIdent(v___x_327_, v___x_328_);
lean_dec(v___x_327_);
if (v___x_329_ == 0)
{
lean_object* v___x_330_; lean_object* v___x_331_; 
lean_dec(v___x_323_);
lean_dec(v___x_318_);
lean_dec(v___x_312_);
v___x_330_ = lean_box(0);
v___x_331_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
lean_ctor_set(v___x_331_, 1, v_a_306_);
return v___x_331_;
}
else
{
lean_object* v___x_332_; uint8_t v___x_333_; 
v___x_332_ = l_Lean_Syntax_getArg(v___x_323_, v___x_317_);
lean_dec(v___x_323_);
lean_inc(v___x_332_);
v___x_333_ = l_Lean_Syntax_matchesNull(v___x_332_, v___x_317_);
if (v___x_333_ == 0)
{
lean_object* v___x_334_; lean_object* v___x_335_; 
lean_dec(v___x_332_);
lean_dec(v___x_318_);
lean_dec(v___x_312_);
v___x_334_ = lean_box(0);
v___x_335_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_335_, 0, v___x_334_);
lean_ctor_set(v___x_335_, 1, v_a_306_);
return v___x_335_;
}
else
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_ref_340_; uint8_t v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_336_ = l_Lean_Syntax_getArg(v___x_332_, v___x_311_);
lean_dec(v___x_332_);
v___x_337_ = l_Lean_Syntax_getArg(v___x_318_, v___x_317_);
v___x_338_ = lean_unsigned_to_nat(2u);
v___x_339_ = l_Lean_Syntax_getArg(v___x_318_, v___x_338_);
lean_dec(v___x_318_);
v_ref_340_ = l_Lean_replaceRef(v___x_312_, v_a_305_);
lean_dec(v___x_312_);
v___x_341_ = 0;
v___x_342_ = l_Lean_SourceInfo_fromRef(v_ref_340_, v___x_341_);
lean_dec(v_ref_340_);
v___x_343_ = ((lean_object*)(lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__1));
v___x_344_ = ((lean_object*)(lp_mathlib_term___u2243_u2097_x5b___x5d___00__closed__2));
lean_inc_n(v___x_342_, 2);
v___x_345_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_342_);
lean_ctor_set(v___x_345_, 1, v___x_344_);
v___x_346_ = ((lean_object*)(lp_mathlib_term___u2243_u209b_u2097_x5b___x5d___00__closed__10));
v___x_347_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_342_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = l_Lean_Syntax_node5(v___x_342_, v___x_343_, v___x_337_, v___x_345_, v___x_336_, v___x_347_, v___x_339_);
v___x_349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_349_, 0, v___x_348_);
lean_ctor_set(v___x_349_, 1, v_a_306_);
return v___x_349_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__2___boxed(lean_object* v_x_350_, lean_object* v_a_351_, lean_object* v_a_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______unexpand__LinearEquiv__2(v_x_350_, v_a_351_, v_a_352_);
lean_dec(v_a_351_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearEquivClass_semilinearEquiv___redArg(lean_object* v_inst_354_, lean_object* v_f_355_){
_start:
{
lean_object* v___x_356_; lean_object* v_toFun_357_; lean_object* v_invFun_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_365_; 
v___x_356_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_354_, v_f_355_);
v_toFun_357_ = lean_ctor_get(v___x_356_, 0);
v_invFun_358_ = lean_ctor_get(v___x_356_, 1);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_356_);
if (v_isSharedCheck_365_ == 0)
{
v___x_360_ = v___x_356_;
v_isShared_361_ = v_isSharedCheck_365_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_invFun_358_);
lean_inc(v_toFun_357_);
lean_dec(v___x_356_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_365_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v___x_363_; 
if (v_isShared_361_ == 0)
{
v___x_363_ = v___x_360_;
goto v_reusejp_362_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v_toFun_357_);
lean_ctor_set(v_reuseFailAlloc_364_, 1, v_invFun_358_);
v___x_363_ = v_reuseFailAlloc_364_;
goto v_reusejp_362_;
}
v_reusejp_362_:
{
return v___x_363_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearEquivClass_semilinearEquiv(lean_object* v_R_366_, lean_object* v_S_367_, lean_object* v_M_368_, lean_object* v_M_u2082_369_, lean_object* v_F_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_00_u03c3_377_, lean_object* v_00_u03c3_x27_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_f_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_SemilinearEquivClass_semilinearEquiv___redArg(v_inst_381_, v_f_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearEquivClass_semilinearEquiv___boxed(lean_object** _args){
lean_object* v_R_385_ = _args[0];
lean_object* v_S_386_ = _args[1];
lean_object* v_M_387_ = _args[2];
lean_object* v_M_u2082_388_ = _args[3];
lean_object* v_F_389_ = _args[4];
lean_object* v_inst_390_ = _args[5];
lean_object* v_inst_391_ = _args[6];
lean_object* v_inst_392_ = _args[7];
lean_object* v_inst_393_ = _args[8];
lean_object* v_inst_394_ = _args[9];
lean_object* v_inst_395_ = _args[10];
lean_object* v_00_u03c3_396_ = _args[11];
lean_object* v_00_u03c3_x27_397_ = _args[12];
lean_object* v_inst_398_ = _args[13];
lean_object* v_inst_399_ = _args[14];
lean_object* v_inst_400_ = _args[15];
lean_object* v_inst_401_ = _args[16];
lean_object* v_f_402_ = _args[17];
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_SemilinearEquivClass_semilinearEquiv(v_R_385_, v_S_386_, v_M_387_, v_M_u2082_388_, v_F_389_, v_inst_390_, v_inst_391_, v_inst_392_, v_inst_393_, v_inst_394_, v_inst_395_, v_00_u03c3_396_, v_00_u03c3_x27_397_, v_inst_398_, v_inst_399_, v_inst_400_, v_inst_401_, v_f_402_);
lean_dec(v_00_u03c3_x27_397_);
lean_dec(v_00_u03c3_396_);
lean_dec(v_inst_395_);
lean_dec(v_inst_394_);
lean_dec_ref(v_inst_393_);
lean_dec_ref(v_inst_392_);
lean_dec_ref(v_inst_391_);
lean_dec_ref(v_inst_390_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap___lam__0(lean_object* v_self_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_toLinearMap_406_; lean_object* v___x_407_; 
v_toLinearMap_406_ = lean_ctor_get(v_self_404_, 0);
lean_inc(v_toLinearMap_406_);
lean_dec_ref(v_self_404_);
v___x_407_ = lean_apply_1(v_toLinearMap_406_, v___y_405_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap(lean_object* v_R_409_, lean_object* v_S_410_, lean_object* v_M_411_, lean_object* v_M_u2082_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_modM_417_, lean_object* v_modM_u2082_418_, lean_object* v_00_u03c3_419_, lean_object* v_00_u03c3_x27_420_, lean_object* v_inst_421_, lean_object* v_inst_422_){
_start:
{
lean_object* v___f_423_; 
v___f_423_ = ((lean_object*)(lp_mathlib_LinearEquiv_instCoeLinearMap___closed__0));
return v___f_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instCoeLinearMap___boxed(lean_object* v_R_424_, lean_object* v_S_425_, lean_object* v_M_426_, lean_object* v_M_u2082_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_modM_432_, lean_object* v_modM_u2082_433_, lean_object* v_00_u03c3_434_, lean_object* v_00_u03c3_x27_435_, lean_object* v_inst_436_, lean_object* v_inst_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_mathlib_LinearEquiv_instCoeLinearMap(v_R_424_, v_S_425_, v_M_426_, v_M_u2082_427_, v_inst_428_, v_inst_429_, v_inst_430_, v_inst_431_, v_modM_432_, v_modM_u2082_433_, v_00_u03c3_434_, v_00_u03c3_x27_435_, v_inst_436_, v_inst_437_);
lean_dec(v_00_u03c3_x27_435_);
lean_dec(v_00_u03c3_434_);
lean_dec(v_modM_u2082_433_);
lean_dec(v_modM_432_);
lean_dec_ref(v_inst_431_);
lean_dec_ref(v_inst_430_);
lean_dec_ref(v_inst_429_);
lean_dec_ref(v_inst_428_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toEquiv___redArg(lean_object* v_e_439_){
_start:
{
lean_object* v___x_440_; 
v___x_440_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_439_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toEquiv(lean_object* v_R_441_, lean_object* v_S_442_, lean_object* v_M_443_, lean_object* v_M_u2082_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_modM_449_, lean_object* v_modM_u2082_450_, lean_object* v_00_u03c3_451_, lean_object* v_00_u03c3_x27_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_e_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_toEquiv___boxed(lean_object* v_R_457_, lean_object* v_S_458_, lean_object* v_M_459_, lean_object* v_M_u2082_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_modM_465_, lean_object* v_modM_u2082_466_, lean_object* v_00_u03c3_467_, lean_object* v_00_u03c3_x27_468_, lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_e_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_LinearEquiv_toEquiv(v_R_457_, v_S_458_, v_M_459_, v_M_u2082_460_, v_inst_461_, v_inst_462_, v_inst_463_, v_inst_464_, v_modM_465_, v_modM_u2082_466_, v_00_u03c3_467_, v_00_u03c3_x27_468_, v_inst_469_, v_inst_470_, v_e_471_);
lean_dec(v_00_u03c3_x27_468_);
lean_dec(v_00_u03c3_467_);
lean_dec(v_modM_u2082_466_);
lean_dec(v_modM_465_);
lean_dec_ref(v_inst_464_);
lean_dec_ref(v_inst_463_);
lean_dec_ref(v_inst_462_);
lean_dec_ref(v_inst_461_);
return v_res_472_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_refl___closed__0(void){
_start:
{
lean_object* v___x_473_; 
v___x_473_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_refl(lean_object* v_R_475_, lean_object* v_M_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_){
_start:
{
lean_object* v___x_480_; lean_object* v_invFun_481_; lean_object* v___f_482_; lean_object* v___x_483_; 
v___x_480_ = lean_obj_once(&lp_mathlib_LinearEquiv_refl___closed__0, &lp_mathlib_LinearEquiv_refl___closed__0_once, _init_lp_mathlib_LinearEquiv_refl___closed__0);
v_invFun_481_ = lean_ctor_get(v___x_480_, 1);
v___f_482_ = ((lean_object*)(lp_mathlib_LinearEquiv_refl___closed__1));
lean_inc(v_invFun_481_);
v___x_483_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_483_, 0, v___f_482_);
lean_ctor_set(v___x_483_, 1, v_invFun_481_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_refl___boxed(lean_object* v_R_484_, lean_object* v_M_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_mathlib_LinearEquiv_refl(v_R_484_, v_M_485_, v_inst_486_, v_inst_487_, v_inst_488_);
lean_dec(v_inst_488_);
lean_dec_ref(v_inst_487_);
lean_dec_ref(v_inst_486_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm___redArg___lam__0(lean_object* v_invFun_490_, lean_object* v___y_491_){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lean_apply_1(v_invFun_490_, v___y_491_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object* v_e_493_){
_start:
{
lean_object* v_invFun_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v_invFun_497_; lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_505_; 
v_invFun_494_ = lean_ctor_get(v_e_493_, 1);
lean_inc(v_invFun_494_);
v___x_495_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_493_);
v___x_496_ = lp_mathlib_Equiv_symm___redArg(v___x_495_);
v_invFun_497_ = lean_ctor_get(v___x_496_, 1);
v_isSharedCheck_505_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_505_ == 0)
{
lean_object* v_unused_506_; 
v_unused_506_ = lean_ctor_get(v___x_496_, 0);
lean_dec(v_unused_506_);
v___x_499_ = v___x_496_;
v_isShared_500_ = v_isSharedCheck_505_;
goto v_resetjp_498_;
}
else
{
lean_inc(v_invFun_497_);
lean_dec(v___x_496_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_505_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
lean_object* v___f_501_; lean_object* v___x_503_; 
v___f_501_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_symm___redArg___lam__0), 2, 1);
lean_closure_set(v___f_501_, 0, v_invFun_494_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 0, v___f_501_);
v___x_503_ = v___x_499_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v___f_501_);
lean_ctor_set(v_reuseFailAlloc_504_, 1, v_invFun_497_);
v___x_503_ = v_reuseFailAlloc_504_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
return v___x_503_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm(lean_object* v_R_507_, lean_object* v_S_508_, lean_object* v_M_509_, lean_object* v_M_u2082_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_module__M_515_, lean_object* v_module__S__M_u2082_516_, lean_object* v_00_u03c3_517_, lean_object* v_00_u03c3_x27_518_, lean_object* v_re_u2081_519_, lean_object* v_re_u2082_520_, lean_object* v_e_521_){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symm___boxed(lean_object* v_R_523_, lean_object* v_S_524_, lean_object* v_M_525_, lean_object* v_M_u2082_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_module__M_531_, lean_object* v_module__S__M_u2082_532_, lean_object* v_00_u03c3_533_, lean_object* v_00_u03c3_x27_534_, lean_object* v_re_u2081_535_, lean_object* v_re_u2082_536_, lean_object* v_e_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_LinearEquiv_symm(v_R_523_, v_S_524_, v_M_525_, v_M_u2082_526_, v_inst_527_, v_inst_528_, v_inst_529_, v_inst_530_, v_module__M_531_, v_module__S__M_u2082_532_, v_00_u03c3_533_, v_00_u03c3_x27_534_, v_re_u2081_535_, v_re_u2082_536_, v_e_537_);
lean_dec(v_00_u03c3_x27_534_);
lean_dec(v_00_u03c3_533_);
lean_dec(v_module__S__M_u2082_532_);
lean_dec(v_module__M_531_);
lean_dec_ref(v_inst_530_);
lean_dec_ref(v_inst_529_);
lean_dec_ref(v_inst_528_);
lean_dec_ref(v_inst_527_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_apply___redArg(lean_object* v_e_539_, lean_object* v_a_540_){
_start:
{
lean_object* v_toLinearMap_541_; lean_object* v___x_542_; 
v_toLinearMap_541_ = lean_ctor_get(v_e_539_, 0);
lean_inc(v_toLinearMap_541_);
lean_dec_ref(v_e_539_);
v___x_542_ = lean_apply_1(v_toLinearMap_541_, v_a_540_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_apply(lean_object* v_R_543_, lean_object* v_S_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_00_u03c3_547_, lean_object* v_00_u03c3_x27_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_M_551_, lean_object* v_M_u2082_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_e_557_, lean_object* v_a_558_){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lp_mathlib_LinearEquiv_Simps_apply___redArg(v_e_557_, v_a_558_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_apply___boxed(lean_object* v_R_560_, lean_object* v_S_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_00_u03c3_564_, lean_object* v_00_u03c3_x27_565_, lean_object* v_inst_566_, lean_object* v_inst_567_, lean_object* v_M_568_, lean_object* v_M_u2082_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_e_574_, lean_object* v_a_575_){
_start:
{
lean_object* v_res_576_; 
v_res_576_ = lp_mathlib_LinearEquiv_Simps_apply(v_R_560_, v_S_561_, v_inst_562_, v_inst_563_, v_00_u03c3_564_, v_00_u03c3_x27_565_, v_inst_566_, v_inst_567_, v_M_568_, v_M_u2082_569_, v_inst_570_, v_inst_571_, v_inst_572_, v_inst_573_, v_e_574_, v_a_575_);
lean_dec(v_inst_573_);
lean_dec(v_inst_572_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
lean_dec(v_00_u03c3_x27_565_);
lean_dec(v_00_u03c3_564_);
lean_dec_ref(v_inst_563_);
lean_dec_ref(v_inst_562_);
return v_res_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_symm__apply___redArg(lean_object* v_e_577_, lean_object* v_a_578_){
_start:
{
lean_object* v___x_579_; lean_object* v_toLinearMap_580_; lean_object* v___x_581_; 
v___x_579_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_577_);
v_toLinearMap_580_ = lean_ctor_get(v___x_579_, 0);
lean_inc(v_toLinearMap_580_);
lean_dec_ref(v___x_579_);
v___x_581_ = lean_apply_1(v_toLinearMap_580_, v_a_578_);
return v___x_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_symm__apply(lean_object* v_R_582_, lean_object* v_S_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_00_u03c3_586_, lean_object* v_00_u03c3_x27_587_, lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_M_590_, lean_object* v_M_u2082_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_e_596_, lean_object* v_a_597_){
_start:
{
lean_object* v___x_598_; 
v___x_598_ = lp_mathlib_LinearEquiv_Simps_symm__apply___redArg(v_e_596_, v_a_597_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_Simps_symm__apply___boxed(lean_object* v_R_599_, lean_object* v_S_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_00_u03c3_603_, lean_object* v_00_u03c3_x27_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_M_607_, lean_object* v_M_u2082_608_, lean_object* v_inst_609_, lean_object* v_inst_610_, lean_object* v_inst_611_, lean_object* v_inst_612_, lean_object* v_e_613_, lean_object* v_a_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_mathlib_LinearEquiv_Simps_symm__apply(v_R_599_, v_S_600_, v_inst_601_, v_inst_602_, v_00_u03c3_603_, v_00_u03c3_x27_604_, v_inst_605_, v_inst_606_, v_M_607_, v_M_u2082_608_, v_inst_609_, v_inst_610_, v_inst_611_, v_inst_612_, v_e_613_, v_a_614_);
lean_dec(v_inst_612_);
lean_dec(v_inst_611_);
lean_dec_ref(v_inst_610_);
lean_dec_ref(v_inst_609_);
lean_dec(v_00_u03c3_x27_604_);
lean_dec(v_00_u03c3_603_);
lean_dec_ref(v_inst_602_);
lean_dec_ref(v_inst_601_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object* v_e_u2081_u2082_616_, lean_object* v_e_u2082_u2083_617_){
_start:
{
lean_object* v_toLinearMap_618_; lean_object* v_toLinearMap_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v_invFun_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_631_; 
v_toLinearMap_618_ = lean_ctor_get(v_e_u2082_u2083_617_, 0);
lean_inc(v_toLinearMap_618_);
v_toLinearMap_619_ = lean_ctor_get(v_e_u2081_u2082_616_, 0);
lean_inc(v_toLinearMap_619_);
v___x_620_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_u2081_u2082_616_);
v___x_621_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_u2082_u2083_617_);
v___x_622_ = lp_mathlib_Equiv_trans___redArg(v___x_620_, v___x_621_);
v_invFun_623_ = lean_ctor_get(v___x_622_, 1);
v_isSharedCheck_631_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_631_ == 0)
{
lean_object* v_unused_632_; 
v_unused_632_ = lean_ctor_get(v___x_622_, 0);
lean_dec(v_unused_632_);
v___x_625_ = v___x_622_;
v_isShared_626_ = v_isSharedCheck_631_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_invFun_623_);
lean_dec(v___x_622_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_631_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___f_627_; lean_object* v___x_629_; 
v___f_627_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_627_, 0, v_toLinearMap_619_);
lean_closure_set(v___f_627_, 1, v_toLinearMap_618_);
if (v_isShared_626_ == 0)
{
lean_ctor_set(v___x_625_, 0, v___f_627_);
v___x_629_ = v___x_625_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_630_; 
v_reuseFailAlloc_630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_630_, 0, v___f_627_);
lean_ctor_set(v_reuseFailAlloc_630_, 1, v_invFun_623_);
v___x_629_ = v_reuseFailAlloc_630_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
return v___x_629_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_trans(lean_object* v_R_u2081_633_, lean_object* v_R_u2082_634_, lean_object* v_R_u2083_635_, lean_object* v_M_u2081_636_, lean_object* v_M_u2082_637_, lean_object* v_M_u2083_638_, lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_module__M_u2081_645_, lean_object* v_module__M_u2082_646_, lean_object* v_module__M_u2083_647_, lean_object* v_00_u03c3_u2081_u2082_648_, lean_object* v_00_u03c3_u2082_u2081_649_, lean_object* v_00_u03c3_u2081_u2083_650_, lean_object* v_00_u03c3_u2083_u2081_651_, lean_object* v_00_u03c3_u2082_u2083_652_, lean_object* v_00_u03c3_u2083_u2082_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_re_u2081_u2082_656_, lean_object* v_re_u2082_u2083_657_, lean_object* v_inst_658_, lean_object* v_re_u2082_u2081_659_, lean_object* v_re_u2083_u2082_660_, lean_object* v_inst_661_, lean_object* v_e_u2081_u2082_662_, lean_object* v_e_u2082_u2083_663_){
_start:
{
lean_object* v___x_664_; 
v___x_664_ = lp_mathlib_LinearEquiv_trans___redArg(v_e_u2081_u2082_662_, v_e_u2082_u2083_663_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_trans___boxed(lean_object** _args){
lean_object* v_R_u2081_665_ = _args[0];
lean_object* v_R_u2082_666_ = _args[1];
lean_object* v_R_u2083_667_ = _args[2];
lean_object* v_M_u2081_668_ = _args[3];
lean_object* v_M_u2082_669_ = _args[4];
lean_object* v_M_u2083_670_ = _args[5];
lean_object* v_inst_671_ = _args[6];
lean_object* v_inst_672_ = _args[7];
lean_object* v_inst_673_ = _args[8];
lean_object* v_inst_674_ = _args[9];
lean_object* v_inst_675_ = _args[10];
lean_object* v_inst_676_ = _args[11];
lean_object* v_module__M_u2081_677_ = _args[12];
lean_object* v_module__M_u2082_678_ = _args[13];
lean_object* v_module__M_u2083_679_ = _args[14];
lean_object* v_00_u03c3_u2081_u2082_680_ = _args[15];
lean_object* v_00_u03c3_u2082_u2081_681_ = _args[16];
lean_object* v_00_u03c3_u2081_u2083_682_ = _args[17];
lean_object* v_00_u03c3_u2083_u2081_683_ = _args[18];
lean_object* v_00_u03c3_u2082_u2083_684_ = _args[19];
lean_object* v_00_u03c3_u2083_u2082_685_ = _args[20];
lean_object* v_inst_686_ = _args[21];
lean_object* v_inst_687_ = _args[22];
lean_object* v_re_u2081_u2082_688_ = _args[23];
lean_object* v_re_u2082_u2083_689_ = _args[24];
lean_object* v_inst_690_ = _args[25];
lean_object* v_re_u2082_u2081_691_ = _args[26];
lean_object* v_re_u2083_u2082_692_ = _args[27];
lean_object* v_inst_693_ = _args[28];
lean_object* v_e_u2081_u2082_694_ = _args[29];
lean_object* v_e_u2082_u2083_695_ = _args[30];
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_mathlib_LinearEquiv_trans(v_R_u2081_665_, v_R_u2082_666_, v_R_u2083_667_, v_M_u2081_668_, v_M_u2082_669_, v_M_u2083_670_, v_inst_671_, v_inst_672_, v_inst_673_, v_inst_674_, v_inst_675_, v_inst_676_, v_module__M_u2081_677_, v_module__M_u2082_678_, v_module__M_u2083_679_, v_00_u03c3_u2081_u2082_680_, v_00_u03c3_u2082_u2081_681_, v_00_u03c3_u2081_u2083_682_, v_00_u03c3_u2083_u2081_683_, v_00_u03c3_u2082_u2083_684_, v_00_u03c3_u2083_u2082_685_, v_inst_686_, v_inst_687_, v_re_u2081_u2082_688_, v_re_u2082_u2083_689_, v_inst_690_, v_re_u2082_u2081_691_, v_re_u2083_u2082_692_, v_inst_693_, v_e_u2081_u2082_694_, v_e_u2082_u2083_695_);
lean_dec(v_00_u03c3_u2083_u2082_685_);
lean_dec(v_00_u03c3_u2082_u2083_684_);
lean_dec(v_00_u03c3_u2083_u2081_683_);
lean_dec(v_00_u03c3_u2081_u2083_682_);
lean_dec(v_00_u03c3_u2082_u2081_681_);
lean_dec(v_00_u03c3_u2081_u2082_680_);
lean_dec(v_module__M_u2083_679_);
lean_dec(v_module__M_u2082_678_);
lean_dec(v_module__M_u2081_677_);
lean_dec_ref(v_inst_676_);
lean_dec_ref(v_inst_675_);
lean_dec_ref(v_inst_674_);
lean_dec_ref(v_inst_673_);
lean_dec_ref(v_inst_672_);
lean_dec_ref(v_inst_671_);
return v_res_696_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__4(void){
_start:
{
lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_724_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__3));
v___x_725_ = l_String_toRawSubstring_x27(v___x_724_);
return v___x_725_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__14(void){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_747_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__13));
v___x_748_ = l_String_toRawSubstring_x27(v___x_747_);
return v___x_748_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__21(void){
_start:
{
lean_object* v___x_761_; lean_object* v___x_762_; 
v___x_761_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__20));
v___x_762_ = l_String_toRawSubstring_x27(v___x_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1(lean_object* v_x_773_, lean_object* v_a_774_, lean_object* v_a_775_){
_start:
{
lean_object* v___x_776_; uint8_t v___x_777_; 
v___x_776_ = ((lean_object*)(lp_mathlib_LinearEquiv_transNotation___closed__1));
lean_inc(v_x_773_);
v___x_777_ = l_Lean_Syntax_isOfKind(v_x_773_, v___x_776_);
if (v___x_777_ == 0)
{
lean_object* v___x_778_; lean_object* v___x_779_; 
lean_dec(v_x_773_);
v___x_778_ = lean_box(1);
v___x_779_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_779_, 0, v___x_778_);
lean_ctor_set(v___x_779_, 1, v_a_775_);
return v___x_779_;
}
else
{
lean_object* v_quotContext_780_; lean_object* v_currMacroScope_781_; lean_object* v_ref_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; uint8_t v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; 
v_quotContext_780_ = lean_ctor_get(v_a_774_, 1);
v_currMacroScope_781_ = lean_ctor_get(v_a_774_, 2);
v_ref_782_ = lean_ctor_get(v_a_774_, 5);
v___x_783_ = lean_unsigned_to_nat(0u);
v___x_784_ = l_Lean_Syntax_getArg(v_x_773_, v___x_783_);
v___x_785_ = lean_unsigned_to_nat(2u);
v___x_786_ = l_Lean_Syntax_getArg(v_x_773_, v___x_785_);
lean_dec(v_x_773_);
v___x_787_ = 0;
v___x_788_ = l_Lean_SourceInfo_fromRef(v_ref_782_, v___x_787_);
v___x_789_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__4));
v___x_790_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__1));
v___x_791_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__2));
lean_inc_n(v___x_788_, 17);
v___x_792_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_792_, 0, v___x_788_);
lean_ctor_set(v___x_792_, 1, v___x_791_);
v___x_793_ = lean_obj_once(&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__4, &lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__4_once, _init_lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__4);
v___x_794_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6));
lean_inc_n(v_currMacroScope_781_, 5);
lean_inc_n(v_quotContext_780_, 5);
v___x_795_ = l_Lean_addMacroScope(v_quotContext_780_, v___x_794_, v_currMacroScope_781_);
v___x_796_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__8));
v___x_797_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_797_, 0, v___x_788_);
lean_ctor_set(v___x_797_, 1, v___x_793_);
lean_ctor_set(v___x_797_, 2, v___x_795_);
lean_ctor_set(v___x_797_, 3, v___x_796_);
v___x_798_ = l_Lean_Syntax_node2(v___x_788_, v___x_790_, v___x_792_, v___x_797_);
v___x_799_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u209b_u2097_x5b___x5d____1___closed__13));
v___x_800_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__10));
v___x_801_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__11));
v___x_802_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_802_, 0, v___x_788_);
lean_ctor_set(v___x_802_, 1, v___x_801_);
v___x_803_ = l_Lean_Syntax_node1(v___x_788_, v___x_800_, v___x_802_);
v___x_804_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__1));
v___x_805_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__3));
v___x_806_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__4));
v___x_807_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_807_, 0, v___x_788_);
lean_ctor_set(v___x_807_, 1, v___x_806_);
v___x_808_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__6));
v___x_809_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8, &lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__8);
v___x_810_ = lean_box(0);
v___x_811_ = l_Lean_addMacroScope(v_quotContext_780_, v___x_810_, v_currMacroScope_781_);
v___x_812_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__12));
v___x_813_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_813_, 0, v___x_788_);
lean_ctor_set(v___x_813_, 1, v___x_809_);
lean_ctor_set(v___x_813_, 2, v___x_811_);
lean_ctor_set(v___x_813_, 3, v___x_812_);
v___x_814_ = l_Lean_Syntax_node1(v___x_788_, v___x_808_, v___x_813_);
v___x_815_ = l_Lean_Syntax_node2(v___x_788_, v___x_805_, v___x_807_, v___x_814_);
v___x_816_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16, &lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__16);
v___x_817_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19));
v___x_818_ = l_Lean_addMacroScope(v_quotContext_780_, v___x_817_, v_currMacroScope_781_);
v___x_819_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__21));
v___x_820_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_820_, 0, v___x_788_);
lean_ctor_set(v___x_820_, 1, v___x_816_);
lean_ctor_set(v___x_820_, 2, v___x_818_);
lean_ctor_set(v___x_820_, 3, v___x_819_);
lean_inc_n(v___x_803_, 15);
v___x_821_ = l_Lean_Syntax_node1(v___x_788_, v___x_799_, v___x_803_);
v___x_822_ = l_Lean_Syntax_node2(v___x_788_, v___x_789_, v___x_820_, v___x_821_);
v___x_823_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__22));
v___x_824_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_824_, 0, v___x_788_);
lean_ctor_set(v___x_824_, 1, v___x_823_);
v___x_825_ = l_Lean_Syntax_node3(v___x_788_, v___x_804_, v___x_815_, v___x_822_, v___x_824_);
v___x_826_ = lean_obj_once(&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__14, &lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__14_once, _init_lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__14);
v___x_827_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__17));
v___x_828_ = l_Lean_addMacroScope(v_quotContext_780_, v___x_827_, v_currMacroScope_781_);
v___x_829_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__19));
v___x_830_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_830_, 0, v___x_788_);
lean_ctor_set(v___x_830_, 1, v___x_826_);
lean_ctor_set(v___x_830_, 2, v___x_828_);
lean_ctor_set(v___x_830_, 3, v___x_829_);
v___x_831_ = lean_obj_once(&lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__21, &lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__21_once, _init_lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__21);
v___x_832_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23));
v___x_833_ = l_Lean_addMacroScope(v_quotContext_780_, v___x_832_, v_currMacroScope_781_);
v___x_834_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__25));
v___x_835_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_835_, 0, v___x_788_);
lean_ctor_set(v___x_835_, 1, v___x_831_);
lean_ctor_set(v___x_835_, 2, v___x_833_);
lean_ctor_set(v___x_835_, 3, v___x_834_);
v___x_836_ = lean_unsigned_to_nat(31u);
v___x_837_ = lean_mk_empty_array_with_capacity(v___x_836_);
v___x_838_ = lean_array_push(v___x_837_, v___x_803_);
v___x_839_ = lean_array_push(v___x_838_, v___x_803_);
v___x_840_ = lean_array_push(v___x_839_, v___x_803_);
v___x_841_ = lean_array_push(v___x_840_, v___x_803_);
v___x_842_ = lean_array_push(v___x_841_, v___x_803_);
v___x_843_ = lean_array_push(v___x_842_, v___x_803_);
v___x_844_ = lean_array_push(v___x_843_, v___x_803_);
v___x_845_ = lean_array_push(v___x_844_, v___x_803_);
v___x_846_ = lean_array_push(v___x_845_, v___x_803_);
v___x_847_ = lean_array_push(v___x_846_, v___x_803_);
v___x_848_ = lean_array_push(v___x_847_, v___x_803_);
v___x_849_ = lean_array_push(v___x_848_, v___x_803_);
v___x_850_ = lean_array_push(v___x_849_, v___x_803_);
v___x_851_ = lean_array_push(v___x_850_, v___x_803_);
v___x_852_ = lean_array_push(v___x_851_, v___x_803_);
lean_inc_n(v___x_825_, 5);
v___x_853_ = lean_array_push(v___x_852_, v___x_825_);
v___x_854_ = lean_array_push(v___x_853_, v___x_825_);
v___x_855_ = lean_array_push(v___x_854_, v___x_825_);
v___x_856_ = lean_array_push(v___x_855_, v___x_825_);
v___x_857_ = lean_array_push(v___x_856_, v___x_825_);
v___x_858_ = lean_array_push(v___x_857_, v___x_825_);
lean_inc_ref(v___x_830_);
v___x_859_ = lean_array_push(v___x_858_, v___x_830_);
v___x_860_ = lean_array_push(v___x_859_, v___x_830_);
lean_inc_ref_n(v___x_835_, 5);
v___x_861_ = lean_array_push(v___x_860_, v___x_835_);
v___x_862_ = lean_array_push(v___x_861_, v___x_835_);
v___x_863_ = lean_array_push(v___x_862_, v___x_835_);
v___x_864_ = lean_array_push(v___x_863_, v___x_835_);
v___x_865_ = lean_array_push(v___x_864_, v___x_835_);
v___x_866_ = lean_array_push(v___x_865_, v___x_835_);
v___x_867_ = lean_array_push(v___x_866_, v___x_784_);
v___x_868_ = lean_array_push(v___x_867_, v___x_786_);
v___x_869_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_869_, 0, v___x_788_);
lean_ctor_set(v___x_869_, 1, v___x_799_);
lean_ctor_set(v___x_869_, 2, v___x_868_);
v___x_870_ = l_Lean_Syntax_node2(v___x_788_, v___x_789_, v___x_798_, v___x_869_);
v___x_871_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_871_, 0, v___x_870_);
lean_ctor_set(v___x_871_, 1, v_a_775_);
return v___x_871_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___boxed(lean_object* v_x_872_, lean_object* v_a_873_, lean_object* v_a_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1(v_x_872_, v_a_873_, v_a_874_);
lean_dec_ref(v_a_873_);
return v_res_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg(lean_object* v___y_876_){
_start:
{
lean_object* v_subExpr_878_; lean_object* v_expr_879_; lean_object* v___x_880_; 
v_subExpr_878_ = lean_ctor_get(v___y_876_, 3);
v_expr_879_ = lean_ctor_get(v_subExpr_878_, 0);
lean_inc_ref(v_expr_879_);
v___x_880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_880_, 0, v_expr_879_);
return v___x_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg___boxed(lean_object* v___y_881_, lean_object* v___y_882_){
_start:
{
lean_object* v_res_883_; 
v_res_883_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg(v___y_881_);
lean_dec_ref(v___y_881_);
return v_res_883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0(lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_){
_start:
{
lean_object* v___x_891_; 
v___x_891_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg(v___y_884_);
return v___x_891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___boxed(lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v_res_899_; 
v_res_899_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0(v___y_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
lean_dec(v___y_897_);
lean_dec_ref(v___y_896_);
lean_dec(v___y_895_);
lean_dec_ref(v___y_894_);
lean_dec(v___y_893_);
lean_dec_ref(v___y_892_);
return v_res_899_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__0(lean_object* v_x_900_){
_start:
{
lean_object* v___x_901_; uint8_t v___x_902_; 
v___x_901_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__6));
v___x_902_ = l_Lean_Expr_isConstOf(v_x_900_, v___x_901_);
return v___x_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__0___boxed(lean_object* v_x_903_){
_start:
{
uint8_t v_res_904_; lean_object* v_r_905_; 
v_res_904_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__0(v_x_903_);
lean_dec_ref(v_x_903_);
v_r_905_ = lean_box(v_res_904_);
return v_r_905_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__1(lean_object* v_x_906_){
_start:
{
lean_object* v___x_907_; uint8_t v___x_908_; 
v___x_907_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__term___u2243_u2097_x5b___x5d____1___closed__19));
v___x_908_ = l_Lean_Expr_isConstOf(v_x_906_, v___x_907_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__1___boxed(lean_object* v_x_909_){
_start:
{
uint8_t v_res_910_; lean_object* v_r_911_; 
v_res_910_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__1(v_x_909_);
lean_dec_ref(v_x_909_);
v_r_911_ = lean_box(v_res_910_);
return v_r_911_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__2(lean_object* v_x_912_){
_start:
{
lean_object* v___x_913_; uint8_t v___x_914_; 
v___x_913_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______macroRules__LinearEquiv__transNotation__1___closed__23));
v___x_914_ = l_Lean_Expr_isConstOf(v_x_912_, v___x_913_);
return v___x_914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__2___boxed(lean_object* v_x_915_){
_start:
{
uint8_t v_res_916_; lean_object* v_r_917_; 
v_res_916_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__2(v_x_915_);
lean_dec_ref(v_x_915_);
v_r_917_ = lean_box(v_res_916_);
return v_r_917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__3(lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
lean_object* v___x_926_; 
v___x_926_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_926_, 0, v___y_918_);
return v___x_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__3___boxed(lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_, lean_object* v___y_934_){
_start:
{
lean_object* v_res_935_; 
v_res_935_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__3(v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_, v___y_933_);
lean_dec(v___y_933_);
lean_dec_ref(v___y_932_);
lean_dec(v___y_931_);
lean_dec_ref(v___y_930_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
return v_res_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__4(lean_object* v_a_936_, lean_object* v_a_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_){
_start:
{
lean_object* v_ref_945_; uint8_t v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; 
v_ref_945_ = lean_ctor_get(v___y_942_, 5);
v___x_946_ = 0;
v___x_947_ = l_Lean_SourceInfo_fromRef(v_ref_945_, v___x_946_);
v___x_948_ = ((lean_object*)(lp_mathlib_LinearEquiv_transNotation___closed__1));
v___x_949_ = ((lean_object*)(lp_mathlib_LinearEquiv_transNotation___closed__2));
lean_inc(v___x_947_);
v___x_950_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_950_, 0, v___x_947_);
lean_ctor_set(v___x_950_, 1, v___x_949_);
v___x_951_ = l_Lean_Syntax_node3(v___x_947_, v___x_948_, v_a_936_, v___x_950_, v_a_937_);
v___x_952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_952_, 0, v___x_951_);
return v___x_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__4___boxed(lean_object* v_a_953_, lean_object* v_a_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__4(v_a_953_, v_a_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5(lean_object* v___f_973_, lean_object* v___f_974_, lean_object* v___f_975_, lean_object* v___f_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_){
_start:
{
lean_object* v___x_984_; lean_object* v_a_985_; lean_object* v___x_987_; uint8_t v_isShared_988_; uint8_t v_isSharedCheck_1050_; 
v___x_984_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1_spec__0___redArg(v___y_977_);
v_a_985_ = lean_ctor_get(v___x_984_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v___x_984_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_987_ = v___x_984_;
v_isShared_988_ = v_isSharedCheck_1050_;
goto v_resetjp_986_;
}
else
{
lean_inc(v_a_985_);
lean_dec(v___x_984_);
v___x_987_ = lean_box(0);
v_isShared_988_ = v_isSharedCheck_1050_;
goto v_resetjp_986_;
}
v_resetjp_986_:
{
lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_989_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_989_, 0, v___f_973_);
lean_inc_ref_n(v___f_974_, 22);
v___x_990_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_990_, 0, v___x_989_);
lean_closure_set(v___x_990_, 1, v___f_974_);
v___x_991_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_991_, 0, v___x_990_);
lean_closure_set(v___x_991_, 1, v___f_974_);
v___x_992_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_992_, 0, v___x_991_);
lean_closure_set(v___x_992_, 1, v___f_974_);
v___x_993_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_993_, 0, v___x_992_);
lean_closure_set(v___x_993_, 1, v___f_974_);
v___x_994_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_994_, 0, v___x_993_);
lean_closure_set(v___x_994_, 1, v___f_974_);
v___x_995_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_995_, 0, v___x_994_);
lean_closure_set(v___x_995_, 1, v___f_974_);
v___x_996_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_996_, 0, v___x_995_);
lean_closure_set(v___x_996_, 1, v___f_974_);
v___x_997_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_997_, 0, v___x_996_);
lean_closure_set(v___x_997_, 1, v___f_974_);
v___x_998_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_998_, 0, v___x_997_);
lean_closure_set(v___x_998_, 1, v___f_974_);
v___x_999_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_999_, 0, v___x_998_);
lean_closure_set(v___x_999_, 1, v___f_974_);
v___x_1000_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1000_, 0, v___x_999_);
lean_closure_set(v___x_1000_, 1, v___f_974_);
v___x_1001_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1001_, 0, v___x_1000_);
lean_closure_set(v___x_1001_, 1, v___f_974_);
v___x_1002_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1002_, 0, v___x_1001_);
lean_closure_set(v___x_1002_, 1, v___f_974_);
v___x_1003_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1003_, 0, v___x_1002_);
lean_closure_set(v___x_1003_, 1, v___f_974_);
v___x_1004_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1004_, 0, v___x_1003_);
lean_closure_set(v___x_1004_, 1, v___f_974_);
v___x_1005_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_1005_, 0, v___f_975_);
v___x_1006_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1006_, 0, v___x_1005_);
lean_closure_set(v___x_1006_, 1, v___f_974_);
v___x_1007_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1007_, 0, v___x_1006_);
lean_closure_set(v___x_1007_, 1, v___f_974_);
lean_inc_ref_n(v___x_1007_, 5);
v___x_1008_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1008_, 0, v___x_1004_);
lean_closure_set(v___x_1008_, 1, v___x_1007_);
v___x_1009_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1009_, 0, v___x_1008_);
lean_closure_set(v___x_1009_, 1, v___x_1007_);
v___x_1010_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1010_, 0, v___x_1009_);
lean_closure_set(v___x_1010_, 1, v___x_1007_);
v___x_1011_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1011_, 0, v___x_1010_);
lean_closure_set(v___x_1011_, 1, v___x_1007_);
v___x_1012_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1012_, 0, v___x_1011_);
lean_closure_set(v___x_1012_, 1, v___x_1007_);
v___x_1013_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1013_, 0, v___x_1012_);
lean_closure_set(v___x_1013_, 1, v___x_1007_);
v___x_1014_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1014_, 0, v___x_1013_);
lean_closure_set(v___x_1014_, 1, v___f_974_);
v___x_1015_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1015_, 0, v___x_1014_);
lean_closure_set(v___x_1015_, 1, v___f_974_);
v___x_1016_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_1016_, 0, v___f_976_);
v___x_1017_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1017_, 0, v___x_1016_);
lean_closure_set(v___x_1017_, 1, v___f_974_);
v___x_1018_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1018_, 0, v___x_1017_);
lean_closure_set(v___x_1018_, 1, v___f_974_);
lean_inc_ref_n(v___x_1018_, 3);
v___x_1019_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1019_, 0, v___x_1015_);
lean_closure_set(v___x_1019_, 1, v___x_1018_);
v___x_1020_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1020_, 0, v___x_1019_);
lean_closure_set(v___x_1020_, 1, v___x_1018_);
v___x_1021_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1021_, 0, v___x_1020_);
lean_closure_set(v___x_1021_, 1, v___f_974_);
v___x_1022_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1022_, 0, v___x_1021_);
lean_closure_set(v___x_1022_, 1, v___x_1018_);
v___x_1023_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1023_, 0, v___x_1022_);
lean_closure_set(v___x_1023_, 1, v___x_1018_);
v___x_1024_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1024_, 0, v___x_1023_);
lean_closure_set(v___x_1024_, 1, v___f_974_);
v___x_1025_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__1));
v___x_1026_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__2));
v___x_1027_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1027_, 0, v___x_1024_);
lean_closure_set(v___x_1027_, 1, v___x_1026_);
v___x_1028_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__4));
v___x_1029_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___closed__5));
v___x_1030_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_1031_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_1027_, v___x_1029_, v___x_1030_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
if (lean_obj_tag(v___x_1031_) == 0)
{
lean_object* v_a_1032_; lean_object* v___x_1034_; 
v_a_1032_ = lean_ctor_get(v___x_1031_, 0);
lean_inc(v_a_1032_);
lean_dec_ref_known(v___x_1031_, 1);
if (v_isShared_988_ == 0)
{
lean_ctor_set_tag(v___x_987_, 1);
v___x_1034_ = v___x_987_;
goto v_reusejp_1033_;
}
else
{
lean_object* v_reuseFailAlloc_1041_; 
v_reuseFailAlloc_1041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1041_, 0, v_a_985_);
v___x_1034_ = v_reuseFailAlloc_1041_;
goto v_reusejp_1033_;
}
v_reusejp_1033_:
{
lean_object* v___x_1035_; 
lean_inc_ref(v___x_1034_);
v___x_1035_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_1032_, v___x_1028_, v___x_1034_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
if (lean_obj_tag(v___x_1035_) == 0)
{
lean_object* v_a_1036_; lean_object* v___x_1037_; 
v_a_1036_ = lean_ctor_get(v___x_1035_, 0);
lean_inc(v_a_1036_);
lean_dec_ref_known(v___x_1035_, 1);
v___x_1037_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_1032_, v___x_1025_, v___x_1034_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
lean_dec(v_a_1032_);
if (lean_obj_tag(v___x_1037_) == 0)
{
lean_object* v_a_1038_; lean_object* v___f_1039_; lean_object* v___x_1040_; 
v_a_1038_ = lean_ctor_get(v___x_1037_, 0);
lean_inc(v_a_1038_);
lean_dec_ref_known(v___x_1037_, 1);
v___f_1039_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__4___boxed), 9, 2);
lean_closure_set(v___f_1039_, 0, v_a_1038_);
lean_closure_set(v___f_1039_, 1, v_a_1036_);
v___x_1040_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_1039_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
return v___x_1040_;
}
else
{
lean_dec(v_a_1036_);
return v___x_1037_;
}
}
else
{
lean_dec_ref(v___x_1034_);
lean_dec(v_a_1032_);
return v___x_1035_;
}
}
}
else
{
lean_object* v_a_1042_; lean_object* v___x_1044_; uint8_t v_isShared_1045_; uint8_t v_isSharedCheck_1049_; 
lean_del_object(v___x_987_);
lean_dec(v_a_985_);
v_a_1042_ = lean_ctor_get(v___x_1031_, 0);
v_isSharedCheck_1049_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_1044_ = v___x_1031_;
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
else
{
lean_inc(v_a_1042_);
lean_dec(v___x_1031_);
v___x_1044_ = lean_box(0);
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
v_resetjp_1043_:
{
lean_object* v___x_1047_; 
if (v_isShared_1045_ == 0)
{
v___x_1047_ = v___x_1044_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v_a_1042_);
v___x_1047_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
return v___x_1047_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5___boxed(lean_object* v___f_1051_, lean_object* v___f_1052_, lean_object* v___f_1053_, lean_object* v___f_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___lam__5(v___f_1051_, v___f_1052_, v___f_1053_, v___f_1054_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
lean_dec(v___y_1060_);
lean_dec_ref(v___y_1059_);
lean_dec(v___y_1058_);
lean_dec_ref(v___y_1057_);
lean_dec(v___y_1056_);
lean_dec_ref(v___y_1055_);
return v_res_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1(lean_object* v_a_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_, lean_object* v_a_1084_, lean_object* v_a_1085_){
_start:
{
lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; 
v___x_1087_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__5));
v___x_1088_ = ((lean_object*)(lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___closed__8));
v___x_1089_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_1087_, v___x_1088_, v_a_1080_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_, v_a_1085_);
return v___x_1089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1___boxed(lean_object* v_a_1090_, lean_object* v_a_1091_, lean_object* v_a_1092_, lean_object* v_a_1093_, lean_object* v_a_1094_, lean_object* v_a_1095_, lean_object* v_a_1096_){
_start:
{
lean_object* v_res_1097_; 
v_res_1097_ = lp_mathlib_LinearEquiv___aux__Mathlib__Algebra__Module__Equiv__Defs______delab__app__LinearEquiv__transNotation__1(v_a_1090_, v_a_1091_, v_a_1092_, v_a_1093_, v_a_1094_, v_a_1095_);
lean_dec(v_a_1095_);
lean_dec_ref(v_a_1094_);
lean_dec(v_a_1093_);
lean_dec_ref(v_a_1092_);
lean_dec(v_a_1091_);
lean_dec_ref(v_a_1090_);
return v_res_1097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symmEquiv___redArg(lean_object* v_inst_1098_, lean_object* v_inst_1099_, lean_object* v_inst_1100_, lean_object* v_inst_1101_, lean_object* v_module__M_1102_, lean_object* v_module__S__M_u2082_1103_, lean_object* v_00_u03c3_1104_, lean_object* v_00_u03c3_x27_1105_){
_start:
{
lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; 
lean_inc(v_00_u03c3_x27_1105_);
lean_inc(v_00_u03c3_1104_);
lean_inc(v_module__S__M_u2082_1103_);
lean_inc(v_module__M_1102_);
lean_inc_ref(v_inst_1101_);
lean_inc_ref(v_inst_1100_);
lean_inc_ref(v_inst_1099_);
lean_inc_ref(v_inst_1098_);
v___x_1106_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_symm___boxed), 15, 14);
lean_closure_set(v___x_1106_, 0, lean_box(0));
lean_closure_set(v___x_1106_, 1, lean_box(0));
lean_closure_set(v___x_1106_, 2, lean_box(0));
lean_closure_set(v___x_1106_, 3, lean_box(0));
lean_closure_set(v___x_1106_, 4, v_inst_1098_);
lean_closure_set(v___x_1106_, 5, v_inst_1099_);
lean_closure_set(v___x_1106_, 6, v_inst_1100_);
lean_closure_set(v___x_1106_, 7, v_inst_1101_);
lean_closure_set(v___x_1106_, 8, v_module__M_1102_);
lean_closure_set(v___x_1106_, 9, v_module__S__M_u2082_1103_);
lean_closure_set(v___x_1106_, 10, v_00_u03c3_1104_);
lean_closure_set(v___x_1106_, 11, v_00_u03c3_x27_1105_);
lean_closure_set(v___x_1106_, 12, lean_box(0));
lean_closure_set(v___x_1106_, 13, lean_box(0));
v___x_1107_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_symm___boxed), 15, 14);
lean_closure_set(v___x_1107_, 0, lean_box(0));
lean_closure_set(v___x_1107_, 1, lean_box(0));
lean_closure_set(v___x_1107_, 2, lean_box(0));
lean_closure_set(v___x_1107_, 3, lean_box(0));
lean_closure_set(v___x_1107_, 4, v_inst_1099_);
lean_closure_set(v___x_1107_, 5, v_inst_1098_);
lean_closure_set(v___x_1107_, 6, v_inst_1101_);
lean_closure_set(v___x_1107_, 7, v_inst_1100_);
lean_closure_set(v___x_1107_, 8, v_module__S__M_u2082_1103_);
lean_closure_set(v___x_1107_, 9, v_module__M_1102_);
lean_closure_set(v___x_1107_, 10, v_00_u03c3_x27_1105_);
lean_closure_set(v___x_1107_, 11, v_00_u03c3_1104_);
lean_closure_set(v___x_1107_, 12, lean_box(0));
lean_closure_set(v___x_1107_, 13, lean_box(0));
v___x_1108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1108_, 0, v___x_1106_);
lean_ctor_set(v___x_1108_, 1, v___x_1107_);
return v___x_1108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_symmEquiv(lean_object* v_R_1109_, lean_object* v_S_1110_, lean_object* v_M_1111_, lean_object* v_M_u2082_1112_, lean_object* v_inst_1113_, lean_object* v_inst_1114_, lean_object* v_inst_1115_, lean_object* v_inst_1116_, lean_object* v_module__M_1117_, lean_object* v_module__S__M_u2082_1118_, lean_object* v_00_u03c3_1119_, lean_object* v_00_u03c3_x27_1120_, lean_object* v_re_u2081_1121_, lean_object* v_re_u2082_1122_){
_start:
{
lean_object* v___x_1123_; 
v___x_1123_ = lp_mathlib_LinearEquiv_symmEquiv___redArg(v_inst_1113_, v_inst_1114_, v_inst_1115_, v_inst_1116_, v_module__M_1117_, v_module__S__M_u2082_1118_, v_00_u03c3_1119_, v_00_u03c3_x27_1120_);
return v___x_1123_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_cast___closed__0(void){
_start:
{
lean_object* v___x_1124_; 
v___x_1124_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_cast(lean_object* v_R_1125_, lean_object* v_inst_1126_, lean_object* v_00_u03b9_1127_, lean_object* v_M_1128_, lean_object* v_inst_1129_, lean_object* v_inst_1130_, lean_object* v_i_1131_, lean_object* v_j_1132_, lean_object* v_h_1133_){
_start:
{
lean_object* v___x_1134_; lean_object* v_toFun_1135_; lean_object* v_invFun_1136_; lean_object* v___x_1137_; 
v___x_1134_ = lean_obj_once(&lp_mathlib_LinearEquiv_cast___closed__0, &lp_mathlib_LinearEquiv_cast___closed__0_once, _init_lp_mathlib_LinearEquiv_cast___closed__0);
v_toFun_1135_ = lean_ctor_get(v___x_1134_, 0);
v_invFun_1136_ = lean_ctor_get(v___x_1134_, 1);
lean_inc(v_invFun_1136_);
lean_inc(v_toFun_1135_);
v___x_1137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1137_, 0, v_toFun_1135_);
lean_ctor_set(v___x_1137_, 1, v_invFun_1136_);
return v___x_1137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_cast___boxed(lean_object* v_R_1138_, lean_object* v_inst_1139_, lean_object* v_00_u03b9_1140_, lean_object* v_M_1141_, lean_object* v_inst_1142_, lean_object* v_inst_1143_, lean_object* v_i_1144_, lean_object* v_j_1145_, lean_object* v_h_1146_){
_start:
{
lean_object* v_res_1147_; 
v_res_1147_ = lp_mathlib_LinearEquiv_cast(v_R_1138_, v_inst_1139_, v_00_u03b9_1140_, v_M_1141_, v_inst_1142_, v_inst_1143_, v_i_1144_, v_j_1145_, v_h_1146_);
lean_dec(v_j_1145_);
lean_dec(v_i_1144_);
lean_dec(v_inst_1143_);
lean_dec_ref(v_inst_1142_);
lean_dec_ref(v_inst_1139_);
return v_res_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv___redArg___lam__0(lean_object* v_toFun_1148_, lean_object* v___y_1149_){
_start:
{
lean_object* v___x_1150_; 
v___x_1150_ = lean_apply_1(v_toFun_1148_, v___y_1149_);
return v___x_1150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv___redArg(lean_object* v_f_1151_){
_start:
{
lean_object* v_toFun_1152_; lean_object* v_invFun_1153_; lean_object* v___x_1155_; uint8_t v_isShared_1156_; uint8_t v_isSharedCheck_1161_; 
v_toFun_1152_ = lean_ctor_get(v_f_1151_, 0);
v_invFun_1153_ = lean_ctor_get(v_f_1151_, 1);
v_isSharedCheck_1161_ = !lean_is_exclusive(v_f_1151_);
if (v_isSharedCheck_1161_ == 0)
{
v___x_1155_ = v_f_1151_;
v_isShared_1156_ = v_isSharedCheck_1161_;
goto v_resetjp_1154_;
}
else
{
lean_inc(v_invFun_1153_);
lean_inc(v_toFun_1152_);
lean_dec(v_f_1151_);
v___x_1155_ = lean_box(0);
v_isShared_1156_ = v_isSharedCheck_1161_;
goto v_resetjp_1154_;
}
v_resetjp_1154_:
{
lean_object* v___f_1157_; lean_object* v___x_1159_; 
v___f_1157_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toSemilinearEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1157_, 0, v_toFun_1152_);
if (v_isShared_1156_ == 0)
{
lean_ctor_set(v___x_1155_, 0, v___f_1157_);
v___x_1159_ = v___x_1155_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1160_; 
v_reuseFailAlloc_1160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1160_, 0, v___f_1157_);
lean_ctor_set(v_reuseFailAlloc_1160_, 1, v_invFun_1153_);
v___x_1159_ = v_reuseFailAlloc_1160_;
goto v_reusejp_1158_;
}
v_reusejp_1158_:
{
return v___x_1159_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv(lean_object* v_R_1162_, lean_object* v_S_1163_, lean_object* v_inst_1164_, lean_object* v_inst_1165_, lean_object* v_f_1166_){
_start:
{
lean_object* v___x_1167_; 
v___x_1167_ = lp_mathlib_RingEquiv_toSemilinearEquiv___redArg(v_f_1166_);
return v___x_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toSemilinearEquiv___boxed(lean_object* v_R_1168_, lean_object* v_S_1169_, lean_object* v_inst_1170_, lean_object* v_inst_1171_, lean_object* v_f_1172_){
_start:
{
lean_object* v_res_1173_; 
v_res_1173_ = lp_mathlib_RingEquiv_toSemilinearEquiv(v_R_1168_, v_S_1169_, v_inst_1170_, v_inst_1171_, v_f_1172_);
lean_dec_ref(v_inst_1171_);
lean_dec_ref(v_inst_1170_);
return v_res_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive___redArg___lam__0(lean_object* v_f_1174_, lean_object* v___y_1175_){
_start:
{
lean_object* v___x_1176_; 
v___x_1176_ = lean_apply_1(v_f_1174_, v___y_1175_);
return v___x_1176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive___redArg(lean_object* v_f_1177_){
_start:
{
lean_object* v___f_1178_; lean_object* v___x_1179_; 
lean_inc(v_f_1177_);
v___f_1178_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_ofInvolutive___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1178_, 0, v_f_1177_);
v___x_1179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1179_, 0, v_f_1177_);
lean_ctor_set(v___x_1179_, 1, v___f_1178_);
return v___x_1179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive(lean_object* v_R_1180_, lean_object* v_M_1181_, lean_object* v_inst_1182_, lean_object* v_inst_1183_, lean_object* v_00_u03c3_1184_, lean_object* v_00_u03c3_x27_1185_, lean_object* v_inst_1186_, lean_object* v_inst_1187_, lean_object* v_x_1188_, lean_object* v_f_1189_, lean_object* v_hf_1190_){
_start:
{
lean_object* v___x_1191_; 
v___x_1191_ = lp_mathlib_LinearEquiv_ofInvolutive___redArg(v_f_1189_);
return v___x_1191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_ofInvolutive___boxed(lean_object* v_R_1192_, lean_object* v_M_1193_, lean_object* v_inst_1194_, lean_object* v_inst_1195_, lean_object* v_00_u03c3_1196_, lean_object* v_00_u03c3_x27_1197_, lean_object* v_inst_1198_, lean_object* v_inst_1199_, lean_object* v_x_1200_, lean_object* v_f_1201_, lean_object* v_hf_1202_){
_start:
{
lean_object* v_res_1203_; 
v_res_1203_ = lp_mathlib_LinearEquiv_ofInvolutive(v_R_1192_, v_M_1193_, v_inst_1194_, v_inst_1195_, v_00_u03c3_1196_, v_00_u03c3_x27_1197_, v_inst_1198_, v_inst_1199_, v_x_1200_, v_f_1201_, v_hf_1202_);
lean_dec(v_x_1200_);
lean_dec(v_00_u03c3_x27_1197_);
lean_dec(v_00_u03c3_1196_);
lean_dec_ref(v_inst_1195_);
lean_dec_ref(v_inst_1194_);
return v_res_1203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__0(lean_object* v_e_1204_, lean_object* v_inst_1205_, lean_object* v_inv_1206_, lean_object* v_x_1207_){
_start:
{
lean_object* v___x_1208_; lean_object* v_toLinearMap_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1208_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_1204_);
v_toLinearMap_1209_ = lean_ctor_get(v___x_1208_, 0);
lean_inc(v_toLinearMap_1209_);
lean_dec_ref(v___x_1208_);
v___x_1210_ = lean_apply_1(v_toLinearMap_1209_, v_x_1207_);
v___x_1211_ = lean_apply_2(v_inst_1205_, v_inv_1206_, v___x_1210_);
return v___x_1211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__1(lean_object* v_toLinearMap_1212_, lean_object* v_inst_1213_, lean_object* v_val_1214_, lean_object* v___y_1215_){
_start:
{
lean_object* v___x_1216_; lean_object* v___x_1217_; 
v___x_1216_ = lean_apply_1(v_toLinearMap_1212_, v___y_1215_);
v___x_1217_ = lean_apply_2(v_inst_1213_, v_val_1214_, v___x_1216_);
return v___x_1217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__2(lean_object* v_inst_1218_, lean_object* v_inst_1219_, lean_object* v_00_u03b1_1220_, lean_object* v_e_1221_){
_start:
{
lean_object* v_val_1222_; lean_object* v_inv_1223_; lean_object* v___x_1225_; uint8_t v_isShared_1226_; uint8_t v_isSharedCheck_1233_; 
v_val_1222_ = lean_ctor_get(v_00_u03b1_1220_, 0);
v_inv_1223_ = lean_ctor_get(v_00_u03b1_1220_, 1);
v_isSharedCheck_1233_ = !lean_is_exclusive(v_00_u03b1_1220_);
if (v_isSharedCheck_1233_ == 0)
{
v___x_1225_ = v_00_u03b1_1220_;
v_isShared_1226_ = v_isSharedCheck_1233_;
goto v_resetjp_1224_;
}
else
{
lean_inc(v_inv_1223_);
lean_inc(v_val_1222_);
lean_dec(v_00_u03b1_1220_);
v___x_1225_ = lean_box(0);
v_isShared_1226_ = v_isSharedCheck_1233_;
goto v_resetjp_1224_;
}
v_resetjp_1224_:
{
lean_object* v_toLinearMap_1227_; lean_object* v___f_1228_; lean_object* v___f_1229_; lean_object* v___x_1231_; 
v_toLinearMap_1227_ = lean_ctor_get(v_e_1221_, 0);
lean_inc(v_toLinearMap_1227_);
v___f_1228_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1228_, 0, v_e_1221_);
lean_closure_set(v___f_1228_, 1, v_inst_1218_);
lean_closure_set(v___f_1228_, 2, v_inv_1223_);
v___f_1229_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__1), 4, 3);
lean_closure_set(v___f_1229_, 0, v_toLinearMap_1227_);
lean_closure_set(v___f_1229_, 1, v_inst_1219_);
lean_closure_set(v___f_1229_, 2, v_val_1222_);
if (v_isShared_1226_ == 0)
{
lean_ctor_set(v___x_1225_, 1, v___f_1228_);
lean_ctor_set(v___x_1225_, 0, v___f_1229_);
v___x_1231_ = v___x_1225_;
goto v_reusejp_1230_;
}
else
{
lean_object* v_reuseFailAlloc_1232_; 
v_reuseFailAlloc_1232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1232_, 0, v___f_1229_);
lean_ctor_set(v_reuseFailAlloc_1232_, 1, v___f_1228_);
v___x_1231_ = v_reuseFailAlloc_1232_;
goto v_reusejp_1230_;
}
v_reusejp_1230_:
{
return v___x_1231_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___redArg(lean_object* v_inst_1234_, lean_object* v_inst_1235_){
_start:
{
lean_object* v___f_1236_; 
v___f_1236_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1236_, 0, v_inst_1234_);
lean_closure_set(v___f_1236_, 1, v_inst_1235_);
return v___f_1236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId(lean_object* v_S_1237_, lean_object* v_R_1238_, lean_object* v_V_1239_, lean_object* v_W_1240_, lean_object* v_inst_1241_, lean_object* v_inst_1242_, lean_object* v_inst_1243_, lean_object* v_inst_1244_, lean_object* v_inst_1245_, lean_object* v_inst_1246_, lean_object* v_inst_1247_, lean_object* v_inst_1248_, lean_object* v_inst_1249_, lean_object* v_inst_1250_, lean_object* v_inst_1251_, lean_object* v_inst_1252_){
_start:
{
lean_object* v___f_1253_; 
v___f_1253_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_instSMulUnitsId___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1253_, 0, v_inst_1245_);
lean_closure_set(v___f_1253_, 1, v_inst_1248_);
return v___f_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_instSMulUnitsId___boxed(lean_object* v_S_1254_, lean_object* v_R_1255_, lean_object* v_V_1256_, lean_object* v_W_1257_, lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_inst_1260_, lean_object* v_inst_1261_, lean_object* v_inst_1262_, lean_object* v_inst_1263_, lean_object* v_inst_1264_, lean_object* v_inst_1265_, lean_object* v_inst_1266_, lean_object* v_inst_1267_, lean_object* v_inst_1268_, lean_object* v_inst_1269_){
_start:
{
lean_object* v_res_1270_; 
v_res_1270_ = lp_mathlib_LinearEquiv_instSMulUnitsId(v_S_1254_, v_R_1255_, v_V_1256_, v_W_1257_, v_inst_1258_, v_inst_1259_, v_inst_1260_, v_inst_1261_, v_inst_1262_, v_inst_1263_, v_inst_1264_, v_inst_1265_, v_inst_1266_, v_inst_1267_, v_inst_1268_, v_inst_1269_);
lean_dec(v_inst_1267_);
lean_dec(v_inst_1264_);
lean_dec_ref(v_inst_1263_);
lean_dec(v_inst_1261_);
lean_dec_ref(v_inst_1260_);
lean_dec_ref(v_inst_1259_);
lean_dec_ref(v_inst_1258_);
return v_res_1270_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
