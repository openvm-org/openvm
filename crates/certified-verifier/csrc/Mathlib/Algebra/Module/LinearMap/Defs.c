// Lean compiler output
// Module: Mathlib.Algebra.Module.LinearMap.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Instances public import Mathlib.Algebra.Module.NatInt public import Mathlib.Algebra.Module.RingHom public import Mathlib.Algebra.Ring.CompTypeclasses public import Mathlib.GroupTheory.GroupAction.Hom
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
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_instMulActionNatOfAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_mulLeft___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_mulRight___redArg(lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 12, .m_data = "term_→ₛₗ[_]_"};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(242, 82, 143, 121, 27, 198, 241, 96)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 5, .m_data = " →ₛₗ["};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__9_value;
static const lean_string_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__10_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__10_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__11 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__11_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__9_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__12 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__12_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__13 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__13_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__12_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__13_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__14 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__14_value;
static const lean_ctor_object lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__14_value)}};
static const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__15 = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u209b_u2097_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__15_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "LinearMap"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(29, 55, 59, 137, 47, 1, 37, 113)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 11, .m_data = "term_→ₗ[_]_"};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 240, 198, 234, 211, 154, 148, 218)}};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " →ₗ["};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__4_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__13_value)}};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__6_value)}};
static const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192_u2097_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__7_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__8;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__11_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__11_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__13_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__9_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__13_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingHom.id"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__15_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "RingHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__17_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__18_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(193, 71, 107, 83, 214, 46, 125, 66)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(77, 63, 71, 191, 78, 103, 81, 221)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ofClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_semilinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_semilinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_semilinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMapClass_linearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMapClass_linearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMapClass_linearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_id___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coeIsScalarTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coeIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp___boxed(lean_object**);
static const lean_string_object lp_mathlib_LinearMap_compNotation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "compNotation"};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__0_value;
static const lean_ctor_object lp_mathlib_LinearMap_compNotation___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(29, 55, 59, 137, 47, 1, 37, 113)}};
static const lean_ctor_object lp_mathlib_LinearMap_compNotation___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__1_value_aux_0),((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 207, 138, 5, 10, 118, 117, 209)}};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__1 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__1_value;
static const lean_string_object lp_mathlib_LinearMap_compNotation___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ∘ₗ "};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__2 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__2_value;
static const lean_ctor_object lp_mathlib_LinearMap_compNotation___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__2_value)}};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__3 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__3_value;
static const lean_ctor_object lp_mathlib_LinearMap_compNotation___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(80) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__4 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__4_value;
static const lean_ctor_object lp_mathlib_LinearMap_compNotation___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__3_value),((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__4_value)}};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__5 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__5_value;
static const lean_ctor_object lp_mathlib_LinearMap_compNotation___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__1_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(81) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap_compNotation___closed__5_value)}};
static const lean_object* lp_mathlib_LinearMap_compNotation___closed__6 = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_LinearMap_compNotation = (const lean_object*)&lp_mathlib_LinearMap_compNotation___closed__6_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "LinearMap.comp"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__0 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__0_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__1;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "comp"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__2 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__2_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(29, 55, 59, 137, 47, 1, 37, 113)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(218, 212, 182, 116, 137, 149, 44, 131)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__4 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__4_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__5 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__5_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "namedArgument"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__6 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__6_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(226, 89, 129, 113, 173, 121, 169, 188)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 3, .m_data = "σ₁₂"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__8 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__8_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__9;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(230, 245, 73, 214, 89, 175, 195, 48)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__10 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__10_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(230, 245, 73, 214, 89, 175, 195, 48)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__11 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__11_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__12 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__12_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__11_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(23, 113, 42, 145, 205, 65, 215, 205)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__13 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__13_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__14 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__14_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__13_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(154, 134, 221, 36, 221, 17, 89, 116)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__15 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__15_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__16 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__16_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__15_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(226, 214, 39, 152, 110, 18, 77, 70)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__17 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__17_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Module"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__18 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__18_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__17_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(46, 62, 20, 235, 247, 160, 73, 240)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__19 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__19_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__19_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(172, 239, 128, 36, 229, 170, 67, 47)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__20 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__20_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Defs"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__21 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__21_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__20_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(100, 89, 193, 182, 36, 185, 210, 64)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__22 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__22_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__23;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__24 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__24_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__25;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__26 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__26_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__27;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__28;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__29;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__30;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__31 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__31_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__32 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__32_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value_aux_2),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__34 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__34_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 3, .m_data = "σ₂₃"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__35 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__35_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__36;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(132, 144, 181, 147, 56, 2, 213, 195)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__37 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__37_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(132, 144, 181, 147, 56, 2, 213, 195)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__38 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__38_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__38_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(13, 231, 120, 71, 202, 4, 58, 0)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__39 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__39_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__39_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(200, 180, 109, 136, 189, 250, 189, 93)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__40 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__40_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__40_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(40, 235, 219, 17, 234, 89, 47, 8)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__41 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__41_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__41_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(236, 246, 71, 149, 23, 1, 62, 30)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__42 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__42_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__42_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(22, 171, 178, 127, 118, 171, 154, 187)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__43 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__43_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__43_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(6, 92, 121, 176, 240, 196, 33, 70)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__44 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__44_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__45;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__46;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__47;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__48;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__49;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__50;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 3, .m_data = "σ₁₃"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__51 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__51_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__52;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__51_value),LEAN_SCALAR_PTR_LITERAL(177, 50, 179, 49, 81, 252, 13, 251)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__53 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__53_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__51_value),LEAN_SCALAR_PTR_LITERAL(177, 50, 179, 49, 81, 252, 13, 251)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__54 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__54_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__54_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(28, 135, 198, 199, 33, 236, 166, 4)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__55 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__55_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__55_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(173, 87, 132, 179, 66, 159, 41, 203)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__56 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__56_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__56_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(241, 49, 90, 126, 9, 14, 135, 140)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__57 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__57_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__57_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(73, 156, 66, 206, 210, 218, 83, 27)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__58 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__58_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__58_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(55, 120, 147, 35, 172, 78, 26, 23)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__59 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__59_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__59_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(251, 141, 35, 228, 125, 182, 99, 244)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__60 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__60_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__61;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__62;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__63;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__64;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__65;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__66;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__0 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__0_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__1 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__1_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__1_value)} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__2 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__2_value;
static const lean_string_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "g"};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__3 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__3_value;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__3_value),LEAN_SCALAR_PTR_LITERAL(30, 12, 229, 162, 1, 36, 3, 29)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__4 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__4_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__4_value)} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__5 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__0 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__0_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__1 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__1_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__2___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__2 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__2_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__0_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__2_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__1_value)} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__3 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__3_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__4 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__4_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__5 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__5_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(21) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__3_value)} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__6 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__6_value;
static const lean_closure_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__5_value),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__6_value)} };
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__7 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 9, .m_data = "term_∘ₛₗ_"};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__0 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__0_value;
static const lean_ctor_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(29, 55, 59, 137, 47, 1, 37, 113)}};
static const lean_ctor_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 90, 71, 25, 223, 156, 9, 162)}};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1_value;
static const lean_string_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 5, .m_data = " ∘ₛₗ "};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__2 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__2_value;
static const lean_ctor_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__2_value)}};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__3 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__3_value;
static const lean_ctor_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(90) << 1) | 1))}};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__4 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__4_value;
static const lean_ctor_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__3_value),((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__4_value)}};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__5 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__5_value;
static const lean_ctor_object lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1_value),((lean_object*)(((size_t)(90) << 1) | 1)),((lean_object*)(((size_t)(91) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__5_value)}};
static const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__6 = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_LinearMap_term___u2218_u209b_u2097__ = (const lean_object*)&lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__6_value;
static lean_once_cell_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__0;
static const lean_ctor_object lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(228, 239, 123, 62, 2, 27, 64, 57)}};
static const lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__1 = (const lean_object*)&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__comp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__comp__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars_u2097___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars_u2097(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars_u2097___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeftRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeftRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeftRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_LinearMap_toMulActionHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom(lean_object* v_R_4_, lean_object* v_S_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_00_u03c3_8_, lean_object* v_M_9_, lean_object* v_M_u2082_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_self_15_){
_start:
{
lean_inc(v_self_15_);
return v_self_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toMulActionHom___boxed(lean_object* v_R_16_, lean_object* v_S_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_00_u03c3_20_, lean_object* v_M_21_, lean_object* v_M_u2082_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_self_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_LinearMap_toMulActionHom(v_R_16_, v_S_17_, v_inst_18_, v_inst_19_, v_00_u03c3_20_, v_M_21_, v_M_u2082_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_self_27_);
lean_dec(v_self_27_);
lean_dec(v_inst_26_);
lean_dec(v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec(v_00_u03c3_20_);
lean_dec_ref(v_inst_19_);
lean_dec_ref(v_inst_18_);
return v_res_28_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_78_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__5));
v___x_79_ = l_String_toRawSubstring_x27(v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1(lean_object* v_x_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v___x_99_; uint8_t v___x_100_; 
v___x_99_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__1));
lean_inc(v_x_96_);
v___x_100_ = l_Lean_Syntax_isOfKind(v_x_96_, v___x_99_);
if (v___x_100_ == 0)
{
lean_object* v___x_101_; lean_object* v___x_102_; 
lean_dec(v_x_96_);
v___x_101_ = lean_box(1);
v___x_102_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v_a_98_);
return v___x_102_;
}
else
{
lean_object* v_quotContext_103_; lean_object* v_currMacroScope_104_; lean_object* v_ref_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; uint8_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v_quotContext_103_ = lean_ctor_get(v_a_97_, 1);
v_currMacroScope_104_ = lean_ctor_get(v_a_97_, 2);
v_ref_105_ = lean_ctor_get(v_a_97_, 5);
v___x_106_ = lean_unsigned_to_nat(0u);
v___x_107_ = l_Lean_Syntax_getArg(v_x_96_, v___x_106_);
v___x_108_ = lean_unsigned_to_nat(2u);
v___x_109_ = l_Lean_Syntax_getArg(v_x_96_, v___x_108_);
v___x_110_ = lean_unsigned_to_nat(4u);
v___x_111_ = l_Lean_Syntax_getArg(v_x_96_, v___x_110_);
lean_dec(v_x_96_);
v___x_112_ = 0;
v___x_113_ = l_Lean_SourceInfo_fromRef(v_ref_105_, v___x_112_);
v___x_114_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
v___x_115_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6);
v___x_116_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7));
lean_inc(v_currMacroScope_104_);
lean_inc(v_quotContext_103_);
v___x_117_ = l_Lean_addMacroScope(v_quotContext_103_, v___x_116_, v_currMacroScope_104_);
v___x_118_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__11));
lean_inc_n(v___x_113_, 2);
v___x_119_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_119_, 0, v___x_113_);
lean_ctor_set(v___x_119_, 1, v___x_115_);
lean_ctor_set(v___x_119_, 2, v___x_117_);
lean_ctor_set(v___x_119_, 3, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13));
v___x_121_ = l_Lean_Syntax_node3(v___x_113_, v___x_120_, v___x_109_, v___x_107_, v___x_111_);
v___x_122_ = l_Lean_Syntax_node2(v___x_113_, v___x_114_, v___x_119_, v___x_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_98_);
return v___x_123_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___boxed(lean_object* v_x_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1(v_x_124_, v_a_125_, v_a_126_);
lean_dec_ref(v_a_125_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1(lean_object* v_x_131_, lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
lean_object* v___x_134_; uint8_t v___x_135_; 
v___x_134_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
lean_inc(v_x_131_);
v___x_135_ = l_Lean_Syntax_isOfKind(v_x_131_, v___x_134_);
if (v___x_135_ == 0)
{
lean_object* v___x_136_; lean_object* v___x_137_; 
lean_dec(v_x_131_);
v___x_136_ = lean_box(0);
v___x_137_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
lean_ctor_set(v___x_137_, 1, v_a_133_);
return v___x_137_;
}
else
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v___x_141_; 
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = l_Lean_Syntax_getArg(v_x_131_, v___x_138_);
v___x_140_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__1));
lean_inc(v___x_139_);
v___x_141_ = l_Lean_Syntax_isOfKind(v___x_139_, v___x_140_);
if (v___x_141_ == 0)
{
lean_object* v___x_142_; lean_object* v___x_143_; 
lean_dec(v___x_139_);
lean_dec(v_x_131_);
v___x_142_ = lean_box(0);
v___x_143_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v_a_133_);
return v___x_143_;
}
else
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; uint8_t v___x_147_; 
v___x_144_ = lean_unsigned_to_nat(1u);
v___x_145_ = l_Lean_Syntax_getArg(v_x_131_, v___x_144_);
lean_dec(v_x_131_);
v___x_146_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_145_);
v___x_147_ = l_Lean_Syntax_matchesNull(v___x_145_, v___x_146_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; lean_object* v___x_149_; 
lean_dec(v___x_145_);
lean_dec(v___x_139_);
v___x_148_ = lean_box(0);
v___x_149_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v_a_133_);
return v___x_149_;
}
else
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v_ref_154_; uint8_t v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_150_ = l_Lean_Syntax_getArg(v___x_145_, v___x_138_);
v___x_151_ = l_Lean_Syntax_getArg(v___x_145_, v___x_144_);
v___x_152_ = lean_unsigned_to_nat(2u);
v___x_153_ = l_Lean_Syntax_getArg(v___x_145_, v___x_152_);
lean_dec(v___x_145_);
v_ref_154_ = l_Lean_replaceRef(v___x_139_, v_a_132_);
lean_dec(v___x_139_);
v___x_155_ = 0;
v___x_156_ = l_Lean_SourceInfo_fromRef(v_ref_154_, v___x_155_);
lean_dec(v_ref_154_);
v___x_157_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__1));
v___x_158_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__4));
lean_inc_n(v___x_156_, 2);
v___x_159_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_156_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
v___x_160_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__10));
v___x_161_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_156_);
lean_ctor_set(v___x_161_, 1, v___x_160_);
v___x_162_ = l_Lean_Syntax_node5(v___x_156_, v___x_157_, v___x_151_, v___x_159_, v___x_150_, v___x_161_, v___x_153_);
v___x_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v_a_133_);
return v___x_163_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___boxed(lean_object* v_x_164_, lean_object* v_a_165_, lean_object* v_a_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1(v_x_164_, v_a_165_, v_a_166_);
lean_dec(v_a_165_);
return v_res_167_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__8(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__7));
v___x_210_ = l_String_toRawSubstring_x27(v___x_209_);
return v___x_210_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_225_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__15));
v___x_226_ = l_String_toRawSubstring_x27(v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1(lean_object* v_x_239_, lean_object* v_a_240_, lean_object* v_a_241_){
_start:
{
lean_object* v___x_242_; uint8_t v___x_243_; 
v___x_242_ = ((lean_object*)(lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__1));
lean_inc(v_x_239_);
v___x_243_ = l_Lean_Syntax_isOfKind(v_x_239_, v___x_242_);
if (v___x_243_ == 0)
{
lean_object* v___x_244_; lean_object* v___x_245_; 
lean_dec(v_x_239_);
v___x_244_ = lean_box(1);
v___x_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v_a_241_);
return v___x_245_;
}
else
{
lean_object* v_quotContext_246_; lean_object* v_currMacroScope_247_; lean_object* v_ref_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; uint8_t v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v_quotContext_246_ = lean_ctor_get(v_a_240_, 1);
v_currMacroScope_247_ = lean_ctor_get(v_a_240_, 2);
v_ref_248_ = lean_ctor_get(v_a_240_, 5);
v___x_249_ = lean_unsigned_to_nat(0u);
v___x_250_ = l_Lean_Syntax_getArg(v_x_239_, v___x_249_);
v___x_251_ = lean_unsigned_to_nat(2u);
v___x_252_ = l_Lean_Syntax_getArg(v_x_239_, v___x_251_);
v___x_253_ = lean_unsigned_to_nat(4u);
v___x_254_ = l_Lean_Syntax_getArg(v_x_239_, v___x_253_);
lean_dec(v_x_239_);
v___x_255_ = 0;
v___x_256_ = l_Lean_SourceInfo_fromRef(v_ref_248_, v___x_255_);
v___x_257_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
v___x_258_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__6);
v___x_259_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__7));
lean_inc_n(v_currMacroScope_247_, 3);
lean_inc_n(v_quotContext_246_, 3);
v___x_260_ = l_Lean_addMacroScope(v_quotContext_246_, v___x_259_, v_currMacroScope_247_);
v___x_261_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__11));
lean_inc_n(v___x_256_, 11);
v___x_262_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_262_, 0, v___x_256_);
lean_ctor_set(v___x_262_, 1, v___x_258_);
lean_ctor_set(v___x_262_, 2, v___x_260_);
lean_ctor_set(v___x_262_, 3, v___x_261_);
v___x_263_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13));
v___x_264_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__1));
v___x_265_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__3));
v___x_266_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__4));
v___x_267_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_256_);
lean_ctor_set(v___x_267_, 1, v___x_266_);
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__6));
v___x_269_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__8, &lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__8_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__8);
v___x_270_ = lean_box(0);
v___x_271_ = l_Lean_addMacroScope(v_quotContext_246_, v___x_270_, v_currMacroScope_247_);
v___x_272_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__14));
v___x_273_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_273_, 0, v___x_256_);
lean_ctor_set(v___x_273_, 1, v___x_269_);
lean_ctor_set(v___x_273_, 2, v___x_271_);
lean_ctor_set(v___x_273_, 3, v___x_272_);
v___x_274_ = l_Lean_Syntax_node1(v___x_256_, v___x_268_, v___x_273_);
v___x_275_ = l_Lean_Syntax_node2(v___x_256_, v___x_265_, v___x_267_, v___x_274_);
v___x_276_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16, &lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16);
v___x_277_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19));
v___x_278_ = l_Lean_addMacroScope(v_quotContext_246_, v___x_277_, v_currMacroScope_247_);
v___x_279_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__21));
v___x_280_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_280_, 0, v___x_256_);
lean_ctor_set(v___x_280_, 1, v___x_276_);
lean_ctor_set(v___x_280_, 2, v___x_278_);
lean_ctor_set(v___x_280_, 3, v___x_279_);
v___x_281_ = l_Lean_Syntax_node1(v___x_256_, v___x_263_, v___x_252_);
v___x_282_ = l_Lean_Syntax_node2(v___x_256_, v___x_257_, v___x_280_, v___x_281_);
v___x_283_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__22));
v___x_284_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_284_, 0, v___x_256_);
lean_ctor_set(v___x_284_, 1, v___x_283_);
v___x_285_ = l_Lean_Syntax_node3(v___x_256_, v___x_264_, v___x_275_, v___x_282_, v___x_284_);
v___x_286_ = l_Lean_Syntax_node3(v___x_256_, v___x_263_, v___x_285_, v___x_250_, v___x_254_);
v___x_287_ = l_Lean_Syntax_node2(v___x_256_, v___x_257_, v___x_262_, v___x_286_);
v___x_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
lean_ctor_set(v___x_288_, 1, v_a_241_);
return v___x_288_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___boxed(lean_object* v_x_289_, lean_object* v_a_290_, lean_object* v_a_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1(v_x_289_, v_a_290_, v_a_291_);
lean_dec_ref(v_a_290_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__2(lean_object* v_x_293_, lean_object* v_a_294_, lean_object* v_a_295_){
_start:
{
lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_296_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
lean_inc(v_x_293_);
v___x_297_ = l_Lean_Syntax_isOfKind(v_x_293_, v___x_296_);
if (v___x_297_ == 0)
{
lean_object* v___x_298_; lean_object* v___x_299_; 
lean_dec(v_x_293_);
v___x_298_ = lean_box(0);
v___x_299_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v_a_295_);
return v___x_299_;
}
else
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; uint8_t v___x_303_; 
v___x_300_ = lean_unsigned_to_nat(0u);
v___x_301_ = l_Lean_Syntax_getArg(v_x_293_, v___x_300_);
v___x_302_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__1));
lean_inc(v___x_301_);
v___x_303_ = l_Lean_Syntax_isOfKind(v___x_301_, v___x_302_);
if (v___x_303_ == 0)
{
lean_object* v___x_304_; lean_object* v___x_305_; 
lean_dec(v___x_301_);
lean_dec(v_x_293_);
v___x_304_ = lean_box(0);
v___x_305_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
lean_ctor_set(v___x_305_, 1, v_a_295_);
return v___x_305_;
}
else
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; uint8_t v___x_309_; 
v___x_306_ = lean_unsigned_to_nat(1u);
v___x_307_ = l_Lean_Syntax_getArg(v_x_293_, v___x_306_);
lean_dec(v_x_293_);
v___x_308_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_307_);
v___x_309_ = l_Lean_Syntax_matchesNull(v___x_307_, v___x_308_);
if (v___x_309_ == 0)
{
lean_object* v___x_310_; lean_object* v___x_311_; 
lean_dec(v___x_307_);
lean_dec(v___x_301_);
v___x_310_ = lean_box(0);
v___x_311_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v_a_295_);
return v___x_311_;
}
else
{
lean_object* v___x_312_; uint8_t v___x_313_; 
v___x_312_ = l_Lean_Syntax_getArg(v___x_307_, v___x_300_);
lean_inc(v___x_312_);
v___x_313_ = l_Lean_Syntax_isOfKind(v___x_312_, v___x_296_);
if (v___x_313_ == 0)
{
lean_object* v___x_314_; lean_object* v___x_315_; 
lean_dec(v___x_312_);
lean_dec(v___x_307_);
lean_dec(v___x_301_);
v___x_314_ = lean_box(0);
v___x_315_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_314_);
lean_ctor_set(v___x_315_, 1, v_a_295_);
return v___x_315_;
}
else
{
lean_object* v___x_316_; lean_object* v___x_317_; uint8_t v___x_318_; 
v___x_316_ = l_Lean_Syntax_getArg(v___x_312_, v___x_300_);
v___x_317_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19));
v___x_318_ = l_Lean_Syntax_matchesIdent(v___x_316_, v___x_317_);
lean_dec(v___x_316_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; lean_object* v___x_320_; 
lean_dec(v___x_312_);
lean_dec(v___x_307_);
lean_dec(v___x_301_);
v___x_319_ = lean_box(0);
v___x_320_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
lean_ctor_set(v___x_320_, 1, v_a_295_);
return v___x_320_;
}
else
{
lean_object* v___x_321_; uint8_t v___x_322_; 
v___x_321_ = l_Lean_Syntax_getArg(v___x_312_, v___x_306_);
lean_dec(v___x_312_);
lean_inc(v___x_321_);
v___x_322_ = l_Lean_Syntax_matchesNull(v___x_321_, v___x_306_);
if (v___x_322_ == 0)
{
lean_object* v___x_323_; lean_object* v___x_324_; 
lean_dec(v___x_321_);
lean_dec(v___x_307_);
lean_dec(v___x_301_);
v___x_323_ = lean_box(0);
v___x_324_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_323_);
lean_ctor_set(v___x_324_, 1, v_a_295_);
return v___x_324_;
}
else
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v_ref_329_; uint8_t v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_325_ = l_Lean_Syntax_getArg(v___x_321_, v___x_300_);
lean_dec(v___x_321_);
v___x_326_ = l_Lean_Syntax_getArg(v___x_307_, v___x_306_);
v___x_327_ = lean_unsigned_to_nat(2u);
v___x_328_ = l_Lean_Syntax_getArg(v___x_307_, v___x_327_);
lean_dec(v___x_307_);
v_ref_329_ = l_Lean_replaceRef(v___x_301_, v_a_294_);
lean_dec(v___x_301_);
v___x_330_ = 0;
v___x_331_ = l_Lean_SourceInfo_fromRef(v_ref_329_, v___x_330_);
lean_dec(v_ref_329_);
v___x_332_ = ((lean_object*)(lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__1));
v___x_333_ = ((lean_object*)(lp_mathlib_term___u2192_u2097_x5b___x5d___00__closed__2));
lean_inc_n(v___x_331_, 2);
v___x_334_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_331_);
lean_ctor_set(v___x_334_, 1, v___x_333_);
v___x_335_ = ((lean_object*)(lp_mathlib_term___u2192_u209b_u2097_x5b___x5d___00__closed__10));
v___x_336_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_331_);
lean_ctor_set(v___x_336_, 1, v___x_335_);
v___x_337_ = l_Lean_Syntax_node5(v___x_331_, v___x_332_, v___x_326_, v___x_334_, v___x_325_, v___x_336_, v___x_328_);
v___x_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v_a_295_);
return v___x_338_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__2___boxed(lean_object* v_x_339_, lean_object* v_a_340_, lean_object* v_a_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__2(v_x_339_, v_a_340_, v_a_341_);
lean_dec(v_a_340_);
return v_res_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ofClass___redArg(lean_object* v_f_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lean_apply_1(v_inst_344_, v_f_343_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ofClass(lean_object* v_R_346_, lean_object* v_S_347_, lean_object* v_M_348_, lean_object* v_M_u2083_349_, lean_object* v_F_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_00_u03c3_357_, lean_object* v_f_358_, lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lean_apply_1(v_inst_359_, v_f_358_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_ofClass___boxed(lean_object* v_R_362_, lean_object* v_S_363_, lean_object* v_M_364_, lean_object* v_M_u2083_365_, lean_object* v_F_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_00_u03c3_373_, lean_object* v_f_374_, lean_object* v_inst_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_LinearMap_ofClass(v_R_362_, v_S_363_, v_M_364_, v_M_u2083_365_, v_F_366_, v_inst_367_, v_inst_368_, v_inst_369_, v_inst_370_, v_inst_371_, v_inst_372_, v_00_u03c3_373_, v_f_374_, v_inst_375_, v_inst_376_);
lean_dec(v_00_u03c3_373_);
lean_dec(v_inst_372_);
lean_dec(v_inst_371_);
lean_dec_ref(v_inst_370_);
lean_dec_ref(v_inst_369_);
lean_dec_ref(v_inst_368_);
lean_dec_ref(v_inst_367_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___redArg___lam__0(lean_object* v_inst_378_, lean_object* v_f_379_, lean_object* v___y_380_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lean_apply_2(v_inst_378_, v_f_379_, v___y_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___redArg(lean_object* v_inst_382_){
_start:
{
lean_object* v___f_383_; 
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_383_, 0, v_inst_382_);
return v___f_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap(lean_object* v_R_384_, lean_object* v_S_385_, lean_object* v_M_386_, lean_object* v_M_u2083_387_, lean_object* v_F_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_00_u03c3_395_, lean_object* v_inst_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_398_, 0, v_inst_396_);
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap___boxed(lean_object* v_R_399_, lean_object* v_S_400_, lean_object* v_M_401_, lean_object* v_M_u2083_402_, lean_object* v_F_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_00_u03c3_410_, lean_object* v_inst_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v_res_413_; 
v_res_413_ = lp_mathlib_SemilinearMapClass_instCoeToSemilinearMap(v_R_399_, v_S_400_, v_M_401_, v_M_u2083_402_, v_F_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_inst_407_, v_inst_408_, v_inst_409_, v_00_u03c3_410_, v_inst_411_, v_inst_412_);
lean_dec(v_00_u03c3_410_);
lean_dec(v_inst_409_);
lean_dec(v_inst_408_);
lean_dec_ref(v_inst_407_);
lean_dec_ref(v_inst_406_);
lean_dec_ref(v_inst_405_);
lean_dec_ref(v_inst_404_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_semilinearMap___redArg(lean_object* v_f_414_, lean_object* v_inst_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lean_apply_1(v_inst_415_, v_f_414_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_semilinearMap(lean_object* v_R_417_, lean_object* v_S_418_, lean_object* v_M_419_, lean_object* v_M_u2083_420_, lean_object* v_F_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_00_u03c3_428_, lean_object* v_f_429_, lean_object* v_inst_430_, lean_object* v_inst_431_){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lean_apply_1(v_inst_430_, v_f_429_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilinearMapClass_semilinearMap___boxed(lean_object* v_R_433_, lean_object* v_S_434_, lean_object* v_M_435_, lean_object* v_M_u2083_436_, lean_object* v_F_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_00_u03c3_444_, lean_object* v_f_445_, lean_object* v_inst_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_mathlib_SemilinearMapClass_semilinearMap(v_R_433_, v_S_434_, v_M_435_, v_M_u2083_436_, v_F_437_, v_inst_438_, v_inst_439_, v_inst_440_, v_inst_441_, v_inst_442_, v_inst_443_, v_00_u03c3_444_, v_f_445_, v_inst_446_, v_inst_447_);
lean_dec(v_00_u03c3_444_);
lean_dec(v_inst_443_);
lean_dec(v_inst_442_);
lean_dec_ref(v_inst_441_);
lean_dec_ref(v_inst_440_);
lean_dec_ref(v_inst_439_);
lean_dec_ref(v_inst_438_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMapClass_linearMap___redArg(lean_object* v_f_449_, lean_object* v_inst_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lean_apply_1(v_inst_450_, v_f_449_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMapClass_linearMap(lean_object* v_R_452_, lean_object* v_S_453_, lean_object* v_M_454_, lean_object* v_M_u2083_455_, lean_object* v_F_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_00_u03c3_463_, lean_object* v_f_464_, lean_object* v_inst_465_, lean_object* v_inst_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lean_apply_1(v_inst_465_, v_f_464_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMapClass_linearMap___boxed(lean_object* v_R_468_, lean_object* v_S_469_, lean_object* v_M_470_, lean_object* v_M_u2083_471_, lean_object* v_F_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_00_u03c3_479_, lean_object* v_f_480_, lean_object* v_inst_481_, lean_object* v_inst_482_){
_start:
{
lean_object* v_res_483_; 
v_res_483_ = lp_mathlib_LinearMapClass_linearMap(v_R_468_, v_S_469_, v_M_470_, v_M_u2083_471_, v_F_472_, v_inst_473_, v_inst_474_, v_inst_475_, v_inst_476_, v_inst_477_, v_inst_478_, v_00_u03c3_479_, v_f_480_, v_inst_481_, v_inst_482_);
lean_dec(v_00_u03c3_479_);
lean_dec(v_inst_478_);
lean_dec(v_inst_477_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_473_);
return v_res_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom___redArg(lean_object* v_f_484_){
_start:
{
lean_inc(v_f_484_);
return v_f_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom___redArg___boxed(lean_object* v_f_485_){
_start:
{
lean_object* v_res_486_; 
v_res_486_ = lp_mathlib_LinearMap_toDistribMulActionHom___redArg(v_f_485_);
lean_dec(v_f_485_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom(lean_object* v_R_487_, lean_object* v_S_488_, lean_object* v_M_489_, lean_object* v_M_u2083_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_inst_496_, lean_object* v_00_u03c3_497_, lean_object* v_f_498_){
_start:
{
lean_inc(v_f_498_);
return v_f_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toDistribMulActionHom___boxed(lean_object* v_R_499_, lean_object* v_S_500_, lean_object* v_M_501_, lean_object* v_M_u2083_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_00_u03c3_509_, lean_object* v_f_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_mathlib_LinearMap_toDistribMulActionHom(v_R_499_, v_S_500_, v_M_501_, v_M_u2083_502_, v_inst_503_, v_inst_504_, v_inst_505_, v_inst_506_, v_inst_507_, v_inst_508_, v_00_u03c3_509_, v_f_510_);
lean_dec(v_f_510_);
lean_dec(v_00_u03c3_509_);
lean_dec(v_inst_508_);
lean_dec(v_inst_507_);
lean_dec_ref(v_inst_506_);
lean_dec_ref(v_inst_505_);
lean_dec_ref(v_inst_504_);
lean_dec_ref(v_inst_503_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy___redArg(lean_object* v_f_x27_512_){
_start:
{
lean_inc(v_f_x27_512_);
return v_f_x27_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy___redArg___boxed(lean_object* v_f_x27_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_LinearMap_copy___redArg(v_f_x27_513_);
lean_dec(v_f_x27_513_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy(lean_object* v_R_515_, lean_object* v_S_516_, lean_object* v_M_517_, lean_object* v_M_u2083_518_, lean_object* v_inst_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_00_u03c3_525_, lean_object* v_f_526_, lean_object* v_f_x27_527_, lean_object* v_h_528_){
_start:
{
lean_inc(v_f_x27_527_);
return v_f_x27_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_copy___boxed(lean_object* v_R_529_, lean_object* v_S_530_, lean_object* v_M_531_, lean_object* v_M_u2083_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_inst_537_, lean_object* v_inst_538_, lean_object* v_00_u03c3_539_, lean_object* v_f_540_, lean_object* v_f_x27_541_, lean_object* v_h_542_){
_start:
{
lean_object* v_res_543_; 
v_res_543_ = lp_mathlib_LinearMap_copy(v_R_529_, v_S_530_, v_M_531_, v_M_u2083_532_, v_inst_533_, v_inst_534_, v_inst_535_, v_inst_536_, v_inst_537_, v_inst_538_, v_00_u03c3_539_, v_f_540_, v_f_x27_541_, v_h_542_);
lean_dec(v_f_x27_541_);
lean_dec(v_f_540_);
lean_dec(v_00_u03c3_539_);
lean_dec(v_inst_538_);
lean_dec(v_inst_537_);
lean_dec_ref(v_inst_536_);
lean_dec_ref(v_inst_535_);
lean_dec_ref(v_inst_534_);
lean_dec_ref(v_inst_533_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object* v_x_544_){
_start:
{
lean_inc(v_x_544_);
return v_x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object* v_x_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_mathlib_LinearMap_id___lam__0(v_x_545_);
lean_dec(v_x_545_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id(lean_object* v_R_548_, lean_object* v_M_549_, lean_object* v_inst_550_, lean_object* v_inst_551_, lean_object* v_inst_552_){
_start:
{
lean_object* v___f_553_; 
v___f_553_ = ((lean_object*)(lp_mathlib_LinearMap_id___closed__0));
return v___f_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id___boxed(lean_object* v_R_554_, lean_object* v_M_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_inst_558_){
_start:
{
lean_object* v_res_559_; 
v_res_559_ = lp_mathlib_LinearMap_id(v_R_554_, v_M_555_, v_inst_556_, v_inst_557_, v_inst_558_);
lean_dec(v_inst_558_);
lean_dec_ref(v_inst_557_);
lean_dec_ref(v_inst_556_);
return v_res_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id_x27(lean_object* v_R_560_, lean_object* v_M_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_00_u03c3_565_, lean_object* v_inst_566_){
_start:
{
lean_object* v___f_567_; 
v___f_567_ = ((lean_object*)(lp_mathlib_LinearMap_id___closed__0));
return v___f_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_id_x27___boxed(lean_object* v_R_568_, lean_object* v_M_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_00_u03c3_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_LinearMap_id_x27(v_R_568_, v_M_569_, v_inst_570_, v_inst_571_, v_inst_572_, v_00_u03c3_573_, v_inst_574_);
lean_dec(v_00_u03c3_573_);
lean_dec(v_inst_572_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object* v_f_576_, lean_object* v___y_577_){
_start:
{
lean_object* v___x_578_; 
v___x_578_ = lean_apply_1(v_f_576_, v___y_577_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg(lean_object* v_f_579_){
_start:
{
lean_object* v___f_580_; 
v___f_580_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_580_, 0, v_f_579_);
return v___f_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom(lean_object* v_R_581_, lean_object* v_S_582_, lean_object* v_M_u2081_583_, lean_object* v_M_u2082_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_inst_587_, lean_object* v_inst_588_, lean_object* v_modM_u2081_589_, lean_object* v_modM_u2082_590_, lean_object* v_00_u03c3_591_, lean_object* v_f_592_){
_start:
{
lean_object* v___f_593_; 
v___f_593_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_593_, 0, v_f_592_);
return v___f_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom___boxed(lean_object* v_R_594_, lean_object* v_S_595_, lean_object* v_M_u2081_596_, lean_object* v_M_u2082_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_modM_u2081_602_, lean_object* v_modM_u2082_603_, lean_object* v_00_u03c3_604_, lean_object* v_f_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_LinearMap_toAddMonoidHom(v_R_594_, v_S_595_, v_M_u2081_596_, v_M_u2082_597_, v_inst_598_, v_inst_599_, v_inst_600_, v_inst_601_, v_modM_u2081_602_, v_modM_u2082_603_, v_00_u03c3_604_, v_f_605_);
lean_dec(v_00_u03c3_604_);
lean_dec(v_modM_u2082_603_);
lean_dec(v_modM_u2081_602_);
lean_dec_ref(v_inst_601_);
lean_dec_ref(v_inst_600_);
lean_dec_ref(v_inst_599_);
lean_dec_ref(v_inst_598_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars___redArg___lam__0(lean_object* v_f_u2097_607_, lean_object* v___y_608_){
_start:
{
lean_object* v___x_609_; 
v___x_609_ = lean_apply_1(v_f_u2097_607_, v___y_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars___redArg(lean_object* v_f_u2097_610_){
_start:
{
lean_object* v___f_611_; 
v___f_611_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_611_, 0, v_f_u2097_610_);
return v___f_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars(lean_object* v_R_612_, lean_object* v_S_613_, lean_object* v_M_614_, lean_object* v_M_u2082_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_inst_620_, lean_object* v_inst_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_f_u2097_625_){
_start:
{
lean_object* v___f_626_; 
v___f_626_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_626_, 0, v_f_u2097_625_);
return v___f_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars___boxed(lean_object* v_R_627_, lean_object* v_S_628_, lean_object* v_M_629_, lean_object* v_M_u2082_630_, lean_object* v_inst_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_inst_638_, lean_object* v_inst_639_, lean_object* v_f_u2097_640_){
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_mathlib_LinearMap_restrictScalars(v_R_627_, v_S_628_, v_M_629_, v_M_u2082_630_, v_inst_631_, v_inst_632_, v_inst_633_, v_inst_634_, v_inst_635_, v_inst_636_, v_inst_637_, v_inst_638_, v_inst_639_, v_f_u2097_640_);
lean_dec(v_inst_638_);
lean_dec(v_inst_637_);
lean_dec(v_inst_636_);
lean_dec(v_inst_635_);
lean_dec_ref(v_inst_634_);
lean_dec_ref(v_inst_633_);
lean_dec_ref(v_inst_632_);
lean_dec_ref(v_inst_631_);
return v_res_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coeIsScalarTower___redArg(lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_inst_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___boxed), 14, 13);
lean_closure_set(v___x_650_, 0, lean_box(0));
lean_closure_set(v___x_650_, 1, lean_box(0));
lean_closure_set(v___x_650_, 2, lean_box(0));
lean_closure_set(v___x_650_, 3, lean_box(0));
lean_closure_set(v___x_650_, 4, v_inst_642_);
lean_closure_set(v___x_650_, 5, v_inst_643_);
lean_closure_set(v___x_650_, 6, v_inst_644_);
lean_closure_set(v___x_650_, 7, v_inst_645_);
lean_closure_set(v___x_650_, 8, v_inst_646_);
lean_closure_set(v___x_650_, 9, v_inst_647_);
lean_closure_set(v___x_650_, 10, v_inst_648_);
lean_closure_set(v___x_650_, 11, v_inst_649_);
lean_closure_set(v___x_650_, 12, lean_box(0));
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coeIsScalarTower(lean_object* v_R_651_, lean_object* v_S_652_, lean_object* v_M_653_, lean_object* v_M_u2082_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_inst_660_, lean_object* v_inst_661_, lean_object* v_inst_662_, lean_object* v_inst_663_){
_start:
{
lean_object* v___x_664_; 
v___x_664_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___boxed), 14, 13);
lean_closure_set(v___x_664_, 0, lean_box(0));
lean_closure_set(v___x_664_, 1, lean_box(0));
lean_closure_set(v___x_664_, 2, lean_box(0));
lean_closure_set(v___x_664_, 3, lean_box(0));
lean_closure_set(v___x_664_, 4, v_inst_655_);
lean_closure_set(v___x_664_, 5, v_inst_656_);
lean_closure_set(v___x_664_, 6, v_inst_657_);
lean_closure_set(v___x_664_, 7, v_inst_658_);
lean_closure_set(v___x_664_, 8, v_inst_659_);
lean_closure_set(v___x_664_, 9, v_inst_660_);
lean_closure_set(v___x_664_, 10, v_inst_661_);
lean_closure_set(v___x_664_, 11, v_inst_662_);
lean_closure_set(v___x_664_, 12, lean_box(0));
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap___redArg(lean_object* v_f_665_){
_start:
{
lean_inc(v_f_665_);
return v_f_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap___redArg___boxed(lean_object* v_f_666_){
_start:
{
lean_object* v_res_667_; 
v_res_667_ = lp_mathlib_RingHom_toSemilinearMap___redArg(v_f_666_);
lean_dec(v_f_666_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap(lean_object* v_R_668_, lean_object* v_S_669_, lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_f_672_){
_start:
{
lean_inc(v_f_672_);
return v_f_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toSemilinearMap___boxed(lean_object* v_R_673_, lean_object* v_S_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_f_677_){
_start:
{
lean_object* v_res_678_; 
v_res_678_ = lp_mathlib_RingHom_toSemilinearMap(v_R_673_, v_S_674_, v_inst_675_, v_inst_676_, v_f_677_);
lean_dec(v_f_677_);
lean_dec_ref(v_inst_676_);
lean_dec_ref(v_inst_675_);
return v_res_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object* v_g_679_, lean_object* v_f_680_, lean_object* v_x_681_){
_start:
{
lean_object* v___x_682_; lean_object* v___x_683_; 
v___x_682_ = lean_apply_1(v_g_679_, v_x_681_);
v___x_683_ = lean_apply_1(v_f_680_, v___x_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object* v_f_684_, lean_object* v_g_685_){
_start:
{
lean_object* v___f_686_; 
v___f_686_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_686_, 0, v_g_685_);
lean_closure_set(v___f_686_, 1, v_f_684_);
return v___f_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp(lean_object* v_R_u2081_687_, lean_object* v_R_u2082_688_, lean_object* v_R_u2083_689_, lean_object* v_M_u2081_690_, lean_object* v_M_u2082_691_, lean_object* v_M_u2083_692_, lean_object* v_inst_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_module__M_u2081_699_, lean_object* v_module__M_u2082_700_, lean_object* v_module__M_u2083_701_, lean_object* v_00_u03c3_u2081_u2082_702_, lean_object* v_00_u03c3_u2082_u2083_703_, lean_object* v_00_u03c3_u2081_u2083_704_, lean_object* v_inst_705_, lean_object* v_f_706_, lean_object* v_g_707_){
_start:
{
lean_object* v___f_708_; 
v___f_708_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_708_, 0, v_g_707_);
lean_closure_set(v___f_708_, 1, v_f_706_);
return v___f_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_comp___boxed(lean_object** _args){
lean_object* v_R_u2081_709_ = _args[0];
lean_object* v_R_u2082_710_ = _args[1];
lean_object* v_R_u2083_711_ = _args[2];
lean_object* v_M_u2081_712_ = _args[3];
lean_object* v_M_u2082_713_ = _args[4];
lean_object* v_M_u2083_714_ = _args[5];
lean_object* v_inst_715_ = _args[6];
lean_object* v_inst_716_ = _args[7];
lean_object* v_inst_717_ = _args[8];
lean_object* v_inst_718_ = _args[9];
lean_object* v_inst_719_ = _args[10];
lean_object* v_inst_720_ = _args[11];
lean_object* v_module__M_u2081_721_ = _args[12];
lean_object* v_module__M_u2082_722_ = _args[13];
lean_object* v_module__M_u2083_723_ = _args[14];
lean_object* v_00_u03c3_u2081_u2082_724_ = _args[15];
lean_object* v_00_u03c3_u2082_u2083_725_ = _args[16];
lean_object* v_00_u03c3_u2081_u2083_726_ = _args[17];
lean_object* v_inst_727_ = _args[18];
lean_object* v_f_728_ = _args[19];
lean_object* v_g_729_ = _args[20];
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_LinearMap_comp(v_R_u2081_709_, v_R_u2082_710_, v_R_u2083_711_, v_M_u2081_712_, v_M_u2082_713_, v_M_u2083_714_, v_inst_715_, v_inst_716_, v_inst_717_, v_inst_718_, v_inst_719_, v_inst_720_, v_module__M_u2081_721_, v_module__M_u2082_722_, v_module__M_u2083_723_, v_00_u03c3_u2081_u2082_724_, v_00_u03c3_u2082_u2083_725_, v_00_u03c3_u2081_u2083_726_, v_inst_727_, v_f_728_, v_g_729_);
lean_dec(v_00_u03c3_u2081_u2083_726_);
lean_dec(v_00_u03c3_u2082_u2083_725_);
lean_dec(v_00_u03c3_u2081_u2082_724_);
lean_dec(v_module__M_u2083_723_);
lean_dec(v_module__M_u2082_722_);
lean_dec(v_module__M_u2081_721_);
lean_dec_ref(v_inst_720_);
lean_dec_ref(v_inst_719_);
lean_dec_ref(v_inst_718_);
lean_dec_ref(v_inst_717_);
lean_dec_ref(v_inst_716_);
lean_dec_ref(v_inst_715_);
return v_res_730_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__1(void){
_start:
{
lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_752_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__0));
v___x_753_ = l_String_toRawSubstring_x27(v___x_752_);
return v___x_753_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__9(void){
_start:
{
lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_771_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__8));
v___x_772_ = l_String_toRawSubstring_x27(v___x_771_);
return v___x_772_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__23(void){
_start:
{
lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; 
v___x_801_ = lean_unsigned_to_nat(3334348489u);
v___x_802_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__22));
v___x_803_ = l_Lean_Name_num___override(v___x_802_, v___x_801_);
return v___x_803_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__25(void){
_start:
{
lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; 
v___x_805_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__24));
v___x_806_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__23, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__23_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__23);
v___x_807_ = l_Lean_Name_str___override(v___x_806_, v___x_805_);
return v___x_807_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__27(void){
_start:
{
lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; 
v___x_809_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__26));
v___x_810_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__25, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__25_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__25);
v___x_811_ = l_Lean_Name_str___override(v___x_810_, v___x_809_);
return v___x_811_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__28(void){
_start:
{
lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; 
v___x_812_ = lean_unsigned_to_nat(67u);
v___x_813_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__27, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__27_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__27);
v___x_814_ = l_Lean_Name_num___override(v___x_813_, v___x_812_);
return v___x_814_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__29(void){
_start:
{
lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; 
v___x_815_ = lean_box(0);
v___x_816_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__28, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__28_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__28);
v___x_817_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_817_, 0, v___x_816_);
lean_ctor_set(v___x_817_, 1, v___x_815_);
return v___x_817_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__30(void){
_start:
{
lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; 
v___x_818_ = lean_box(0);
v___x_819_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__29, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__29_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__29);
v___x_820_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_820_, 0, v___x_819_);
lean_ctor_set(v___x_820_, 1, v___x_818_);
return v___x_820_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__36(void){
_start:
{
lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_830_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__35));
v___x_831_ = l_String_toRawSubstring_x27(v___x_830_);
return v___x_831_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__45(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_855_ = lean_unsigned_to_nat(3334348489u);
v___x_856_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__44));
v___x_857_ = l_Lean_Name_num___override(v___x_856_, v___x_855_);
return v___x_857_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__46(void){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_858_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__24));
v___x_859_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__45, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__45_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__45);
v___x_860_ = l_Lean_Name_str___override(v___x_859_, v___x_858_);
return v___x_860_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__47(void){
_start:
{
lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; 
v___x_861_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__26));
v___x_862_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__46, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__46_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__46);
v___x_863_ = l_Lean_Name_str___override(v___x_862_, v___x_861_);
return v___x_863_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__48(void){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v___x_864_ = lean_unsigned_to_nat(68u);
v___x_865_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__47, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__47_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__47);
v___x_866_ = l_Lean_Name_num___override(v___x_865_, v___x_864_);
return v___x_866_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__49(void){
_start:
{
lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; 
v___x_867_ = lean_box(0);
v___x_868_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__48, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__48_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__48);
v___x_869_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_869_, 0, v___x_868_);
lean_ctor_set(v___x_869_, 1, v___x_867_);
return v___x_869_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__50(void){
_start:
{
lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; 
v___x_870_ = lean_box(0);
v___x_871_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__49, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__49_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__49);
v___x_872_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_872_, 0, v___x_871_);
lean_ctor_set(v___x_872_, 1, v___x_870_);
return v___x_872_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__52(void){
_start:
{
lean_object* v___x_874_; lean_object* v___x_875_; 
v___x_874_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__51));
v___x_875_ = l_String_toRawSubstring_x27(v___x_874_);
return v___x_875_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__61(void){
_start:
{
lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; 
v___x_899_ = lean_unsigned_to_nat(3334348489u);
v___x_900_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__60));
v___x_901_ = l_Lean_Name_num___override(v___x_900_, v___x_899_);
return v___x_901_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__62(void){
_start:
{
lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; 
v___x_902_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__24));
v___x_903_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__61, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__61_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__61);
v___x_904_ = l_Lean_Name_str___override(v___x_903_, v___x_902_);
return v___x_904_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__63(void){
_start:
{
lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; 
v___x_905_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__26));
v___x_906_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__62, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__62_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__62);
v___x_907_ = l_Lean_Name_str___override(v___x_906_, v___x_905_);
return v___x_907_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__64(void){
_start:
{
lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; 
v___x_908_ = lean_unsigned_to_nat(69u);
v___x_909_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__63, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__63_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__63);
v___x_910_ = l_Lean_Name_num___override(v___x_909_, v___x_908_);
return v___x_910_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__65(void){
_start:
{
lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; 
v___x_911_ = lean_box(0);
v___x_912_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__64, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__64_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__64);
v___x_913_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_913_, 0, v___x_912_);
lean_ctor_set(v___x_913_, 1, v___x_911_);
return v___x_913_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__66(void){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; 
v___x_914_ = lean_box(0);
v___x_915_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__65, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__65_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__65);
v___x_916_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_916_, 0, v___x_915_);
lean_ctor_set(v___x_916_, 1, v___x_914_);
return v___x_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1(lean_object* v_x_917_, lean_object* v_a_918_, lean_object* v_a_919_){
_start:
{
lean_object* v___x_920_; uint8_t v___x_921_; 
v___x_920_ = ((lean_object*)(lp_mathlib_LinearMap_compNotation___closed__1));
lean_inc(v_x_917_);
v___x_921_ = l_Lean_Syntax_isOfKind(v_x_917_, v___x_920_);
if (v___x_921_ == 0)
{
lean_object* v___x_922_; lean_object* v___x_923_; 
lean_dec(v_x_917_);
v___x_922_ = lean_box(1);
v___x_923_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_923_, 0, v___x_922_);
lean_ctor_set(v___x_923_, 1, v_a_919_);
return v___x_923_;
}
else
{
lean_object* v_quotContext_924_; lean_object* v_currMacroScope_925_; lean_object* v_ref_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; uint8_t v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; 
v_quotContext_924_ = lean_ctor_get(v_a_918_, 1);
v_currMacroScope_925_ = lean_ctor_get(v_a_918_, 2);
v_ref_926_ = lean_ctor_get(v_a_918_, 5);
v___x_927_ = lean_unsigned_to_nat(0u);
v___x_928_ = l_Lean_Syntax_getArg(v_x_917_, v___x_927_);
v___x_929_ = lean_unsigned_to_nat(2u);
v___x_930_ = l_Lean_Syntax_getArg(v_x_917_, v___x_929_);
lean_dec(v_x_917_);
v___x_931_ = 0;
v___x_932_ = l_Lean_SourceInfo_fromRef(v_ref_926_, v___x_931_);
v___x_933_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
v___x_934_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__1, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__1_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__1);
v___x_935_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3));
lean_inc_n(v_currMacroScope_925_, 5);
lean_inc_n(v_quotContext_924_, 5);
v___x_936_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_935_, v_currMacroScope_925_);
v___x_937_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__5));
lean_inc_n(v___x_932_, 16);
v___x_938_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_938_, 0, v___x_932_);
lean_ctor_set(v___x_938_, 1, v___x_934_);
lean_ctor_set(v___x_938_, 2, v___x_936_);
lean_ctor_set(v___x_938_, 3, v___x_937_);
v___x_939_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13));
v___x_940_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__7));
v___x_941_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__4));
v___x_942_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_942_, 0, v___x_932_);
lean_ctor_set(v___x_942_, 1, v___x_941_);
v___x_943_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__9, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__9_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__9);
v___x_944_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__10));
v___x_945_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_944_, v_currMacroScope_925_);
v___x_946_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__30, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__30_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__30);
v___x_947_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_947_, 0, v___x_932_);
lean_ctor_set(v___x_947_, 1, v___x_943_);
lean_ctor_set(v___x_947_, 2, v___x_945_);
lean_ctor_set(v___x_947_, 3, v___x_946_);
v___x_948_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__31));
v___x_949_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_949_, 0, v___x_932_);
lean_ctor_set(v___x_949_, 1, v___x_948_);
v___x_950_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16, &lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__16);
v___x_951_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19));
v___x_952_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_951_, v_currMacroScope_925_);
v___x_953_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__21));
v___x_954_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_954_, 0, v___x_932_);
lean_ctor_set(v___x_954_, 1, v___x_950_);
lean_ctor_set(v___x_954_, 2, v___x_952_);
lean_ctor_set(v___x_954_, 3, v___x_953_);
v___x_955_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__33));
v___x_956_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__34));
v___x_957_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_957_, 0, v___x_932_);
lean_ctor_set(v___x_957_, 1, v___x_956_);
v___x_958_ = l_Lean_Syntax_node1(v___x_932_, v___x_955_, v___x_957_);
v___x_959_ = l_Lean_Syntax_node1(v___x_932_, v___x_939_, v___x_958_);
v___x_960_ = l_Lean_Syntax_node2(v___x_932_, v___x_933_, v___x_954_, v___x_959_);
v___x_961_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__22));
v___x_962_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_962_, 0, v___x_932_);
lean_ctor_set(v___x_962_, 1, v___x_961_);
lean_inc_ref_n(v___x_962_, 2);
lean_inc_n(v___x_960_, 2);
lean_inc_ref_n(v___x_949_, 2);
lean_inc_ref_n(v___x_942_, 2);
v___x_963_ = l_Lean_Syntax_node5(v___x_932_, v___x_940_, v___x_942_, v___x_947_, v___x_949_, v___x_960_, v___x_962_);
v___x_964_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__36, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__36_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__36);
v___x_965_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__37));
v___x_966_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_965_, v_currMacroScope_925_);
v___x_967_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__50, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__50_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__50);
v___x_968_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_968_, 0, v___x_932_);
lean_ctor_set(v___x_968_, 1, v___x_964_);
lean_ctor_set(v___x_968_, 2, v___x_966_);
lean_ctor_set(v___x_968_, 3, v___x_967_);
v___x_969_ = l_Lean_Syntax_node5(v___x_932_, v___x_940_, v___x_942_, v___x_968_, v___x_949_, v___x_960_, v___x_962_);
v___x_970_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__52, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__52_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__52);
v___x_971_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__53));
v___x_972_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_971_, v_currMacroScope_925_);
v___x_973_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__66, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__66_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__66);
v___x_974_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_974_, 0, v___x_932_);
lean_ctor_set(v___x_974_, 1, v___x_970_);
lean_ctor_set(v___x_974_, 2, v___x_972_);
lean_ctor_set(v___x_974_, 3, v___x_973_);
v___x_975_ = l_Lean_Syntax_node5(v___x_932_, v___x_940_, v___x_942_, v___x_974_, v___x_949_, v___x_960_, v___x_962_);
v___x_976_ = l_Lean_Syntax_node5(v___x_932_, v___x_939_, v___x_963_, v___x_969_, v___x_975_, v___x_928_, v___x_930_);
v___x_977_ = l_Lean_Syntax_node2(v___x_932_, v___x_933_, v___x_938_, v___x_976_);
v___x_978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_978_, 0, v___x_977_);
lean_ctor_set(v___x_978_, 1, v_a_919_);
return v___x_978_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___boxed(lean_object* v_x_979_, lean_object* v_a_980_, lean_object* v_a_981_){
_start:
{
lean_object* v_res_982_; 
v_res_982_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1(v_x_979_, v_a_980_, v_a_981_);
lean_dec_ref(v_a_980_);
return v_res_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg(lean_object* v___y_983_){
_start:
{
lean_object* v_subExpr_985_; lean_object* v_expr_986_; lean_object* v___x_987_; 
v_subExpr_985_ = lean_ctor_get(v___y_983_, 3);
v_expr_986_ = lean_ctor_get(v_subExpr_985_, 0);
lean_inc_ref(v_expr_986_);
v___x_987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_987_, 0, v_expr_986_);
return v___x_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg___boxed(lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
lean_object* v_res_990_; 
v_res_990_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg(v___y_988_);
lean_dec_ref(v___y_988_);
return v_res_990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0(lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
lean_object* v___x_998_; 
v___x_998_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg(v___y_991_);
return v___x_998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___boxed(lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_){
_start:
{
lean_object* v_res_1006_; 
v_res_1006_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0(v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_, v___y_1003_, v___y_1004_);
lean_dec(v___y_1004_);
lean_dec_ref(v___y_1003_);
lean_dec(v___y_1002_);
lean_dec_ref(v___y_1001_);
lean_dec(v___y_1000_);
lean_dec_ref(v___y_999_);
return v_res_1006_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__0(lean_object* v_x_1007_){
_start:
{
lean_object* v___x_1008_; uint8_t v___x_1009_; 
v___x_1008_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__3));
v___x_1009_ = l_Lean_Expr_isConstOf(v_x_1007_, v___x_1008_);
return v___x_1009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__0___boxed(lean_object* v_x_1010_){
_start:
{
uint8_t v_res_1011_; lean_object* v_r_1012_; 
v_res_1011_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__0(v_x_1010_);
lean_dec_ref(v_x_1010_);
v_r_1012_ = lean_box(v_res_1011_);
return v_r_1012_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__1(lean_object* v_x_1013_){
_start:
{
lean_object* v___x_1014_; uint8_t v___x_1015_; 
v___x_1014_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u2097_x5b___x5d____1___closed__19));
v___x_1015_ = l_Lean_Expr_isConstOf(v_x_1013_, v___x_1014_);
return v___x_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__1___boxed(lean_object* v_x_1016_){
_start:
{
uint8_t v_res_1017_; lean_object* v_r_1018_; 
v_res_1017_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__1(v_x_1016_);
lean_dec_ref(v_x_1016_);
v_r_1018_ = lean_box(v_res_1017_);
return v_r_1018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__2(lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_){
_start:
{
lean_object* v___x_1027_; 
v___x_1027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1027_, 0, v___y_1019_);
return v___x_1027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__2___boxed(lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v_res_1036_; 
v_res_1036_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__2(v___y_1028_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_);
lean_dec(v___y_1034_);
lean_dec_ref(v___y_1033_);
lean_dec(v___y_1032_);
lean_dec_ref(v___y_1031_);
lean_dec(v___y_1030_);
lean_dec_ref(v___y_1029_);
return v_res_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__3(lean_object* v_a_1037_, lean_object* v_a_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_){
_start:
{
lean_object* v_ref_1046_; uint8_t v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v_ref_1046_ = lean_ctor_get(v___y_1043_, 5);
v___x_1047_ = 0;
v___x_1048_ = l_Lean_SourceInfo_fromRef(v_ref_1046_, v___x_1047_);
v___x_1049_ = ((lean_object*)(lp_mathlib_LinearMap_compNotation___closed__1));
v___x_1050_ = ((lean_object*)(lp_mathlib_LinearMap_compNotation___closed__2));
lean_inc(v___x_1048_);
v___x_1051_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1051_, 0, v___x_1048_);
lean_ctor_set(v___x_1051_, 1, v___x_1050_);
v___x_1052_ = l_Lean_Syntax_node3(v___x_1048_, v___x_1049_, v_a_1037_, v___x_1051_, v_a_1038_);
v___x_1053_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__3___boxed(lean_object* v_a_1054_, lean_object* v_a_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__3(v_a_1054_, v_a_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
lean_dec(v___y_1057_);
lean_dec_ref(v___y_1056_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4(lean_object* v___f_1074_, lean_object* v___f_1075_, lean_object* v___f_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v___x_1084_; lean_object* v_a_1085_; lean_object* v___x_1087_; uint8_t v_isShared_1088_; uint8_t v_isSharedCheck_1137_; 
v___x_1084_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1_spec__0___redArg(v___y_1077_);
v_a_1085_ = lean_ctor_get(v___x_1084_, 0);
v_isSharedCheck_1137_ = !lean_is_exclusive(v___x_1084_);
if (v_isSharedCheck_1137_ == 0)
{
v___x_1087_ = v___x_1084_;
v_isShared_1088_ = v_isSharedCheck_1137_;
goto v_resetjp_1086_;
}
else
{
lean_inc(v_a_1085_);
lean_dec(v___x_1084_);
v___x_1087_ = lean_box(0);
v_isShared_1088_ = v_isSharedCheck_1137_;
goto v_resetjp_1086_;
}
v_resetjp_1086_:
{
lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; 
v___x_1089_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_1089_, 0, v___f_1074_);
lean_inc_ref_n(v___f_1075_, 17);
v___x_1090_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1090_, 0, v___x_1089_);
lean_closure_set(v___x_1090_, 1, v___f_1075_);
v___x_1091_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1091_, 0, v___x_1090_);
lean_closure_set(v___x_1091_, 1, v___f_1075_);
v___x_1092_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1092_, 0, v___x_1091_);
lean_closure_set(v___x_1092_, 1, v___f_1075_);
v___x_1093_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1093_, 0, v___x_1092_);
lean_closure_set(v___x_1093_, 1, v___f_1075_);
v___x_1094_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1094_, 0, v___x_1093_);
lean_closure_set(v___x_1094_, 1, v___f_1075_);
v___x_1095_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1095_, 0, v___x_1094_);
lean_closure_set(v___x_1095_, 1, v___f_1075_);
v___x_1096_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1096_, 0, v___x_1095_);
lean_closure_set(v___x_1096_, 1, v___f_1075_);
v___x_1097_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1097_, 0, v___x_1096_);
lean_closure_set(v___x_1097_, 1, v___f_1075_);
v___x_1098_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1098_, 0, v___x_1097_);
lean_closure_set(v___x_1098_, 1, v___f_1075_);
v___x_1099_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1099_, 0, v___x_1098_);
lean_closure_set(v___x_1099_, 1, v___f_1075_);
v___x_1100_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1100_, 0, v___x_1099_);
lean_closure_set(v___x_1100_, 1, v___f_1075_);
v___x_1101_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1101_, 0, v___x_1100_);
lean_closure_set(v___x_1101_, 1, v___f_1075_);
v___x_1102_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1102_, 0, v___x_1101_);
lean_closure_set(v___x_1102_, 1, v___f_1075_);
v___x_1103_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1103_, 0, v___x_1102_);
lean_closure_set(v___x_1103_, 1, v___f_1075_);
v___x_1104_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1104_, 0, v___x_1103_);
lean_closure_set(v___x_1104_, 1, v___f_1075_);
v___x_1105_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_1105_, 0, v___f_1076_);
v___x_1106_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1106_, 0, v___x_1105_);
lean_closure_set(v___x_1106_, 1, v___f_1075_);
v___x_1107_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1107_, 0, v___x_1106_);
lean_closure_set(v___x_1107_, 1, v___f_1075_);
lean_inc_ref_n(v___x_1107_, 2);
v___x_1108_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1108_, 0, v___x_1104_);
lean_closure_set(v___x_1108_, 1, v___x_1107_);
v___x_1109_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1109_, 0, v___x_1108_);
lean_closure_set(v___x_1109_, 1, v___x_1107_);
v___x_1110_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1110_, 0, v___x_1109_);
lean_closure_set(v___x_1110_, 1, v___x_1107_);
v___x_1111_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1111_, 0, v___x_1110_);
lean_closure_set(v___x_1111_, 1, v___f_1075_);
v___x_1112_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__1));
v___x_1113_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__2));
v___x_1114_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1114_, 0, v___x_1111_);
lean_closure_set(v___x_1114_, 1, v___x_1113_);
v___x_1115_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__4));
v___x_1116_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___closed__5));
v___x_1117_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_1118_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_1114_, v___x_1116_, v___x_1117_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; lean_object* v___x_1121_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
lean_inc(v_a_1119_);
lean_dec_ref_known(v___x_1118_, 1);
if (v_isShared_1088_ == 0)
{
lean_ctor_set_tag(v___x_1087_, 1);
v___x_1121_ = v___x_1087_;
goto v_reusejp_1120_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v_a_1085_);
v___x_1121_ = v_reuseFailAlloc_1128_;
goto v_reusejp_1120_;
}
v_reusejp_1120_:
{
lean_object* v___x_1122_; 
lean_inc_ref(v___x_1121_);
v___x_1122_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_1119_, v___x_1115_, v___x_1121_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_);
if (lean_obj_tag(v___x_1122_) == 0)
{
lean_object* v_a_1123_; lean_object* v___x_1124_; 
v_a_1123_ = lean_ctor_get(v___x_1122_, 0);
lean_inc(v_a_1123_);
lean_dec_ref_known(v___x_1122_, 1);
v___x_1124_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_1119_, v___x_1112_, v___x_1121_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_);
lean_dec(v_a_1119_);
if (lean_obj_tag(v___x_1124_) == 0)
{
lean_object* v_a_1125_; lean_object* v___f_1126_; lean_object* v___x_1127_; 
v_a_1125_ = lean_ctor_get(v___x_1124_, 0);
lean_inc(v_a_1125_);
lean_dec_ref_known(v___x_1124_, 1);
v___f_1126_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__3___boxed), 9, 2);
lean_closure_set(v___f_1126_, 0, v_a_1125_);
lean_closure_set(v___f_1126_, 1, v_a_1123_);
v___x_1127_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_1126_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_);
return v___x_1127_;
}
else
{
lean_dec(v_a_1123_);
return v___x_1124_;
}
}
else
{
lean_dec_ref(v___x_1121_);
lean_dec(v_a_1119_);
return v___x_1122_;
}
}
}
else
{
lean_object* v_a_1129_; lean_object* v___x_1131_; uint8_t v_isShared_1132_; uint8_t v_isSharedCheck_1136_; 
lean_del_object(v___x_1087_);
lean_dec(v_a_1085_);
v_a_1129_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1136_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1136_ == 0)
{
v___x_1131_ = v___x_1118_;
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
else
{
lean_inc(v_a_1129_);
lean_dec(v___x_1118_);
v___x_1131_ = lean_box(0);
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
v_resetjp_1130_:
{
lean_object* v___x_1134_; 
if (v_isShared_1132_ == 0)
{
v___x_1134_ = v___x_1131_;
goto v_reusejp_1133_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v_a_1129_);
v___x_1134_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1133_;
}
v_reusejp_1133_:
{
return v___x_1134_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4___boxed(lean_object* v___f_1138_, lean_object* v___f_1139_, lean_object* v___f_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_){
_start:
{
lean_object* v_res_1148_; 
v_res_1148_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___lam__4(v___f_1138_, v___f_1139_, v___f_1140_, v___y_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_);
lean_dec(v___y_1146_);
lean_dec_ref(v___y_1145_);
lean_dec(v___y_1144_);
lean_dec_ref(v___y_1143_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
return v_res_1148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1(lean_object* v_a_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_){
_start:
{
lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; 
v___x_1171_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__4));
v___x_1172_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___closed__7));
v___x_1173_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_1171_, v___x_1172_, v_a_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1___boxed(lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_){
_start:
{
lean_object* v_res_1181_; 
v_res_1181_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______delab__app__LinearMap__compNotation__1(v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_);
lean_dec(v_a_1179_);
lean_dec_ref(v_a_1178_);
lean_dec(v_a_1177_);
lean_dec_ref(v_a_1176_);
lean_dec(v_a_1175_);
lean_dec_ref(v_a_1174_);
return v_res_1181_;
}
}
static lean_object* _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__0(void){
_start:
{
lean_object* v___x_1202_; lean_object* v___x_1203_; 
v___x_1202_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__2));
v___x_1203_ = l_String_toRawSubstring_x27(v___x_1202_);
return v___x_1203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1(lean_object* v_x_1206_, lean_object* v_a_1207_, lean_object* v_a_1208_){
_start:
{
lean_object* v___x_1209_; uint8_t v___x_1210_; 
v___x_1209_ = ((lean_object*)(lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1));
lean_inc(v_x_1206_);
v___x_1210_ = l_Lean_Syntax_isOfKind(v_x_1206_, v___x_1209_);
if (v___x_1210_ == 0)
{
lean_object* v___x_1211_; lean_object* v___x_1212_; 
lean_dec(v_x_1206_);
v___x_1211_ = lean_box(1);
v___x_1212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1212_, 0, v___x_1211_);
lean_ctor_set(v___x_1212_, 1, v_a_1208_);
return v___x_1212_;
}
else
{
lean_object* v_quotContext_1213_; lean_object* v_currMacroScope_1214_; lean_object* v_ref_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; uint8_t v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; 
v_quotContext_1213_ = lean_ctor_get(v_a_1207_, 1);
v_currMacroScope_1214_ = lean_ctor_get(v_a_1207_, 2);
v_ref_1215_ = lean_ctor_get(v_a_1207_, 5);
v___x_1216_ = lean_unsigned_to_nat(0u);
v___x_1217_ = l_Lean_Syntax_getArg(v_x_1206_, v___x_1216_);
v___x_1218_ = lean_unsigned_to_nat(2u);
v___x_1219_ = l_Lean_Syntax_getArg(v_x_1206_, v___x_1218_);
lean_dec(v_x_1206_);
v___x_1220_ = 0;
v___x_1221_ = l_Lean_SourceInfo_fromRef(v_ref_1215_, v___x_1220_);
v___x_1222_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
v___x_1223_ = lean_obj_once(&lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__0, &lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__0_once, _init_lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__0);
v___x_1224_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___closed__1));
lean_inc(v_currMacroScope_1214_);
lean_inc(v_quotContext_1213_);
v___x_1225_ = l_Lean_addMacroScope(v_quotContext_1213_, v___x_1224_, v_currMacroScope_1214_);
v___x_1226_ = ((lean_object*)(lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__compNotation__1___closed__5));
lean_inc_n(v___x_1221_, 2);
v___x_1227_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1227_, 0, v___x_1221_);
lean_ctor_set(v___x_1227_, 1, v___x_1223_);
lean_ctor_set(v___x_1227_, 2, v___x_1225_);
lean_ctor_set(v___x_1227_, 3, v___x_1226_);
v___x_1228_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__13));
v___x_1229_ = l_Lean_Syntax_node2(v___x_1221_, v___x_1228_, v___x_1217_, v___x_1219_);
v___x_1230_ = l_Lean_Syntax_node2(v___x_1221_, v___x_1222_, v___x_1227_, v___x_1229_);
v___x_1231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1230_);
lean_ctor_set(v___x_1231_, 1, v_a_1208_);
return v___x_1231_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1___boxed(lean_object* v_x_1232_, lean_object* v_a_1233_, lean_object* v_a_1234_){
_start:
{
lean_object* v_res_1235_; 
v_res_1235_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__LinearMap__term___u2218_u209b_u2097____1(v_x_1232_, v_a_1233_, v_a_1234_);
lean_dec_ref(v_a_1233_);
return v_res_1235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__comp__1(lean_object* v_x_1236_, lean_object* v_a_1237_, lean_object* v_a_1238_){
_start:
{
lean_object* v___x_1239_; uint8_t v___x_1240_; 
v___x_1239_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______macroRules__term___u2192_u209b_u2097_x5b___x5d____1___closed__4));
lean_inc(v_x_1236_);
v___x_1240_ = l_Lean_Syntax_isOfKind(v_x_1236_, v___x_1239_);
if (v___x_1240_ == 0)
{
lean_object* v___x_1241_; lean_object* v___x_1242_; 
lean_dec(v_x_1236_);
v___x_1241_ = lean_box(0);
v___x_1242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1242_, 0, v___x_1241_);
lean_ctor_set(v___x_1242_, 1, v_a_1238_);
return v___x_1242_;
}
else
{
lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; uint8_t v___x_1246_; 
v___x_1243_ = lean_unsigned_to_nat(0u);
v___x_1244_ = l_Lean_Syntax_getArg(v_x_1236_, v___x_1243_);
v___x_1245_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__1___closed__1));
lean_inc(v___x_1244_);
v___x_1246_ = l_Lean_Syntax_isOfKind(v___x_1244_, v___x_1245_);
if (v___x_1246_ == 0)
{
lean_object* v___x_1247_; lean_object* v___x_1248_; 
lean_dec(v___x_1244_);
lean_dec(v_x_1236_);
v___x_1247_ = lean_box(0);
v___x_1248_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
lean_ctor_set(v___x_1248_, 1, v_a_1238_);
return v___x_1248_;
}
else
{
lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; uint8_t v___x_1252_; 
v___x_1249_ = lean_unsigned_to_nat(1u);
v___x_1250_ = l_Lean_Syntax_getArg(v_x_1236_, v___x_1249_);
lean_dec(v_x_1236_);
v___x_1251_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_1250_);
v___x_1252_ = l_Lean_Syntax_matchesNull(v___x_1250_, v___x_1251_);
if (v___x_1252_ == 0)
{
lean_object* v___x_1253_; lean_object* v___x_1254_; 
lean_dec(v___x_1250_);
lean_dec(v___x_1244_);
v___x_1253_ = lean_box(0);
v___x_1254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1254_, 0, v___x_1253_);
lean_ctor_set(v___x_1254_, 1, v_a_1238_);
return v___x_1254_;
}
else
{
lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v_ref_1257_; uint8_t v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; 
v___x_1255_ = l_Lean_Syntax_getArg(v___x_1250_, v___x_1243_);
v___x_1256_ = l_Lean_Syntax_getArg(v___x_1250_, v___x_1249_);
lean_dec(v___x_1250_);
v_ref_1257_ = l_Lean_replaceRef(v___x_1244_, v_a_1237_);
lean_dec(v___x_1244_);
v___x_1258_ = 0;
v___x_1259_ = l_Lean_SourceInfo_fromRef(v_ref_1257_, v___x_1258_);
lean_dec(v_ref_1257_);
v___x_1260_ = ((lean_object*)(lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__1));
v___x_1261_ = ((lean_object*)(lp_mathlib_LinearMap_term___u2218_u209b_u2097___00__closed__2));
lean_inc(v___x_1259_);
v___x_1262_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1262_, 0, v___x_1259_);
lean_ctor_set(v___x_1262_, 1, v___x_1261_);
v___x_1263_ = l_Lean_Syntax_node3(v___x_1259_, v___x_1260_, v___x_1255_, v___x_1262_, v___x_1256_);
v___x_1264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1263_);
lean_ctor_set(v___x_1264_, 1, v_a_1238_);
return v___x_1264_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__comp__1___boxed(lean_object* v_x_1265_, lean_object* v_a_1266_, lean_object* v_a_1267_){
_start:
{
lean_object* v_res_1268_; 
v_res_1268_ = lp_mathlib_LinearMap___aux__Mathlib__Algebra__Module__LinearMap__Defs______unexpand__LinearMap__comp__1(v_x_1265_, v_a_1266_, v_a_1267_);
lean_dec(v_a_1266_);
return v_res_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse___redArg(lean_object* v_g_1269_){
_start:
{
lean_inc(v_g_1269_);
return v_g_1269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse___redArg___boxed(lean_object* v_g_1270_){
_start:
{
lean_object* v_res_1271_; 
v_res_1271_ = lp_mathlib_LinearMap_inverse___redArg(v_g_1270_);
lean_dec(v_g_1270_);
return v_res_1271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse(lean_object* v_R_1272_, lean_object* v_S_1273_, lean_object* v_M_1274_, lean_object* v_M_u2082_1275_, lean_object* v_inst_1276_, lean_object* v_inst_1277_, lean_object* v_inst_1278_, lean_object* v_inst_1279_, lean_object* v_inst_1280_, lean_object* v_inst_1281_, lean_object* v_00_u03c3_1282_, lean_object* v_00_u03c3_x27_1283_, lean_object* v_inst_1284_, lean_object* v_f_1285_, lean_object* v_g_1286_, lean_object* v_h_u2081_1287_, lean_object* v_h_u2082_1288_){
_start:
{
lean_inc(v_g_1286_);
return v_g_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inverse___boxed(lean_object** _args){
lean_object* v_R_1289_ = _args[0];
lean_object* v_S_1290_ = _args[1];
lean_object* v_M_1291_ = _args[2];
lean_object* v_M_u2082_1292_ = _args[3];
lean_object* v_inst_1293_ = _args[4];
lean_object* v_inst_1294_ = _args[5];
lean_object* v_inst_1295_ = _args[6];
lean_object* v_inst_1296_ = _args[7];
lean_object* v_inst_1297_ = _args[8];
lean_object* v_inst_1298_ = _args[9];
lean_object* v_00_u03c3_1299_ = _args[10];
lean_object* v_00_u03c3_x27_1300_ = _args[11];
lean_object* v_inst_1301_ = _args[12];
lean_object* v_f_1302_ = _args[13];
lean_object* v_g_1303_ = _args[14];
lean_object* v_h_u2081_1304_ = _args[15];
lean_object* v_h_u2082_1305_ = _args[16];
_start:
{
lean_object* v_res_1306_; 
v_res_1306_ = lp_mathlib_LinearMap_inverse(v_R_1289_, v_S_1290_, v_M_1291_, v_M_u2082_1292_, v_inst_1293_, v_inst_1294_, v_inst_1295_, v_inst_1296_, v_inst_1297_, v_inst_1298_, v_00_u03c3_1299_, v_00_u03c3_x27_1300_, v_inst_1301_, v_f_1302_, v_g_1303_, v_h_u2081_1304_, v_h_u2082_1305_);
lean_dec(v_g_1303_);
lean_dec(v_f_1302_);
lean_dec(v_00_u03c3_x27_1300_);
lean_dec(v_00_u03c3_1299_);
lean_dec(v_inst_1298_);
lean_dec(v_inst_1297_);
lean_dec_ref(v_inst_1296_);
lean_dec_ref(v_inst_1295_);
lean_dec_ref(v_inst_1294_);
lean_dec_ref(v_inst_1293_);
return v_res_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap___redArg___lam__0(lean_object* v_g_1307_, lean_object* v___y_1308_){
_start:
{
lean_object* v___x_1309_; 
v___x_1309_ = lean_apply_1(v_g_1307_, v___y_1308_);
return v___x_1309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap___redArg(lean_object* v_g_1310_){
_start:
{
lean_object* v___f_1311_; 
v___f_1311_ = lean_alloc_closure((void*)(lp_mathlib_Module_compHom_toLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1311_, 0, v_g_1310_);
return v___f_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap(lean_object* v_R_1312_, lean_object* v_S_1313_, lean_object* v_inst_1314_, lean_object* v_inst_1315_, lean_object* v_g_1316_){
_start:
{
lean_object* v___f_1317_; 
v___f_1317_ = lean_alloc_closure((void*)(lp_mathlib_Module_compHom_toLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1317_, 0, v_g_1316_);
return v___f_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_compHom_toLinearMap___boxed(lean_object* v_R_1318_, lean_object* v_S_1319_, lean_object* v_inst_1320_, lean_object* v_inst_1321_, lean_object* v_g_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_mathlib_Module_compHom_toLinearMap(v_R_1318_, v_S_1319_, v_inst_1320_, v_inst_1321_, v_g_1322_);
lean_dec_ref(v_inst_1321_);
lean_dec_ref(v_inst_1320_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27___redArg(lean_object* v_f_1324_){
_start:
{
lean_inc(v_f_1324_);
return v_f_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27___redArg___boxed(lean_object* v_f_1325_){
_start:
{
lean_object* v_res_1326_; 
v_res_1326_ = lp_mathlib_IsLinearMap_mk_x27___redArg(v_f_1325_);
lean_dec(v_f_1325_);
return v_res_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27(lean_object* v_R_1327_, lean_object* v_M_1328_, lean_object* v_M_u2082_1329_, lean_object* v_inst_1330_, lean_object* v_inst_1331_, lean_object* v_inst_1332_, lean_object* v_inst_1333_, lean_object* v_inst_1334_, lean_object* v_f_1335_, lean_object* v_lin_1336_){
_start:
{
lean_inc(v_f_1335_);
return v_f_1335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsLinearMap_mk_x27___boxed(lean_object* v_R_1337_, lean_object* v_M_1338_, lean_object* v_M_u2082_1339_, lean_object* v_inst_1340_, lean_object* v_inst_1341_, lean_object* v_inst_1342_, lean_object* v_inst_1343_, lean_object* v_inst_1344_, lean_object* v_f_1345_, lean_object* v_lin_1346_){
_start:
{
lean_object* v_res_1347_; 
v_res_1347_ = lp_mathlib_IsLinearMap_mk_x27(v_R_1337_, v_M_1338_, v_M_u2082_1339_, v_inst_1340_, v_inst_1341_, v_inst_1342_, v_inst_1343_, v_inst_1344_, v_f_1345_, v_lin_1346_);
lean_dec(v_f_1345_);
lean_dec(v_inst_1344_);
lean_dec(v_inst_1343_);
lean_dec_ref(v_inst_1342_);
lean_dec_ref(v_inst_1341_);
lean_dec_ref(v_inst_1340_);
return v_res_1347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap___redArg(lean_object* v_f_1348_){
_start:
{
lean_object* v___f_1349_; 
v___f_1349_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1349_, 0, v_f_1348_);
return v___f_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap(lean_object* v_M_1350_, lean_object* v_M_u2082_1351_, lean_object* v_inst_1352_, lean_object* v_inst_1353_, lean_object* v_f_1354_){
_start:
{
lean_object* v___f_1355_; 
v___f_1355_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1355_, 0, v_f_1354_);
return v___f_1355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toNatLinearMap___boxed(lean_object* v_M_1356_, lean_object* v_M_u2082_1357_, lean_object* v_inst_1358_, lean_object* v_inst_1359_, lean_object* v_f_1360_){
_start:
{
lean_object* v_res_1361_; 
v_res_1361_ = lp_mathlib_AddMonoidHom_toNatLinearMap(v_M_1356_, v_M_u2082_1357_, v_inst_1358_, v_inst_1359_, v_f_1360_);
lean_dec_ref(v_inst_1359_);
lean_dec_ref(v_inst_1358_);
return v_res_1361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap___redArg(lean_object* v_f_1362_){
_start:
{
lean_object* v___f_1363_; 
v___f_1363_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1363_, 0, v_f_1362_);
return v___f_1363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap(lean_object* v_M_1364_, lean_object* v_M_u2082_1365_, lean_object* v_inst_1366_, lean_object* v_inst_1367_, lean_object* v_f_1368_){
_start:
{
lean_object* v___f_1369_; 
v___f_1369_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1369_, 0, v_f_1368_);
return v___f_1369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toIntLinearMap___boxed(lean_object* v_M_1370_, lean_object* v_M_u2082_1371_, lean_object* v_inst_1372_, lean_object* v_inst_1373_, lean_object* v_f_1374_){
_start:
{
lean_object* v_res_1375_; 
v_res_1375_ = lp_mathlib_AddMonoidHom_toIntLinearMap(v_M_1370_, v_M_u2082_1371_, v_inst_1372_, v_inst_1373_, v_f_1374_);
lean_dec_ref(v_inst_1373_);
lean_dec_ref(v_inst_1372_);
return v_res_1375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul___redArg___lam__0(lean_object* v_inst_1376_, lean_object* v_a_1377_, lean_object* v_f_1378_, lean_object* v___y_1379_){
_start:
{
lean_object* v___x_1380_; lean_object* v___x_1381_; 
v___x_1380_ = lean_apply_1(v_f_1378_, v___y_1379_);
v___x_1381_ = lean_apply_2(v_inst_1376_, v_a_1377_, v___x_1380_);
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul___redArg(lean_object* v_inst_1382_){
_start:
{
lean_object* v___f_1383_; 
v___f_1383_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1383_, 0, v_inst_1382_);
return v___f_1383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul(lean_object* v_R_1384_, lean_object* v_R_u2082_1385_, lean_object* v_S_1386_, lean_object* v_M_1387_, lean_object* v_M_u2082_1388_, lean_object* v_inst_1389_, lean_object* v_inst_1390_, lean_object* v_inst_1391_, lean_object* v_inst_1392_, lean_object* v_inst_1393_, lean_object* v_inst_1394_, lean_object* v_00_u03c3_u2081_u2082_1395_, lean_object* v_inst_1396_, lean_object* v_inst_1397_){
_start:
{
lean_object* v___f_1398_; 
v___f_1398_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1398_, 0, v_inst_1396_);
return v___f_1398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSMul___boxed(lean_object* v_R_1399_, lean_object* v_R_u2082_1400_, lean_object* v_S_1401_, lean_object* v_M_1402_, lean_object* v_M_u2082_1403_, lean_object* v_inst_1404_, lean_object* v_inst_1405_, lean_object* v_inst_1406_, lean_object* v_inst_1407_, lean_object* v_inst_1408_, lean_object* v_inst_1409_, lean_object* v_00_u03c3_u2081_u2082_1410_, lean_object* v_inst_1411_, lean_object* v_inst_1412_){
_start:
{
lean_object* v_res_1413_; 
v_res_1413_ = lp_mathlib_LinearMap_instSMul(v_R_1399_, v_R_u2082_1400_, v_S_1401_, v_M_1402_, v_M_u2082_1403_, v_inst_1404_, v_inst_1405_, v_inst_1406_, v_inst_1407_, v_inst_1408_, v_inst_1409_, v_00_u03c3_u2081_u2082_1410_, v_inst_1411_, v_inst_1412_);
lean_dec(v_00_u03c3_u2081_u2082_1410_);
lean_dec(v_inst_1409_);
lean_dec(v_inst_1408_);
lean_dec_ref(v_inst_1407_);
lean_dec_ref(v_inst_1406_);
lean_dec_ref(v_inst_1405_);
lean_dec_ref(v_inst_1404_);
return v_res_1413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg___lam__0(lean_object* v_toZero_1414_, lean_object* v_x_1415_){
_start:
{
lean_inc(v_toZero_1414_);
return v_toZero_1414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg___lam__0___boxed(lean_object* v_toZero_1416_, lean_object* v_x_1417_){
_start:
{
lean_object* v_res_1418_; 
v_res_1418_ = lp_mathlib_LinearMap_instZero___redArg___lam__0(v_toZero_1416_, v_x_1417_);
lean_dec(v_x_1417_);
lean_dec(v_toZero_1416_);
return v_res_1418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg(lean_object* v_inst_1419_){
_start:
{
lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v_toZero_1422_; lean_object* v___f_1423_; 
v___x_1420_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1419_);
v___x_1421_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1420_);
v_toZero_1422_ = lean_ctor_get(v___x_1421_, 0);
lean_inc(v_toZero_1422_);
lean_dec_ref(v___x_1421_);
v___f_1423_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1423_, 0, v_toZero_1422_);
return v___f_1423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___redArg___boxed(lean_object* v_inst_1424_){
_start:
{
lean_object* v_res_1425_; 
v_res_1425_ = lp_mathlib_LinearMap_instZero___redArg(v_inst_1424_);
lean_dec_ref(v_inst_1424_);
return v_res_1425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero(lean_object* v_R_u2081_1426_, lean_object* v_R_u2082_1427_, lean_object* v_M_1428_, lean_object* v_M_u2082_1429_, lean_object* v_inst_1430_, lean_object* v_inst_1431_, lean_object* v_inst_1432_, lean_object* v_inst_1433_, lean_object* v_inst_1434_, lean_object* v_inst_1435_, lean_object* v_00_u03c3_u2081_u2082_1436_){
_start:
{
lean_object* v___x_1437_; 
v___x_1437_ = lp_mathlib_LinearMap_instZero___redArg(v_inst_1433_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instZero___boxed(lean_object* v_R_u2081_1438_, lean_object* v_R_u2082_1439_, lean_object* v_M_1440_, lean_object* v_M_u2082_1441_, lean_object* v_inst_1442_, lean_object* v_inst_1443_, lean_object* v_inst_1444_, lean_object* v_inst_1445_, lean_object* v_inst_1446_, lean_object* v_inst_1447_, lean_object* v_00_u03c3_u2081_u2082_1448_){
_start:
{
lean_object* v_res_1449_; 
v_res_1449_ = lp_mathlib_LinearMap_instZero(v_R_u2081_1438_, v_R_u2082_1439_, v_M_1440_, v_M_u2082_1441_, v_inst_1442_, v_inst_1443_, v_inst_1444_, v_inst_1445_, v_inst_1446_, v_inst_1447_, v_00_u03c3_u2081_u2082_1448_);
lean_dec(v_00_u03c3_u2081_u2082_1448_);
lean_dec(v_inst_1447_);
lean_dec(v_inst_1446_);
lean_dec_ref(v_inst_1445_);
lean_dec_ref(v_inst_1444_);
lean_dec_ref(v_inst_1443_);
lean_dec_ref(v_inst_1442_);
return v_res_1449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited___redArg(lean_object* v_inst_1450_){
_start:
{
lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v_toZero_1453_; lean_object* v___f_1454_; 
v___x_1451_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_1450_);
v___x_1452_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1451_);
v_toZero_1453_ = lean_ctor_get(v___x_1452_, 0);
lean_inc(v_toZero_1453_);
lean_dec_ref(v___x_1452_);
v___f_1454_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instZero___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1454_, 0, v_toZero_1453_);
return v___f_1454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited___redArg___boxed(lean_object* v_inst_1455_){
_start:
{
lean_object* v_res_1456_; 
v_res_1456_ = lp_mathlib_LinearMap_instInhabited___redArg(v_inst_1455_);
lean_dec_ref(v_inst_1455_);
return v_res_1456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited(lean_object* v_R_u2081_1457_, lean_object* v_R_u2082_1458_, lean_object* v_M_1459_, lean_object* v_M_u2082_1460_, lean_object* v_inst_1461_, lean_object* v_inst_1462_, lean_object* v_inst_1463_, lean_object* v_inst_1464_, lean_object* v_inst_1465_, lean_object* v_inst_1466_, lean_object* v_00_u03c3_u2081_u2082_1467_){
_start:
{
lean_object* v___x_1468_; 
v___x_1468_ = lp_mathlib_LinearMap_instInhabited___redArg(v_inst_1464_);
return v___x_1468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instInhabited___boxed(lean_object* v_R_u2081_1469_, lean_object* v_R_u2082_1470_, lean_object* v_M_1471_, lean_object* v_M_u2082_1472_, lean_object* v_inst_1473_, lean_object* v_inst_1474_, lean_object* v_inst_1475_, lean_object* v_inst_1476_, lean_object* v_inst_1477_, lean_object* v_inst_1478_, lean_object* v_00_u03c3_u2081_u2082_1479_){
_start:
{
lean_object* v_res_1480_; 
v_res_1480_ = lp_mathlib_LinearMap_instInhabited(v_R_u2081_1469_, v_R_u2082_1470_, v_M_1471_, v_M_u2082_1472_, v_inst_1473_, v_inst_1474_, v_inst_1475_, v_inst_1476_, v_inst_1477_, v_inst_1478_, v_00_u03c3_u2081_u2082_1479_);
lean_dec(v_00_u03c3_u2081_u2082_1479_);
lean_dec(v_inst_1478_);
lean_dec(v_inst_1477_);
lean_dec_ref(v_inst_1476_);
lean_dec_ref(v_inst_1475_);
lean_dec_ref(v_inst_1474_);
lean_dec_ref(v_inst_1473_);
return v_res_1480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft___redArg(lean_object* v_inst_1481_){
_start:
{
lean_object* v___x_1482_; 
v___x_1482_ = lp_mathlib_LinearMap_instInhabited___redArg(v_inst_1481_);
return v___x_1482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft___redArg___boxed(lean_object* v_inst_1483_){
_start:
{
lean_object* v_res_1484_; 
v_res_1484_ = lp_mathlib_LinearMap_uniqueOfLeft___redArg(v_inst_1483_);
lean_dec_ref(v_inst_1483_);
return v_res_1484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft(lean_object* v_R_u2081_1485_, lean_object* v_R_u2082_1486_, lean_object* v_M_1487_, lean_object* v_M_u2082_1488_, lean_object* v_inst_1489_, lean_object* v_inst_1490_, lean_object* v_inst_1491_, lean_object* v_inst_1492_, lean_object* v_inst_1493_, lean_object* v_inst_1494_, lean_object* v_00_u03c3_u2081_u2082_1495_, lean_object* v_inst_1496_){
_start:
{
lean_object* v___x_1497_; 
v___x_1497_ = lp_mathlib_LinearMap_instInhabited___redArg(v_inst_1492_);
return v___x_1497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfLeft___boxed(lean_object* v_R_u2081_1498_, lean_object* v_R_u2082_1499_, lean_object* v_M_1500_, lean_object* v_M_u2082_1501_, lean_object* v_inst_1502_, lean_object* v_inst_1503_, lean_object* v_inst_1504_, lean_object* v_inst_1505_, lean_object* v_inst_1506_, lean_object* v_inst_1507_, lean_object* v_00_u03c3_u2081_u2082_1508_, lean_object* v_inst_1509_){
_start:
{
lean_object* v_res_1510_; 
v_res_1510_ = lp_mathlib_LinearMap_uniqueOfLeft(v_R_u2081_1498_, v_R_u2082_1499_, v_M_1500_, v_M_u2082_1501_, v_inst_1502_, v_inst_1503_, v_inst_1504_, v_inst_1505_, v_inst_1506_, v_inst_1507_, v_00_u03c3_u2081_u2082_1508_, v_inst_1509_);
lean_dec(v_00_u03c3_u2081_u2082_1508_);
lean_dec(v_inst_1507_);
lean_dec(v_inst_1506_);
lean_dec_ref(v_inst_1505_);
lean_dec_ref(v_inst_1504_);
lean_dec_ref(v_inst_1503_);
lean_dec_ref(v_inst_1502_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight___redArg(lean_object* v_inst_1511_){
_start:
{
lean_object* v___x_1512_; 
v___x_1512_ = lp_mathlib_LinearMap_instInhabited___redArg(v_inst_1511_);
return v___x_1512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight___redArg___boxed(lean_object* v_inst_1513_){
_start:
{
lean_object* v_res_1514_; 
v_res_1514_ = lp_mathlib_LinearMap_uniqueOfRight___redArg(v_inst_1513_);
lean_dec_ref(v_inst_1513_);
return v_res_1514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight(lean_object* v_R_u2081_1515_, lean_object* v_R_u2082_1516_, lean_object* v_M_1517_, lean_object* v_M_u2082_1518_, lean_object* v_inst_1519_, lean_object* v_inst_1520_, lean_object* v_inst_1521_, lean_object* v_inst_1522_, lean_object* v_inst_1523_, lean_object* v_inst_1524_, lean_object* v_00_u03c3_u2081_u2082_1525_, lean_object* v_inst_1526_){
_start:
{
lean_object* v___x_1527_; 
v___x_1527_ = lp_mathlib_LinearMap_instInhabited___redArg(v_inst_1522_);
return v___x_1527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_uniqueOfRight___boxed(lean_object* v_R_u2081_1528_, lean_object* v_R_u2082_1529_, lean_object* v_M_1530_, lean_object* v_M_u2082_1531_, lean_object* v_inst_1532_, lean_object* v_inst_1533_, lean_object* v_inst_1534_, lean_object* v_inst_1535_, lean_object* v_inst_1536_, lean_object* v_inst_1537_, lean_object* v_00_u03c3_u2081_u2082_1538_, lean_object* v_inst_1539_){
_start:
{
lean_object* v_res_1540_; 
v_res_1540_ = lp_mathlib_LinearMap_uniqueOfRight(v_R_u2081_1528_, v_R_u2082_1529_, v_M_1530_, v_M_u2082_1531_, v_inst_1532_, v_inst_1533_, v_inst_1534_, v_inst_1535_, v_inst_1536_, v_inst_1537_, v_00_u03c3_u2081_u2082_1538_, v_inst_1539_);
lean_dec(v_00_u03c3_u2081_u2082_1538_);
lean_dec(v_inst_1537_);
lean_dec(v_inst_1536_);
lean_dec_ref(v_inst_1535_);
lean_dec_ref(v_inst_1534_);
lean_dec_ref(v_inst_1533_);
lean_dec_ref(v_inst_1532_);
return v_res_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd___redArg___lam__0(lean_object* v_toAdd_1541_, lean_object* v_f_1542_, lean_object* v_g_1543_, lean_object* v___y_1544_){
_start:
{
lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; 
lean_inc(v___y_1544_);
v___x_1545_ = lean_apply_1(v_f_1542_, v___y_1544_);
v___x_1546_ = lean_apply_1(v_g_1543_, v___y_1544_);
v___x_1547_ = lean_apply_2(v_toAdd_1541_, v___x_1545_, v___x_1546_);
return v___x_1547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd___redArg(lean_object* v_inst_1548_){
_start:
{
lean_object* v_toAdd_1549_; lean_object* v___f_1550_; 
v_toAdd_1549_ = lean_ctor_get(v_inst_1548_, 1);
lean_inc(v_toAdd_1549_);
lean_dec_ref(v_inst_1548_);
v___f_1550_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instAdd___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1550_, 0, v_toAdd_1549_);
return v___f_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd(lean_object* v_R_u2081_1551_, lean_object* v_R_u2082_1552_, lean_object* v_M_1553_, lean_object* v_M_u2082_1554_, lean_object* v_inst_1555_, lean_object* v_inst_1556_, lean_object* v_inst_1557_, lean_object* v_inst_1558_, lean_object* v_inst_1559_, lean_object* v_inst_1560_, lean_object* v_00_u03c3_u2081_u2082_1561_){
_start:
{
lean_object* v___x_1562_; 
v___x_1562_ = lp_mathlib_LinearMap_instAdd___redArg(v_inst_1558_);
return v___x_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instAdd___boxed(lean_object* v_R_u2081_1563_, lean_object* v_R_u2082_1564_, lean_object* v_M_1565_, lean_object* v_M_u2082_1566_, lean_object* v_inst_1567_, lean_object* v_inst_1568_, lean_object* v_inst_1569_, lean_object* v_inst_1570_, lean_object* v_inst_1571_, lean_object* v_inst_1572_, lean_object* v_00_u03c3_u2081_u2082_1573_){
_start:
{
lean_object* v_res_1574_; 
v_res_1574_ = lp_mathlib_LinearMap_instAdd(v_R_u2081_1563_, v_R_u2082_1564_, v_M_1565_, v_M_u2082_1566_, v_inst_1567_, v_inst_1568_, v_inst_1569_, v_inst_1570_, v_inst_1571_, v_inst_1572_, v_00_u03c3_u2081_u2082_1573_);
lean_dec(v_00_u03c3_u2081_u2082_1573_);
lean_dec(v_inst_1572_);
lean_dec(v_inst_1571_);
lean_dec_ref(v_inst_1569_);
lean_dec_ref(v_inst_1568_);
lean_dec_ref(v_inst_1567_);
return v_res_1574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addMonoid___redArg(lean_object* v_inst_1575_){
_start:
{
lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___f_1579_; lean_object* v___f_1580_; lean_object* v___x_1581_; 
v___x_1576_ = lp_mathlib_LinearMap_instZero___redArg(v_inst_1575_);
lean_inc_ref(v_inst_1575_);
v___x_1577_ = lp_mathlib_LinearMap_instAdd___redArg(v_inst_1575_);
v___x_1578_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v_inst_1575_);
v___f_1579_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1579_, 0, v___x_1578_);
v___f_1580_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1580_, 0, v___f_1579_);
v___x_1581_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1581_, 0, v___x_1576_);
lean_ctor_set(v___x_1581_, 1, v___x_1577_);
lean_ctor_set(v___x_1581_, 2, v___f_1580_);
return v___x_1581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addMonoid(lean_object* v_R_u2081_1582_, lean_object* v_R_u2082_1583_, lean_object* v_M_1584_, lean_object* v_M_u2082_1585_, lean_object* v_inst_1586_, lean_object* v_inst_1587_, lean_object* v_inst_1588_, lean_object* v_inst_1589_, lean_object* v_inst_1590_, lean_object* v_inst_1591_, lean_object* v_00_u03c3_u2081_u2082_1592_){
_start:
{
lean_object* v___x_1593_; 
v___x_1593_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_1589_);
return v___x_1593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addMonoid___boxed(lean_object* v_R_u2081_1594_, lean_object* v_R_u2082_1595_, lean_object* v_M_1596_, lean_object* v_M_u2082_1597_, lean_object* v_inst_1598_, lean_object* v_inst_1599_, lean_object* v_inst_1600_, lean_object* v_inst_1601_, lean_object* v_inst_1602_, lean_object* v_inst_1603_, lean_object* v_00_u03c3_u2081_u2082_1604_){
_start:
{
lean_object* v_res_1605_; 
v_res_1605_ = lp_mathlib_LinearMap_addMonoid(v_R_u2081_1594_, v_R_u2082_1595_, v_M_1596_, v_M_u2082_1597_, v_inst_1598_, v_inst_1599_, v_inst_1600_, v_inst_1601_, v_inst_1602_, v_inst_1603_, v_00_u03c3_u2081_u2082_1604_);
lean_dec(v_00_u03c3_u2081_u2082_1604_);
lean_dec(v_inst_1603_);
lean_dec(v_inst_1602_);
lean_dec_ref(v_inst_1600_);
lean_dec_ref(v_inst_1599_);
lean_dec_ref(v_inst_1598_);
return v_res_1605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommMonoid___redArg(lean_object* v_inst_1606_){
_start:
{
lean_object* v___x_1607_; 
v___x_1607_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_1606_);
return v___x_1607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommMonoid(lean_object* v_R_u2081_1608_, lean_object* v_R_u2082_1609_, lean_object* v_M_1610_, lean_object* v_M_u2082_1611_, lean_object* v_inst_1612_, lean_object* v_inst_1613_, lean_object* v_inst_1614_, lean_object* v_inst_1615_, lean_object* v_inst_1616_, lean_object* v_inst_1617_, lean_object* v_00_u03c3_u2081_u2082_1618_){
_start:
{
lean_object* v___x_1619_; 
v___x_1619_ = lp_mathlib_LinearMap_addMonoid___redArg(v_inst_1615_);
return v___x_1619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommMonoid___boxed(lean_object* v_R_u2081_1620_, lean_object* v_R_u2082_1621_, lean_object* v_M_1622_, lean_object* v_M_u2082_1623_, lean_object* v_inst_1624_, lean_object* v_inst_1625_, lean_object* v_inst_1626_, lean_object* v_inst_1627_, lean_object* v_inst_1628_, lean_object* v_inst_1629_, lean_object* v_00_u03c3_u2081_u2082_1630_){
_start:
{
lean_object* v_res_1631_; 
v_res_1631_ = lp_mathlib_LinearMap_addCommMonoid(v_R_u2081_1620_, v_R_u2082_1621_, v_M_1622_, v_M_u2082_1623_, v_inst_1624_, v_inst_1625_, v_inst_1626_, v_inst_1627_, v_inst_1628_, v_inst_1629_, v_00_u03c3_u2081_u2082_1630_);
lean_dec(v_00_u03c3_u2081_u2082_1630_);
lean_dec(v_inst_1629_);
lean_dec(v_inst_1628_);
lean_dec_ref(v_inst_1626_);
lean_dec_ref(v_inst_1625_);
lean_dec_ref(v_inst_1624_);
return v_res_1631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___redArg___lam__0(lean_object* v_toNeg_1632_, lean_object* v_f_1633_, lean_object* v___y_1634_){
_start:
{
lean_object* v___x_1635_; lean_object* v___x_1636_; 
v___x_1635_ = lean_apply_1(v_f_1633_, v___y_1634_);
v___x_1636_ = lean_apply_1(v_toNeg_1632_, v___x_1635_);
return v___x_1636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___redArg(lean_object* v_inst_1637_){
_start:
{
lean_object* v___x_1638_; lean_object* v_toNeg_1639_; lean_object* v___f_1640_; 
v___x_1638_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_1637_);
v_toNeg_1639_ = lean_ctor_get(v___x_1638_, 1);
lean_inc(v_toNeg_1639_);
lean_dec_ref(v___x_1638_);
v___f_1640_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instNeg___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1640_, 0, v_toNeg_1639_);
return v___f_1640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___redArg___boxed(lean_object* v_inst_1641_){
_start:
{
lean_object* v_res_1642_; 
v_res_1642_ = lp_mathlib_LinearMap_instNeg___redArg(v_inst_1641_);
lean_dec_ref(v_inst_1641_);
return v_res_1642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg(lean_object* v_R_u2081_1643_, lean_object* v_R_u2082_1644_, lean_object* v_M_1645_, lean_object* v_N_u2082_1646_, lean_object* v_inst_1647_, lean_object* v_inst_1648_, lean_object* v_inst_1649_, lean_object* v_inst_1650_, lean_object* v_inst_1651_, lean_object* v_inst_1652_, lean_object* v_00_u03c3_u2081_u2082_1653_){
_start:
{
lean_object* v___x_1654_; 
v___x_1654_ = lp_mathlib_LinearMap_instNeg___redArg(v_inst_1650_);
return v___x_1654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instNeg___boxed(lean_object* v_R_u2081_1655_, lean_object* v_R_u2082_1656_, lean_object* v_M_1657_, lean_object* v_N_u2082_1658_, lean_object* v_inst_1659_, lean_object* v_inst_1660_, lean_object* v_inst_1661_, lean_object* v_inst_1662_, lean_object* v_inst_1663_, lean_object* v_inst_1664_, lean_object* v_00_u03c3_u2081_u2082_1665_){
_start:
{
lean_object* v_res_1666_; 
v_res_1666_ = lp_mathlib_LinearMap_instNeg(v_R_u2081_1655_, v_R_u2082_1656_, v_M_1657_, v_N_u2082_1658_, v_inst_1659_, v_inst_1660_, v_inst_1661_, v_inst_1662_, v_inst_1663_, v_inst_1664_, v_00_u03c3_u2081_u2082_1665_);
lean_dec(v_00_u03c3_u2081_u2082_1665_);
lean_dec(v_inst_1664_);
lean_dec(v_inst_1663_);
lean_dec_ref(v_inst_1662_);
lean_dec_ref(v_inst_1661_);
lean_dec_ref(v_inst_1660_);
lean_dec_ref(v_inst_1659_);
return v_res_1666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub___redArg___lam__0(lean_object* v_toSub_1667_, lean_object* v_f_1668_, lean_object* v_g_1669_, lean_object* v___y_1670_){
_start:
{
lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; 
lean_inc(v___y_1670_);
v___x_1671_ = lean_apply_1(v_f_1668_, v___y_1670_);
v___x_1672_ = lean_apply_1(v_g_1669_, v___y_1670_);
v___x_1673_ = lean_apply_2(v_toSub_1667_, v___x_1671_, v___x_1672_);
return v___x_1673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub___redArg(lean_object* v_inst_1674_){
_start:
{
lean_object* v_toSub_1675_; lean_object* v___f_1676_; 
v_toSub_1675_ = lean_ctor_get(v_inst_1674_, 2);
lean_inc(v_toSub_1675_);
lean_dec_ref(v_inst_1674_);
v___f_1676_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSub___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1676_, 0, v_toSub_1675_);
return v___f_1676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub(lean_object* v_R_u2081_1677_, lean_object* v_R_u2082_1678_, lean_object* v_M_1679_, lean_object* v_N_u2082_1680_, lean_object* v_inst_1681_, lean_object* v_inst_1682_, lean_object* v_inst_1683_, lean_object* v_inst_1684_, lean_object* v_inst_1685_, lean_object* v_inst_1686_, lean_object* v_00_u03c3_u2081_u2082_1687_){
_start:
{
lean_object* v___x_1688_; 
v___x_1688_ = lp_mathlib_LinearMap_instSub___redArg(v_inst_1684_);
return v___x_1688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instSub___boxed(lean_object* v_R_u2081_1689_, lean_object* v_R_u2082_1690_, lean_object* v_M_1691_, lean_object* v_N_u2082_1692_, lean_object* v_inst_1693_, lean_object* v_inst_1694_, lean_object* v_inst_1695_, lean_object* v_inst_1696_, lean_object* v_inst_1697_, lean_object* v_inst_1698_, lean_object* v_00_u03c3_u2081_u2082_1699_){
_start:
{
lean_object* v_res_1700_; 
v_res_1700_ = lp_mathlib_LinearMap_instSub(v_R_u2081_1689_, v_R_u2082_1690_, v_M_1691_, v_N_u2082_1692_, v_inst_1693_, v_inst_1694_, v_inst_1695_, v_inst_1696_, v_inst_1697_, v_inst_1698_, v_00_u03c3_u2081_u2082_1699_);
lean_dec(v_00_u03c3_u2081_u2082_1699_);
lean_dec(v_inst_1698_);
lean_dec(v_inst_1697_);
lean_dec_ref(v_inst_1695_);
lean_dec_ref(v_inst_1694_);
lean_dec_ref(v_inst_1693_);
return v_res_1700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommGroup___redArg(lean_object* v_inst_1701_){
_start:
{
lean_object* v_toAddMonoid_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___f_1707_; lean_object* v___f_1708_; lean_object* v___x_1709_; 
v_toAddMonoid_1702_ = lean_ctor_get(v_inst_1701_, 0);
lean_inc_ref(v_toAddMonoid_1702_);
v___x_1703_ = lp_mathlib_LinearMap_addMonoid___redArg(v_toAddMonoid_1702_);
v___x_1704_ = lp_mathlib_LinearMap_instNeg___redArg(v_inst_1701_);
lean_inc_ref(v_inst_1701_);
v___x_1705_ = lp_mathlib_LinearMap_instSub___redArg(v_inst_1701_);
v___x_1706_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_1701_);
v___f_1707_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1707_, 0, v___x_1706_);
v___f_1708_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1708_, 0, v___f_1707_);
v___x_1709_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1709_, 0, v___x_1703_);
lean_ctor_set(v___x_1709_, 1, v___x_1704_);
lean_ctor_set(v___x_1709_, 2, v___x_1705_);
lean_ctor_set(v___x_1709_, 3, v___f_1708_);
return v___x_1709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommGroup(lean_object* v_R_u2081_1710_, lean_object* v_R_u2082_1711_, lean_object* v_M_1712_, lean_object* v_N_u2082_1713_, lean_object* v_inst_1714_, lean_object* v_inst_1715_, lean_object* v_inst_1716_, lean_object* v_inst_1717_, lean_object* v_inst_1718_, lean_object* v_inst_1719_, lean_object* v_00_u03c3_u2081_u2082_1720_){
_start:
{
lean_object* v___x_1721_; 
v___x_1721_ = lp_mathlib_LinearMap_addCommGroup___redArg(v_inst_1717_);
return v___x_1721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_addCommGroup___boxed(lean_object* v_R_u2081_1722_, lean_object* v_R_u2082_1723_, lean_object* v_M_1724_, lean_object* v_N_u2082_1725_, lean_object* v_inst_1726_, lean_object* v_inst_1727_, lean_object* v_inst_1728_, lean_object* v_inst_1729_, lean_object* v_inst_1730_, lean_object* v_inst_1731_, lean_object* v_00_u03c3_u2081_u2082_1732_){
_start:
{
lean_object* v_res_1733_; 
v_res_1733_ = lp_mathlib_LinearMap_addCommGroup(v_R_u2081_1722_, v_R_u2082_1723_, v_M_1724_, v_N_u2082_1725_, v_inst_1726_, v_inst_1727_, v_inst_1728_, v_inst_1729_, v_inst_1730_, v_inst_1731_, v_00_u03c3_u2081_u2082_1732_);
lean_dec(v_00_u03c3_u2081_u2082_1732_);
lean_dec(v_inst_1731_);
lean_dec(v_inst_1730_);
lean_dec_ref(v_inst_1728_);
lean_dec_ref(v_inst_1727_);
lean_dec_ref(v_inst_1726_);
return v_res_1733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom___redArg___lam__0(lean_object* v_a_1734_, lean_object* v_f_1735_){
_start:
{
lean_object* v___x_1736_; 
v___x_1736_ = lean_apply_1(v_f_1735_, v_a_1734_);
return v___x_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom___redArg(lean_object* v_a_1737_){
_start:
{
lean_object* v___f_1738_; 
v___f_1738_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_evalAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1738_, 0, v_a_1737_);
return v___f_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom(lean_object* v_R_u2081_1739_, lean_object* v_R_u2082_1740_, lean_object* v_M_1741_, lean_object* v_M_u2082_1742_, lean_object* v_inst_1743_, lean_object* v_inst_1744_, lean_object* v_inst_1745_, lean_object* v_inst_1746_, lean_object* v_inst_1747_, lean_object* v_inst_1748_, lean_object* v_00_u03c3_u2081_u2082_1749_, lean_object* v_a_1750_){
_start:
{
lean_object* v___f_1751_; 
v___f_1751_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_evalAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1751_, 0, v_a_1750_);
return v___f_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_evalAddMonoidHom___boxed(lean_object* v_R_u2081_1752_, lean_object* v_R_u2082_1753_, lean_object* v_M_1754_, lean_object* v_M_u2082_1755_, lean_object* v_inst_1756_, lean_object* v_inst_1757_, lean_object* v_inst_1758_, lean_object* v_inst_1759_, lean_object* v_inst_1760_, lean_object* v_inst_1761_, lean_object* v_00_u03c3_u2081_u2082_1762_, lean_object* v_a_1763_){
_start:
{
lean_object* v_res_1764_; 
v_res_1764_ = lp_mathlib_LinearMap_evalAddMonoidHom(v_R_u2081_1752_, v_R_u2082_1753_, v_M_1754_, v_M_u2082_1755_, v_inst_1756_, v_inst_1757_, v_inst_1758_, v_inst_1759_, v_inst_1760_, v_inst_1761_, v_00_u03c3_u2081_u2082_1762_, v_a_1763_);
lean_dec(v_00_u03c3_u2081_u2082_1762_);
lean_dec(v_inst_1761_);
lean_dec(v_inst_1760_);
lean_dec_ref(v_inst_1759_);
lean_dec_ref(v_inst_1758_);
lean_dec_ref(v_inst_1757_);
lean_dec_ref(v_inst_1756_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom_x27___redArg(lean_object* v_inst_1765_, lean_object* v_inst_1766_, lean_object* v_inst_1767_, lean_object* v_inst_1768_, lean_object* v_inst_1769_, lean_object* v_inst_1770_, lean_object* v_00_u03c3_u2081_u2082_1771_){
_start:
{
lean_object* v___x_1772_; 
v___x_1772_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___boxed), 12, 11);
lean_closure_set(v___x_1772_, 0, lean_box(0));
lean_closure_set(v___x_1772_, 1, lean_box(0));
lean_closure_set(v___x_1772_, 2, lean_box(0));
lean_closure_set(v___x_1772_, 3, lean_box(0));
lean_closure_set(v___x_1772_, 4, v_inst_1765_);
lean_closure_set(v___x_1772_, 5, v_inst_1766_);
lean_closure_set(v___x_1772_, 6, v_inst_1767_);
lean_closure_set(v___x_1772_, 7, v_inst_1768_);
lean_closure_set(v___x_1772_, 8, v_inst_1769_);
lean_closure_set(v___x_1772_, 9, v_inst_1770_);
lean_closure_set(v___x_1772_, 10, v_00_u03c3_u2081_u2082_1771_);
return v___x_1772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toAddMonoidHom_x27(lean_object* v_R_u2081_1773_, lean_object* v_R_u2082_1774_, lean_object* v_M_1775_, lean_object* v_M_u2082_1776_, lean_object* v_inst_1777_, lean_object* v_inst_1778_, lean_object* v_inst_1779_, lean_object* v_inst_1780_, lean_object* v_inst_1781_, lean_object* v_inst_1782_, lean_object* v_00_u03c3_u2081_u2082_1783_){
_start:
{
lean_object* v___x_1784_; 
v___x_1784_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___boxed), 12, 11);
lean_closure_set(v___x_1784_, 0, lean_box(0));
lean_closure_set(v___x_1784_, 1, lean_box(0));
lean_closure_set(v___x_1784_, 2, lean_box(0));
lean_closure_set(v___x_1784_, 3, lean_box(0));
lean_closure_set(v___x_1784_, 4, v_inst_1777_);
lean_closure_set(v___x_1784_, 5, v_inst_1778_);
lean_closure_set(v___x_1784_, 6, v_inst_1779_);
lean_closure_set(v___x_1784_, 7, v_inst_1780_);
lean_closure_set(v___x_1784_, 8, v_inst_1781_);
lean_closure_set(v___x_1784_, 9, v_inst_1782_);
lean_closure_set(v___x_1784_, 10, v_00_u03c3_u2081_u2082_1783_);
return v___x_1784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulAction___redArg(lean_object* v_inst_1785_){
_start:
{
lean_object* v___f_1786_; 
v___f_1786_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1786_, 0, v_inst_1785_);
return v___f_1786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulAction(lean_object* v_R_1787_, lean_object* v_R_u2082_1788_, lean_object* v_S_1789_, lean_object* v_M_1790_, lean_object* v_M_u2082_1791_, lean_object* v_inst_1792_, lean_object* v_inst_1793_, lean_object* v_inst_1794_, lean_object* v_inst_1795_, lean_object* v_inst_1796_, lean_object* v_inst_1797_, lean_object* v_00_u03c3_u2081_u2082_1798_, lean_object* v_inst_1799_, lean_object* v_inst_1800_, lean_object* v_inst_1801_){
_start:
{
lean_object* v___f_1802_; 
v___f_1802_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1802_, 0, v_inst_1800_);
return v___f_1802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_instDistribMulAction___boxed(lean_object* v_R_1803_, lean_object* v_R_u2082_1804_, lean_object* v_S_1805_, lean_object* v_M_1806_, lean_object* v_M_u2082_1807_, lean_object* v_inst_1808_, lean_object* v_inst_1809_, lean_object* v_inst_1810_, lean_object* v_inst_1811_, lean_object* v_inst_1812_, lean_object* v_inst_1813_, lean_object* v_00_u03c3_u2081_u2082_1814_, lean_object* v_inst_1815_, lean_object* v_inst_1816_, lean_object* v_inst_1817_){
_start:
{
lean_object* v_res_1818_; 
v_res_1818_ = lp_mathlib_LinearMap_instDistribMulAction(v_R_1803_, v_R_u2082_1804_, v_S_1805_, v_M_1806_, v_M_u2082_1807_, v_inst_1808_, v_inst_1809_, v_inst_1810_, v_inst_1811_, v_inst_1812_, v_inst_1813_, v_00_u03c3_u2081_u2082_1814_, v_inst_1815_, v_inst_1816_, v_inst_1817_);
lean_dec_ref(v_inst_1815_);
lean_dec(v_00_u03c3_u2081_u2082_1814_);
lean_dec(v_inst_1813_);
lean_dec(v_inst_1812_);
lean_dec_ref(v_inst_1811_);
lean_dec_ref(v_inst_1810_);
lean_dec_ref(v_inst_1809_);
lean_dec_ref(v_inst_1808_);
return v_res_1818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_module___redArg(lean_object* v_inst_1819_){
_start:
{
lean_object* v___f_1820_; 
v___f_1820_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1820_, 0, v_inst_1819_);
return v___f_1820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_module(lean_object* v_R_1821_, lean_object* v_R_u2082_1822_, lean_object* v_S_1823_, lean_object* v_M_1824_, lean_object* v_M_u2082_1825_, lean_object* v_inst_1826_, lean_object* v_inst_1827_, lean_object* v_inst_1828_, lean_object* v_inst_1829_, lean_object* v_inst_1830_, lean_object* v_inst_1831_, lean_object* v_00_u03c3_u2081_u2082_1832_, lean_object* v_inst_1833_, lean_object* v_inst_1834_, lean_object* v_inst_1835_){
_start:
{
lean_object* v___f_1836_; 
v___f_1836_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1836_, 0, v_inst_1834_);
return v___f_1836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_module___boxed(lean_object* v_R_1837_, lean_object* v_R_u2082_1838_, lean_object* v_S_1839_, lean_object* v_M_1840_, lean_object* v_M_u2082_1841_, lean_object* v_inst_1842_, lean_object* v_inst_1843_, lean_object* v_inst_1844_, lean_object* v_inst_1845_, lean_object* v_inst_1846_, lean_object* v_inst_1847_, lean_object* v_00_u03c3_u2081_u2082_1848_, lean_object* v_inst_1849_, lean_object* v_inst_1850_, lean_object* v_inst_1851_){
_start:
{
lean_object* v_res_1852_; 
v_res_1852_ = lp_mathlib_LinearMap_module(v_R_1837_, v_R_u2082_1838_, v_S_1839_, v_M_1840_, v_M_u2082_1841_, v_inst_1842_, v_inst_1843_, v_inst_1844_, v_inst_1845_, v_inst_1846_, v_inst_1847_, v_00_u03c3_u2081_u2082_1848_, v_inst_1849_, v_inst_1850_, v_inst_1851_);
lean_dec_ref(v_inst_1849_);
lean_dec(v_00_u03c3_u2081_u2082_1848_);
lean_dec(v_inst_1847_);
lean_dec(v_inst_1846_);
lean_dec_ref(v_inst_1845_);
lean_dec_ref(v_inst_1844_);
lean_dec_ref(v_inst_1843_);
lean_dec_ref(v_inst_1842_);
return v_res_1852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars_u2097___redArg(lean_object* v_inst_1853_, lean_object* v_inst_1854_, lean_object* v_inst_1855_, lean_object* v_inst_1856_, lean_object* v_inst_1857_, lean_object* v_inst_1858_, lean_object* v_inst_1859_, lean_object* v_inst_1860_){
_start:
{
lean_object* v___x_1861_; 
v___x_1861_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___boxed), 14, 13);
lean_closure_set(v___x_1861_, 0, lean_box(0));
lean_closure_set(v___x_1861_, 1, lean_box(0));
lean_closure_set(v___x_1861_, 2, lean_box(0));
lean_closure_set(v___x_1861_, 3, lean_box(0));
lean_closure_set(v___x_1861_, 4, v_inst_1853_);
lean_closure_set(v___x_1861_, 5, v_inst_1854_);
lean_closure_set(v___x_1861_, 6, v_inst_1855_);
lean_closure_set(v___x_1861_, 7, v_inst_1856_);
lean_closure_set(v___x_1861_, 8, v_inst_1857_);
lean_closure_set(v___x_1861_, 9, v_inst_1858_);
lean_closure_set(v___x_1861_, 10, v_inst_1859_);
lean_closure_set(v___x_1861_, 11, v_inst_1860_);
lean_closure_set(v___x_1861_, 12, lean_box(0));
return v___x_1861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars_u2097(lean_object* v_R_1862_, lean_object* v_S_1863_, lean_object* v_M_1864_, lean_object* v_N_1865_, lean_object* v_inst_1866_, lean_object* v_inst_1867_, lean_object* v_inst_1868_, lean_object* v_inst_1869_, lean_object* v_inst_1870_, lean_object* v_inst_1871_, lean_object* v_inst_1872_, lean_object* v_inst_1873_, lean_object* v_inst_1874_, lean_object* v_R_u2081_1875_, lean_object* v_inst_1876_, lean_object* v_inst_1877_, lean_object* v_inst_1878_, lean_object* v_inst_1879_){
_start:
{
lean_object* v___x_1880_; 
v___x_1880_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_restrictScalars___boxed), 14, 13);
lean_closure_set(v___x_1880_, 0, lean_box(0));
lean_closure_set(v___x_1880_, 1, lean_box(0));
lean_closure_set(v___x_1880_, 2, lean_box(0));
lean_closure_set(v___x_1880_, 3, lean_box(0));
lean_closure_set(v___x_1880_, 4, v_inst_1866_);
lean_closure_set(v___x_1880_, 5, v_inst_1867_);
lean_closure_set(v___x_1880_, 6, v_inst_1868_);
lean_closure_set(v___x_1880_, 7, v_inst_1869_);
lean_closure_set(v___x_1880_, 8, v_inst_1870_);
lean_closure_set(v___x_1880_, 9, v_inst_1871_);
lean_closure_set(v___x_1880_, 10, v_inst_1872_);
lean_closure_set(v___x_1880_, 11, v_inst_1873_);
lean_closure_set(v___x_1880_, 12, lean_box(0));
return v___x_1880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_restrictScalars_u2097___boxed(lean_object** _args){
lean_object* v_R_1881_ = _args[0];
lean_object* v_S_1882_ = _args[1];
lean_object* v_M_1883_ = _args[2];
lean_object* v_N_1884_ = _args[3];
lean_object* v_inst_1885_ = _args[4];
lean_object* v_inst_1886_ = _args[5];
lean_object* v_inst_1887_ = _args[6];
lean_object* v_inst_1888_ = _args[7];
lean_object* v_inst_1889_ = _args[8];
lean_object* v_inst_1890_ = _args[9];
lean_object* v_inst_1891_ = _args[10];
lean_object* v_inst_1892_ = _args[11];
lean_object* v_inst_1893_ = _args[12];
lean_object* v_R_u2081_1894_ = _args[13];
lean_object* v_inst_1895_ = _args[14];
lean_object* v_inst_1896_ = _args[15];
lean_object* v_inst_1897_ = _args[16];
lean_object* v_inst_1898_ = _args[17];
_start:
{
lean_object* v_res_1899_; 
v_res_1899_ = lp_mathlib_LinearMap_restrictScalars_u2097(v_R_1881_, v_S_1882_, v_M_1883_, v_N_1884_, v_inst_1885_, v_inst_1886_, v_inst_1887_, v_inst_1888_, v_inst_1889_, v_inst_1890_, v_inst_1891_, v_inst_1892_, v_inst_1893_, v_R_u2081_1894_, v_inst_1895_, v_inst_1896_, v_inst_1897_, v_inst_1898_);
lean_dec(v_inst_1896_);
lean_dec_ref(v_inst_1895_);
return v_res_1899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeft___redArg(lean_object* v_inst_1900_, lean_object* v_a_1901_){
_start:
{
lean_object* v___x_1902_; 
v___x_1902_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v_inst_1900_, v_a_1901_);
return v___x_1902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeft(lean_object* v_R_1903_, lean_object* v_A_1904_, lean_object* v_inst_1905_, lean_object* v_inst_1906_, lean_object* v_inst_1907_, lean_object* v_inst_1908_, lean_object* v_a_1909_){
_start:
{
lean_object* v___x_1910_; 
v___x_1910_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v_inst_1906_, v_a_1909_);
return v___x_1910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeft___boxed(lean_object* v_R_1911_, lean_object* v_A_1912_, lean_object* v_inst_1913_, lean_object* v_inst_1914_, lean_object* v_inst_1915_, lean_object* v_inst_1916_, lean_object* v_a_1917_){
_start:
{
lean_object* v_res_1918_; 
v_res_1918_ = lp_mathlib_LinearMap_mulLeft(v_R_1911_, v_A_1912_, v_inst_1913_, v_inst_1914_, v_inst_1915_, v_inst_1916_, v_a_1917_);
lean_dec(v_inst_1915_);
lean_dec_ref(v_inst_1913_);
return v_res_1918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulRight___redArg(lean_object* v_inst_1919_, lean_object* v_b_1920_){
_start:
{
lean_object* v___x_1921_; 
v___x_1921_ = lp_mathlib_AddMonoidHom_mulRight___redArg(v_inst_1919_, v_b_1920_);
return v___x_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulRight(lean_object* v_R_1922_, lean_object* v_A_1923_, lean_object* v_inst_1924_, lean_object* v_inst_1925_, lean_object* v_inst_1926_, lean_object* v_inst_1927_, lean_object* v_b_1928_){
_start:
{
lean_object* v___x_1929_; 
v___x_1929_ = lp_mathlib_AddMonoidHom_mulRight___redArg(v_inst_1925_, v_b_1928_);
return v___x_1929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulRight___boxed(lean_object* v_R_1930_, lean_object* v_A_1931_, lean_object* v_inst_1932_, lean_object* v_inst_1933_, lean_object* v_inst_1934_, lean_object* v_inst_1935_, lean_object* v_b_1936_){
_start:
{
lean_object* v_res_1937_; 
v_res_1937_ = lp_mathlib_LinearMap_mulRight(v_R_1930_, v_A_1931_, v_inst_1932_, v_inst_1933_, v_inst_1934_, v_inst_1935_, v_b_1936_);
lean_dec(v_inst_1934_);
lean_dec_ref(v_inst_1932_);
return v_res_1937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeftRight___redArg(lean_object* v_inst_1938_, lean_object* v_ab_1939_){
_start:
{
lean_object* v_fst_1940_; lean_object* v_snd_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___f_1944_; 
v_fst_1940_ = lean_ctor_get(v_ab_1939_, 0);
lean_inc(v_fst_1940_);
v_snd_1941_ = lean_ctor_get(v_ab_1939_, 1);
lean_inc(v_snd_1941_);
lean_dec_ref(v_ab_1939_);
lean_inc_ref(v_inst_1938_);
v___x_1942_ = lp_mathlib_AddMonoidHom_mulRight___redArg(v_inst_1938_, v_snd_1941_);
v___x_1943_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v_inst_1938_, v_fst_1940_);
v___f_1944_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1944_, 0, v___x_1943_);
lean_closure_set(v___f_1944_, 1, v___x_1942_);
return v___f_1944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeftRight(lean_object* v_R_1945_, lean_object* v_A_1946_, lean_object* v_inst_1947_, lean_object* v_inst_1948_, lean_object* v_inst_1949_, lean_object* v_inst_1950_, lean_object* v_inst_1951_, lean_object* v_ab_1952_){
_start:
{
lean_object* v___x_1953_; 
v___x_1953_ = lp_mathlib_LinearMap_mulLeftRight___redArg(v_inst_1948_, v_ab_1952_);
return v___x_1953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mulLeftRight___boxed(lean_object* v_R_1954_, lean_object* v_A_1955_, lean_object* v_inst_1956_, lean_object* v_inst_1957_, lean_object* v_inst_1958_, lean_object* v_inst_1959_, lean_object* v_inst_1960_, lean_object* v_ab_1961_){
_start:
{
lean_object* v_res_1962_; 
v_res_1962_ = lp_mathlib_LinearMap_mulLeftRight(v_R_1954_, v_A_1955_, v_inst_1956_, v_inst_1957_, v_inst_1958_, v_inst_1959_, v_inst_1960_, v_ab_1961_);
lean_dec(v_inst_1958_);
lean_dec_ref(v_inst_1956_);
return v_res_1962_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_RingHom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_RingHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_RingHom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_RingHom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
