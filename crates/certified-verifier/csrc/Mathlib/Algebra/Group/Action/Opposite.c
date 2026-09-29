// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Defs public import Mathlib.Algebra.Group.Opposite
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "RightActions"};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__0 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__0_value;
static const lean_string_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_•>_"};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__1 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__1_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 0, 39, 209, 154, 166, 250, 78)}};
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 89, 231, 107, 149, 210, 34, 200)}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__2 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__2_value;
static const lean_string_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__3 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__4 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__4_value;
static const lean_string_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " •> "};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__5 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__5_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__5_value)}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__6 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__6_value;
static const lean_string_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__7 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__7_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__8 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__8_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__8_value),((lean_object*)(((size_t)(74) << 1) | 1))}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__9 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__9_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__4_value),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__6_value),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__9_value)}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__10 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__10_value;
static const lean_ctor_object lp_mathlib_RightActions_term___u2022_x3e___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__2_value),((lean_object*)(((size_t)(74) << 1) | 1)),((lean_object*)(((size_t)(75) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__10_value)}};
static const lean_object* lp_mathlib_RightActions_term___u2022_x3e___00__closed__11 = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_RightActions_term___u2022_x3e__ = (const lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__11_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_•_"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__0_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(39, 170, 60, 237, 168, 151, 8, 86)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__1_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "•"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HSMul"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hSMul"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 107, 25, 48, 80, 144, 236, 217)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(23, 127, 6, 115, 121, 139, 223, 188)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "m"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(165, 239, 73, 172, 230, 126, 139, 134)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__0_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__0_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__2_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__2_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__5 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__5_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__5_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__6 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_<•_"};
static const lean_object* lp_mathlib_RightActions_term___x3c_u2022___00__closed__0 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__0_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 0, 39, 209, 154, 166, 250, 78)}};
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(39, 24, 18, 1, 93, 43, 143, 153)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_u2022___00__closed__1 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__1_value;
static const lean_string_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " <• "};
static const lean_object* lp_mathlib_RightActions_term___x3c_u2022___00__closed__2 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__2_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__2_value)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_u2022___00__closed__3 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__4_value),((lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__3_value),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__9_value)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_u2022___00__closed__4 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__4_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_u2022___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__1_value),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__4_value)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_u2022___00__closed__5 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_RightActions_term___x3c_u2022__ = (const lean_object*)&lp_mathlib_RightActions_term___x3c_u2022___00__closed__5_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__0_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__1_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__2_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "MulOpposite.op"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__5 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__5_value;
static lean_once_cell_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__6;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "MulOpposite"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__7 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__7_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "op"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__8 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__8_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(167, 55, 13, 115, 157, 142, 229, 51)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(54, 16, 66, 233, 86, 82, 84, 251)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__10 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__10_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__11 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__11_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__12 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__12_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__13 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(167, 55, 13, 115, 157, 142, 229, 51)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__0_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__1_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__2___boxed, .m_arity = 11, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__0_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__0_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__1_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__2_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__2_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__3_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__3_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__4 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_+ᵥ>_"};
static const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__0 = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__0_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 0, 39, 209, 154, 166, 250, 78)}};
static const lean_ctor_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 43, 226, 153, 123, 250, 251, 0)}};
static const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1 = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1_value;
static const lean_string_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " +ᵥ> "};
static const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__2 = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__2_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__2_value)}};
static const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__3 = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__4_value),((lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__3_value),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__9_value)}};
static const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__4 = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__4_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1_value),((lean_object*)(((size_t)(74) << 1) | 1)),((lean_object*)(((size_t)(75) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__4_value)}};
static const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__5 = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_RightActions_term___x2b_u1d65_x3e__ = (const lean_object*)&lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__5_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_+ᵥ_"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__0_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 242, 40, 27, 208, 97, 34, 167)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__1_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "+ᵥ"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HVAdd"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hVAdd"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 135, 107, 242, 117, 15, 176, 86)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(90, 24, 198, 227, 204, 199, 190, 118)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__2_value;
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__0_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__0_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__1_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__1_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__2_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__2_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_<+ᵥ_"};
static const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__0 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__0_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 0, 39, 209, 154, 166, 250, 78)}};
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 99, 142, 219, 108, 68, 208, 195)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1_value;
static const lean_string_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " <+ᵥ "};
static const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__2 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__2_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__2_value)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__3 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__4_value),((lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_RightActions_term___u2022_x3e___00__closed__9_value)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__4 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__4_value;
static const lean_ctor_object lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1_value),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)(((size_t)(73) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__4_value)}};
static const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__5 = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_RightActions_term___x3c_x2b_u1d65__ = (const lean_object*)&lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__5_value;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "AddOpposite.op"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__0_value;
static lean_once_cell_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__1;
static const lean_string_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "AddOpposite"};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__2_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(248, 170, 135, 100, 51, 49, 73, 76)}};
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(21, 212, 86, 111, 184, 86, 135, 236)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__4_value;
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__5 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(248, 170, 135, 100, 51, 49, 73, 76)}};
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__0_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__1 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__1_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__2___boxed, .m_arity = 11, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__0_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__0_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__1_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__1_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__2_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__2_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__3_value;
static const lean_closure_object lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__4_value),((lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__3_value)} };
static const lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulAction___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___f_2_; 
v___f_2_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2_, 0, v_inst_1_);
return v___f_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulAction(lean_object* v_M_3_, lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulAction___boxed(lean_object* v_M_8_, lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_MulOpposite_instMulAction(v_M_8_, v_00_u03b1_9_, v_inst_10_, v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddAction___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_inst_13_);
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddAction(lean_object* v_M_15_, lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___f_19_; 
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_19_, 0, v_inst_18_);
return v___f_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddAction___boxed(lean_object* v_M_20_, lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_AddOpposite_instAddAction(v_M_20_, v_00_u03b1_21_, v_inst_22_, v_inst_23_);
lean_dec_ref(v_inst_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1(lean_object* v_x_56_, lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib_RightActions_term___u2022_x3e___00__closed__2));
lean_inc(v_x_56_);
v___x_60_ = l_Lean_Syntax_isOfKind(v_x_56_, v___x_59_);
if (v___x_60_ == 0)
{
lean_object* v___x_61_; lean_object* v___x_62_; 
lean_dec(v_x_56_);
v___x_61_ = lean_box(1);
v___x_62_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v_a_58_);
return v___x_62_;
}
else
{
lean_object* v_ref_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v_ref_63_ = lean_ctor_get(v_a_57_, 5);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = l_Lean_Syntax_getArg(v_x_56_, v___x_64_);
v___x_66_ = lean_unsigned_to_nat(2u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_56_, v___x_66_);
lean_dec(v_x_56_);
v___x_68_ = 0;
v___x_69_ = l_Lean_SourceInfo_fromRef(v_ref_63_, v___x_68_);
v___x_70_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__1));
v___x_71_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__2));
lean_inc(v___x_69_);
v___x_72_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_72_, 0, v___x_69_);
lean_ctor_set(v___x_72_, 1, v___x_71_);
v___x_73_ = l_Lean_Syntax_node3(v___x_69_, v___x_70_, v___x_65_, v___x_72_, v___x_67_);
v___x_74_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_a_58_);
return v___x_74_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___boxed(lean_object* v_x_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1(v_x_75_, v_a_76_, v_a_77_);
lean_dec_ref(v_a_76_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(lean_object* v___y_79_){
_start:
{
lean_object* v_subExpr_81_; lean_object* v_expr_82_; lean_object* v___x_83_; 
v_subExpr_81_ = lean_ctor_get(v___y_79_, 3);
v_expr_82_ = lean_ctor_get(v_subExpr_81_, 0);
lean_inc_ref(v_expr_82_);
v___x_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_83_, 0, v_expr_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg___boxed(lean_object* v___y_84_, lean_object* v___y_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(v___y_84_);
lean_dec_ref(v___y_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0(lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(v___y_87_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___boxed(lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0(v___y_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_);
lean_dec(v___y_100_);
lean_dec_ref(v___y_99_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
return v_res_102_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0(lean_object* v_x_108_){
_start:
{
lean_object* v___x_109_; uint8_t v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___closed__2));
v___x_110_ = l_Lean_Expr_isConstOf(v_x_108_, v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0___boxed(lean_object* v_x_111_){
_start:
{
uint8_t v_res_112_; lean_object* v_r_113_; 
v_res_112_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__0(v_x_111_);
lean_dec_ref(v_x_111_);
v_r_113_ = lean_box(v_res_112_);
return v_r_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__1(lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_122_, 0, v___y_114_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__1___boxed(lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__1(v___y_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__2(lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
lean_object* v_ref_141_; uint8_t v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_ref_141_ = lean_ctor_get(v___y_138_, 5);
v___x_142_ = 0;
v___x_143_ = l_Lean_SourceInfo_fromRef(v_ref_141_, v___x_142_);
v___x_144_ = ((lean_object*)(lp_mathlib_RightActions_term___u2022_x3e___00__closed__2));
v___x_145_ = ((lean_object*)(lp_mathlib_RightActions_term___u2022_x3e___00__closed__5));
lean_inc(v___x_143_);
v___x_146_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_143_);
lean_ctor_set(v___x_146_, 1, v___x_145_);
v___x_147_ = l_Lean_Syntax_node3(v___x_143_, v___x_144_, v_a_132_, v___x_146_, v_a_133_);
v___x_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__2___boxed(lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__2(v_a_149_, v_a_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_);
lean_dec(v___y_156_);
lean_dec_ref(v___y_155_);
lean_dec(v___y_154_);
lean_dec_ref(v___y_153_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3(lean_object* v___f_169_, lean_object* v___f_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_){
_start:
{
lean_object* v___x_178_; lean_object* v_a_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_213_; 
v___x_178_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(v___y_171_);
v_a_179_ = lean_ctor_get(v___x_178_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v___x_178_);
if (v_isSharedCheck_213_ == 0)
{
v___x_181_ = v___x_178_;
v_isShared_182_ = v_isSharedCheck_213_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_a_179_);
lean_dec(v___x_178_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_213_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_183_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_183_, 0, v___f_169_);
lean_inc_ref_n(v___f_170_, 3);
v___x_184_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_184_, 0, v___x_183_);
lean_closure_set(v___x_184_, 1, v___f_170_);
v___x_185_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_185_, 0, v___x_184_);
lean_closure_set(v___x_185_, 1, v___f_170_);
v___x_186_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_186_, 0, v___x_185_);
lean_closure_set(v___x_186_, 1, v___f_170_);
v___x_187_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_187_, 0, v___x_186_);
lean_closure_set(v___x_187_, 1, v___f_170_);
v___x_188_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1));
v___x_189_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2));
v___x_190_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_190_, 0, v___x_187_);
lean_closure_set(v___x_190_, 1, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4));
v___x_192_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5));
v___x_193_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_194_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_190_, v___x_192_, v___x_193_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_);
if (lean_obj_tag(v___x_194_) == 0)
{
lean_object* v_a_195_; lean_object* v___x_197_; 
v_a_195_ = lean_ctor_get(v___x_194_, 0);
lean_inc(v_a_195_);
lean_dec_ref_known(v___x_194_, 1);
if (v_isShared_182_ == 0)
{
lean_ctor_set_tag(v___x_181_, 1);
v___x_197_ = v___x_181_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_a_179_);
v___x_197_ = v_reuseFailAlloc_204_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
lean_object* v___x_198_; 
lean_inc_ref(v___x_197_);
v___x_198_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_195_, v___x_188_, v___x_197_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_);
if (lean_obj_tag(v___x_198_) == 0)
{
lean_object* v_a_199_; lean_object* v___x_200_; 
v_a_199_ = lean_ctor_get(v___x_198_, 0);
lean_inc(v_a_199_);
lean_dec_ref_known(v___x_198_, 1);
v___x_200_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_195_, v___x_191_, v___x_197_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_);
lean_dec(v_a_195_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v_a_201_; lean_object* v___f_202_; lean_object* v___x_203_; 
v_a_201_ = lean_ctor_get(v___x_200_, 0);
lean_inc(v_a_201_);
lean_dec_ref_known(v___x_200_, 1);
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__2___boxed), 9, 2);
lean_closure_set(v___f_202_, 0, v_a_199_);
lean_closure_set(v___f_202_, 1, v_a_201_);
v___x_203_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_202_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_);
return v___x_203_;
}
else
{
lean_dec(v_a_199_);
return v___x_200_;
}
}
else
{
lean_dec_ref(v___x_197_);
lean_dec(v_a_195_);
return v___x_198_;
}
}
}
else
{
lean_object* v_a_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_212_; 
lean_del_object(v___x_181_);
lean_dec(v_a_179_);
v_a_205_ = lean_ctor_get(v___x_194_, 0);
v_isSharedCheck_212_ = !lean_is_exclusive(v___x_194_);
if (v_isSharedCheck_212_ == 0)
{
v___x_207_ = v___x_194_;
v_isShared_208_ = v_isSharedCheck_212_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_a_205_);
lean_dec(v___x_194_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_212_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_210_; 
if (v_isShared_208_ == 0)
{
v___x_210_ = v___x_207_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_a_205_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___boxed(lean_object* v___f_214_, lean_object* v___f_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3(v___f_214_, v___f_215_, v___y_216_, v___y_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1(lean_object* v_a_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_244_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3));
v___x_245_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__6));
v___x_246_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_244_, v___x_245_, v_a_237_, v_a_238_, v_a_239_, v_a_240_, v_a_241_, v_a_242_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___boxed(lean_object* v_a_247_, lean_object* v_a_248_, lean_object* v_a_249_, lean_object* v_a_250_, lean_object* v_a_251_, lean_object* v_a_252_, lean_object* v_a_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1(v_a_247_, v_a_248_, v_a_249_, v_a_250_, v_a_251_, v_a_252_);
lean_dec(v_a_252_);
lean_dec_ref(v_a_251_);
lean_dec(v_a_250_);
lean_dec_ref(v_a_249_);
lean_dec(v_a_248_);
lean_dec_ref(v_a_247_);
return v_res_254_;
}
}
static lean_object* _init_lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__6(void){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_281_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__5));
v___x_282_ = l_String_toRawSubstring_x27(v___x_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1(lean_object* v_x_297_, lean_object* v_a_298_, lean_object* v_a_299_){
_start:
{
lean_object* v___x_300_; uint8_t v___x_301_; 
v___x_300_ = ((lean_object*)(lp_mathlib_RightActions_term___x3c_u2022___00__closed__1));
lean_inc(v_x_297_);
v___x_301_ = l_Lean_Syntax_isOfKind(v_x_297_, v___x_300_);
if (v___x_301_ == 0)
{
lean_object* v___x_302_; lean_object* v___x_303_; 
lean_dec(v_x_297_);
v___x_302_ = lean_box(1);
v___x_303_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_303_, 0, v___x_302_);
lean_ctor_set(v___x_303_, 1, v_a_299_);
return v___x_303_;
}
else
{
lean_object* v_quotContext_304_; lean_object* v_currMacroScope_305_; lean_object* v_ref_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; uint8_t v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v_quotContext_304_ = lean_ctor_get(v_a_298_, 1);
v_currMacroScope_305_ = lean_ctor_get(v_a_298_, 2);
v_ref_306_ = lean_ctor_get(v_a_298_, 5);
v___x_307_ = lean_unsigned_to_nat(0u);
v___x_308_ = l_Lean_Syntax_getArg(v_x_297_, v___x_307_);
v___x_309_ = lean_unsigned_to_nat(2u);
v___x_310_ = l_Lean_Syntax_getArg(v_x_297_, v___x_309_);
lean_dec(v_x_297_);
v___x_311_ = 0;
v___x_312_ = l_Lean_SourceInfo_fromRef(v_ref_306_, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__1));
v___x_314_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4));
v___x_315_ = lean_obj_once(&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__6, &lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__6_once, _init_lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__6);
v___x_316_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9));
lean_inc(v_currMacroScope_305_);
lean_inc(v_quotContext_304_);
v___x_317_ = l_Lean_addMacroScope(v_quotContext_304_, v___x_316_, v_currMacroScope_305_);
v___x_318_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__11));
lean_inc_n(v___x_312_, 4);
v___x_319_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_319_, 0, v___x_312_);
lean_ctor_set(v___x_319_, 1, v___x_315_);
lean_ctor_set(v___x_319_, 2, v___x_317_);
lean_ctor_set(v___x_319_, 3, v___x_318_);
v___x_320_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__13));
v___x_321_ = l_Lean_Syntax_node1(v___x_312_, v___x_320_, v___x_310_);
v___x_322_ = l_Lean_Syntax_node2(v___x_312_, v___x_314_, v___x_319_, v___x_321_);
v___x_323_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___u2022_x3e____1___closed__2));
v___x_324_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_312_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
v___x_325_ = l_Lean_Syntax_node3(v___x_312_, v___x_313_, v___x_322_, v___x_324_, v___x_308_);
v___x_326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
lean_ctor_set(v___x_326_, 1, v_a_299_);
return v___x_326_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___boxed(lean_object* v_x_327_, lean_object* v_a_328_, lean_object* v_a_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1(v_x_327_, v_a_328_, v_a_329_);
lean_dec_ref(v_a_328_);
return v_res_330_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1(lean_object* v_x_333_){
_start:
{
lean_object* v___x_334_; uint8_t v___x_335_; 
v___x_334_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___closed__0));
v___x_335_ = l_Lean_Expr_isConstOf(v_x_333_, v___x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1___boxed(lean_object* v_x_336_){
_start:
{
uint8_t v_res_337_; lean_object* v_r_338_; 
v_res_337_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__1(v_x_336_);
lean_dec_ref(v_x_336_);
v_r_338_ = lean_box(v_res_337_);
return v_r_338_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__0(lean_object* v_x_339_){
_start:
{
lean_object* v___x_340_; uint8_t v___x_341_; 
v___x_340_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__9));
v___x_341_ = l_Lean_Expr_isConstOf(v_x_339_, v___x_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__0___boxed(lean_object* v_x_342_){
_start:
{
uint8_t v_res_343_; lean_object* v_r_344_; 
v_res_343_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__0(v_x_342_);
lean_dec_ref(v_x_342_);
v_r_344_ = lean_box(v_res_343_);
return v_r_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__3(lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_){
_start:
{
lean_object* v_ref_354_; uint8_t v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; 
v_ref_354_ = lean_ctor_get(v___y_351_, 5);
v___x_355_ = 0;
v___x_356_ = l_Lean_SourceInfo_fromRef(v_ref_354_, v___x_355_);
v___x_357_ = ((lean_object*)(lp_mathlib_RightActions_term___x3c_u2022___00__closed__1));
v___x_358_ = ((lean_object*)(lp_mathlib_RightActions_term___x3c_u2022___00__closed__2));
lean_inc(v___x_356_);
v___x_359_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_359_, 0, v___x_356_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
v___x_360_ = l_Lean_Syntax_node3(v___x_356_, v___x_357_, v_a_345_, v___x_359_, v_a_346_);
v___x_361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_361_, 0, v___x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__3___boxed(lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__3(v_a_362_, v_a_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_, v___y_368_, v___y_369_);
lean_dec(v___y_369_);
lean_dec_ref(v___y_368_);
lean_dec(v___y_367_);
lean_dec_ref(v___y_366_);
lean_dec(v___y_365_);
lean_dec_ref(v___y_364_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__2(lean_object* v___f_372_, lean_object* v___f_373_, lean_object* v___f_374_, lean_object* v___f_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v___x_383_; lean_object* v_a_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_423_; 
v___x_383_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(v___y_376_);
v_a_384_ = lean_ctor_get(v___x_383_, 0);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_383_);
if (v_isSharedCheck_423_ == 0)
{
v___x_386_ = v___x_383_;
v_isShared_387_ = v_isSharedCheck_423_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_a_384_);
lean_dec(v___x_383_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_423_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_388_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_388_, 0, v___f_372_);
v___x_389_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_389_, 0, v___f_373_);
lean_inc_ref_n(v___f_374_, 4);
v___x_390_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_390_, 0, v___x_389_);
lean_closure_set(v___x_390_, 1, v___f_374_);
v___x_391_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_391_, 0, v___x_388_);
lean_closure_set(v___x_391_, 1, v___x_390_);
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_392_, 0, v___x_391_);
lean_closure_set(v___x_392_, 1, v___f_374_);
v___x_393_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_393_, 0, v___x_392_);
lean_closure_set(v___x_393_, 1, v___f_374_);
v___x_394_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_394_, 0, v___x_393_);
lean_closure_set(v___x_394_, 1, v___f_374_);
v___x_395_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_395_, 0, v___f_375_);
v___x_396_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_396_, 0, v___x_395_);
lean_closure_set(v___x_396_, 1, v___f_374_);
v___x_397_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1));
v___x_398_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2));
v___x_399_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_399_, 0, v___x_396_);
lean_closure_set(v___x_399_, 1, v___x_398_);
v___x_400_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_400_, 0, v___x_394_);
lean_closure_set(v___x_400_, 1, v___x_399_);
v___x_401_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4));
v___x_402_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5));
v___x_403_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_404_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_400_, v___x_402_, v___x_403_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
if (lean_obj_tag(v___x_404_) == 0)
{
lean_object* v_a_405_; lean_object* v___x_407_; 
v_a_405_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_a_405_);
lean_dec_ref_known(v___x_404_, 1);
if (v_isShared_387_ == 0)
{
lean_ctor_set_tag(v___x_386_, 1);
v___x_407_ = v___x_386_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_a_384_);
v___x_407_ = v_reuseFailAlloc_414_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
lean_object* v___x_408_; 
lean_inc_ref(v___x_407_);
v___x_408_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_405_, v___x_397_, v___x_407_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
if (lean_obj_tag(v___x_408_) == 0)
{
lean_object* v_a_409_; lean_object* v___x_410_; 
v_a_409_ = lean_ctor_get(v___x_408_, 0);
lean_inc(v_a_409_);
lean_dec_ref_known(v___x_408_, 1);
v___x_410_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_405_, v___x_401_, v___x_407_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
lean_dec(v_a_405_);
if (lean_obj_tag(v___x_410_) == 0)
{
lean_object* v_a_411_; lean_object* v___f_412_; lean_object* v___x_413_; 
v_a_411_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_a_411_);
lean_dec_ref_known(v___x_410_, 1);
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__3___boxed), 9, 2);
lean_closure_set(v___f_412_, 0, v_a_411_);
lean_closure_set(v___f_412_, 1, v_a_409_);
v___x_413_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_412_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
return v___x_413_;
}
else
{
lean_dec(v_a_409_);
return v___x_410_;
}
}
else
{
lean_dec_ref(v___x_407_);
lean_dec(v_a_405_);
return v___x_408_;
}
}
}
else
{
lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_422_; 
lean_del_object(v___x_386_);
lean_dec(v_a_384_);
v_a_415_ = lean_ctor_get(v___x_404_, 0);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_404_);
if (v_isSharedCheck_422_ == 0)
{
v___x_417_ = v___x_404_;
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_404_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_420_; 
if (v_isShared_418_ == 0)
{
v___x_420_ = v___x_417_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_a_415_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__2___boxed(lean_object* v___f_424_, lean_object* v___f_425_, lean_object* v___f_426_, lean_object* v___f_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___lam__2(v___f_424_, v___f_425_, v___f_426_, v___f_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_);
lean_dec(v___y_433_);
lean_dec_ref(v___y_432_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1(lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_456_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3));
v___x_457_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___closed__4));
v___x_458_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_456_, v___x_457_, v_a_449_, v_a_450_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1___boxed(lean_object* v_a_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_u2022____1(v_a_459_, v_a_460_, v_a_461_, v_a_462_, v_a_463_, v_a_464_);
lean_dec(v_a_464_);
lean_dec_ref(v_a_463_);
lean_dec(v_a_462_);
lean_dec_ref(v_a_461_);
lean_dec(v_a_460_);
lean_dec_ref(v_a_459_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1(lean_object* v_x_488_, lean_object* v_a_489_, lean_object* v_a_490_){
_start:
{
lean_object* v___x_491_; uint8_t v___x_492_; 
v___x_491_ = ((lean_object*)(lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1));
lean_inc(v_x_488_);
v___x_492_ = l_Lean_Syntax_isOfKind(v_x_488_, v___x_491_);
if (v___x_492_ == 0)
{
lean_object* v___x_493_; lean_object* v___x_494_; 
lean_dec(v_x_488_);
v___x_493_ = lean_box(1);
v___x_494_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_493_);
lean_ctor_set(v___x_494_, 1, v_a_490_);
return v___x_494_;
}
else
{
lean_object* v_ref_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; uint8_t v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; 
v_ref_495_ = lean_ctor_get(v_a_489_, 5);
v___x_496_ = lean_unsigned_to_nat(0u);
v___x_497_ = l_Lean_Syntax_getArg(v_x_488_, v___x_496_);
v___x_498_ = lean_unsigned_to_nat(2u);
v___x_499_ = l_Lean_Syntax_getArg(v_x_488_, v___x_498_);
lean_dec(v_x_488_);
v___x_500_ = 0;
v___x_501_ = l_Lean_SourceInfo_fromRef(v_ref_495_, v___x_500_);
v___x_502_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__1));
v___x_503_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__2));
lean_inc(v___x_501_);
v___x_504_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_504_, 0, v___x_501_);
lean_ctor_set(v___x_504_, 1, v___x_503_);
v___x_505_ = l_Lean_Syntax_node3(v___x_501_, v___x_502_, v___x_497_, v___x_504_, v___x_499_);
v___x_506_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_506_, 0, v___x_505_);
lean_ctor_set(v___x_506_, 1, v_a_490_);
return v___x_506_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___boxed(lean_object* v_x_507_, lean_object* v_a_508_, lean_object* v_a_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1(v_x_507_, v_a_508_, v_a_509_);
lean_dec_ref(v_a_508_);
return v_res_510_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0(lean_object* v_x_516_){
_start:
{
lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_517_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___closed__2));
v___x_518_ = l_Lean_Expr_isConstOf(v_x_516_, v___x_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0___boxed(lean_object* v_x_519_){
_start:
{
uint8_t v_res_520_; lean_object* v_r_521_; 
v_res_520_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__0(v_x_519_);
lean_dec_ref(v_x_519_);
v_r_521_ = lean_box(v_res_520_);
return v_r_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__2(lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
lean_object* v_ref_531_; uint8_t v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; 
v_ref_531_ = lean_ctor_get(v___y_528_, 5);
v___x_532_ = 0;
v___x_533_ = l_Lean_SourceInfo_fromRef(v_ref_531_, v___x_532_);
v___x_534_ = ((lean_object*)(lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__1));
v___x_535_ = ((lean_object*)(lp_mathlib_RightActions_term___x2b_u1d65_x3e___00__closed__2));
lean_inc(v___x_533_);
v___x_536_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_536_, 0, v___x_533_);
lean_ctor_set(v___x_536_, 1, v___x_535_);
v___x_537_ = l_Lean_Syntax_node3(v___x_533_, v___x_534_, v_a_522_, v___x_536_, v_a_523_);
v___x_538_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_538_, 0, v___x_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__2___boxed(lean_object* v_a_539_, lean_object* v_a_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__2(v_a_539_, v_a_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_);
lean_dec(v___y_546_);
lean_dec_ref(v___y_545_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__1(lean_object* v___f_549_, lean_object* v___f_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_){
_start:
{
lean_object* v___x_558_; lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_593_; 
v___x_558_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(v___y_551_);
v_a_559_ = lean_ctor_get(v___x_558_, 0);
v_isSharedCheck_593_ = !lean_is_exclusive(v___x_558_);
if (v_isSharedCheck_593_ == 0)
{
v___x_561_ = v___x_558_;
v_isShared_562_ = v_isSharedCheck_593_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_558_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_593_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_563_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_563_, 0, v___f_549_);
lean_inc_ref_n(v___f_550_, 3);
v___x_564_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_564_, 0, v___x_563_);
lean_closure_set(v___x_564_, 1, v___f_550_);
v___x_565_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_565_, 0, v___x_564_);
lean_closure_set(v___x_565_, 1, v___f_550_);
v___x_566_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_566_, 0, v___x_565_);
lean_closure_set(v___x_566_, 1, v___f_550_);
v___x_567_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_567_, 0, v___x_566_);
lean_closure_set(v___x_567_, 1, v___f_550_);
v___x_568_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1));
v___x_569_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2));
v___x_570_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_570_, 0, v___x_567_);
lean_closure_set(v___x_570_, 1, v___x_569_);
v___x_571_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4));
v___x_572_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5));
v___x_573_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_574_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_570_, v___x_572_, v___x_573_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
if (lean_obj_tag(v___x_574_) == 0)
{
lean_object* v_a_575_; lean_object* v___x_577_; 
v_a_575_ = lean_ctor_get(v___x_574_, 0);
lean_inc(v_a_575_);
lean_dec_ref_known(v___x_574_, 1);
if (v_isShared_562_ == 0)
{
lean_ctor_set_tag(v___x_561_, 1);
v___x_577_ = v___x_561_;
goto v_reusejp_576_;
}
else
{
lean_object* v_reuseFailAlloc_584_; 
v_reuseFailAlloc_584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_584_, 0, v_a_559_);
v___x_577_ = v_reuseFailAlloc_584_;
goto v_reusejp_576_;
}
v_reusejp_576_:
{
lean_object* v___x_578_; 
lean_inc_ref(v___x_577_);
v___x_578_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_575_, v___x_568_, v___x_577_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
if (lean_obj_tag(v___x_578_) == 0)
{
lean_object* v_a_579_; lean_object* v___x_580_; 
v_a_579_ = lean_ctor_get(v___x_578_, 0);
lean_inc(v_a_579_);
lean_dec_ref_known(v___x_578_, 1);
v___x_580_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_575_, v___x_571_, v___x_577_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
lean_dec(v_a_575_);
if (lean_obj_tag(v___x_580_) == 0)
{
lean_object* v_a_581_; lean_object* v___f_582_; lean_object* v___x_583_; 
v_a_581_ = lean_ctor_get(v___x_580_, 0);
lean_inc(v_a_581_);
lean_dec_ref_known(v___x_580_, 1);
v___f_582_ = lean_alloc_closure((void*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__2___boxed), 9, 2);
lean_closure_set(v___f_582_, 0, v_a_579_);
lean_closure_set(v___f_582_, 1, v_a_581_);
v___x_583_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_582_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
return v___x_583_;
}
else
{
lean_dec(v_a_579_);
return v___x_580_;
}
}
else
{
lean_dec_ref(v___x_577_);
lean_dec(v_a_575_);
return v___x_578_;
}
}
}
else
{
lean_object* v_a_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
lean_del_object(v___x_561_);
lean_dec(v_a_559_);
v_a_585_ = lean_ctor_get(v___x_574_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_592_ == 0)
{
v___x_587_ = v___x_574_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_a_585_);
lean_dec(v___x_574_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_a_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__1___boxed(lean_object* v___f_594_, lean_object* v___f_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___lam__1(v___f_594_, v___f_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_, v___y_600_, v___y_601_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
lean_dec(v___y_599_);
lean_dec_ref(v___y_598_);
lean_dec(v___y_597_);
lean_dec_ref(v___y_596_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1(lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_){
_start:
{
lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_621_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3));
v___x_622_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___closed__3));
v___x_623_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_621_, v___x_622_, v_a_614_, v_a_615_, v_a_616_, v_a_617_, v_a_618_, v_a_619_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1___boxed(lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x2b_u1d65_x3e____1(v_a_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_, v_a_629_);
lean_dec(v_a_629_);
lean_dec_ref(v_a_628_);
lean_dec(v_a_627_);
lean_dec_ref(v_a_626_);
lean_dec(v_a_625_);
lean_dec_ref(v_a_624_);
return v_res_631_;
}
}
static lean_object* _init_lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__1(void){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_649_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__0));
v___x_650_ = l_String_toRawSubstring_x27(v___x_649_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1(lean_object* v_x_661_, lean_object* v_a_662_, lean_object* v_a_663_){
_start:
{
lean_object* v___x_664_; uint8_t v___x_665_; 
v___x_664_ = ((lean_object*)(lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1));
lean_inc(v_x_661_);
v___x_665_ = l_Lean_Syntax_isOfKind(v_x_661_, v___x_664_);
if (v___x_665_ == 0)
{
lean_object* v___x_666_; lean_object* v___x_667_; 
lean_dec(v_x_661_);
v___x_666_ = lean_box(1);
v___x_667_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_667_, 0, v___x_666_);
lean_ctor_set(v___x_667_, 1, v_a_663_);
return v___x_667_;
}
else
{
lean_object* v_quotContext_668_; lean_object* v_currMacroScope_669_; lean_object* v_ref_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; uint8_t v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v_quotContext_668_ = lean_ctor_get(v_a_662_, 1);
v_currMacroScope_669_ = lean_ctor_get(v_a_662_, 2);
v_ref_670_ = lean_ctor_get(v_a_662_, 5);
v___x_671_ = lean_unsigned_to_nat(0u);
v___x_672_ = l_Lean_Syntax_getArg(v_x_661_, v___x_671_);
v___x_673_ = lean_unsigned_to_nat(2u);
v___x_674_ = l_Lean_Syntax_getArg(v_x_661_, v___x_673_);
lean_dec(v_x_661_);
v___x_675_ = 0;
v___x_676_ = l_Lean_SourceInfo_fromRef(v_ref_670_, v___x_675_);
v___x_677_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__1));
v___x_678_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__4));
v___x_679_ = lean_obj_once(&lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__1, &lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__1_once, _init_lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__1);
v___x_680_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3));
lean_inc(v_currMacroScope_669_);
lean_inc(v_quotContext_668_);
v___x_681_ = l_Lean_addMacroScope(v_quotContext_668_, v___x_680_, v_currMacroScope_669_);
v___x_682_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__5));
lean_inc_n(v___x_676_, 4);
v___x_683_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_683_, 0, v___x_676_);
lean_ctor_set(v___x_683_, 1, v___x_679_);
lean_ctor_set(v___x_683_, 2, v___x_681_);
lean_ctor_set(v___x_683_, 3, v___x_682_);
v___x_684_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_u2022____1___closed__13));
v___x_685_ = l_Lean_Syntax_node1(v___x_676_, v___x_684_, v___x_674_);
v___x_686_ = l_Lean_Syntax_node2(v___x_676_, v___x_678_, v___x_683_, v___x_685_);
v___x_687_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x2b_u1d65_x3e____1___closed__2));
v___x_688_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_688_, 0, v___x_676_);
lean_ctor_set(v___x_688_, 1, v___x_687_);
v___x_689_ = l_Lean_Syntax_node3(v___x_676_, v___x_677_, v___x_686_, v___x_688_, v___x_672_);
v___x_690_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_690_, 0, v___x_689_);
lean_ctor_set(v___x_690_, 1, v_a_663_);
return v___x_690_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___boxed(lean_object* v_x_691_, lean_object* v_a_692_, lean_object* v_a_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1(v_x_691_, v_a_692_, v_a_693_);
lean_dec_ref(v_a_692_);
return v_res_694_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1(lean_object* v_x_697_){
_start:
{
lean_object* v___x_698_; uint8_t v___x_699_; 
v___x_698_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___closed__0));
v___x_699_ = l_Lean_Expr_isConstOf(v_x_697_, v___x_698_);
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1___boxed(lean_object* v_x_700_){
_start:
{
uint8_t v_res_701_; lean_object* v_r_702_; 
v_res_701_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__1(v_x_700_);
lean_dec_ref(v_x_700_);
v_r_702_ = lean_box(v_res_701_);
return v_r_702_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__0(lean_object* v_x_703_){
_start:
{
lean_object* v___x_704_; uint8_t v___x_705_; 
v___x_704_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______macroRules__RightActions__term___x3c_x2b_u1d65____1___closed__3));
v___x_705_ = l_Lean_Expr_isConstOf(v_x_703_, v___x_704_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__0___boxed(lean_object* v_x_706_){
_start:
{
uint8_t v_res_707_; lean_object* v_r_708_; 
v_res_707_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__0(v_x_706_);
lean_dec_ref(v_x_706_);
v_r_708_ = lean_box(v_res_707_);
return v_r_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__3(lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_){
_start:
{
lean_object* v_ref_718_; uint8_t v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; 
v_ref_718_ = lean_ctor_get(v___y_715_, 5);
v___x_719_ = 0;
v___x_720_ = l_Lean_SourceInfo_fromRef(v_ref_718_, v___x_719_);
v___x_721_ = ((lean_object*)(lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__1));
v___x_722_ = ((lean_object*)(lp_mathlib_RightActions_term___x3c_x2b_u1d65___00__closed__2));
lean_inc(v___x_720_);
v___x_723_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_720_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
v___x_724_ = l_Lean_Syntax_node3(v___x_720_, v___x_721_, v_a_709_, v___x_723_, v_a_710_);
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
return v___x_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__3___boxed(lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_){
_start:
{
lean_object* v_res_735_; 
v_res_735_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__3(v_a_726_, v_a_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_, v___y_732_, v___y_733_);
lean_dec(v___y_733_);
lean_dec_ref(v___y_732_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
return v_res_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__2(lean_object* v___f_736_, lean_object* v___f_737_, lean_object* v___f_738_, lean_object* v___f_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
lean_object* v___x_747_; lean_object* v_a_748_; lean_object* v___x_750_; uint8_t v_isShared_751_; uint8_t v_isSharedCheck_787_; 
v___x_747_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1_spec__0___redArg(v___y_740_);
v_a_748_ = lean_ctor_get(v___x_747_, 0);
v_isSharedCheck_787_ = !lean_is_exclusive(v___x_747_);
if (v_isSharedCheck_787_ == 0)
{
v___x_750_ = v___x_747_;
v_isShared_751_ = v_isSharedCheck_787_;
goto v_resetjp_749_;
}
else
{
lean_inc(v_a_748_);
lean_dec(v___x_747_);
v___x_750_ = lean_box(0);
v_isShared_751_ = v_isSharedCheck_787_;
goto v_resetjp_749_;
}
v_resetjp_749_:
{
lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_752_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_752_, 0, v___f_736_);
v___x_753_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_753_, 0, v___f_737_);
lean_inc_ref_n(v___f_738_, 4);
v___x_754_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_754_, 0, v___x_753_);
lean_closure_set(v___x_754_, 1, v___f_738_);
v___x_755_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_755_, 0, v___x_752_);
lean_closure_set(v___x_755_, 1, v___x_754_);
v___x_756_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_756_, 0, v___x_755_);
lean_closure_set(v___x_756_, 1, v___f_738_);
v___x_757_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_757_, 0, v___x_756_);
lean_closure_set(v___x_757_, 1, v___f_738_);
v___x_758_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_758_, 0, v___x_757_);
lean_closure_set(v___x_758_, 1, v___f_738_);
v___x_759_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_759_, 0, v___f_739_);
v___x_760_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_760_, 0, v___x_759_);
lean_closure_set(v___x_760_, 1, v___f_738_);
v___x_761_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__1));
v___x_762_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__2));
v___x_763_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_763_, 0, v___x_760_);
lean_closure_set(v___x_763_, 1, v___x_762_);
v___x_764_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_764_, 0, v___x_758_);
lean_closure_set(v___x_764_, 1, v___x_763_);
v___x_765_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__4));
v___x_766_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___lam__3___closed__5));
v___x_767_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_768_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_764_, v___x_766_, v___x_767_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
if (lean_obj_tag(v___x_768_) == 0)
{
lean_object* v_a_769_; lean_object* v___x_771_; 
v_a_769_ = lean_ctor_get(v___x_768_, 0);
lean_inc(v_a_769_);
lean_dec_ref_known(v___x_768_, 1);
if (v_isShared_751_ == 0)
{
lean_ctor_set_tag(v___x_750_, 1);
v___x_771_ = v___x_750_;
goto v_reusejp_770_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v_a_748_);
v___x_771_ = v_reuseFailAlloc_778_;
goto v_reusejp_770_;
}
v_reusejp_770_:
{
lean_object* v___x_772_; 
lean_inc_ref(v___x_771_);
v___x_772_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_769_, v___x_761_, v___x_771_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
if (lean_obj_tag(v___x_772_) == 0)
{
lean_object* v_a_773_; lean_object* v___x_774_; 
v_a_773_ = lean_ctor_get(v___x_772_, 0);
lean_inc(v_a_773_);
lean_dec_ref_known(v___x_772_, 1);
v___x_774_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_769_, v___x_765_, v___x_771_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
lean_dec(v_a_769_);
if (lean_obj_tag(v___x_774_) == 0)
{
lean_object* v_a_775_; lean_object* v___f_776_; lean_object* v___x_777_; 
v_a_775_ = lean_ctor_get(v___x_774_, 0);
lean_inc(v_a_775_);
lean_dec_ref_known(v___x_774_, 1);
v___f_776_ = lean_alloc_closure((void*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__3___boxed), 9, 2);
lean_closure_set(v___f_776_, 0, v_a_775_);
lean_closure_set(v___f_776_, 1, v_a_773_);
v___x_777_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_776_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
return v___x_777_;
}
else
{
lean_dec(v_a_773_);
return v___x_774_;
}
}
else
{
lean_dec_ref(v___x_771_);
lean_dec(v_a_769_);
return v___x_772_;
}
}
}
else
{
lean_object* v_a_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_786_; 
lean_del_object(v___x_750_);
lean_dec(v_a_748_);
v_a_779_ = lean_ctor_get(v___x_768_, 0);
v_isSharedCheck_786_ = !lean_is_exclusive(v___x_768_);
if (v_isSharedCheck_786_ == 0)
{
v___x_781_ = v___x_768_;
v_isShared_782_ = v_isSharedCheck_786_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_a_779_);
lean_dec(v___x_768_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_786_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v___x_784_; 
if (v_isShared_782_ == 0)
{
v___x_784_ = v___x_781_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_785_; 
v_reuseFailAlloc_785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_785_, 0, v_a_779_);
v___x_784_ = v_reuseFailAlloc_785_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
return v___x_784_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__2___boxed(lean_object* v___f_788_, lean_object* v___f_789_, lean_object* v___f_790_, lean_object* v___f_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___lam__2(v___f_788_, v___f_789_, v___f_790_, v___f_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_, v___y_796_, v___y_797_);
lean_dec(v___y_797_);
lean_dec_ref(v___y_796_);
lean_dec(v___y_795_);
lean_dec_ref(v___y_794_);
lean_dec(v___y_793_);
lean_dec_ref(v___y_792_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1(lean_object* v_a_813_, lean_object* v_a_814_, lean_object* v_a_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_){
_start:
{
lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_820_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___u2022_x3e____1___closed__3));
v___x_821_ = ((lean_object*)(lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___closed__4));
v___x_822_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_820_, v___x_821_, v_a_813_, v_a_814_, v_a_815_, v_a_816_, v_a_817_, v_a_818_);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1___boxed(lean_object* v_a_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_mathlib_RightActions___aux__Mathlib__Algebra__Group__Action__Opposite______delab__app__RightActions__term___x3c_x2b_u1d65____1(v_a_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_, v_a_828_);
lean_dec(v_a_828_);
lean_dec_ref(v_a_827_);
lean_dec(v_a_826_);
lean_dec_ref(v_a_825_);
lean_dec(v_a_824_);
lean_dec_ref(v_a_823_);
return v_res_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction___redArg(lean_object* v_inst_831_){
_start:
{
lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v_toMul_834_; lean_object* v___f_835_; 
v___x_832_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_831_);
v___x_833_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_832_);
v_toMul_834_ = lean_ctor_get(v___x_833_, 1);
lean_inc(v_toMul_834_);
lean_dec_ref(v___x_833_);
v___f_835_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_835_, 0, v_toMul_834_);
return v___f_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction___redArg___boxed(lean_object* v_inst_836_){
_start:
{
lean_object* v_res_837_; 
v_res_837_ = lp_mathlib_Monoid_toOppositeMulAction___redArg(v_inst_836_);
lean_dec_ref(v_inst_836_);
return v_res_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction(lean_object* v_00_u03b1_838_, lean_object* v_inst_839_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_mathlib_Monoid_toOppositeMulAction___redArg(v_inst_839_);
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toOppositeMulAction___boxed(lean_object* v_00_u03b1_841_, lean_object* v_inst_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_mathlib_Monoid_toOppositeMulAction(v_00_u03b1_841_, v_inst_842_);
lean_dec_ref(v_inst_842_);
return v_res_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction___redArg(lean_object* v_inst_844_){
_start:
{
lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v_toAdd_847_; lean_object* v___f_848_; 
v___x_845_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_844_);
v___x_846_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_845_);
v_toAdd_847_ = lean_ctor_get(v___x_846_, 1);
lean_inc(v_toAdd_847_);
lean_dec_ref(v___x_846_);
v___f_848_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_848_, 0, v_toAdd_847_);
return v___f_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction___redArg___boxed(lean_object* v_inst_849_){
_start:
{
lean_object* v_res_850_; 
v_res_850_ = lp_mathlib_AddMonoid_toOppositeAddAction___redArg(v_inst_849_);
lean_dec_ref(v_inst_849_);
return v_res_850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction(lean_object* v_00_u03b1_851_, lean_object* v_inst_852_){
_start:
{
lean_object* v___x_853_; 
v___x_853_ = lp_mathlib_AddMonoid_toOppositeAddAction___redArg(v_inst_852_);
return v___x_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toOppositeAddAction___boxed(lean_object* v_00_u03b1_854_, lean_object* v_inst_855_){
_start:
{
lean_object* v_res_856_; 
v_res_856_ = lp_mathlib_AddMonoid_toOppositeAddAction(v_00_u03b1_854_, v_inst_855_);
lean_dec_ref(v_inst_855_);
return v_res_856_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
