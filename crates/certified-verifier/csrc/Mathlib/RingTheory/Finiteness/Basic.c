// Lean compiler output
// Module: Mathlib.RingTheory.Finiteness.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Shrink public import Mathlib.Algebra.Algebra.Tower public import Mathlib.Algebra.Order.Nonneg.Module public import Mathlib.LinearAlgebra.Pi public import Mathlib.LinearAlgebra.Quotient.Defs public import Mathlib.RingTheory.Finiteness.Defs
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
lean_object* lp_mathlib_Submodule_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_isType_x27___boxed(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchFVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeFG(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeFG___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingTheory"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__4_value),LEAN_SCALAR_PTR_LITERAL(81, 182, 200, 127, 246, 185, 232, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finiteness"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__6_value),LEAN_SCALAR_PTR_LITERAL(38, 6, 54, 37, 134, 66, 184, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__8_value),LEAN_SCALAR_PTR_LITERAL(221, 95, 26, 108, 200, 13, 19, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(232, 159, 64, 75, 83, 199, 67, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "termR≥0"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__11_value),LEAN_SCALAR_PTR_LITERAL(68, 3, 183, 35, 96, 151, 56, 57)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "R≥0"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__12_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Nonneg"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(221, 99, 24, 205, 44, 73, 232, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "R"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__15;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(10, 150, 1, 122, 163, 250, 19, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(10, 150, 1, 122, 163, 250, 19, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(99, 199, 61, 239, 123, 123, 35, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__19_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__2_value),LEAN_SCALAR_PTR_LITERAL(62, 116, 43, 91, 63, 55, 139, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__20_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__4_value),LEAN_SCALAR_PTR_LITERAL(141, 69, 137, 210, 240, 17, 125, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__21_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__6_value),LEAN_SCALAR_PTR_LITERAL(58, 203, 47, 11, 65, 5, 14, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__22_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__8_value),LEAN_SCALAR_PTR_LITERAL(217, 169, 83, 100, 197, 57, 68, 17)}};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__23_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__24;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__25_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__26;
static const lean_string_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__27_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__28;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__29;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__30;
static lean_once_cell_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__31;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_isType_x27___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchExpr___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchFVar___boxed, .m_arity = 10, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6_value;
static const lean_closure_object lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6_value)} };
static const lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___lam__0(lean_object* v_P_1_, lean_object* v_Q_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___f_8_ = ((lean_object*)(lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___closed__0));
v___x_9_ = lp_mathlib_Submodule_instPartialOrder(lean_box(0), lean_box(0), v_inst_5_, v_inst_6_, v_inst_7_);
v___x_10_ = lp_mathlib_Subtype_preorder(lean_box(0), v___x_9_, lean_box(0));
lean_dec_ref(v___x_9_);
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___f_8_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg___boxed(lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg(v_inst_12_, v_inst_13_, v_inst_14_);
lean_dec(v_inst_14_);
lean_dec_ref(v_inst_13_);
lean_dec_ref(v_inst_12_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG(lean_object* v_R_16_, lean_object* v_M_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___redArg(v_inst_18_, v_inst_19_, v_inst_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instSemilatticeSupSubtypeFG___boxed(lean_object* v_R_22_, lean_object* v_M_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Submodule_instSemilatticeSupSubtypeFG(v_R_22_, v_M_23_, v_inst_24_, v_inst_25_, v_inst_26_);
lean_dec(v_inst_26_);
lean_dec_ref(v_inst_25_);
lean_dec_ref(v_inst_24_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeFG(lean_object* v_R_28_, lean_object* v_M_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_box(0);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInhabitedSubtypeFG___boxed(lean_object* v_R_34_, lean_object* v_M_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Submodule_instInhabitedSubtypeFG(v_R_34_, v_M_35_, v_inst_36_, v_inst_37_, v_inst_38_);
lean_dec(v_inst_38_);
lean_dec_ref(v_inst_37_);
lean_dec_ref(v_inst_36_);
return v_res_39_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__5));
v___x_86_ = l_String_toRawSubstring_x27(v___x_85_);
return v___x_86_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__15(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__14));
v___x_105_ = l_String_toRawSubstring_x27(v___x_104_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__24(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = lean_unsigned_to_nat(3057824669u);
v___x_128_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__23));
v___x_129_ = l_Lean_Name_num___override(v___x_128_, v___x_127_);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__26(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__25));
v___x_132_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__24, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__24_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__24);
v___x_133_ = l_Lean_Name_str___override(v___x_132_, v___x_131_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__28(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_135_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__27));
v___x_136_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__26, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__26_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__26);
v___x_137_ = l_Lean_Name_str___override(v___x_136_, v___x_135_);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__29(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_unsigned_to_nat(21u);
v___x_139_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__28, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__28_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__28);
v___x_140_ = l_Lean_Name_num___override(v___x_139_, v___x_138_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__30(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_141_ = lean_box(0);
v___x_142_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__29, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__29_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__29);
v___x_143_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v___x_141_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__31(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_144_ = lean_box(0);
v___x_145_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__30, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__30_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__30);
v___x_146_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v___x_144_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_150_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__12));
v___x_151_ = l_Lean_Syntax_isOfKind(v_x_147_, v___x_150_);
if (v___x_151_ == 0)
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = lean_box(1);
v___x_153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v_a_149_);
return v___x_153_;
}
else
{
lean_object* v_quotContext_154_; lean_object* v_currMacroScope_155_; lean_object* v_ref_156_; uint8_t v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v_quotContext_154_ = lean_ctor_get(v_a_148_, 1);
v_currMacroScope_155_ = lean_ctor_get(v_a_148_, 2);
v_ref_156_ = lean_ctor_get(v_a_148_, 5);
v___x_157_ = 0;
v___x_158_ = l_Lean_SourceInfo_fromRef(v_ref_156_, v___x_157_);
v___x_159_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4));
v___x_160_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__6);
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7));
lean_inc_n(v_currMacroScope_155_, 2);
lean_inc_n(v_quotContext_154_, 2);
v___x_162_ = l_Lean_addMacroScope(v_quotContext_154_, v___x_161_, v_currMacroScope_155_);
v___x_163_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__11));
lean_inc_n(v___x_158_, 3);
v___x_164_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_164_, 0, v___x_158_);
lean_ctor_set(v___x_164_, 1, v___x_160_);
lean_ctor_set(v___x_164_, 2, v___x_162_);
lean_ctor_set(v___x_164_, 3, v___x_163_);
v___x_165_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__13));
v___x_166_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__15, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__15_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__15);
v___x_167_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__16));
v___x_168_ = l_Lean_addMacroScope(v_quotContext_154_, v___x_167_, v_currMacroScope_155_);
v___x_169_ = lean_obj_once(&lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__31, &lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__31_once, _init_lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__31);
v___x_170_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_170_, 0, v___x_158_);
lean_ctor_set(v___x_170_, 1, v___x_166_);
lean_ctor_set(v___x_170_, 2, v___x_168_);
lean_ctor_set(v___x_170_, 3, v___x_169_);
v___x_171_ = l_Lean_Syntax_node1(v___x_158_, v___x_165_, v___x_170_);
v___x_172_ = l_Lean_Syntax_node2(v___x_158_, v___x_159_, v___x_164_, v___x_171_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
lean_ctor_set(v___x_173_, 1, v_a_149_);
return v___x_173_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___boxed(lean_object* v_x_174_, lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1(v_x_174_, v_a_175_, v_a_176_);
lean_dec_ref(v_a_175_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg(lean_object* v___y_178_){
_start:
{
lean_object* v_subExpr_180_; lean_object* v_expr_181_; lean_object* v___x_182_; 
v_subExpr_180_ = lean_ctor_get(v___y_178_, 3);
v_expr_181_ = lean_ctor_get(v_subExpr_180_, 0);
lean_inc_ref(v_expr_181_);
v___x_182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_182_, 0, v_expr_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg___boxed(lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg(v___y_183_);
lean_dec_ref(v___y_183_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0(lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg(v___y_186_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___boxed(lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0(v___y_194_, v___y_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
lean_dec(v___y_195_);
lean_dec_ref(v___y_194_);
return v_res_201_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__0(lean_object* v_x_202_){
_start:
{
lean_object* v___x_203_; uint8_t v___x_204_; 
v___x_203_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______macroRules____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7));
v___x_204_ = l_Lean_Expr_isConstOf(v_x_202_, v___x_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__0___boxed(lean_object* v_x_205_){
_start:
{
uint8_t v_res_206_; lean_object* v_r_207_; 
v_res_206_ = lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__0(v_x_205_);
lean_dec_ref(v_x_205_);
v_r_207_ = lean_box(v_res_206_);
return v_r_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__1(lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_216_, 0, v___y_208_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__1___boxed(lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__1(v___y_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__2(lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_){
_start:
{
lean_object* v_ref_233_; uint8_t v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v_ref_233_ = lean_ctor_get(v___y_230_, 5);
v___x_234_ = 0;
v___x_235_ = l_Lean_SourceInfo_fromRef(v_ref_233_, v___x_234_);
v___x_236_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__12));
v___x_237_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0__termR_u22650___closed__13));
lean_inc(v___x_235_);
v___x_238_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_235_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = l_Lean_Syntax_node1(v___x_235_, v___x_236_, v___x_238_);
v___x_240_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__2___boxed(lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__2(v___y_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
lean_dec(v___y_246_);
lean_dec_ref(v___y_245_);
lean_dec(v___y_244_);
lean_dec_ref(v___y_243_);
lean_dec(v___y_242_);
lean_dec_ref(v___y_241_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3(lean_object* v___f_255_, lean_object* v___f_256_, lean_object* v___f_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_265_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1_spec__0___redArg(v___y_258_);
lean_dec_ref(v___x_265_);
v___x_266_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_266_, 0, v___f_255_);
v___x_267_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___closed__2));
v___x_268_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_268_, 0, v___x_266_);
lean_closure_set(v___x_268_, 1, v___x_267_);
lean_inc_ref(v___f_256_);
v___x_269_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_269_, 0, v___x_268_);
lean_closure_set(v___x_269_, 1, v___f_256_);
v___x_270_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_271_ = lp_mathlib_Mathlib_Notation3_matchApp(v___x_269_, v___f_256_, v___x_270_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
if (lean_obj_tag(v___x_271_) == 0)
{
lean_object* v___x_272_; 
lean_dec_ref_known(v___x_271_, 1);
v___x_272_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_257_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
return v___x_272_;
}
else
{
lean_object* v_a_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_280_; 
lean_dec_ref(v___f_257_);
v_a_273_ = lean_ctor_get(v___x_271_, 0);
v_isSharedCheck_280_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_280_ == 0)
{
v___x_275_ = v___x_271_;
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_a_273_);
lean_dec(v___x_271_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_278_; 
if (v_isShared_276_ == 0)
{
v___x_278_ = v___x_275_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v_a_273_);
v___x_278_ = v_reuseFailAlloc_279_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
return v___x_278_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3___boxed(lean_object* v___f_281_, lean_object* v___f_282_, lean_object* v___f_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___lam__3(v___f_281_, v___f_282_, v___f_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
lean_dec(v___y_289_);
lean_dec_ref(v___y_288_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1(lean_object* v_a_307_, lean_object* v_a_308_, lean_object* v_a_309_, lean_object* v_a_310_, lean_object* v_a_311_, lean_object* v_a_312_){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_314_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__4));
v___x_315_ = ((lean_object*)(lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___closed__7));
v___x_316_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_314_, v___x_315_, v_a_307_, v_a_308_, v_a_309_, v_a_310_, v_a_311_, v_a_312_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1___boxed(lean_object* v_a_317_, lean_object* v_a_318_, lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v_a_321_, lean_object* v_a_322_, lean_object* v_a_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib___private_Mathlib_RingTheory_Finiteness_Basic_0____aux__Mathlib__RingTheory__Finiteness__Basic______delab__app____private__Mathlib__RingTheory__Finiteness__Basic__0__termR_u22650__1(v_a_317_, v_a_318_, v_a_319_, v_a_320_, v_a_321_, v_a_322_);
lean_dec(v_a_322_);
lean_dec_ref(v_a_321_);
lean_dec(v_a_320_);
lean_dec_ref(v_a_319_);
lean_dec(v_a_318_);
lean_dec_ref(v_a_317_);
return v_res_324_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Shrink(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Shrink(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Shrink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Finiteness_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Finiteness_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
