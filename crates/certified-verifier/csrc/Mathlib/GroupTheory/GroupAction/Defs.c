// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Basic public import Mathlib.Algebra.Group.Pointwise.Set.Scalar public import Mathlib.Algebra.Group.Subgroup.Defs public import Mathlib.Algebra.Group.Submonoid.MulAction public import Mathlib.Data.Set.BooleanAlgebra public meta import Mathlib.Tactic.ToDual
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
lean_object* lp_mathlib_Equiv_subtypeEquivRight(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
lean_object* l_Quotient_mk_x27___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizerSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizerSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizerAddSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizerAddSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_submonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_submonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_subgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_subgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_orbitRel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_orbitRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_orbitRel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_orbitRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "GroupTheory"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__4_value),LEAN_SCALAR_PTR_LITERAL(21, 126, 254, 74, 51, 201, 216, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "GroupAction"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__6_value),LEAN_SCALAR_PTR_LITERAL(176, 146, 31, 162, 21, 62, 246, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Defs"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(104, 167, 230, 144, 195, 181, 64, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(65, 62, 162, 9, 126, 41, 250, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MulAction"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__11_value),LEAN_SCALAR_PTR_LITERAL(14, 107, 230, 230, 74, 118, 121, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termΩ"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__13_value),LEAN_SCALAR_PTR_LITERAL(248, 22, 196, 196, 155, 137, 21, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "Ω"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__14_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "orbitRel.Quotient"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "orbitRel"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Quotient"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(98, 123, 177, 95, 99, 141, 29, 30)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(62, 235, 132, 72, 175, 249, 143, 207)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__11_value),LEAN_SCALAR_PTR_LITERAL(28, 59, 20, 151, 135, 131, 123, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(65, 252, 211, 246, 104, 52, 3, 94)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(73, 115, 172, 144, 11, 6, 217, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "G"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__18;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__20_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(216, 211, 51, 65, 145, 150, 96, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__22_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__2_value),LEAN_SCALAR_PTR_LITERAL(193, 52, 230, 80, 73, 178, 33, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__23_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 242, 112, 36, 24, 65, 120, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__24_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__6_value),LEAN_SCALAR_PTR_LITERAL(99, 191, 166, 110, 130, 162, 202, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__25_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(183, 156, 185, 157, 170, 89, 87, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__26_value),((lean_object*)(((size_t)(927793915) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(177, 120, 123, 163, 118, 80, 227, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__27_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__27_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(210, 237, 40, 56, 104, 221, 20, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__29_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__29_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(254, 31, 40, 174, 164, 180, 61, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__31_value),((lean_object*)(((size_t)(17) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(144, 55, 255, 16, 11, 117, 223, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__32_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__32_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__34_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__35_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__36;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__37_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__38_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__38_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 206, 72, 126, 215, 111, 61)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__39_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__39_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__2_value),LEAN_SCALAR_PTR_LITERAL(26, 242, 46, 175, 203, 191, 82, 215)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__40_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__4_value),LEAN_SCALAR_PTR_LITERAL(197, 178, 114, 128, 70, 140, 93, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__41_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__41_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__6_value),LEAN_SCALAR_PTR_LITERAL(160, 181, 204, 53, 228, 229, 226, 23)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__42_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__42_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__8_value),LEAN_SCALAR_PTR_LITERAL(184, 141, 74, 166, 17, 236, 178, 20)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__43_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__43_value),((lean_object*)(((size_t)(927793915) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(74, 42, 245, 32, 7, 127, 62, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__44_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(69, 121, 53, 232, 167, 246, 201, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__45_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__45_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(229, 218, 92, 252, 68, 231, 126, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__46_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__46_value),((lean_object*)(((size_t)(18) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(49, 127, 226, 110, 105, 189, 72, 55)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__47_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__47_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__48 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__48_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__48_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__49 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__49_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__0 = (const lean_object*)&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__0_value;
static const lean_closure_object lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Quotient_mk_x27___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__1 = (const lean_object*)&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__1_value;
static lean_once_cell_t lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__2;
static lean_once_cell_t lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__3;
static lean_once_cell_t lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__4;
static lean_once_cell_t lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_selfEquivSigmaOrbits_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_selfEquivSigmaOrbits_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_m_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_m_2_, v___y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit(lean_object* v_M_7_, lean_object* v_00_u03b1_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_a_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit___redArg___lam__0), 3, 1);
lean_closure_set(v___f_12_, 0, v_inst_10_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit___boxed(lean_object* v_M_13_, lean_object* v_00_u03b1_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_a_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_MulAction_instElemOrbit(v_M_13_, v_00_u03b1_14_, v_inst_15_, v_inst_16_, v_a_17_);
lean_dec(v_a_17_);
lean_dec_ref(v_inst_15_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit(lean_object* v_M_21_, lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_a_25_){
_start:
{
lean_object* v___f_26_; 
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit___redArg___lam__0), 3, 1);
lean_closure_set(v___f_26_, 0, v_inst_24_);
return v___f_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit___boxed(lean_object* v_M_27_, lean_object* v_00_u03b1_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_AddAction_instElemOrbit(v_M_27_, v_00_u03b1_28_, v_inst_29_, v_inst_30_, v_a_31_);
lean_dec(v_a_31_);
lean_dec_ref(v_inst_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizerSubmonoid(lean_object* v_M_33_, lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_a_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_box(0);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizerSubmonoid___boxed(lean_object* v_M_39_, lean_object* v_00_u03b1_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_a_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_MulAction_stabilizerSubmonoid(v_M_39_, v_00_u03b1_40_, v_inst_41_, v_inst_42_, v_a_43_);
lean_dec(v_a_43_);
lean_dec(v_inst_42_);
lean_dec_ref(v_inst_41_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizerAddSubmonoid(lean_object* v_M_45_, lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_a_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_box(0);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizerAddSubmonoid___boxed(lean_object* v_M_51_, lean_object* v_00_u03b1_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_a_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_AddAction_stabilizerAddSubmonoid(v_M_51_, v_00_u03b1_52_, v_inst_53_, v_inst_54_, v_a_55_);
lean_dec(v_a_55_);
lean_dec(v_inst_54_);
lean_dec_ref(v_inst_53_);
return v_res_56_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg(lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_a_59_, lean_object* v_x_60_){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; uint8_t v___x_63_; 
lean_inc(v_a_59_);
v___x_61_ = lean_apply_2(v_inst_57_, v_x_60_, v_a_59_);
v___x_62_ = lean_apply_2(v_inst_58_, v___x_61_, v_a_59_);
v___x_63_ = lean_unbox(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg___boxed(lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_a_66_, lean_object* v_x_67_){
_start:
{
uint8_t v_res_68_; lean_object* v_r_69_; 
v_res_68_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg(v_inst_64_, v_inst_65_, v_a_66_, v_x_67_);
v_r_69_ = lean_box(v_res_68_);
return v_r_69_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1(lean_object* v_M_70_, lean_object* v_00_u03b1_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_a_75_, lean_object* v_x_76_){
_start:
{
uint8_t v___x_77_; 
v___x_77_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg(v_inst_73_, v_inst_74_, v_a_75_, v_x_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___boxed(lean_object* v_M_78_, lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_a_83_, lean_object* v_x_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1(v_M_78_, v_00_u03b1_79_, v_inst_80_, v_inst_81_, v_inst_82_, v_a_83_, v_x_84_);
lean_dec_ref(v_inst_80_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_a_89_, lean_object* v_x_90_){
_start:
{
uint8_t v___x_91_; 
v___x_91_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg(v_inst_87_, v_inst_88_, v_a_89_, v_x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___redArg___boxed(lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_a_94_, lean_object* v_x_95_){
_start:
{
uint8_t v_res_96_; lean_object* v_r_97_; 
v_res_96_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___redArg(v_inst_92_, v_inst_93_, v_a_94_, v_x_95_);
v_r_97_ = lean_box(v_res_96_);
return v_r_97_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq(lean_object* v_M_98_, lean_object* v_00_u03b1_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_a_103_, lean_object* v_x_104_){
_start:
{
uint8_t v___x_105_; 
v___x_105_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___aux__1___redArg(v_inst_101_, v_inst_102_, v_a_103_, v_x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq___boxed(lean_object* v_M_106_, lean_object* v_00_u03b1_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_a_111_, lean_object* v_x_112_){
_start:
{
uint8_t v_res_113_; lean_object* v_r_114_; 
v_res_113_ = lp_mathlib_MulAction_instDecidablePredMemSubmonoidStabilizerSubmonoidOfDecidableEq(v_M_106_, v_00_u03b1_107_, v_inst_108_, v_inst_109_, v_inst_110_, v_a_111_, v_x_112_);
lean_dec_ref(v_inst_108_);
v_r_114_ = lean_box(v_res_113_);
return v_r_114_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg(lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_a_117_, lean_object* v_x_118_){
_start:
{
lean_object* v___x_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
lean_inc(v_a_117_);
v___x_119_ = lean_apply_2(v_inst_115_, v_x_118_, v_a_117_);
v___x_120_ = lean_apply_2(v_inst_116_, v___x_119_, v_a_117_);
v___x_121_ = lean_unbox(v___x_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg___boxed(lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_a_124_, lean_object* v_x_125_){
_start:
{
uint8_t v_res_126_; lean_object* v_r_127_; 
v_res_126_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg(v_inst_122_, v_inst_123_, v_a_124_, v_x_125_);
v_r_127_ = lean_box(v_res_126_);
return v_r_127_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1(lean_object* v_M_128_, lean_object* v_00_u03b1_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_a_133_, lean_object* v_x_134_){
_start:
{
uint8_t v___x_135_; 
v___x_135_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg(v_inst_131_, v_inst_132_, v_a_133_, v_x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___boxed(lean_object* v_M_136_, lean_object* v_00_u03b1_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_a_141_, lean_object* v_x_142_){
_start:
{
uint8_t v_res_143_; lean_object* v_r_144_; 
v_res_143_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1(v_M_136_, v_00_u03b1_137_, v_inst_138_, v_inst_139_, v_inst_140_, v_a_141_, v_x_142_);
lean_dec_ref(v_inst_138_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___redArg(lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_a_147_, lean_object* v_x_148_){
_start:
{
uint8_t v___x_149_; 
v___x_149_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg(v_inst_145_, v_inst_146_, v_a_147_, v_x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___redArg___boxed(lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_a_152_, lean_object* v_x_153_){
_start:
{
uint8_t v_res_154_; lean_object* v_r_155_; 
v_res_154_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___redArg(v_inst_150_, v_inst_151_, v_a_152_, v_x_153_);
v_r_155_ = lean_box(v_res_154_);
return v_r_155_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq(lean_object* v_M_156_, lean_object* v_00_u03b1_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_a_161_, lean_object* v_x_162_){
_start:
{
uint8_t v___x_163_; 
v___x_163_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___aux__1___redArg(v_inst_159_, v_inst_160_, v_a_161_, v_x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq___boxed(lean_object* v_M_164_, lean_object* v_00_u03b1_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_a_169_, lean_object* v_x_170_){
_start:
{
uint8_t v_res_171_; lean_object* v_r_172_; 
v_res_171_ = lp_mathlib_AddAction_instDecidablePredMemAddSubmonoidStabilizerAddSubmonoidOfDecidableEq(v_M_164_, v_00_u03b1_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_a_169_, v_x_170_);
lean_dec_ref(v_inst_166_);
v_r_172_ = lean_box(v_res_171_);
return v_r_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_submonoid(lean_object* v_M_173_, lean_object* v_00_u03b1_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lean_box(0);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_submonoid___boxed(lean_object* v_M_179_, lean_object* v_00_u03b1_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_FixedPoints_submonoid(v_M_179_, v_00_u03b1_180_, v_inst_181_, v_inst_182_, v_inst_183_);
lean_dec(v_inst_183_);
lean_dec_ref(v_inst_182_);
lean_dec_ref(v_inst_181_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_subgroup(lean_object* v_M_185_, lean_object* v_00_u03b1_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lean_box(0);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FixedPoints_subgroup___boxed(lean_object* v_M_191_, lean_object* v_00_u03b1_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_FixedPoints_subgroup(v_M_191_, v_00_u03b1_192_, v_inst_193_, v_inst_194_, v_inst_195_);
lean_dec(v_inst_195_);
lean_dec_ref(v_inst_194_);
lean_dec_ref(v_inst_193_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_orbitRel(lean_object* v_G_197_, lean_object* v_00_u03b1_198_, lean_object* v_inst_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lean_box(0);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_orbitRel___boxed(lean_object* v_G_202_, lean_object* v_00_u03b1_203_, lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_MulAction_orbitRel(v_G_202_, v_00_u03b1_203_, v_inst_204_, v_inst_205_);
lean_dec(v_inst_205_);
lean_dec_ref(v_inst_204_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_orbitRel(lean_object* v_G_207_, lean_object* v_00_u03b1_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lean_box(0);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_orbitRel___boxed(lean_object* v_G_212_, lean_object* v_00_u03b1_213_, lean_object* v_inst_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_AddAction_orbitRel(v_G_212_, v_00_u03b1_213_, v_inst_214_, v_inst_215_);
lean_dec(v_inst_215_);
lean_dec_ref(v_inst_214_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1___redArg___lam__0(lean_object* v_inst_217_, lean_object* v_g_218_, lean_object* v___y_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lean_apply_2(v_inst_217_, v_g_218_, v___y_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1___redArg(lean_object* v_inst_221_){
_start:
{
lean_object* v___f_222_; 
v___f_222_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_222_, 0, v_inst_221_);
return v___f_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1(lean_object* v_G_223_, lean_object* v_00_u03b1_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_x_227_){
_start:
{
lean_object* v___f_228_; 
v___f_228_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_228_, 0, v_inst_226_);
return v___f_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instElemOrbit__1___boxed(lean_object* v_G_229_, lean_object* v_00_u03b1_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_x_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_MulAction_instElemOrbit__1(v_G_229_, v_00_u03b1_230_, v_inst_231_, v_inst_232_, v_x_233_);
lean_dec(v_x_233_);
lean_dec_ref(v_inst_231_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit__1___redArg(lean_object* v_inst_235_){
_start:
{
lean_object* v___f_236_; 
v___f_236_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_236_, 0, v_inst_235_);
return v___f_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit__1(lean_object* v_G_237_, lean_object* v_00_u03b1_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_x_241_){
_start:
{
lean_object* v___f_242_; 
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_instElemOrbit__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_242_, 0, v_inst_240_);
return v___f_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instElemOrbit__1___boxed(lean_object* v_G_243_, lean_object* v_00_u03b1_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_x_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_AddAction_instElemOrbit__1(v_G_243_, v_00_u03b1_244_, v_inst_245_, v_inst_246_, v_x_247_);
lean_dec(v_x_247_);
lean_dec_ref(v_inst_245_);
return v_res_248_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__6(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__5));
v___x_299_ = l_String_toRawSubstring_x27(v___x_298_);
return v___x_299_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__18(void){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_324_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__17));
v___x_325_ = l_String_toRawSubstring_x27(v___x_324_);
return v___x_325_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__36(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_368_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__35));
v___x_369_ = l_String_toRawSubstring_x27(v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1(lean_object* v_x_408_, lean_object* v_a_409_, lean_object* v_a_410_){
_start:
{
lean_object* v___x_411_; uint8_t v___x_412_; 
v___x_411_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__14));
v___x_412_ = l_Lean_Syntax_isOfKind(v_x_408_, v___x_411_);
if (v___x_412_ == 0)
{
lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_413_ = lean_box(1);
v___x_414_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_414_, 0, v___x_413_);
lean_ctor_set(v___x_414_, 1, v_a_410_);
return v___x_414_;
}
else
{
lean_object* v_quotContext_415_; lean_object* v_currMacroScope_416_; lean_object* v_ref_417_; uint8_t v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; 
v_quotContext_415_ = lean_ctor_get(v_a_409_, 1);
v_currMacroScope_416_ = lean_ctor_get(v_a_409_, 2);
v_ref_417_ = lean_ctor_get(v_a_409_, 5);
v___x_418_ = 0;
v___x_419_ = l_Lean_SourceInfo_fromRef(v_ref_417_, v___x_418_);
v___x_420_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4));
v___x_421_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__6, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__6_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__6);
v___x_422_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__9));
lean_inc_n(v_currMacroScope_416_, 3);
lean_inc_n(v_quotContext_415_, 3);
v___x_423_ = l_Lean_addMacroScope(v_quotContext_415_, v___x_422_, v_currMacroScope_416_);
v___x_424_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__14));
lean_inc_n(v___x_419_, 4);
v___x_425_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_425_, 0, v___x_419_);
lean_ctor_set(v___x_425_, 1, v___x_421_);
lean_ctor_set(v___x_425_, 2, v___x_423_);
lean_ctor_set(v___x_425_, 3, v___x_424_);
v___x_426_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__16));
v___x_427_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__18, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__18_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__18);
v___x_428_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__19));
v___x_429_ = l_Lean_addMacroScope(v_quotContext_415_, v___x_428_, v_currMacroScope_416_);
v___x_430_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__34));
v___x_431_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_431_, 0, v___x_419_);
lean_ctor_set(v___x_431_, 1, v___x_427_);
lean_ctor_set(v___x_431_, 2, v___x_429_);
lean_ctor_set(v___x_431_, 3, v___x_430_);
v___x_432_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__36, &lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__36_once, _init_lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__36);
v___x_433_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__37));
v___x_434_ = l_Lean_addMacroScope(v_quotContext_415_, v___x_433_, v_currMacroScope_416_);
v___x_435_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__49));
v___x_436_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_436_, 0, v___x_419_);
lean_ctor_set(v___x_436_, 1, v___x_432_);
lean_ctor_set(v___x_436_, 2, v___x_434_);
lean_ctor_set(v___x_436_, 3, v___x_435_);
v___x_437_ = l_Lean_Syntax_node2(v___x_419_, v___x_426_, v___x_431_, v___x_436_);
v___x_438_ = l_Lean_Syntax_node2(v___x_419_, v___x_420_, v___x_425_, v___x_437_);
v___x_439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_439_, 0, v___x_438_);
lean_ctor_set(v___x_439_, 1, v_a_410_);
return v___x_439_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___boxed(lean_object* v_x_440_, lean_object* v_a_441_, lean_object* v_a_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1(v_x_440_, v_a_441_, v_a_442_);
lean_dec_ref(v_a_441_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1(lean_object* v_x_447_, lean_object* v_a_448_, lean_object* v_a_449_){
_start:
{
lean_object* v___x_450_; uint8_t v___x_451_; 
v___x_450_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__4));
lean_inc(v_x_447_);
v___x_451_ = l_Lean_Syntax_isOfKind(v_x_447_, v___x_450_);
if (v___x_451_ == 0)
{
lean_object* v___x_452_; lean_object* v___x_453_; 
lean_dec(v_x_447_);
v___x_452_ = lean_box(0);
v___x_453_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_453_, 0, v___x_452_);
lean_ctor_set(v___x_453_, 1, v_a_449_);
return v___x_453_;
}
else
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_454_ = lean_unsigned_to_nat(0u);
v___x_455_ = l_Lean_Syntax_getArg(v_x_447_, v___x_454_);
v___x_456_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___closed__1));
lean_inc(v___x_455_);
v___x_457_ = l_Lean_Syntax_isOfKind(v___x_455_, v___x_456_);
if (v___x_457_ == 0)
{
lean_object* v___x_458_; lean_object* v___x_459_; 
lean_dec(v___x_455_);
lean_dec(v_x_447_);
v___x_458_ = lean_box(0);
v___x_459_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_459_, 0, v___x_458_);
lean_ctor_set(v___x_459_, 1, v_a_449_);
return v___x_459_;
}
else
{
lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; uint8_t v___x_463_; 
v___x_460_ = lean_unsigned_to_nat(1u);
v___x_461_ = l_Lean_Syntax_getArg(v_x_447_, v___x_460_);
lean_dec(v_x_447_);
v___x_462_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_461_);
v___x_463_ = l_Lean_Syntax_matchesNull(v___x_461_, v___x_462_);
if (v___x_463_ == 0)
{
lean_object* v___x_464_; lean_object* v___x_465_; 
lean_dec(v___x_461_);
lean_dec(v___x_455_);
v___x_464_ = lean_box(0);
v___x_465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
lean_ctor_set(v___x_465_, 1, v_a_449_);
return v___x_465_;
}
else
{
lean_object* v___x_466_; lean_object* v___x_467_; uint8_t v___x_468_; 
v___x_466_ = l_Lean_Syntax_getArg(v___x_461_, v___x_454_);
v___x_467_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__19));
v___x_468_ = l_Lean_Syntax_matchesIdent(v___x_466_, v___x_467_);
lean_dec(v___x_466_);
if (v___x_468_ == 0)
{
lean_object* v___x_469_; lean_object* v___x_470_; 
lean_dec(v___x_461_);
lean_dec(v___x_455_);
v___x_469_ = lean_box(0);
v___x_470_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_469_);
lean_ctor_set(v___x_470_, 1, v_a_449_);
return v___x_470_;
}
else
{
lean_object* v___x_471_; lean_object* v___x_472_; uint8_t v___x_473_; 
v___x_471_ = l_Lean_Syntax_getArg(v___x_461_, v___x_460_);
lean_dec(v___x_461_);
v___x_472_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______macroRules____private__Mathlib__GroupTheory__GroupAction__Defs__0__MulAction__term_u03a9__1___closed__37));
v___x_473_ = l_Lean_Syntax_matchesIdent(v___x_471_, v___x_472_);
lean_dec(v___x_471_);
if (v___x_473_ == 0)
{
lean_object* v___x_474_; lean_object* v___x_475_; 
lean_dec(v___x_455_);
v___x_474_ = lean_box(0);
v___x_475_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
lean_ctor_set(v___x_475_, 1, v_a_449_);
return v___x_475_;
}
else
{
lean_object* v_ref_476_; uint8_t v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; 
v_ref_476_ = l_Lean_replaceRef(v___x_455_, v_a_448_);
lean_dec(v___x_455_);
v___x_477_ = 0;
v___x_478_ = l_Lean_SourceInfo_fromRef(v_ref_476_, v___x_477_);
lean_dec(v_ref_476_);
v___x_479_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__14));
v___x_480_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction_term_u03a9___closed__15));
lean_inc(v___x_478_);
v___x_481_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_478_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
v___x_482_ = l_Lean_Syntax_node1(v___x_478_, v___x_479_, v___x_481_);
v___x_483_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_483_, 0, v___x_482_);
lean_ctor_set(v___x_483_, 1, v_a_449_);
return v___x_483_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1___boxed(lean_object* v_x_484_, lean_object* v_a_485_, lean_object* v_a_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib___private_Mathlib_GroupTheory_GroupAction_Defs_0__MulAction___aux__Mathlib__GroupTheory__GroupAction__Defs______unexpand__MulAction__orbitRel__Quotient__1(v_x_484_, v_a_485_, v_a_486_);
lean_dec(v_a_485_);
return v_res_487_;
}
}
static lean_object* _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___closed__0(void){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lp_mathlib_Equiv_subtypeEquivRight(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0(lean_object* v_x_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lean_obj_once(&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___closed__0, &lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___closed__0_once, _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___closed__0);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0___boxed(lean_object* v_x_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___lam__0(v_x_491_);
lean_dec(v_x_491_);
return v_res_492_;
}
}
static lean_object* _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__2(void){
_start:
{
lean_object* v___x_496_; lean_object* v___x_497_; 
v___x_496_ = ((lean_object*)(lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__1));
v___x_497_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v___x_496_);
return v___x_497_;
}
}
static lean_object* _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__3(void){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_498_ = lean_obj_once(&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__2, &lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__2_once, _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__2);
v___x_499_ = lp_mathlib_Equiv_symm___redArg(v___x_498_);
return v___x_499_;
}
}
static lean_object* _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__4(void){
_start:
{
lean_object* v___f_500_; lean_object* v___x_501_; 
v___f_500_ = ((lean_object*)(lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__0));
v___x_501_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v___f_500_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_502_ = lean_obj_once(&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__4, &lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__4_once, _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__4);
v___x_503_ = lean_obj_once(&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__3, &lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__3_once, _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__3);
v___x_504_ = lp_mathlib_Equiv_trans___redArg(v___x_503_, v___x_502_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27(lean_object* v_G_505_, lean_object* v_00_u03b1_506_, lean_object* v_inst_507_, lean_object* v_inst_508_){
_start:
{
lean_object* v___x_509_; 
v___x_509_ = lean_obj_once(&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5, &lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5_once, _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___boxed(lean_object* v_G_510_, lean_object* v_00_u03b1_511_, lean_object* v_inst_512_, lean_object* v_inst_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_MulAction_selfEquivSigmaOrbits_x27(v_G_510_, v_00_u03b1_511_, v_inst_512_, v_inst_513_);
lean_dec(v_inst_513_);
lean_dec_ref(v_inst_512_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_selfEquivSigmaOrbits_x27(lean_object* v_G_515_, lean_object* v_00_u03b1_516_, lean_object* v_inst_517_, lean_object* v_inst_518_){
_start:
{
lean_object* v___x_519_; 
v___x_519_ = lean_obj_once(&lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5, &lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5_once, _init_lp_mathlib_MulAction_selfEquivSigmaOrbits_x27___closed__5);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_selfEquivSigmaOrbits_x27___boxed(lean_object* v_G_520_, lean_object* v_00_u03b1_521_, lean_object* v_inst_522_, lean_object* v_inst_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_AddAction_selfEquivSigmaOrbits_x27(v_G_520_, v_00_u03b1_521_, v_inst_522_, v_inst_523_);
lean_dec(v_inst_523_);
lean_dec_ref(v_inst_522_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizer(lean_object* v_G_525_, lean_object* v_00_u03b1_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_a_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lean_box(0);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_stabilizer___boxed(lean_object* v_G_531_, lean_object* v_00_u03b1_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_a_535_){
_start:
{
lean_object* v_res_536_; 
v_res_536_ = lp_mathlib_MulAction_stabilizer(v_G_531_, v_00_u03b1_532_, v_inst_533_, v_inst_534_, v_a_535_);
lean_dec(v_a_535_);
lean_dec(v_inst_534_);
lean_dec_ref(v_inst_533_);
return v_res_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizer(lean_object* v_G_537_, lean_object* v_00_u03b1_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_a_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lean_box(0);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_stabilizer___boxed(lean_object* v_G_543_, lean_object* v_00_u03b1_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_a_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_AddAction_stabilizer(v_G_543_, v_00_u03b1_544_, v_inst_545_, v_inst_546_, v_a_547_);
lean_dec(v_a_547_);
lean_dec(v_inst_546_);
lean_dec_ref(v_inst_545_);
return v_res_548_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg(lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_a_551_, lean_object* v_x_552_){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; uint8_t v___x_555_; 
lean_inc(v_a_551_);
v___x_553_ = lean_apply_2(v_inst_549_, v_x_552_, v_a_551_);
v___x_554_ = lean_apply_2(v_inst_550_, v___x_553_, v_a_551_);
v___x_555_ = lean_unbox(v___x_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg___boxed(lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_a_558_, lean_object* v_x_559_){
_start:
{
uint8_t v_res_560_; lean_object* v_r_561_; 
v_res_560_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_556_, v_inst_557_, v_a_558_, v_x_559_);
v_r_561_ = lean_box(v_res_560_);
return v_r_561_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1(lean_object* v_G_562_, lean_object* v_00_u03b1_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_a_567_, lean_object* v_x_568_){
_start:
{
uint8_t v___x_569_; 
v___x_569_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_565_, v_inst_566_, v_a_567_, v_x_568_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___boxed(lean_object* v_G_570_, lean_object* v_00_u03b1_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_a_575_, lean_object* v_x_576_){
_start:
{
uint8_t v_res_577_; lean_object* v_r_578_; 
v_res_577_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1(v_G_570_, v_00_u03b1_571_, v_inst_572_, v_inst_573_, v_inst_574_, v_a_575_, v_x_576_);
lean_dec_ref(v_inst_572_);
v_r_578_ = lean_box(v_res_577_);
return v_r_578_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___redArg(lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_a_581_, lean_object* v_x_582_){
_start:
{
uint8_t v___x_583_; 
v___x_583_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_579_, v_inst_580_, v_a_581_, v_x_582_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___redArg___boxed(lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_a_586_, lean_object* v_x_587_){
_start:
{
uint8_t v_res_588_; lean_object* v_r_589_; 
v_res_588_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___redArg(v_inst_584_, v_inst_585_, v_a_586_, v_x_587_);
v_r_589_ = lean_box(v_res_588_);
return v_r_589_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq(lean_object* v_G_590_, lean_object* v_00_u03b1_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_a_595_, lean_object* v_x_596_){
_start:
{
uint8_t v___x_597_; 
v___x_597_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_593_, v_inst_594_, v_a_595_, v_x_596_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq___boxed(lean_object* v_G_598_, lean_object* v_00_u03b1_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_a_603_, lean_object* v_x_604_){
_start:
{
uint8_t v_res_605_; lean_object* v_r_606_; 
v_res_605_ = lp_mathlib_MulAction_instDecidablePredMemSubgroupStabilizerOfDecidableEq(v_G_598_, v_00_u03b1_599_, v_inst_600_, v_inst_601_, v_inst_602_, v_a_603_, v_x_604_);
lean_dec_ref(v_inst_600_);
v_r_606_ = lean_box(v_res_605_);
return v_r_606_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg(lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_a_609_, lean_object* v_x_610_){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; uint8_t v___x_613_; 
lean_inc(v_a_609_);
v___x_611_ = lean_apply_2(v_inst_607_, v_x_610_, v_a_609_);
v___x_612_ = lean_apply_2(v_inst_608_, v___x_611_, v_a_609_);
v___x_613_ = lean_unbox(v___x_612_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg___boxed(lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_a_616_, lean_object* v_x_617_){
_start:
{
uint8_t v_res_618_; lean_object* v_r_619_; 
v_res_618_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_614_, v_inst_615_, v_a_616_, v_x_617_);
v_r_619_ = lean_box(v_res_618_);
return v_r_619_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1(lean_object* v_G_620_, lean_object* v_00_u03b1_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_a_625_, lean_object* v_x_626_){
_start:
{
uint8_t v___x_627_; 
v___x_627_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_623_, v_inst_624_, v_a_625_, v_x_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___boxed(lean_object* v_G_628_, lean_object* v_00_u03b1_629_, lean_object* v_inst_630_, lean_object* v_inst_631_, lean_object* v_inst_632_, lean_object* v_a_633_, lean_object* v_x_634_){
_start:
{
uint8_t v_res_635_; lean_object* v_r_636_; 
v_res_635_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1(v_G_628_, v_00_u03b1_629_, v_inst_630_, v_inst_631_, v_inst_632_, v_a_633_, v_x_634_);
lean_dec_ref(v_inst_630_);
v_r_636_ = lean_box(v_res_635_);
return v_r_636_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___redArg(lean_object* v_inst_637_, lean_object* v_inst_638_, lean_object* v_a_639_, lean_object* v_x_640_){
_start:
{
uint8_t v___x_641_; 
v___x_641_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_637_, v_inst_638_, v_a_639_, v_x_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___redArg___boxed(lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_a_644_, lean_object* v_x_645_){
_start:
{
uint8_t v_res_646_; lean_object* v_r_647_; 
v_res_646_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___redArg(v_inst_642_, v_inst_643_, v_a_644_, v_x_645_);
v_r_647_ = lean_box(v_res_646_);
return v_r_647_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq(lean_object* v_G_648_, lean_object* v_00_u03b1_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_a_653_, lean_object* v_x_654_){
_start:
{
uint8_t v___x_655_; 
v___x_655_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___aux__1___redArg(v_inst_651_, v_inst_652_, v_a_653_, v_x_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq___boxed(lean_object* v_G_656_, lean_object* v_00_u03b1_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_inst_660_, lean_object* v_a_661_, lean_object* v_x_662_){
_start:
{
uint8_t v_res_663_; lean_object* v_r_664_; 
v_res_663_ = lp_mathlib_AddAction_instDecidablePredMemAddSubgroupStabilizerOfDecidableEq(v_G_656_, v_00_u03b1_657_, v_inst_658_, v_inst_659_, v_inst_660_, v_a_661_, v_x_662_);
lean_dec_ref(v_inst_658_);
v_r_664_ = lean_box(v_res_663_);
return v_r_664_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
