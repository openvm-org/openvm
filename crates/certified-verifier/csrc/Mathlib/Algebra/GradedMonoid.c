// Lean compiler output
// Module: Mathlib.Algebra.GradedMonoid
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.List.Lemmas public import Mathlib.Algebra.Group.Action.Hom public import Mathlib.Algebra.Group.Submonoid.Defs public import Mathlib.Data.List.FinRange public import Mathlib.Data.SetLike.Basic public import Mathlib.Data.Sigma.Basic public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_foldrRecOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mk(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GOne_toOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GOne_toOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "GradedMonoid"};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0_value;
static const lean_string_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "tacticApply_gmonoid_gnpowRec_zero_tac"};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__1 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__1_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 131, 118, 169, 110, 89, 176, 172)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__1_value),LEAN_SCALAR_PTR_LITERAL(190, 237, 103, 162, 137, 65, 232, 140)}};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2_value;
static const lean_string_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "apply_gmonoid_gnpowRec_zero_tac"};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__3 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__3_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__4 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__4_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__4_value)}};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__5 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__5_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__0 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__0_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__1 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__1_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__2 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__2_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__3 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__3_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "GMonoid.gnpowRec_zero"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__5 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__5_value;
static lean_once_cell_t lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__6;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "GMonoid"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "gnpowRec_zero"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__8 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__8_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(130, 156, 101, 118, 17, 164, 157, 227)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(127, 173, 249, 152, 206, 72, 217, 140)}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__9 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__9_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 131, 118, 169, 110, 89, 176, 172)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(23, 29, 204, 206, 64, 236, 197, 83)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(14, 36, 29, 237, 53, 46, 218, 76)}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__11 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__11_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__12 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "tacticApply_gmonoid_gnpowRec_succ_tac"};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__0 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__0_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 131, 118, 169, 110, 89, 176, 172)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(15, 129, 146, 124, 205, 53, 167, 247)}};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1_value;
static const lean_string_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "apply_gmonoid_gnpowRec_succ_tac"};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__2 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__2_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__3 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__3_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__3_value)}};
static const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__4 = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac = (const lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__4_value;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "GMonoid.gnpowRec_succ"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__0 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__0_value;
static lean_once_cell_t lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__1;
static const lean_string_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "gnpowRec_succ"};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__2 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__2_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(130, 156, 101, 118, 17, 164, 157, 227)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(162, 207, 37, 100, 96, 14, 221, 227)}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__3 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__3_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 131, 118, 169, 110, 89, 176, 172)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(23, 29, 204, 206, 64, 236, 197, 83)}};
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 16, 19, 200, 25, 78, 131, 103)}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__5 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__5_value;
static const lean_ctor_object lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__6 = (const lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__0 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__0_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value_aux_1),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value_aux_2),((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1_value;
static const lean_array_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__3 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4_value;
static const lean_string_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__5 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__5_value;
static const lean_ctor_object lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__6 = (const lean_object*)&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__6_value;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__7;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__8;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__9;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__10;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__11;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__13;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__14;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__0;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__1;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__2;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__3;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__4;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__5;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__6;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__7;
static lean_once_cell_t lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GCommMonoid_toCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GCommMonoid_toCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_smul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dProdIndex___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dProdIndex___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dProdIndex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dProd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_One_gOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_One_gOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_One_gOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_One_gOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_gCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_gCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_gCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_submonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_submonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_homogeneousSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_homogeneousSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault___aux__1___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v_inst_1_);
lean_ctor_set(v___x_3_, 1, v_inst_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault___aux__1(lean_object* v_00_u03b9_4_, lean_object* v_A_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v_inst_6_);
lean_ctor_set(v___x_8_, 1, v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault___redArg(lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v_inst_9_);
lean_ctor_set(v___x_11_, 1, v_inst_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instInhabitedOfDefault(lean_object* v_00_u03b9_12_, lean_object* v_A_13_, lean_object* v_inst_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_16_, 0, v_inst_14_);
lean_ctor_set(v___x_16_, 1, v_inst_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mk___redArg(lean_object* v_fst_17_, lean_object* v_snd_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_19_, 0, v_fst_17_);
lean_ctor_set(v___x_19_, 1, v_snd_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mk(lean_object* v_00_u03b9_20_, lean_object* v_A_21_, lean_object* v_fst_22_, lean_object* v_snd_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_24_, 0, v_fst_22_);
lean_ctor_set(v___x_24_, 1, v_snd_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instSMul___redArg___lam__0(lean_object* v_inst_25_, lean_object* v_r_26_, lean_object* v_g_27_){
_start:
{
lean_object* v_fst_28_; lean_object* v_snd_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_37_; 
v_fst_28_ = lean_ctor_get(v_g_27_, 0);
v_snd_29_ = lean_ctor_get(v_g_27_, 1);
v_isSharedCheck_37_ = !lean_is_exclusive(v_g_27_);
if (v_isSharedCheck_37_ == 0)
{
v___x_31_ = v_g_27_;
v_isShared_32_ = v_isSharedCheck_37_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_snd_29_);
lean_inc(v_fst_28_);
lean_dec(v_g_27_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_37_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
lean_object* v___x_33_; lean_object* v___x_35_; 
lean_inc(v_fst_28_);
v___x_33_ = lean_apply_3(v_inst_25_, v_fst_28_, v_r_26_, v_snd_29_);
if (v_isShared_32_ == 0)
{
lean_ctor_set(v___x_31_, 1, v___x_33_);
v___x_35_ = v___x_31_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_fst_28_);
lean_ctor_set(v_reuseFailAlloc_36_, 1, v___x_33_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instSMul___redArg(lean_object* v_inst_38_){
_start:
{
lean_object* v___f_39_; 
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_39_, 0, v_inst_38_);
return v___f_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instSMul(lean_object* v_00_u03b9_40_, lean_object* v_00_u03b1_41_, lean_object* v_A_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; 
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_44_, 0, v_inst_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction___redArg___lam__0(lean_object* v_inst_45_, lean_object* v_i_46_, lean_object* v___y_47_, lean_object* v___y_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_apply_3(v_inst_45_, v_i_46_, v___y_47_, v___y_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction___redArg(lean_object* v_inst_50_){
_start:
{
lean_object* v___f_51_; lean_object* v___f_52_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_instMulAction___redArg___lam__0), 4, 1);
lean_closure_set(v___f_51_, 0, v_inst_50_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_52_, 0, v___f_51_);
return v___f_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction(lean_object* v_00_u03b9_53_, lean_object* v_00_u03b1_54_, lean_object* v_A_55_, lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_GradedMonoid_instMulAction___redArg(v_inst_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instMulAction___boxed(lean_object* v_00_u03b9_59_, lean_object* v_00_u03b1_60_, lean_object* v_A_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_GradedMonoid_instMulAction(v_00_u03b9_59_, v_00_u03b1_60_, v_A_61_, v_inst_62_, v_inst_63_);
lean_dec_ref(v_inst_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GOne_toOne___redArg(lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_67_, 0, v_inst_65_);
lean_ctor_set(v___x_67_, 1, v_inst_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GOne_toOne(lean_object* v_00_u03b9_68_, lean_object* v_A_69_, lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v_inst_70_);
lean_ctor_set(v___x_72_, 1, v_inst_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toMul___redArg___lam__0(lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_x_75_, lean_object* v_y_76_){
_start:
{
lean_object* v_fst_77_; lean_object* v_snd_78_; lean_object* v_fst_79_; lean_object* v_snd_80_; lean_object* v___x_82_; uint8_t v_isShared_83_; uint8_t v_isSharedCheck_89_; 
v_fst_77_ = lean_ctor_get(v_x_75_, 0);
lean_inc(v_fst_77_);
v_snd_78_ = lean_ctor_get(v_x_75_, 1);
lean_inc(v_snd_78_);
lean_dec_ref(v_x_75_);
v_fst_79_ = lean_ctor_get(v_y_76_, 0);
v_snd_80_ = lean_ctor_get(v_y_76_, 1);
v_isSharedCheck_89_ = !lean_is_exclusive(v_y_76_);
if (v_isSharedCheck_89_ == 0)
{
v___x_82_ = v_y_76_;
v_isShared_83_ = v_isSharedCheck_89_;
goto v_resetjp_81_;
}
else
{
lean_inc(v_snd_80_);
lean_inc(v_fst_79_);
lean_dec(v_y_76_);
v___x_82_ = lean_box(0);
v_isShared_83_ = v_isSharedCheck_89_;
goto v_resetjp_81_;
}
v_resetjp_81_:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_87_; 
lean_inc(v_fst_79_);
lean_inc(v_fst_77_);
v___x_84_ = lean_apply_2(v_inst_73_, v_fst_77_, v_fst_79_);
v___x_85_ = lean_apply_4(v_inst_74_, v_fst_77_, v_fst_79_, v_snd_78_, v_snd_80_);
if (v_isShared_83_ == 0)
{
lean_ctor_set(v___x_82_, 1, v___x_85_);
lean_ctor_set(v___x_82_, 0, v___x_84_);
v___x_87_ = v___x_82_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v___x_84_);
lean_ctor_set(v_reuseFailAlloc_88_, 1, v___x_85_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toMul___redArg(lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___f_92_; 
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMul_toMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_92_, 0, v_inst_90_);
lean_closure_set(v___f_92_, 1, v_inst_91_);
return v___f_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMul_toMul(lean_object* v_00_u03b9_93_, lean_object* v_A_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___f_97_; 
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMul_toMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_97_, 0, v_inst_95_);
lean_closure_set(v___f_97_, 1, v_inst_96_);
return v___f_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg(lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_x_101_, lean_object* v_x_102_, lean_object* v_x_103_){
_start:
{
lean_object* v_toNSMul_104_; lean_object* v_zero_105_; uint8_t v_isZero_106_; 
v_toNSMul_104_ = lean_ctor_get(v_inst_98_, 2);
v_zero_105_ = lean_unsigned_to_nat(0u);
v_isZero_106_ = lean_nat_dec_eq(v_x_101_, v_zero_105_);
if (v_isZero_106_ == 1)
{
lean_dec(v_x_103_);
lean_dec(v_x_102_);
lean_dec(v_inst_99_);
lean_dec_ref(v_inst_98_);
lean_inc(v_inst_100_);
return v_inst_100_;
}
else
{
lean_object* v_one_107_; lean_object* v_n_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_one_107_ = lean_unsigned_to_nat(1u);
v_n_108_ = lean_nat_sub(v_x_101_, v_one_107_);
lean_inc(v_toNSMul_104_);
lean_inc_n(v_x_102_, 2);
lean_inc(v_n_108_);
v___x_109_ = lean_apply_2(v_toNSMul_104_, v_n_108_, v_x_102_);
lean_inc(v_x_103_);
lean_inc(v_inst_99_);
v___x_110_ = lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg(v_inst_98_, v_inst_99_, v_inst_100_, v_n_108_, v_x_102_, v_x_103_);
lean_dec(v_n_108_);
v___x_111_ = lean_apply_4(v_inst_99_, v___x_109_, v_x_102_, v___x_110_, v_x_103_);
return v___x_111_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg___boxed(lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_x_115_, lean_object* v_x_116_, lean_object* v_x_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg(v_inst_112_, v_inst_113_, v_inst_114_, v_x_115_, v_x_116_, v_x_117_);
lean_dec(v_x_115_);
lean_dec(v_inst_114_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec(lean_object* v_00_u03b9_119_, lean_object* v_A_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_x_124_, lean_object* v_x_125_, lean_object* v_x_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_GradedMonoid_GMonoid_gnpowRec___redArg(v_inst_121_, v_inst_122_, v_inst_123_, v_x_124_, v_x_125_, v_x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_gnpowRec___boxed(lean_object* v_00_u03b9_128_, lean_object* v_A_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_x_133_, lean_object* v_x_134_, lean_object* v_x_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_GradedMonoid_GMonoid_gnpowRec(v_00_u03b9_128_, v_A_129_, v_inst_130_, v_inst_131_, v_inst_132_, v_x_133_, v_x_134_, v_x_135_);
lean_dec(v_x_133_);
lean_dec(v_inst_132_);
return v_res_136_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__6(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__5));
v___x_162_ = l_String_toRawSubstring_x27(v___x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1(lean_object* v_x_178_, lean_object* v_a_179_, lean_object* v_a_180_){
_start:
{
lean_object* v___x_181_; uint8_t v___x_182_; 
v___x_181_ = ((lean_object*)(lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2));
v___x_182_ = l_Lean_Syntax_isOfKind(v_x_178_, v___x_181_);
if (v___x_182_ == 0)
{
lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_183_ = lean_box(1);
v___x_184_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v_a_180_);
return v___x_184_;
}
else
{
lean_object* v_quotContext_185_; lean_object* v_currMacroScope_186_; lean_object* v_ref_187_; uint8_t v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v_quotContext_185_ = lean_ctor_get(v_a_179_, 1);
v_currMacroScope_186_ = lean_ctor_get(v_a_179_, 2);
v_ref_187_ = lean_ctor_get(v_a_179_, 5);
v___x_188_ = 0;
v___x_189_ = l_Lean_SourceInfo_fromRef(v_ref_187_, v___x_188_);
v___x_190_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__3));
v___x_191_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4));
lean_inc_n(v___x_189_, 2);
v___x_192_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_192_, 0, v___x_189_);
lean_ctor_set(v___x_192_, 1, v___x_190_);
v___x_193_ = lean_obj_once(&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__6, &lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__6_once, _init_lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__6);
v___x_194_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__9));
lean_inc(v_currMacroScope_186_);
lean_inc(v_quotContext_185_);
v___x_195_ = l_Lean_addMacroScope(v_quotContext_185_, v___x_194_, v_currMacroScope_186_);
v___x_196_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__12));
v___x_197_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_197_, 0, v___x_189_);
lean_ctor_set(v___x_197_, 1, v___x_193_);
lean_ctor_set(v___x_197_, 2, v___x_195_);
lean_ctor_set(v___x_197_, 3, v___x_196_);
v___x_198_ = l_Lean_Syntax_node2(v___x_189_, v___x_191_, v___x_192_, v___x_197_);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
lean_ctor_set(v___x_199_, 1, v_a_180_);
return v___x_199_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___boxed(lean_object* v_x_200_, lean_object* v_a_201_, lean_object* v_a_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1(v_x_200_, v_a_201_, v_a_202_);
lean_dec_ref(v_a_201_);
return v_res_203_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__1(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__0));
v___x_219_ = l_String_toRawSubstring_x27(v___x_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1(lean_object* v_x_234_, lean_object* v_a_235_, lean_object* v_a_236_){
_start:
{
lean_object* v___x_237_; uint8_t v___x_238_; 
v___x_237_ = ((lean_object*)(lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1));
v___x_238_ = l_Lean_Syntax_isOfKind(v_x_234_, v___x_237_);
if (v___x_238_ == 0)
{
lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_239_ = lean_box(1);
v___x_240_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
lean_ctor_set(v___x_240_, 1, v_a_236_);
return v___x_240_;
}
else
{
lean_object* v_quotContext_241_; lean_object* v_currMacroScope_242_; lean_object* v_ref_243_; uint8_t v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v_quotContext_241_ = lean_ctor_get(v_a_235_, 1);
v_currMacroScope_242_ = lean_ctor_get(v_a_235_, 2);
v_ref_243_ = lean_ctor_get(v_a_235_, 5);
v___x_244_ = 0;
v___x_245_ = l_Lean_SourceInfo_fromRef(v_ref_243_, v___x_244_);
v___x_246_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__3));
v___x_247_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__zero__tac__1___closed__4));
lean_inc_n(v___x_245_, 2);
v___x_248_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_245_);
lean_ctor_set(v___x_248_, 1, v___x_246_);
v___x_249_ = lean_obj_once(&lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__1, &lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__1_once, _init_lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__1);
v___x_250_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__3));
lean_inc(v_currMacroScope_242_);
lean_inc(v_quotContext_241_);
v___x_251_ = l_Lean_addMacroScope(v_quotContext_241_, v___x_250_, v_currMacroScope_242_);
v___x_252_ = ((lean_object*)(lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___closed__6));
v___x_253_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_253_, 0, v___x_245_);
lean_ctor_set(v___x_253_, 1, v___x_249_);
lean_ctor_set(v___x_253_, 2, v___x_251_);
lean_ctor_set(v___x_253_, 3, v___x_252_);
v___x_254_ = l_Lean_Syntax_node2(v___x_245_, v___x_247_, v___x_248_, v___x_253_);
v___x_255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v_a_236_);
return v___x_255_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1___boxed(lean_object* v_x_256_, lean_object* v_a_257_, lean_object* v_a_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_GradedMonoid___aux__Mathlib__Algebra__GradedMonoid______macroRules__GradedMonoid__tacticApply__gmonoid__gnpowRec__succ__tac__1(v_x_256_, v_a_257_, v_a_258_);
lean_dec_ref(v_a_257_);
return v_res_259_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__7(void){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = ((lean_object*)(lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__3));
v___x_278_ = l_Lean_mkAtom(v___x_277_);
return v___x_278_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__8(void){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_279_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__7, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__7_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__7);
v___x_280_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_281_ = lean_array_push(v___x_280_, v___x_279_);
return v___x_281_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__9(void){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_282_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__8, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__8_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__8);
v___x_283_ = ((lean_object*)(lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__zero__tac___closed__2));
v___x_284_ = lean_box(2);
v___x_285_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_285_, 0, v___x_284_);
lean_ctor_set(v___x_285_, 1, v___x_283_);
lean_ctor_set(v___x_285_, 2, v___x_282_);
return v___x_285_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__10(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_286_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__9, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__9_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__9);
v___x_287_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_288_ = lean_array_push(v___x_287_, v___x_286_);
return v___x_288_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__11(void){
_start:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_289_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__10, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__10_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__10);
v___x_290_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__6));
v___x_291_ = lean_box(2);
v___x_292_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_292_, 0, v___x_291_);
lean_ctor_set(v___x_292_, 1, v___x_290_);
lean_ctor_set(v___x_292_, 2, v___x_289_);
return v___x_292_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__12(void){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_293_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__11, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__11_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__11);
v___x_294_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_295_ = lean_array_push(v___x_294_, v___x_293_);
return v___x_295_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__13(void){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_296_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__12, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__12_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__12);
v___x_297_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4));
v___x_298_ = lean_box(2);
v___x_299_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v___x_297_);
lean_ctor_set(v___x_299_, 2, v___x_296_);
return v___x_299_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__14(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_300_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__13, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__13_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__13);
v___x_301_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_302_ = lean_array_push(v___x_301_, v___x_300_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__15(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_303_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__14, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__14_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__14);
v___x_304_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1));
v___x_305_ = lean_box(2);
v___x_306_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_306_, 0, v___x_305_);
lean_ctor_set(v___x_306_, 1, v___x_304_);
lean_ctor_set(v___x_306_, 2, v___x_303_);
return v___x_306_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam(void){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__15, &lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__15_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__15);
return v___x_307_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__0(void){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_308_ = ((lean_object*)(lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__2));
v___x_309_ = l_Lean_mkAtom(v___x_308_);
return v___x_309_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__1(void){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_310_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__0, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__0_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__0);
v___x_311_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_312_ = lean_array_push(v___x_311_, v___x_310_);
return v___x_312_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__2(void){
_start:
{
lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_313_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__1, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__1_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__1);
v___x_314_ = ((lean_object*)(lp_mathlib_GradedMonoid_tacticApply__gmonoid__gnpowRec__succ__tac___closed__1));
v___x_315_ = lean_box(2);
v___x_316_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v___x_314_);
lean_ctor_set(v___x_316_, 2, v___x_313_);
return v___x_316_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__3(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_317_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__2, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__2_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__2);
v___x_318_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_319_ = lean_array_push(v___x_318_, v___x_317_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__4(void){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v___x_320_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__3, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__3_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__3);
v___x_321_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__6));
v___x_322_ = lean_box(2);
v___x_323_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_323_, 0, v___x_322_);
lean_ctor_set(v___x_323_, 1, v___x_321_);
lean_ctor_set(v___x_323_, 2, v___x_320_);
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__5(void){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_324_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__4, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__4_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__4);
v___x_325_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_326_ = lean_array_push(v___x_325_, v___x_324_);
return v___x_326_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__6(void){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_327_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__5, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__5_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__5);
v___x_328_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__4));
v___x_329_ = lean_box(2);
v___x_330_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
lean_ctor_set(v___x_330_, 1, v___x_328_);
lean_ctor_set(v___x_330_, 2, v___x_327_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__7(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_331_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__6, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__6_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__6);
v___x_332_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__2));
v___x_333_ = lean_array_push(v___x_332_, v___x_331_);
return v___x_333_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__8(void){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_334_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__7, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__7_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__7);
v___x_335_ = ((lean_object*)(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam___closed__1));
v___x_336_ = lean_box(2);
v___x_337_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_337_, 0, v___x_336_);
lean_ctor_set(v___x_337_, 1, v___x_335_);
lean_ctor_set(v___x_337_, 2, v___x_334_);
return v___x_337_;
}
}
static lean_object* _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam(void){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lean_obj_once(&lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__8, &lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__8_once, _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam___closed__8);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg___lam__0(lean_object* v_toNSMul_339_, lean_object* v_gnpow_340_, lean_object* v_n_341_, lean_object* v_a_342_){
_start:
{
lean_object* v_fst_343_; lean_object* v_snd_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_353_; 
v_fst_343_ = lean_ctor_get(v_a_342_, 0);
v_snd_344_ = lean_ctor_get(v_a_342_, 1);
v_isSharedCheck_353_ = !lean_is_exclusive(v_a_342_);
if (v_isSharedCheck_353_ == 0)
{
v___x_346_ = v_a_342_;
v_isShared_347_ = v_isSharedCheck_353_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_snd_344_);
lean_inc(v_fst_343_);
lean_dec(v_a_342_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_353_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_351_; 
lean_inc(v_fst_343_);
lean_inc(v_n_341_);
v___x_348_ = lean_apply_2(v_toNSMul_339_, v_n_341_, v_fst_343_);
v___x_349_ = lean_apply_3(v_gnpow_340_, v_n_341_, v_fst_343_, v_snd_344_);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 1, v___x_349_);
lean_ctor_set(v___x_346_, 0, v___x_348_);
v___x_351_ = v___x_346_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v___x_349_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg(lean_object* v_inst_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v_toZero_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_380_; 
v___x_356_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_354_);
v___x_357_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_356_);
v_toZero_358_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_380_ == 0)
{
lean_object* v_unused_381_; 
v_unused_381_ = lean_ctor_get(v___x_357_, 1);
lean_dec(v_unused_381_);
v___x_360_ = v___x_357_;
v_isShared_361_ = v_isSharedCheck_380_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_toZero_358_);
lean_dec(v___x_357_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_380_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v_toGMul_362_; lean_object* v_toGOne_363_; lean_object* v_gnpow_364_; lean_object* v_toAdd_365_; lean_object* v_toNSMul_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_378_; 
v_toGMul_362_ = lean_ctor_get(v_inst_355_, 0);
lean_inc(v_toGMul_362_);
v_toGOne_363_ = lean_ctor_get(v_inst_355_, 1);
lean_inc(v_toGOne_363_);
v_gnpow_364_ = lean_ctor_get(v_inst_355_, 2);
lean_inc(v_gnpow_364_);
lean_dec_ref(v_inst_355_);
v_toAdd_365_ = lean_ctor_get(v_inst_354_, 1);
v_toNSMul_366_ = lean_ctor_get(v_inst_354_, 2);
v_isSharedCheck_378_ = !lean_is_exclusive(v_inst_354_);
if (v_isSharedCheck_378_ == 0)
{
lean_object* v_unused_379_; 
v_unused_379_ = lean_ctor_get(v_inst_354_, 0);
lean_dec(v_unused_379_);
v___x_368_ = v_inst_354_;
v_isShared_369_ = v_isSharedCheck_378_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_toNSMul_366_);
lean_inc(v_toAdd_365_);
lean_dec(v_inst_354_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_378_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_371_; 
if (v_isShared_361_ == 0)
{
lean_ctor_set(v___x_360_, 1, v_toGOne_363_);
v___x_371_ = v___x_360_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_toZero_358_);
lean_ctor_set(v_reuseFailAlloc_377_, 1, v_toGOne_363_);
v___x_371_ = v_reuseFailAlloc_377_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
lean_object* v___f_372_; lean_object* v___f_373_; lean_object* v___x_375_; 
v___f_372_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg___lam__0), 4, 2);
lean_closure_set(v___f_372_, 0, v_toNSMul_366_);
lean_closure_set(v___f_372_, 1, v_gnpow_364_);
v___f_373_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GMul_toMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_373_, 0, v_toAdd_365_);
lean_closure_set(v___f_373_, 1, v_toGMul_362_);
if (v_isShared_369_ == 0)
{
lean_ctor_set(v___x_368_, 2, v___f_372_);
lean_ctor_set(v___x_368_, 1, v___f_373_);
lean_ctor_set(v___x_368_, 0, v___x_371_);
v___x_375_ = v___x_368_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v___x_371_);
lean_ctor_set(v_reuseFailAlloc_376_, 1, v___f_373_);
lean_ctor_set(v_reuseFailAlloc_376_, 2, v___f_372_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GMonoid_toMonoid(lean_object* v_00_u03b9_382_, lean_object* v_A_383_, lean_object* v_inst_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg(v_inst_384_, v_inst_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GCommMonoid_toCommMonoid___redArg(lean_object* v_inst_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg(v_inst_387_, v_inst_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GCommMonoid_toCommMonoid(lean_object* v_00_u03b9_390_, lean_object* v_A_391_, lean_object* v_inst_392_, lean_object* v_inst_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_mathlib_GradedMonoid_GMonoid_toMonoid___redArg(v_inst_392_, v_inst_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one___redArg(lean_object* v_inst_395_){
_start:
{
lean_inc(v_inst_395_);
return v_inst_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one___redArg___boxed(lean_object* v_inst_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_GradedMonoid_GradeZero_one___redArg(v_inst_396_);
lean_dec(v_inst_396_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one(lean_object* v_00_u03b9_398_, lean_object* v_A_399_, lean_object* v_inst_400_, lean_object* v_inst_401_){
_start:
{
lean_inc(v_inst_401_);
return v_inst_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_one___boxed(lean_object* v_00_u03b9_402_, lean_object* v_A_403_, lean_object* v_inst_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_mathlib_GradedMonoid_GradeZero_one(v_00_u03b9_402_, v_A_403_, v_inst_404_, v_inst_405_);
lean_dec(v_inst_405_);
lean_dec(v_inst_404_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_smul___redArg___lam__0(lean_object* v_inst_407_, lean_object* v_toZero_408_, lean_object* v_i_409_, lean_object* v_x_410_, lean_object* v_y_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lean_apply_4(v_inst_407_, v_toZero_408_, v_i_409_, v_x_410_, v_y_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_smul___redArg(lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_i_415_){
_start:
{
lean_object* v___x_416_; lean_object* v_toZero_417_; lean_object* v___f_418_; 
v___x_416_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_413_);
v_toZero_417_ = lean_ctor_get(v___x_416_, 0);
lean_inc(v_toZero_417_);
lean_dec_ref(v___x_416_);
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GradeZero_smul___redArg___lam__0), 5, 3);
lean_closure_set(v___f_418_, 0, v_inst_414_);
lean_closure_set(v___f_418_, 1, v_toZero_417_);
lean_closure_set(v___f_418_, 2, v_i_415_);
return v___f_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_smul(lean_object* v_00_u03b9_419_, lean_object* v_A_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_i_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(v_inst_421_, v_inst_422_, v_i_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mul___redArg___lam__0(lean_object* v_inst_425_, lean_object* v_toZero_426_, lean_object* v_x1_427_, lean_object* v_x2_428_){
_start:
{
lean_object* v___x_429_; 
lean_inc(v_toZero_426_);
v___x_429_ = lean_apply_4(v_inst_425_, v_toZero_426_, v_toZero_426_, v_x1_427_, v_x2_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mul___redArg(lean_object* v_inst_430_, lean_object* v_inst_431_){
_start:
{
lean_object* v___x_432_; lean_object* v_toZero_433_; lean_object* v___f_434_; 
v___x_432_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_430_);
v_toZero_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc(v_toZero_433_);
lean_dec_ref(v___x_432_);
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GradeZero_mul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_434_, 0, v_inst_431_);
lean_closure_set(v___f_434_, 1, v_toZero_433_);
return v___f_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mul(lean_object* v_00_u03b9_435_, lean_object* v_A_436_, lean_object* v_inst_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v_inst_437_, v_inst_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___redArg___lam__0(lean_object* v_inst_440_, lean_object* v_toZero_441_, lean_object* v_x_442_, lean_object* v_n_443_){
_start:
{
lean_object* v_gnpow_444_; lean_object* v___x_445_; 
v_gnpow_444_ = lean_ctor_get(v_inst_440_, 2);
lean_inc(v_gnpow_444_);
lean_dec_ref(v_inst_440_);
v___x_445_ = lean_apply_3(v_gnpow_444_, v_n_443_, v_toZero_441_, v_x_442_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___redArg(lean_object* v_inst_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v_toZero_450_; lean_object* v___f_451_; 
v___x_448_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_446_);
v___x_449_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_448_);
v_toZero_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_toZero_450_);
lean_dec_ref(v___x_449_);
v___f_451_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_instNatPowOfNat___redArg___lam__0), 4, 2);
lean_closure_set(v___f_451_, 0, v_inst_447_);
lean_closure_set(v___f_451_, 1, v_toZero_450_);
return v___f_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___redArg___boxed(lean_object* v_inst_452_, lean_object* v_inst_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_mathlib_GradedMonoid_instNatPowOfNat___redArg(v_inst_452_, v_inst_453_);
lean_dec_ref(v_inst_452_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat(lean_object* v_00_u03b9_455_, lean_object* v_A_456_, lean_object* v_inst_457_, lean_object* v_inst_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_mathlib_GradedMonoid_instNatPowOfNat___redArg(v_inst_457_, v_inst_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_instNatPowOfNat___boxed(lean_object* v_00_u03b9_460_, lean_object* v_A_461_, lean_object* v_inst_462_, lean_object* v_inst_463_){
_start:
{
lean_object* v_res_464_; 
v_res_464_ = lp_mathlib_GradedMonoid_instNatPowOfNat(v_00_u03b9_460_, v_A_461_, v_inst_462_, v_inst_463_);
lean_dec_ref(v_inst_462_);
return v_res_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid___redArg___lam__0(lean_object* v_inst_465_, lean_object* v_gnpow_466_, lean_object* v_n_467_, lean_object* v_x_468_){
_start:
{
lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v_toZero_471_; lean_object* v___x_472_; 
v___x_469_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_465_);
v___x_470_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_469_);
v_toZero_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_toZero_471_);
lean_dec_ref(v___x_470_);
v___x_472_ = lean_apply_3(v_gnpow_466_, v_n_467_, v_toZero_471_, v_x_468_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid___redArg___lam__0___boxed(lean_object* v_inst_473_, lean_object* v_gnpow_474_, lean_object* v_n_475_, lean_object* v_x_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_GradedMonoid_GradeZero_monoid___redArg___lam__0(v_inst_473_, v_gnpow_474_, v_n_475_, v_x_476_);
lean_dec_ref(v_inst_473_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid___redArg(lean_object* v_inst_478_, lean_object* v_inst_479_){
_start:
{
lean_object* v___x_480_; lean_object* v_toGMul_481_; lean_object* v_toGOne_482_; lean_object* v_gnpow_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_492_; 
v___x_480_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_478_);
v_toGMul_481_ = lean_ctor_get(v_inst_479_, 0);
v_toGOne_482_ = lean_ctor_get(v_inst_479_, 1);
v_gnpow_483_ = lean_ctor_get(v_inst_479_, 2);
v_isSharedCheck_492_ = !lean_is_exclusive(v_inst_479_);
if (v_isSharedCheck_492_ == 0)
{
v___x_485_ = v_inst_479_;
v_isShared_486_ = v_isSharedCheck_492_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_gnpow_483_);
lean_inc(v_toGOne_482_);
lean_inc(v_toGMul_481_);
lean_dec(v_inst_479_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_492_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___f_487_; lean_object* v___x_488_; lean_object* v___x_490_; 
v___f_487_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GradeZero_monoid___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_487_, 0, v_inst_478_);
lean_closure_set(v___f_487_, 1, v_gnpow_483_);
v___x_488_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v___x_480_, v_toGMul_481_);
if (v_isShared_486_ == 0)
{
lean_ctor_set(v___x_485_, 2, v___f_487_);
lean_ctor_set(v___x_485_, 1, v___x_488_);
lean_ctor_set(v___x_485_, 0, v_toGOne_482_);
v___x_490_ = v___x_485_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v_toGOne_482_);
lean_ctor_set(v_reuseFailAlloc_491_, 1, v___x_488_);
lean_ctor_set(v_reuseFailAlloc_491_, 2, v___f_487_);
v___x_490_ = v_reuseFailAlloc_491_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
return v___x_490_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_monoid(lean_object* v_00_u03b9_493_, lean_object* v_A_494_, lean_object* v_inst_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib_GradedMonoid_GradeZero_monoid___redArg(v_inst_495_, v_inst_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg___lam__0(lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_n_500_, lean_object* v_x_501_){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v_toZero_504_; lean_object* v_gnpow_505_; lean_object* v___x_506_; 
v___x_502_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_498_);
v___x_503_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_502_);
v_toZero_504_ = lean_ctor_get(v___x_503_, 0);
lean_inc(v_toZero_504_);
lean_dec_ref(v___x_503_);
v_gnpow_505_ = lean_ctor_get(v_inst_499_, 2);
lean_inc(v_gnpow_505_);
lean_dec_ref(v_inst_499_);
v___x_506_ = lean_apply_3(v_gnpow_505_, v_n_500_, v_toZero_504_, v_x_501_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg___lam__0___boxed(lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_n_509_, lean_object* v_x_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg___lam__0(v_inst_507_, v_inst_508_, v_n_509_, v_x_510_);
lean_dec_ref(v_inst_507_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg(lean_object* v_inst_512_, lean_object* v_inst_513_){
_start:
{
lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v_toOne_517_; lean_object* v_toMul_518_; lean_object* v___f_519_; lean_object* v___x_520_; 
lean_inc_ref(v_inst_513_);
lean_inc_ref(v_inst_512_);
v___x_514_ = lp_mathlib_GradedMonoid_GradeZero_monoid___redArg(v_inst_512_, v_inst_513_);
v___x_515_ = lp_mathlib_Monoid_toMulOneClass___redArg(v___x_514_);
lean_dec_ref(v___x_514_);
v___x_516_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_515_);
v_toOne_517_ = lean_ctor_get(v___x_516_, 0);
lean_inc(v_toOne_517_);
v_toMul_518_ = lean_ctor_get(v___x_516_, 1);
lean_inc(v_toMul_518_);
lean_dec_ref(v___x_516_);
v___f_519_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_519_, 0, v_inst_512_);
lean_closure_set(v___f_519_, 1, v_inst_513_);
v___x_520_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_520_, 0, v_toOne_517_);
lean_ctor_set(v___x_520_, 1, v_toMul_518_);
lean_ctor_set(v___x_520_, 2, v___f_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_commMonoid(lean_object* v_00_u03b9_521_, lean_object* v_A_522_, lean_object* v_inst_523_, lean_object* v_inst_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_GradedMonoid_GradeZero_commMonoid___redArg(v_inst_523_, v_inst_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom___redArg(lean_object* v_inst_526_){
_start:
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v_toZero_529_; lean_object* v___x_530_; 
v___x_527_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_526_);
v___x_528_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_527_);
v_toZero_529_ = lean_ctor_get(v___x_528_, 0);
lean_inc(v_toZero_529_);
lean_dec_ref(v___x_528_);
v___x_530_ = lean_alloc_closure((void*)(lp_mathlib_GradedMonoid_mk), 4, 3);
lean_closure_set(v___x_530_, 0, lean_box(0));
lean_closure_set(v___x_530_, 1, lean_box(0));
lean_closure_set(v___x_530_, 2, v_toZero_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom___redArg___boxed(lean_object* v_inst_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_GradedMonoid_mkZeroMonoidHom___redArg(v_inst_531_);
lean_dec_ref(v_inst_531_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom(lean_object* v_00_u03b9_533_, lean_object* v_A_534_, lean_object* v_inst_535_, lean_object* v_inst_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib_GradedMonoid_mkZeroMonoidHom___redArg(v_inst_535_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_mkZeroMonoidHom___boxed(lean_object* v_00_u03b9_538_, lean_object* v_A_539_, lean_object* v_inst_540_, lean_object* v_inst_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_mathlib_GradedMonoid_mkZeroMonoidHom(v_00_u03b9_538_, v_A_539_, v_inst_540_, v_inst_541_);
lean_dec_ref(v_inst_541_);
lean_dec_ref(v_inst_540_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction___redArg(lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_i_545_){
_start:
{
lean_object* v___x_546_; lean_object* v_toGMul_547_; lean_object* v___x_548_; 
v___x_546_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_543_);
v_toGMul_547_ = lean_ctor_get(v_inst_544_, 0);
lean_inc(v_toGMul_547_);
lean_dec_ref(v_inst_544_);
v___x_548_ = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(v___x_546_, v_toGMul_547_, v_i_545_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction___redArg___boxed(lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_i_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_mathlib_GradedMonoid_GradeZero_mulAction___redArg(v_inst_549_, v_inst_550_, v_i_551_);
lean_dec_ref(v_inst_549_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction(lean_object* v_00_u03b9_553_, lean_object* v_A_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_i_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lp_mathlib_GradedMonoid_GradeZero_mulAction___redArg(v_inst_555_, v_inst_556_, v_i_557_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedMonoid_GradeZero_mulAction___boxed(lean_object* v_00_u03b9_559_, lean_object* v_A_560_, lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_i_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_GradedMonoid_GradeZero_mulAction(v_00_u03b9_559_, v_A_560_, v_inst_561_, v_inst_562_, v_i_563_);
lean_dec_ref(v_inst_561_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dProdIndex___redArg___lam__0(lean_object* v_f_u03b9_565_, lean_object* v_toAdd_566_, lean_object* v_i_567_, lean_object* v_b_568_){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_569_ = lean_apply_1(v_f_u03b9_565_, v_i_567_);
v___x_570_ = lean_apply_2(v_toAdd_566_, v___x_569_, v_b_568_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dProdIndex___redArg(lean_object* v_inst_571_, lean_object* v_l_572_, lean_object* v_f_u03b9_573_){
_start:
{
lean_object* v_toAdd_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v_toZero_577_; lean_object* v___f_578_; lean_object* v___x_579_; 
v_toAdd_574_ = lean_ctor_get(v_inst_571_, 1);
lean_inc(v_toAdd_574_);
v___x_575_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_571_);
lean_dec_ref(v_inst_571_);
v___x_576_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_575_);
v_toZero_577_ = lean_ctor_get(v___x_576_, 0);
lean_inc(v_toZero_577_);
lean_dec_ref(v___x_576_);
v___f_578_ = lean_alloc_closure((void*)(lp_mathlib_List_dProdIndex___redArg___lam__0), 4, 2);
lean_closure_set(v___f_578_, 0, v_f_u03b9_573_);
lean_closure_set(v___f_578_, 1, v_toAdd_574_);
v___x_579_ = l_List_foldrTR___redArg(v___f_578_, v_toZero_577_, v_l_572_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dProdIndex(lean_object* v_00_u03b9_580_, lean_object* v_00_u03b1_581_, lean_object* v_inst_582_, lean_object* v_l_583_, lean_object* v_f_u03b9_584_){
_start:
{
lean_object* v___x_585_; 
v___x_585_ = lp_mathlib_List_dProdIndex___redArg(v_inst_582_, v_l_583_, v_f_u03b9_584_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dProd___redArg___lam__1(lean_object* v_f_u03b9_586_, lean_object* v_fA_587_, lean_object* v_toGMul_588_, lean_object* v_x_589_, lean_object* v_x_590_, lean_object* v_a_591_, lean_object* v_x_592_){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; 
lean_inc(v_a_591_);
v___x_593_ = lean_apply_1(v_f_u03b9_586_, v_a_591_);
v___x_594_ = lean_apply_1(v_fA_587_, v_a_591_);
v___x_595_ = lean_apply_4(v_toGMul_588_, v___x_593_, v_x_589_, v___x_594_, v_x_590_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dProd___redArg(lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_l_598_, lean_object* v_f_u03b9_599_, lean_object* v_fA_600_){
_start:
{
lean_object* v_toAdd_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v_toZero_604_; lean_object* v_toGMul_605_; lean_object* v_toGOne_606_; lean_object* v___f_607_; lean_object* v___f_608_; lean_object* v___x_609_; 
v_toAdd_601_ = lean_ctor_get(v_inst_596_, 1);
lean_inc(v_toAdd_601_);
v___x_602_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_596_);
lean_dec_ref(v_inst_596_);
v___x_603_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_602_);
v_toZero_604_ = lean_ctor_get(v___x_603_, 0);
lean_inc(v_toZero_604_);
lean_dec_ref(v___x_603_);
v_toGMul_605_ = lean_ctor_get(v_inst_597_, 0);
lean_inc(v_toGMul_605_);
v_toGOne_606_ = lean_ctor_get(v_inst_597_, 1);
lean_inc(v_toGOne_606_);
lean_dec_ref(v_inst_597_);
lean_inc(v_f_u03b9_599_);
v___f_607_ = lean_alloc_closure((void*)(lp_mathlib_List_dProdIndex___redArg___lam__0), 4, 2);
lean_closure_set(v___f_607_, 0, v_f_u03b9_599_);
lean_closure_set(v___f_607_, 1, v_toAdd_601_);
v___f_608_ = lean_alloc_closure((void*)(lp_mathlib_List_dProd___redArg___lam__1), 7, 3);
lean_closure_set(v___f_608_, 0, v_f_u03b9_599_);
lean_closure_set(v___f_608_, 1, v_fA_600_);
lean_closure_set(v___f_608_, 2, v_toGMul_605_);
v___x_609_ = l_List_foldrRecOn___redArg(v_l_598_, v___f_607_, v_toZero_604_, v_toGOne_606_, v___f_608_);
lean_dec(v_toGOne_606_);
lean_dec(v_toZero_604_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dProd(lean_object* v_00_u03b9_610_, lean_object* v_00_u03b1_611_, lean_object* v_A_612_, lean_object* v_inst_613_, lean_object* v_inst_614_, lean_object* v_l_615_, lean_object* v_f_u03b9_616_, lean_object* v_fA_617_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = lp_mathlib_List_dProd___redArg(v_inst_613_, v_inst_614_, v_l_615_, v_f_u03b9_616_, v_fA_617_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_One_gOne___redArg(lean_object* v_inst_619_){
_start:
{
lean_inc(v_inst_619_);
return v_inst_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_One_gOne___redArg___boxed(lean_object* v_inst_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_mathlib_One_gOne___redArg(v_inst_620_);
lean_dec(v_inst_620_);
return v_res_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_One_gOne(lean_object* v_00_u03b9_622_, lean_object* v_R_623_, lean_object* v_inst_624_, lean_object* v_inst_625_){
_start:
{
lean_inc(v_inst_625_);
return v_inst_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_One_gOne___boxed(lean_object* v_00_u03b9_626_, lean_object* v_R_627_, lean_object* v_inst_628_, lean_object* v_inst_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_One_gOne(v_00_u03b9_626_, v_R_627_, v_inst_628_, v_inst_629_);
lean_dec(v_inst_629_);
lean_dec(v_inst_628_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___redArg___lam__0(lean_object* v_inst_631_, lean_object* v_i_632_, lean_object* v_j_633_, lean_object* v_x_634_, lean_object* v_y_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lean_apply_2(v_inst_631_, v_x_634_, v_y_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___redArg___lam__0___boxed(lean_object* v_inst_637_, lean_object* v_i_638_, lean_object* v_j_639_, lean_object* v_x_640_, lean_object* v_y_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_Mul_gMul___redArg___lam__0(v_inst_637_, v_i_638_, v_j_639_, v_x_640_, v_y_641_);
lean_dec(v_j_639_);
lean_dec(v_i_638_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___redArg(lean_object* v_inst_643_){
_start:
{
lean_object* v___f_644_; 
v___f_644_ = lean_alloc_closure((void*)(lp_mathlib_Mul_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_644_, 0, v_inst_643_);
return v___f_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul(lean_object* v_00_u03b9_645_, lean_object* v_R_646_, lean_object* v_inst_647_, lean_object* v_inst_648_){
_start:
{
lean_object* v___f_649_; 
v___f_649_ = lean_alloc_closure((void*)(lp_mathlib_Mul_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_649_, 0, v_inst_648_);
return v___f_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_gMul___boxed(lean_object* v_00_u03b9_650_, lean_object* v_R_651_, lean_object* v_inst_652_, lean_object* v_inst_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Mul_gMul(v_00_u03b9_650_, v_R_651_, v_inst_652_, v_inst_653_);
lean_dec(v_inst_652_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___redArg___lam__0(lean_object* v_toNPow_655_, lean_object* v_n_656_, lean_object* v_x_657_, lean_object* v_a_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lean_apply_2(v_toNPow_655_, v_n_656_, v_a_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___redArg___lam__0___boxed(lean_object* v_toNPow_660_, lean_object* v_n_661_, lean_object* v_x_662_, lean_object* v_a_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_Monoid_gMonoid___redArg___lam__0(v_toNPow_660_, v_n_661_, v_x_662_, v_a_663_);
lean_dec(v_x_662_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___redArg(lean_object* v_inst_665_){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v_toOne_668_; lean_object* v_toMul_669_; lean_object* v_toNPow_670_; lean_object* v___x_672_; uint8_t v_isShared_673_; uint8_t v_isSharedCheck_679_; 
v___x_666_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_665_);
v___x_667_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_666_);
v_toOne_668_ = lean_ctor_get(v___x_667_, 0);
lean_inc(v_toOne_668_);
v_toMul_669_ = lean_ctor_get(v___x_667_, 1);
lean_inc(v_toMul_669_);
lean_dec_ref(v___x_667_);
v_toNPow_670_ = lean_ctor_get(v_inst_665_, 2);
v_isSharedCheck_679_ = !lean_is_exclusive(v_inst_665_);
if (v_isSharedCheck_679_ == 0)
{
lean_object* v_unused_680_; lean_object* v_unused_681_; 
v_unused_680_ = lean_ctor_get(v_inst_665_, 1);
lean_dec(v_unused_680_);
v_unused_681_ = lean_ctor_get(v_inst_665_, 0);
lean_dec(v_unused_681_);
v___x_672_ = v_inst_665_;
v_isShared_673_ = v_isSharedCheck_679_;
goto v_resetjp_671_;
}
else
{
lean_inc(v_toNPow_670_);
lean_dec(v_inst_665_);
v___x_672_ = lean_box(0);
v_isShared_673_ = v_isSharedCheck_679_;
goto v_resetjp_671_;
}
v_resetjp_671_:
{
lean_object* v___f_674_; lean_object* v___f_675_; lean_object* v___x_677_; 
v___f_674_ = lean_alloc_closure((void*)(lp_mathlib_Mul_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_674_, 0, v_toMul_669_);
v___f_675_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_gMonoid___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_675_, 0, v_toNPow_670_);
if (v_isShared_673_ == 0)
{
lean_ctor_set(v___x_672_, 2, v___f_675_);
lean_ctor_set(v___x_672_, 1, v_toOne_668_);
lean_ctor_set(v___x_672_, 0, v___f_674_);
v___x_677_ = v___x_672_;
goto v_reusejp_676_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v___f_674_);
lean_ctor_set(v_reuseFailAlloc_678_, 1, v_toOne_668_);
lean_ctor_set(v_reuseFailAlloc_678_, 2, v___f_675_);
v___x_677_ = v_reuseFailAlloc_678_;
goto v_reusejp_676_;
}
v_reusejp_676_:
{
return v___x_677_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid(lean_object* v_00_u03b9_682_, lean_object* v_R_683_, lean_object* v_inst_684_, lean_object* v_inst_685_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_mathlib_Monoid_gMonoid___redArg(v_inst_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_gMonoid___boxed(lean_object* v_00_u03b9_687_, lean_object* v_R_688_, lean_object* v_inst_689_, lean_object* v_inst_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib_Monoid_gMonoid(v_00_u03b9_687_, v_R_688_, v_inst_689_, v_inst_690_);
lean_dec_ref(v_inst_689_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_gCommMonoid___redArg(lean_object* v_inst_692_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lp_mathlib_Monoid_gMonoid___redArg(v_inst_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_gCommMonoid(lean_object* v_00_u03b9_694_, lean_object* v_R_695_, lean_object* v_inst_696_, lean_object* v_inst_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_Monoid_gMonoid___redArg(v_inst_697_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommMonoid_gCommMonoid___boxed(lean_object* v_00_u03b9_699_, lean_object* v_R_700_, lean_object* v_inst_701_, lean_object* v_inst_702_){
_start:
{
lean_object* v_res_703_; 
v_res_703_ = lp_mathlib_CommMonoid_gCommMonoid(v_00_u03b9_699_, v_R_700_, v_inst_701_, v_inst_702_);
lean_dec_ref(v_inst_701_);
return v_res_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne___redArg(lean_object* v_inst_704_){
_start:
{
lean_inc(v_inst_704_);
return v_inst_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne___redArg___boxed(lean_object* v_inst_705_){
_start:
{
lean_object* v_res_706_; 
v_res_706_ = lp_mathlib_SetLike_gOne___redArg(v_inst_705_);
lean_dec(v_inst_705_);
return v_res_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne(lean_object* v_00_u03b9_707_, lean_object* v_R_708_, lean_object* v_S_709_, lean_object* v_inst_710_, lean_object* v_inst_711_, lean_object* v_inst_712_, lean_object* v_A_713_, lean_object* v_inst_714_){
_start:
{
lean_inc(v_inst_711_);
return v_inst_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gOne___boxed(lean_object* v_00_u03b9_715_, lean_object* v_R_716_, lean_object* v_S_717_, lean_object* v_inst_718_, lean_object* v_inst_719_, lean_object* v_inst_720_, lean_object* v_A_721_, lean_object* v_inst_722_){
_start:
{
lean_object* v_res_723_; 
v_res_723_ = lp_mathlib_SetLike_gOne(v_00_u03b9_715_, v_R_716_, v_S_717_, v_inst_718_, v_inst_719_, v_inst_720_, v_A_721_, v_inst_722_);
lean_dec(v_A_721_);
lean_dec(v_inst_720_);
lean_dec(v_inst_719_);
return v_res_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___redArg___lam__0(lean_object* v_inst_724_, lean_object* v_i_725_, lean_object* v_j_726_, lean_object* v_a_727_, lean_object* v_b_728_){
_start:
{
lean_object* v___x_729_; 
v___x_729_ = lean_apply_2(v_inst_724_, v_a_727_, v_b_728_);
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___redArg___lam__0___boxed(lean_object* v_inst_730_, lean_object* v_i_731_, lean_object* v_j_732_, lean_object* v_a_733_, lean_object* v_b_734_){
_start:
{
lean_object* v_res_735_; 
v_res_735_ = lp_mathlib_SetLike_gMul___redArg___lam__0(v_inst_730_, v_i_731_, v_j_732_, v_a_733_, v_b_734_);
lean_dec(v_j_732_);
lean_dec(v_i_731_);
return v_res_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___redArg(lean_object* v_inst_736_){
_start:
{
lean_object* v___f_737_; 
v___f_737_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_737_, 0, v_inst_736_);
return v___f_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul(lean_object* v_00_u03b9_738_, lean_object* v_R_739_, lean_object* v_S_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_inst_743_, lean_object* v_A_744_, lean_object* v_inst_745_){
_start:
{
lean_object* v___f_746_; 
v___f_746_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_746_, 0, v_inst_742_);
return v___f_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMul___boxed(lean_object* v_00_u03b9_747_, lean_object* v_R_748_, lean_object* v_S_749_, lean_object* v_inst_750_, lean_object* v_inst_751_, lean_object* v_inst_752_, lean_object* v_A_753_, lean_object* v_inst_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_SetLike_gMul(v_00_u03b9_747_, v_R_748_, v_S_749_, v_inst_750_, v_inst_751_, v_inst_752_, v_A_753_, v_inst_754_);
lean_dec(v_A_753_);
lean_dec(v_inst_752_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_submonoid(lean_object* v_00_u03b9_756_, lean_object* v_R_757_, lean_object* v_S_758_, lean_object* v_inst_759_, lean_object* v_inst_760_, lean_object* v_inst_761_, lean_object* v_A_762_, lean_object* v_inst_763_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lean_box(0);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_submonoid___boxed(lean_object* v_00_u03b9_765_, lean_object* v_R_766_, lean_object* v_S_767_, lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_inst_770_, lean_object* v_A_771_, lean_object* v_inst_772_){
_start:
{
lean_object* v_res_773_; 
v_res_773_ = lp_mathlib_SetLike_GradeZero_submonoid(v_00_u03b9_765_, v_R_766_, v_S_767_, v_inst_768_, v_inst_769_, v_inst_770_, v_A_771_, v_inst_772_);
lean_dec(v_A_771_);
lean_dec_ref(v_inst_770_);
lean_dec_ref(v_inst_769_);
return v_res_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___redArg(lean_object* v_inst_774_){
_start:
{
lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v_toOne_777_; 
v___x_775_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_774_);
v___x_776_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_775_);
v_toOne_777_ = lean_ctor_get(v___x_776_, 0);
lean_inc(v_toOne_777_);
lean_dec_ref(v___x_776_);
return v_toOne_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___redArg___boxed(lean_object* v_inst_778_){
_start:
{
lean_object* v_res_779_; 
v_res_779_ = lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___redArg(v_inst_778_);
lean_dec_ref(v_inst_778_);
return v_res_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1(lean_object* v_00_u03b9_780_, lean_object* v_R_781_, lean_object* v_S_782_, lean_object* v_inst_783_, lean_object* v_inst_784_, lean_object* v_inst_785_, lean_object* v_A_786_, lean_object* v_inst_787_){
_start:
{
lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v_toOne_790_; 
v___x_788_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_784_);
v___x_789_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_788_);
v_toOne_790_ = lean_ctor_get(v___x_789_, 0);
lean_inc(v_toOne_790_);
lean_dec_ref(v___x_789_);
return v_toOne_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__1___boxed(lean_object* v_00_u03b9_791_, lean_object* v_R_792_, lean_object* v_S_793_, lean_object* v_inst_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_A_797_, lean_object* v_inst_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_mathlib_SetLike_GradeZero_instMonoid___aux__1(v_00_u03b9_791_, v_R_792_, v_S_793_, v_inst_794_, v_inst_795_, v_inst_796_, v_A_797_, v_inst_798_);
lean_dec(v_A_797_);
lean_dec_ref(v_inst_796_);
lean_dec_ref(v_inst_795_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___redArg(lean_object* v_inst_800_, lean_object* v_a_801_, lean_object* v_b_802_){
_start:
{
lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v_toMul_805_; lean_object* v___x_806_; 
v___x_803_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_800_);
v___x_804_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_803_);
v_toMul_805_ = lean_ctor_get(v___x_804_, 1);
lean_inc(v_toMul_805_);
lean_dec_ref(v___x_804_);
v___x_806_ = lean_apply_2(v_toMul_805_, v_a_801_, v_b_802_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___redArg___boxed(lean_object* v_inst_807_, lean_object* v_a_808_, lean_object* v_b_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___redArg(v_inst_807_, v_a_808_, v_b_809_);
lean_dec_ref(v_inst_807_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3(lean_object* v_00_u03b9_811_, lean_object* v_R_812_, lean_object* v_S_813_, lean_object* v_inst_814_, lean_object* v_inst_815_, lean_object* v_inst_816_, lean_object* v_A_817_, lean_object* v_inst_818_, lean_object* v_a_819_, lean_object* v_b_820_){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v_toMul_823_; lean_object* v___x_824_; 
v___x_821_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_815_);
v___x_822_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_821_);
v_toMul_823_ = lean_ctor_get(v___x_822_, 1);
lean_inc(v_toMul_823_);
lean_dec_ref(v___x_822_);
v___x_824_ = lean_apply_2(v_toMul_823_, v_a_819_, v_b_820_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___boxed(lean_object* v_00_u03b9_825_, lean_object* v_R_826_, lean_object* v_S_827_, lean_object* v_inst_828_, lean_object* v_inst_829_, lean_object* v_inst_830_, lean_object* v_A_831_, lean_object* v_inst_832_, lean_object* v_a_833_, lean_object* v_b_834_){
_start:
{
lean_object* v_res_835_; 
v_res_835_ = lp_mathlib_SetLike_GradeZero_instMonoid___aux__3(v_00_u03b9_825_, v_R_826_, v_S_827_, v_inst_828_, v_inst_829_, v_inst_830_, v_A_831_, v_inst_832_, v_a_833_, v_b_834_);
lean_dec(v_A_831_);
lean_dec_ref(v_inst_830_);
lean_dec_ref(v_inst_829_);
return v_res_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__8___redArg(lean_object* v_inst_836_, lean_object* v_n_837_, lean_object* v_x_838_){
_start:
{
lean_object* v_toNPow_839_; lean_object* v___x_840_; 
v_toNPow_839_ = lean_ctor_get(v_inst_836_, 2);
lean_inc(v_toNPow_839_);
lean_dec_ref(v_inst_836_);
v___x_840_ = lean_apply_2(v_toNPow_839_, v_n_837_, v_x_838_);
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__8(lean_object* v_00_u03b9_841_, lean_object* v_R_842_, lean_object* v_S_843_, lean_object* v_inst_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_A_847_, lean_object* v_inst_848_, lean_object* v_n_849_, lean_object* v_x_850_){
_start:
{
lean_object* v_toNPow_851_; lean_object* v___x_852_; 
v_toNPow_851_ = lean_ctor_get(v_inst_845_, 2);
lean_inc(v_toNPow_851_);
lean_dec_ref(v_inst_845_);
v___x_852_ = lean_apply_2(v_toNPow_851_, v_n_849_, v_x_850_);
return v___x_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___aux__8___boxed(lean_object* v_00_u03b9_853_, lean_object* v_R_854_, lean_object* v_S_855_, lean_object* v_inst_856_, lean_object* v_inst_857_, lean_object* v_inst_858_, lean_object* v_A_859_, lean_object* v_inst_860_, lean_object* v_n_861_, lean_object* v_x_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_mathlib_SetLike_GradeZero_instMonoid___aux__8(v_00_u03b9_853_, v_R_854_, v_S_855_, v_inst_856_, v_inst_857_, v_inst_858_, v_A_859_, v_inst_860_, v_n_861_, v_x_862_);
lean_dec(v_A_859_);
lean_dec_ref(v_inst_858_);
return v_res_863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid___redArg(lean_object* v_inst_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_A_867_){
_start:
{
lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v_toOne_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
v___x_868_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_865_);
v___x_869_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_868_);
v_toOne_870_ = lean_ctor_get(v___x_869_, 0);
lean_inc(v_toOne_870_);
lean_dec_ref(v___x_869_);
lean_inc(v_A_867_);
lean_inc_ref(v_inst_866_);
lean_inc_ref(v_inst_865_);
v___x_871_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_GradeZero_instMonoid___aux__3___boxed), 10, 8);
lean_closure_set(v___x_871_, 0, lean_box(0));
lean_closure_set(v___x_871_, 1, lean_box(0));
lean_closure_set(v___x_871_, 2, lean_box(0));
lean_closure_set(v___x_871_, 3, v_inst_864_);
lean_closure_set(v___x_871_, 4, v_inst_865_);
lean_closure_set(v___x_871_, 5, v_inst_866_);
lean_closure_set(v___x_871_, 6, v_A_867_);
lean_closure_set(v___x_871_, 7, lean_box(0));
v___x_872_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_GradeZero_instMonoid___aux__8___boxed), 10, 8);
lean_closure_set(v___x_872_, 0, lean_box(0));
lean_closure_set(v___x_872_, 1, lean_box(0));
lean_closure_set(v___x_872_, 2, lean_box(0));
lean_closure_set(v___x_872_, 3, v_inst_864_);
lean_closure_set(v___x_872_, 4, v_inst_865_);
lean_closure_set(v___x_872_, 5, v_inst_866_);
lean_closure_set(v___x_872_, 6, v_A_867_);
lean_closure_set(v___x_872_, 7, lean_box(0));
v___x_873_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_873_, 0, v_toOne_870_);
lean_ctor_set(v___x_873_, 1, v___x_871_);
lean_ctor_set(v___x_873_, 2, v___x_872_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instMonoid(lean_object* v_00_u03b9_874_, lean_object* v_R_875_, lean_object* v_S_876_, lean_object* v_inst_877_, lean_object* v_inst_878_, lean_object* v_inst_879_, lean_object* v_A_880_, lean_object* v_inst_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lp_mathlib_SetLike_GradeZero_instMonoid___redArg(v_inst_877_, v_inst_878_, v_inst_879_, v_A_880_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommMonoid___redArg(lean_object* v_inst_883_, lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_A_886_){
_start:
{
lean_object* v___x_887_; 
v___x_887_ = lp_mathlib_SetLike_GradeZero_instMonoid___redArg(v_inst_884_, v_inst_885_, v_inst_883_, v_A_886_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_GradeZero_instCommMonoid(lean_object* v_00_u03b9_888_, lean_object* v_inst_889_, lean_object* v_R_890_, lean_object* v_S_891_, lean_object* v_inst_892_, lean_object* v_inst_893_, lean_object* v_A_894_, lean_object* v_inst_895_){
_start:
{
lean_object* v___x_896_; 
v___x_896_ = lp_mathlib_SetLike_GradeZero_instMonoid___redArg(v_inst_892_, v_inst_893_, v_inst_889_, v_A_894_);
return v___x_896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMonoid___redArg(lean_object* v_inst_897_){
_start:
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v_toOne_900_; lean_object* v_toMul_901_; lean_object* v_toNPow_902_; lean_object* v___x_904_; uint8_t v_isShared_905_; uint8_t v_isSharedCheck_911_; 
v___x_898_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_897_);
v___x_899_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_898_);
v_toOne_900_ = lean_ctor_get(v___x_899_, 0);
lean_inc(v_toOne_900_);
v_toMul_901_ = lean_ctor_get(v___x_899_, 1);
lean_inc(v_toMul_901_);
lean_dec_ref(v___x_899_);
v_toNPow_902_ = lean_ctor_get(v_inst_897_, 2);
v_isSharedCheck_911_ = !lean_is_exclusive(v_inst_897_);
if (v_isSharedCheck_911_ == 0)
{
lean_object* v_unused_912_; lean_object* v_unused_913_; 
v_unused_912_ = lean_ctor_get(v_inst_897_, 1);
lean_dec(v_unused_912_);
v_unused_913_ = lean_ctor_get(v_inst_897_, 0);
lean_dec(v_unused_913_);
v___x_904_ = v_inst_897_;
v_isShared_905_ = v_isSharedCheck_911_;
goto v_resetjp_903_;
}
else
{
lean_inc(v_toNPow_902_);
lean_dec(v_inst_897_);
v___x_904_ = lean_box(0);
v_isShared_905_ = v_isSharedCheck_911_;
goto v_resetjp_903_;
}
v_resetjp_903_:
{
lean_object* v___f_906_; lean_object* v___f_907_; lean_object* v___x_909_; 
v___f_906_ = lean_alloc_closure((void*)(lp_mathlib_SetLike_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_906_, 0, v_toMul_901_);
v___f_907_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_gMonoid___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_907_, 0, v_toNPow_902_);
if (v_isShared_905_ == 0)
{
lean_ctor_set(v___x_904_, 2, v___f_907_);
lean_ctor_set(v___x_904_, 1, v_toOne_900_);
lean_ctor_set(v___x_904_, 0, v___f_906_);
v___x_909_ = v___x_904_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_910_; 
v_reuseFailAlloc_910_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_910_, 0, v___f_906_);
lean_ctor_set(v_reuseFailAlloc_910_, 1, v_toOne_900_);
lean_ctor_set(v_reuseFailAlloc_910_, 2, v___f_907_);
v___x_909_ = v_reuseFailAlloc_910_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
return v___x_909_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMonoid(lean_object* v_00_u03b9_914_, lean_object* v_R_915_, lean_object* v_S_916_, lean_object* v_inst_917_, lean_object* v_inst_918_, lean_object* v_inst_919_, lean_object* v_A_920_, lean_object* v_inst_921_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = lp_mathlib_SetLike_gMonoid___redArg(v_inst_918_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gMonoid___boxed(lean_object* v_00_u03b9_923_, lean_object* v_R_924_, lean_object* v_S_925_, lean_object* v_inst_926_, lean_object* v_inst_927_, lean_object* v_inst_928_, lean_object* v_A_929_, lean_object* v_inst_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_mathlib_SetLike_gMonoid(v_00_u03b9_923_, v_R_924_, v_S_925_, v_inst_926_, v_inst_927_, v_inst_928_, v_A_929_, v_inst_930_);
lean_dec(v_A_929_);
lean_dec_ref(v_inst_928_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gCommMonoid___redArg(lean_object* v_inst_932_){
_start:
{
lean_object* v___x_933_; 
v___x_933_ = lp_mathlib_SetLike_gMonoid___redArg(v_inst_932_);
return v___x_933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gCommMonoid(lean_object* v_00_u03b9_934_, lean_object* v_R_935_, lean_object* v_S_936_, lean_object* v_inst_937_, lean_object* v_inst_938_, lean_object* v_inst_939_, lean_object* v_A_940_, lean_object* v_inst_941_){
_start:
{
lean_object* v___x_942_; 
v___x_942_ = lp_mathlib_SetLike_gMonoid___redArg(v_inst_938_);
return v___x_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_gCommMonoid___boxed(lean_object* v_00_u03b9_943_, lean_object* v_R_944_, lean_object* v_S_945_, lean_object* v_inst_946_, lean_object* v_inst_947_, lean_object* v_inst_948_, lean_object* v_A_949_, lean_object* v_inst_950_){
_start:
{
lean_object* v_res_951_; 
v_res_951_ = lp_mathlib_SetLike_gCommMonoid(v_00_u03b9_943_, v_R_944_, v_S_945_, v_inst_946_, v_inst_947_, v_inst_948_, v_A_949_, v_inst_950_);
lean_dec(v_A_949_);
lean_dec_ref(v_inst_948_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_homogeneousSubmonoid(lean_object* v_00_u03b9_952_, lean_object* v_R_953_, lean_object* v_S_954_, lean_object* v_inst_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_A_958_, lean_object* v_inst_959_){
_start:
{
lean_object* v___x_960_; 
v___x_960_ = lean_box(0);
return v___x_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_homogeneousSubmonoid___boxed(lean_object* v_00_u03b9_961_, lean_object* v_R_962_, lean_object* v_S_963_, lean_object* v_inst_964_, lean_object* v_inst_965_, lean_object* v_inst_966_, lean_object* v_A_967_, lean_object* v_inst_968_){
_start:
{
lean_object* v_res_969_; 
v_res_969_ = lp_mathlib_SetLike_homogeneousSubmonoid(v_00_u03b9_961_, v_R_962_, v_S_963_, v_inst_964_, v_inst_965_, v_inst_966_, v_A_967_, v_inst_968_);
lean_dec(v_A_967_);
lean_dec_ref(v_inst_966_);
lean_dec_ref(v_inst_965_);
return v_res_969_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_FinRange(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_FinRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam = _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam();
lean_mark_persistent(lp_mathlib_GradedMonoid_GMonoid_gnpow__zero_x27___autoParam);
lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam = _init_lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam();
lean_mark_persistent(lp_mathlib_GradedMonoid_GMonoid_gnpow__succ_x27___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_FinRange(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_FinRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
}
#ifdef __cplusplus
}
#endif
