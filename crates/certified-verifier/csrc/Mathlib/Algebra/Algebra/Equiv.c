// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Equiv
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Hom public import Mathlib.Algebra.Ring.Action.Group
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_ULift_ringEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemiring;
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toIntAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
extern lean_object* lp_mathlib_Nat_instSemiring;
lean_object* lp_mathlib_Semiring_toNatAlgebra___redArg(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 11, .m_data = "term_≃ₐ[_]_"};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(202, 39, 159, 2, 24, 78, 66, 35)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ≃ₐ["};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__9_value;
static const lean_string_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__10_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__10_value)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__11 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__11_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__9_value),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__11_value)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__12 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__12_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__12_value),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__13 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__13_value;
static const lean_ctor_object lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__13_value)}};
static const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__14 = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_u2090_x5b___x5d__ = (const lean_object*)&lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "AlgEquiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(65, 206, 77, 118, 12, 82, 247, 147)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquivClass_toAlgEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquivClass_toAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquivClass_toAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutLinearEquivId___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutLinearEquivId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutRingEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutAlgHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AlgEquiv_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgEquiv_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_refl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_refl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__1 = (const lean_object*)&lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__0_value),((lean_object*)&lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__2 = (const lean_object*)&lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AlgEquiv_cast___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgEquiv_cast___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_aut___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_aut___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_aut___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_aut___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_aut___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_AlgEquiv_aut___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgEquiv_aut___redArg___closed__1;
static lean_once_cell_t lp_mathlib_AlgEquiv_aut___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgEquiv_aut___redArg___closed__2;
static lean_once_cell_t lp_mathlib_AlgEquiv_aut___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgEquiv_aut___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_aut___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_aut(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_applyMulSemiringAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_applyMulSemiringAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_applyMulSemiringAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHomHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHomHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMapHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMapHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__1 = (const lean_object*)&lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__0_value),((lean_object*)&lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__2 = (const lean_object*)&lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNatAlgEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNatAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNatAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivNatAlgEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivNatAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toIntAlgEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toIntAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toIntAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivIntAlgEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivIntAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgAut___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgAut(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_algEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ulift___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ulift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ulift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjAlgEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjAlgEquiv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv___redArg(lean_object* v_self_1_){
_start:
{
lean_inc_ref(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_AlgEquiv_toMulEquiv___redArg(v_self_2_);
lean_dec_ref(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv(lean_object* v_R_4_, lean_object* v_A_5_, lean_object* v_B_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_self_12_){
_start:
{
lean_inc_ref(v_self_12_);
return v_self_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toMulEquiv___boxed(lean_object* v_R_13_, lean_object* v_A_14_, lean_object* v_B_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_self_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_AlgEquiv_toMulEquiv(v_R_13_, v_A_14_, v_B_15_, v_inst_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_inst_20_, v_self_21_);
lean_dec_ref(v_self_21_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_19_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
lean_dec_ref(v_inst_16_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv___redArg(lean_object* v_self_23_){
_start:
{
lean_inc_ref(v_self_23_);
return v_self_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv___redArg___boxed(lean_object* v_self_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_AlgEquiv_toAddEquiv___redArg(v_self_24_);
lean_dec_ref(v_self_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv(lean_object* v_R_26_, lean_object* v_A_27_, lean_object* v_B_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_self_34_){
_start:
{
lean_inc_ref(v_self_34_);
return v_self_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAddEquiv___boxed(lean_object* v_R_35_, lean_object* v_A_36_, lean_object* v_B_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_self_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_AlgEquiv_toAddEquiv(v_R_35_, v_A_36_, v_B_37_, v_inst_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_inst_42_, v_self_43_);
lean_dec_ref(v_self_43_);
lean_dec_ref(v_inst_42_);
lean_dec_ref(v_inst_41_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_38_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___redArg(lean_object* v_self_45_){
_start:
{
lean_inc_ref(v_self_45_);
return v_self_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___redArg___boxed(lean_object* v_self_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_AlgEquiv_toRingEquiv___redArg(v_self_46_);
lean_dec_ref(v_self_46_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv(lean_object* v_R_48_, lean_object* v_A_49_, lean_object* v_B_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_self_56_){
_start:
{
lean_inc_ref(v_self_56_);
return v_self_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toRingEquiv___boxed(lean_object* v_R_57_, lean_object* v_A_58_, lean_object* v_B_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_self_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_AlgEquiv_toRingEquiv(v_R_57_, v_A_58_, v_B_59_, v_inst_60_, v_inst_61_, v_inst_62_, v_inst_63_, v_inst_64_, v_self_65_);
lean_dec_ref(v_self_65_);
lean_dec_ref(v_inst_64_);
lean_dec_ref(v_inst_63_);
lean_dec_ref(v_inst_62_);
lean_dec_ref(v_inst_61_);
lean_dec_ref(v_inst_60_);
return v_res_66_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__6(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__5));
v___x_114_ = l_String_toRawSubstring_x27(v___x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1(lean_object* v_x_131_, lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
lean_object* v___x_134_; uint8_t v___x_135_; 
v___x_134_ = ((lean_object*)(lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__1));
lean_inc(v_x_131_);
v___x_135_ = l_Lean_Syntax_isOfKind(v_x_131_, v___x_134_);
if (v___x_135_ == 0)
{
lean_object* v___x_136_; lean_object* v___x_137_; 
lean_dec(v_x_131_);
v___x_136_ = lean_box(1);
v___x_137_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
lean_ctor_set(v___x_137_, 1, v_a_133_);
return v___x_137_;
}
else
{
lean_object* v_quotContext_138_; lean_object* v_currMacroScope_139_; lean_object* v_ref_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; uint8_t v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v_quotContext_138_ = lean_ctor_get(v_a_132_, 1);
v_currMacroScope_139_ = lean_ctor_get(v_a_132_, 2);
v_ref_140_ = lean_ctor_get(v_a_132_, 5);
v___x_141_ = lean_unsigned_to_nat(0u);
v___x_142_ = l_Lean_Syntax_getArg(v_x_131_, v___x_141_);
v___x_143_ = lean_unsigned_to_nat(2u);
v___x_144_ = l_Lean_Syntax_getArg(v_x_131_, v___x_143_);
v___x_145_ = lean_unsigned_to_nat(4u);
v___x_146_ = l_Lean_Syntax_getArg(v_x_131_, v___x_145_);
lean_dec(v_x_131_);
v___x_147_ = 0;
v___x_148_ = l_Lean_SourceInfo_fromRef(v_ref_140_, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4));
v___x_150_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__6);
v___x_151_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__7));
lean_inc(v_currMacroScope_139_);
lean_inc(v_quotContext_138_);
v___x_152_ = l_Lean_addMacroScope(v_quotContext_138_, v___x_151_, v_currMacroScope_139_);
v___x_153_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__11));
lean_inc_n(v___x_148_, 2);
v___x_154_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_154_, 0, v___x_148_);
lean_ctor_set(v___x_154_, 1, v___x_150_);
lean_ctor_set(v___x_154_, 2, v___x_152_);
lean_ctor_set(v___x_154_, 3, v___x_153_);
v___x_155_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__13));
v___x_156_ = l_Lean_Syntax_node3(v___x_148_, v___x_155_, v___x_144_, v___x_142_, v___x_146_);
v___x_157_ = l_Lean_Syntax_node2(v___x_148_, v___x_149_, v___x_154_, v___x_156_);
v___x_158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v_a_133_);
return v___x_158_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___boxed(lean_object* v_x_159_, lean_object* v_a_160_, lean_object* v_a_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1(v_x_159_, v_a_160_, v_a_161_);
lean_dec_ref(v_a_160_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1(lean_object* v_x_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______macroRules__term___u2243_u2090_x5b___x5d____1___closed__4));
lean_inc(v_x_166_);
v___x_170_ = l_Lean_Syntax_isOfKind(v_x_166_, v___x_169_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; lean_object* v___x_172_; 
lean_dec(v_x_166_);
v___x_171_ = lean_box(0);
v___x_172_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v_a_168_);
return v___x_172_;
}
else
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; uint8_t v___x_176_; 
v___x_173_ = lean_unsigned_to_nat(0u);
v___x_174_ = l_Lean_Syntax_getArg(v_x_166_, v___x_173_);
v___x_175_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___closed__1));
lean_inc(v___x_174_);
v___x_176_ = l_Lean_Syntax_isOfKind(v___x_174_, v___x_175_);
if (v___x_176_ == 0)
{
lean_object* v___x_177_; lean_object* v___x_178_; 
lean_dec(v___x_174_);
lean_dec(v_x_166_);
v___x_177_ = lean_box(0);
v___x_178_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v_a_168_);
return v___x_178_;
}
else
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; uint8_t v___x_182_; 
v___x_179_ = lean_unsigned_to_nat(1u);
v___x_180_ = l_Lean_Syntax_getArg(v_x_166_, v___x_179_);
lean_dec(v_x_166_);
v___x_181_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_180_);
v___x_182_ = l_Lean_Syntax_matchesNull(v___x_180_, v___x_181_);
if (v___x_182_ == 0)
{
lean_object* v___x_183_; lean_object* v___x_184_; 
lean_dec(v___x_180_);
lean_dec(v___x_174_);
v___x_183_ = lean_box(0);
v___x_184_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v_a_168_);
return v___x_184_;
}
else
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v_ref_189_; uint8_t v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_185_ = l_Lean_Syntax_getArg(v___x_180_, v___x_173_);
v___x_186_ = l_Lean_Syntax_getArg(v___x_180_, v___x_179_);
v___x_187_ = lean_unsigned_to_nat(2u);
v___x_188_ = l_Lean_Syntax_getArg(v___x_180_, v___x_187_);
lean_dec(v___x_180_);
v_ref_189_ = l_Lean_replaceRef(v___x_174_, v_a_167_);
lean_dec(v___x_174_);
v___x_190_ = 0;
v___x_191_ = l_Lean_SourceInfo_fromRef(v_ref_189_, v___x_190_);
lean_dec(v_ref_189_);
v___x_192_ = ((lean_object*)(lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__1));
v___x_193_ = ((lean_object*)(lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__4));
lean_inc_n(v___x_191_, 2);
v___x_194_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_191_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v___x_195_ = ((lean_object*)(lp_mathlib_term___u2243_u2090_x5b___x5d___00__closed__10));
v___x_196_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_191_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
v___x_197_ = l_Lean_Syntax_node5(v___x_191_, v___x_192_, v___x_186_, v___x_194_, v___x_185_, v___x_196_, v___x_188_);
v___x_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set(v___x_198_, 1, v_a_168_);
return v___x_198_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1___boxed(lean_object* v_x_199_, lean_object* v_a_200_, lean_object* v_a_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib___aux__Mathlib__Algebra__Algebra__Equiv______unexpand__AlgEquiv__1(v_x_199_, v_a_200_, v_a_201_);
lean_dec(v_a_200_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofClass___redArg(lean_object* v_inst_203_, lean_object* v_f_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_203_, v_f_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofClass(lean_object* v_F_206_, lean_object* v_R_207_, lean_object* v_A_208_, lean_object* v_B_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_215_, v_f_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofClass___boxed(lean_object* v_F_219_, lean_object* v_R_220_, lean_object* v_A_221_, lean_object* v_B_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_f_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_AlgEquiv_ofClass(v_F_219_, v_R_220_, v_A_221_, v_B_222_, v_inst_223_, v_inst_224_, v_inst_225_, v_inst_226_, v_inst_227_, v_inst_228_, v_inst_229_, v_f_230_);
lean_dec_ref(v_inst_227_);
lean_dec_ref(v_inst_226_);
lean_dec_ref(v_inst_225_);
lean_dec_ref(v_inst_224_);
lean_dec_ref(v_inst_223_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquivClass_toAlgEquiv___redArg(lean_object* v_inst_232_, lean_object* v_f_233_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_232_, v_f_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquivClass_toAlgEquiv(lean_object* v_F_235_, lean_object* v_R_236_, lean_object* v_A_237_, lean_object* v_B_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_f_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_244_, v_f_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquivClass_toAlgEquiv___boxed(lean_object* v_F_248_, lean_object* v_R_249_, lean_object* v_A_250_, lean_object* v_B_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_f_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_AlgEquivClass_toAlgEquiv(v_F_248_, v_R_249_, v_A_250_, v_B_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_, v_inst_257_, v_inst_258_, v_f_259_);
lean_dec_ref(v_inst_256_);
lean_dec_ref(v_inst_255_);
lean_dec_ref(v_inst_254_);
lean_dec_ref(v_inst_253_);
lean_dec_ref(v_inst_252_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearEquiv___redArg(lean_object* v_e_261_){
_start:
{
lean_object* v_toFun_262_; lean_object* v_invFun_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_270_; 
v_toFun_262_ = lean_ctor_get(v_e_261_, 0);
v_invFun_263_ = lean_ctor_get(v_e_261_, 1);
v_isSharedCheck_270_ = !lean_is_exclusive(v_e_261_);
if (v_isSharedCheck_270_ == 0)
{
v___x_265_ = v_e_261_;
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_invFun_263_);
lean_inc(v_toFun_262_);
lean_dec(v_e_261_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_268_; 
if (v_isShared_266_ == 0)
{
v___x_268_ = v___x_265_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_toFun_262_);
lean_ctor_set(v_reuseFailAlloc_269_, 1, v_invFun_263_);
v___x_268_ = v_reuseFailAlloc_269_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
return v___x_268_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearEquiv(lean_object* v_R_271_, lean_object* v_A_u2081_272_, lean_object* v_A_u2082_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_e_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_mathlib_AlgEquiv_toLinearEquiv___redArg(v_e_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearEquiv___boxed(lean_object* v_R_281_, lean_object* v_A_u2081_282_, lean_object* v_A_u2082_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_e_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_AlgEquiv_toLinearEquiv(v_R_281_, v_A_u2081_282_, v_A_u2082_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_inst_288_, v_e_289_);
lean_dec_ref(v_inst_288_);
lean_dec_ref(v_inst_287_);
lean_dec_ref(v_inst_286_);
lean_dec_ref(v_inst_285_);
lean_dec_ref(v_inst_284_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutLinearEquivId___redArg(lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toLinearEquiv___boxed), 9, 8);
lean_closure_set(v___x_296_, 0, lean_box(0));
lean_closure_set(v___x_296_, 1, lean_box(0));
lean_closure_set(v___x_296_, 2, lean_box(0));
lean_closure_set(v___x_296_, 3, v_inst_291_);
lean_closure_set(v___x_296_, 4, v_inst_292_);
lean_closure_set(v___x_296_, 5, v_inst_293_);
lean_closure_set(v___x_296_, 6, v_inst_294_);
lean_closure_set(v___x_296_, 7, v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutLinearEquivId(lean_object* v_R_297_, lean_object* v_A_u2081_298_, lean_object* v_A_u2082_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toLinearEquiv___boxed), 9, 8);
lean_closure_set(v___x_305_, 0, lean_box(0));
lean_closure_set(v___x_305_, 1, lean_box(0));
lean_closure_set(v___x_305_, 2, lean_box(0));
lean_closure_set(v___x_305_, 3, v_inst_300_);
lean_closure_set(v___x_305_, 4, v_inst_301_);
lean_closure_set(v___x_305_, 5, v_inst_302_);
lean_closure_set(v___x_305_, 6, v_inst_303_);
lean_closure_set(v___x_305_, 7, v_inst_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutRingEquiv___redArg(lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toRingEquiv___boxed), 9, 8);
lean_closure_set(v___x_311_, 0, lean_box(0));
lean_closure_set(v___x_311_, 1, lean_box(0));
lean_closure_set(v___x_311_, 2, lean_box(0));
lean_closure_set(v___x_311_, 3, v_inst_306_);
lean_closure_set(v___x_311_, 4, v_inst_307_);
lean_closure_set(v___x_311_, 5, v_inst_308_);
lean_closure_set(v___x_311_, 6, v_inst_309_);
lean_closure_set(v___x_311_, 7, v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutRingEquiv(lean_object* v_R_312_, lean_object* v_A_u2081_313_, lean_object* v_A_u2082_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toRingEquiv___boxed), 9, 8);
lean_closure_set(v___x_320_, 0, lean_box(0));
lean_closure_set(v___x_320_, 1, lean_box(0));
lean_closure_set(v___x_320_, 2, lean_box(0));
lean_closure_set(v___x_320_, 3, v_inst_315_);
lean_closure_set(v___x_320_, 4, v_inst_316_);
lean_closure_set(v___x_320_, 5, v_inst_317_);
lean_closure_set(v___x_320_, 6, v_inst_318_);
lean_closure_set(v___x_320_, 7, v_inst_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom___redArg(lean_object* v_e_321_){
_start:
{
lean_object* v_toFun_322_; 
v_toFun_322_ = lean_ctor_get(v_e_321_, 0);
lean_inc(v_toFun_322_);
return v_toFun_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom___redArg___boxed(lean_object* v_e_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_AlgEquiv_toAlgHom___redArg(v_e_323_);
lean_dec_ref(v_e_323_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom(lean_object* v_R_325_, lean_object* v_A_u2081_326_, lean_object* v_A_u2082_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_e_333_){
_start:
{
lean_object* v_toFun_334_; 
v_toFun_334_ = lean_ctor_get(v_e_333_, 0);
lean_inc(v_toFun_334_);
return v_toFun_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHom___boxed(lean_object* v_R_335_, lean_object* v_A_u2081_336_, lean_object* v_A_u2082_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_e_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_AlgEquiv_toAlgHom(v_R_335_, v_A_u2081_336_, v_A_u2082_337_, v_inst_338_, v_inst_339_, v_inst_340_, v_inst_341_, v_inst_342_, v_e_343_);
lean_dec_ref(v_e_343_);
lean_dec_ref(v_inst_342_);
lean_dec_ref(v_inst_341_);
lean_dec_ref(v_inst_340_);
lean_dec_ref(v_inst_339_);
lean_dec_ref(v_inst_338_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutAlgHom___redArg(lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toAlgHom___boxed), 9, 8);
lean_closure_set(v___x_350_, 0, lean_box(0));
lean_closure_set(v___x_350_, 1, lean_box(0));
lean_closure_set(v___x_350_, 2, lean_box(0));
lean_closure_set(v___x_350_, 3, v_inst_345_);
lean_closure_set(v___x_350_, 4, v_inst_346_);
lean_closure_set(v___x_350_, 5, v_inst_347_);
lean_closure_set(v___x_350_, 6, v_inst_348_);
lean_closure_set(v___x_350_, 7, v_inst_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instCoeOutAlgHom(lean_object* v_R_351_, lean_object* v_A_u2081_352_, lean_object* v_A_u2082_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toAlgHom___boxed), 9, 8);
lean_closure_set(v___x_359_, 0, lean_box(0));
lean_closure_set(v___x_359_, 1, lean_box(0));
lean_closure_set(v___x_359_, 2, lean_box(0));
lean_closure_set(v___x_359_, 3, v_inst_354_);
lean_closure_set(v___x_359_, 4, v_inst_355_);
lean_closure_set(v___x_359_, 5, v_inst_356_);
lean_closure_set(v___x_359_, 6, v_inst_357_);
lean_closure_set(v___x_359_, 7, v_inst_358_);
return v___x_359_;
}
}
static lean_object* _init_lp_mathlib_AlgEquiv_refl___closed__0(void){
_start:
{
lean_object* v___x_360_; 
v___x_360_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_refl(lean_object* v_R_361_, lean_object* v_A_u2081_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_obj_once(&lp_mathlib_AlgEquiv_refl___closed__0, &lp_mathlib_AlgEquiv_refl___closed__0_once, _init_lp_mathlib_AlgEquiv_refl___closed__0);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_refl___boxed(lean_object* v_R_367_, lean_object* v_A_u2081_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_AlgEquiv_refl(v_R_367_, v_A_u2081_368_, v_inst_369_, v_inst_370_, v_inst_371_);
lean_dec_ref(v_inst_371_);
lean_dec_ref(v_inst_370_);
lean_dec_ref(v_inst_369_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instInhabited(lean_object* v_R_373_, lean_object* v_A_u2081_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lean_obj_once(&lp_mathlib_AlgEquiv_refl___closed__0, &lp_mathlib_AlgEquiv_refl___closed__0_once, _init_lp_mathlib_AlgEquiv_refl___closed__0);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instInhabited___boxed(lean_object* v_R_379_, lean_object* v_A_u2081_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_AlgEquiv_instInhabited(v_R_379_, v_A_u2081_380_, v_inst_381_, v_inst_382_, v_inst_383_);
lean_dec_ref(v_inst_383_);
lean_dec_ref(v_inst_382_);
lean_dec_ref(v_inst_381_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___redArg(lean_object* v_e_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_mathlib_Equiv_symm___redArg(v_e_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm(lean_object* v_R_387_, lean_object* v_A_u2081_388_, lean_object* v_A_u2082_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_e_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_Equiv_symm___redArg(v_e_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_symm___boxed(lean_object* v_R_397_, lean_object* v_A_u2081_398_, lean_object* v_A_u2082_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_e_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_mathlib_AlgEquiv_symm(v_R_397_, v_A_u2081_398_, v_A_u2082_399_, v_inst_400_, v_inst_401_, v_inst_402_, v_inst_403_, v_inst_404_, v_e_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_inst_402_);
lean_dec_ref(v_inst_401_);
lean_dec_ref(v_inst_400_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_apply___redArg(lean_object* v_e_407_, lean_object* v_a_408_){
_start:
{
lean_object* v_toFun_409_; lean_object* v___x_410_; 
v_toFun_409_ = lean_ctor_get(v_e_407_, 0);
lean_inc(v_toFun_409_);
lean_dec_ref(v_e_407_);
v___x_410_ = lean_apply_1(v_toFun_409_, v_a_408_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_apply(lean_object* v_R_411_, lean_object* v_A_u2081_412_, lean_object* v_A_u2082_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_e_419_, lean_object* v_a_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_mathlib_AlgEquiv_Simps_apply___redArg(v_e_419_, v_a_420_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_apply___boxed(lean_object* v_R_422_, lean_object* v_A_u2081_423_, lean_object* v_A_u2082_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_e_430_, lean_object* v_a_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_AlgEquiv_Simps_apply(v_R_422_, v_A_u2081_423_, v_A_u2082_424_, v_inst_425_, v_inst_426_, v_inst_427_, v_inst_428_, v_inst_429_, v_e_430_, v_a_431_);
lean_dec_ref(v_inst_429_);
lean_dec_ref(v_inst_428_);
lean_dec_ref(v_inst_427_);
lean_dec_ref(v_inst_426_);
lean_dec_ref(v_inst_425_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__0(lean_object* v_f_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_toFun_435_; lean_object* v___x_436_; 
v_toFun_435_ = lean_ctor_get(v_f_433_, 0);
lean_inc(v_toFun_435_);
lean_dec_ref(v_f_433_);
v___x_436_ = lean_apply_1(v_toFun_435_, v___y_434_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__1(lean_object* v_f_437_, lean_object* v___y_438_){
_start:
{
lean_object* v_invFun_439_; lean_object* v___x_440_; 
v_invFun_439_ = lean_ctor_get(v_f_437_, 1);
lean_inc(v_invFun_439_);
lean_dec_ref(v_f_437_);
v___x_440_ = lean_apply_1(v_invFun_439_, v___y_438_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___redArg(lean_object* v_e_446_){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_447_ = ((lean_object*)(lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__2));
v___x_448_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_447_, v_e_446_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv(lean_object* v_R_449_, lean_object* v_A_u2081_450_, lean_object* v_A_u2082_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_e_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_mathlib_AlgEquiv_Simps_toEquiv___redArg(v_e_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_toEquiv___boxed(lean_object* v_R_459_, lean_object* v_A_u2081_460_, lean_object* v_A_u2082_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_e_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_AlgEquiv_Simps_toEquiv(v_R_459_, v_A_u2081_460_, v_A_u2082_461_, v_inst_462_, v_inst_463_, v_inst_464_, v_inst_465_, v_inst_466_, v_e_467_);
lean_dec_ref(v_inst_466_);
lean_dec_ref(v_inst_465_);
lean_dec_ref(v_inst_464_);
lean_dec_ref(v_inst_463_);
lean_dec_ref(v_inst_462_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_symm__apply___redArg(lean_object* v_e_469_, lean_object* v_a_470_){
_start:
{
lean_object* v___x_471_; lean_object* v_toFun_472_; lean_object* v___x_473_; 
v___x_471_ = lp_mathlib_Equiv_symm___redArg(v_e_469_);
v_toFun_472_ = lean_ctor_get(v___x_471_, 0);
lean_inc(v_toFun_472_);
lean_dec_ref(v___x_471_);
v___x_473_ = lean_apply_1(v_toFun_472_, v_a_470_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_symm__apply(lean_object* v_R_474_, lean_object* v_A_u2081_475_, lean_object* v_A_u2082_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_e_482_, lean_object* v_a_483_){
_start:
{
lean_object* v___x_484_; 
v___x_484_ = lp_mathlib_AlgEquiv_Simps_symm__apply___redArg(v_e_482_, v_a_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_Simps_symm__apply___boxed(lean_object* v_R_485_, lean_object* v_A_u2081_486_, lean_object* v_A_u2082_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_e_493_, lean_object* v_a_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_mathlib_AlgEquiv_Simps_symm__apply(v_R_485_, v_A_u2081_486_, v_A_u2082_487_, v_inst_488_, v_inst_489_, v_inst_490_, v_inst_491_, v_inst_492_, v_e_493_, v_a_494_);
lean_dec_ref(v_inst_492_);
lean_dec_ref(v_inst_491_);
lean_dec_ref(v_inst_490_);
lean_dec_ref(v_inst_489_);
lean_dec_ref(v_inst_488_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_trans___redArg(lean_object* v_e_u2081_496_, lean_object* v_e_u2082_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_Equiv_trans___redArg(v_e_u2081_496_, v_e_u2082_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_trans(lean_object* v_R_499_, lean_object* v_A_u2081_500_, lean_object* v_A_u2082_501_, lean_object* v_A_u2083_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_e_u2081_510_, lean_object* v_e_u2082_511_){
_start:
{
lean_object* v___x_512_; 
v___x_512_ = lp_mathlib_Equiv_trans___redArg(v_e_u2081_510_, v_e_u2082_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_trans___boxed(lean_object* v_R_513_, lean_object* v_A_u2081_514_, lean_object* v_A_u2082_515_, lean_object* v_A_u2083_516_, lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_e_u2081_524_, lean_object* v_e_u2082_525_){
_start:
{
lean_object* v_res_526_; 
v_res_526_ = lp_mathlib_AlgEquiv_trans(v_R_513_, v_A_u2081_514_, v_A_u2082_515_, v_A_u2083_516_, v_inst_517_, v_inst_518_, v_inst_519_, v_inst_520_, v_inst_521_, v_inst_522_, v_inst_523_, v_e_u2081_524_, v_e_u2082_525_);
lean_dec_ref(v_inst_523_);
lean_dec_ref(v_inst_522_);
lean_dec_ref(v_inst_521_);
lean_dec_ref(v_inst_520_);
lean_dec_ref(v_inst_519_);
lean_dec_ref(v_inst_518_);
lean_dec_ref(v_inst_517_);
return v_res_526_;
}
}
static lean_object* _init_lp_mathlib_AlgEquiv_cast___closed__0(void){
_start:
{
lean_object* v___x_527_; 
v___x_527_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_cast(lean_object* v_R_528_, lean_object* v_inst_529_, lean_object* v_00_u03b9_530_, lean_object* v_A_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_i_534_, lean_object* v_j_535_, lean_object* v_h_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lean_obj_once(&lp_mathlib_AlgEquiv_cast___closed__0, &lp_mathlib_AlgEquiv_cast___closed__0_once, _init_lp_mathlib_AlgEquiv_cast___closed__0);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_cast___boxed(lean_object* v_R_538_, lean_object* v_inst_539_, lean_object* v_00_u03b9_540_, lean_object* v_A_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_i_544_, lean_object* v_j_545_, lean_object* v_h_546_){
_start:
{
lean_object* v_res_547_; 
v_res_547_ = lp_mathlib_AlgEquiv_cast(v_R_538_, v_inst_539_, v_00_u03b9_540_, v_A_541_, v_inst_542_, v_inst_543_, v_i_544_, v_j_545_, v_h_546_);
lean_dec(v_j_545_);
lean_dec(v_i_544_);
lean_dec_ref(v_inst_543_);
lean_dec_ref(v_inst_542_);
lean_dec_ref(v_inst_539_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg___lam__0(lean_object* v_e_u2082_548_, lean_object* v_e_u2081_549_, lean_object* v_f_550_, lean_object* v___y_551_){
_start:
{
lean_object* v_toFun_552_; lean_object* v___x_553_; lean_object* v_toFun_554_; lean_object* v___x_555_; lean_object* v___x_43__overap_556_; lean_object* v___x_557_; 
v_toFun_552_ = lean_ctor_get(v_e_u2082_548_, 0);
lean_inc(v_toFun_552_);
lean_dec_ref(v_e_u2082_548_);
v___x_553_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_549_);
v_toFun_554_ = lean_ctor_get(v___x_553_, 0);
lean_inc(v_toFun_554_);
lean_dec_ref(v___x_553_);
v___x_555_ = lp_mathlib_AlgHom_comp___redArg(v_toFun_552_, v_f_550_);
v___x_43__overap_556_ = lp_mathlib_AlgHom_comp___redArg(v___x_555_, v_toFun_554_);
v___x_557_ = lean_apply_1(v___x_43__overap_556_, v___y_551_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg___lam__1(lean_object* v_e_u2082_558_, lean_object* v_e_u2081_559_, lean_object* v_f_560_, lean_object* v___y_561_){
_start:
{
lean_object* v___x_562_; lean_object* v_toFun_563_; lean_object* v_toFun_564_; lean_object* v___x_565_; lean_object* v___x_47__overap_566_; lean_object* v___x_567_; 
v___x_562_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_558_);
v_toFun_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc(v_toFun_563_);
lean_dec_ref(v___x_562_);
v_toFun_564_ = lean_ctor_get(v_e_u2081_559_, 0);
lean_inc(v_toFun_564_);
lean_dec_ref(v_e_u2081_559_);
v___x_565_ = lp_mathlib_AlgHom_comp___redArg(v_toFun_563_, v_f_560_);
v___x_47__overap_566_ = lp_mathlib_AlgHom_comp___redArg(v___x_565_, v_toFun_564_);
v___x_567_ = lean_apply_1(v___x_47__overap_566_, v___y_561_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg(lean_object* v_e_u2081_568_, lean_object* v_e_u2082_569_){
_start:
{
lean_object* v___f_570_; lean_object* v___f_571_; lean_object* v___x_572_; 
lean_inc_ref(v_e_u2081_568_);
lean_inc_ref(v_e_u2082_569_);
v___f_570_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_arrowCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_570_, 0, v_e_u2082_569_);
lean_closure_set(v___f_570_, 1, v_e_u2081_568_);
v___f_571_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_arrowCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_571_, 0, v_e_u2082_569_);
lean_closure_set(v___f_571_, 1, v_e_u2081_568_);
v___x_572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_572_, 0, v___f_570_);
lean_ctor_set(v___x_572_, 1, v___f_571_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr(lean_object* v_R_573_, lean_object* v_A_u2081_574_, lean_object* v_A_u2082_575_, lean_object* v_A_u2081_x27_576_, lean_object* v_A_u2082_x27_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_e_u2081_587_, lean_object* v_e_u2082_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_mathlib_AlgEquiv_arrowCongr___redArg(v_e_u2081_587_, v_e_u2082_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_arrowCongr___boxed(lean_object* v_R_590_, lean_object* v_A_u2081_591_, lean_object* v_A_u2082_592_, lean_object* v_A_u2081_x27_593_, lean_object* v_A_u2082_x27_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_inst_603_, lean_object* v_e_u2081_604_, lean_object* v_e_u2082_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_AlgEquiv_arrowCongr(v_R_590_, v_A_u2081_591_, v_A_u2082_592_, v_A_u2081_x27_593_, v_A_u2082_x27_594_, v_inst_595_, v_inst_596_, v_inst_597_, v_inst_598_, v_inst_599_, v_inst_600_, v_inst_601_, v_inst_602_, v_inst_603_, v_e_u2081_604_, v_e_u2082_605_);
lean_dec_ref(v_inst_603_);
lean_dec_ref(v_inst_602_);
lean_dec_ref(v_inst_601_);
lean_dec_ref(v_inst_600_);
lean_dec_ref(v_inst_599_);
lean_dec_ref(v_inst_598_);
lean_dec_ref(v_inst_597_);
lean_dec_ref(v_inst_596_);
lean_dec_ref(v_inst_595_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg___lam__0(lean_object* v_e_607_, lean_object* v_e_x27_608_, lean_object* v_00_u03c8_609_){
_start:
{
lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; 
v___x_610_ = lp_mathlib_Equiv_symm___redArg(v_e_607_);
v___x_611_ = lp_mathlib_Equiv_trans___redArg(v_00_u03c8_609_, v_e_x27_608_);
v___x_612_ = lp_mathlib_Equiv_trans___redArg(v___x_610_, v___x_611_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg___lam__1(lean_object* v_e_x27_613_, lean_object* v_e_614_, lean_object* v_00_u03c8_615_){
_start:
{
lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; 
v___x_616_ = lp_mathlib_Equiv_symm___redArg(v_e_x27_613_);
v___x_617_ = lp_mathlib_Equiv_trans___redArg(v_00_u03c8_615_, v___x_616_);
v___x_618_ = lp_mathlib_Equiv_trans___redArg(v_e_614_, v___x_617_);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg(lean_object* v_e_619_, lean_object* v_e_x27_620_){
_start:
{
lean_object* v___f_621_; lean_object* v___f_622_; lean_object* v___x_623_; 
lean_inc_ref(v_e_x27_620_);
lean_inc_ref(v_e_619_);
v___f_621_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_equivCongr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_621_, 0, v_e_619_);
lean_closure_set(v___f_621_, 1, v_e_x27_620_);
v___f_622_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_equivCongr___redArg___lam__1), 3, 2);
lean_closure_set(v___f_622_, 0, v_e_x27_620_);
lean_closure_set(v___f_622_, 1, v_e_619_);
v___x_623_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_623_, 0, v___f_621_);
lean_ctor_set(v___x_623_, 1, v___f_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr(lean_object* v_R_624_, lean_object* v_A_u2081_625_, lean_object* v_A_u2082_626_, lean_object* v_A_u2081_x27_627_, lean_object* v_A_u2082_x27_628_, lean_object* v_inst_629_, lean_object* v_inst_630_, lean_object* v_inst_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_e_638_, lean_object* v_e_x27_639_){
_start:
{
lean_object* v___x_640_; 
v___x_640_ = lp_mathlib_AlgEquiv_equivCongr___redArg(v_e_638_, v_e_x27_639_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_equivCongr___boxed(lean_object* v_R_641_, lean_object* v_A_u2081_642_, lean_object* v_A_u2082_643_, lean_object* v_A_u2081_x27_644_, lean_object* v_A_u2082_x27_645_, lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_e_655_, lean_object* v_e_x27_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_mathlib_AlgEquiv_equivCongr(v_R_641_, v_A_u2081_642_, v_A_u2082_643_, v_A_u2081_x27_644_, v_A_u2082_x27_645_, v_inst_646_, v_inst_647_, v_inst_648_, v_inst_649_, v_inst_650_, v_inst_651_, v_inst_652_, v_inst_653_, v_inst_654_, v_e_655_, v_e_x27_656_);
lean_dec_ref(v_inst_654_);
lean_dec_ref(v_inst_653_);
lean_dec_ref(v_inst_652_);
lean_dec_ref(v_inst_651_);
lean_dec_ref(v_inst_650_);
lean_dec_ref(v_inst_649_);
lean_dec_ref(v_inst_648_);
lean_dec_ref(v_inst_647_);
lean_dec_ref(v_inst_646_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg___lam__0(lean_object* v_f_658_, lean_object* v___y_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lean_apply_1(v_f_658_, v___y_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg___lam__1(lean_object* v_g_661_, lean_object* v___y_662_){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = lean_apply_1(v_g_661_, v___y_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg(lean_object* v_f_664_, lean_object* v_g_665_){
_start:
{
lean_object* v___f_666_; lean_object* v___f_667_; lean_object* v___x_668_; 
v___f_666_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_ofAlgHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_666_, 0, v_f_664_);
v___f_667_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_ofAlgHom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_667_, 0, v_g_665_);
v___x_668_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_668_, 0, v___f_666_);
lean_ctor_set(v___x_668_, 1, v___f_667_);
return v___x_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom(lean_object* v_R_669_, lean_object* v_A_u2081_670_, lean_object* v_A_u2082_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_f_677_, lean_object* v_g_678_, lean_object* v_h_u2081_679_, lean_object* v_h_u2082_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lp_mathlib_AlgEquiv_ofAlgHom___redArg(v_f_677_, v_g_678_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofAlgHom___boxed(lean_object* v_R_682_, lean_object* v_A_u2081_683_, lean_object* v_A_u2082_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_f_690_, lean_object* v_g_691_, lean_object* v_h_u2081_692_, lean_object* v_h_u2082_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_mathlib_AlgEquiv_ofAlgHom(v_R_682_, v_A_u2081_683_, v_A_u2082_684_, v_inst_685_, v_inst_686_, v_inst_687_, v_inst_688_, v_inst_689_, v_f_690_, v_g_691_, v_h_u2081_692_, v_h_u2082_693_);
lean_dec_ref(v_inst_689_);
lean_dec_ref(v_inst_688_);
lean_dec_ref(v_inst_687_);
lean_dec_ref(v_inst_686_);
lean_dec_ref(v_inst_685_);
return v_res_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMap___redArg(lean_object* v_e_695_){
_start:
{
lean_object* v___x_696_; lean_object* v_toLinearMap_697_; 
v___x_696_ = lp_mathlib_AlgEquiv_toLinearEquiv___redArg(v_e_695_);
v_toLinearMap_697_ = lean_ctor_get(v___x_696_, 0);
lean_inc(v_toLinearMap_697_);
lean_dec_ref(v___x_696_);
return v_toLinearMap_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMap(lean_object* v_R_698_, lean_object* v_A_u2081_699_, lean_object* v_A_u2082_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_inst_704_, lean_object* v_inst_705_, lean_object* v_e_706_){
_start:
{
lean_object* v___x_707_; lean_object* v_toLinearMap_708_; 
v___x_707_ = lp_mathlib_AlgEquiv_toLinearEquiv___redArg(v_e_706_);
v_toLinearMap_708_ = lean_ctor_get(v___x_707_, 0);
lean_inc(v_toLinearMap_708_);
lean_dec_ref(v___x_707_);
return v_toLinearMap_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMap___boxed(lean_object* v_R_709_, lean_object* v_A_u2081_710_, lean_object* v_A_u2082_711_, lean_object* v_inst_712_, lean_object* v_inst_713_, lean_object* v_inst_714_, lean_object* v_inst_715_, lean_object* v_inst_716_, lean_object* v_e_717_){
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_mathlib_AlgEquiv_toLinearMap(v_R_709_, v_A_u2081_710_, v_A_u2082_711_, v_inst_712_, v_inst_713_, v_inst_714_, v_inst_715_, v_inst_716_, v_e_717_);
lean_dec_ref(v_inst_716_);
lean_dec_ref(v_inst_715_);
lean_dec_ref(v_inst_714_);
lean_dec_ref(v_inst_713_);
lean_dec_ref(v_inst_712_);
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg___lam__0(lean_object* v_l_719_, lean_object* v___y_720_){
_start:
{
lean_object* v_toLinearMap_721_; lean_object* v___x_722_; 
v_toLinearMap_721_ = lean_ctor_get(v_l_719_, 0);
lean_inc(v_toLinearMap_721_);
lean_dec_ref(v_l_719_);
v___x_722_ = lean_apply_1(v_toLinearMap_721_, v___y_720_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg___lam__1(lean_object* v___x_723_, lean_object* v___y_724_){
_start:
{
lean_object* v_toLinearMap_725_; lean_object* v___x_726_; 
v_toLinearMap_725_ = lean_ctor_get(v___x_723_, 0);
lean_inc(v_toLinearMap_725_);
lean_dec_ref(v___x_723_);
v___x_726_ = lean_apply_1(v_toLinearMap_725_, v___y_724_);
return v___x_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(lean_object* v_l_727_){
_start:
{
lean_object* v___f_728_; lean_object* v___x_729_; lean_object* v___f_730_; lean_object* v___x_731_; 
lean_inc_ref(v_l_727_);
v___f_728_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_ofLinearEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_728_, 0, v_l_727_);
v___x_729_ = lp_mathlib_LinearEquiv_symm___redArg(v_l_727_);
v___f_730_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_ofLinearEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_730_, 0, v___x_729_);
v___x_731_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_731_, 0, v___f_728_);
lean_ctor_set(v___x_731_, 1, v___f_730_);
return v___x_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv(lean_object* v_R_732_, lean_object* v_A_u2081_733_, lean_object* v_A_u2082_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_l_740_, lean_object* v_map__one_741_, lean_object* v_map__mul_742_){
_start:
{
lean_object* v___x_743_; 
v___x_743_ = lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(v_l_740_);
return v___x_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv___boxed(lean_object* v_R_744_, lean_object* v_A_u2081_745_, lean_object* v_A_u2082_746_, lean_object* v_inst_747_, lean_object* v_inst_748_, lean_object* v_inst_749_, lean_object* v_inst_750_, lean_object* v_inst_751_, lean_object* v_l_752_, lean_object* v_map__one_753_, lean_object* v_map__mul_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_AlgEquiv_ofLinearEquiv(v_R_744_, v_A_u2081_745_, v_A_u2082_746_, v_inst_747_, v_inst_748_, v_inst_749_, v_inst_750_, v_inst_751_, v_l_752_, v_map__one_753_, v_map__mul_754_);
lean_dec_ref(v_inst_751_);
lean_dec_ref(v_inst_750_);
lean_dec_ref(v_inst_749_);
lean_dec_ref(v_inst_748_);
lean_dec_ref(v_inst_747_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux___redArg(lean_object* v_l_756_){
_start:
{
lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_757_ = lp_mathlib_AlgEquiv_ofLinearEquiv___redArg(v_l_756_);
v___x_758_ = lp_mathlib_Equiv_symm___redArg(v___x_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux(lean_object* v_R_759_, lean_object* v_A_u2081_760_, lean_object* v_A_u2082_761_, lean_object* v_inst_762_, lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_l_767_, lean_object* v_map__one_768_, lean_object* v_map__mul_769_){
_start:
{
lean_object* v___x_770_; 
v___x_770_ = lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux___redArg(v_l_767_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux___boxed(lean_object* v_R_771_, lean_object* v_A_u2081_772_, lean_object* v_A_u2082_773_, lean_object* v_inst_774_, lean_object* v_inst_775_, lean_object* v_inst_776_, lean_object* v_inst_777_, lean_object* v_inst_778_, lean_object* v_l_779_, lean_object* v_map__one_780_, lean_object* v_map__mul_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_mathlib_AlgEquiv_ofLinearEquiv__symm_aux(v_R_771_, v_A_u2081_772_, v_A_u2082_773_, v_inst_774_, v_inst_775_, v_inst_776_, v_inst_777_, v_inst_778_, v_l_779_, v_map__one_780_, v_map__mul_781_);
lean_dec_ref(v_inst_778_);
lean_dec_ref(v_inst_777_);
lean_dec_ref(v_inst_776_);
lean_dec_ref(v_inst_775_);
lean_dec_ref(v_inst_774_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___redArg___lam__1(lean_object* v___x_783_, lean_object* v___y_784_){
_start:
{
lean_object* v_toFun_785_; lean_object* v___x_786_; 
v_toFun_785_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_toFun_785_);
lean_dec_ref(v___x_783_);
v___x_786_ = lean_apply_1(v_toFun_785_, v___y_784_);
return v___x_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___redArg(lean_object* v_f_787_){
_start:
{
lean_object* v___f_788_; lean_object* v___x_789_; lean_object* v___f_790_; lean_object* v___x_791_; 
lean_inc_ref(v_f_787_);
v___f_788_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_788_, 0, v_f_787_);
v___x_789_ = lp_mathlib_Equiv_symm___redArg(v_f_787_);
v___f_790_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_ofRingEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_790_, 0, v___x_789_);
v___x_791_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_791_, 0, v___f_788_);
lean_ctor_set(v___x_791_, 1, v___f_790_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv(lean_object* v_R_792_, lean_object* v_A_u2081_793_, lean_object* v_A_u2082_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_inst_798_, lean_object* v_inst_799_, lean_object* v_f_800_, lean_object* v_hf_801_){
_start:
{
lean_object* v___x_802_; 
v___x_802_ = lp_mathlib_AlgEquiv_ofRingEquiv___redArg(v_f_800_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___boxed(lean_object* v_R_803_, lean_object* v_A_u2081_804_, lean_object* v_A_u2082_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_, lean_object* v_inst_809_, lean_object* v_inst_810_, lean_object* v_f_811_, lean_object* v_hf_812_){
_start:
{
lean_object* v_res_813_; 
v_res_813_ = lp_mathlib_AlgEquiv_ofRingEquiv(v_R_803_, v_A_u2081_804_, v_A_u2082_805_, v_inst_806_, v_inst_807_, v_inst_808_, v_inst_809_, v_inst_810_, v_f_811_, v_hf_812_);
lean_dec_ref(v_inst_810_);
lean_dec_ref(v_inst_809_);
lean_dec_ref(v_inst_808_);
lean_dec_ref(v_inst_807_);
lean_dec_ref(v_inst_806_);
return v_res_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_aut___redArg___lam__0(lean_object* v_00_u03d5_814_, lean_object* v_00_u03c8_815_){
_start:
{
lean_object* v___x_816_; 
v___x_816_ = lp_mathlib_Equiv_trans___redArg(v_00_u03c8_815_, v_00_u03d5_814_);
return v___x_816_;
}
}
static lean_object* _init_lp_mathlib_AlgEquiv_aut___redArg___closed__1(void){
_start:
{
lean_object* v___x_818_; lean_object* v___f_819_; lean_object* v___x_820_; 
v___x_818_ = lean_obj_once(&lp_mathlib_AlgEquiv_refl___closed__0, &lp_mathlib_AlgEquiv_refl___closed__0_once, _init_lp_mathlib_AlgEquiv_refl___closed__0);
v___f_819_ = ((lean_object*)(lp_mathlib_AlgEquiv_aut___redArg___closed__0));
v___x_820_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_820_, 0, lean_box(0));
lean_closure_set(v___x_820_, 1, v___f_819_);
lean_closure_set(v___x_820_, 2, v___x_818_);
return v___x_820_;
}
}
static lean_object* _init_lp_mathlib_AlgEquiv_aut___redArg___closed__2(void){
_start:
{
lean_object* v___x_821_; lean_object* v___f_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_821_ = lean_obj_once(&lp_mathlib_AlgEquiv_aut___redArg___closed__1, &lp_mathlib_AlgEquiv_aut___redArg___closed__1_once, _init_lp_mathlib_AlgEquiv_aut___redArg___closed__1);
v___f_822_ = ((lean_object*)(lp_mathlib_AlgEquiv_aut___redArg___closed__0));
v___x_823_ = lean_obj_once(&lp_mathlib_AlgEquiv_refl___closed__0, &lp_mathlib_AlgEquiv_refl___closed__0_once, _init_lp_mathlib_AlgEquiv_refl___closed__0);
v___x_824_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_824_, 0, v___x_823_);
lean_ctor_set(v___x_824_, 1, v___f_822_);
lean_ctor_set(v___x_824_, 2, v___x_821_);
return v___x_824_;
}
}
static lean_object* _init_lp_mathlib_AlgEquiv_aut___redArg___closed__3(void){
_start:
{
lean_object* v___f_825_; lean_object* v___x_826_; lean_object* v___x_827_; 
v___f_825_ = ((lean_object*)(lp_mathlib_AlgEquiv_aut___redArg___closed__0));
v___x_826_ = lean_obj_once(&lp_mathlib_AlgEquiv_refl___closed__0, &lp_mathlib_AlgEquiv_refl___closed__0_once, _init_lp_mathlib_AlgEquiv_refl___closed__0);
v___x_827_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_827_, 0, lean_box(0));
lean_closure_set(v___x_827_, 1, v___x_826_);
lean_closure_set(v___x_827_, 2, v___f_825_);
return v___x_827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_aut___redArg(lean_object* v_inst_828_, lean_object* v_inst_829_, lean_object* v_inst_830_){
_start:
{
lean_object* v___f_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v___f_831_ = ((lean_object*)(lp_mathlib_AlgEquiv_aut___redArg___closed__0));
v___x_832_ = lean_obj_once(&lp_mathlib_AlgEquiv_refl___closed__0, &lp_mathlib_AlgEquiv_refl___closed__0_once, _init_lp_mathlib_AlgEquiv_refl___closed__0);
v___x_833_ = lean_obj_once(&lp_mathlib_AlgEquiv_aut___redArg___closed__2, &lp_mathlib_AlgEquiv_aut___redArg___closed__2_once, _init_lp_mathlib_AlgEquiv_aut___redArg___closed__2);
lean_inc_ref(v_inst_830_);
lean_inc_ref(v_inst_829_);
v___x_834_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_symm___boxed), 9, 8);
lean_closure_set(v___x_834_, 0, lean_box(0));
lean_closure_set(v___x_834_, 1, lean_box(0));
lean_closure_set(v___x_834_, 2, lean_box(0));
lean_closure_set(v___x_834_, 3, v_inst_828_);
lean_closure_set(v___x_834_, 4, v_inst_829_);
lean_closure_set(v___x_834_, 5, v_inst_829_);
lean_closure_set(v___x_834_, 6, v_inst_830_);
lean_closure_set(v___x_834_, 7, v_inst_830_);
lean_inc_ref_n(v___x_834_, 2);
v___x_835_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_835_, 0, lean_box(0));
lean_closure_set(v___x_835_, 1, v___x_833_);
lean_closure_set(v___x_835_, 2, v___x_834_);
v___x_836_ = lean_obj_once(&lp_mathlib_AlgEquiv_aut___redArg___closed__3, &lp_mathlib_AlgEquiv_aut___redArg___closed__3_once, _init_lp_mathlib_AlgEquiv_aut___redArg___closed__3);
v___x_837_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_837_, 0, lean_box(0));
lean_closure_set(v___x_837_, 1, v___x_832_);
lean_closure_set(v___x_837_, 2, v___f_831_);
lean_closure_set(v___x_837_, 3, v___x_834_);
lean_closure_set(v___x_837_, 4, v___x_836_);
v___x_838_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_838_, 0, v___x_833_);
lean_ctor_set(v___x_838_, 1, v___x_834_);
lean_ctor_set(v___x_838_, 2, v___x_835_);
lean_ctor_set(v___x_838_, 3, v___x_837_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_aut(lean_object* v_R_839_, lean_object* v_A_u2081_840_, lean_object* v_inst_841_, lean_object* v_inst_842_, lean_object* v_inst_843_){
_start:
{
lean_object* v___x_844_; 
v___x_844_ = lp_mathlib_AlgEquiv_aut___redArg(v_inst_841_, v_inst_842_, v_inst_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___redArg___lam__0(lean_object* v_00_u03d5_845_, lean_object* v_00_u03c8_846_){
_start:
{
lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; 
lean_inc_ref(v_00_u03d5_845_);
v___x_847_ = lp_mathlib_Equiv_symm___redArg(v_00_u03d5_845_);
v___x_848_ = lp_mathlib_Equiv_trans___redArg(v_00_u03c8_846_, v_00_u03d5_845_);
v___x_849_ = lp_mathlib_Equiv_trans___redArg(v___x_847_, v___x_848_);
return v___x_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___redArg___lam__1(lean_object* v_00_u03d5_850_, lean_object* v_00_u03c8_851_){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; 
lean_inc_ref(v_00_u03d5_850_);
v___x_852_ = lp_mathlib_Equiv_symm___redArg(v_00_u03d5_850_);
v___x_853_ = lp_mathlib_Equiv_trans___redArg(v_00_u03c8_851_, v___x_852_);
v___x_854_ = lp_mathlib_Equiv_trans___redArg(v_00_u03d5_850_, v___x_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___redArg(lean_object* v_00_u03d5_855_){
_start:
{
lean_object* v___f_856_; lean_object* v___f_857_; lean_object* v___x_858_; 
lean_inc_ref(v_00_u03d5_855_);
v___f_856_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_autCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_856_, 0, v_00_u03d5_855_);
v___f_857_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_autCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_857_, 0, v_00_u03d5_855_);
v___x_858_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_858_, 0, v___f_856_);
lean_ctor_set(v___x_858_, 1, v___f_857_);
return v___x_858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr(lean_object* v_R_859_, lean_object* v_A_u2081_860_, lean_object* v_A_u2082_861_, lean_object* v_inst_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_00_u03d5_867_){
_start:
{
lean_object* v___x_868_; 
v___x_868_ = lp_mathlib_AlgEquiv_autCongr___redArg(v_00_u03d5_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_autCongr___boxed(lean_object* v_R_869_, lean_object* v_A_u2081_870_, lean_object* v_A_u2082_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_00_u03d5_877_){
_start:
{
lean_object* v_res_878_; 
v_res_878_ = lp_mathlib_AlgEquiv_autCongr(v_R_869_, v_A_u2081_870_, v_A_u2082_871_, v_inst_872_, v_inst_873_, v_inst_874_, v_inst_875_, v_inst_876_, v_00_u03d5_877_);
lean_dec_ref(v_inst_876_);
lean_dec_ref(v_inst_875_);
lean_dec_ref(v_inst_874_);
lean_dec_ref(v_inst_873_);
lean_dec_ref(v_inst_872_);
return v_res_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction___lam__0(lean_object* v_x1_879_, lean_object* v_x2_880_){
_start:
{
lean_object* v_toFun_881_; lean_object* v___x_882_; 
v_toFun_881_ = lean_ctor_get(v_x1_879_, 0);
lean_inc(v_toFun_881_);
lean_dec_ref(v_x1_879_);
v___x_882_ = lean_apply_1(v_toFun_881_, v_x2_880_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction(lean_object* v_R_884_, lean_object* v_A_u2081_885_, lean_object* v_inst_886_, lean_object* v_inst_887_, lean_object* v_inst_888_){
_start:
{
lean_object* v___f_889_; 
v___f_889_ = ((lean_object*)(lp_mathlib_AlgEquiv_applyMulSemiringAction___closed__0));
return v___f_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction___boxed(lean_object* v_R_890_, lean_object* v_A_u2081_891_, lean_object* v_inst_892_, lean_object* v_inst_893_, lean_object* v_inst_894_){
_start:
{
lean_object* v_res_895_; 
v_res_895_ = lp_mathlib_AlgEquiv_applyMulSemiringAction(v_R_890_, v_A_u2081_891_, v_inst_892_, v_inst_893_, v_inst_894_);
lean_dec_ref(v_inst_894_);
lean_dec_ref(v_inst_893_);
lean_dec_ref(v_inst_892_);
return v_res_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___lam__1(lean_object* v_f_896_, lean_object* v___y_897_){
_start:
{
lean_object* v___f_898_; lean_object* v___x_899_; 
v___f_898_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_898_, 0, v_f_896_);
v___x_899_ = lp_mathlib_Units_map___redArg___lam__0(v___f_898_, v___y_897_);
return v___x_899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits(lean_object* v_R_901_, lean_object* v_A_u2081_902_, lean_object* v_inst_903_, lean_object* v_inst_904_, lean_object* v_inst_905_){
_start:
{
lean_object* v___f_906_; 
v___f_906_ = ((lean_object*)(lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___closed__0));
return v___f_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_instMulDistribMulActionUnits___boxed(lean_object* v_R_907_, lean_object* v_A_u2081_908_, lean_object* v_inst_909_, lean_object* v_inst_910_, lean_object* v_inst_911_){
_start:
{
lean_object* v_res_912_; 
v_res_912_ = lp_mathlib_AlgEquiv_instMulDistribMulActionUnits(v_R_907_, v_A_u2081_908_, v_inst_909_, v_inst_910_, v_inst_911_);
lean_dec_ref(v_inst_911_);
lean_dec_ref(v_inst_910_);
lean_dec_ref(v_inst_909_);
return v_res_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHomHom___redArg(lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_inst_915_){
_start:
{
lean_object* v___x_916_; 
lean_inc_ref(v_inst_915_);
lean_inc_ref(v_inst_914_);
v___x_916_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toAlgHom___boxed), 9, 8);
lean_closure_set(v___x_916_, 0, lean_box(0));
lean_closure_set(v___x_916_, 1, lean_box(0));
lean_closure_set(v___x_916_, 2, lean_box(0));
lean_closure_set(v___x_916_, 3, v_inst_913_);
lean_closure_set(v___x_916_, 4, v_inst_914_);
lean_closure_set(v___x_916_, 5, v_inst_914_);
lean_closure_set(v___x_916_, 6, v_inst_915_);
lean_closure_set(v___x_916_, 7, v_inst_915_);
return v___x_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toAlgHomHom(lean_object* v_R_917_, lean_object* v_A_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_inst_921_){
_start:
{
lean_object* v___x_922_; 
lean_inc_ref(v_inst_921_);
lean_inc_ref(v_inst_920_);
v___x_922_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toAlgHom___boxed), 9, 8);
lean_closure_set(v___x_922_, 0, lean_box(0));
lean_closure_set(v___x_922_, 1, lean_box(0));
lean_closure_set(v___x_922_, 2, lean_box(0));
lean_closure_set(v___x_922_, 3, v_inst_919_);
lean_closure_set(v___x_922_, 4, v_inst_920_);
lean_closure_set(v___x_922_, 5, v_inst_920_);
lean_closure_set(v___x_922_, 6, v_inst_921_);
lean_closure_set(v___x_922_, 7, v_inst_921_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMapHom___redArg(lean_object* v_inst_923_, lean_object* v_inst_924_, lean_object* v_inst_925_){
_start:
{
lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___f_928_; 
lean_inc_ref_n(v_inst_925_, 3);
lean_inc_ref_n(v_inst_924_, 3);
lean_inc_ref(v_inst_923_);
v___x_926_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_926_, 0, lean_box(0));
lean_closure_set(v___x_926_, 1, lean_box(0));
lean_closure_set(v___x_926_, 2, lean_box(0));
lean_closure_set(v___x_926_, 3, v_inst_923_);
lean_closure_set(v___x_926_, 4, v_inst_924_);
lean_closure_set(v___x_926_, 5, v_inst_924_);
lean_closure_set(v___x_926_, 6, v_inst_925_);
lean_closure_set(v___x_926_, 7, v_inst_925_);
v___x_927_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toAlgHom___boxed), 9, 8);
lean_closure_set(v___x_927_, 0, lean_box(0));
lean_closure_set(v___x_927_, 1, lean_box(0));
lean_closure_set(v___x_927_, 2, lean_box(0));
lean_closure_set(v___x_927_, 3, v_inst_923_);
lean_closure_set(v___x_927_, 4, v_inst_924_);
lean_closure_set(v___x_927_, 5, v_inst_924_);
lean_closure_set(v___x_927_, 6, v_inst_925_);
lean_closure_set(v___x_927_, 7, v_inst_925_);
v___f_928_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_928_, 0, v___x_927_);
lean_closure_set(v___f_928_, 1, v___x_926_);
return v___f_928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toLinearMapHom(lean_object* v_R_929_, lean_object* v_A_930_, lean_object* v_inst_931_, lean_object* v_inst_932_, lean_object* v_inst_933_){
_start:
{
lean_object* v___x_934_; 
v___x_934_ = lp_mathlib_AlgEquiv_toLinearMapHom___redArg(v_inst_931_, v_inst_932_, v_inst_933_);
return v___x_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__0(lean_object* v_inv_935_, lean_object* v___y_936_){
_start:
{
lean_object* v___x_937_; 
v___x_937_ = lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0(v_inv_935_, v___y_936_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__1(lean_object* v_f_938_){
_start:
{
lean_object* v_val_939_; lean_object* v_inv_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_948_; 
v_val_939_ = lean_ctor_get(v_f_938_, 0);
v_inv_940_ = lean_ctor_get(v_f_938_, 1);
v_isSharedCheck_948_ = !lean_is_exclusive(v_f_938_);
if (v_isSharedCheck_948_ == 0)
{
v___x_942_ = v_f_938_;
v_isShared_943_ = v_isSharedCheck_948_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_inv_940_);
lean_inc(v_val_939_);
lean_dec(v_f_938_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_948_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v___f_944_; lean_object* v___x_946_; 
v___f_944_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__0), 2, 1);
lean_closure_set(v___f_944_, 0, v_inv_940_);
if (v_isShared_943_ == 0)
{
lean_ctor_set(v___x_942_, 1, v___f_944_);
v___x_946_ = v___x_942_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v_val_939_);
lean_ctor_set(v_reuseFailAlloc_947_, 1, v___f_944_);
v___x_946_ = v_reuseFailAlloc_947_;
goto v_reusejp_945_;
}
v_reusejp_945_:
{
return v___x_946_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___lam__2(lean_object* v_f_949_){
_start:
{
lean_object* v_toFun_950_; lean_object* v___x_951_; lean_object* v_toFun_952_; lean_object* v___x_954_; uint8_t v_isShared_955_; uint8_t v_isSharedCheck_959_; 
v_toFun_950_ = lean_ctor_get(v_f_949_, 0);
lean_inc(v_toFun_950_);
v___x_951_ = lp_mathlib_Equiv_symm___redArg(v_f_949_);
v_toFun_952_ = lean_ctor_get(v___x_951_, 0);
v_isSharedCheck_959_ = !lean_is_exclusive(v___x_951_);
if (v_isSharedCheck_959_ == 0)
{
lean_object* v_unused_960_; 
v_unused_960_ = lean_ctor_get(v___x_951_, 1);
lean_dec(v_unused_960_);
v___x_954_ = v___x_951_;
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
else
{
lean_inc(v_toFun_952_);
lean_dec(v___x_951_);
v___x_954_ = lean_box(0);
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
v_resetjp_953_:
{
lean_object* v___x_957_; 
if (v_isShared_955_ == 0)
{
lean_ctor_set(v___x_954_, 1, v_toFun_952_);
lean_ctor_set(v___x_954_, 0, v_toFun_950_);
v___x_957_ = v___x_954_;
goto v_reusejp_956_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v_toFun_950_);
lean_ctor_set(v_reuseFailAlloc_958_, 1, v_toFun_952_);
v___x_957_ = v_reuseFailAlloc_958_;
goto v_reusejp_956_;
}
v_reusejp_956_:
{
return v___x_957_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv(lean_object* v_R_966_, lean_object* v_S_967_, lean_object* v_inst_968_, lean_object* v_inst_969_, lean_object* v_inst_970_){
_start:
{
lean_object* v___x_971_; 
v___x_971_ = ((lean_object*)(lp_mathlib_AlgEquiv_algHomUnitsEquiv___closed__2));
return v___x_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_algHomUnitsEquiv___boxed(lean_object* v_R_972_, lean_object* v_S_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_inst_976_){
_start:
{
lean_object* v_res_977_; 
v_res_977_ = lp_mathlib_AlgEquiv_algHomUnitsEquiv(v_R_972_, v_S_973_, v_inst_974_, v_inst_975_, v_inst_976_);
lean_dec_ref(v_inst_976_);
lean_dec_ref(v_inst_975_);
lean_dec_ref(v_inst_974_);
return v_res_977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNatAlgEquiv___redArg(lean_object* v_f_978_){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = ((lean_object*)(lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__2));
v___x_980_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_979_, v_f_978_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNatAlgEquiv(lean_object* v_R_981_, lean_object* v_S_982_, lean_object* v_inst_983_, lean_object* v_inst_984_, lean_object* v_f_985_){
_start:
{
lean_object* v___x_986_; 
v___x_986_ = lp_mathlib_RingEquiv_toNatAlgEquiv___redArg(v_f_985_);
return v___x_986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNatAlgEquiv___boxed(lean_object* v_R_987_, lean_object* v_S_988_, lean_object* v_inst_989_, lean_object* v_inst_990_, lean_object* v_f_991_){
_start:
{
lean_object* v_res_992_; 
v_res_992_ = lp_mathlib_RingEquiv_toNatAlgEquiv(v_R_987_, v_S_988_, v_inst_989_, v_inst_990_, v_f_991_);
lean_dec_ref(v_inst_990_);
lean_dec_ref(v_inst_989_);
return v_res_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivNatAlgEquiv___redArg(lean_object* v_inst_993_, lean_object* v_inst_994_){
_start:
{
lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
v___x_995_ = lp_mathlib_Nat_instSemiring;
lean_inc_ref_n(v_inst_994_, 2);
lean_inc_ref_n(v_inst_993_, 2);
v___x_996_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toNatAlgEquiv___boxed), 5, 4);
lean_closure_set(v___x_996_, 0, lean_box(0));
lean_closure_set(v___x_996_, 1, lean_box(0));
lean_closure_set(v___x_996_, 2, v_inst_993_);
lean_closure_set(v___x_996_, 3, v_inst_994_);
v___x_997_ = lp_mathlib_Semiring_toNatAlgebra___redArg(v_inst_993_);
v___x_998_ = lp_mathlib_Semiring_toNatAlgebra___redArg(v_inst_994_);
v___x_999_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toRingEquiv___boxed), 9, 8);
lean_closure_set(v___x_999_, 0, lean_box(0));
lean_closure_set(v___x_999_, 1, lean_box(0));
lean_closure_set(v___x_999_, 2, lean_box(0));
lean_closure_set(v___x_999_, 3, v___x_995_);
lean_closure_set(v___x_999_, 4, v_inst_993_);
lean_closure_set(v___x_999_, 5, v_inst_994_);
lean_closure_set(v___x_999_, 6, v___x_997_);
lean_closure_set(v___x_999_, 7, v___x_998_);
v___x_1000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1000_, 0, v___x_996_);
lean_ctor_set(v___x_1000_, 1, v___x_999_);
return v___x_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivNatAlgEquiv(lean_object* v_R_1001_, lean_object* v_S_1002_, lean_object* v_inst_1003_, lean_object* v_inst_1004_){
_start:
{
lean_object* v___x_1005_; 
v___x_1005_ = lp_mathlib_RingEquiv_equivNatAlgEquiv___redArg(v_inst_1003_, v_inst_1004_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toIntAlgEquiv___redArg(lean_object* v_f_1006_){
_start:
{
lean_object* v___x_1007_; lean_object* v___x_1008_; 
v___x_1007_ = ((lean_object*)(lp_mathlib_AlgEquiv_Simps_toEquiv___redArg___closed__2));
v___x_1008_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_1007_, v_f_1006_);
return v___x_1008_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toIntAlgEquiv(lean_object* v_R_1009_, lean_object* v_S_1010_, lean_object* v_inst_1011_, lean_object* v_inst_1012_, lean_object* v_f_1013_){
_start:
{
lean_object* v___x_1014_; 
v___x_1014_ = lp_mathlib_RingEquiv_toIntAlgEquiv___redArg(v_f_1013_);
return v___x_1014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toIntAlgEquiv___boxed(lean_object* v_R_1015_, lean_object* v_S_1016_, lean_object* v_inst_1017_, lean_object* v_inst_1018_, lean_object* v_f_1019_){
_start:
{
lean_object* v_res_1020_; 
v_res_1020_ = lp_mathlib_RingEquiv_toIntAlgEquiv(v_R_1015_, v_S_1016_, v_inst_1017_, v_inst_1018_, v_f_1019_);
lean_dec_ref(v_inst_1018_);
lean_dec_ref(v_inst_1017_);
return v_res_1020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivIntAlgEquiv___redArg(lean_object* v_inst_1021_, lean_object* v_inst_1022_){
_start:
{
lean_object* v___x_1023_; lean_object* v_toSemiring_1024_; lean_object* v_toSemiring_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; 
v___x_1023_ = lp_mathlib_Int_instCommSemiring;
v_toSemiring_1024_ = lean_ctor_get(v_inst_1021_, 0);
lean_inc_ref(v_toSemiring_1024_);
v_toSemiring_1025_ = lean_ctor_get(v_inst_1022_, 0);
lean_inc_ref(v_toSemiring_1025_);
lean_inc_ref(v_inst_1022_);
lean_inc_ref(v_inst_1021_);
v___x_1026_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_toIntAlgEquiv___boxed), 5, 4);
lean_closure_set(v___x_1026_, 0, lean_box(0));
lean_closure_set(v___x_1026_, 1, lean_box(0));
lean_closure_set(v___x_1026_, 2, v_inst_1021_);
lean_closure_set(v___x_1026_, 3, v_inst_1022_);
v___x_1027_ = lp_mathlib_Ring_toIntAlgebra___redArg(v_inst_1021_);
v___x_1028_ = lp_mathlib_Ring_toIntAlgebra___redArg(v_inst_1022_);
v___x_1029_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_toRingEquiv___boxed), 9, 8);
lean_closure_set(v___x_1029_, 0, lean_box(0));
lean_closure_set(v___x_1029_, 1, lean_box(0));
lean_closure_set(v___x_1029_, 2, lean_box(0));
lean_closure_set(v___x_1029_, 3, v___x_1023_);
lean_closure_set(v___x_1029_, 4, v_toSemiring_1024_);
lean_closure_set(v___x_1029_, 5, v_toSemiring_1025_);
lean_closure_set(v___x_1029_, 6, v___x_1027_);
lean_closure_set(v___x_1029_, 7, v___x_1028_);
v___x_1030_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1026_);
lean_ctor_set(v___x_1030_, 1, v___x_1029_);
return v___x_1030_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_equivIntAlgEquiv(lean_object* v_R_1031_, lean_object* v_S_1032_, lean_object* v_inst_1033_, lean_object* v_inst_1034_){
_start:
{
lean_object* v___x_1035_; 
v___x_1035_ = lp_mathlib_RingEquiv_equivIntAlgEquiv___redArg(v_inst_1033_, v_inst_1034_);
return v___x_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___redArg(lean_object* v_inst_1036_, lean_object* v_inst_1037_, lean_object* v_g_1038_){
_start:
{
lean_object* v___x_1039_; 
v___x_1039_ = lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0(v_inst_1036_, v_inst_1037_, v_g_1038_);
return v___x_1039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___redArg___boxed(lean_object* v_inst_1040_, lean_object* v_inst_1041_, lean_object* v_g_1042_){
_start:
{
lean_object* v_res_1043_; 
v_res_1043_ = lp_mathlib_MulSemiringAction_toAlgEquiv___redArg(v_inst_1040_, v_inst_1041_, v_g_1042_);
lean_dec_ref(v_inst_1040_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv(lean_object* v_G_1044_, lean_object* v_R_1045_, lean_object* v_A_1046_, lean_object* v_inst_1047_, lean_object* v_inst_1048_, lean_object* v_inst_1049_, lean_object* v_inst_1050_, lean_object* v_inst_1051_, lean_object* v_inst_1052_, lean_object* v_g_1053_){
_start:
{
lean_object* v___x_1054_; 
v___x_1054_ = lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0(v_inst_1050_, v_inst_1051_, v_g_1053_);
return v___x_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___boxed(lean_object* v_G_1055_, lean_object* v_R_1056_, lean_object* v_A_1057_, lean_object* v_inst_1058_, lean_object* v_inst_1059_, lean_object* v_inst_1060_, lean_object* v_inst_1061_, lean_object* v_inst_1062_, lean_object* v_inst_1063_, lean_object* v_g_1064_){
_start:
{
lean_object* v_res_1065_; 
v_res_1065_ = lp_mathlib_MulSemiringAction_toAlgEquiv(v_G_1055_, v_R_1056_, v_A_1057_, v_inst_1058_, v_inst_1059_, v_inst_1060_, v_inst_1061_, v_inst_1062_, v_inst_1063_, v_g_1064_);
lean_dec_ref(v_inst_1061_);
lean_dec_ref(v_inst_1060_);
lean_dec_ref(v_inst_1059_);
lean_dec_ref(v_inst_1058_);
return v_res_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgAut___redArg(lean_object* v_inst_1066_, lean_object* v_inst_1067_, lean_object* v_inst_1068_, lean_object* v_inst_1069_, lean_object* v_inst_1070_){
_start:
{
lean_object* v___x_1071_; 
v___x_1071_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toAlgEquiv___boxed), 10, 9);
lean_closure_set(v___x_1071_, 0, lean_box(0));
lean_closure_set(v___x_1071_, 1, lean_box(0));
lean_closure_set(v___x_1071_, 2, lean_box(0));
lean_closure_set(v___x_1071_, 3, v_inst_1066_);
lean_closure_set(v___x_1071_, 4, v_inst_1067_);
lean_closure_set(v___x_1071_, 5, v_inst_1068_);
lean_closure_set(v___x_1071_, 6, v_inst_1069_);
lean_closure_set(v___x_1071_, 7, v_inst_1070_);
lean_closure_set(v___x_1071_, 8, lean_box(0));
return v___x_1071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toAlgAut(lean_object* v_G_1072_, lean_object* v_R_1073_, lean_object* v_A_1074_, lean_object* v_inst_1075_, lean_object* v_inst_1076_, lean_object* v_inst_1077_, lean_object* v_inst_1078_, lean_object* v_inst_1079_, lean_object* v_inst_1080_){
_start:
{
lean_object* v___x_1081_; 
v___x_1081_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toAlgEquiv___boxed), 10, 9);
lean_closure_set(v___x_1081_, 0, lean_box(0));
lean_closure_set(v___x_1081_, 1, lean_box(0));
lean_closure_set(v___x_1081_, 2, lean_box(0));
lean_closure_set(v___x_1081_, 3, v_inst_1075_);
lean_closure_set(v___x_1081_, 4, v_inst_1076_);
lean_closure_set(v___x_1081_, 5, v_inst_1077_);
lean_closure_set(v___x_1081_, 6, v_inst_1078_);
lean_closure_set(v___x_1081_, 7, v_inst_1079_);
lean_closure_set(v___x_1081_, 8, lean_box(0));
return v___x_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0(lean_object* v_toZero_1082_, lean_object* v_x_1083_){
_start:
{
lean_inc(v_toZero_1082_);
return v_toZero_1082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0___boxed(lean_object* v_toZero_1084_, lean_object* v_x_1085_){
_start:
{
lean_object* v_res_1086_; 
v_res_1086_ = lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0(v_toZero_1084_, v_x_1085_);
lean_dec(v_x_1085_);
lean_dec(v_toZero_1084_);
return v_res_1086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg(lean_object* v_inst_1087_, lean_object* v_inst_1088_){
_start:
{
lean_object* v_toAddCommMonoid_1089_; lean_object* v_toAddCommMonoid_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v_toZero_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v_toZero_1096_; lean_object* v___f_1097_; lean_object* v___f_1098_; lean_object* v___f_1099_; lean_object* v___f_1100_; lean_object* v___x_1101_; 
v_toAddCommMonoid_1089_ = lean_ctor_get(v_inst_1087_, 0);
v_toAddCommMonoid_1090_ = lean_ctor_get(v_inst_1088_, 0);
v___x_1091_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_1090_);
v___x_1092_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1091_);
v_toZero_1093_ = lean_ctor_get(v___x_1092_, 0);
lean_inc(v_toZero_1093_);
lean_dec_ref(v___x_1092_);
v___x_1094_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_1089_);
v___x_1095_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_1094_);
v_toZero_1096_ = lean_ctor_get(v___x_1095_, 0);
lean_inc(v_toZero_1096_);
lean_dec_ref(v___x_1095_);
v___f_1097_ = lean_alloc_closure((void*)(lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1097_, 0, v_toZero_1093_);
v___f_1098_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1098_, 0, v___f_1097_);
v___f_1099_ = lean_alloc_closure((void*)(lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1099_, 0, v_toZero_1096_);
v___f_1100_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toMonoidHom_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1100_, 0, v___f_1099_);
v___x_1101_ = lp_mathlib_AlgEquiv_ofAlgHom___redArg(v___f_1098_, v___f_1100_);
return v___x_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg___boxed(lean_object* v_inst_1102_, lean_object* v_inst_1103_){
_start:
{
lean_object* v_res_1104_; 
v_res_1104_ = lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg(v_inst_1102_, v_inst_1103_);
lean_dec_ref(v_inst_1103_);
lean_dec_ref(v_inst_1102_);
return v_res_1104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton(lean_object* v_R_1105_, lean_object* v_S_1106_, lean_object* v_T_1107_, lean_object* v_inst_1108_, lean_object* v_inst_1109_, lean_object* v_inst_1110_, lean_object* v_inst_1111_, lean_object* v_inst_1112_, lean_object* v_inst_1113_, lean_object* v_inst_1114_){
_start:
{
lean_object* v___x_1115_; 
v___x_1115_ = lp_mathlib_instUniqueAlgEquivOfSubsingleton___redArg(v_inst_1109_, v_inst_1110_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAlgEquivOfSubsingleton___boxed(lean_object* v_R_1116_, lean_object* v_S_1117_, lean_object* v_T_1118_, lean_object* v_inst_1119_, lean_object* v_inst_1120_, lean_object* v_inst_1121_, lean_object* v_inst_1122_, lean_object* v_inst_1123_, lean_object* v_inst_1124_, lean_object* v_inst_1125_){
_start:
{
lean_object* v_res_1126_; 
v_res_1126_ = lp_mathlib_instUniqueAlgEquivOfSubsingleton(v_R_1116_, v_S_1117_, v_T_1118_, v_inst_1119_, v_inst_1120_, v_inst_1121_, v_inst_1122_, v_inst_1123_, v_inst_1124_, v_inst_1125_);
lean_dec_ref(v_inst_1123_);
lean_dec_ref(v_inst_1122_);
lean_dec_ref(v_inst_1121_);
lean_dec_ref(v_inst_1120_);
lean_dec_ref(v_inst_1119_);
return v_res_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algEquiv___redArg(lean_object* v_inst_1127_){
_start:
{
lean_object* v___x_1128_; lean_object* v_toNonUnitalNonAssocSemiring_1129_; lean_object* v___x_1130_; 
v___x_1128_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1127_);
v_toNonUnitalNonAssocSemiring_1129_ = lean_ctor_get(v___x_1128_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_1129_);
lean_dec_ref(v___x_1128_);
v___x_1130_ = lp_mathlib_ULift_ringEquiv(lean_box(0), v_toNonUnitalNonAssocSemiring_1129_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_1129_);
return v___x_1130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algEquiv(lean_object* v_R_1131_, lean_object* v_A_1132_, lean_object* v_inst_1133_, lean_object* v_inst_1134_, lean_object* v_inst_1135_){
_start:
{
lean_object* v___x_1136_; 
v___x_1136_ = lp_mathlib_ULift_algEquiv___redArg(v_inst_1134_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_algEquiv___boxed(lean_object* v_R_1137_, lean_object* v_A_1138_, lean_object* v_inst_1139_, lean_object* v_inst_1140_, lean_object* v_inst_1141_){
_start:
{
lean_object* v_res_1142_; 
v_res_1142_ = lp_mathlib_ULift_algEquiv(v_R_1137_, v_A_1138_, v_inst_1139_, v_inst_1140_, v_inst_1141_);
lean_dec_ref(v_inst_1141_);
lean_dec_ref(v_inst_1139_);
return v_res_1142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ulift___redArg(lean_object* v_inst_1143_, lean_object* v_inst_1144_, lean_object* v_f_1145_){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v_toFun_1148_; lean_object* v___x_1149_; lean_object* v_toFun_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; 
v___x_1146_ = lp_mathlib_ULift_algEquiv___redArg(v_inst_1144_);
v___x_1147_ = lp_mathlib_Equiv_symm___redArg(v___x_1146_);
v_toFun_1148_ = lean_ctor_get(v___x_1147_, 0);
lean_inc(v_toFun_1148_);
lean_dec_ref(v___x_1147_);
v___x_1149_ = lp_mathlib_ULift_algEquiv___redArg(v_inst_1143_);
v_toFun_1150_ = lean_ctor_get(v___x_1149_, 0);
lean_inc(v_toFun_1150_);
lean_dec_ref(v___x_1149_);
v___x_1151_ = lp_mathlib_AlgHom_comp___redArg(v_f_1145_, v_toFun_1150_);
v___x_1152_ = lp_mathlib_AlgHom_comp___redArg(v_toFun_1148_, v___x_1151_);
return v___x_1152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ulift(lean_object* v_R_1153_, lean_object* v_S_1154_, lean_object* v_T_1155_, lean_object* v_inst_1156_, lean_object* v_inst_1157_, lean_object* v_inst_1158_, lean_object* v_inst_1159_, lean_object* v_inst_1160_, lean_object* v_f_1161_){
_start:
{
lean_object* v___x_1162_; 
v___x_1162_ = lp_mathlib_AlgHom_ulift___redArg(v_inst_1157_, v_inst_1158_, v_f_1161_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_ulift___boxed(lean_object* v_R_1163_, lean_object* v_S_1164_, lean_object* v_T_1165_, lean_object* v_inst_1166_, lean_object* v_inst_1167_, lean_object* v_inst_1168_, lean_object* v_inst_1169_, lean_object* v_inst_1170_, lean_object* v_f_1171_){
_start:
{
lean_object* v_res_1172_; 
v_res_1172_ = lp_mathlib_AlgHom_ulift(v_R_1163_, v_S_1164_, v_T_1165_, v_inst_1166_, v_inst_1167_, v_inst_1168_, v_inst_1169_, v_inst_1170_, v_f_1171_);
lean_dec_ref(v_inst_1170_);
lean_dec_ref(v_inst_1169_);
lean_dec_ref(v_inst_1166_);
return v_res_1172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___redArg___lam__0(lean_object* v_e_1173_, lean_object* v_toOne_1174_, lean_object* v_toMul_1175_, lean_object* v_x_1176_){
_start:
{
lean_object* v_toLinearMap_1177_; lean_object* v___x_1178_; lean_object* v_toLinearMap_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; 
v_toLinearMap_1177_ = lean_ctor_get(v_e_1173_, 0);
lean_inc(v_toLinearMap_1177_);
v___x_1178_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_1173_);
v_toLinearMap_1179_ = lean_ctor_get(v___x_1178_, 0);
lean_inc(v_toLinearMap_1179_);
lean_dec_ref(v___x_1178_);
v___x_1180_ = lean_apply_1(v_toLinearMap_1177_, v_toOne_1174_);
v___x_1181_ = lean_apply_2(v_toMul_1175_, v___x_1180_, v_x_1176_);
v___x_1182_ = lean_apply_1(v_toLinearMap_1179_, v___x_1181_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___redArg(lean_object* v_inst_1183_, lean_object* v_inst_1184_, lean_object* v_inst_1185_, lean_object* v_e_1186_){
_start:
{
lean_object* v_algebraMap_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v_toMul_1190_; lean_object* v___x_1192_; uint8_t v_isShared_1193_; uint8_t v_isSharedCheck_1200_; 
v_algebraMap_1187_ = lean_ctor_get(v_inst_1185_, 1);
v___x_1188_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1183_);
v___x_1189_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_1184_);
v_toMul_1190_ = lean_ctor_get(v___x_1189_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1189_);
if (v_isSharedCheck_1200_ == 0)
{
lean_object* v_unused_1201_; 
v_unused_1201_ = lean_ctor_get(v___x_1189_, 1);
lean_dec(v_unused_1201_);
v___x_1192_ = v___x_1189_;
v_isShared_1193_ = v_isSharedCheck_1200_;
goto v_resetjp_1191_;
}
else
{
lean_inc(v_toMul_1190_);
lean_dec(v___x_1189_);
v___x_1192_ = lean_box(0);
v_isShared_1193_ = v_isSharedCheck_1200_;
goto v_resetjp_1191_;
}
v_resetjp_1191_:
{
lean_object* v___x_1194_; lean_object* v_toOne_1195_; lean_object* v___f_1196_; lean_object* v___x_1198_; 
v___x_1194_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_1188_);
v_toOne_1195_ = lean_ctor_get(v___x_1194_, 2);
lean_inc(v_toOne_1195_);
lean_dec_ref(v___x_1194_);
v___f_1196_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_algEquivOfRing___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1196_, 0, v_e_1186_);
lean_closure_set(v___f_1196_, 1, v_toOne_1195_);
lean_closure_set(v___f_1196_, 2, v_toMul_1190_);
lean_inc(v_algebraMap_1187_);
if (v_isShared_1193_ == 0)
{
lean_ctor_set(v___x_1192_, 1, v___f_1196_);
lean_ctor_set(v___x_1192_, 0, v_algebraMap_1187_);
v___x_1198_ = v___x_1192_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_algebraMap_1187_);
lean_ctor_set(v_reuseFailAlloc_1199_, 1, v___f_1196_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___redArg___boxed(lean_object* v_inst_1202_, lean_object* v_inst_1203_, lean_object* v_inst_1204_, lean_object* v_e_1205_){
_start:
{
lean_object* v_res_1206_; 
v_res_1206_ = lp_mathlib_LinearEquiv_algEquivOfRing___redArg(v_inst_1202_, v_inst_1203_, v_inst_1204_, v_e_1205_);
lean_dec_ref(v_inst_1204_);
return v_res_1206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing(lean_object* v_R_1207_, lean_object* v_A_1208_, lean_object* v_inst_1209_, lean_object* v_inst_1210_, lean_object* v_inst_1211_, lean_object* v_e_1212_){
_start:
{
lean_object* v___x_1213_; 
v___x_1213_ = lp_mathlib_LinearEquiv_algEquivOfRing___redArg(v_inst_1209_, v_inst_1210_, v_inst_1211_, v_e_1212_);
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_algEquivOfRing___boxed(lean_object* v_R_1214_, lean_object* v_A_1215_, lean_object* v_inst_1216_, lean_object* v_inst_1217_, lean_object* v_inst_1218_, lean_object* v_e_1219_){
_start:
{
lean_object* v_res_1220_; 
v_res_1220_ = lp_mathlib_LinearEquiv_algEquivOfRing(v_R_1214_, v_A_1215_, v_inst_1216_, v_inst_1217_, v_inst_1218_, v_e_1219_);
lean_dec_ref(v_inst_1218_);
return v_res_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjAlgEquiv___redArg(lean_object* v_e_1221_){
_start:
{
lean_object* v___x_1222_; 
lean_inc_ref(v_e_1221_);
v___x_1222_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_1221_, v_e_1221_);
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjAlgEquiv(lean_object* v_R_1223_, lean_object* v_S_1224_, lean_object* v_M_u2081_1225_, lean_object* v_M_u2082_1226_, lean_object* v_inst_1227_, lean_object* v_inst_1228_, lean_object* v_inst_1229_, lean_object* v_inst_1230_, lean_object* v_inst_1231_, lean_object* v_inst_1232_, lean_object* v_inst_1233_, lean_object* v_inst_1234_, lean_object* v_inst_1235_, lean_object* v_inst_1236_, lean_object* v_inst_1237_, lean_object* v_inst_1238_, lean_object* v_inst_1239_, lean_object* v_e_1240_){
_start:
{
lean_object* v___x_1241_; 
lean_inc_ref(v_e_1240_);
v___x_1241_ = lp_mathlib_LinearEquiv_arrowCongrAddEquiv___redArg(v_e_1240_, v_e_1240_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_conjAlgEquiv___boxed(lean_object** _args){
lean_object* v_R_1242_ = _args[0];
lean_object* v_S_1243_ = _args[1];
lean_object* v_M_u2081_1244_ = _args[2];
lean_object* v_M_u2082_1245_ = _args[3];
lean_object* v_inst_1246_ = _args[4];
lean_object* v_inst_1247_ = _args[5];
lean_object* v_inst_1248_ = _args[6];
lean_object* v_inst_1249_ = _args[7];
lean_object* v_inst_1250_ = _args[8];
lean_object* v_inst_1251_ = _args[9];
lean_object* v_inst_1252_ = _args[10];
lean_object* v_inst_1253_ = _args[11];
lean_object* v_inst_1254_ = _args[12];
lean_object* v_inst_1255_ = _args[13];
lean_object* v_inst_1256_ = _args[14];
lean_object* v_inst_1257_ = _args[15];
lean_object* v_inst_1258_ = _args[16];
lean_object* v_e_1259_ = _args[17];
_start:
{
lean_object* v_res_1260_; 
v_res_1260_ = lp_mathlib_LinearEquiv_conjAlgEquiv(v_R_1242_, v_S_1243_, v_M_u2081_1244_, v_M_u2082_1245_, v_inst_1246_, v_inst_1247_, v_inst_1248_, v_inst_1249_, v_inst_1250_, v_inst_1251_, v_inst_1252_, v_inst_1253_, v_inst_1254_, v_inst_1255_, v_inst_1256_, v_inst_1257_, v_inst_1258_, v_e_1259_);
lean_dec(v_inst_1256_);
lean_dec(v_inst_1253_);
lean_dec(v_inst_1252_);
lean_dec_ref(v_inst_1251_);
lean_dec(v_inst_1250_);
lean_dec_ref(v_inst_1249_);
lean_dec(v_inst_1248_);
lean_dec_ref(v_inst_1247_);
lean_dec_ref(v_inst_1246_);
return v_res_1260_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
}
#ifdef __cplusplus
}
#endif
