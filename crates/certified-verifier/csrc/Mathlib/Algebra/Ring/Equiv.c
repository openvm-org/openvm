// Lean compiler output
// Module: Mathlib.Algebra.Ring.Equiv
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Opposite public import Mathlib.Algebra.GroupWithZero.Equiv public import Mathlib.Algebra.GroupWithZero.InjSurj public import Mathlib.Algebra.Notation.Prod public import Mathlib.Algebra.Ring.Hom.Defs public import Mathlib.Logic.Equiv.Set public import Mathlib.Util.Delaborators import Mathlib.Tactic.DSimpPercent
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
lean_object* lp_mathlib_AddEquiv_mulOp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_piOptionEquivProd(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ofUnique___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_MulEquiv_opOp(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_x2b_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_≃+*_"};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 115, 93, 167, 24, 120, 153, 155)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_x2b_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2243_x2b_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ≃+* "};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2243_x2b_x2a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__7_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b_x2a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b_x2a___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_x2b_x2a__ = (const lean_object*)&lp_mathlib_term___u2243_x2b_x2a___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "RingEquiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(137, 180, 103, 8, 69, 243, 92, 209)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquivClass_toRingEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquivClass_toRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquivClass_toRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_unop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_toOpposite___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_toOpposite___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toOpposite___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_cast___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_cast___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piEquivPiSubtypeProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piEquivPiSubtypeProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piEquivPiSubtypeProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_piMulOpposite___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_piMulOpposite___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_piMulOpposite___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_piMulOpposite___closed__0_value;
static const lean_closure_object lp_mathlib_RingEquiv_piMulOpposite___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_piMulOpposite___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_piMulOpposite___closed__1 = (const lean_object*)&lp_mathlib_RingEquiv_piMulOpposite___closed__1_value;
static const lean_ctor_object lp_mathlib_RingEquiv_piMulOpposite___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RingEquiv_piMulOpposite___closed__0_value),((lean_object*)&lp_mathlib_RingEquiv_piMulOpposite___closed__1_value)}};
static const lean_object* lp_mathlib_RingEquiv_piMulOpposite___closed__2 = (const lean_object*)&lp_mathlib_RingEquiv_piMulOpposite___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_prodCongr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_prodCongr___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_prodCongr___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_RingEquiv_prodCongr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_prodCongr___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___closed__1 = (const lean_object*)&lp_mathlib_RingEquiv_prodCongr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_RingEquiv_prodCongr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RingEquiv_prodCongr___redArg___closed__0_value),((lean_object*)&lp_mathlib_RingEquiv_prodCongr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___closed__2 = (const lean_object*)&lp_mathlib_RingEquiv_prodCongr___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_piOptionEquivProd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_piOptionEquivProd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piOptionEquivProd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piOptionEquivProd___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toRingEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toRingEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_sumArrowEquivProdArrow___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_sumArrowEquivProdArrow___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sumArrowEquivProdArrow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sumArrowEquivProdArrow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse___redArg(lean_object* v_g_1_){
_start:
{
lean_inc(v_g_1_);
return v_g_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse___redArg___boxed(lean_object* v_g_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_NonUnitalRingHom_inverse___redArg(v_g_2_);
lean_dec(v_g_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse(lean_object* v_R_4_, lean_object* v_S_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_f_8_, lean_object* v_g_9_, lean_object* v_h_u2081_10_, lean_object* v_h_u2082_11_){
_start:
{
lean_inc(v_g_9_);
return v_g_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_inverse___boxed(lean_object* v_R_12_, lean_object* v_S_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_, lean_object* v_g_17_, lean_object* v_h_u2081_18_, lean_object* v_h_u2082_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_NonUnitalRingHom_inverse(v_R_12_, v_S_13_, v_inst_14_, v_inst_15_, v_f_16_, v_g_17_, v_h_u2081_18_, v_h_u2082_19_);
lean_dec(v_g_17_);
lean_dec(v_f_16_);
lean_dec_ref(v_inst_15_);
lean_dec_ref(v_inst_14_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse___redArg(lean_object* v_g_21_){
_start:
{
lean_inc(v_g_21_);
return v_g_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse___redArg___boxed(lean_object* v_g_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_RingHom_inverse___redArg(v_g_22_);
lean_dec(v_g_22_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse(lean_object* v_R_24_, lean_object* v_S_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_f_28_, lean_object* v_g_29_, lean_object* v_h_u2081_30_, lean_object* v_h_u2082_31_){
_start:
{
lean_inc(v_g_29_);
return v_g_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_inverse___boxed(lean_object* v_R_32_, lean_object* v_S_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_f_36_, lean_object* v_g_37_, lean_object* v_h_u2081_38_, lean_object* v_h_u2082_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_RingHom_inverse(v_R_32_, v_S_33_, v_inst_34_, v_inst_35_, v_f_36_, v_g_37_, v_h_u2081_38_, v_h_u2082_39_);
lean_dec(v_g_37_);
lean_dec(v_f_36_);
lean_dec_ref(v_inst_35_);
lean_dec_ref(v_inst_34_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___redArg(lean_object* v_self_41_){
_start:
{
lean_inc_ref(v_self_41_);
return v_self_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___redArg___boxed(lean_object* v_self_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_RingEquiv_toMulEquiv___redArg(v_self_42_);
lean_dec_ref(v_self_42_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv(lean_object* v_R_44_, lean_object* v_S_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_self_50_){
_start:
{
lean_inc_ref(v_self_50_);
return v_self_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMulEquiv___boxed(lean_object* v_R_51_, lean_object* v_S_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_self_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_RingEquiv_toMulEquiv(v_R_51_, v_S_52_, v_inst_53_, v_inst_54_, v_inst_55_, v_inst_56_, v_self_57_);
lean_dec_ref(v_self_57_);
lean_dec(v_inst_56_);
lean_dec(v_inst_55_);
lean_dec(v_inst_54_);
lean_dec(v_inst_53_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv___redArg(lean_object* v_self_59_){
_start:
{
lean_inc_ref(v_self_59_);
return v_self_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv___redArg___boxed(lean_object* v_self_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_RingEquiv_toAddEquiv___redArg(v_self_60_);
lean_dec_ref(v_self_60_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv(lean_object* v_R_62_, lean_object* v_S_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_self_68_){
_start:
{
lean_inc_ref(v_self_68_);
return v_self_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddEquiv___boxed(lean_object* v_R_69_, lean_object* v_S_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_self_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_RingEquiv_toAddEquiv(v_R_69_, v_S_70_, v_inst_71_, v_inst_72_, v_inst_73_, v_inst_74_, v_self_75_);
lean_dec_ref(v_self_75_);
lean_dec(v_inst_74_);
lean_dec(v_inst_73_);
lean_dec(v_inst_72_);
lean_dec(v_inst_71_);
return v_res_76_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__6(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_111_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__5));
v___x_112_ = l_String_toRawSubstring_x27(v___x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1(lean_object* v_x_129_, lean_object* v_a_130_, lean_object* v_a_131_){
_start:
{
lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_132_ = ((lean_object*)(lp_mathlib_term___u2243_x2b_x2a___00__closed__1));
lean_inc(v_x_129_);
v___x_133_ = l_Lean_Syntax_isOfKind(v_x_129_, v___x_132_);
if (v___x_133_ == 0)
{
lean_object* v___x_134_; lean_object* v___x_135_; 
lean_dec(v_x_129_);
v___x_134_ = lean_box(1);
v___x_135_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v_a_131_);
return v___x_135_;
}
else
{
lean_object* v_quotContext_136_; lean_object* v_currMacroScope_137_; lean_object* v_ref_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; uint8_t v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v_quotContext_136_ = lean_ctor_get(v_a_130_, 1);
v_currMacroScope_137_ = lean_ctor_get(v_a_130_, 2);
v_ref_138_ = lean_ctor_get(v_a_130_, 5);
v___x_139_ = lean_unsigned_to_nat(0u);
v___x_140_ = l_Lean_Syntax_getArg(v_x_129_, v___x_139_);
v___x_141_ = lean_unsigned_to_nat(2u);
v___x_142_ = l_Lean_Syntax_getArg(v_x_129_, v___x_141_);
lean_dec(v_x_129_);
v___x_143_ = 0;
v___x_144_ = l_Lean_SourceInfo_fromRef(v_ref_138_, v___x_143_);
v___x_145_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4));
v___x_146_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__6);
v___x_147_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__7));
lean_inc(v_currMacroScope_137_);
lean_inc(v_quotContext_136_);
v___x_148_ = l_Lean_addMacroScope(v_quotContext_136_, v___x_147_, v_currMacroScope_137_);
v___x_149_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__11));
lean_inc_n(v___x_144_, 2);
v___x_150_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_150_, 0, v___x_144_);
lean_ctor_set(v___x_150_, 1, v___x_146_);
lean_ctor_set(v___x_150_, 2, v___x_148_);
lean_ctor_set(v___x_150_, 3, v___x_149_);
v___x_151_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__13));
v___x_152_ = l_Lean_Syntax_node2(v___x_144_, v___x_151_, v___x_140_, v___x_142_);
v___x_153_ = l_Lean_Syntax_node2(v___x_144_, v___x_145_, v___x_150_, v___x_152_);
v___x_154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_131_);
return v___x_154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___boxed(lean_object* v_x_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1(v_x_155_, v_a_156_, v_a_157_);
lean_dec_ref(v_a_156_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1(lean_object* v_x_162_, lean_object* v_a_163_, lean_object* v_a_164_){
_start:
{
lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_165_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______macroRules__term___u2243_x2b_x2a____1___closed__4));
lean_inc(v_x_162_);
v___x_166_ = l_Lean_Syntax_isOfKind(v_x_162_, v___x_165_);
if (v___x_166_ == 0)
{
lean_object* v___x_167_; lean_object* v___x_168_; 
lean_dec(v_x_162_);
v___x_167_ = lean_box(0);
v___x_168_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
lean_ctor_set(v___x_168_, 1, v_a_164_);
return v___x_168_;
}
else
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_169_ = lean_unsigned_to_nat(0u);
v___x_170_ = l_Lean_Syntax_getArg(v_x_162_, v___x_169_);
v___x_171_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___closed__1));
lean_inc(v___x_170_);
v___x_172_ = l_Lean_Syntax_isOfKind(v___x_170_, v___x_171_);
if (v___x_172_ == 0)
{
lean_object* v___x_173_; lean_object* v___x_174_; 
lean_dec(v___x_170_);
lean_dec(v_x_162_);
v___x_173_ = lean_box(0);
v___x_174_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_a_164_);
return v___x_174_;
}
else
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_175_ = lean_unsigned_to_nat(1u);
v___x_176_ = l_Lean_Syntax_getArg(v_x_162_, v___x_175_);
lean_dec(v_x_162_);
v___x_177_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_176_);
v___x_178_ = l_Lean_Syntax_matchesNull(v___x_176_, v___x_177_);
if (v___x_178_ == 0)
{
lean_object* v___x_179_; lean_object* v___x_180_; 
lean_dec(v___x_176_);
lean_dec(v___x_170_);
v___x_179_ = lean_box(0);
v___x_180_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
lean_ctor_set(v___x_180_, 1, v_a_164_);
return v___x_180_;
}
else
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v_ref_183_; uint8_t v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_181_ = l_Lean_Syntax_getArg(v___x_176_, v___x_169_);
v___x_182_ = l_Lean_Syntax_getArg(v___x_176_, v___x_175_);
lean_dec(v___x_176_);
v_ref_183_ = l_Lean_replaceRef(v___x_170_, v_a_163_);
lean_dec(v___x_170_);
v___x_184_ = 0;
v___x_185_ = l_Lean_SourceInfo_fromRef(v_ref_183_, v___x_184_);
lean_dec(v_ref_183_);
v___x_186_ = ((lean_object*)(lp_mathlib_term___u2243_x2b_x2a___00__closed__1));
v___x_187_ = ((lean_object*)(lp_mathlib_term___u2243_x2b_x2a___00__closed__4));
lean_inc(v___x_185_);
v___x_188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_185_);
lean_ctor_set(v___x_188_, 1, v___x_187_);
v___x_189_ = l_Lean_Syntax_node3(v___x_185_, v___x_186_, v___x_181_, v___x_188_, v___x_182_);
v___x_190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
lean_ctor_set(v___x_190_, 1, v_a_164_);
return v___x_190_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1___boxed(lean_object* v_x_191_, lean_object* v_a_192_, lean_object* v_a_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib___aux__Mathlib__Algebra__Ring__Equiv______unexpand__RingEquiv__1(v_x_191_, v_a_192_, v_a_193_);
lean_dec(v_a_192_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquivClass_toRingEquiv___redArg(lean_object* v_inst_195_, lean_object* v_f_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_195_, v_f_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquivClass_toRingEquiv(lean_object* v_F_198_, lean_object* v_00_u03b1_199_, lean_object* v_00_u03b2_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_f_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_205_, v_f_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquivClass_toRingEquiv___boxed(lean_object* v_F_209_, lean_object* v_00_u03b1_210_, lean_object* v_00_u03b2_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_f_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_RingEquivClass_toRingEquiv(v_F_209_, v_00_u03b1_210_, v_00_u03b2_211_, v_inst_212_, v_inst_213_, v_inst_214_, v_inst_215_, v_inst_216_, v_inst_217_, v_f_218_);
lean_dec(v_inst_215_);
lean_dec(v_inst_214_);
lean_dec(v_inst_213_);
lean_dec(v_inst_212_);
return v_res_219_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_refl___closed__0(void){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl(lean_object* v_R_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_obj_once(&lp_mathlib_RingEquiv_refl___closed__0, &lp_mathlib_RingEquiv_refl___closed__0_once, _init_lp_mathlib_RingEquiv_refl___closed__0);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_refl___boxed(lean_object* v_R_225_, lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_RingEquiv_refl(v_R_225_, v_inst_226_, v_inst_227_);
lean_dec(v_inst_227_);
lean_dec(v_inst_226_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instInhabited(lean_object* v_R_229_, lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lean_obj_once(&lp_mathlib_RingEquiv_refl___closed__0, &lp_mathlib_RingEquiv_refl___closed__0_once, _init_lp_mathlib_RingEquiv_refl___closed__0);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instInhabited___boxed(lean_object* v_R_233_, lean_object* v_inst_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_RingEquiv_instInhabited(v_R_233_, v_inst_234_, v_inst_235_);
lean_dec(v_inst_235_);
lean_dec(v_inst_234_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___redArg(lean_object* v_e_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_mathlib_Equiv_symm___redArg(v_e_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm(lean_object* v_R_239_, lean_object* v_S_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_e_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lp_mathlib_Equiv_symm___redArg(v_e_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_symm___boxed(lean_object* v_R_247_, lean_object* v_S_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_e_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_RingEquiv_symm(v_R_247_, v_S_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_inst_252_, v_e_253_);
lean_dec(v_inst_252_);
lean_dec(v_inst_251_);
lean_dec(v_inst_250_);
lean_dec(v_inst_249_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_Simps_symm__apply___redArg(lean_object* v_e_255_, lean_object* v_a_256_){
_start:
{
lean_object* v___x_257_; lean_object* v_toFun_258_; lean_object* v___x_259_; 
v___x_257_ = lp_mathlib_Equiv_symm___redArg(v_e_255_);
v_toFun_258_ = lean_ctor_get(v___x_257_, 0);
lean_inc(v_toFun_258_);
lean_dec_ref(v___x_257_);
v___x_259_ = lean_apply_1(v_toFun_258_, v_a_256_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_Simps_symm__apply(lean_object* v_R_260_, lean_object* v_S_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_e_266_, lean_object* v_a_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_mathlib_RingEquiv_Simps_symm__apply___redArg(v_e_266_, v_a_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_Simps_symm__apply___boxed(lean_object* v_R_269_, lean_object* v_S_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_e_275_, lean_object* v_a_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_RingEquiv_Simps_symm__apply(v_R_269_, v_S_270_, v_inst_271_, v_inst_272_, v_inst_273_, v_inst_274_, v_e_275_, v_a_276_);
lean_dec(v_inst_274_);
lean_dec(v_inst_273_);
lean_dec(v_inst_272_);
lean_dec(v_inst_271_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_trans___redArg(lean_object* v_e_u2081_278_, lean_object* v_e_u2082_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_mathlib_Equiv_trans___redArg(v_e_u2081_278_, v_e_u2082_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_trans(lean_object* v_R_281_, lean_object* v_S_282_, lean_object* v_S_x27_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_e_u2081_290_, lean_object* v_e_u2082_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Equiv_trans___redArg(v_e_u2081_290_, v_e_u2082_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_trans___boxed(lean_object* v_R_293_, lean_object* v_S_294_, lean_object* v_S_x27_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_e_u2081_302_, lean_object* v_e_u2082_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_RingEquiv_trans(v_R_293_, v_S_294_, v_S_x27_295_, v_inst_296_, v_inst_297_, v_inst_298_, v_inst_299_, v_inst_300_, v_inst_301_, v_e_u2081_302_, v_e_u2082_303_);
lean_dec(v_inst_301_);
lean_dec(v_inst_300_);
lean_dec(v_inst_299_);
lean_dec(v_inst_298_);
lean_dec(v_inst_297_);
lean_dec(v_inst_296_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofUnique___redArg(lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_305_, v_inst_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofUnique(lean_object* v_M_308_, lean_object* v_N_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_310_, v_inst_311_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofUnique___boxed(lean_object* v_M_317_, lean_object* v_N_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib_RingEquiv_ofUnique(v_M_317_, v_N_318_, v_inst_319_, v_inst_320_, v_inst_321_, v_inst_322_, v_inst_323_, v_inst_324_);
lean_dec(v_inst_324_);
lean_dec(v_inst_323_);
lean_dec(v_inst_322_);
lean_dec(v_inst_321_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instUnique___redArg(lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_326_, v_inst_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instUnique(lean_object* v_M_329_, lean_object* v_N_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_331_, v_inst_332_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_instUnique___boxed(lean_object* v_M_338_, lean_object* v_N_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_mathlib_RingEquiv_instUnique(v_M_338_, v_N_339_, v_inst_340_, v_inst_341_, v_inst_342_, v_inst_343_, v_inst_344_, v_inst_345_);
lean_dec(v_inst_345_);
lean_dec(v_inst_344_);
lean_dec(v_inst_343_);
lean_dec(v_inst_342_);
return v_res_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__0(lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_f_349_){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v_toFun_352_; lean_object* v___x_353_; 
v___x_350_ = lp_mathlib_AddEquiv_mulOp(lean_box(0), lean_box(0), v_inst_347_, v_inst_348_);
v___x_351_ = lp_mathlib_Equiv_symm___redArg(v___x_350_);
v_toFun_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_toFun_352_);
lean_dec_ref(v___x_351_);
v___x_353_ = lean_apply_1(v_toFun_352_, v_f_349_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__0___boxed(lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_f_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_RingEquiv_op___redArg___lam__0(v_inst_354_, v_inst_355_, v_f_356_);
lean_dec(v_inst_355_);
lean_dec(v_inst_354_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__1(lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_f_360_){
_start:
{
lean_object* v___x_361_; lean_object* v_toFun_362_; lean_object* v___x_363_; 
v___x_361_ = lp_mathlib_AddEquiv_mulOp(lean_box(0), lean_box(0), v_inst_358_, v_inst_359_);
v_toFun_362_ = lean_ctor_get(v___x_361_, 0);
lean_inc(v_toFun_362_);
lean_dec_ref(v___x_361_);
v___x_363_ = lean_apply_1(v_toFun_362_, v_f_360_);
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg___lam__1___boxed(lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_f_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_RingEquiv_op___redArg___lam__1(v_inst_364_, v_inst_365_, v_f_366_);
lean_dec(v_inst_365_);
lean_dec(v_inst_364_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___redArg(lean_object* v_inst_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___f_370_; lean_object* v___f_371_; lean_object* v___x_372_; 
lean_inc(v_inst_369_);
lean_inc(v_inst_368_);
v___f_370_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_op___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_370_, 0, v_inst_368_);
lean_closure_set(v___f_370_, 1, v_inst_369_);
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_op___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_371_, 0, v_inst_368_);
lean_closure_set(v___f_371_, 1, v_inst_369_);
v___x_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_372_, 0, v___f_371_);
lean_ctor_set(v___x_372_, 1, v___f_370_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op(lean_object* v_00_u03b1_373_, lean_object* v_00_u03b2_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_RingEquiv_op___redArg(v_inst_375_, v_inst_377_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_op___boxed(lean_object* v_00_u03b1_380_, lean_object* v_00_u03b2_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_RingEquiv_op(v_00_u03b1_380_, v_00_u03b2_381_, v_inst_382_, v_inst_383_, v_inst_384_, v_inst_385_);
lean_dec(v_inst_385_);
lean_dec(v_inst_383_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_unop___redArg(lean_object* v_inst_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v___x_389_; lean_object* v___x_390_; 
v___x_389_ = lp_mathlib_RingEquiv_op___redArg(v_inst_387_, v_inst_388_);
v___x_390_ = lp_mathlib_Equiv_symm___redArg(v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_unop(lean_object* v_00_u03b1_391_, lean_object* v_00_u03b2_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lp_mathlib_RingEquiv_unop___redArg(v_inst_393_, v_inst_395_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_unop___boxed(lean_object* v_00_u03b1_398_, lean_object* v_00_u03b2_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_RingEquiv_unop(v_00_u03b1_398_, v_00_u03b2_399_, v_inst_400_, v_inst_401_, v_inst_402_, v_inst_403_);
lean_dec(v_inst_403_);
lean_dec(v_inst_401_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp___redArg(lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_MulEquiv_opOp(lean_box(0), v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp___redArg___boxed(lean_object* v_inst_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_RingEquiv_opOp___redArg(v_inst_407_);
lean_dec(v_inst_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp(lean_object* v_R_409_, lean_object* v_inst_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_MulEquiv_opOp(lean_box(0), v_inst_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_opOp___boxed(lean_object* v_R_413_, lean_object* v_inst_414_, lean_object* v_inst_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_RingEquiv_opOp(v_R_413_, v_inst_414_, v_inst_415_);
lean_dec(v_inst_415_);
lean_dec(v_inst_414_);
return v_res_416_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_toOpposite___closed__0(void){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toOpposite(lean_object* v_R_418_, lean_object* v_inst_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lean_obj_once(&lp_mathlib_RingEquiv_toOpposite___closed__0, &lp_mathlib_RingEquiv_toOpposite___closed__0_once, _init_lp_mathlib_RingEquiv_toOpposite___closed__0);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toOpposite___boxed(lean_object* v_R_421_, lean_object* v_inst_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib_RingEquiv_toOpposite(v_R_421_, v_inst_422_);
lean_dec_ref(v_inst_422_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piUnique___redArg(lean_object* v_inst_424_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_424_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piUnique(lean_object* v_00_u03b9_426_, lean_object* v_R_427_, lean_object* v_inst_428_, lean_object* v_inst_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_428_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piUnique___boxed(lean_object* v_00_u03b9_431_, lean_object* v_R_432_, lean_object* v_inst_433_, lean_object* v_inst_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_RingEquiv_piUnique(v_00_u03b9_431_, v_R_432_, v_inst_433_, v_inst_434_);
lean_dec_ref(v_inst_434_);
return v_res_435_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_cast___closed__0(void){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_cast(lean_object* v_00_u03b9_437_, lean_object* v_R_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_i_441_, lean_object* v_j_442_, lean_object* v_h_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lean_obj_once(&lp_mathlib_RingEquiv_cast___closed__0, &lp_mathlib_RingEquiv_cast___closed__0_once, _init_lp_mathlib_RingEquiv_cast___closed__0);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_cast___boxed(lean_object* v_00_u03b9_445_, lean_object* v_R_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_i_449_, lean_object* v_j_450_, lean_object* v_h_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_mathlib_RingEquiv_cast(v_00_u03b9_445_, v_R_446_, v_inst_447_, v_inst_448_, v_i_449_, v_j_450_, v_h_451_);
lean_dec(v_j_450_);
lean_dec(v_i_449_);
lean_dec(v_inst_448_);
lean_dec(v_inst_447_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___redArg___lam__0(lean_object* v_e_453_, lean_object* v_x_454_, lean_object* v_j_455_){
_start:
{
lean_object* v___x_456_; lean_object* v_toFun_457_; lean_object* v___x_458_; lean_object* v___x_459_; 
lean_inc(v_j_455_);
v___x_456_ = lean_apply_1(v_e_453_, v_j_455_);
v_toFun_457_ = lean_ctor_get(v___x_456_, 0);
lean_inc(v_toFun_457_);
lean_dec_ref(v___x_456_);
v___x_458_ = lean_apply_1(v_x_454_, v_j_455_);
v___x_459_ = lean_apply_1(v_toFun_457_, v___x_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___redArg___lam__1(lean_object* v_e_460_, lean_object* v_x_461_, lean_object* v_j_462_){
_start:
{
lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v_toFun_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
lean_inc(v_j_462_);
v___x_463_ = lean_apply_1(v_e_460_, v_j_462_);
v___x_464_ = lp_mathlib_Equiv_symm___redArg(v___x_463_);
v_toFun_465_ = lean_ctor_get(v___x_464_, 0);
lean_inc(v_toFun_465_);
lean_dec_ref(v___x_464_);
v___x_466_ = lean_apply_1(v_x_461_, v_j_462_);
v___x_467_ = lean_apply_1(v_toFun_465_, v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___redArg(lean_object* v_e_468_){
_start:
{
lean_object* v___f_469_; lean_object* v___f_470_; lean_object* v___x_471_; 
lean_inc_ref(v_e_468_);
v___f_469_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_469_, 0, v_e_468_);
v___f_470_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_piCongrRight___redArg___lam__1), 3, 1);
lean_closure_set(v___f_470_, 0, v_e_468_);
v___x_471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_471_, 0, v___f_469_);
lean_ctor_set(v___x_471_, 1, v___f_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight(lean_object* v_00_u03b9_472_, lean_object* v_R_473_, lean_object* v_S_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_e_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_mathlib_RingEquiv_piCongrRight___redArg(v_e_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrRight___boxed(lean_object* v_00_u03b9_479_, lean_object* v_R_480_, lean_object* v_S_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_e_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_mathlib_RingEquiv_piCongrRight(v_00_u03b9_479_, v_R_480_, v_S_481_, v_inst_482_, v_inst_483_, v_e_484_);
lean_dec_ref(v_inst_483_);
lean_dec_ref(v_inst_482_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft_x27___redArg(lean_object* v_e_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_Equiv_piCongrLeft_x27___redArg(v_e_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft_x27(lean_object* v_00_u03b9_488_, lean_object* v_00_u03b9_x27_489_, lean_object* v_R_490_, lean_object* v_e_491_, lean_object* v_inst_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_Equiv_piCongrLeft_x27___redArg(v_e_491_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft_x27___boxed(lean_object* v_00_u03b9_494_, lean_object* v_00_u03b9_x27_495_, lean_object* v_R_496_, lean_object* v_e_497_, lean_object* v_inst_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_RingEquiv_piCongrLeft_x27(v_00_u03b9_494_, v_00_u03b9_x27_495_, v_R_496_, v_e_497_, v_inst_498_);
lean_dec_ref(v_inst_498_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft___redArg(lean_object* v_e_500_){
_start:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v___x_501_ = lp_mathlib_Equiv_symm___redArg(v_e_500_);
v___x_502_ = lp_mathlib_Equiv_piCongrLeft_x27___redArg(v___x_501_);
v___x_503_ = lp_mathlib_Equiv_symm___redArg(v___x_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft(lean_object* v_00_u03b9_504_, lean_object* v_00_u03b9_x27_505_, lean_object* v_S_506_, lean_object* v_e_507_, lean_object* v_inst_508_){
_start:
{
lean_object* v___x_509_; 
v___x_509_ = lp_mathlib_RingEquiv_piCongrLeft___redArg(v_e_507_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piCongrLeft___boxed(lean_object* v_00_u03b9_510_, lean_object* v_00_u03b9_x27_511_, lean_object* v_S_512_, lean_object* v_e_513_, lean_object* v_inst_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_mathlib_RingEquiv_piCongrLeft(v_00_u03b9_510_, v_00_u03b9_x27_511_, v_S_512_, v_e_513_, v_inst_514_);
lean_dec_ref(v_inst_514_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piEquivPiSubtypeProd___redArg(lean_object* v_inst_516_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg(v_inst_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piEquivPiSubtypeProd(lean_object* v_00_u03b9_518_, lean_object* v_p_519_, lean_object* v_inst_520_, lean_object* v_Y_521_, lean_object* v_inst_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg(v_inst_520_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piEquivPiSubtypeProd___boxed(lean_object* v_00_u03b9_524_, lean_object* v_p_525_, lean_object* v_inst_526_, lean_object* v_Y_527_, lean_object* v_inst_528_){
_start:
{
lean_object* v_res_529_; 
v_res_529_ = lp_mathlib_RingEquiv_piEquivPiSubtypeProd(v_00_u03b9_524_, v_p_525_, v_inst_526_, v_Y_527_, v_inst_528_);
lean_dec_ref(v_inst_528_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite___lam__0(lean_object* v_f_530_, lean_object* v_i_531_){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = lean_apply_1(v_f_530_, v_i_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite___lam__1(lean_object* v_f_533_, lean_object* v___y_534_){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = lean_apply_1(v_f_533_, v___y_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite(lean_object* v_00_u03b9_541_, lean_object* v_S_542_, lean_object* v_inst_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = ((lean_object*)(lp_mathlib_RingEquiv_piMulOpposite___closed__2));
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piMulOpposite___boxed(lean_object* v_00_u03b9_545_, lean_object* v_S_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_RingEquiv_piMulOpposite(v_00_u03b9_545_, v_S_546_, v_inst_547_);
lean_dec_ref(v_inst_547_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___lam__0(lean_object* v_f_549_, lean_object* v___y_550_){
_start:
{
lean_object* v_toFun_551_; lean_object* v___x_552_; 
v_toFun_551_ = lean_ctor_get(v_f_549_, 0);
lean_inc(v_toFun_551_);
lean_dec_ref(v_f_549_);
v___x_552_ = lean_apply_1(v_toFun_551_, v___y_550_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___redArg___lam__1(lean_object* v_f_553_, lean_object* v___y_554_){
_start:
{
lean_object* v_invFun_555_; lean_object* v___x_556_; 
v_invFun_555_ = lean_ctor_get(v_f_553_, 1);
lean_inc(v_invFun_555_);
lean_dec_ref(v_f_553_);
v___x_556_ = lean_apply_1(v_invFun_555_, v___y_554_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___redArg(lean_object* v_f_562_, lean_object* v_g_563_){
_start:
{
lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
v___x_564_ = ((lean_object*)(lp_mathlib_RingEquiv_prodCongr___redArg___closed__2));
v___x_565_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_564_, v_f_562_);
v___x_566_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_564_, v_g_563_);
v___x_567_ = lp_mathlib_Equiv_prodCongr___redArg(v___x_565_, v___x_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr(lean_object* v_R_568_, lean_object* v_R_x27_569_, lean_object* v_S_570_, lean_object* v_S_x27_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_f_576_, lean_object* v_g_577_){
_start:
{
lean_object* v___x_578_; 
v___x_578_ = lp_mathlib_RingEquiv_prodCongr___redArg(v_f_576_, v_g_577_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_prodCongr___boxed(lean_object* v_R_579_, lean_object* v_R_x27_580_, lean_object* v_S_581_, lean_object* v_S_x27_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_f_587_, lean_object* v_g_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_mathlib_RingEquiv_prodCongr(v_R_579_, v_R_x27_580_, v_S_581_, v_S_x27_582_, v_inst_583_, v_inst_584_, v_inst_585_, v_inst_586_, v_f_587_, v_g_588_);
lean_dec_ref(v_inst_586_);
lean_dec_ref(v_inst_585_);
lean_dec_ref(v_inst_584_);
lean_dec_ref(v_inst_583_);
return v_res_589_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_piOptionEquivProd___closed__0(void){
_start:
{
lean_object* v___x_590_; 
v___x_590_ = lp_mathlib_Equiv_piOptionEquivProd(lean_box(0), lean_box(0));
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piOptionEquivProd(lean_object* v_00_u03b9_591_, lean_object* v_R_592_, lean_object* v_inst_593_){
_start:
{
lean_object* v___x_594_; 
v___x_594_ = lean_obj_once(&lp_mathlib_RingEquiv_piOptionEquivProd___closed__0, &lp_mathlib_RingEquiv_piOptionEquivProd___closed__0_once, _init_lp_mathlib_RingEquiv_piOptionEquivProd___closed__0);
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_piOptionEquivProd___boxed(lean_object* v_00_u03b9_595_, lean_object* v_R_596_, lean_object* v_inst_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_RingEquiv_piOptionEquivProd(v_00_u03b9_595_, v_R_596_, v_inst_597_);
lean_dec_ref(v_inst_597_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom___redArg(lean_object* v_e_599_){
_start:
{
lean_object* v_toFun_600_; 
v_toFun_600_ = lean_ctor_get(v_e_599_, 0);
lean_inc(v_toFun_600_);
return v_toFun_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom___redArg___boxed(lean_object* v_e_601_){
_start:
{
lean_object* v_res_602_; 
v_res_602_ = lp_mathlib_RingEquiv_toNonUnitalRingHom___redArg(v_e_601_);
lean_dec_ref(v_e_601_);
return v_res_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom(lean_object* v_R_603_, lean_object* v_S_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_e_607_){
_start:
{
lean_object* v_toFun_608_; 
v_toFun_608_ = lean_ctor_get(v_e_607_, 0);
lean_inc(v_toFun_608_);
return v_toFun_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toNonUnitalRingHom___boxed(lean_object* v_R_609_, lean_object* v_S_610_, lean_object* v_inst_611_, lean_object* v_inst_612_, lean_object* v_e_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_mathlib_RingEquiv_toNonUnitalRingHom(v_R_609_, v_S_610_, v_inst_611_, v_inst_612_, v_e_613_);
lean_dec_ref(v_e_613_);
lean_dec_ref(v_inst_612_);
lean_dec_ref(v_inst_611_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom___redArg(lean_object* v_e_615_){
_start:
{
lean_object* v_toFun_616_; 
v_toFun_616_ = lean_ctor_get(v_e_615_, 0);
lean_inc(v_toFun_616_);
return v_toFun_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom___redArg___boxed(lean_object* v_e_617_){
_start:
{
lean_object* v_res_618_; 
v_res_618_ = lp_mathlib_RingEquiv_toRingHom___redArg(v_e_617_);
lean_dec_ref(v_e_617_);
return v_res_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom(lean_object* v_R_619_, lean_object* v_S_620_, lean_object* v_inst_621_, lean_object* v_inst_622_, lean_object* v_e_623_){
_start:
{
lean_object* v_toFun_624_; 
v_toFun_624_ = lean_ctor_get(v_e_623_, 0);
lean_inc(v_toFun_624_);
return v_toFun_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toRingHom___boxed(lean_object* v_R_625_, lean_object* v_S_626_, lean_object* v_inst_627_, lean_object* v_inst_628_, lean_object* v_e_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_RingEquiv_toRingHom(v_R_625_, v_S_626_, v_inst_627_, v_inst_628_, v_e_629_);
lean_dec_ref(v_e_629_);
lean_dec_ref(v_inst_628_);
lean_dec_ref(v_inst_627_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom___redArg(lean_object* v_e_631_){
_start:
{
lean_object* v_toFun_632_; 
v_toFun_632_ = lean_ctor_get(v_e_631_, 0);
lean_inc(v_toFun_632_);
return v_toFun_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom___redArg___boxed(lean_object* v_e_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_RingEquiv_toMonoidHom___redArg(v_e_633_);
lean_dec_ref(v_e_633_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom(lean_object* v_R_635_, lean_object* v_S_636_, lean_object* v_inst_637_, lean_object* v_inst_638_, lean_object* v_e_639_){
_start:
{
lean_object* v_toFun_640_; 
v_toFun_640_ = lean_ctor_get(v_e_639_, 0);
lean_inc(v_toFun_640_);
return v_toFun_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toMonoidHom___boxed(lean_object* v_R_641_, lean_object* v_S_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_e_645_){
_start:
{
lean_object* v_res_646_; 
v_res_646_ = lp_mathlib_RingEquiv_toMonoidHom(v_R_641_, v_S_642_, v_inst_643_, v_inst_644_, v_e_645_);
lean_dec_ref(v_e_645_);
lean_dec_ref(v_inst_644_);
lean_dec_ref(v_inst_643_);
return v_res_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom___redArg(lean_object* v_e_647_){
_start:
{
lean_object* v_toFun_648_; 
v_toFun_648_ = lean_ctor_get(v_e_647_, 0);
lean_inc(v_toFun_648_);
return v_toFun_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom___redArg___boxed(lean_object* v_e_649_){
_start:
{
lean_object* v_res_650_; 
v_res_650_ = lp_mathlib_RingEquiv_toAddMonoidHom___redArg(v_e_649_);
lean_dec_ref(v_e_649_);
return v_res_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom(lean_object* v_R_651_, lean_object* v_S_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_e_655_){
_start:
{
lean_object* v_toFun_656_; 
v_toFun_656_ = lean_ctor_get(v_e_655_, 0);
lean_inc(v_toFun_656_);
return v_toFun_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_toAddMonoidHom___boxed(lean_object* v_R_657_, lean_object* v_S_658_, lean_object* v_inst_659_, lean_object* v_inst_660_, lean_object* v_e_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_mathlib_RingEquiv_toAddMonoidHom(v_R_657_, v_S_658_, v_inst_659_, v_inst_660_, v_e_661_);
lean_dec_ref(v_e_661_);
lean_dec_ref(v_inst_660_);
lean_dec_ref(v_inst_659_);
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toRingEquiv___redArg(lean_object* v_inst_663_, lean_object* v_f_664_){
_start:
{
lean_object* v___x_665_; 
v___x_665_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_663_, v_f_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toRingEquiv(lean_object* v_R_666_, lean_object* v_S_667_, lean_object* v_F_668_, lean_object* v_inst_669_, lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_f_675_, lean_object* v_H_676_){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_673_, v_f_675_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toRingEquiv___boxed(lean_object* v_R_678_, lean_object* v_S_679_, lean_object* v_F_680_, lean_object* v_inst_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_f_687_, lean_object* v_H_688_){
_start:
{
lean_object* v_res_689_; 
v_res_689_ = lp_mathlib_MulEquiv_toRingEquiv(v_R_678_, v_S_679_, v_F_680_, v_inst_681_, v_inst_682_, v_inst_683_, v_inst_684_, v_inst_685_, v_inst_686_, v_f_687_, v_H_688_);
lean_dec(v_inst_684_);
lean_dec(v_inst_683_);
lean_dec(v_inst_682_);
lean_dec(v_inst_681_);
return v_res_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toRingEquiv___redArg(lean_object* v_inst_690_, lean_object* v_f_691_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_690_, v_f_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toRingEquiv(lean_object* v_R_693_, lean_object* v_S_694_, lean_object* v_F_695_, lean_object* v_inst_696_, lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_inst_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_f_702_, lean_object* v_H_703_){
_start:
{
lean_object* v___x_704_; 
v___x_704_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_700_, v_f_702_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toRingEquiv___boxed(lean_object* v_R_705_, lean_object* v_S_706_, lean_object* v_F_707_, lean_object* v_inst_708_, lean_object* v_inst_709_, lean_object* v_inst_710_, lean_object* v_inst_711_, lean_object* v_inst_712_, lean_object* v_inst_713_, lean_object* v_f_714_, lean_object* v_H_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_AddEquiv_toRingEquiv(v_R_705_, v_S_706_, v_F_707_, v_inst_708_, v_inst_709_, v_inst_710_, v_inst_711_, v_inst_712_, v_inst_713_, v_f_714_, v_H_715_);
lean_dec(v_inst_711_);
lean_dec(v_inst_710_);
lean_dec(v_inst_709_);
lean_dec(v_inst_708_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg___lam__0(lean_object* v_hom_717_, lean_object* v___y_718_){
_start:
{
lean_object* v___x_719_; 
v___x_719_ = lean_apply_1(v_hom_717_, v___y_718_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg___lam__1(lean_object* v_inv_720_, lean_object* v___y_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lean_apply_1(v_inv_720_, v___y_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg(lean_object* v_hom_723_, lean_object* v_inv_724_){
_start:
{
lean_object* v___f_725_; lean_object* v___f_726_; lean_object* v___x_727_; 
v___f_725_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_725_, 0, v_hom_723_);
v___f_726_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_726_, 0, v_inv_724_);
v___x_727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_727_, 0, v___f_725_);
lean_ctor_set(v___x_727_, 1, v___f_726_);
return v___x_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom(lean_object* v_R_728_, lean_object* v_S_729_, lean_object* v_inst_730_, lean_object* v_inst_731_, lean_object* v_hom_732_, lean_object* v_inv_733_, lean_object* v_hom__inv__id_734_, lean_object* v_inv__hom__id_735_){
_start:
{
lean_object* v___x_736_; 
v___x_736_ = lp_mathlib_RingEquiv_ofNonUnitalRingHom___redArg(v_hom_732_, v_inv_733_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofNonUnitalRingHom___boxed(lean_object* v_R_737_, lean_object* v_S_738_, lean_object* v_inst_739_, lean_object* v_inst_740_, lean_object* v_hom_741_, lean_object* v_inv_742_, lean_object* v_hom__inv__id_743_, lean_object* v_inv__hom__id_744_){
_start:
{
lean_object* v_res_745_; 
v_res_745_ = lp_mathlib_RingEquiv_ofNonUnitalRingHom(v_R_737_, v_S_738_, v_inst_739_, v_inst_740_, v_hom_741_, v_inv_742_, v_hom__inv__id_743_, v_inv__hom__id_744_);
lean_dec_ref(v_inst_740_);
lean_dec_ref(v_inst_739_);
return v_res_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom___redArg___lam__1(lean_object* v_g_746_, lean_object* v___y_747_){
_start:
{
lean_object* v___x_748_; 
v___x_748_ = lean_apply_1(v_g_746_, v___y_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom___redArg(lean_object* v_f_749_, lean_object* v_g_750_){
_start:
{
lean_object* v___f_751_; lean_object* v___f_752_; lean_object* v___x_753_; 
v___f_751_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_piMulOpposite___lam__1), 2, 1);
lean_closure_set(v___f_751_, 0, v_f_749_);
v___f_752_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofRingHom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_752_, 0, v_g_750_);
v___x_753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_753_, 0, v___f_751_);
lean_ctor_set(v___x_753_, 1, v___f_752_);
return v___x_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom(lean_object* v_R_754_, lean_object* v_S_755_, lean_object* v_inst_756_, lean_object* v_inst_757_, lean_object* v_f_758_, lean_object* v_g_759_, lean_object* v_h_u2081_760_, lean_object* v_h_u2082_761_){
_start:
{
lean_object* v___x_762_; 
v___x_762_ = lp_mathlib_RingEquiv_ofRingHom___redArg(v_f_758_, v_g_759_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofRingHom___boxed(lean_object* v_R_763_, lean_object* v_S_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_f_767_, lean_object* v_g_768_, lean_object* v_h_u2081_769_, lean_object* v_h_u2082_770_){
_start:
{
lean_object* v_res_771_; 
v_res_771_ = lp_mathlib_RingEquiv_ofRingHom(v_R_763_, v_S_764_, v_inst_765_, v_inst_766_, v_f_767_, v_g_768_, v_h_u2081_769_, v_h_u2082_770_);
lean_dec_ref(v_inst_766_);
lean_dec_ref(v_inst_765_);
return v_res_771_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_sumArrowEquivProdArrow___closed__0(void){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_mathlib_Equiv_sumArrowEquivProdArrow(lean_box(0), lean_box(0), lean_box(0));
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sumArrowEquivProdArrow(lean_object* v_00_u03b1_773_, lean_object* v_00_u03b2_774_, lean_object* v_R_775_, lean_object* v_inst_776_){
_start:
{
lean_object* v___x_777_; 
v___x_777_ = lean_obj_once(&lp_mathlib_RingEquiv_sumArrowEquivProdArrow___closed__0, &lp_mathlib_RingEquiv_sumArrowEquivProdArrow___closed__0_once, _init_lp_mathlib_RingEquiv_sumArrowEquivProdArrow___closed__0);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_sumArrowEquivProdArrow___boxed(lean_object* v_00_u03b1_778_, lean_object* v_00_u03b2_779_, lean_object* v_R_780_, lean_object* v_inst_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_mathlib_RingEquiv_sumArrowEquivProdArrow(v_00_u03b1_778_, v_00_u03b2_779_, v_R_780_, v_inst_781_);
lean_dec_ref(v_inst_781_);
return v_res_782_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
}
#ifdef __cplusplus
}
#endif
