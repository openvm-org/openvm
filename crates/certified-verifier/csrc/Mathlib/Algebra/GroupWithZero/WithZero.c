// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.WithZero
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.TypeTags.Basic public import Mathlib.Algebra.Group.WithOne.Defs public import Mathlib.Algebra.GroupWithZero.Equiv public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.Data.Nat.Cast.Defs public import Mathlib.Data.Option.NAry public import Mathlib.Util.CompileInductive
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
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Option_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_WithZero_coe(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_recZeroCoe___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_instAddMonoid___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Units_mk0___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroOneClass(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithZero_coeMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_coe, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_WithZero_coeMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_WithZero_coeMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coeMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coeMonoidHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithZero_lift_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_lift_x27___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_lift_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_WithZero_lift_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonoidWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_invOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_invOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_div(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvOneMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvOneMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instInvolutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instInvolutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__7(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_withZero___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_withZero___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_withZero___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_withZero___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoidWithOne___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoidWithOne(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_WithZero_term___u1d50_u2070___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "WithZero"};
static const lean_object* lp_mathlib_WithZero_term___u1d50_u2070___closed__0 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__0_value;
static const lean_string_object lp_mathlib_WithZero_term___u1d50_u2070___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "term_ᵐ⁰"};
static const lean_object* lp_mathlib_WithZero_term___u1d50_u2070___closed__1 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__1_value;
static const lean_ctor_object lp_mathlib_WithZero_term___u1d50_u2070___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__0_value),LEAN_SCALAR_PTR_LITERAL(4, 24, 249, 20, 56, 201, 66, 156)}};
static const lean_ctor_object lp_mathlib_WithZero_term___u1d50_u2070___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__2_value_aux_0),((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 165, 245, 214, 228, 209, 49, 56)}};
static const lean_object* lp_mathlib_WithZero_term___u1d50_u2070___closed__2 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__2_value;
static const lean_string_object lp_mathlib_WithZero_term___u1d50_u2070___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "ᵐ⁰"};
static const lean_object* lp_mathlib_WithZero_term___u1d50_u2070___closed__3 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__3_value;
static const lean_ctor_object lp_mathlib_WithZero_term___u1d50_u2070___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__3_value)}};
static const lean_object* lp_mathlib_WithZero_term___u1d50_u2070___closed__4 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__4_value;
static const lean_ctor_object lp_mathlib_WithZero_term___u1d50_u2070___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__4_value)}};
static const lean_object* lp_mathlib_WithZero_term___u1d50_u2070___closed__5 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_WithZero_term___u1d50_u2070 = (const lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__5_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_<|_"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__0 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__0_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 38, 96, 140, 215, 46, 31, 82)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__1 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__1_value;
static lean_once_cell_t lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__2;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero_term___u1d50_u2070___closed__0_value),LEAN_SCALAR_PTR_LITERAL(4, 24, 249, 20, 56, 201, 66, 156)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__3 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__3_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__4 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__4_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__3_value)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__5 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__5_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__6 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__6_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__4_value),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__6_value)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__7 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__7_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "<|"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__8 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__8_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__9 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__9_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__10 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__10_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__11 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__11_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__12 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__12_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Multiplicative"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__14 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__14_value;
static lean_once_cell_t lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__15;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(179, 156, 201, 191, 87, 50, 86, 127)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__16 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__16_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__17 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__17_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__16_value)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__18 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__18_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__19 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__19_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__17_value),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__19_value)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__20 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__20_value;
static const lean_string_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__21 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__21_value;
static const lean_ctor_object lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__22 = (const lean_object*)&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithZero_exp___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_exp___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_exp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_exp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithZero_log___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_log___redArg___closed__0;
static lean_once_cell_t lp_mathlib_WithZero_log___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithZero_log___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_one___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2_, 0, v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_one(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5_, 0, v_inst_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0(lean_object* v_inst_6_, lean_object* v_x1_7_, lean_object* v_x2_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_apply_2(v_inst_6_, v_x1_7_, v_x2_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroClass___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_11_, 0, v_inst_10_);
v___x_12_ = lean_box(0);
v___x_13_ = lean_alloc_closure((void*)(lp_mathlib_Option_map_u2082), 6, 4);
lean_closure_set(v___x_13_, 0, lean_box(0));
lean_closure_set(v___x_13_, 1, lean_box(0));
lean_closure_set(v___x_13_, 2, lean_box(0));
lean_closure_set(v___x_13_, 3, v___f_11_);
v___x_14_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v___x_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroClass(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_WithZero_instMulZeroClass___redArg(v_inst_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instSemigroupWithZero___redArg(lean_object* v_inst_18_){
_start:
{
lean_object* v___f_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_19_, 0, v_inst_18_);
v___x_20_ = lean_alloc_closure((void*)(lp_mathlib_Option_map_u2082), 6, 4);
lean_closure_set(v___x_20_, 0, lean_box(0));
lean_closure_set(v___x_20_, 1, lean_box(0));
lean_closure_set(v___x_20_, 2, lean_box(0));
lean_closure_set(v___x_20_, 3, v___f_19_);
v___x_21_ = lean_box(0);
v___x_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_22_, 0, v___x_20_);
lean_ctor_set(v___x_22_, 1, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instSemigroupWithZero(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_WithZero_instSemigroupWithZero___redArg(v_inst_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommSemigroup___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___f_27_; lean_object* v___x_28_; 
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_27_, 0, v_inst_26_);
v___x_28_ = lean_alloc_closure((void*)(lp_mathlib_Option_map_u2082), 6, 4);
lean_closure_set(v___x_28_, 0, lean_box(0));
lean_closure_set(v___x_28_, 1, lean_box(0));
lean_closure_set(v___x_28_, 2, lean_box(0));
lean_closure_set(v___x_28_, 3, v___f_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommSemigroup(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_WithZero_instCommSemigroup___redArg(v_inst_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroOneClass___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; lean_object* v_toOne_34_; lean_object* v_toMul_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_47_; 
v___x_33_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_32_);
v_toOne_34_ = lean_ctor_get(v___x_33_, 0);
v_toMul_35_ = lean_ctor_get(v___x_33_, 1);
v_isSharedCheck_47_ = !lean_is_exclusive(v___x_33_);
if (v_isSharedCheck_47_ == 0)
{
v___x_37_ = v___x_33_;
v_isShared_38_ = v_isSharedCheck_47_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_toMul_35_);
lean_inc(v_toOne_34_);
lean_dec(v___x_33_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_47_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_39_; lean_object* v___f_40_; lean_object* v___x_41_; lean_object* v___x_43_; 
v___x_39_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_39_, 0, v_toOne_34_);
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_40_, 0, v_toMul_35_);
v___x_41_ = lean_alloc_closure((void*)(lp_mathlib_Option_map_u2082), 6, 4);
lean_closure_set(v___x_41_, 0, lean_box(0));
lean_closure_set(v___x_41_, 1, lean_box(0));
lean_closure_set(v___x_41_, 2, lean_box(0));
lean_closure_set(v___x_41_, 3, v___f_40_);
if (v_isShared_38_ == 0)
{
lean_ctor_set(v___x_37_, 1, v___x_41_);
lean_ctor_set(v___x_37_, 0, v___x_39_);
v___x_43_ = v___x_37_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v___x_39_);
lean_ctor_set(v_reuseFailAlloc_46_, 1, v___x_41_);
v___x_43_ = v_reuseFailAlloc_46_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_44_ = lean_box(0);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_43_);
lean_ctor_set(v___x_45_, 1, v___x_44_);
return v___x_45_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMulZeroOneClass(lean_object* v_00_u03b1_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_WithZero_instMulZeroOneClass___redArg(v_inst_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coeMonoidHom(lean_object* v_00_u03b1_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = ((lean_object*)(lp_mathlib_WithZero_coeMonoidHom___closed__0));
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coeMonoidHom___boxed(lean_object* v_00_u03b1_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_WithZero_coeMonoidHom(v_00_u03b1_55_, v_inst_56_);
lean_dec_ref(v_inst_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__0(lean_object* v_F_58_, lean_object* v___y_59_){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_60_ = ((lean_object*)(lp_mathlib_WithZero_coeMonoidHom___closed__0));
v___x_61_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_60_, v_F_58_, v___y_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__1(lean_object* v_f_62_, lean_object* v___y_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lean_apply_1(v_f_62_, v___y_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__2(lean_object* v_toZero_65_, lean_object* v_f_66_, lean_object* v___y_67_){
_start:
{
lean_object* v___f_68_; lean_object* v___x_69_; 
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_lift_x27___redArg___lam__1), 2, 1);
lean_closure_set(v___f_68_, 0, v_f_66_);
v___x_69_ = lp_mathlib_WithZero_recZeroCoe___redArg(v_toZero_65_, v___f_68_, v___y_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg___lam__2___boxed(lean_object* v_toZero_70_, lean_object* v_f_71_, lean_object* v___y_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_WithZero_lift_x27___redArg___lam__2(v_toZero_70_, v_f_71_, v___y_72_);
lean_dec(v_toZero_70_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___redArg(lean_object* v_inst_75_){
_start:
{
lean_object* v___x_76_; lean_object* v_toZero_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_86_; 
v___x_76_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_75_);
v_toZero_77_ = lean_ctor_get(v___x_76_, 1);
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_76_);
if (v_isSharedCheck_86_ == 0)
{
lean_object* v_unused_87_; 
v_unused_87_ = lean_ctor_get(v___x_76_, 0);
lean_dec(v_unused_87_);
v___x_79_ = v___x_76_;
v_isShared_80_ = v_isSharedCheck_86_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_toZero_77_);
lean_dec(v___x_76_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_86_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___f_81_; lean_object* v___f_82_; lean_object* v___x_84_; 
v___f_81_ = ((lean_object*)(lp_mathlib_WithZero_lift_x27___redArg___closed__0));
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_lift_x27___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_82_, 0, v_toZero_77_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 1, v___f_81_);
lean_ctor_set(v___x_79_, 0, v___f_82_);
v___x_84_ = v___x_79_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v___f_82_);
lean_ctor_set(v_reuseFailAlloc_85_, 1, v___f_81_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27(lean_object* v_00_u03b1_88_, lean_object* v_00_u03b2_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_WithZero_lift_x27___redArg(v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_lift_x27___boxed(lean_object* v_00_u03b1_93_, lean_object* v_00_u03b2_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_WithZero_lift_x27(v_00_u03b1_93_, v_00_u03b2_94_, v_inst_95_, v_inst_96_);
lean_dec_ref(v_inst_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_x27___redArg(lean_object* v_inst_98_, lean_object* v_f_99_){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v_toFun_102_; lean_object* v___x_103_; lean_object* v___f_104_; lean_object* v___x_105_; 
v___x_100_ = lp_mathlib_WithZero_instMulZeroOneClass___redArg(v_inst_98_);
v___x_101_ = lp_mathlib_WithZero_lift_x27___redArg(v___x_100_);
v_toFun_102_ = lean_ctor_get(v___x_101_, 0);
lean_inc(v_toFun_102_);
lean_dec_ref(v___x_101_);
v___x_103_ = ((lean_object*)(lp_mathlib_WithZero_coeMonoidHom___closed__0));
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_104_, 0, v_f_99_);
lean_closure_set(v___f_104_, 1, v___x_103_);
v___x_105_ = lean_apply_1(v_toFun_102_, v___f_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_x27(lean_object* v_00_u03b1_106_, lean_object* v_00_u03b2_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_f_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_WithZero_map_x27___redArg(v_inst_109_, v_f_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_x27___boxed(lean_object* v_00_u03b1_112_, lean_object* v_00_u03b2_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_f_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_WithZero_map_x27(v_00_u03b1_112_, v_00_u03b2_113_, v_inst_114_, v_inst_115_, v_f_116_);
lean_dec_ref(v_inst_114_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow___redArg___lam__0(lean_object* v_inst_118_, lean_object* v___x_119_, lean_object* v_inst_120_, lean_object* v_x_121_, lean_object* v_x_122_){
_start:
{
if (lean_obj_tag(v_x_121_) == 0)
{
lean_object* v_zero_123_; uint8_t v_isZero_124_; 
lean_dec(v_inst_120_);
v_zero_123_ = lean_unsigned_to_nat(0u);
v_isZero_124_ = lean_nat_dec_eq(v_x_122_, v_zero_123_);
lean_dec(v_x_122_);
if (v_isZero_124_ == 1)
{
lean_object* v___x_125_; 
v___x_125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_125_, 0, v_inst_118_);
return v___x_125_;
}
else
{
lean_dec(v_inst_118_);
lean_inc(v___x_119_);
return v___x_119_;
}
}
else
{
lean_object* v_val_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_134_; 
lean_dec(v_inst_118_);
v_val_126_ = lean_ctor_get(v_x_121_, 0);
v_isSharedCheck_134_ = !lean_is_exclusive(v_x_121_);
if (v_isSharedCheck_134_ == 0)
{
v___x_128_ = v_x_121_;
v_isShared_129_ = v_isSharedCheck_134_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_val_126_);
lean_dec(v_x_121_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_134_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_130_; lean_object* v___x_132_; 
v___x_130_ = lean_apply_2(v_inst_120_, v_val_126_, v_x_122_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 0, v___x_130_);
v___x_132_ = v___x_128_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_130_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow___redArg___lam__0___boxed(lean_object* v_inst_135_, lean_object* v___x_136_, lean_object* v_inst_137_, lean_object* v_x_138_, lean_object* v_x_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_WithZero_pow___redArg___lam__0(v_inst_135_, v___x_136_, v_inst_137_, v_x_138_, v_x_139_);
lean_dec(v___x_136_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v___x_143_; lean_object* v___f_144_; 
v___x_143_ = lean_box(0);
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_pow___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_144_, 0, v_inst_141_);
lean_closure_set(v___f_144_, 1, v___x_143_);
lean_closure_set(v___f_144_, 2, v_inst_142_);
return v___f_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_pow(lean_object* v_00_u03b1_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_mathlib_WithZero_pow___redArg(v_inst_146_, v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonoidWithZero___redArg___lam__0(lean_object* v_toOne_149_, lean_object* v_toNPow_150_, lean_object* v_n_151_, lean_object* v_a_152_){
_start:
{
if (lean_obj_tag(v_a_152_) == 0)
{
lean_object* v_zero_153_; uint8_t v_isZero_154_; 
lean_dec(v_toNPow_150_);
v_zero_153_ = lean_unsigned_to_nat(0u);
v_isZero_154_ = lean_nat_dec_eq(v_n_151_, v_zero_153_);
lean_dec(v_n_151_);
if (v_isZero_154_ == 1)
{
lean_object* v___x_155_; 
v___x_155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_155_, 0, v_toOne_149_);
return v___x_155_;
}
else
{
lean_object* v___x_156_; 
lean_dec(v_toOne_149_);
v___x_156_ = lean_box(0);
return v___x_156_;
}
}
else
{
lean_object* v_val_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_165_; 
lean_dec(v_toOne_149_);
v_val_157_ = lean_ctor_get(v_a_152_, 0);
v_isSharedCheck_165_ = !lean_is_exclusive(v_a_152_);
if (v_isSharedCheck_165_ == 0)
{
v___x_159_ = v_a_152_;
v_isShared_160_ = v_isSharedCheck_165_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_val_157_);
lean_dec(v_a_152_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_165_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___x_161_; lean_object* v___x_163_; 
v___x_161_ = lean_apply_2(v_toNPow_150_, v_n_151_, v_val_157_);
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 0, v___x_161_);
v___x_163_ = v___x_159_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v___x_161_);
v___x_163_ = v_reuseFailAlloc_164_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
return v___x_163_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonoidWithZero___redArg(lean_object* v_inst_166_){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v_toMulOneClass_169_; lean_object* v___x_170_; lean_object* v_toOne_171_; lean_object* v_toMul_172_; lean_object* v___x_173_; lean_object* v_toOne_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_192_; 
v___x_167_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_166_);
lean_inc_ref(v___x_167_);
v___x_168_ = lp_mathlib_WithZero_instMulZeroOneClass___redArg(v___x_167_);
v_toMulOneClass_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc_ref(v_toMulOneClass_169_);
lean_dec_ref(v___x_168_);
v___x_170_ = lean_box(0);
v_toOne_171_ = lean_ctor_get(v_toMulOneClass_169_, 0);
lean_inc(v_toOne_171_);
v_toMul_172_ = lean_ctor_get(v_toMulOneClass_169_, 1);
lean_inc(v_toMul_172_);
lean_dec_ref(v_toMulOneClass_169_);
v___x_173_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_167_);
v_toOne_174_ = lean_ctor_get(v___x_173_, 0);
v_isSharedCheck_192_ = !lean_is_exclusive(v___x_173_);
if (v_isSharedCheck_192_ == 0)
{
lean_object* v_unused_193_; 
v_unused_193_ = lean_ctor_get(v___x_173_, 1);
lean_dec(v_unused_193_);
v___x_176_ = v___x_173_;
v_isShared_177_ = v_isSharedCheck_192_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_toOne_174_);
lean_dec(v___x_173_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_192_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v_toNPow_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_189_; 
v_toNPow_178_ = lean_ctor_get(v_inst_166_, 2);
v_isSharedCheck_189_ = !lean_is_exclusive(v_inst_166_);
if (v_isSharedCheck_189_ == 0)
{
lean_object* v_unused_190_; lean_object* v_unused_191_; 
v_unused_190_ = lean_ctor_get(v_inst_166_, 1);
lean_dec(v_unused_190_);
v_unused_191_ = lean_ctor_get(v_inst_166_, 0);
lean_dec(v_unused_191_);
v___x_180_ = v_inst_166_;
v_isShared_181_ = v_isSharedCheck_189_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_toNPow_178_);
lean_dec(v_inst_166_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_189_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___f_182_; lean_object* v___x_184_; 
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instMonoidWithZero___redArg___lam__0), 4, 2);
lean_closure_set(v___f_182_, 0, v_toOne_174_);
lean_closure_set(v___f_182_, 1, v_toNPow_178_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 2, v___f_182_);
lean_ctor_set(v___x_180_, 1, v_toMul_172_);
lean_ctor_set(v___x_180_, 0, v_toOne_171_);
v___x_184_ = v___x_180_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_toOne_171_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v_toMul_172_);
lean_ctor_set(v_reuseFailAlloc_188_, 2, v___f_182_);
v___x_184_ = v_reuseFailAlloc_188_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
lean_object* v___x_186_; 
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 1, v___x_170_);
lean_ctor_set(v___x_176_, 0, v___x_184_);
v___x_186_ = v___x_176_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_184_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v___x_170_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
return v___x_186_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonoidWithZero(lean_object* v_00_u03b1_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib_WithZero_instMonoidWithZero___redArg(v_inst_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommMonoidWithZero___redArg(lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; lean_object* v_toMonoid_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_207_; 
v___x_198_ = lp_mathlib_WithZero_instMonoidWithZero___redArg(v_inst_197_);
v_toMonoid_199_ = lean_ctor_get(v___x_198_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_198_);
if (v_isSharedCheck_207_ == 0)
{
lean_object* v_unused_208_; 
v_unused_208_ = lean_ctor_get(v___x_198_, 1);
lean_dec(v_unused_208_);
v___x_201_ = v___x_198_;
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_toMonoid_199_);
lean_dec(v___x_198_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_203_; lean_object* v___x_205_; 
v___x_203_ = lean_box(0);
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 1, v___x_203_);
v___x_205_ = v___x_201_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_toMonoid_199_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v___x_203_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommMonoidWithZero(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_WithZero_instCommMonoidWithZero___redArg(v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inv___redArg___lam__0(lean_object* v_inst_212_, lean_object* v_a_213_){
_start:
{
if (lean_obj_tag(v_a_213_) == 0)
{
lean_dec(v_inst_212_);
return v_a_213_;
}
else
{
lean_object* v_val_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_222_; 
v_val_214_ = lean_ctor_get(v_a_213_, 0);
v_isSharedCheck_222_ = !lean_is_exclusive(v_a_213_);
if (v_isSharedCheck_222_ == 0)
{
v___x_216_ = v_a_213_;
v_isShared_217_ = v_isSharedCheck_222_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_val_214_);
lean_dec(v_a_213_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_222_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_218_; lean_object* v___x_220_; 
v___x_218_ = lean_apply_1(v_inst_212_, v_val_214_);
if (v_isShared_217_ == 0)
{
lean_ctor_set(v___x_216_, 0, v___x_218_);
v___x_220_ = v___x_216_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v___x_218_);
v___x_220_ = v_reuseFailAlloc_221_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
return v___x_220_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inv___redArg(lean_object* v_inst_223_){
_start:
{
lean_object* v___f_224_; 
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_224_, 0, v_inst_223_);
return v___f_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inv(lean_object* v_00_u03b1_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___f_227_; 
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_227_, 0, v_inst_226_);
return v___f_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_invOneClass___redArg(lean_object* v_inst_228_){
_start:
{
lean_object* v_toOne_229_; lean_object* v_toInv_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_239_; 
v_toOne_229_ = lean_ctor_get(v_inst_228_, 0);
v_toInv_230_ = lean_ctor_get(v_inst_228_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v_inst_228_);
if (v_isSharedCheck_239_ == 0)
{
v___x_232_ = v_inst_228_;
v_isShared_233_ = v_isSharedCheck_239_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_toInv_230_);
lean_inc(v_toOne_229_);
lean_dec(v_inst_228_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_239_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_234_; lean_object* v___f_235_; lean_object* v___x_237_; 
v___x_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_234_, 0, v_toOne_229_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_235_, 0, v_toInv_230_);
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 1, v___f_235_);
lean_ctor_set(v___x_232_, 0, v___x_234_);
v___x_237_ = v___x_232_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v___x_234_);
lean_ctor_set(v_reuseFailAlloc_238_, 1, v___f_235_);
v___x_237_ = v_reuseFailAlloc_238_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
return v___x_237_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_invOneClass(lean_object* v_00_u03b1_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_WithZero_invOneClass___redArg(v_inst_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_div___redArg(lean_object* v_inst_243_){
_start:
{
lean_object* v___f_244_; lean_object* v___x_245_; 
v___f_244_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instMulZeroClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_244_, 0, v_inst_243_);
v___x_245_ = lean_alloc_closure((void*)(lp_mathlib_Option_map_u2082), 6, 4);
lean_closure_set(v___x_245_, 0, lean_box(0));
lean_closure_set(v___x_245_, 1, lean_box(0));
lean_closure_set(v___x_245_, 2, lean_box(0));
lean_closure_set(v___x_245_, 3, v___f_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_div(lean_object* v_00_u03b1_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib_WithZero_div___redArg(v_inst_247_);
return v___x_248_;
}
}
static lean_object* _init_lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v_natZero_249_; lean_object* v_intZero_250_; 
v_natZero_249_ = lean_unsigned_to_nat(0u);
v_intZero_250_ = lean_nat_to_int(v_natZero_249_);
return v_intZero_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt___redArg___lam__0(lean_object* v_inst_251_, lean_object* v___x_252_, lean_object* v_inst_253_, lean_object* v_x_254_, lean_object* v_x_255_){
_start:
{
if (lean_obj_tag(v_x_254_) == 0)
{
lean_object* v_natZero_256_; lean_object* v_intZero_257_; uint8_t v_isNeg_258_; 
lean_dec(v_inst_253_);
v_natZero_256_ = lean_unsigned_to_nat(0u);
v_intZero_257_ = lean_obj_once(&lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0, &lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0_once, _init_lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0);
v_isNeg_258_ = lean_int_dec_lt(v_x_255_, v_intZero_257_);
if (v_isNeg_258_ == 0)
{
lean_object* v_a_259_; uint8_t v_isZero_260_; 
v_a_259_ = lean_nat_abs(v_x_255_);
lean_dec(v_x_255_);
v_isZero_260_ = lean_nat_dec_eq(v_a_259_, v_natZero_256_);
lean_dec(v_a_259_);
if (v_isZero_260_ == 1)
{
lean_object* v___x_261_; 
v___x_261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_261_, 0, v_inst_251_);
return v___x_261_;
}
else
{
lean_dec(v_inst_251_);
lean_inc(v___x_252_);
return v___x_252_;
}
}
else
{
lean_dec(v_x_255_);
lean_dec(v_inst_251_);
lean_inc(v___x_252_);
return v___x_252_;
}
}
else
{
lean_object* v_val_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_270_; 
lean_dec(v_inst_251_);
v_val_262_ = lean_ctor_get(v_x_254_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v_x_254_);
if (v_isSharedCheck_270_ == 0)
{
v___x_264_ = v_x_254_;
v_isShared_265_ = v_isSharedCheck_270_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_val_262_);
lean_dec(v_x_254_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_270_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v___x_266_; lean_object* v___x_268_; 
v___x_266_ = lean_apply_2(v_inst_253_, v_val_262_, v_x_255_);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 0, v___x_266_);
v___x_268_ = v___x_264_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v___x_266_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt___redArg___lam__0___boxed(lean_object* v_inst_271_, lean_object* v___x_272_, lean_object* v_inst_273_, lean_object* v_x_274_, lean_object* v_x_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_WithZero_instPowInt___redArg___lam__0(v_inst_271_, v___x_272_, v_inst_273_, v_x_274_, v_x_275_);
lean_dec(v___x_272_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt___redArg(lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; lean_object* v___f_280_; 
v___x_279_ = lean_box(0);
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instPowInt___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_280_, 0, v_inst_277_);
lean_closure_set(v___f_280_, 1, v___x_279_);
lean_closure_set(v___f_280_, 2, v_inst_278_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instPowInt(lean_object* v_00_u03b1_281_, lean_object* v_inst_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_WithZero_instPowInt___redArg(v_inst_282_, v_inst_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvMonoid___redArg___lam__0(lean_object* v_toOne_285_, lean_object* v_toZPow_286_, lean_object* v_n_287_, lean_object* v_a_288_){
_start:
{
if (lean_obj_tag(v_a_288_) == 0)
{
lean_object* v___x_289_; lean_object* v_natZero_290_; lean_object* v_intZero_291_; uint8_t v_isNeg_292_; 
lean_dec(v_toZPow_286_);
v___x_289_ = lean_box(0);
v_natZero_290_ = lean_unsigned_to_nat(0u);
v_intZero_291_ = lean_obj_once(&lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0, &lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0_once, _init_lp_mathlib_WithZero_instPowInt___redArg___lam__0___closed__0);
v_isNeg_292_ = lean_int_dec_lt(v_n_287_, v_intZero_291_);
if (v_isNeg_292_ == 0)
{
lean_object* v_a_293_; uint8_t v_isZero_294_; 
v_a_293_ = lean_nat_abs(v_n_287_);
lean_dec(v_n_287_);
v_isZero_294_ = lean_nat_dec_eq(v_a_293_, v_natZero_290_);
lean_dec(v_a_293_);
if (v_isZero_294_ == 1)
{
lean_object* v___x_295_; 
v___x_295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_295_, 0, v_toOne_285_);
return v___x_295_;
}
else
{
lean_dec(v_toOne_285_);
return v___x_289_;
}
}
else
{
lean_dec(v_n_287_);
lean_dec(v_toOne_285_);
return v___x_289_;
}
}
else
{
lean_object* v_val_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_304_; 
lean_dec(v_toOne_285_);
v_val_296_ = lean_ctor_get(v_a_288_, 0);
v_isSharedCheck_304_ = !lean_is_exclusive(v_a_288_);
if (v_isSharedCheck_304_ == 0)
{
v___x_298_ = v_a_288_;
v_isShared_299_ = v_isSharedCheck_304_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_val_296_);
lean_dec(v_a_288_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_304_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v___x_300_; lean_object* v___x_302_; 
v___x_300_ = lean_apply_2(v_toZPow_286_, v_n_287_, v_val_296_);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 0, v___x_300_);
v___x_302_ = v___x_298_;
goto v_reusejp_301_;
}
else
{
lean_object* v_reuseFailAlloc_303_; 
v_reuseFailAlloc_303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_303_, 0, v___x_300_);
v___x_302_ = v_reuseFailAlloc_303_;
goto v_reusejp_301_;
}
v_reusejp_301_:
{
return v___x_302_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvMonoid___redArg(lean_object* v_inst_305_){
_start:
{
lean_object* v_toMonoid_306_; lean_object* v_toInv_307_; lean_object* v_toDiv_308_; lean_object* v_toZPow_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_324_; 
v_toMonoid_306_ = lean_ctor_get(v_inst_305_, 0);
v_toInv_307_ = lean_ctor_get(v_inst_305_, 1);
v_toDiv_308_ = lean_ctor_get(v_inst_305_, 2);
v_toZPow_309_ = lean_ctor_get(v_inst_305_, 3);
v_isSharedCheck_324_ = !lean_is_exclusive(v_inst_305_);
if (v_isSharedCheck_324_ == 0)
{
v___x_311_ = v_inst_305_;
v_isShared_312_ = v_isSharedCheck_324_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_toZPow_309_);
lean_inc(v_toDiv_308_);
lean_inc(v_toInv_307_);
lean_inc(v_toMonoid_306_);
lean_dec(v_inst_305_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_324_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_313_; lean_object* v_toMonoid_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v_toOne_317_; lean_object* v___f_318_; lean_object* v___x_319_; lean_object* v___f_320_; lean_object* v___x_322_; 
lean_inc_ref(v_toMonoid_306_);
v___x_313_ = lp_mathlib_WithZero_instMonoidWithZero___redArg(v_toMonoid_306_);
v_toMonoid_314_ = lean_ctor_get(v___x_313_, 0);
lean_inc_ref(v_toMonoid_314_);
lean_dec_ref(v___x_313_);
v___x_315_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_306_);
lean_dec_ref(v_toMonoid_306_);
v___x_316_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_315_);
v_toOne_317_ = lean_ctor_get(v___x_316_, 0);
lean_inc(v_toOne_317_);
lean_dec_ref(v___x_316_);
v___f_318_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_318_, 0, v_toInv_307_);
v___x_319_ = lp_mathlib_WithZero_div___redArg(v_toDiv_308_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instDivInvMonoid___redArg___lam__0), 4, 2);
lean_closure_set(v___f_320_, 0, v_toOne_317_);
lean_closure_set(v___f_320_, 1, v_toZPow_309_);
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 3, v___f_320_);
lean_ctor_set(v___x_311_, 2, v___x_319_);
lean_ctor_set(v___x_311_, 1, v___f_318_);
lean_ctor_set(v___x_311_, 0, v_toMonoid_314_);
v___x_322_ = v___x_311_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v_toMonoid_314_);
lean_ctor_set(v_reuseFailAlloc_323_, 1, v___f_318_);
lean_ctor_set(v_reuseFailAlloc_323_, 2, v___x_319_);
lean_ctor_set(v_reuseFailAlloc_323_, 3, v___f_320_);
v___x_322_ = v_reuseFailAlloc_323_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
return v___x_322_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvMonoid(lean_object* v_00_u03b1_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; 
v___x_327_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_326_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvOneMonoid___redArg(lean_object* v_inst_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivInvOneMonoid(lean_object* v_00_u03b1_330_, lean_object* v_inst_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instInvolutiveInv___redArg(lean_object* v_inst_333_){
_start:
{
lean_object* v___f_334_; 
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_334_, 0, v_inst_333_);
return v___f_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instInvolutiveInv(lean_object* v_00_u03b1_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___f_337_; 
v___f_337_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_337_, 0, v_inst_336_);
return v___f_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionMonoid___redArg(lean_object* v_inst_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionMonoid(lean_object* v_00_u03b1_340_, lean_object* v_inst_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionCommMonoid___redArg(lean_object* v_inst_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instDivisionCommMonoid(lean_object* v_00_u03b1_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instGroupWithZero___redArg(lean_object* v_inst_348_){
_start:
{
lean_object* v_toMonoid_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v_toInv_352_; lean_object* v_toDiv_353_; lean_object* v_toZPow_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
v_toMonoid_349_ = lean_ctor_get(v_inst_348_, 0);
lean_inc_ref(v_toMonoid_349_);
v___x_350_ = lp_mathlib_WithZero_instMonoidWithZero___redArg(v_toMonoid_349_);
v___x_351_ = lp_mathlib_WithZero_instDivInvMonoid___redArg(v_inst_348_);
v_toInv_352_ = lean_ctor_get(v___x_351_, 1);
v_toDiv_353_ = lean_ctor_get(v___x_351_, 2);
v_toZPow_354_ = lean_ctor_get(v___x_351_, 3);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_351_);
if (v_isSharedCheck_361_ == 0)
{
lean_object* v_unused_362_; 
v_unused_362_ = lean_ctor_get(v___x_351_, 0);
lean_dec(v_unused_362_);
v___x_356_ = v___x_351_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_toZPow_354_);
lean_inc(v_toDiv_353_);
lean_inc(v_toInv_352_);
lean_dec(v___x_351_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v___x_350_);
v___x_359_ = v___x_356_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v_toInv_352_);
lean_ctor_set(v_reuseFailAlloc_360_, 2, v_toDiv_353_);
lean_ctor_set(v_reuseFailAlloc_360_, 3, v_toZPow_354_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instGroupWithZero(lean_object* v_00_u03b1_363_, lean_object* v_inst_364_){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lp_mathlib_WithZero_instGroupWithZero___redArg(v_inst_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__0(lean_object* v_a_366_){
_start:
{
lean_object* v_val_367_; lean_object* v_val_368_; 
v_val_367_ = lean_ctor_get(v_a_366_, 0);
v_val_368_ = lean_ctor_get(v_val_367_, 0);
lean_inc(v_val_368_);
return v_val_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__0___boxed(lean_object* v_a_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__0(v_a_369_);
lean_dec_ref(v_a_369_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__1(lean_object* v___x_371_, lean_object* v_a_372_){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_373_, 0, v_a_372_);
v___x_374_ = lp_mathlib_Units_mk0___redArg(v___x_371_, v___x_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(lean_object* v_inst_376_){
_start:
{
lean_object* v___f_377_; lean_object* v___x_378_; lean_object* v___f_379_; lean_object* v___x_380_; 
v___f_377_ = ((lean_object*)(lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___closed__0));
v___x_378_ = lp_mathlib_WithZero_instGroupWithZero___redArg(v_inst_376_);
v___f_379_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_unitsWithZeroEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_379_, 0, v___x_378_);
v___x_380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_380_, 0, v___f_377_);
lean_ctor_set(v___x_380_, 1, v___f_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unitsWithZeroEquiv(lean_object* v_00_u03b1_381_, lean_object* v_inst_382_){
_start:
{
lean_object* v___x_383_; 
v___x_383_ = lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(v_inst_382_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__0(lean_object* v_self_384_){
_start:
{
lean_object* v_val_385_; 
v_val_385_ = lean_ctor_get(v_self_384_, 0);
lean_inc(v_val_385_);
return v_val_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__0___boxed(lean_object* v_self_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__0(v_self_386_);
lean_dec_ref(v_self_386_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__1(lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_a_390_){
_start:
{
lean_object* v___x_391_; uint8_t v___x_392_; 
lean_inc(v_a_390_);
v___x_391_ = lean_apply_1(v_inst_388_, v_a_390_);
v___x_392_ = lean_unbox(v___x_391_);
if (v___x_392_ == 0)
{
lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_393_ = lp_mathlib_Units_mk0___redArg(v_inst_389_, v_a_390_);
v___x_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
return v___x_394_;
}
else
{
lean_object* v___x_395_; 
lean_dec(v_a_390_);
lean_dec_ref(v_inst_389_);
v___x_395_ = lean_box(0);
return v___x_395_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__2(lean_object* v_toZero_396_, lean_object* v___f_397_, lean_object* v_n_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib_WithZero_recZeroCoe___redArg(v_toZero_396_, v___f_397_, v_n_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__2___boxed(lean_object* v_toZero_400_, lean_object* v___f_401_, lean_object* v_n_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__2(v_toZero_400_, v___f_401_, v_n_402_);
lean_dec(v_toZero_400_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(lean_object* v_inst_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v_toMonoidWithZero_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v_toZero_410_; lean_object* v___x_412_; uint8_t v_isShared_413_; uint8_t v_isSharedCheck_420_; 
v_toMonoidWithZero_407_ = lean_ctor_get(v_inst_405_, 0);
lean_inc_ref(v_toMonoidWithZero_407_);
v___x_408_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_toMonoidWithZero_407_);
v___x_409_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_408_);
v_toZero_410_ = lean_ctor_get(v___x_409_, 1);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_409_);
if (v_isSharedCheck_420_ == 0)
{
lean_object* v_unused_421_; 
v_unused_421_ = lean_ctor_get(v___x_409_, 0);
lean_dec(v_unused_421_);
v___x_412_ = v___x_409_;
v_isShared_413_ = v_isSharedCheck_420_;
goto v_resetjp_411_;
}
else
{
lean_inc(v_toZero_410_);
lean_dec(v___x_409_);
v___x_412_ = lean_box(0);
v_isShared_413_ = v_isSharedCheck_420_;
goto v_resetjp_411_;
}
v_resetjp_411_:
{
lean_object* v___f_414_; lean_object* v___f_415_; lean_object* v___f_416_; lean_object* v___x_418_; 
v___f_414_ = ((lean_object*)(lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___closed__0));
v___f_415_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__1), 3, 2);
lean_closure_set(v___f_415_, 0, v_inst_406_);
lean_closure_set(v___f_415_, 1, v_inst_405_);
v___f_416_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_withZeroUnitsEquiv___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_416_, 0, v_toZero_410_);
lean_closure_set(v___f_416_, 1, v___f_414_);
if (v_isShared_413_ == 0)
{
lean_ctor_set(v___x_412_, 1, v___f_415_);
lean_ctor_set(v___x_412_, 0, v___f_416_);
v___x_418_ = v___x_412_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___f_416_);
lean_ctor_set(v_reuseFailAlloc_419_, 1, v___f_415_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv(lean_object* v_G_422_, lean_object* v_inst_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(v_inst_423_, v_inst_424_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__0(lean_object* v_e_426_, lean_object* v_x_427_){
_start:
{
lean_object* v_toFun_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v_val_431_; 
v_toFun_428_ = lean_ctor_get(v_e_426_, 0);
lean_inc(v_toFun_428_);
lean_dec_ref(v_e_426_);
v___x_429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_429_, 0, v_x_427_);
v___x_430_ = lean_apply_1(v_toFun_428_, v___x_429_);
v_val_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc(v_val_431_);
lean_dec(v___x_430_);
return v_val_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__1(lean_object* v_e_432_, lean_object* v_x_433_){
_start:
{
lean_object* v___x_434_; lean_object* v_toFun_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v_val_438_; 
v___x_434_ = lp_mathlib_Equiv_symm___redArg(v_e_432_);
v_toFun_435_ = lean_ctor_get(v___x_434_, 0);
lean_inc(v_toFun_435_);
lean_dec_ref(v___x_434_);
v___x_436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_436_, 0, v_x_433_);
v___x_437_ = lean_apply_1(v_toFun_435_, v___x_436_);
v_val_438_ = lean_ctor_get(v___x_437_, 0);
lean_inc(v_val_438_);
lean_dec(v___x_437_);
return v_val_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__2(lean_object* v_e_439_){
_start:
{
lean_object* v___f_440_; lean_object* v___f_441_; lean_object* v___x_442_; 
lean_inc_ref(v_e_439_);
v___f_440_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__0), 2, 1);
lean_closure_set(v___f_440_, 0, v_e_439_);
v___f_441_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__1), 2, 1);
lean_closure_set(v___f_441_, 0, v_e_439_);
v___x_442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_442_, 0, v___f_440_);
lean_ctor_set(v___x_442_, 1, v___f_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__3(lean_object* v_e_443_, lean_object* v___y_444_){
_start:
{
lean_object* v_toFun_445_; lean_object* v___x_446_; 
v_toFun_445_ = lean_ctor_get(v_e_443_, 0);
lean_inc(v_toFun_445_);
lean_dec_ref(v_e_443_);
v___x_446_ = lean_apply_1(v_toFun_445_, v___y_444_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__4(lean_object* v___x_447_, lean_object* v___f_448_, lean_object* v___y_449_){
_start:
{
lean_object* v___x_201__overap_450_; lean_object* v___x_451_; 
v___x_201__overap_450_ = lp_mathlib_WithZero_map_x27___redArg(v___x_447_, v___f_448_);
v___x_451_ = lean_apply_1(v___x_201__overap_450_, v___y_449_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__5(lean_object* v___x_452_, lean_object* v___y_453_){
_start:
{
lean_object* v_toFun_454_; lean_object* v___x_455_; 
v_toFun_454_ = lean_ctor_get(v___x_452_, 0);
lean_inc(v_toFun_454_);
lean_dec_ref(v___x_452_);
v___x_455_ = lean_apply_1(v_toFun_454_, v___y_453_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___lam__7(lean_object* v___x_456_, lean_object* v___x_457_, lean_object* v_e_458_){
_start:
{
lean_object* v___f_459_; lean_object* v___f_460_; lean_object* v___x_461_; lean_object* v___f_462_; lean_object* v___f_463_; lean_object* v___x_464_; 
lean_inc_ref(v_e_458_);
v___f_459_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__3), 2, 1);
lean_closure_set(v___f_459_, 0, v_e_458_);
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__4), 3, 2);
lean_closure_set(v___f_460_, 0, v___x_456_);
lean_closure_set(v___f_460_, 1, v___f_459_);
v___x_461_ = lp_mathlib_Equiv_symm___redArg(v_e_458_);
v___f_462_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__5), 2, 1);
lean_closure_set(v___f_462_, 0, v___x_461_);
v___f_463_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__4), 3, 2);
lean_closure_set(v___f_463_, 0, v___x_457_);
lean_closure_set(v___f_463_, 1, v___f_462_);
v___x_464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_464_, 0, v___f_460_);
lean_ctor_set(v___x_464_, 1, v___f_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg(lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v_toMonoid_468_; lean_object* v___x_469_; lean_object* v_toMonoid_470_; lean_object* v___f_471_; lean_object* v___x_472_; lean_object* v___f_473_; lean_object* v___x_474_; 
v_toMonoid_468_ = lean_ctor_get(v_inst_466_, 0);
v___x_469_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_468_);
v_toMonoid_470_ = lean_ctor_get(v_inst_467_, 0);
v___f_471_ = ((lean_object*)(lp_mathlib_MulEquiv_withZero___redArg___closed__0));
v___x_472_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_470_);
v___f_473_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__7), 3, 2);
lean_closure_set(v___f_473_, 0, v___x_472_);
lean_closure_set(v___f_473_, 1, v___x_469_);
v___x_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_474_, 0, v___f_473_);
lean_ctor_set(v___x_474_, 1, v___f_471_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___redArg___boxed(lean_object* v_inst_475_, lean_object* v_inst_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_MulEquiv_withZero___redArg(v_inst_475_, v_inst_476_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero(lean_object* v_00_u03b1_478_, lean_object* v_00_u03b2_479_, lean_object* v_inst_480_, lean_object* v_inst_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib_MulEquiv_withZero___redArg(v_inst_480_, v_inst_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_withZero___boxed(lean_object* v_00_u03b1_483_, lean_object* v_00_u03b2_484_, lean_object* v_inst_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_MulEquiv_withZero(v_00_u03b1_483_, v_00_u03b2_484_, v_inst_485_, v_inst_486_);
lean_dec_ref(v_inst_486_);
lean_dec_ref(v_inst_485_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero___redArg(lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_e_490_){
_start:
{
lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v_toFun_493_; lean_object* v___x_494_; 
v___x_491_ = lp_mathlib_MulEquiv_withZero___redArg(v_inst_488_, v_inst_489_);
v___x_492_ = lp_mathlib_Equiv_symm___redArg(v___x_491_);
v_toFun_493_ = lean_ctor_get(v___x_492_, 0);
lean_inc(v_toFun_493_);
lean_dec_ref(v___x_492_);
v___x_494_ = lean_apply_1(v_toFun_493_, v_e_490_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero___redArg___boxed(lean_object* v_inst_495_, lean_object* v_inst_496_, lean_object* v_e_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_MulEquiv_unzero___redArg(v_inst_495_, v_inst_496_, v_e_497_);
lean_dec_ref(v_inst_496_);
lean_dec_ref(v_inst_495_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero(lean_object* v_00_u03b1_499_, lean_object* v_00_u03b2_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_e_503_){
_start:
{
lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v_toFun_506_; lean_object* v___x_507_; 
v___x_504_ = lp_mathlib_MulEquiv_withZero___redArg(v_inst_501_, v_inst_502_);
v___x_505_ = lp_mathlib_Equiv_symm___redArg(v___x_504_);
v_toFun_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc(v_toFun_506_);
lean_dec_ref(v___x_505_);
v___x_507_ = lean_apply_1(v_toFun_506_, v_e_503_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unzero___boxed(lean_object* v_00_u03b1_508_, lean_object* v_00_u03b2_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_e_512_){
_start:
{
lean_object* v_res_513_; 
v_res_513_ = lp_mathlib_MulEquiv_unzero(v_00_u03b1_508_, v_00_u03b2_509_, v_inst_510_, v_inst_511_, v_e_512_);
lean_dec_ref(v_inst_511_);
lean_dec_ref(v_inst_510_);
return v_res_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommGroupWithZero___redArg(lean_object* v_inst_514_){
_start:
{
lean_object* v_toMonoid_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v_toInv_518_; lean_object* v_toDiv_519_; lean_object* v_toZPow_520_; lean_object* v___x_522_; uint8_t v_isShared_523_; uint8_t v_isSharedCheck_527_; 
v_toMonoid_515_ = lean_ctor_get(v_inst_514_, 0);
lean_inc_ref(v_toMonoid_515_);
v___x_516_ = lp_mathlib_WithZero_instCommMonoidWithZero___redArg(v_toMonoid_515_);
v___x_517_ = lp_mathlib_WithZero_instGroupWithZero___redArg(v_inst_514_);
v_toInv_518_ = lean_ctor_get(v___x_517_, 1);
v_toDiv_519_ = lean_ctor_get(v___x_517_, 2);
v_toZPow_520_ = lean_ctor_get(v___x_517_, 3);
v_isSharedCheck_527_ = !lean_is_exclusive(v___x_517_);
if (v_isSharedCheck_527_ == 0)
{
lean_object* v_unused_528_; 
v_unused_528_ = lean_ctor_get(v___x_517_, 0);
lean_dec(v_unused_528_);
v___x_522_ = v___x_517_;
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
else
{
lean_inc(v_toZPow_520_);
lean_inc(v_toDiv_519_);
lean_inc(v_toInv_518_);
lean_dec(v___x_517_);
v___x_522_ = lean_box(0);
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
v_resetjp_521_:
{
lean_object* v___x_525_; 
if (v_isShared_523_ == 0)
{
lean_ctor_set(v___x_522_, 0, v___x_516_);
v___x_525_ = v___x_522_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v___x_516_);
lean_ctor_set(v_reuseFailAlloc_526_, 1, v_toInv_518_);
lean_ctor_set(v_reuseFailAlloc_526_, 2, v_toDiv_519_);
lean_ctor_set(v_reuseFailAlloc_526_, 3, v_toZPow_520_);
v___x_525_ = v_reuseFailAlloc_526_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
return v___x_525_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCommGroupWithZero(lean_object* v_00_u03b1_529_, lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lp_mathlib_WithZero_instCommGroupWithZero___redArg(v_inst_530_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoidWithOne___redArg___lam__0(lean_object* v_toNatCast_532_, lean_object* v_n_533_){
_start:
{
lean_object* v___x_534_; uint8_t v___x_535_; 
v___x_534_ = lean_unsigned_to_nat(0u);
v___x_535_ = lean_nat_dec_eq(v_n_533_, v___x_534_);
if (v___x_535_ == 0)
{
lean_object* v___x_536_; lean_object* v___x_537_; 
v___x_536_ = lean_apply_1(v_toNatCast_532_, v_n_533_);
v___x_537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_537_, 0, v___x_536_);
return v___x_537_;
}
else
{
lean_object* v___x_538_; 
lean_dec(v_n_533_);
lean_dec(v_toNatCast_532_);
v___x_538_ = lean_box(0);
return v___x_538_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoidWithOne___redArg(lean_object* v_inst_539_){
_start:
{
lean_object* v_toAddMonoid_540_; lean_object* v_toNatCast_541_; lean_object* v_toOne_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_553_; 
v_toAddMonoid_540_ = lean_ctor_get(v_inst_539_, 1);
v_toNatCast_541_ = lean_ctor_get(v_inst_539_, 0);
v_toOne_542_ = lean_ctor_get(v_inst_539_, 2);
v_isSharedCheck_553_ = !lean_is_exclusive(v_inst_539_);
if (v_isSharedCheck_553_ == 0)
{
v___x_544_ = v_inst_539_;
v_isShared_545_ = v_isSharedCheck_553_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_toOne_542_);
lean_inc(v_toAddMonoid_540_);
lean_inc(v_toNatCast_541_);
lean_dec(v_inst_539_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_553_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v_toAdd_546_; lean_object* v___f_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_551_; 
v_toAdd_546_ = lean_ctor_get(v_toAddMonoid_540_, 1);
lean_inc(v_toAdd_546_);
lean_dec_ref(v_toAddMonoid_540_);
v___f_547_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instAddMonoidWithOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_547_, 0, v_toNatCast_541_);
v___x_548_ = lp_mathlib_WithZero_instAddMonoid___redArg(v_toAdd_546_);
v___x_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_549_, 0, v_toOne_542_);
if (v_isShared_545_ == 0)
{
lean_ctor_set(v___x_544_, 2, v___x_549_);
lean_ctor_set(v___x_544_, 1, v___x_548_);
lean_ctor_set(v___x_544_, 0, v___f_547_);
v___x_551_ = v___x_544_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v___f_547_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v___x_548_);
lean_ctor_set(v_reuseFailAlloc_552_, 2, v___x_549_);
v___x_551_ = v_reuseFailAlloc_552_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
return v___x_551_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoidWithOne(lean_object* v_00_u03b1_554_, lean_object* v_inst_555_){
_start:
{
lean_object* v___x_556_; 
v___x_556_ = lp_mathlib_WithZero_instAddMonoidWithOne___redArg(v_inst_555_);
return v___x_556_;
}
}
static lean_object* _init_lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__2(void){
_start:
{
lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_573_ = ((lean_object*)(lp_mathlib_WithZero_term___u1d50_u2070___closed__0));
v___x_574_ = l_String_toRawSubstring_x27(v___x_573_);
return v___x_574_;
}
}
static lean_object* _init_lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__15(void){
_start:
{
lean_object* v___x_599_; lean_object* v___x_600_; 
v___x_599_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__14));
v___x_600_ = l_String_toRawSubstring_x27(v___x_599_);
return v___x_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1(lean_object* v_x_617_, lean_object* v_a_618_, lean_object* v_a_619_){
_start:
{
lean_object* v___x_620_; uint8_t v___x_621_; 
v___x_620_ = ((lean_object*)(lp_mathlib_WithZero_term___u1d50_u2070___closed__2));
lean_inc(v_x_617_);
v___x_621_ = l_Lean_Syntax_isOfKind(v_x_617_, v___x_620_);
if (v___x_621_ == 0)
{
lean_object* v___x_622_; lean_object* v___x_623_; 
lean_dec(v_x_617_);
v___x_622_ = lean_box(1);
v___x_623_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_623_, 0, v___x_622_);
lean_ctor_set(v___x_623_, 1, v_a_619_);
return v___x_623_;
}
else
{
lean_object* v_quotContext_624_; lean_object* v_currMacroScope_625_; lean_object* v_ref_626_; lean_object* v___x_627_; lean_object* v___x_628_; uint8_t v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
v_quotContext_624_ = lean_ctor_get(v_a_618_, 1);
v_currMacroScope_625_ = lean_ctor_get(v_a_618_, 2);
v_ref_626_ = lean_ctor_get(v_a_618_, 5);
v___x_627_ = lean_unsigned_to_nat(0u);
v___x_628_ = l_Lean_Syntax_getArg(v_x_617_, v___x_627_);
lean_dec(v_x_617_);
v___x_629_ = 0;
v___x_630_ = l_Lean_SourceInfo_fromRef(v_ref_626_, v___x_629_);
v___x_631_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__1));
v___x_632_ = lean_obj_once(&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__2, &lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__2_once, _init_lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__2);
v___x_633_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__3));
lean_inc_n(v_currMacroScope_625_, 2);
lean_inc_n(v_quotContext_624_, 2);
v___x_634_ = l_Lean_addMacroScope(v_quotContext_624_, v___x_633_, v_currMacroScope_625_);
v___x_635_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__7));
lean_inc_n(v___x_630_, 5);
v___x_636_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_636_, 0, v___x_630_);
lean_ctor_set(v___x_636_, 1, v___x_632_);
lean_ctor_set(v___x_636_, 2, v___x_634_);
lean_ctor_set(v___x_636_, 3, v___x_635_);
v___x_637_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__8));
v___x_638_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_630_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
v___x_639_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__13));
v___x_640_ = lean_obj_once(&lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__15, &lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__15_once, _init_lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__15);
v___x_641_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__16));
v___x_642_ = l_Lean_addMacroScope(v_quotContext_624_, v___x_641_, v_currMacroScope_625_);
v___x_643_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__20));
v___x_644_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_644_, 0, v___x_630_);
lean_ctor_set(v___x_644_, 1, v___x_640_);
lean_ctor_set(v___x_644_, 2, v___x_642_);
lean_ctor_set(v___x_644_, 3, v___x_643_);
v___x_645_ = ((lean_object*)(lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___closed__22));
v___x_646_ = l_Lean_Syntax_node1(v___x_630_, v___x_645_, v___x_628_);
v___x_647_ = l_Lean_Syntax_node2(v___x_630_, v___x_639_, v___x_644_, v___x_646_);
v___x_648_ = l_Lean_Syntax_node3(v___x_630_, v___x_631_, v___x_636_, v___x_638_, v___x_647_);
v___x_649_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_649_, 0, v___x_648_);
lean_ctor_set(v___x_649_, 1, v_a_619_);
return v___x_649_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1___boxed(lean_object* v_x_650_, lean_object* v_a_651_, lean_object* v_a_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_WithZero___aux__Mathlib__Algebra__GroupWithZero__WithZero______macroRules__WithZero__term___u1d50_u2070__1(v_x_650_, v_a_651_, v_a_652_);
lean_dec_ref(v_a_651_);
return v_res_653_;
}
}
static lean_object* _init_lp_mathlib_WithZero_exp___redArg___closed__0(void){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_exp___redArg(lean_object* v_a_655_){
_start:
{
lean_object* v___x_656_; lean_object* v_toFun_657_; lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_656_ = lean_obj_once(&lp_mathlib_WithZero_exp___redArg___closed__0, &lp_mathlib_WithZero_exp___redArg___closed__0_once, _init_lp_mathlib_WithZero_exp___redArg___closed__0);
v_toFun_657_ = lean_ctor_get(v___x_656_, 0);
lean_inc(v_toFun_657_);
v___x_658_ = lean_apply_1(v_toFun_657_, v_a_655_);
v___x_659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_659_, 0, v___x_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_exp(lean_object* v_M_660_, lean_object* v_a_661_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = lp_mathlib_WithZero_exp___redArg(v_a_661_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn___redArg(lean_object* v_x_663_, lean_object* v_zero_664_, lean_object* v_exp_665_){
_start:
{
lean_object* v___x_666_; 
v___x_666_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_zero_664_, v_exp_665_, v_x_663_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn___redArg___boxed(lean_object* v_x_667_, lean_object* v_zero_668_, lean_object* v_exp_669_){
_start:
{
lean_object* v_res_670_; 
v_res_670_ = lp_mathlib_WithZero_expRecOn___redArg(v_x_667_, v_zero_668_, v_exp_669_);
lean_dec(v_zero_668_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn(lean_object* v_M_671_, lean_object* v_motive_672_, lean_object* v_x_673_, lean_object* v_zero_674_, lean_object* v_exp_675_){
_start:
{
lean_object* v___x_676_; 
v___x_676_ = lp_mathlib_Option_rec___redArg_00___x40_Mathlib_Util_CompileInductive_3128836237____hygCtx___hyg_3_(v_zero_674_, v_exp_675_, v_x_673_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expRecOn___boxed(lean_object* v_M_677_, lean_object* v_motive_678_, lean_object* v_x_679_, lean_object* v_zero_680_, lean_object* v_exp_681_){
_start:
{
lean_object* v_res_682_; 
v_res_682_ = lp_mathlib_WithZero_expRecOn(v_M_677_, v_motive_678_, v_x_679_, v_zero_680_, v_exp_681_);
lean_dec(v_zero_680_);
return v_res_682_;
}
}
static lean_object* _init_lp_mathlib_WithZero_log___redArg___closed__0(void){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_683_;
}
}
static lean_object* _init_lp_mathlib_WithZero_log___redArg___closed__1(void){
_start:
{
lean_object* v___x_684_; lean_object* v___f_685_; 
v___x_684_ = lean_obj_once(&lp_mathlib_WithZero_log___redArg___closed__0, &lp_mathlib_WithZero_log___redArg___closed__0_once, _init_lp_mathlib_WithZero_log___redArg___closed__0);
v___f_685_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_withZero___redArg___lam__5), 2, 1);
lean_closure_set(v___f_685_, 0, v___x_684_);
return v___f_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log___redArg(lean_object* v_inst_686_, lean_object* v_x_687_){
_start:
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v_toZero_690_; lean_object* v___f_691_; lean_object* v___x_692_; 
v___x_688_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_686_);
v___x_689_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_688_);
v_toZero_690_ = lean_ctor_get(v___x_689_, 0);
lean_inc(v_toZero_690_);
lean_dec_ref(v___x_689_);
v___f_691_ = lean_obj_once(&lp_mathlib_WithZero_log___redArg___closed__1, &lp_mathlib_WithZero_log___redArg___closed__1_once, _init_lp_mathlib_WithZero_log___redArg___closed__1);
v___x_692_ = lp_mathlib_WithZero_recZeroCoe___redArg(v_toZero_690_, v___f_691_, v_x_687_);
lean_dec(v_toZero_690_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log___redArg___boxed(lean_object* v_inst_693_, lean_object* v_x_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib_WithZero_log___redArg(v_inst_693_, v_x_694_);
lean_dec_ref(v_inst_693_);
return v_res_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log(lean_object* v_M_696_, lean_object* v_inst_697_, lean_object* v_x_698_){
_start:
{
lean_object* v___x_699_; 
v___x_699_ = lp_mathlib_WithZero_log___redArg(v_inst_697_, v_x_698_);
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_log___boxed(lean_object* v_M_700_, lean_object* v_inst_701_, lean_object* v_x_702_){
_start:
{
lean_object* v_res_703_; 
v_res_703_ = lp_mathlib_WithZero_log(v_M_700_, v_inst_701_, v_x_702_);
lean_dec_ref(v_inst_701_);
return v_res_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expEquiv___redArg(lean_object* v_inst_704_){
_start:
{
lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; 
v___x_705_ = lean_obj_once(&lp_mathlib_WithZero_exp___redArg___closed__0, &lp_mathlib_WithZero_exp___redArg___closed__0_once, _init_lp_mathlib_WithZero_exp___redArg___closed__0);
v___x_706_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_704_);
v___x_707_ = lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(v___x_706_);
v___x_708_ = lp_mathlib_Equiv_symm___redArg(v___x_707_);
v___x_709_ = lp_mathlib_Equiv_trans___redArg(v___x_705_, v___x_708_);
return v___x_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_expEquiv(lean_object* v_G_710_, lean_object* v_inst_711_){
_start:
{
lean_object* v___x_712_; 
v___x_712_ = lp_mathlib_WithZero_expEquiv___redArg(v_inst_711_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logEquiv___redArg(lean_object* v_inst_713_){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_714_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_713_);
v___x_715_ = lp_mathlib_WithZero_unitsWithZeroEquiv___redArg(v___x_714_);
v___x_716_ = lean_obj_once(&lp_mathlib_WithZero_log___redArg___closed__0, &lp_mathlib_WithZero_log___redArg___closed__0_once, _init_lp_mathlib_WithZero_log___redArg___closed__0);
v___x_717_ = lp_mathlib_Equiv_trans___redArg(v___x_715_, v___x_716_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_logEquiv(lean_object* v_G_718_, lean_object* v_inst_719_){
_start:
{
lean_object* v___x_720_; 
v___x_720_ = lp_mathlib_WithZero_logEquiv___redArg(v_inst_719_);
return v___x_720_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
}
#ifdef __cplusplus
}
#endif
