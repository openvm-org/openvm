// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Data.DFinsupp.Sigma public import Mathlib.Data.DFinsupp.Submonoid
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
lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemiring;
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_instMulActionNatOfAddMonoid___redArg(lean_object*);
extern lean_object* lp_mathlib_Nat_instSemiring;
lean_object* lp_mathlib_DFinsupp_zipWith___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_mapRange___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_liftAddHom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_DFinsupp_instInhabited___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_getBinders(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchScoped(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_sigmaCurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0(lean_object*);
lean_object* lp_mathlib_DFinsupp_singleAddHom___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_sigmaUncurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_compHom___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunDirectSumForall___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instCoeFunDirectSumForall___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instCoeFunDirectSumForall___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCoeFunDirectSumForall___closed__0 = (const lean_object*)&lp_mathlib_instCoeFunDirectSumForall___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunDirectSumForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunDirectSumForall___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "DirectSum"};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__0_value;
static const lean_string_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term⨁_,_"};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__1_value;
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(131, 7, 104, 240, 110, 231, 213, 224)}};
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(47, 217, 240, 52, 63, 59, 146, 42)}};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2_value;
static const lean_string_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__3_value;
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__4_value;
static const lean_string_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⨁"};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__5_value;
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__5_value)}};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__6 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__6_value;
static lean_once_cell_t lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__7;
static const lean_string_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__8_value;
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__8_value)}};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__9 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__9_value;
static lean_once_cell_t lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__10;
static const lean_string_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__12_value;
static const lean_ctor_object lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__13 = (const lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__13_value;
static lean_once_cell_t lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__14;
static lean_once_cell_t lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__15;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_term_u2a01___x2c__;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__0_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Notation3"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__1_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termExpand_binders%(_=>_)_,_"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 176, 22, 214, 10, 13, 147, 22)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(120, 7, 237, 26, 3, 243, 131, 214)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "expand_binders%"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__4_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__5_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__6 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__7;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__8 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__8_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__9 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__9_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__10 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__10_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__11 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__11_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__12 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__12_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__13 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14_value;
static lean_once_cell_t lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__15;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(131, 7, 104, 240, 110, 231, 213, 224)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__17 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__17_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16_value)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__18 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__18_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__19 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__19_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__17_value),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__19_value)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__20 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__20_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__21 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__21_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__22 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__22_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__23 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__23_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value_aux_0),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value_aux_1),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value_aux_2),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__25 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__25_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__26 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__26_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__27 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__27_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⨁ "};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__3_value;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extBinders"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5_value_aux_0),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5_value_aux_1),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(142, 202, 111, 171, 129, 134, 17, 161)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5_value;
static lean_once_cell_t lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__6;
static const lean_string_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "extBinderCollection"};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__7 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8_value_aux_0),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8_value_aux_1),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(144, 58, 22, 199, 215, 82, 42, 232)}};
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__8_value)} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__9 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__0_value),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__3_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__4_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__5_value;
static const lean_closure_object lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__4_value),((lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__5_value)} };
static const lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__6 = (const lean_object*)&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_coeFnAddMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_coeFnAddMonoidHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_coeFnAddMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_of___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_of___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_of(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAddMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_setToSet___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_unique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_uniqueOfIsEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_uniqueOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_id___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_id___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_id___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_id___redArg___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_id___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_DirectSum_id___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_id___redArg___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_id___redArg___closed__1 = (const lean_object*)&lp_mathlib_DirectSum_id___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_equivCongrLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_equivCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_DirectSum_Basic_0__DFinsupp_extendWith_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_DirectSum_Basic_0__DFinsupp_extendWith_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurry___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaUncurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaUncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurryEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurryEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_coeAddMonoidHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_coeAddMonoidHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_addEquivProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_addEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_addEquivProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunDirectSumForall___lam__0(lean_object* v_f_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v_f_1_, 0);
lean_inc(v_toFun_3_);
lean_dec_ref(v_f_1_);
v___x_4_ = lean_apply_1(v_toFun_3_, v___y_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunDirectSumForall(lean_object* v_00_u03b9_6_, lean_object* v_00_u03b2_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___f_9_; 
v___f_9_ = ((lean_object*)(lp_mathlib_instCoeFunDirectSumForall___closed__0));
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunDirectSumForall___boxed(lean_object* v_00_u03b9_10_, lean_object* v_00_u03b2_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_instCoeFunDirectSumForall(v_00_u03b9_10_, v_00_u03b2_11_, v_inst_12_);
lean_dec_ref(v_inst_12_);
return v_res_13_;
}
}
static lean_object* _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_25_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_26_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__6));
v___x_27_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__4));
v___x_28_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
lean_ctor_set(v___x_28_, 1, v___x_26_);
lean_ctor_set(v___x_28_, 2, v___x_25_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__10(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_32_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__9));
v___x_33_ = lean_obj_once(&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__7, &lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__7_once, _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__7);
v___x_34_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__4));
v___x_35_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_35_, 0, v___x_34_);
lean_ctor_set(v___x_35_, 1, v___x_33_);
lean_ctor_set(v___x_35_, 2, v___x_32_);
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__14(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_42_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__13));
v___x_43_ = lean_obj_once(&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__10, &lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__10_once, _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__10);
v___x_44_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__4));
v___x_45_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_42_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__15(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_46_ = lean_obj_once(&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__14, &lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__14_once, _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__14);
v___x_47_ = lean_unsigned_to_nat(1022u);
v___x_48_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2));
v___x_49_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_49_, 0, v___x_48_);
lean_ctor_set(v___x_49_, 1, v___x_47_);
lean_ctor_set(v___x_49_, 2, v___x_46_);
return v___x_49_;
}
}
static lean_object* _init_lp_mathlib_DirectSum_term_u2a01___x2c__(void){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_obj_once(&lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__15, &lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__15_once, _init_lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__15);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__7(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_61_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__6));
v___x_62_ = l_String_toRawSubstring_x27(v___x_61_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__15(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__0));
v___x_76_ = l_String_toRawSubstring_x27(v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1(lean_object* v_x_102_, lean_object* v_a_103_, lean_object* v_a_104_){
_start:
{
lean_object* v___x_105_; uint8_t v___x_106_; 
v___x_105_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2));
lean_inc(v_x_102_);
v___x_106_ = l_Lean_Syntax_isOfKind(v_x_102_, v___x_105_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; lean_object* v___x_108_; 
lean_dec(v_x_102_);
v___x_107_ = lean_box(1);
v___x_108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v_a_104_);
return v___x_108_;
}
else
{
lean_object* v_quotContext_109_; lean_object* v_currMacroScope_110_; lean_object* v_ref_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; uint8_t v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v_quotContext_109_ = lean_ctor_get(v_a_103_, 1);
v_currMacroScope_110_ = lean_ctor_get(v_a_103_, 2);
v_ref_111_ = lean_ctor_get(v_a_103_, 5);
v___x_112_ = lean_unsigned_to_nat(1u);
v___x_113_ = l_Lean_Syntax_getArg(v_x_102_, v___x_112_);
v___x_114_ = lean_unsigned_to_nat(3u);
v___x_115_ = l_Lean_Syntax_getArg(v_x_102_, v___x_114_);
lean_dec(v_x_102_);
v___x_116_ = 0;
v___x_117_ = l_Lean_SourceInfo_fromRef(v_ref_111_, v___x_116_);
v___x_118_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__3));
v___x_119_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__4));
lean_inc_n(v___x_117_, 11);
v___x_120_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_117_);
lean_ctor_set(v___x_120_, 1, v___x_119_);
v___x_121_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__5));
v___x_122_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_117_);
lean_ctor_set(v___x_122_, 1, v___x_121_);
v___x_123_ = lean_obj_once(&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__7, &lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__7_once, _init_lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__7);
v___x_124_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_110_, 2);
lean_inc_n(v_quotContext_109_, 2);
v___x_125_ = l_Lean_addMacroScope(v_quotContext_109_, v___x_124_, v_currMacroScope_110_);
v___x_126_ = lean_box(0);
v___x_127_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_127_, 0, v___x_117_);
lean_ctor_set(v___x_127_, 1, v___x_123_);
lean_ctor_set(v___x_127_, 2, v___x_125_);
lean_ctor_set(v___x_127_, 3, v___x_126_);
v___x_128_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__9));
v___x_129_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_117_);
lean_ctor_set(v___x_129_, 1, v___x_128_);
v___x_130_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__14));
v___x_131_ = lean_obj_once(&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__15, &lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__15_once, _init_lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__15);
v___x_132_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16));
v___x_133_ = l_Lean_addMacroScope(v_quotContext_109_, v___x_132_, v_currMacroScope_110_);
v___x_134_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__20));
v___x_135_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_135_, 0, v___x_117_);
lean_ctor_set(v___x_135_, 1, v___x_131_);
lean_ctor_set(v___x_135_, 2, v___x_133_);
lean_ctor_set(v___x_135_, 3, v___x_134_);
v___x_136_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__22));
v___x_137_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__24));
v___x_138_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__25));
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_117_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = l_Lean_Syntax_node1(v___x_117_, v___x_137_, v___x_139_);
lean_inc_ref(v___x_127_);
v___x_141_ = l_Lean_Syntax_node2(v___x_117_, v___x_136_, v___x_140_, v___x_127_);
v___x_142_ = l_Lean_Syntax_node2(v___x_117_, v___x_130_, v___x_135_, v___x_141_);
v___x_143_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__26));
v___x_144_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_117_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v___x_145_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__27));
v___x_146_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_117_);
lean_ctor_set(v___x_146_, 1, v___x_145_);
v___x_147_ = lean_unsigned_to_nat(9u);
v___x_148_ = lean_mk_empty_array_with_capacity(v___x_147_);
v___x_149_ = lean_array_push(v___x_148_, v___x_120_);
v___x_150_ = lean_array_push(v___x_149_, v___x_122_);
v___x_151_ = lean_array_push(v___x_150_, v___x_127_);
v___x_152_ = lean_array_push(v___x_151_, v___x_129_);
v___x_153_ = lean_array_push(v___x_152_, v___x_142_);
v___x_154_ = lean_array_push(v___x_153_, v___x_144_);
v___x_155_ = lean_array_push(v___x_154_, v___x_113_);
v___x_156_ = lean_array_push(v___x_155_, v___x_146_);
v___x_157_ = lean_array_push(v___x_156_, v___x_115_);
v___x_158_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_158_, 0, v___x_117_);
lean_ctor_set(v___x_158_, 1, v___x_118_);
lean_ctor_set(v___x_158_, 2, v___x_157_);
v___x_159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
lean_ctor_set(v___x_159_, 1, v_a_104_);
return v___x_159_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___boxed(lean_object* v_x_160_, lean_object* v_a_161_, lean_object* v_a_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1(v_x_160_, v_a_161_, v_a_162_);
lean_dec_ref(v_a_161_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg(lean_object* v___y_164_){
_start:
{
lean_object* v_subExpr_166_; lean_object* v_expr_167_; lean_object* v___x_168_; 
v_subExpr_166_ = lean_ctor_get(v___y_164_, 3);
v_expr_167_ = lean_ctor_get(v_subExpr_166_, 0);
lean_inc_ref(v_expr_167_);
v___x_168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_168_, 0, v_expr_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg___boxed(lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg(v___y_169_);
lean_dec_ref(v___y_169_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0(lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg(v___y_172_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___boxed(lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0(v___y_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_);
lean_dec(v___y_185_);
lean_dec_ref(v___y_184_);
lean_dec(v___y_183_);
lean_dec_ref(v___y_182_);
lean_dec(v___y_181_);
lean_dec_ref(v___y_180_);
return v_res_187_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__0(lean_object* v_x_188_){
_start:
{
lean_object* v___x_189_; uint8_t v___x_190_; 
v___x_189_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__16));
v___x_190_ = l_Lean_Expr_isConstOf(v_x_188_, v___x_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__0___boxed(lean_object* v_x_191_){
_start:
{
uint8_t v_res_192_; lean_object* v_r_193_; 
v_res_192_ = lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__0(v_x_191_);
lean_dec_ref(v_x_191_);
v_r_193_ = lean_box(v_res_192_);
return v_r_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__1(lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v___x_202_; 
v___x_202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_202_, 0, v___y_194_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__1___boxed(lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__1(v___y_203_, v___y_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_, v___y_209_);
lean_dec(v___y_209_);
lean_dec_ref(v___y_208_);
lean_dec(v___y_207_);
lean_dec_ref(v___y_206_);
lean_dec(v___y_205_);
lean_dec_ref(v___y_204_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2(uint8_t v___x_213_, lean_object* v___x_214_, lean_object* v_a_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v_ref_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v_ref_223_ = lean_ctor_get(v___y_220_, 5);
v___x_224_ = l_Lean_SourceInfo_fromRef(v_ref_223_, v___x_213_);
v___x_225_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__2));
v___x_226_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_224_, 2);
v___x_227_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_224_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = ((lean_object*)(lp_mathlib_DirectSum_term_u2a01___x2c___00__closed__8));
v___x_229_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_224_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v___x_230_ = l_Lean_Syntax_node4(v___x_224_, v___x_225_, v___x_227_, v___x_214_, v___x_229_, v_a_215_);
v___x_231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___boxed(lean_object* v___x_232_, lean_object* v___x_233_, lean_object* v_a_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
uint8_t v___x_7043__boxed_242_; lean_object* v_res_243_; 
v___x_7043__boxed_242_ = lean_unbox(v___x_232_);
v_res_243_ = lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2(v___x_7043__boxed_242_, v___x_233_, v_a_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
return v_res_243_;
}
}
static lean_object* _init_lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__6(void){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = l_Array_mkArray0(lean_box(0));
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3(lean_object* v___f_262_, lean_object* v___f_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
lean_object* v___x_271_; lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_318_; 
v___x_271_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1_spec__0___redArg(v___y_264_);
v_a_272_ = lean_ctor_get(v___x_271_, 0);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_318_ == 0)
{
v___x_274_ = v___x_271_;
v_isShared_275_ = v_isSharedCheck_318_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v___x_271_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_318_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_276_; lean_object* v___y_278_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_276_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__1));
v___x_308_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_309_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_276_, v___x_308_, v___y_264_, v___y_266_);
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v_a_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
v_a_310_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_a_310_);
lean_dec_ref_known(v___x_309_, 1);
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_311_, 0, v___f_262_);
lean_inc_ref(v___f_263_);
v___x_312_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_312_, 0, v___x_311_);
lean_closure_set(v___x_312_, 1, v___f_263_);
v___x_313_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__8));
v___x_314_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__9));
v___x_315_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_315_, 0, v___x_312_);
lean_closure_set(v___x_315_, 1, v___x_314_);
v___x_316_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_316_, 0, v___x_315_);
lean_closure_set(v___x_316_, 1, v___f_263_);
v___x_317_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_276_, v___x_313_, v___x_316_, v_a_310_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
v___y_278_ = v___x_317_;
goto v___jp_277_;
}
else
{
lean_dec_ref(v___f_263_);
lean_dec_ref(v___f_262_);
v___y_278_ = v___x_309_;
goto v___jp_277_;
}
v___jp_277_:
{
if (lean_obj_tag(v___y_278_) == 0)
{
lean_object* v_a_279_; lean_object* v_ref_280_; lean_object* v___x_282_; 
v_a_279_ = lean_ctor_get(v___y_278_, 0);
lean_inc(v_a_279_);
lean_dec_ref_known(v___y_278_, 1);
v_ref_280_ = lean_ctor_get(v___y_268_, 5);
if (v_isShared_275_ == 0)
{
lean_ctor_set_tag(v___x_274_, 1);
v___x_282_ = v___x_274_;
goto v_reusejp_281_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_a_272_);
v___x_282_ = v_reuseFailAlloc_299_;
goto v_reusejp_281_;
}
v_reusejp_281_:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_279_, v___x_276_, v___x_282_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_283_) == 0)
{
lean_object* v_a_284_; uint8_t v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___f_297_; lean_object* v___x_298_; 
v_a_284_ = lean_ctor_get(v___x_283_, 0);
lean_inc(v_a_284_);
lean_dec_ref_known(v___x_283_, 1);
v___x_285_ = 0;
v___x_286_ = l_Lean_SourceInfo_fromRef(v_ref_280_, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__5));
v___x_288_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______macroRules__DirectSum__term_u2a01___x2c____1___closed__22));
v___x_289_ = lean_obj_once(&lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__6, &lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__6_once, _init_lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__6);
v___x_290_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_279_);
lean_dec(v_a_279_);
v___x_291_ = l_Array_append___redArg(v___x_289_, v___x_290_);
lean_dec_ref(v___x_290_);
lean_inc_n(v___x_286_, 2);
v___x_292_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_292_, 0, v___x_286_);
lean_ctor_set(v___x_292_, 1, v___x_288_);
lean_ctor_set(v___x_292_, 2, v___x_291_);
v___x_293_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___closed__8));
v___x_294_ = l_Lean_Syntax_node1(v___x_286_, v___x_293_, v___x_292_);
v___x_295_ = l_Lean_Syntax_node1(v___x_286_, v___x_287_, v___x_294_);
v___x_296_ = lean_box(v___x_285_);
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_297_, 0, v___x_296_);
lean_closure_set(v___f_297_, 1, v___x_295_);
lean_closure_set(v___f_297_, 2, v_a_284_);
v___x_298_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_297_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
return v___x_298_;
}
else
{
lean_dec(v_a_279_);
return v___x_283_;
}
}
}
else
{
lean_object* v_a_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_307_; 
lean_del_object(v___x_274_);
lean_dec(v_a_272_);
v_a_300_ = lean_ctor_get(v___y_278_, 0);
v_isSharedCheck_307_ = !lean_is_exclusive(v___y_278_);
if (v_isSharedCheck_307_ == 0)
{
v___x_302_ = v___y_278_;
v_isShared_303_ = v_isSharedCheck_307_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_a_300_);
lean_dec(v___y_278_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_307_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_305_; 
if (v_isShared_303_ == 0)
{
v___x_305_ = v___x_302_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v_a_300_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3___boxed(lean_object* v___f_319_, lean_object* v___f_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___lam__3(v___f_319_, v___f_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1(lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_349_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__3));
v___x_350_ = ((lean_object*)(lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___closed__6));
v___x_351_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_349_, v___x_350_, v_a_342_, v_a_343_, v_a_344_, v_a_345_, v_a_346_, v_a_347_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1___boxed(lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_DirectSum___aux__Mathlib__Algebra__DirectSum__Basic______delab__app__DirectSum__term_u2a01___x2c____1(v_a_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_, v_a_357_);
lean_dec(v_a_357_);
lean_dec_ref(v_a_356_);
lean_dec(v_a_355_);
lean_dec_ref(v_a_354_);
lean_dec(v_a_353_);
lean_dec_ref(v_a_352_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___redArg___lam__0(lean_object* v_inst_360_, lean_object* v_c_361_, lean_object* v_x_362_, lean_object* v_x_363_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lean_apply_3(v_inst_360_, v_x_362_, v_c_361_, v_x_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___redArg(lean_object* v_inst_365_, lean_object* v_c_366_, lean_object* v_v_367_){
_start:
{
lean_object* v___f_368_; lean_object* v___x_369_; 
v___f_368_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___redArg___lam__0), 4, 2);
lean_closure_set(v___f_368_, 0, v_inst_365_);
lean_closure_set(v___f_368_, 1, v_c_366_);
v___x_369_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_368_, v_v_367_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1(lean_object* v_00_u03b9_370_, lean_object* v_00_u03b2_371_, lean_object* v_R_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_c_376_, lean_object* v_v_377_){
_start:
{
lean_object* v___f_378_; lean_object* v___x_379_; 
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___redArg___lam__0), 4, 2);
lean_closure_set(v___f_378_, 0, v_inst_375_);
lean_closure_set(v___f_378_, 1, v_c_376_);
v___x_379_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_378_, v_v_377_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed(lean_object* v_00_u03b9_380_, lean_object* v_00_u03b2_381_, lean_object* v_R_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_c_386_, lean_object* v_v_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_mathlib_DirectSum_instSMulOfModule___aux__1(v_00_u03b9_380_, v_00_u03b2_381_, v_R_382_, v_inst_383_, v_inst_384_, v_inst_385_, v_c_386_, v_v_387_);
lean_dec_ref(v_inst_384_);
lean_dec_ref(v_inst_383_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule___redArg(lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_inst_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_392_, 0, lean_box(0));
lean_closure_set(v___x_392_, 1, lean_box(0));
lean_closure_set(v___x_392_, 2, lean_box(0));
lean_closure_set(v___x_392_, 3, v_inst_389_);
lean_closure_set(v___x_392_, 4, v_inst_390_);
lean_closure_set(v___x_392_, 5, v_inst_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulOfModule(lean_object* v_00_u03b9_393_, lean_object* v_00_u03b2_394_, lean_object* v_R_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_399_, 0, lean_box(0));
lean_closure_set(v___x_399_, 1, lean_box(0));
lean_closure_set(v___x_399_, 2, lean_box(0));
lean_closure_set(v___x_399_, 3, v_inst_396_);
lean_closure_set(v___x_399_, 4, v_inst_397_);
lean_closure_set(v___x_399_, 5, v_inst_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0(lean_object* v_inst_400_, lean_object* v_x_401_){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v_toZero_405_; 
v___x_402_ = lean_apply_1(v_inst_400_, v_x_401_);
v___x_403_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_402_);
lean_dec_ref(v___x_402_);
v___x_404_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_403_);
v_toZero_405_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_toZero_405_);
lean_dec_ref(v___x_404_);
return v_toZero_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg(lean_object* v_inst_406_){
_start:
{
lean_object* v___f_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v___f_407_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_407_, 0, v_inst_406_);
v___x_408_ = lean_box(0);
v___x_409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_409_, 0, v___f_407_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__1(lean_object* v_00_u03b9_410_, lean_object* v_00_u03b2_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v___f_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_413_, 0, v_inst_412_);
v___x_414_ = lean_box(0);
v___x_415_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_415_, 0, v___f_413_);
lean_ctor_set(v___x_415_, 1, v___x_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__3___redArg___lam__0(lean_object* v_inst_416_, lean_object* v_x_417_, lean_object* v_x1_418_, lean_object* v_x2_419_){
_start:
{
lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v_toAdd_423_; lean_object* v___x_424_; 
v___x_420_ = lean_apply_1(v_inst_416_, v_x_417_);
v___x_421_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_420_);
lean_dec_ref(v___x_420_);
v___x_422_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_421_);
v_toAdd_423_ = lean_ctor_get(v___x_422_, 1);
lean_inc(v_toAdd_423_);
lean_dec_ref(v___x_422_);
v___x_424_ = lean_apply_2(v_toAdd_423_, v_x1_418_, v_x2_419_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__3___redArg(lean_object* v_inst_425_, lean_object* v_x_426_, lean_object* v_y_427_){
_start:
{
lean_object* v___f_428_; lean_object* v___x_429_; 
v___f_428_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__3___redArg___lam__0), 4, 1);
lean_closure_set(v___f_428_, 0, v_inst_425_);
v___x_429_ = lp_mathlib_DFinsupp_zipWith___redArg(v___f_428_, v_x_426_, v_y_427_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___aux__3(lean_object* v_00_u03b9_430_, lean_object* v_00_u03b2_431_, lean_object* v_inst_432_, lean_object* v_x_433_, lean_object* v_y_434_){
_start:
{
lean_object* v___f_435_; lean_object* v___x_436_; 
v___f_435_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__3___redArg___lam__0), 4, 1);
lean_closure_set(v___f_435_, 0, v_inst_432_);
v___x_436_ = lp_mathlib_DFinsupp_zipWith___redArg(v___f_435_, v_x_433_, v_y_434_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___redArg___lam__1(lean_object* v_inst_437_, lean_object* v_i_438_, lean_object* v___y_439_, lean_object* v___y_440_){
_start:
{
lean_object* v___x_441_; lean_object* v___x_79__overap_442_; lean_object* v___x_443_; 
v___x_441_ = lean_apply_1(v_inst_437_, v_i_438_);
v___x_79__overap_442_ = lp_mathlib_instMulActionNatOfAddMonoid___redArg(v___x_441_);
v___x_443_ = lean_apply_2(v___x_79__overap_442_, v___y_439_, v___y_440_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid___redArg(lean_object* v_inst_444_){
_start:
{
lean_object* v___f_445_; lean_object* v___f_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___f_452_; lean_object* v___x_453_; 
lean_inc_ref_n(v_inst_444_, 3);
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_445_, 0, v_inst_444_);
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___redArg___lam__1), 4, 1);
lean_closure_set(v___f_446_, 0, v_inst_444_);
v___x_447_ = lp_mathlib_Nat_instSemiring;
v___x_448_ = lean_box(0);
v___x_449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_449_, 0, v___f_445_);
lean_ctor_set(v___x_449_, 1, v___x_448_);
v___x_450_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__3), 5, 3);
lean_closure_set(v___x_450_, 0, lean_box(0));
lean_closure_set(v___x_450_, 1, lean_box(0));
lean_closure_set(v___x_450_, 2, v_inst_444_);
v___x_451_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_451_, 0, lean_box(0));
lean_closure_set(v___x_451_, 1, lean_box(0));
lean_closure_set(v___x_451_, 2, lean_box(0));
lean_closure_set(v___x_451_, 3, v___x_447_);
lean_closure_set(v___x_451_, 4, v_inst_444_);
lean_closure_set(v___x_451_, 5, v___f_446_);
v___f_452_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_452_, 0, v___x_451_);
v___x_453_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_453_, 0, v___x_449_);
lean_ctor_set(v___x_453_, 1, v___x_450_);
lean_ctor_set(v___x_453_, 2, v___f_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommMonoid(lean_object* v_00_u03b9_454_, lean_object* v_00_u03b2_455_, lean_object* v_inst_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v_inst_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited___aux__1___redArg(lean_object* v_inst_458_){
_start:
{
lean_object* v___f_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v___f_459_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_459_, 0, v_inst_458_);
v___x_460_ = lean_box(0);
v___x_461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_461_, 0, v___f_459_);
lean_ctor_set(v___x_461_, 1, v___x_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited___aux__1(lean_object* v_00_u03b9_462_, lean_object* v_00_u03b2_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___f_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_465_, 0, v_inst_464_);
v___x_466_ = lean_box(0);
v___x_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_467_, 0, v___f_465_);
lean_ctor_set(v___x_467_, 1, v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited___redArg(lean_object* v_inst_468_){
_start:
{
lean_object* v___f_469_; lean_object* v___x_470_; lean_object* v___x_471_; 
v___f_469_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommMonoid___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_469_, 0, v_inst_468_);
v___x_470_ = lean_box(0);
v___x_471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_471_, 0, v___f_469_);
lean_ctor_set(v___x_471_, 1, v___x_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instInhabited(lean_object* v_00_u03b9_472_, lean_object* v_00_u03b2_473_, lean_object* v_inst_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_mathlib_DirectSum_instInhabited___redArg(v_inst_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___aux__1___redArg(lean_object* v_f_476_, lean_object* v_i_477_){
_start:
{
lean_object* v_toFun_478_; lean_object* v___x_479_; 
v_toFun_478_ = lean_ctor_get(v_f_476_, 0);
lean_inc(v_toFun_478_);
lean_dec_ref(v_f_476_);
v___x_479_ = lean_apply_1(v_toFun_478_, v_i_477_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___aux__1(lean_object* v_00_u03b9_480_, lean_object* v_00_u03b2_481_, lean_object* v_inst_482_, lean_object* v_f_483_, lean_object* v_i_484_){
_start:
{
lean_object* v_toFun_485_; lean_object* v___x_486_; 
v_toFun_485_ = lean_ctor_get(v_f_483_, 0);
lean_inc(v_toFun_485_);
lean_dec_ref(v_f_483_);
v___x_486_ = lean_apply_1(v_toFun_485_, v_i_484_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___aux__1___boxed(lean_object* v_00_u03b9_487_, lean_object* v_00_u03b2_488_, lean_object* v_inst_489_, lean_object* v_f_490_, lean_object* v_i_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_DirectSum_instDFunLike___aux__1(v_00_u03b9_487_, v_00_u03b2_488_, v_inst_489_, v_f_490_, v_i_491_);
lean_dec_ref(v_inst_489_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike___redArg(lean_object* v_inst_493_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDFunLike___aux__1___boxed), 5, 3);
lean_closure_set(v___x_494_, 0, lean_box(0));
lean_closure_set(v___x_494_, 1, lean_box(0));
lean_closure_set(v___x_494_, 2, v_inst_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDFunLike(lean_object* v_00_u03b9_495_, lean_object* v_00_u03b2_496_, lean_object* v_inst_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDFunLike___aux__1___boxed), 5, 3);
lean_closure_set(v___x_498_, 0, lean_box(0));
lean_closure_set(v___x_498_, 1, lean_box(0));
lean_closure_set(v___x_498_, 2, v_inst_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0(lean_object* v_inst_499_, lean_object* v_i_500_){
_start:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v_toZero_504_; 
v___x_501_ = lean_apply_1(v_inst_499_, v_i_500_);
v___x_502_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_501_);
lean_dec_ref(v___x_501_);
v___x_503_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_502_);
v_toZero_504_ = lean_ctor_get(v___x_503_, 0);
lean_inc(v_toZero_504_);
lean_dec_ref(v___x_503_);
return v_toZero_504_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg(lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_a_508_, lean_object* v_b_509_){
_start:
{
lean_object* v___f_510_; uint8_t v___x_511_; 
v___f_510_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_510_, 0, v_inst_506_);
v___x_511_ = lp_mathlib_DFinsupp_instDecidableEq___redArg(v_inst_505_, v___f_510_, v_inst_507_, v_a_508_, v_b_509_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___boxed(lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_a_515_, lean_object* v_b_516_){
_start:
{
uint8_t v_res_517_; lean_object* v_r_518_; 
v_res_517_ = lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg(v_inst_512_, v_inst_513_, v_inst_514_, v_a_515_, v_b_516_);
v_r_518_ = lean_box(v_res_517_);
return v_r_518_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq___aux__1(lean_object* v_00_u03b9_519_, lean_object* v_00_u03b2_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_a_524_, lean_object* v_b_525_){
_start:
{
uint8_t v___x_526_; 
v___x_526_ = lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg(v_inst_521_, v_inst_522_, v_inst_523_, v_a_524_, v_b_525_);
return v___x_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___aux__1___boxed(lean_object* v_00_u03b9_527_, lean_object* v_00_u03b2_528_, lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_inst_531_, lean_object* v_a_532_, lean_object* v_b_533_){
_start:
{
uint8_t v_res_534_; lean_object* v_r_535_; 
v_res_534_ = lp_mathlib_DirectSum_instDecidableEq___aux__1(v_00_u03b9_527_, v_00_u03b2_528_, v_inst_529_, v_inst_530_, v_inst_531_, v_a_532_, v_b_533_);
v_r_535_ = lean_box(v_res_534_);
return v_r_535_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq___redArg(lean_object* v_inst_536_, lean_object* v_inst_537_, lean_object* v_inst_538_, lean_object* v_a_539_, lean_object* v_b_540_){
_start:
{
uint8_t v___x_541_; 
v___x_541_ = lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg(v_inst_536_, v_inst_537_, v_inst_538_, v_a_539_, v_b_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___redArg___boxed(lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_a_545_, lean_object* v_b_546_){
_start:
{
uint8_t v_res_547_; lean_object* v_r_548_; 
v_res_547_ = lp_mathlib_DirectSum_instDecidableEq___redArg(v_inst_542_, v_inst_543_, v_inst_544_, v_a_545_, v_b_546_);
v_r_548_ = lean_box(v_res_547_);
return v_r_548_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_instDecidableEq(lean_object* v_00_u03b9_549_, lean_object* v_00_u03b2_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_a_554_, lean_object* v_b_555_){
_start:
{
uint8_t v___x_556_; 
v___x_556_ = lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg(v_inst_551_, v_inst_552_, v_inst_553_, v_a_554_, v_b_555_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instDecidableEq___boxed(lean_object* v_00_u03b9_557_, lean_object* v_00_u03b2_558_, lean_object* v_inst_559_, lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_a_562_, lean_object* v_b_563_){
_start:
{
uint8_t v_res_564_; lean_object* v_r_565_; 
v_res_564_ = lp_mathlib_DirectSum_instDecidableEq(v_00_u03b9_557_, v_00_u03b2_558_, v_inst_559_, v_inst_560_, v_inst_561_, v_a_562_, v_b_563_);
v_r_565_ = lean_box(v_res_564_);
return v_r_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom___lam__0(lean_object* v_x_566_, lean_object* v___y_567_){
_start:
{
lean_object* v_toFun_568_; lean_object* v___x_569_; 
v_toFun_568_ = lean_ctor_get(v_x_566_, 0);
lean_inc(v_toFun_568_);
lean_dec_ref(v_x_566_);
v___x_569_ = lean_apply_1(v_toFun_568_, v___y_567_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom(lean_object* v_00_u03b9_571_, lean_object* v_00_u03b2_572_, lean_object* v_inst_573_){
_start:
{
lean_object* v___f_574_; 
v___f_574_ = ((lean_object*)(lp_mathlib_DirectSum_coeFnAddMonoidHom___closed__0));
return v___f_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnAddMonoidHom___boxed(lean_object* v_00_u03b9_575_, lean_object* v_00_u03b2_576_, lean_object* v_inst_577_){
_start:
{
lean_object* v_res_578_; 
v_res_578_ = lp_mathlib_DirectSum_coeFnAddMonoidHom(v_00_u03b9_575_, v_00_u03b2_576_, v_inst_577_);
lean_dec_ref(v_inst_577_);
return v_res_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__1___redArg___lam__0(lean_object* v_inst_579_, lean_object* v_x_580_, lean_object* v___y_581_){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v_toNeg_584_; lean_object* v___x_585_; 
v___x_582_ = lean_apply_1(v_inst_579_, v_x_580_);
v___x_583_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_582_);
lean_dec_ref(v___x_582_);
v_toNeg_584_ = lean_ctor_get(v___x_583_, 1);
lean_inc(v_toNeg_584_);
lean_dec_ref(v___x_583_);
v___x_585_ = lean_apply_1(v_toNeg_584_, v___y_581_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__1___redArg(lean_object* v_inst_586_, lean_object* v_f_587_){
_start:
{
lean_object* v___f_588_; lean_object* v___x_589_; 
v___f_588_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___aux__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_588_, 0, v_inst_586_);
v___x_589_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_588_, v_f_587_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__1(lean_object* v_00_u03b9_590_, lean_object* v_00_u03b2_591_, lean_object* v_inst_592_, lean_object* v_f_593_){
_start:
{
lean_object* v___f_594_; lean_object* v___x_595_; 
v___f_594_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___aux__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_594_, 0, v_inst_592_);
v___x_595_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_594_, v_f_593_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__3___redArg___lam__0(lean_object* v_inst_596_, lean_object* v_x_597_, lean_object* v___y_598_, lean_object* v___y_599_){
_start:
{
lean_object* v___x_600_; lean_object* v_toSub_601_; lean_object* v___x_602_; 
v___x_600_ = lean_apply_1(v_inst_596_, v_x_597_);
v_toSub_601_ = lean_ctor_get(v___x_600_, 2);
lean_inc(v_toSub_601_);
lean_dec_ref(v___x_600_);
v___x_602_ = lean_apply_2(v_toSub_601_, v___y_598_, v___y_599_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__3___redArg(lean_object* v_inst_603_, lean_object* v_x_604_, lean_object* v_y_605_){
_start:
{
lean_object* v___f_606_; lean_object* v___x_607_; 
v___f_606_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___aux__3___redArg___lam__0), 4, 1);
lean_closure_set(v___f_606_, 0, v_inst_603_);
v___x_607_ = lp_mathlib_DFinsupp_zipWith___redArg(v___f_606_, v_x_604_, v_y_605_);
return v___x_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___aux__3(lean_object* v_00_u03b9_608_, lean_object* v_00_u03b2_609_, lean_object* v_inst_610_, lean_object* v_x_611_, lean_object* v_y_612_){
_start:
{
lean_object* v___f_613_; lean_object* v___x_614_; 
v___f_613_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___aux__3___redArg___lam__0), 4, 1);
lean_closure_set(v___f_613_, 0, v_inst_610_);
v___x_614_ = lp_mathlib_DFinsupp_zipWith___redArg(v___f_613_, v_x_611_, v_y_612_);
return v___x_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg___lam__0(lean_object* v_inst_615_, lean_object* v_i_616_){
_start:
{
lean_object* v___x_617_; lean_object* v_toAddMonoid_618_; 
v___x_617_ = lean_apply_1(v_inst_615_, v_i_616_);
v_toAddMonoid_618_ = lean_ctor_get(v___x_617_, 0);
lean_inc_ref(v_toAddMonoid_618_);
lean_dec_ref(v___x_617_);
return v_toAddMonoid_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg___lam__1(lean_object* v_inst_619_, lean_object* v_i_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
lean_object* v___x_623_; lean_object* v___x_35__overap_624_; lean_object* v___x_625_; 
v___x_623_ = lean_apply_1(v_inst_619_, v_i_620_);
v___x_35__overap_624_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v___x_623_);
v___x_625_ = lean_apply_2(v___x_35__overap_624_, v___y_621_, v___y_622_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg(lean_object* v_inst_626_){
_start:
{
lean_object* v___f_627_; lean_object* v___f_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___f_634_; lean_object* v___x_635_; 
lean_inc_ref_n(v_inst_626_, 3);
v___f_627_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_627_, 0, v_inst_626_);
v___f_628_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___redArg___lam__1), 4, 1);
lean_closure_set(v___f_628_, 0, v_inst_626_);
v___x_629_ = lp_mathlib_Int_instCommSemiring;
lean_inc_ref(v___f_627_);
v___x_630_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v___f_627_);
v___x_631_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___aux__1), 4, 3);
lean_closure_set(v___x_631_, 0, lean_box(0));
lean_closure_set(v___x_631_, 1, lean_box(0));
lean_closure_set(v___x_631_, 2, v_inst_626_);
v___x_632_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAddCommGroup___aux__3), 5, 3);
lean_closure_set(v___x_632_, 0, lean_box(0));
lean_closure_set(v___x_632_, 1, lean_box(0));
lean_closure_set(v___x_632_, 2, v_inst_626_);
v___x_633_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_633_, 0, lean_box(0));
lean_closure_set(v___x_633_, 1, lean_box(0));
lean_closure_set(v___x_633_, 2, lean_box(0));
lean_closure_set(v___x_633_, 3, v___x_629_);
lean_closure_set(v___x_633_, 4, v___f_627_);
lean_closure_set(v___x_633_, 5, v___f_628_);
v___f_634_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_634_, 0, v___x_633_);
v___x_635_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_635_, 0, v___x_630_);
lean_ctor_set(v___x_635_, 1, v___x_631_);
lean_ctor_set(v___x_635_, 2, v___x_632_);
lean_ctor_set(v___x_635_, 3, v___f_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAddCommGroup(lean_object* v_00_u03b9_636_, lean_object* v_00_u03b2_637_, lean_object* v_inst_638_){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lp_mathlib_DirectSum_instAddCommGroup___redArg(v_inst_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mk___redArg(lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_s_642_){
_start:
{
lean_object* v___f_643_; lean_object* v___x_644_; 
v___f_643_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_643_, 0, v_inst_640_);
v___x_644_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mk), 6, 5);
lean_closure_set(v___x_644_, 0, lean_box(0));
lean_closure_set(v___x_644_, 1, lean_box(0));
lean_closure_set(v___x_644_, 2, v___f_643_);
lean_closure_set(v___x_644_, 3, v_inst_641_);
lean_closure_set(v___x_644_, 4, v_s_642_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mk(lean_object* v_00_u03b9_645_, lean_object* v_00_u03b2_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_s_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lp_mathlib_DirectSum_mk___redArg(v_inst_647_, v_inst_648_, v_s_649_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_of___redArg___lam__0(lean_object* v_inst_651_, lean_object* v_i_652_){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_653_ = lean_apply_1(v_inst_651_, v_i_652_);
v___x_654_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_653_);
lean_dec_ref(v___x_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_of___redArg(lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_i_657_){
_start:
{
lean_object* v___f_658_; lean_object* v___x_659_; 
v___f_658_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_of___redArg___lam__0), 2, 1);
lean_closure_set(v___f_658_, 0, v_inst_655_);
v___x_659_ = lp_mathlib_DFinsupp_singleAddHom___redArg(v_inst_656_, v___f_658_, v_i_657_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_of(lean_object* v_00_u03b9_660_, lean_object* v_00_u03b2_661_, lean_object* v_inst_662_, lean_object* v_inst_663_, lean_object* v_i_664_){
_start:
{
lean_object* v___x_665_; 
v___x_665_ = lp_mathlib_DirectSum_of___redArg(v_inst_662_, v_inst_663_, v_i_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAddMonoid___redArg(lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_inst_668_, lean_object* v_00_u03c6_669_){
_start:
{
lean_object* v___f_670_; lean_object* v___x_671_; lean_object* v_toFun_672_; lean_object* v___x_673_; 
v___f_670_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_of___redArg___lam__0), 2, 1);
lean_closure_set(v___f_670_, 0, v_inst_666_);
v___x_671_ = lp_mathlib_DFinsupp_liftAddHom___redArg(v_inst_667_, v___f_670_, v_inst_668_);
v_toFun_672_ = lean_ctor_get(v___x_671_, 0);
lean_inc(v_toFun_672_);
lean_dec_ref(v___x_671_);
v___x_673_ = lean_apply_1(v_toFun_672_, v_00_u03c6_669_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAddMonoid(lean_object* v_00_u03b9_674_, lean_object* v_00_u03b2_675_, lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_00_u03b3_678_, lean_object* v_inst_679_, lean_object* v_00_u03c6_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v_inst_676_, v_inst_677_, v_inst_679_, v_00_u03c6_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___redArg___lam__0(lean_object* v_inst_682_, lean_object* v_i_683_){
_start:
{
lean_object* v___x_684_; lean_object* v___x_685_; 
v___x_684_ = lean_apply_1(v_inst_682_, v_i_683_);
v___x_685_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v___x_684_);
return v___x_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___redArg___lam__1(lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_i_688_, lean_object* v___y_689_, lean_object* v___y_690_){
_start:
{
lean_object* v___x_691_; lean_object* v___x_692_; 
v___x_691_ = lp_mathlib_DirectSum_of___redArg(v_inst_686_, v_inst_687_, v_i_688_);
v___x_692_ = lp_mathlib_MonoidHom_compHom___lam__0(v___x_691_, v___y_689_, v___y_690_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___redArg(lean_object* v_inst_693_, lean_object* v_inst_694_){
_start:
{
lean_object* v___f_695_; lean_object* v___f_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; 
lean_inc_ref_n(v_inst_693_, 2);
v___f_695_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_fromAddMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_695_, 0, v_inst_693_);
lean_inc_ref(v_inst_694_);
v___f_696_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_fromAddMonoid___redArg___lam__1), 5, 2);
lean_closure_set(v___f_696_, 0, v_inst_693_);
lean_closure_set(v___f_696_, 1, v_inst_694_);
v___x_697_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v_inst_693_);
v___x_698_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v___x_697_);
v___x_699_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v___f_695_, v_inst_694_, v___x_698_, v___f_696_);
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid(lean_object* v_00_u03b9_700_, lean_object* v_00_u03b2_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_00_u03b3_704_, lean_object* v_inst_705_){
_start:
{
lean_object* v___x_706_; 
v___x_706_ = lp_mathlib_DirectSum_fromAddMonoid___redArg(v_inst_702_, v_inst_703_);
return v___x_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_fromAddMonoid___boxed(lean_object* v_00_u03b9_707_, lean_object* v_00_u03b2_708_, lean_object* v_inst_709_, lean_object* v_inst_710_, lean_object* v_00_u03b3_711_, lean_object* v_inst_712_){
_start:
{
lean_object* v_res_713_; 
v_res_713_ = lp_mathlib_DirectSum_fromAddMonoid(v_00_u03b9_707_, v_00_u03b2_708_, v_inst_709_, v_inst_710_, v_00_u03b3_711_, v_inst_712_);
lean_dec_ref(v_inst_712_);
return v_res_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg___lam__0(lean_object* v_inst_714_, lean_object* v_i_715_){
_start:
{
lean_object* v___x_716_; 
v___x_716_ = lean_apply_1(v_inst_714_, v_i_715_);
return v___x_716_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_setToSet___redArg___lam__1(lean_object* v_inst_717_, lean_object* v_a_718_, lean_object* v_b_719_){
_start:
{
lean_object* v___x_720_; uint8_t v___x_721_; 
v___x_720_ = lean_apply_2(v_inst_717_, v_a_718_, v_b_719_);
v___x_721_ = lean_unbox(v___x_720_);
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg___lam__1___boxed(lean_object* v_inst_722_, lean_object* v_a_723_, lean_object* v_b_724_){
_start:
{
uint8_t v_res_725_; lean_object* v_r_726_; 
v_res_725_ = lp_mathlib_DirectSum_setToSet___redArg___lam__1(v_inst_722_, v_a_723_, v_b_724_);
v_r_726_ = lean_box(v_res_725_);
return v_r_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg___lam__4(lean_object* v___f_727_, lean_object* v___f_728_, lean_object* v_i_729_, lean_object* v___y_730_){
_start:
{
lean_object* v___x_38__overap_731_; lean_object* v___x_732_; 
v___x_38__overap_731_ = lp_mathlib_DirectSum_of___redArg(v___f_727_, v___f_728_, v_i_729_);
v___x_732_ = lean_apply_1(v___x_38__overap_731_, v___y_730_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet___redArg(lean_object* v_inst_733_, lean_object* v_inst_734_){
_start:
{
lean_object* v___f_735_; lean_object* v___f_736_; lean_object* v___f_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___f_735_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_setToSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_735_, 0, v_inst_733_);
v___f_736_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_setToSet___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_736_, 0, v_inst_734_);
lean_inc_ref(v___f_736_);
lean_inc_ref_n(v___f_735_, 2);
v___f_737_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_setToSet___redArg___lam__4), 4, 2);
lean_closure_set(v___f_737_, 0, v___f_735_);
lean_closure_set(v___f_737_, 1, v___f_736_);
v___x_738_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v___f_735_);
v___x_739_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v___f_735_, v___f_736_, v___x_738_, v___f_737_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_setToSet(lean_object* v_00_u03b9_740_, lean_object* v_00_u03b2_741_, lean_object* v_inst_742_, lean_object* v_inst_743_, lean_object* v_S_744_, lean_object* v_T_745_, lean_object* v_H_746_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lp_mathlib_DirectSum_setToSet___redArg(v_inst_742_, v_inst_743_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_unique___redArg(lean_object* v_inst_748_){
_start:
{
lean_object* v___f_749_; lean_object* v___x_750_; 
v___f_749_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_749_, 0, v_inst_748_);
v___x_750_ = lp_mathlib_DFinsupp_instInhabited___redArg(v___f_749_);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_unique(lean_object* v_00_u03b9_751_, lean_object* v_00_u03b2_752_, lean_object* v_inst_753_, lean_object* v_inst_754_){
_start:
{
lean_object* v___x_755_; 
v___x_755_ = lp_mathlib_DirectSum_unique___redArg(v_inst_753_);
return v___x_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_uniqueOfIsEmpty___redArg(lean_object* v_inst_756_){
_start:
{
lean_object* v___f_757_; lean_object* v___x_758_; 
v___f_757_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_757_, 0, v_inst_756_);
v___x_758_ = lp_mathlib_DFinsupp_instInhabited___redArg(v___f_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_uniqueOfIsEmpty(lean_object* v_00_u03b9_759_, lean_object* v_00_u03b2_760_, lean_object* v_inst_761_, lean_object* v_inst_762_){
_start:
{
lean_object* v___x_763_; 
v___x_763_ = lp_mathlib_DirectSum_uniqueOfIsEmpty___redArg(v_inst_761_);
return v___x_763_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_id___redArg___lam__0(lean_object* v_a_764_, lean_object* v_b_765_){
_start:
{
uint8_t v___x_766_; 
v___x_766_ = 1;
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__0___boxed(lean_object* v_a_767_, lean_object* v_b_768_){
_start:
{
uint8_t v_res_769_; lean_object* v_r_770_; 
v_res_769_ = lp_mathlib_DirectSum_id___redArg___lam__0(v_a_767_, v_b_768_);
lean_dec(v_b_768_);
lean_dec(v_a_767_);
v_r_770_ = lean_box(v_res_769_);
return v_r_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__1(lean_object* v_inst_771_, lean_object* v_i_772_){
_start:
{
lean_inc_ref(v_inst_771_);
return v_inst_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__1___boxed(lean_object* v_inst_773_, lean_object* v_i_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_DirectSum_id___redArg___lam__1(v_inst_773_, v_i_774_);
lean_dec(v_i_774_);
lean_dec_ref(v_inst_773_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__2(lean_object* v_x_776_, lean_object* v___y_777_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lp_mathlib_OneHom_id___lam__0(v___y_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__2___boxed(lean_object* v_x_779_, lean_object* v___y_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_mathlib_DirectSum_id___redArg___lam__2(v_x_779_, v___y_780_);
lean_dec(v___y_780_);
lean_dec(v_x_779_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__3(lean_object* v___f_782_, lean_object* v___f_783_, lean_object* v_inst_784_, lean_object* v___f_785_, lean_object* v___y_786_){
_start:
{
lean_object* v___x_77__overap_787_; lean_object* v___x_788_; 
v___x_77__overap_787_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v___f_782_, v___f_783_, v_inst_784_, v___f_785_);
v___x_788_ = lean_apply_1(v___x_77__overap_787_, v___y_786_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg___lam__4(lean_object* v___f_789_, lean_object* v___f_790_, lean_object* v_inst_791_, lean_object* v___y_792_){
_start:
{
lean_object* v___x_82__overap_793_; lean_object* v___x_794_; 
v___x_82__overap_793_ = lp_mathlib_DirectSum_of___redArg(v___f_789_, v___f_790_, v_inst_791_);
v___x_794_ = lean_apply_1(v___x_82__overap_793_, v___y_792_);
return v___x_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id___redArg(lean_object* v_inst_797_, lean_object* v_inst_798_){
_start:
{
lean_object* v___f_799_; lean_object* v___f_800_; lean_object* v___f_801_; lean_object* v___f_802_; lean_object* v___f_803_; lean_object* v___x_804_; 
v___f_799_ = ((lean_object*)(lp_mathlib_DirectSum_id___redArg___closed__0));
lean_inc_ref(v_inst_797_);
v___f_800_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_id___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_800_, 0, v_inst_797_);
v___f_801_ = ((lean_object*)(lp_mathlib_DirectSum_id___redArg___closed__1));
lean_inc_ref(v___f_800_);
v___f_802_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_id___redArg___lam__3), 5, 4);
lean_closure_set(v___f_802_, 0, v___f_800_);
lean_closure_set(v___f_802_, 1, v___f_799_);
lean_closure_set(v___f_802_, 2, v_inst_797_);
lean_closure_set(v___f_802_, 3, v___f_801_);
v___f_803_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_id___redArg___lam__4), 4, 3);
lean_closure_set(v___f_803_, 0, v___f_800_);
lean_closure_set(v___f_803_, 1, v___f_799_);
lean_closure_set(v___f_803_, 2, v_inst_798_);
v___x_804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_804_, 0, v___f_802_);
lean_ctor_set(v___x_804_, 1, v___f_803_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_id(lean_object* v_M_805_, lean_object* v_00_u03b9_806_, lean_object* v_inst_807_, lean_object* v_inst_808_){
_start:
{
lean_object* v___x_809_; 
v___x_809_ = lp_mathlib_DirectSum_id___redArg(v_inst_807_, v_inst_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_equivCongrLeft___redArg(lean_object* v_inst_810_, lean_object* v_h_811_){
_start:
{
lean_object* v___f_812_; lean_object* v___x_813_; 
v___f_812_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instDecidableEq___aux__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_812_, 0, v_inst_810_);
v___x_813_ = lp_mathlib_DFinsupp_equivCongrLeft___redArg(v___f_812_, v_h_811_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_equivCongrLeft(lean_object* v_00_u03b9_814_, lean_object* v_00_u03b2_815_, lean_object* v_inst_816_, lean_object* v_00_u03ba_817_, lean_object* v_h_818_){
_start:
{
lean_object* v___x_819_; 
v___x_819_ = lp_mathlib_DirectSum_equivCongrLeft___redArg(v_inst_816_, v_h_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_DirectSum_Basic_0__DFinsupp_extendWith_match__1_splitter___redArg(lean_object* v_i_820_, lean_object* v_h__1_821_, lean_object* v_h__2_822_){
_start:
{
if (lean_obj_tag(v_i_820_) == 0)
{
lean_object* v___x_823_; lean_object* v___x_824_; 
lean_dec(v_h__2_822_);
v___x_823_ = lean_box(0);
v___x_824_ = lean_apply_1(v_h__1_821_, v___x_823_);
return v___x_824_;
}
else
{
lean_object* v_val_825_; lean_object* v___x_826_; 
lean_dec(v_h__1_821_);
v_val_825_ = lean_ctor_get(v_i_820_, 0);
lean_inc(v_val_825_);
lean_dec_ref_known(v_i_820_, 1);
v___x_826_ = lean_apply_1(v_h__2_822_, v_val_825_);
return v___x_826_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_DirectSum_Basic_0__DFinsupp_extendWith_match__1_splitter(lean_object* v_00_u03b9_827_, lean_object* v_motive_828_, lean_object* v_i_829_, lean_object* v_h__1_830_, lean_object* v_h__2_831_){
_start:
{
if (lean_obj_tag(v_i_829_) == 0)
{
lean_object* v___x_832_; lean_object* v___x_833_; 
lean_dec(v_h__2_831_);
v___x_832_ = lean_box(0);
v___x_833_ = lean_apply_1(v_h__1_830_, v___x_832_);
return v___x_833_;
}
else
{
lean_object* v_val_834_; lean_object* v___x_835_; 
lean_dec(v_h__1_830_);
v_val_834_ = lean_ctor_get(v_i_829_, 0);
lean_inc(v_val_834_);
lean_dec_ref_known(v_i_829_, 1);
v___x_835_ = lean_apply_1(v_h__2_831_, v_val_834_);
return v___x_835_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurry___redArg___lam__0(lean_object* v_inst_836_, lean_object* v_i_837_, lean_object* v_j_838_){
_start:
{
lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v_toZero_842_; 
v___x_839_ = lean_apply_2(v_inst_836_, v_i_837_, v_j_838_);
v___x_840_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_839_);
lean_dec_ref(v___x_839_);
v___x_841_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_840_);
v_toZero_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc(v_toZero_842_);
lean_dec_ref(v___x_841_);
return v_toZero_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurry___redArg(lean_object* v_inst_843_, lean_object* v_inst_844_){
_start:
{
lean_object* v___f_845_; lean_object* v___x_846_; 
v___f_845_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_sigmaCurry___redArg___lam__0), 3, 1);
lean_closure_set(v___f_845_, 0, v_inst_844_);
v___x_846_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaCurry___boxed), 6, 5);
lean_closure_set(v___x_846_, 0, lean_box(0));
lean_closure_set(v___x_846_, 1, lean_box(0));
lean_closure_set(v___x_846_, 2, lean_box(0));
lean_closure_set(v___x_846_, 3, v_inst_843_);
lean_closure_set(v___x_846_, 4, v___f_845_);
return v___x_846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurry(lean_object* v_00_u03b9_847_, lean_object* v_inst_848_, lean_object* v_00_u03b1_849_, lean_object* v_00_u03b4_850_, lean_object* v_inst_851_){
_start:
{
lean_object* v___x_852_; 
v___x_852_ = lp_mathlib_DirectSum_sigmaCurry___redArg(v_inst_848_, v_inst_851_);
return v___x_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaUncurry___redArg(lean_object* v_inst_853_, lean_object* v_inst_854_){
_start:
{
lean_object* v___f_855_; lean_object* v___x_856_; 
v___f_855_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_sigmaCurry___redArg___lam__0), 3, 1);
lean_closure_set(v___f_855_, 0, v_inst_854_);
v___x_856_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___boxed), 6, 5);
lean_closure_set(v___x_856_, 0, lean_box(0));
lean_closure_set(v___x_856_, 1, lean_box(0));
lean_closure_set(v___x_856_, 2, lean_box(0));
lean_closure_set(v___x_856_, 3, v_inst_853_);
lean_closure_set(v___x_856_, 4, v___f_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaUncurry(lean_object* v_00_u03b9_857_, lean_object* v_inst_858_, lean_object* v_00_u03b1_859_, lean_object* v_00_u03b4_860_, lean_object* v_inst_861_){
_start:
{
lean_object* v___x_862_; 
v___x_862_ = lp_mathlib_DirectSum_sigmaUncurry___redArg(v_inst_858_, v_inst_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurryEquiv___redArg(lean_object* v_inst_863_, lean_object* v_inst_864_){
_start:
{
lean_object* v___f_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v_invFun_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_875_; 
lean_inc_ref(v_inst_864_);
v___f_865_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_sigmaCurry___redArg___lam__0), 3, 1);
lean_closure_set(v___f_865_, 0, v_inst_864_);
lean_inc_ref(v_inst_863_);
v___x_866_ = lp_mathlib_DirectSum_sigmaCurry___redArg(v_inst_863_, v_inst_864_);
v___x_867_ = lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(v_inst_863_, v___f_865_);
v_invFun_868_ = lean_ctor_get(v___x_867_, 1);
v_isSharedCheck_875_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_875_ == 0)
{
lean_object* v_unused_876_; 
v_unused_876_ = lean_ctor_get(v___x_867_, 0);
lean_dec(v_unused_876_);
v___x_870_ = v___x_867_;
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_invFun_868_);
lean_dec(v___x_867_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_873_; 
if (v_isShared_871_ == 0)
{
lean_ctor_set(v___x_870_, 0, v___x_866_);
v___x_873_ = v___x_870_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_866_);
lean_ctor_set(v_reuseFailAlloc_874_, 1, v_invFun_868_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaCurryEquiv(lean_object* v_00_u03b9_877_, lean_object* v_inst_878_, lean_object* v_00_u03b1_879_, lean_object* v_00_u03b4_880_, lean_object* v_inst_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lp_mathlib_DirectSum_sigmaCurryEquiv___redArg(v_inst_878_, v_inst_881_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg___lam__0(lean_object* v_inst_883_, lean_object* v_i_884_, lean_object* v_j_885_){
_start:
{
lean_object* v___x_886_; 
v___x_886_ = lean_apply_1(v_inst_883_, v_j_885_);
return v___x_886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg___lam__0___boxed(lean_object* v_inst_887_, lean_object* v_i_888_, lean_object* v_j_889_){
_start:
{
lean_object* v_res_890_; 
v_res_890_ = lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg___lam__0(v_inst_887_, v_i_888_, v_j_889_);
lean_dec(v_i_888_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg(lean_object* v_inst_891_, lean_object* v_f_892_, lean_object* v_inst_893_){
_start:
{
lean_object* v___f_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; 
lean_inc_ref(v_inst_893_);
v___f_894_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_894_, 0, v_inst_893_);
v___x_895_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_892_);
v___x_896_ = lp_mathlib_Equiv_symm___redArg(v___x_895_);
v___x_897_ = lp_mathlib_DirectSum_equivCongrLeft___redArg(v_inst_893_, v___x_896_);
v___x_898_ = lp_mathlib_DirectSum_sigmaCurryEquiv___redArg(v_inst_891_, v___f_894_);
v___x_899_ = lp_mathlib_Equiv_trans___redArg(v___x_897_, v___x_898_);
return v___x_899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaFiberAddEquiv(lean_object* v_00_u03b9_u2081_900_, lean_object* v_00_u03b9_u2082_901_, lean_object* v_inst_902_, lean_object* v_f_903_, lean_object* v_00_u03b2_904_, lean_object* v_inst_905_){
_start:
{
lean_object* v___x_906_; 
v___x_906_ = lp_mathlib_DirectSum_sigmaFiberAddEquiv___redArg(v_inst_902_, v_f_903_, v_inst_905_);
return v___x_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__0(lean_object* v_inst_907_, lean_object* v_i_908_){
_start:
{
lean_object* v___x_909_; 
v___x_909_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_907_);
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__0___boxed(lean_object* v_inst_910_, lean_object* v_i_911_){
_start:
{
lean_object* v_res_912_; 
v_res_912_ = lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__0(v_inst_910_, v_i_911_);
lean_dec(v_i_911_);
return v_res_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__1(lean_object* v_i_913_, lean_object* v___y_914_){
_start:
{
lean_object* v___x_915_; 
v___x_915_ = lp_mathlib_SubmonoidClass_subtype___lam__0(v___y_914_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__1___boxed(lean_object* v_i_916_, lean_object* v___y_917_){
_start:
{
lean_object* v_res_918_; 
v_res_918_ = lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__1(v_i_916_, v___y_917_);
lean_dec(v___y_917_);
lean_dec(v_i_916_);
return v_res_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg(lean_object* v_inst_920_, lean_object* v_inst_921_){
_start:
{
lean_object* v___f_922_; lean_object* v___f_923_; lean_object* v___x_924_; 
lean_inc_ref(v_inst_921_);
v___f_922_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_coeAddMonoidHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_922_, 0, v_inst_921_);
v___f_923_ = ((lean_object*)(lp_mathlib_DirectSum_coeAddMonoidHom___redArg___closed__0));
v___x_924_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v___f_922_, v_inst_920_, v_inst_921_, v___f_923_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom(lean_object* v_00_u03b9_925_, lean_object* v_M_926_, lean_object* v_S_927_, lean_object* v_inst_928_, lean_object* v_inst_929_, lean_object* v_inst_930_, lean_object* v_inst_931_, lean_object* v_A_932_){
_start:
{
lean_object* v___x_933_; 
v___x_933_ = lp_mathlib_DirectSum_coeAddMonoidHom___redArg(v_inst_928_, v_inst_929_);
return v___x_933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___boxed(lean_object* v_00_u03b9_934_, lean_object* v_M_935_, lean_object* v_S_936_, lean_object* v_inst_937_, lean_object* v_inst_938_, lean_object* v_inst_939_, lean_object* v_inst_940_, lean_object* v_A_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_mathlib_DirectSum_coeAddMonoidHom(v_00_u03b9_934_, v_M_935_, v_S_936_, v_inst_937_, v_inst_938_, v_inst_939_, v_inst_940_, v_A_941_);
lean_dec(v_A_941_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_map___redArg(lean_object* v_inst_943_, lean_object* v_inst_944_, lean_object* v_f_945_){
_start:
{
lean_object* v___f_946_; lean_object* v___f_947_; lean_object* v___x_948_; 
v___f_946_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_of___redArg___lam__0), 2, 1);
lean_closure_set(v___f_946_, 0, v_inst_943_);
v___f_947_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_of___redArg___lam__0), 2, 1);
lean_closure_set(v___f_947_, 0, v_inst_944_);
v___x_948_ = lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg(v___f_946_, v___f_947_, v_f_945_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_map(lean_object* v_00_u03b9_949_, lean_object* v_00_u03b1_950_, lean_object* v_00_u03b2_951_, lean_object* v_inst_952_, lean_object* v_inst_953_, lean_object* v_f_954_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_mathlib_DirectSum_map___redArg(v_inst_952_, v_inst_953_, v_f_954_);
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_addEquivProd___redArg(lean_object* v_inst_956_){
_start:
{
lean_object* v___x_957_; 
v___x_957_ = lp_mathlib_DFinsupp_equivFunOnFintype___redArg(v_inst_956_);
return v___x_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_addEquivProd(lean_object* v_00_u03b9_958_, lean_object* v_inst_959_, lean_object* v_G_960_, lean_object* v_inst_961_){
_start:
{
lean_object* v___x_962_; 
v___x_962_ = lp_mathlib_DFinsupp_equivFunOnFintype___redArg(v_inst_959_);
return v___x_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_addEquivProd___boxed(lean_object* v_00_u03b9_963_, lean_object* v_inst_964_, lean_object* v_G_965_, lean_object* v_inst_966_){
_start:
{
lean_object* v_res_967_; 
v_res_967_ = lp_mathlib_DirectSum_addEquivProd(v_00_u03b9_963_, v_inst_964_, v_G_965_, v_inst_966_);
lean_dec_ref(v_inst_966_);
return v_res_967_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_DirectSum_term_u2a01___x2c__ = _init_lp_mathlib_DirectSum_term_u2a01___x2c__();
lean_mark_persistent(lp_mathlib_DirectSum_term_u2a01___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
