// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Subalgebra.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Operations public import Mathlib.Algebra.Algebra.Subalgebra.Basic
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_AlgHom_codRestrict___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Subalgebra_val___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Subalgebra_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
lean_object* lp_mathlib_Subsemiring_toSemiring___redArg(lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemiring;
lean_object* lp_mathlib_Ring_toIntAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Subalgebra_equivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instSemiring;
lean_object* lp_mathlib_Semiring_toNatAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Algebra_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_gi___closed__0 = (const lean_object*)&lp_mathlib_Algebra_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_gi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_gi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__0 = (const lean_object*)&lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__1 = (const lean_object*)&lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__2 = (const lean_object*)&lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_inclusion___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___closed__0 = (const lean_object*)&lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instInhabitedSubalgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instInhabitedSubalgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_toTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_toTop___closed__0 = (const lean_object*)&lp_mathlib_Algebra_toTop___closed__0_value;
static lean_once_cell_t lp_mathlib_Algebra_toTop___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Algebra_toTop___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subalgebra_topEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subalgebra_val___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subalgebra_topEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Subalgebra_topEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instUnique___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_saturation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_saturation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__0 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__0_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "subalgebra_adjoin"};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__1 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__1_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__1_value),LEAN_SCALAR_PTR_LITERAL(4, 197, 140, 101, 198, 8, 50, 217)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__2 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__2_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__3 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__3_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__4 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__4_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__5 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__5_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__5_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__6 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__6_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__7 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__7_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__8 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__8_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__9 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__9_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__10 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__10_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__11 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__11_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__11_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__12 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__12_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__9_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__10_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__12_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__13 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__13_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__4_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__6_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__13_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__14 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__14_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__15 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__15_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__15_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__16 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__16_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__17 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__17_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__17_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__18 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__18_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__4_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__18_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__9_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__19 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__19_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__16_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__19_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__20 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__20_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__4_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__14_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__20_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__21 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__21_value;
static const lean_string_object lp_mathlib_Algebra_subalgebra__adjoin___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__22 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__22_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__22_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__23 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__23_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__4_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__21_value),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__23_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__24 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__24_value;
static const lean_ctor_object lp_mathlib_Algebra_subalgebra__adjoin___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__24_value)}};
static const lean_object* lp_mathlib_Algebra_subalgebra__adjoin___closed__25 = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__25_value;
LEAN_EXPORT const lean_object* lp_mathlib_Algebra_subalgebra__adjoin = (const lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__25_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__9_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__12_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__13_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Subsemiring"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__14 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(105, 98, 152, 245, 133, 69, 8, 105)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__15_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__16 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__16_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Submodule"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__17 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(67, 59, 114, 255, 83, 15, 173, 8)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__18 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__18_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__19 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__20 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__16_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__20_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__21 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__13_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__21_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__22 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__22_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__23 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__23_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__24 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__24_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__26 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__26_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__0 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1_value;
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Algebra.adjoin"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__2 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3;
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "adjoin"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__4 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_subalgebra__adjoin___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(252, 215, 251, 69, 243, 204, 61, 65)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__6 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__7 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__8 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__16_value),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__8_value)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__9 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__13_value),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__9_value)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__10 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__10_value;
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term{_}"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__11 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(225, 26, 220, 95, 138, 254, 219, 101)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__12 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__12_value;
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__13 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14;
static lean_once_cell_t lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__15;
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__16 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__16_value;
static const lean_string_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__17 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__18;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__19 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__20 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__19_value)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__21 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__22 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__20_value),((lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__22_value)}};
static const lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__23 = (const lean_object*)&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "EmptyCollection"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "emptyCollection"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 209, 69, 209, 212, 29, 83, 196)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__1_value),LEAN_SCALAR_PTR_LITERAL(3, 53, 136, 5, 91, 228, 156, 207)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Singleton"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "singleton"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__3_value),LEAN_SCALAR_PTR_LITERAL(190, 73, 36, 155, 228, 35, 161, 122)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__4_value),LEAN_SCALAR_PTR_LITERAL(185, 48, 115, 60, 21, 14, 217, 215)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Insert"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "insert"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__6_value),LEAN_SCALAR_PTR_LITERAL(126, 209, 156, 174, 188, 62, 109, 85)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__7_value),LEAN_SCALAR_PTR_LITERAL(12, 132, 219, 243, 180, 219, 203, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__8_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_delabAdjoinNotation___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_delabAdjoinNotation___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Algebra_delabAdjoinNotation___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_delabAdjoinNotation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_delabAdjoinNotation___closed__0 = (const lean_object*)&lp_mathlib_Algebra_delabAdjoinNotation___closed__0_value;
static const lean_closure_object lp_mathlib_Algebra_delabAdjoinNotation___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Algebra_delabAdjoinNotation___lam__0___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__9_value)} };
static const lean_object* lp_mathlib_Algebra_delabAdjoinNotation___closed__1 = (const lean_object*)&lp_mathlib_Algebra_delabAdjoinNotation___closed__1_value;
static const lean_closure_object lp_mathlib_Algebra_delabAdjoinNotation___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_mathlib_Algebra_delabAdjoinNotation___closed__1_value)} };
static const lean_object* lp_mathlib_Algebra_delabAdjoinNotation___closed__2 = (const lean_object*)&lp_mathlib_Algebra_delabAdjoinNotation___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommSemiringOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommSemiringOfComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommSemiringOfComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommRingOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommRingOfComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommRingOfComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureEquivAdjoinNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureEquivAdjoinNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureEquivAdjoinInt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureEquivAdjoinInt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebraOrderEmbedding___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebraOrderEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoin(lean_object* v_R_1_, lean_object* v_A_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_s_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoin___boxed(lean_object* v_R_8_, lean_object* v_A_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_s_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Algebra_adjoin(v_R_8_, v_A_9_, v_inst_10_, v_inst_11_, v_inst_12_, v_s_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_gi___lam__0(lean_object* v_s_15_, lean_object* v_hs_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_box(0);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_gi(lean_object* v_R_19_, lean_object* v_A_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = ((lean_object*)(lp_mathlib_Algebra_gi___closed__0));
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_gi___boxed(lean_object* v_R_25_, lean_object* v_A_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Algebra_gi(v_R_25_, v_A_26_, v_inst_27_, v_inst_28_, v_inst_29_);
lean_dec_ref(v_inst_29_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_27_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___lam__0(lean_object* v_s_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_box(0);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___lam__1(lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_box(0);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg(lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___f_43_; lean_object* v___f_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___f_43_ = ((lean_object*)(lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__0));
v___f_44_ = ((lean_object*)(lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__1));
v___x_45_ = lp_mathlib_Subalgebra_instPartialOrder(lean_box(0), lean_box(0), v_inst_40_, v_inst_41_, v_inst_42_);
v___x_46_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_46_, 0, v___x_45_);
lean_ctor_set(v___x_46_, 1, v___f_44_);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
lean_ctor_set(v___x_47_, 1, v___f_44_);
v___x_48_ = ((lean_object*)(lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___closed__2));
v___x_49_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_49_, 0, v___x_47_);
lean_ctor_set(v___x_49_, 1, v___f_43_);
lean_ctor_set(v___x_49_, 2, v___f_43_);
lean_ctor_set(v___x_49_, 3, v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg___boxed(lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg(v_inst_50_, v_inst_51_, v_inst_52_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
lean_dec_ref(v_inst_50_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra(lean_object* v_R_54_, lean_object* v_A_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Algebra_instCompleteLatticeSubalgebra___redArg(v_inst_56_, v_inst_57_, v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCompleteLatticeSubalgebra___boxed(lean_object* v_R_60_, lean_object* v_A_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Algebra_instCompleteLatticeSubalgebra(v_R_60_, v_A_61_, v_inst_62_, v_inst_63_, v_inst_64_);
lean_dec_ref(v_inst_64_);
lean_dec_ref(v_inst_63_);
lean_dec_ref(v_inst_62_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___lam__0(lean_object* v_inst_66_, lean_object* v_c_67_, lean_object* v_x_68_){
_start:
{
lean_object* v___x_69_; lean_object* v_toNonUnitalNonAssocSemiring_70_; lean_object* v___x_71_; lean_object* v_toMul_72_; lean_object* v___x_73_; 
v___x_69_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_66_);
v_toNonUnitalNonAssocSemiring_70_ = lean_ctor_get(v___x_69_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_70_);
lean_dec_ref(v___x_69_);
v___x_71_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_toNonUnitalNonAssocSemiring_70_);
v_toMul_72_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_toMul_72_);
lean_dec_ref(v___x_71_);
v___x_73_ = lean_apply_2(v_toMul_72_, v_c_67_, v_x_68_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg(lean_object* v_inst_75_){
_start:
{
lean_object* v___f_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___f_76_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_76_, 0, v_inst_75_);
v___x_77_ = ((lean_object*)(lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___closed__0));
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___f_76_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin(lean_object* v_R_79_, lean_object* v_inst_80_, lean_object* v_C_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_S_u2081_84_, lean_object* v_S_u2082_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg(v_inst_82_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___boxed(lean_object* v_R_87_, lean_object* v_inst_88_, lean_object* v_C_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_S_u2081_92_, lean_object* v_S_u2082_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Algebra_instSubtypeMemSubalgebraMin(v_R_87_, v_inst_88_, v_C_89_, v_inst_90_, v_inst_91_, v_S_u2081_92_, v_S_u2082_93_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_88_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1___redArg(lean_object* v_inst_95_){
_start:
{
lean_object* v___f_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_96_, 0, v_inst_95_);
v___x_97_ = ((lean_object*)(lp_mathlib_Algebra_instSubtypeMemSubalgebraMin___redArg___closed__0));
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___f_96_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1(lean_object* v_R_99_, lean_object* v_inst_100_, lean_object* v_C_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_S_u2081_104_, lean_object* v_S_u2082_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1___redArg(v_inst_102_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1___boxed(lean_object* v_R_107_, lean_object* v_inst_108_, lean_object* v_C_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_S_u2081_112_, lean_object* v_S_u2082_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Algebra_instSubtypeMemSubalgebraMin__1(v_R_107_, v_inst_108_, v_C_109_, v_inst_110_, v_inst_111_, v_S_u2081_112_, v_S_u2082_113_);
lean_dec_ref(v_inst_111_);
lean_dec_ref(v_inst_108_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instInhabitedSubalgebra(lean_object* v_R_115_, lean_object* v_A_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_box(0);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instInhabitedSubalgebra___boxed(lean_object* v_R_121_, lean_object* v_A_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Algebra_instInhabitedSubalgebra(v_R_121_, v_A_122_, v_inst_123_, v_inst_124_, v_inst_125_);
lean_dec_ref(v_inst_125_);
lean_dec_ref(v_inst_124_);
lean_dec_ref(v_inst_123_);
return v_res_126_;
}
}
static lean_object* _init_lp_mathlib_Algebra_toTop___closed__1(void){
_start:
{
lean_object* v___f_128_; lean_object* v___x_129_; 
v___f_128_ = ((lean_object*)(lp_mathlib_Algebra_toTop___closed__0));
v___x_129_ = lp_mathlib_AlgHom_codRestrict___redArg(v___f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toTop(lean_object* v_R_130_, lean_object* v_A_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lean_obj_once(&lp_mathlib_Algebra_toTop___closed__1, &lp_mathlib_Algebra_toTop___closed__1_once, _init_lp_mathlib_Algebra_toTop___closed__1);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toTop___boxed(lean_object* v_R_136_, lean_object* v_A_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Algebra_toTop(v_R_136_, v_A_137_, v_inst_138_, v_inst_139_, v_inst_140_);
lean_dec_ref(v_inst_140_);
lean_dec_ref(v_inst_139_);
lean_dec_ref(v_inst_138_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv___redArg(lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v___f_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___f_146_ = ((lean_object*)(lp_mathlib_Subalgebra_topEquiv___redArg___closed__0));
v___x_147_ = lp_mathlib_Algebra_toTop(lean_box(0), lean_box(0), v_inst_143_, v_inst_144_, v_inst_145_);
v___x_148_ = lp_mathlib_AlgEquiv_ofAlgHom___redArg(v___f_146_, v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv___redArg___boxed(lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_Subalgebra_topEquiv___redArg(v_inst_149_, v_inst_150_, v_inst_151_);
lean_dec_ref(v_inst_151_);
lean_dec_ref(v_inst_150_);
lean_dec_ref(v_inst_149_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv(lean_object* v_R_153_, lean_object* v_A_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Subalgebra_topEquiv___redArg(v_inst_155_, v_inst_156_, v_inst_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_topEquiv___boxed(lean_object* v_R_159_, lean_object* v_A_160_, lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Subalgebra_topEquiv(v_R_159_, v_A_160_, v_inst_161_, v_inst_162_, v_inst_163_);
lean_dec_ref(v_inst_163_);
lean_dec_ref(v_inst_162_);
lean_dec_ref(v_inst_161_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instUnique(lean_object* v_R_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_box(0);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_instUnique___boxed(lean_object* v_R_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Subalgebra_instUnique(v_R_168_, v_inst_169_);
lean_dec_ref(v_inst_169_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_saturation(lean_object* v_R_171_, lean_object* v_S_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_s_176_, lean_object* v_M_177_, lean_object* v_H_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_box(0);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_saturation___boxed(lean_object* v_R_180_, lean_object* v_S_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_s_185_, lean_object* v_M_186_, lean_object* v_H_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_Subalgebra_saturation(v_R_180_, v_S_181_, v_inst_182_, v_inst_183_, v_inst_184_, v_s_185_, v_M_186_, v_H_187_);
lean_dec_ref(v_inst_184_);
lean_dec_ref(v_inst_183_);
lean_dec_ref(v_inst_182_);
return v_res_188_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11(void){
_start:
{
lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_269_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__10));
v___x_270_ = l_String_toRawSubstring_x27(v___x_269_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0(uint8_t v___x_299_, lean_object* v___x_300_, size_t v_sz_301_, size_t v_i_302_, lean_object* v_bs_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
uint8_t v___x_306_; 
v___x_306_ = lean_usize_dec_lt(v_i_302_, v_sz_301_);
if (v___x_306_ == 0)
{
lean_object* v___x_307_; 
lean_dec(v___x_300_);
v___x_307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_307_, 0, v_bs_303_);
lean_ctor_set(v___x_307_, 1, v___y_305_);
return v___x_307_;
}
else
{
lean_object* v_quotContext_308_; lean_object* v_currMacroScope_309_; lean_object* v_ref_310_; lean_object* v_v_311_; lean_object* v___x_312_; lean_object* v_bs_x27_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; size_t v___x_334_; size_t v___x_335_; lean_object* v___x_336_; 
v_quotContext_308_ = lean_ctor_get(v___y_304_, 1);
v_currMacroScope_309_ = lean_ctor_get(v___y_304_, 2);
v_ref_310_ = lean_ctor_get(v___y_304_, 5);
v_v_311_ = lean_array_uget(v_bs_303_, v_i_302_);
v___x_312_ = lean_unsigned_to_nat(0u);
v_bs_x27_313_ = lean_array_uset(v_bs_303_, v_i_302_, v___x_312_);
v___x_314_ = l_Lean_SourceInfo_fromRef(v_ref_310_, v___x_299_);
v___x_315_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4));
v___x_316_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6));
v___x_317_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__7));
lean_inc_n(v___x_314_, 7);
v___x_318_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_314_);
lean_ctor_set(v___x_318_, 1, v___x_317_);
v___x_319_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__9));
v___x_320_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11);
v___x_321_ = lean_box(0);
lean_inc(v_currMacroScope_309_);
lean_inc(v_quotContext_308_);
v___x_322_ = l_Lean_addMacroScope(v_quotContext_308_, v___x_321_, v_currMacroScope_309_);
v___x_323_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__22));
v___x_324_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_324_, 0, v___x_314_);
lean_ctor_set(v___x_324_, 1, v___x_320_);
lean_ctor_set(v___x_324_, 2, v___x_322_);
lean_ctor_set(v___x_324_, 3, v___x_323_);
v___x_325_ = l_Lean_Syntax_node1(v___x_314_, v___x_319_, v___x_324_);
v___x_326_ = l_Lean_Syntax_node2(v___x_314_, v___x_316_, v___x_318_, v___x_325_);
v___x_327_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__23));
v___x_328_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_314_);
lean_ctor_set(v___x_328_, 1, v___x_327_);
v___x_329_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25));
lean_inc(v___x_300_);
v___x_330_ = l_Lean_Syntax_node1(v___x_314_, v___x_329_, v___x_300_);
v___x_331_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__26));
v___x_332_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_314_);
lean_ctor_set(v___x_332_, 1, v___x_331_);
v___x_333_ = l_Lean_Syntax_node5(v___x_314_, v___x_315_, v___x_326_, v_v_311_, v___x_328_, v___x_330_, v___x_332_);
v___x_334_ = ((size_t)1ULL);
v___x_335_ = lean_usize_add(v_i_302_, v___x_334_);
v___x_336_ = lean_array_uset(v_bs_x27_313_, v_i_302_, v___x_333_);
v_i_302_ = v___x_335_;
v_bs_303_ = v___x_336_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___boxed(lean_object* v___x_338_, lean_object* v___x_339_, lean_object* v_sz_340_, lean_object* v_i_341_, lean_object* v_bs_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
uint8_t v___x_8360__boxed_345_; size_t v_sz_boxed_346_; size_t v_i_boxed_347_; lean_object* v_res_348_; 
v___x_8360__boxed_345_ = lean_unbox(v___x_338_);
v_sz_boxed_346_ = lean_unbox_usize(v_sz_340_);
lean_dec(v_sz_340_);
v_i_boxed_347_ = lean_unbox_usize(v_i_341_);
lean_dec(v_i_341_);
v_res_348_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0(v___x_8360__boxed_345_, v___x_339_, v_sz_boxed_346_, v_i_boxed_347_, v_bs_342_, v___y_343_, v___y_344_);
lean_dec_ref(v___y_343_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__1(size_t v_sz_349_, size_t v_i_350_, lean_object* v_bs_351_){
_start:
{
uint8_t v___x_352_; 
v___x_352_ = lean_usize_dec_lt(v_i_350_, v_sz_349_);
if (v___x_352_ == 0)
{
return v_bs_351_;
}
else
{
lean_object* v_v_353_; lean_object* v___x_354_; lean_object* v_bs_x27_355_; size_t v___x_356_; size_t v___x_357_; lean_object* v___x_358_; 
v_v_353_ = lean_array_uget(v_bs_351_, v_i_350_);
v___x_354_ = lean_unsigned_to_nat(0u);
v_bs_x27_355_ = lean_array_uset(v_bs_351_, v_i_350_, v___x_354_);
v___x_356_ = ((size_t)1ULL);
v___x_357_ = lean_usize_add(v_i_350_, v___x_356_);
v___x_358_ = lean_array_uset(v_bs_x27_355_, v_i_350_, v_v_353_);
v_i_350_ = v___x_357_;
v_bs_351_ = v___x_358_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__1___boxed(lean_object* v_sz_360_, lean_object* v_i_361_, lean_object* v_bs_362_){
_start:
{
size_t v_sz_boxed_363_; size_t v_i_boxed_364_; lean_object* v_res_365_; 
v_sz_boxed_363_ = lean_unbox_usize(v_sz_360_);
lean_dec(v_sz_360_);
v_i_boxed_364_ = lean_unbox_usize(v_i_361_);
lean_dec(v_i_361_);
v_res_365_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__1(v_sz_boxed_363_, v_i_boxed_364_, v_bs_362_);
return v_res_365_;
}
}
static lean_object* _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__2));
v___x_374_ = l_String_toRawSubstring_x27(v___x_373_);
return v___x_374_;
}
}
static lean_object* _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14(void){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = l_Array_mkArray0(lean_box(0));
return v___x_398_;
}
}
static lean_object* _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__15(void){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_399_ = ((lean_object*)(lp_mathlib_Algebra_subalgebra__adjoin___closed__10));
v___x_400_ = l_Lean_mkAtom(v___x_399_);
return v___x_400_;
}
}
static lean_object* _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__18(void){
_start:
{
lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_403_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__17));
v___x_404_ = l_String_toRawSubstring_x27(v___x_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1(lean_object* v_x_418_, lean_object* v_a_419_, lean_object* v_a_420_){
_start:
{
lean_object* v___x_421_; uint8_t v___x_422_; 
v___x_421_ = ((lean_object*)(lp_mathlib_Algebra_subalgebra__adjoin___closed__2));
lean_inc(v_x_418_);
v___x_422_ = l_Lean_Syntax_isOfKind(v_x_418_, v___x_421_);
if (v___x_422_ == 0)
{
lean_object* v___x_423_; lean_object* v___x_424_; 
lean_dec(v_x_418_);
v___x_423_ = lean_box(1);
v___x_424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
lean_ctor_set(v___x_424_, 1, v_a_420_);
return v___x_424_;
}
else
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; uint8_t v___x_431_; 
v___x_425_ = lean_unsigned_to_nat(0u);
v___x_426_ = l_Lean_Syntax_getArg(v_x_418_, v___x_425_);
v___x_427_ = lean_unsigned_to_nat(2u);
v___x_428_ = l_Lean_Syntax_getArg(v_x_418_, v___x_427_);
v___x_429_ = lean_unsigned_to_nat(3u);
v___x_430_ = l_Lean_Syntax_getArg(v_x_418_, v___x_429_);
lean_dec(v_x_418_);
lean_inc(v___x_430_);
v___x_431_ = l_Lean_Syntax_matchesNull(v___x_430_, v___x_425_);
if (v___x_431_ == 0)
{
uint8_t v___x_432_; 
lean_inc(v___x_430_);
v___x_432_ = l_Lean_Syntax_matchesNull(v___x_430_, v___x_427_);
if (v___x_432_ == 0)
{
lean_object* v___x_433_; lean_object* v___x_434_; 
lean_dec(v___x_430_);
lean_dec(v___x_428_);
lean_dec(v___x_426_);
v___x_433_ = lean_box(1);
v___x_434_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_434_, 0, v___x_433_);
lean_ctor_set(v___x_434_, 1, v_a_420_);
return v___x_434_;
}
else
{
lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v_xs_437_; lean_object* v___x_438_; size_t v_sz_439_; size_t v___x_440_; lean_object* v___x_441_; 
v___x_435_ = lean_unsigned_to_nat(1u);
v___x_436_ = l_Lean_Syntax_getArg(v___x_430_, v___x_435_);
lean_dec(v___x_430_);
v_xs_437_ = l_Lean_Syntax_getArgs(v___x_428_);
lean_dec(v___x_428_);
v___x_438_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_xs_437_);
lean_dec_ref(v_xs_437_);
v_sz_439_ = lean_array_size(v___x_438_);
v___x_440_ = ((size_t)0ULL);
lean_inc(v___x_436_);
v___x_441_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0(v___x_431_, v___x_436_, v_sz_439_, v___x_440_, v___x_438_, v_a_419_, v_a_420_);
if (lean_obj_tag(v___x_441_) == 0)
{
lean_object* v_a_442_; lean_object* v_a_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_501_; 
v_a_442_ = lean_ctor_get(v___x_441_, 0);
v_a_443_ = lean_ctor_get(v___x_441_, 1);
v_isSharedCheck_501_ = !lean_is_exclusive(v___x_441_);
if (v_isSharedCheck_501_ == 0)
{
v___x_445_ = v___x_441_;
v_isShared_446_ = v_isSharedCheck_501_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_a_443_);
lean_inc(v_a_442_);
lean_dec(v___x_441_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_501_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v_quotContext_447_; lean_object* v_currMacroScope_448_; lean_object* v_ref_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; size_t v_sz_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_499_; 
v_quotContext_447_ = lean_ctor_get(v_a_419_, 1);
v_currMacroScope_448_ = lean_ctor_get(v_a_419_, 2);
v_ref_449_ = lean_ctor_get(v_a_419_, 5);
v___x_450_ = l_Lean_SourceInfo_fromRef(v_ref_449_, v___x_431_);
v___x_451_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1));
v___x_452_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3);
v___x_453_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5));
lean_inc_n(v_currMacroScope_448_, 3);
lean_inc_n(v_quotContext_447_, 3);
v___x_454_ = l_Lean_addMacroScope(v_quotContext_447_, v___x_453_, v_currMacroScope_448_);
v___x_455_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__7));
lean_inc_n(v___x_450_, 17);
v___x_456_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_456_, 0, v___x_450_);
lean_ctor_set(v___x_456_, 1, v___x_452_);
lean_ctor_set(v___x_456_, 2, v___x_454_);
lean_ctor_set(v___x_456_, 3, v___x_455_);
v___x_457_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25));
v___x_458_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__4));
v___x_459_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__6));
v___x_460_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__7));
v___x_461_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_461_, 0, v___x_450_);
lean_ctor_set(v___x_461_, 1, v___x_460_);
v___x_462_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__9));
v___x_463_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__11);
v___x_464_ = lean_box(0);
v___x_465_ = l_Lean_addMacroScope(v_quotContext_447_, v___x_464_, v_currMacroScope_448_);
v___x_466_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__10));
v___x_467_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_467_, 0, v___x_450_);
lean_ctor_set(v___x_467_, 1, v___x_463_);
lean_ctor_set(v___x_467_, 2, v___x_465_);
lean_ctor_set(v___x_467_, 3, v___x_466_);
v___x_468_ = l_Lean_Syntax_node1(v___x_450_, v___x_462_, v___x_467_);
v___x_469_ = l_Lean_Syntax_node2(v___x_450_, v___x_459_, v___x_461_, v___x_468_);
v___x_470_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__12));
v___x_471_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__13));
v___x_472_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_472_, 0, v___x_450_);
lean_ctor_set(v___x_472_, 1, v___x_471_);
v___x_473_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14);
v_sz_474_ = lean_array_size(v_a_442_);
v___x_475_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__1(v_sz_474_, v___x_440_, v_a_442_);
v___x_476_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__15, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__15_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__15);
v___x_477_ = l_Lean_mkSepArray(v___x_475_, v___x_476_);
lean_dec_ref(v___x_475_);
v___x_478_ = l_Array_append___redArg(v___x_473_, v___x_477_);
lean_dec_ref(v___x_477_);
v___x_479_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_479_, 0, v___x_450_);
lean_ctor_set(v___x_479_, 1, v___x_457_);
lean_ctor_set(v___x_479_, 2, v___x_478_);
v___x_480_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__16));
v___x_481_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_481_, 0, v___x_450_);
lean_ctor_set(v___x_481_, 1, v___x_480_);
v___x_482_ = l_Lean_Syntax_node3(v___x_450_, v___x_470_, v___x_472_, v___x_479_, v___x_481_);
v___x_483_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__23));
v___x_484_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_484_, 0, v___x_450_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
v___x_485_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__18, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__18_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__18);
v___x_486_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__19));
v___x_487_ = l_Lean_addMacroScope(v_quotContext_447_, v___x_486_, v_currMacroScope_448_);
v___x_488_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__23));
v___x_489_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_489_, 0, v___x_450_);
lean_ctor_set(v___x_489_, 1, v___x_485_);
lean_ctor_set(v___x_489_, 2, v___x_487_);
lean_ctor_set(v___x_489_, 3, v___x_488_);
v___x_490_ = l_Lean_Syntax_node1(v___x_450_, v___x_457_, v___x_436_);
v___x_491_ = l_Lean_Syntax_node2(v___x_450_, v___x_451_, v___x_489_, v___x_490_);
v___x_492_ = l_Lean_Syntax_node1(v___x_450_, v___x_457_, v___x_491_);
v___x_493_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__26));
v___x_494_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_450_);
lean_ctor_set(v___x_494_, 1, v___x_493_);
v___x_495_ = l_Lean_Syntax_node5(v___x_450_, v___x_458_, v___x_469_, v___x_482_, v___x_484_, v___x_492_, v___x_494_);
v___x_496_ = l_Lean_Syntax_node2(v___x_450_, v___x_457_, v___x_426_, v___x_495_);
v___x_497_ = l_Lean_Syntax_node2(v___x_450_, v___x_451_, v___x_456_, v___x_496_);
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 0, v___x_497_);
v___x_499_ = v___x_445_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_497_);
lean_ctor_set(v_reuseFailAlloc_500_, 1, v_a_443_);
v___x_499_ = v_reuseFailAlloc_500_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
return v___x_499_;
}
}
}
else
{
lean_object* v_a_502_; lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_510_; 
lean_dec(v___x_436_);
lean_dec(v___x_426_);
v_a_502_ = lean_ctor_get(v___x_441_, 0);
v_a_503_ = lean_ctor_get(v___x_441_, 1);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_441_);
if (v_isSharedCheck_510_ == 0)
{
v___x_505_ = v___x_441_;
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_inc(v_a_502_);
lean_dec(v___x_441_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_508_; 
if (v_isShared_506_ == 0)
{
v___x_508_ = v___x_505_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v_a_502_);
lean_ctor_set(v_reuseFailAlloc_509_, 1, v_a_503_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
}
else
{
lean_object* v_quotContext_511_; lean_object* v_currMacroScope_512_; lean_object* v_ref_513_; lean_object* v___x_514_; uint8_t v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; 
lean_dec(v___x_430_);
v_quotContext_511_ = lean_ctor_get(v_a_419_, 1);
v_currMacroScope_512_ = lean_ctor_get(v_a_419_, 2);
v_ref_513_ = lean_ctor_get(v_a_419_, 5);
v___x_514_ = l_Lean_Syntax_getArgs(v___x_428_);
lean_dec(v___x_428_);
v___x_515_ = 0;
v___x_516_ = l_Lean_SourceInfo_fromRef(v_ref_513_, v___x_515_);
v___x_517_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__1));
v___x_518_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__3);
v___x_519_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__5));
lean_inc(v_currMacroScope_512_);
lean_inc(v_quotContext_511_);
v___x_520_ = l_Lean_addMacroScope(v_quotContext_511_, v___x_519_, v_currMacroScope_512_);
v___x_521_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__7));
lean_inc_n(v___x_516_, 6);
v___x_522_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_522_, 0, v___x_516_);
lean_ctor_set(v___x_522_, 1, v___x_518_);
lean_ctor_set(v___x_522_, 2, v___x_520_);
lean_ctor_set(v___x_522_, 3, v___x_521_);
v___x_523_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25));
v___x_524_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__12));
v___x_525_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__13));
v___x_526_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_526_, 0, v___x_516_);
lean_ctor_set(v___x_526_, 1, v___x_525_);
v___x_527_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14);
v___x_528_ = l_Array_append___redArg(v___x_527_, v___x_514_);
lean_dec_ref(v___x_514_);
v___x_529_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_529_, 0, v___x_516_);
lean_ctor_set(v___x_529_, 1, v___x_523_);
lean_ctor_set(v___x_529_, 2, v___x_528_);
v___x_530_ = ((lean_object*)(lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__16));
v___x_531_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_516_);
lean_ctor_set(v___x_531_, 1, v___x_530_);
v___x_532_ = l_Lean_Syntax_node3(v___x_516_, v___x_524_, v___x_526_, v___x_529_, v___x_531_);
v___x_533_ = l_Lean_Syntax_node2(v___x_516_, v___x_523_, v___x_426_, v___x_532_);
v___x_534_ = l_Lean_Syntax_node2(v___x_516_, v___x_517_, v___x_522_, v___x_533_);
v___x_535_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_535_, 0, v___x_534_);
lean_ctor_set(v___x_535_, 1, v_a_420_);
return v___x_535_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___boxed(lean_object* v_x_536_, lean_object* v_a_537_, lean_object* v_a_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1(v_x_536_, v_a_537_, v_a_538_);
lean_dec_ref(v_a_537_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg(lean_object* v___y_540_){
_start:
{
lean_object* v_subExpr_542_; lean_object* v_expr_543_; lean_object* v___x_544_; 
v_subExpr_542_ = lean_ctor_get(v___y_540_, 3);
v_expr_543_ = lean_ctor_get(v_subExpr_542_, 0);
lean_inc_ref(v_expr_543_);
v___x_544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_544_, 0, v_expr_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg___boxed(lean_object* v___y_545_, lean_object* v___y_546_){
_start:
{
lean_object* v_res_547_; 
v_res_547_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg(v___y_545_);
lean_dec_ref(v___y_545_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0(lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg(v___y_548_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___boxed(lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0(v___y_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_);
lean_dec(v___y_561_);
lean_dec_ref(v___y_560_);
lean_dec(v___y_559_);
lean_dec_ref(v___y_558_);
lean_dec(v___y_557_);
lean_dec_ref(v___y_556_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg(lean_object* v___y_564_){
_start:
{
lean_object* v_subExpr_566_; lean_object* v_pos_567_; lean_object* v___x_568_; 
v_subExpr_566_ = lean_ctor_get(v___y_564_, 3);
v_pos_567_ = lean_ctor_get(v_subExpr_566_, 1);
lean_inc(v_pos_567_);
v___x_568_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_568_, 0, v_pos_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg___boxed(lean_object* v___y_569_, lean_object* v___y_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg(v___y_569_);
lean_dec_ref(v___y_569_);
return v_res_571_;
}
}
static lean_object* _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_572_; lean_object* v_dummy_573_; 
v___x_572_ = lean_box(0);
v_dummy_573_ = l_Lean_Expr_sort___override(v___x_572_);
return v_dummy_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(lean_object* v_argIdx_574_, lean_object* v_x_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
lean_object* v___x_583_; lean_object* v_a_584_; lean_object* v___x_585_; lean_object* v_a_586_; lean_object* v_optionsPerPos_587_; lean_object* v_currNamespace_588_; lean_object* v_openDecls_589_; uint8_t v_inPattern_590_; lean_object* v_depth_591_; lean_object* v_lctxInitIndices_592_; lean_object* v_nargs_593_; lean_object* v___x_594_; lean_object* v_dummy_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v_args_599_; lean_object* v___x_600_; lean_object* v_newPos_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_583_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg(v___y_576_);
v_a_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_584_);
lean_dec_ref(v___x_583_);
v___x_585_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg(v___y_576_);
v_a_586_ = lean_ctor_get(v___x_585_, 0);
lean_inc(v_a_586_);
lean_dec_ref(v___x_585_);
v_optionsPerPos_587_ = lean_ctor_get(v___y_576_, 0);
v_currNamespace_588_ = lean_ctor_get(v___y_576_, 1);
v_openDecls_589_ = lean_ctor_get(v___y_576_, 2);
v_inPattern_590_ = lean_ctor_get_uint8(v___y_576_, sizeof(void*)*6);
v_depth_591_ = lean_ctor_get(v___y_576_, 4);
v_lctxInitIndices_592_ = lean_ctor_get(v___y_576_, 5);
v_nargs_593_ = l_Lean_Expr_getAppNumArgs(v_a_584_);
v___x_594_ = l_Lean_instInhabitedExpr;
v_dummy_595_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___closed__0);
lean_inc(v_nargs_593_);
v___x_596_ = lean_mk_array(v_nargs_593_, v_dummy_595_);
v___x_597_ = lean_unsigned_to_nat(1u);
v___x_598_ = lean_nat_sub(v_nargs_593_, v___x_597_);
lean_dec(v_nargs_593_);
v_args_599_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_584_, v___x_596_, v___x_598_);
v___x_600_ = lean_array_get_size(v_args_599_);
v_newPos_601_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_600_, v_argIdx_574_, v_a_586_);
lean_dec(v_a_586_);
v___x_602_ = lean_array_get(v___x_594_, v_args_599_, v_argIdx_574_);
lean_dec_ref(v_args_599_);
v___x_603_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_602_);
lean_ctor_set(v___x_603_, 1, v_newPos_601_);
lean_inc(v_lctxInitIndices_592_);
lean_inc(v_depth_591_);
lean_inc(v_openDecls_589_);
lean_inc(v_currNamespace_588_);
lean_inc(v_optionsPerPos_587_);
v___x_604_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_604_, 0, v_optionsPerPos_587_);
lean_ctor_set(v___x_604_, 1, v_currNamespace_588_);
lean_ctor_set(v___x_604_, 2, v_openDecls_589_);
lean_ctor_set(v___x_604_, 3, v___x_603_);
lean_ctor_set(v___x_604_, 4, v_depth_591_);
lean_ctor_set(v___x_604_, 5, v_lctxInitIndices_592_);
lean_ctor_set_uint8(v___x_604_, sizeof(void*)*6, v_inPattern_590_);
lean_inc(v___y_581_);
lean_inc_ref(v___y_580_);
lean_inc(v___y_579_);
lean_inc_ref(v___y_578_);
lean_inc(v___y_577_);
v___x_605_ = lean_apply_7(v_x_575_, v___x_604_, v___y_577_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, lean_box(0));
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg___boxed(lean_object* v_argIdx_606_, lean_object* v_x_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v_argIdx_606_, v_x_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_, v___y_612_, v___y_613_);
lean_dec(v___y_613_);
lean_dec_ref(v___y_612_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
lean_dec(v___y_609_);
lean_dec_ref(v___y_608_);
lean_dec(v_argIdx_606_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___boxed(lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray(v_a_632_, v_a_633_, v_a_634_, v_a_635_, v_a_636_, v_a_637_);
lean_dec(v_a_637_);
lean_dec_ref(v_a_636_);
lean_dec(v_a_635_);
lean_dec_ref(v_a_634_);
lean_dec(v_a_633_);
lean_dec_ref(v_a_632_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray(lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_, lean_object* v_a_643_, lean_object* v_a_644_, lean_object* v_a_645_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__0___redArg(v_a_640_);
if (lean_obj_tag(v___x_647_) == 0)
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_710_; 
v_a_648_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_710_ == 0)
{
v___x_650_ = v___x_647_;
v_isShared_651_ = v_isSharedCheck_710_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_647_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_710_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_652_; lean_object* v___x_653_; uint8_t v___x_654_; 
v___x_652_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__2));
v___x_653_ = lean_unsigned_to_nat(2u);
v___x_654_ = l_Lean_Expr_isAppOfArity(v_a_648_, v___x_652_, v___x_653_);
if (v___x_654_ == 0)
{
lean_object* v___x_655_; lean_object* v___x_656_; uint8_t v___x_657_; 
lean_del_object(v___x_650_);
v___x_655_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__5));
v___x_656_ = lean_unsigned_to_nat(4u);
v___x_657_ = l_Lean_Expr_isAppOfArity(v_a_648_, v___x_655_, v___x_656_);
if (v___x_657_ == 0)
{
lean_object* v___x_658_; lean_object* v___x_659_; uint8_t v___x_660_; 
v___x_658_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__8));
v___x_659_ = lean_unsigned_to_nat(5u);
v___x_660_ = l_Lean_Expr_isAppOfArity(v_a_648_, v___x_658_, v___x_659_);
lean_dec(v_a_648_);
if (v___x_660_ == 0)
{
lean_object* v___x_661_; 
v___x_661_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_661_;
}
else
{
lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_662_ = lean_unsigned_to_nat(3u);
v___x_663_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__9));
v___x_664_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v___x_662_, v___x_663_, v_a_640_, v_a_641_, v_a_642_, v_a_643_, v_a_644_, v_a_645_);
if (lean_obj_tag(v___x_664_) == 0)
{
lean_object* v_a_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
v_a_665_ = lean_ctor_get(v___x_664_, 0);
lean_inc(v_a_665_);
lean_dec_ref_known(v___x_664_, 1);
v___x_666_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___boxed), 7, 0);
v___x_667_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v___x_656_, v___x_666_, v_a_640_, v_a_641_, v_a_642_, v_a_643_, v_a_644_, v_a_645_);
if (lean_obj_tag(v___x_667_) == 0)
{
lean_object* v_a_668_; lean_object* v___x_670_; uint8_t v_isShared_671_; uint8_t v_isSharedCheck_676_; 
v_a_668_ = lean_ctor_get(v___x_667_, 0);
v_isSharedCheck_676_ = !lean_is_exclusive(v___x_667_);
if (v_isSharedCheck_676_ == 0)
{
v___x_670_ = v___x_667_;
v_isShared_671_ = v_isSharedCheck_676_;
goto v_resetjp_669_;
}
else
{
lean_inc(v_a_668_);
lean_dec(v___x_667_);
v___x_670_ = lean_box(0);
v_isShared_671_ = v_isSharedCheck_676_;
goto v_resetjp_669_;
}
v_resetjp_669_:
{
lean_object* v___x_672_; lean_object* v___x_674_; 
v___x_672_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_672_, 0, v_a_665_);
lean_ctor_set(v___x_672_, 1, v_a_668_);
if (v_isShared_671_ == 0)
{
lean_ctor_set(v___x_670_, 0, v___x_672_);
v___x_674_ = v___x_670_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_675_; 
v_reuseFailAlloc_675_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_675_, 0, v___x_672_);
v___x_674_ = v_reuseFailAlloc_675_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
return v___x_674_;
}
}
}
else
{
lean_dec(v_a_665_);
return v___x_667_;
}
}
else
{
lean_object* v_a_677_; lean_object* v___x_679_; uint8_t v_isShared_680_; uint8_t v_isSharedCheck_684_; 
v_a_677_ = lean_ctor_get(v___x_664_, 0);
v_isSharedCheck_684_ = !lean_is_exclusive(v___x_664_);
if (v_isSharedCheck_684_ == 0)
{
v___x_679_ = v___x_664_;
v_isShared_680_ = v_isSharedCheck_684_;
goto v_resetjp_678_;
}
else
{
lean_inc(v_a_677_);
lean_dec(v___x_664_);
v___x_679_ = lean_box(0);
v_isShared_680_ = v_isSharedCheck_684_;
goto v_resetjp_678_;
}
v_resetjp_678_:
{
lean_object* v___x_682_; 
if (v_isShared_680_ == 0)
{
v___x_682_ = v___x_679_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_683_; 
v_reuseFailAlloc_683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_683_, 0, v_a_677_);
v___x_682_ = v_reuseFailAlloc_683_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
return v___x_682_;
}
}
}
}
}
else
{
lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
lean_dec(v_a_648_);
v___x_685_ = lean_unsigned_to_nat(3u);
v___x_686_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray___closed__9));
v___x_687_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v___x_685_, v___x_686_, v_a_640_, v_a_641_, v_a_642_, v_a_643_, v_a_644_, v_a_645_);
if (lean_obj_tag(v___x_687_) == 0)
{
lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_697_; 
v_a_688_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_697_ == 0)
{
v___x_690_ = v___x_687_;
v_isShared_691_ = v_isSharedCheck_697_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_687_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_697_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_695_; 
v___x_692_ = lean_box(0);
v___x_693_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_693_, 0, v_a_688_);
lean_ctor_set(v___x_693_, 1, v___x_692_);
if (v_isShared_691_ == 0)
{
lean_ctor_set(v___x_690_, 0, v___x_693_);
v___x_695_ = v___x_690_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v___x_693_);
v___x_695_ = v_reuseFailAlloc_696_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
return v___x_695_;
}
}
}
else
{
lean_object* v_a_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_705_; 
v_a_698_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_705_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_705_ == 0)
{
v___x_700_ = v___x_687_;
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_a_698_);
lean_dec(v___x_687_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_703_; 
if (v_isShared_701_ == 0)
{
v___x_703_ = v___x_700_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v_a_698_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
return v___x_703_;
}
}
}
}
}
else
{
lean_object* v___x_706_; lean_object* v___x_708_; 
lean_dec(v_a_648_);
v___x_706_ = lean_box(0);
if (v_isShared_651_ == 0)
{
lean_ctor_set(v___x_650_, 0, v___x_706_);
v___x_708_ = v___x_650_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v___x_706_);
v___x_708_ = v_reuseFailAlloc_709_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
return v___x_708_;
}
}
}
}
else
{
lean_object* v_a_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_718_; 
v_a_711_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_718_ == 0)
{
v___x_713_ = v___x_647_;
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_a_711_);
lean_dec(v___x_647_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_716_; 
if (v_isShared_714_ == 0)
{
v___x_716_ = v___x_713_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_a_711_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
return v___x_716_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1(lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_){
_start:
{
lean_object* v___x_726_; 
v___x_726_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___redArg(v___y_719_);
return v___x_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1___boxed(lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_){
_start:
{
lean_object* v_res_734_; 
v_res_734_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1_spec__1(v___y_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_, v___y_732_);
lean_dec(v___y_732_);
lean_dec_ref(v___y_731_);
lean_dec(v___y_730_);
lean_dec_ref(v___y_729_);
lean_dec(v___y_728_);
lean_dec_ref(v___y_727_);
return v_res_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1(lean_object* v_00_u03b1_735_, lean_object* v_argIdx_736_, lean_object* v_x_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v___x_745_; 
v___x_745_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v_argIdx_736_, v_x_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___boxed(lean_object* v_00_u03b1_746_, lean_object* v_argIdx_747_, lean_object* v_x_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_){
_start:
{
lean_object* v_res_756_; 
v_res_756_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1(v_00_u03b1_746_, v_argIdx_747_, v_x_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec(v_argIdx_747_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation___lam__0(lean_object* v___x_758_, lean_object* v___x_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_){
_start:
{
lean_object* v___x_767_; 
v___x_767_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v___x_758_, v___x_759_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
if (lean_obj_tag(v___x_767_) == 0)
{
lean_object* v_a_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; 
v_a_768_ = lean_ctor_get(v___x_767_, 0);
lean_inc(v_a_768_);
lean_dec_ref_known(v___x_767_, 1);
v___x_769_ = lean_unsigned_to_nat(5u);
v___x_770_ = ((lean_object*)(lp_mathlib_Algebra_delabAdjoinNotation___lam__0___closed__0));
v___x_771_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_Mathlib_Algebra_Algebra_Subalgebra_Lattice_0__Algebra_delabAdjoinNotation_delabInsertArray_spec__1___redArg(v___x_769_, v___x_770_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
if (lean_obj_tag(v___x_771_) == 0)
{
lean_object* v_a_772_; lean_object* v___x_774_; uint8_t v_isShared_775_; uint8_t v_isSharedCheck_796_; 
v_a_772_ = lean_ctor_get(v___x_771_, 0);
v_isSharedCheck_796_ = !lean_is_exclusive(v___x_771_);
if (v_isSharedCheck_796_ == 0)
{
v___x_774_ = v___x_771_;
v_isShared_775_ = v_isSharedCheck_796_;
goto v_resetjp_773_;
}
else
{
lean_inc(v_a_772_);
lean_dec(v___x_771_);
v___x_774_ = lean_box(0);
v_isShared_775_ = v_isSharedCheck_796_;
goto v_resetjp_773_;
}
v_resetjp_773_:
{
lean_object* v_ref_776_; uint8_t v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_794_; 
v_ref_776_ = lean_ctor_get(v___y_764_, 5);
v___x_777_ = 0;
v___x_778_ = l_Lean_SourceInfo_fromRef(v_ref_776_, v___x_777_);
v___x_779_ = ((lean_object*)(lp_mathlib_Algebra_subalgebra__adjoin___closed__2));
v___x_780_ = ((lean_object*)(lp_mathlib_Algebra_subalgebra__adjoin___closed__5));
lean_inc_n(v___x_778_, 4);
v___x_781_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_781_, 0, v___x_778_);
lean_ctor_set(v___x_781_, 1, v___x_780_);
v___x_782_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1_spec__0___closed__25));
v___x_783_ = lean_obj_once(&lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14, &lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14_once, _init_lp_mathlib_Algebra___aux__Mathlib__Algebra__Algebra__Subalgebra__Lattice______macroRules__Algebra__subalgebra__adjoin__1___closed__14);
v___x_784_ = ((lean_object*)(lp_mathlib_Algebra_subalgebra__adjoin___closed__10));
v___x_785_ = lean_array_mk(v_a_772_);
v___x_786_ = l_Lean_Syntax_SepArray_ofElems(v___x_784_, v___x_785_);
lean_dec_ref(v___x_785_);
v___x_787_ = l_Array_append___redArg(v___x_783_, v___x_786_);
lean_dec_ref(v___x_786_);
v___x_788_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_788_, 0, v___x_778_);
lean_ctor_set(v___x_788_, 1, v___x_782_);
lean_ctor_set(v___x_788_, 2, v___x_787_);
v___x_789_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_789_, 0, v___x_778_);
lean_ctor_set(v___x_789_, 1, v___x_782_);
lean_ctor_set(v___x_789_, 2, v___x_783_);
v___x_790_ = ((lean_object*)(lp_mathlib_Algebra_subalgebra__adjoin___closed__22));
v___x_791_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_791_, 0, v___x_778_);
lean_ctor_set(v___x_791_, 1, v___x_790_);
v___x_792_ = l_Lean_Syntax_node5(v___x_778_, v___x_779_, v_a_768_, v___x_781_, v___x_788_, v___x_789_, v___x_791_);
if (v_isShared_775_ == 0)
{
lean_ctor_set(v___x_774_, 0, v___x_792_);
v___x_794_ = v___x_774_;
goto v_reusejp_793_;
}
else
{
lean_object* v_reuseFailAlloc_795_; 
v_reuseFailAlloc_795_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_795_, 0, v___x_792_);
v___x_794_ = v_reuseFailAlloc_795_;
goto v_reusejp_793_;
}
v_reusejp_793_:
{
return v___x_794_;
}
}
}
else
{
lean_object* v_a_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_804_; 
lean_dec(v_a_768_);
v_a_797_ = lean_ctor_get(v___x_771_, 0);
v_isSharedCheck_804_ = !lean_is_exclusive(v___x_771_);
if (v_isSharedCheck_804_ == 0)
{
v___x_799_ = v___x_771_;
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_a_797_);
lean_dec(v___x_771_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_802_; 
if (v_isShared_800_ == 0)
{
v___x_802_ = v___x_799_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v_a_797_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
return v___x_802_;
}
}
}
}
else
{
return v___x_767_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation___lam__0___boxed(lean_object* v___x_805_, lean_object* v___x_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_mathlib_Algebra_delabAdjoinNotation___lam__0(v___x_805_, v___x_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
lean_dec(v___y_810_);
lean_dec_ref(v___y_809_);
lean_dec(v___y_808_);
lean_dec_ref(v___y_807_);
lean_dec(v___x_805_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation(lean_object* v_a_822_, lean_object* v_a_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_829_ = ((lean_object*)(lp_mathlib_Algebra_delabAdjoinNotation___closed__0));
v___x_830_ = ((lean_object*)(lp_mathlib_Algebra_delabAdjoinNotation___closed__2));
v___x_831_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_829_, v___x_830_, v_a_822_, v_a_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_delabAdjoinNotation___boxed(lean_object* v_a_832_, lean_object* v_a_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_mathlib_Algebra_delabAdjoinNotation(v_a_832_, v_a_833_, v_a_834_, v_a_835_, v_a_836_, v_a_837_);
lean_dec(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec(v_a_835_);
lean_dec_ref(v_a_834_);
lean_dec(v_a_833_);
lean_dec_ref(v_a_832_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___redArg(lean_object* v_x_840_){
_start:
{
lean_inc(v_x_840_);
return v_x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___redArg___boxed(lean_object* v_x_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___redArg(v_x_841_);
lean_dec(v_x_841_);
return v_res_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux(lean_object* v_A_843_, lean_object* v_B_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_inst_847_, lean_object* v_s_848_, lean_object* v_x_849_, lean_object* v_inst_850_){
_start:
{
lean_inc(v_x_849_);
return v_x_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux___boxed(lean_object* v_A_851_, lean_object* v_B_852_, lean_object* v_inst_853_, lean_object* v_inst_854_, lean_object* v_inst_855_, lean_object* v_s_856_, lean_object* v_x_857_, lean_object* v_inst_858_){
_start:
{
lean_object* v_res_859_; 
v_res_859_ = lp_mathlib_Algebra_instCoeDepSubtypeMemSubalgebraAdjoinOfCoeAdjoinAux(v_A_851_, v_B_852_, v_inst_853_, v_inst_854_, v_inst_855_, v_s_856_, v_x_857_, v_inst_858_);
lean_dec(v_x_857_);
lean_dec_ref(v_inst_855_);
lean_dec_ref(v_inst_854_);
lean_dec_ref(v_inst_853_);
return v_res_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommSemiringOfComm___redArg(lean_object* v_inst_860_){
_start:
{
lean_object* v___x_861_; 
v___x_861_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_860_);
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommSemiringOfComm(lean_object* v_R_862_, lean_object* v_A_863_, lean_object* v_inst_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_s_867_, lean_object* v_hcomm_868_){
_start:
{
lean_object* v___x_869_; 
v___x_869_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_865_);
return v___x_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommSemiringOfComm___boxed(lean_object* v_R_870_, lean_object* v_A_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_s_875_, lean_object* v_hcomm_876_){
_start:
{
lean_object* v_res_877_; 
v_res_877_ = lp_mathlib_Algebra_adjoinCommSemiringOfComm(v_R_870_, v_A_871_, v_inst_872_, v_inst_873_, v_inst_874_, v_s_875_, v_hcomm_876_);
lean_dec_ref(v_inst_874_);
lean_dec_ref(v_inst_872_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommRingOfComm___redArg(lean_object* v_inst_878_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommRingOfComm(lean_object* v_R_880_, lean_object* v_A_881_, lean_object* v_inst_882_, lean_object* v_inst_883_, lean_object* v_inst_884_, lean_object* v_s_885_, lean_object* v_hcomm_886_){
_start:
{
lean_object* v___x_887_; 
v___x_887_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_883_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_adjoinCommRingOfComm___boxed(lean_object* v_R_888_, lean_object* v_A_889_, lean_object* v_inst_890_, lean_object* v_inst_891_, lean_object* v_inst_892_, lean_object* v_s_893_, lean_object* v_hcomm_894_){
_start:
{
lean_object* v_res_895_; 
v_res_895_ = lp_mathlib_Algebra_adjoinCommRingOfComm(v_R_888_, v_A_889_, v_inst_890_, v_inst_891_, v_inst_892_, v_s_893_, v_hcomm_894_);
lean_dec_ref(v_inst_892_);
lean_dec_ref(v_inst_890_);
return v_res_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureEquivAdjoinNat___redArg(lean_object* v_inst_896_){
_start:
{
lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; 
v___x_897_ = lp_mathlib_Nat_instSemiring;
lean_inc_ref(v_inst_896_);
v___x_898_ = lp_mathlib_Semiring_toNatAlgebra___redArg(v_inst_896_);
v___x_899_ = lean_box(0);
v___x_900_ = lp_mathlib_Subalgebra_equivOfEq(lean_box(0), lean_box(0), v___x_897_, v_inst_896_, v___x_898_, v___x_899_, v___x_899_, lean_box(0));
lean_dec_ref(v___x_898_);
lean_dec_ref(v_inst_896_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureEquivAdjoinNat(lean_object* v_R_901_, lean_object* v_inst_902_, lean_object* v_s_903_){
_start:
{
lean_object* v___x_904_; 
v___x_904_ = lp_mathlib_Subsemiring_closureEquivAdjoinNat___redArg(v_inst_902_);
return v___x_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureEquivAdjoinInt___redArg(lean_object* v_inst_905_){
_start:
{
lean_object* v___x_906_; lean_object* v_toSemiring_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; 
v___x_906_ = lp_mathlib_Int_instCommSemiring;
v_toSemiring_907_ = lean_ctor_get(v_inst_905_, 0);
lean_inc_ref(v_toSemiring_907_);
v___x_908_ = lp_mathlib_Ring_toIntAlgebra___redArg(v_inst_905_);
v___x_909_ = lean_box(0);
v___x_910_ = lp_mathlib_Subalgebra_equivOfEq(lean_box(0), lean_box(0), v___x_906_, v_toSemiring_907_, v___x_908_, v___x_909_, v___x_909_, lean_box(0));
lean_dec_ref(v___x_908_);
lean_dec_ref(v_toSemiring_907_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureEquivAdjoinInt(lean_object* v_R_911_, lean_object* v_inst_912_, lean_object* v_s_913_){
_start:
{
lean_object* v___x_914_; 
v___x_914_ = lp_mathlib_Subring_closureEquivAdjoinInt___redArg(v_inst_912_);
return v___x_914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebraOrderEmbedding___redArg(lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_){
_start:
{
lean_object* v___x_918_; 
v___x_918_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_toNonUnitalSubalgebra___boxed), 6, 5);
lean_closure_set(v___x_918_, 0, lean_box(0));
lean_closure_set(v___x_918_, 1, lean_box(0));
lean_closure_set(v___x_918_, 2, v_inst_915_);
lean_closure_set(v___x_918_, 3, v_inst_916_);
lean_closure_set(v___x_918_, 4, v_inst_917_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subalgebra_toNonUnitalSubalgebraOrderEmbedding(lean_object* v_R_919_, lean_object* v_A_920_, lean_object* v_inst_921_, lean_object* v_inst_922_, lean_object* v_inst_923_){
_start:
{
lean_object* v___x_924_; 
v___x_924_ = lean_alloc_closure((void*)(lp_mathlib_Subalgebra_toNonUnitalSubalgebra___boxed), 6, 5);
lean_closure_set(v___x_924_, 0, lean_box(0));
lean_closure_set(v___x_924_, 1, lean_box(0));
lean_closure_set(v___x_924_, 2, v_inst_921_);
lean_closure_set(v___x_924_, 3, v_inst_922_);
lean_closure_set(v___x_924_, 4, v_inst_923_);
return v___x_924_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
