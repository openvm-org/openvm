// Lean compiler output
// Module: Mathlib.Util.TermReduce
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Delta public import Mathlib.Init
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
lean_object* lp_mathlib_Lean_Expr_reduceProjStruct_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_ExprStructEq_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
extern lean_object* l_Lean_interruptExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t l_Lean_ExprStructEq_beq(lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_IO_CancelToken_isSet(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_zetaReduce(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_delta_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Util"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "TermReduce"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "betaStx"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 193, 160, 41, 66, 209, 63, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 26, 109, 107, 185, 12, 19, 22)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 116, 129, 127, 230, 3, 22, 205)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "beta% "};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__9_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Util_TermReduce_betaStx = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__13_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabBeta(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabBeta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "deltaStx"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 193, 160, 41, 66, 209, 63, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 26, 109, 107, 185, 12, 19, 22)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 67, 192, 249, 25, 189, 189, 193)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "delta% "};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Util_TermReduce_deltaStx = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "cannot delta reduce "};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "zetaStx"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 193, 160, 41, 66, 209, 63, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 26, 109, 107, 185, 12, 19, 22)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 125, 4, 201, 212, 46, 66, 85)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "zeta% "};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Util_TermReduce_zetaStx = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabZeta(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabZeta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "reduceProjStx"};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(144, 193, 160, 41, 66, 209, 63, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 26, 109, 107, 185, 12, 19, 22)}};
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 131, 108, 60, 117, 65, 72, 58)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "reduceProj% "};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "transform"};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12_spec__13(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_31_ = lean_box(0);
v___x_32_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_33_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___x_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg(){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___closed__0);
v___x_36_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg___boxed(lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0(lean_object* v_00_u03b1_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___boxed(lean_object* v_00_u03b1_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0(v_00_u03b1_48_, v___y_49_, v___y_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(lean_object* v_e_57_, lean_object* v___y_58_){
_start:
{
uint8_t v___x_60_; 
v___x_60_ = l_Lean_Expr_hasMVar(v_e_57_);
if (v___x_60_ == 0)
{
lean_object* v___x_61_; 
v___x_61_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_61_, 0, v_e_57_);
return v___x_61_;
}
else
{
lean_object* v___x_62_; lean_object* v_mctx_63_; lean_object* v___x_64_; lean_object* v_fst_65_; lean_object* v_snd_66_; lean_object* v___x_67_; lean_object* v_cache_68_; lean_object* v_zetaDeltaFVarIds_69_; lean_object* v_postponed_70_; lean_object* v_diag_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_80_; 
v___x_62_ = lean_st_ref_get(v___y_58_);
v_mctx_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc_ref(v_mctx_63_);
lean_dec(v___x_62_);
v___x_64_ = l_Lean_instantiateMVarsCore(v_mctx_63_, v_e_57_);
v_fst_65_ = lean_ctor_get(v___x_64_, 0);
lean_inc(v_fst_65_);
v_snd_66_ = lean_ctor_get(v___x_64_, 1);
lean_inc(v_snd_66_);
lean_dec_ref(v___x_64_);
v___x_67_ = lean_st_ref_take(v___y_58_);
v_cache_68_ = lean_ctor_get(v___x_67_, 1);
v_zetaDeltaFVarIds_69_ = lean_ctor_get(v___x_67_, 2);
v_postponed_70_ = lean_ctor_get(v___x_67_, 3);
v_diag_71_ = lean_ctor_get(v___x_67_, 4);
v_isSharedCheck_80_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_80_ == 0)
{
lean_object* v_unused_81_; 
v_unused_81_ = lean_ctor_get(v___x_67_, 0);
lean_dec(v_unused_81_);
v___x_73_ = v___x_67_;
v_isShared_74_ = v_isSharedCheck_80_;
goto v_resetjp_72_;
}
else
{
lean_inc(v_diag_71_);
lean_inc(v_postponed_70_);
lean_inc(v_zetaDeltaFVarIds_69_);
lean_inc(v_cache_68_);
lean_dec(v___x_67_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_80_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
lean_object* v___x_76_; 
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 0, v_snd_66_);
v___x_76_ = v___x_73_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_79_; 
v_reuseFailAlloc_79_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_79_, 0, v_snd_66_);
lean_ctor_set(v_reuseFailAlloc_79_, 1, v_cache_68_);
lean_ctor_set(v_reuseFailAlloc_79_, 2, v_zetaDeltaFVarIds_69_);
lean_ctor_set(v_reuseFailAlloc_79_, 3, v_postponed_70_);
lean_ctor_set(v_reuseFailAlloc_79_, 4, v_diag_71_);
v___x_76_ = v_reuseFailAlloc_79_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_77_ = lean_st_ref_set(v___y_58_, v___x_76_);
v___x_78_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_78_, 0, v_fst_65_);
return v___x_78_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg___boxed(lean_object* v_e_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(v_e_82_, v___y_83_);
lean_dec(v___y_83_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1(lean_object* v_e_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(v_e_86_, v___y_90_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___boxed(lean_object* v_e_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1(v_e_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_);
lean_dec(v___y_101_);
lean_dec_ref(v___y_100_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabBeta(lean_object* v_stx_104_, lean_object* v_expectedType_x3f_105_, lean_object* v_a_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_betaStx___closed__4));
lean_inc(v_stx_104_);
v___x_114_ = l_Lean_Syntax_isOfKind(v_stx_104_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; 
lean_dec(v_expectedType_x3f_105_);
lean_dec(v_stx_104_);
v___x_115_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_116_ = lean_unsigned_to_nat(1u);
v___x_117_ = l_Lean_Syntax_getArg(v_stx_104_, v___x_116_);
lean_dec(v_stx_104_);
v___x_118_ = l_Lean_Elab_Term_elabTerm(v___x_117_, v_expectedType_x3f_105_, v___x_114_, v___x_114_, v_a_106_, v_a_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_object* v_a_119_; lean_object* v___x_120_; lean_object* v_a_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_129_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
lean_inc(v_a_119_);
lean_dec_ref_known(v___x_118_, 1);
v___x_120_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(v_a_119_, v_a_109_);
v_a_121_ = lean_ctor_get(v___x_120_, 0);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_129_ == 0)
{
v___x_123_ = v___x_120_;
v_isShared_124_ = v_isSharedCheck_129_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_a_121_);
lean_dec(v___x_120_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_129_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v___x_125_; lean_object* v___x_127_; 
v___x_125_ = l_Lean_Expr_headBeta(v_a_121_);
if (v_isShared_124_ == 0)
{
lean_ctor_set(v___x_123_, 0, v___x_125_);
v___x_127_ = v___x_123_;
goto v_reusejp_126_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v___x_125_);
v___x_127_ = v_reuseFailAlloc_128_;
goto v_reusejp_126_;
}
v_reusejp_126_:
{
return v___x_127_;
}
}
}
else
{
return v___x_118_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabBeta___boxed(lean_object* v_stx_130_, lean_object* v_expectedType_x3f_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_, lean_object* v_a_137_, lean_object* v_a_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Mathlib_Util_TermReduce_elabBeta(v_stx_130_, v_expectedType_x3f_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_, v_a_136_, v_a_137_);
lean_dec(v_a_137_);
lean_dec_ref(v_a_136_);
lean_dec(v_a_135_);
lean_dec_ref(v_a_134_);
lean_dec(v_a_133_);
lean_dec_ref(v_a_132_);
return v_res_139_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__0(uint8_t v___x_158_, lean_object* v_x_159_){
_start:
{
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__0___boxed(lean_object* v___x_160_, lean_object* v_x_161_){
_start:
{
uint8_t v___x_7071__boxed_162_; uint8_t v_res_163_; lean_object* v_r_164_; 
v___x_7071__boxed_162_ = lean_unbox(v___x_160_);
v_res_163_ = lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__0(v___x_7071__boxed_162_, v_x_161_);
lean_dec(v_x_161_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__1(lean_object* v_a_165_, lean_object* v___f_166_, uint8_t v___x_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = l_Lean_Meta_delta_x3f(v_a_165_, v___f_166_, v___x_167_, v___y_172_, v___y_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__1___boxed(lean_object* v_a_176_, lean_object* v___f_177_, lean_object* v___x_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
uint8_t v___x_7079__boxed_186_; lean_object* v_res_187_; 
v___x_7079__boxed_186_ = lean_unbox(v___x_178_);
v_res_187_ = lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__1(v_a_176_, v___f_177_, v___x_7079__boxed_186_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0(lean_object* v___y_188_, uint8_t v_isExporting_189_, lean_object* v___x_190_, lean_object* v___y_191_, lean_object* v___x_192_, lean_object* v_a_x3f_193_){
_start:
{
lean_object* v___x_195_; lean_object* v_env_196_; lean_object* v_nextMacroScope_197_; lean_object* v_ngen_198_; lean_object* v_auxDeclNGen_199_; lean_object* v_traceState_200_; lean_object* v_messages_201_; lean_object* v_infoState_202_; lean_object* v_snapshotTasks_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_228_; 
v___x_195_ = lean_st_ref_take(v___y_188_);
v_env_196_ = lean_ctor_get(v___x_195_, 0);
v_nextMacroScope_197_ = lean_ctor_get(v___x_195_, 1);
v_ngen_198_ = lean_ctor_get(v___x_195_, 2);
v_auxDeclNGen_199_ = lean_ctor_get(v___x_195_, 3);
v_traceState_200_ = lean_ctor_get(v___x_195_, 4);
v_messages_201_ = lean_ctor_get(v___x_195_, 6);
v_infoState_202_ = lean_ctor_get(v___x_195_, 7);
v_snapshotTasks_203_ = lean_ctor_get(v___x_195_, 8);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_195_);
if (v_isSharedCheck_228_ == 0)
{
lean_object* v_unused_229_; 
v_unused_229_ = lean_ctor_get(v___x_195_, 5);
lean_dec(v_unused_229_);
v___x_205_ = v___x_195_;
v_isShared_206_ = v_isSharedCheck_228_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_snapshotTasks_203_);
lean_inc(v_infoState_202_);
lean_inc(v_messages_201_);
lean_inc(v_traceState_200_);
lean_inc(v_auxDeclNGen_199_);
lean_inc(v_ngen_198_);
lean_inc(v_nextMacroScope_197_);
lean_inc(v_env_196_);
lean_dec(v___x_195_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_228_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_207_; lean_object* v___x_209_; 
v___x_207_ = l_Lean_Environment_setExporting(v_env_196_, v_isExporting_189_);
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 5, v___x_190_);
lean_ctor_set(v___x_205_, 0, v___x_207_);
v___x_209_ = v___x_205_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v___x_207_);
lean_ctor_set(v_reuseFailAlloc_227_, 1, v_nextMacroScope_197_);
lean_ctor_set(v_reuseFailAlloc_227_, 2, v_ngen_198_);
lean_ctor_set(v_reuseFailAlloc_227_, 3, v_auxDeclNGen_199_);
lean_ctor_set(v_reuseFailAlloc_227_, 4, v_traceState_200_);
lean_ctor_set(v_reuseFailAlloc_227_, 5, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_227_, 6, v_messages_201_);
lean_ctor_set(v_reuseFailAlloc_227_, 7, v_infoState_202_);
lean_ctor_set(v_reuseFailAlloc_227_, 8, v_snapshotTasks_203_);
v___x_209_ = v_reuseFailAlloc_227_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v_mctx_212_; lean_object* v_zetaDeltaFVarIds_213_; lean_object* v_postponed_214_; lean_object* v_diag_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_225_; 
v___x_210_ = lean_st_ref_set(v___y_188_, v___x_209_);
v___x_211_ = lean_st_ref_take(v___y_191_);
v_mctx_212_ = lean_ctor_get(v___x_211_, 0);
v_zetaDeltaFVarIds_213_ = lean_ctor_get(v___x_211_, 2);
v_postponed_214_ = lean_ctor_get(v___x_211_, 3);
v_diag_215_ = lean_ctor_get(v___x_211_, 4);
v_isSharedCheck_225_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_225_ == 0)
{
lean_object* v_unused_226_; 
v_unused_226_ = lean_ctor_get(v___x_211_, 1);
lean_dec(v_unused_226_);
v___x_217_ = v___x_211_;
v_isShared_218_ = v_isSharedCheck_225_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_diag_215_);
lean_inc(v_postponed_214_);
lean_inc(v_zetaDeltaFVarIds_213_);
lean_inc(v_mctx_212_);
lean_dec(v___x_211_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_225_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v___x_220_; 
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 1, v___x_192_);
v___x_220_ = v___x_217_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_224_; 
v_reuseFailAlloc_224_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_224_, 0, v_mctx_212_);
lean_ctor_set(v_reuseFailAlloc_224_, 1, v___x_192_);
lean_ctor_set(v_reuseFailAlloc_224_, 2, v_zetaDeltaFVarIds_213_);
lean_ctor_set(v_reuseFailAlloc_224_, 3, v_postponed_214_);
lean_ctor_set(v_reuseFailAlloc_224_, 4, v_diag_215_);
v___x_220_ = v_reuseFailAlloc_224_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_221_ = lean_st_ref_set(v___y_191_, v___x_220_);
v___x_222_ = lean_box(0);
v___x_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
return v___x_223_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v___y_230_, lean_object* v_isExporting_231_, lean_object* v___x_232_, lean_object* v___y_233_, lean_object* v___x_234_, lean_object* v_a_x3f_235_, lean_object* v___y_236_){
_start:
{
uint8_t v_isExporting_boxed_237_; lean_object* v_res_238_; 
v_isExporting_boxed_237_ = lean_unbox(v_isExporting_231_);
v_res_238_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0(v___y_230_, v_isExporting_boxed_237_, v___x_232_, v___y_233_, v___x_234_, v_a_x3f_235_);
lean_dec(v_a_x3f_235_);
lean_dec(v___y_233_);
lean_dec(v___y_230_);
return v_res_238_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_239_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_240_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__0);
v___x_241_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
return v___x_241_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
return v___x_243_;
}
}
static lean_object* _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_244_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__1);
v___x_245_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v___x_244_);
lean_ctor_set(v___x_245_, 2, v___x_244_);
lean_ctor_set(v___x_245_, 3, v___x_244_);
lean_ctor_set(v___x_245_, 4, v___x_244_);
lean_ctor_set(v___x_245_, 5, v___x_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg(lean_object* v_x_246_, uint8_t v_isExporting_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_){
_start:
{
lean_object* v___x_255_; lean_object* v_env_256_; uint8_t v_isExporting_257_; lean_object* v___x_323_; uint8_t v_isModule_324_; 
v___x_255_ = lean_st_ref_get(v___y_253_);
v_env_256_ = lean_ctor_get(v___x_255_, 0);
lean_inc_ref(v_env_256_);
lean_dec(v___x_255_);
v_isExporting_257_ = lean_ctor_get_uint8(v_env_256_, sizeof(void*)*8);
v___x_323_ = l_Lean_Environment_header(v_env_256_);
lean_dec_ref(v_env_256_);
v_isModule_324_ = lean_ctor_get_uint8(v___x_323_, sizeof(void*)*7 + 4);
lean_dec_ref(v___x_323_);
if (v_isModule_324_ == 0)
{
lean_object* v___x_325_; 
lean_inc(v___y_253_);
lean_inc_ref(v___y_252_);
lean_inc(v___y_251_);
lean_inc_ref(v___y_250_);
lean_inc(v___y_249_);
lean_inc_ref(v___y_248_);
v___x_325_ = lean_apply_7(v_x_246_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, lean_box(0));
return v___x_325_;
}
else
{
if (v_isExporting_257_ == 0)
{
if (v_isExporting_247_ == 0)
{
lean_object* v___x_326_; 
lean_inc(v___y_253_);
lean_inc_ref(v___y_252_);
lean_inc(v___y_251_);
lean_inc_ref(v___y_250_);
lean_inc(v___y_249_);
lean_inc_ref(v___y_248_);
v___x_326_ = lean_apply_7(v_x_246_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, lean_box(0));
return v___x_326_;
}
else
{
goto v___jp_258_;
}
}
else
{
if (v_isExporting_247_ == 0)
{
goto v___jp_258_;
}
else
{
lean_object* v___x_327_; 
lean_inc(v___y_253_);
lean_inc_ref(v___y_252_);
lean_inc(v___y_251_);
lean_inc_ref(v___y_250_);
lean_inc(v___y_249_);
lean_inc_ref(v___y_248_);
v___x_327_ = lean_apply_7(v_x_246_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, lean_box(0));
return v___x_327_;
}
}
}
v___jp_258_:
{
lean_object* v___x_259_; lean_object* v_env_260_; lean_object* v_nextMacroScope_261_; lean_object* v_ngen_262_; lean_object* v_auxDeclNGen_263_; lean_object* v_traceState_264_; lean_object* v_messages_265_; lean_object* v_infoState_266_; lean_object* v_snapshotTasks_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_321_; 
v___x_259_ = lean_st_ref_take(v___y_253_);
v_env_260_ = lean_ctor_get(v___x_259_, 0);
v_nextMacroScope_261_ = lean_ctor_get(v___x_259_, 1);
v_ngen_262_ = lean_ctor_get(v___x_259_, 2);
v_auxDeclNGen_263_ = lean_ctor_get(v___x_259_, 3);
v_traceState_264_ = lean_ctor_get(v___x_259_, 4);
v_messages_265_ = lean_ctor_get(v___x_259_, 6);
v_infoState_266_ = lean_ctor_get(v___x_259_, 7);
v_snapshotTasks_267_ = lean_ctor_get(v___x_259_, 8);
v_isSharedCheck_321_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_321_ == 0)
{
lean_object* v_unused_322_; 
v_unused_322_ = lean_ctor_get(v___x_259_, 5);
lean_dec(v_unused_322_);
v___x_269_ = v___x_259_;
v_isShared_270_ = v_isSharedCheck_321_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_snapshotTasks_267_);
lean_inc(v_infoState_266_);
lean_inc(v_messages_265_);
lean_inc(v_traceState_264_);
lean_inc(v_auxDeclNGen_263_);
lean_inc(v_ngen_262_);
lean_inc(v_nextMacroScope_261_);
lean_inc(v_env_260_);
lean_dec(v___x_259_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_321_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_274_; 
v___x_271_ = l_Lean_Environment_setExporting(v_env_260_, v_isExporting_247_);
v___x_272_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__2, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__2);
if (v_isShared_270_ == 0)
{
lean_ctor_set(v___x_269_, 5, v___x_272_);
lean_ctor_set(v___x_269_, 0, v___x_271_);
v___x_274_ = v___x_269_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_271_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v_nextMacroScope_261_);
lean_ctor_set(v_reuseFailAlloc_320_, 2, v_ngen_262_);
lean_ctor_set(v_reuseFailAlloc_320_, 3, v_auxDeclNGen_263_);
lean_ctor_set(v_reuseFailAlloc_320_, 4, v_traceState_264_);
lean_ctor_set(v_reuseFailAlloc_320_, 5, v___x_272_);
lean_ctor_set(v_reuseFailAlloc_320_, 6, v_messages_265_);
lean_ctor_set(v_reuseFailAlloc_320_, 7, v_infoState_266_);
lean_ctor_set(v_reuseFailAlloc_320_, 8, v_snapshotTasks_267_);
v___x_274_ = v_reuseFailAlloc_320_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v_mctx_277_; lean_object* v_zetaDeltaFVarIds_278_; lean_object* v_postponed_279_; lean_object* v_diag_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_318_; 
v___x_275_ = lean_st_ref_set(v___y_253_, v___x_274_);
v___x_276_ = lean_st_ref_take(v___y_251_);
v_mctx_277_ = lean_ctor_get(v___x_276_, 0);
v_zetaDeltaFVarIds_278_ = lean_ctor_get(v___x_276_, 2);
v_postponed_279_ = lean_ctor_get(v___x_276_, 3);
v_diag_280_ = lean_ctor_get(v___x_276_, 4);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_276_);
if (v_isSharedCheck_318_ == 0)
{
lean_object* v_unused_319_; 
v_unused_319_ = lean_ctor_get(v___x_276_, 1);
lean_dec(v_unused_319_);
v___x_282_ = v___x_276_;
v_isShared_283_ = v_isSharedCheck_318_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_diag_280_);
lean_inc(v_postponed_279_);
lean_inc(v_zetaDeltaFVarIds_278_);
lean_inc(v_mctx_277_);
lean_dec(v___x_276_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_318_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___x_284_; lean_object* v___x_286_; 
v___x_284_ = lean_obj_once(&lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__3, &lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___closed__3);
if (v_isShared_283_ == 0)
{
lean_ctor_set(v___x_282_, 1, v___x_284_);
v___x_286_ = v___x_282_;
goto v_reusejp_285_;
}
else
{
lean_object* v_reuseFailAlloc_317_; 
v_reuseFailAlloc_317_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_317_, 0, v_mctx_277_);
lean_ctor_set(v_reuseFailAlloc_317_, 1, v___x_284_);
lean_ctor_set(v_reuseFailAlloc_317_, 2, v_zetaDeltaFVarIds_278_);
lean_ctor_set(v_reuseFailAlloc_317_, 3, v_postponed_279_);
lean_ctor_set(v_reuseFailAlloc_317_, 4, v_diag_280_);
v___x_286_ = v_reuseFailAlloc_317_;
goto v_reusejp_285_;
}
v_reusejp_285_:
{
lean_object* v___x_287_; lean_object* v_r_288_; 
v___x_287_ = lean_st_ref_set(v___y_251_, v___x_286_);
lean_inc(v___y_253_);
lean_inc_ref(v___y_252_);
lean_inc(v___y_251_);
lean_inc_ref(v___y_250_);
lean_inc(v___y_249_);
lean_inc_ref(v___y_248_);
v_r_288_ = lean_apply_7(v_x_246_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, lean_box(0));
if (lean_obj_tag(v_r_288_) == 0)
{
lean_object* v_a_289_; lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_305_; 
v_a_289_ = lean_ctor_get(v_r_288_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v_r_288_);
if (v_isSharedCheck_305_ == 0)
{
v___x_291_ = v_r_288_;
v_isShared_292_ = v_isSharedCheck_305_;
goto v_resetjp_290_;
}
else
{
lean_inc(v_a_289_);
lean_dec(v_r_288_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_305_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
lean_object* v___x_294_; 
lean_inc(v_a_289_);
if (v_isShared_292_ == 0)
{
lean_ctor_set_tag(v___x_291_, 1);
v___x_294_ = v___x_291_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_289_);
v___x_294_ = v_reuseFailAlloc_304_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
lean_object* v___x_295_; lean_object* v___x_297_; uint8_t v_isShared_298_; uint8_t v_isSharedCheck_302_; 
v___x_295_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0(v___y_253_, v_isExporting_257_, v___x_272_, v___y_251_, v___x_284_, v___x_294_);
lean_dec_ref(v___x_294_);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_295_);
if (v_isSharedCheck_302_ == 0)
{
lean_object* v_unused_303_; 
v_unused_303_ = lean_ctor_get(v___x_295_, 0);
lean_dec(v_unused_303_);
v___x_297_ = v___x_295_;
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
else
{
lean_dec(v___x_295_);
v___x_297_ = lean_box(0);
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
v_resetjp_296_:
{
lean_object* v___x_300_; 
if (v_isShared_298_ == 0)
{
lean_ctor_set(v___x_297_, 0, v_a_289_);
v___x_300_ = v___x_297_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_a_289_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
}
else
{
lean_object* v_a_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_315_; 
v_a_306_ = lean_ctor_get(v_r_288_, 0);
lean_inc(v_a_306_);
lean_dec_ref_known(v_r_288_, 1);
v___x_307_ = lean_box(0);
v___x_308_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___lam__0(v___y_253_, v_isExporting_257_, v___x_272_, v___y_251_, v___x_284_, v___x_307_);
v_isSharedCheck_315_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_315_ == 0)
{
lean_object* v_unused_316_; 
v_unused_316_ = lean_ctor_get(v___x_308_, 0);
lean_dec(v_unused_316_);
v___x_310_ = v___x_308_;
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
else
{
lean_dec(v___x_308_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_313_; 
if (v_isShared_311_ == 0)
{
lean_ctor_set_tag(v___x_310_, 1);
lean_ctor_set(v___x_310_, 0, v_a_306_);
v___x_313_ = v___x_310_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_314_; 
v_reuseFailAlloc_314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_314_, 0, v_a_306_);
v___x_313_ = v_reuseFailAlloc_314_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
return v___x_313_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg___boxed(lean_object* v_x_328_, lean_object* v_isExporting_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
uint8_t v_isExporting_boxed_337_; lean_object* v_res_338_; 
v_isExporting_boxed_337_ = lean_unbox(v_isExporting_329_);
v_res_338_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg(v_x_328_, v_isExporting_boxed_337_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
lean_dec(v___y_331_);
lean_dec_ref(v___y_330_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg(lean_object* v_x_339_, uint8_t v_when_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
if (v_when_340_ == 0)
{
lean_object* v___x_348_; 
lean_inc(v___y_346_);
lean_inc_ref(v___y_345_);
lean_inc(v___y_344_);
lean_inc_ref(v___y_343_);
lean_inc(v___y_342_);
lean_inc_ref(v___y_341_);
v___x_348_ = lean_apply_7(v_x_339_, v___y_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, lean_box(0));
return v___x_348_;
}
else
{
uint8_t v___x_349_; lean_object* v___x_350_; 
v___x_349_ = 0;
v___x_350_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg(v_x_339_, v___x_349_, v___y_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_);
return v___x_350_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg___boxed(lean_object* v_x_351_, lean_object* v_when_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
uint8_t v_when_boxed_360_; lean_object* v_res_361_; 
v_when_boxed_360_ = lean_unbox(v_when_352_);
v_res_361_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg(v_x_351_, v_when_boxed_360_, v___y_353_, v___y_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_);
lean_dec(v___y_358_);
lean_dec_ref(v___y_357_);
lean_dec(v___y_356_);
lean_dec_ref(v___y_355_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
return v_res_361_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0(void){
_start:
{
lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_362_ = lean_box(1);
v___x_363_ = l_Lean_MessageData_ofFormat(v___x_362_);
return v___x_363_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__3(void){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_367_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__2));
v___x_368_ = l_Lean_MessageData_ofFormat(v___x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5(lean_object* v_x_369_, lean_object* v_x_370_){
_start:
{
if (lean_obj_tag(v_x_370_) == 0)
{
return v_x_369_;
}
else
{
lean_object* v_head_371_; lean_object* v_tail_372_; lean_object* v___x_374_; uint8_t v_isShared_375_; uint8_t v_isSharedCheck_394_; 
v_head_371_ = lean_ctor_get(v_x_370_, 0);
v_tail_372_ = lean_ctor_get(v_x_370_, 1);
v_isSharedCheck_394_ = !lean_is_exclusive(v_x_370_);
if (v_isSharedCheck_394_ == 0)
{
v___x_374_ = v_x_370_;
v_isShared_375_ = v_isSharedCheck_394_;
goto v_resetjp_373_;
}
else
{
lean_inc(v_tail_372_);
lean_inc(v_head_371_);
lean_dec(v_x_370_);
v___x_374_ = lean_box(0);
v_isShared_375_ = v_isSharedCheck_394_;
goto v_resetjp_373_;
}
v_resetjp_373_:
{
lean_object* v_before_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_392_; 
v_before_376_ = lean_ctor_get(v_head_371_, 0);
v_isSharedCheck_392_ = !lean_is_exclusive(v_head_371_);
if (v_isSharedCheck_392_ == 0)
{
lean_object* v_unused_393_; 
v_unused_393_ = lean_ctor_get(v_head_371_, 1);
lean_dec(v_unused_393_);
v___x_378_ = v_head_371_;
v_isShared_379_ = v_isSharedCheck_392_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_before_376_);
lean_dec(v_head_371_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_392_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_380_; lean_object* v___x_382_; 
v___x_380_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0);
if (v_isShared_379_ == 0)
{
lean_ctor_set_tag(v___x_378_, 7);
lean_ctor_set(v___x_378_, 1, v___x_380_);
lean_ctor_set(v___x_378_, 0, v_x_369_);
v___x_382_ = v___x_378_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v_x_369_);
lean_ctor_set(v_reuseFailAlloc_391_, 1, v___x_380_);
v___x_382_ = v_reuseFailAlloc_391_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
lean_object* v___x_383_; lean_object* v___x_385_; 
v___x_383_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__3);
if (v_isShared_375_ == 0)
{
lean_ctor_set_tag(v___x_374_, 7);
lean_ctor_set(v___x_374_, 1, v___x_383_);
lean_ctor_set(v___x_374_, 0, v___x_382_);
v___x_385_ = v___x_374_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v___x_382_);
lean_ctor_set(v_reuseFailAlloc_390_, 1, v___x_383_);
v___x_385_ = v_reuseFailAlloc_390_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_386_ = l_Lean_MessageData_ofSyntax(v_before_376_);
v___x_387_ = l_Lean_indentD(v___x_386_);
v___x_388_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_385_);
lean_ctor_set(v___x_388_, 1, v___x_387_);
v_x_369_ = v___x_388_;
v_x_370_ = v_tail_372_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__4(lean_object* v_opts_395_, lean_object* v_opt_396_){
_start:
{
lean_object* v_name_397_; lean_object* v_defValue_398_; lean_object* v_map_399_; lean_object* v___x_400_; 
v_name_397_ = lean_ctor_get(v_opt_396_, 0);
v_defValue_398_ = lean_ctor_get(v_opt_396_, 1);
v_map_399_ = lean_ctor_get(v_opts_395_, 0);
v___x_400_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_399_, v_name_397_);
if (lean_obj_tag(v___x_400_) == 0)
{
uint8_t v___x_401_; 
v___x_401_ = lean_unbox(v_defValue_398_);
return v___x_401_;
}
else
{
lean_object* v_val_402_; 
v_val_402_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_val_402_);
lean_dec_ref_known(v___x_400_, 1);
if (lean_obj_tag(v_val_402_) == 1)
{
uint8_t v_v_403_; 
v_v_403_ = lean_ctor_get_uint8(v_val_402_, 0);
lean_dec_ref_known(v_val_402_, 0);
return v_v_403_;
}
else
{
uint8_t v___x_404_; 
lean_dec(v_val_402_);
v___x_404_ = lean_unbox(v_defValue_398_);
return v___x_404_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__4___boxed(lean_object* v_opts_405_, lean_object* v_opt_406_){
_start:
{
uint8_t v_res_407_; lean_object* v_r_408_; 
v_res_407_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__4(v_opts_405_, v_opt_406_);
lean_dec_ref(v_opt_406_);
lean_dec_ref(v_opts_405_);
v_r_408_ = lean_box(v_res_407_);
return v_r_408_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__1));
v___x_413_ = l_Lean_MessageData_ofFormat(v___x_412_);
return v___x_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg(lean_object* v_msgData_414_, lean_object* v_macroStack_415_, lean_object* v___y_416_){
_start:
{
lean_object* v_options_418_; lean_object* v___x_419_; uint8_t v___x_420_; 
v_options_418_ = lean_ctor_get(v___y_416_, 2);
v___x_419_ = l_Lean_Elab_pp_macroStack;
v___x_420_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__4(v_options_418_, v___x_419_);
if (v___x_420_ == 0)
{
lean_object* v___x_421_; 
lean_dec(v_macroStack_415_);
v___x_421_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_421_, 0, v_msgData_414_);
return v___x_421_;
}
else
{
if (lean_obj_tag(v_macroStack_415_) == 0)
{
lean_object* v___x_422_; 
v___x_422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_422_, 0, v_msgData_414_);
return v___x_422_;
}
else
{
lean_object* v_head_423_; lean_object* v_after_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_439_; 
v_head_423_ = lean_ctor_get(v_macroStack_415_, 0);
lean_inc(v_head_423_);
v_after_424_ = lean_ctor_get(v_head_423_, 1);
v_isSharedCheck_439_ = !lean_is_exclusive(v_head_423_);
if (v_isSharedCheck_439_ == 0)
{
lean_object* v_unused_440_; 
v_unused_440_ = lean_ctor_get(v_head_423_, 0);
lean_dec(v_unused_440_);
v___x_426_ = v_head_423_;
v_isShared_427_ = v_isSharedCheck_439_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_after_424_);
lean_dec(v_head_423_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_439_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_428_; lean_object* v___x_430_; 
v___x_428_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5___closed__0);
if (v_isShared_427_ == 0)
{
lean_ctor_set_tag(v___x_426_, 7);
lean_ctor_set(v___x_426_, 1, v___x_428_);
lean_ctor_set(v___x_426_, 0, v_msgData_414_);
v___x_430_ = v___x_426_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_msgData_414_);
lean_ctor_set(v_reuseFailAlloc_438_, 1, v___x_428_);
v___x_430_ = v_reuseFailAlloc_438_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v_msgData_435_; lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_431_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___closed__2);
v___x_432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_430_);
lean_ctor_set(v___x_432_, 1, v___x_431_);
v___x_433_ = l_Lean_MessageData_ofSyntax(v_after_424_);
v___x_434_ = l_Lean_indentD(v___x_433_);
v_msgData_435_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_435_, 0, v___x_432_);
lean_ctor_set(v_msgData_435_, 1, v___x_434_);
v___x_436_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3_spec__5(v_msgData_435_, v_macroStack_415_);
v___x_437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_437_, 0, v___x_436_);
return v___x_437_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_441_, lean_object* v_macroStack_442_, lean_object* v___y_443_, lean_object* v___y_444_){
_start:
{
lean_object* v_res_445_; 
v_res_445_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg(v_msgData_441_, v_macroStack_442_, v___y_443_);
lean_dec_ref(v___y_443_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__2(lean_object* v_msgData_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v___x_452_; lean_object* v_env_453_; lean_object* v___x_454_; lean_object* v_mctx_455_; lean_object* v_lctx_456_; lean_object* v_options_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_452_ = lean_st_ref_get(v___y_450_);
v_env_453_ = lean_ctor_get(v___x_452_, 0);
lean_inc_ref(v_env_453_);
lean_dec(v___x_452_);
v___x_454_ = lean_st_ref_get(v___y_448_);
v_mctx_455_ = lean_ctor_get(v___x_454_, 0);
lean_inc_ref(v_mctx_455_);
lean_dec(v___x_454_);
v_lctx_456_ = lean_ctor_get(v___y_447_, 2);
v_options_457_ = lean_ctor_get(v___y_449_, 2);
lean_inc_ref(v_options_457_);
lean_inc_ref(v_lctx_456_);
v___x_458_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_458_, 0, v_env_453_);
lean_ctor_set(v___x_458_, 1, v_mctx_455_);
lean_ctor_set(v___x_458_, 2, v_lctx_456_);
lean_ctor_set(v___x_458_, 3, v_options_457_);
v___x_459_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_459_, 0, v___x_458_);
lean_ctor_set(v___x_459_, 1, v_msgData_446_);
v___x_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__2___boxed(lean_object* v_msgData_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__2(v_msgData_461_, v___y_462_, v___y_463_, v___y_464_, v___y_465_);
lean_dec(v___y_465_);
lean_dec_ref(v___y_464_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg(lean_object* v_msg_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v_ref_476_; lean_object* v___x_477_; lean_object* v_a_478_; lean_object* v_macroStack_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_490_; 
v_ref_476_ = lean_ctor_get(v___y_473_, 5);
v___x_477_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__2(v_msg_468_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
v_a_478_ = lean_ctor_get(v___x_477_, 0);
lean_inc(v_a_478_);
lean_dec_ref(v___x_477_);
v_macroStack_479_ = lean_ctor_get(v___y_469_, 1);
v___x_480_ = l_Lean_Elab_getBetterRef(v_ref_476_, v_macroStack_479_);
lean_inc(v_macroStack_479_);
v___x_481_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg(v_a_478_, v_macroStack_479_, v___y_473_);
v_a_482_ = lean_ctor_get(v___x_481_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_490_ == 0)
{
v___x_484_ = v___x_481_;
v_isShared_485_ = v_isSharedCheck_490_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v___x_481_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_490_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; lean_object* v___x_488_; 
v___x_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_480_);
lean_ctor_set(v___x_486_, 1, v_a_482_);
if (v_isShared_485_ == 0)
{
lean_ctor_set_tag(v___x_484_, 1);
lean_ctor_set(v___x_484_, 0, v___x_486_);
v___x_488_ = v___x_484_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v___x_486_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg___boxed(lean_object* v_msg_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg(v_msg_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_);
lean_dec(v___y_497_);
lean_dec_ref(v___y_496_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
return v_res_499_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__1(void){
_start:
{
lean_object* v___x_501_; lean_object* v___x_502_; 
v___x_501_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__0));
v___x_502_ = l_Lean_stringToMessageData(v___x_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta(lean_object* v_stx_503_, lean_object* v_expectedType_x3f_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_, lean_object* v_a_509_, lean_object* v_a_510_){
_start:
{
lean_object* v___x_512_; uint8_t v___x_513_; 
v___x_512_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_deltaStx___closed__1));
lean_inc(v_stx_503_);
v___x_513_ = l_Lean_Syntax_isOfKind(v_stx_503_, v___x_512_);
if (v___x_513_ == 0)
{
lean_object* v___x_514_; 
lean_dec(v_expectedType_x3f_504_);
lean_dec(v_stx_503_);
v___x_514_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
return v___x_514_;
}
else
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; uint8_t v___x_520_; lean_object* v___x_521_; 
v___x_515_ = lean_unsigned_to_nat(1u);
v___x_516_ = l_Lean_Syntax_getArg(v_stx_503_, v___x_515_);
lean_dec(v_stx_503_);
v___x_517_ = lean_box(v___x_513_);
v___x_518_ = lean_box(v___x_513_);
v___x_519_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTerm___boxed), 11, 4);
lean_closure_set(v___x_519_, 0, v___x_516_);
lean_closure_set(v___x_519_, 1, v_expectedType_x3f_504_);
lean_closure_set(v___x_519_, 2, v___x_517_);
lean_closure_set(v___x_519_, 3, v___x_518_);
v___x_520_ = 2;
v___x_521_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_519_, v___x_520_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_);
if (lean_obj_tag(v___x_521_) == 0)
{
lean_object* v_a_522_; uint8_t v___x_523_; uint8_t v___x_524_; lean_object* v___x_525_; 
v_a_522_ = lean_ctor_get(v___x_521_, 0);
lean_inc(v_a_522_);
lean_dec_ref_known(v___x_521_, 1);
v___x_523_ = 0;
v___x_524_ = 0;
v___x_525_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_523_, v___x_524_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_);
if (lean_obj_tag(v___x_525_) == 0)
{
lean_object* v___x_526_; lean_object* v_a_527_; lean_object* v___x_528_; lean_object* v___f_529_; lean_object* v___x_530_; lean_object* v___f_531_; lean_object* v___x_532_; 
lean_dec_ref_known(v___x_525_, 1);
v___x_526_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(v_a_522_, v_a_508_);
v_a_527_ = lean_ctor_get(v___x_526_, 0);
lean_inc_n(v_a_527_, 2);
lean_dec_ref(v___x_526_);
v___x_528_ = lean_box(v___x_513_);
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__0___boxed), 2, 1);
lean_closure_set(v___f_529_, 0, v___x_528_);
v___x_530_ = lean_box(v___x_524_);
v___f_531_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Util_TermReduce_elabDelta___lam__1___boxed), 10, 3);
lean_closure_set(v___f_531_, 0, v_a_527_);
lean_closure_set(v___f_531_, 1, v___f_529_);
lean_closure_set(v___f_531_, 2, v___x_530_);
v___x_532_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg(v___f_531_, v___x_513_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_545_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_545_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_545_ == 0)
{
v___x_535_ = v___x_532_;
v_isShared_536_ = v_isSharedCheck_545_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_532_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_545_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
if (lean_obj_tag(v_a_533_) == 1)
{
lean_object* v_val_537_; lean_object* v___x_539_; 
lean_dec(v_a_527_);
v_val_537_ = lean_ctor_get(v_a_533_, 0);
lean_inc(v_val_537_);
lean_dec_ref_known(v_a_533_, 1);
if (v_isShared_536_ == 0)
{
lean_ctor_set(v___x_535_, 0, v_val_537_);
v___x_539_ = v___x_535_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v_val_537_);
v___x_539_ = v_reuseFailAlloc_540_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
return v___x_539_;
}
}
else
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
lean_del_object(v___x_535_);
lean_dec(v_a_533_);
v___x_541_ = lean_obj_once(&lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__1, &lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__1_once, _init_lp_mathlib_Mathlib_Util_TermReduce_elabDelta___closed__1);
v___x_542_ = l_Lean_MessageData_ofExpr(v_a_527_);
v___x_543_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_541_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg(v___x_543_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_, v_a_510_);
return v___x_544_;
}
}
}
else
{
lean_object* v_a_546_; lean_object* v___x_548_; uint8_t v_isShared_549_; uint8_t v_isSharedCheck_553_; 
lean_dec(v_a_527_);
v_a_546_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_553_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_553_ == 0)
{
v___x_548_ = v___x_532_;
v_isShared_549_ = v_isSharedCheck_553_;
goto v_resetjp_547_;
}
else
{
lean_inc(v_a_546_);
lean_dec(v___x_532_);
v___x_548_ = lean_box(0);
v_isShared_549_ = v_isSharedCheck_553_;
goto v_resetjp_547_;
}
v_resetjp_547_:
{
lean_object* v___x_551_; 
if (v_isShared_549_ == 0)
{
v___x_551_ = v___x_548_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v_a_546_);
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
else
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_561_; 
lean_dec(v_a_522_);
v_a_554_ = lean_ctor_get(v___x_525_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_525_);
if (v_isSharedCheck_561_ == 0)
{
v___x_556_ = v___x_525_;
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_525_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___x_559_; 
if (v_isShared_557_ == 0)
{
v___x_559_ = v___x_556_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v_a_554_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
}
}
else
{
return v___x_521_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabDelta___boxed(lean_object* v_stx_562_, lean_object* v_expectedType_x3f_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib_Mathlib_Util_TermReduce_elabDelta(v_stx_562_, v_expectedType_x3f_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_, v_a_569_);
lean_dec(v_a_569_);
lean_dec_ref(v_a_568_);
lean_dec(v_a_567_);
lean_dec_ref(v_a_566_);
lean_dec(v_a_565_);
lean_dec_ref(v_a_564_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0(lean_object* v_00_u03b1_572_, lean_object* v_x_573_, uint8_t v_isExporting_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_){
_start:
{
lean_object* v___x_582_; 
v___x_582_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___redArg(v_x_573_, v_isExporting_574_, v___y_575_, v___y_576_, v___y_577_, v___y_578_, v___y_579_, v___y_580_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0___boxed(lean_object* v_00_u03b1_583_, lean_object* v_x_584_, lean_object* v_isExporting_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_){
_start:
{
uint8_t v_isExporting_boxed_593_; lean_object* v_res_594_; 
v_isExporting_boxed_593_ = lean_unbox(v_isExporting_585_);
v_res_594_ = lp_mathlib_Lean_withExporting___at___00Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0_spec__0(v_00_u03b1_583_, v_x_584_, v_isExporting_boxed_593_, v___y_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
lean_dec(v___y_589_);
lean_dec_ref(v___y_588_);
lean_dec(v___y_587_);
lean_dec_ref(v___y_586_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0(lean_object* v_00_u03b1_595_, lean_object* v_x_596_, uint8_t v_when_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
lean_object* v___x_605_; 
v___x_605_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___redArg(v_x_596_, v_when_597_, v___y_598_, v___y_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0___boxed(lean_object* v_00_u03b1_606_, lean_object* v_x_607_, lean_object* v_when_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
uint8_t v_when_boxed_616_; lean_object* v_res_617_; 
v_when_boxed_616_ = lean_unbox(v_when_608_);
v_res_617_ = lp_mathlib_Lean_withoutExporting___at___00Mathlib_Util_TermReduce_elabDelta_spec__0(v_00_u03b1_606_, v_x_607_, v_when_boxed_616_, v___y_609_, v___y_610_, v___y_611_, v___y_612_, v___y_613_, v___y_614_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
lean_dec(v___y_612_);
lean_dec_ref(v___y_611_);
lean_dec(v___y_610_);
lean_dec_ref(v___y_609_);
return v_res_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1(lean_object* v_00_u03b1_618_, lean_object* v_msg_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___redArg(v_msg_619_, v___y_620_, v___y_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1___boxed(lean_object* v_00_u03b1_628_, lean_object* v_msg_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1(v_00_u03b1_628_, v_msg_629_, v___y_630_, v___y_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_);
lean_dec(v___y_635_);
lean_dec_ref(v___y_634_);
lean_dec(v___y_633_);
lean_dec_ref(v___y_632_);
lean_dec(v___y_631_);
lean_dec_ref(v___y_630_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3(lean_object* v_msgData_638_, lean_object* v_macroStack_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___redArg(v_msgData_638_, v_macroStack_639_, v___y_644_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3___boxed(lean_object* v_msgData_648_, lean_object* v_macroStack_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Util_TermReduce_elabDelta_spec__1_spec__3(v_msgData_648_, v_macroStack_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabZeta(lean_object* v_stx_676_, lean_object* v_expectedType_x3f_677_, lean_object* v_a_678_, lean_object* v_a_679_, lean_object* v_a_680_, lean_object* v_a_681_, lean_object* v_a_682_, lean_object* v_a_683_){
_start:
{
lean_object* v___x_685_; uint8_t v___x_686_; 
v___x_685_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_zetaStx___closed__1));
lean_inc(v_stx_676_);
v___x_686_ = l_Lean_Syntax_isOfKind(v_stx_676_, v___x_685_);
if (v___x_686_ == 0)
{
lean_object* v___x_687_; 
lean_dec(v_expectedType_x3f_677_);
lean_dec(v_stx_676_);
v___x_687_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
return v___x_687_;
}
else
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; uint8_t v___x_693_; lean_object* v___x_694_; 
v___x_688_ = lean_unsigned_to_nat(1u);
v___x_689_ = l_Lean_Syntax_getArg(v_stx_676_, v___x_688_);
lean_dec(v_stx_676_);
v___x_690_ = lean_box(v___x_686_);
v___x_691_ = lean_box(v___x_686_);
v___x_692_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTerm___boxed), 11, 4);
lean_closure_set(v___x_692_, 0, v___x_689_);
lean_closure_set(v___x_692_, 1, v_expectedType_x3f_677_);
lean_closure_set(v___x_692_, 2, v___x_690_);
lean_closure_set(v___x_692_, 3, v___x_691_);
v___x_693_ = 2;
v___x_694_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_692_, v___x_693_, v_a_678_, v_a_679_, v_a_680_, v_a_681_, v_a_682_, v_a_683_);
if (lean_obj_tag(v___x_694_) == 0)
{
lean_object* v_a_695_; uint8_t v___x_696_; uint8_t v___x_697_; lean_object* v___x_698_; 
v_a_695_ = lean_ctor_get(v___x_694_, 0);
lean_inc(v_a_695_);
lean_dec_ref_known(v___x_694_, 1);
v___x_696_ = 0;
v___x_697_ = 0;
v___x_698_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_696_, v___x_697_, v_a_678_, v_a_679_, v_a_680_, v_a_681_, v_a_682_, v_a_683_);
if (lean_obj_tag(v___x_698_) == 0)
{
lean_object* v___x_699_; lean_object* v_a_700_; lean_object* v___x_701_; 
lean_dec_ref_known(v___x_698_, 1);
v___x_699_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(v_a_695_, v_a_681_);
v_a_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_a_700_);
lean_dec_ref(v___x_699_);
v___x_701_ = l_Lean_Meta_zetaReduce(v_a_700_, v___x_686_, v___x_686_, v___x_686_, v_a_680_, v_a_681_, v_a_682_, v_a_683_);
return v___x_701_;
}
else
{
lean_object* v_a_702_; lean_object* v___x_704_; uint8_t v_isShared_705_; uint8_t v_isSharedCheck_709_; 
lean_dec(v_a_695_);
v_a_702_ = lean_ctor_get(v___x_698_, 0);
v_isSharedCheck_709_ = !lean_is_exclusive(v___x_698_);
if (v_isSharedCheck_709_ == 0)
{
v___x_704_ = v___x_698_;
v_isShared_705_ = v_isSharedCheck_709_;
goto v_resetjp_703_;
}
else
{
lean_inc(v_a_702_);
lean_dec(v___x_698_);
v___x_704_ = lean_box(0);
v_isShared_705_ = v_isSharedCheck_709_;
goto v_resetjp_703_;
}
v_resetjp_703_:
{
lean_object* v___x_707_; 
if (v_isShared_705_ == 0)
{
v___x_707_ = v___x_704_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v_a_702_);
v___x_707_ = v_reuseFailAlloc_708_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
return v___x_707_;
}
}
}
}
else
{
return v___x_694_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabZeta___boxed(lean_object* v_stx_710_, lean_object* v_expectedType_x3f_711_, lean_object* v_a_712_, lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_, lean_object* v_a_716_, lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_mathlib_Mathlib_Util_TermReduce_elabZeta(v_stx_710_, v_expectedType_x3f_711_, v_a_712_, v_a_713_, v_a_714_, v_a_715_, v_a_716_, v_a_717_);
lean_dec(v_a_717_);
lean_dec_ref(v_a_716_);
lean_dec(v_a_715_);
lean_dec_ref(v_a_714_);
lean_dec(v_a_713_);
lean_dec_ref(v_a_712_);
return v_res_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0(lean_object* v_x_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
lean_object* v___x_748_; lean_object* v___x_749_; 
v___x_748_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___closed__0));
v___x_749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_749_, 0, v___x_748_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0___boxed(lean_object* v_x_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_){
_start:
{
lean_object* v_res_758_; 
v_res_758_ = lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__0(v_x_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_);
lean_dec(v___y_756_);
lean_dec_ref(v___y_755_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec_ref(v_x_750_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__1(lean_object* v_e_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_){
_start:
{
lean_object* v___x_767_; 
v___x_767_ = lp_mathlib_Lean_Expr_reduceProjStruct_x3f(v_e_759_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
if (lean_obj_tag(v___x_767_) == 0)
{
lean_object* v_a_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_776_; 
v_a_768_ = lean_ctor_get(v___x_767_, 0);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_767_);
if (v_isSharedCheck_776_ == 0)
{
v___x_770_ = v___x_767_;
v_isShared_771_ = v_isSharedCheck_776_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_a_768_);
lean_dec(v___x_767_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_776_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_772_; lean_object* v___x_774_; 
v___x_772_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_772_, 0, v_a_768_);
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 0, v___x_772_);
v___x_774_ = v___x_770_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v___x_772_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
return v___x_774_;
}
}
}
else
{
lean_object* v_a_777_; lean_object* v___x_779_; uint8_t v_isShared_780_; uint8_t v_isSharedCheck_784_; 
v_a_777_ = lean_ctor_get(v___x_767_, 0);
v_isSharedCheck_784_ = !lean_is_exclusive(v___x_767_);
if (v_isSharedCheck_784_ == 0)
{
v___x_779_ = v___x_767_;
v_isShared_780_ = v_isSharedCheck_784_;
goto v_resetjp_778_;
}
else
{
lean_inc(v_a_777_);
lean_dec(v___x_767_);
v___x_779_ = lean_box(0);
v_isShared_780_ = v_isSharedCheck_784_;
goto v_resetjp_778_;
}
v_resetjp_778_:
{
lean_object* v___x_782_; 
if (v_isShared_780_ == 0)
{
v___x_782_ = v___x_779_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v_a_777_);
v___x_782_ = v_reuseFailAlloc_783_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
return v___x_782_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__1___boxed(lean_object* v_e_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___lam__1(v_e_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_);
lean_dec(v___y_791_);
lean_dec_ref(v___y_790_);
lean_dec(v___y_789_);
lean_dec_ref(v___y_788_);
lean_dec(v___y_787_);
lean_dec_ref(v___y_786_);
return v_res_793_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg(lean_object* v_a_794_, lean_object* v_x_795_){
_start:
{
if (lean_obj_tag(v_x_795_) == 0)
{
uint8_t v___x_796_; 
v___x_796_ = 0;
return v___x_796_;
}
else
{
lean_object* v_key_797_; lean_object* v_tail_798_; uint8_t v___x_799_; 
v_key_797_ = lean_ctor_get(v_x_795_, 0);
v_tail_798_ = lean_ctor_get(v_x_795_, 2);
v___x_799_ = l_Lean_ExprStructEq_beq(v_key_797_, v_a_794_);
if (v___x_799_ == 0)
{
v_x_795_ = v_tail_798_;
goto _start;
}
else
{
return v___x_799_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg___boxed(lean_object* v_a_801_, lean_object* v_x_802_){
_start:
{
uint8_t v_res_803_; lean_object* v_r_804_; 
v_res_803_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg(v_a_801_, v_x_802_);
lean_dec(v_x_802_);
lean_dec_ref(v_a_801_);
v_r_804_ = lean_box(v_res_803_);
return v_r_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12___redArg(lean_object* v_a_805_, lean_object* v_b_806_, lean_object* v_x_807_){
_start:
{
if (lean_obj_tag(v_x_807_) == 0)
{
lean_dec(v_b_806_);
lean_dec_ref(v_a_805_);
return v_x_807_;
}
else
{
lean_object* v_key_808_; lean_object* v_value_809_; lean_object* v_tail_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_822_; 
v_key_808_ = lean_ctor_get(v_x_807_, 0);
v_value_809_ = lean_ctor_get(v_x_807_, 1);
v_tail_810_ = lean_ctor_get(v_x_807_, 2);
v_isSharedCheck_822_ = !lean_is_exclusive(v_x_807_);
if (v_isSharedCheck_822_ == 0)
{
v___x_812_ = v_x_807_;
v_isShared_813_ = v_isSharedCheck_822_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_tail_810_);
lean_inc(v_value_809_);
lean_inc(v_key_808_);
lean_dec(v_x_807_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_822_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
uint8_t v___x_814_; 
v___x_814_ = l_Lean_ExprStructEq_beq(v_key_808_, v_a_805_);
if (v___x_814_ == 0)
{
lean_object* v___x_815_; lean_object* v___x_817_; 
v___x_815_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12___redArg(v_a_805_, v_b_806_, v_tail_810_);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 2, v___x_815_);
v___x_817_ = v___x_812_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v_key_808_);
lean_ctor_set(v_reuseFailAlloc_818_, 1, v_value_809_);
lean_ctor_set(v_reuseFailAlloc_818_, 2, v___x_815_);
v___x_817_ = v_reuseFailAlloc_818_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
return v___x_817_;
}
}
else
{
lean_object* v___x_820_; 
lean_dec(v_value_809_);
lean_dec(v_key_808_);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 1, v_b_806_);
lean_ctor_set(v___x_812_, 0, v_a_805_);
v___x_820_ = v___x_812_;
goto v_reusejp_819_;
}
else
{
lean_object* v_reuseFailAlloc_821_; 
v_reuseFailAlloc_821_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_821_, 0, v_a_805_);
lean_ctor_set(v_reuseFailAlloc_821_, 1, v_b_806_);
lean_ctor_set(v_reuseFailAlloc_821_, 2, v_tail_810_);
v___x_820_ = v_reuseFailAlloc_821_;
goto v_reusejp_819_;
}
v_reusejp_819_:
{
return v___x_820_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12_spec__13___redArg(lean_object* v_x_823_, lean_object* v_x_824_){
_start:
{
if (lean_obj_tag(v_x_824_) == 0)
{
return v_x_823_;
}
else
{
lean_object* v_key_825_; lean_object* v_value_826_; lean_object* v_tail_827_; lean_object* v___x_829_; uint8_t v_isShared_830_; uint8_t v_isSharedCheck_850_; 
v_key_825_ = lean_ctor_get(v_x_824_, 0);
v_value_826_ = lean_ctor_get(v_x_824_, 1);
v_tail_827_ = lean_ctor_get(v_x_824_, 2);
v_isSharedCheck_850_ = !lean_is_exclusive(v_x_824_);
if (v_isSharedCheck_850_ == 0)
{
v___x_829_ = v_x_824_;
v_isShared_830_ = v_isSharedCheck_850_;
goto v_resetjp_828_;
}
else
{
lean_inc(v_tail_827_);
lean_inc(v_value_826_);
lean_inc(v_key_825_);
lean_dec(v_x_824_);
v___x_829_ = lean_box(0);
v_isShared_830_ = v_isSharedCheck_850_;
goto v_resetjp_828_;
}
v_resetjp_828_:
{
lean_object* v___x_831_; uint64_t v___x_832_; uint64_t v___x_833_; uint64_t v___x_834_; uint64_t v_fold_835_; uint64_t v___x_836_; uint64_t v___x_837_; uint64_t v___x_838_; size_t v___x_839_; size_t v___x_840_; size_t v___x_841_; size_t v___x_842_; size_t v___x_843_; lean_object* v___x_844_; lean_object* v___x_846_; 
v___x_831_ = lean_array_get_size(v_x_823_);
v___x_832_ = l_Lean_ExprStructEq_hash(v_key_825_);
v___x_833_ = 32ULL;
v___x_834_ = lean_uint64_shift_right(v___x_832_, v___x_833_);
v_fold_835_ = lean_uint64_xor(v___x_832_, v___x_834_);
v___x_836_ = 16ULL;
v___x_837_ = lean_uint64_shift_right(v_fold_835_, v___x_836_);
v___x_838_ = lean_uint64_xor(v_fold_835_, v___x_837_);
v___x_839_ = lean_uint64_to_usize(v___x_838_);
v___x_840_ = lean_usize_of_nat(v___x_831_);
v___x_841_ = ((size_t)1ULL);
v___x_842_ = lean_usize_sub(v___x_840_, v___x_841_);
v___x_843_ = lean_usize_land(v___x_839_, v___x_842_);
v___x_844_ = lean_array_uget_borrowed(v_x_823_, v___x_843_);
lean_inc(v___x_844_);
if (v_isShared_830_ == 0)
{
lean_ctor_set(v___x_829_, 2, v___x_844_);
v___x_846_ = v___x_829_;
goto v_reusejp_845_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v_key_825_);
lean_ctor_set(v_reuseFailAlloc_849_, 1, v_value_826_);
lean_ctor_set(v_reuseFailAlloc_849_, 2, v___x_844_);
v___x_846_ = v_reuseFailAlloc_849_;
goto v_reusejp_845_;
}
v_reusejp_845_:
{
lean_object* v___x_847_; 
v___x_847_ = lean_array_uset(v_x_823_, v___x_843_, v___x_846_);
v_x_823_ = v___x_847_;
v_x_824_ = v_tail_827_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12___redArg(lean_object* v_i_851_, lean_object* v_source_852_, lean_object* v_target_853_){
_start:
{
lean_object* v___x_854_; uint8_t v___x_855_; 
v___x_854_ = lean_array_get_size(v_source_852_);
v___x_855_ = lean_nat_dec_lt(v_i_851_, v___x_854_);
if (v___x_855_ == 0)
{
lean_dec_ref(v_source_852_);
lean_dec(v_i_851_);
return v_target_853_;
}
else
{
lean_object* v_es_856_; lean_object* v___x_857_; lean_object* v_source_858_; lean_object* v_target_859_; lean_object* v___x_860_; lean_object* v___x_861_; 
v_es_856_ = lean_array_fget(v_source_852_, v_i_851_);
v___x_857_ = lean_box(0);
v_source_858_ = lean_array_fset(v_source_852_, v_i_851_, v___x_857_);
v_target_859_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12_spec__13___redArg(v_target_853_, v_es_856_);
v___x_860_ = lean_unsigned_to_nat(1u);
v___x_861_ = lean_nat_add(v_i_851_, v___x_860_);
lean_dec(v_i_851_);
v_i_851_ = v___x_861_;
v_source_852_ = v_source_858_;
v_target_853_ = v_target_859_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11___redArg(lean_object* v_data_863_){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v_nbuckets_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_864_ = lean_array_get_size(v_data_863_);
v___x_865_ = lean_unsigned_to_nat(2u);
v_nbuckets_866_ = lean_nat_mul(v___x_864_, v___x_865_);
v___x_867_ = lean_unsigned_to_nat(0u);
v___x_868_ = lean_box(0);
v___x_869_ = lean_mk_array(v_nbuckets_866_, v___x_868_);
v___x_870_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12___redArg(v___x_867_, v_data_863_, v___x_869_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6___redArg(lean_object* v_m_871_, lean_object* v_a_872_, lean_object* v_b_873_){
_start:
{
lean_object* v_size_874_; lean_object* v_buckets_875_; lean_object* v___x_877_; uint8_t v_isShared_878_; uint8_t v_isSharedCheck_918_; 
v_size_874_ = lean_ctor_get(v_m_871_, 0);
v_buckets_875_ = lean_ctor_get(v_m_871_, 1);
v_isSharedCheck_918_ = !lean_is_exclusive(v_m_871_);
if (v_isSharedCheck_918_ == 0)
{
v___x_877_ = v_m_871_;
v_isShared_878_ = v_isSharedCheck_918_;
goto v_resetjp_876_;
}
else
{
lean_inc(v_buckets_875_);
lean_inc(v_size_874_);
lean_dec(v_m_871_);
v___x_877_ = lean_box(0);
v_isShared_878_ = v_isSharedCheck_918_;
goto v_resetjp_876_;
}
v_resetjp_876_:
{
lean_object* v___x_879_; uint64_t v___x_880_; uint64_t v___x_881_; uint64_t v___x_882_; uint64_t v_fold_883_; uint64_t v___x_884_; uint64_t v___x_885_; uint64_t v___x_886_; size_t v___x_887_; size_t v___x_888_; size_t v___x_889_; size_t v___x_890_; size_t v___x_891_; lean_object* v_bkt_892_; uint8_t v___x_893_; 
v___x_879_ = lean_array_get_size(v_buckets_875_);
v___x_880_ = l_Lean_ExprStructEq_hash(v_a_872_);
v___x_881_ = 32ULL;
v___x_882_ = lean_uint64_shift_right(v___x_880_, v___x_881_);
v_fold_883_ = lean_uint64_xor(v___x_880_, v___x_882_);
v___x_884_ = 16ULL;
v___x_885_ = lean_uint64_shift_right(v_fold_883_, v___x_884_);
v___x_886_ = lean_uint64_xor(v_fold_883_, v___x_885_);
v___x_887_ = lean_uint64_to_usize(v___x_886_);
v___x_888_ = lean_usize_of_nat(v___x_879_);
v___x_889_ = ((size_t)1ULL);
v___x_890_ = lean_usize_sub(v___x_888_, v___x_889_);
v___x_891_ = lean_usize_land(v___x_887_, v___x_890_);
v_bkt_892_ = lean_array_uget_borrowed(v_buckets_875_, v___x_891_);
v___x_893_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg(v_a_872_, v_bkt_892_);
if (v___x_893_ == 0)
{
lean_object* v___x_894_; lean_object* v_size_x27_895_; lean_object* v___x_896_; lean_object* v_buckets_x27_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; uint8_t v___x_903_; 
v___x_894_ = lean_unsigned_to_nat(1u);
v_size_x27_895_ = lean_nat_add(v_size_874_, v___x_894_);
lean_dec(v_size_874_);
lean_inc(v_bkt_892_);
v___x_896_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_896_, 0, v_a_872_);
lean_ctor_set(v___x_896_, 1, v_b_873_);
lean_ctor_set(v___x_896_, 2, v_bkt_892_);
v_buckets_x27_897_ = lean_array_uset(v_buckets_875_, v___x_891_, v___x_896_);
v___x_898_ = lean_unsigned_to_nat(4u);
v___x_899_ = lean_nat_mul(v_size_x27_895_, v___x_898_);
v___x_900_ = lean_unsigned_to_nat(3u);
v___x_901_ = lean_nat_div(v___x_899_, v___x_900_);
lean_dec(v___x_899_);
v___x_902_ = lean_array_get_size(v_buckets_x27_897_);
v___x_903_ = lean_nat_dec_le(v___x_901_, v___x_902_);
lean_dec(v___x_901_);
if (v___x_903_ == 0)
{
lean_object* v_val_904_; lean_object* v___x_906_; 
v_val_904_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11___redArg(v_buckets_x27_897_);
if (v_isShared_878_ == 0)
{
lean_ctor_set(v___x_877_, 1, v_val_904_);
lean_ctor_set(v___x_877_, 0, v_size_x27_895_);
v___x_906_ = v___x_877_;
goto v_reusejp_905_;
}
else
{
lean_object* v_reuseFailAlloc_907_; 
v_reuseFailAlloc_907_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_907_, 0, v_size_x27_895_);
lean_ctor_set(v_reuseFailAlloc_907_, 1, v_val_904_);
v___x_906_ = v_reuseFailAlloc_907_;
goto v_reusejp_905_;
}
v_reusejp_905_:
{
return v___x_906_;
}
}
else
{
lean_object* v___x_909_; 
if (v_isShared_878_ == 0)
{
lean_ctor_set(v___x_877_, 1, v_buckets_x27_897_);
lean_ctor_set(v___x_877_, 0, v_size_x27_895_);
v___x_909_ = v___x_877_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_910_; 
v_reuseFailAlloc_910_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_910_, 0, v_size_x27_895_);
lean_ctor_set(v_reuseFailAlloc_910_, 1, v_buckets_x27_897_);
v___x_909_ = v_reuseFailAlloc_910_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
return v___x_909_;
}
}
}
else
{
lean_object* v___x_911_; lean_object* v_buckets_x27_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_916_; 
lean_inc(v_bkt_892_);
v___x_911_ = lean_box(0);
v_buckets_x27_912_ = lean_array_uset(v_buckets_875_, v___x_891_, v___x_911_);
v___x_913_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12___redArg(v_a_872_, v_b_873_, v_bkt_892_);
v___x_914_ = lean_array_uset(v_buckets_x27_912_, v___x_891_, v___x_913_);
if (v_isShared_878_ == 0)
{
lean_ctor_set(v___x_877_, 1, v___x_914_);
v___x_916_ = v___x_877_;
goto v_reusejp_915_;
}
else
{
lean_object* v_reuseFailAlloc_917_; 
v_reuseFailAlloc_917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_917_, 0, v_size_874_);
lean_ctor_set(v_reuseFailAlloc_917_, 1, v___x_914_);
v___x_916_ = v_reuseFailAlloc_917_;
goto v_reusejp_915_;
}
v_reusejp_915_:
{
return v___x_916_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__2(lean_object* v_a_919_, lean_object* v_e_920_, lean_object* v_a_921_){
_start:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; 
v___x_923_ = lean_st_ref_take(v_a_919_);
v___x_924_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6___redArg(v___x_923_, v_e_920_, v_a_921_);
v___x_925_ = lean_st_ref_set(v_a_919_, v___x_924_);
v___x_926_ = lean_box(0);
return v___x_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__2___boxed(lean_object* v_a_927_, lean_object* v_e_928_, lean_object* v_a_929_, lean_object* v___y_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__2(v_a_927_, v_e_928_, v_a_929_);
lean_dec(v_a_927_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg(lean_object* v_a_932_, lean_object* v_x_933_){
_start:
{
if (lean_obj_tag(v_x_933_) == 0)
{
lean_object* v___x_934_; 
v___x_934_ = lean_box(0);
return v___x_934_;
}
else
{
lean_object* v_key_935_; lean_object* v_value_936_; lean_object* v_tail_937_; uint8_t v___x_938_; 
v_key_935_ = lean_ctor_get(v_x_933_, 0);
v_value_936_ = lean_ctor_get(v_x_933_, 1);
v_tail_937_ = lean_ctor_get(v_x_933_, 2);
v___x_938_ = l_Lean_ExprStructEq_beq(v_key_935_, v_a_932_);
if (v___x_938_ == 0)
{
v_x_933_ = v_tail_937_;
goto _start;
}
else
{
lean_object* v___x_940_; 
lean_inc(v_value_936_);
v___x_940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_940_, 0, v_value_936_);
return v___x_940_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object* v_a_941_, lean_object* v_x_942_){
_start:
{
lean_object* v_res_943_; 
v_res_943_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg(v_a_941_, v_x_942_);
lean_dec(v_x_942_);
lean_dec_ref(v_a_941_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg(lean_object* v_m_944_, lean_object* v_a_945_){
_start:
{
lean_object* v_buckets_946_; lean_object* v___x_947_; uint64_t v___x_948_; uint64_t v___x_949_; uint64_t v___x_950_; uint64_t v_fold_951_; uint64_t v___x_952_; uint64_t v___x_953_; uint64_t v___x_954_; size_t v___x_955_; size_t v___x_956_; size_t v___x_957_; size_t v___x_958_; size_t v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; 
v_buckets_946_ = lean_ctor_get(v_m_944_, 1);
v___x_947_ = lean_array_get_size(v_buckets_946_);
v___x_948_ = l_Lean_ExprStructEq_hash(v_a_945_);
v___x_949_ = 32ULL;
v___x_950_ = lean_uint64_shift_right(v___x_948_, v___x_949_);
v_fold_951_ = lean_uint64_xor(v___x_948_, v___x_950_);
v___x_952_ = 16ULL;
v___x_953_ = lean_uint64_shift_right(v_fold_951_, v___x_952_);
v___x_954_ = lean_uint64_xor(v_fold_951_, v___x_953_);
v___x_955_ = lean_uint64_to_usize(v___x_954_);
v___x_956_ = lean_usize_of_nat(v___x_947_);
v___x_957_ = ((size_t)1ULL);
v___x_958_ = lean_usize_sub(v___x_956_, v___x_957_);
v___x_959_ = lean_usize_land(v___x_955_, v___x_958_);
v___x_960_ = lean_array_uget_borrowed(v_buckets_946_, v___x_959_);
v___x_961_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg(v_a_945_, v___x_960_);
return v___x_961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_m_962_, lean_object* v_a_963_){
_start:
{
lean_object* v_res_964_; 
v_res_964_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg(v_m_962_, v_a_963_);
lean_dec_ref(v_a_963_);
lean_dec_ref(v_m_962_);
return v_res_964_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0(lean_object* v_00_u03b1_965_, lean_object* v_x_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_){
_start:
{
lean_object* v___x_974_; lean_object* v___x_975_; 
v___x_974_ = lean_apply_1(v_x_966_, lean_box(0));
v___x_975_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_975_, 0, v___x_974_);
return v___x_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0___boxed(lean_object* v_00_u03b1_976_, lean_object* v_x_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
lean_object* v_res_985_; 
v_res_985_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0(v_00_u03b1_976_, v_x_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_);
lean_dec(v___y_983_);
lean_dec_ref(v___y_982_);
lean_dec(v___y_981_);
lean_dec_ref(v___y_980_);
lean_dec(v___y_979_);
lean_dec_ref(v___y_978_);
return v_res_985_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___closed__0(void){
_start:
{
lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_986_ = lean_box(0);
v___x_987_ = l_Lean_interruptExceptionId;
v___x_988_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_988_, 0, v___x_987_);
lean_ctor_set(v___x_988_, 1, v___x_986_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg(){
_start:
{
lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_990_ = lean_obj_once(&lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___closed__0, &lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___closed__0_once, _init_lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___closed__0);
v___x_991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_991_, 0, v___x_990_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg___boxed(lean_object* v___y_992_){
_start:
{
lean_object* v_res_993_; 
v_res_993_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg();
return v_res_993_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__3(void){
_start:
{
lean_object* v___x_999_; lean_object* v___x_1000_; 
v___x_999_ = l_Lean_maxRecDepthErrorMessage;
v___x_1000_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1000_, 0, v___x_999_);
return v___x_1000_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__4(void){
_start:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_1001_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__3);
v___x_1002_ = l_Lean_MessageData_ofFormat(v___x_1001_);
return v___x_1002_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__5(void){
_start:
{
lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; 
v___x_1003_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__4);
v___x_1004_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__2));
v___x_1005_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1005_, 0, v___x_1004_);
lean_ctor_set(v___x_1005_, 1, v___x_1003_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg(lean_object* v_ref_1006_){
_start:
{
lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v___x_1008_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___closed__5);
v___x_1009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1009_, 0, v_ref_1006_);
lean_ctor_set(v___x_1009_, 1, v___x_1008_);
v___x_1010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1009_);
return v___x_1010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg___boxed(lean_object* v_ref_1011_, lean_object* v___y_1012_){
_start:
{
lean_object* v_res_1013_; 
v_res_1013_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg(v_ref_1011_);
return v_res_1013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg(lean_object* v_x_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_){
_start:
{
lean_object* v___y_1024_; lean_object* v___y_1034_; lean_object* v___y_1035_; lean_object* v___y_1036_; lean_object* v___y_1037_; lean_object* v___y_1038_; lean_object* v___y_1039_; lean_object* v___y_1040_; lean_object* v___y_1041_; uint8_t v___y_1042_; uint8_t v___y_1043_; lean_object* v___y_1044_; lean_object* v___y_1045_; lean_object* v___y_1046_; lean_object* v___y_1047_; lean_object* v___y_1048_; lean_object* v___y_1049_; lean_object* v_fileName_1054_; lean_object* v_fileMap_1055_; lean_object* v_options_1056_; lean_object* v_currRecDepth_1057_; lean_object* v_maxRecDepth_1058_; lean_object* v_ref_1059_; lean_object* v_currNamespace_1060_; lean_object* v_openDecls_1061_; lean_object* v_initHeartbeats_1062_; lean_object* v_maxHeartbeats_1063_; lean_object* v_quotContext_1064_; lean_object* v_currMacroScope_1065_; uint8_t v_diag_1066_; lean_object* v_cancelTk_x3f_1067_; uint8_t v_suppressElabErrors_1068_; lean_object* v_inheritedTraceOptions_1069_; 
v_fileName_1054_ = lean_ctor_get(v___y_1020_, 0);
v_fileMap_1055_ = lean_ctor_get(v___y_1020_, 1);
v_options_1056_ = lean_ctor_get(v___y_1020_, 2);
v_currRecDepth_1057_ = lean_ctor_get(v___y_1020_, 3);
v_maxRecDepth_1058_ = lean_ctor_get(v___y_1020_, 4);
v_ref_1059_ = lean_ctor_get(v___y_1020_, 5);
v_currNamespace_1060_ = lean_ctor_get(v___y_1020_, 6);
v_openDecls_1061_ = lean_ctor_get(v___y_1020_, 7);
v_initHeartbeats_1062_ = lean_ctor_get(v___y_1020_, 8);
v_maxHeartbeats_1063_ = lean_ctor_get(v___y_1020_, 9);
v_quotContext_1064_ = lean_ctor_get(v___y_1020_, 10);
v_currMacroScope_1065_ = lean_ctor_get(v___y_1020_, 11);
v_diag_1066_ = lean_ctor_get_uint8(v___y_1020_, sizeof(void*)*14);
v_cancelTk_x3f_1067_ = lean_ctor_get(v___y_1020_, 12);
v_suppressElabErrors_1068_ = lean_ctor_get_uint8(v___y_1020_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1069_ = lean_ctor_get(v___y_1020_, 13);
if (lean_obj_tag(v_cancelTk_x3f_1067_) == 1)
{
lean_object* v_val_1075_; uint8_t v___x_1076_; 
v_val_1075_ = lean_ctor_get(v_cancelTk_x3f_1067_, 0);
v___x_1076_ = l_IO_CancelToken_isSet(v_val_1075_);
if (v___x_1076_ == 0)
{
goto v___jp_1070_;
}
else
{
lean_object* v___x_1077_; lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
lean_dec_ref(v_x_1014_);
v___x_1077_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg();
v_a_1078_ = lean_ctor_get(v___x_1077_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_1077_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1077_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1078_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
return v___x_1083_;
}
}
}
}
else
{
goto v___jp_1070_;
}
v___jp_1023_:
{
if (lean_obj_tag(v___y_1024_) == 0)
{
return v___y_1024_;
}
else
{
lean_object* v_a_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1032_; 
v_a_1025_ = lean_ctor_get(v___y_1024_, 0);
v_isSharedCheck_1032_ = !lean_is_exclusive(v___y_1024_);
if (v_isSharedCheck_1032_ == 0)
{
v___x_1027_ = v___y_1024_;
v_isShared_1028_ = v_isSharedCheck_1032_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_a_1025_);
lean_dec(v___y_1024_);
v___x_1027_ = lean_box(0);
v_isShared_1028_ = v_isSharedCheck_1032_;
goto v_resetjp_1026_;
}
v_resetjp_1026_:
{
lean_object* v___x_1030_; 
if (v_isShared_1028_ == 0)
{
v___x_1030_ = v___x_1027_;
goto v_reusejp_1029_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v_a_1025_);
v___x_1030_ = v_reuseFailAlloc_1031_;
goto v_reusejp_1029_;
}
v_reusejp_1029_:
{
return v___x_1030_;
}
}
}
}
v___jp_1033_:
{
lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1050_ = lean_unsigned_to_nat(1u);
v___x_1051_ = lean_nat_add(v___y_1041_, v___x_1050_);
lean_inc_ref(v___y_1047_);
lean_inc(v___y_1036_);
lean_inc(v___y_1046_);
lean_inc(v___y_1035_);
lean_inc(v___y_1049_);
lean_inc(v___y_1034_);
lean_inc(v___y_1048_);
lean_inc(v___y_1040_);
lean_inc(v___y_1039_);
lean_inc_ref(v___y_1045_);
lean_inc_ref(v___y_1044_);
lean_inc_ref(v___y_1037_);
v___x_1052_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1052_, 0, v___y_1037_);
lean_ctor_set(v___x_1052_, 1, v___y_1044_);
lean_ctor_set(v___x_1052_, 2, v___y_1045_);
lean_ctor_set(v___x_1052_, 3, v___x_1051_);
lean_ctor_set(v___x_1052_, 4, v___y_1039_);
lean_ctor_set(v___x_1052_, 5, v___y_1038_);
lean_ctor_set(v___x_1052_, 6, v___y_1040_);
lean_ctor_set(v___x_1052_, 7, v___y_1048_);
lean_ctor_set(v___x_1052_, 8, v___y_1034_);
lean_ctor_set(v___x_1052_, 9, v___y_1049_);
lean_ctor_set(v___x_1052_, 10, v___y_1035_);
lean_ctor_set(v___x_1052_, 11, v___y_1046_);
lean_ctor_set(v___x_1052_, 12, v___y_1036_);
lean_ctor_set(v___x_1052_, 13, v___y_1047_);
lean_ctor_set_uint8(v___x_1052_, sizeof(void*)*14, v___y_1042_);
lean_ctor_set_uint8(v___x_1052_, sizeof(void*)*14 + 1, v___y_1043_);
lean_inc(v___y_1021_);
lean_inc(v___y_1019_);
lean_inc_ref(v___y_1018_);
lean_inc(v___y_1017_);
lean_inc_ref(v___y_1016_);
lean_inc(v___y_1015_);
v___x_1053_ = lean_apply_8(v_x_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___x_1052_, v___y_1021_, lean_box(0));
v___y_1024_ = v___x_1053_;
goto v___jp_1023_;
}
v___jp_1070_:
{
lean_object* v___x_1071_; uint8_t v___x_1072_; 
v___x_1071_ = lean_unsigned_to_nat(0u);
v___x_1072_ = lean_nat_dec_eq(v_maxRecDepth_1058_, v___x_1071_);
if (v___x_1072_ == 0)
{
uint8_t v___x_1073_; 
v___x_1073_ = lean_nat_dec_eq(v_currRecDepth_1057_, v_maxRecDepth_1058_);
if (v___x_1073_ == 0)
{
lean_inc(v_ref_1059_);
v___y_1034_ = v_initHeartbeats_1062_;
v___y_1035_ = v_quotContext_1064_;
v___y_1036_ = v_cancelTk_x3f_1067_;
v___y_1037_ = v_fileName_1054_;
v___y_1038_ = v_ref_1059_;
v___y_1039_ = v_maxRecDepth_1058_;
v___y_1040_ = v_currNamespace_1060_;
v___y_1041_ = v_currRecDepth_1057_;
v___y_1042_ = v_diag_1066_;
v___y_1043_ = v_suppressElabErrors_1068_;
v___y_1044_ = v_fileMap_1055_;
v___y_1045_ = v_options_1056_;
v___y_1046_ = v_currMacroScope_1065_;
v___y_1047_ = v_inheritedTraceOptions_1069_;
v___y_1048_ = v_openDecls_1061_;
v___y_1049_ = v_maxHeartbeats_1063_;
goto v___jp_1033_;
}
else
{
lean_object* v___x_1074_; 
lean_dec_ref(v_x_1014_);
lean_inc(v_ref_1059_);
v___x_1074_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg(v_ref_1059_);
v___y_1024_ = v___x_1074_;
goto v___jp_1023_;
}
}
else
{
lean_inc(v_ref_1059_);
v___y_1034_ = v_initHeartbeats_1062_;
v___y_1035_ = v_quotContext_1064_;
v___y_1036_ = v_cancelTk_x3f_1067_;
v___y_1037_ = v_fileName_1054_;
v___y_1038_ = v_ref_1059_;
v___y_1039_ = v_maxRecDepth_1058_;
v___y_1040_ = v_currNamespace_1060_;
v___y_1041_ = v_currRecDepth_1057_;
v___y_1042_ = v_diag_1066_;
v___y_1043_ = v_suppressElabErrors_1068_;
v___y_1044_ = v_fileMap_1055_;
v___y_1045_ = v_options_1056_;
v___y_1046_ = v_currMacroScope_1065_;
v___y_1047_ = v_inheritedTraceOptions_1069_;
v___y_1048_ = v_openDecls_1061_;
v___y_1049_ = v_maxHeartbeats_1063_;
goto v___jp_1033_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg___boxed(lean_object* v_x_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_){
_start:
{
lean_object* v_res_1095_; 
v_res_1095_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg(v_x_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
lean_dec(v___y_1093_);
lean_dec_ref(v___y_1092_);
lean_dec(v___y_1091_);
lean_dec_ref(v___y_1090_);
lean_dec(v___y_1089_);
lean_dec_ref(v___y_1088_);
lean_dec(v___y_1087_);
return v_res_1095_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1097_; lean_object* v_dummy_1098_; 
v___x_1097_ = lean_box(0);
v_dummy_1098_ = l_Lean_Expr_sort___override(v___x_1097_);
return v_dummy_1098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__1(lean_object* v_pre_1099_, lean_object* v_post_1100_, size_t v_sz_1101_, size_t v_i_1102_, lean_object* v_bs_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
uint8_t v___x_1112_; 
v___x_1112_ = lean_usize_dec_lt(v_i_1102_, v_sz_1101_);
if (v___x_1112_ == 0)
{
lean_object* v___x_1113_; 
lean_dec_ref(v_post_1100_);
lean_dec_ref(v_pre_1099_);
v___x_1113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1113_, 0, v_bs_1103_);
return v___x_1113_;
}
else
{
lean_object* v_v_1114_; lean_object* v___x_1115_; 
v_v_1114_ = lean_array_uget_borrowed(v_bs_1103_, v_i_1102_);
lean_inc(v_v_1114_);
lean_inc_ref(v_post_1100_);
lean_inc_ref(v_pre_1099_);
v___x_1115_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1099_, v_post_1100_, v_v_1114_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_);
if (lean_obj_tag(v___x_1115_) == 0)
{
lean_object* v_a_1116_; lean_object* v___x_1117_; lean_object* v_bs_x27_1118_; size_t v___x_1119_; size_t v___x_1120_; lean_object* v___x_1121_; 
v_a_1116_ = lean_ctor_get(v___x_1115_, 0);
lean_inc(v_a_1116_);
lean_dec_ref_known(v___x_1115_, 1);
v___x_1117_ = lean_unsigned_to_nat(0u);
v_bs_x27_1118_ = lean_array_uset(v_bs_1103_, v_i_1102_, v___x_1117_);
v___x_1119_ = ((size_t)1ULL);
v___x_1120_ = lean_usize_add(v_i_1102_, v___x_1119_);
v___x_1121_ = lean_array_uset(v_bs_x27_1118_, v_i_1102_, v_a_1116_);
v_i_1102_ = v___x_1120_;
v_bs_1103_ = v___x_1121_;
goto _start;
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1130_; 
lean_dec_ref(v_bs_1103_);
lean_dec_ref(v_post_1100_);
lean_dec_ref(v_pre_1099_);
v_a_1123_ = lean_ctor_get(v___x_1115_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1115_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1125_ = v___x_1115_;
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1115_);
v___x_1125_ = lean_box(0);
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
v_resetjp_1124_:
{
lean_object* v___x_1128_; 
if (v_isShared_1126_ == 0)
{
v___x_1128_ = v___x_1125_;
goto v_reusejp_1127_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_a_1123_);
v___x_1128_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1127_;
}
v_reusejp_1127_:
{
return v___x_1128_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__4(lean_object* v_pre_1131_, lean_object* v_post_1132_, lean_object* v_x_1133_, lean_object* v_x_1134_, lean_object* v_x_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_){
_start:
{
if (lean_obj_tag(v_x_1133_) == 5)
{
lean_object* v_fn_1144_; lean_object* v_arg_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; 
v_fn_1144_ = lean_ctor_get(v_x_1133_, 0);
lean_inc_ref(v_fn_1144_);
v_arg_1145_ = lean_ctor_get(v_x_1133_, 1);
lean_inc_ref(v_arg_1145_);
lean_dec_ref_known(v_x_1133_, 2);
v___x_1146_ = lean_array_set(v_x_1134_, v_x_1135_, v_arg_1145_);
v___x_1147_ = lean_unsigned_to_nat(1u);
v___x_1148_ = lean_nat_sub(v_x_1135_, v___x_1147_);
lean_dec(v_x_1135_);
v_x_1133_ = v_fn_1144_;
v_x_1134_ = v___x_1146_;
v_x_1135_ = v___x_1148_;
goto _start;
}
else
{
lean_object* v___x_1150_; 
lean_dec(v_x_1135_);
lean_inc_ref(v_post_1132_);
lean_inc_ref(v_pre_1131_);
v___x_1150_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1131_, v_post_1132_, v_x_1133_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
if (lean_obj_tag(v___x_1150_) == 0)
{
lean_object* v_a_1151_; size_t v_sz_1152_; size_t v___x_1153_; lean_object* v___x_1154_; 
v_a_1151_ = lean_ctor_get(v___x_1150_, 0);
lean_inc(v_a_1151_);
lean_dec_ref_known(v___x_1150_, 1);
v_sz_1152_ = lean_array_size(v_x_1134_);
v___x_1153_ = ((size_t)0ULL);
lean_inc_ref(v_post_1132_);
lean_inc_ref(v_pre_1131_);
v___x_1154_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__1(v_pre_1131_, v_post_1132_, v_sz_1152_, v___x_1153_, v_x_1134_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
if (lean_obj_tag(v___x_1154_) == 0)
{
lean_object* v_a_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; 
v_a_1155_ = lean_ctor_get(v___x_1154_, 0);
lean_inc(v_a_1155_);
lean_dec_ref_known(v___x_1154_, 1);
v___x_1156_ = l_Lean_mkAppN(v_a_1151_, v_a_1155_);
lean_dec(v_a_1155_);
v___x_1157_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1131_, v_post_1132_, v___x_1156_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
return v___x_1157_;
}
else
{
lean_object* v_a_1158_; lean_object* v___x_1160_; uint8_t v_isShared_1161_; uint8_t v_isSharedCheck_1165_; 
lean_dec(v_a_1151_);
lean_dec_ref(v_post_1132_);
lean_dec_ref(v_pre_1131_);
v_a_1158_ = lean_ctor_get(v___x_1154_, 0);
v_isSharedCheck_1165_ = !lean_is_exclusive(v___x_1154_);
if (v_isSharedCheck_1165_ == 0)
{
v___x_1160_ = v___x_1154_;
v_isShared_1161_ = v_isSharedCheck_1165_;
goto v_resetjp_1159_;
}
else
{
lean_inc(v_a_1158_);
lean_dec(v___x_1154_);
v___x_1160_ = lean_box(0);
v_isShared_1161_ = v_isSharedCheck_1165_;
goto v_resetjp_1159_;
}
v_resetjp_1159_:
{
lean_object* v___x_1163_; 
if (v_isShared_1161_ == 0)
{
v___x_1163_ = v___x_1160_;
goto v_reusejp_1162_;
}
else
{
lean_object* v_reuseFailAlloc_1164_; 
v_reuseFailAlloc_1164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1164_, 0, v_a_1158_);
v___x_1163_ = v_reuseFailAlloc_1164_;
goto v_reusejp_1162_;
}
v_reusejp_1162_:
{
return v___x_1163_;
}
}
}
}
else
{
lean_dec_ref(v_x_1134_);
lean_dec_ref(v_post_1132_);
lean_dec_ref(v_pre_1131_);
return v___x_1150_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1(lean_object* v___x_1166_, lean_object* v_pre_1167_, lean_object* v_e_1168_, lean_object* v_post_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; uint8_t v___y_1182_; lean_object* v___y_1183_; lean_object* v___y_1184_; lean_object* v___y_1185_; uint8_t v___y_1186_; lean_object* v___y_1196_; lean_object* v___y_1197_; lean_object* v___y_1198_; lean_object* v___y_1199_; uint8_t v___y_1200_; uint8_t v___y_1201_; uint8_t v___y_1209_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v___y_1212_; lean_object* v___y_1213_; uint8_t v___y_1214_; lean_object* v___x_1221_; 
v___x_1221_ = l_Lean_Core_checkSystem(v___x_1166_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1221_) == 0)
{
lean_object* v___x_1222_; 
lean_dec_ref_known(v___x_1221_, 1);
lean_inc_ref(v_pre_1167_);
lean_inc(v___y_1176_);
lean_inc_ref(v___y_1175_);
lean_inc(v___y_1174_);
lean_inc_ref(v___y_1173_);
lean_inc(v___y_1172_);
lean_inc_ref(v___y_1171_);
lean_inc_ref(v_e_1168_);
v___x_1222_ = lean_apply_8(v_pre_1167_, v_e_1168_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_, lean_box(0));
if (lean_obj_tag(v___x_1222_) == 0)
{
lean_object* v_a_1223_; lean_object* v___x_1225_; uint8_t v_isShared_1226_; uint8_t v_isSharedCheck_1312_; 
v_a_1223_ = lean_ctor_get(v___x_1222_, 0);
v_isSharedCheck_1312_ = !lean_is_exclusive(v___x_1222_);
if (v_isSharedCheck_1312_ == 0)
{
v___x_1225_ = v___x_1222_;
v_isShared_1226_ = v_isSharedCheck_1312_;
goto v_resetjp_1224_;
}
else
{
lean_inc(v_a_1223_);
lean_dec(v___x_1222_);
v___x_1225_ = lean_box(0);
v_isShared_1226_ = v_isSharedCheck_1312_;
goto v_resetjp_1224_;
}
v_resetjp_1224_:
{
lean_object* v___y_1228_; 
switch(lean_obj_tag(v_a_1223_))
{
case 0:
{
lean_object* v_e_1302_; lean_object* v___x_1304_; 
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_e_1168_);
lean_dec_ref(v_pre_1167_);
v_e_1302_ = lean_ctor_get(v_a_1223_, 0);
lean_inc_ref(v_e_1302_);
lean_dec_ref_known(v_a_1223_, 1);
if (v_isShared_1226_ == 0)
{
lean_ctor_set(v___x_1225_, 0, v_e_1302_);
v___x_1304_ = v___x_1225_;
goto v_reusejp_1303_;
}
else
{
lean_object* v_reuseFailAlloc_1305_; 
v_reuseFailAlloc_1305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1305_, 0, v_e_1302_);
v___x_1304_ = v_reuseFailAlloc_1305_;
goto v_reusejp_1303_;
}
v_reusejp_1303_:
{
return v___x_1304_;
}
}
case 1:
{
lean_object* v_e_1306_; lean_object* v___x_1307_; 
lean_del_object(v___x_1225_);
lean_dec_ref(v_e_1168_);
v_e_1306_ = lean_ctor_get(v_a_1223_, 0);
lean_inc_ref(v_e_1306_);
lean_dec_ref_known(v_a_1223_, 1);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1307_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_e_1306_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1307_) == 0)
{
lean_object* v_a_1308_; lean_object* v___x_1309_; 
v_a_1308_ = lean_ctor_get(v___x_1307_, 0);
lean_inc(v_a_1308_);
lean_dec_ref_known(v___x_1307_, 1);
v___x_1309_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v_a_1308_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1309_;
}
else
{
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1307_;
}
}
default: 
{
lean_object* v_e_x3f_1310_; 
lean_del_object(v___x_1225_);
v_e_x3f_1310_ = lean_ctor_get(v_a_1223_, 0);
lean_inc(v_e_x3f_1310_);
lean_dec_ref_known(v_a_1223_, 1);
if (lean_obj_tag(v_e_x3f_1310_) == 0)
{
v___y_1228_ = v_e_1168_;
goto v___jp_1227_;
}
else
{
lean_object* v_val_1311_; 
lean_dec_ref(v_e_1168_);
v_val_1311_ = lean_ctor_get(v_e_x3f_1310_, 0);
lean_inc(v_val_1311_);
lean_dec_ref_known(v_e_x3f_1310_, 1);
v___y_1228_ = v_val_1311_;
goto v___jp_1227_;
}
}
}
v___jp_1227_:
{
switch(lean_obj_tag(v___y_1228_))
{
case 7:
{
lean_object* v_binderName_1229_; lean_object* v_binderType_1230_; lean_object* v_body_1231_; uint8_t v_binderInfo_1232_; lean_object* v___x_1233_; 
v_binderName_1229_ = lean_ctor_get(v___y_1228_, 0);
lean_inc(v_binderName_1229_);
v_binderType_1230_ = lean_ctor_get(v___y_1228_, 1);
v_body_1231_ = lean_ctor_get(v___y_1228_, 2);
v_binderInfo_1232_ = lean_ctor_get_uint8(v___y_1228_, sizeof(void*)*3 + 8);
lean_inc_ref(v_binderType_1230_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1233_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_binderType_1230_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1233_) == 0)
{
lean_object* v_a_1234_; lean_object* v___x_1235_; 
v_a_1234_ = lean_ctor_get(v___x_1233_, 0);
lean_inc(v_a_1234_);
lean_dec_ref_known(v___x_1233_, 1);
lean_inc_ref(v_body_1231_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1235_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_body_1231_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1235_) == 0)
{
lean_object* v_a_1236_; size_t v___x_1237_; size_t v___x_1238_; uint8_t v___x_1239_; 
v_a_1236_ = lean_ctor_get(v___x_1235_, 0);
lean_inc(v_a_1236_);
lean_dec_ref_known(v___x_1235_, 1);
v___x_1237_ = lean_ptr_addr(v_binderType_1230_);
v___x_1238_ = lean_ptr_addr(v_a_1234_);
v___x_1239_ = lean_usize_dec_eq(v___x_1237_, v___x_1238_);
if (v___x_1239_ == 0)
{
v___y_1209_ = v_binderInfo_1232_;
v___y_1210_ = v___y_1228_;
v___y_1211_ = v_a_1236_;
v___y_1212_ = v_a_1234_;
v___y_1213_ = v_binderName_1229_;
v___y_1214_ = v___x_1239_;
goto v___jp_1208_;
}
else
{
size_t v___x_1240_; size_t v___x_1241_; uint8_t v___x_1242_; 
v___x_1240_ = lean_ptr_addr(v_body_1231_);
v___x_1241_ = lean_ptr_addr(v_a_1236_);
v___x_1242_ = lean_usize_dec_eq(v___x_1240_, v___x_1241_);
v___y_1209_ = v_binderInfo_1232_;
v___y_1210_ = v___y_1228_;
v___y_1211_ = v_a_1236_;
v___y_1212_ = v_a_1234_;
v___y_1213_ = v_binderName_1229_;
v___y_1214_ = v___x_1242_;
goto v___jp_1208_;
}
}
else
{
lean_dec(v_a_1234_);
lean_dec_ref_known(v___y_1228_, 3);
lean_dec(v_binderName_1229_);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1235_;
}
}
else
{
lean_dec_ref_known(v___y_1228_, 3);
lean_dec(v_binderName_1229_);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1233_;
}
}
case 6:
{
lean_object* v_binderName_1243_; lean_object* v_binderType_1244_; lean_object* v_body_1245_; uint8_t v_binderInfo_1246_; lean_object* v___x_1247_; 
v_binderName_1243_ = lean_ctor_get(v___y_1228_, 0);
lean_inc(v_binderName_1243_);
v_binderType_1244_ = lean_ctor_get(v___y_1228_, 1);
v_body_1245_ = lean_ctor_get(v___y_1228_, 2);
v_binderInfo_1246_ = lean_ctor_get_uint8(v___y_1228_, sizeof(void*)*3 + 8);
lean_inc_ref(v_binderType_1244_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1247_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_binderType_1244_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1247_) == 0)
{
lean_object* v_a_1248_; lean_object* v___x_1249_; 
v_a_1248_ = lean_ctor_get(v___x_1247_, 0);
lean_inc(v_a_1248_);
lean_dec_ref_known(v___x_1247_, 1);
lean_inc_ref(v_body_1245_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1249_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_body_1245_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1249_) == 0)
{
lean_object* v_a_1250_; size_t v___x_1251_; size_t v___x_1252_; uint8_t v___x_1253_; 
v_a_1250_ = lean_ctor_get(v___x_1249_, 0);
lean_inc(v_a_1250_);
lean_dec_ref_known(v___x_1249_, 1);
v___x_1251_ = lean_ptr_addr(v_binderType_1244_);
v___x_1252_ = lean_ptr_addr(v_a_1248_);
v___x_1253_ = lean_usize_dec_eq(v___x_1251_, v___x_1252_);
if (v___x_1253_ == 0)
{
v___y_1196_ = v___y_1228_;
v___y_1197_ = v_a_1250_;
v___y_1198_ = v_a_1248_;
v___y_1199_ = v_binderName_1243_;
v___y_1200_ = v_binderInfo_1246_;
v___y_1201_ = v___x_1253_;
goto v___jp_1195_;
}
else
{
size_t v___x_1254_; size_t v___x_1255_; uint8_t v___x_1256_; 
v___x_1254_ = lean_ptr_addr(v_body_1245_);
v___x_1255_ = lean_ptr_addr(v_a_1250_);
v___x_1256_ = lean_usize_dec_eq(v___x_1254_, v___x_1255_);
v___y_1196_ = v___y_1228_;
v___y_1197_ = v_a_1250_;
v___y_1198_ = v_a_1248_;
v___y_1199_ = v_binderName_1243_;
v___y_1200_ = v_binderInfo_1246_;
v___y_1201_ = v___x_1256_;
goto v___jp_1195_;
}
}
else
{
lean_dec(v_a_1248_);
lean_dec(v_binderName_1243_);
lean_dec_ref_known(v___y_1228_, 3);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1249_;
}
}
else
{
lean_dec_ref_known(v___y_1228_, 3);
lean_dec(v_binderName_1243_);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1247_;
}
}
case 8:
{
lean_object* v_declName_1257_; lean_object* v_type_1258_; lean_object* v_value_1259_; lean_object* v_body_1260_; uint8_t v_nondep_1261_; lean_object* v___x_1262_; 
v_declName_1257_ = lean_ctor_get(v___y_1228_, 0);
lean_inc(v_declName_1257_);
v_type_1258_ = lean_ctor_get(v___y_1228_, 1);
v_value_1259_ = lean_ctor_get(v___y_1228_, 2);
v_body_1260_ = lean_ctor_get(v___y_1228_, 3);
lean_inc_ref(v_body_1260_);
v_nondep_1261_ = lean_ctor_get_uint8(v___y_1228_, sizeof(void*)*4 + 8);
lean_inc_ref(v_type_1258_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1262_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_type_1258_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1262_) == 0)
{
lean_object* v_a_1263_; lean_object* v___x_1264_; 
v_a_1263_ = lean_ctor_get(v___x_1262_, 0);
lean_inc(v_a_1263_);
lean_dec_ref_known(v___x_1262_, 1);
lean_inc_ref(v_value_1259_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1264_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_value_1259_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1264_) == 0)
{
lean_object* v_a_1265_; lean_object* v___x_1266_; 
v_a_1265_ = lean_ctor_get(v___x_1264_, 0);
lean_inc(v_a_1265_);
lean_dec_ref_known(v___x_1264_, 1);
lean_inc_ref(v_body_1260_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1266_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_body_1260_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1266_) == 0)
{
lean_object* v_a_1267_; size_t v___x_1268_; size_t v___x_1269_; uint8_t v___x_1270_; 
v_a_1267_ = lean_ctor_get(v___x_1266_, 0);
lean_inc(v_a_1267_);
lean_dec_ref_known(v___x_1266_, 1);
v___x_1268_ = lean_ptr_addr(v_type_1258_);
v___x_1269_ = lean_ptr_addr(v_a_1263_);
v___x_1270_ = lean_usize_dec_eq(v___x_1268_, v___x_1269_);
if (v___x_1270_ == 0)
{
v___y_1179_ = v_a_1267_;
v___y_1180_ = v_a_1265_;
v___y_1181_ = v___y_1228_;
v___y_1182_ = v_nondep_1261_;
v___y_1183_ = v_a_1263_;
v___y_1184_ = v_body_1260_;
v___y_1185_ = v_declName_1257_;
v___y_1186_ = v___x_1270_;
goto v___jp_1178_;
}
else
{
size_t v___x_1271_; size_t v___x_1272_; uint8_t v___x_1273_; 
v___x_1271_ = lean_ptr_addr(v_value_1259_);
v___x_1272_ = lean_ptr_addr(v_a_1265_);
v___x_1273_ = lean_usize_dec_eq(v___x_1271_, v___x_1272_);
v___y_1179_ = v_a_1267_;
v___y_1180_ = v_a_1265_;
v___y_1181_ = v___y_1228_;
v___y_1182_ = v_nondep_1261_;
v___y_1183_ = v_a_1263_;
v___y_1184_ = v_body_1260_;
v___y_1185_ = v_declName_1257_;
v___y_1186_ = v___x_1273_;
goto v___jp_1178_;
}
}
else
{
lean_dec(v_a_1265_);
lean_dec(v_a_1263_);
lean_dec_ref(v_body_1260_);
lean_dec_ref_known(v___y_1228_, 4);
lean_dec(v_declName_1257_);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1266_;
}
}
else
{
lean_dec(v_a_1263_);
lean_dec_ref(v_body_1260_);
lean_dec_ref_known(v___y_1228_, 4);
lean_dec(v_declName_1257_);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1264_;
}
}
else
{
lean_dec_ref(v_body_1260_);
lean_dec(v_declName_1257_);
lean_dec_ref_known(v___y_1228_, 4);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1262_;
}
}
case 5:
{
lean_object* v_dummy_1274_; lean_object* v_nargs_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; 
v_dummy_1274_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___closed__0, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___closed__0_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___closed__0);
v_nargs_1275_ = l_Lean_Expr_getAppNumArgs(v___y_1228_);
lean_inc(v_nargs_1275_);
v___x_1276_ = lean_mk_array(v_nargs_1275_, v_dummy_1274_);
v___x_1277_ = lean_unsigned_to_nat(1u);
v___x_1278_ = lean_nat_sub(v_nargs_1275_, v___x_1277_);
lean_dec(v_nargs_1275_);
v___x_1279_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__4(v_pre_1167_, v_post_1169_, v___y_1228_, v___x_1276_, v___x_1278_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1279_;
}
case 10:
{
lean_object* v_data_1280_; lean_object* v_expr_1281_; lean_object* v___x_1282_; 
v_data_1280_ = lean_ctor_get(v___y_1228_, 0);
v_expr_1281_ = lean_ctor_get(v___y_1228_, 1);
lean_inc_ref(v_expr_1281_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1282_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_expr_1281_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1282_) == 0)
{
lean_object* v_a_1283_; size_t v___x_1284_; size_t v___x_1285_; uint8_t v___x_1286_; 
v_a_1283_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1283_);
lean_dec_ref_known(v___x_1282_, 1);
v___x_1284_ = lean_ptr_addr(v_expr_1281_);
v___x_1285_ = lean_ptr_addr(v_a_1283_);
v___x_1286_ = lean_usize_dec_eq(v___x_1284_, v___x_1285_);
if (v___x_1286_ == 0)
{
lean_object* v___x_1287_; lean_object* v___x_1288_; 
lean_inc(v_data_1280_);
lean_dec_ref_known(v___y_1228_, 2);
v___x_1287_ = l_Lean_Expr_mdata___override(v_data_1280_, v_a_1283_);
v___x_1288_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1287_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1288_;
}
else
{
lean_object* v___x_1289_; 
lean_dec(v_a_1283_);
v___x_1289_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___y_1228_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1289_;
}
}
else
{
lean_dec_ref_known(v___y_1228_, 2);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1282_;
}
}
case 11:
{
lean_object* v_typeName_1290_; lean_object* v_idx_1291_; lean_object* v_struct_1292_; lean_object* v___x_1293_; 
v_typeName_1290_ = lean_ctor_get(v___y_1228_, 0);
v_idx_1291_ = lean_ctor_get(v___y_1228_, 1);
v_struct_1292_ = lean_ctor_get(v___y_1228_, 2);
lean_inc_ref(v_struct_1292_);
lean_inc_ref(v_post_1169_);
lean_inc_ref(v_pre_1167_);
v___x_1293_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1167_, v_post_1169_, v_struct_1292_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
if (lean_obj_tag(v___x_1293_) == 0)
{
lean_object* v_a_1294_; size_t v___x_1295_; size_t v___x_1296_; uint8_t v___x_1297_; 
v_a_1294_ = lean_ctor_get(v___x_1293_, 0);
lean_inc(v_a_1294_);
lean_dec_ref_known(v___x_1293_, 1);
v___x_1295_ = lean_ptr_addr(v_struct_1292_);
v___x_1296_ = lean_ptr_addr(v_a_1294_);
v___x_1297_ = lean_usize_dec_eq(v___x_1295_, v___x_1296_);
if (v___x_1297_ == 0)
{
lean_object* v___x_1298_; lean_object* v___x_1299_; 
lean_inc(v_idx_1291_);
lean_inc(v_typeName_1290_);
lean_dec_ref_known(v___y_1228_, 3);
v___x_1298_ = l_Lean_Expr_proj___override(v_typeName_1290_, v_idx_1291_, v_a_1294_);
v___x_1299_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1298_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1299_;
}
else
{
lean_object* v___x_1300_; 
lean_dec(v_a_1294_);
v___x_1300_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___y_1228_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1300_;
}
}
else
{
lean_dec_ref_known(v___y_1228_, 3);
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_pre_1167_);
return v___x_1293_;
}
}
default: 
{
lean_object* v___x_1301_; 
v___x_1301_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___y_1228_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1301_;
}
}
}
}
}
else
{
lean_object* v_a_1313_; lean_object* v___x_1315_; uint8_t v_isShared_1316_; uint8_t v_isSharedCheck_1320_; 
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_e_1168_);
lean_dec_ref(v_pre_1167_);
v_a_1313_ = lean_ctor_get(v___x_1222_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v___x_1222_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1315_ = v___x_1222_;
v_isShared_1316_ = v_isSharedCheck_1320_;
goto v_resetjp_1314_;
}
else
{
lean_inc(v_a_1313_);
lean_dec(v___x_1222_);
v___x_1315_ = lean_box(0);
v_isShared_1316_ = v_isSharedCheck_1320_;
goto v_resetjp_1314_;
}
v_resetjp_1314_:
{
lean_object* v___x_1318_; 
if (v_isShared_1316_ == 0)
{
v___x_1318_ = v___x_1315_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v_a_1313_);
v___x_1318_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1317_;
}
v_reusejp_1317_:
{
return v___x_1318_;
}
}
}
}
else
{
lean_object* v_a_1321_; lean_object* v___x_1323_; uint8_t v_isShared_1324_; uint8_t v_isSharedCheck_1328_; 
lean_dec_ref(v_post_1169_);
lean_dec_ref(v_e_1168_);
lean_dec_ref(v_pre_1167_);
v_a_1321_ = lean_ctor_get(v___x_1221_, 0);
v_isSharedCheck_1328_ = !lean_is_exclusive(v___x_1221_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1323_ = v___x_1221_;
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
else
{
lean_inc(v_a_1321_);
lean_dec(v___x_1221_);
v___x_1323_ = lean_box(0);
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
v_resetjp_1322_:
{
lean_object* v___x_1326_; 
if (v_isShared_1324_ == 0)
{
v___x_1326_ = v___x_1323_;
goto v_reusejp_1325_;
}
else
{
lean_object* v_reuseFailAlloc_1327_; 
v_reuseFailAlloc_1327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1327_, 0, v_a_1321_);
v___x_1326_ = v_reuseFailAlloc_1327_;
goto v_reusejp_1325_;
}
v_reusejp_1325_:
{
return v___x_1326_;
}
}
}
v___jp_1178_:
{
if (v___y_1186_ == 0)
{
lean_object* v___x_1187_; lean_object* v___x_1188_; 
lean_dec_ref(v___y_1184_);
lean_dec_ref(v___y_1181_);
v___x_1187_ = l_Lean_Expr_letE___override(v___y_1185_, v___y_1183_, v___y_1180_, v___y_1179_, v___y_1182_);
v___x_1188_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1187_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1188_;
}
else
{
size_t v___x_1189_; size_t v___x_1190_; uint8_t v___x_1191_; 
v___x_1189_ = lean_ptr_addr(v___y_1184_);
lean_dec_ref(v___y_1184_);
v___x_1190_ = lean_ptr_addr(v___y_1179_);
v___x_1191_ = lean_usize_dec_eq(v___x_1189_, v___x_1190_);
if (v___x_1191_ == 0)
{
lean_object* v___x_1192_; lean_object* v___x_1193_; 
lean_dec_ref(v___y_1181_);
v___x_1192_ = l_Lean_Expr_letE___override(v___y_1185_, v___y_1183_, v___y_1180_, v___y_1179_, v___y_1182_);
v___x_1193_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1192_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1193_;
}
else
{
lean_object* v___x_1194_; 
lean_dec(v___y_1185_);
lean_dec_ref(v___y_1183_);
lean_dec_ref(v___y_1180_);
lean_dec_ref(v___y_1179_);
v___x_1194_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___y_1181_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1194_;
}
}
}
v___jp_1195_:
{
if (v___y_1201_ == 0)
{
lean_object* v___x_1202_; lean_object* v___x_1203_; 
lean_dec_ref(v___y_1196_);
v___x_1202_ = l_Lean_Expr_lam___override(v___y_1199_, v___y_1198_, v___y_1197_, v___y_1200_);
v___x_1203_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1202_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1203_;
}
else
{
uint8_t v___x_1204_; 
v___x_1204_ = l_Lean_instBEqBinderInfo_beq(v___y_1200_, v___y_1200_);
if (v___x_1204_ == 0)
{
lean_object* v___x_1205_; lean_object* v___x_1206_; 
lean_dec_ref(v___y_1196_);
v___x_1205_ = l_Lean_Expr_lam___override(v___y_1199_, v___y_1198_, v___y_1197_, v___y_1200_);
v___x_1206_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1205_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1206_;
}
else
{
lean_object* v___x_1207_; 
lean_dec(v___y_1199_);
lean_dec_ref(v___y_1198_);
lean_dec_ref(v___y_1197_);
v___x_1207_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___y_1196_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1207_;
}
}
}
v___jp_1208_:
{
if (v___y_1214_ == 0)
{
lean_object* v___x_1215_; lean_object* v___x_1216_; 
lean_dec_ref(v___y_1210_);
v___x_1215_ = l_Lean_Expr_forallE___override(v___y_1213_, v___y_1212_, v___y_1211_, v___y_1209_);
v___x_1216_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1215_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1216_;
}
else
{
uint8_t v___x_1217_; 
v___x_1217_ = l_Lean_instBEqBinderInfo_beq(v___y_1209_, v___y_1209_);
if (v___x_1217_ == 0)
{
lean_object* v___x_1218_; lean_object* v___x_1219_; 
lean_dec_ref(v___y_1210_);
v___x_1218_ = l_Lean_Expr_forallE___override(v___y_1213_, v___y_1212_, v___y_1211_, v___y_1209_);
v___x_1219_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___x_1218_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1219_;
}
else
{
lean_object* v___x_1220_; 
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec_ref(v___y_1211_);
v___x_1220_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1167_, v_post_1169_, v___y_1210_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_);
return v___x_1220_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___boxed(lean_object* v___x_1329_, lean_object* v_pre_1330_, lean_object* v_e_1331_, lean_object* v_post_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_){
_start:
{
lean_object* v_res_1341_; 
v_res_1341_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1(v___x_1329_, v_pre_1330_, v_e_1331_, v_post_1332_, v___y_1333_, v___y_1334_, v___y_1335_, v___y_1336_, v___y_1337_, v___y_1338_, v___y_1339_);
lean_dec(v___y_1339_);
lean_dec_ref(v___y_1338_);
lean_dec(v___y_1337_);
lean_dec_ref(v___y_1336_);
lean_dec(v___y_1335_);
lean_dec_ref(v___y_1334_);
lean_dec(v___y_1333_);
return v_res_1341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(lean_object* v_pre_1342_, lean_object* v_post_1343_, lean_object* v_e_1344_, lean_object* v_a_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_){
_start:
{
lean_object* v___x_1353_; lean_object* v___x_1354_; 
lean_inc(v_a_1345_);
v___x_1353_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_1353_, 0, lean_box(0));
lean_closure_set(v___x_1353_, 1, lean_box(0));
lean_closure_set(v___x_1353_, 2, v_a_1345_);
v___x_1354_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0(lean_box(0), v___x_1353_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_);
if (lean_obj_tag(v___x_1354_) == 0)
{
lean_object* v_a_1355_; lean_object* v___x_1357_; uint8_t v_isShared_1358_; uint8_t v_isSharedCheck_1386_; 
v_a_1355_ = lean_ctor_get(v___x_1354_, 0);
v_isSharedCheck_1386_ = !lean_is_exclusive(v___x_1354_);
if (v_isSharedCheck_1386_ == 0)
{
v___x_1357_ = v___x_1354_;
v_isShared_1358_ = v_isSharedCheck_1386_;
goto v_resetjp_1356_;
}
else
{
lean_inc(v_a_1355_);
lean_dec(v___x_1354_);
v___x_1357_ = lean_box(0);
v_isShared_1358_ = v_isSharedCheck_1386_;
goto v_resetjp_1356_;
}
v_resetjp_1356_:
{
lean_object* v___x_1359_; 
v___x_1359_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg(v_a_1355_, v_e_1344_);
lean_dec(v_a_1355_);
if (lean_obj_tag(v___x_1359_) == 0)
{
lean_object* v___x_1360_; lean_object* v___f_1361_; lean_object* v___x_1362_; 
lean_del_object(v___x_1357_);
v___x_1360_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___closed__0));
lean_inc_ref(v_e_1344_);
v___f_1361_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__1___boxed), 12, 4);
lean_closure_set(v___f_1361_, 0, v___x_1360_);
lean_closure_set(v___f_1361_, 1, v_pre_1342_);
lean_closure_set(v___f_1361_, 2, v_e_1344_);
lean_closure_set(v___f_1361_, 3, v_post_1343_);
v___x_1362_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg(v___f_1361_, v_a_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_);
if (lean_obj_tag(v___x_1362_) == 0)
{
lean_object* v_a_1363_; lean_object* v___f_1364_; lean_object* v___x_1365_; 
v_a_1363_ = lean_ctor_get(v___x_1362_, 0);
lean_inc_n(v_a_1363_, 2);
lean_dec_ref_known(v___x_1362_, 1);
lean_inc(v_a_1345_);
v___f_1364_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__2___boxed), 4, 3);
lean_closure_set(v___f_1364_, 0, v_a_1345_);
lean_closure_set(v___f_1364_, 1, v_e_1344_);
lean_closure_set(v___f_1364_, 2, v_a_1363_);
v___x_1365_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___lam__0(lean_box(0), v___f_1364_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_);
if (lean_obj_tag(v___x_1365_) == 0)
{
lean_object* v___x_1367_; uint8_t v_isShared_1368_; uint8_t v_isSharedCheck_1372_; 
v_isSharedCheck_1372_ = !lean_is_exclusive(v___x_1365_);
if (v_isSharedCheck_1372_ == 0)
{
lean_object* v_unused_1373_; 
v_unused_1373_ = lean_ctor_get(v___x_1365_, 0);
lean_dec(v_unused_1373_);
v___x_1367_ = v___x_1365_;
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
else
{
lean_dec(v___x_1365_);
v___x_1367_ = lean_box(0);
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
v_resetjp_1366_:
{
lean_object* v___x_1370_; 
if (v_isShared_1368_ == 0)
{
lean_ctor_set(v___x_1367_, 0, v_a_1363_);
v___x_1370_ = v___x_1367_;
goto v_reusejp_1369_;
}
else
{
lean_object* v_reuseFailAlloc_1371_; 
v_reuseFailAlloc_1371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1371_, 0, v_a_1363_);
v___x_1370_ = v_reuseFailAlloc_1371_;
goto v_reusejp_1369_;
}
v_reusejp_1369_:
{
return v___x_1370_;
}
}
}
else
{
lean_object* v_a_1374_; lean_object* v___x_1376_; uint8_t v_isShared_1377_; uint8_t v_isSharedCheck_1381_; 
lean_dec(v_a_1363_);
v_a_1374_ = lean_ctor_get(v___x_1365_, 0);
v_isSharedCheck_1381_ = !lean_is_exclusive(v___x_1365_);
if (v_isSharedCheck_1381_ == 0)
{
v___x_1376_ = v___x_1365_;
v_isShared_1377_ = v_isSharedCheck_1381_;
goto v_resetjp_1375_;
}
else
{
lean_inc(v_a_1374_);
lean_dec(v___x_1365_);
v___x_1376_ = lean_box(0);
v_isShared_1377_ = v_isSharedCheck_1381_;
goto v_resetjp_1375_;
}
v_resetjp_1375_:
{
lean_object* v___x_1379_; 
if (v_isShared_1377_ == 0)
{
v___x_1379_ = v___x_1376_;
goto v_reusejp_1378_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_a_1374_);
v___x_1379_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1378_;
}
v_reusejp_1378_:
{
return v___x_1379_;
}
}
}
}
else
{
lean_dec_ref(v_e_1344_);
return v___x_1362_;
}
}
else
{
lean_object* v_val_1382_; lean_object* v___x_1384_; 
lean_dec_ref(v_e_1344_);
lean_dec_ref(v_post_1343_);
lean_dec_ref(v_pre_1342_);
v_val_1382_ = lean_ctor_get(v___x_1359_, 0);
lean_inc(v_val_1382_);
lean_dec_ref_known(v___x_1359_, 1);
if (v_isShared_1358_ == 0)
{
lean_ctor_set(v___x_1357_, 0, v_val_1382_);
v___x_1384_ = v___x_1357_;
goto v_reusejp_1383_;
}
else
{
lean_object* v_reuseFailAlloc_1385_; 
v_reuseFailAlloc_1385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1385_, 0, v_val_1382_);
v___x_1384_ = v_reuseFailAlloc_1385_;
goto v_reusejp_1383_;
}
v_reusejp_1383_:
{
return v___x_1384_;
}
}
}
}
else
{
lean_object* v_a_1387_; lean_object* v___x_1389_; uint8_t v_isShared_1390_; uint8_t v_isSharedCheck_1394_; 
lean_dec_ref(v_e_1344_);
lean_dec_ref(v_post_1343_);
lean_dec_ref(v_pre_1342_);
v_a_1387_ = lean_ctor_get(v___x_1354_, 0);
v_isSharedCheck_1394_ = !lean_is_exclusive(v___x_1354_);
if (v_isSharedCheck_1394_ == 0)
{
v___x_1389_ = v___x_1354_;
v_isShared_1390_ = v_isSharedCheck_1394_;
goto v_resetjp_1388_;
}
else
{
lean_inc(v_a_1387_);
lean_dec(v___x_1354_);
v___x_1389_ = lean_box(0);
v_isShared_1390_ = v_isSharedCheck_1394_;
goto v_resetjp_1388_;
}
v_resetjp_1388_:
{
lean_object* v___x_1392_; 
if (v_isShared_1390_ == 0)
{
v___x_1392_ = v___x_1389_;
goto v_reusejp_1391_;
}
else
{
lean_object* v_reuseFailAlloc_1393_; 
v_reuseFailAlloc_1393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1393_, 0, v_a_1387_);
v___x_1392_ = v_reuseFailAlloc_1393_;
goto v_reusejp_1391_;
}
v_reusejp_1391_:
{
return v___x_1392_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(lean_object* v_pre_1395_, lean_object* v_post_1396_, lean_object* v_e_1397_, lean_object* v_a_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_){
_start:
{
lean_object* v___x_1406_; 
lean_inc_ref(v_post_1396_);
lean_inc(v___y_1404_);
lean_inc_ref(v___y_1403_);
lean_inc(v___y_1402_);
lean_inc_ref(v___y_1401_);
lean_inc(v___y_1400_);
lean_inc_ref(v___y_1399_);
lean_inc_ref(v_e_1397_);
v___x_1406_ = lean_apply_8(v_post_1396_, v_e_1397_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_, v___y_1404_, lean_box(0));
if (lean_obj_tag(v___x_1406_) == 0)
{
lean_object* v_a_1407_; lean_object* v___x_1409_; uint8_t v_isShared_1410_; uint8_t v_isSharedCheck_1425_; 
v_a_1407_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1425_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1425_ == 0)
{
v___x_1409_ = v___x_1406_;
v_isShared_1410_ = v_isSharedCheck_1425_;
goto v_resetjp_1408_;
}
else
{
lean_inc(v_a_1407_);
lean_dec(v___x_1406_);
v___x_1409_ = lean_box(0);
v_isShared_1410_ = v_isSharedCheck_1425_;
goto v_resetjp_1408_;
}
v_resetjp_1408_:
{
switch(lean_obj_tag(v_a_1407_))
{
case 0:
{
lean_object* v_e_1411_; lean_object* v___x_1413_; 
lean_dec_ref(v_e_1397_);
lean_dec_ref(v_post_1396_);
lean_dec_ref(v_pre_1395_);
v_e_1411_ = lean_ctor_get(v_a_1407_, 0);
lean_inc_ref(v_e_1411_);
lean_dec_ref_known(v_a_1407_, 1);
if (v_isShared_1410_ == 0)
{
lean_ctor_set(v___x_1409_, 0, v_e_1411_);
v___x_1413_ = v___x_1409_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1414_; 
v_reuseFailAlloc_1414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1414_, 0, v_e_1411_);
v___x_1413_ = v_reuseFailAlloc_1414_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
return v___x_1413_;
}
}
case 1:
{
lean_object* v_e_1415_; lean_object* v___x_1416_; 
lean_del_object(v___x_1409_);
lean_dec_ref(v_e_1397_);
v_e_1415_ = lean_ctor_get(v_a_1407_, 0);
lean_inc_ref(v_e_1415_);
lean_dec_ref_known(v_a_1407_, 1);
v___x_1416_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1395_, v_post_1396_, v_e_1415_, v_a_1398_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_, v___y_1403_, v___y_1404_);
return v___x_1416_;
}
default: 
{
lean_object* v_e_x3f_1417_; 
lean_dec_ref(v_post_1396_);
lean_dec_ref(v_pre_1395_);
v_e_x3f_1417_ = lean_ctor_get(v_a_1407_, 0);
lean_inc(v_e_x3f_1417_);
lean_dec_ref_known(v_a_1407_, 1);
if (lean_obj_tag(v_e_x3f_1417_) == 0)
{
lean_object* v___x_1419_; 
if (v_isShared_1410_ == 0)
{
lean_ctor_set(v___x_1409_, 0, v_e_1397_);
v___x_1419_ = v___x_1409_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1420_; 
v_reuseFailAlloc_1420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1420_, 0, v_e_1397_);
v___x_1419_ = v_reuseFailAlloc_1420_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
return v___x_1419_;
}
}
else
{
lean_object* v_val_1421_; lean_object* v___x_1423_; 
lean_dec_ref(v_e_1397_);
v_val_1421_ = lean_ctor_get(v_e_x3f_1417_, 0);
lean_inc(v_val_1421_);
lean_dec_ref_known(v_e_x3f_1417_, 1);
if (v_isShared_1410_ == 0)
{
lean_ctor_set(v___x_1409_, 0, v_val_1421_);
v___x_1423_ = v___x_1409_;
goto v_reusejp_1422_;
}
else
{
lean_object* v_reuseFailAlloc_1424_; 
v_reuseFailAlloc_1424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1424_, 0, v_val_1421_);
v___x_1423_ = v_reuseFailAlloc_1424_;
goto v_reusejp_1422_;
}
v_reusejp_1422_:
{
return v___x_1423_;
}
}
}
}
}
}
else
{
lean_object* v_a_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1433_; 
lean_dec_ref(v_e_1397_);
lean_dec_ref(v_post_1396_);
lean_dec_ref(v_pre_1395_);
v_a_1426_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1433_ == 0)
{
v___x_1428_ = v___x_1406_;
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_a_1426_);
lean_dec(v___x_1406_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v___x_1431_; 
if (v_isShared_1429_ == 0)
{
v___x_1431_ = v___x_1428_;
goto v_reusejp_1430_;
}
else
{
lean_object* v_reuseFailAlloc_1432_; 
v_reuseFailAlloc_1432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1432_, 0, v_a_1426_);
v___x_1431_ = v_reuseFailAlloc_1432_;
goto v_reusejp_1430_;
}
v_reusejp_1430_:
{
return v___x_1431_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2___boxed(lean_object* v_pre_1434_, lean_object* v_post_1435_, lean_object* v_e_1436_, lean_object* v_a_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_){
_start:
{
lean_object* v_res_1445_; 
v_res_1445_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__2(v_pre_1434_, v_post_1435_, v_e_1436_, v_a_1437_, v___y_1438_, v___y_1439_, v___y_1440_, v___y_1441_, v___y_1442_, v___y_1443_);
lean_dec(v___y_1443_);
lean_dec_ref(v___y_1442_);
lean_dec(v___y_1441_);
lean_dec_ref(v___y_1440_);
lean_dec(v___y_1439_);
lean_dec_ref(v___y_1438_);
lean_dec(v_a_1437_);
return v_res_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__1___boxed(lean_object* v_pre_1446_, lean_object* v_post_1447_, lean_object* v_sz_1448_, lean_object* v_i_1449_, lean_object* v_bs_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_){
_start:
{
size_t v_sz_boxed_1459_; size_t v_i_boxed_1460_; lean_object* v_res_1461_; 
v_sz_boxed_1459_ = lean_unbox_usize(v_sz_1448_);
lean_dec(v_sz_1448_);
v_i_boxed_1460_ = lean_unbox_usize(v_i_1449_);
lean_dec(v_i_1449_);
v_res_1461_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__1(v_pre_1446_, v_post_1447_, v_sz_boxed_1459_, v_i_boxed_1460_, v_bs_1450_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_, v___y_1457_);
lean_dec(v___y_1457_);
lean_dec_ref(v___y_1456_);
lean_dec(v___y_1455_);
lean_dec_ref(v___y_1454_);
lean_dec(v___y_1453_);
lean_dec_ref(v___y_1452_);
lean_dec(v___y_1451_);
return v_res_1461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__4___boxed(lean_object* v_pre_1462_, lean_object* v_post_1463_, lean_object* v_x_1464_, lean_object* v_x_1465_, lean_object* v_x_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_){
_start:
{
lean_object* v_res_1475_; 
v_res_1475_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__4(v_pre_1462_, v_post_1463_, v_x_1464_, v_x_1465_, v_x_1466_, v___y_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1472_);
lean_dec(v___y_1471_);
lean_dec_ref(v___y_1470_);
lean_dec(v___y_1469_);
lean_dec_ref(v___y_1468_);
lean_dec(v___y_1467_);
return v_res_1475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0___boxed(lean_object* v_pre_1476_, lean_object* v_post_1477_, lean_object* v_e_1478_, lean_object* v_a_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_, lean_object* v___y_1486_){
_start:
{
lean_object* v_res_1487_; 
v_res_1487_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1476_, v_post_1477_, v_e_1478_, v_a_1479_, v___y_1480_, v___y_1481_, v___y_1482_, v___y_1483_, v___y_1484_, v___y_1485_);
lean_dec(v___y_1485_);
lean_dec_ref(v___y_1484_);
lean_dec(v___y_1483_);
lean_dec_ref(v___y_1482_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v_a_1479_);
return v_res_1487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0(lean_object* v_00_u03b1_1488_, lean_object* v_x_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_){
_start:
{
lean_object* v___x_1497_; lean_object* v___x_1498_; 
v___x_1497_ = lean_apply_1(v_x_1489_, lean_box(0));
v___x_1498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1497_);
return v___x_1498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0___boxed(lean_object* v_00_u03b1_1499_, lean_object* v_x_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_){
_start:
{
lean_object* v_res_1508_; 
v_res_1508_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0(v_00_u03b1_1499_, v_x_1500_, v___y_1501_, v___y_1502_, v___y_1503_, v___y_1504_, v___y_1505_, v___y_1506_);
lean_dec(v___y_1506_);
lean_dec_ref(v___y_1505_);
lean_dec(v___y_1504_);
lean_dec_ref(v___y_1503_);
lean_dec(v___y_1502_);
lean_dec_ref(v___y_1501_);
return v_res_1508_;
}
}
static lean_object* _init_lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; 
v___x_1509_ = lean_box(0);
v___x_1510_ = lean_unsigned_to_nat(16u);
v___x_1511_ = lean_mk_array(v___x_1510_, v___x_1509_);
return v___x_1511_;
}
}
static lean_object* _init_lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; 
v___x_1512_ = lean_obj_once(&lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__0, &lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__0_once, _init_lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__0);
v___x_1513_ = lean_unsigned_to_nat(0u);
v___x_1514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1514_, 0, v___x_1513_);
lean_ctor_set(v___x_1514_, 1, v___x_1512_);
return v___x_1514_;
}
}
static lean_object* _init_lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1515_ = lean_obj_once(&lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__1, &lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__1_once, _init_lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__1);
v___x_1516_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_1516_, 0, lean_box(0));
lean_closure_set(v___x_1516_, 1, lean_box(0));
lean_closure_set(v___x_1516_, 2, v___x_1515_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0(lean_object* v_input_1517_, lean_object* v_pre_1518_, lean_object* v_post_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_, lean_object* v___y_1522_, lean_object* v___y_1523_, lean_object* v___y_1524_, lean_object* v___y_1525_){
_start:
{
lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v_a_1529_; lean_object* v___x_1530_; 
v___x_1527_ = lean_obj_once(&lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__2, &lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__2_once, _init_lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___closed__2);
v___x_1528_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0(lean_box(0), v___x_1527_, v___y_1520_, v___y_1521_, v___y_1522_, v___y_1523_, v___y_1524_, v___y_1525_);
v_a_1529_ = lean_ctor_get(v___x_1528_, 0);
lean_inc(v_a_1529_);
lean_dec_ref(v___x_1528_);
v___x_1530_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0(v_pre_1518_, v_post_1519_, v_input_1517_, v_a_1529_, v___y_1520_, v___y_1521_, v___y_1522_, v___y_1523_, v___y_1524_, v___y_1525_);
if (lean_obj_tag(v___x_1530_) == 0)
{
lean_object* v_a_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1535_; uint8_t v_isShared_1536_; uint8_t v_isSharedCheck_1540_; 
v_a_1531_ = lean_ctor_get(v___x_1530_, 0);
lean_inc(v_a_1531_);
lean_dec_ref_known(v___x_1530_, 1);
v___x_1532_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_1532_, 0, lean_box(0));
lean_closure_set(v___x_1532_, 1, lean_box(0));
lean_closure_set(v___x_1532_, 2, v_a_1529_);
v___x_1533_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___lam__0(lean_box(0), v___x_1532_, v___y_1520_, v___y_1521_, v___y_1522_, v___y_1523_, v___y_1524_, v___y_1525_);
v_isSharedCheck_1540_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1540_ == 0)
{
lean_object* v_unused_1541_; 
v_unused_1541_ = lean_ctor_get(v___x_1533_, 0);
lean_dec(v_unused_1541_);
v___x_1535_ = v___x_1533_;
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
else
{
lean_dec(v___x_1533_);
v___x_1535_ = lean_box(0);
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
v_resetjp_1534_:
{
lean_object* v___x_1538_; 
if (v_isShared_1536_ == 0)
{
lean_ctor_set(v___x_1535_, 0, v_a_1531_);
v___x_1538_ = v___x_1535_;
goto v_reusejp_1537_;
}
else
{
lean_object* v_reuseFailAlloc_1539_; 
v_reuseFailAlloc_1539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1539_, 0, v_a_1531_);
v___x_1538_ = v_reuseFailAlloc_1539_;
goto v_reusejp_1537_;
}
v_reusejp_1537_:
{
return v___x_1538_;
}
}
}
else
{
lean_dec(v_a_1529_);
return v___x_1530_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0___boxed(lean_object* v_input_1542_, lean_object* v_pre_1543_, lean_object* v_post_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_){
_start:
{
lean_object* v_res_1552_; 
v_res_1552_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0(v_input_1542_, v_pre_1543_, v_post_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_, v___y_1550_);
lean_dec(v___y_1550_);
lean_dec_ref(v___y_1549_);
lean_dec(v___y_1548_);
lean_dec_ref(v___y_1547_);
lean_dec(v___y_1546_);
lean_dec_ref(v___y_1545_);
return v_res_1552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj(lean_object* v_stx_1555_, lean_object* v_expectedType_x3f_1556_, lean_object* v_a_1557_, lean_object* v_a_1558_, lean_object* v_a_1559_, lean_object* v_a_1560_, lean_object* v_a_1561_, lean_object* v_a_1562_){
_start:
{
lean_object* v___x_1564_; uint8_t v___x_1565_; 
v___x_1564_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_reduceProjStx___closed__1));
lean_inc(v_stx_1555_);
v___x_1565_ = l_Lean_Syntax_isOfKind(v_stx_1555_, v___x_1564_);
if (v___x_1565_ == 0)
{
lean_object* v___x_1566_; 
lean_dec(v_expectedType_x3f_1556_);
lean_dec(v_stx_1555_);
v___x_1566_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Util_TermReduce_elabBeta_spec__0___redArg();
return v___x_1566_;
}
else
{
lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; uint8_t v___x_1572_; lean_object* v___x_1573_; 
v___x_1567_ = lean_unsigned_to_nat(1u);
v___x_1568_ = l_Lean_Syntax_getArg(v_stx_1555_, v___x_1567_);
lean_dec(v_stx_1555_);
v___x_1569_ = lean_box(v___x_1565_);
v___x_1570_ = lean_box(v___x_1565_);
v___x_1571_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTerm___boxed), 11, 4);
lean_closure_set(v___x_1571_, 0, v___x_1568_);
lean_closure_set(v___x_1571_, 1, v_expectedType_x3f_1556_);
lean_closure_set(v___x_1571_, 2, v___x_1569_);
lean_closure_set(v___x_1571_, 3, v___x_1570_);
v___x_1572_ = 2;
v___x_1573_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1571_, v___x_1572_, v_a_1557_, v_a_1558_, v_a_1559_, v_a_1560_, v_a_1561_, v_a_1562_);
if (lean_obj_tag(v___x_1573_) == 0)
{
lean_object* v_a_1574_; uint8_t v___x_1575_; uint8_t v___x_1576_; lean_object* v___x_1577_; 
v_a_1574_ = lean_ctor_get(v___x_1573_, 0);
lean_inc(v_a_1574_);
lean_dec_ref_known(v___x_1573_, 1);
v___x_1575_ = 0;
v___x_1576_ = 0;
v___x_1577_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_1575_, v___x_1576_, v_a_1557_, v_a_1558_, v_a_1559_, v_a_1560_, v_a_1561_, v_a_1562_);
if (lean_obj_tag(v___x_1577_) == 0)
{
lean_object* v___x_1578_; lean_object* v_a_1579_; lean_object* v___f_1580_; lean_object* v___f_1581_; lean_object* v___x_1582_; 
lean_dec_ref_known(v___x_1577_, 1);
v___x_1578_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Util_TermReduce_elabBeta_spec__1___redArg(v_a_1574_, v_a_1560_);
v_a_1579_ = lean_ctor_get(v___x_1578_, 0);
lean_inc(v_a_1579_);
lean_dec_ref(v___x_1578_);
v___f_1580_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__0));
v___f_1581_ = ((lean_object*)(lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___closed__1));
v___x_1582_ = lp_mathlib_Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0(v_a_1579_, v___f_1580_, v___f_1581_, v_a_1557_, v_a_1558_, v_a_1559_, v_a_1560_, v_a_1561_, v_a_1562_);
return v___x_1582_;
}
else
{
lean_object* v_a_1583_; lean_object* v___x_1585_; uint8_t v_isShared_1586_; uint8_t v_isSharedCheck_1590_; 
lean_dec(v_a_1574_);
v_a_1583_ = lean_ctor_get(v___x_1577_, 0);
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1577_);
if (v_isSharedCheck_1590_ == 0)
{
v___x_1585_ = v___x_1577_;
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
else
{
lean_inc(v_a_1583_);
lean_dec(v___x_1577_);
v___x_1585_ = lean_box(0);
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
v_resetjp_1584_:
{
lean_object* v___x_1588_; 
if (v_isShared_1586_ == 0)
{
v___x_1588_ = v___x_1585_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v_a_1583_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
}
else
{
return v___x_1573_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj___boxed(lean_object* v_stx_1591_, lean_object* v_expectedType_x3f_1592_, lean_object* v_a_1593_, lean_object* v_a_1594_, lean_object* v_a_1595_, lean_object* v_a_1596_, lean_object* v_a_1597_, lean_object* v_a_1598_, lean_object* v_a_1599_){
_start:
{
lean_object* v_res_1600_; 
v_res_1600_ = lp_mathlib_Mathlib_Util_TermReduce_elabReduceProj(v_stx_1591_, v_expectedType_x3f_1592_, v_a_1593_, v_a_1594_, v_a_1595_, v_a_1596_, v_a_1597_, v_a_1598_);
lean_dec(v_a_1598_);
lean_dec_ref(v_a_1597_);
lean_dec(v_a_1596_);
lean_dec_ref(v_a_1595_);
lean_dec(v_a_1594_);
lean_dec_ref(v_a_1593_);
return v_res_1600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_1601_, lean_object* v_m_1602_, lean_object* v_a_1603_){
_start:
{
lean_object* v___x_1604_; 
v___x_1604_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___redArg(v_m_1602_, v_a_1603_);
return v___x_1604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_1605_, lean_object* v_m_1606_, lean_object* v_a_1607_){
_start:
{
lean_object* v_res_1608_; 
v_res_1608_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3(v_00_u03b2_1605_, v_m_1606_, v_a_1607_);
lean_dec_ref(v_a_1607_);
lean_dec_ref(v_m_1606_);
return v_res_1608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7(lean_object* v_00_u03b1_1609_, lean_object* v_ref_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
lean_object* v___x_1614_; 
v___x_1614_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___redArg(v_ref_1610_);
return v___x_1614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7___boxed(lean_object* v_00_u03b1_1615_, lean_object* v_ref_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_){
_start:
{
lean_object* v_res_1620_; 
v_res_1620_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__7(v_00_u03b1_1615_, v_ref_1616_, v___y_1617_, v___y_1618_);
lean_dec(v___y_1618_);
lean_dec_ref(v___y_1617_);
return v_res_1620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8(lean_object* v_00_u03b1_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___x_1625_; 
v___x_1625_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___redArg();
return v___x_1625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8___boxed(lean_object* v_00_u03b1_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_){
_start:
{
lean_object* v_res_1630_; 
v_res_1630_ = lp_mathlib_Lean_throwInterruptException___at___00Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5_spec__8(v_00_u03b1_1626_, v___y_1627_, v___y_1628_);
lean_dec(v___y_1628_);
lean_dec_ref(v___y_1627_);
return v_res_1630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5(lean_object* v_00_u03b1_1631_, lean_object* v_x_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_){
_start:
{
lean_object* v___x_1641_; 
v___x_1641_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___redArg(v_x_1632_, v___y_1633_, v___y_1634_, v___y_1635_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_);
return v___x_1641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5___boxed(lean_object* v_00_u03b1_1642_, lean_object* v_x_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_){
_start:
{
lean_object* v_res_1652_; 
v_res_1652_ = lp_mathlib_Lean_Core_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__5(v_00_u03b1_1642_, v_x_1643_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_, v___y_1648_, v___y_1649_, v___y_1650_);
lean_dec(v___y_1650_);
lean_dec_ref(v___y_1649_);
lean_dec(v___y_1648_);
lean_dec_ref(v___y_1647_);
lean_dec(v___y_1646_);
lean_dec_ref(v___y_1645_);
lean_dec(v___y_1644_);
return v_res_1652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6(lean_object* v_00_u03b2_1653_, lean_object* v_m_1654_, lean_object* v_a_1655_, lean_object* v_b_1656_){
_start:
{
lean_object* v___x_1657_; 
v___x_1657_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6___redArg(v_m_1654_, v_a_1655_, v_b_1656_);
return v___x_1657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4(lean_object* v_00_u03b2_1658_, lean_object* v_a_1659_, lean_object* v_x_1660_){
_start:
{
lean_object* v___x_1661_; 
v___x_1661_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___redArg(v_a_1659_, v_x_1660_);
return v___x_1661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4___boxed(lean_object* v_00_u03b2_1662_, lean_object* v_a_1663_, lean_object* v_x_1664_){
_start:
{
lean_object* v_res_1665_; 
v_res_1665_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__3_spec__4(v_00_u03b2_1662_, v_a_1663_, v_x_1664_);
lean_dec(v_x_1664_);
lean_dec_ref(v_a_1663_);
return v_res_1665_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10(lean_object* v_00_u03b2_1666_, lean_object* v_a_1667_, lean_object* v_x_1668_){
_start:
{
uint8_t v___x_1669_; 
v___x_1669_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___redArg(v_a_1667_, v_x_1668_);
return v___x_1669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10___boxed(lean_object* v_00_u03b2_1670_, lean_object* v_a_1671_, lean_object* v_x_1672_){
_start:
{
uint8_t v_res_1673_; lean_object* v_r_1674_; 
v_res_1673_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__10(v_00_u03b2_1670_, v_a_1671_, v_x_1672_);
lean_dec(v_x_1672_);
lean_dec_ref(v_a_1671_);
v_r_1674_ = lean_box(v_res_1673_);
return v_r_1674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11(lean_object* v_00_u03b2_1675_, lean_object* v_data_1676_){
_start:
{
lean_object* v___x_1677_; 
v___x_1677_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11___redArg(v_data_1676_);
return v___x_1677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12(lean_object* v_00_u03b2_1678_, lean_object* v_a_1679_, lean_object* v_b_1680_, lean_object* v_x_1681_){
_start:
{
lean_object* v___x_1682_; 
v___x_1682_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__12___redArg(v_a_1679_, v_b_1680_, v_x_1681_);
return v___x_1682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12(lean_object* v_00_u03b2_1683_, lean_object* v_i_1684_, lean_object* v_source_1685_, lean_object* v_target_1686_){
_start:
{
lean_object* v___x_1687_; 
v___x_1687_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12___redArg(v_i_1684_, v_source_1685_, v_target_1686_);
return v___x_1687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12_spec__13(lean_object* v_00_u03b2_1688_, lean_object* v_x_1689_, lean_object* v_x_1690_){
_start:
{
lean_object* v___x_1691_; 
v___x_1691_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Core_transform_visit___at___00Lean_Core_transform___at___00Mathlib_Util_TermReduce_elabReduceProj_spec__0_spec__0_spec__6_spec__11_spec__12_spec__13___redArg(v_x_1689_, v_x_1690_);
return v___x_1691_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_TermReduce(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Delta(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_TermReduce(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Delta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Delta(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_TermReduce(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Delta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_TermReduce(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_TermReduce(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_TermReduce(builtin);
}
#ifdef __cplusplus
}
#endif
