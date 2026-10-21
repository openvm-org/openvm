// Lean compiler output
// Module: Qq.AssertInstancesCommute
// Imports: public import Init public meta import Init public import Qq.MetaM
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
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* lp_Qq_Qq_Impl_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_Elab_Term_tryPostpone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_mk_array(lean_object*, lean_object*);
extern lean_object* l_Lean_LocalContext_empty;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_Qq_Qq_Impl_unquoteLCtx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_determineLocalInstances(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_hasValue(lean_object*, uint8_t);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* lp_Qq_Qq_Impl_makeDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_quoteLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_quoteExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* lean_name_append_after(lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Qq"};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value;
static const lean_string_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termAssumeInstancesCommute'_"};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__1 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__1_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2_value_aux_0),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(172, 197, 36, 16, 129, 93, 240, 70)}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2_value;
static const lean_string_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__3 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__3_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__4 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__4_value;
static const lean_string_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "assumeInstancesCommute'"};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__5 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__5_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__5_value)}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__6 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__6_value;
static const lean_string_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__7 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__7_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__8 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__8_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__9 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__9_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__4_value),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__6_value),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__9_value)}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__10 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__10_value;
static const lean_ctor_object lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__10_value)}};
static const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__11 = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_termAssumeInstancesCommute_x27__ = (const lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__11_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___closed__0 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Quoted"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__0 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__0_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1_value_aux_0),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(115, 104, 38, 134, 26, 50, 120, 141)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__0;
static lean_once_cell_t lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__1;
static const lean_array_object lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Impl"};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__0_value;
static const lean_string_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "termAssertInstancesCommuteImpl_"};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__1_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 194, 239, 50, 239, 219, 54, 130)}};
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 52, 96, 68, 253, 143, 132, 6)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value;
static const lean_string_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "assertInstancesCommuteImpl"};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__3_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__4 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__4_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__4_value),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__4_value),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__9_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__5 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__5_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__5_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__6 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl__ = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__6_value;
static lean_once_cell_t lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "_eq"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "MetaM"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(66, 183, 9, 249, 10, 54, 148, 83)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "PLift"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__6 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__6_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(199, 82, 227, 164, 10, 97, 128, 84)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__7 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__7_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__9;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "QuotedDefEq"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__10 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__10_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__11_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(117, 77, 146, 58, 70, 164, 104, 0)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__11 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__11_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "withNewMCtxDepth"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__13 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__13_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(152, 233, 69, 104, 117, 58, 129, 215)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__15;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__16;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "instMonadControlTOfPure"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__17 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__17_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(217, 64, 180, 246, 100, 162, 50, 42)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__18 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__18_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8_value)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__19 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__19_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__20;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__21;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Applicative"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__22 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__22_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toPure"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__23 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__23_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__22_value),LEAN_SCALAR_PTR_LITERAL(225, 21, 170, 15, 195, 130, 155, 116)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__24_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__23_value),LEAN_SCALAR_PTR_LITERAL(222, 75, 18, 17, 200, 253, 193, 106)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__24 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__24_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__25;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__26;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ReaderT"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__27 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__27_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "instApplicativeOfMonad"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__28 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__28_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__27_value),LEAN_SCALAR_PTR_LITERAL(67, 31, 123, 97, 193, 212, 110, 40)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__29_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__28_value),LEAN_SCALAR_PTR_LITERAL(240, 83, 123, 188, 25, 206, 11, 223)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__29 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__29_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__30;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Context"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__31 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__31_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__31_value),LEAN_SCALAR_PTR_LITERAL(48, 31, 16, 58, 163, 100, 109, 238)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__33;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__34;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "StateRefT'"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__35 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__35_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__35_value),LEAN_SCALAR_PTR_LITERAL(198, 203, 212, 130, 115, 32, 16, 150)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__36 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__36_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__37;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "IO"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__38 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__38_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "RealWorld"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__39 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__39_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__38_value),LEAN_SCALAR_PTR_LITERAL(2, 76, 19, 202, 4, 69, 238, 60)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__40_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__39_value),LEAN_SCALAR_PTR_LITERAL(186, 219, 141, 147, 12, 104, 171, 84)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__40 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__40_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__42;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "State"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__43 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__43_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__43_value),LEAN_SCALAR_PTR_LITERAL(134, 8, 176, 15, 180, 2, 111, 49)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__46;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Core"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__47 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__47_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "CoreM"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__48 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__48_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__47_value),LEAN_SCALAR_PTR_LITERAL(194, 126, 120, 188, 150, 235, 117, 203)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__48_value),LEAN_SCALAR_PTR_LITERAL(115, 114, 191, 177, 45, 189, 121, 141)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__51;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__52;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instMonad"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__53 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__53_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__35_value),LEAN_SCALAR_PTR_LITERAL(198, 203, 212, 130, 115, 32, 16, 150)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__54_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(227, 254, 242, 24, 106, 88, 158, 153)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__54 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__54_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__55;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__56;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__57;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__58;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instMonadCoreM"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__59 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__59_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__47_value),LEAN_SCALAR_PTR_LITERAL(194, 126, 120, 188, 150, 235, 117, 203)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__59_value),LEAN_SCALAR_PTR_LITERAL(29, 229, 198, 35, 240, 93, 181, 204)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__61;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__62;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__63;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__64;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__66;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instMonadMetaM"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__67 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__67_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__67_value),LEAN_SCALAR_PTR_LITERAL(3, 19, 135, 207, 197, 7, 154, 145)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__70;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "withDefault"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__71 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__71_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__71_value),LEAN_SCALAR_PTR_LITERAL(20, 194, 85, 202, 61, 138, 210, 252)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__73;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__74_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__74;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__75;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__76;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "assertDefEqQ"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__77 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__77_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__78_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__78_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__77_value),LEAN_SCALAR_PTR_LITERAL(178, 63, 224, 92, 152, 169, 195, 191)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__78 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__78_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__79;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__80 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__80_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__81 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__81_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__82_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__80_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__82_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__81_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__82 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__82_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__83;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term_>>=_"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__84 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__84_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__84_value),LEAN_SCALAR_PTR_LITERAL(143, 92, 54, 199, 40, 32, 117, 253)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__85 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__85_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ">>="};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__86 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__86_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__89 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__89_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__89_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__91 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__91_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__91_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__93 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__93_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__93_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__94 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__94_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "__defeqres"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__95 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__95_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__96_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__96;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__95_value),LEAN_SCALAR_PTR_LITERAL(166, 144, 236, 76, 52, 226, 180, 201)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__97 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__97_value;
static lean_once_cell_t lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__99 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__99_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "have"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__100 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__100_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__100_value),LEAN_SCALAR_PTR_LITERAL(55, 91, 239, 116, 115, 0, 62, 115)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__102 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__102_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__102_value),LEAN_SCALAR_PTR_LITERAL(5, 186, 227, 151, 19, 40, 136, 241)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__104 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__104_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__104_value),LEAN_SCALAR_PTR_LITERAL(61, 47, 121, 206, 37, 68, 134, 111)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIdDecl"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__106 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__106_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__106_value),LEAN_SCALAR_PTR_LITERAL(82, 96, 243, 36, 251, 209, 136, 237)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "letId"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__108 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__108_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__108_value),LEAN_SCALAR_PTR_LITERAL(67, 92, 92, 51, 38, 250, 60, 190)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__110 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__110_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__111_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__111 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__111_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__111_value),LEAN_SCALAR_PTR_LITERAL(103, 149, 207, 196, 17, 4, 77, 74)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__113 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__113_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__114_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fieldIdx"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__114 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__114_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__115_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__114_value),LEAN_SCALAR_PTR_LITERAL(243, 141, 165, 29, 238, 211, 61, 163)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__115 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__115_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__116_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__116 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__116_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__2_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__5 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__5_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__6 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "termAssertInstancesCommuteDummy"};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 194, 239, 50, 239, 219, 54, 130)}};
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 113, 144, 86, 170, 187, 62, 0)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "assertInstancesCommuteDummy"};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__2_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__2_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__3_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__4_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy = (const lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__4_value;
static const lean_string_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "assert"};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value_aux_2),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 125, 133, 119, 190, 55, 66, 188)}};
static const lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "termAssumeInstancesCommuteDummy"};
static const lean_object* lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 194, 239, 50, 239, 219, 54, 130)}};
static const lean_ctor_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 51, 29, 214, 0, 205, 204, 203)}};
static const lean_object* lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "assumeInstancesCommuteDummy"};
static const lean_object* lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__2_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__2_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__3_value)}};
static const lean_object* lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__4_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy = (const lean_object*)&lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__4_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_doElemAssertInstancesCommute___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "doElemAssertInstancesCommute"};
static const lean_object* lp_Qq_Qq_doElemAssertInstancesCommute___closed__0 = (const lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_doElemAssertInstancesCommute___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_doElemAssertInstancesCommute___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(252, 162, 155, 32, 109, 198, 86, 176)}};
static const lean_object* lp_Qq_Qq_doElemAssertInstancesCommute___closed__1 = (const lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__1_value;
static const lean_string_object lp_Qq_Qq_doElemAssertInstancesCommute___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "assertInstancesCommute"};
static const lean_object* lp_Qq_Qq_doElemAssertInstancesCommute___closed__2 = (const lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__2_value;
static const lean_ctor_object lp_Qq_Qq_doElemAssertInstancesCommute___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__2_value)}};
static const lean_object* lp_Qq_Qq_doElemAssertInstancesCommute___closed__3 = (const lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_doElemAssertInstancesCommute___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__3_value)}};
static const lean_object* lp_Qq_Qq_doElemAssertInstancesCommute___closed__4 = (const lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__4_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_doElemAssertInstancesCommute = (const lean_object*)&lp_Qq_Qq_doElemAssertInstancesCommute___closed__4_value;
static const lean_string_object lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doAssert"};
static const lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__0 = (const lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__0_value;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__87_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value_aux_1),((lean_object*)&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__88_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value_aux_2),((lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(171, 15, 212, 125, 46, 208, 251, 33)}};
static const lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1 = (const lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1_value;
static const lean_string_object lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "assert!"};
static const lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__2 = (const lean_object*)&lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_doElemAssumeInstancesCommute___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "doElemAssumeInstancesCommute"};
static const lean_object* lp_Qq_Qq_doElemAssumeInstancesCommute___closed__0 = (const lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 160, 7, 69, 149, 206, 60, 225)}};
static const lean_object* lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1 = (const lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1_value;
static const lean_string_object lp_Qq_Qq_doElemAssumeInstancesCommute___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "assumeInstancesCommute"};
static const lean_object* lp_Qq_Qq_doElemAssumeInstancesCommute___closed__2 = (const lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__2_value;
static const lean_ctor_object lp_Qq_Qq_doElemAssumeInstancesCommute___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__2_value)}};
static const lean_object* lp_Qq_Qq_doElemAssumeInstancesCommute___closed__3 = (const lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_doElemAssumeInstancesCommute___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__3_value)}};
static const lean_object* lp_Qq_Qq_doElemAssumeInstancesCommute___closed__4 = (const lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__4_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_doElemAssumeInstancesCommute = (const lean_object*)&lp_Qq_Qq_doElemAssumeInstancesCommute___closed__4_value;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssumeInstancesCommute__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssumeInstancesCommute__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg(lean_object* v_lctx_27_, lean_object* v_localInsts_28_, lean_object* v_x_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_27_, v_localInsts_28_, v_x_29_, v___y_30_, v___y_31_, v___y_32_, v___y_33_);
if (lean_obj_tag(v___x_35_) == 0)
{
lean_object* v_a_36_; lean_object* v___x_38_; uint8_t v_isShared_39_; uint8_t v_isSharedCheck_43_; 
v_a_36_ = lean_ctor_get(v___x_35_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_35_);
if (v_isSharedCheck_43_ == 0)
{
v___x_38_ = v___x_35_;
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
else
{
lean_inc(v_a_36_);
lean_dec(v___x_35_);
v___x_38_ = lean_box(0);
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
v_resetjp_37_:
{
lean_object* v___x_41_; 
if (v_isShared_39_ == 0)
{
v___x_41_ = v___x_38_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_a_36_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
else
{
lean_object* v_a_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_51_; 
v_a_44_ = lean_ctor_get(v___x_35_, 0);
v_isSharedCheck_51_ = !lean_is_exclusive(v___x_35_);
if (v_isSharedCheck_51_ == 0)
{
v___x_46_ = v___x_35_;
v_isShared_47_ = v_isSharedCheck_51_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_a_44_);
lean_dec(v___x_35_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_51_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_49_; 
if (v_isShared_47_ == 0)
{
v___x_49_ = v___x_46_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v_a_44_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg___boxed(lean_object* v_lctx_52_, lean_object* v_localInsts_53_, lean_object* v_x_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg(v_lctx_52_, v_localInsts_53_, v_x_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
lean_dec(v___y_56_);
lean_dec_ref(v___y_55_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0(lean_object* v_00_u03b1_61_, lean_object* v_lctx_62_, lean_object* v_localInsts_63_, lean_object* v_x_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg(v_lctx_62_, v_localInsts_63_, v_x_64_, v___y_65_, v___y_66_, v___y_67_, v___y_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___boxed(lean_object* v_00_u03b1_71_, lean_object* v_lctx_72_, lean_object* v_localInsts_73_, lean_object* v_x_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0(v_00_u03b1_71_, v_lctx_72_, v_localInsts_73_, v_x_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___lam__0(lean_object* v___x_81_, lean_object* v___x_82_, lean_object* v_a_83_, uint8_t v___x_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = l_Lean_Meta_synthInstance_x3f(v___x_81_, v___x_82_, v___y_85_, v___y_86_, v___y_87_, v___y_88_);
if (lean_obj_tag(v___x_90_) == 0)
{
lean_object* v_a_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_125_; 
v_a_91_ = lean_ctor_get(v___x_90_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_90_);
if (v_isSharedCheck_125_ == 0)
{
v___x_93_ = v___x_90_;
v_isShared_94_ = v_isSharedCheck_125_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_a_91_);
lean_dec(v___x_90_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_125_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
if (lean_obj_tag(v_a_91_) == 1)
{
lean_object* v_val_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
lean_del_object(v___x_93_);
v_val_95_ = lean_ctor_get(v_a_91_, 0);
v___x_96_ = l_Lean_LocalDecl_toExpr(v_a_83_);
lean_inc(v_val_95_);
v___x_97_ = lp_Qq_Qq_Impl_makeDefEq(v___x_96_, v_val_95_, v___y_85_, v___y_86_, v___y_87_, v___y_88_);
if (lean_obj_tag(v___x_97_) == 0)
{
lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_112_; 
v_a_98_ = lean_ctor_get(v___x_97_, 0);
v_isSharedCheck_112_ = !lean_is_exclusive(v___x_97_);
if (v_isSharedCheck_112_ == 0)
{
v___x_100_ = v___x_97_;
v_isShared_101_ = v_isSharedCheck_112_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_97_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_112_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
if (lean_obj_tag(v_a_98_) == 0)
{
if (v___x_84_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_104_; 
lean_dec_ref_known(v_a_91_, 1);
v___x_102_ = lean_box(0);
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 0, v___x_102_);
v___x_104_ = v___x_100_;
goto v_reusejp_103_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v___x_102_);
v___x_104_ = v_reuseFailAlloc_105_;
goto v_reusejp_103_;
}
v_reusejp_103_:
{
return v___x_104_;
}
}
else
{
lean_object* v___x_107_; 
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 0, v_a_91_);
v___x_107_ = v___x_100_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v_a_91_);
v___x_107_ = v_reuseFailAlloc_108_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
return v___x_107_;
}
}
}
else
{
lean_object* v___x_110_; 
lean_dec_ref_known(v_a_98_, 1);
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 0, v_a_91_);
v___x_110_ = v___x_100_;
goto v_reusejp_109_;
}
else
{
lean_object* v_reuseFailAlloc_111_; 
v_reuseFailAlloc_111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_111_, 0, v_a_91_);
v___x_110_ = v_reuseFailAlloc_111_;
goto v_reusejp_109_;
}
v_reusejp_109_:
{
return v___x_110_;
}
}
}
}
else
{
lean_object* v_a_113_; lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_120_; 
lean_dec_ref_known(v_a_91_, 1);
v_a_113_ = lean_ctor_get(v___x_97_, 0);
v_isSharedCheck_120_ = !lean_is_exclusive(v___x_97_);
if (v_isSharedCheck_120_ == 0)
{
v___x_115_ = v___x_97_;
v_isShared_116_ = v_isSharedCheck_120_;
goto v_resetjp_114_;
}
else
{
lean_inc(v_a_113_);
lean_dec(v___x_97_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_120_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v___x_118_; 
if (v_isShared_116_ == 0)
{
v___x_118_ = v___x_115_;
goto v_reusejp_117_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v_a_113_);
v___x_118_ = v_reuseFailAlloc_119_;
goto v_reusejp_117_;
}
v_reusejp_117_:
{
return v___x_118_;
}
}
}
}
else
{
lean_object* v___x_121_; lean_object* v___x_123_; 
lean_dec(v_a_91_);
lean_dec_ref(v_a_83_);
v___x_121_ = lean_box(0);
if (v_isShared_94_ == 0)
{
lean_ctor_set(v___x_93_, 0, v___x_121_);
v___x_123_ = v___x_93_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v___x_121_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
else
{
lean_dec_ref(v_a_83_);
return v___x_90_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___lam__0___boxed(lean_object* v___x_126_, lean_object* v___x_127_, lean_object* v_a_128_, lean_object* v___x_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
uint8_t v___x_1777__boxed_135_; lean_object* v_res_136_; 
v___x_1777__boxed_135_ = lean_unbox(v___x_129_);
v_res_136_ = lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___lam__0(v___x_126_, v___x_127_, v_a_128_, v___x_1777__boxed_135_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1(lean_object* v_inst_137_, lean_object* v_as_138_, size_t v_i_139_, size_t v_stop_140_, lean_object* v_b_141_){
_start:
{
lean_object* v___y_143_; uint8_t v___x_147_; 
v___x_147_ = lean_usize_dec_eq(v_i_139_, v_stop_140_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; lean_object* v_fvar_149_; lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_148_ = lean_array_uget_borrowed(v_as_138_, v_i_139_);
v_fvar_149_ = lean_ctor_get(v___x_148_, 1);
lean_inc(v_inst_137_);
v___x_150_ = l_Lean_Expr_fvar___override(v_inst_137_);
v___x_151_ = lean_expr_eqv(v_fvar_149_, v___x_150_);
lean_dec_ref(v___x_150_);
if (v___x_151_ == 0)
{
lean_object* v___x_152_; 
lean_inc(v___x_148_);
v___x_152_ = lean_array_push(v_b_141_, v___x_148_);
v___y_143_ = v___x_152_;
goto v___jp_142_;
}
else
{
v___y_143_ = v_b_141_;
goto v___jp_142_;
}
}
else
{
lean_dec(v_inst_137_);
return v_b_141_;
}
v___jp_142_:
{
size_t v___x_144_; size_t v___x_145_; 
v___x_144_ = ((size_t)1ULL);
v___x_145_ = lean_usize_add(v_i_139_, v___x_144_);
v_i_139_ = v___x_145_;
v_b_141_ = v___y_143_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1___boxed(lean_object* v_inst_153_, lean_object* v_as_154_, lean_object* v_i_155_, lean_object* v_stop_156_, lean_object* v_b_157_){
_start:
{
size_t v_i_boxed_158_; size_t v_stop_boxed_159_; lean_object* v_res_160_; 
v_i_boxed_158_ = lean_unbox_usize(v_i_155_);
lean_dec(v_i_155_);
v_stop_boxed_159_ = lean_unbox_usize(v_stop_156_);
lean_dec(v_stop_156_);
v_res_160_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1(v_inst_153_, v_as_154_, v_i_boxed_158_, v_stop_boxed_159_, v_b_157_);
lean_dec_ref(v_as_154_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f(lean_object* v_inst_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_){
_start:
{
lean_object* v___x_169_; 
lean_inc(v_inst_163_);
v___x_169_ = l_Lean_FVarId_getDecl___redArg(v_inst_163_, v_a_164_, v_a_166_, v_a_167_);
if (lean_obj_tag(v___x_169_) == 0)
{
lean_object* v_a_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_200_; 
v_a_170_ = lean_ctor_get(v___x_169_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_200_ == 0)
{
v___x_172_ = v___x_169_;
v_isShared_173_ = v_isSharedCheck_200_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_a_170_);
lean_dec(v___x_169_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_200_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
uint8_t v___x_174_; uint8_t v___x_175_; 
v___x_174_ = 0;
v___x_175_ = l_Lean_LocalDecl_hasValue(v_a_170_, v___x_174_);
if (v___x_175_ == 0)
{
lean_object* v_lctx_176_; lean_object* v_localInstances_177_; lean_object* v___y_179_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; uint8_t v___x_188_; 
lean_del_object(v___x_172_);
v_lctx_176_ = lean_ctor_get(v_a_164_, 2);
v_localInstances_177_ = lean_ctor_get(v_a_164_, 3);
v___x_185_ = lean_unsigned_to_nat(0u);
v___x_186_ = lean_array_get_size(v_localInstances_177_);
v___x_187_ = ((lean_object*)(lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___closed__0));
v___x_188_ = lean_nat_dec_lt(v___x_185_, v___x_186_);
if (v___x_188_ == 0)
{
lean_dec(v_inst_163_);
v___y_179_ = v___x_187_;
goto v___jp_178_;
}
else
{
uint8_t v___x_189_; 
v___x_189_ = lean_nat_dec_le(v___x_186_, v___x_186_);
if (v___x_189_ == 0)
{
if (v___x_188_ == 0)
{
lean_dec(v_inst_163_);
v___y_179_ = v___x_187_;
goto v___jp_178_;
}
else
{
size_t v___x_190_; size_t v___x_191_; lean_object* v___x_192_; 
v___x_190_ = ((size_t)0ULL);
v___x_191_ = lean_usize_of_nat(v___x_186_);
v___x_192_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1(v_inst_163_, v_localInstances_177_, v___x_190_, v___x_191_, v___x_187_);
v___y_179_ = v___x_192_;
goto v___jp_178_;
}
}
else
{
size_t v___x_193_; size_t v___x_194_; lean_object* v___x_195_; 
v___x_193_ = ((size_t)0ULL);
v___x_194_ = lean_usize_of_nat(v___x_186_);
v___x_195_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__1(v_inst_163_, v_localInstances_177_, v___x_193_, v___x_194_, v___x_187_);
v___y_179_ = v___x_195_;
goto v___jp_178_;
}
}
v___jp_178_:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___f_183_; lean_object* v___x_184_; 
v___x_180_ = l_Lean_LocalDecl_type(v_a_170_);
v___x_181_ = lean_box(0);
v___x_182_ = lean_box(v___x_175_);
v___f_183_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___lam__0___boxed), 9, 4);
lean_closure_set(v___f_183_, 0, v___x_180_);
lean_closure_set(v___f_183_, 1, v___x_181_);
lean_closure_set(v___f_183_, 2, v_a_170_);
lean_closure_set(v___f_183_, 3, v___x_182_);
lean_inc_ref(v_lctx_176_);
v___x_184_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_isRedundantLocalInst_x3f_spec__0___redArg(v_lctx_176_, v___y_179_, v___f_183_, v_a_164_, v_a_165_, v_a_166_, v_a_167_);
return v___x_184_;
}
}
else
{
lean_object* v___x_196_; lean_object* v___x_198_; 
lean_dec(v_a_170_);
lean_dec(v_inst_163_);
v___x_196_ = lean_box(0);
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 0, v___x_196_);
v___x_198_ = v___x_172_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_196_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_dec(v_inst_163_);
v_a_201_ = lean_ctor_get(v___x_169_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_169_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_169_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_isRedundantLocalInst_x3f___boxed(lean_object* v_inst_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_Qq_Qq_Impl_isRedundantLocalInst_x3f(v_inst_209_, v_a_210_, v_a_211_, v_a_212_, v_a_213_);
lean_dec(v_a_213_);
lean_dec_ref(v_a_212_);
lean_dec(v_a_211_);
lean_dec_ref(v_a_210_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___lam__0(lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_){
_start:
{
lean_object* v_localInstances_222_; lean_object* v___x_223_; 
v_localInstances_222_ = lean_ctor_get(v___y_217_, 3);
lean_inc_ref(v_localInstances_222_);
v___x_223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_223_, 0, v_localInstances_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___lam__0___boxed(lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___lam__0(v___y_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_);
lean_dec(v___y_228_);
lean_dec_ref(v___y_227_);
lean_dec(v___y_226_);
lean_dec_ref(v___y_225_);
lean_dec_ref(v___y_224_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___lam__0(lean_object* v___x_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_Qq_Qq_Impl_isRedundantLocalInst_x3f(v___x_231_, v___y_233_, v___y_234_, v___y_235_, v___y_236_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___lam__0___boxed(lean_object* v___x_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___lam__0(v___x_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
lean_dec(v___y_244_);
lean_dec_ref(v___y_243_);
lean_dec(v___y_242_);
lean_dec_ref(v___y_241_);
lean_dec_ref(v___y_240_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___lam__0(lean_object* v_x_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
lean_object* v___x_254_; 
lean_inc_ref(v___y_248_);
v___x_254_ = lean_apply_6(v_x_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, lean_box(0));
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_x_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___lam__0(v_x_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_, v___y_260_);
lean_dec_ref(v___y_256_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg(lean_object* v_lctx_263_, lean_object* v_localInsts_264_, lean_object* v_x_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
lean_object* v___f_272_; lean_object* v___x_273_; 
lean_inc_ref(v___y_266_);
v___f_272_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_272_, 0, v_x_265_);
lean_closure_set(v___f_272_, 1, v___y_266_);
v___x_273_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_263_, v_localInsts_264_, v___f_272_, v___y_267_, v___y_268_, v___y_269_, v___y_270_);
if (lean_obj_tag(v___x_273_) == 0)
{
return v___x_273_;
}
else
{
lean_object* v_a_274_; lean_object* v___x_276_; uint8_t v_isShared_277_; uint8_t v_isSharedCheck_281_; 
v_a_274_ = lean_ctor_get(v___x_273_, 0);
v_isSharedCheck_281_ = !lean_is_exclusive(v___x_273_);
if (v_isSharedCheck_281_ == 0)
{
v___x_276_ = v___x_273_;
v_isShared_277_ = v_isSharedCheck_281_;
goto v_resetjp_275_;
}
else
{
lean_inc(v_a_274_);
lean_dec(v___x_273_);
v___x_276_ = lean_box(0);
v_isShared_277_ = v_isSharedCheck_281_;
goto v_resetjp_275_;
}
v_resetjp_275_:
{
lean_object* v___x_279_; 
if (v_isShared_277_ == 0)
{
v___x_279_ = v___x_276_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v_a_274_);
v___x_279_ = v_reuseFailAlloc_280_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
return v___x_279_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_lctx_282_, lean_object* v_localInsts_283_, lean_object* v_x_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg(v_lctx_282_, v_localInsts_283_, v_x_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
lean_dec(v___y_289_);
lean_dec_ref(v___y_288_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
lean_dec_ref(v___y_285_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg(lean_object* v_k_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_){
_start:
{
lean_object* v_unquoted_299_; lean_object* v___x_300_; 
v_unquoted_299_ = lean_ctor_get(v___y_293_, 3);
v___x_300_ = lp_Qq_Qq_Impl_determineLocalInstances(v_unquoted_299_, v___y_294_, v___y_295_, v___y_296_, v___y_297_);
if (lean_obj_tag(v___x_300_) == 0)
{
lean_object* v_a_301_; lean_object* v___x_302_; 
v_a_301_ = lean_ctor_get(v___x_300_, 0);
lean_inc(v_a_301_);
lean_dec_ref_known(v___x_300_, 1);
lean_inc_ref(v_unquoted_299_);
v___x_302_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg(v_unquoted_299_, v_a_301_, v_k_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_);
return v___x_302_;
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
lean_dec_ref(v_k_292_);
v_a_303_ = lean_ctor_get(v___x_300_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_300_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_300_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_300_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg___boxed(lean_object* v_k_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg(v_k_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_);
lean_dec(v___y_316_);
lean_dec_ref(v___y_315_);
lean_dec(v___y_314_);
lean_dec_ref(v___y_313_);
lean_dec_ref(v___y_312_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg(lean_object* v_a_319_, lean_object* v_x_320_){
_start:
{
if (lean_obj_tag(v_x_320_) == 0)
{
lean_object* v___x_321_; 
v___x_321_ = lean_box(0);
return v___x_321_;
}
else
{
lean_object* v_key_322_; lean_object* v_value_323_; lean_object* v_tail_324_; uint8_t v___x_325_; 
v_key_322_ = lean_ctor_get(v_x_320_, 0);
v_value_323_ = lean_ctor_get(v_x_320_, 1);
v_tail_324_ = lean_ctor_get(v_x_320_, 2);
v___x_325_ = lean_expr_eqv(v_key_322_, v_a_319_);
if (v___x_325_ == 0)
{
v_x_320_ = v_tail_324_;
goto _start;
}
else
{
lean_object* v___x_327_; 
lean_inc(v_value_323_);
v___x_327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_327_, 0, v_value_323_);
return v___x_327_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg___boxed(lean_object* v_a_328_, lean_object* v_x_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg(v_a_328_, v_x_329_);
lean_dec(v_x_329_);
lean_dec_ref(v_a_328_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg(lean_object* v_m_331_, lean_object* v_a_332_){
_start:
{
lean_object* v_buckets_333_; lean_object* v___x_334_; uint64_t v___x_335_; uint64_t v___x_336_; uint64_t v___x_337_; uint64_t v_fold_338_; uint64_t v___x_339_; uint64_t v___x_340_; uint64_t v___x_341_; size_t v___x_342_; size_t v___x_343_; size_t v___x_344_; size_t v___x_345_; size_t v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v_buckets_333_ = lean_ctor_get(v_m_331_, 1);
v___x_334_ = lean_array_get_size(v_buckets_333_);
v___x_335_ = l_Lean_Expr_hash(v_a_332_);
v___x_336_ = 32ULL;
v___x_337_ = lean_uint64_shift_right(v___x_335_, v___x_336_);
v_fold_338_ = lean_uint64_xor(v___x_335_, v___x_337_);
v___x_339_ = 16ULL;
v___x_340_ = lean_uint64_shift_right(v_fold_338_, v___x_339_);
v___x_341_ = lean_uint64_xor(v_fold_338_, v___x_340_);
v___x_342_ = lean_uint64_to_usize(v___x_341_);
v___x_343_ = lean_usize_of_nat(v___x_334_);
v___x_344_ = ((size_t)1ULL);
v___x_345_ = lean_usize_sub(v___x_343_, v___x_344_);
v___x_346_ = lean_usize_land(v___x_342_, v___x_345_);
v___x_347_ = lean_array_uget_borrowed(v_buckets_333_, v___x_346_);
v___x_348_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg(v_a_332_, v___x_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg___boxed(lean_object* v_m_349_, lean_object* v_a_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg(v_m_349_, v_a_350_);
lean_dec_ref(v_a_350_);
lean_dec_ref(v_m_349_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2(lean_object* v_as_355_, size_t v_sz_356_, size_t v_i_357_, lean_object* v_b_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_a_366_; uint8_t v___x_370_; 
v___x_370_ = lean_usize_dec_lt(v_i_357_, v_sz_356_);
if (v___x_370_ == 0)
{
lean_object* v___x_371_; 
v___x_371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_371_, 0, v_b_358_);
return v___x_371_;
}
else
{
lean_object* v_a_372_; lean_object* v_fvar_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_434_; 
lean_dec_ref(v_b_358_);
v_a_372_ = lean_array_uget(v_as_355_, v_i_357_);
v_fvar_373_ = lean_ctor_get(v_a_372_, 1);
v_isSharedCheck_434_ = !lean_is_exclusive(v_a_372_);
if (v_isSharedCheck_434_ == 0)
{
lean_object* v_unused_435_; 
v_unused_435_ = lean_ctor_get(v_a_372_, 0);
lean_dec(v_unused_435_);
v___x_375_ = v_a_372_;
v_isShared_376_ = v_isSharedCheck_434_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_fvar_373_);
lean_dec(v_a_372_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_434_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
lean_object* v_exprBackSubst_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
v_exprBackSubst_377_ = lean_ctor_get(v___y_359_, 4);
v___x_378_ = lean_box(0);
v___x_379_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___closed__0));
v___x_380_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg(v_exprBackSubst_377_, v_fvar_373_);
if (lean_obj_tag(v___x_380_) == 1)
{
lean_object* v_val_381_; lean_object* v___x_383_; uint8_t v_isShared_384_; uint8_t v_isSharedCheck_433_; 
v_val_381_ = lean_ctor_get(v___x_380_, 0);
v_isSharedCheck_433_ = !lean_is_exclusive(v___x_380_);
if (v_isSharedCheck_433_ == 0)
{
v___x_383_ = v___x_380_;
v_isShared_384_ = v_isSharedCheck_433_;
goto v_resetjp_382_;
}
else
{
lean_inc(v_val_381_);
lean_dec(v___x_380_);
v___x_383_ = lean_box(0);
v_isShared_384_ = v_isSharedCheck_433_;
goto v_resetjp_382_;
}
v_resetjp_382_:
{
if (lean_obj_tag(v_val_381_) == 0)
{
lean_object* v_e_385_; 
v_e_385_ = lean_ctor_get(v_val_381_, 0);
lean_inc_ref(v_e_385_);
lean_dec_ref_known(v_val_381_, 1);
if (lean_obj_tag(v_e_385_) == 1)
{
lean_object* v_fvarId_386_; lean_object* v___x_387_; 
v_fvarId_386_ = lean_ctor_get(v_e_385_, 0);
lean_inc(v_fvarId_386_);
lean_dec_ref_known(v_e_385_, 1);
v___x_387_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_386_, v___y_360_, v___y_362_, v___y_363_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v_a_388_; uint8_t v___x_389_; uint8_t v___x_390_; 
v_a_388_ = lean_ctor_get(v___x_387_, 0);
lean_inc(v_a_388_);
lean_dec_ref_known(v___x_387_, 1);
v___x_389_ = 0;
v___x_390_ = l_Lean_LocalDecl_hasValue(v_a_388_, v___x_389_);
lean_dec(v_a_388_);
if (v___x_390_ == 0)
{
lean_object* v___x_391_; lean_object* v___f_392_; lean_object* v___x_393_; 
v___x_391_ = l_Lean_Expr_fvarId_x21(v_fvar_373_);
lean_dec_ref(v_fvar_373_);
lean_inc(v___x_391_);
v___f_392_ = lean_alloc_closure((void*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___lam__0___boxed), 7, 1);
lean_closure_set(v___f_392_, 0, v___x_391_);
v___x_393_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg(v___f_392_, v___y_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_);
if (lean_obj_tag(v___x_393_) == 0)
{
lean_object* v_a_394_; lean_object* v___x_396_; uint8_t v_isShared_397_; uint8_t v_isSharedCheck_416_; 
v_a_394_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_416_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_416_ == 0)
{
v___x_396_ = v___x_393_;
v_isShared_397_ = v_isSharedCheck_416_;
goto v_resetjp_395_;
}
else
{
lean_inc(v_a_394_);
lean_dec(v___x_393_);
v___x_396_ = lean_box(0);
v_isShared_397_ = v_isSharedCheck_416_;
goto v_resetjp_395_;
}
v_resetjp_395_:
{
if (lean_obj_tag(v_a_394_) == 1)
{
lean_object* v_val_398_; lean_object* v___x_400_; uint8_t v_isShared_401_; uint8_t v_isSharedCheck_415_; 
v_val_398_ = lean_ctor_get(v_a_394_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v_a_394_);
if (v_isSharedCheck_415_ == 0)
{
v___x_400_ = v_a_394_;
v_isShared_401_ = v_isSharedCheck_415_;
goto v_resetjp_399_;
}
else
{
lean_inc(v_val_398_);
lean_dec(v_a_394_);
v___x_400_ = lean_box(0);
v_isShared_401_ = v_isSharedCheck_415_;
goto v_resetjp_399_;
}
v_resetjp_399_:
{
lean_object* v___x_403_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 1, v_val_398_);
lean_ctor_set(v___x_375_, 0, v___x_391_);
v___x_403_ = v___x_375_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v___x_391_);
lean_ctor_set(v_reuseFailAlloc_414_, 1, v_val_398_);
v___x_403_ = v_reuseFailAlloc_414_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
lean_object* v___x_405_; 
if (v_isShared_401_ == 0)
{
lean_ctor_set(v___x_400_, 0, v___x_403_);
v___x_405_ = v___x_400_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v___x_403_);
v___x_405_ = v_reuseFailAlloc_413_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
lean_object* v___x_407_; 
if (v_isShared_384_ == 0)
{
lean_ctor_set(v___x_383_, 0, v___x_405_);
v___x_407_ = v___x_383_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_405_);
v___x_407_ = v_reuseFailAlloc_412_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
lean_object* v___x_408_; lean_object* v___x_410_; 
v___x_408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_407_);
lean_ctor_set(v___x_408_, 1, v___x_378_);
if (v_isShared_397_ == 0)
{
lean_ctor_set(v___x_396_, 0, v___x_408_);
v___x_410_ = v___x_396_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v___x_408_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_396_);
lean_dec(v_a_394_);
lean_dec(v___x_391_);
lean_del_object(v___x_383_);
lean_del_object(v___x_375_);
v_a_366_ = v___x_379_;
goto v___jp_365_;
}
}
}
else
{
lean_object* v_a_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_424_; 
lean_dec(v___x_391_);
lean_del_object(v___x_383_);
lean_del_object(v___x_375_);
v_a_417_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_424_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_424_ == 0)
{
v___x_419_ = v___x_393_;
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_a_417_);
lean_dec(v___x_393_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
lean_object* v___x_422_; 
if (v_isShared_420_ == 0)
{
v___x_422_ = v___x_419_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_a_417_);
v___x_422_ = v_reuseFailAlloc_423_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
return v___x_422_;
}
}
}
}
else
{
lean_del_object(v___x_383_);
lean_del_object(v___x_375_);
lean_dec_ref(v_fvar_373_);
v_a_366_ = v___x_379_;
goto v___jp_365_;
}
}
else
{
lean_object* v_a_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_432_; 
lean_del_object(v___x_383_);
lean_del_object(v___x_375_);
lean_dec_ref(v_fvar_373_);
v_a_425_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_432_ == 0)
{
v___x_427_ = v___x_387_;
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_a_425_);
lean_dec(v___x_387_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v___x_430_; 
if (v_isShared_428_ == 0)
{
v___x_430_ = v___x_427_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v_a_425_);
v___x_430_ = v_reuseFailAlloc_431_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
return v___x_430_;
}
}
}
}
else
{
lean_dec_ref(v_e_385_);
lean_del_object(v___x_383_);
lean_del_object(v___x_375_);
lean_dec_ref(v_fvar_373_);
v_a_366_ = v___x_379_;
goto v___jp_365_;
}
}
else
{
lean_del_object(v___x_383_);
lean_dec(v_val_381_);
lean_del_object(v___x_375_);
lean_dec_ref(v_fvar_373_);
v_a_366_ = v___x_379_;
goto v___jp_365_;
}
}
}
else
{
lean_dec(v___x_380_);
lean_del_object(v___x_375_);
lean_dec_ref(v_fvar_373_);
v_a_366_ = v___x_379_;
goto v___jp_365_;
}
}
}
v___jp_365_:
{
size_t v___x_367_; size_t v___x_368_; 
v___x_367_ = ((size_t)1ULL);
v___x_368_ = lean_usize_add(v_i_357_, v___x_367_);
lean_inc_ref(v_a_366_);
v_i_357_ = v___x_368_;
v_b_358_ = v_a_366_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___boxed(lean_object* v_as_436_, lean_object* v_sz_437_, lean_object* v_i_438_, lean_object* v_b_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
size_t v_sz_boxed_446_; size_t v_i_boxed_447_; lean_object* v_res_448_; 
v_sz_boxed_446_ = lean_unbox_usize(v_sz_437_);
lean_dec(v_sz_437_);
v_i_boxed_447_ = lean_unbox_usize(v_i_438_);
lean_dec(v_i_438_);
v_res_448_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2(v_as_436_, v_sz_boxed_446_, v_i_boxed_447_, v_b_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec_ref(v___y_440_);
lean_dec_ref(v_as_436_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f(lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v___f_456_; lean_object* v___x_457_; 
v___f_456_ = ((lean_object*)(lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___closed__0));
v___x_457_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg(v___f_456_, v_a_450_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
if (lean_obj_tag(v___x_457_) == 0)
{
lean_object* v_a_458_; lean_object* v___x_459_; lean_object* v___x_460_; size_t v_sz_461_; size_t v___x_462_; lean_object* v___x_463_; 
v_a_458_ = lean_ctor_get(v___x_457_, 0);
lean_inc(v_a_458_);
lean_dec_ref_known(v___x_457_, 1);
v___x_459_ = lean_box(0);
v___x_460_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2___closed__0));
v_sz_461_ = lean_array_size(v_a_458_);
v___x_462_ = ((size_t)0ULL);
v___x_463_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__2(v_a_458_, v_sz_461_, v___x_462_, v___x_460_, v_a_450_, v_a_451_, v_a_452_, v_a_453_, v_a_454_);
lean_dec(v_a_458_);
if (lean_obj_tag(v___x_463_) == 0)
{
lean_object* v_a_464_; lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_476_; 
v_a_464_ = lean_ctor_get(v___x_463_, 0);
v_isSharedCheck_476_ = !lean_is_exclusive(v___x_463_);
if (v_isSharedCheck_476_ == 0)
{
v___x_466_ = v___x_463_;
v_isShared_467_ = v_isSharedCheck_476_;
goto v_resetjp_465_;
}
else
{
lean_inc(v_a_464_);
lean_dec(v___x_463_);
v___x_466_ = lean_box(0);
v_isShared_467_ = v_isSharedCheck_476_;
goto v_resetjp_465_;
}
v_resetjp_465_:
{
lean_object* v_fst_468_; 
v_fst_468_ = lean_ctor_get(v_a_464_, 0);
lean_inc(v_fst_468_);
lean_dec(v_a_464_);
if (lean_obj_tag(v_fst_468_) == 0)
{
lean_object* v___x_470_; 
if (v_isShared_467_ == 0)
{
lean_ctor_set(v___x_466_, 0, v___x_459_);
v___x_470_ = v___x_466_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v___x_459_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
else
{
lean_object* v_val_472_; lean_object* v___x_474_; 
v_val_472_ = lean_ctor_get(v_fst_468_, 0);
lean_inc(v_val_472_);
lean_dec_ref_known(v_fst_468_, 1);
if (v_isShared_467_ == 0)
{
lean_ctor_set(v___x_466_, 0, v_val_472_);
v___x_474_ = v___x_466_;
goto v_reusejp_473_;
}
else
{
lean_object* v_reuseFailAlloc_475_; 
v_reuseFailAlloc_475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_475_, 0, v_val_472_);
v___x_474_ = v_reuseFailAlloc_475_;
goto v_reusejp_473_;
}
v_reusejp_473_:
{
return v___x_474_;
}
}
}
}
else
{
lean_object* v_a_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_484_; 
v_a_477_ = lean_ctor_get(v___x_463_, 0);
v_isSharedCheck_484_ = !lean_is_exclusive(v___x_463_);
if (v_isSharedCheck_484_ == 0)
{
v___x_479_ = v___x_463_;
v_isShared_480_ = v_isSharedCheck_484_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_a_477_);
lean_dec(v___x_463_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_484_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_482_; 
if (v_isShared_480_ == 0)
{
v___x_482_ = v___x_479_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v_a_477_);
v___x_482_ = v_reuseFailAlloc_483_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
return v___x_482_;
}
}
}
}
else
{
lean_object* v_a_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_492_; 
v_a_485_ = lean_ctor_get(v___x_457_, 0);
v_isSharedCheck_492_ = !lean_is_exclusive(v___x_457_);
if (v_isSharedCheck_492_ == 0)
{
v___x_487_ = v___x_457_;
v_isShared_488_ = v_isSharedCheck_492_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_a_485_);
lean_dec(v___x_457_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_492_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_490_; 
if (v_isShared_488_ == 0)
{
v___x_490_ = v___x_487_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v_a_485_);
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
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInst_x3f___boxed(lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_, lean_object* v_a_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_Qq_Qq_Impl_findRedundantLocalInst_x3f(v_a_493_, v_a_494_, v_a_495_, v_a_496_, v_a_497_);
lean_dec(v_a_497_);
lean_dec_ref(v_a_496_);
lean_dec(v_a_495_);
lean_dec_ref(v_a_494_);
lean_dec_ref(v_a_493_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0(lean_object* v_00_u03b1_500_, lean_object* v_lctx_501_, lean_object* v_localInsts_502_, lean_object* v_x_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___redArg(v_lctx_501_, v_localInsts_502_, v_x_503_, v___y_504_, v___y_505_, v___y_506_, v___y_507_, v___y_508_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0___boxed(lean_object* v_00_u03b1_511_, lean_object* v_lctx_512_, lean_object* v_localInsts_513_, lean_object* v_x_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0_spec__0(v_00_u03b1_511_, v_lctx_512_, v_localInsts_513_, v_x_514_, v___y_515_, v___y_516_, v___y_517_, v___y_518_, v___y_519_);
lean_dec(v___y_519_);
lean_dec_ref(v___y_518_);
lean_dec(v___y_517_);
lean_dec_ref(v___y_516_);
lean_dec_ref(v___y_515_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0(lean_object* v_00_u03b1_522_, lean_object* v_k_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___redArg(v_k_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0___boxed(lean_object* v_00_u03b1_531_, lean_object* v_k_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__0(v_00_u03b1_531_, v_k_532_, v___y_533_, v___y_534_, v___y_535_, v___y_536_, v___y_537_);
lean_dec(v___y_537_);
lean_dec_ref(v___y_536_);
lean_dec(v___y_535_);
lean_dec_ref(v___y_534_);
lean_dec_ref(v___y_533_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1(lean_object* v_00_u03b2_540_, lean_object* v_m_541_, lean_object* v_a_542_){
_start:
{
lean_object* v___x_543_; 
v___x_543_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___redArg(v_m_541_, v_a_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1___boxed(lean_object* v_00_u03b2_544_, lean_object* v_m_545_, lean_object* v_a_546_){
_start:
{
lean_object* v_res_547_; 
v_res_547_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1(v_00_u03b2_544_, v_m_545_, v_a_546_);
lean_dec_ref(v_a_546_);
lean_dec_ref(v_m_545_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2(lean_object* v_00_u03b2_548_, lean_object* v_a_549_, lean_object* v_x_550_){
_start:
{
lean_object* v___x_551_; 
v___x_551_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___redArg(v_a_549_, v_x_550_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2___boxed(lean_object* v_00_u03b2_552_, lean_object* v_a_553_, lean_object* v_x_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_findRedundantLocalInst_x3f_spec__1_spec__2(v_00_u03b2_552_, v_a_553_, v_x_554_);
lean_dec(v_x_554_);
lean_dec_ref(v_a_553_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(lean_object* v_e_556_, lean_object* v___y_557_){
_start:
{
uint8_t v___x_559_; 
v___x_559_ = l_Lean_Expr_hasMVar(v_e_556_);
if (v___x_559_ == 0)
{
lean_object* v___x_560_; 
v___x_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_560_, 0, v_e_556_);
return v___x_560_;
}
else
{
lean_object* v___x_561_; lean_object* v_mctx_562_; lean_object* v___x_563_; lean_object* v_fst_564_; lean_object* v_snd_565_; lean_object* v___x_566_; lean_object* v_cache_567_; lean_object* v_zetaDeltaFVarIds_568_; lean_object* v_postponed_569_; lean_object* v_diag_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_579_; 
v___x_561_ = lean_st_ref_get(v___y_557_);
v_mctx_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc_ref(v_mctx_562_);
lean_dec(v___x_561_);
v___x_563_ = l_Lean_instantiateMVarsCore(v_mctx_562_, v_e_556_);
v_fst_564_ = lean_ctor_get(v___x_563_, 0);
lean_inc(v_fst_564_);
v_snd_565_ = lean_ctor_get(v___x_563_, 1);
lean_inc(v_snd_565_);
lean_dec_ref(v___x_563_);
v___x_566_ = lean_st_ref_take(v___y_557_);
v_cache_567_ = lean_ctor_get(v___x_566_, 1);
v_zetaDeltaFVarIds_568_ = lean_ctor_get(v___x_566_, 2);
v_postponed_569_ = lean_ctor_get(v___x_566_, 3);
v_diag_570_ = lean_ctor_get(v___x_566_, 4);
v_isSharedCheck_579_ = !lean_is_exclusive(v___x_566_);
if (v_isSharedCheck_579_ == 0)
{
lean_object* v_unused_580_; 
v_unused_580_ = lean_ctor_get(v___x_566_, 0);
lean_dec(v_unused_580_);
v___x_572_ = v___x_566_;
v_isShared_573_ = v_isSharedCheck_579_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_diag_570_);
lean_inc(v_postponed_569_);
lean_inc(v_zetaDeltaFVarIds_568_);
lean_inc(v_cache_567_);
lean_dec(v___x_566_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_579_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_575_; 
if (v_isShared_573_ == 0)
{
lean_ctor_set(v___x_572_, 0, v_snd_565_);
v___x_575_ = v___x_572_;
goto v_reusejp_574_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_snd_565_);
lean_ctor_set(v_reuseFailAlloc_578_, 1, v_cache_567_);
lean_ctor_set(v_reuseFailAlloc_578_, 2, v_zetaDeltaFVarIds_568_);
lean_ctor_set(v_reuseFailAlloc_578_, 3, v_postponed_569_);
lean_ctor_set(v_reuseFailAlloc_578_, 4, v_diag_570_);
v___x_575_ = v_reuseFailAlloc_578_;
goto v_reusejp_574_;
}
v_reusejp_574_:
{
lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_576_ = lean_st_ref_set(v___y_557_, v___x_575_);
v___x_577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_577_, 0, v_fst_564_);
return v___x_577_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg___boxed(lean_object* v_e_581_, lean_object* v___y_582_, lean_object* v___y_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(v_e_581_, v___y_582_);
lean_dec(v___y_582_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0(lean_object* v_e_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(v_e_585_, v___y_589_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___boxed(lean_object* v_e_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_){
_start:
{
lean_object* v_res_602_; 
v_res_602_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0(v_e_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_, v___y_600_);
lean_dec(v___y_600_);
lean_dec_ref(v___y_599_);
lean_dec(v___y_598_);
lean_dec_ref(v___y_597_);
lean_dec(v___y_596_);
lean_dec_ref(v___y_595_);
return v_res_602_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__0(lean_object* v___x_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lean_infer_type(v___x_603_, v___y_605_, v___y_606_, v___y_607_, v___y_608_);
if (lean_obj_tag(v___x_610_) == 0)
{
lean_object* v_a_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_619_; 
v_a_611_ = lean_ctor_get(v___x_610_, 0);
v_isSharedCheck_619_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_619_ == 0)
{
v___x_613_ = v___x_610_;
v_isShared_614_ = v_isSharedCheck_619_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_a_611_);
lean_dec(v___x_610_);
v___x_613_ = lean_box(0);
v_isShared_614_ = v_isSharedCheck_619_;
goto v_resetjp_612_;
}
v_resetjp_612_:
{
lean_object* v___x_615_; lean_object* v___x_617_; 
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v_a_611_);
lean_ctor_set(v___x_615_, 1, v___y_604_);
if (v_isShared_614_ == 0)
{
lean_ctor_set(v___x_613_, 0, v___x_615_);
v___x_617_ = v___x_613_;
goto v_reusejp_616_;
}
else
{
lean_object* v_reuseFailAlloc_618_; 
v_reuseFailAlloc_618_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_618_, 0, v___x_615_);
v___x_617_ = v_reuseFailAlloc_618_;
goto v_reusejp_616_;
}
v_reusejp_616_:
{
return v___x_617_;
}
}
}
else
{
lean_object* v_a_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_627_; 
lean_dec_ref(v___y_604_);
v_a_620_ = lean_ctor_get(v___x_610_, 0);
v_isSharedCheck_627_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_627_ == 0)
{
v___x_622_ = v___x_610_;
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_a_620_);
lean_dec(v___x_610_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_625_; 
if (v_isShared_623_ == 0)
{
v___x_625_ = v___x_622_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v_a_620_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__0___boxed(lean_object* v___x_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__0(v___x_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_, v___y_633_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__1(lean_object* v_fst_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_){
_start:
{
lean_object* v___x_643_; 
v___x_643_ = l_Lean_Meta_getLevel(v_fst_636_, v___y_638_, v___y_639_, v___y_640_, v___y_641_);
if (lean_obj_tag(v___x_643_) == 0)
{
lean_object* v_a_644_; lean_object* v___x_646_; uint8_t v_isShared_647_; uint8_t v_isSharedCheck_652_; 
v_a_644_ = lean_ctor_get(v___x_643_, 0);
v_isSharedCheck_652_ = !lean_is_exclusive(v___x_643_);
if (v_isSharedCheck_652_ == 0)
{
v___x_646_ = v___x_643_;
v_isShared_647_ = v_isSharedCheck_652_;
goto v_resetjp_645_;
}
else
{
lean_inc(v_a_644_);
lean_dec(v___x_643_);
v___x_646_ = lean_box(0);
v_isShared_647_ = v_isSharedCheck_652_;
goto v_resetjp_645_;
}
v_resetjp_645_:
{
lean_object* v___x_648_; lean_object* v___x_650_; 
v___x_648_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_648_, 0, v_a_644_);
lean_ctor_set(v___x_648_, 1, v___y_637_);
if (v_isShared_647_ == 0)
{
lean_ctor_set(v___x_646_, 0, v___x_648_);
v___x_650_ = v___x_646_;
goto v_reusejp_649_;
}
else
{
lean_object* v_reuseFailAlloc_651_; 
v_reuseFailAlloc_651_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_651_, 0, v___x_648_);
v___x_650_ = v_reuseFailAlloc_651_;
goto v_reusejp_649_;
}
v_reusejp_649_:
{
return v___x_650_;
}
}
}
else
{
lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_660_; 
lean_dec_ref(v___y_637_);
v_a_653_ = lean_ctor_get(v___x_643_, 0);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_643_);
if (v_isSharedCheck_660_ == 0)
{
v___x_655_ = v___x_643_;
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_dec(v___x_643_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___x_658_; 
if (v_isShared_656_ == 0)
{
v___x_658_ = v___x_655_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v_a_653_);
v___x_658_ = v_reuseFailAlloc_659_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
return v___x_658_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__1___boxed(lean_object* v_fst_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_){
_start:
{
lean_object* v_res_668_; 
v_res_668_ = lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__1(v_fst_661_, v___y_662_, v___y_663_, v___y_664_, v___y_665_, v___y_666_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
lean_dec(v___y_664_);
lean_dec_ref(v___y_663_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___lam__0(lean_object* v_x_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v___x_676_; 
v___x_676_ = lean_apply_6(v_x_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_, v___y_674_, lean_box(0));
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___lam__0___boxed(lean_object* v_x_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___lam__0(v_x_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg(lean_object* v_lctx_685_, lean_object* v_localInsts_686_, lean_object* v_x_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_){
_start:
{
lean_object* v___f_694_; lean_object* v___x_695_; 
v___f_694_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_694_, 0, v_x_687_);
lean_closure_set(v___f_694_, 1, v___y_688_);
v___x_695_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_685_, v_localInsts_686_, v___f_694_, v___y_689_, v___y_690_, v___y_691_, v___y_692_);
if (lean_obj_tag(v___x_695_) == 0)
{
lean_object* v_a_696_; lean_object* v___x_698_; uint8_t v_isShared_699_; uint8_t v_isSharedCheck_703_; 
v_a_696_ = lean_ctor_get(v___x_695_, 0);
v_isSharedCheck_703_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_703_ == 0)
{
v___x_698_ = v___x_695_;
v_isShared_699_ = v_isSharedCheck_703_;
goto v_resetjp_697_;
}
else
{
lean_inc(v_a_696_);
lean_dec(v___x_695_);
v___x_698_ = lean_box(0);
v_isShared_699_ = v_isSharedCheck_703_;
goto v_resetjp_697_;
}
v_resetjp_697_:
{
lean_object* v___x_701_; 
if (v_isShared_699_ == 0)
{
v___x_701_ = v___x_698_;
goto v_reusejp_700_;
}
else
{
lean_object* v_reuseFailAlloc_702_; 
v_reuseFailAlloc_702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_702_, 0, v_a_696_);
v___x_701_ = v_reuseFailAlloc_702_;
goto v_reusejp_700_;
}
v_reusejp_700_:
{
return v___x_701_;
}
}
}
else
{
lean_object* v_a_704_; lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_711_; 
v_a_704_ = lean_ctor_get(v___x_695_, 0);
v_isSharedCheck_711_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_711_ == 0)
{
v___x_706_ = v___x_695_;
v_isShared_707_ = v_isSharedCheck_711_;
goto v_resetjp_705_;
}
else
{
lean_inc(v_a_704_);
lean_dec(v___x_695_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_711_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_709_; 
if (v_isShared_707_ == 0)
{
v___x_709_ = v___x_706_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_710_; 
v_reuseFailAlloc_710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_710_, 0, v_a_704_);
v___x_709_ = v_reuseFailAlloc_710_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
return v___x_709_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg___boxed(lean_object* v_lctx_712_, lean_object* v_localInsts_713_, lean_object* v_x_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_){
_start:
{
lean_object* v_res_721_; 
v_res_721_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg(v_lctx_712_, v_localInsts_713_, v_x_714_, v___y_715_, v___y_716_, v___y_717_, v___y_718_, v___y_719_);
lean_dec(v___y_719_);
lean_dec_ref(v___y_718_);
lean_dec(v___y_717_);
lean_dec_ref(v___y_716_);
return v_res_721_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg(lean_object* v_k_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_unquoted_729_; lean_object* v___x_730_; 
v_unquoted_729_ = lean_ctor_get(v___y_723_, 3);
lean_inc_ref(v_unquoted_729_);
v___x_730_ = lp_Qq_Qq_Impl_determineLocalInstances(v_unquoted_729_, v___y_724_, v___y_725_, v___y_726_, v___y_727_);
if (lean_obj_tag(v___x_730_) == 0)
{
lean_object* v_a_731_; lean_object* v___x_732_; 
v_a_731_ = lean_ctor_get(v___x_730_, 0);
lean_inc(v_a_731_);
lean_dec_ref_known(v___x_730_, 1);
v___x_732_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg(v_unquoted_729_, v_a_731_, v_k_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_, v___y_727_);
return v___x_732_;
}
else
{
lean_object* v_a_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_740_; 
lean_dec_ref(v_unquoted_729_);
lean_dec_ref(v___y_723_);
lean_dec_ref(v_k_722_);
v_a_733_ = lean_ctor_get(v___x_730_, 0);
v_isSharedCheck_740_ = !lean_is_exclusive(v___x_730_);
if (v_isSharedCheck_740_ == 0)
{
v___x_735_ = v___x_730_;
v_isShared_736_ = v_isSharedCheck_740_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_a_733_);
lean_dec(v___x_730_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_740_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v___x_738_; 
if (v_isShared_736_ == 0)
{
v___x_738_ = v___x_735_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v_a_733_);
v___x_738_ = v_reuseFailAlloc_739_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
return v___x_738_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg___boxed(lean_object* v_k_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
lean_object* v_res_748_; 
v_res_748_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg(v_k_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
lean_dec(v___y_746_);
lean_dec_ref(v___y_745_);
lean_dec(v___y_744_);
lean_dec_ref(v___y_743_);
return v_res_748_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6(lean_object* v_as_753_, size_t v_sz_754_, size_t v_i_755_, lean_object* v_b_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_){
_start:
{
uint8_t v___x_764_; 
v___x_764_ = lean_usize_dec_lt(v_i_755_, v_sz_754_);
if (v___x_764_ == 0)
{
lean_object* v___x_765_; 
v___x_765_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_765_, 0, v_b_756_);
return v___x_765_;
}
else
{
lean_object* v_snd_766_; lean_object* v___x_768_; uint8_t v_isShared_769_; uint8_t v_isSharedCheck_833_; 
v_snd_766_ = lean_ctor_get(v_b_756_, 1);
v_isSharedCheck_833_ = !lean_is_exclusive(v_b_756_);
if (v_isSharedCheck_833_ == 0)
{
lean_object* v_unused_834_; 
v_unused_834_ = lean_ctor_get(v_b_756_, 0);
lean_dec(v_unused_834_);
v___x_768_ = v_b_756_;
v_isShared_769_ = v_isSharedCheck_833_;
goto v_resetjp_767_;
}
else
{
lean_inc(v_snd_766_);
lean_dec(v_b_756_);
v___x_768_ = lean_box(0);
v_isShared_769_ = v_isSharedCheck_833_;
goto v_resetjp_767_;
}
v_resetjp_767_:
{
lean_object* v___x_770_; lean_object* v_a_772_; lean_object* v_a_779_; 
v___x_770_ = lean_box(0);
v_a_779_ = lean_array_uget_borrowed(v_as_753_, v_i_755_);
if (lean_obj_tag(v_a_779_) == 0)
{
v_a_772_ = v_snd_766_;
goto v___jp_771_;
}
else
{
lean_object* v_val_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
lean_dec(v_snd_766_);
v_val_780_ = lean_ctor_get(v_a_779_, 0);
v___x_781_ = l_Lean_LocalDecl_type(v_val_780_);
v___x_782_ = lp_Qq_Qq_Impl_whnfR(v___x_781_, v___y_759_, v___y_760_, v___y_761_, v___y_762_);
if (lean_obj_tag(v___x_782_) == 0)
{
lean_object* v_a_783_; lean_object* v___x_784_; lean_object* v___y_786_; lean_object* v___y_787_; lean_object* v___y_788_; lean_object* v___y_789_; lean_object* v___y_790_; lean_object* v___y_791_; uint8_t v___x_815_; 
v_a_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_a_783_);
lean_dec_ref_known(v___x_782_, 1);
v___x_784_ = lean_box(0);
v___x_815_ = l_Lean_Expr_isMVar(v_a_783_);
if (v___x_815_ == 0)
{
v___y_786_ = v___y_757_;
v___y_787_ = v___y_758_;
v___y_788_ = v___y_759_;
v___y_789_ = v___y_760_;
v___y_790_ = v___y_761_;
v___y_791_ = v___y_762_;
goto v___jp_785_;
}
else
{
lean_object* v___x_816_; 
v___x_816_ = l_Lean_Elab_Term_tryPostpone(v___y_757_, v___y_758_, v___y_759_, v___y_760_, v___y_761_, v___y_762_);
if (lean_obj_tag(v___x_816_) == 0)
{
lean_dec_ref_known(v___x_816_, 1);
v___y_786_ = v___y_757_;
v___y_787_ = v___y_758_;
v___y_788_ = v___y_759_;
v___y_789_ = v___y_760_;
v___y_790_ = v___y_761_;
v___y_791_ = v___y_762_;
goto v___jp_785_;
}
else
{
lean_object* v_a_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_824_; 
lean_dec(v_a_783_);
lean_del_object(v___x_768_);
v_a_817_ = lean_ctor_get(v___x_816_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_816_);
if (v_isSharedCheck_824_ == 0)
{
v___x_819_ = v___x_816_;
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_a_817_);
lean_dec(v___x_816_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
lean_object* v___x_822_; 
if (v_isShared_820_ == 0)
{
v___x_822_ = v___x_819_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_a_817_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
}
v___jp_785_:
{
lean_object* v___x_792_; uint8_t v___x_793_; 
v___x_792_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1));
v___x_793_ = l_Lean_Expr_isAppOf(v_a_783_, v___x_792_);
if (v___x_793_ == 0)
{
lean_dec(v_a_783_);
v_a_772_ = v___x_784_;
goto v___jp_771_;
}
else
{
lean_object* v___x_794_; lean_object* v___x_795_; 
v___x_794_ = l_Lean_Expr_appArg_x21(v_a_783_);
lean_dec(v_a_783_);
v___x_795_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(v___x_794_, v___y_789_);
if (lean_obj_tag(v___x_795_) == 0)
{
lean_object* v_a_796_; uint8_t v___x_797_; 
v_a_796_ = lean_ctor_get(v___x_795_, 0);
lean_inc(v_a_796_);
lean_dec_ref_known(v___x_795_, 1);
v___x_797_ = l_Lean_Expr_hasExprMVar(v_a_796_);
lean_dec(v_a_796_);
if (v___x_797_ == 0)
{
v_a_772_ = v___x_784_;
goto v___jp_771_;
}
else
{
lean_object* v___x_798_; 
v___x_798_ = l_Lean_Elab_Term_tryPostpone(v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_dec_ref_known(v___x_798_, 1);
v_a_772_ = v___x_784_;
goto v___jp_771_;
}
else
{
lean_object* v_a_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_806_; 
lean_del_object(v___x_768_);
v_a_799_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_806_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_806_ == 0)
{
v___x_801_ = v___x_798_;
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_a_799_);
lean_dec(v___x_798_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v___x_804_; 
if (v_isShared_802_ == 0)
{
v___x_804_ = v___x_801_;
goto v_reusejp_803_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v_a_799_);
v___x_804_ = v_reuseFailAlloc_805_;
goto v_reusejp_803_;
}
v_reusejp_803_:
{
return v___x_804_;
}
}
}
}
}
else
{
lean_object* v_a_807_; lean_object* v___x_809_; uint8_t v_isShared_810_; uint8_t v_isSharedCheck_814_; 
lean_del_object(v___x_768_);
v_a_807_ = lean_ctor_get(v___x_795_, 0);
v_isSharedCheck_814_ = !lean_is_exclusive(v___x_795_);
if (v_isSharedCheck_814_ == 0)
{
v___x_809_ = v___x_795_;
v_isShared_810_ = v_isSharedCheck_814_;
goto v_resetjp_808_;
}
else
{
lean_inc(v_a_807_);
lean_dec(v___x_795_);
v___x_809_ = lean_box(0);
v_isShared_810_ = v_isSharedCheck_814_;
goto v_resetjp_808_;
}
v_resetjp_808_:
{
lean_object* v___x_812_; 
if (v_isShared_810_ == 0)
{
v___x_812_ = v___x_809_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v_a_807_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
}
}
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
lean_del_object(v___x_768_);
v_a_825_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_782_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_782_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_830_; 
if (v_isShared_828_ == 0)
{
v___x_830_ = v___x_827_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_825_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
v___jp_771_:
{
lean_object* v___x_774_; 
if (v_isShared_769_ == 0)
{
lean_ctor_set(v___x_768_, 1, v_a_772_);
lean_ctor_set(v___x_768_, 0, v___x_770_);
v___x_774_ = v___x_768_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v___x_770_);
lean_ctor_set(v_reuseFailAlloc_778_, 1, v_a_772_);
v___x_774_ = v_reuseFailAlloc_778_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
size_t v___x_775_; size_t v___x_776_; 
v___x_775_ = ((size_t)1ULL);
v___x_776_ = lean_usize_add(v_i_755_, v___x_775_);
v_i_755_ = v___x_776_;
v_b_756_ = v___x_774_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___boxed(lean_object* v_as_835_, lean_object* v_sz_836_, lean_object* v_i_837_, lean_object* v_b_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_){
_start:
{
size_t v_sz_boxed_846_; size_t v_i_boxed_847_; lean_object* v_res_848_; 
v_sz_boxed_846_ = lean_unbox_usize(v_sz_836_);
lean_dec(v_sz_836_);
v_i_boxed_847_ = lean_unbox_usize(v_i_837_);
lean_dec(v_i_837_);
v_res_848_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6(v_as_835_, v_sz_boxed_846_, v_i_boxed_847_, v_b_838_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_, v___y_844_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
lean_dec(v___y_842_);
lean_dec_ref(v___y_841_);
lean_dec(v___y_840_);
lean_dec_ref(v___y_839_);
lean_dec_ref(v_as_835_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3(lean_object* v_as_849_, size_t v_sz_850_, size_t v_i_851_, lean_object* v_b_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_){
_start:
{
uint8_t v___x_860_; 
v___x_860_ = lean_usize_dec_lt(v_i_851_, v_sz_850_);
if (v___x_860_ == 0)
{
lean_object* v___x_861_; 
v___x_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_861_, 0, v_b_852_);
return v___x_861_;
}
else
{
lean_object* v_snd_862_; lean_object* v___x_864_; uint8_t v_isShared_865_; uint8_t v_isSharedCheck_929_; 
v_snd_862_ = lean_ctor_get(v_b_852_, 1);
v_isSharedCheck_929_ = !lean_is_exclusive(v_b_852_);
if (v_isSharedCheck_929_ == 0)
{
lean_object* v_unused_930_; 
v_unused_930_ = lean_ctor_get(v_b_852_, 0);
lean_dec(v_unused_930_);
v___x_864_ = v_b_852_;
v_isShared_865_ = v_isSharedCheck_929_;
goto v_resetjp_863_;
}
else
{
lean_inc(v_snd_862_);
lean_dec(v_b_852_);
v___x_864_ = lean_box(0);
v_isShared_865_ = v_isSharedCheck_929_;
goto v_resetjp_863_;
}
v_resetjp_863_:
{
lean_object* v___x_866_; lean_object* v_a_868_; lean_object* v_a_875_; 
v___x_866_ = lean_box(0);
v_a_875_ = lean_array_uget_borrowed(v_as_849_, v_i_851_);
if (lean_obj_tag(v_a_875_) == 0)
{
v_a_868_ = v_snd_862_;
goto v___jp_867_;
}
else
{
lean_object* v_val_876_; lean_object* v___x_877_; lean_object* v___x_878_; 
lean_dec(v_snd_862_);
v_val_876_ = lean_ctor_get(v_a_875_, 0);
v___x_877_ = l_Lean_LocalDecl_type(v_val_876_);
v___x_878_ = lp_Qq_Qq_Impl_whnfR(v___x_877_, v___y_855_, v___y_856_, v___y_857_, v___y_858_);
if (lean_obj_tag(v___x_878_) == 0)
{
lean_object* v_a_879_; lean_object* v___x_880_; lean_object* v___y_882_; lean_object* v___y_883_; lean_object* v___y_884_; lean_object* v___y_885_; lean_object* v___y_886_; lean_object* v___y_887_; uint8_t v___x_911_; 
v_a_879_ = lean_ctor_get(v___x_878_, 0);
lean_inc(v_a_879_);
lean_dec_ref_known(v___x_878_, 1);
v___x_880_ = lean_box(0);
v___x_911_ = l_Lean_Expr_isMVar(v_a_879_);
if (v___x_911_ == 0)
{
v___y_882_ = v___y_853_;
v___y_883_ = v___y_854_;
v___y_884_ = v___y_855_;
v___y_885_ = v___y_856_;
v___y_886_ = v___y_857_;
v___y_887_ = v___y_858_;
goto v___jp_881_;
}
else
{
lean_object* v___x_912_; 
v___x_912_ = l_Lean_Elab_Term_tryPostpone(v___y_853_, v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_);
if (lean_obj_tag(v___x_912_) == 0)
{
lean_dec_ref_known(v___x_912_, 1);
v___y_882_ = v___y_853_;
v___y_883_ = v___y_854_;
v___y_884_ = v___y_855_;
v___y_885_ = v___y_856_;
v___y_886_ = v___y_857_;
v___y_887_ = v___y_858_;
goto v___jp_881_;
}
else
{
lean_object* v_a_913_; lean_object* v___x_915_; uint8_t v_isShared_916_; uint8_t v_isSharedCheck_920_; 
lean_dec(v_a_879_);
lean_del_object(v___x_864_);
v_a_913_ = lean_ctor_get(v___x_912_, 0);
v_isSharedCheck_920_ = !lean_is_exclusive(v___x_912_);
if (v_isSharedCheck_920_ == 0)
{
v___x_915_ = v___x_912_;
v_isShared_916_ = v_isSharedCheck_920_;
goto v_resetjp_914_;
}
else
{
lean_inc(v_a_913_);
lean_dec(v___x_912_);
v___x_915_ = lean_box(0);
v_isShared_916_ = v_isSharedCheck_920_;
goto v_resetjp_914_;
}
v_resetjp_914_:
{
lean_object* v___x_918_; 
if (v_isShared_916_ == 0)
{
v___x_918_ = v___x_915_;
goto v_reusejp_917_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v_a_913_);
v___x_918_ = v_reuseFailAlloc_919_;
goto v_reusejp_917_;
}
v_reusejp_917_:
{
return v___x_918_;
}
}
}
}
v___jp_881_:
{
lean_object* v___x_888_; uint8_t v___x_889_; 
v___x_888_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1));
v___x_889_ = l_Lean_Expr_isAppOf(v_a_879_, v___x_888_);
if (v___x_889_ == 0)
{
lean_dec(v_a_879_);
v_a_868_ = v___x_880_;
goto v___jp_867_;
}
else
{
lean_object* v___x_890_; lean_object* v___x_891_; 
v___x_890_ = l_Lean_Expr_appArg_x21(v_a_879_);
lean_dec(v_a_879_);
v___x_891_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(v___x_890_, v___y_885_);
if (lean_obj_tag(v___x_891_) == 0)
{
lean_object* v_a_892_; uint8_t v___x_893_; 
v_a_892_ = lean_ctor_get(v___x_891_, 0);
lean_inc(v_a_892_);
lean_dec_ref_known(v___x_891_, 1);
v___x_893_ = l_Lean_Expr_hasExprMVar(v_a_892_);
lean_dec(v_a_892_);
if (v___x_893_ == 0)
{
v_a_868_ = v___x_880_;
goto v___jp_867_;
}
else
{
lean_object* v___x_894_; 
v___x_894_ = l_Lean_Elab_Term_tryPostpone(v___y_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_);
if (lean_obj_tag(v___x_894_) == 0)
{
lean_dec_ref_known(v___x_894_, 1);
v_a_868_ = v___x_880_;
goto v___jp_867_;
}
else
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
lean_del_object(v___x_864_);
v_a_895_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___x_894_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_894_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
}
}
else
{
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_910_; 
lean_del_object(v___x_864_);
v_a_903_ = lean_ctor_get(v___x_891_, 0);
v_isSharedCheck_910_ = !lean_is_exclusive(v___x_891_);
if (v_isSharedCheck_910_ == 0)
{
v___x_905_ = v___x_891_;
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_891_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v___x_908_; 
if (v_isShared_906_ == 0)
{
v___x_908_ = v___x_905_;
goto v_reusejp_907_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_a_903_);
v___x_908_ = v_reuseFailAlloc_909_;
goto v_reusejp_907_;
}
v_reusejp_907_:
{
return v___x_908_;
}
}
}
}
}
}
else
{
lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_928_; 
lean_del_object(v___x_864_);
v_a_921_ = lean_ctor_get(v___x_878_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_878_);
if (v_isSharedCheck_928_ == 0)
{
v___x_923_ = v___x_878_;
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_878_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_926_; 
if (v_isShared_924_ == 0)
{
v___x_926_ = v___x_923_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v_a_921_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
}
v___jp_867_:
{
lean_object* v___x_870_; 
if (v_isShared_865_ == 0)
{
lean_ctor_set(v___x_864_, 1, v_a_868_);
lean_ctor_set(v___x_864_, 0, v___x_866_);
v___x_870_ = v___x_864_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_866_);
lean_ctor_set(v_reuseFailAlloc_874_, 1, v_a_868_);
v___x_870_ = v_reuseFailAlloc_874_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
size_t v___x_871_; size_t v___x_872_; lean_object* v___x_873_; 
v___x_871_ = ((size_t)1ULL);
v___x_872_ = lean_usize_add(v_i_851_, v___x_871_);
v___x_873_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6(v_as_849_, v_sz_850_, v___x_872_, v___x_870_, v___y_853_, v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_);
return v___x_873_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3___boxed(lean_object* v_as_931_, lean_object* v_sz_932_, lean_object* v_i_933_, lean_object* v_b_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_){
_start:
{
size_t v_sz_boxed_942_; size_t v_i_boxed_943_; lean_object* v_res_944_; 
v_sz_boxed_942_ = lean_unbox_usize(v_sz_932_);
lean_dec(v_sz_932_);
v_i_boxed_943_ = lean_unbox_usize(v_i_933_);
lean_dec(v_i_933_);
v_res_944_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3(v_as_931_, v_sz_boxed_942_, v_i_boxed_943_, v_b_934_, v___y_935_, v___y_936_, v___y_937_, v___y_938_, v___y_939_, v___y_940_);
lean_dec(v___y_940_);
lean_dec_ref(v___y_939_);
lean_dec(v___y_938_);
lean_dec_ref(v___y_937_);
lean_dec(v___y_936_);
lean_dec_ref(v___y_935_);
lean_dec_ref(v_as_931_);
return v_res_944_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1(lean_object* v_init_945_, lean_object* v_n_946_, lean_object* v_b_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
if (lean_obj_tag(v_n_946_) == 0)
{
lean_object* v_cs_955_; lean_object* v___x_956_; lean_object* v___x_957_; size_t v_sz_958_; size_t v___x_959_; lean_object* v___x_960_; 
v_cs_955_ = lean_ctor_get(v_n_946_, 0);
v___x_956_ = lean_box(0);
v___x_957_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_957_, 0, v___x_956_);
lean_ctor_set(v___x_957_, 1, v_b_947_);
v_sz_958_ = lean_array_size(v_cs_955_);
v___x_959_ = ((size_t)0ULL);
v___x_960_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__2(v_init_945_, v_cs_955_, v_sz_958_, v___x_959_, v___x_957_, v___y_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_);
if (lean_obj_tag(v___x_960_) == 0)
{
lean_object* v_a_961_; lean_object* v___x_963_; uint8_t v_isShared_964_; uint8_t v_isSharedCheck_975_; 
v_a_961_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_975_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_975_ == 0)
{
v___x_963_ = v___x_960_;
v_isShared_964_ = v_isSharedCheck_975_;
goto v_resetjp_962_;
}
else
{
lean_inc(v_a_961_);
lean_dec(v___x_960_);
v___x_963_ = lean_box(0);
v_isShared_964_ = v_isSharedCheck_975_;
goto v_resetjp_962_;
}
v_resetjp_962_:
{
lean_object* v_fst_965_; 
v_fst_965_ = lean_ctor_get(v_a_961_, 0);
if (lean_obj_tag(v_fst_965_) == 0)
{
lean_object* v_snd_966_; lean_object* v___x_967_; lean_object* v___x_969_; 
v_snd_966_ = lean_ctor_get(v_a_961_, 1);
lean_inc(v_snd_966_);
lean_dec(v_a_961_);
v___x_967_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_967_, 0, v_snd_966_);
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 0, v___x_967_);
v___x_969_ = v___x_963_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v___x_967_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
else
{
lean_object* v_val_971_; lean_object* v___x_973_; 
lean_inc_ref(v_fst_965_);
lean_dec(v_a_961_);
v_val_971_ = lean_ctor_get(v_fst_965_, 0);
lean_inc(v_val_971_);
lean_dec_ref_known(v_fst_965_, 1);
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 0, v_val_971_);
v___x_973_ = v___x_963_;
goto v_reusejp_972_;
}
else
{
lean_object* v_reuseFailAlloc_974_; 
v_reuseFailAlloc_974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_974_, 0, v_val_971_);
v___x_973_ = v_reuseFailAlloc_974_;
goto v_reusejp_972_;
}
v_reusejp_972_:
{
return v___x_973_;
}
}
}
}
else
{
lean_object* v_a_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_983_; 
v_a_976_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_983_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_983_ == 0)
{
v___x_978_ = v___x_960_;
v_isShared_979_ = v_isSharedCheck_983_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_a_976_);
lean_dec(v___x_960_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_983_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_981_; 
if (v_isShared_979_ == 0)
{
v___x_981_ = v___x_978_;
goto v_reusejp_980_;
}
else
{
lean_object* v_reuseFailAlloc_982_; 
v_reuseFailAlloc_982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_982_, 0, v_a_976_);
v___x_981_ = v_reuseFailAlloc_982_;
goto v_reusejp_980_;
}
v_reusejp_980_:
{
return v___x_981_;
}
}
}
}
else
{
lean_object* v_vs_984_; lean_object* v___x_985_; lean_object* v___x_986_; size_t v_sz_987_; size_t v___x_988_; lean_object* v___x_989_; 
v_vs_984_ = lean_ctor_get(v_n_946_, 0);
v___x_985_ = lean_box(0);
v___x_986_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_986_, 0, v___x_985_);
lean_ctor_set(v___x_986_, 1, v_b_947_);
v_sz_987_ = lean_array_size(v_vs_984_);
v___x_988_ = ((size_t)0ULL);
v___x_989_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3(v_vs_984_, v_sz_987_, v___x_988_, v___x_986_, v___y_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_);
if (lean_obj_tag(v___x_989_) == 0)
{
lean_object* v_a_990_; lean_object* v___x_992_; uint8_t v_isShared_993_; uint8_t v_isSharedCheck_1004_; 
v_a_990_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1004_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1004_ == 0)
{
v___x_992_ = v___x_989_;
v_isShared_993_ = v_isSharedCheck_1004_;
goto v_resetjp_991_;
}
else
{
lean_inc(v_a_990_);
lean_dec(v___x_989_);
v___x_992_ = lean_box(0);
v_isShared_993_ = v_isSharedCheck_1004_;
goto v_resetjp_991_;
}
v_resetjp_991_:
{
lean_object* v_fst_994_; 
v_fst_994_ = lean_ctor_get(v_a_990_, 0);
if (lean_obj_tag(v_fst_994_) == 0)
{
lean_object* v_snd_995_; lean_object* v___x_996_; lean_object* v___x_998_; 
v_snd_995_ = lean_ctor_get(v_a_990_, 1);
lean_inc(v_snd_995_);
lean_dec(v_a_990_);
v___x_996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_996_, 0, v_snd_995_);
if (v_isShared_993_ == 0)
{
lean_ctor_set(v___x_992_, 0, v___x_996_);
v___x_998_ = v___x_992_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_999_; 
v_reuseFailAlloc_999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_999_, 0, v___x_996_);
v___x_998_ = v_reuseFailAlloc_999_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
return v___x_998_;
}
}
else
{
lean_object* v_val_1000_; lean_object* v___x_1002_; 
lean_inc_ref(v_fst_994_);
lean_dec(v_a_990_);
v_val_1000_ = lean_ctor_get(v_fst_994_, 0);
lean_inc(v_val_1000_);
lean_dec_ref_known(v_fst_994_, 1);
if (v_isShared_993_ == 0)
{
lean_ctor_set(v___x_992_, 0, v_val_1000_);
v___x_1002_ = v___x_992_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v_val_1000_);
v___x_1002_ = v_reuseFailAlloc_1003_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
return v___x_1002_;
}
}
}
}
else
{
lean_object* v_a_1005_; lean_object* v___x_1007_; uint8_t v_isShared_1008_; uint8_t v_isSharedCheck_1012_; 
v_a_1005_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_1007_ = v___x_989_;
v_isShared_1008_ = v_isSharedCheck_1012_;
goto v_resetjp_1006_;
}
else
{
lean_inc(v_a_1005_);
lean_dec(v___x_989_);
v___x_1007_ = lean_box(0);
v_isShared_1008_ = v_isSharedCheck_1012_;
goto v_resetjp_1006_;
}
v_resetjp_1006_:
{
lean_object* v___x_1010_; 
if (v_isShared_1008_ == 0)
{
v___x_1010_ = v___x_1007_;
goto v_reusejp_1009_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v_a_1005_);
v___x_1010_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1009_;
}
v_reusejp_1009_:
{
return v___x_1010_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__2(lean_object* v_init_1013_, lean_object* v_as_1014_, size_t v_sz_1015_, size_t v_i_1016_, lean_object* v_b_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
uint8_t v___x_1025_; 
v___x_1025_ = lean_usize_dec_lt(v_i_1016_, v_sz_1015_);
if (v___x_1025_ == 0)
{
lean_object* v___x_1026_; 
v___x_1026_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1026_, 0, v_b_1017_);
return v___x_1026_;
}
else
{
lean_object* v_snd_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1061_; 
v_snd_1027_ = lean_ctor_get(v_b_1017_, 1);
v_isSharedCheck_1061_ = !lean_is_exclusive(v_b_1017_);
if (v_isSharedCheck_1061_ == 0)
{
lean_object* v_unused_1062_; 
v_unused_1062_ = lean_ctor_get(v_b_1017_, 0);
lean_dec(v_unused_1062_);
v___x_1029_ = v_b_1017_;
v_isShared_1030_ = v_isSharedCheck_1061_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_snd_1027_);
lean_dec(v_b_1017_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1061_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v_a_1031_; lean_object* v___x_1032_; 
v_a_1031_ = lean_array_uget_borrowed(v_as_1014_, v_i_1016_);
lean_inc(v_snd_1027_);
v___x_1032_ = lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1(v_init_1013_, v_a_1031_, v_snd_1027_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_);
if (lean_obj_tag(v___x_1032_) == 0)
{
lean_object* v_a_1033_; lean_object* v___x_1035_; uint8_t v_isShared_1036_; uint8_t v_isSharedCheck_1052_; 
v_a_1033_ = lean_ctor_get(v___x_1032_, 0);
v_isSharedCheck_1052_ = !lean_is_exclusive(v___x_1032_);
if (v_isSharedCheck_1052_ == 0)
{
v___x_1035_ = v___x_1032_;
v_isShared_1036_ = v_isSharedCheck_1052_;
goto v_resetjp_1034_;
}
else
{
lean_inc(v_a_1033_);
lean_dec(v___x_1032_);
v___x_1035_ = lean_box(0);
v_isShared_1036_ = v_isSharedCheck_1052_;
goto v_resetjp_1034_;
}
v_resetjp_1034_:
{
if (lean_obj_tag(v_a_1033_) == 0)
{
lean_object* v___x_1037_; lean_object* v___x_1039_; 
v___x_1037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1037_, 0, v_a_1033_);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 0, v___x_1037_);
v___x_1039_ = v___x_1029_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v___x_1037_);
lean_ctor_set(v_reuseFailAlloc_1043_, 1, v_snd_1027_);
v___x_1039_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
lean_object* v___x_1041_; 
if (v_isShared_1036_ == 0)
{
lean_ctor_set(v___x_1035_, 0, v___x_1039_);
v___x_1041_ = v___x_1035_;
goto v_reusejp_1040_;
}
else
{
lean_object* v_reuseFailAlloc_1042_; 
v_reuseFailAlloc_1042_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1042_, 0, v___x_1039_);
v___x_1041_ = v_reuseFailAlloc_1042_;
goto v_reusejp_1040_;
}
v_reusejp_1040_:
{
return v___x_1041_;
}
}
}
else
{
lean_object* v_a_1044_; lean_object* v___x_1045_; lean_object* v___x_1047_; 
lean_del_object(v___x_1035_);
lean_dec(v_snd_1027_);
v_a_1044_ = lean_ctor_get(v_a_1033_, 0);
lean_inc(v_a_1044_);
lean_dec_ref_known(v_a_1033_, 1);
v___x_1045_ = lean_box(0);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 1, v_a_1044_);
lean_ctor_set(v___x_1029_, 0, v___x_1045_);
v___x_1047_ = v___x_1029_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v___x_1045_);
lean_ctor_set(v_reuseFailAlloc_1051_, 1, v_a_1044_);
v___x_1047_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
size_t v___x_1048_; size_t v___x_1049_; 
v___x_1048_ = ((size_t)1ULL);
v___x_1049_ = lean_usize_add(v_i_1016_, v___x_1048_);
v_i_1016_ = v___x_1049_;
v_b_1017_ = v___x_1047_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1060_; 
lean_del_object(v___x_1029_);
lean_dec(v_snd_1027_);
v_a_1053_ = lean_ctor_get(v___x_1032_, 0);
v_isSharedCheck_1060_ = !lean_is_exclusive(v___x_1032_);
if (v_isSharedCheck_1060_ == 0)
{
v___x_1055_ = v___x_1032_;
v_isShared_1056_ = v_isSharedCheck_1060_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_a_1053_);
lean_dec(v___x_1032_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1060_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v___x_1058_; 
if (v_isShared_1056_ == 0)
{
v___x_1058_ = v___x_1055_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v_a_1053_);
v___x_1058_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
return v___x_1058_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__2___boxed(lean_object* v_init_1063_, lean_object* v_as_1064_, lean_object* v_sz_1065_, lean_object* v_i_1066_, lean_object* v_b_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_){
_start:
{
size_t v_sz_boxed_1075_; size_t v_i_boxed_1076_; lean_object* v_res_1077_; 
v_sz_boxed_1075_ = lean_unbox_usize(v_sz_1065_);
lean_dec(v_sz_1065_);
v_i_boxed_1076_ = lean_unbox_usize(v_i_1066_);
lean_dec(v_i_1066_);
v_res_1077_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__2(v_init_1063_, v_as_1064_, v_sz_boxed_1075_, v_i_boxed_1076_, v_b_1067_, v___y_1068_, v___y_1069_, v___y_1070_, v___y_1071_, v___y_1072_, v___y_1073_);
lean_dec(v___y_1073_);
lean_dec_ref(v___y_1072_);
lean_dec(v___y_1071_);
lean_dec_ref(v___y_1070_);
lean_dec(v___y_1069_);
lean_dec_ref(v___y_1068_);
lean_dec_ref(v_as_1064_);
return v_res_1077_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1___boxed(lean_object* v_init_1078_, lean_object* v_n_1079_, lean_object* v_b_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_){
_start:
{
lean_object* v_res_1088_; 
v_res_1088_ = lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1(v_init_1078_, v_n_1079_, v_b_1080_, v___y_1081_, v___y_1082_, v___y_1083_, v___y_1084_, v___y_1085_, v___y_1086_);
lean_dec(v___y_1086_);
lean_dec_ref(v___y_1085_);
lean_dec(v___y_1084_);
lean_dec_ref(v___y_1083_);
lean_dec(v___y_1082_);
lean_dec_ref(v___y_1081_);
lean_dec_ref(v_n_1079_);
return v_res_1088_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2_spec__5(lean_object* v_as_1089_, size_t v_sz_1090_, size_t v_i_1091_, lean_object* v_b_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_){
_start:
{
uint8_t v___x_1100_; 
v___x_1100_ = lean_usize_dec_lt(v_i_1091_, v_sz_1090_);
if (v___x_1100_ == 0)
{
lean_object* v___x_1101_; 
v___x_1101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1101_, 0, v_b_1092_);
return v___x_1101_;
}
else
{
lean_object* v_snd_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1169_; 
v_snd_1102_ = lean_ctor_get(v_b_1092_, 1);
v_isSharedCheck_1169_ = !lean_is_exclusive(v_b_1092_);
if (v_isSharedCheck_1169_ == 0)
{
lean_object* v_unused_1170_; 
v_unused_1170_ = lean_ctor_get(v_b_1092_, 0);
lean_dec(v_unused_1170_);
v___x_1104_ = v_b_1092_;
v_isShared_1105_ = v_isSharedCheck_1169_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_snd_1102_);
lean_dec(v_b_1092_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1169_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1106_; lean_object* v_a_1108_; lean_object* v_a_1115_; 
v___x_1106_ = lean_box(0);
v_a_1115_ = lean_array_uget_borrowed(v_as_1089_, v_i_1091_);
if (lean_obj_tag(v_a_1115_) == 0)
{
v_a_1108_ = v_snd_1102_;
goto v___jp_1107_;
}
else
{
lean_object* v_val_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; 
lean_dec(v_snd_1102_);
v_val_1116_ = lean_ctor_get(v_a_1115_, 0);
v___x_1117_ = l_Lean_LocalDecl_type(v_val_1116_);
v___x_1118_ = lp_Qq_Qq_Impl_whnfR(v___x_1117_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; lean_object* v___x_1120_; lean_object* v___y_1122_; lean_object* v___y_1123_; lean_object* v___y_1124_; lean_object* v___y_1125_; lean_object* v___y_1126_; lean_object* v___y_1127_; uint8_t v___x_1151_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
lean_inc(v_a_1119_);
lean_dec_ref_known(v___x_1118_, 1);
v___x_1120_ = lean_box(0);
v___x_1151_ = l_Lean_Expr_isMVar(v_a_1119_);
if (v___x_1151_ == 0)
{
v___y_1122_ = v___y_1093_;
v___y_1123_ = v___y_1094_;
v___y_1124_ = v___y_1095_;
v___y_1125_ = v___y_1096_;
v___y_1126_ = v___y_1097_;
v___y_1127_ = v___y_1098_;
goto v___jp_1121_;
}
else
{
lean_object* v___x_1152_; 
v___x_1152_ = l_Lean_Elab_Term_tryPostpone(v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
if (lean_obj_tag(v___x_1152_) == 0)
{
lean_dec_ref_known(v___x_1152_, 1);
v___y_1122_ = v___y_1093_;
v___y_1123_ = v___y_1094_;
v___y_1124_ = v___y_1095_;
v___y_1125_ = v___y_1096_;
v___y_1126_ = v___y_1097_;
v___y_1127_ = v___y_1098_;
goto v___jp_1121_;
}
else
{
lean_object* v_a_1153_; lean_object* v___x_1155_; uint8_t v_isShared_1156_; uint8_t v_isSharedCheck_1160_; 
lean_dec(v_a_1119_);
lean_del_object(v___x_1104_);
v_a_1153_ = lean_ctor_get(v___x_1152_, 0);
v_isSharedCheck_1160_ = !lean_is_exclusive(v___x_1152_);
if (v_isSharedCheck_1160_ == 0)
{
v___x_1155_ = v___x_1152_;
v_isShared_1156_ = v_isSharedCheck_1160_;
goto v_resetjp_1154_;
}
else
{
lean_inc(v_a_1153_);
lean_dec(v___x_1152_);
v___x_1155_ = lean_box(0);
v_isShared_1156_ = v_isSharedCheck_1160_;
goto v_resetjp_1154_;
}
v_resetjp_1154_:
{
lean_object* v___x_1158_; 
if (v_isShared_1156_ == 0)
{
v___x_1158_ = v___x_1155_;
goto v_reusejp_1157_;
}
else
{
lean_object* v_reuseFailAlloc_1159_; 
v_reuseFailAlloc_1159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1159_, 0, v_a_1153_);
v___x_1158_ = v_reuseFailAlloc_1159_;
goto v_reusejp_1157_;
}
v_reusejp_1157_:
{
return v___x_1158_;
}
}
}
}
v___jp_1121_:
{
lean_object* v___x_1128_; uint8_t v___x_1129_; 
v___x_1128_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1));
v___x_1129_ = l_Lean_Expr_isAppOf(v_a_1119_, v___x_1128_);
if (v___x_1129_ == 0)
{
lean_dec(v_a_1119_);
v_a_1108_ = v___x_1120_;
goto v___jp_1107_;
}
else
{
lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1130_ = l_Lean_Expr_appArg_x21(v_a_1119_);
lean_dec(v_a_1119_);
v___x_1131_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(v___x_1130_, v___y_1125_);
if (lean_obj_tag(v___x_1131_) == 0)
{
lean_object* v_a_1132_; uint8_t v___x_1133_; 
v_a_1132_ = lean_ctor_get(v___x_1131_, 0);
lean_inc(v_a_1132_);
lean_dec_ref_known(v___x_1131_, 1);
v___x_1133_ = l_Lean_Expr_hasExprMVar(v_a_1132_);
lean_dec(v_a_1132_);
if (v___x_1133_ == 0)
{
v_a_1108_ = v___x_1120_;
goto v___jp_1107_;
}
else
{
lean_object* v___x_1134_; 
v___x_1134_ = l_Lean_Elab_Term_tryPostpone(v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_, v___y_1127_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_dec_ref_known(v___x_1134_, 1);
v_a_1108_ = v___x_1120_;
goto v___jp_1107_;
}
else
{
lean_object* v_a_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1142_; 
lean_del_object(v___x_1104_);
v_a_1135_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1137_ = v___x_1134_;
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_a_1135_);
lean_dec(v___x_1134_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1140_; 
if (v_isShared_1138_ == 0)
{
v___x_1140_ = v___x_1137_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_a_1135_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
}
else
{
lean_object* v_a_1143_; lean_object* v___x_1145_; uint8_t v_isShared_1146_; uint8_t v_isSharedCheck_1150_; 
lean_del_object(v___x_1104_);
v_a_1143_ = lean_ctor_get(v___x_1131_, 0);
v_isSharedCheck_1150_ = !lean_is_exclusive(v___x_1131_);
if (v_isSharedCheck_1150_ == 0)
{
v___x_1145_ = v___x_1131_;
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
else
{
lean_inc(v_a_1143_);
lean_dec(v___x_1131_);
v___x_1145_ = lean_box(0);
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
v_resetjp_1144_:
{
lean_object* v___x_1148_; 
if (v_isShared_1146_ == 0)
{
v___x_1148_ = v___x_1145_;
goto v_reusejp_1147_;
}
else
{
lean_object* v_reuseFailAlloc_1149_; 
v_reuseFailAlloc_1149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1149_, 0, v_a_1143_);
v___x_1148_ = v_reuseFailAlloc_1149_;
goto v_reusejp_1147_;
}
v_reusejp_1147_:
{
return v___x_1148_;
}
}
}
}
}
}
else
{
lean_object* v_a_1161_; lean_object* v___x_1163_; uint8_t v_isShared_1164_; uint8_t v_isSharedCheck_1168_; 
lean_del_object(v___x_1104_);
v_a_1161_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1168_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1168_ == 0)
{
v___x_1163_ = v___x_1118_;
v_isShared_1164_ = v_isSharedCheck_1168_;
goto v_resetjp_1162_;
}
else
{
lean_inc(v_a_1161_);
lean_dec(v___x_1118_);
v___x_1163_ = lean_box(0);
v_isShared_1164_ = v_isSharedCheck_1168_;
goto v_resetjp_1162_;
}
v_resetjp_1162_:
{
lean_object* v___x_1166_; 
if (v_isShared_1164_ == 0)
{
v___x_1166_ = v___x_1163_;
goto v_reusejp_1165_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v_a_1161_);
v___x_1166_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1165_;
}
v_reusejp_1165_:
{
return v___x_1166_;
}
}
}
}
v___jp_1107_:
{
lean_object* v___x_1110_; 
if (v_isShared_1105_ == 0)
{
lean_ctor_set(v___x_1104_, 1, v_a_1108_);
lean_ctor_set(v___x_1104_, 0, v___x_1106_);
v___x_1110_ = v___x_1104_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1114_; 
v_reuseFailAlloc_1114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1114_, 0, v___x_1106_);
lean_ctor_set(v_reuseFailAlloc_1114_, 1, v_a_1108_);
v___x_1110_ = v_reuseFailAlloc_1114_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
size_t v___x_1111_; size_t v___x_1112_; 
v___x_1111_ = ((size_t)1ULL);
v___x_1112_ = lean_usize_add(v_i_1091_, v___x_1111_);
v_i_1091_ = v___x_1112_;
v_b_1092_ = v___x_1110_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2_spec__5___boxed(lean_object* v_as_1171_, lean_object* v_sz_1172_, lean_object* v_i_1173_, lean_object* v_b_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
size_t v_sz_boxed_1182_; size_t v_i_boxed_1183_; lean_object* v_res_1184_; 
v_sz_boxed_1182_ = lean_unbox_usize(v_sz_1172_);
lean_dec(v_sz_1172_);
v_i_boxed_1183_ = lean_unbox_usize(v_i_1173_);
lean_dec(v_i_1173_);
v_res_1184_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2_spec__5(v_as_1171_, v_sz_boxed_1182_, v_i_boxed_1183_, v_b_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
lean_dec(v___y_1178_);
lean_dec_ref(v___y_1177_);
lean_dec(v___y_1176_);
lean_dec_ref(v___y_1175_);
lean_dec_ref(v_as_1171_);
return v_res_1184_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2(lean_object* v_as_1185_, size_t v_sz_1186_, size_t v_i_1187_, lean_object* v_b_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_){
_start:
{
uint8_t v___x_1196_; 
v___x_1196_ = lean_usize_dec_lt(v_i_1187_, v_sz_1186_);
if (v___x_1196_ == 0)
{
lean_object* v___x_1197_; 
v___x_1197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1197_, 0, v_b_1188_);
return v___x_1197_;
}
else
{
lean_object* v_snd_1198_; lean_object* v___x_1200_; uint8_t v_isShared_1201_; uint8_t v_isSharedCheck_1265_; 
v_snd_1198_ = lean_ctor_get(v_b_1188_, 1);
v_isSharedCheck_1265_ = !lean_is_exclusive(v_b_1188_);
if (v_isSharedCheck_1265_ == 0)
{
lean_object* v_unused_1266_; 
v_unused_1266_ = lean_ctor_get(v_b_1188_, 0);
lean_dec(v_unused_1266_);
v___x_1200_ = v_b_1188_;
v_isShared_1201_ = v_isSharedCheck_1265_;
goto v_resetjp_1199_;
}
else
{
lean_inc(v_snd_1198_);
lean_dec(v_b_1188_);
v___x_1200_ = lean_box(0);
v_isShared_1201_ = v_isSharedCheck_1265_;
goto v_resetjp_1199_;
}
v_resetjp_1199_:
{
lean_object* v___x_1202_; lean_object* v_a_1204_; lean_object* v_a_1211_; 
v___x_1202_ = lean_box(0);
v_a_1211_ = lean_array_uget_borrowed(v_as_1185_, v_i_1187_);
if (lean_obj_tag(v_a_1211_) == 0)
{
v_a_1204_ = v_snd_1198_;
goto v___jp_1203_;
}
else
{
lean_object* v_val_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; 
lean_dec(v_snd_1198_);
v_val_1212_ = lean_ctor_get(v_a_1211_, 0);
v___x_1213_ = l_Lean_LocalDecl_type(v_val_1212_);
v___x_1214_ = lp_Qq_Qq_Impl_whnfR(v___x_1213_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_);
if (lean_obj_tag(v___x_1214_) == 0)
{
lean_object* v_a_1215_; lean_object* v___x_1216_; lean_object* v___y_1218_; lean_object* v___y_1219_; lean_object* v___y_1220_; lean_object* v___y_1221_; lean_object* v___y_1222_; lean_object* v___y_1223_; uint8_t v___x_1247_; 
v_a_1215_ = lean_ctor_get(v___x_1214_, 0);
lean_inc(v_a_1215_);
lean_dec_ref_known(v___x_1214_, 1);
v___x_1216_ = lean_box(0);
v___x_1247_ = l_Lean_Expr_isMVar(v_a_1215_);
if (v___x_1247_ == 0)
{
v___y_1218_ = v___y_1189_;
v___y_1219_ = v___y_1190_;
v___y_1220_ = v___y_1191_;
v___y_1221_ = v___y_1192_;
v___y_1222_ = v___y_1193_;
v___y_1223_ = v___y_1194_;
goto v___jp_1217_;
}
else
{
lean_object* v___x_1248_; 
v___x_1248_ = l_Lean_Elab_Term_tryPostpone(v___y_1189_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_);
if (lean_obj_tag(v___x_1248_) == 0)
{
lean_dec_ref_known(v___x_1248_, 1);
v___y_1218_ = v___y_1189_;
v___y_1219_ = v___y_1190_;
v___y_1220_ = v___y_1191_;
v___y_1221_ = v___y_1192_;
v___y_1222_ = v___y_1193_;
v___y_1223_ = v___y_1194_;
goto v___jp_1217_;
}
else
{
lean_object* v_a_1249_; lean_object* v___x_1251_; uint8_t v_isShared_1252_; uint8_t v_isSharedCheck_1256_; 
lean_dec(v_a_1215_);
lean_del_object(v___x_1200_);
v_a_1249_ = lean_ctor_get(v___x_1248_, 0);
v_isSharedCheck_1256_ = !lean_is_exclusive(v___x_1248_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1251_ = v___x_1248_;
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
else
{
lean_inc(v_a_1249_);
lean_dec(v___x_1248_);
v___x_1251_ = lean_box(0);
v_isShared_1252_ = v_isSharedCheck_1256_;
goto v_resetjp_1250_;
}
v_resetjp_1250_:
{
lean_object* v___x_1254_; 
if (v_isShared_1252_ == 0)
{
v___x_1254_ = v___x_1251_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v_a_1249_);
v___x_1254_ = v_reuseFailAlloc_1255_;
goto v_reusejp_1253_;
}
v_reusejp_1253_:
{
return v___x_1254_;
}
}
}
}
v___jp_1217_:
{
lean_object* v___x_1224_; uint8_t v___x_1225_; 
v___x_1224_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1_spec__3_spec__6___closed__1));
v___x_1225_ = l_Lean_Expr_isAppOf(v_a_1215_, v___x_1224_);
if (v___x_1225_ == 0)
{
lean_dec(v_a_1215_);
v_a_1204_ = v___x_1216_;
goto v___jp_1203_;
}
else
{
lean_object* v___x_1226_; lean_object* v___x_1227_; 
v___x_1226_ = l_Lean_Expr_appArg_x21(v_a_1215_);
lean_dec(v_a_1215_);
v___x_1227_ = lp_Qq_Lean_instantiateMVars___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__0___redArg(v___x_1226_, v___y_1221_);
if (lean_obj_tag(v___x_1227_) == 0)
{
lean_object* v_a_1228_; uint8_t v___x_1229_; 
v_a_1228_ = lean_ctor_get(v___x_1227_, 0);
lean_inc(v_a_1228_);
lean_dec_ref_known(v___x_1227_, 1);
v___x_1229_ = l_Lean_Expr_hasExprMVar(v_a_1228_);
lean_dec(v_a_1228_);
if (v___x_1229_ == 0)
{
v_a_1204_ = v___x_1216_;
goto v___jp_1203_;
}
else
{
lean_object* v___x_1230_; 
v___x_1230_ = l_Lean_Elab_Term_tryPostpone(v___y_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_, v___y_1223_);
if (lean_obj_tag(v___x_1230_) == 0)
{
lean_dec_ref_known(v___x_1230_, 1);
v_a_1204_ = v___x_1216_;
goto v___jp_1203_;
}
else
{
lean_object* v_a_1231_; lean_object* v___x_1233_; uint8_t v_isShared_1234_; uint8_t v_isSharedCheck_1238_; 
lean_del_object(v___x_1200_);
v_a_1231_ = lean_ctor_get(v___x_1230_, 0);
v_isSharedCheck_1238_ = !lean_is_exclusive(v___x_1230_);
if (v_isSharedCheck_1238_ == 0)
{
v___x_1233_ = v___x_1230_;
v_isShared_1234_ = v_isSharedCheck_1238_;
goto v_resetjp_1232_;
}
else
{
lean_inc(v_a_1231_);
lean_dec(v___x_1230_);
v___x_1233_ = lean_box(0);
v_isShared_1234_ = v_isSharedCheck_1238_;
goto v_resetjp_1232_;
}
v_resetjp_1232_:
{
lean_object* v___x_1236_; 
if (v_isShared_1234_ == 0)
{
v___x_1236_ = v___x_1233_;
goto v_reusejp_1235_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v_a_1231_);
v___x_1236_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1235_;
}
v_reusejp_1235_:
{
return v___x_1236_;
}
}
}
}
}
else
{
lean_object* v_a_1239_; lean_object* v___x_1241_; uint8_t v_isShared_1242_; uint8_t v_isSharedCheck_1246_; 
lean_del_object(v___x_1200_);
v_a_1239_ = lean_ctor_get(v___x_1227_, 0);
v_isSharedCheck_1246_ = !lean_is_exclusive(v___x_1227_);
if (v_isSharedCheck_1246_ == 0)
{
v___x_1241_ = v___x_1227_;
v_isShared_1242_ = v_isSharedCheck_1246_;
goto v_resetjp_1240_;
}
else
{
lean_inc(v_a_1239_);
lean_dec(v___x_1227_);
v___x_1241_ = lean_box(0);
v_isShared_1242_ = v_isSharedCheck_1246_;
goto v_resetjp_1240_;
}
v_resetjp_1240_:
{
lean_object* v___x_1244_; 
if (v_isShared_1242_ == 0)
{
v___x_1244_ = v___x_1241_;
goto v_reusejp_1243_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v_a_1239_);
v___x_1244_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1243_;
}
v_reusejp_1243_:
{
return v___x_1244_;
}
}
}
}
}
}
else
{
lean_object* v_a_1257_; lean_object* v___x_1259_; uint8_t v_isShared_1260_; uint8_t v_isSharedCheck_1264_; 
lean_del_object(v___x_1200_);
v_a_1257_ = lean_ctor_get(v___x_1214_, 0);
v_isSharedCheck_1264_ = !lean_is_exclusive(v___x_1214_);
if (v_isSharedCheck_1264_ == 0)
{
v___x_1259_ = v___x_1214_;
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
else
{
lean_inc(v_a_1257_);
lean_dec(v___x_1214_);
v___x_1259_ = lean_box(0);
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
v_resetjp_1258_:
{
lean_object* v___x_1262_; 
if (v_isShared_1260_ == 0)
{
v___x_1262_ = v___x_1259_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v_a_1257_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
}
}
v___jp_1203_:
{
lean_object* v___x_1206_; 
if (v_isShared_1201_ == 0)
{
lean_ctor_set(v___x_1200_, 1, v_a_1204_);
lean_ctor_set(v___x_1200_, 0, v___x_1202_);
v___x_1206_ = v___x_1200_;
goto v_reusejp_1205_;
}
else
{
lean_object* v_reuseFailAlloc_1210_; 
v_reuseFailAlloc_1210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1210_, 0, v___x_1202_);
lean_ctor_set(v_reuseFailAlloc_1210_, 1, v_a_1204_);
v___x_1206_ = v_reuseFailAlloc_1210_;
goto v_reusejp_1205_;
}
v_reusejp_1205_:
{
size_t v___x_1207_; size_t v___x_1208_; lean_object* v___x_1209_; 
v___x_1207_ = ((size_t)1ULL);
v___x_1208_ = lean_usize_add(v_i_1187_, v___x_1207_);
v___x_1209_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2_spec__5(v_as_1185_, v_sz_1186_, v___x_1208_, v___x_1206_, v___y_1189_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_);
return v___x_1209_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2___boxed(lean_object* v_as_1267_, lean_object* v_sz_1268_, lean_object* v_i_1269_, lean_object* v_b_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_){
_start:
{
size_t v_sz_boxed_1278_; size_t v_i_boxed_1279_; lean_object* v_res_1280_; 
v_sz_boxed_1278_ = lean_unbox_usize(v_sz_1268_);
lean_dec(v_sz_1268_);
v_i_boxed_1279_ = lean_unbox_usize(v_i_1269_);
lean_dec(v_i_1269_);
v_res_1280_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2(v_as_1267_, v_sz_boxed_1278_, v_i_boxed_1279_, v_b_1270_, v___y_1271_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_, v___y_1276_);
lean_dec(v___y_1276_);
lean_dec_ref(v___y_1275_);
lean_dec(v___y_1274_);
lean_dec_ref(v___y_1273_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec_ref(v_as_1267_);
return v_res_1280_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1(lean_object* v_t_1281_, lean_object* v_init_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
lean_object* v_root_1290_; lean_object* v_tail_1291_; lean_object* v___x_1292_; 
v_root_1290_ = lean_ctor_get(v_t_1281_, 0);
v_tail_1291_ = lean_ctor_get(v_t_1281_, 1);
v___x_1292_ = lp_Qq_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__1(v_init_1282_, v_root_1290_, v_init_1282_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_, v___y_1288_);
if (lean_obj_tag(v___x_1292_) == 0)
{
lean_object* v_a_1293_; lean_object* v___x_1295_; uint8_t v_isShared_1296_; uint8_t v_isSharedCheck_1329_; 
v_a_1293_ = lean_ctor_get(v___x_1292_, 0);
v_isSharedCheck_1329_ = !lean_is_exclusive(v___x_1292_);
if (v_isSharedCheck_1329_ == 0)
{
v___x_1295_ = v___x_1292_;
v_isShared_1296_ = v_isSharedCheck_1329_;
goto v_resetjp_1294_;
}
else
{
lean_inc(v_a_1293_);
lean_dec(v___x_1292_);
v___x_1295_ = lean_box(0);
v_isShared_1296_ = v_isSharedCheck_1329_;
goto v_resetjp_1294_;
}
v_resetjp_1294_:
{
if (lean_obj_tag(v_a_1293_) == 0)
{
lean_object* v_a_1297_; lean_object* v___x_1299_; 
v_a_1297_ = lean_ctor_get(v_a_1293_, 0);
lean_inc(v_a_1297_);
lean_dec_ref_known(v_a_1293_, 1);
if (v_isShared_1296_ == 0)
{
lean_ctor_set(v___x_1295_, 0, v_a_1297_);
v___x_1299_ = v___x_1295_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v_a_1297_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
else
{
lean_object* v_a_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; size_t v_sz_1304_; size_t v___x_1305_; lean_object* v___x_1306_; 
lean_del_object(v___x_1295_);
v_a_1301_ = lean_ctor_get(v_a_1293_, 0);
lean_inc(v_a_1301_);
lean_dec_ref_known(v_a_1293_, 1);
v___x_1302_ = lean_box(0);
v___x_1303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1303_, 0, v___x_1302_);
lean_ctor_set(v___x_1303_, 1, v_a_1301_);
v_sz_1304_ = lean_array_size(v_tail_1291_);
v___x_1305_ = ((size_t)0ULL);
v___x_1306_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1_spec__2(v_tail_1291_, v_sz_1304_, v___x_1305_, v___x_1303_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_, v___y_1288_);
if (lean_obj_tag(v___x_1306_) == 0)
{
lean_object* v_a_1307_; lean_object* v___x_1309_; uint8_t v_isShared_1310_; uint8_t v_isSharedCheck_1320_; 
v_a_1307_ = lean_ctor_get(v___x_1306_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v___x_1306_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1309_ = v___x_1306_;
v_isShared_1310_ = v_isSharedCheck_1320_;
goto v_resetjp_1308_;
}
else
{
lean_inc(v_a_1307_);
lean_dec(v___x_1306_);
v___x_1309_ = lean_box(0);
v_isShared_1310_ = v_isSharedCheck_1320_;
goto v_resetjp_1308_;
}
v_resetjp_1308_:
{
lean_object* v_fst_1311_; 
v_fst_1311_ = lean_ctor_get(v_a_1307_, 0);
if (lean_obj_tag(v_fst_1311_) == 0)
{
lean_object* v_snd_1312_; lean_object* v___x_1314_; 
v_snd_1312_ = lean_ctor_get(v_a_1307_, 1);
lean_inc(v_snd_1312_);
lean_dec(v_a_1307_);
if (v_isShared_1310_ == 0)
{
lean_ctor_set(v___x_1309_, 0, v_snd_1312_);
v___x_1314_ = v___x_1309_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1315_; 
v_reuseFailAlloc_1315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1315_, 0, v_snd_1312_);
v___x_1314_ = v_reuseFailAlloc_1315_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
return v___x_1314_;
}
}
else
{
lean_object* v_val_1316_; lean_object* v___x_1318_; 
lean_inc_ref(v_fst_1311_);
lean_dec(v_a_1307_);
v_val_1316_ = lean_ctor_get(v_fst_1311_, 0);
lean_inc(v_val_1316_);
lean_dec_ref_known(v_fst_1311_, 1);
if (v_isShared_1310_ == 0)
{
lean_ctor_set(v___x_1309_, 0, v_val_1316_);
v___x_1318_ = v___x_1309_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v_val_1316_);
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
v_a_1321_ = lean_ctor_get(v___x_1306_, 0);
v_isSharedCheck_1328_ = !lean_is_exclusive(v___x_1306_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1323_ = v___x_1306_;
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
else
{
lean_inc(v_a_1321_);
lean_dec(v___x_1306_);
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
}
}
}
else
{
lean_object* v_a_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1337_; 
v_a_1330_ = lean_ctor_get(v___x_1292_, 0);
v_isSharedCheck_1337_ = !lean_is_exclusive(v___x_1292_);
if (v_isSharedCheck_1337_ == 0)
{
v___x_1332_ = v___x_1292_;
v_isShared_1333_ = v_isSharedCheck_1337_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_a_1330_);
lean_dec(v___x_1292_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1337_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
lean_object* v___x_1335_; 
if (v_isShared_1333_ == 0)
{
v___x_1335_ = v___x_1332_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v_a_1330_);
v___x_1335_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
return v___x_1335_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1___boxed(lean_object* v_t_1338_, lean_object* v_init_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_){
_start:
{
lean_object* v_res_1347_; 
v_res_1347_ = lp_Qq_Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1(v_t_1338_, v_init_1339_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_, v___y_1344_, v___y_1345_);
lean_dec(v___y_1345_);
lean_dec_ref(v___y_1344_);
lean_dec(v___y_1343_);
lean_dec_ref(v___y_1342_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec_ref(v_t_1338_);
return v_res_1347_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__0(void){
_start:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; 
v___x_1348_ = lean_box(0);
v___x_1349_ = lean_unsigned_to_nat(16u);
v___x_1350_ = lean_mk_array(v___x_1349_, v___x_1348_);
return v___x_1350_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__1(void){
_start:
{
lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; 
v___x_1351_ = lean_obj_once(&lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__0, &lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__0_once, _init_lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__0);
v___x_1352_ = lean_unsigned_to_nat(0u);
v___x_1353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1352_);
lean_ctor_set(v___x_1353_, 1, v___x_1351_);
return v___x_1353_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f(lean_object* v_a_1356_, lean_object* v_a_1357_, lean_object* v_a_1358_, lean_object* v_a_1359_, lean_object* v_a_1360_, lean_object* v_a_1361_){
_start:
{
lean_object* v_lctx_1363_; lean_object* v_decls_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; 
v_lctx_1363_ = lean_ctor_get(v_a_1358_, 2);
v_decls_1364_ = lean_ctor_get(v_lctx_1363_, 1);
v___x_1365_ = lean_box(0);
v___x_1366_ = lp_Qq_Lean_PersistentArray_forIn___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__1(v_decls_1364_, v___x_1365_, v_a_1356_, v_a_1357_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1366_) == 0)
{
lean_object* v___x_1367_; uint8_t v_mayPostpone_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; 
lean_dec_ref_known(v___x_1366_, 1);
v___x_1367_ = lean_obj_once(&lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__1, &lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__1_once, _init_lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__1);
v_mayPostpone_1368_ = lean_ctor_get_uint8(v_a_1356_, sizeof(void*)*8);
v___x_1369_ = lean_box(0);
v___x_1370_ = l_Lean_LocalContext_empty;
v___x_1371_ = ((lean_object*)(lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___closed__2));
v___x_1372_ = lean_alloc_ctor(0, 8, 1);
lean_ctor_set(v___x_1372_, 0, v___x_1369_);
lean_ctor_set(v___x_1372_, 1, v___x_1367_);
lean_ctor_set(v___x_1372_, 2, v___x_1367_);
lean_ctor_set(v___x_1372_, 3, v___x_1370_);
lean_ctor_set(v___x_1372_, 4, v___x_1367_);
lean_ctor_set(v___x_1372_, 5, v___x_1367_);
lean_ctor_set(v___x_1372_, 6, v___x_1371_);
lean_ctor_set(v___x_1372_, 7, v___x_1369_);
lean_ctor_set_uint8(v___x_1372_, sizeof(void*)*8, v_mayPostpone_1368_);
v___x_1373_ = lp_Qq_Qq_Impl_unquoteLCtx(v___x_1372_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1373_) == 0)
{
lean_object* v_a_1374_; lean_object* v_snd_1375_; lean_object* v___x_1377_; uint8_t v_isShared_1378_; uint8_t v_isSharedCheck_1505_; 
v_a_1374_ = lean_ctor_get(v___x_1373_, 0);
lean_inc(v_a_1374_);
lean_dec_ref_known(v___x_1373_, 1);
v_snd_1375_ = lean_ctor_get(v_a_1374_, 1);
v_isSharedCheck_1505_ = !lean_is_exclusive(v_a_1374_);
if (v_isSharedCheck_1505_ == 0)
{
lean_object* v_unused_1506_; 
v_unused_1506_ = lean_ctor_get(v_a_1374_, 0);
lean_dec(v_unused_1506_);
v___x_1377_ = v_a_1374_;
v_isShared_1378_ = v_isSharedCheck_1505_;
goto v_resetjp_1376_;
}
else
{
lean_inc(v_snd_1375_);
lean_dec(v_a_1374_);
v___x_1377_ = lean_box(0);
v_isShared_1378_ = v_isSharedCheck_1505_;
goto v_resetjp_1376_;
}
v_resetjp_1376_:
{
lean_object* v___x_1379_; 
v___x_1379_ = lp_Qq_Qq_Impl_findRedundantLocalInst_x3f(v_snd_1375_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1379_) == 0)
{
lean_object* v_a_1380_; lean_object* v___x_1382_; uint8_t v_isShared_1383_; uint8_t v_isSharedCheck_1496_; 
v_a_1380_ = lean_ctor_get(v___x_1379_, 0);
v_isSharedCheck_1496_ = !lean_is_exclusive(v___x_1379_);
if (v_isSharedCheck_1496_ == 0)
{
v___x_1382_ = v___x_1379_;
v_isShared_1383_ = v_isSharedCheck_1496_;
goto v_resetjp_1381_;
}
else
{
lean_inc(v_a_1380_);
lean_dec(v___x_1379_);
v___x_1382_ = lean_box(0);
v_isShared_1383_ = v_isSharedCheck_1496_;
goto v_resetjp_1381_;
}
v_resetjp_1381_:
{
if (lean_obj_tag(v_a_1380_) == 0)
{
lean_object* v___x_1384_; lean_object* v___x_1386_; 
lean_del_object(v___x_1377_);
lean_dec(v_snd_1375_);
v___x_1384_ = lean_box(0);
if (v_isShared_1383_ == 0)
{
lean_ctor_set(v___x_1382_, 0, v___x_1384_);
v___x_1386_ = v___x_1382_;
goto v_reusejp_1385_;
}
else
{
lean_object* v_reuseFailAlloc_1387_; 
v_reuseFailAlloc_1387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1387_, 0, v___x_1384_);
v___x_1386_ = v_reuseFailAlloc_1387_;
goto v_reusejp_1385_;
}
v_reusejp_1385_:
{
return v___x_1386_;
}
}
else
{
lean_object* v_val_1388_; lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1495_; 
lean_del_object(v___x_1382_);
v_val_1388_ = lean_ctor_get(v_a_1380_, 0);
v_isSharedCheck_1495_ = !lean_is_exclusive(v_a_1380_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1390_ = v_a_1380_;
v_isShared_1391_ = v_isSharedCheck_1495_;
goto v_resetjp_1389_;
}
else
{
lean_inc(v_val_1388_);
lean_dec(v_a_1380_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1495_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
lean_object* v_fst_1392_; lean_object* v_snd_1393_; lean_object* v___x_1395_; uint8_t v_isShared_1396_; uint8_t v_isSharedCheck_1494_; 
v_fst_1392_ = lean_ctor_get(v_val_1388_, 0);
v_snd_1393_ = lean_ctor_get(v_val_1388_, 1);
v_isSharedCheck_1494_ = !lean_is_exclusive(v_val_1388_);
if (v_isSharedCheck_1494_ == 0)
{
v___x_1395_ = v_val_1388_;
v_isShared_1396_ = v_isSharedCheck_1494_;
goto v_resetjp_1394_;
}
else
{
lean_inc(v_snd_1393_);
lean_inc(v_fst_1392_);
lean_dec(v_val_1388_);
v___x_1395_ = lean_box(0);
v_isShared_1396_ = v_isSharedCheck_1494_;
goto v_resetjp_1394_;
}
v_resetjp_1394_:
{
lean_object* v___x_1397_; lean_object* v___f_1398_; lean_object* v___x_1399_; 
lean_inc(v_fst_1392_);
v___x_1397_ = l_Lean_Expr_fvar___override(v_fst_1392_);
lean_inc_ref(v___x_1397_);
v___f_1398_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1398_, 0, v___x_1397_);
v___x_1399_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg(v___f_1398_, v_snd_1375_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1399_) == 0)
{
lean_object* v_a_1400_; lean_object* v_fst_1401_; lean_object* v_snd_1402_; lean_object* v___x_1404_; uint8_t v_isShared_1405_; uint8_t v_isSharedCheck_1485_; 
v_a_1400_ = lean_ctor_get(v___x_1399_, 0);
lean_inc(v_a_1400_);
lean_dec_ref_known(v___x_1399_, 1);
v_fst_1401_ = lean_ctor_get(v_a_1400_, 0);
v_snd_1402_ = lean_ctor_get(v_a_1400_, 1);
v_isSharedCheck_1485_ = !lean_is_exclusive(v_a_1400_);
if (v_isSharedCheck_1485_ == 0)
{
v___x_1404_ = v_a_1400_;
v_isShared_1405_ = v_isSharedCheck_1485_;
goto v_resetjp_1403_;
}
else
{
lean_inc(v_snd_1402_);
lean_inc(v_fst_1401_);
lean_dec(v_a_1400_);
v___x_1404_ = lean_box(0);
v_isShared_1405_ = v_isSharedCheck_1485_;
goto v_resetjp_1403_;
}
v_resetjp_1403_:
{
lean_object* v___f_1406_; lean_object* v___x_1407_; 
lean_inc(v_fst_1401_);
v___f_1406_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___lam__1___boxed), 7, 1);
lean_closure_set(v___f_1406_, 0, v_fst_1401_);
v___x_1407_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg(v___f_1406_, v_snd_1402_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1407_) == 0)
{
lean_object* v_a_1408_; lean_object* v_fst_1409_; lean_object* v_snd_1410_; lean_object* v___x_1412_; uint8_t v_isShared_1413_; uint8_t v_isSharedCheck_1476_; 
v_a_1408_ = lean_ctor_get(v___x_1407_, 0);
lean_inc(v_a_1408_);
lean_dec_ref_known(v___x_1407_, 1);
v_fst_1409_ = lean_ctor_get(v_a_1408_, 0);
v_snd_1410_ = lean_ctor_get(v_a_1408_, 1);
v_isSharedCheck_1476_ = !lean_is_exclusive(v_a_1408_);
if (v_isSharedCheck_1476_ == 0)
{
v___x_1412_ = v_a_1408_;
v_isShared_1413_ = v_isSharedCheck_1476_;
goto v_resetjp_1411_;
}
else
{
lean_inc(v_snd_1410_);
lean_inc(v_fst_1409_);
lean_dec(v_a_1408_);
v___x_1412_ = lean_box(0);
v_isShared_1413_ = v_isSharedCheck_1476_;
goto v_resetjp_1411_;
}
v_resetjp_1411_:
{
lean_object* v___x_1414_; 
v___x_1414_ = lp_Qq_Qq_Impl_quoteLevel(v_fst_1409_, v_snd_1410_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1414_) == 0)
{
lean_object* v_a_1415_; lean_object* v___x_1416_; 
v_a_1415_ = lean_ctor_get(v___x_1414_, 0);
lean_inc(v_a_1415_);
lean_dec_ref_known(v___x_1414_, 1);
v___x_1416_ = lp_Qq_Qq_Impl_quoteExpr(v_fst_1401_, v_snd_1410_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1416_) == 0)
{
lean_object* v_a_1417_; lean_object* v___x_1418_; 
v_a_1417_ = lean_ctor_get(v___x_1416_, 0);
lean_inc(v_a_1417_);
lean_dec_ref_known(v___x_1416_, 1);
v___x_1418_ = lp_Qq_Qq_Impl_quoteExpr(v___x_1397_, v_snd_1410_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
if (lean_obj_tag(v___x_1418_) == 0)
{
lean_object* v_a_1419_; lean_object* v___x_1420_; 
v_a_1419_ = lean_ctor_get(v___x_1418_, 0);
lean_inc(v_a_1419_);
lean_dec_ref_known(v___x_1418_, 1);
v___x_1420_ = lp_Qq_Qq_Impl_quoteExpr(v_snd_1393_, v_snd_1410_, v_a_1358_, v_a_1359_, v_a_1360_, v_a_1361_);
lean_dec(v_snd_1410_);
if (lean_obj_tag(v___x_1420_) == 0)
{
lean_object* v_a_1421_; lean_object* v___x_1423_; uint8_t v_isShared_1424_; uint8_t v_isSharedCheck_1443_; 
v_a_1421_ = lean_ctor_get(v___x_1420_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1420_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1423_ = v___x_1420_;
v_isShared_1424_ = v_isSharedCheck_1443_;
goto v_resetjp_1422_;
}
else
{
lean_inc(v_a_1421_);
lean_dec(v___x_1420_);
v___x_1423_ = lean_box(0);
v_isShared_1424_ = v_isSharedCheck_1443_;
goto v_resetjp_1422_;
}
v_resetjp_1422_:
{
lean_object* v___x_1426_; 
if (v_isShared_1413_ == 0)
{
lean_ctor_set(v___x_1412_, 1, v_a_1421_);
lean_ctor_set(v___x_1412_, 0, v_a_1419_);
v___x_1426_ = v___x_1412_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1442_; 
v_reuseFailAlloc_1442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1442_, 0, v_a_1419_);
lean_ctor_set(v_reuseFailAlloc_1442_, 1, v_a_1421_);
v___x_1426_ = v_reuseFailAlloc_1442_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
lean_object* v___x_1428_; 
if (v_isShared_1396_ == 0)
{
lean_ctor_set(v___x_1395_, 1, v___x_1426_);
lean_ctor_set(v___x_1395_, 0, v_a_1417_);
v___x_1428_ = v___x_1395_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1441_; 
v_reuseFailAlloc_1441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1441_, 0, v_a_1417_);
lean_ctor_set(v_reuseFailAlloc_1441_, 1, v___x_1426_);
v___x_1428_ = v_reuseFailAlloc_1441_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
lean_object* v___x_1430_; 
if (v_isShared_1378_ == 0)
{
lean_ctor_set(v___x_1377_, 1, v___x_1428_);
lean_ctor_set(v___x_1377_, 0, v_a_1415_);
v___x_1430_ = v___x_1377_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1440_; 
v_reuseFailAlloc_1440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1440_, 0, v_a_1415_);
lean_ctor_set(v_reuseFailAlloc_1440_, 1, v___x_1428_);
v___x_1430_ = v_reuseFailAlloc_1440_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
lean_object* v___x_1432_; 
if (v_isShared_1405_ == 0)
{
lean_ctor_set(v___x_1404_, 1, v___x_1430_);
lean_ctor_set(v___x_1404_, 0, v_fst_1392_);
v___x_1432_ = v___x_1404_;
goto v_reusejp_1431_;
}
else
{
lean_object* v_reuseFailAlloc_1439_; 
v_reuseFailAlloc_1439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1439_, 0, v_fst_1392_);
lean_ctor_set(v_reuseFailAlloc_1439_, 1, v___x_1430_);
v___x_1432_ = v_reuseFailAlloc_1439_;
goto v_reusejp_1431_;
}
v_reusejp_1431_:
{
lean_object* v___x_1434_; 
if (v_isShared_1391_ == 0)
{
lean_ctor_set(v___x_1390_, 0, v___x_1432_);
v___x_1434_ = v___x_1390_;
goto v_reusejp_1433_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v___x_1432_);
v___x_1434_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1433_;
}
v_reusejp_1433_:
{
lean_object* v___x_1436_; 
if (v_isShared_1424_ == 0)
{
lean_ctor_set(v___x_1423_, 0, v___x_1434_);
v___x_1436_ = v___x_1423_;
goto v_reusejp_1435_;
}
else
{
lean_object* v_reuseFailAlloc_1437_; 
v_reuseFailAlloc_1437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1437_, 0, v___x_1434_);
v___x_1436_ = v_reuseFailAlloc_1437_;
goto v_reusejp_1435_;
}
v_reusejp_1435_:
{
return v___x_1436_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_1444_; lean_object* v___x_1446_; uint8_t v_isShared_1447_; uint8_t v_isSharedCheck_1451_; 
lean_dec(v_a_1419_);
lean_dec(v_a_1417_);
lean_dec(v_a_1415_);
lean_del_object(v___x_1412_);
lean_del_object(v___x_1404_);
lean_del_object(v___x_1395_);
lean_dec(v_fst_1392_);
lean_del_object(v___x_1390_);
lean_del_object(v___x_1377_);
v_a_1444_ = lean_ctor_get(v___x_1420_, 0);
v_isSharedCheck_1451_ = !lean_is_exclusive(v___x_1420_);
if (v_isSharedCheck_1451_ == 0)
{
v___x_1446_ = v___x_1420_;
v_isShared_1447_ = v_isSharedCheck_1451_;
goto v_resetjp_1445_;
}
else
{
lean_inc(v_a_1444_);
lean_dec(v___x_1420_);
v___x_1446_ = lean_box(0);
v_isShared_1447_ = v_isSharedCheck_1451_;
goto v_resetjp_1445_;
}
v_resetjp_1445_:
{
lean_object* v___x_1449_; 
if (v_isShared_1447_ == 0)
{
v___x_1449_ = v___x_1446_;
goto v_reusejp_1448_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v_a_1444_);
v___x_1449_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1448_;
}
v_reusejp_1448_:
{
return v___x_1449_;
}
}
}
}
else
{
lean_object* v_a_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1459_; 
lean_dec(v_a_1417_);
lean_dec(v_a_1415_);
lean_del_object(v___x_1412_);
lean_dec(v_snd_1410_);
lean_del_object(v___x_1404_);
lean_del_object(v___x_1395_);
lean_dec(v_snd_1393_);
lean_dec(v_fst_1392_);
lean_del_object(v___x_1390_);
lean_del_object(v___x_1377_);
v_a_1452_ = lean_ctor_get(v___x_1418_, 0);
v_isSharedCheck_1459_ = !lean_is_exclusive(v___x_1418_);
if (v_isSharedCheck_1459_ == 0)
{
v___x_1454_ = v___x_1418_;
v_isShared_1455_ = v_isSharedCheck_1459_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_a_1452_);
lean_dec(v___x_1418_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1459_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1457_; 
if (v_isShared_1455_ == 0)
{
v___x_1457_ = v___x_1454_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1458_; 
v_reuseFailAlloc_1458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1458_, 0, v_a_1452_);
v___x_1457_ = v_reuseFailAlloc_1458_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
return v___x_1457_;
}
}
}
}
else
{
lean_object* v_a_1460_; lean_object* v___x_1462_; uint8_t v_isShared_1463_; uint8_t v_isSharedCheck_1467_; 
lean_dec(v_a_1415_);
lean_del_object(v___x_1412_);
lean_dec(v_snd_1410_);
lean_del_object(v___x_1404_);
lean_dec_ref(v___x_1397_);
lean_del_object(v___x_1395_);
lean_dec(v_snd_1393_);
lean_dec(v_fst_1392_);
lean_del_object(v___x_1390_);
lean_del_object(v___x_1377_);
v_a_1460_ = lean_ctor_get(v___x_1416_, 0);
v_isSharedCheck_1467_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1467_ == 0)
{
v___x_1462_ = v___x_1416_;
v_isShared_1463_ = v_isSharedCheck_1467_;
goto v_resetjp_1461_;
}
else
{
lean_inc(v_a_1460_);
lean_dec(v___x_1416_);
v___x_1462_ = lean_box(0);
v_isShared_1463_ = v_isSharedCheck_1467_;
goto v_resetjp_1461_;
}
v_resetjp_1461_:
{
lean_object* v___x_1465_; 
if (v_isShared_1463_ == 0)
{
v___x_1465_ = v___x_1462_;
goto v_reusejp_1464_;
}
else
{
lean_object* v_reuseFailAlloc_1466_; 
v_reuseFailAlloc_1466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1466_, 0, v_a_1460_);
v___x_1465_ = v_reuseFailAlloc_1466_;
goto v_reusejp_1464_;
}
v_reusejp_1464_:
{
return v___x_1465_;
}
}
}
}
else
{
lean_object* v_a_1468_; lean_object* v___x_1470_; uint8_t v_isShared_1471_; uint8_t v_isSharedCheck_1475_; 
lean_del_object(v___x_1412_);
lean_dec(v_snd_1410_);
lean_del_object(v___x_1404_);
lean_dec(v_fst_1401_);
lean_dec_ref(v___x_1397_);
lean_del_object(v___x_1395_);
lean_dec(v_snd_1393_);
lean_dec(v_fst_1392_);
lean_del_object(v___x_1390_);
lean_del_object(v___x_1377_);
v_a_1468_ = lean_ctor_get(v___x_1414_, 0);
v_isSharedCheck_1475_ = !lean_is_exclusive(v___x_1414_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1470_ = v___x_1414_;
v_isShared_1471_ = v_isSharedCheck_1475_;
goto v_resetjp_1469_;
}
else
{
lean_inc(v_a_1468_);
lean_dec(v___x_1414_);
v___x_1470_ = lean_box(0);
v_isShared_1471_ = v_isSharedCheck_1475_;
goto v_resetjp_1469_;
}
v_resetjp_1469_:
{
lean_object* v___x_1473_; 
if (v_isShared_1471_ == 0)
{
v___x_1473_ = v___x_1470_;
goto v_reusejp_1472_;
}
else
{
lean_object* v_reuseFailAlloc_1474_; 
v_reuseFailAlloc_1474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1474_, 0, v_a_1468_);
v___x_1473_ = v_reuseFailAlloc_1474_;
goto v_reusejp_1472_;
}
v_reusejp_1472_:
{
return v___x_1473_;
}
}
}
}
}
else
{
lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1484_; 
lean_del_object(v___x_1404_);
lean_dec(v_fst_1401_);
lean_dec_ref(v___x_1397_);
lean_del_object(v___x_1395_);
lean_dec(v_snd_1393_);
lean_dec(v_fst_1392_);
lean_del_object(v___x_1390_);
lean_del_object(v___x_1377_);
v_a_1477_ = lean_ctor_get(v___x_1407_, 0);
v_isSharedCheck_1484_ = !lean_is_exclusive(v___x_1407_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1479_ = v___x_1407_;
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1407_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v___x_1482_; 
if (v_isShared_1480_ == 0)
{
v___x_1482_ = v___x_1479_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v_a_1477_);
v___x_1482_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1481_;
}
v_reusejp_1481_:
{
return v___x_1482_;
}
}
}
}
}
else
{
lean_object* v_a_1486_; lean_object* v___x_1488_; uint8_t v_isShared_1489_; uint8_t v_isSharedCheck_1493_; 
lean_dec_ref(v___x_1397_);
lean_del_object(v___x_1395_);
lean_dec(v_snd_1393_);
lean_dec(v_fst_1392_);
lean_del_object(v___x_1390_);
lean_del_object(v___x_1377_);
v_a_1486_ = lean_ctor_get(v___x_1399_, 0);
v_isSharedCheck_1493_ = !lean_is_exclusive(v___x_1399_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1488_ = v___x_1399_;
v_isShared_1489_ = v_isSharedCheck_1493_;
goto v_resetjp_1487_;
}
else
{
lean_inc(v_a_1486_);
lean_dec(v___x_1399_);
v___x_1488_ = lean_box(0);
v_isShared_1489_ = v_isSharedCheck_1493_;
goto v_resetjp_1487_;
}
v_resetjp_1487_:
{
lean_object* v___x_1491_; 
if (v_isShared_1489_ == 0)
{
v___x_1491_ = v___x_1488_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v_a_1486_);
v___x_1491_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
return v___x_1491_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1504_; 
lean_del_object(v___x_1377_);
lean_dec(v_snd_1375_);
v_a_1497_ = lean_ctor_get(v___x_1379_, 0);
v_isSharedCheck_1504_ = !lean_is_exclusive(v___x_1379_);
if (v_isSharedCheck_1504_ == 0)
{
v___x_1499_ = v___x_1379_;
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1379_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v___x_1502_; 
if (v_isShared_1500_ == 0)
{
v___x_1502_ = v___x_1499_;
goto v_reusejp_1501_;
}
else
{
lean_object* v_reuseFailAlloc_1503_; 
v_reuseFailAlloc_1503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1503_, 0, v_a_1497_);
v___x_1502_ = v_reuseFailAlloc_1503_;
goto v_reusejp_1501_;
}
v_reusejp_1501_:
{
return v___x_1502_;
}
}
}
}
}
else
{
lean_object* v_a_1507_; lean_object* v___x_1509_; uint8_t v_isShared_1510_; uint8_t v_isSharedCheck_1514_; 
v_a_1507_ = lean_ctor_get(v___x_1373_, 0);
v_isSharedCheck_1514_ = !lean_is_exclusive(v___x_1373_);
if (v_isSharedCheck_1514_ == 0)
{
v___x_1509_ = v___x_1373_;
v_isShared_1510_ = v_isSharedCheck_1514_;
goto v_resetjp_1508_;
}
else
{
lean_inc(v_a_1507_);
lean_dec(v___x_1373_);
v___x_1509_ = lean_box(0);
v_isShared_1510_ = v_isSharedCheck_1514_;
goto v_resetjp_1508_;
}
v_resetjp_1508_:
{
lean_object* v___x_1512_; 
if (v_isShared_1510_ == 0)
{
v___x_1512_ = v___x_1509_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1513_; 
v_reuseFailAlloc_1513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1513_, 0, v_a_1507_);
v___x_1512_ = v_reuseFailAlloc_1513_;
goto v_reusejp_1511_;
}
v_reusejp_1511_:
{
return v___x_1512_;
}
}
}
}
else
{
lean_object* v_a_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1522_; 
v_a_1515_ = lean_ctor_get(v___x_1366_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1366_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1517_ = v___x_1366_;
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_a_1515_);
lean_dec(v___x_1366_);
v___x_1517_ = lean_box(0);
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
v_resetjp_1516_:
{
lean_object* v___x_1520_; 
if (v_isShared_1518_ == 0)
{
v___x_1520_ = v___x_1517_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v_a_1515_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f___boxed(lean_object* v_a_1523_, lean_object* v_a_1524_, lean_object* v_a_1525_, lean_object* v_a_1526_, lean_object* v_a_1527_, lean_object* v_a_1528_, lean_object* v_a_1529_){
_start:
{
lean_object* v_res_1530_; 
v_res_1530_ = lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f(v_a_1523_, v_a_1524_, v_a_1525_, v_a_1526_, v_a_1527_, v_a_1528_);
lean_dec(v_a_1528_);
lean_dec_ref(v_a_1527_);
lean_dec(v_a_1526_);
lean_dec_ref(v_a_1525_);
lean_dec(v_a_1524_);
lean_dec_ref(v_a_1523_);
return v_res_1530_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4(lean_object* v_00_u03b1_1531_, lean_object* v_lctx_1532_, lean_object* v_localInsts_1533_, lean_object* v_x_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_){
_start:
{
lean_object* v___x_1541_; 
v___x_1541_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___redArg(v_lctx_1532_, v_localInsts_1533_, v_x_1534_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_, v___y_1539_);
return v___x_1541_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4___boxed(lean_object* v_00_u03b1_1542_, lean_object* v_lctx_1543_, lean_object* v_localInsts_1544_, lean_object* v_x_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_){
_start:
{
lean_object* v_res_1552_; 
v_res_1552_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2_spec__4(v_00_u03b1_1542_, v_lctx_1543_, v_localInsts_1544_, v_x_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_, v___y_1550_);
lean_dec(v___y_1550_);
lean_dec_ref(v___y_1549_);
lean_dec(v___y_1548_);
lean_dec_ref(v___y_1547_);
return v_res_1552_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2(lean_object* v_00_u03b1_1553_, lean_object* v_k_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_){
_start:
{
lean_object* v___x_1561_; 
v___x_1561_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___redArg(v_k_1554_, v___y_1555_, v___y_1556_, v___y_1557_, v___y_1558_, v___y_1559_);
return v___x_1561_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2___boxed(lean_object* v_00_u03b1_1562_, lean_object* v_k_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_){
_start:
{
lean_object* v_res_1570_; 
v_res_1570_ = lp_Qq_Qq_Impl_withUnquotedLCtx___at___00Qq_Impl_findRedundantLocalInstQuoted_x3f_spec__2(v_00_u03b1_1562_, v_k_1563_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_);
lean_dec(v___y_1568_);
lean_dec_ref(v___y_1567_);
lean_dec(v___y_1566_);
lean_dec_ref(v___y_1565_);
return v_res_1570_;
}
}
static lean_object* _init_lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; 
v___x_1589_ = lean_box(0);
v___x_1590_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1591_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1591_, 0, v___x_1590_);
lean_ctor_set(v___x_1591_, 1, v___x_1589_);
return v___x_1591_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg(){
_start:
{
lean_object* v___x_1593_; lean_object* v___x_1594_; 
v___x_1593_ = lean_obj_once(&lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___closed__0, &lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___closed__0_once, _init_lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___closed__0);
v___x_1594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1594_, 0, v___x_1593_);
return v___x_1594_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg___boxed(lean_object* v___y_1595_){
_start:
{
lean_object* v_res_1596_; 
v_res_1596_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg();
return v_res_1596_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0(lean_object* v_00_u03b1_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_){
_start:
{
lean_object* v___x_1605_; 
v___x_1605_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg();
return v___x_1605_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___boxed(lean_object* v_00_u03b1_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_){
_start:
{
lean_object* v_res_1614_; 
v_res_1614_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0(v_00_u03b1_1606_, v___y_1607_, v___y_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_);
lean_dec(v___y_1612_);
lean_dec_ref(v___y_1611_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
lean_dec(v___y_1608_);
lean_dec_ref(v___y_1607_);
return v_res_1614_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; 
v___x_1623_ = lean_box(0);
v___x_1624_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__4));
v___x_1625_ = l_Lean_Expr_const___override(v___x_1624_, v___x_1623_);
return v___x_1625_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__9(void){
_start:
{
lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1632_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8));
v___x_1633_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__7));
v___x_1634_ = l_Lean_Expr_const___override(v___x_1633_, v___x_1632_);
return v___x_1634_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12(void){
_start:
{
lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; 
v___x_1639_ = lean_box(0);
v___x_1640_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__11));
v___x_1641_ = l_Lean_Expr_const___override(v___x_1640_, v___x_1639_);
return v___x_1641_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__15(void){
_start:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; 
v___x_1647_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8));
v___x_1648_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__14));
v___x_1649_ = l_Lean_Expr_const___override(v___x_1648_, v___x_1647_);
return v___x_1649_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__16(void){
_start:
{
lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; 
v___x_1650_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5);
v___x_1651_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__15, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__15_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__15);
v___x_1652_ = l_Lean_Expr_app___override(v___x_1651_, v___x_1650_);
return v___x_1652_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__20(void){
_start:
{
lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; 
v___x_1659_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__19));
v___x_1660_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__18));
v___x_1661_ = l_Lean_Expr_const___override(v___x_1660_, v___x_1659_);
return v___x_1661_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__21(void){
_start:
{
lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; 
v___x_1662_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5);
v___x_1663_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__20, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__20_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__20);
v___x_1664_ = l_Lean_Expr_app___override(v___x_1663_, v___x_1662_);
return v___x_1664_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__25(void){
_start:
{
lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; 
v___x_1670_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__19));
v___x_1671_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__24));
v___x_1672_ = l_Lean_Expr_const___override(v___x_1671_, v___x_1670_);
return v___x_1672_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__26(void){
_start:
{
lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; 
v___x_1673_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5);
v___x_1674_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__25, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__25_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__25);
v___x_1675_ = l_Lean_Expr_app___override(v___x_1674_, v___x_1673_);
return v___x_1675_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__30(void){
_start:
{
lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; 
v___x_1681_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__19));
v___x_1682_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__29));
v___x_1683_ = l_Lean_Expr_const___override(v___x_1682_, v___x_1681_);
return v___x_1683_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__33(void){
_start:
{
lean_object* v___x_1689_; lean_object* v___x_1690_; lean_object* v___x_1691_; 
v___x_1689_ = lean_box(0);
v___x_1690_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__32));
v___x_1691_ = l_Lean_Expr_const___override(v___x_1690_, v___x_1689_);
return v___x_1691_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__34(void){
_start:
{
lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; 
v___x_1692_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__33, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__33_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__33);
v___x_1693_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__30, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__30_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__30);
v___x_1694_ = l_Lean_Expr_app___override(v___x_1693_, v___x_1692_);
return v___x_1694_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__37(void){
_start:
{
lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; 
v___x_1698_ = lean_box(0);
v___x_1699_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__36));
v___x_1700_ = l_Lean_Expr_const___override(v___x_1699_, v___x_1698_);
return v___x_1700_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41(void){
_start:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; 
v___x_1706_ = lean_box(0);
v___x_1707_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__40));
v___x_1708_ = l_Lean_Expr_const___override(v___x_1707_, v___x_1706_);
return v___x_1708_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__42(void){
_start:
{
lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1709_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41);
v___x_1710_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__37, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__37_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__37);
v___x_1711_ = l_Lean_Expr_app___override(v___x_1710_, v___x_1709_);
return v___x_1711_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45(void){
_start:
{
lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; 
v___x_1717_ = lean_box(0);
v___x_1718_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__44));
v___x_1719_ = l_Lean_Expr_const___override(v___x_1718_, v___x_1717_);
return v___x_1719_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__46(void){
_start:
{
lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1720_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45);
v___x_1721_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__42, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__42_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__42);
v___x_1722_ = l_Lean_Expr_app___override(v___x_1721_, v___x_1720_);
return v___x_1722_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50(void){
_start:
{
lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; 
v___x_1729_ = lean_box(0);
v___x_1730_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__49));
v___x_1731_ = l_Lean_Expr_const___override(v___x_1730_, v___x_1729_);
return v___x_1731_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__51(void){
_start:
{
lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; 
v___x_1732_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50);
v___x_1733_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__46, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__46_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__46);
v___x_1734_ = l_Lean_Expr_app___override(v___x_1733_, v___x_1732_);
return v___x_1734_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__52(void){
_start:
{
lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; 
v___x_1735_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__51, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__51_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__51);
v___x_1736_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__34, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__34_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__34);
v___x_1737_ = l_Lean_Expr_app___override(v___x_1736_, v___x_1735_);
return v___x_1737_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__55(void){
_start:
{
lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; 
v___x_1742_ = lean_box(0);
v___x_1743_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__54));
v___x_1744_ = l_Lean_Expr_const___override(v___x_1743_, v___x_1742_);
return v___x_1744_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__56(void){
_start:
{
lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; 
v___x_1745_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__41);
v___x_1746_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__55, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__55_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__55);
v___x_1747_ = l_Lean_Expr_app___override(v___x_1746_, v___x_1745_);
return v___x_1747_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__57(void){
_start:
{
lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; 
v___x_1748_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__45);
v___x_1749_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__56, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__56_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__56);
v___x_1750_ = l_Lean_Expr_app___override(v___x_1749_, v___x_1748_);
return v___x_1750_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__58(void){
_start:
{
lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; 
v___x_1751_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__50);
v___x_1752_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__57, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__57_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__57);
v___x_1753_ = l_Lean_Expr_app___override(v___x_1752_, v___x_1751_);
return v___x_1753_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__61(void){
_start:
{
lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; 
v___x_1759_ = lean_box(0);
v___x_1760_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__60));
v___x_1761_ = l_Lean_Expr_const___override(v___x_1760_, v___x_1759_);
return v___x_1761_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__62(void){
_start:
{
lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; 
v___x_1762_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__61, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__61_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__61);
v___x_1763_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__58, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__58_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__58);
v___x_1764_ = l_Lean_Expr_app___override(v___x_1763_, v___x_1762_);
return v___x_1764_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__63(void){
_start:
{
lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; 
v___x_1765_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__62, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__62_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__62);
v___x_1766_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__52, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__52_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__52);
v___x_1767_ = l_Lean_Expr_app___override(v___x_1766_, v___x_1765_);
return v___x_1767_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__64(void){
_start:
{
lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; 
v___x_1768_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__63, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__63_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__63);
v___x_1769_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__26, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__26_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__26);
v___x_1770_ = l_Lean_Expr_app___override(v___x_1769_, v___x_1768_);
return v___x_1770_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65(void){
_start:
{
lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; 
v___x_1771_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__64, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__64_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__64);
v___x_1772_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__21, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__21_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__21);
v___x_1773_ = l_Lean_Expr_app___override(v___x_1772_, v___x_1771_);
return v___x_1773_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__66(void){
_start:
{
lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; 
v___x_1774_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65);
v___x_1775_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__16, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__16_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__16);
v___x_1776_ = l_Lean_Expr_app___override(v___x_1775_, v___x_1774_);
return v___x_1776_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69(void){
_start:
{
lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; 
v___x_1782_ = lean_box(0);
v___x_1783_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__68));
v___x_1784_ = l_Lean_Expr_const___override(v___x_1783_, v___x_1782_);
return v___x_1784_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__70(void){
_start:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; 
v___x_1785_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69);
v___x_1786_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__66, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__66_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__66);
v___x_1787_ = l_Lean_Expr_app___override(v___x_1786_, v___x_1785_);
return v___x_1787_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__73(void){
_start:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; 
v___x_1793_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__8));
v___x_1794_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__72));
v___x_1795_ = l_Lean_Expr_const___override(v___x_1794_, v___x_1793_);
return v___x_1795_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__74(void){
_start:
{
lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; 
v___x_1796_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__5);
v___x_1797_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__73, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__73_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__73);
v___x_1798_ = l_Lean_Expr_app___override(v___x_1797_, v___x_1796_);
return v___x_1798_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__75(void){
_start:
{
lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; 
v___x_1799_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__65);
v___x_1800_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__74, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__74_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__74);
v___x_1801_ = l_Lean_Expr_app___override(v___x_1800_, v___x_1799_);
return v___x_1801_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__76(void){
_start:
{
lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; 
v___x_1802_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__69);
v___x_1803_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__75, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__75_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__75);
v___x_1804_ = l_Lean_Expr_app___override(v___x_1803_, v___x_1802_);
return v___x_1804_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__79(void){
_start:
{
lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; 
v___x_1809_ = lean_box(0);
v___x_1810_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__78));
v___x_1811_ = l_Lean_Expr_const___override(v___x_1810_, v___x_1809_);
return v___x_1811_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__83(void){
_start:
{
lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; 
v___x_1817_ = lean_box(0);
v___x_1818_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__82));
v___x_1819_ = l_Lean_Expr_const___override(v___x_1818_, v___x_1817_);
return v___x_1819_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__96(void){
_start:
{
lean_object* v___x_1842_; lean_object* v___x_1843_; 
v___x_1842_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__95));
v___x_1843_ = l_String_toRawSubstring_x27(v___x_1842_);
return v___x_1843_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98(void){
_start:
{
lean_object* v___x_1846_; 
v___x_1846_ = l_Array_mkArray0(lean_box(0));
return v___x_1846_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0(lean_object* v_stx_1890_, lean_object* v_expectedType_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_){
_start:
{
lean_object* v___x_1899_; uint8_t v___x_1900_; 
v___x_1899_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2));
lean_inc(v_stx_1890_);
v___x_1900_ = l_Lean_Syntax_isOfKind(v_stx_1890_, v___x_1899_);
if (v___x_1900_ == 0)
{
lean_object* v___x_1901_; 
lean_dec_ref(v_expectedType_1891_);
lean_dec(v_stx_1890_);
v___x_1901_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg();
return v___x_1901_;
}
else
{
lean_object* v___x_1902_; 
v___x_1902_ = lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f(v___y_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
if (lean_obj_tag(v___x_1902_) == 0)
{
lean_object* v_a_1903_; lean_object* v___x_1904_; lean_object* v___x_1905_; 
v_a_1903_ = lean_ctor_get(v___x_1902_, 0);
lean_inc(v_a_1903_);
lean_dec_ref_known(v___x_1902_, 1);
v___x_1904_ = lean_unsigned_to_nat(1u);
v___x_1905_ = l_Lean_Syntax_getArg(v_stx_1890_, v___x_1904_);
lean_dec(v_stx_1890_);
if (lean_obj_tag(v_a_1903_) == 0)
{
lean_object* v___x_1906_; lean_object* v___x_1907_; 
v___x_1906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1906_, 0, v_expectedType_1891_);
v___x_1907_ = l_Lean_Elab_Term_elabTerm(v___x_1905_, v___x_1906_, v___x_1900_, v___x_1900_, v___y_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
return v___x_1907_;
}
else
{
lean_object* v_val_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_2056_; 
v_val_1908_ = lean_ctor_get(v_a_1903_, 0);
v_isSharedCheck_2056_ = !lean_is_exclusive(v_a_1903_);
if (v_isSharedCheck_2056_ == 0)
{
v___x_1910_ = v_a_1903_;
v_isShared_1911_ = v_isSharedCheck_2056_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_val_1908_);
lean_dec(v_a_1903_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_2056_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v_snd_1912_; lean_object* v_snd_1913_; lean_object* v_snd_1914_; lean_object* v_fst_1915_; lean_object* v___x_1917_; uint8_t v_isShared_1918_; uint8_t v_isSharedCheck_2054_; 
v_snd_1912_ = lean_ctor_get(v_val_1908_, 1);
lean_inc(v_snd_1912_);
v_snd_1913_ = lean_ctor_get(v_snd_1912_, 1);
lean_inc(v_snd_1913_);
v_snd_1914_ = lean_ctor_get(v_snd_1913_, 1);
lean_inc(v_snd_1914_);
v_fst_1915_ = lean_ctor_get(v_val_1908_, 0);
v_isSharedCheck_2054_ = !lean_is_exclusive(v_val_1908_);
if (v_isSharedCheck_2054_ == 0)
{
lean_object* v_unused_2055_; 
v_unused_2055_ = lean_ctor_get(v_val_1908_, 1);
lean_dec(v_unused_2055_);
v___x_1917_ = v_val_1908_;
v_isShared_1918_ = v_isSharedCheck_2054_;
goto v_resetjp_1916_;
}
else
{
lean_inc(v_fst_1915_);
lean_dec(v_val_1908_);
v___x_1917_ = lean_box(0);
v_isShared_1918_ = v_isSharedCheck_2054_;
goto v_resetjp_1916_;
}
v_resetjp_1916_:
{
lean_object* v_fst_1919_; lean_object* v___x_1921_; uint8_t v_isShared_1922_; uint8_t v_isSharedCheck_2052_; 
v_fst_1919_ = lean_ctor_get(v_snd_1912_, 0);
v_isSharedCheck_2052_ = !lean_is_exclusive(v_snd_1912_);
if (v_isSharedCheck_2052_ == 0)
{
lean_object* v_unused_2053_; 
v_unused_2053_ = lean_ctor_get(v_snd_1912_, 1);
lean_dec(v_unused_2053_);
v___x_1921_ = v_snd_1912_;
v_isShared_1922_ = v_isSharedCheck_2052_;
goto v_resetjp_1920_;
}
else
{
lean_inc(v_fst_1919_);
lean_dec(v_snd_1912_);
v___x_1921_ = lean_box(0);
v_isShared_1922_ = v_isSharedCheck_2052_;
goto v_resetjp_1920_;
}
v_resetjp_1920_:
{
lean_object* v_fst_1923_; lean_object* v___x_1925_; uint8_t v_isShared_1926_; uint8_t v_isSharedCheck_2050_; 
v_fst_1923_ = lean_ctor_get(v_snd_1913_, 0);
v_isSharedCheck_2050_ = !lean_is_exclusive(v_snd_1913_);
if (v_isSharedCheck_2050_ == 0)
{
lean_object* v_unused_2051_; 
v_unused_2051_ = lean_ctor_get(v_snd_1913_, 1);
lean_dec(v_unused_2051_);
v___x_1925_ = v_snd_1913_;
v_isShared_1926_ = v_isSharedCheck_2050_;
goto v_resetjp_1924_;
}
else
{
lean_inc(v_fst_1923_);
lean_dec(v_snd_1913_);
v___x_1925_ = lean_box(0);
v_isShared_1926_ = v_isSharedCheck_2050_;
goto v_resetjp_1924_;
}
v_resetjp_1924_:
{
lean_object* v_fst_1927_; lean_object* v_snd_1928_; lean_object* v___x_1930_; uint8_t v_isShared_1931_; uint8_t v_isSharedCheck_2049_; 
v_fst_1927_ = lean_ctor_get(v_snd_1914_, 0);
v_snd_1928_ = lean_ctor_get(v_snd_1914_, 1);
v_isSharedCheck_2049_ = !lean_is_exclusive(v_snd_1914_);
if (v_isSharedCheck_2049_ == 0)
{
v___x_1930_ = v_snd_1914_;
v_isShared_1931_ = v_isSharedCheck_2049_;
goto v_resetjp_1929_;
}
else
{
lean_inc(v_snd_1928_);
lean_inc(v_fst_1927_);
lean_dec(v_snd_1914_);
v___x_1930_ = lean_box(0);
v_isShared_1931_ = v_isSharedCheck_2049_;
goto v_resetjp_1929_;
}
v_resetjp_1929_:
{
lean_object* v___x_1932_; 
v___x_1932_ = l_Lean_FVarId_getUserName___redArg(v_fst_1915_, v___y_1894_, v___y_1896_, v___y_1897_);
if (lean_obj_tag(v___x_1932_) == 0)
{
lean_object* v_a_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; 
v_a_1933_ = lean_ctor_get(v___x_1932_, 0);
lean_inc(v_a_1933_);
lean_dec_ref_known(v___x_1932_, 1);
v___x_1934_ = l_Lean_Name_eraseMacroScopes(v_a_1933_);
lean_dec(v_a_1933_);
v___x_1935_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__0));
v___x_1936_ = lean_name_append_after(v___x_1934_, v___x_1935_);
v___x_1937_ = l_Lean_Core_mkFreshUserName(v___x_1936_, v___y_1896_, v___y_1897_);
if (lean_obj_tag(v___x_1937_) == 0)
{
lean_object* v_a_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; 
v_a_1938_ = lean_ctor_get(v___x_1937_, 0);
lean_inc(v_a_1938_);
lean_dec_ref_known(v___x_1937_, 1);
v___x_1939_ = lean_box(0);
v___x_1940_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__9, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__9_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__9);
v___x_1941_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12);
lean_inc(v_fst_1919_);
v___x_1942_ = l_Lean_Expr_app___override(v___x_1941_, v_fst_1919_);
lean_inc(v_fst_1923_);
v___x_1943_ = l_Lean_Expr_app___override(v___x_1942_, v_fst_1923_);
lean_inc(v_fst_1927_);
v___x_1944_ = l_Lean_Expr_app___override(v___x_1943_, v_fst_1927_);
lean_inc(v_snd_1928_);
v___x_1945_ = l_Lean_Expr_app___override(v___x_1944_, v_snd_1928_);
v___x_1946_ = l_Lean_Expr_app___override(v___x_1940_, v___x_1945_);
v___x_1947_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__70, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__70_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__70);
lean_inc_ref(v___x_1946_);
v___x_1948_ = l_Lean_Expr_app___override(v___x_1947_, v___x_1946_);
v___x_1949_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__76, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__76_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__76);
v___x_1950_ = l_Lean_Expr_app___override(v___x_1949_, v___x_1946_);
v___x_1951_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__79, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__79_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__79);
v___x_1952_ = l_Lean_Expr_app___override(v___x_1951_, v_fst_1919_);
v___x_1953_ = l_Lean_Expr_app___override(v___x_1952_, v_fst_1923_);
v___x_1954_ = l_Lean_Expr_app___override(v___x_1953_, v_fst_1927_);
v___x_1955_ = l_Lean_Expr_app___override(v___x_1954_, v_snd_1928_);
v___x_1956_ = l_Lean_Expr_app___override(v___x_1950_, v___x_1955_);
v___x_1957_ = l_Lean_Expr_app___override(v___x_1948_, v___x_1956_);
v___x_1958_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__83, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__83_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__83);
v___x_1959_ = l_Lean_Expr_app___override(v___x_1957_, v___x_1958_);
v___x_1960_ = l_Lean_Elab_Term_exprToSyntax(v___x_1959_, v___y_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
if (lean_obj_tag(v___x_1960_) == 0)
{
lean_object* v_a_1961_; lean_object* v_ref_1962_; lean_object* v_quotContext_1963_; lean_object* v_currMacroScope_1964_; uint8_t v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1970_; 
v_a_1961_ = lean_ctor_get(v___x_1960_, 0);
lean_inc(v_a_1961_);
lean_dec_ref_known(v___x_1960_, 1);
v_ref_1962_ = lean_ctor_get(v___y_1896_, 5);
v_quotContext_1963_ = lean_ctor_get(v___y_1896_, 10);
v_currMacroScope_1964_ = lean_ctor_get(v___y_1896_, 11);
v___x_1965_ = 0;
v___x_1966_ = l_Lean_SourceInfo_fromRef(v_ref_1962_, v___x_1965_);
v___x_1967_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__85));
v___x_1968_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__86));
lean_inc(v___x_1966_);
if (v_isShared_1931_ == 0)
{
lean_ctor_set_tag(v___x_1930_, 2);
lean_ctor_set(v___x_1930_, 1, v___x_1968_);
lean_ctor_set(v___x_1930_, 0, v___x_1966_);
v___x_1970_ = v___x_1930_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_2024_; 
v_reuseFailAlloc_2024_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2024_, 0, v___x_1966_);
lean_ctor_set(v_reuseFailAlloc_2024_, 1, v___x_1968_);
v___x_1970_ = v_reuseFailAlloc_2024_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1974_; 
v___x_1971_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__89));
v___x_1972_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__90));
lean_inc(v___x_1966_);
if (v_isShared_1926_ == 0)
{
lean_ctor_set_tag(v___x_1925_, 2);
lean_ctor_set(v___x_1925_, 1, v___x_1971_);
lean_ctor_set(v___x_1925_, 0, v___x_1966_);
v___x_1974_ = v___x_1925_;
goto v_reusejp_1973_;
}
else
{
lean_object* v_reuseFailAlloc_2023_; 
v_reuseFailAlloc_2023_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2023_, 0, v___x_1966_);
lean_ctor_set(v_reuseFailAlloc_2023_, 1, v___x_1971_);
v___x_1974_ = v_reuseFailAlloc_2023_;
goto v_reusejp_1973_;
}
v_reusejp_1973_:
{
lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1986_; 
v___x_1975_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__92));
v___x_1976_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__94));
v___x_1977_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__96, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__96_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__96);
v___x_1978_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__97));
lean_inc(v_currMacroScope_1964_);
lean_inc(v_quotContext_1963_);
v___x_1979_ = l_Lean_addMacroScope(v_quotContext_1963_, v___x_1978_, v_currMacroScope_1964_);
lean_inc_n(v___x_1966_, 4);
v___x_1980_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1980_, 0, v___x_1966_);
lean_ctor_set(v___x_1980_, 1, v___x_1977_);
lean_ctor_set(v___x_1980_, 2, v___x_1979_);
lean_ctor_set(v___x_1980_, 3, v___x_1939_);
lean_inc_ref(v___x_1980_);
v___x_1981_ = l_Lean_Syntax_node1(v___x_1966_, v___x_1976_, v___x_1980_);
v___x_1982_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98);
v___x_1983_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1983_, 0, v___x_1966_);
lean_ctor_set(v___x_1983_, 1, v___x_1976_);
lean_ctor_set(v___x_1983_, 2, v___x_1982_);
v___x_1984_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__99));
if (v_isShared_1922_ == 0)
{
lean_ctor_set_tag(v___x_1921_, 2);
lean_ctor_set(v___x_1921_, 1, v___x_1984_);
lean_ctor_set(v___x_1921_, 0, v___x_1966_);
v___x_1986_ = v___x_1921_;
goto v_reusejp_1985_;
}
else
{
lean_object* v_reuseFailAlloc_2022_; 
v_reuseFailAlloc_2022_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2022_, 0, v___x_1966_);
lean_ctor_set(v_reuseFailAlloc_2022_, 1, v___x_1984_);
v___x_1986_ = v_reuseFailAlloc_2022_;
goto v_reusejp_1985_;
}
v_reusejp_1985_:
{
lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1990_; 
v___x_1987_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__100));
v___x_1988_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101));
lean_inc(v___x_1966_);
if (v_isShared_1918_ == 0)
{
lean_ctor_set_tag(v___x_1917_, 2);
lean_ctor_set(v___x_1917_, 1, v___x_1987_);
lean_ctor_set(v___x_1917_, 0, v___x_1966_);
v___x_1990_ = v___x_1917_;
goto v_reusejp_1989_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v___x_1966_);
lean_ctor_set(v_reuseFailAlloc_2021_, 1, v___x_1987_);
v___x_1990_ = v_reuseFailAlloc_2021_;
goto v_reusejp_1989_;
}
v_reusejp_1989_:
{
lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2018_; 
v___x_1991_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103));
lean_inc_ref_n(v___x_1983_, 4);
lean_inc_n(v___x_1966_, 14);
v___x_1992_ = l_Lean_Syntax_node1(v___x_1966_, v___x_1991_, v___x_1983_);
v___x_1993_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105));
v___x_1994_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107));
v___x_1995_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109));
v___x_1996_ = l_Lean_mkIdent(v_a_1938_);
v___x_1997_ = l_Lean_Syntax_node1(v___x_1966_, v___x_1995_, v___x_1996_);
v___x_1998_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__110));
v___x_1999_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1999_, 0, v___x_1966_);
lean_ctor_set(v___x_1999_, 1, v___x_1998_);
v___x_2000_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__112));
v___x_2001_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__113));
v___x_2002_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2002_, 0, v___x_1966_);
lean_ctor_set(v___x_2002_, 1, v___x_2001_);
v___x_2003_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__115));
v___x_2004_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__116));
v___x_2005_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2005_, 0, v___x_1966_);
lean_ctor_set(v___x_2005_, 1, v___x_2004_);
v___x_2006_ = l_Lean_Syntax_node1(v___x_1966_, v___x_2003_, v___x_2005_);
v___x_2007_ = l_Lean_Syntax_node3(v___x_1966_, v___x_2000_, v___x_1980_, v___x_2002_, v___x_2006_);
v___x_2008_ = l_Lean_Syntax_node5(v___x_1966_, v___x_1994_, v___x_1997_, v___x_1983_, v___x_1983_, v___x_1999_, v___x_2007_);
v___x_2009_ = l_Lean_Syntax_node1(v___x_1966_, v___x_1993_, v___x_2008_);
v___x_2010_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__3));
v___x_2011_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2011_, 0, v___x_1966_);
lean_ctor_set(v___x_2011_, 1, v___x_2010_);
v___x_2012_ = l_Lean_Syntax_node2(v___x_1966_, v___x_1899_, v___x_2011_, v___x_1905_);
v___x_2013_ = l_Lean_Syntax_node5(v___x_1966_, v___x_1988_, v___x_1990_, v___x_1992_, v___x_2009_, v___x_1983_, v___x_2012_);
v___x_2014_ = l_Lean_Syntax_node4(v___x_1966_, v___x_1975_, v___x_1981_, v___x_1983_, v___x_1986_, v___x_2013_);
v___x_2015_ = l_Lean_Syntax_node2(v___x_1966_, v___x_1972_, v___x_1974_, v___x_2014_);
v___x_2016_ = l_Lean_Syntax_node3(v___x_1966_, v___x_1967_, v_a_1961_, v___x_1970_, v___x_2015_);
if (v_isShared_1911_ == 0)
{
lean_ctor_set(v___x_1910_, 0, v_expectedType_1891_);
v___x_2018_ = v___x_1910_;
goto v_reusejp_2017_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v_expectedType_1891_);
v___x_2018_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2017_;
}
v_reusejp_2017_:
{
lean_object* v___x_2019_; 
v___x_2019_ = l_Lean_Elab_Term_elabTerm(v___x_2016_, v___x_2018_, v___x_1900_, v___x_1900_, v___y_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
return v___x_2019_;
}
}
}
}
}
}
else
{
lean_object* v_a_2025_; lean_object* v___x_2027_; uint8_t v_isShared_2028_; uint8_t v_isSharedCheck_2032_; 
lean_dec(v_a_1938_);
lean_del_object(v___x_1930_);
lean_del_object(v___x_1925_);
lean_del_object(v___x_1921_);
lean_del_object(v___x_1917_);
lean_del_object(v___x_1910_);
lean_dec(v___x_1905_);
lean_dec_ref(v_expectedType_1891_);
v_a_2025_ = lean_ctor_get(v___x_1960_, 0);
v_isSharedCheck_2032_ = !lean_is_exclusive(v___x_1960_);
if (v_isSharedCheck_2032_ == 0)
{
v___x_2027_ = v___x_1960_;
v_isShared_2028_ = v_isSharedCheck_2032_;
goto v_resetjp_2026_;
}
else
{
lean_inc(v_a_2025_);
lean_dec(v___x_1960_);
v___x_2027_ = lean_box(0);
v_isShared_2028_ = v_isSharedCheck_2032_;
goto v_resetjp_2026_;
}
v_resetjp_2026_:
{
lean_object* v___x_2030_; 
if (v_isShared_2028_ == 0)
{
v___x_2030_ = v___x_2027_;
goto v_reusejp_2029_;
}
else
{
lean_object* v_reuseFailAlloc_2031_; 
v_reuseFailAlloc_2031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2031_, 0, v_a_2025_);
v___x_2030_ = v_reuseFailAlloc_2031_;
goto v_reusejp_2029_;
}
v_reusejp_2029_:
{
return v___x_2030_;
}
}
}
}
else
{
lean_object* v_a_2033_; lean_object* v___x_2035_; uint8_t v_isShared_2036_; uint8_t v_isSharedCheck_2040_; 
lean_del_object(v___x_1930_);
lean_dec(v_snd_1928_);
lean_dec(v_fst_1927_);
lean_del_object(v___x_1925_);
lean_dec(v_fst_1923_);
lean_del_object(v___x_1921_);
lean_dec(v_fst_1919_);
lean_del_object(v___x_1917_);
lean_del_object(v___x_1910_);
lean_dec(v___x_1905_);
lean_dec_ref(v_expectedType_1891_);
v_a_2033_ = lean_ctor_get(v___x_1937_, 0);
v_isSharedCheck_2040_ = !lean_is_exclusive(v___x_1937_);
if (v_isSharedCheck_2040_ == 0)
{
v___x_2035_ = v___x_1937_;
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
else
{
lean_inc(v_a_2033_);
lean_dec(v___x_1937_);
v___x_2035_ = lean_box(0);
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
v_resetjp_2034_:
{
lean_object* v___x_2038_; 
if (v_isShared_2036_ == 0)
{
v___x_2038_ = v___x_2035_;
goto v_reusejp_2037_;
}
else
{
lean_object* v_reuseFailAlloc_2039_; 
v_reuseFailAlloc_2039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2039_, 0, v_a_2033_);
v___x_2038_ = v_reuseFailAlloc_2039_;
goto v_reusejp_2037_;
}
v_reusejp_2037_:
{
return v___x_2038_;
}
}
}
}
else
{
lean_object* v_a_2041_; lean_object* v___x_2043_; uint8_t v_isShared_2044_; uint8_t v_isSharedCheck_2048_; 
lean_del_object(v___x_1930_);
lean_dec(v_snd_1928_);
lean_dec(v_fst_1927_);
lean_del_object(v___x_1925_);
lean_dec(v_fst_1923_);
lean_del_object(v___x_1921_);
lean_dec(v_fst_1919_);
lean_del_object(v___x_1917_);
lean_del_object(v___x_1910_);
lean_dec(v___x_1905_);
lean_dec_ref(v_expectedType_1891_);
v_a_2041_ = lean_ctor_get(v___x_1932_, 0);
v_isSharedCheck_2048_ = !lean_is_exclusive(v___x_1932_);
if (v_isSharedCheck_2048_ == 0)
{
v___x_2043_ = v___x_1932_;
v_isShared_2044_ = v_isSharedCheck_2048_;
goto v_resetjp_2042_;
}
else
{
lean_inc(v_a_2041_);
lean_dec(v___x_1932_);
v___x_2043_ = lean_box(0);
v_isShared_2044_ = v_isSharedCheck_2048_;
goto v_resetjp_2042_;
}
v_resetjp_2042_:
{
lean_object* v___x_2046_; 
if (v_isShared_2044_ == 0)
{
v___x_2046_ = v___x_2043_;
goto v_reusejp_2045_;
}
else
{
lean_object* v_reuseFailAlloc_2047_; 
v_reuseFailAlloc_2047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2047_, 0, v_a_2041_);
v___x_2046_ = v_reuseFailAlloc_2047_;
goto v_reusejp_2045_;
}
v_reusejp_2045_:
{
return v___x_2046_;
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
else
{
lean_object* v_a_2057_; lean_object* v___x_2059_; uint8_t v_isShared_2060_; uint8_t v_isSharedCheck_2064_; 
lean_dec_ref(v_expectedType_1891_);
lean_dec(v_stx_1890_);
v_a_2057_ = lean_ctor_get(v___x_1902_, 0);
v_isSharedCheck_2064_ = !lean_is_exclusive(v___x_1902_);
if (v_isSharedCheck_2064_ == 0)
{
v___x_2059_ = v___x_1902_;
v_isShared_2060_ = v_isSharedCheck_2064_;
goto v_resetjp_2058_;
}
else
{
lean_inc(v_a_2057_);
lean_dec(v___x_1902_);
v___x_2059_ = lean_box(0);
v_isShared_2060_ = v_isSharedCheck_2064_;
goto v_resetjp_2058_;
}
v_resetjp_2058_:
{
lean_object* v___x_2062_; 
if (v_isShared_2060_ == 0)
{
v___x_2062_ = v___x_2059_;
goto v_reusejp_2061_;
}
else
{
lean_object* v_reuseFailAlloc_2063_; 
v_reuseFailAlloc_2063_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2063_, 0, v_a_2057_);
v___x_2062_ = v_reuseFailAlloc_2063_;
goto v_reusejp_2061_;
}
v_reusejp_2061_:
{
return v___x_2062_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___boxed(lean_object* v_stx_2065_, lean_object* v_expectedType_2066_, lean_object* v___y_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_){
_start:
{
lean_object* v_res_2074_; 
v_res_2074_ = lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0(v_stx_2065_, v_expectedType_2066_, v___y_2067_, v___y_2068_, v___y_2069_, v___y_2070_, v___y_2071_, v___y_2072_);
lean_dec(v___y_2072_);
lean_dec_ref(v___y_2071_);
lean_dec(v___y_2070_);
lean_dec_ref(v___y_2069_);
lean_dec(v___y_2068_);
lean_dec_ref(v___y_2067_);
return v_res_2074_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1(lean_object* v_stx_2075_, lean_object* v_expectedType_x3f_2076_, lean_object* v_a_2077_, lean_object* v_a_2078_, lean_object* v_a_2079_, lean_object* v_a_2080_, lean_object* v_a_2081_, lean_object* v_a_2082_){
_start:
{
lean_object* v___f_2084_; lean_object* v___x_2085_; 
v___f_2084_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___boxed), 9, 1);
lean_closure_set(v___f_2084_, 0, v_stx_2075_);
v___x_2085_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_2076_, v___f_2084_, v_a_2077_, v_a_2078_, v_a_2079_, v_a_2080_, v_a_2081_, v_a_2082_);
return v___x_2085_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___boxed(lean_object* v_stx_2086_, lean_object* v_expectedType_x3f_2087_, lean_object* v_a_2088_, lean_object* v_a_2089_, lean_object* v_a_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_, lean_object* v_a_2093_, lean_object* v_a_2094_){
_start:
{
lean_object* v_res_2095_; 
v_res_2095_ = lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1(v_stx_2086_, v_expectedType_x3f_2087_, v_a_2088_, v_a_2089_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_);
lean_dec(v_a_2093_);
lean_dec_ref(v_a_2092_);
lean_dec(v_a_2091_);
lean_dec_ref(v_a_2090_);
lean_dec(v_a_2089_);
lean_dec_ref(v_a_2088_);
return v_res_2095_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0(lean_object* v_stx_2111_, lean_object* v_expectedType_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_){
_start:
{
lean_object* v___x_2120_; uint8_t v___x_2121_; 
v___x_2120_ = ((lean_object*)(lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2));
lean_inc(v_stx_2111_);
v___x_2121_ = l_Lean_Syntax_isOfKind(v_stx_2111_, v___x_2120_);
if (v___x_2121_ == 0)
{
lean_object* v___x_2122_; 
lean_dec_ref(v_expectedType_2112_);
lean_dec(v_stx_2111_);
v___x_2122_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1_spec__0___redArg();
return v___x_2122_;
}
else
{
lean_object* v___x_2123_; 
v___x_2123_ = lp_Qq_Qq_Impl_findRedundantLocalInstQuoted_x3f(v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_);
if (lean_obj_tag(v___x_2123_) == 0)
{
lean_object* v_a_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; 
v_a_2124_ = lean_ctor_get(v___x_2123_, 0);
lean_inc(v_a_2124_);
lean_dec_ref_known(v___x_2123_, 1);
v___x_2125_ = lean_unsigned_to_nat(1u);
v___x_2126_ = l_Lean_Syntax_getArg(v_stx_2111_, v___x_2125_);
lean_dec(v_stx_2111_);
if (lean_obj_tag(v_a_2124_) == 0)
{
lean_object* v___x_2127_; lean_object* v___x_2128_; 
v___x_2127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2127_, 0, v_expectedType_2112_);
v___x_2128_ = l_Lean_Elab_Term_elabTerm(v___x_2126_, v___x_2127_, v___x_2121_, v___x_2121_, v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_);
return v___x_2128_;
}
else
{
lean_object* v_val_2129_; lean_object* v___x_2131_; uint8_t v_isShared_2132_; uint8_t v_isSharedCheck_2245_; 
v_val_2129_ = lean_ctor_get(v_a_2124_, 0);
v_isSharedCheck_2245_ = !lean_is_exclusive(v_a_2124_);
if (v_isSharedCheck_2245_ == 0)
{
v___x_2131_ = v_a_2124_;
v_isShared_2132_ = v_isSharedCheck_2245_;
goto v_resetjp_2130_;
}
else
{
lean_inc(v_val_2129_);
lean_dec(v_a_2124_);
v___x_2131_ = lean_box(0);
v_isShared_2132_ = v_isSharedCheck_2245_;
goto v_resetjp_2130_;
}
v_resetjp_2130_:
{
lean_object* v_snd_2133_; lean_object* v_snd_2134_; lean_object* v_snd_2135_; lean_object* v_fst_2136_; lean_object* v___x_2138_; uint8_t v_isShared_2139_; uint8_t v_isSharedCheck_2243_; 
v_snd_2133_ = lean_ctor_get(v_val_2129_, 1);
lean_inc(v_snd_2133_);
v_snd_2134_ = lean_ctor_get(v_snd_2133_, 1);
lean_inc(v_snd_2134_);
v_snd_2135_ = lean_ctor_get(v_snd_2134_, 1);
lean_inc(v_snd_2135_);
v_fst_2136_ = lean_ctor_get(v_val_2129_, 0);
v_isSharedCheck_2243_ = !lean_is_exclusive(v_val_2129_);
if (v_isSharedCheck_2243_ == 0)
{
lean_object* v_unused_2244_; 
v_unused_2244_ = lean_ctor_get(v_val_2129_, 1);
lean_dec(v_unused_2244_);
v___x_2138_ = v_val_2129_;
v_isShared_2139_ = v_isSharedCheck_2243_;
goto v_resetjp_2137_;
}
else
{
lean_inc(v_fst_2136_);
lean_dec(v_val_2129_);
v___x_2138_ = lean_box(0);
v_isShared_2139_ = v_isSharedCheck_2243_;
goto v_resetjp_2137_;
}
v_resetjp_2137_:
{
lean_object* v_fst_2140_; lean_object* v___x_2142_; uint8_t v_isShared_2143_; uint8_t v_isSharedCheck_2241_; 
v_fst_2140_ = lean_ctor_get(v_snd_2133_, 0);
v_isSharedCheck_2241_ = !lean_is_exclusive(v_snd_2133_);
if (v_isSharedCheck_2241_ == 0)
{
lean_object* v_unused_2242_; 
v_unused_2242_ = lean_ctor_get(v_snd_2133_, 1);
lean_dec(v_unused_2242_);
v___x_2142_ = v_snd_2133_;
v_isShared_2143_ = v_isSharedCheck_2241_;
goto v_resetjp_2141_;
}
else
{
lean_inc(v_fst_2140_);
lean_dec(v_snd_2133_);
v___x_2142_ = lean_box(0);
v_isShared_2143_ = v_isSharedCheck_2241_;
goto v_resetjp_2141_;
}
v_resetjp_2141_:
{
lean_object* v_fst_2144_; lean_object* v___x_2146_; uint8_t v_isShared_2147_; uint8_t v_isSharedCheck_2239_; 
v_fst_2144_ = lean_ctor_get(v_snd_2134_, 0);
v_isSharedCheck_2239_ = !lean_is_exclusive(v_snd_2134_);
if (v_isSharedCheck_2239_ == 0)
{
lean_object* v_unused_2240_; 
v_unused_2240_ = lean_ctor_get(v_snd_2134_, 1);
lean_dec(v_unused_2240_);
v___x_2146_ = v_snd_2134_;
v_isShared_2147_ = v_isSharedCheck_2239_;
goto v_resetjp_2145_;
}
else
{
lean_inc(v_fst_2144_);
lean_dec(v_snd_2134_);
v___x_2146_ = lean_box(0);
v_isShared_2147_ = v_isSharedCheck_2239_;
goto v_resetjp_2145_;
}
v_resetjp_2145_:
{
lean_object* v_fst_2148_; lean_object* v_snd_2149_; lean_object* v___x_2151_; uint8_t v_isShared_2152_; uint8_t v_isSharedCheck_2238_; 
v_fst_2148_ = lean_ctor_get(v_snd_2135_, 0);
v_snd_2149_ = lean_ctor_get(v_snd_2135_, 1);
v_isSharedCheck_2238_ = !lean_is_exclusive(v_snd_2135_);
if (v_isSharedCheck_2238_ == 0)
{
v___x_2151_ = v_snd_2135_;
v_isShared_2152_ = v_isSharedCheck_2238_;
goto v_resetjp_2150_;
}
else
{
lean_inc(v_snd_2149_);
lean_inc(v_fst_2148_);
lean_dec(v_snd_2135_);
v___x_2151_ = lean_box(0);
v_isShared_2152_ = v_isSharedCheck_2238_;
goto v_resetjp_2150_;
}
v_resetjp_2150_:
{
lean_object* v___x_2153_; 
v___x_2153_ = l_Lean_FVarId_getUserName___redArg(v_fst_2136_, v___y_2115_, v___y_2117_, v___y_2118_);
if (lean_obj_tag(v___x_2153_) == 0)
{
lean_object* v_a_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; 
v_a_2154_ = lean_ctor_get(v___x_2153_, 0);
lean_inc(v_a_2154_);
lean_dec_ref_known(v___x_2153_, 1);
v___x_2155_ = l_Lean_Name_eraseMacroScopes(v_a_2154_);
lean_dec(v_a_2154_);
v___x_2156_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__0));
v___x_2157_ = lean_name_append_after(v___x_2155_, v___x_2156_);
v___x_2158_ = l_Lean_Core_mkFreshUserName(v___x_2157_, v___y_2117_, v___y_2118_);
if (lean_obj_tag(v___x_2158_) == 0)
{
lean_object* v_a_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; 
v_a_2159_ = lean_ctor_get(v___x_2158_, 0);
lean_inc(v_a_2159_);
lean_dec_ref_known(v___x_2158_, 1);
v___x_2160_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__12);
v___x_2161_ = l_Lean_Expr_app___override(v___x_2160_, v_fst_2140_);
v___x_2162_ = l_Lean_Expr_app___override(v___x_2161_, v_fst_2144_);
v___x_2163_ = l_Lean_Expr_app___override(v___x_2162_, v_fst_2148_);
v___x_2164_ = l_Lean_Expr_app___override(v___x_2163_, v_snd_2149_);
v___x_2165_ = l_Lean_Elab_Term_exprToSyntax(v___x_2164_, v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_);
if (lean_obj_tag(v___x_2165_) == 0)
{
lean_object* v_a_2166_; lean_object* v_ref_2167_; uint8_t v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2173_; 
v_a_2166_ = lean_ctor_get(v___x_2165_, 0);
lean_inc(v_a_2166_);
lean_dec_ref_known(v___x_2165_, 1);
v_ref_2167_ = lean_ctor_get(v___y_2117_, 5);
v___x_2168_ = 0;
v___x_2169_ = l_Lean_SourceInfo_fromRef(v_ref_2167_, v___x_2168_);
v___x_2170_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__100));
v___x_2171_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__101));
lean_inc(v___x_2169_);
if (v_isShared_2152_ == 0)
{
lean_ctor_set_tag(v___x_2151_, 2);
lean_ctor_set(v___x_2151_, 1, v___x_2170_);
lean_ctor_set(v___x_2151_, 0, v___x_2169_);
v___x_2173_ = v___x_2151_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2213_; 
v_reuseFailAlloc_2213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2213_, 0, v___x_2169_);
lean_ctor_set(v_reuseFailAlloc_2213_, 1, v___x_2170_);
v___x_2173_ = v_reuseFailAlloc_2213_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2187_; 
v___x_2174_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__103));
v___x_2175_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__94));
v___x_2176_ = lean_obj_once(&lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98, &lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98_once, _init_lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__98);
lean_inc_n(v___x_2169_, 4);
v___x_2177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2177_, 0, v___x_2169_);
lean_ctor_set(v___x_2177_, 1, v___x_2175_);
lean_ctor_set(v___x_2177_, 2, v___x_2176_);
lean_inc_ref(v___x_2177_);
v___x_2178_ = l_Lean_Syntax_node1(v___x_2169_, v___x_2174_, v___x_2177_);
v___x_2179_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__105));
v___x_2180_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__107));
v___x_2181_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__109));
v___x_2182_ = l_Lean_mkIdent(v_a_2159_);
v___x_2183_ = l_Lean_Syntax_node1(v___x_2169_, v___x_2181_, v___x_2182_);
v___x_2184_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__1));
v___x_2185_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__2));
if (v_isShared_2147_ == 0)
{
lean_ctor_set_tag(v___x_2146_, 2);
lean_ctor_set(v___x_2146_, 1, v___x_2185_);
lean_ctor_set(v___x_2146_, 0, v___x_2169_);
v___x_2187_ = v___x_2146_;
goto v_reusejp_2186_;
}
else
{
lean_object* v_reuseFailAlloc_2212_; 
v_reuseFailAlloc_2212_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2212_, 0, v___x_2169_);
lean_ctor_set(v_reuseFailAlloc_2212_, 1, v___x_2185_);
v___x_2187_ = v_reuseFailAlloc_2212_;
goto v_reusejp_2186_;
}
v_reusejp_2186_:
{
lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2192_; 
lean_inc_n(v___x_2169_, 3);
v___x_2188_ = l_Lean_Syntax_node2(v___x_2169_, v___x_2184_, v___x_2187_, v_a_2166_);
v___x_2189_ = l_Lean_Syntax_node1(v___x_2169_, v___x_2175_, v___x_2188_);
v___x_2190_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__Impl__termAssertInstancesCommuteImpl____1___lam__0___closed__110));
if (v_isShared_2143_ == 0)
{
lean_ctor_set_tag(v___x_2142_, 2);
lean_ctor_set(v___x_2142_, 1, v___x_2190_);
lean_ctor_set(v___x_2142_, 0, v___x_2169_);
v___x_2192_ = v___x_2142_;
goto v_reusejp_2191_;
}
else
{
lean_object* v_reuseFailAlloc_2211_; 
v_reuseFailAlloc_2211_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2211_, 0, v___x_2169_);
lean_ctor_set(v_reuseFailAlloc_2211_, 1, v___x_2190_);
v___x_2192_ = v_reuseFailAlloc_2211_;
goto v_reusejp_2191_;
}
v_reusejp_2191_:
{
lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2196_; 
v___x_2193_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__4));
v___x_2194_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__5));
lean_inc(v___x_2169_);
if (v_isShared_2139_ == 0)
{
lean_ctor_set_tag(v___x_2138_, 2);
lean_ctor_set(v___x_2138_, 1, v___x_2194_);
lean_ctor_set(v___x_2138_, 0, v___x_2169_);
v___x_2196_ = v___x_2138_;
goto v_reusejp_2195_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v___x_2169_);
lean_ctor_set(v_reuseFailAlloc_2210_, 1, v___x_2194_);
v___x_2196_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2195_;
}
v_reusejp_2195_:
{
lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2207_; 
v___x_2197_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___closed__6));
lean_inc_n(v___x_2169_, 6);
v___x_2198_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2198_, 0, v___x_2169_);
lean_ctor_set(v___x_2198_, 1, v___x_2197_);
lean_inc_ref_n(v___x_2177_, 2);
v___x_2199_ = l_Lean_Syntax_node3(v___x_2169_, v___x_2193_, v___x_2196_, v___x_2177_, v___x_2198_);
v___x_2200_ = l_Lean_Syntax_node5(v___x_2169_, v___x_2180_, v___x_2183_, v___x_2177_, v___x_2189_, v___x_2192_, v___x_2199_);
v___x_2201_ = l_Lean_Syntax_node1(v___x_2169_, v___x_2179_, v___x_2200_);
v___x_2202_ = ((lean_object*)(lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__5));
v___x_2203_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2203_, 0, v___x_2169_);
lean_ctor_set(v___x_2203_, 1, v___x_2202_);
v___x_2204_ = l_Lean_Syntax_node2(v___x_2169_, v___x_2120_, v___x_2203_, v___x_2126_);
v___x_2205_ = l_Lean_Syntax_node5(v___x_2169_, v___x_2171_, v___x_2173_, v___x_2178_, v___x_2201_, v___x_2177_, v___x_2204_);
if (v_isShared_2132_ == 0)
{
lean_ctor_set(v___x_2131_, 0, v_expectedType_2112_);
v___x_2207_ = v___x_2131_;
goto v_reusejp_2206_;
}
else
{
lean_object* v_reuseFailAlloc_2209_; 
v_reuseFailAlloc_2209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2209_, 0, v_expectedType_2112_);
v___x_2207_ = v_reuseFailAlloc_2209_;
goto v_reusejp_2206_;
}
v_reusejp_2206_:
{
lean_object* v___x_2208_; 
v___x_2208_ = l_Lean_Elab_Term_elabTerm(v___x_2205_, v___x_2207_, v___x_2121_, v___x_2121_, v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_);
return v___x_2208_;
}
}
}
}
}
}
else
{
lean_object* v_a_2214_; lean_object* v___x_2216_; uint8_t v_isShared_2217_; uint8_t v_isSharedCheck_2221_; 
lean_dec(v_a_2159_);
lean_del_object(v___x_2151_);
lean_del_object(v___x_2146_);
lean_del_object(v___x_2142_);
lean_del_object(v___x_2138_);
lean_del_object(v___x_2131_);
lean_dec(v___x_2126_);
lean_dec_ref(v_expectedType_2112_);
v_a_2214_ = lean_ctor_get(v___x_2165_, 0);
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2165_);
if (v_isSharedCheck_2221_ == 0)
{
v___x_2216_ = v___x_2165_;
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
else
{
lean_inc(v_a_2214_);
lean_dec(v___x_2165_);
v___x_2216_ = lean_box(0);
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
v_resetjp_2215_:
{
lean_object* v___x_2219_; 
if (v_isShared_2217_ == 0)
{
v___x_2219_ = v___x_2216_;
goto v_reusejp_2218_;
}
else
{
lean_object* v_reuseFailAlloc_2220_; 
v_reuseFailAlloc_2220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2220_, 0, v_a_2214_);
v___x_2219_ = v_reuseFailAlloc_2220_;
goto v_reusejp_2218_;
}
v_reusejp_2218_:
{
return v___x_2219_;
}
}
}
}
else
{
lean_object* v_a_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2229_; 
lean_del_object(v___x_2151_);
lean_dec(v_snd_2149_);
lean_dec(v_fst_2148_);
lean_del_object(v___x_2146_);
lean_dec(v_fst_2144_);
lean_del_object(v___x_2142_);
lean_dec(v_fst_2140_);
lean_del_object(v___x_2138_);
lean_del_object(v___x_2131_);
lean_dec(v___x_2126_);
lean_dec_ref(v_expectedType_2112_);
v_a_2222_ = lean_ctor_get(v___x_2158_, 0);
v_isSharedCheck_2229_ = !lean_is_exclusive(v___x_2158_);
if (v_isSharedCheck_2229_ == 0)
{
v___x_2224_ = v___x_2158_;
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_a_2222_);
lean_dec(v___x_2158_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v___x_2227_; 
if (v_isShared_2225_ == 0)
{
v___x_2227_ = v___x_2224_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2228_; 
v_reuseFailAlloc_2228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2228_, 0, v_a_2222_);
v___x_2227_ = v_reuseFailAlloc_2228_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
return v___x_2227_;
}
}
}
}
else
{
lean_object* v_a_2230_; lean_object* v___x_2232_; uint8_t v_isShared_2233_; uint8_t v_isSharedCheck_2237_; 
lean_del_object(v___x_2151_);
lean_dec(v_snd_2149_);
lean_dec(v_fst_2148_);
lean_del_object(v___x_2146_);
lean_dec(v_fst_2144_);
lean_del_object(v___x_2142_);
lean_dec(v_fst_2140_);
lean_del_object(v___x_2138_);
lean_del_object(v___x_2131_);
lean_dec(v___x_2126_);
lean_dec_ref(v_expectedType_2112_);
v_a_2230_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2237_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2237_ == 0)
{
v___x_2232_ = v___x_2153_;
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
else
{
lean_inc(v_a_2230_);
lean_dec(v___x_2153_);
v___x_2232_ = lean_box(0);
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
v_resetjp_2231_:
{
lean_object* v___x_2235_; 
if (v_isShared_2233_ == 0)
{
v___x_2235_ = v___x_2232_;
goto v_reusejp_2234_;
}
else
{
lean_object* v_reuseFailAlloc_2236_; 
v_reuseFailAlloc_2236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2236_, 0, v_a_2230_);
v___x_2235_ = v_reuseFailAlloc_2236_;
goto v_reusejp_2234_;
}
v_reusejp_2234_:
{
return v___x_2235_;
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
else
{
lean_object* v_a_2246_; lean_object* v___x_2248_; uint8_t v_isShared_2249_; uint8_t v_isSharedCheck_2253_; 
lean_dec_ref(v_expectedType_2112_);
lean_dec(v_stx_2111_);
v_a_2246_ = lean_ctor_get(v___x_2123_, 0);
v_isSharedCheck_2253_ = !lean_is_exclusive(v___x_2123_);
if (v_isSharedCheck_2253_ == 0)
{
v___x_2248_ = v___x_2123_;
v_isShared_2249_ = v_isSharedCheck_2253_;
goto v_resetjp_2247_;
}
else
{
lean_inc(v_a_2246_);
lean_dec(v___x_2123_);
v___x_2248_ = lean_box(0);
v_isShared_2249_ = v_isSharedCheck_2253_;
goto v_resetjp_2247_;
}
v_resetjp_2247_:
{
lean_object* v___x_2251_; 
if (v_isShared_2249_ == 0)
{
v___x_2251_ = v___x_2248_;
goto v_reusejp_2250_;
}
else
{
lean_object* v_reuseFailAlloc_2252_; 
v_reuseFailAlloc_2252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2252_, 0, v_a_2246_);
v___x_2251_ = v_reuseFailAlloc_2252_;
goto v_reusejp_2250_;
}
v_reusejp_2250_:
{
return v___x_2251_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___boxed(lean_object* v_stx_2254_, lean_object* v_expectedType_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_){
_start:
{
lean_object* v_res_2263_; 
v_res_2263_ = lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0(v_stx_2254_, v_expectedType_2255_, v___y_2256_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_, v___y_2261_);
lean_dec(v___y_2261_);
lean_dec_ref(v___y_2260_);
lean_dec(v___y_2259_);
lean_dec_ref(v___y_2258_);
lean_dec(v___y_2257_);
lean_dec_ref(v___y_2256_);
return v_res_2263_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1(lean_object* v_stx_2264_, lean_object* v_expectedType_x3f_2265_, lean_object* v_a_2266_, lean_object* v_a_2267_, lean_object* v_a_2268_, lean_object* v_a_2269_, lean_object* v_a_2270_, lean_object* v_a_2271_){
_start:
{
lean_object* v___f_2273_; lean_object* v___x_2274_; 
v___f_2273_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___lam__0___boxed), 9, 1);
lean_closure_set(v___f_2273_, 0, v_stx_2264_);
v___x_2274_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_2265_, v___f_2273_, v_a_2266_, v_a_2267_, v_a_2268_, v_a_2269_, v_a_2270_, v_a_2271_);
return v___x_2274_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1___boxed(lean_object* v_stx_2275_, lean_object* v_expectedType_x3f_2276_, lean_object* v_a_2277_, lean_object* v_a_2278_, lean_object* v_a_2279_, lean_object* v_a_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_, lean_object* v_a_2283_){
_start:
{
lean_object* v_res_2284_; 
v_res_2284_ = lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______elabRules__Qq__termAssumeInstancesCommute_x27____1(v_stx_2275_, v_expectedType_x3f_2276_, v_a_2277_, v_a_2278_, v_a_2279_, v_a_2280_, v_a_2281_, v_a_2282_);
lean_dec(v_a_2282_);
lean_dec_ref(v_a_2281_);
lean_dec(v_a_2280_);
lean_dec_ref(v_a_2279_);
lean_dec(v_a_2278_);
lean_dec_ref(v_a_2277_);
return v_res_2284_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1(lean_object* v_x_2304_, lean_object* v_a_2305_, lean_object* v_a_2306_){
_start:
{
lean_object* v___x_2307_; uint8_t v___x_2308_; 
v___x_2307_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1));
lean_inc(v_x_2304_);
v___x_2308_ = l_Lean_Syntax_isOfKind(v_x_2304_, v___x_2307_);
if (v___x_2308_ == 0)
{
lean_object* v___x_2309_; lean_object* v___x_2310_; 
lean_dec(v_x_2304_);
v___x_2309_ = lean_box(1);
v___x_2310_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2310_, 0, v___x_2309_);
lean_ctor_set(v___x_2310_, 1, v_a_2306_);
return v___x_2310_;
}
else
{
lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; uint8_t v___x_2314_; 
v___x_2311_ = lean_unsigned_to_nat(1u);
v___x_2312_ = l_Lean_Syntax_getArg(v_x_2304_, v___x_2311_);
v___x_2313_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1));
v___x_2314_ = l_Lean_Syntax_isOfKind(v___x_2312_, v___x_2313_);
if (v___x_2314_ == 0)
{
lean_object* v___x_2315_; lean_object* v___x_2316_; 
lean_dec(v_x_2304_);
v___x_2315_ = lean_box(1);
v___x_2316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2316_, 0, v___x_2315_);
lean_ctor_set(v___x_2316_, 1, v_a_2306_);
return v___x_2316_;
}
else
{
lean_object* v_ref_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; uint8_t v___x_2320_; lean_object* v___x_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; 
v_ref_2317_ = lean_ctor_get(v_a_2305_, 5);
v___x_2318_ = lean_unsigned_to_nat(3u);
v___x_2319_ = l_Lean_Syntax_getArg(v_x_2304_, v___x_2318_);
lean_dec(v_x_2304_);
v___x_2320_ = 0;
v___x_2321_ = l_Lean_SourceInfo_fromRef(v_ref_2317_, v___x_2320_);
v___x_2322_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__2));
v___x_2323_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteImpl___00__closed__3));
lean_inc(v___x_2321_);
v___x_2324_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2324_, 0, v___x_2321_);
lean_ctor_set(v___x_2324_, 1, v___x_2323_);
v___x_2325_ = l_Lean_Syntax_node2(v___x_2321_, v___x_2322_, v___x_2324_, v___x_2319_);
v___x_2326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2326_, 0, v___x_2325_);
lean_ctor_set(v___x_2326_, 1, v_a_2306_);
return v___x_2326_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___boxed(lean_object* v_x_2327_, lean_object* v_a_2328_, lean_object* v_a_2329_){
_start:
{
lean_object* v_res_2330_; 
v_res_2330_ = lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1(v_x_2327_, v_a_2328_, v_a_2329_);
lean_dec_ref(v_a_2328_);
return v_res_2330_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__2(lean_object* v_x_2344_, lean_object* v_a_2345_, lean_object* v_a_2346_){
_start:
{
lean_object* v___x_2347_; uint8_t v___x_2348_; 
v___x_2347_ = ((lean_object*)(lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__1___closed__1));
lean_inc(v_x_2344_);
v___x_2348_ = l_Lean_Syntax_isOfKind(v_x_2344_, v___x_2347_);
if (v___x_2348_ == 0)
{
lean_object* v___x_2349_; lean_object* v___x_2350_; 
lean_dec(v_x_2344_);
v___x_2349_ = lean_box(1);
v___x_2350_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2350_, 0, v___x_2349_);
lean_ctor_set(v___x_2350_, 1, v_a_2346_);
return v___x_2350_;
}
else
{
lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; uint8_t v___x_2354_; 
v___x_2351_ = lean_unsigned_to_nat(1u);
v___x_2352_ = l_Lean_Syntax_getArg(v_x_2344_, v___x_2351_);
v___x_2353_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1));
v___x_2354_ = l_Lean_Syntax_isOfKind(v___x_2352_, v___x_2353_);
if (v___x_2354_ == 0)
{
lean_object* v___x_2355_; lean_object* v___x_2356_; 
lean_dec(v_x_2344_);
v___x_2355_ = lean_box(1);
v___x_2356_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2356_, 0, v___x_2355_);
lean_ctor_set(v___x_2356_, 1, v_a_2346_);
return v___x_2356_;
}
else
{
lean_object* v_ref_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; uint8_t v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; 
v_ref_2357_ = lean_ctor_get(v_a_2345_, 5);
v___x_2358_ = lean_unsigned_to_nat(3u);
v___x_2359_ = l_Lean_Syntax_getArg(v_x_2344_, v___x_2358_);
lean_dec(v_x_2344_);
v___x_2360_ = 0;
v___x_2361_ = l_Lean_SourceInfo_fromRef(v_ref_2357_, v___x_2360_);
v___x_2362_ = ((lean_object*)(lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__2));
v___x_2363_ = ((lean_object*)(lp_Qq_Qq_termAssumeInstancesCommute_x27___00__closed__5));
lean_inc(v___x_2361_);
v___x_2364_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2364_, 0, v___x_2361_);
lean_ctor_set(v___x_2364_, 1, v___x_2363_);
v___x_2365_ = l_Lean_Syntax_node2(v___x_2361_, v___x_2362_, v___x_2364_, v___x_2359_);
v___x_2366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2366_, 0, v___x_2365_);
lean_ctor_set(v___x_2366_, 1, v_a_2346_);
return v___x_2366_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__2___boxed(lean_object* v_x_2367_, lean_object* v_a_2368_, lean_object* v_a_2369_){
_start:
{
lean_object* v_res_2370_; 
v_res_2370_ = lp_Qq_Qq_Impl___aux__Qq__AssertInstancesCommute______macroRules__Lean__Parser__Term__assert__2(v_x_2367_, v_a_2368_, v_a_2369_);
lean_dec_ref(v_a_2368_);
return v_res_2370_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1(lean_object* v_x_2390_, lean_object* v_a_2391_, lean_object* v_a_2392_){
_start:
{
lean_object* v___x_2393_; uint8_t v___x_2394_; 
v___x_2393_ = ((lean_object*)(lp_Qq_Qq_doElemAssertInstancesCommute___closed__1));
v___x_2394_ = l_Lean_Syntax_isOfKind(v_x_2390_, v___x_2393_);
if (v___x_2394_ == 0)
{
lean_object* v___x_2395_; lean_object* v___x_2396_; 
v___x_2395_ = lean_box(1);
v___x_2396_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2396_, 0, v___x_2395_);
lean_ctor_set(v___x_2396_, 1, v_a_2392_);
return v___x_2396_;
}
else
{
lean_object* v_ref_2397_; uint8_t v___x_2398_; lean_object* v___x_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; 
v_ref_2397_ = lean_ctor_get(v_a_2391_, 5);
v___x_2398_ = 0;
v___x_2399_ = l_Lean_SourceInfo_fromRef(v_ref_2397_, v___x_2398_);
v___x_2400_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1));
v___x_2401_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__2));
lean_inc_n(v___x_2399_, 3);
v___x_2402_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2402_, 0, v___x_2399_);
lean_ctor_set(v___x_2402_, 1, v___x_2401_);
v___x_2403_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__1));
v___x_2404_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssertInstancesCommuteDummy___closed__2));
v___x_2405_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2405_, 0, v___x_2399_);
lean_ctor_set(v___x_2405_, 1, v___x_2404_);
v___x_2406_ = l_Lean_Syntax_node1(v___x_2399_, v___x_2403_, v___x_2405_);
v___x_2407_ = l_Lean_Syntax_node2(v___x_2399_, v___x_2400_, v___x_2402_, v___x_2406_);
v___x_2408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2408_, 0, v___x_2407_);
lean_ctor_set(v___x_2408_, 1, v_a_2392_);
return v___x_2408_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___boxed(lean_object* v_x_2409_, lean_object* v_a_2410_, lean_object* v_a_2411_){
_start:
{
lean_object* v_res_2412_; 
v_res_2412_ = lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1(v_x_2409_, v_a_2410_, v_a_2411_);
lean_dec_ref(v_a_2410_);
return v_res_2412_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssumeInstancesCommute__1(lean_object* v_x_2425_, lean_object* v_a_2426_, lean_object* v_a_2427_){
_start:
{
lean_object* v___x_2428_; uint8_t v___x_2429_; 
v___x_2428_ = ((lean_object*)(lp_Qq_Qq_doElemAssumeInstancesCommute___closed__1));
v___x_2429_ = l_Lean_Syntax_isOfKind(v_x_2425_, v___x_2428_);
if (v___x_2429_ == 0)
{
lean_object* v___x_2430_; lean_object* v___x_2431_; 
v___x_2430_ = lean_box(1);
v___x_2431_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2430_);
lean_ctor_set(v___x_2431_, 1, v_a_2427_);
return v___x_2431_;
}
else
{
lean_object* v_ref_2432_; uint8_t v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; 
v_ref_2432_ = lean_ctor_get(v_a_2426_, 5);
v___x_2433_ = 0;
v___x_2434_ = l_Lean_SourceInfo_fromRef(v_ref_2432_, v___x_2433_);
v___x_2435_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__1));
v___x_2436_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssertInstancesCommute__1___closed__2));
lean_inc_n(v___x_2434_, 3);
v___x_2437_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2437_, 0, v___x_2434_);
lean_ctor_set(v___x_2437_, 1, v___x_2436_);
v___x_2438_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__1));
v___x_2439_ = ((lean_object*)(lp_Qq_Qq_Impl_termAssumeInstancesCommuteDummy___closed__2));
v___x_2440_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2440_, 0, v___x_2434_);
lean_ctor_set(v___x_2440_, 1, v___x_2439_);
v___x_2441_ = l_Lean_Syntax_node1(v___x_2434_, v___x_2438_, v___x_2440_);
v___x_2442_ = l_Lean_Syntax_node2(v___x_2434_, v___x_2435_, v___x_2437_, v___x_2441_);
v___x_2443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2443_, 0, v___x_2442_);
lean_ctor_set(v___x_2443_, 1, v_a_2427_);
return v___x_2443_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssumeInstancesCommute__1___boxed(lean_object* v_x_2444_, lean_object* v_a_2445_, lean_object* v_a_2446_){
_start:
{
lean_object* v_res_2447_; 
v_res_2447_ = lp_Qq_Qq___aux__Qq__AssertInstancesCommute______macroRules__Qq__doElemAssumeInstancesCommute__1(v_x_2444_, v_a_2445_, v_a_2446_);
lean_dec_ref(v_a_2445_);
return v_res_2447_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_MetaM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_AssertInstancesCommute(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_AssertInstancesCommute(uint8_t builtin) {
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
lean_object* initialize_Qq_Qq_MetaM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_AssertInstancesCommute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_AssertInstancesCommute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_AssertInstancesCommute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_AssertInstancesCommute(builtin);
}
#ifdef __cplusplus
}
#endif
