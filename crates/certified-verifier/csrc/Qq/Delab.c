// Lean compiler output
// Module: Qq.Delab
// Imports: public import Init public meta import Init public import Qq.Macro public import Qq.Typ
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_removeDollar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_SubExpr_HoleIterator_next(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_PrettyPrinter_Delaborator_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_Array_reverse___redArg(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_LocalContext_empty;
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_unquoteLCtx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_determineLocalInstances(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_unquoteExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_Qq_Qq_Impl_unquoteLevelLCtx(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_unquoteLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delabLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_register___at___00__private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_register___at___00__private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__0_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__0_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__0_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_string_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__1_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "qq"};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__1_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__1_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__0_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__1_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(12, 180, 126, 50, 197, 149, 46, 177)}};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_string_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__3_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "(pretty printer) print quotations as q(...) and Q(...)"};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__3_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__3_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__4_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__3_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__4_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__4_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_string_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Qq"};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_string_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__6_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Impl"};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__6_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__6_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__6_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(204, 194, 239, 50, 239, 219, 54, 130)}};
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__0_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(74, 23, 198, 232, 186, 117, 254, 102)}};
static const lean_ctor_object lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__1_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(155, 51, 205, 32, 197, 66, 97, 201)}};
static const lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_ = (const lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_pp_qq;
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_unquote(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_unquote___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Qq_Qq_Impl_checkQqDelabOptions___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions___lam__0___boxed(lean_object*);
static const lean_closure_object lp_Qq_Qq_Impl_checkQqDelabOptions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_checkQqDelabOptions___closed__0_value;
static const lean_closure_object lp_Qq_Qq_Impl_checkQqDelabOptions___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Qq_Impl_checkQqDelabOptions___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_checkQqDelabOptions___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___closed__0_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM = (const lean_object*)&lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_delabQuoted___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_Qq_Qq_Impl_delabQuoted___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuoted___closed__0_value;
static lean_once_cell_t lp_Qq_Qq_Impl_delabQuoted___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_delabQuoted___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__0 = (const lean_object*)&lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__1 = (const lean_object*)&lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__0 = (const lean_object*)&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__0_value;
static const lean_string_object lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__1 = (const lean_object*)&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__1_value;
static const lean_ctor_object lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__2_value_aux_0),((lean_object*)&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__2 = (const lean_object*)&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__2_value;
static lean_once_cell_t lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__3;
static lean_once_cell_t lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__4;
static lean_once_cell_t lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__5;
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_nested"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__0 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__0_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__0_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1_value_aux_0),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__1_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(12, 180, 126, 50, 197, 149, 46, 177)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1_value_aux_1),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 234, 31, 158, 24, 230, 149, 148)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "let"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__5 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__5_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value_aux_0),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value_aux_1),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value_aux_2),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__5_value),LEAN_SCALAR_PTR_LITERAL(155, 166, 195, 152, 24, 103, 8, 2)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__7 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__7_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value_aux_0),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value_aux_1),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value_aux_2),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__7_value),LEAN_SCALAR_PTR_LITERAL(5, 186, 227, 151, 19, 40, 136, 241)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__9 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__9_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__9_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__10 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__10_value;
static lean_once_cell_t lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__12 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__12_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value_aux_0),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value_aux_1),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value_aux_2),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__12_value),LEAN_SCALAR_PTR_LITERAL(61, 47, 121, 206, 37, 68, 134, 111)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIdDecl"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__14 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__14_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value_aux_0),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value_aux_1),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value_aux_2),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__14_value),LEAN_SCALAR_PTR_LITERAL(82, 96, 243, 36, 251, 209, 136, 237)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "letId"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__16 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__16_value;
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value_aux_0),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value_aux_1),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value_aux_2),((lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__16_value),LEAN_SCALAR_PTR_LITERAL(67, 92, 92, 51, 38, 250, 60, 190)}};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__18 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__18_value;
static const lean_string_object lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__19 = (const lean_object*)&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__19_value;
static lean_once_cell_t lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__20;
static lean_once_cell_t lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__21;
static lean_once_cell_t lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__22;
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq_Impl_withDelabQuoted___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_withDelabQuoted___closed__0;
static lean_once_cell_t lp_Qq_Qq_Impl_withDelabQuoted___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_withDelabQuoted___closed__1;
static const lean_array_object lp_Qq_Qq_Impl_withDelabQuoted___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq_Qq_Impl_withDelabQuoted___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_withDelabQuoted___closed__2_value;
static lean_once_cell_t lp_Qq_Qq_Impl_withDelabQuoted___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_withDelabQuoted___closed__3;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_delabQ___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "termQ(__)_1"};
static const lean_object* lp_Qq_Qq_Impl_delabQ___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_delabQ___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_delabQ___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_delabQ___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_delabQ___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_delabQ___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 116, 92, 169, 40, 159, 67, 140)}};
static const lean_object* lp_Qq_Qq_Impl_delabQ___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_delabQ___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl_delabQ___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Q("};
static const lean_object* lp_Qq_Qq_Impl_delabQ___lam__0___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_delabQ___lam__0___closed__2_value;
static const lean_string_object lp_Qq_Qq_Impl_delabQ___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_Qq_Qq_Impl_delabQ___lam__0___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_delabQ___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq_Impl_delabQ___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_delabQ___closed__0;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_delabq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "termQ(__)"};
static const lean_object* lp_Qq_Qq_Impl_delabq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_delabq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_delabq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_delabq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_delabq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_delabq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(101, 189, 163, 187, 112, 4, 232, 151)}};
static const lean_object* lp_Qq_Qq_Impl_delabq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_delabq___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl_delabq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "q("};
static const lean_object* lp_Qq_Qq_Impl_delabq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_delabq___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq_Impl_delabq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_delabq___closed__0;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_=Q_"};
static const lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(184, 137, 18, 100, 15, 93, 119, 77)}};
static const lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__1_value;
static const lean_string_object lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=Q"};
static const lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq_Impl_delabQuotedDefEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___closed__0;
static lean_once_cell_t lp_Qq_Qq_Impl_delabQuotedDefEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__0;
static const lean_string_object lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term_=QL_"};
static const lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__1_value;
static const lean_ctor_object lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__5_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__2_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__1_value),LEAN_SCALAR_PTR_LITERAL(50, 120, 42, 219, 90, 165, 37, 102)}};
static const lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__2_value;
static const lean_string_object lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "=QL"};
static const lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__3_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_register___at___00__private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
v___x_8_ = lean_alloc_ctor(1, 0, 1);
v___x_9_ = lean_unbox(v_defValue_5_);
lean_ctor_set_uint8(v___x_8_, 0, v___x_9_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_10_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_10_, 0, v_name_1_);
lean_ctor_set(v___x_10_, 1, v_ref_3_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v_descr_6_);
lean_ctor_set(v___x_10_, 4, v_deprecation_x3f_7_);
v___x_11_ = lean_register_option(v_name_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_19_; 
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_20_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_17_; 
lean_inc(v_defValue_5_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_name_1_);
lean_ctor_set(v___x_15_, 1, v_defValue_5_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_15_);
v___x_17_ = v___x_13_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
else
{
lean_object* v_a_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_28_; 
lean_dec(v_name_1_);
v_a_21_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_28_ == 0)
{
v___x_23_ = v___x_11_;
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_a_21_);
lean_dec(v___x_11_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_26_; 
if (v_isShared_24_ == 0)
{
v___x_26_ = v___x_23_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_a_21_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_register___at___00__private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_Qq_Lean_Option_register___at___00__private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = ((lean_object*)(lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_));
v___x_54_ = ((lean_object*)(lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__4_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_));
v___x_55_ = ((lean_object*)(lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__7_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_));
v___x_56_ = lp_Qq_Lean_Option_register___at___00__private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4__spec__0(v___x_53_, v___x_54_, v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4____boxed(lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_();
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg(lean_object* v_x_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_){
_start:
{
lean_object* v___x_65_; 
lean_inc(v_a_63_);
lean_inc_ref(v_a_62_);
lean_inc(v_a_61_);
lean_inc_ref(v_a_60_);
v___x_65_ = lean_apply_5(v_x_59_, v_a_60_, v_a_61_, v_a_62_, v_a_63_, lean_box(0));
if (lean_obj_tag(v___x_65_) == 0)
{
return v___x_65_;
}
else
{
lean_object* v_a_66_; uint8_t v___y_68_; uint8_t v___x_70_; 
v_a_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc(v_a_66_);
v___x_70_ = l_Lean_Exception_isInterrupt(v_a_66_);
if (v___x_70_ == 0)
{
uint8_t v___x_71_; 
v___x_71_ = l_Lean_Exception_isRuntime(v_a_66_);
v___y_68_ = v___x_71_;
goto v___jp_67_;
}
else
{
lean_dec(v_a_66_);
v___y_68_ = v___x_70_;
goto v___jp_67_;
}
v___jp_67_:
{
if (v___y_68_ == 0)
{
lean_object* v___x_69_; 
lean_dec_ref_known(v___x_65_, 1);
v___x_69_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_69_;
}
else
{
return v___x_65_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg___boxed(lean_object* v_x_72_, lean_object* v_a_73_, lean_object* v_a_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg(v_x_72_, v_a_73_, v_a_74_, v_a_75_, v_a_76_);
lean_dec(v_a_76_);
lean_dec_ref(v_a_75_);
lean_dec(v_a_74_);
lean_dec_ref(v_a_73_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError(lean_object* v_00_u03b1_79_, lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg(v_x_80_, v_a_83_, v_a_84_, v_a_85_, v_a_86_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___boxed(lean_object* v_00_u03b1_89_, lean_object* v_x_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_, lean_object* v_a_95_, lean_object* v_a_96_, lean_object* v_a_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError(v_00_u03b1_89_, v_x_90_, v_a_91_, v_a_92_, v_a_93_, v_a_94_, v_a_95_, v_a_96_);
lean_dec(v_a_96_);
lean_dec_ref(v_a_95_);
lean_dec(v_a_94_);
lean_dec_ref(v_a_93_);
lean_dec(v_a_92_);
lean_dec_ref(v_a_91_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_unquote(lean_object* v_e_99_, lean_object* v_a_100_, lean_object* v_a_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_Qq_Qq_Impl_unquoteLCtx(v_a_100_, v_a_101_, v_a_102_, v_a_103_, v_a_104_);
if (lean_obj_tag(v___x_106_) == 0)
{
lean_object* v_a_107_; lean_object* v_snd_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_142_; 
v_a_107_ = lean_ctor_get(v___x_106_, 0);
lean_inc(v_a_107_);
lean_dec_ref_known(v___x_106_, 1);
v_snd_108_ = lean_ctor_get(v_a_107_, 1);
v_isSharedCheck_142_ = !lean_is_exclusive(v_a_107_);
if (v_isSharedCheck_142_ == 0)
{
lean_object* v_unused_143_; 
v_unused_143_ = lean_ctor_get(v_a_107_, 0);
lean_dec(v_unused_143_);
v___x_110_ = v_a_107_;
v_isShared_111_ = v_isSharedCheck_142_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_snd_108_);
lean_dec(v_a_107_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_142_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v___x_112_; 
v___x_112_ = lp_Qq_Qq_Impl_unquoteExpr(v_e_99_, v_snd_108_, v_a_101_, v_a_102_, v_a_103_, v_a_104_);
if (lean_obj_tag(v___x_112_) == 0)
{
lean_object* v_a_113_; lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_133_; 
v_a_113_ = lean_ctor_get(v___x_112_, 0);
v_isSharedCheck_133_ = !lean_is_exclusive(v___x_112_);
if (v_isSharedCheck_133_ == 0)
{
v___x_115_ = v___x_112_;
v_isShared_116_ = v_isSharedCheck_133_;
goto v_resetjp_114_;
}
else
{
lean_inc(v_a_113_);
lean_dec(v___x_112_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_133_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v_snd_117_; lean_object* v_fst_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_132_; 
v_snd_117_ = lean_ctor_get(v_a_113_, 1);
v_fst_118_ = lean_ctor_get(v_a_113_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v_a_113_);
if (v_isSharedCheck_132_ == 0)
{
v___x_120_ = v_a_113_;
v_isShared_121_ = v_isSharedCheck_132_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_snd_117_);
lean_inc(v_fst_118_);
lean_dec(v_a_113_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_132_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v_unquoted_122_; lean_object* v___x_124_; 
v_unquoted_122_ = lean_ctor_get(v_snd_117_, 3);
lean_inc_ref(v_unquoted_122_);
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 1, v_unquoted_122_);
v___x_124_ = v___x_120_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_fst_118_);
lean_ctor_set(v_reuseFailAlloc_131_, 1, v_unquoted_122_);
v___x_124_ = v_reuseFailAlloc_131_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
lean_object* v___x_126_; 
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 1, v_snd_117_);
lean_ctor_set(v___x_110_, 0, v___x_124_);
v___x_126_ = v___x_110_;
goto v_reusejp_125_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v___x_124_);
lean_ctor_set(v_reuseFailAlloc_130_, 1, v_snd_117_);
v___x_126_ = v_reuseFailAlloc_130_;
goto v_reusejp_125_;
}
v_reusejp_125_:
{
lean_object* v___x_128_; 
if (v_isShared_116_ == 0)
{
lean_ctor_set(v___x_115_, 0, v___x_126_);
v___x_128_ = v___x_115_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v___x_126_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
return v___x_128_;
}
}
}
}
}
}
else
{
lean_object* v_a_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_141_; 
lean_del_object(v___x_110_);
v_a_134_ = lean_ctor_get(v___x_112_, 0);
v_isSharedCheck_141_ = !lean_is_exclusive(v___x_112_);
if (v_isSharedCheck_141_ == 0)
{
v___x_136_ = v___x_112_;
v_isShared_137_ = v_isSharedCheck_141_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_a_134_);
lean_dec(v___x_112_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_141_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___x_139_; 
if (v_isShared_137_ == 0)
{
v___x_139_ = v___x_136_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v_a_134_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
}
}
}
else
{
lean_object* v_a_144_; lean_object* v___x_146_; uint8_t v_isShared_147_; uint8_t v_isSharedCheck_151_; 
lean_dec_ref(v_e_99_);
v_a_144_ = lean_ctor_get(v___x_106_, 0);
v_isSharedCheck_151_ = !lean_is_exclusive(v___x_106_);
if (v_isSharedCheck_151_ == 0)
{
v___x_146_ = v___x_106_;
v_isShared_147_ = v_isSharedCheck_151_;
goto v_resetjp_145_;
}
else
{
lean_inc(v_a_144_);
lean_dec(v___x_106_);
v___x_146_ = lean_box(0);
v_isShared_147_ = v_isSharedCheck_151_;
goto v_resetjp_145_;
}
v_resetjp_145_:
{
lean_object* v___x_149_; 
if (v_isShared_147_ == 0)
{
v___x_149_ = v___x_146_;
goto v_reusejp_148_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v_a_144_);
v___x_149_ = v_reuseFailAlloc_150_;
goto v_reusejp_148_;
}
v_reusejp_148_:
{
return v___x_149_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Delab_0__Qq_Impl_unquote___boxed(lean_object* v_e_152_, lean_object* v_a_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_Qq___private_Qq_Delab_0__Qq_Impl_unquote(v_e_152_, v_a_153_, v_a_154_, v_a_155_, v_a_156_, v_a_157_);
lean_dec(v_a_157_);
lean_dec_ref(v_a_156_);
lean_dec(v_a_155_);
lean_dec_ref(v_a_154_);
return v_res_159_;
}
}
LEAN_EXPORT uint8_t lp_Qq_Qq_Impl_checkQqDelabOptions___lam__0(lean_object* v_x_160_){
_start:
{
lean_object* v_map_161_; lean_object* v___x_162_; uint8_t v___x_163_; lean_object* v___x_164_; 
v_map_161_ = lean_ctor_get(v_x_160_, 0);
v___x_162_ = ((lean_object*)(lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn___closed__2_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_));
v___x_163_ = 1;
v___x_164_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_161_, v___x_162_);
if (lean_obj_tag(v___x_164_) == 0)
{
return v___x_163_;
}
else
{
lean_object* v_val_165_; 
v_val_165_ = lean_ctor_get(v___x_164_, 0);
lean_inc(v_val_165_);
lean_dec_ref_known(v___x_164_, 1);
if (lean_obj_tag(v_val_165_) == 1)
{
uint8_t v_v_166_; 
v_v_166_ = lean_ctor_get_uint8(v_val_165_, 0);
lean_dec_ref_known(v_val_165_, 0);
return v_v_166_;
}
else
{
lean_dec(v_val_165_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions___lam__0___boxed(lean_object* v_x_167_){
_start:
{
uint8_t v_res_168_; lean_object* v_r_169_; 
v_res_168_ = lp_Qq_Qq_Impl_checkQqDelabOptions___lam__0(v_x_167_);
lean_dec_ref(v_x_167_);
v_r_169_ = lean_box(v_res_168_);
return v_r_169_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions(lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v___y_180_; lean_object* v___y_181_; lean_object* v___y_182_; lean_object* v___y_183_; lean_object* v___y_184_; lean_object* v___y_185_; lean_object* v___f_207_; lean_object* v___x_208_; 
v___f_207_ = ((lean_object*)(lp_Qq_Qq_Impl_checkQqDelabOptions___closed__1));
v___x_208_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___f_207_, v_a_172_, v_a_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
if (lean_obj_tag(v___x_208_) == 0)
{
lean_object* v_a_209_; uint8_t v___x_210_; 
v_a_209_ = lean_ctor_get(v___x_208_, 0);
lean_inc(v_a_209_);
lean_dec_ref_known(v___x_208_, 1);
v___x_210_ = lean_unbox(v_a_209_);
lean_dec(v_a_209_);
if (v___x_210_ == 0)
{
lean_object* v___x_211_; 
v___x_211_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_211_) == 0)
{
lean_dec_ref_known(v___x_211_, 1);
v___y_180_ = v_a_172_;
v___y_181_ = v_a_173_;
v___y_182_ = v_a_174_;
v___y_183_ = v_a_175_;
v___y_184_ = v_a_176_;
v___y_185_ = v_a_177_;
goto v___jp_179_;
}
else
{
return v___x_211_;
}
}
else
{
v___y_180_ = v_a_172_;
v___y_181_ = v_a_173_;
v___y_182_ = v_a_174_;
v___y_183_ = v_a_175_;
v___y_184_ = v_a_176_;
v___y_185_ = v_a_177_;
goto v___jp_179_;
}
}
else
{
lean_object* v_a_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
v_a_212_ = lean_ctor_get(v___x_208_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_208_);
if (v_isSharedCheck_219_ == 0)
{
v___x_214_ = v___x_208_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_a_212_);
lean_dec(v___x_208_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_217_; 
if (v_isShared_215_ == 0)
{
v___x_217_ = v___x_214_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_a_212_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
v___jp_179_:
{
lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_186_ = ((lean_object*)(lp_Qq_Qq_Impl_checkQqDelabOptions___closed__0));
v___x_187_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_186_, v___y_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_);
if (lean_obj_tag(v___x_187_) == 0)
{
lean_object* v_a_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_198_; 
v_a_188_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_198_ == 0)
{
v___x_190_ = v___x_187_;
v_isShared_191_ = v_isSharedCheck_198_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_a_188_);
lean_dec(v___x_187_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_198_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
uint8_t v___x_192_; 
v___x_192_ = lean_unbox(v_a_188_);
lean_dec(v_a_188_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; lean_object* v___x_195_; 
v___x_193_ = lean_box(0);
if (v_isShared_191_ == 0)
{
lean_ctor_set(v___x_190_, 0, v___x_193_);
v___x_195_ = v___x_190_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_193_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
else
{
lean_object* v___x_197_; 
lean_del_object(v___x_190_);
v___x_197_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_197_;
}
}
}
else
{
lean_object* v_a_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_206_; 
v_a_199_ = lean_ctor_get(v___x_187_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_206_ == 0)
{
v___x_201_ = v___x_187_;
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_a_199_);
lean_dec(v___x_187_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_204_; 
if (v_isShared_202_ == 0)
{
v___x_204_ = v___x_201_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v_a_199_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_checkQqDelabOptions___boxed(lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_, lean_object* v_a_224_, lean_object* v_a_225_, lean_object* v_a_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_Qq_Qq_Impl_checkQqDelabOptions(v_a_220_, v_a_221_, v_a_222_, v_a_223_, v_a_224_, v_a_225_);
lean_dec(v_a_225_);
lean_dec_ref(v_a_224_);
lean_dec(v_a_223_);
lean_dec_ref(v_a_222_);
lean_dec(v_a_221_);
lean_dec_ref(v_a_220_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___lam__0(lean_object* v_00_u03b1_228_, lean_object* v_k_229_, lean_object* v_s_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v___x_238_; 
lean_inc(v___y_236_);
lean_inc_ref(v___y_235_);
lean_inc(v___y_234_);
lean_inc_ref(v___y_233_);
v___x_238_ = lean_apply_6(v_k_229_, v_s_230_, v___y_233_, v___y_234_, v___y_235_, v___y_236_, lean_box(0));
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___lam__0___boxed(lean_object* v_00_u03b1_239_, lean_object* v_k_240_, lean_object* v_s_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_Qq_Qq_Impl_instMonadLiftUnquoteMStateTUnquoteStateDelabM___lam__0(v_00_u03b1_239_, v_k_240_, v_s_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_);
lean_dec(v___y_247_);
lean_dec_ref(v___y_246_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___lam__0(lean_object* v_x_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
lean_object* v___x_261_; 
lean_inc(v___y_255_);
lean_inc_ref(v___y_254_);
v___x_261_ = lean_apply_8(v_x_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_, lean_box(0));
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___lam__0___boxed(lean_object* v_x_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___lam__0(v_x_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg(lean_object* v_lctx_272_, lean_object* v_localInsts_273_, lean_object* v_x_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v___f_283_; lean_object* v___x_284_; 
lean_inc(v___y_277_);
lean_inc_ref(v___y_276_);
v___f_283_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_283_, 0, v_x_274_);
lean_closure_set(v___f_283_, 1, v___y_275_);
lean_closure_set(v___f_283_, 2, v___y_276_);
lean_closure_set(v___f_283_, 3, v___y_277_);
v___x_284_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_272_, v_localInsts_273_, v___f_283_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_292_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_284_);
if (v_isSharedCheck_292_ == 0)
{
v___x_287_ = v___x_284_;
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_a_285_);
lean_dec(v___x_284_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_290_; 
if (v_isShared_288_ == 0)
{
v___x_290_ = v___x_287_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v_a_285_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
}
else
{
lean_object* v_a_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_300_; 
v_a_293_ = lean_ctor_get(v___x_284_, 0);
v_isSharedCheck_300_ = !lean_is_exclusive(v___x_284_);
if (v_isSharedCheck_300_ == 0)
{
v___x_295_ = v___x_284_;
v_isShared_296_ = v_isSharedCheck_300_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_a_293_);
lean_dec(v___x_284_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_300_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v___x_298_; 
if (v_isShared_296_ == 0)
{
v___x_298_ = v___x_295_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_a_293_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg___boxed(lean_object* v_lctx_301_, lean_object* v_localInsts_302_, lean_object* v_x_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg(v_lctx_301_, v_localInsts_302_, v_x_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0(lean_object* v_00_u03b1_313_, lean_object* v_lctx_314_, lean_object* v_localInsts_315_, lean_object* v_x_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg(v_lctx_314_, v_localInsts_315_, v_x_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___boxed(lean_object* v_00_u03b1_326_, lean_object* v_lctx_327_, lean_object* v_localInsts_328_, lean_object* v_x_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0(v_00_u03b1_326_, v_lctx_327_, v_localInsts_328_, v_x_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
lean_dec(v___y_332_);
lean_dec_ref(v___y_331_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_subExpr_342_; lean_object* v_expr_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v_subExpr_342_ = lean_ctor_get(v___y_340_, 3);
v_expr_343_ = lean_ctor_get(v_subExpr_342_, 0);
lean_inc_ref(v_expr_343_);
v___x_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_344_, 0, v_expr_343_);
lean_ctor_set(v___x_344_, 1, v___y_339_);
v___x_345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_345_, 0, v___x_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg___boxed(lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(v___y_346_, v___y_347_);
lean_dec_ref(v___y_347_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1(lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(v___y_350_, v___y_351_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___boxed(lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1(v___y_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_, v___y_364_, v___y_365_);
lean_dec(v___y_365_);
lean_dec_ref(v___y_364_);
lean_dec(v___y_363_);
lean_dec_ref(v___y_362_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted___lam__0(lean_object* v_val_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_subExpr_377_; lean_object* v_optionsPerPos_378_; lean_object* v_currNamespace_379_; lean_object* v_openDecls_380_; uint8_t v_inPattern_381_; lean_object* v_depth_382_; lean_object* v_lctxInitIndices_383_; lean_object* v_pos_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v_subExpr_377_ = lean_ctor_get(v___y_370_, 3);
v_optionsPerPos_378_ = lean_ctor_get(v___y_370_, 0);
v_currNamespace_379_ = lean_ctor_get(v___y_370_, 1);
v_openDecls_380_ = lean_ctor_get(v___y_370_, 2);
v_inPattern_381_ = lean_ctor_get_uint8(v___y_370_, sizeof(void*)*6);
v_depth_382_ = lean_ctor_get(v___y_370_, 4);
v_lctxInitIndices_383_ = lean_ctor_get(v___y_370_, 5);
v_pos_384_ = lean_ctor_get(v_subExpr_377_, 1);
lean_inc(v_pos_384_);
v___x_385_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_385_, 0, v_val_368_);
lean_ctor_set(v___x_385_, 1, v_pos_384_);
lean_inc(v_lctxInitIndices_383_);
lean_inc(v_depth_382_);
lean_inc(v_openDecls_380_);
lean_inc(v_currNamespace_379_);
lean_inc(v_optionsPerPos_378_);
v___x_386_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_386_, 0, v_optionsPerPos_378_);
lean_ctor_set(v___x_386_, 1, v_currNamespace_379_);
lean_ctor_set(v___x_386_, 2, v_openDecls_380_);
lean_ctor_set(v___x_386_, 3, v___x_385_);
lean_ctor_set(v___x_386_, 4, v_depth_382_);
lean_ctor_set(v___x_386_, 5, v_lctxInitIndices_383_);
lean_ctor_set_uint8(v___x_386_, sizeof(void*)*6, v_inPattern_381_);
v___x_387_ = l_Lean_PrettyPrinter_Delaborator_delab(v___x_386_, v___y_371_, v___y_372_, v___y_373_, v___y_374_, v___y_375_);
lean_dec_ref_known(v___x_386_, 6);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_396_; 
v_a_388_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_396_ == 0)
{
v___x_390_ = v___x_387_;
v_isShared_391_ = v_isSharedCheck_396_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_387_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_396_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_392_; lean_object* v___x_394_; 
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v_a_388_);
lean_ctor_set(v___x_392_, 1, v___y_369_);
if (v_isShared_391_ == 0)
{
lean_ctor_set(v___x_390_, 0, v___x_392_);
v___x_394_ = v___x_390_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v___x_392_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
lean_dec_ref(v___y_369_);
v_a_397_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_404_ == 0)
{
v___x_399_ = v___x_387_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_387_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
if (v_isShared_400_ == 0)
{
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_a_397_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted___lam__0___boxed(lean_object* v_val_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_Qq_Qq_Impl_delabQuoted___lam__0(v_val_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
lean_dec(v___y_408_);
lean_dec_ref(v___y_407_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2_spec__2(lean_object* v_msgData_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_){
_start:
{
lean_object* v___x_421_; lean_object* v_env_422_; lean_object* v___x_423_; lean_object* v_mctx_424_; lean_object* v_lctx_425_; lean_object* v_options_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v___x_421_ = lean_st_ref_get(v___y_419_);
v_env_422_ = lean_ctor_get(v___x_421_, 0);
lean_inc_ref(v_env_422_);
lean_dec(v___x_421_);
v___x_423_ = lean_st_ref_get(v___y_417_);
v_mctx_424_ = lean_ctor_get(v___x_423_, 0);
lean_inc_ref(v_mctx_424_);
lean_dec(v___x_423_);
v_lctx_425_ = lean_ctor_get(v___y_416_, 2);
v_options_426_ = lean_ctor_get(v___y_418_, 2);
lean_inc_ref(v_options_426_);
lean_inc_ref(v_lctx_425_);
v___x_427_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_427_, 0, v_env_422_);
lean_ctor_set(v___x_427_, 1, v_mctx_424_);
lean_ctor_set(v___x_427_, 2, v_lctx_425_);
lean_ctor_set(v___x_427_, 3, v_options_426_);
v___x_428_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
lean_ctor_set(v___x_428_, 1, v_msgData_415_);
v___x_429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2_spec__2___boxed(lean_object* v_msgData_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_){
_start:
{
lean_object* v_res_436_; 
v_res_436_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2_spec__2(v_msgData_430_, v___y_431_, v___y_432_, v___y_433_, v___y_434_);
lean_dec(v___y_434_);
lean_dec_ref(v___y_433_);
lean_dec(v___y_432_);
lean_dec_ref(v___y_431_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg(lean_object* v_msg_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_ref_443_; lean_object* v___x_444_; lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_453_; 
v_ref_443_ = lean_ctor_get(v___y_440_, 5);
v___x_444_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2_spec__2(v_msg_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_);
v_a_445_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_453_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_453_ == 0)
{
v___x_447_ = v___x_444_;
v_isShared_448_ = v_isSharedCheck_453_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_444_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_453_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_449_; lean_object* v___x_451_; 
lean_inc(v_ref_443_);
v___x_449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_449_, 0, v_ref_443_);
lean_ctor_set(v___x_449_, 1, v_a_445_);
if (v_isShared_448_ == 0)
{
lean_ctor_set_tag(v___x_447_, 1);
lean_ctor_set(v___x_447_, 0, v___x_449_);
v___x_451_ = v___x_447_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_452_; 
v_reuseFailAlloc_452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_452_, 0, v___x_449_);
v___x_451_ = v_reuseFailAlloc_452_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
return v___x_451_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg___boxed(lean_object* v_msg_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg(v_msg_454_, v___y_455_, v___y_456_, v___y_457_, v___y_458_);
lean_dec(v___y_458_);
lean_dec_ref(v___y_457_);
lean_dec(v___y_456_);
lean_dec_ref(v___y_455_);
return v_res_460_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_delabQuoted___closed__1(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_462_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQuoted___closed__0));
v___x_463_ = l_Lean_stringToMessageData(v___x_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted(lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_, lean_object* v_a_469_, lean_object* v_a_470_){
_start:
{
lean_object* v_val_473_; lean_object* v_snd_474_; lean_object* v___x_488_; lean_object* v_a_489_; lean_object* v_fst_490_; lean_object* v_snd_491_; lean_object* v___x_492_; 
v___x_488_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(v_a_464_, v_a_465_);
v_a_489_ = lean_ctor_get(v___x_488_, 0);
lean_inc(v_a_489_);
lean_dec_ref(v___x_488_);
v_fst_490_ = lean_ctor_get(v_a_489_, 0);
lean_inc(v_fst_490_);
v_snd_491_ = lean_ctor_get(v_a_489_, 1);
lean_inc(v_snd_491_);
lean_dec(v_a_489_);
v___x_492_ = lp_Qq_Qq_Impl_unquoteExpr(v_fst_490_, v_snd_491_, v_a_467_, v_a_468_, v_a_469_, v_a_470_);
if (lean_obj_tag(v___x_492_) == 0)
{
lean_object* v_a_493_; lean_object* v_fst_494_; lean_object* v_snd_495_; 
v_a_493_ = lean_ctor_get(v___x_492_, 0);
lean_inc(v_a_493_);
lean_dec_ref_known(v___x_492_, 1);
v_fst_494_ = lean_ctor_get(v_a_493_, 0);
lean_inc(v_fst_494_);
v_snd_495_ = lean_ctor_get(v_a_493_, 1);
lean_inc(v_snd_495_);
lean_dec(v_a_493_);
v_val_473_ = v_fst_494_;
v_snd_474_ = v_snd_495_;
goto v___jp_472_;
}
else
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_517_; 
v_a_496_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_517_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_517_ == 0)
{
v___x_498_ = v___x_492_;
v_isShared_499_ = v_isSharedCheck_517_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_492_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_517_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
uint8_t v___y_501_; uint8_t v___x_515_; 
v___x_515_ = l_Lean_Exception_isInterrupt(v_a_496_);
if (v___x_515_ == 0)
{
uint8_t v___x_516_; 
lean_inc(v_a_496_);
v___x_516_ = l_Lean_Exception_isRuntime(v_a_496_);
v___y_501_ = v___x_516_;
goto v___jp_500_;
}
else
{
v___y_501_ = v___x_515_;
goto v___jp_500_;
}
v___jp_500_:
{
if (v___y_501_ == 0)
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_511_; 
lean_del_object(v___x_498_);
lean_dec(v_a_496_);
v___x_502_ = lean_obj_once(&lp_Qq_Qq_Impl_delabQuoted___closed__1, &lp_Qq_Qq_Impl_delabQuoted___closed__1_once, _init_lp_Qq_Qq_Impl_delabQuoted___closed__1);
v___x_503_ = lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg(v___x_502_, v_a_467_, v_a_468_, v_a_469_, v_a_470_);
v_a_504_ = lean_ctor_get(v___x_503_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_503_);
if (v_isSharedCheck_511_ == 0)
{
v___x_506_ = v___x_503_;
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_503_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_509_; 
if (v_isShared_507_ == 0)
{
v___x_509_ = v___x_506_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v_a_504_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
else
{
lean_object* v___x_513_; 
if (v_isShared_499_ == 0)
{
v___x_513_ = v___x_498_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v_a_496_);
v___x_513_ = v_reuseFailAlloc_514_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
return v___x_513_;
}
}
}
}
}
v___jp_472_:
{
lean_object* v_unquoted_475_; lean_object* v___x_476_; 
v_unquoted_475_ = lean_ctor_get(v_snd_474_, 3);
lean_inc_ref(v_unquoted_475_);
v___x_476_ = lp_Qq_Qq_Impl_determineLocalInstances(v_unquoted_475_, v_a_467_, v_a_468_, v_a_469_, v_a_470_);
if (lean_obj_tag(v___x_476_) == 0)
{
lean_object* v_a_477_; lean_object* v___f_478_; lean_object* v___x_479_; 
v_a_477_ = lean_ctor_get(v___x_476_, 0);
lean_inc(v_a_477_);
lean_dec_ref_known(v___x_476_, 1);
v___f_478_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuoted___lam__0___boxed), 9, 1);
lean_closure_set(v___f_478_, 0, v_val_473_);
v___x_479_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq_Impl_delabQuoted_spec__0___redArg(v_unquoted_475_, v_a_477_, v___f_478_, v_snd_474_, v_a_465_, v_a_466_, v_a_467_, v_a_468_, v_a_469_, v_a_470_);
return v___x_479_;
}
else
{
lean_object* v_a_480_; lean_object* v___x_482_; uint8_t v_isShared_483_; uint8_t v_isSharedCheck_487_; 
lean_dec_ref(v_unquoted_475_);
lean_dec_ref(v_snd_474_);
lean_dec_ref(v_val_473_);
v_a_480_ = lean_ctor_get(v___x_476_, 0);
v_isSharedCheck_487_ = !lean_is_exclusive(v___x_476_);
if (v_isSharedCheck_487_ == 0)
{
v___x_482_ = v___x_476_;
v_isShared_483_ = v_isSharedCheck_487_;
goto v_resetjp_481_;
}
else
{
lean_inc(v_a_480_);
lean_dec(v___x_476_);
v___x_482_ = lean_box(0);
v_isShared_483_ = v_isSharedCheck_487_;
goto v_resetjp_481_;
}
v_resetjp_481_:
{
lean_object* v___x_485_; 
if (v_isShared_483_ == 0)
{
v___x_485_ = v___x_482_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v_a_480_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuoted___boxed(lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_, lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v_a_524_, lean_object* v_a_525_){
_start:
{
lean_object* v_res_526_; 
v_res_526_ = lp_Qq_Qq_Impl_delabQuoted(v_a_518_, v_a_519_, v_a_520_, v_a_521_, v_a_522_, v_a_523_, v_a_524_);
lean_dec(v_a_524_);
lean_dec_ref(v_a_523_);
lean_dec(v_a_522_);
lean_dec_ref(v_a_521_);
lean_dec(v_a_520_);
lean_dec_ref(v_a_519_);
return v_res_526_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2(lean_object* v_00_u03b1_527_, lean_object* v_msg_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___redArg(v_msg_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2___boxed(lean_object* v_00_u03b1_535_, lean_object* v_msg_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_Qq_Lean_throwError___at___00Qq_Impl_delabQuoted_spec__2(v_00_u03b1_535_, v_msg_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_);
lean_dec(v___y_540_);
lean_dec_ref(v___y_539_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg(lean_object* v___y_543_, lean_object* v___y_544_){
_start:
{
lean_object* v___x_546_; lean_object* v_holeIter_547_; lean_object* v_curr_548_; lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_569_; 
v___x_546_ = lean_st_ref_get(v___y_544_);
v_holeIter_547_ = lean_ctor_get(v___x_546_, 2);
lean_inc_ref(v_holeIter_547_);
lean_dec(v___x_546_);
v_curr_548_ = lean_ctor_get(v_holeIter_547_, 0);
v_isSharedCheck_569_ = !lean_is_exclusive(v_holeIter_547_);
if (v_isSharedCheck_569_ == 0)
{
lean_object* v_unused_570_; 
v_unused_570_ = lean_ctor_get(v_holeIter_547_, 1);
lean_dec(v_unused_570_);
v___x_550_ = v_holeIter_547_;
v_isShared_551_ = v_isSharedCheck_569_;
goto v_resetjp_549_;
}
else
{
lean_inc(v_curr_548_);
lean_dec(v_holeIter_547_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_569_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___x_552_; lean_object* v_steps_553_; lean_object* v_infos_554_; lean_object* v_holeIter_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_568_; 
v___x_552_ = lean_st_ref_take(v___y_544_);
v_steps_553_ = lean_ctor_get(v___x_552_, 0);
v_infos_554_ = lean_ctor_get(v___x_552_, 1);
v_holeIter_555_ = lean_ctor_get(v___x_552_, 2);
v_isSharedCheck_568_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_568_ == 0)
{
v___x_557_ = v___x_552_;
v_isShared_558_ = v_isSharedCheck_568_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_holeIter_555_);
lean_inc(v_infos_554_);
lean_inc(v_steps_553_);
lean_dec(v___x_552_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_568_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_559_; lean_object* v___x_561_; 
v___x_559_ = l_Lean_PrettyPrinter_Delaborator_SubExpr_HoleIterator_next(v_holeIter_555_);
if (v_isShared_558_ == 0)
{
lean_ctor_set(v___x_557_, 2, v___x_559_);
v___x_561_ = v___x_557_;
goto v_reusejp_560_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v_steps_553_);
lean_ctor_set(v_reuseFailAlloc_567_, 1, v_infos_554_);
lean_ctor_set(v_reuseFailAlloc_567_, 2, v___x_559_);
v___x_561_ = v_reuseFailAlloc_567_;
goto v_reusejp_560_;
}
v_reusejp_560_:
{
lean_object* v___x_562_; lean_object* v___x_564_; 
v___x_562_ = lean_st_ref_set(v___y_544_, v___x_561_);
if (v_isShared_551_ == 0)
{
lean_ctor_set(v___x_550_, 1, v___y_543_);
v___x_564_ = v___x_550_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v_curr_548_);
lean_ctor_set(v_reuseFailAlloc_566_, 1, v___y_543_);
v___x_564_ = v_reuseFailAlloc_566_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
lean_object* v___x_565_; 
v___x_565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_565_, 0, v___x_564_);
return v___x_565_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg___boxed(lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg(v___y_571_, v___y_572_);
lean_dec(v___y_572_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1(lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
lean_object* v___x_583_; 
v___x_583_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg(v___y_575_, v___y_577_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___boxed(lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1(v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2(lean_object* v_o_596_, lean_object* v_k_597_, uint8_t v_v_598_){
_start:
{
lean_object* v_map_599_; uint8_t v_hasTrace_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_614_; 
v_map_599_ = lean_ctor_get(v_o_596_, 0);
v_hasTrace_600_ = lean_ctor_get_uint8(v_o_596_, sizeof(void*)*1);
v_isSharedCheck_614_ = !lean_is_exclusive(v_o_596_);
if (v_isSharedCheck_614_ == 0)
{
v___x_602_ = v_o_596_;
v_isShared_603_ = v_isSharedCheck_614_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_map_599_);
lean_dec(v_o_596_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_614_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_604_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_604_, 0, v_v_598_);
lean_inc(v_k_597_);
v___x_605_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_597_, v___x_604_, v_map_599_);
if (v_hasTrace_600_ == 0)
{
lean_object* v___x_606_; uint8_t v___x_607_; lean_object* v___x_609_; 
v___x_606_ = ((lean_object*)(lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___closed__1));
v___x_607_ = l_Lean_Name_isPrefixOf(v___x_606_, v_k_597_);
lean_dec(v_k_597_);
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_605_);
v___x_609_ = v___x_602_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v___x_605_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
lean_ctor_set_uint8(v___x_609_, sizeof(void*)*1, v___x_607_);
return v___x_609_;
}
}
else
{
lean_object* v___x_612_; 
lean_dec(v_k_597_);
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_605_);
v___x_612_ = v___x_602_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_605_);
lean_ctor_set_uint8(v_reuseFailAlloc_613_, sizeof(void*)*1, v_hasTrace_600_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2___boxed(lean_object* v_o_615_, lean_object* v_k_616_, lean_object* v_v_617_){
_start:
{
uint8_t v_v_boxed_618_; lean_object* v_res_619_; 
v_v_boxed_618_ = lean_unbox(v_v_617_);
v_res_619_ = lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2(v_o_615_, v_k_616_, v_v_boxed_618_);
return v_res_619_;
}
}
LEAN_EXPORT uint8_t lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__3(lean_object* v_opts_620_, lean_object* v_opt_621_){
_start:
{
lean_object* v_name_622_; lean_object* v_defValue_623_; lean_object* v_map_624_; lean_object* v___x_625_; 
v_name_622_ = lean_ctor_get(v_opt_621_, 0);
v_defValue_623_ = lean_ctor_get(v_opt_621_, 1);
v_map_624_ = lean_ctor_get(v_opts_620_, 0);
v___x_625_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_624_, v_name_622_);
if (lean_obj_tag(v___x_625_) == 0)
{
uint8_t v___x_626_; 
v___x_626_ = lean_unbox(v_defValue_623_);
return v___x_626_;
}
else
{
lean_object* v_val_627_; 
v_val_627_ = lean_ctor_get(v___x_625_, 0);
lean_inc(v_val_627_);
lean_dec_ref_known(v___x_625_, 1);
if (lean_obj_tag(v_val_627_) == 1)
{
uint8_t v_v_628_; 
v_v_628_ = lean_ctor_get_uint8(v_val_627_, 0);
lean_dec_ref_known(v_val_627_, 0);
return v_v_628_;
}
else
{
uint8_t v___x_629_; 
lean_dec(v_val_627_);
v___x_629_ = lean_unbox(v_defValue_623_);
return v___x_629_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__3___boxed(lean_object* v_opts_630_, lean_object* v_opt_631_){
_start:
{
uint8_t v_res_632_; lean_object* v_r_633_; 
v_res_632_ = lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__3(v_opts_630_, v_opt_631_);
lean_dec_ref(v_opt_631_);
lean_dec_ref(v_opts_630_);
v_r_633_ = lean_box(v_res_632_);
return v_r_633_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__4(lean_object* v_opts_634_, lean_object* v_opt_635_){
_start:
{
lean_object* v_name_636_; lean_object* v_defValue_637_; lean_object* v_map_638_; lean_object* v___x_639_; 
v_name_636_ = lean_ctor_get(v_opt_635_, 0);
v_defValue_637_ = lean_ctor_get(v_opt_635_, 1);
v_map_638_ = lean_ctor_get(v_opts_634_, 0);
v___x_639_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_638_, v_name_636_);
if (lean_obj_tag(v___x_639_) == 0)
{
lean_inc(v_defValue_637_);
return v_defValue_637_;
}
else
{
lean_object* v_val_640_; 
v_val_640_ = lean_ctor_get(v___x_639_, 0);
lean_inc(v_val_640_);
lean_dec_ref_known(v___x_639_, 1);
if (lean_obj_tag(v_val_640_) == 3)
{
lean_object* v_v_641_; 
v_v_641_ = lean_ctor_get(v_val_640_, 0);
lean_inc(v_v_641_);
lean_dec_ref_known(v_val_640_, 1);
return v_v_641_;
}
else
{
lean_dec(v_val_640_);
lean_inc(v_defValue_637_);
return v_defValue_637_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__4___boxed(lean_object* v_opts_642_, lean_object* v_opt_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__4(v_opts_642_, v_opt_643_);
lean_dec_ref(v_opt_643_);
lean_dec_ref(v_opts_642_);
return v_res_644_;
}
}
static lean_object* _init_lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_650_ = l_Lean_maxRecDepthErrorMessage;
v___x_651_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_651_, 0, v___x_650_);
return v___x_651_;
}
}
static lean_object* _init_lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__4(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; 
v___x_652_ = lean_obj_once(&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__3, &lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__3_once, _init_lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__3);
v___x_653_ = l_Lean_MessageData_ofFormat(v___x_652_);
return v___x_653_;
}
}
static lean_object* _init_lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_654_ = lean_obj_once(&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__4, &lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__4_once, _init_lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__4);
v___x_655_ = ((lean_object*)(lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__2));
v___x_656_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_656_, 0, v___x_655_);
lean_ctor_set(v___x_656_, 1, v___x_654_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg(lean_object* v_ref_657_){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_659_ = lean_obj_once(&lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__5, &lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__5_once, _init_lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___closed__5);
v___x_660_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_660_, 0, v_ref_657_);
lean_ctor_set(v___x_660_, 1, v___x_659_);
v___x_661_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_661_, 0, v___x_660_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg___boxed(lean_object* v_ref_662_, lean_object* v___y_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg(v_ref_662_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6(lean_object* v_00_u03b1_665_, lean_object* v_ref_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg(v_ref_666_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___boxed(lean_object* v_00_u03b1_675_, lean_object* v_ref_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6(v_00_u03b1_675_, v_ref_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted___lam__0(lean_object* v_____r_685_, lean_object* v_res_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_){
_start:
{
lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_695_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_695_, 0, v_res_686_);
lean_ctor_set(v___x_695_, 1, v___y_687_);
v___x_696_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_696_, 0, v___x_695_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted___lam__0___boxed(lean_object* v_____r_697_, lean_object* v_res_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_Qq_Qq_Impl_withDelabQuoted___lam__0(v_____r_697_, v_res_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_);
lean_dec(v___y_705_);
lean_dec_ref(v___y_704_);
lean_dec(v___y_703_);
lean_dec_ref(v___y_702_);
lean_dec(v___y_701_);
lean_dec_ref(v___y_700_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg(lean_object* v_a_708_, lean_object* v_x_709_){
_start:
{
if (lean_obj_tag(v_x_709_) == 0)
{
lean_object* v___x_710_; 
v___x_710_ = lean_box(0);
return v___x_710_;
}
else
{
lean_object* v_key_711_; lean_object* v_value_712_; lean_object* v_tail_713_; uint8_t v___x_714_; 
v_key_711_ = lean_ctor_get(v_x_709_, 0);
v_value_712_ = lean_ctor_get(v_x_709_, 1);
v_tail_713_ = lean_ctor_get(v_x_709_, 2);
v___x_714_ = lean_expr_eqv(v_key_711_, v_a_708_);
if (v___x_714_ == 0)
{
v_x_709_ = v_tail_713_;
goto _start;
}
else
{
lean_object* v___x_716_; 
lean_inc(v_value_712_);
v___x_716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_716_, 0, v_value_712_);
return v___x_716_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg___boxed(lean_object* v_a_717_, lean_object* v_x_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg(v_a_717_, v_x_718_);
lean_dec(v_x_718_);
lean_dec_ref(v_a_717_);
return v_res_719_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg(lean_object* v_m_720_, lean_object* v_a_721_){
_start:
{
lean_object* v_buckets_722_; lean_object* v___x_723_; uint64_t v___x_724_; uint64_t v___x_725_; uint64_t v___x_726_; uint64_t v_fold_727_; uint64_t v___x_728_; uint64_t v___x_729_; uint64_t v___x_730_; size_t v___x_731_; size_t v___x_732_; size_t v___x_733_; size_t v___x_734_; size_t v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; 
v_buckets_722_ = lean_ctor_get(v_m_720_, 1);
v___x_723_ = lean_array_get_size(v_buckets_722_);
v___x_724_ = l_Lean_Expr_hash(v_a_721_);
v___x_725_ = 32ULL;
v___x_726_ = lean_uint64_shift_right(v___x_724_, v___x_725_);
v_fold_727_ = lean_uint64_xor(v___x_724_, v___x_726_);
v___x_728_ = 16ULL;
v___x_729_ = lean_uint64_shift_right(v_fold_727_, v___x_728_);
v___x_730_ = lean_uint64_xor(v_fold_727_, v___x_729_);
v___x_731_ = lean_uint64_to_usize(v___x_730_);
v___x_732_ = lean_usize_of_nat(v___x_723_);
v___x_733_ = ((size_t)1ULL);
v___x_734_ = lean_usize_sub(v___x_732_, v___x_733_);
v___x_735_ = lean_usize_land(v___x_731_, v___x_734_);
v___x_736_ = lean_array_uget_borrowed(v_buckets_722_, v___x_735_);
v___x_737_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg(v_a_721_, v___x_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg___boxed(lean_object* v_m_738_, lean_object* v_a_739_){
_start:
{
lean_object* v_res_740_; 
v_res_740_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg(v_m_738_, v_a_739_);
lean_dec_ref(v_a_739_);
lean_dec_ref(v_m_738_);
return v_res_740_;
}
}
LEAN_EXPORT uint8_t lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___lam__0(lean_object* v_val_741_, lean_object* v_x_742_){
_start:
{
lean_object* v___x_743_; lean_object* v___x_744_; uint8_t v___x_745_; 
v___x_743_ = l_Lean_Syntax_getId(v_x_742_);
v___x_744_ = l_Lean_LocalDecl_userName(v_val_741_);
v___x_745_ = lean_name_eq(v___x_743_, v___x_744_);
lean_dec(v___x_744_);
lean_dec(v___x_743_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___lam__0___boxed(lean_object* v_val_746_, lean_object* v_x_747_){
_start:
{
uint8_t v_res_748_; lean_object* v_r_749_; 
v_res_748_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___lam__0(v_val_746_, v_x_747_);
lean_dec(v_x_747_);
lean_dec_ref(v_val_746_);
v_r_749_ = lean_box(v_res_748_);
return v_r_749_;
}
}
static lean_object* _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11(void){
_start:
{
lean_object* v___x_773_; 
v___x_773_ = l_Array_mkArray0(lean_box(0));
return v___x_773_;
}
}
static lean_object* _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__20(void){
_start:
{
lean_object* v___x_794_; 
v___x_794_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_794_;
}
}
static lean_object* _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__21(void){
_start:
{
lean_object* v___x_795_; lean_object* v___x_796_; 
v___x_795_ = lean_obj_once(&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__20, &lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__20_once, _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__20);
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v___x_795_);
return v___x_796_;
}
}
static lean_object* _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__22(void){
_start:
{
lean_object* v___x_797_; lean_object* v___x_798_; 
v___x_797_ = lean_obj_once(&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__21, &lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__21_once, _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__21);
v___x_798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_798_, 0, v___x_797_);
lean_ctor_set(v___x_798_, 1, v___x_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5(lean_object* v_as_799_, size_t v_sz_800_, size_t v_i_801_, lean_object* v_b_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_){
_start:
{
lean_object* v_a_812_; lean_object* v_snd_813_; uint8_t v___x_817_; 
v___x_817_ = lean_usize_dec_lt(v_i_801_, v_sz_800_);
if (v___x_817_ == 0)
{
lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_818_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_818_, 0, v_b_802_);
lean_ctor_set(v___x_818_, 1, v___y_803_);
v___x_819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
return v___x_819_;
}
else
{
lean_object* v_unquoted_820_; lean_object* v_exprBackSubst_821_; lean_object* v_a_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v_unquoted_820_ = lean_ctor_get(v___y_803_, 3);
v_exprBackSubst_821_ = lean_ctor_get(v___y_803_, 4);
v_a_822_ = lean_array_uget_borrowed(v_as_799_, v_i_801_);
lean_inc(v_a_822_);
v___x_823_ = l_Lean_Expr_fvar___override(v_a_822_);
v___x_824_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg(v_exprBackSubst_821_, v___x_823_);
lean_dec_ref(v___x_823_);
if (lean_obj_tag(v___x_824_) == 1)
{
lean_object* v_val_825_; 
v_val_825_ = lean_ctor_get(v___x_824_, 0);
lean_inc(v_val_825_);
lean_dec_ref_known(v___x_824_, 1);
if (lean_obj_tag(v_val_825_) == 0)
{
lean_object* v_e_826_; lean_object* v___x_827_; 
v_e_826_ = lean_ctor_get(v_val_825_, 0);
lean_inc_ref(v_e_826_);
lean_dec_ref_known(v_val_825_, 1);
lean_inc(v_a_822_);
lean_inc_ref(v_unquoted_820_);
v___x_827_ = lean_local_ctx_find(v_unquoted_820_, v_a_822_);
if (lean_obj_tag(v___x_827_) == 1)
{
lean_object* v_val_828_; lean_object* v___f_829_; lean_object* v___x_830_; 
v_val_828_ = lean_ctor_get(v___x_827_, 0);
lean_inc_n(v_val_828_, 2);
lean_dec_ref_known(v___x_827_, 1);
v___f_829_ = lean_alloc_closure((void*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___lam__0___boxed), 2, 1);
lean_closure_set(v___f_829_, 0, v_val_828_);
lean_inc(v_b_802_);
v___x_830_ = l_Lean_Syntax_find_x3f(v_b_802_, v___f_829_);
if (lean_obj_tag(v___x_830_) == 0)
{
lean_dec(v_val_828_);
lean_dec_ref(v_e_826_);
v_a_812_ = v_b_802_;
v_snd_813_ = v___y_803_;
goto v___jp_811_;
}
else
{
lean_object* v___x_831_; lean_object* v___x_832_; 
lean_dec_ref_known(v___x_830_, 1);
v___x_831_ = l_Lean_LocalDecl_userName(v_val_828_);
lean_dec(v_val_828_);
v___x_832_ = lp_Qq_Qq_Impl_removeDollar(v___x_831_);
if (lean_obj_tag(v___x_832_) == 1)
{
lean_object* v_val_833_; lean_object* v___x_834_; 
v_val_833_ = lean_ctor_get(v___x_832_, 0);
lean_inc(v_val_833_);
lean_dec_ref_known(v___x_832_, 1);
v___x_834_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_nextExtraPos___at___00Qq_Impl_withDelabQuoted_spec__1___redArg(v___y_803_, v___y_805_);
if (lean_obj_tag(v___x_834_) == 0)
{
lean_object* v_a_835_; lean_object* v_fst_836_; lean_object* v_snd_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_944_; 
v_a_835_ = lean_ctor_get(v___x_834_, 0);
lean_inc(v_a_835_);
lean_dec_ref_known(v___x_834_, 1);
v_fst_836_ = lean_ctor_get(v_a_835_, 0);
v_snd_837_ = lean_ctor_get(v_a_835_, 1);
v_isSharedCheck_944_ = !lean_is_exclusive(v_a_835_);
if (v_isSharedCheck_944_ == 0)
{
v___x_839_ = v_a_835_;
v_isShared_840_ = v_isSharedCheck_944_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_snd_837_);
lean_inc(v_fst_836_);
lean_dec(v_a_835_);
v___x_839_ = lean_box(0);
v_isShared_840_ = v_isSharedCheck_944_;
goto v_resetjp_838_;
}
v_resetjp_838_:
{
lean_object* v___x_841_; lean_object* v_optionsPerPos_842_; lean_object* v_currNamespace_843_; lean_object* v_openDecls_844_; uint8_t v_inPattern_845_; lean_object* v_depth_846_; lean_object* v_lctxInitIndices_847_; lean_object* v_fileName_848_; lean_object* v_fileMap_849_; lean_object* v_options_850_; lean_object* v_currRecDepth_851_; lean_object* v_ref_852_; lean_object* v_currNamespace_853_; lean_object* v_openDecls_854_; lean_object* v_initHeartbeats_855_; lean_object* v_maxHeartbeats_856_; lean_object* v_quotContext_857_; lean_object* v_currMacroScope_858_; lean_object* v_cancelTk_x3f_859_; uint8_t v_suppressElabErrors_860_; lean_object* v_inheritedTraceOptions_861_; lean_object* v_env_862_; lean_object* v___x_863_; lean_object* v___x_865_; 
v___x_841_ = lean_st_ref_get(v___y_809_);
v_optionsPerPos_842_ = lean_ctor_get(v___y_804_, 0);
v_currNamespace_843_ = lean_ctor_get(v___y_804_, 1);
v_openDecls_844_ = lean_ctor_get(v___y_804_, 2);
v_inPattern_845_ = lean_ctor_get_uint8(v___y_804_, sizeof(void*)*6);
v_depth_846_ = lean_ctor_get(v___y_804_, 4);
v_lctxInitIndices_847_ = lean_ctor_get(v___y_804_, 5);
v_fileName_848_ = lean_ctor_get(v___y_808_, 0);
v_fileMap_849_ = lean_ctor_get(v___y_808_, 1);
v_options_850_ = lean_ctor_get(v___y_808_, 2);
v_currRecDepth_851_ = lean_ctor_get(v___y_808_, 3);
v_ref_852_ = lean_ctor_get(v___y_808_, 5);
v_currNamespace_853_ = lean_ctor_get(v___y_808_, 6);
v_openDecls_854_ = lean_ctor_get(v___y_808_, 7);
v_initHeartbeats_855_ = lean_ctor_get(v___y_808_, 8);
v_maxHeartbeats_856_ = lean_ctor_get(v___y_808_, 9);
v_quotContext_857_ = lean_ctor_get(v___y_808_, 10);
v_currMacroScope_858_ = lean_ctor_get(v___y_808_, 11);
v_cancelTk_x3f_859_ = lean_ctor_get(v___y_808_, 12);
v_suppressElabErrors_860_ = lean_ctor_get_uint8(v___y_808_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_861_ = lean_ctor_get(v___y_808_, 13);
v_env_862_ = lean_ctor_get(v___x_841_, 0);
lean_inc_ref(v_env_862_);
lean_dec(v___x_841_);
v___x_863_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1));
if (v_isShared_840_ == 0)
{
lean_ctor_set(v___x_839_, 1, v_fst_836_);
lean_ctor_set(v___x_839_, 0, v_e_826_);
v___x_865_ = v___x_839_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_e_826_);
lean_ctor_set(v_reuseFailAlloc_943_, 1, v_fst_836_);
v___x_865_ = v_reuseFailAlloc_943_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
lean_object* v___x_866_; uint8_t v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; uint8_t v___x_870_; lean_object* v_fileName_872_; lean_object* v_fileMap_873_; lean_object* v_currRecDepth_874_; lean_object* v_ref_875_; lean_object* v_currNamespace_876_; lean_object* v_openDecls_877_; lean_object* v_initHeartbeats_878_; lean_object* v_maxHeartbeats_879_; lean_object* v_quotContext_880_; lean_object* v_currMacroScope_881_; lean_object* v_cancelTk_x3f_882_; uint8_t v_suppressElabErrors_883_; lean_object* v_inheritedTraceOptions_884_; lean_object* v___y_885_; uint8_t v___y_921_; uint8_t v___x_942_; 
lean_inc(v_lctxInitIndices_847_);
lean_inc(v_depth_846_);
lean_inc(v_openDecls_844_);
lean_inc(v_currNamespace_843_);
lean_inc(v_optionsPerPos_842_);
v___x_866_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_866_, 0, v_optionsPerPos_842_);
lean_ctor_set(v___x_866_, 1, v_currNamespace_843_);
lean_ctor_set(v___x_866_, 2, v_openDecls_844_);
lean_ctor_set(v___x_866_, 3, v___x_865_);
lean_ctor_set(v___x_866_, 4, v_depth_846_);
lean_ctor_set(v___x_866_, 5, v_lctxInitIndices_847_);
lean_ctor_set_uint8(v___x_866_, sizeof(void*)*6, v_inPattern_845_);
v___x_867_ = 0;
lean_inc_ref(v_options_850_);
v___x_868_ = lp_Qq_Lean_Options_set___at___00Qq_Impl_withDelabQuoted_spec__2(v_options_850_, v___x_863_, v___x_867_);
v___x_869_ = l_Lean_diagnostics;
v___x_870_ = lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__3(v___x_868_, v___x_869_);
v___x_942_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_862_);
lean_dec_ref(v_env_862_);
if (v___x_942_ == 0)
{
if (v___x_870_ == 0)
{
v___y_921_ = v___x_817_;
goto v___jp_920_;
}
else
{
v___y_921_ = v___x_942_;
goto v___jp_920_;
}
}
else
{
v___y_921_ = v___x_870_;
goto v___jp_920_;
}
v___jp_871_:
{
lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; 
v___x_886_ = l_Lean_maxRecDepth;
v___x_887_ = lp_Qq_Lean_Option_get___at___00Qq_Impl_withDelabQuoted_spec__4(v___x_868_, v___x_886_);
lean_inc_ref(v_inheritedTraceOptions_884_);
lean_inc(v_cancelTk_x3f_882_);
lean_inc(v_currMacroScope_881_);
lean_inc(v_quotContext_880_);
lean_inc(v_maxHeartbeats_879_);
lean_inc(v_initHeartbeats_878_);
lean_inc(v_openDecls_877_);
lean_inc(v_currNamespace_876_);
lean_inc(v_ref_875_);
lean_inc(v_currRecDepth_874_);
lean_inc_ref(v_fileMap_873_);
lean_inc_ref(v_fileName_872_);
v___x_888_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_888_, 0, v_fileName_872_);
lean_ctor_set(v___x_888_, 1, v_fileMap_873_);
lean_ctor_set(v___x_888_, 2, v___x_868_);
lean_ctor_set(v___x_888_, 3, v_currRecDepth_874_);
lean_ctor_set(v___x_888_, 4, v___x_887_);
lean_ctor_set(v___x_888_, 5, v_ref_875_);
lean_ctor_set(v___x_888_, 6, v_currNamespace_876_);
lean_ctor_set(v___x_888_, 7, v_openDecls_877_);
lean_ctor_set(v___x_888_, 8, v_initHeartbeats_878_);
lean_ctor_set(v___x_888_, 9, v_maxHeartbeats_879_);
lean_ctor_set(v___x_888_, 10, v_quotContext_880_);
lean_ctor_set(v___x_888_, 11, v_currMacroScope_881_);
lean_ctor_set(v___x_888_, 12, v_cancelTk_x3f_882_);
lean_ctor_set(v___x_888_, 13, v_inheritedTraceOptions_884_);
lean_ctor_set_uint8(v___x_888_, sizeof(void*)*14, v___x_870_);
lean_ctor_set_uint8(v___x_888_, sizeof(void*)*14 + 1, v_suppressElabErrors_883_);
v___x_889_ = l_Lean_PrettyPrinter_Delaborator_delab(v___x_866_, v___y_805_, v___y_806_, v___y_807_, v___x_888_, v___y_885_);
lean_dec_ref_known(v___x_888_, 14);
lean_dec_ref_known(v___x_866_, 6);
if (lean_obj_tag(v___x_889_) == 0)
{
lean_object* v_a_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; 
v_a_890_ = lean_ctor_get(v___x_889_, 0);
lean_inc(v_a_890_);
lean_dec_ref_known(v___x_889_, 1);
v___x_891_ = l_Lean_SourceInfo_fromRef(v_ref_875_, v___x_867_);
v___x_892_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__5));
v___x_893_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__6));
lean_inc_n(v___x_891_, 8);
v___x_894_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_894_, 0, v___x_891_);
lean_ctor_set(v___x_894_, 1, v___x_892_);
v___x_895_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__8));
v___x_896_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__10));
v___x_897_ = lean_obj_once(&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11, &lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11_once, _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11);
v___x_898_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_898_, 0, v___x_891_);
lean_ctor_set(v___x_898_, 1, v___x_896_);
lean_ctor_set(v___x_898_, 2, v___x_897_);
lean_inc_ref_n(v___x_898_, 2);
v___x_899_ = l_Lean_Syntax_node1(v___x_891_, v___x_895_, v___x_898_);
v___x_900_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__13));
v___x_901_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__15));
v___x_902_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__17));
v___x_903_ = l_Lean_mkIdent(v_val_833_);
v___x_904_ = l_Lean_Syntax_node1(v___x_891_, v___x_902_, v___x_903_);
v___x_905_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__18));
v___x_906_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_906_, 0, v___x_891_);
lean_ctor_set(v___x_906_, 1, v___x_905_);
v___x_907_ = l_Lean_Syntax_node5(v___x_891_, v___x_901_, v___x_904_, v___x_898_, v___x_898_, v___x_906_, v_a_890_);
v___x_908_ = l_Lean_Syntax_node1(v___x_891_, v___x_900_, v___x_907_);
v___x_909_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__19));
v___x_910_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_910_, 0, v___x_891_);
lean_ctor_set(v___x_910_, 1, v___x_909_);
v___x_911_ = l_Lean_Syntax_node5(v___x_891_, v___x_893_, v___x_894_, v___x_899_, v___x_908_, v___x_910_, v_b_802_);
v_a_812_ = v___x_911_;
v_snd_813_ = v_snd_837_;
goto v___jp_811_;
}
else
{
lean_object* v_a_912_; lean_object* v___x_914_; uint8_t v_isShared_915_; uint8_t v_isSharedCheck_919_; 
lean_dec(v_snd_837_);
lean_dec(v_val_833_);
lean_dec(v_b_802_);
v_a_912_ = lean_ctor_get(v___x_889_, 0);
v_isSharedCheck_919_ = !lean_is_exclusive(v___x_889_);
if (v_isSharedCheck_919_ == 0)
{
v___x_914_ = v___x_889_;
v_isShared_915_ = v_isSharedCheck_919_;
goto v_resetjp_913_;
}
else
{
lean_inc(v_a_912_);
lean_dec(v___x_889_);
v___x_914_ = lean_box(0);
v_isShared_915_ = v_isSharedCheck_919_;
goto v_resetjp_913_;
}
v_resetjp_913_:
{
lean_object* v___x_917_; 
if (v_isShared_915_ == 0)
{
v___x_917_ = v___x_914_;
goto v_reusejp_916_;
}
else
{
lean_object* v_reuseFailAlloc_918_; 
v_reuseFailAlloc_918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_918_, 0, v_a_912_);
v___x_917_ = v_reuseFailAlloc_918_;
goto v_reusejp_916_;
}
v_reusejp_916_:
{
return v___x_917_;
}
}
}
}
v___jp_920_:
{
if (v___y_921_ == 0)
{
lean_object* v___x_922_; lean_object* v_env_923_; lean_object* v_nextMacroScope_924_; lean_object* v_ngen_925_; lean_object* v_auxDeclNGen_926_; lean_object* v_traceState_927_; lean_object* v_messages_928_; lean_object* v_infoState_929_; lean_object* v_snapshotTasks_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_940_; 
v___x_922_ = lean_st_ref_take(v___y_809_);
v_env_923_ = lean_ctor_get(v___x_922_, 0);
v_nextMacroScope_924_ = lean_ctor_get(v___x_922_, 1);
v_ngen_925_ = lean_ctor_get(v___x_922_, 2);
v_auxDeclNGen_926_ = lean_ctor_get(v___x_922_, 3);
v_traceState_927_ = lean_ctor_get(v___x_922_, 4);
v_messages_928_ = lean_ctor_get(v___x_922_, 6);
v_infoState_929_ = lean_ctor_get(v___x_922_, 7);
v_snapshotTasks_930_ = lean_ctor_get(v___x_922_, 8);
v_isSharedCheck_940_ = !lean_is_exclusive(v___x_922_);
if (v_isSharedCheck_940_ == 0)
{
lean_object* v_unused_941_; 
v_unused_941_ = lean_ctor_get(v___x_922_, 5);
lean_dec(v_unused_941_);
v___x_932_ = v___x_922_;
v_isShared_933_ = v_isSharedCheck_940_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_snapshotTasks_930_);
lean_inc(v_infoState_929_);
lean_inc(v_messages_928_);
lean_inc(v_traceState_927_);
lean_inc(v_auxDeclNGen_926_);
lean_inc(v_ngen_925_);
lean_inc(v_nextMacroScope_924_);
lean_inc(v_env_923_);
lean_dec(v___x_922_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_940_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_937_; 
v___x_934_ = l_Lean_Kernel_enableDiag(v_env_923_, v___x_870_);
v___x_935_ = lean_obj_once(&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__22, &lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__22_once, _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__22);
if (v_isShared_933_ == 0)
{
lean_ctor_set(v___x_932_, 5, v___x_935_);
lean_ctor_set(v___x_932_, 0, v___x_934_);
v___x_937_ = v___x_932_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v___x_934_);
lean_ctor_set(v_reuseFailAlloc_939_, 1, v_nextMacroScope_924_);
lean_ctor_set(v_reuseFailAlloc_939_, 2, v_ngen_925_);
lean_ctor_set(v_reuseFailAlloc_939_, 3, v_auxDeclNGen_926_);
lean_ctor_set(v_reuseFailAlloc_939_, 4, v_traceState_927_);
lean_ctor_set(v_reuseFailAlloc_939_, 5, v___x_935_);
lean_ctor_set(v_reuseFailAlloc_939_, 6, v_messages_928_);
lean_ctor_set(v_reuseFailAlloc_939_, 7, v_infoState_929_);
lean_ctor_set(v_reuseFailAlloc_939_, 8, v_snapshotTasks_930_);
v___x_937_ = v_reuseFailAlloc_939_;
goto v_reusejp_936_;
}
v_reusejp_936_:
{
lean_object* v___x_938_; 
v___x_938_ = lean_st_ref_set(v___y_809_, v___x_937_);
v_fileName_872_ = v_fileName_848_;
v_fileMap_873_ = v_fileMap_849_;
v_currRecDepth_874_ = v_currRecDepth_851_;
v_ref_875_ = v_ref_852_;
v_currNamespace_876_ = v_currNamespace_853_;
v_openDecls_877_ = v_openDecls_854_;
v_initHeartbeats_878_ = v_initHeartbeats_855_;
v_maxHeartbeats_879_ = v_maxHeartbeats_856_;
v_quotContext_880_ = v_quotContext_857_;
v_currMacroScope_881_ = v_currMacroScope_858_;
v_cancelTk_x3f_882_ = v_cancelTk_x3f_859_;
v_suppressElabErrors_883_ = v_suppressElabErrors_860_;
v_inheritedTraceOptions_884_ = v_inheritedTraceOptions_861_;
v___y_885_ = v___y_809_;
goto v___jp_871_;
}
}
}
else
{
v_fileName_872_ = v_fileName_848_;
v_fileMap_873_ = v_fileMap_849_;
v_currRecDepth_874_ = v_currRecDepth_851_;
v_ref_875_ = v_ref_852_;
v_currNamespace_876_ = v_currNamespace_853_;
v_openDecls_877_ = v_openDecls_854_;
v_initHeartbeats_878_ = v_initHeartbeats_855_;
v_maxHeartbeats_879_ = v_maxHeartbeats_856_;
v_quotContext_880_ = v_quotContext_857_;
v_currMacroScope_881_ = v_currMacroScope_858_;
v_cancelTk_x3f_882_ = v_cancelTk_x3f_859_;
v_suppressElabErrors_883_ = v_suppressElabErrors_860_;
v_inheritedTraceOptions_884_ = v_inheritedTraceOptions_861_;
v___y_885_ = v___y_809_;
goto v___jp_871_;
}
}
}
}
}
else
{
lean_object* v_a_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_952_; 
lean_dec(v_val_833_);
lean_dec_ref(v_e_826_);
lean_dec(v_b_802_);
v_a_945_ = lean_ctor_get(v___x_834_, 0);
v_isSharedCheck_952_ = !lean_is_exclusive(v___x_834_);
if (v_isSharedCheck_952_ == 0)
{
v___x_947_ = v___x_834_;
v_isShared_948_ = v_isSharedCheck_952_;
goto v_resetjp_946_;
}
else
{
lean_inc(v_a_945_);
lean_dec(v___x_834_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_952_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v___x_950_; 
if (v_isShared_948_ == 0)
{
v___x_950_ = v___x_947_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_951_; 
v_reuseFailAlloc_951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_951_, 0, v_a_945_);
v___x_950_ = v_reuseFailAlloc_951_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
return v___x_950_;
}
}
}
}
else
{
lean_dec(v___x_832_);
lean_dec_ref(v_e_826_);
v_a_812_ = v_b_802_;
v_snd_813_ = v___y_803_;
goto v___jp_811_;
}
}
}
else
{
lean_dec(v___x_827_);
lean_dec_ref(v_e_826_);
v_a_812_ = v_b_802_;
v_snd_813_ = v___y_803_;
goto v___jp_811_;
}
}
else
{
lean_dec(v_val_825_);
v_a_812_ = v_b_802_;
v_snd_813_ = v___y_803_;
goto v___jp_811_;
}
}
else
{
lean_dec(v___x_824_);
v_a_812_ = v_b_802_;
v_snd_813_ = v___y_803_;
goto v___jp_811_;
}
}
v___jp_811_:
{
size_t v___x_814_; size_t v___x_815_; 
v___x_814_ = ((size_t)1ULL);
v___x_815_ = lean_usize_add(v_i_801_, v___x_814_);
v_i_801_ = v___x_815_;
v_b_802_ = v_a_812_;
v___y_803_ = v_snd_813_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___boxed(lean_object* v_as_953_, lean_object* v_sz_954_, lean_object* v_i_955_, lean_object* v_b_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_){
_start:
{
size_t v_sz_boxed_965_; size_t v_i_boxed_966_; lean_object* v_res_967_; 
v_sz_boxed_965_ = lean_unbox_usize(v_sz_954_);
lean_dec(v_sz_954_);
v_i_boxed_966_ = lean_unbox_usize(v_i_955_);
lean_dec(v_i_955_);
v_res_967_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5(v_as_953_, v_sz_boxed_965_, v_i_boxed_966_, v_b_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_, v___y_961_, v___y_962_, v___y_963_);
lean_dec(v___y_963_);
lean_dec_ref(v___y_962_);
lean_dec(v___y_961_);
lean_dec_ref(v___y_960_);
lean_dec(v___y_959_);
lean_dec_ref(v___y_958_);
lean_dec_ref(v_as_953_);
return v_res_967_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__0(void){
_start:
{
lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_968_ = lean_box(0);
v___x_969_ = lean_unsigned_to_nat(16u);
v___x_970_ = lean_mk_array(v___x_969_, v___x_968_);
return v___x_970_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__1(void){
_start:
{
lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; 
v___x_971_ = lean_obj_once(&lp_Qq_Qq_Impl_withDelabQuoted___closed__0, &lp_Qq_Qq_Impl_withDelabQuoted___closed__0_once, _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__0);
v___x_972_ = lean_unsigned_to_nat(0u);
v___x_973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_973_, 0, v___x_972_);
lean_ctor_set(v___x_973_, 1, v___x_971_);
return v___x_973_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__3(void){
_start:
{
uint8_t v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; 
v___x_976_ = 0;
v___x_977_ = ((lean_object*)(lp_Qq_Qq_Impl_withDelabQuoted___closed__2));
v___x_978_ = l_Lean_LocalContext_empty;
v___x_979_ = lean_obj_once(&lp_Qq_Qq_Impl_withDelabQuoted___closed__1, &lp_Qq_Qq_Impl_withDelabQuoted___closed__1_once, _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__1);
v___x_980_ = lean_box(0);
v___x_981_ = lean_alloc_ctor(0, 8, 1);
lean_ctor_set(v___x_981_, 0, v___x_980_);
lean_ctor_set(v___x_981_, 1, v___x_979_);
lean_ctor_set(v___x_981_, 2, v___x_979_);
lean_ctor_set(v___x_981_, 3, v___x_978_);
lean_ctor_set(v___x_981_, 4, v___x_979_);
lean_ctor_set(v___x_981_, 5, v___x_979_);
lean_ctor_set(v___x_981_, 6, v___x_977_);
lean_ctor_set(v___x_981_, 7, v___x_980_);
lean_ctor_set_uint8(v___x_981_, sizeof(void*)*8, v___x_976_);
return v___x_981_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted(lean_object* v_k_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_, lean_object* v_a_988_){
_start:
{
lean_object* v___y_991_; lean_object* v___x_1009_; lean_object* v_fileName_1010_; lean_object* v_fileMap_1011_; lean_object* v_options_1012_; lean_object* v_currRecDepth_1013_; lean_object* v_maxRecDepth_1014_; lean_object* v_ref_1015_; lean_object* v_currNamespace_1016_; lean_object* v_openDecls_1017_; lean_object* v_initHeartbeats_1018_; lean_object* v_maxHeartbeats_1019_; lean_object* v_quotContext_1020_; lean_object* v_currMacroScope_1021_; uint8_t v_diag_1022_; lean_object* v_cancelTk_x3f_1023_; uint8_t v_suppressElabErrors_1024_; lean_object* v_inheritedTraceOptions_1025_; lean_object* v___y_1027_; lean_object* v___y_1028_; lean_object* v___y_1029_; lean_object* v___x_1040_; uint8_t v___x_1075_; 
v___x_1009_ = lean_unsigned_to_nat(0u);
v_fileName_1010_ = lean_ctor_get(v_a_987_, 0);
v_fileMap_1011_ = lean_ctor_get(v_a_987_, 1);
v_options_1012_ = lean_ctor_get(v_a_987_, 2);
v_currRecDepth_1013_ = lean_ctor_get(v_a_987_, 3);
v_maxRecDepth_1014_ = lean_ctor_get(v_a_987_, 4);
v_ref_1015_ = lean_ctor_get(v_a_987_, 5);
v_currNamespace_1016_ = lean_ctor_get(v_a_987_, 6);
v_openDecls_1017_ = lean_ctor_get(v_a_987_, 7);
v_initHeartbeats_1018_ = lean_ctor_get(v_a_987_, 8);
v_maxHeartbeats_1019_ = lean_ctor_get(v_a_987_, 9);
v_quotContext_1020_ = lean_ctor_get(v_a_987_, 10);
v_currMacroScope_1021_ = lean_ctor_get(v_a_987_, 11);
v_diag_1022_ = lean_ctor_get_uint8(v_a_987_, sizeof(void*)*14);
v_cancelTk_x3f_1023_ = lean_ctor_get(v_a_987_, 12);
v_suppressElabErrors_1024_ = lean_ctor_get_uint8(v_a_987_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1025_ = lean_ctor_get(v_a_987_, 13);
v___x_1040_ = lean_obj_once(&lp_Qq_Qq_Impl_withDelabQuoted___closed__3, &lp_Qq_Qq_Impl_withDelabQuoted___closed__3_once, _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__3);
v___x_1075_ = lean_nat_dec_eq(v_maxRecDepth_1014_, v___x_1009_);
if (v___x_1075_ == 0)
{
uint8_t v___x_1076_; 
v___x_1076_ = lean_nat_dec_eq(v_currRecDepth_1013_, v_maxRecDepth_1014_);
if (v___x_1076_ == 0)
{
goto v___jp_1041_;
}
else
{
lean_object* v___x_1077_; 
lean_dec_ref(v_k_982_);
lean_inc(v_ref_1015_);
v___x_1077_ = lp_Qq_Lean_throwMaxRecDepthAt___at___00Qq_Impl_withDelabQuoted_spec__6___redArg(v_ref_1015_);
return v___x_1077_;
}
}
else
{
goto v___jp_1041_;
}
v___jp_990_:
{
if (lean_obj_tag(v___y_991_) == 0)
{
lean_object* v_a_992_; lean_object* v___x_994_; uint8_t v_isShared_995_; uint8_t v_isSharedCheck_1000_; 
v_a_992_ = lean_ctor_get(v___y_991_, 0);
v_isSharedCheck_1000_ = !lean_is_exclusive(v___y_991_);
if (v_isSharedCheck_1000_ == 0)
{
v___x_994_ = v___y_991_;
v_isShared_995_ = v_isSharedCheck_1000_;
goto v_resetjp_993_;
}
else
{
lean_inc(v_a_992_);
lean_dec(v___y_991_);
v___x_994_ = lean_box(0);
v_isShared_995_ = v_isSharedCheck_1000_;
goto v_resetjp_993_;
}
v_resetjp_993_:
{
lean_object* v_fst_996_; lean_object* v___x_998_; 
v_fst_996_ = lean_ctor_get(v_a_992_, 0);
lean_inc(v_fst_996_);
lean_dec(v_a_992_);
if (v_isShared_995_ == 0)
{
lean_ctor_set(v___x_994_, 0, v_fst_996_);
v___x_998_ = v___x_994_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_999_; 
v_reuseFailAlloc_999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_999_, 0, v_fst_996_);
v___x_998_ = v_reuseFailAlloc_999_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
return v___x_998_;
}
}
}
else
{
lean_object* v_a_1001_; lean_object* v___x_1003_; uint8_t v_isShared_1004_; uint8_t v_isSharedCheck_1008_; 
v_a_1001_ = lean_ctor_get(v___y_991_, 0);
v_isSharedCheck_1008_ = !lean_is_exclusive(v___y_991_);
if (v_isSharedCheck_1008_ == 0)
{
v___x_1003_ = v___y_991_;
v_isShared_1004_ = v_isSharedCheck_1008_;
goto v_resetjp_1002_;
}
else
{
lean_inc(v_a_1001_);
lean_dec(v___y_991_);
v___x_1003_ = lean_box(0);
v_isShared_1004_ = v_isSharedCheck_1008_;
goto v_resetjp_1002_;
}
v_resetjp_1002_:
{
lean_object* v___x_1006_; 
if (v_isShared_1004_ == 0)
{
v___x_1006_ = v___x_1003_;
goto v_reusejp_1005_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v_a_1001_);
v___x_1006_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1005_;
}
v_reusejp_1005_:
{
return v___x_1006_;
}
}
}
}
v___jp_1026_:
{
lean_object* v_abstractedFVars_1030_; lean_object* v___x_1031_; size_t v_sz_1032_; size_t v___x_1033_; lean_object* v___x_1034_; 
v_abstractedFVars_1030_ = lean_ctor_get(v___y_1029_, 6);
lean_inc_ref(v_abstractedFVars_1030_);
v___x_1031_ = l_Array_reverse___redArg(v_abstractedFVars_1030_);
v_sz_1032_ = lean_array_size(v___x_1031_);
v___x_1033_ = ((size_t)0ULL);
v___x_1034_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5(v___x_1031_, v_sz_1032_, v___x_1033_, v___y_1028_, v___y_1029_, v_a_983_, v_a_984_, v_a_985_, v_a_986_, v___y_1027_, v_a_988_);
lean_dec_ref(v___x_1031_);
if (lean_obj_tag(v___x_1034_) == 0)
{
lean_object* v_a_1035_; lean_object* v_fst_1036_; lean_object* v_snd_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; 
v_a_1035_ = lean_ctor_get(v___x_1034_, 0);
lean_inc(v_a_1035_);
lean_dec_ref_known(v___x_1034_, 1);
v_fst_1036_ = lean_ctor_get(v_a_1035_, 0);
lean_inc(v_fst_1036_);
v_snd_1037_ = lean_ctor_get(v_a_1035_, 1);
lean_inc(v_snd_1037_);
lean_dec(v_a_1035_);
v___x_1038_ = lean_box(0);
v___x_1039_ = lp_Qq_Qq_Impl_withDelabQuoted___lam__0(v___x_1038_, v_fst_1036_, v_snd_1037_, v_a_983_, v_a_984_, v_a_985_, v_a_986_, v___y_1027_, v_a_988_);
lean_dec_ref(v___y_1027_);
v___y_991_ = v___x_1039_;
goto v___jp_990_;
}
else
{
lean_dec_ref(v___y_1027_);
v___y_991_ = v___x_1034_;
goto v___jp_990_;
}
}
v___jp_1041_:
{
lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; 
v___x_1042_ = lean_unsigned_to_nat(1u);
v___x_1043_ = lean_nat_add(v_currRecDepth_1013_, v___x_1042_);
lean_inc_ref(v_inheritedTraceOptions_1025_);
lean_inc(v_cancelTk_x3f_1023_);
lean_inc(v_currMacroScope_1021_);
lean_inc(v_quotContext_1020_);
lean_inc(v_maxHeartbeats_1019_);
lean_inc(v_initHeartbeats_1018_);
lean_inc(v_openDecls_1017_);
lean_inc(v_currNamespace_1016_);
lean_inc(v_ref_1015_);
lean_inc(v_maxRecDepth_1014_);
lean_inc_ref(v_options_1012_);
lean_inc_ref(v_fileMap_1011_);
lean_inc_ref(v_fileName_1010_);
v___x_1044_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1044_, 0, v_fileName_1010_);
lean_ctor_set(v___x_1044_, 1, v_fileMap_1011_);
lean_ctor_set(v___x_1044_, 2, v_options_1012_);
lean_ctor_set(v___x_1044_, 3, v___x_1043_);
lean_ctor_set(v___x_1044_, 4, v_maxRecDepth_1014_);
lean_ctor_set(v___x_1044_, 5, v_ref_1015_);
lean_ctor_set(v___x_1044_, 6, v_currNamespace_1016_);
lean_ctor_set(v___x_1044_, 7, v_openDecls_1017_);
lean_ctor_set(v___x_1044_, 8, v_initHeartbeats_1018_);
lean_ctor_set(v___x_1044_, 9, v_maxHeartbeats_1019_);
lean_ctor_set(v___x_1044_, 10, v_quotContext_1020_);
lean_ctor_set(v___x_1044_, 11, v_currMacroScope_1021_);
lean_ctor_set(v___x_1044_, 12, v_cancelTk_x3f_1023_);
lean_ctor_set(v___x_1044_, 13, v_inheritedTraceOptions_1025_);
lean_ctor_set_uint8(v___x_1044_, sizeof(void*)*14, v_diag_1022_);
lean_ctor_set_uint8(v___x_1044_, sizeof(void*)*14 + 1, v_suppressElabErrors_1024_);
v___x_1045_ = lp_Qq_Qq_Impl_unquoteLCtx(v___x_1040_, v_a_985_, v_a_986_, v___x_1044_, v_a_988_);
if (lean_obj_tag(v___x_1045_) == 0)
{
lean_object* v_a_1046_; lean_object* v_snd_1047_; lean_object* v___x_1048_; 
v_a_1046_ = lean_ctor_get(v___x_1045_, 0);
lean_inc(v_a_1046_);
lean_dec_ref_known(v___x_1045_, 1);
v_snd_1047_ = lean_ctor_get(v_a_1046_, 1);
lean_inc(v_snd_1047_);
lean_dec(v_a_1046_);
lean_inc(v_a_988_);
lean_inc_ref(v___x_1044_);
lean_inc(v_a_986_);
lean_inc_ref(v_a_985_);
lean_inc(v_a_984_);
lean_inc_ref(v_a_983_);
v___x_1048_ = lean_apply_8(v_k_982_, v_snd_1047_, v_a_983_, v_a_984_, v_a_985_, v_a_986_, v___x_1044_, v_a_988_, lean_box(0));
if (lean_obj_tag(v___x_1048_) == 0)
{
lean_object* v_a_1049_; lean_object* v_fst_1050_; lean_object* v_snd_1051_; lean_object* v_map_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; 
v_a_1049_ = lean_ctor_get(v___x_1048_, 0);
lean_inc(v_a_1049_);
lean_dec_ref_known(v___x_1048_, 1);
v_fst_1050_ = lean_ctor_get(v_a_1049_, 0);
lean_inc(v_fst_1050_);
v_snd_1051_ = lean_ctor_get(v_a_1049_, 1);
lean_inc(v_snd_1051_);
lean_dec(v_a_1049_);
v_map_1052_ = lean_ctor_get(v_options_1012_, 0);
v___x_1053_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__1));
v___x_1054_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1052_, v___x_1053_);
if (lean_obj_tag(v___x_1054_) == 0)
{
v___y_1027_ = v___x_1044_;
v___y_1028_ = v_fst_1050_;
v___y_1029_ = v_snd_1051_;
goto v___jp_1026_;
}
else
{
lean_object* v_val_1055_; 
v_val_1055_ = lean_ctor_get(v___x_1054_, 0);
lean_inc(v_val_1055_);
lean_dec_ref_known(v___x_1054_, 1);
if (lean_obj_tag(v_val_1055_) == 1)
{
uint8_t v_v_1056_; 
v_v_1056_ = lean_ctor_get_uint8(v_val_1055_, 0);
lean_dec_ref_known(v_val_1055_, 0);
if (v_v_1056_ == 0)
{
lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1057_ = lean_box(0);
v___x_1058_ = lp_Qq_Qq_Impl_withDelabQuoted___lam__0(v___x_1057_, v_fst_1050_, v_snd_1051_, v_a_983_, v_a_984_, v_a_985_, v_a_986_, v___x_1044_, v_a_988_);
lean_dec_ref_known(v___x_1044_, 14);
v___y_991_ = v___x_1058_;
goto v___jp_990_;
}
else
{
v___y_1027_ = v___x_1044_;
v___y_1028_ = v_fst_1050_;
v___y_1029_ = v_snd_1051_;
goto v___jp_1026_;
}
}
else
{
lean_dec(v_val_1055_);
v___y_1027_ = v___x_1044_;
v___y_1028_ = v_fst_1050_;
v___y_1029_ = v_snd_1051_;
goto v___jp_1026_;
}
}
}
else
{
lean_object* v_a_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1066_; 
lean_dec_ref_known(v___x_1044_, 14);
v_a_1059_ = lean_ctor_get(v___x_1048_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_1048_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1061_ = v___x_1048_;
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_a_1059_);
lean_dec(v___x_1048_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1064_; 
if (v_isShared_1062_ == 0)
{
v___x_1064_ = v___x_1061_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_a_1059_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
}
}
else
{
lean_object* v_a_1067_; lean_object* v___x_1069_; uint8_t v_isShared_1070_; uint8_t v_isSharedCheck_1074_; 
lean_dec_ref_known(v___x_1044_, 14);
lean_dec_ref(v_k_982_);
v_a_1067_ = lean_ctor_get(v___x_1045_, 0);
v_isSharedCheck_1074_ = !lean_is_exclusive(v___x_1045_);
if (v_isSharedCheck_1074_ == 0)
{
v___x_1069_ = v___x_1045_;
v_isShared_1070_ = v_isSharedCheck_1074_;
goto v_resetjp_1068_;
}
else
{
lean_inc(v_a_1067_);
lean_dec(v___x_1045_);
v___x_1069_ = lean_box(0);
v_isShared_1070_ = v_isSharedCheck_1074_;
goto v_resetjp_1068_;
}
v_resetjp_1068_:
{
lean_object* v___x_1072_; 
if (v_isShared_1070_ == 0)
{
v___x_1072_ = v___x_1069_;
goto v_reusejp_1071_;
}
else
{
lean_object* v_reuseFailAlloc_1073_; 
v_reuseFailAlloc_1073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1073_, 0, v_a_1067_);
v___x_1072_ = v_reuseFailAlloc_1073_;
goto v_reusejp_1071_;
}
v_reusejp_1071_:
{
return v___x_1072_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_withDelabQuoted___boxed(lean_object* v_k_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_, lean_object* v_a_1084_, lean_object* v_a_1085_){
_start:
{
lean_object* v_res_1086_; 
v_res_1086_ = lp_Qq_Qq_Impl_withDelabQuoted(v_k_1078_, v_a_1079_, v_a_1080_, v_a_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
lean_dec(v_a_1084_);
lean_dec_ref(v_a_1083_);
lean_dec(v_a_1082_);
lean_dec_ref(v_a_1081_);
lean_dec(v_a_1080_);
lean_dec_ref(v_a_1079_);
return v_res_1086_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0(lean_object* v_00_u03b2_1087_, lean_object* v_m_1088_, lean_object* v_a_1089_){
_start:
{
lean_object* v___x_1090_; 
v___x_1090_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___redArg(v_m_1088_, v_a_1089_);
return v___x_1090_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0___boxed(lean_object* v_00_u03b2_1091_, lean_object* v_m_1092_, lean_object* v_a_1093_){
_start:
{
lean_object* v_res_1094_; 
v_res_1094_ = lp_Qq_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0(v_00_u03b2_1091_, v_m_1092_, v_a_1093_);
lean_dec_ref(v_a_1093_);
lean_dec_ref(v_m_1092_);
return v_res_1094_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0(lean_object* v_00_u03b2_1095_, lean_object* v_a_1096_, lean_object* v_x_1097_){
_start:
{
lean_object* v___x_1098_; 
v___x_1098_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___redArg(v_a_1096_, v_x_1097_);
return v___x_1098_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1099_, lean_object* v_a_1100_, lean_object* v_x_1101_){
_start:
{
lean_object* v_res_1102_; 
v_res_1102_ = lp_Qq_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Qq_Impl_withDelabQuoted_spec__0_spec__0(v_00_u03b2_1099_, v_a_1100_, v_x_1101_);
lean_dec(v_x_1101_);
lean_dec_ref(v_a_1100_);
return v_res_1102_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(lean_object* v___y_1103_){
_start:
{
lean_object* v_subExpr_1105_; lean_object* v_expr_1106_; lean_object* v___x_1107_; 
v_subExpr_1105_ = lean_ctor_get(v___y_1103_, 3);
v_expr_1106_ = lean_ctor_get(v_subExpr_1105_, 0);
lean_inc_ref(v_expr_1106_);
v___x_1107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1107_, 0, v_expr_1106_);
return v___x_1107_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg___boxed(lean_object* v___y_1108_, lean_object* v___y_1109_){
_start:
{
lean_object* v_res_1110_; 
v_res_1110_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v___y_1108_);
lean_dec_ref(v___y_1108_);
return v_res_1110_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0(lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_){
_start:
{
lean_object* v___x_1118_; 
v___x_1118_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v___y_1111_);
return v___x_1118_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___boxed(lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_){
_start:
{
lean_object* v_res_1126_; 
v_res_1126_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0(v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_);
lean_dec(v___y_1124_);
lean_dec_ref(v___y_1123_);
lean_dec(v___y_1122_);
lean_dec_ref(v___y_1121_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
return v_res_1126_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel___lam__0(uint8_t v___x_1127_, lean_object* v___x_1128_, lean_object* v_a_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_){
_start:
{
lean_object* v___x_1135_; 
v___x_1135_ = lp_Qq_Qq_Impl_unquoteLevelLCtx(v___x_1127_, v___x_1128_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_);
if (lean_obj_tag(v___x_1135_) == 0)
{
lean_object* v_a_1136_; lean_object* v_snd_1137_; lean_object* v___x_1138_; 
v_a_1136_ = lean_ctor_get(v___x_1135_, 0);
lean_inc(v_a_1136_);
lean_dec_ref_known(v___x_1135_, 1);
v_snd_1137_ = lean_ctor_get(v_a_1136_, 1);
lean_inc(v_snd_1137_);
lean_dec(v_a_1136_);
v___x_1138_ = lp_Qq_Qq_Impl_unquoteLevel(v_a_1129_, v_snd_1137_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_);
return v___x_1138_;
}
else
{
lean_object* v_a_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1146_; 
lean_dec_ref(v_a_1129_);
v_a_1139_ = lean_ctor_get(v___x_1135_, 0);
v_isSharedCheck_1146_ = !lean_is_exclusive(v___x_1135_);
if (v_isSharedCheck_1146_ == 0)
{
v___x_1141_ = v___x_1135_;
v_isShared_1142_ = v_isSharedCheck_1146_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_a_1139_);
lean_dec(v___x_1135_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1146_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___x_1144_; 
if (v_isShared_1142_ == 0)
{
v___x_1144_ = v___x_1141_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v_a_1139_);
v___x_1144_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1143_;
}
v_reusejp_1143_:
{
return v___x_1144_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel___lam__0___boxed(lean_object* v___x_1147_, lean_object* v___x_1148_, lean_object* v_a_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_){
_start:
{
uint8_t v___x_1438__boxed_1155_; lean_object* v_res_1156_; 
v___x_1438__boxed_1155_ = lean_unbox(v___x_1147_);
v_res_1156_ = lp_Qq_Qq_Impl_delabQuotedLevel___lam__0(v___x_1438__boxed_1155_, v___x_1148_, v_a_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_);
lean_dec(v___y_1153_);
lean_dec_ref(v___y_1152_);
lean_dec(v___y_1151_);
lean_dec_ref(v___y_1150_);
return v_res_1156_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel(lean_object* v_a_1157_, lean_object* v_a_1158_, lean_object* v_a_1159_, lean_object* v_a_1160_, lean_object* v_a_1161_, lean_object* v_a_1162_){
_start:
{
lean_object* v___x_1164_; lean_object* v_a_1165_; uint8_t v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___f_1169_; lean_object* v___x_1170_; 
v___x_1164_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v_a_1157_);
v_a_1165_ = lean_ctor_get(v___x_1164_, 0);
lean_inc(v_a_1165_);
lean_dec_ref(v___x_1164_);
v___x_1166_ = 0;
v___x_1167_ = lean_obj_once(&lp_Qq_Qq_Impl_withDelabQuoted___closed__3, &lp_Qq_Qq_Impl_withDelabQuoted___closed__3_once, _init_lp_Qq_Qq_Impl_withDelabQuoted___closed__3);
v___x_1168_ = lean_box(v___x_1166_);
v___f_1169_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuotedLevel___lam__0___boxed), 8, 3);
lean_closure_set(v___f_1169_, 0, v___x_1168_);
lean_closure_set(v___f_1169_, 1, v___x_1167_);
lean_closure_set(v___f_1169_, 2, v_a_1165_);
v___x_1170_ = lp_Qq___private_Qq_Delab_0__Qq_Impl_failureOnError___redArg(v___f_1169_, v_a_1159_, v_a_1160_, v_a_1161_, v_a_1162_);
if (lean_obj_tag(v___x_1170_) == 0)
{
lean_object* v_a_1171_; lean_object* v_fst_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v_a_1171_ = lean_ctor_get(v___x_1170_, 0);
lean_inc(v_a_1171_);
lean_dec_ref_known(v___x_1170_, 1);
v_fst_1172_ = lean_ctor_get(v_a_1171_, 0);
lean_inc(v_fst_1172_);
lean_dec(v_a_1171_);
v___x_1173_ = lean_unsigned_to_nat(1024u);
v___x_1174_ = l_Lean_PrettyPrinter_Delaborator_delabLevel(v_fst_1172_, v___x_1173_, v_a_1157_, v_a_1158_, v_a_1159_, v_a_1160_, v_a_1161_, v_a_1162_);
return v___x_1174_;
}
else
{
lean_object* v_a_1175_; lean_object* v___x_1177_; uint8_t v_isShared_1178_; uint8_t v_isSharedCheck_1182_; 
v_a_1175_ = lean_ctor_get(v___x_1170_, 0);
v_isSharedCheck_1182_ = !lean_is_exclusive(v___x_1170_);
if (v_isSharedCheck_1182_ == 0)
{
v___x_1177_ = v___x_1170_;
v_isShared_1178_ = v_isSharedCheck_1182_;
goto v_resetjp_1176_;
}
else
{
lean_inc(v_a_1175_);
lean_dec(v___x_1170_);
v___x_1177_ = lean_box(0);
v_isShared_1178_ = v_isSharedCheck_1182_;
goto v_resetjp_1176_;
}
v_resetjp_1176_:
{
lean_object* v___x_1180_; 
if (v_isShared_1178_ == 0)
{
v___x_1180_ = v___x_1177_;
goto v_reusejp_1179_;
}
else
{
lean_object* v_reuseFailAlloc_1181_; 
v_reuseFailAlloc_1181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1181_, 0, v_a_1175_);
v___x_1180_ = v_reuseFailAlloc_1181_;
goto v_reusejp_1179_;
}
v_reusejp_1179_:
{
return v___x_1180_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevel___boxed(lean_object* v_a_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_, lean_object* v_a_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_Qq_Qq_Impl_delabQuotedLevel(v_a_1183_, v_a_1184_, v_a_1185_, v_a_1186_, v_a_1187_, v_a_1188_);
lean_dec(v_a_1188_);
lean_dec_ref(v_a_1187_);
lean_dec(v_a_1186_);
lean_dec_ref(v_a_1185_);
lean_dec(v_a_1184_);
lean_dec_ref(v_a_1183_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg(lean_object* v_child_1191_, lean_object* v_childIdx_1192_, lean_object* v_x_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_){
_start:
{
lean_object* v_subExpr_1202_; lean_object* v_optionsPerPos_1203_; lean_object* v_currNamespace_1204_; lean_object* v_openDecls_1205_; uint8_t v_inPattern_1206_; lean_object* v_depth_1207_; lean_object* v_lctxInitIndices_1208_; lean_object* v_pos_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; 
v_subExpr_1202_ = lean_ctor_get(v___y_1195_, 3);
v_optionsPerPos_1203_ = lean_ctor_get(v___y_1195_, 0);
v_currNamespace_1204_ = lean_ctor_get(v___y_1195_, 1);
v_openDecls_1205_ = lean_ctor_get(v___y_1195_, 2);
v_inPattern_1206_ = lean_ctor_get_uint8(v___y_1195_, sizeof(void*)*6);
v_depth_1207_ = lean_ctor_get(v___y_1195_, 4);
v_lctxInitIndices_1208_ = lean_ctor_get(v___y_1195_, 5);
v_pos_1209_ = lean_ctor_get(v_subExpr_1202_, 1);
v___x_1210_ = l_Lean_SubExpr_Pos_push(v_pos_1209_, v_childIdx_1192_);
v___x_1211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1211_, 0, v_child_1191_);
lean_ctor_set(v___x_1211_, 1, v___x_1210_);
lean_inc(v_lctxInitIndices_1208_);
lean_inc(v_depth_1207_);
lean_inc(v_openDecls_1205_);
lean_inc(v_currNamespace_1204_);
lean_inc(v_optionsPerPos_1203_);
v___x_1212_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1212_, 0, v_optionsPerPos_1203_);
lean_ctor_set(v___x_1212_, 1, v_currNamespace_1204_);
lean_ctor_set(v___x_1212_, 2, v_openDecls_1205_);
lean_ctor_set(v___x_1212_, 3, v___x_1211_);
lean_ctor_set(v___x_1212_, 4, v_depth_1207_);
lean_ctor_set(v___x_1212_, 5, v_lctxInitIndices_1208_);
lean_ctor_set_uint8(v___x_1212_, sizeof(void*)*6, v_inPattern_1206_);
lean_inc(v___y_1200_);
lean_inc_ref(v___y_1199_);
lean_inc(v___y_1198_);
lean_inc_ref(v___y_1197_);
lean_inc(v___y_1196_);
v___x_1213_ = lean_apply_8(v_x_1193_, v___y_1194_, v___x_1212_, v___y_1196_, v___y_1197_, v___y_1198_, v___y_1199_, v___y_1200_, lean_box(0));
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg___boxed(lean_object* v_child_1214_, lean_object* v_childIdx_1215_, lean_object* v_x_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_){
_start:
{
lean_object* v_res_1225_; 
v_res_1225_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg(v_child_1214_, v_childIdx_1215_, v_x_1216_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_, v___y_1223_);
lean_dec(v___y_1223_);
lean_dec_ref(v___y_1222_);
lean_dec(v___y_1221_);
lean_dec_ref(v___y_1220_);
lean_dec(v___y_1219_);
lean_dec_ref(v___y_1218_);
return v_res_1225_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(lean_object* v_x_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
lean_object* v___x_1235_; lean_object* v_a_1236_; lean_object* v_fst_1237_; lean_object* v_snd_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; 
v___x_1235_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(v___y_1227_, v___y_1228_);
v_a_1236_ = lean_ctor_get(v___x_1235_, 0);
lean_inc(v_a_1236_);
lean_dec_ref(v___x_1235_);
v_fst_1237_ = lean_ctor_get(v_a_1236_, 0);
lean_inc(v_fst_1237_);
v_snd_1238_ = lean_ctor_get(v_a_1236_, 1);
lean_inc(v_snd_1238_);
lean_dec(v_a_1236_);
v___x_1239_ = l_Lean_Expr_appArg_x21(v_fst_1237_);
lean_dec(v_fst_1237_);
v___x_1240_ = lean_unsigned_to_nat(1u);
v___x_1241_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg(v___x_1239_, v___x_1240_, v_x_1226_, v_snd_1238_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_, v___y_1232_, v___y_1233_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg___boxed(lean_object* v_x_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
lean_object* v_res_1251_; 
v_res_1251_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(v_x_1242_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
lean_dec(v___y_1247_);
lean_dec_ref(v___y_1246_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
return v_res_1251_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ___lam__0(lean_object* v___x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_){
_start:
{
lean_object* v___x_1267_; 
v___x_1267_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(v___x_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_, v___y_1265_);
if (lean_obj_tag(v___x_1267_) == 0)
{
lean_object* v_a_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1296_; 
v_a_1268_ = lean_ctor_get(v___x_1267_, 0);
v_isSharedCheck_1296_ = !lean_is_exclusive(v___x_1267_);
if (v_isSharedCheck_1296_ == 0)
{
v___x_1270_ = v___x_1267_;
v_isShared_1271_ = v_isSharedCheck_1296_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_a_1268_);
lean_dec(v___x_1267_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1296_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
lean_object* v_fst_1272_; lean_object* v_snd_1273_; lean_object* v___x_1275_; uint8_t v_isShared_1276_; uint8_t v_isSharedCheck_1295_; 
v_fst_1272_ = lean_ctor_get(v_a_1268_, 0);
v_snd_1273_ = lean_ctor_get(v_a_1268_, 1);
v_isSharedCheck_1295_ = !lean_is_exclusive(v_a_1268_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1275_ = v_a_1268_;
v_isShared_1276_ = v_isSharedCheck_1295_;
goto v_resetjp_1274_;
}
else
{
lean_inc(v_snd_1273_);
lean_inc(v_fst_1272_);
lean_dec(v_a_1268_);
v___x_1275_ = lean_box(0);
v_isShared_1276_ = v_isSharedCheck_1295_;
goto v_resetjp_1274_;
}
v_resetjp_1274_:
{
lean_object* v_ref_1277_; uint8_t v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1290_; 
v_ref_1277_ = lean_ctor_get(v___y_1264_, 5);
v___x_1278_ = 0;
v___x_1279_ = l_Lean_SourceInfo_fromRef(v_ref_1277_, v___x_1278_);
v___x_1280_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQ___lam__0___closed__1));
v___x_1281_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQ___lam__0___closed__2));
lean_inc_n(v___x_1279_, 3);
v___x_1282_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1282_, 0, v___x_1279_);
lean_ctor_set(v___x_1282_, 1, v___x_1281_);
v___x_1283_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__10));
v___x_1284_ = lean_obj_once(&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11, &lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11_once, _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11);
v___x_1285_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1285_, 0, v___x_1279_);
lean_ctor_set(v___x_1285_, 1, v___x_1283_);
lean_ctor_set(v___x_1285_, 2, v___x_1284_);
v___x_1286_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQ___lam__0___closed__3));
v___x_1287_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1279_);
lean_ctor_set(v___x_1287_, 1, v___x_1286_);
v___x_1288_ = l_Lean_Syntax_node4(v___x_1279_, v___x_1280_, v___x_1282_, v_fst_1272_, v___x_1285_, v___x_1287_);
if (v_isShared_1276_ == 0)
{
lean_ctor_set(v___x_1275_, 0, v___x_1288_);
v___x_1290_ = v___x_1275_;
goto v_reusejp_1289_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v___x_1288_);
lean_ctor_set(v_reuseFailAlloc_1294_, 1, v_snd_1273_);
v___x_1290_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1289_;
}
v_reusejp_1289_:
{
lean_object* v___x_1292_; 
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 0, v___x_1290_);
v___x_1292_ = v___x_1270_;
goto v_reusejp_1291_;
}
else
{
lean_object* v_reuseFailAlloc_1293_; 
v_reuseFailAlloc_1293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1293_, 0, v___x_1290_);
v___x_1292_ = v_reuseFailAlloc_1293_;
goto v_reusejp_1291_;
}
v_reusejp_1291_:
{
return v___x_1292_;
}
}
}
}
}
else
{
return v___x_1267_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ___lam__0___boxed(lean_object* v___x_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_){
_start:
{
lean_object* v_res_1306_; 
v_res_1306_ = lp_Qq_Qq_Impl_delabQ___lam__0(v___x_1297_, v___y_1298_, v___y_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
lean_dec(v___y_1304_);
lean_dec_ref(v___y_1303_);
lean_dec(v___y_1302_);
lean_dec_ref(v___y_1301_);
lean_dec(v___y_1300_);
lean_dec_ref(v___y_1299_);
return v_res_1306_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_delabQ___closed__0(void){
_start:
{
lean_object* v___x_1307_; lean_object* v___f_1308_; 
v___x_1307_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuoted___boxed), 8, 0);
v___f_1308_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQ___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1308_, 0, v___x_1307_);
return v___f_1308_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ(lean_object* v_a_1309_, lean_object* v_a_1310_, lean_object* v_a_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_, lean_object* v_a_1314_){
_start:
{
lean_object* v___x_1328_; lean_object* v_a_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; uint8_t v___x_1332_; 
v___x_1328_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v_a_1309_);
v_a_1329_ = lean_ctor_get(v___x_1328_, 0);
lean_inc(v_a_1329_);
lean_dec_ref(v___x_1328_);
v___x_1330_ = l_Lean_Expr_getAppNumArgs(v_a_1329_);
lean_dec(v_a_1329_);
v___x_1331_ = lean_unsigned_to_nat(1u);
v___x_1332_ = lean_nat_dec_eq(v___x_1330_, v___x_1331_);
lean_dec(v___x_1330_);
if (v___x_1332_ == 0)
{
lean_object* v___x_1333_; 
v___x_1333_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1333_) == 0)
{
lean_dec_ref_known(v___x_1333_, 1);
goto v___jp_1316_;
}
else
{
lean_object* v_a_1334_; lean_object* v___x_1336_; uint8_t v_isShared_1337_; uint8_t v_isSharedCheck_1341_; 
v_a_1334_ = lean_ctor_get(v___x_1333_, 0);
v_isSharedCheck_1341_ = !lean_is_exclusive(v___x_1333_);
if (v_isSharedCheck_1341_ == 0)
{
v___x_1336_ = v___x_1333_;
v_isShared_1337_ = v_isSharedCheck_1341_;
goto v_resetjp_1335_;
}
else
{
lean_inc(v_a_1334_);
lean_dec(v___x_1333_);
v___x_1336_ = lean_box(0);
v_isShared_1337_ = v_isSharedCheck_1341_;
goto v_resetjp_1335_;
}
v_resetjp_1335_:
{
lean_object* v___x_1339_; 
if (v_isShared_1337_ == 0)
{
v___x_1339_ = v___x_1336_;
goto v_reusejp_1338_;
}
else
{
lean_object* v_reuseFailAlloc_1340_; 
v_reuseFailAlloc_1340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1340_, 0, v_a_1334_);
v___x_1339_ = v_reuseFailAlloc_1340_;
goto v_reusejp_1338_;
}
v_reusejp_1338_:
{
return v___x_1339_;
}
}
}
}
else
{
goto v___jp_1316_;
}
v___jp_1316_:
{
lean_object* v___x_1317_; 
v___x_1317_ = lp_Qq_Qq_Impl_checkQqDelabOptions(v_a_1309_, v_a_1310_, v_a_1311_, v_a_1312_, v_a_1313_, v_a_1314_);
if (lean_obj_tag(v___x_1317_) == 0)
{
lean_object* v___f_1318_; lean_object* v___x_1319_; 
lean_dec_ref_known(v___x_1317_, 1);
v___f_1318_ = lean_obj_once(&lp_Qq_Qq_Impl_delabQ___closed__0, &lp_Qq_Qq_Impl_delabQ___closed__0_once, _init_lp_Qq_Qq_Impl_delabQ___closed__0);
v___x_1319_ = lp_Qq_Qq_Impl_withDelabQuoted(v___f_1318_, v_a_1309_, v_a_1310_, v_a_1311_, v_a_1312_, v_a_1313_, v_a_1314_);
return v___x_1319_;
}
else
{
lean_object* v_a_1320_; lean_object* v___x_1322_; uint8_t v_isShared_1323_; uint8_t v_isSharedCheck_1327_; 
v_a_1320_ = lean_ctor_get(v___x_1317_, 0);
v_isSharedCheck_1327_ = !lean_is_exclusive(v___x_1317_);
if (v_isSharedCheck_1327_ == 0)
{
v___x_1322_ = v___x_1317_;
v_isShared_1323_ = v_isSharedCheck_1327_;
goto v_resetjp_1321_;
}
else
{
lean_inc(v_a_1320_);
lean_dec(v___x_1317_);
v___x_1322_ = lean_box(0);
v_isShared_1323_ = v_isSharedCheck_1327_;
goto v_resetjp_1321_;
}
v_resetjp_1321_:
{
lean_object* v___x_1325_; 
if (v_isShared_1323_ == 0)
{
v___x_1325_ = v___x_1322_;
goto v_reusejp_1324_;
}
else
{
lean_object* v_reuseFailAlloc_1326_; 
v_reuseFailAlloc_1326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1326_, 0, v_a_1320_);
v___x_1325_ = v_reuseFailAlloc_1326_;
goto v_reusejp_1324_;
}
v_reusejp_1324_:
{
return v___x_1325_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQ___boxed(lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_){
_start:
{
lean_object* v_res_1349_; 
v_res_1349_ = lp_Qq_Qq_Impl_delabQ(v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_, v_a_1346_, v_a_1347_);
lean_dec(v_a_1347_);
lean_dec_ref(v_a_1346_);
lean_dec(v_a_1345_);
lean_dec_ref(v_a_1344_);
lean_dec(v_a_1343_);
lean_dec_ref(v_a_1342_);
return v_res_1349_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0(lean_object* v_00_u03b1_1350_, lean_object* v_child_1351_, lean_object* v_childIdx_1352_, lean_object* v_x_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_){
_start:
{
lean_object* v___x_1362_; 
v___x_1362_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg(v_child_1351_, v_childIdx_1352_, v_x_1353_, v___y_1354_, v___y_1355_, v___y_1356_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_);
return v___x_1362_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1363_, lean_object* v_child_1364_, lean_object* v_childIdx_1365_, lean_object* v_x_1366_, lean_object* v___y_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_){
_start:
{
lean_object* v_res_1375_; 
v_res_1375_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0(v_00_u03b1_1363_, v_child_1364_, v_childIdx_1365_, v_x_1366_, v___y_1367_, v___y_1368_, v___y_1369_, v___y_1370_, v___y_1371_, v___y_1372_, v___y_1373_);
lean_dec(v___y_1373_);
lean_dec_ref(v___y_1372_);
lean_dec(v___y_1371_);
lean_dec_ref(v___y_1370_);
lean_dec(v___y_1369_);
lean_dec_ref(v___y_1368_);
return v_res_1375_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0(lean_object* v_00_u03b1_1376_, lean_object* v_x_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_){
_start:
{
lean_object* v___x_1386_; 
v___x_1386_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(v_x_1377_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_, v___y_1382_, v___y_1383_, v___y_1384_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___boxed(lean_object* v_00_u03b1_1387_, lean_object* v_x_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_){
_start:
{
lean_object* v_res_1397_; 
v_res_1397_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0(v_00_u03b1_1387_, v_x_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_);
lean_dec(v___y_1395_);
lean_dec_ref(v___y_1394_);
lean_dec(v___y_1393_);
lean_dec_ref(v___y_1392_);
lean_dec(v___y_1391_);
lean_dec_ref(v___y_1390_);
return v_res_1397_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq___lam__0(lean_object* v___x_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_){
_start:
{
lean_object* v___x_1412_; 
v___x_1412_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(v___x_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
if (lean_obj_tag(v___x_1412_) == 0)
{
lean_object* v_a_1413_; lean_object* v___x_1415_; uint8_t v_isShared_1416_; uint8_t v_isSharedCheck_1441_; 
v_a_1413_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1441_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1441_ == 0)
{
v___x_1415_ = v___x_1412_;
v_isShared_1416_ = v_isSharedCheck_1441_;
goto v_resetjp_1414_;
}
else
{
lean_inc(v_a_1413_);
lean_dec(v___x_1412_);
v___x_1415_ = lean_box(0);
v_isShared_1416_ = v_isSharedCheck_1441_;
goto v_resetjp_1414_;
}
v_resetjp_1414_:
{
lean_object* v_fst_1417_; lean_object* v_snd_1418_; lean_object* v___x_1420_; uint8_t v_isShared_1421_; uint8_t v_isSharedCheck_1440_; 
v_fst_1417_ = lean_ctor_get(v_a_1413_, 0);
v_snd_1418_ = lean_ctor_get(v_a_1413_, 1);
v_isSharedCheck_1440_ = !lean_is_exclusive(v_a_1413_);
if (v_isSharedCheck_1440_ == 0)
{
v___x_1420_ = v_a_1413_;
v_isShared_1421_ = v_isSharedCheck_1440_;
goto v_resetjp_1419_;
}
else
{
lean_inc(v_snd_1418_);
lean_inc(v_fst_1417_);
lean_dec(v_a_1413_);
v___x_1420_ = lean_box(0);
v_isShared_1421_ = v_isSharedCheck_1440_;
goto v_resetjp_1419_;
}
v_resetjp_1419_:
{
lean_object* v_ref_1422_; uint8_t v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1435_; 
v_ref_1422_ = lean_ctor_get(v___y_1409_, 5);
v___x_1423_ = 0;
v___x_1424_ = l_Lean_SourceInfo_fromRef(v_ref_1422_, v___x_1423_);
v___x_1425_ = ((lean_object*)(lp_Qq_Qq_Impl_delabq___lam__0___closed__1));
v___x_1426_ = ((lean_object*)(lp_Qq_Qq_Impl_delabq___lam__0___closed__2));
lean_inc_n(v___x_1424_, 3);
v___x_1427_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1427_, 0, v___x_1424_);
lean_ctor_set(v___x_1427_, 1, v___x_1426_);
v___x_1428_ = ((lean_object*)(lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__10));
v___x_1429_ = lean_obj_once(&lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11, &lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11_once, _init_lp_Qq___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Qq_Impl_withDelabQuoted_spec__5___closed__11);
v___x_1430_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1424_);
lean_ctor_set(v___x_1430_, 1, v___x_1428_);
lean_ctor_set(v___x_1430_, 2, v___x_1429_);
v___x_1431_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQ___lam__0___closed__3));
v___x_1432_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1432_, 0, v___x_1424_);
lean_ctor_set(v___x_1432_, 1, v___x_1431_);
v___x_1433_ = l_Lean_Syntax_node4(v___x_1424_, v___x_1425_, v___x_1427_, v_fst_1417_, v___x_1430_, v___x_1432_);
if (v_isShared_1421_ == 0)
{
lean_ctor_set(v___x_1420_, 0, v___x_1433_);
v___x_1435_ = v___x_1420_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1439_; 
v_reuseFailAlloc_1439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1439_, 0, v___x_1433_);
lean_ctor_set(v_reuseFailAlloc_1439_, 1, v_snd_1418_);
v___x_1435_ = v_reuseFailAlloc_1439_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
lean_object* v___x_1437_; 
if (v_isShared_1416_ == 0)
{
lean_ctor_set(v___x_1415_, 0, v___x_1435_);
v___x_1437_ = v___x_1415_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v___x_1435_);
v___x_1437_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
return v___x_1437_;
}
}
}
}
}
else
{
return v___x_1412_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq___lam__0___boxed(lean_object* v___x_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_){
_start:
{
lean_object* v_res_1451_; 
v_res_1451_ = lp_Qq_Qq_Impl_delabq___lam__0(v___x_1442_, v___y_1443_, v___y_1444_, v___y_1445_, v___y_1446_, v___y_1447_, v___y_1448_, v___y_1449_);
lean_dec(v___y_1449_);
lean_dec_ref(v___y_1448_);
lean_dec(v___y_1447_);
lean_dec_ref(v___y_1446_);
lean_dec(v___y_1445_);
lean_dec_ref(v___y_1444_);
return v_res_1451_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_delabq___closed__0(void){
_start:
{
lean_object* v___x_1452_; lean_object* v___f_1453_; 
v___x_1452_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuoted___boxed), 8, 0);
v___f_1453_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabq___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1453_, 0, v___x_1452_);
return v___f_1453_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq(lean_object* v_a_1454_, lean_object* v_a_1455_, lean_object* v_a_1456_, lean_object* v_a_1457_, lean_object* v_a_1458_, lean_object* v_a_1459_){
_start:
{
lean_object* v___x_1473_; lean_object* v_a_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; uint8_t v___x_1477_; 
v___x_1473_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v_a_1454_);
v_a_1474_ = lean_ctor_get(v___x_1473_, 0);
lean_inc(v_a_1474_);
lean_dec_ref(v___x_1473_);
v___x_1475_ = l_Lean_Expr_getAppNumArgs(v_a_1474_);
lean_dec(v_a_1474_);
v___x_1476_ = lean_unsigned_to_nat(2u);
v___x_1477_ = lean_nat_dec_eq(v___x_1475_, v___x_1476_);
lean_dec(v___x_1475_);
if (v___x_1477_ == 0)
{
lean_object* v___x_1478_; 
v___x_1478_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1478_) == 0)
{
lean_dec_ref_known(v___x_1478_, 1);
goto v___jp_1461_;
}
else
{
lean_object* v_a_1479_; lean_object* v___x_1481_; uint8_t v_isShared_1482_; uint8_t v_isSharedCheck_1486_; 
v_a_1479_ = lean_ctor_get(v___x_1478_, 0);
v_isSharedCheck_1486_ = !lean_is_exclusive(v___x_1478_);
if (v_isSharedCheck_1486_ == 0)
{
v___x_1481_ = v___x_1478_;
v_isShared_1482_ = v_isSharedCheck_1486_;
goto v_resetjp_1480_;
}
else
{
lean_inc(v_a_1479_);
lean_dec(v___x_1478_);
v___x_1481_ = lean_box(0);
v_isShared_1482_ = v_isSharedCheck_1486_;
goto v_resetjp_1480_;
}
v_resetjp_1480_:
{
lean_object* v___x_1484_; 
if (v_isShared_1482_ == 0)
{
v___x_1484_ = v___x_1481_;
goto v_reusejp_1483_;
}
else
{
lean_object* v_reuseFailAlloc_1485_; 
v_reuseFailAlloc_1485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1485_, 0, v_a_1479_);
v___x_1484_ = v_reuseFailAlloc_1485_;
goto v_reusejp_1483_;
}
v_reusejp_1483_:
{
return v___x_1484_;
}
}
}
}
else
{
goto v___jp_1461_;
}
v___jp_1461_:
{
lean_object* v___x_1462_; 
v___x_1462_ = lp_Qq_Qq_Impl_checkQqDelabOptions(v_a_1454_, v_a_1455_, v_a_1456_, v_a_1457_, v_a_1458_, v_a_1459_);
if (lean_obj_tag(v___x_1462_) == 0)
{
lean_object* v___f_1463_; lean_object* v___x_1464_; 
lean_dec_ref_known(v___x_1462_, 1);
v___f_1463_ = lean_obj_once(&lp_Qq_Qq_Impl_delabq___closed__0, &lp_Qq_Qq_Impl_delabq___closed__0_once, _init_lp_Qq_Qq_Impl_delabq___closed__0);
v___x_1464_ = lp_Qq_Qq_Impl_withDelabQuoted(v___f_1463_, v_a_1454_, v_a_1455_, v_a_1456_, v_a_1457_, v_a_1458_, v_a_1459_);
return v___x_1464_;
}
else
{
lean_object* v_a_1465_; lean_object* v___x_1467_; uint8_t v_isShared_1468_; uint8_t v_isSharedCheck_1472_; 
v_a_1465_ = lean_ctor_get(v___x_1462_, 0);
v_isSharedCheck_1472_ = !lean_is_exclusive(v___x_1462_);
if (v_isSharedCheck_1472_ == 0)
{
v___x_1467_ = v___x_1462_;
v_isShared_1468_ = v_isSharedCheck_1472_;
goto v_resetjp_1466_;
}
else
{
lean_inc(v_a_1465_);
lean_dec(v___x_1462_);
v___x_1467_ = lean_box(0);
v_isShared_1468_ = v_isSharedCheck_1472_;
goto v_resetjp_1466_;
}
v_resetjp_1466_:
{
lean_object* v___x_1470_; 
if (v_isShared_1468_ == 0)
{
v___x_1470_ = v___x_1467_;
goto v_reusejp_1469_;
}
else
{
lean_object* v_reuseFailAlloc_1471_; 
v_reuseFailAlloc_1471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1471_, 0, v_a_1465_);
v___x_1470_ = v_reuseFailAlloc_1471_;
goto v_reusejp_1469_;
}
v_reusejp_1469_:
{
return v___x_1470_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabq___boxed(lean_object* v_a_1487_, lean_object* v_a_1488_, lean_object* v_a_1489_, lean_object* v_a_1490_, lean_object* v_a_1491_, lean_object* v_a_1492_, lean_object* v_a_1493_){
_start:
{
lean_object* v_res_1494_; 
v_res_1494_ = lp_Qq_Qq_Impl_delabq(v_a_1487_, v_a_1488_, v_a_1489_, v_a_1490_, v_a_1491_, v_a_1492_);
lean_dec(v_a_1492_);
lean_dec_ref(v_a_1491_);
lean_dec(v_a_1490_);
lean_dec_ref(v_a_1489_);
lean_dec(v_a_1488_);
lean_dec_ref(v_a_1487_);
return v_res_1494_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg(lean_object* v_x_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_){
_start:
{
lean_object* v___x_1504_; lean_object* v_a_1505_; lean_object* v_fst_1506_; lean_object* v_snd_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; 
v___x_1504_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuoted_spec__1___redArg(v___y_1496_, v___y_1497_);
v_a_1505_ = lean_ctor_get(v___x_1504_, 0);
lean_inc(v_a_1505_);
lean_dec_ref(v___x_1504_);
v_fst_1506_ = lean_ctor_get(v_a_1505_, 0);
lean_inc(v_fst_1506_);
v_snd_1507_ = lean_ctor_get(v_a_1505_, 1);
lean_inc(v_snd_1507_);
lean_dec(v_a_1505_);
v___x_1508_ = l_Lean_Expr_appFn_x21(v_fst_1506_);
lean_dec(v_fst_1506_);
v___x_1509_ = lean_unsigned_to_nat(0u);
v___x_1510_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0_spec__0___redArg(v___x_1508_, v___x_1509_, v_x_1495_, v_snd_1507_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_);
return v___x_1510_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg___boxed(lean_object* v_x_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_, lean_object* v___y_1514_, lean_object* v___y_1515_, lean_object* v___y_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_, lean_object* v___y_1519_){
_start:
{
lean_object* v_res_1520_; 
v_res_1520_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg(v_x_1511_, v___y_1512_, v___y_1513_, v___y_1514_, v___y_1515_, v___y_1516_, v___y_1517_, v___y_1518_);
lean_dec(v___y_1518_);
lean_dec_ref(v___y_1517_);
lean_dec(v___y_1516_);
lean_dec_ref(v___y_1515_);
lean_dec(v___y_1514_);
lean_dec_ref(v___y_1513_);
return v_res_1520_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0(lean_object* v___x_1526_, lean_object* v___x_1527_, lean_object* v___y_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
lean_object* v___x_1536_; 
v___x_1536_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg(v___x_1526_, v___y_1528_, v___y_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_, v___y_1534_);
if (lean_obj_tag(v___x_1536_) == 0)
{
lean_object* v_a_1537_; lean_object* v_fst_1538_; lean_object* v_snd_1539_; lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1570_; 
v_a_1537_ = lean_ctor_get(v___x_1536_, 0);
lean_inc(v_a_1537_);
lean_dec_ref_known(v___x_1536_, 1);
v_fst_1538_ = lean_ctor_get(v_a_1537_, 0);
v_snd_1539_ = lean_ctor_get(v_a_1537_, 1);
v_isSharedCheck_1570_ = !lean_is_exclusive(v_a_1537_);
if (v_isSharedCheck_1570_ == 0)
{
v___x_1541_ = v_a_1537_;
v_isShared_1542_ = v_isSharedCheck_1570_;
goto v_resetjp_1540_;
}
else
{
lean_inc(v_snd_1539_);
lean_inc(v_fst_1538_);
lean_dec(v_a_1537_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1570_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v___x_1543_; 
v___x_1543_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___redArg(v___x_1527_, v_snd_1539_, v___y_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_, v___y_1534_);
if (lean_obj_tag(v___x_1543_) == 0)
{
lean_object* v_a_1544_; lean_object* v___x_1546_; uint8_t v_isShared_1547_; uint8_t v_isSharedCheck_1569_; 
v_a_1544_ = lean_ctor_get(v___x_1543_, 0);
v_isSharedCheck_1569_ = !lean_is_exclusive(v___x_1543_);
if (v_isSharedCheck_1569_ == 0)
{
v___x_1546_ = v___x_1543_;
v_isShared_1547_ = v_isSharedCheck_1569_;
goto v_resetjp_1545_;
}
else
{
lean_inc(v_a_1544_);
lean_dec(v___x_1543_);
v___x_1546_ = lean_box(0);
v_isShared_1547_ = v_isSharedCheck_1569_;
goto v_resetjp_1545_;
}
v_resetjp_1545_:
{
lean_object* v_fst_1548_; lean_object* v_snd_1549_; lean_object* v___x_1551_; uint8_t v_isShared_1552_; uint8_t v_isSharedCheck_1568_; 
v_fst_1548_ = lean_ctor_get(v_a_1544_, 0);
v_snd_1549_ = lean_ctor_get(v_a_1544_, 1);
v_isSharedCheck_1568_ = !lean_is_exclusive(v_a_1544_);
if (v_isSharedCheck_1568_ == 0)
{
v___x_1551_ = v_a_1544_;
v_isShared_1552_ = v_isSharedCheck_1568_;
goto v_resetjp_1550_;
}
else
{
lean_inc(v_snd_1549_);
lean_inc(v_fst_1548_);
lean_dec(v_a_1544_);
v___x_1551_ = lean_box(0);
v_isShared_1552_ = v_isSharedCheck_1568_;
goto v_resetjp_1550_;
}
v_resetjp_1550_:
{
lean_object* v_ref_1553_; uint8_t v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1559_; 
v_ref_1553_ = lean_ctor_get(v___y_1533_, 5);
v___x_1554_ = 0;
v___x_1555_ = l_Lean_SourceInfo_fromRef(v_ref_1553_, v___x_1554_);
v___x_1556_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__1));
v___x_1557_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___closed__2));
lean_inc(v___x_1555_);
if (v_isShared_1542_ == 0)
{
lean_ctor_set_tag(v___x_1541_, 2);
lean_ctor_set(v___x_1541_, 1, v___x_1557_);
lean_ctor_set(v___x_1541_, 0, v___x_1555_);
v___x_1559_ = v___x_1541_;
goto v_reusejp_1558_;
}
else
{
lean_object* v_reuseFailAlloc_1567_; 
v_reuseFailAlloc_1567_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1567_, 0, v___x_1555_);
lean_ctor_set(v_reuseFailAlloc_1567_, 1, v___x_1557_);
v___x_1559_ = v_reuseFailAlloc_1567_;
goto v_reusejp_1558_;
}
v_reusejp_1558_:
{
lean_object* v___x_1560_; lean_object* v___x_1562_; 
v___x_1560_ = l_Lean_Syntax_node3(v___x_1555_, v___x_1556_, v_fst_1538_, v___x_1559_, v_fst_1548_);
if (v_isShared_1552_ == 0)
{
lean_ctor_set(v___x_1551_, 0, v___x_1560_);
v___x_1562_ = v___x_1551_;
goto v_reusejp_1561_;
}
else
{
lean_object* v_reuseFailAlloc_1566_; 
v_reuseFailAlloc_1566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1566_, 0, v___x_1560_);
lean_ctor_set(v_reuseFailAlloc_1566_, 1, v_snd_1549_);
v___x_1562_ = v_reuseFailAlloc_1566_;
goto v_reusejp_1561_;
}
v_reusejp_1561_:
{
lean_object* v___x_1564_; 
if (v_isShared_1547_ == 0)
{
lean_ctor_set(v___x_1546_, 0, v___x_1562_);
v___x_1564_ = v___x_1546_;
goto v_reusejp_1563_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v___x_1562_);
v___x_1564_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1563_;
}
v_reusejp_1563_:
{
return v___x_1564_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_1541_);
lean_dec(v_fst_1538_);
return v___x_1543_;
}
}
}
else
{
lean_dec_ref(v___x_1527_);
return v___x_1536_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___boxed(lean_object* v___x_1571_, lean_object* v___x_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_){
_start:
{
lean_object* v_res_1581_; 
v_res_1581_ = lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0(v___x_1571_, v___x_1572_, v___y_1573_, v___y_1574_, v___y_1575_, v___y_1576_, v___y_1577_, v___y_1578_, v___y_1579_);
lean_dec(v___y_1579_);
lean_dec_ref(v___y_1578_);
lean_dec(v___y_1577_);
lean_dec_ref(v___y_1576_);
lean_dec(v___y_1575_);
lean_dec_ref(v___y_1574_);
return v_res_1581_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_delabQuotedDefEq___closed__0(void){
_start:
{
lean_object* v___x_1582_; lean_object* v___x_1583_; 
v___x_1582_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuoted___boxed), 8, 0);
v___x_1583_ = lean_alloc_closure((void*)(lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQ_spec__0___boxed), 10, 2);
lean_closure_set(v___x_1583_, 0, lean_box(0));
lean_closure_set(v___x_1583_, 1, v___x_1582_);
return v___x_1583_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_delabQuotedDefEq___closed__1(void){
_start:
{
lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___f_1586_; 
v___x_1584_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuoted___boxed), 8, 0);
v___x_1585_ = lean_obj_once(&lp_Qq_Qq_Impl_delabQuotedDefEq___closed__0, &lp_Qq_Qq_Impl_delabQuotedDefEq___closed__0_once, _init_lp_Qq_Qq_Impl_delabQuotedDefEq___closed__0);
v___f_1586_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuotedDefEq___lam__0___boxed), 10, 2);
lean_closure_set(v___f_1586_, 0, v___x_1585_);
lean_closure_set(v___f_1586_, 1, v___x_1584_);
return v___f_1586_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq(lean_object* v_a_1587_, lean_object* v_a_1588_, lean_object* v_a_1589_, lean_object* v_a_1590_, lean_object* v_a_1591_, lean_object* v_a_1592_){
_start:
{
lean_object* v___x_1606_; lean_object* v_a_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; uint8_t v___x_1610_; 
v___x_1606_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v_a_1587_);
v_a_1607_ = lean_ctor_get(v___x_1606_, 0);
lean_inc(v_a_1607_);
lean_dec_ref(v___x_1606_);
v___x_1608_ = l_Lean_Expr_getAppNumArgs(v_a_1607_);
lean_dec(v_a_1607_);
v___x_1609_ = lean_unsigned_to_nat(4u);
v___x_1610_ = lean_nat_dec_eq(v___x_1608_, v___x_1609_);
lean_dec(v___x_1608_);
if (v___x_1610_ == 0)
{
lean_object* v___x_1611_; 
v___x_1611_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1611_) == 0)
{
lean_dec_ref_known(v___x_1611_, 1);
goto v___jp_1594_;
}
else
{
lean_object* v_a_1612_; lean_object* v___x_1614_; uint8_t v_isShared_1615_; uint8_t v_isSharedCheck_1619_; 
v_a_1612_ = lean_ctor_get(v___x_1611_, 0);
v_isSharedCheck_1619_ = !lean_is_exclusive(v___x_1611_);
if (v_isSharedCheck_1619_ == 0)
{
v___x_1614_ = v___x_1611_;
v_isShared_1615_ = v_isSharedCheck_1619_;
goto v_resetjp_1613_;
}
else
{
lean_inc(v_a_1612_);
lean_dec(v___x_1611_);
v___x_1614_ = lean_box(0);
v_isShared_1615_ = v_isSharedCheck_1619_;
goto v_resetjp_1613_;
}
v_resetjp_1613_:
{
lean_object* v___x_1617_; 
if (v_isShared_1615_ == 0)
{
v___x_1617_ = v___x_1614_;
goto v_reusejp_1616_;
}
else
{
lean_object* v_reuseFailAlloc_1618_; 
v_reuseFailAlloc_1618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1618_, 0, v_a_1612_);
v___x_1617_ = v_reuseFailAlloc_1618_;
goto v_reusejp_1616_;
}
v_reusejp_1616_:
{
return v___x_1617_;
}
}
}
}
else
{
goto v___jp_1594_;
}
v___jp_1594_:
{
lean_object* v___x_1595_; 
v___x_1595_ = lp_Qq_Qq_Impl_checkQqDelabOptions(v_a_1587_, v_a_1588_, v_a_1589_, v_a_1590_, v_a_1591_, v_a_1592_);
if (lean_obj_tag(v___x_1595_) == 0)
{
lean_object* v___f_1596_; lean_object* v___x_1597_; 
lean_dec_ref_known(v___x_1595_, 1);
v___f_1596_ = lean_obj_once(&lp_Qq_Qq_Impl_delabQuotedDefEq___closed__1, &lp_Qq_Qq_Impl_delabQuotedDefEq___closed__1_once, _init_lp_Qq_Qq_Impl_delabQuotedDefEq___closed__1);
v___x_1597_ = lp_Qq_Qq_Impl_withDelabQuoted(v___f_1596_, v_a_1587_, v_a_1588_, v_a_1589_, v_a_1590_, v_a_1591_, v_a_1592_);
return v___x_1597_;
}
else
{
lean_object* v_a_1598_; lean_object* v___x_1600_; uint8_t v_isShared_1601_; uint8_t v_isSharedCheck_1605_; 
v_a_1598_ = lean_ctor_get(v___x_1595_, 0);
v_isSharedCheck_1605_ = !lean_is_exclusive(v___x_1595_);
if (v_isSharedCheck_1605_ == 0)
{
v___x_1600_ = v___x_1595_;
v_isShared_1601_ = v_isSharedCheck_1605_;
goto v_resetjp_1599_;
}
else
{
lean_inc(v_a_1598_);
lean_dec(v___x_1595_);
v___x_1600_ = lean_box(0);
v_isShared_1601_ = v_isSharedCheck_1605_;
goto v_resetjp_1599_;
}
v_resetjp_1599_:
{
lean_object* v___x_1603_; 
if (v_isShared_1601_ == 0)
{
v___x_1603_ = v___x_1600_;
goto v_reusejp_1602_;
}
else
{
lean_object* v_reuseFailAlloc_1604_; 
v_reuseFailAlloc_1604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1604_, 0, v_a_1598_);
v___x_1603_ = v_reuseFailAlloc_1604_;
goto v_reusejp_1602_;
}
v_reusejp_1602_:
{
return v___x_1603_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedDefEq___boxed(lean_object* v_a_1620_, lean_object* v_a_1621_, lean_object* v_a_1622_, lean_object* v_a_1623_, lean_object* v_a_1624_, lean_object* v_a_1625_, lean_object* v_a_1626_){
_start:
{
lean_object* v_res_1627_; 
v_res_1627_ = lp_Qq_Qq_Impl_delabQuotedDefEq(v_a_1620_, v_a_1621_, v_a_1622_, v_a_1623_, v_a_1624_, v_a_1625_);
lean_dec(v_a_1625_);
lean_dec_ref(v_a_1624_);
lean_dec(v_a_1623_);
lean_dec_ref(v_a_1622_);
lean_dec(v_a_1621_);
lean_dec_ref(v_a_1620_);
return v_res_1627_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0(lean_object* v_00_u03b1_1628_, lean_object* v_x_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_){
_start:
{
lean_object* v___x_1638_; 
v___x_1638_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___redArg(v_x_1629_, v___y_1630_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_, v___y_1636_);
return v___x_1638_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0___boxed(lean_object* v_00_u03b1_1639_, lean_object* v_x_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_){
_start:
{
lean_object* v_res_1649_; 
v_res_1649_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedDefEq_spec__0(v_00_u03b1_1639_, v_x_1640_, v___y_1641_, v___y_1642_, v___y_1643_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_);
lean_dec(v___y_1647_);
lean_dec_ref(v___y_1646_);
lean_dec(v___y_1645_);
lean_dec_ref(v___y_1644_);
lean_dec(v___y_1643_);
lean_dec_ref(v___y_1642_);
return v_res_1649_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg(lean_object* v_child_1650_, lean_object* v_childIdx_1651_, lean_object* v_x_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_){
_start:
{
lean_object* v_subExpr_1660_; lean_object* v_optionsPerPos_1661_; lean_object* v_currNamespace_1662_; lean_object* v_openDecls_1663_; uint8_t v_inPattern_1664_; lean_object* v_depth_1665_; lean_object* v_lctxInitIndices_1666_; lean_object* v_pos_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; 
v_subExpr_1660_ = lean_ctor_get(v___y_1653_, 3);
v_optionsPerPos_1661_ = lean_ctor_get(v___y_1653_, 0);
v_currNamespace_1662_ = lean_ctor_get(v___y_1653_, 1);
v_openDecls_1663_ = lean_ctor_get(v___y_1653_, 2);
v_inPattern_1664_ = lean_ctor_get_uint8(v___y_1653_, sizeof(void*)*6);
v_depth_1665_ = lean_ctor_get(v___y_1653_, 4);
v_lctxInitIndices_1666_ = lean_ctor_get(v___y_1653_, 5);
v_pos_1667_ = lean_ctor_get(v_subExpr_1660_, 1);
v___x_1668_ = l_Lean_SubExpr_Pos_push(v_pos_1667_, v_childIdx_1651_);
v___x_1669_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1669_, 0, v_child_1650_);
lean_ctor_set(v___x_1669_, 1, v___x_1668_);
lean_inc(v_lctxInitIndices_1666_);
lean_inc(v_depth_1665_);
lean_inc(v_openDecls_1663_);
lean_inc(v_currNamespace_1662_);
lean_inc(v_optionsPerPos_1661_);
v___x_1670_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1670_, 0, v_optionsPerPos_1661_);
lean_ctor_set(v___x_1670_, 1, v_currNamespace_1662_);
lean_ctor_set(v___x_1670_, 2, v_openDecls_1663_);
lean_ctor_set(v___x_1670_, 3, v___x_1669_);
lean_ctor_set(v___x_1670_, 4, v_depth_1665_);
lean_ctor_set(v___x_1670_, 5, v_lctxInitIndices_1666_);
lean_ctor_set_uint8(v___x_1670_, sizeof(void*)*6, v_inPattern_1664_);
lean_inc(v___y_1658_);
lean_inc_ref(v___y_1657_);
lean_inc(v___y_1656_);
lean_inc_ref(v___y_1655_);
lean_inc(v___y_1654_);
v___x_1671_ = lean_apply_7(v_x_1652_, v___x_1670_, v___y_1654_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_, lean_box(0));
return v___x_1671_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg___boxed(lean_object* v_child_1672_, lean_object* v_childIdx_1673_, lean_object* v_x_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_){
_start:
{
lean_object* v_res_1682_; 
v_res_1682_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg(v_child_1672_, v_childIdx_1673_, v_x_1674_, v___y_1675_, v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_);
lean_dec(v___y_1680_);
lean_dec_ref(v___y_1679_);
lean_dec(v___y_1678_);
lean_dec_ref(v___y_1677_);
lean_dec(v___y_1676_);
lean_dec_ref(v___y_1675_);
return v_res_1682_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg(lean_object* v_x_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_){
_start:
{
lean_object* v___x_1691_; lean_object* v_a_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; 
v___x_1691_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v___y_1684_);
v_a_1692_ = lean_ctor_get(v___x_1691_, 0);
lean_inc(v_a_1692_);
lean_dec_ref(v___x_1691_);
v___x_1693_ = l_Lean_Expr_appFn_x21(v_a_1692_);
lean_dec(v_a_1692_);
v___x_1694_ = lean_unsigned_to_nat(0u);
v___x_1695_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg(v___x_1693_, v___x_1694_, v_x_1683_, v___y_1684_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_, v___y_1689_);
return v___x_1695_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg___boxed(lean_object* v_x_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_){
_start:
{
lean_object* v_res_1704_; 
v_res_1704_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg(v_x_1696_, v___y_1697_, v___y_1698_, v___y_1699_, v___y_1700_, v___y_1701_, v___y_1702_);
lean_dec(v___y_1702_);
lean_dec_ref(v___y_1701_);
lean_dec(v___y_1700_);
lean_dec_ref(v___y_1699_);
lean_dec(v___y_1698_);
lean_dec_ref(v___y_1697_);
return v_res_1704_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg(lean_object* v_x_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_){
_start:
{
lean_object* v___x_1713_; lean_object* v_a_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; 
v___x_1713_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v___y_1706_);
v_a_1714_ = lean_ctor_get(v___x_1713_, 0);
lean_inc(v_a_1714_);
lean_dec_ref(v___x_1713_);
v___x_1715_ = l_Lean_Expr_appArg_x21(v_a_1714_);
lean_dec(v_a_1714_);
v___x_1716_ = lean_unsigned_to_nat(1u);
v___x_1717_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg(v___x_1715_, v___x_1716_, v_x_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_);
return v___x_1717_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg___boxed(lean_object* v_x_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_){
_start:
{
lean_object* v_res_1726_; 
v_res_1726_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg(v_x_1718_, v___y_1719_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_);
lean_dec(v___y_1724_);
lean_dec_ref(v___y_1723_);
lean_dec(v___y_1722_);
lean_dec_ref(v___y_1721_);
lean_dec(v___y_1720_);
lean_dec_ref(v___y_1719_);
return v_res_1726_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0(lean_object* v_00_u03b1_1727_, lean_object* v_x_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_){
_start:
{
lean_object* v___x_1736_; 
v___x_1736_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg(v_x_1728_, v___y_1729_, v___y_1730_, v___y_1731_, v___y_1732_, v___y_1733_, v___y_1734_);
return v___x_1736_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___boxed(lean_object* v_00_u03b1_1737_, lean_object* v_x_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_){
_start:
{
lean_object* v_res_1746_; 
v_res_1746_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0(v_00_u03b1_1737_, v_x_1738_, v___y_1739_, v___y_1740_, v___y_1741_, v___y_1742_, v___y_1743_, v___y_1744_);
lean_dec(v___y_1744_);
lean_dec_ref(v___y_1743_);
lean_dec(v___y_1742_);
lean_dec_ref(v___y_1741_);
lean_dec(v___y_1740_);
lean_dec_ref(v___y_1739_);
return v_res_1746_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__0(void){
_start:
{
lean_object* v___x_1747_; lean_object* v___x_1748_; 
v___x_1747_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuotedLevel___boxed), 7, 0);
v___x_1748_ = lean_alloc_closure((void*)(lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___boxed), 9, 2);
lean_closure_set(v___x_1748_, 0, lean_box(0));
lean_closure_set(v___x_1748_, 1, v___x_1747_);
return v___x_1748_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq(lean_object* v_a_1754_, lean_object* v_a_1755_, lean_object* v_a_1756_, lean_object* v_a_1757_, lean_object* v_a_1758_, lean_object* v_a_1759_){
_start:
{
lean_object* v___x_1791_; lean_object* v_a_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; uint8_t v___x_1795_; 
v___x_1791_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Qq_Impl_delabQuotedLevel_spec__0___redArg(v_a_1754_);
v_a_1792_ = lean_ctor_get(v___x_1791_, 0);
lean_inc(v_a_1792_);
lean_dec_ref(v___x_1791_);
v___x_1793_ = l_Lean_Expr_getAppNumArgs(v_a_1792_);
lean_dec(v_a_1792_);
v___x_1794_ = lean_unsigned_to_nat(2u);
v___x_1795_ = lean_nat_dec_eq(v___x_1793_, v___x_1794_);
lean_dec(v___x_1793_);
if (v___x_1795_ == 0)
{
lean_object* v___x_1796_; 
v___x_1796_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1796_) == 0)
{
lean_dec_ref_known(v___x_1796_, 1);
goto v___jp_1761_;
}
else
{
lean_object* v_a_1797_; lean_object* v___x_1799_; uint8_t v_isShared_1800_; uint8_t v_isSharedCheck_1804_; 
v_a_1797_ = lean_ctor_get(v___x_1796_, 0);
v_isSharedCheck_1804_ = !lean_is_exclusive(v___x_1796_);
if (v_isSharedCheck_1804_ == 0)
{
v___x_1799_ = v___x_1796_;
v_isShared_1800_ = v_isSharedCheck_1804_;
goto v_resetjp_1798_;
}
else
{
lean_inc(v_a_1797_);
lean_dec(v___x_1796_);
v___x_1799_ = lean_box(0);
v_isShared_1800_ = v_isSharedCheck_1804_;
goto v_resetjp_1798_;
}
v_resetjp_1798_:
{
lean_object* v___x_1802_; 
if (v_isShared_1800_ == 0)
{
v___x_1802_ = v___x_1799_;
goto v_reusejp_1801_;
}
else
{
lean_object* v_reuseFailAlloc_1803_; 
v_reuseFailAlloc_1803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1803_, 0, v_a_1797_);
v___x_1802_ = v_reuseFailAlloc_1803_;
goto v_reusejp_1801_;
}
v_reusejp_1801_:
{
return v___x_1802_;
}
}
}
}
else
{
goto v___jp_1761_;
}
v___jp_1761_:
{
lean_object* v___x_1762_; 
v___x_1762_ = lp_Qq_Qq_Impl_checkQqDelabOptions(v_a_1754_, v_a_1755_, v_a_1756_, v_a_1757_, v_a_1758_, v_a_1759_);
if (lean_obj_tag(v___x_1762_) == 0)
{
lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; 
lean_dec_ref_known(v___x_1762_, 1);
v___x_1763_ = lean_alloc_closure((void*)(lp_Qq_Qq_Impl_delabQuotedLevel___boxed), 7, 0);
v___x_1764_ = lean_obj_once(&lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__0, &lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__0_once, _init_lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__0);
v___x_1765_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg(v___x_1764_, v_a_1754_, v_a_1755_, v_a_1756_, v_a_1757_, v_a_1758_, v_a_1759_);
if (lean_obj_tag(v___x_1765_) == 0)
{
lean_object* v_a_1766_; lean_object* v___x_1767_; 
v_a_1766_ = lean_ctor_get(v___x_1765_, 0);
lean_inc(v_a_1766_);
lean_dec_ref_known(v___x_1765_, 1);
v___x_1767_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0___redArg(v___x_1763_, v_a_1754_, v_a_1755_, v_a_1756_, v_a_1757_, v_a_1758_, v_a_1759_);
if (lean_obj_tag(v___x_1767_) == 0)
{
lean_object* v_a_1768_; lean_object* v___x_1770_; uint8_t v_isShared_1771_; uint8_t v_isSharedCheck_1782_; 
v_a_1768_ = lean_ctor_get(v___x_1767_, 0);
v_isSharedCheck_1782_ = !lean_is_exclusive(v___x_1767_);
if (v_isSharedCheck_1782_ == 0)
{
v___x_1770_ = v___x_1767_;
v_isShared_1771_ = v_isSharedCheck_1782_;
goto v_resetjp_1769_;
}
else
{
lean_inc(v_a_1768_);
lean_dec(v___x_1767_);
v___x_1770_ = lean_box(0);
v_isShared_1771_ = v_isSharedCheck_1782_;
goto v_resetjp_1769_;
}
v_resetjp_1769_:
{
lean_object* v_ref_1772_; uint8_t v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1780_; 
v_ref_1772_ = lean_ctor_get(v_a_1758_, 5);
v___x_1773_ = 0;
v___x_1774_ = l_Lean_SourceInfo_fromRef(v_ref_1772_, v___x_1773_);
v___x_1775_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__2));
v___x_1776_ = ((lean_object*)(lp_Qq_Qq_Impl_delabQuotedLevelDefEq___closed__3));
lean_inc(v___x_1774_);
v___x_1777_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1777_, 0, v___x_1774_);
lean_ctor_set(v___x_1777_, 1, v___x_1776_);
v___x_1778_ = l_Lean_Syntax_node3(v___x_1774_, v___x_1775_, v_a_1766_, v___x_1777_, v_a_1768_);
if (v_isShared_1771_ == 0)
{
lean_ctor_set(v___x_1770_, 0, v___x_1778_);
v___x_1780_ = v___x_1770_;
goto v_reusejp_1779_;
}
else
{
lean_object* v_reuseFailAlloc_1781_; 
v_reuseFailAlloc_1781_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1781_, 0, v___x_1778_);
v___x_1780_ = v_reuseFailAlloc_1781_;
goto v_reusejp_1779_;
}
v_reusejp_1779_:
{
return v___x_1780_;
}
}
}
else
{
lean_dec(v_a_1766_);
return v___x_1767_;
}
}
else
{
lean_dec_ref(v___x_1763_);
return v___x_1765_;
}
}
else
{
lean_object* v_a_1783_; lean_object* v___x_1785_; uint8_t v_isShared_1786_; uint8_t v_isSharedCheck_1790_; 
v_a_1783_ = lean_ctor_get(v___x_1762_, 0);
v_isSharedCheck_1790_ = !lean_is_exclusive(v___x_1762_);
if (v_isSharedCheck_1790_ == 0)
{
v___x_1785_ = v___x_1762_;
v_isShared_1786_ = v_isSharedCheck_1790_;
goto v_resetjp_1784_;
}
else
{
lean_inc(v_a_1783_);
lean_dec(v___x_1762_);
v___x_1785_ = lean_box(0);
v_isShared_1786_ = v_isSharedCheck_1790_;
goto v_resetjp_1784_;
}
v_resetjp_1784_:
{
lean_object* v___x_1788_; 
if (v_isShared_1786_ == 0)
{
v___x_1788_ = v___x_1785_;
goto v_reusejp_1787_;
}
else
{
lean_object* v_reuseFailAlloc_1789_; 
v_reuseFailAlloc_1789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1789_, 0, v_a_1783_);
v___x_1788_ = v_reuseFailAlloc_1789_;
goto v_reusejp_1787_;
}
v_reusejp_1787_:
{
return v___x_1788_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_delabQuotedLevelDefEq___boxed(lean_object* v_a_1805_, lean_object* v_a_1806_, lean_object* v_a_1807_, lean_object* v_a_1808_, lean_object* v_a_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_){
_start:
{
lean_object* v_res_1812_; 
v_res_1812_ = lp_Qq_Qq_Impl_delabQuotedLevelDefEq(v_a_1805_, v_a_1806_, v_a_1807_, v_a_1808_, v_a_1809_, v_a_1810_);
lean_dec(v_a_1810_);
lean_dec_ref(v_a_1809_);
lean_dec(v_a_1808_);
lean_dec_ref(v_a_1807_);
lean_dec(v_a_1806_);
lean_dec_ref(v_a_1805_);
return v_res_1812_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0(lean_object* v_00_u03b1_1813_, lean_object* v_child_1814_, lean_object* v_childIdx_1815_, lean_object* v_x_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_){
_start:
{
lean_object* v___x_1824_; 
v___x_1824_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___redArg(v_child_1814_, v_childIdx_1815_, v_x_1816_, v___y_1817_, v___y_1818_, v___y_1819_, v___y_1820_, v___y_1821_, v___y_1822_);
return v___x_1824_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1825_, lean_object* v_child_1826_, lean_object* v_childIdx_1827_, lean_object* v_x_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_, lean_object* v___y_1832_, lean_object* v___y_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_){
_start:
{
lean_object* v_res_1836_; 
v_res_1836_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Qq_Impl_delabQuotedLevelDefEq_spec__0_spec__0(v_00_u03b1_1825_, v_child_1826_, v_childIdx_1827_, v_x_1828_, v___y_1829_, v___y_1830_, v___y_1831_, v___y_1832_, v___y_1833_, v___y_1834_);
lean_dec(v___y_1834_);
lean_dec_ref(v___y_1833_);
lean_dec(v___y_1832_);
lean_dec_ref(v___y_1831_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
return v_res_1836_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1(lean_object* v_00_u03b1_1837_, lean_object* v_x_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_){
_start:
{
lean_object* v___x_1846_; 
v___x_1846_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___redArg(v_x_1838_, v___y_1839_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
return v___x_1846_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1___boxed(lean_object* v_00_u03b1_1847_, lean_object* v_x_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_){
_start:
{
lean_object* v_res_1856_; 
v_res_1856_ = lp_Qq_Lean_PrettyPrinter_Delaborator_SubExpr_withAppFn___at___00Qq_Impl_delabQuotedLevelDefEq_spec__1(v_00_u03b1_1847_, v_x_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_);
lean_dec(v___y_1854_);
lean_dec_ref(v___y_1853_);
lean_dec(v___y_1852_);
lean_dec_ref(v___y_1851_);
lean_dec(v___y_1850_);
lean_dec_ref(v___y_1849_);
return v_res_1856_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Macro(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_Delab(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Macro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_Qq___private_Qq_Delab_0__Qq_Impl_initFn_00___x40_Qq_Delab_3595483951____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_Qq_Qq_Impl_pp_qq = lean_io_result_get_value(res);
lean_mark_persistent(lp_Qq_Qq_Impl_pp_qq);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_Delab(uint8_t builtin) {
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
lean_object* initialize_Qq_Qq_Macro(uint8_t builtin);
lean_object* initialize_Qq_Qq_Typ(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_Delab(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Macro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Delab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_Delab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_Delab(builtin);
}
#ifdef __cplusplus
}
#endif
